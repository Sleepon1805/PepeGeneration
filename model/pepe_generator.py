import torch
from lightning import LightningModule
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

from model.embedding import EmbeddingModel
from model.unet import UNetModel
from SDE_sampling.sde_sampler import PC_Sampler
from config import Config, save_config


class PepeGenerator(LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.data_config = config.data_config
        self.training_config = config.training_config

        self.sampler = PC_Sampler(self, config.sampler_config)
        self.embedding_model = EmbeddingModel(config.model_config)
        self.backbone_model = UNetModel(config.model_config)

        self.loss_func = torch.nn.MSELoss()

        self.example_input_array = (
            (
                torch.rand(
                    self.training_config.batch_size,
                    3,
                    self.data_config.image_size,
                    self.data_config.image_size
                ),
                torch.ones(
                    self.training_config.batch_size,
                    self.data_config.condition_size
                )
            ),
            torch.ones(self.training_config.batch_size),
        )
        self.save_hyperparameters(self.config.__dict__)

    def forward(self, batch, t):
        x, cond = batch

        # drift and diffusion coefficients instead of time itself
        time_scales = self.sampler.sde.get_sde_scales(t)

        emb = self.embedding_model(*time_scales, cond)
        output = self.backbone_model(x, emb)
        return output

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch)
        self.log("val_loss", loss)

        if batch_idx == 0 and self.global_step > 0:  # to skip sanity check
            self.log_generated_images(batch)
        return

    def _calculate_loss(self, batch):
        image_batch, *labels = batch
        ts = self.sampler.sample_timesteps(image_batch.shape[0])
        noised_image_batch, noise = self.sampler.noise_images(image_batch, ts)
        noised_batch = (noised_image_batch, *labels)
        output = self.forward(noised_batch, ts)
        loss = self.loss_func(output, noise)
        return loss

    """ LightningModule methods """

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.training_config.lr)
        scheduler_name = (self.training_config.scheduler or 'none').lower()
        if scheduler_name in ('no', 'none'):
            print('No scheduler')
            return optimizer
        elif scheduler_name == 'multisteplr':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, gamma=0.1, milestones=(5, 10)
            )
            return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler}}
        elif scheduler_name == 'reducelronplateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=0.1, patience=4, min_lr=1e-6, mode='min'
            )
            return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss'}}
        else:
            raise NotImplemented(self.training_config.scheduler)

    def to(self, device):
        super().to(device)
        self.sampler.to(device)
        return self

    def on_train_start(self) -> None:
        # save hparams
        save_config(self.config, self.logger.log_dir)
        self.logger.log_hyperparams(self.config.__dict__)

    """
    Methods for evaluation
    """

    def log_generated_images(self, batch, grid_size=(3, 3)):
        """
        Generate some images to log them, their distribution and FID metric
        """

        if not self.training_config.calculate_fid:
            num_images = min(batch[0].shape[0], grid_size[0] * grid_size[1])
            batch[0] = batch[0][:num_images]
            batch[1] = batch[1][:num_images]

        gen_samples = self.sampler.generate_samples(self, tuple(batch))

        # images
        images = self.sampler.generated_samples_to_images(gen_samples, grid_size)
        self.logger.experiment.add_image(  # noqa
            'generated images', images, self.current_epoch, dataformats="HWC"
        )

        # distributions
        try:
            self.logger.experiment.add_histogram(  # noqa
                'generated_distribution', gen_samples.clip(-2.5, 2.5), global_step=self.current_epoch
            )
        except:
            print('Could not log histogram')

        if self.training_config.calculate_fid:
            # Frechet Inception Distance
            fid_loss = self.calculate_fid(gen_samples)
            self.log('fid_metric', fid_loss)

            # Inception Score
            inception_score = self.calculate_inception_score(gen_samples)
            self.log('inception_score', inception_score)
        else:
            self.log('fid_metric', -1)
            self.log('inception_score', -1)

    def calculate_fid(self, gen_samples: torch.Tensor):
        fid_metric = FrechetInceptionDistance(feature=192, reset_real_features=False, normalize=True)

        # add real images to fid
        first_val_batch = next(iter(self.trainer.val_dataloaders))
        samples = first_val_batch[0]
        images = (samples + 1) / 2
        fid_metric.update(images, real=True)

        fid_metric.update((gen_samples.cpu().clamp(-1, 1) + 1) / 2, real=False)
        fid_loss = fid_metric.compute().item()
        return fid_loss

    @staticmethod
    def calculate_inception_score(gen_samples: torch.Tensor):
        inception_score = InceptionScore(normalize=True)

        inception_score.update((gen_samples.cpu().clamp(-1, 1) + 1) / 2)
        inception_score = inception_score.compute()[0].item()
        return inception_score
