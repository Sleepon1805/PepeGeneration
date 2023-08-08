import numpy as np
import torch
import pickle
from typing import Tuple
from rich.progress import Progress
from lightning import LightningModule
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

from model.unet import UNetModel
from model.diffusion_sampler import Sampler, DDPM_Diffusion
from config import Config


class PepeGenerator(LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.sampler: Sampler = DDPM_Diffusion(config)
        self.model = UNetModel(config)

        self.loss_func = torch.nn.MSELoss()

        self.example_input_array = (
            (
                torch.Tensor(config.batch_size, 3, config.image_size, config.image_size),
                torch.ones(config.batch_size, config.condition_size)
            ),
            torch.ones(config.batch_size),
        )
        self.save_hyperparameters(self.config.__dict__)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr)
        if self.config.scheduler is None or self.config.scheduler.lower() in ('no', 'none'):
            print('No scheduler')
            return optimizer
        elif self.config.scheduler.lower() == 'multisteplr':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma=0.1, milestones=(5, ),
                                                             verbose=False)
            return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler}}
        elif self.config.scheduler.lower() == 'reducelronplateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=4, min_lr=1e-6,
                                                                   mode='min', verbose=False)
            return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss'}}
        else:
            raise NotImplemented(self.config.scheduler)

    def to(self, device):
        super().to(device)
        self.sampler.to(device)

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

    def on_train_start(self) -> None:
        # save hparams
        with open(self.logger.log_dir + '/config.pkl', 'wb') as f:
            pickle.dump(self.config, f, pickle.HIGHEST_PROTOCOL)
        self.logger.log_hyperparams(self.config.__dict__)

    def forward(self, batch, t):
        x, *labels = batch

        if len(labels) == 0:
            cond = torch.bernoulli(torch.full((x.shape[0], self.config.condition_size),
                                              0.5, device=self.device)) * 2 - 1
        elif len(labels) == 1:
            cond = labels[0]
        else:
            raise ValueError

        return self.model(x, t, cond=cond)

    def _calculate_loss(self, batch):
        image_batch, *labels = batch
        ts = self.sampler.sample_timesteps(image_batch.shape[0])
        noised_image_batch, noise = self.sampler.noise_images(image_batch, ts)
        noised_batch = (noised_image_batch, *labels)
        output = self.forward(noised_batch, ts)
        loss = self.loss_func(output, noise)
        return loss

    def generate_samples(self, batch, progress: Progress = None, seed=42) -> torch.Tensor:
        return self.sampler.generate_samples(self, batch, progress, seed)

    """
    Methods for evaluation
    """

    def log_generated_images(self, batch, grid_size=(3, 3)):
        """
        Generate some images to log them, their distribution and FID metric
        """

        with self.trainer.progress_bar_callback.progress as progress:
            gen_samples = self.generate_samples(batch, progress)

        # images
        images = self.generated_samples_to_images(gen_samples, grid_size)
        self.logger.experiment.add_image('generated images', images, self.current_epoch, dataformats="HWC")

        # distributions
        try:
            self.logger.experiment.add_histogram('generated_distribution', gen_samples.clip(-10, 10),
                                                 global_step=self.current_epoch)
        except:
            print('Could not log histogram')

        # Frechet Inception Distance
        fid_loss = self.calculate_fid(gen_samples)
        self.log('fid_metric', fid_loss)

        # Inception Score
        inception_score = self.calculate_inception_score(gen_samples)
        self.log('inception_score', inception_score)

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
