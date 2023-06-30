import numpy as np
import torch
import pickle
from lightning import LightningModule
from rich.progress import Progress

from model.unet import UNetModel
from model.diffusion import Diffusion
from config import Config


class PepeGenerator(LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.diffusion = Diffusion(config)
        self.model = UNetModel(config)

        self.loss_func = torch.nn.MSELoss()

        self.example_input_array = torch.Tensor(config.batch_size, 3, config.image_size, config.image_size), \
            torch.ones(config.batch_size)
        self.save_hyperparameters(self.config.__dict__)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr)
        if self.config.scheduler is None or self.config.scheduler.lower() in ('no', 'none'):
            print('No scheduler')
            return optimizer
        elif self.config.scheduler.lower() == 'multisteplr':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma=0.1, milestones=(10, 20, 30),
                                                             verbose=False)
            return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler}}
        elif self.config.scheduler.lower() == 'reducelronplateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=4, min_lr=1e-6,
                                                                   mode='min', verbose=False)
            return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss'}}
        else:
            raise NotImplemented(self.config.scheduler)

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch)
        self.log("val_loss", loss)
        return

    def on_train_start(self) -> None:
        # log hparams
        self.logger.log_hyperparams(params=self.config.__dict__)
        # save hparams
        with open(self.logger.log_dir + '/config.pkl', 'wb') as f:
            pickle.dump(self.config, f, pickle.HIGHEST_PROTOCOL)

    def on_validation_end(self) -> None:
        if self.global_step > 0:  # to skip sanity check
            self._log_images_and_dists()

    def forward(self, x, t):
        return self.model(x, t)

    def _calculate_loss(self, batch):
        ts = self.diffusion.sample_timesteps(batch.shape[0])
        noised_batch, noise = self.diffusion.noise_images(batch, ts)
        output = self.forward(noised_batch, ts)
        loss = self.loss_func(output, noise)
        return loss

    def denoise_sample(self, x, t):
        predicted_noise = self.forward(x, t.repeat(x.shape[0]))
        x = self.diffusion.denoise_images(x, t, predicted_noise)
        return x

    def generate_samples(self, num_images, progress: Progress) -> torch.Tensor:
        torch.manual_seed(137)
        progress.generating_progress_bar_id = progress.add_task("[white]Generating images",
                                                                total=self.config.diffusion_steps-1)
        # Generate samples from denoising process
        x = torch.randn((num_images, 3, self.config.image_size, self.config.image_size), device=self.device)
        sample_steps = torch.arange(self.config.diffusion_steps - 1, 0, -1, device=self.device)
        for t in sample_steps:
            progress.update(progress.generating_progress_bar_id, advance=1, visible=True)
            progress.refresh()
            x = self.denoise_sample(x, t)
        return x

    @staticmethod
    def generated_samples_to_images(gen_samples: torch.Tensor, grid_size=(4, 4)) -> np.ndarray:
        assert gen_samples.shape[0] == grid_size[0] * grid_size[1]

        gen_samples = (gen_samples.clamp(-1, 1) + 1) / 2
        gen_samples = (gen_samples * 255).type(torch.uint8)

        # stack images
        gen_samples = torch.cat(torch.split(gen_samples, grid_size[0], dim=0), dim=2)
        gen_samples = torch.cat(torch.split(gen_samples, 1, dim=0), dim=3)
        gen_samples = gen_samples.squeeze().cpu().numpy()
        gen_samples = np.moveaxis(gen_samples, 0, -1)

        return gen_samples

    def _log_images_and_dists(self, grid_size=(3, 3)):
        """
        Generate some images to log them and their distribution
        """

        with self.trainer.progress_bar_callback.progress as progress:
            gen_samples = self.generate_samples(grid_size[0] * grid_size[1], progress)

        # distributions
        self.logger.experiment.add_histogram('generated_distribution', gen_samples,
                                             global_step=self.current_epoch)

        # images
        images = self.generated_samples_to_images(gen_samples, grid_size)
        self.logger.experiment.add_image('generated images', images, self.current_epoch, dataformats="HWC")
