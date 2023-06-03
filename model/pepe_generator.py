import torch
import pytorch_lightning as pl
from rich.progress import Progress

from model.unet import UNetModel
from model.diffusion import Diffusion
from config import Config


class PepeGenerator(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.diffusion = Diffusion(config)
        self.model = UNetModel(config)

        self.loss_func = torch.nn.MSELoss()

        self.example_input_array = torch.Tensor(config.batch_size, 3, config.image_size, config.image_size), \
            torch.ones(config.batch_size)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr)
        if self.config.scheduler is None or self.config.scheduler['name'].lower() in ('no', 'none'):
            print('No scheduler')
            return optimizer
        elif self.config.scheduler['name'].lower() == 'multisteplr':
            print(f'Using MultiStepLR({str(self.config.scheduler["params"])})')
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                             **self.config.scheduler["params"],
                                                             verbose=True)
            return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler}}
        elif self.config.scheduler['name'].lower() == 'reducelronplateau':
            print(f'Using ReduceLROnPlateau({str(self.config.scheduler["params"])})')
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                   **self.config.scheduler["params"],
                                                                   mode='min', verbose=True)
            return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss'}}
        else:
            raise NotImplemented(self.config.scheduler['name'])

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

    def on_validation_end(self) -> None:
        # generate some images to log their distribution
        if self.global_step > 0:  # to skip sanity check
            grid_size = (3, 3)
            progress = self.trainer.progress_bar_callback.progress
            gen_samples = self.generate_samples(grid_size[0] * grid_size[1], progress)
            self.logger.experiment.add_histogram('result dist', gen_samples, global_step=self.current_epoch)
            images = self.generated_samples_to_images(gen_samples, grid_size)
            self.logger.experiment.add_image('generated images', images, self.current_epoch, dataformats="CHW")

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

    def generate_samples(self, num_images, progress):
        progress.generating_progress_bar_id = progress.add_task("[white]Generating images...",
                                                                total=self.config.diffusion_steps-1)
        # Generate samples from denoising process
        x = torch.randn((num_images, 3, self.config.image_size, self.config.image_size), device=self.device)
        sample_steps = torch.arange(self.config.diffusion_steps - 1, 0, -1, device=self.device)
        for t in sample_steps:
            progress.update(progress.generating_progress_bar_id, advance=1, visible=True)
            progress.refresh()
            x = self.denoise_sample(x, t)
        progress.update(progress.generating_progress_bar_id, comleted=0, visible=False)
        return x

    @staticmethod
    def generated_samples_to_images(gen_samples, grid_size=(4, 4)):
        assert gen_samples.shape[0] == grid_size[0] * grid_size[1]

        gen_samples = (gen_samples.clamp(-1, 1) + 1) / 2
        gen_samples = (gen_samples * 255).type(torch.uint8)

        # stack images
        gen_samples = torch.cat(torch.split(gen_samples, grid_size[0], dim=0), dim=2)
        gen_samples = torch.cat(torch.split(gen_samples, 1, dim=0), dim=3)
        gen_samples = gen_samples.squeeze().cpu().numpy()

        return gen_samples
