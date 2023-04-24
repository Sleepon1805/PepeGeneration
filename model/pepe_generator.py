import pytorch_lightning as pl
import torch

from model.unet import UNetModel
from model.diffusion import Diffusion


class PepeGenerator(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.diffusion = Diffusion(config)
        self.model = UNetModel(config)

        self.loss_func = torch.nn.MSELoss()

        self.example_input_array = torch.Tensor(config.batch_size, 3, config.image_size, config.image_size), \
            torch.ones(config.batch_size)
        self.save_hyperparameters()

    def forward(self, x, t):
        return self.model(x, t)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.init_lr)
        return optimizer
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15], gamma=0.1)
        # return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss'}}

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch)
        self.log("val_loss", loss)
        return

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
