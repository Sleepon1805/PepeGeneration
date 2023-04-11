import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from tqdm import tqdm

from model.unet import UNet


class PepeGenerator(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.example_input_array = torch.Tensor(self.cfg.batch_size, 3, *self.cfg.image_size)
        self.t_range = self.cfg.diffusion_steps
        self.in_size = self.cfg.image_size[0]

        # gaussian noise constants
        self.beta = torch.linspace(1e-4, 0.02, self.t_range)
        self.alpha_hat = torch.cumprod(1 - self.beta, dim=0)

        # model
        self.model = UNet(c_in=3, c_out=3, init_channels=self.cfg.init_channels, time_dim=self.cfg.time_channels)

        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        return optimizer

    # def on_train_start(self):
    #     # log model graph FIXME
    #     self.logger.log_graph(self, self.example_input_array)

    def forward(self, x, t=None):
        if t is None:
            t = torch.zeros(x.shape[0])  # for sanity check
        return self.model(x, t)

    def noise_images(self, x, t):
        alpha_hats = self.alpha_hat[t].view(-1, 1, 1, 1).to(self.device)
        noise = torch.randn_like(x)
        noised_x = torch.sqrt(alpha_hats) * x + torch.sqrt(1 - alpha_hats) * noise
        return noised_x, noise

    def get_loss(self, batch):
        sampled_ts = torch.randint(1, self.t_range, size=[batch.shape[0]])
        x = 2 * batch - 1
        x_t, noise = self.noise_images(x, sampled_ts)
        predicted_noise = self.forward(x_t, sampled_ts)
        loss = F.mse_loss(predicted_noise, noise)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        self.log("val_loss", loss)
        return

    def sample(self, num_images):
        with torch.no_grad():
            x = torch.randn((num_images, 3, *self.cfg.image_size))
            for i in tqdm(range(self.t_range - 1, 0, -1)):
                t = torch.ones(num_images, dtype=torch.int) * i
                predicted_noise = self.forward(x, t)
                alpha = 1 - self.beta[t].view(-1, 1, 1, 1)
                alpha_hat = self.alpha_hat[t].view(-1, 1, 1, 1)
                beta = self.beta[t].view(-1, 1, 1, 1)
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - (beta / (torch.sqrt(1 - alpha_hat))) * predicted_noise) \
                    + torch.sqrt(beta) * noise
        images = (torch.clip(x, -1, 1) + 1) / 2
        return images
