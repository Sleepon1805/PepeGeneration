import torch
import numpy as np
from typing import Tuple
import torch.nn.functional as F
from rich.progress import Progress
from torchmetrics.image.fid import FrechetInceptionDistance

from model.unet import UNetModel
from model.pepe_generator import PepeGenerator
from config import Config, HIGHRES_IMAGE_SIZE_MULT


class HighResUNetModel(UNetModel):
    """
    A UNetModel that performs super-resolution.

    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, config: Config):
        super().__init__(config, in_channels=6)

    def forward(self, x, timesteps, low_res=None, cond=None):
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        x = torch.cat([x, upsampled], dim=1)
        out = super().forward(x, timesteps, cond)
        return out


class HighResPepeGenerator(PepeGenerator):
    def __init__(self, config: Config):
        super().__init__(config)
        config.res_factor = HIGHRES_IMAGE_SIZE_MULT  # to save with config
        self.model = HighResUNetModel(config)
        self.example_input_array = (
            torch.Tensor(config.batch_size, 3,
                         config.image_size * HIGHRES_IMAGE_SIZE_MULT, config.image_size * HIGHRES_IMAGE_SIZE_MULT),
            torch.ones(config.batch_size),
            torch.Tensor(config.batch_size, 3, config.image_size, config.image_size),
            torch.ones(config.batch_size, config.condition_size),
        )

    def forward(self, x, t, low_res=None, cond=None):
        return self.model(x, t, low_res=low_res, cond=cond)

    def _calculate_loss(self, batch):
        highres_image_batch, lowres_image_batch, cond_batch = batch
        ts = self.diffusion.sample_timesteps(highres_image_batch.shape[0])
        noised_batch, noise = self.diffusion.noise_images(highres_image_batch, ts)
        output = self.forward(noised_batch, ts, lowres_image_batch, cond_batch)
        loss = self.loss_func(output, noise)
        return loss

    """
    Methods for inference and evaluation
    """

    def denoise_samples(self, x, t, low_res=None, cond=None):
        predicted_noise = self.forward(x, t.repeat(x.shape[0]), low_res, cond)
        x = self.diffusion.denoise_images(x, t, predicted_noise)
        return x

    def generate_samples(self, batch, progress: Progress = None, seed=42) -> torch.Tensor:
        torch.manual_seed(seed)
        highres_images_batch, lowres_image_batch, cond_batch = batch

        if progress is not None:
            progress.generating_progress_bar_id = progress.add_task(
                f"[white]Generating {highres_images_batch.shape[0]} images",
                total=self.config.diffusion_steps-1
            )

        # Generate samples from denoising process
        x = torch.randn_like(highres_images_batch, device=self.device)
        sample_steps = torch.arange(self.config.diffusion_steps - 1, 0, -1, device=self.device)
        for t in sample_steps:
            if progress is not None:
                progress.update(progress.generating_progress_bar_id, advance=1, visible=True)
                progress.refresh()
            x = self.denoise_samples(x, t, lowres_image_batch, cond_batch)
        return x
