import torch
from lightning import LightningModule
from abc import ABC, abstractmethod
from rich.progress import Progress
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt

from config import Config


class Sampler(ABC):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.device = 'cpu'

    def to(self, device):
        self.device = device
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, torch.Tensor):
                self.__setattr__(attr_name, attr_value.to(device))

    @abstractmethod
    def init_timesteps(self):
        pass

    @abstractmethod
    def sample_timesteps(self, n):
        pass

    def prior_sampling(self, shape):
        pass

    @abstractmethod
    def noise_images(self, images, t):
        pass

    @abstractmethod
    def denoise_step(self, model: LightningModule, batch, t):
        pass

    def generate_samples(self, model, batch, progress: Progress = None, seed=42) -> torch.Tensor:
        torch.manual_seed(seed)
        images_batch, *labels = batch

        # move to device
        for i in range(len(labels)):
            labels[i] = labels[i].to(self.device)

        # init
        x = self.prior_sampling(images_batch.shape)
        timesteps = self.init_timesteps()

        if progress is not None:
            progress.generating_progress_bar_id = progress.add_task(
                f"[white]Generating {images_batch.shape[0]} images",
                total=len(timesteps)
            )

        # Generate samples from denoising process
        for t in timesteps:
            if progress is not None:
                progress.update(progress.generating_progress_bar_id, advance=1, visible=True)
                progress.refresh()
            batch = (x, *labels)
            x = self.denoise_step(model, batch, t)
        return x

    @staticmethod
    def generated_samples_to_images(gen_samples: torch.Tensor, grid_size: Tuple[int, int]) -> np.ndarray:
        gen_samples = gen_samples[:grid_size[0] * grid_size[1]]

        gen_samples = (gen_samples.clamp(-1, 1) + 1) / 2
        gen_samples = (gen_samples * 255).type(torch.uint8)

        # stack images
        gen_samples = torch.cat(torch.split(gen_samples, grid_size[0], dim=0), dim=2)
        gen_samples = torch.cat(torch.split(gen_samples, 1, dim=0), dim=3)
        gen_samples = gen_samples.squeeze().cpu().numpy()
        gen_samples = np.moveaxis(gen_samples, 0, -1)

        return gen_samples


class DDPM_Diffusion(Sampler):
    def __init__(self, config: Config):
        super().__init__(config)
        self.beta_min = config.beta_min
        self.beta_max = config.beta_max
        self.diffusion_steps = config.diffusion_steps

        self.beta = torch.linspace(self.beta_min, self.beta_max, self.diffusion_steps)
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def init_timesteps(self):
        return torch.arange(self.config.diffusion_steps - 1, 0, -1, device=self.device)

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.diffusion_steps, size=(n,), device=self.device)

    def prior_sampling(self, shape):
        return torch.randn(shape, device=self.device)

    def noise_images(self, images, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t]).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t]).view(-1, 1, 1, 1)
        epsilon = torch.randn_like(images, device=self.device)
        noised_image = sqrt_alpha_hat * images + sqrt_one_minus_alpha_hat * epsilon
        return noised_image, epsilon

    def noise_images_one_step(self, prev_step_images, t):
        beta = self.beta[t].view(-1, 1, 1, 1)
        epsilon = torch.randn_like(prev_step_images, device=self.device)
        noised_image = torch.sqrt(1 - beta) * prev_step_images + torch.sqrt(beta) * epsilon
        return noised_image, epsilon

    def denoise_step(self, model: LightningModule, batch, t):
        images = batch[0]
        predicted_noise = model(batch, t.repeat(images.shape[0]))

        if t > 1:
            noise = torch.randn_like(images)
        else:
            noise = torch.zeros_like(images)

        alpha = self.alpha[t].view(-1, 1, 1, 1)
        alpha_hat = self.alpha_hat[t].view(-1, 1, 1, 1)
        beta = self.beta[t].view(-1, 1, 1, 1)

        denoised_image = 1 / torch.sqrt(alpha) * (images - ((1 - alpha) / (torch.sqrt(1 - alpha_hat)))
                                                  * predicted_noise) + torch.sqrt(beta) * noise
        return denoised_image

    def visualize_generation_process(self, model, batch, progress: Progress = None, seed=42):
        torch.manual_seed(seed)
        images_batch, *labels = batch

        # move to device
        images_batch = images_batch.to(self.device)
        for i in range(len(labels)):
            labels[i] = labels[i].to(self.device)

        # init
        timesteps = self.init_timesteps()

        if progress is not None:
            progress.generating_progress_bar_id = progress.add_task(
                f"[white]Generating {images_batch.shape[0]} images",
                total=len(timesteps)
            )

        noising_images = []
        x = images_batch
        # Generate samples from denoising process
        for t in torch.flip(timesteps, dims=(0,)):
            x, noise = self.noise_images_one_step(x, t)
            noising_images.append(x)
        noising_images = noising_images[::100] + [noising_images[-1]]

        denoising_images = []
        # Generate samples from denoising process
        for t in timesteps:
            if progress is not None:
                progress.update(progress.generating_progress_bar_id, advance=1, visible=True)
                progress.refresh()
            x = self.denoise_step(model, (x, *labels), t)
            denoising_images.append(x)
        denoising_images = denoising_images[::100] + [denoising_images[-1]]

        plt.imshow(self.generated_samples_to_images(torch.concat(noising_images, dim=0), (11, 1)))
        plt.axis('off')
        plt.show()

        plt.imshow(self.generated_samples_to_images(torch.concat(denoising_images[::-1], dim=0), (11, 1)))
        plt.axis('off')
        plt.show()
