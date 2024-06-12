import torch
import numpy as np
from typing import Tuple, Callable
from abc import ABC, abstractmethod
from lightning import LightningModule

from config import Config, progress_bar
from SDE_sampling.sde_lib import get_sde
from SDE_sampling.predictors_correctors import get_predictor, get_corrector


class Sampler(ABC):
    def __init__(self):
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

    def generate_samples(self, model, batch, seed=42) -> torch.Tensor:
        torch.manual_seed(seed)
        images_batch, *labels = batch

        # move to device
        for i in range(len(labels)):
            labels[i] = labels[i].to(self.device)

        # init
        x = self.prior_sampling(images_batch.shape)
        timesteps = self.init_timesteps()

        # Generate samples from denoising process
        for t in progress_bar(timesteps, desc=f"Generating {images_batch.shape[0]} images"):
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


class PC_Sampler(Sampler):
    def __init__(self, model: LightningModule, config: Config):
        super().__init__()
        self.model = model
        self.sde = get_sde(
            config.sde_name,
            config.schedule_param_start,
            config.schedule_param_end,
            config.num_scales
        )
        self.score_fn = self.get_score_fn()

        self.predictor = get_predictor(
            config.predictor_name,
            self.sde,
            self.score_fn,
            config.probability_flow
        )
        self.corrector = get_corrector(
            config.corrector_name,
            self.sde,
            self.score_fn,
            config.snr
        )

        self.num_corrector_steps = config.num_corrector_steps
        self.denoise = config.denoise

    def get_score_fn(self) -> Callable:
        """
        Orig:
        https://github.com/yang-song/score_sde_pytorch/blob/cb1f359f4aadf0ff9a5e122fe8fffc9451fd6e44/models/utils.py#L129
        """
        def score_fn(batch, t):
            std = self.sde.marginal_prob(torch.zeros_like(batch[0]), t)[1]
            score = self.model(batch, t)
            score = -score / std[:, None, None, None]
            return score
        return score_fn

    def init_timesteps(self):
        return torch.linspace(self.sde.T, self.sde.eps, self.sde.N, device=self.device)

    def sample_timesteps(self, n):
        return torch.rand(n, device=self.device) * (self.sde.T - self.sde.eps) + self.sde.eps

    def prior_sampling(self, shape):
        return self.sde.prior_sampling(shape).to(self.device)

    def noise_images(self, images, t):
        epsilon = torch.randn_like(images)
        mean, std = self.sde.marginal_prob(images, t)
        noised_images = mean + std[:, None, None, None] * epsilon
        return noised_images, epsilon

    def denoise_step(self, model, batch, t):
        x, *labels = batch

        x, x_mean = self.predictor.update(batch, t.repeat(x.shape[0]))
        for _ in range(self.num_corrector_steps):
            x, x_mean = self.corrector.update((x, *labels), t.repeat(x.shape[0]))
        out = x_mean if self.denoise else x
        return out
