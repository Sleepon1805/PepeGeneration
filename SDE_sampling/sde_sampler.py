import torch
from typing import Tuple, Callable
from lightning import LightningModule

from config import PCSamplerConfig
from utils.typings import TrainImagesType, BatchType, BatchedFloatType
from model.sampler import Sampler
from SDE_sampling.sde_lib import get_sde
from SDE_sampling.predictors_correctors import get_predictor, get_corrector


class PC_Sampler(Sampler):
    def __init__(self, model: LightningModule, config: PCSamplerConfig):
        super().__init__()
        self.model = model
        self.sde = get_sde(
            config.sde_config.sde_name,
            config.sde_config.schedule_param_start,
            config.sde_config.schedule_param_end,
            config.sde_config.num_scales
        )
        self.score_fn = self.get_score_fn()

        self.predictor = get_predictor(
            config.pc_config.predictor_name,
            self.sde,
            self.score_fn,
            config.pc_config.probability_flow
        )
        self.corrector = get_corrector(
            config.pc_config.corrector_name,
            self.sde,
            self.score_fn,
            config.pc_config.snr
        )

        self.denoise = config.denoise
        self.num_corrector_steps = config.num_corrector_steps

    def get_score_fn(self) -> Callable:
        # Orig: https://github.com/yang-song/score_sde_pytorch/blob/cb1f359f4aadf0ff9a5e122fe8fffc9451fd6e44/models/utils.py#L129
        def score_fn(batch, t):
            std = self.sde.marginal_prob(torch.zeros_like(batch[0]), t)[1]
            score = self.model(batch, t)
            score = -score / std[:, None, None, None]
            return score
        return score_fn

    def init_timesteps(self) -> BatchedFloatType:
        return torch.linspace(self.sde.T, self.sde.eps, self.sde.N, device=self.device)

    def sample_timesteps(self, n: int) -> BatchedFloatType:
        return torch.rand(n, device=self.device) * (self.sde.T - self.sde.eps) + self.sde.eps

    def prior_sampling(self, shape: Tuple) -> TrainImagesType:
        return self.sde.prior_sampling(shape).to(self.device)

    def noise_images(self, images: TrainImagesType, t: BatchedFloatType) -> Tuple[TrainImagesType, TrainImagesType]:
        epsilon = torch.randn_like(images)
        mean, std = self.sde.marginal_prob(images, t)
        noised_images = mean + std[:, None, None, None] * epsilon
        return noised_images, epsilon

    def denoise_step(self, model: LightningModule, batch: BatchType, t: BatchedFloatType) -> TrainImagesType:
        x, *labels = batch

        x, x_mean = self.predictor.update(batch, t.repeat(x.shape[0]))
        for _ in range(self.num_corrector_steps):
            x, x_mean = self.corrector.update((x, *labels), t.repeat(x.shape[0]))
        out = x_mean if self.denoise else x
        return out
