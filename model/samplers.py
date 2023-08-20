import torch
import numpy as np
from time import time
from typing import Tuple
from scipy import integrate
import matplotlib.pyplot as plt
from rich.progress import Progress
from abc import ABC, abstractmethod
from lightning import LightningModule

from config import DDPMSamplingConfig, PCSamplingConfig, ODESamplingConfig
from SDE_sampling.sde_lib import get_sde, VPSDE, subVPSDE, VESDE
from SDE_sampling.predictors_correctors import get_predictor, get_corrector, Predictor, Corrector


class Sampler(ABC):
    def __init__(self, config):
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


class DDPM_Sampler(Sampler):
    def __init__(self, config: DDPMSamplingConfig):
        assert isinstance(config, DDPMSamplingConfig)
        super().__init__(config)
        self.beta_min = config.beta_min
        self.beta_max = config.beta_max
        self.diffusion_steps = config.diffusion_steps

        self.beta = torch.linspace(self.beta_min, self.beta_max, self.diffusion_steps)
        self.alpha = torch.cumprod(1 - self.beta, dim=0)

    def init_timesteps(self):
        return torch.arange(self.diffusion_steps - 1, 0, -1, device=self.device)

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.diffusion_steps, size=(n,), device=self.device)

    def prior_sampling(self, shape):
        return torch.randn(shape, device=self.device)

    def noise_images(self, images, t):
        sqrt_alpha = torch.sqrt(self.alpha[t]).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = torch.sqrt(1 - self.alpha[t]).view(-1, 1, 1, 1)
        noise = torch.randn_like(images, device=self.device)
        noised_image = sqrt_alpha * images + sqrt_one_minus_alpha * noise
        return noised_image, noise

    def noise_images_one_step(self, prev_step_images, t):
        beta = self.beta[t].view(-1, 1, 1, 1)
        noise = torch.randn_like(prev_step_images, device=self.device)
        noised_image = torch.sqrt(1 - beta) * prev_step_images + torch.sqrt(beta) * noise
        return noised_image, noise

    def denoise_step(self, model: LightningModule, batch, t):
        images = batch[0]
        predicted_noise = model(batch, t.repeat(images.shape[0]))

        if t > 1:
            noise = torch.randn_like(images)
        else:
            noise = torch.zeros_like(images)

        alpha = self.alpha[t].view(-1, 1, 1, 1)
        beta = self.beta[t].view(-1, 1, 1, 1)

        denoised_image = 1 / torch.sqrt(1 - beta) * (images - (beta / (torch.sqrt(1 - alpha)))
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


class PC_Sampler(Sampler):
    def __init__(self, config: PCSamplingConfig):
        assert isinstance(config, PCSamplingConfig)
        super().__init__(config)

        self.num_corrector_steps = config.num_corrector_steps
        self.probability_flow = config.probability_flow
        self.denoise = config.denoise
        self.snr = config.snr

        self.sde = get_sde(config)
        self.predictor_name = config.predictor_name
        self.corrector_name = config.corrector_name

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

        predictor, corrector = self._get_pc(model)

        x, x_mean = predictor.update(batch, t.repeat(x.shape[0]))
        for _ in range(self.num_corrector_steps):
            x, x_mean = corrector.update((x, *labels), t.repeat(x.shape[0]))
        out = x_mean if self.denoise else x
        return out

    def _get_pc(self, model) -> (Predictor, Corrector):
        score_fn = get_score_fn(self.sde, model)

        predictor = get_predictor(self.predictor_name, self.sde, score_fn, self.probability_flow)
        corrector = get_corrector(self.corrector_name, self.sde, score_fn, self.snr)
        return predictor, corrector


class ODE_Sampler(Sampler):
    def __init__(self, config: ODESamplingConfig):
        assert isinstance(config, ODESamplingConfig)
        super().__init__(config)
        self.sde = get_sde(config)

        self.method = config.method
        self.rtol = config.rtol
        self.atol = config.atol
        self.denoise = config.denoise

    def init_timesteps(self):
        raise NotImplementedError

    def sample_timesteps(self, n):
        raise NotImplementedError

    def noise_images(self, images, t):
        raise NotImplementedError

    def denoise_step(self, model, batch, t):
        raise NotImplementedError

    def prior_sampling(self, shape):
        return self.sde.prior_sampling(shape).to(self.device)

    def generate_samples(self, model, batch, progress=None, seed=42):
        x = batch[0]
        shape = x.shape

        # Initial sample
        x = self.prior_sampling(shape)
        x = x.detach().cpu().numpy().reshape((-1,))

        score_fn = get_score_fn(self.sde, model)

        ode_func = self.get_ode_func(score_fn, shape)

        # Black-box ODE solver for the probability flow ODE
        print('Generating samples with ODE Solver. It will take some time.')
        ttime = time()
        solution = integrate.solve_ivp(ode_func, (self.sde.T, self.sde.eps), x,
                                       rtol=self.rtol, atol=self.atol, method=self.method)
        x = torch.tensor(solution.y[:, -1]).reshape(shape).to(self.device).type(torch.float32)
        print(f'Generating samples with ODE Solver took {time() - ttime} seconds.')

        # Denoising is equivalent to running one predictor step without adding noise
        if self.denoise:
            x = self.denoise_update(x, score_fn)
        return x

    def get_ode_func(self, score_fn, shape):
        def ode_func(t, x):
            x = torch.from_numpy(x.reshape(shape)).to(self.device).type(torch.float32)
            vec_t = torch.ones(shape[0], device=self.device) * t
            rsde = self.sde.reverse(score_fn, probability_flow=True)
            drift = rsde.sde((x, ), vec_t)[0]
            return drift.detach().cpu().numpy().reshape((-1,))
        return ode_func

    def denoise_update(self, batch, score_fn):
        x = batch[0]
        # Reverse diffusion predictor for denoising
        predictor = get_predictor('reverse_diffusion', self.sde, score_fn, probability_flow=True)
        vec_eps = torch.ones(x.shape[0], device=self.device) * self.sde.eps
        _, x = predictor.update(batch, vec_eps)
        return x


def get_score_fn(sde, model):
    continuous = isinstance(model.config.sampler_config, PCSamplingConfig)
    if (isinstance(sde, VPSDE) and continuous) or isinstance(sde, subVPSDE):
        def score_fn(batch, t):
            # For VP-trained models, t=0 corresponds to the lowest noise level
            # The maximum value of time embedding is assumed to 999 for
            # continuously-trained models.
            images = batch[0]
            ts = t * 999
            score = model(batch, ts)
            std = sde.marginal_prob(torch.zeros_like(images), t)[1]
            score = -score / std[:, None, None, None]
            return score

    elif isinstance(sde, VPSDE) and not continuous:
        def score_fn(batch, t):
            # For VP-trained models, t=0 corresponds to the lowest noise level
            ts = t * (sde.N - 1)
            score = model(batch, ts)
            std = sde.sqrt_1m_alphas_cumprod.to(ts.device)[ts.long()]
            score = -score / std[:, None, None, None]
            return score

    elif isinstance(sde, VESDE) and continuous:
        def score_fn(x, t):
            labels = sde.marginal_prob(torch.zeros_like(x), t)[1]
            score = model(x, labels)
            return score

    elif isinstance(sde, VESDE) and not continuous:
        def score_fn(x, t):
            # For VE-trained models, t=0 corresponds to the highest noise level
            labels = sde.T - t
            labels *= sde.N - 1
            labels = torch.round(labels).long()
            score = model(x, labels)
            return score

    else:
        raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

    return score_fn
