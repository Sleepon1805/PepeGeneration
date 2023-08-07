import abc
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from scipy import integrate
import matplotlib.pyplot as plt
from typing import Tuple

from config import Config
from evaluate import load_model_and_config
from SDE_sampling.sde_lib import get_sde, VPSDE, subVPSDE, VESDE
from SDE_sampling.predictors_correctors import PREDICTORS, CORRECTORS, ReverseDiffusionPredictor


class Sampler(abc.ABC):
    def __init__(self, model, config: Config):
        self.model = model
        self.device = model.device
        self.config = config

    @abc.abstractmethod
    def sample(self, num_samples) -> torch.Tensor:
        pass

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
    def __init__(self, model, config: Config, sde_name: str, predictor_name: str, corrector_name: str, snr: float,
                 n_steps=1, probability_flow=False, denoise=True):
        super().__init__(model, config)
        self.n_steps = n_steps
        self.probability_flow = probability_flow
        self.denoise = denoise
        self.snr = snr

        self.sde, self.eps = get_sde(sde_name, self.config)
        self.predictor, self.corrector = self.get_pc(predictor_name, corrector_name)

    def sample(self, num_samples):
        shape = (num_samples, 3, self.config.image_size, self.config.image_size)

        # Initial sample
        x = self.sde.prior_sampling(shape).to(self.device)
        timesteps = torch.linspace(self.sde.T, self.eps, self.sde.N, device=self.device)

        for i in tqdm(range(self.sde.N)):
            t = timesteps[i]
            vec_t = torch.ones(shape[0], device=self.device) * t
            x, x_mean = self.predictor.update_fn(x, vec_t)
            x, x_mean = self.corrector.update_fn(x, vec_t)

        x = x_mean if self.denoise else x
        return x

    def get_pc(self, predictor_name, corrector_name):
        score_fn = get_score_fn(self.sde, self.model, continuous=self.config.continuous_training)

        predictor = PREDICTORS[predictor_name](self.sde, score_fn, self.probability_flow)
        corrector = CORRECTORS[corrector_name](self.sde, score_fn, self.snr, self.n_steps)
        return predictor, corrector


class ODE_Sampler(Sampler):
    def __init__(self, model, config, sde_name, method='RK45', rtol=1e-5, atol=1e-5, denoise=False):
        super().__init__(model, config)
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.denoise = denoise

        self.sde, self.eps = get_sde(sde_name, self.config)
        self.score_fn = get_score_fn(self.sde, self.model, continuous=True)
        self.predictor = ReverseDiffusionPredictor(self.sde, self.score_fn, probability_flow=False)

    def sample(self, num_samples, z=None):
        shape = (num_samples, 3, self.config.image_size, self.config.image_size)

        # Initial sample
        if z is None:
            # If not represent, sample the latent code from the prior distibution of the SDE.
            x = self.sde.prior_sampling(shape).to(self.device)
        else:
            x = z
        x = x.detach().cpu().numpy().reshape((-1,))

        ode_func = self.get_ode_func(shape)

        # Black-box ODE solver for the probability flow ODE
        solution = integrate.solve_ivp(ode_func, (self.sde.T, self.eps), x,
                                       rtol=self.rtol, atol=self.atol, method=self.method)
        n = solution.nfev
        x = torch.tensor(solution.y[:, -1]).reshape(shape).to(self.device).type(torch.float32)

        # Denoising is equivalent to running one predictor step without adding noise
        if self.denoise:
            x = self.denoise_update_fn(x)

        return x, n

    def get_ode_func(self, shape):
        def ode_func(x, t):
            x = torch.from_numpy(x.reshape(shape)).to(self.device).type(torch.float32)
            vec_t = torch.ones(shape[0], device=self.device) * t
            rsde = self.sde.reverse(self.score_fn, probability_flow=True)
            drift = rsde.sde(x, vec_t)[0]
            return drift.detach().cpu().numpy().reshape((-1,))
        return ode_func

    def denoise_update_fn(self, x):
        # Reverse diffusion predictor for denoising
        vec_eps = torch.ones(x.shape[0], device=x.device) * self.eps
        _, x = self.predictor.update_fn(x, vec_eps)
        return x


def get_score_fn(sde, model, continuous=False):
    if isinstance(sde, VPSDE) or isinstance(sde, subVPSDE):
        def score_fn(x, t):
            # Scale neural network output by standard deviation and flip sign
            if continuous or isinstance(sde, subVPSDE):
                # For VP-trained models, t=0 corresponds to the lowest noise level
                # The maximum value of time embedding is assumed to 999 for
                # continuously-trained models.
                labels = t * 999
                score = model(x, labels)
                std = sde.marginal_prob(torch.zeros_like(x), t)[1]
            else:
                # For VP-trained models, t=0 corresponds to the lowest noise level
                labels = t * (sde.N - 1)
                score = model(x, labels)
                std = sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()]

            score = -score / std[:, None, None, None]
            return score

    elif isinstance(sde, VESDE):
        def score_fn(x, t):
            if continuous:
                labels = sde.marginal_prob(torch.zeros_like(x), t)[1]
            else:
                # For VE-trained models, t=0 corresponds to the highest noise level
                labels = sde.T - t
                labels *= sde.N - 1
                labels = torch.round(labels).long()

            score = model(x, labels)
            return score

    else:
        raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

    return score_fn


def inference(ckpt_path: Path, grid_shape=(4, 4), on_gpu=True):
    device = 'cuda' if (on_gpu and torch.cuda.is_available()) else 'cpu'
    model, config = load_model_and_config(ckpt_path, device)

    num_samples = grid_shape[0] * grid_shape[1]
    sde_name = 'vpsde'

    sampler = PC_Sampler(model, config, sde_name,
                         predictor_name='reverse_diffusion',
                         corrector_name='langevin',
                         snr=0.16,
                         n_steps=1,
                         probability_flow=False,
                         denoise=False)
    # sampler = ODE_Sampler(model, config, sde_name, denoise=True)

    gen_samples = sampler.sample(num_samples)

    gen_images = sampler.generated_samples_to_images(gen_samples, grid_shape)
    plt.imshow(gen_images)
    plt.show()


if __name__ == '__main__':
    dataset_name = 'celeba'
    version = 10
    ckpt = Path(f'../lightning_logs/{dataset_name}/version_{version}/checkpoints/last.ckpt')

    inference(ckpt, grid_shape=(4, 4), on_gpu=True)
