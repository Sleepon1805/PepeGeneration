import torch
from pathlib import Path
from scipy import integrate
import matplotlib.pyplot as plt

from config import Config
from model.diffusion_sampler import Sampler
from SDE_sampling.sde_lib import get_sde, VPSDE, subVPSDE, VESDE
from SDE_sampling.predictors_correctors import PREDICTORS, CORRECTORS, ReverseDiffusionPredictor, Predictor, Corrector


class PC_Sampler(Sampler):
    def __init__(self, config: Config):
        super().__init__(config)
        # assert not self.config.use_condition, 'Conditional prediction with SDE sampler is not implemented yet'

        self.n_steps = config.num_corrector_steps
        self.probability_flow = config.probability_flow
        self.denoise = config.denoise
        self.snr = config.snr

        self.sde = get_sde(config.sde_name, self.config)
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

        x, x_mean = predictor.update_fn(batch, t.repeat(x.shape[0]))
        x, x_mean = corrector.update_fn((x, *labels), t.repeat(x.shape[0]))
        out = x_mean if self.denoise else x
        return out

    def _get_pc(self, model) -> (Predictor, Corrector):
        score_fn = get_score_fn(self.sde, model, continuous=self.config.sde_training)

        predictor = PREDICTORS[self.predictor_name](self.sde, score_fn, self.probability_flow)
        corrector = CORRECTORS[self.corrector_name](self.sde, score_fn, self.snr, self.n_steps)
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
