import torch
from scipy import integrate

from config import Config, SDE_Config
from model.diffusion_sampler import Sampler
from SDE_sampling.sde_lib import get_sde, VPSDE, subVPSDE, VESDE
from SDE_sampling.predictors_correctors import PREDICTORS, CORRECTORS, ReverseDiffusionPredictor, Predictor, Corrector


class PC_Sampler(Sampler):
    def __init__(self, config: Config, sde_config: SDE_Config):
        super().__init__(config)
        self.sde_config = sde_config

        self.n_steps = sde_config.num_corrector_steps
        self.probability_flow = sde_config.probability_flow
        self.denoise = sde_config.denoise
        self.snr = sde_config.snr

        self.sde = get_sde(sde_config.sde_name, sde_config)
        self.predictor_name = sde_config.predictor_name
        self.corrector_name = sde_config.corrector_name

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
    def __init__(self, config, sde_config, method='RK45', rtol=1e-5, atol=1e-5):
        super().__init__(config)
        self.sde_config = sde_config
        self.sde = get_sde(sde_config.sde_name, sde_config)

        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.denoise = sde_config.denoise

    def init_timesteps(self):
        pass

    def sample_timesteps(self, n):
        pass

    def prior_sampling(self, shape):
        return self.sde.prior_sampling(shape).to(self.device)

    def noise_images(self, images, t):
        pass

    def denoise_step(self, model, batch, t):
        pass

    def generate_samples(self, model, batch, progress=None, seed=42):
        x = batch[0]
        shape = x.shape

        # Initial sample
        x = self.prior_sampling(shape)
        x = x.detach().cpu().numpy().reshape((-1,))

        score_fn = get_score_fn(self.sde, model, continuous=self.config.sde_training)

        ode_func = self.get_ode_func(score_fn, shape)

        # Black-box ODE solver for the probability flow ODE
        solution = integrate.solve_ivp(ode_func, (self.sde.T, self.sde.eps), x,
                                       rtol=self.rtol, atol=self.atol, method=self.method)
        x = torch.tensor(solution.y[:, -1]).reshape(shape).to(self.device).type(torch.float32)

        # Denoising is equivalent to running one predictor step without adding noise
        if self.denoise:
            x = self.denoise_update_fn(x, score_fn)

        return x

    def get_ode_func(self, score_fn, shape):
        def ode_func(t, x):
            x = torch.from_numpy(x.reshape(shape)).to(self.device).type(torch.float32)
            vec_t = torch.ones(shape[0], device=self.device) * t
            rsde = self.sde.reverse(score_fn, probability_flow=True)
            drift = rsde.sde((x, ), vec_t)[0]
            return drift.detach().cpu().numpy().reshape((-1,))
        return ode_func

    def denoise_update_fn(self, x, score_fn):
        # Reverse diffusion predictor for denoising
        predictor = ReverseDiffusionPredictor(self.sde, score_fn, probability_flow=True)
        vec_eps = torch.ones(x.shape[0], device=x.device) * self.sde.eps
        _, x = predictor.update_fn((x, ), vec_eps)
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
