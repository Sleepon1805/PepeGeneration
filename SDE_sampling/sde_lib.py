""" Abstract SDE classes, Reverse SDE, and VE/VP SDEs. """
import torch
from enum import Enum
from typing import Callable, Tuple
from abc import abstractmethod

from utils.typings import TrainImagesType, BatchType, BatchedFloatType, ABCTypeChecked


class SDE(ABCTypeChecked):
    """ SDE abstract class. Functions are designed for a mini-batch of inputs. """

    def __init__(self, N: int, T: float, eps: float):
        """
        Construct an SDE.

        Args:
            N: number of discretization time steps.
            T: time horizon.
            eps: start time > 0 for numerical stability.
        """
        self.N = N
        self.T = T
        self.eps = eps

    @abstractmethod
    def get_sde(self, batch: BatchType, t: BatchedFloatType) -> Tuple[TrainImagesType, TrainImagesType]:
        """
        Get current drift and diffusion for X and t (vectorized).
        :param batch:
        :param t:
        :return:
        """
        pass

    @abstractmethod
    def marginal_prob(self, x_0: TrainImagesType, t: BatchedFloatType) -> Tuple[TrainImagesType, TrainImagesType]:
        """
        Noise image(s) x_0 to timestep t: $p_t(x)$
        :param x_0: Images
        :param t: timesteps
        :return: mean and std of the marginal distribution
        """
        # TODO: use Eq. (5.50), (5.51) in Särkkä & Solin (2019): Applied Stochastic Differential Equations
        pass

    @abstractmethod
    def prior_sampling(self, shape: Tuple) -> TrainImagesType:
        """
        Generate one sample from the prior distribution: $p_T(X)$ (which should not depend on X).
        :param shape: shape of the sample
        :return: sample from the prior distribution
        """
        pass

    def discretize(self, batch: BatchType, t: BatchedFloatType) -> Tuple[TrainImagesType, TrainImagesType]:
        """
        Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.
        Useful for reverse diffusion sampling and probabiliy flow sampling.
        Defaults to Euler-Maruyama discretization.

        Args:
            batch: input batch
            t: a torch float representing the time step (from 0 to `self.T`)
        Returns:
            f, G
        """
        dt = 1 / self.N
        drift, diffusion = self.get_sde(batch, t)
        f = drift * dt
        G = diffusion * (dt ** 0.5)
        return f, G


class ReverseSDE(SDE):
    def __init__(self, sde: SDE, score_fn: Callable, probability_flow=False):
        super().__init__(sde.N, sde.T, sde.eps)
        self.forward_sde = sde
        self.score_fn = score_fn
        self.probability_flow = probability_flow

    def get_sde(self, batch: BatchType, t: BatchedFloatType) -> Tuple[TrainImagesType, TrainImagesType]:
        drift, diffusion = self.forward_sde.get_sde(batch, t)
        score = self.score_fn(batch, t)
        drift = drift - diffusion[:, None, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
        # Set the diffusion function to zero for ODEs.
        diffusion *= (1 - self.probability_flow)
        return drift, diffusion

    def marginal_prob(self, x_0: TrainImagesType, t: BatchedFloatType) -> Tuple[TrainImagesType, TrainImagesType]:
        return self.forward_sde.marginal_prob(x_0, t)

    def prior_sampling(self, shape: Tuple) -> TrainImagesType:
        return self.forward_sde.prior_sampling(shape)

    def discretize(self, batch: BatchType, t: BatchedFloatType) -> Tuple[TrainImagesType, TrainImagesType]:
        f, G = self.forward_sde.discretize(batch, t)
        rev_f = f - G[:, None, None, None] ** 2 * self.score_fn(batch, t) * (0.5 if self.probability_flow else 1.)
        rev_G = torch.zeros_like(G) if self.probability_flow else G
        return rev_f, rev_G


class VPSDE(SDE):
    def __init__(self, beta_min=0.1, beta_max=20., N=1000):
        """
        Construct a Variance Preserving SDE.

        Args:
            beta_min: value of beta(0)
            beta_max: value of beta(1)
            N: number of discretization steps
        """
        super().__init__(N, T=1.0, eps=1e-3)
        self.beta_0 = beta_min
        self.beta_1 = beta_max

        self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1. - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    def get_sde(self, batch: BatchType, t: BatchedFloatType) -> Tuple[TrainImagesType, TrainImagesType]:
        x = batch[0]
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t[:, None, None, None] * x
        diffusion = beta_t ** 0.5
        return drift, diffusion

    def marginal_prob(self, x_0: TrainImagesType, t: BatchedFloatType) -> Tuple[TrainImagesType, BatchedFloatType]:
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = torch.exp(log_mean_coeff[:, None, None, None]) * x_0
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean, std

    def prior_sampling(self, shape: Tuple) -> TrainImagesType:
        return torch.randn(*shape)

    def discretize(self, batch: BatchType, t: BatchedFloatType) -> Tuple[TrainImagesType, TrainImagesType]:
        """ DDPM discretization. """
        x = batch[0]
        timestep = (t * (self.N - 1) / self.T).long()
        beta = self.discrete_betas.to(x.device)[timestep]
        alpha = self.alphas.to(x.device)[timestep]
        sqrt_beta = torch.sqrt(beta)
        f = torch.sqrt(alpha)[:, None, None, None] * x - x
        G = sqrt_beta
        return f, G


class subVPSDE(SDE):
    def __init__(self, beta_min=0.1, beta_max=20., N=1000):
        """
        Construct the sub-VP SDE that excels at likelihoods.

        Args:
            beta_min: value of beta(0)
            beta_max: value of beta(1)
            N: number of discretization steps
        """
        super().__init__(N, T=1.0, eps=1e-3)
        self.beta_0 = beta_min
        self.beta_1 = beta_max

        self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1. - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    def get_sde(self, batch: BatchType, t: BatchedFloatType) -> Tuple[TrainImagesType, TrainImagesType]:
        x = batch[0]
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t[:, None, None, None] * x
        discount = 1. - torch.exp(-2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t ** 2).item()
        diffusion = (beta_t * discount) ** 0.5
        return drift, diffusion

    def marginal_prob(self, x_0: TrainImagesType, t: BatchedFloatType) -> Tuple[TrainImagesType, TrainImagesType]:
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = torch.exp(log_mean_coeff)[:, None, None, None] * x_0
        std = 1 - torch.exp(2. * log_mean_coeff)
        return mean, std

    def prior_sampling(self, shape: Tuple) -> TrainImagesType:
        return torch.randn(*shape)


class VESDE(SDE):
    def __init__(self, sigma_min=0.01, sigma_max=50., N=1000):
        """
        Construct a Variance Exploding SDE.

        Args:
            sigma_min: smallest sigma.
            sigma_max: largest sigma.
            N: number of discretization steps
        """
        super().__init__(N, T=1.0, eps=1e-5)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.discrete_sigmas = torch.exp(torch.linspace(torch.log(self.sigma_min), torch.log(self.sigma_max), N))

    def get_sde(self, batch: BatchType, t: BatchedFloatType) -> Tuple[TrainImagesType, TrainImagesType]:
        x = batch[0]
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        drift = torch.zeros_like(x)
        diffusion = sigma * (2 * (torch.log(self.sigma_max) - torch.log(self.sigma_min))) ** 0.5
        return drift, diffusion

    def marginal_prob(self, x_0: TrainImagesType, t: BatchedFloatType) -> Tuple[TrainImagesType, TrainImagesType]:
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        mean = x_0
        return mean, std

    def prior_sampling(self, shape: Tuple) -> TrainImagesType:
        return torch.randn(*shape) * self.sigma_max

    def discretize(self, batch: BatchType, t: BatchedFloatType) -> Tuple[TrainImagesType, TrainImagesType]:
        """ SMLD(NCSN) discretization. """
        x = batch[0]
        timestep = (t * (self.N - 1) / self.T).long()
        sigma = self.discrete_sigmas.to(t.device)[timestep]
        adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t),
                                     self.discrete_sigmas.to(t.device)[timestep - 1])
        f = torch.zeros_like(x)
        G = torch.sqrt(sigma ** 2 - adjacent_sigma ** 2)
        return f, G


class SDEName(Enum):
    VESDE = VESDE
    VPSDE = VPSDE
    subVPSDE = subVPSDE


def get_sde(sde_name: SDEName, schedule_param_start: float, schedule_param_end: float, num_scales: int) -> SDE:
    sde_instance = sde_name.value
    sde = sde_instance(schedule_param_start, schedule_param_end, num_scales)
    return sde
