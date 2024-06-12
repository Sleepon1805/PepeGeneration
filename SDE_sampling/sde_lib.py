""" Abstract SDE classes, Reverse SDE, and VE/VP SDEs. """
import abc
import torch
import numpy as np
from enum import Enum
from typing import Callable


class SDE(abc.ABC):
    """ SDE abstract class. Functions are designed for a mini-batch of inputs. """

    def __init__(self, N, T, eps):
        """
        Construct an SDE.

        Args:
            N: number of discretization time steps.
        """
        super().__init__()
        self.N = N
        self.T = T
        self.eps = eps

    @abc.abstractmethod
    def get_sde(self, x: torch.Tensor, t):
        pass

    @abc.abstractmethod
    def marginal_prob(self, x_0: torch.Tensor, t):
        """ Parameters to determine the marginal distribution of the SDE, $p_t(x)$. """
        # TODO: use Eq. (5.50), (5.51) in Särkkä & Solin (2019): Applied Stochastic Differential Equations
        pass

    @abc.abstractmethod
    def prior_sampling(self, shape):
        """ Generate one sample from the prior distribution, $p_T(x)$. """
        pass

    @abc.abstractmethod
    def prior_logp(self, z: torch.Tensor):
        """
        Compute log-density of the prior distribution.
        Useful for computing the log-SDE_sampling via probability flow ODE.

        Args:
            z: latent code
        Returns:
            log probability density
        """
        pass

    def discretize(self, x: torch.Tensor, t):
        """
        Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.
        Useful for reverse diffusion sampling and probabiliy flow sampling.
        Defaults to Euler-Maruyama discretization.

        Args:
            x: a torch tensor
            t: a torch float representing the time step (from 0 to `self.T`)
        Returns:
            f, G
        """
        dt = 1 / self.N
        drift, diffusion = self.get_sde(x, t)
        f = drift * dt
        G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
        return f, G


class ReverseSDE(SDE):
    def __init__(self, sde: SDE, score_fn: Callable, probability_flow=False):
        super().__init__(sde.N, sde.T, sde.eps)
        self.forward_sde = sde
        self.score_fn = score_fn
        self.probability_flow = probability_flow

    def get_sde(self, batch: tuple[torch.Tensor, ...], t):
        """ Create the drift and diffusion functions for the reverse SDE/ODE. """
        x = batch[0]
        drift, diffusion = self.forward_sde.get_sde(x, t)
        score = self.score_fn(batch, t)
        drift = drift - diffusion[:, None, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
        # Set the diffusion function to zero for ODEs.
        diffusion *= (1 - self.probability_flow)
        return drift, diffusion

    def marginal_prob(self, x: torch.Tensor, t):
        return self.forward_sde.marginal_prob(x, t)

    def prior_sampling(self, shape):
        return self.forward_sde.prior_sampling(shape)

    def prior_logp(self, z: torch.Tensor):
        return self.forward_sde.prior_logp(z)

    def discretize(self, batch: tuple[torch.Tensor, ...], t):
        """ Create discretized iteration rules for the reverse diffusion sampler. """
        x = batch[0]
        f, G = self.forward_sde.discretize(x, t)
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
        super().__init__(N, T=1, eps=1e-3)
        self.beta_0 = beta_min
        self.beta_1 = beta_max

        self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1. - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    def get_sde(self, x, t):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t[:, None, None, None] * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(self, x, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = torch.exp(log_mean_coeff[:, None, None, None]) * x
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape)

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        logps = -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.
        return logps

    def discretize(self, x, t):
        """ DDPM discretization. """
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
        super().__init__(N, T=1, eps=1e-3)
        self.beta_0 = beta_min
        self.beta_1 = beta_max

        self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1. - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    def get_sde(self, x, t):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t[:, None, None, None] * x
        discount = 1. - torch.exp(-2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t ** 2)
        diffusion = torch.sqrt(beta_t * discount)
        return drift, diffusion

    def marginal_prob(self, x, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = torch.exp(log_mean_coeff)[:, None, None, None] * x
        std = 1 - torch.exp(2. * log_mean_coeff)
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape)

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        return -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.


class VESDE(SDE):
    def __init__(self, sigma_min=0.01, sigma_max=50., N=1000):
        """
        Construct a Variance Exploding SDE.

        Args:
            sigma_min: smallest sigma.
            sigma_max: largest sigma.
            N: number of discretization steps
        """
        super().__init__(N, T=1, eps=1e-5)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.discrete_sigmas = torch.exp(torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N))

    def get_sde(self, x, t):
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        drift = torch.zeros_like(x)
        diffusion = sigma * torch.sqrt(torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)),
                                                    device=t.device))
        return drift, diffusion

    def marginal_prob(self, x, t):
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        mean = x
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape) * self.sigma_max

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        return -N / 2. * np.log(2 * np.pi * self.sigma_max ** 2) - torch.sum(z ** 2, dim=(1, 2, 3)) / (
                    2 * self.sigma_max ** 2)

    def discretize(self, x, t):
        """ SMLD(NCSN) discretization. """
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


def get_sde(sde_name: SDEName, schedule_param_start, schedule_param_end, num_scales) -> SDE:
    sde_instance = sde_name.value
    sde = sde_instance(schedule_param_start, schedule_param_end, num_scales)
    return sde
