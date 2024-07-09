""" Abstract SDE classes, Reverse SDE, and VE/VP SDEs. """
import sys
import torch
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
    def get_sde(self, batch: BatchType, t: BatchedFloatType) -> Tuple[TrainImagesType, BatchedFloatType]:
        """
        Get current drift and diffusion for X and t (vectorized).
        :param batch:
        :param t:
        :return:
        """
        pass

    @abstractmethod
    def get_param(self, t: BatchedFloatType) -> BatchedFloatType:
        """
        Get the parameter for the SDE at time t.
        :param t: time steps
        :return: parameter at time t
        """
        pass

    @abstractmethod
    def marginal_prob(self, x_0: TrainImagesType, t: BatchedFloatType) -> Tuple[TrainImagesType, BatchedFloatType]:
        """
        Noise image(s) x_0 to timestep t: $p_t(x)$
        :param x_0: Images
        :param t: timesteps
        :return: mean and std of the marginal distribution
        """
        # TODO: use Eq. (5.50), (5.51) in Särkkä & Solin (2019): Applied Stochastic Differential Equations
        pass

    def prior_sampling(self, shape: Tuple) -> TrainImagesType:
        """
        Generate one sample from the prior distribution: $p_T(X)$ (which should not depend on X).
        :param shape: shape of the sample
        :return: sample from the prior distribution
        """
        std = self.marginal_prob(torch.zeros(shape), torch.ones(shape[0]) * self.T)[1]
        p_t = torch.randn(shape) * std[:, None, None, None]
        return p_t

    def discretize(self, batch: BatchType, t: BatchedFloatType) -> Tuple[TrainImagesType, BatchedFloatType]:
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

    def get_sde(self, batch: BatchType, t: BatchedFloatType) -> Tuple[TrainImagesType, BatchedFloatType]:
        drift, diffusion = self.forward_sde.get_sde(batch, t)
        score = self.score_fn(batch, t)
        drift = drift - diffusion[:, None, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
        # Set the diffusion function to zero for ODEs.
        diffusion *= (1 - self.probability_flow)
        return drift, diffusion

    def get_param(self, t: BatchedFloatType) -> BatchedFloatType:
        return self.forward_sde.get_param(t)

    def marginal_prob(self, x_0: TrainImagesType, t: BatchedFloatType) -> Tuple[TrainImagesType, BatchedFloatType]:
        return self.forward_sde.marginal_prob(x_0, t)


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

    def get_param(self, t: BatchedFloatType) -> BatchedFloatType:
        return self.beta_0 + (self.beta_1 - self.beta_0) * (t / self.T)

    def get_sde(self, batch: BatchType, t: BatchedFloatType) -> Tuple[TrainImagesType, BatchedFloatType]:
        x = batch[0]
        beta_t = self.get_param(t)
        drift = -0.5 * beta_t[:, None, None, None] * x
        diffusion = beta_t ** 0.5
        return drift, diffusion

    def marginal_prob(self, x_0: TrainImagesType, t: BatchedFloatType) -> Tuple[TrainImagesType, BatchedFloatType]:
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = torch.exp(log_mean_coeff[:, None, None, None]) * x_0
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean, std


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

    def get_param(self, t: BatchedFloatType) -> BatchedFloatType:
        return self.beta_0 + (self.beta_1 - self.beta_0) * (t / self.T)

    def get_sde(self, batch: BatchType, t: BatchedFloatType) -> Tuple[TrainImagesType, BatchedFloatType]:
        x = batch[0]
        beta_t = self.get_param(t)
        drift = -0.5 * beta_t[:, None, None, None] * x
        discount = 1. - torch.exp(-2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t ** 2)
        diffusion = (beta_t * discount) ** 0.5
        return drift, diffusion

    def marginal_prob(self, x_0: TrainImagesType, t: BatchedFloatType) -> Tuple[TrainImagesType, BatchedFloatType]:
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = torch.exp(log_mean_coeff)[:, None, None, None] * x_0
        std = 1 - torch.exp(2. * log_mean_coeff)
        return mean, std


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
        self.sigma_min = torch.tensor(sigma_min)
        self.sigma_max = torch.tensor(sigma_max)
        self.discrete_sigmas = torch.exp(torch.linspace(torch.log(self.sigma_min), torch.log(self.sigma_max), N))

    def get_param(self, t: BatchedFloatType) -> BatchedFloatType:
        self.sigma_min * (self.sigma_max / self.sigma_min) ** (t / self.T)

    def get_sde(self, batch: BatchType, t: BatchedFloatType) -> Tuple[TrainImagesType, BatchedFloatType]:
        x = batch[0]
        sigma = self.get_param(t)
        drift = torch.zeros_like(x)
        diffusion = sigma * (2 * (torch.log(self.sigma_max) - torch.log(self.sigma_min))) ** 0.5
        return drift, diffusion

    def marginal_prob(self, x_0: TrainImagesType, t: BatchedFloatType) -> Tuple[TrainImagesType, BatchedFloatType]:
        std = self.get_param(t)
        mean = x_0
        return mean, std


def get_sde(sde_name: str, schedule_param_start: float, schedule_param_end: float, num_scales: int) -> SDE:
    try:
        # get class name from string
        sde_instance = getattr(sys.modules[__name__], sde_name)
    except AttributeError:
        raise ValueError(f"SDE {sde_name} not found.")
    sde = sde_instance(schedule_param_start, schedule_param_end, num_scales)
    return sde
