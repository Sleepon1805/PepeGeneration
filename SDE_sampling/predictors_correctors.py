import abc
import torch
import numpy as np
from typing import Callable

from SDE_sampling import sde_lib


class Predictor(abc.ABC):
    """ The abstract class for a predictor algorithm. """

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__()
        self.sde = sde
        # Compute the reverse SDE/ODE
        self.rsde = sde_lib.ReverseSDE(sde, score_fn, probability_flow)
        self.score_fn = score_fn

    @abc.abstractmethod
    def update(self, batch: tuple[torch.Tensor, ...], t):
        """
        One update of the predictor.

        Args:
            batch: A Batch representing the current state
            t: A Pytorch tensor representing the current time step.
        Returns:
            x: A PyTorch tensor of the next state.
            x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass


class EulerMaruyamaPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update(self, batch: tuple[torch.Tensor, ...], t):
        x = batch[0]
        dt = -1. / self.rsde.N
        z = torch.randn_like(x)
        drift, diffusion = self.rsde.get_sde(batch, t)
        x_mean = x + drift * dt
        x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z
        return x, x_mean


class ReverseDiffusionPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update(self, batch: tuple[torch.Tensor, ...], t):
        x = batch[0]
        f, G = self.rsde.discretize(batch, t)
        z = torch.randn_like(x)
        x_mean = x - f
        x = x_mean + G[:, None, None, None] * z
        return x, x_mean


class AncestralSamplingPredictor(Predictor):
    """ The ancestral sampling predictor. Currently only supports VE/VP SDEs. """

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)
        if not isinstance(sde, sde_lib.VPSDE) and not isinstance(sde, sde_lib.VESDE):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")
        assert not probability_flow, "Probability flow not supported by ancestral sampling"

    def vesde_update_fn(self, batch: tuple[torch.Tensor, ...], t):
        x = batch[0]
        timestep = (t * (self.sde.N - 1) / self.sde.T).long()
        sigma = self.sde.discrete_sigmas.to(t.device)[timestep]
        adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t), self.sde.discrete_sigmas.to(t.device)[timestep - 1])
        score = self.score_fn(batch, t)
        x_mean = x + score * (sigma ** 2 - adjacent_sigma ** 2)[:, None, None, None]
        std = torch.sqrt((adjacent_sigma ** 2 * (sigma ** 2 - adjacent_sigma ** 2)) / (sigma ** 2))
        noise = torch.randn_like(x)
        x = x_mean + std[:, None, None, None] * noise
        return x, x_mean

    def vpsde_update_fn(self, batch: tuple[torch.Tensor, ...], t):
        x = batch[0]
        timestep = (t * (self.sde.N - 1) / self.sde.T).long()
        beta = self.sde.discrete_betas.to(t.device)[timestep]
        score = self.score_fn(batch, t)
        x_mean = (x + beta[:, None, None, None] * score) / torch.sqrt(1. - beta)[:, None, None, None]
        # x_mean = (2 - torch.sqrt(1 - beta)[:, None, None, None]) * x + beta[:, None, None, None] * score
        noise = torch.randn_like(x)
        x = x_mean + torch.sqrt(beta)[:, None, None, None] * noise
        return x, x_mean

    def update(self, batch: tuple[torch.Tensor, ...], t):
        if isinstance(self.sde, sde_lib.VESDE):
            return self.vesde_update_fn(batch, t)
        elif isinstance(self.sde, sde_lib.VPSDE):
            return self.vpsde_update_fn(batch, t)


class NonePredictor(Predictor):
    """ An empty predictor that does nothing. """

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update(self, batch: tuple[torch.Tensor, ...], t):
        x = batch[0]
        return x, x


class Corrector(abc.ABC):
    """ The abstract class for a corrector algorithm. """

    def __init__(self, sde, score_fn, snr):
        super().__init__()
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr

    @abc.abstractmethod
    def update(self, batch: tuple[torch.Tensor, ...], t):
        """
        One update of the corrector.

        Args:
            batch: A Batch representing the current state
            t: A PyTorch tensor representing the current time step.
        Returns:
            x: A PyTorch tensor of the next state.
            x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass


class LangevinCorrector(Corrector):
    def __init__(self, sde, score_fn, snr):
        super().__init__(sde, score_fn, snr)
        if not isinstance(sde, sde_lib.VPSDE) \
                and not isinstance(sde, sde_lib.VESDE) \
                and not isinstance(sde, sde_lib.subVPSDE):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

    def update(self, batch: tuple[torch.Tensor, ...], t):
        x = batch[0]
        if isinstance(self.sde, sde_lib.VPSDE) or isinstance(self.sde, sde_lib.subVPSDE):
            timestep = (t * (self.sde.N - 1) / self.sde.T).long()
            alpha = self.sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        grad = self.score_fn(batch, t)
        noise = torch.randn_like(x)
        grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
        noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
        step_size = (self.snr * noise_norm / grad_norm) ** 2 * 2 * alpha
        x_mean = x + step_size[:, None, None, None] * grad
        x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

        return x, x_mean


class AnnealedLangevinDynamics(Corrector):
    """
    The original annealed Langevin dynamics predictor in NCSN/NCSNv2.
    We include this corrector only for completeness. It was not directly used in our paper.
    """

    def __init__(self, sde, score_fn, snr):
        super().__init__(sde, score_fn, snr)
        if not isinstance(sde, sde_lib.VPSDE) \
                and not isinstance(sde, sde_lib.VESDE) \
                and not isinstance(sde, sde_lib.subVPSDE):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

    def update(self, batch: tuple[torch.Tensor, ...], t):
        x = batch[0]
        score_fn = self.score_fn
        target_snr = self.snr
        if isinstance(self.sde, sde_lib.VPSDE) or isinstance(self.sde, sde_lib.subVPSDE):
            timestep = (t * (self.sde.N - 1) / self.sde.T).long()
            alpha = self.sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        std = self.sde.marginal_prob(x, t)[1]

        grad = score_fn(batch, t)
        noise = torch.randn_like(x)
        step_size = (target_snr * std) ** 2 * 2 * alpha
        x_mean = x + step_size[:, None, None, None] * grad
        x = x_mean + noise * torch.sqrt(step_size * 2)[:, None, None, None]

        return x, x_mean


class NoneCorrector(Corrector):
    """ An empty corrector that does nothing. """

    def __init__(self, sde, score_fn, snr):
        super().__init__(sde, score_fn, snr)

    def update(self, batch: tuple[torch.Tensor, ...], t):
        x = batch[0]
        return x, x


def get_predictor(predictor_name: str, sde: sde_lib.SDE, score_fn: Callable, probability_flow: bool) -> Predictor:
    if predictor_name.lower() == 'euler_maruyama':
        return EulerMaruyamaPredictor(sde, score_fn, probability_flow)
    elif predictor_name.lower() == 'reverse_diffusion':
        return ReverseDiffusionPredictor(sde, score_fn, probability_flow)
    elif predictor_name.lower() == 'ancestral_sampling':
        return AncestralSamplingPredictor(sde, score_fn, probability_flow)
    elif predictor_name.lower() == 'none':
        return NonePredictor(sde, score_fn, probability_flow)
    else:
        raise ValueError(predictor_name)


def get_corrector(corrector_name: str, sde: sde_lib.SDE, score_fn: Callable, snr: float) -> Corrector:
    if corrector_name.lower() == 'langevin':
        return LangevinCorrector(sde, score_fn, snr)
    elif corrector_name.lower() == 'ald':
        return AnnealedLangevinDynamics(sde, score_fn, snr)
    elif corrector_name.lower() == 'none':
        return NoneCorrector(sde, score_fn, snr)
    else:
        raise ValueError(corrector_name)
