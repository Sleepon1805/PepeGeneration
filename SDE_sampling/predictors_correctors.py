import sys
import torch
from typing import Callable, Tuple
from abc import abstractmethod

from utils.typings import TrainImagesType, BatchType, BatchedFloatType, ABCTypeChecked
from SDE_sampling import sde_lib


class Predictor(ABCTypeChecked):
    """ The abstract class for a predictor algorithm. """

    def __init__(self, sde: sde_lib.SDE, score_fn: Callable, probability_flow=False):
        self.sde = sde
        self.score_fn = score_fn
        self.rsde = sde_lib.ReverseSDE(sde, score_fn, probability_flow)

    @abstractmethod
    def update(self, batch: BatchType, t: BatchedFloatType) -> Tuple[TrainImagesType, TrainImagesType]:
        """
        One update of the predictor.
        :param batch: Batch with noised images and labels
        :param t: Tensor with time steps for each image.
        :return: Tuple of two Tensors: Images with random noise and images without noise.
        """
        pass


class EulerMaruyamaPredictor(Predictor):
    def __init__(self, sde: sde_lib.SDE, score_fn: Callable, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update(self, batch: BatchType, t: BatchedFloatType) -> Tuple[TrainImagesType, TrainImagesType]:
        x = batch[0]
        dt = -1. / self.rsde.N
        z = torch.randn_like(x)
        drift, diffusion = self.rsde.get_sde(batch, t)
        x_mean = x + drift * dt
        x = x_mean + diffusion[:, None, None, None] * (-dt) ** 0.5 * z
        return x, x_mean


class ReverseDiffusionPredictor(Predictor):
    def __init__(self, sde: sde_lib.SDE, score_fn: Callable, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update(self, batch: BatchType, t: BatchedFloatType) -> Tuple[TrainImagesType, TrainImagesType]:
        x = batch[0]
        f, G = self.rsde.discretize(batch, t)
        z = torch.randn_like(x)
        x_mean = x - f
        x = x_mean + G[:, None, None, None] * z
        return x, x_mean


class AncestralSamplingPredictor(Predictor):
    """ The ancestral sampling predictor. Currently only supports VE/VP SDEs. """

    def __init__(self, sde: sde_lib.SDE, score_fn: Callable, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)
        if not isinstance(sde, sde_lib.VPSDE) and not isinstance(sde, sde_lib.VESDE):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")
        assert not probability_flow, "Probability flow not supported by ancestral sampling"

    def vesde_update_fn(self, batch: BatchType, t: BatchedFloatType) -> Tuple[TrainImagesType, TrainImagesType]:
        assert isinstance(self.sde, sde_lib.VESDE)
        x = batch[0]
        timestep = (t * (self.sde.N - 1) / self.sde.T).long()
        discrete_sigmas = torch.exp(torch.linspace(
            torch.log(self.sde.sigma_min),
            torch.log(self.sde.sigma_max),
            self.sde.N
        ))
        sigma = discrete_sigmas.to(t.device)[timestep]
        adjacent_sigma = torch.where(  # noqa
            timestep == 0, torch.zeros_like(t),
            self.sde.discrete_sigmas.to(t.device)[timestep - 1]
        )
        score = self.score_fn(batch, t)
        x_mean = x + score * (sigma ** 2 - adjacent_sigma ** 2)[:, None, None, None]
        std = torch.sqrt((adjacent_sigma ** 2 * (sigma ** 2 - adjacent_sigma ** 2)) / (sigma ** 2))
        noise = torch.randn_like(x)
        x = x_mean + std[:, None, None, None] * noise
        return x, x_mean

    def vpsde_update_fn(self, batch: BatchType, t: BatchedFloatType) -> Tuple[TrainImagesType, TrainImagesType]:
        assert isinstance(self.sde, sde_lib.VPSDE)
        x = batch[0]
        timestep = (t * (self.sde.N - 1) / self.sde.T).long()
        discrete_betas = torch.linspace(self.sde.beta_0 / self.sde.N, self.sde.beta_1 / self.sde.N, self.sde.N)
        beta = discrete_betas.to(t.device)[timestep]
        score = self.score_fn(batch, t)
        x_mean = (x + beta[:, None, None, None] * score) / torch.sqrt(1. - beta)[:, None, None, None]
        # x_mean = (2 - torch.sqrt(1 - beta)[:, None, None, None]) * x + beta[:, None, None, None] * score
        noise = torch.randn_like(x)
        x = x_mean + torch.sqrt(beta)[:, None, None, None] * noise
        return x, x_mean

    def update(self, batch: BatchType, t: BatchedFloatType) -> Tuple[TrainImagesType, TrainImagesType]:
        if isinstance(self.sde, sde_lib.VESDE):
            return self.vesde_update_fn(batch, t)
        elif isinstance(self.sde, sde_lib.VPSDE):
            return self.vpsde_update_fn(batch, t)


class NonePredictor(Predictor):
    """ An empty predictor that does nothing. """

    def __init__(self, sde: sde_lib.SDE, score_fn: Callable, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update(self, batch: BatchType, t: BatchedFloatType) -> Tuple[TrainImagesType, TrainImagesType]:
        x = batch[0]
        return x, x


class Corrector(ABCTypeChecked):
    """ The abstract class for a corrector algorithm. """

    def __init__(self, sde: sde_lib.SDE, score_fn: Callable, snr: float):
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr

    @abstractmethod
    def update(self, batch: BatchType, t: BatchedFloatType) -> Tuple[TrainImagesType, TrainImagesType]:
        """
        One update of the predictor.
        :param batch: Batch with noised images and labels
        :param t: Tensor with time steps for each image.
        :return: Tuple of two Tensors: Images with random noise and images without noise.
        """
        pass


class LangevinCorrector(Corrector):
    def __init__(self, sde: sde_lib.SDE, score_fn: Callable, snr: float):
        super().__init__(sde, score_fn, snr)
        if not isinstance(sde, sde_lib.VPSDE) \
                and not isinstance(sde, sde_lib.VESDE) \
                and not isinstance(sde, sde_lib.subVPSDE):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

    def update(self, batch: BatchType, t: BatchedFloatType) -> Tuple[TrainImagesType, TrainImagesType]:
        x = batch[0]
        if isinstance(self.sde, sde_lib.VPSDE) or isinstance(self.sde, sde_lib.subVPSDE):
            beta_t = self.sde.get_param(t)
            alpha = 1. - beta_t / self.sde.N
        elif isinstance(self.sde, sde_lib.VESDE):
            alpha = torch.ones_like(t)
        else:
            raise NotImplementedError(f"SDE class {self.sde.__class__.__name__} not yet supported.")

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

    def __init__(self, sde: sde_lib.SDE, score_fn: Callable, snr: float):
        super().__init__(sde, score_fn, snr)
        if not isinstance(sde, sde_lib.VPSDE) \
                and not isinstance(sde, sde_lib.VESDE) \
                and not isinstance(sde, sde_lib.subVPSDE):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

    def update(self, batch: BatchType, t: BatchedFloatType) -> Tuple[TrainImagesType, TrainImagesType]:
        x = batch[0]
        score_fn = self.score_fn
        target_snr = self.snr
        if isinstance(self.sde, sde_lib.VPSDE) or isinstance(self.sde, sde_lib.subVPSDE):
            beta_t = self.sde.get_param(t)
            alpha = 1. - beta_t / self.sde.N
        elif isinstance(self.sde, sde_lib.VESDE):
            alpha = torch.ones_like(t)
        else:
            raise NotImplementedError(f"SDE class {self.sde.__class__.__name__} not yet supported.")

        std = self.sde.marginal_prob(x, t)[1]

        grad = score_fn(batch, t)
        noise = torch.randn_like(x)
        step_size = (target_snr * std) ** 2 * 2 * alpha
        x_mean = x + step_size[:, None, None, None] * grad
        x = x_mean + noise * torch.sqrt(step_size * 2)[:, None, None, None]

        return x, x_mean


class NoneCorrector(Corrector):
    """ An empty corrector that does nothing. """

    def __init__(self, sde: sde_lib.SDE, score_fn: Callable, snr: float):
        super().__init__(sde, score_fn, snr)

    def update(self, batch: BatchType, t: BatchedFloatType) -> Tuple[TrainImagesType, TrainImagesType]:
        x = batch[0]
        return x, x


def get_predictor(
        predictor_name: str, sde: sde_lib.SDE, score_fn: Callable, probability_flow: bool
) -> Predictor:
    try:
        # get class name from string
        predictor_instance = getattr(sys.modules[__name__], predictor_name)
    except AttributeError:
        raise ValueError(f"Predictor {predictor_name} not found.")
    return predictor_instance(sde, score_fn, probability_flow)


def get_corrector(
        corrector_name: str, sde: sde_lib.SDE, score_fn: Callable, snr: float
) -> Corrector:
    try:
        # get class name from string
        corrector_instance = getattr(sys.modules[__name__], corrector_name)
    except AttributeError:
        raise ValueError(f"Corrector {corrector_name} not found.")
    return corrector_instance(sde, score_fn, snr)
