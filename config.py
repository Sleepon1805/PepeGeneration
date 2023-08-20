import os
import git
from dataclasses import dataclass
from typing import Tuple

from data.condition_utils import CONDITION_SIZE

HIGHRES_IMAGE_SIZE_MULT = 4


def curr_git_hash():
    repo = git.Repo('.', search_parent_directories=True)
    return repo.head.commit.hexsha[:7] + '+' * repo.is_dirty()


@dataclass
class Paths:
    pepe_data_path: str = '/home/sleepon/data/TwitchPepeDataset/'
    celeba_data_path: str = '/home/sleepon/data/CelebFaces/img_align_celeba/img_align_celeba/'
    parsed_datasets: str = os.path.dirname(__file__) + '/data/parsed_data/'


@dataclass
class DDPMSamplingConfig:
    # default DDPM sampler
    beta_min: float = 0.0001
    beta_max: float = 0.02
    diffusion_steps: int = 1000


@dataclass
class PCSamplingConfig:
    # Predictor-Corrector Sampler
    sde_name: str = 'VPSDE'  # VPSDE, subVPSDE, VESDE
    beta_min: float = 0.1  # VPSDE, subVPSDE param
    beta_max: float = 20.  # VPSDE, subVPSDE param
    sigma_min: float = 0.01  # VESDE param
    sigma_max: float = 50.  # VESDE param
    num_scales: int = 1000
    predictor_name: str = 'euler_maruyama'  # none, ancestral_sampling, reverse_diffusion, euler_maruyama
    corrector_name: str = 'langevin'  # none, langevin, ald
    snr: float = 0.01
    num_corrector_steps: int = 1
    probability_flow: bool = False
    denoise: bool = False


@dataclass
class ODESamplingConfig:
    # ODE Solver
    sde_name: str = 'VPSDE'  # VPSDE, subVPSDE, VESDE
    beta_min: float = 0.1  # VPSDE, subVPSDE param
    beta_max: float = 20.  # VPSDE, subVPSDE param
    sigma_min: float = 0.01  # VESDE param
    sigma_max: float = 50.  # VESDE param
    num_scales: int = 1000
    method: str = 'RK45'
    rtol: float = 1e-5
    atol: float = 1e-5
    denoise: bool = False


@dataclass
class Config:
    # git commit hash for logging
    git_hash: str = curr_git_hash()

    # training params
    batch_size: int = 64
    precision: str = '16-mixed'
    image_size: int = 64  # size of image NxN
    lr: float = 1e-4  # learning rate on training start
    scheduler: str = 'MultiStepLR'
    gradient_clip_algorithm: str = "norm"
    gradient_clip_val: float = 0.5
    dataset_split: Tuple[float, float] = (0.8, 0.2)

    # pretrained backbone and current dataset
    dataset_name: str = 'celeba'
    use_condition: bool = False
    condition_size: int = CONDITION_SIZE
    pretrained_ckpt: str = None
    # pretrained_ckpt: str = './lightning_logs/celeba/version_6/checkpoints/last.ckpt'

    # model params
    init_channels: int = 128
    channel_mult: Tuple[int, int, int, int] = (1, 2, 4, 4)
    conv_resample: bool = True
    num_heads: int = 1
    dropout: float = 0.3
    use_second_attention: bool = True

    sampler_config = DDPMSamplingConfig()  # PCSamplingConfig(), ODESamplingConfig()
