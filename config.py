import os
import git
import pickle
from typing import Tuple, Literal
from pathlib import Path
from dataclasses import dataclass

from data.condition_utils import CONDITION_SIZE

HIGHRES_IMAGE_SIZE_MULT = 4


def curr_git_hash():
    repo = git.Repo('.', search_parent_directories=True)
    return repo.head.commit.hexsha[:7] + '+' * repo.is_dirty()


@dataclass
class Paths:
    pepe_data_path: str = '/home/sleepon/data/TwitchPepeDataset/'
    celeba_data_path: str = '/home/sleepon/data/CelebFaces/img_align_celeba/img_align_celeba/'
    twitch_emotes_data_path: str = '/home/sleepon/data/AllFFZEmotes/'
    parsed_datasets: str = os.path.dirname(__file__) + '/data/parsed_data/'


@dataclass
class DDPMSamplingConfig:
    """ default DDPM sampler """
    # DDPM params
    beta_min: float = 0.0001
    beta_max: float = 0.02
    diffusion_steps: int = 1000


@dataclass
class PCSamplingConfig:
    """ Predictor-Corrector Sampler """
    # sde params
    sde_name: Literal['VPSDE', 'subVPSDE', 'VESDE'] = 'VESDE'
    num_scales: int = 1000  # number of discretization timesteps
    beta_min: float = 0.1  # VPSDE, subVPSDE param
    beta_max: float = 20.  # VPSDE, subVPSDE param
    sigma_min: float = 0.01  # VESDE param
    sigma_max: float = 50.  # VESDE param
    # predictor params
    predictor_name: Literal['none', 'ancestral_sampling', 'reverse_diffusion', 'euler_maruyama'] = 'euler_maruyama'
    # corrector params
    corrector_name: Literal['none', 'langevin', 'ald'] = 'langevin'
    snr: float = 0.01  # signal-to-noise ratio
    num_corrector_steps: int = 1
    # sampler params
    probability_flow: bool = False
    denoise: bool = False


@dataclass
class ODESamplingConfig:
    """ ODE Solver """
    # sde params
    sde_name: Literal['VPSDE', 'subVPSDE', 'VESDE'] = 'VESDE'
    num_scales: int = 1000  # number of discretization timesteps
    beta_min: float = 0.1  # VPSDE, subVPSDE param
    beta_max: float = 20.  # VPSDE, subVPSDE param
    sigma_min: float = 0.01  # VESDE param
    sigma_max: float = 50.  # VESDE param
    # ode solver params
    method: str = 'RK45'
    rtol: float = 1e-5
    atol: float = 1e-5
    # sampler params
    denoise: bool = False


@dataclass
class Config:
    # git commit hash for logging
    git_hash: str = curr_git_hash()

    # training params
    batch_size: int = 16
    precision: str = '16-mixed'
    image_size: int = 128  # size of image NxN
    lr: float = 1e-4  # learning rate on training start
    scheduler: str = 'no'
    gradient_clip_algorithm: str = "norm"
    gradient_clip_val: float = 0.5
    dataset_split: Tuple[float, float] = (0.8, 0.2)

    # pretrained backbone and current dataset
    dataset_name: Literal['celeba', 'pepe', 'twitch_emotes'] = 'pepe'
    use_condition: bool = False
    condition_size: int = CONDITION_SIZE
    pretrained_ckpt: str = None
    # pretrained_ckpt: str = './lightning_logs/celeba/version_12/checkpoints/last.ckpt'

    # model params
    init_channels: int = 128
    channel_mult: Tuple[int, int, int, int] = (1, 2, 4, 4)
    conv_resample: bool = True
    num_heads: int = 1
    dropout: float = 0.3
    use_second_attention: bool = True

    sampler_config = PCSamplingConfig()  # one of DDPMSamplingConfig(), PCSamplingConfig(), ODESamplingConfig()


def save_config(config: Config, save_folder: str | Path):
    if isinstance(save_folder, str):
        save_folder = Path(save_folder)

    assert os.path.isdir(save_folder)
    os.makedirs(save_folder, exist_ok=True)

    # save config and sampler config
    with open(save_folder.joinpath('config.pkl'), 'wb') as f:
        pickle.dump(config, f, pickle.HIGHEST_PROTOCOL)
    with open(save_folder.joinpath('sampler_config.pkl'), 'wb') as f:
        # must be saved separately due to different object classes of different sampler_configs
        pickle.dump(config.sampler_config, f, pickle.HIGHEST_PROTOCOL)


def load_config(path: str | Path, path_to_checkpoint: bool):
    if isinstance(path, str):
        path = Path(path)
    if path_to_checkpoint:
        path = path.parents[1]

    try:
        with open(path.joinpath('config.pkl'), 'rb') as config_file:
            config = pickle.load(config_file)
    except Exception as e:
        print(f'Could not read config from .pkl file: {e}')
        print('Using default Config().')
        config = Config()

    try:
        with open(path.joinpath('sampler_config.pkl'), 'rb') as sampler_config_file:
            sampler_config = pickle.load(sampler_config_file)
    except Exception as e:
        print(f'Could not read sampler_config from .pkl file: {e}')
        print('Using DDPMSamplingConfig().')
        sampler_config = DDPMSamplingConfig()

    config.sampler_config = sampler_config
    return config
