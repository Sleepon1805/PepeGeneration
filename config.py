import os
import git
import yaml
from typing import Tuple, Literal
from pathlib import Path
from dataclasses import dataclass

from data.condition_utils import CONDITION_SIZE
from progress_bar import progress_bar
from SDE_sampling.sde_lib import SDEName
from SDE_sampling.predictors_correctors import PredictorName, CorrectorName

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
class Config:
    # git commit hash for logging
    git_hash: str = curr_git_hash()

    # training params
    batch_size: int = 32
    precision: str = 32
    image_size: int = 64  # size of image NxN
    lr: float = 1e-3  # learning rate on training start
    scheduler: str = 'multisteplr'
    gradient_clip_algorithm: str = "norm"
    gradient_clip_val: float = 0.5
    dataset_split: Tuple[float, float] = (0.8, 0.2)
    calculate_fid: bool = False

    # pretrained backbone and current dataset
    dataset_name: Literal['celeba', 'pepe', 'twitch_emotes'] = 'celeba'
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

    # sde params
    sde_name: SDEName = SDEName.VPSDE
    num_scales: int = 1000
    schedule_param_start: float = 0.1  # 0.1 for VPSDE, subVPSDE; 0.01 for VESDE
    schedule_param_end: float = 20.  # 20. for VPSDE, subVPSDE; 50. for VESDE

    # predictor params
    predictor_name: PredictorName = PredictorName.EULER_MARUYAMA
    probability_flow: bool = False

    # corrector params
    corrector_name: CorrectorName = CorrectorName.NONE
    snr: float = 0.01

    # sampler params
    num_corrector_steps: int = 1
    denoise: bool = False


def save_config(config: Config, save_folder: str | Path):
    if isinstance(save_folder, str):
        save_folder = Path(save_folder)

    assert os.path.isdir(save_folder)
    os.makedirs(save_folder, exist_ok=True)

    # save config and sampler config
    with open(save_folder.joinpath('config.yaml'), 'w') as f:
        yaml.dump(config, f)


def load_config(path: str | Path, path_to_checkpoint: bool):
    if isinstance(path, str):
        path = Path(path)
    if path_to_checkpoint:
        path = path.parents[1]

    try:
        with open(path.joinpath('config.yaml'), 'r') as config_file:
            config = yaml.load(config_file, Loader=yaml.Loader)
    except Exception as e:
        print(f'Could not read config from .yaml file: {e}')
        print('Using default Config().')
        config = Config()
    return config