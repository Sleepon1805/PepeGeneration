import os
import subprocess
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Paths:
    pepe_data_path: str = '/home/sleepon/data/PepeDataset/'
    celeba_data_path: str = '/home/sleepon/data/CelebFaces/img_align_celeba/img_align_celeba/'
    twitch_emotes_data_path: str = '/home/sleepon/data/TwitchPepeDataset/'
    parsed_datasets: str = os.path.dirname(__file__) + '/dataset/parsed_data/'


@dataclass
class Config:
    # git commit hash for logging
    git_hash: str = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()

    # hparams
    batch_size: int = 32
    image_size: int = 64  # size of image NxN
    lr: float = 1e-3  # learning rate on training start
    scheduler: str = 'MultiStepLR'
    gradient_clip_algorithm: str = "norm"
    gradient_clip_val: float = 0.5

    # pretrained backbone and current dataset
    dataset_name: str = 'twitch_emotes'
    # pretrained_ckpt: str = None
    pretrained_ckpt: str = './lightning_logs/celeba/version_1/checkpoints/last.ckpt'

    # gaussian noise hparams
    diffusion_steps: int = 1000
    beta_min: float = 1e-4
    beta_max: float = 0.02

    # model params
    init_channels: int = 128
    channel_mult: Tuple[int, int, int, int] = (1, 2, 4, 4)
    conv_resample: bool = True
    num_heads: int = 1
    dropout: float = 0.3

    # training params
    dataset_split: Tuple[float, float] = (0.8, 0.2)
