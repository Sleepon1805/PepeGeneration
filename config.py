import os
import subprocess
from dataclasses import dataclass
from typing import Tuple
from types import MappingProxyType


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
    lr: float = 1e-4  # learning rate on training start
    scheduler: MappingProxyType = None
    # scheduler: MappingProxyType = MappingProxyType(
    #     {'name': 'ReduceLROnPlateau', 'params': {'factor': 0.1, 'patience': 4, 'min_lr': 1e-6}}
    # )
    gradient_clip_algorithm: str = "norm"
    gradient_clip_val: float = 0.5

    # pretrained backbone and current dataset
    dataset_name: str = 'celeba'
    pretrained_ckpt: str = None
    # pretrained_ckpt: str = './lightning_logs/version_0/checkpoints/epoch=11-val_loss=0.0244.ckpt'

    # gaussian noise hparams
    diffusion_steps: int = 1000
    beta_min: float = 1e-4
    beta_max: float = 0.02

    # model params
    init_channels: int = 256
    channel_mult: Tuple[int, int, int, int] = (1, 1, 2, 2)
    conv_resample: bool = True
    num_heads: int = 1
    dropout: float = 0

    # training params
    dataset_split: Tuple[float, float] = (0.8, 0.2)
