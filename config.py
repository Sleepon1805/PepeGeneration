import os
import git
from dataclasses import dataclass
from typing import Tuple

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
class Config:
    # git commit hash for logging
    git_hash: str = curr_git_hash()

    # training params
    batch_size: int = 64
    image_size: int = 64  # size of image NxN
    lr: float = 1e-4  # learning rate on training start
    scheduler: str = 'MultiStepLR'
    gradient_clip_algorithm: str = "norm"
    gradient_clip_val: float = 0.5
    dataset_split: Tuple[float, float] = (0.8, 0.2)

    # pretrained backbone and current dataset
    dataset_name: str = 'celeba'
    pretrained_ckpt: str = None
    # pretrained_ckpt: str = './lightning_logs/celeba/version_6/checkpoints/last.ckpt'

    # gaussian noise hparams
    diffusion_steps: int = 4000
    beta_min: float = 1e-4
    beta_max: float = 0.02

    # model params
    init_channels: int = 128
    channel_mult: Tuple[int, int, int, int] = (1, 2, 4, 4)
    conv_resample: bool = True
    num_heads: int = 1
    dropout: float = 0.3
    use_second_attention: bool = True
