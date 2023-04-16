import os
import torch
from easydict import EasyDict


cfg = EasyDict()

cfg.device = torch.cuda.is_available()

# paths
cfg.pepe_data_path = '/home/sleepon/data/PepeDataset/'
cfg.celeba_data_path = '/home/sleepon/data/CelebFaces/img_align_celeba/img_align_celeba/'
cfg.parsed_datasets = './parsed_data/'

# hparams
cfg.batch_size = 128
cfg.image_size = (32, 32)

# model params
cfg.diffusion_steps = 1000
cfg.init_channels = 64
cfg.time_channels = 128

# training params
cfg.dataset_split = [0.8, 0.2]
