import os
import torch
from easydict import EasyDict


cfg = EasyDict()

cfg.device = torch.cuda.is_available()

# paths
cfg.source_data = '/home/sleepon/data/PepeDataset/'
cfg.dataset = './parsed_data/'

# hparams
cfg.batch_size = 32
cfg.image_size = (32, 32)

# model params
cfg.diffusion_steps = 500
cfg.init_channels = 64
cfg.time_channels = 128

# training params
cfg.dataset_split = [0.8, 0.2]
