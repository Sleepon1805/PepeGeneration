import os
import torch
from easydict import EasyDict


cfg = EasyDict()

cfg.device = torch.cuda.is_available()

# paths
cfg.source_data = '/home/sleepon/data/PepeDataset/'
cfg.dataset = './parsed_data/'

# hparams
cfg.batch_size = 8
cfg.image_size = (128, 128)

# model params
cfg.diffusion_steps = 1000
cfg.init_channels = 32
cfg.time_channels = 64

# training params
cfg.dataset_split = [0.8, 0.2]
