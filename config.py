from easydict import EasyDict
import subprocess

cfg = EasyDict()

# paths
cfg.pepe_data_path = '/home/sleepon/data/PepeDataset/'
cfg.celeba_data_path = '/home/sleepon/data/CelebFaces/img_align_celeba/img_align_celeba/'
cfg.parsed_datasets = './parsed_data/'

# git commit hash for logging
cfg.git_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()

# hparams
cfg.batch_size = 64
cfg.image_size = 64  # size of image NxN
cfg.lr = 1e-4  # learning rate on training start
cfg.scheduler_name = 'None'
# cfg.scheduler_params = {'factor': 0.1, 'patience': 4, 'min_lr': 1e-6}
cfg.pretrained_ckpt = None
# cfg.pretrained_ckpt = './lightning_logs/version_11/checkpoints/epoch=16-val_loss=0.0242.ckpt'

# gaussian noise hparams
cfg.diffusion_steps = 1000
cfg.beta_min = 1e-4
cfg.beta_max = 0.02

# model params
cfg.init_channels = 256
cfg.channel_mult = (1, 1, 2, 2)
cfg.conv_resample = True
cfg.num_heads = 1
cfg.dropout = 0

# training params
cfg.dataset_split = [0.8, 0.2]
