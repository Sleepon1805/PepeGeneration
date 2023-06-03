import os
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
from rich.progress import track
import torchmetrics
import yaml

from model.pepe_generator import PepeGenerator
from dataset.dataset import PepeDataset
from config import Paths, Config


def inference(version, grid_shape=(4, 4), calculate_fid: bool = False, on_gpu: bool = True):
    folder_to_save = f'./lightning_logs/version_{version}/results/'
    if not os.path.exists(folder_to_save):
        os.makedirs(folder_to_save)

    if on_gpu and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    model, config = load_model_and_config(version, device)

    # gen_samples: torch.Tensor (grid_shape[0] * grid_shape[1], 3, cfg.image_size, cfg.image_size)
    gen_samples = model.generate_samples(grid_shape[0] * grid_shape[1])
    gen_images = model.generated_samples_to_images(gen_samples, grid_shape)

    if calculate_fid:
        calculate_fid_loss(gen_samples, config, device)

    save_results(gen_samples, gen_images, folder_to_save)


def load_model_and_config(version, device: str):
    checkpoint = sorted(glob.glob(f'./lightning_logs/version_{version}/checkpoints/epoch=*.ckpt'))[-1]
    print(f'Loaded checkpoint is {checkpoint}')

    with open(f'./lightning_logs/version_{version}/hparams.yaml', "r") as hparams:
        try:
            config = yaml.safe_load(hparams)
        except:
            print('Could not read config from .yaml file. Using default Config().')
            config = Config()

    model = PepeGenerator.load_from_checkpoint(checkpoint, config=config)
    model.eval(), model.freeze(), model.to(device)

    return model, config


def save_results(gen_samples: torch.Tensor, gen_images: np.ndarray, folder_to_save):
    # save distribution of color values
    fig, axs = plt.subplots(1, 3, sharey='all')
    axs[0].hist(gen_samples[..., 0], bins=100, histtype='step')
    axs[0].set_xticks([-1, 0, 1])
    axs[0].set_title('blue')
    axs[1].hist(gen_samples[..., 1], bins=100, histtype='step')
    axs[1].set_xticks([-1, 0, 1])
    axs[1].set_title('green')
    axs[2].hist(gen_samples[..., 2], bins=100, histtype='step')
    axs[2].set_xticks([-1, 0, 1])
    axs[2].set_title('red')
    plt.savefig(folder_to_save + 'distribution.png')

    # save resulting pics
    plt.imsave(folder_to_save + '/final_pred.png', gen_images)

    print(f'Saved results at {folder_to_save}')


def calculate_fid_loss(gen_samples, config: Config, device: str):
    fid_metric = torchmetrics.image.fid.FrechetInceptionDistance(normalize=True)

    # real images
    dataset = PepeDataset(config.dataset_name, paths=Paths(), augments=None)
    train_set, val_set = torch.utils.data.random_split(dataset, config.dataset_split,
                                                       generator=torch.Generator().manual_seed(42))
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=config.batch_size, pin_memory=True,
                                             num_workers=0, device=device)
    for batch in track(val_loader, description='Adding real images to FID'):
        fid_metric.update(batch, real=True)

    # generated images
    fid_metric.update(gen_samples, real=False)

    fid_loss = fid_metric.compute()
    print(fid_loss)
    return fid_loss


if __name__ == '__main__':
    model_version = 14

    inference(model_version, calculate_fid=False, grid_shape=(4, 4), on_gpu=True)
    # inference(model_version, calculate_fid=True, grid_shape=(8, 8), on_gpu=True)
