import os
import glob
import numpy as np
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm

from dataset import PepeDataset
from model.pepe_generator import PepeGenerator
from config import cfg


def stack_samples(samples, stack_dim):
    samples = list(torch.split(samples, 1, dim=1))
    for i in range(len(samples)):
        samples[i] = samples[i].squeeze(1)
    return torch.cat(samples, dim=stack_dim)


if __name__ == '__main__':
    version = 0
    checkpoint = glob.glob(f'./lightning_logs/version_{version}/checkpoints/*.ckpt')[0]
    folder_to_save = f'./lightning_logs/version_{version}/'

    gif_shape = [3, 3]
    sample_batch_size = gif_shape[0] * gif_shape[1]
    n_hold_final = 10
    model = PepeGenerator.load_from_checkpoint(checkpoint, in_size=cfg.image_size[0] * cfg.image_size[1],
                                               t_range=cfg.diffusion_steps, img_depth=3)

    # Generate samples from denoising process
    gen_samples = []
    x = torch.randn((sample_batch_size, 3, *cfg.image_size))
    sample_steps = torch.arange(cfg.diffusion_steps - 1, 0, -1)
    for t in tqdm(sample_steps):
        x = model.denoise_sample(x, t)
        if t % 50 == 0:
            gen_samples.append(x)
    for _ in range(n_hold_final):
        gen_samples.append(x)
    gen_samples = torch.stack(gen_samples, dim=0).moveaxis(2, 4).squeeze(-1)
    gen_samples = (gen_samples.clamp(-1, 1) + 1) / 2

    gen_samples = (gen_samples * 255).type(torch.uint8)
    gen_samples = gen_samples.reshape(-1, gif_shape[0], gif_shape[1], *cfg.image_size, 3)

    gen_samples = stack_samples(gen_samples, 2)
    gen_samples = stack_samples(gen_samples, 2)

    # save resulting pics
    plt.imsave(folder_to_save + '/final_pred.png', gen_samples[-1].numpy())

    # save distribution of color values
    plt.hist(gen_samples[-1].flatten().numpy(), bins=100)
    plt.savefig(folder_to_save + 'distribution.png')

    # save gif
    imageio.mimsave(folder_to_save + 'pred.gif', list(gen_samples), fps=5)


