import os
import glob
import torch
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm

from model.pepe_generator import PepeGenerator
from config import cfg


def stack_samples(samples, stack_dim):
    samples = list(torch.split(samples, 1, dim=1))
    for i in range(len(samples)):
        samples[i] = samples[i].squeeze(1)
    return torch.cat(samples, dim=stack_dim)


if __name__ == '__main__':
    version = 1
    checkpoint = glob.glob(f'./lightning_logs/version_{version}/checkpoints/epoch=*.ckpt')[-1]
    print(f'Loaded checkpoint is {checkpoint}')
    folder_to_save = f'./lightning_logs/version_{version}/results/'
    if not os.path.exists(folder_to_save):
        os.makedirs(folder_to_save)

    gif_shape = [4, 4]
    sample_batch_size = gif_shape[0] * gif_shape[1]
    n_hold_final = 10
    model = PepeGenerator.load_from_checkpoint(checkpoint, config=cfg)
    model.eval(), model.freeze()

    # Generate samples from denoising process
    gen_samples = []
    x = torch.randn((sample_batch_size, 3, cfg.image_size, cfg.image_size))
    sample_steps = torch.arange(cfg.diffusion_steps - 1, 0, -1)
    for t in tqdm(sample_steps):
        x = model.denoise_sample(x, t)
        if t % 50 == 0:
            gen_samples.append(x)
    for _ in range(n_hold_final):
        gen_samples.append(x)

    # save distribution of color values
    values = gen_samples[-1].moveaxis(2, -1).reshape((-1, 3))
    fig, axs = plt.subplots(1, 3, sharey='all')
    axs[0].hist(values[:, 0], bins=100, color='red')
    axs[0].set_xticks([-1, 0, 1])
    axs[1].hist(values[:, 1], bins=100, color='green')
    axs[1].set_xticks([-1, 0, 1])
    axs[2].hist(values[:, 2], bins=100, color='blue')
    axs[2].set_xticks([-1, 0, 1])
    plt.savefig(folder_to_save + 'distribution.png')

    gen_samples = torch.stack(gen_samples, dim=0).moveaxis(2, 4).squeeze(-1)
    gen_samples = (gen_samples.clamp(-1, 1) + 1) / 2

    gen_samples = (gen_samples * 255).type(torch.uint8)
    gen_samples = gen_samples.reshape(-1, gif_shape[0], gif_shape[1], cfg.image_size, cfg.image_size, 3)

    gen_samples = stack_samples(gen_samples, 2)
    gen_samples = stack_samples(gen_samples, 2)

    # save resulting pics
    plt.imsave(folder_to_save + '/final_pred.png', gen_samples[-1].numpy())

    # save gif
    imageio.mimsave(folder_to_save + 'pred.gif', list(gen_samples), fps=5)


