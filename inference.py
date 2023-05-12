import os
import glob
import torch
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm
import torchmetrics

from model.pepe_generator import PepeGenerator
from dataset import PepeDataset
from config import cfg


def inference(version, gif_shape=(4, 4)):
    folder_to_save = f'./lightning_logs/version_{version}/results/'
    if not os.path.exists(folder_to_save):
        os.makedirs(folder_to_save)

    # gen_samples: (num_steps, gif_shape[0]*gif_shape[1], cfg.image_size, cfg.image_size, 3)
    gen_samples = generate_images(version, gif_shape[0]*gif_shape[1])

    save_results(gen_samples, folder_to_save, gif_shape)


def generate_images(version, num_images):
    checkpoint = sorted(glob.glob(f'./lightning_logs/version_{version}/checkpoints/epoch=*.ckpt'))[-1]
    print(f'Loaded checkpoint is {checkpoint}')
    n_hold_final = 10

    model = PepeGenerator.load_from_checkpoint(checkpoint, config=cfg)
    model.eval(), model.freeze()

    # Generate samples from denoising process
    gen_samples = []
    x = torch.randn((num_images, 3, cfg.image_size, cfg.image_size))
    sample_steps = torch.arange(cfg.diffusion_steps - 1, 0, -1)
    for t in tqdm(sample_steps, desc='Generating images'):
        x = model.denoise_sample(x, t)
        if t % 50 == 0:
            gen_samples.append(x)
    for _ in range(n_hold_final):
        gen_samples.append(x)

    gen_samples = torch.stack(gen_samples, dim=0).moveaxis(2, 4).squeeze(-1)

    return gen_samples


def save_results(gen_samples, folder_to_save, gif_shape):
    # save distribution of color values
    values = gen_samples[-1].reshape((-1, 3))
    fig, axs = plt.subplots(1, 3, sharey='all')
    axs[0].hist(values[:, 0], bins=100, color='red')
    axs[0].set_xticks([-1, 0, 1])
    axs[1].hist(values[:, 1], bins=100, color='green')
    axs[1].set_xticks([-1, 0, 1])
    axs[2].hist(values[:, 2], bins=100, color='blue')
    axs[2].set_xticks([-1, 0, 1])
    plt.savefig(folder_to_save + 'distribution.png')

    gen_samples = (gen_samples.clamp(-1, 1) + 1) / 2
    gen_samples = (gen_samples * 255).type(torch.uint8)
    gen_samples = gen_samples.reshape(-1, gif_shape[0], gif_shape[1], cfg.image_size, cfg.image_size, 3)

    gen_samples = stack_samples(gen_samples, 2)
    gen_samples = stack_samples(gen_samples, 2)

    # save resulting pics
    plt.imsave(folder_to_save + '/final_pred.png', gen_samples[-1].numpy())

    # save gif
    imageio.mimsave(folder_to_save + 'pred.gif', list(gen_samples), fps=5)

    print(f'Saved results at {folder_to_save}')


def stack_samples(samples, stack_dim):
    samples = list(torch.split(samples, 1, dim=1))
    for i in range(len(samples)):
        samples[i] = samples[i].squeeze(1)
    return torch.cat(samples, dim=stack_dim)


def calculate_fid_loss(version, num_samples, dataset_name='celeba'):
    fid_metric = torchmetrics.image.fid.FrechetInceptionDistance(normalize=True)

    # real images
    dataset = PepeDataset(dataset_name, config=cfg, augments=None)
    train_set, val_set = torch.utils.data.random_split(dataset, cfg.dataset_split,
                                                       generator=torch.Generator().manual_seed(42))
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=cfg.batch_size, pin_memory=True,
                                             num_workers=0)
    for batch in tqdm(val_loader, desc='Adding real images to FID'):
        fid_metric.update(batch, real=True)

    # generated images
    gen_samples = generate_images(version, num_samples)
    gen_samples = gen_samples.clamp(-1, 1)
    gen_samples = gen_samples[-1].moveaxis(-1, 2).reshape(-1, 3, cfg.image_size, cfg.image_size)
    fid_metric.update(gen_samples, real=False)

    fid_loss = fid_metric.compute()
    print(fid_loss)
    return fid_loss


if __name__ == '__main__':
    model_version = 4

    inference(model_version, gif_shape=(4, 4))
    calculate_fid_loss(model_version, num_samples=20, dataset_name='celeba')
