import os
import glob
import torch
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm
import torchmetrics
import yaml

from model.pepe_generator import PepeGenerator
from dataset.dataset import PepeDataset
from config import Paths, Config


def inference(version, gif_shape=(4, 4)):
    folder_to_save = f'./lightning_logs/version_{version}/results/'
    if not os.path.exists(folder_to_save):
        os.makedirs(folder_to_save)

    # gen_samples: (num_steps, gif_shape[0]*gif_shape[1], cfg.image_size, cfg.image_size, 3)
    gen_samples = generate_images(version, gif_shape[0]*gif_shape[1])

    save_results(gen_samples, folder_to_save)


def generate_images(version, num_images):
    checkpoint = sorted(glob.glob(f'./lightning_logs/version_{version}/checkpoints/epoch=*.ckpt'))[-1]
    print(f'Loaded checkpoint is {checkpoint}')

    with open(f'./lightning_logs/version_{version}/hparams.yaml', "r") as hparams:
        try:
            config = yaml.safe_load(hparams)
        except:
            print('Could not read config from .yaml file. Using default Config().')
            config = Config()

    model = PepeGenerator.load_from_checkpoint(checkpoint, config=config)
    model.eval(), model.freeze()

    # Generate samples from denoising process
    gen_samples = []
    x = torch.randn((num_images, 3, config.image_size, config.image_size))
    sample_steps = torch.arange(config.diffusion_steps - 1, 0, -1)
    for t in tqdm(sample_steps, desc='Generating images'):
        x = model.denoise_sample(x, t)
        if (t % 50 == 0) or (t == sample_steps[-1]):
            gen_samples.append(x)

    gen_samples = torch.stack(gen_samples, dim=0).moveaxis(2, 4).squeeze(-1)

    return gen_samples


def save_results(gen_samples: torch.Tensor, folder_to_save):
    # save distribution of color values
    fig, axs = plt.subplots(1, 3, sharey='all')
    for i in range(gen_samples.shape[0]):
        values = gen_samples[i].reshape((-1, 3))
        axs[0].hist(values[:, 0], bins=100, histtype='step', )
        axs[1].hist(values[:, 1], bins=100, histtype='step')
        axs[2].hist(values[:, 2], bins=100, histtype='step')
    axs[0].set_xticks([-1, 0, 1])
    axs[0].set_title('red')
    axs[1].set_xticks([-1, 0, 1])
    axs[1].set_title('green')
    axs[2].set_xticks([-1, 0, 1])
    axs[2].set_title('blue')
    plt.savefig(folder_to_save + 'distribution.png')

    gen_samples = (gen_samples.clamp(-1, 1) + 1) / 2
    gen_samples = (gen_samples * 255).type(torch.uint8)

    # stack images
    gen_samples = torch.cat(torch.split(gen_samples, 4, dim=1), dim=2)
    gen_samples = torch.cat(torch.split(gen_samples, 1, dim=1), dim=3).squeeze()

    # save resulting pics
    plt.imsave(folder_to_save + '/final_pred.png', gen_samples[-1].numpy())

    # save gif
    imageio.mimsave(folder_to_save + 'pred.gif', list(gen_samples), fps=5)

    print(f'Saved results at {folder_to_save}')


def calculate_fid_loss(version, num_samples, config: Config, dataset_name='celeba'):
    fid_metric = torchmetrics.image.fid.FrechetInceptionDistance(normalize=True)

    # real images
    dataset = PepeDataset(dataset_name, paths=Paths(), augments=None)
    train_set, val_set = torch.utils.data.random_split(dataset, config.dataset_split,
                                                       generator=torch.Generator().manual_seed(42))
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=config.batch_size, pin_memory=True,
                                             num_workers=0)
    for batch in tqdm(val_loader, desc='Adding real images to FID'):
        fid_metric.update(batch, real=True)

    # generated images
    gen_samples = generate_images(version, num_samples)
    gen_samples = gen_samples.clamp(-1, 1)
    gen_samples = gen_samples[-1].moveaxis(-1, 2).reshape(-1, 3, config.image_size, config.image_size)
    fid_metric.update(gen_samples, real=False)

    fid_loss = fid_metric.compute()
    print(fid_loss)
    return fid_loss


if __name__ == '__main__':
    model_version = 12

    inference(model_version, gif_shape=(4, 4))
    # calculate_fid_loss(model_version, num_samples=20, config=Config(), dataset_name='celeba')
