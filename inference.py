import os
import glob

import numpy as np
import torch
import matplotlib.pyplot as plt
import torchmetrics
import pickle
from pathlib import Path
from typing import List
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, BarColumn, TextColumn, TimeRemainingColumn, \
    MofNCompleteColumn

from model.pepe_generator import PepeGenerator
from data.dataset import PepeDataset
from config import Paths, Config


def inference(checkpoint: Path, condition=None, grid_shape=(4, 4), calculate_fid=False, save_images=True, on_gpu=True):
    folder_to_save = checkpoint.parents[1].joinpath('results/')
    if not os.path.exists(folder_to_save):
        os.makedirs(folder_to_save)

    if on_gpu and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # rich progress bar
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn()
    )

    model, config = load_model_and_config(checkpoint, device)

    # decode conditions
    if dataset_name == 'celeba':
        cond_size = (grid_shape[0] * grid_shape[1], config.condition_size)
    elif dataset_name == 'pepe':
        cond_size = (grid_shape[0] * grid_shape[1], 26, config.condition_size)
    else:
        raise ValueError
    if condition is not None:
        cond_str = '-'.join(condition)
        cond_batch = decode_condition(config.dataset_name, condition, cond_size, device)
        print(f'Generating images for condition {cond_str}')
    else:
        cond_str = 'RNG'
        cond_batch = torch.bernoulli(torch.full(cond_size, 0.5, device=device)) * 2 - 1

    # fake batch
    fake_image_batch = torch.zeros((grid_shape[0] * grid_shape[1], 3, config.image_size, config.image_size))
    fake_batch = (fake_image_batch, cond_batch)

    # generate images
    with progress:
        # [grid_shape[0] * grid_shape[1] x 3 x cfg.image_size x cfg.image_size]
        gen_samples = model.generate_samples(fake_batch, progress=progress)
    gen_images = model.generated_samples_to_images(gen_samples, grid_shape)

    # save distributions of generated samples
    if save_images:
        fig, ax = plt.subplots(3, sharex='all', sharey='all', figsize=(10, 5))
        sampled_data = gen_samples.moveaxis(1, 0).flatten(1).cpu()
        ax[0].hist(sampled_data[0], bins=100, color='red')
        ax[1].hist(sampled_data[1], bins=100, color='green')
        ax[2].hist(sampled_data[2], bins=100, color='blue')
        # plt.savefig(folder_to_save.joinpath(f'distribution-{cond_str}.png'))
        plt.show()

    # save generated images
    plt.imshow(gen_images)
    if save_images:
        plt.imsave(folder_to_save.joinpath(f'final_pred-{cond_str}.png'), gen_images)
    plt.title(cond_str)
    plt.show()
    print(f'Saved results at {folder_to_save}')

    if calculate_fid:
        with progress:
            calculate_fid_loss(gen_samples, config, device, progress=progress)


def load_model_and_config(checkpoint: Path, device: str):
    print(f'Loaded model from {checkpoint}')

    with open(checkpoint.parents[1].joinpath('config.pkl'), 'rb') as config_file:
        try:
            config = pickle.load(config_file)
        except:
            print('Could not read config from .yaml file. Using default Config().')
            config = Config()

    model = PepeGenerator.load_from_checkpoint(checkpoint, config=config)
    model.eval(), model.freeze(), model.to(device)

    return model, config


def decode_condition(dataset: str, condition, cond_size, device) -> torch.Tensor:
    if dataset == 'pepe':
        assert isinstance(condition, str)
        name = ''.join(i.lower() for i in condition if i.isalpha())  # take only letters in lower case
        enum_letter = (lambda s: ord(s) - 97)  # enumerate lower case letters from 0 to 25
        decoded_cond = np.zeros(cond_size)
        for i, letter in enumerate(name):
            decoded_cond[enum_letter(letter), i] = 1
    elif dataset == 'celeba':
        assert isinstance(condition, list)
        all_cond_features = {
            "5_o_Clock_Shadow": 0,
            "Arched_Eyebrows": 1,
            "Attractive": 2,
            "Bags_Under_Eyes": 3,
            "Bald": 4,
            "Bangs": 5,
            "Big_Lips": 6,
            "Big_Nose": 7,
            "Black_Hair": 8,
            "Blond_Hair": 9,
            "Blurry": 10,
            "Brown_Hair": 11,
            "Bushy_Eyebrows": 12,
            "Chubby": 13,
            "Double_Chin": 14,
            "Eyeglasses": 15,
            "Goatee": 16,
            "Gray_Hair": 17,
            "Heavy_Makeup": 18,
            "High_Cheekbones": 19,
            "Male": 20,
            "Mouth_Slightly_Open": 21,
            "Mustache": 22,
            "Narrow_Eyes": 23,
            "No_Beard": 24,
            "Oval_Face": 25,
            "Pale_Skin": 26,
            "Pointy_Nose": 27,
            "Receding_Hairline": 28,
            "Rosy_Cheeks": 29,
            "Sideburns": 30,
            "Smiling": 31,
            "Straight_Hair": 32,
            "Wavy_Hair": 33,
            "Wearing_Earrings": 34,
            "Wearing_Hat": 35,
            "Wearing_Lipstick": 36,
            "Wearing_Necklace": 37,
            "Wearing_Necktie": 38,
            "Young": 39,
        }

        decoded_cond = torch.full(cond_size, -1, device=device, dtype=torch.float32)
        for feature in condition:
            if feature in all_cond_features.keys():
                decoded_cond[..., all_cond_features[feature]] = 1
    else:
        raise ValueError

    return decoded_cond


def calculate_fid_loss(gen_samples, config: Config, device: str, progress: Progress):
    fid_metric = torchmetrics.image.fid.FrechetInceptionDistance(feature=192, normalize=True).to(device)

    # real images
    dataset = PepeDataset(config.dataset_name, config.image_size, paths=Paths(), augments=None)
    train_set, val_set = torch.utils.data.random_split(dataset, config.dataset_split,
                                                       generator=torch.Generator().manual_seed(137))
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=config.batch_size, pin_memory=True,
                                             num_workers=0)
    with progress:
        progress_bar_task = progress.add_task("[white]Adding real images to FID", total=len(val_loader))
        for samples, cond in val_loader:
            progress.update(progress_bar_task, advance=1)
            images = (samples.to(device) + 1) / 2
            fid_metric.update(images, real=True)

    # generated images
    fid_metric.update((gen_samples.clamp(-1, 1) + 1) / 2, real=False)

    fid_loss = fid_metric.compute().item()
    print(f'FID loss: {fid_loss}')
    return fid_loss


if __name__ == '__main__':
    dataset_name = 'celeba'
    version = 10
    ckpt = Path(sorted(glob.glob(f'./lightning_logs/{dataset_name}/version_{version}/checkpoints/last.ckpt'))[-1])

    inference(
        ckpt,
        condition=None,
        grid_shape=(4, 4),
        calculate_fid=False,
        save_images=True,
        on_gpu=True
    )

    # for features in [[], ["Male"], ["Eyeglasses"], ["Male", "Eyeglasses"]]:
    #     inference(
    #         ckpt,
    #         condition=features,
    #         grid_shape=(3, 3),
    #         calculate_fid=True,
    #         save_images=False,
    #         on_gpu=True
    #     )
