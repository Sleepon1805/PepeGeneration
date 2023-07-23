import os
import glob
import torch
import matplotlib.pyplot as plt
import torchmetrics
import pickle
from pathlib import Path
from typing import List
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, BarColumn, TextColumn, TimeRemainingColumn, \
    MofNCompleteColumn

from model.pepe_generator import PepeGenerator
from dataset.dataset import PepeDataset
from config import Paths, Config


def inference(checkpoint: Path, grid_shape=(4, 4), calculate_fid: bool = False, on_gpu: bool = True):
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

    with progress:
        # [grid_shape[0] * grid_shape[1] x 3 x cfg.image_size x cfg.image_size]
        gen_samples = model.generate_samples(grid_shape[0] * grid_shape[1], progress=progress)

    # save resulting pics
    fig, ax = plt.subplots(3, sharex='all', sharey='all', figsize=(10, 5))
    sampled_data = gen_samples.moveaxis(1, 0).flatten(1).cpu()
    ax[0].hist(sampled_data[0], bins=100, color='red')
    ax[1].hist(sampled_data[1], bins=100, color='green')
    ax[2].hist(sampled_data[2], bins=100, color='blue')
    plt.savefig(folder_to_save.joinpath('distribution.png'))
    plt.show()

    gen_images = model.generated_samples_to_images(gen_samples, grid_shape)

    plt.imshow(gen_images)
    plt.imsave(folder_to_save.joinpath('final_pred.png'), gen_images)
    plt.show()
    print(f'Saved results at {folder_to_save}')

    if calculate_fid:
        with progress:
            calculate_fid_loss(gen_samples, config, device, progress=progress)


def load_model_and_config(checkpoint: Path, device: str):
    print(f'Loaded checkpoint is {checkpoint}')

    with open(checkpoint.parents[1].joinpath('config.pkl'), 'rb') as config_file:
        try:
            config = pickle.load(config_file)
        except:
            print('Could not read config from .yaml file. Using default Config().')
            config = Config()

    model = PepeGenerator.load_from_checkpoint(checkpoint, config=config)
    model.eval(), model.freeze(), model.to(device)

    return model, config


def calculate_fid_loss(gen_samples, config: Config, device: str, progress: Progress):
    fid_metric = torchmetrics.image.fid.FrechetInceptionDistance(feature=192, normalize=True).to(device)

    # real images
    dataset = PepeDataset(config.dataset_name, paths=Paths(), augments=None)
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


def gen_images_with_condition(checkpoint: Path, features: List[str], grid_size=(3, 3), on_gpu: bool = True):
    if on_gpu and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

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

    # condition
    cond = torch.full((grid_size[0] * grid_size[1], 40), -1, device=device, dtype=torch.float32)
    for feature in features:
        if feature in all_cond_features.keys():
            cond[:, all_cond_features[feature]] = 1

    with progress:
        # [grid_shape[0] * grid_shape[1] x 3 x cfg.image_size x cfg.image_size]
        gen_samples = model.generate_samples(grid_size[0] * grid_size[1], cond=cond, progress=progress, seed=322)

    gen_images = model.generated_samples_to_images(gen_samples, grid_size)

    plt.imshow(gen_images)
    plt.title(features)
    plt.show()
    print(f'Generated image for features = {features}')


if __name__ == '__main__':
    dataset = 'celeba'
    version = 5
    ckpt = Path(sorted(glob.glob(f'./lightning_logs/{dataset}/version_{version}/checkpoints/last.ckpt'))[-1])

    # inference(ckpt, calculate_fid=True, grid_shape=(16, 16), on_gpu=True)

    for condition in [[], ["Black_Hair"], ["Blond_Hair"], ["Black_Hair", "Blond_Hair"]]:
        gen_images_with_condition(checkpoint=ckpt, features=condition, on_gpu=True)
