import os
import glob
import torch
import matplotlib.pyplot as plt
import torchmetrics
import pickle
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, BarColumn, TextColumn, TimeRemainingColumn, \
    MofNCompleteColumn

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

    # rich progress bar
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn()
    )

    model, config = load_model_and_config(version, device)

    with progress:
        # [grid_shape[0] * grid_shape[1] x 3 x cfg.image_size x cfg.image_size]
        gen_samples = model.generate_samples(grid_shape[0] * grid_shape[1], progress=progress)
        gen_images = model.generated_samples_to_images(gen_samples, grid_shape)

        if calculate_fid:
            calculate_fid_loss(gen_samples, config, device, progress=progress)

    # save resulting pics
    fig, ax = plt.subplots(3, sharey='all', figsize=(10, 5))
    ax[0].hist(gen_images.reshape((-1, 3))[:, 0], bins=100, color='red')
    ax[1].hist(gen_images.reshape((-1, 3))[:, 1], bins=100, color='green')
    ax[2].hist(gen_images.reshape((-1, 3))[:, 2], bins=100, color='blue')
    plt.savefig(folder_to_save + '/distribution.png')
    plt.show()

    plt.imshow(gen_images)
    plt.imsave(folder_to_save + '/final_pred.png', gen_images)
    plt.show()
    print(f'Saved results at {folder_to_save}')


def load_model_and_config(version, device: str):
    checkpoint = sorted(glob.glob(f'./lightning_logs/version_{version}/checkpoints/epoch=*.ckpt'))[-1]
    print(f'Loaded checkpoint is {checkpoint}')

    with open(f'./lightning_logs/version_{version}/config.pkl', 'rb') as config_file:
        try:
            config = pickle.load(config_file)
        except:
            print('Could not read config from .yaml file. Using default Config().')
            config = Config()

    model = PepeGenerator.load_from_checkpoint(checkpoint, config=config)
    model.eval(), model.freeze(), model.to(device)

    return model, config


def calculate_fid_loss(gen_samples, config: Config, device: str, progress: Progress):
    fid_metric = torchmetrics.image.fid.FrechetInceptionDistance(normalize=True).to(device)

    # real images
    dataset = PepeDataset(config.dataset_name, paths=Paths(), augments=None)
    train_set, val_set = torch.utils.data.random_split(dataset, config.dataset_split,
                                                       generator=torch.Generator().manual_seed(137))
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=config.batch_size, pin_memory=True,
                                             num_workers=0)
    with progress:
        progress_bar_task = progress.add_task("[white]Adding real images to FID", total=len(val_loader))
        for batch in val_loader:
            progress.update(progress_bar_task, advance=1)
            fid_metric.update(batch.to(device), real=True)

    # generated images
    fid_metric.update(gen_samples, real=False)

    fid_loss = fid_metric.compute().item()
    print(f'FID loss: {fid_loss}')
    return fid_loss


if __name__ == '__main__':
    model_version = 0

    inference(model_version, calculate_fid=False, grid_shape=(4, 4), on_gpu=True)
    # inference(model_version, calculate_fid=True, grid_shape=(2, 2), on_gpu=True)
