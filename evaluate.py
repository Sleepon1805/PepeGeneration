import os
import glob
import torch
import pickle
import torchmetrics
from pathlib import Path
import matplotlib.pyplot as plt
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, BarColumn, TextColumn, TimeRemainingColumn, \
    MofNCompleteColumn

from config import Paths, Config, SamplingConfig
from data.dataset import PepeDataset
from model.pepe_generator import PepeGenerator
from data.condition_utils import encode_condition

from SDE_sampling.sde_samplers import PC_Sampler, ODE_Sampler


def inference(checkpoint: Path, condition=None, grid_shape=(4, 4), calculate_metrics=False,
              save_images=True, on_gpu=True):
    folder_to_save = checkpoint.parents[1].joinpath('results/')
    if not os.path.exists(folder_to_save):
        os.makedirs(folder_to_save)

    device = 'cuda' if (on_gpu and torch.cuda.is_available()) else 'cpu'

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

    # set up sampler
    sampling_config = SamplingConfig()
    if sampling_config.sampler.lower() in ('ddpm', 'default'):
        print('Using default DDPM Sampler as evaluation sampler.')
    elif sampling_config.sampler.lower() == 'ode_solver':
        print('Using ODE Solver as evaluation sampler.')
        model.sampler = ODE_Sampler(config, sampling_config)
    elif sampling_config.sampler.lower() == 'pc_sampler':
        print('Using PC Sampler as evaluation sampler.')
        model.sampler = PC_Sampler(config, sampling_config)
    else:
        raise ValueError
    model.sampler.to(device)

    # get fake batch with zero'ed images and encoded condition
    fake_batch = create_input_batch(condition, grid_shape[0] * grid_shape[1], config)

    # with progress:
    #     model.sampler.visualize_generation_process(model, fake_batch, progress)

    # generate images
    with progress:
        # [grid_shape[0] * grid_shape[1] x 3 x cfg.image_size x cfg.image_size]
        gen_samples = model.generate_samples(fake_batch, progress=progress)
    gen_images = model.sampler.generated_samples_to_images(gen_samples, grid_shape)

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
        plt.imsave(folder_to_save.joinpath(f'final_pred.png'), gen_images)
        print(f'Saved results at {folder_to_save}')
    plt.title(condition)
    plt.show()

    if calculate_metrics:
        with progress:
            calculate_fid_loss(gen_samples, config, device, progress=progress)
        inception_score = model.calculate_inception_score(gen_samples=gen_samples)
        print(f'Inception score: {inception_score}')


def load_model_and_config(checkpoint: Path | str, device: str) -> (PepeGenerator, Config):
    print(f'Loaded model from {checkpoint}')
    if isinstance(checkpoint, str):
        checkpoint = Path(checkpoint)

    with open(checkpoint.parents[1].joinpath('config.pkl'), 'rb') as config_file:
        try:
            config = pickle.load(config_file)
        except:
            print('Could not read config from .yaml file. Using default Config().')
            config = Config()

    model = PepeGenerator.load_from_checkpoint(checkpoint, config=config, strict=False)
    model.eval(), model.freeze(), model.to(device)

    return model, config


def create_input_batch(condition, num_samples, config):
    if condition is None:
        print('Got no condition. Taking input images and conditions from first val batch.')
        dataset = PepeDataset(config.dataset_name, config.image_size, paths=Paths(), augments=None)
        train_set, val_set = torch.utils.data.random_split(dataset, config.dataset_split,
                                                           generator=torch.Generator().manual_seed(137))
        loader = torch.utils.data.DataLoader(val_set, batch_size=num_samples, pin_memory=True,
                                             num_workers=0)
        batch = next(iter(loader))
        return batch
    else:
        print(f'Generating images for condition {condition}, input images are set to zeros.')
        encoded_cond = encode_condition(config.dataset_name, condition)
        encoded_cond = torch.from_numpy(encoded_cond)
        cond_batch = torch.repeat_interleave(encoded_cond, num_samples, dim=0)

        # fake batch
        fake_image_batch = torch.zeros((num_samples, 3, config.image_size, config.image_size))
        fake_batch = (fake_image_batch, cond_batch)
        return fake_batch


def calculate_fid_loss(gen_samples, config: Config, device: str, progress: Progress):
    fid_metric = torchmetrics.image.fid.FrechetInceptionDistance(feature=192, normalize=True).to(device)

    # real images
    dataset = PepeDataset(config.dataset_name, config.image_size, paths=Paths(), augments=None)
    train_set, val_set = torch.utils.data.random_split(dataset, config.dataset_split,
                                                       generator=torch.Generator().manual_seed(137))
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=gen_samples.shape[0], pin_memory=True,
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
    version = 11
    ckpt = Path(sorted(glob.glob(f'/home/sleepon/Downloads/version_11/checkpoints/epoch=07-fid_metric=1.83-val_loss=0.0254.ckpt'))[-1])

    inference(
        ckpt,
        condition=None,
        grid_shape=(3, 3),
        calculate_metrics=False,
        save_images=False,
        on_gpu=True,
    )
