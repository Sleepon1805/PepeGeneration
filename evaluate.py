import os
import torch
import torchmetrics
from pathlib import Path
import matplotlib.pyplot as plt

from data.dataset import PepeDataset
from model.pepe_generator import PepeGenerator
from data.condition_utils import encode_condition
from config import Paths, Config, load_config, CONDITION_SIZE
from utils.progress_bar import progress_bar


def inference(checkpoint: Path, condition=None, grid_shape=(4, 4), calculate_metrics=False,
              save_images=True, on_gpu=True, seed=42):
    folder_to_save = checkpoint.parents[1].joinpath('results/')
    if save_images and not os.path.exists(folder_to_save):
        os.makedirs(folder_to_save)

    device = 'cuda' if (on_gpu and torch.cuda.is_available()) else 'cpu'

    model, config = load_model_and_config(checkpoint, device)

    # get fake batch with zero'ed images and encoded condition
    fake_batch = create_input_batch(condition, grid_shape[0] * grid_shape[1], config)

    # with progress:
    #     model.sampler.visualize_generation_process(model, fake_batch, progress)

    # generate images: [grid_shape[0] * grid_shape[1] x 3 x cfg.image_size x cfg.image_size]
    gen_samples = model.sampler.generate_samples(model, fake_batch, seed=seed)
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
        calculate_fid_loss(gen_samples, config, device)
        inception_score = model.calculate_inception_score(gen_samples=gen_samples)
        print(f'Inception score: {inception_score}')


def load_model_and_config(checkpoint: Path | str, device: str) -> (PepeGenerator, Config):
    config = load_config(checkpoint, path_to_checkpoint=True)

    model = PepeGenerator.load_from_checkpoint(checkpoint, config=config, strict=False)
    model.eval(), model.freeze(), model.to(device)

    print(f'Loaded model from {checkpoint}')
    return model, config


def create_input_batch(condition, num_samples, config: Config):
    data_config = config.data_config
    if condition is None:
        try:
            print('Got no condition. Taking input images and conditions from first val batch.')
            dataset = PepeDataset(data_config, paths=Paths(), augments=None)
            _, val_set = torch.utils.data.random_split(
                dataset, config.training_config.dataset_split, generator=torch.Generator().manual_seed(137)
            )
            dataloader = torch.utils.data.DataLoader(
                val_set, batch_size=num_samples, pin_memory=True, num_workers=0
            )
            batch = next(iter(dataloader))
            return batch
        except AssertionError:
            print('Simulating input batch and condition with zeros')
            fake_image_batch = torch.zeros((num_samples, 3, data_config.image_size, data_config.image_size))
            fake_cond_batch = torch.ones(num_samples, CONDITION_SIZE)
            fake_batch = (fake_image_batch, fake_cond_batch)
            return fake_batch
    else:
        print(f'Generating images for condition {condition}, input images are set to zeros.')
        encoded_cond = encode_condition(data_config.dataset_name, condition)
        encoded_cond = torch.from_numpy(encoded_cond)
        cond_batch = torch.repeat_interleave(encoded_cond, num_samples, dim=0)

        # fake batch
        fake_image_batch = torch.zeros((num_samples, 3, data_config.image_size, data_config.image_size))
        fake_batch = (fake_image_batch, cond_batch)
        return fake_batch


def calculate_fid_loss(gen_samples, config: Config, device: str):
    data_config = config.data_config
    fid_metric = torchmetrics.image.fid.FrechetInceptionDistance(feature=192, normalize=True).to(device)

    # real images
    dataset = PepeDataset(data_config, paths=Paths(), augments=None)
    _, val_set = torch.utils.data.random_split(
        dataset, config.training_config.dataset_split, generator=torch.Generator().manual_seed(137)
    )
    dataloader = torch.utils.data.DataLoader(
        val_set, batch_size=gen_samples.shape[0], pin_memory=True, num_workers=0
    )
    for samples, cond in progress_bar(dataloader, desc="Adding real images to FID"):
        images = (samples.to(device) + 1) / 2
        fid_metric.update(images, real=True)

    # generated images
    fid_metric.update((gen_samples.clamp(-1, 1) + 1) / 2, real=False)

    fid_loss = fid_metric.compute().item()
    print(f'FID loss: {fid_loss}')
    return fid_loss


if __name__ == '__main__':
    version = 12
    dataset_name = 'celeba'
    ckpt_identifier = 'epoch=00'  # 'last' or 'epoch=00'
    ckpt = next(
        Path(
            f'./lightning_logs/{dataset_name}/version_{version}/checkpoints/'
        ).rglob(f'*{ckpt_identifier}*.ckpt')  # glob search for checkpoint for specific epoch
    )

    inference(
        ckpt,
        condition=None,
        grid_shape=(2, 2),
        calculate_metrics=False,
        save_images=False,
        on_gpu=True,
        seed=137,
    )
