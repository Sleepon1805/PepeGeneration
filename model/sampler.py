import torch
import numpy as np
from typing import Tuple
from abc import abstractmethod
from lightning import LightningModule

from utils.progress_bar import progress_bar
from utils.typings import TrainImagesType, BatchType, BatchedFloatType, ABCTypeChecked


class Sampler(ABCTypeChecked):
    def __init__(self):
        self.device = 'cpu'

    def to(self, device):
        """
        Move all tensors to device.
        :param device: 'cpu', 'cuda' or other torch device
        :return:
        """
        self.device = device
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, torch.Tensor):
                self.__setattr__(attr_name, attr_value.to(device))

    @abstractmethod
    def init_timesteps(self) -> torch.Tensor:
        """
        Initialize time steps for the SDE.
        :return: Tensor of shape (N,)
        """
        pass

    @abstractmethod
    def sample_timesteps(self, n: int) -> torch.Tensor:
        """
        Sample time steps for the SDE.
        :param n: int: number of time steps to sample
        :return: Tensor of shape (n,)
        """
        pass

    @abstractmethod
    def prior_sampling(self, shape: Tuple) -> torch.Tensor:
        """
        Generate one sample from the prior distribution, $p_T(x)$.
        :param shape: shape of the sample
        :return: Tensor of shape `shape`
        """
        pass

    @abstractmethod
    def noise_images(self, images: TrainImagesType, t: BatchedFloatType) -> Tuple[TrainImagesType, TrainImagesType]:
        """
        Add noise to images corresponding to given times.
        :param images: Tensor of shape (N, 3, img_size, img_size)
        :param t: Tensor of shape (N,) or float
        :return: Tuple of two Tensors: noised images and noise, both shapes (N, 3, img_size, img_size)
        """
        pass

    @abstractmethod
    def denoise_step(self, model: LightningModule, batch: BatchType, t: BatchedFloatType) -> TrainImagesType:
        """
        Perform one denoising step.
        :param model: neural network model
        :param batch: BatchType: batch of noised images and labels
        :param t: current time step
        :return: Tensor of shape (N, 3, img_size, img_size): images with noise ~ timestep t-1
        """
        pass

    def generate_samples(self, model: LightningModule, batch: BatchType, seed=42) -> TrainImagesType:
        """
        Run the denoising process from begin to the end. Generation params are stored in the batch:
        batch = (images, *labels), where images only needed for shape, labels are used for conditioning.
        :param model: trained neural network
        :param batch: input batch
        :param seed: seed for random number generator
        :return: Tensor of shape (N, 3, img_size, img_size): generated samples
        """
        torch.manual_seed(seed)
        images_batch, *labels = batch

        # move to device
        for i in range(len(labels)):
            labels[i] = labels[i].to(self.device)

        # init
        x = self.prior_sampling(images_batch.shape)
        timesteps = self.init_timesteps()

        # Generate samples from denoising process
        for t in progress_bar(timesteps, desc=f"Generating {images_batch.shape[0]} images"):
            batch = (x, *labels)
            x = self.denoise_step(model, batch, t)
        return x

    @staticmethod
    def generated_samples_to_images(gen_samples: TrainImagesType, grid_size: Tuple[int, int]) -> np.ndarray:
        """
        Convert generated samples to images. Convert to uint8 and stack images in a grid.
        Only grid_size[0] * grid_size[1] images are taken.
        :param gen_samples: Tensor of shape (N, 3, img_size, img_size) with values (approx.) in [-1, 1]
        :param grid_size: Tuple of two ints: number of images in each row and column
        :return: Tensor of shape (3 * img_size, 3 * img_size, 3): stacked images
        """
        gen_samples = gen_samples[:grid_size[0] * grid_size[1]]

        gen_samples = (gen_samples.clamp(-1, 1) + 1) / 2
        gen_samples = (gen_samples * 255).type(torch.uint8)

        # stack images
        gen_samples = torch.cat(torch.split(gen_samples, grid_size[0], dim=0), dim=2)
        gen_samples = torch.cat(torch.split(gen_samples, 1, dim=0), dim=3)
        gen_samples = gen_samples.squeeze().cpu().numpy()
        gen_samples = np.moveaxis(gen_samples, 0, -1)

        return gen_samples
