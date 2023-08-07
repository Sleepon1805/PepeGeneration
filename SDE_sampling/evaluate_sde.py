import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from evaluate import load_model_and_config
from SDE_sampling.likelihood import get_likelihood_fn
from SDE_sampling.sde_lib import VESDE, VPSDE, subVPSDE
from SDE_sampling.samplers import (ReverseDiffusionPredictor,
                                   LangevinCorrector,
                                   EulerMaruyamaPredictor,
                                   AncestralSamplingPredictor,
                                   NoneCorrector,
                                   NonePredictor,
                                   AnnealedLangevinDynamics,
                                   get_pc_sampler)


def inference(ckpt_path: Path, on_gpu=True):
    device = 'cuda' if (on_gpu and torch.cuda.is_available()) else 'cpu'

    model, config = load_model_and_config(ckpt_path, device)

    if config.sde_type.lower() == 'vesde':
        raise NotImplementedError
    elif config.sde_type.lower() == 'vpsde':
        sde = VPSDE(beta_min=config.beta_min, beta_max=config.beta_max, N=config.num_scales)
        sampling_eps = 1e-3
    elif config.sde_type.lower() == 'subvpsde':
        sde = subVPSDE(beta_min=config.beta_min, beta_max=config.beta_max, N=config.num_scales)
        sampling_eps = 1e-3
    else:
        raise ValueError

    num_samples = 4
    shape = (num_samples, 3, config.image_size, config.image_size)
    predictor = ReverseDiffusionPredictor
    corrector = LangevinCorrector
    snr = 0.16
    n_steps = 1
    probability_flow = False
    inverse_scaler = (lambda xx: xx)
    sampling_fn = get_pc_sampler(sde, shape, predictor, corrector,
                                 inverse_scaler, snr, n_steps=n_steps,
                                 probability_flow=probability_flow,
                                 continuous=config.continuous_training,
                                 eps=sampling_eps, device=device)

    x, n = sampling_fn(model)
    show_samples(x, config)


def image_grid(x, config):
    size = config.image_size
    channels = 3
    img = x.reshape(-1, size, size, channels)
    w = int(np.sqrt(img.shape[0]))
    img = img.reshape((w, w, size, size, channels)).transpose((0, 2, 1, 3, 4)).reshape((w * size, w * size, channels))
    return img


def show_samples(x, config):
    x = x.permute(0, 2, 3, 1).detach().cpu().numpy()
    img = image_grid(x, config)
    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    dataset_name = 'celeba'
    version = 10
    ckpt = Path(f'../lightning_logs/{dataset_name}/version_{version}/checkpoints/last.ckpt')

    inference(ckpt, on_gpu=True)
