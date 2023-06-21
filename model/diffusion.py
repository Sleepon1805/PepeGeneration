import torch
from lightning import LightningModule

from config import Config


class Diffusion(LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.beta_min = config.beta_min
        self.beta_max = config.beta_max
        self.diffusion_steps = config.diffusion_steps

        self._init_values()

    def _init_values(self):
        beta = torch.linspace(self.beta_min, self.beta_max, self.diffusion_steps)
        alpha = 1 - beta
        alpha_hat = torch.cumprod(alpha, dim=0)

        # to automatically move them to gpu, it basically creates self.beta, ...
        self.register_buffer('beta', beta)
        self.register_buffer('alpha', alpha)
        self.register_buffer('alpha_hat', alpha_hat)

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.diffusion_steps, size=(n,), device=self.device)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t]).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t]).view(-1, 1, 1, 1)
        epsilon = torch.randn_like(x, device=self.device)
        noised_image = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon
        return noised_image, epsilon

    def denoise_images(self, x, t, predicted_noise):
        if t > 1:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)

        alpha = self.alpha[t].view(-1, 1, 1, 1)
        alpha_hat = self.alpha_hat[t].view(-1, 1, 1, 1)
        beta = self.beta[t].view(-1, 1, 1, 1)

        denoised_image = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) \
            + torch.sqrt(beta) * noise
        return denoised_image
