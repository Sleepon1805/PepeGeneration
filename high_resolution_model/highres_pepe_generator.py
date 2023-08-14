import torch
import torch.nn.functional as F
from rich.progress import Progress

from model.unet import UNetModel
from model.pepe_generator import PepeGenerator
from config import Config, HIGHRES_IMAGE_SIZE_MULT


class HighResUNetModel(UNetModel):
    """
    A UNetModel that performs super-resolution.

    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, config: Config):
        super().__init__(config, in_channels=6)

    def forward(self, x, timesteps, low_res=None, cond=None):
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        x = torch.cat([x, upsampled], dim=1)
        out = super().forward(x, timesteps, cond)
        return out


class HighResPepeGenerator(PepeGenerator):
    def __init__(self, config: Config):
        super().__init__(config)
        config.res_factor = HIGHRES_IMAGE_SIZE_MULT  # to save with config
        self.model = HighResUNetModel(config)
        self.example_input_array = (
            (
                torch.Tensor(config.batch_size, 3,
                             config.image_size * HIGHRES_IMAGE_SIZE_MULT, config.image_size * HIGHRES_IMAGE_SIZE_MULT),
                torch.Tensor(config.batch_size, 3, config.image_size, config.image_size),
                torch.ones(config.batch_size, config.condition_size)
            ),
            torch.ones(config.batch_size),
        )

    def forward(self, batch, t):
        x, *labels = batch

        if len(labels) == 1:
            low_res = labels[0]
            cond = torch.bernoulli(torch.full((x.shape[0], self.config.condition_size),
                                              0.5, device=self.device)) * 2 - 1
        elif len(labels) == 2:
            low_res, cond = labels
        else:
            raise ValueError

        return self.model(x, t, low_res=low_res, cond=cond)
