import torch
import torch.nn as nn
from lightning import LightningModule

from model.modules import SinTimestepEmbedding
from config import CONDITION_SIZE, ModelConfig
from utils.typings import BatchedFloatType, ConditionType


class EmbeddingModel(LightningModule):
    """
    Model to convert additional information (time, condition) into embeddings.
    """
    def __init__(
        self,
        model_config: ModelConfig,
    ):
        super().__init__()
        self.in_channels: int = model_config.init_channels
        self.out_channels: int = model_config.embed_channels
        self.use_condition: bool = model_config.use_condition

        if self.use_condition:
            print('Using condition in model')
            self.drift_channels = self.out_channels // 4
            self.diffusion_channels = self.out_channels // 4
            self.cond_channels = self.out_channels // 2

            self.drift_embed = TimeEmbedding(self.in_channels, self.drift_channels)
            self.diffusion_embed = TimeEmbedding(self.in_channels, self.diffusion_channels)
            self.condition_embed = ConditionEmbedding(self.in_channels, self.cond_channels, CONDITION_SIZE)
        else:
            print('Not using condition in model')
            self.drift_embed_dim = self.out_channels // 2
            self.diffusion_channels = self.out_channels // 2
            self.cond_channels = 0

            self.drift_embed = TimeEmbedding(self.in_channels, self.drift_channels)
            self.diffusion_embed = TimeEmbedding(self.in_channels, self.diffusion_channels)
            self.condition_embed = None

        assert self.drift_channels + self.diffusion_channels + self.cond_channels == self.out_channels, \
            'Something went wrong with splitting the embedding channels'

    def forward(
        self,
        drift_scale: BatchedFloatType,
        diffusion_scale: BatchedFloatType,
        cond: ConditionType = None
    ) -> torch.Tensor:
        """
        Apply the model to an input batch.

        :param drift_scale: torch.Tensor with shape (B)
        :param diffusion_scale: torch.Tensor with shape (B)
        :param cond: torch.Tensor with shape (B, C_cond) or None
        :return: output - torch.Tensor with shape (B, C_embed)
        """

        assert not self.use_condition or cond is not None, 'Model requires condition input'

        drift_emb = self.drift_embed(drift_scale)
        diffusion_emb = self.diffusion_embed(diffusion_scale)
        if self.use_condition:
            cond_emb = self.condition_embed(cond)
            emb = torch.concat([drift_emb, diffusion_emb, cond_emb], dim=-1)
        else:
            emb = torch.concat([drift_emb, diffusion_emb], dim=-1)

        return emb

    def example_input_array(self, batch_size: int = 1):
        """
        Return an example input array to be used for the model.
        """
        example_drift = torch.rand(batch_size, device=self.device)  # uniform on [0, 1)
        example_diffusion = torch.rand(batch_size, device=self.device)  # uniform on [0, 1)
        example_condition = torch.bernoulli(
                torch.full((batch_size, CONDITION_SIZE), 0.5, device=self.device)
            ) * 2 - 1  # either -1 or 1
        return example_drift, example_diffusion, example_condition


class TimeEmbedding(nn.Module):
    def __init__(self, model_channels, time_embed_dim):
        super().__init__()

        self.time_embedding = SinTimestepEmbedding(model_channels)
        self.linear_0 = nn.Linear(model_channels, time_embed_dim)
        self.silu = nn.SiLU()
        self.linear_1 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, x):
        """
        :param x: input - torch.Tensor with shape (B,)
        :return: time embedding - torch.Tensor with shape (B, C_time)
        """
        x = self.time_embedding(x)
        x = self.linear_0(x)
        x = self.silu(x)
        x = self.linear_1(x)
        return x


class ConditionEmbedding(nn.Module):
    def __init__(self, model_channels, time_embed_dim, embedding_length=40):
        super().__init__()

        self.cond_embedding = nn.Linear(embedding_length, model_channels)
        self.linear_0 = nn.Linear(model_channels, time_embed_dim)
        self.linear_1 = nn.Linear(time_embed_dim, time_embed_dim)
        self.silu = nn.SiLU()

    def forward(self, x):
        """
        :param x: input - torch.Tensor with shape (B, emb_length)
        :return: time embedding - torch.Tensor with shape (B, C_time)
        """
        x = self.cond_embedding(x)
        x = self.silu(x)
        x = self.linear_0(x)
        x = self.silu(x)
        x = self.linear_1(x)
        x = self.silu(x)
        return x


if __name__ == '__main__':
    from time import time

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    to_compile = False  # pytorch 2.0 feature
    torch.set_float32_matmul_precision('high')

    model_cfg = ModelConfig()
    model = EmbeddingModel(model_cfg)
    model.to(device)

    example_input_array = model.example_input_array(batch_size=32)

    if to_compile:
        model = torch.compile(model, fullgraph=True)

    for i in range(10):
        print(f"Forward pass {i+1}/10:")
        ttime = time()
        res = model(*example_input_array)
        print(f"Time: {time() - ttime}")

