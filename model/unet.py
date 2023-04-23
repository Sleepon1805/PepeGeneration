from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from config import cfg
from model.utils import (
    zero_module,
    normalization,
)


class UNetModel(LightningModule):
    """
    https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/unet.py

    The full UNet model with attention and timestep embedding.
    :param model_channels: base channel count for the model.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param num_heads: the number of attention heads in each attention layer.
    """

    def __init__(
            self,
            model_channels=64,
            num_res_blocks=2,
            attention_resolutions=(cfg.image_size // 16, cfg.image_size // 8),
            dropout=0,
            channel_mult=(1, 2, 4, 8) if cfg.image_size == 64 else (1, 1, 2, 2),
            conv_resample=True,
            num_heads=1,
    ):
        super().__init__()

        self.in_channels = 3
        self.out_channels = 3

        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads

        time_embed_dim = self.model_channels * 4
        self.time_embed = nn.Sequential(
            SinTimestepEmbedding(self.model_channels),
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # downsample layers
        self.init_conv = TimestepEmbedSequential(
            nn.Conv2d(self.in_channels, model_channels, 3, padding=1)
        )

        self.downsample_0 = TimestepEmbedSequential(
            ResBlock(
                model_channels, time_embed_dim, dropout, out_channels=channel_mult[0] * model_channels,
            ),
            ResBlock(
                channel_mult[0] * model_channels, time_embed_dim, dropout, out_channels=channel_mult[0] * model_channels,
            ),
            Downsample(
                channel_mult[0] * model_channels, self.conv_resample,
            ),
        )
        self.downsample_1 = TimestepEmbedSequential(
            ResBlock(
                channel_mult[0] * model_channels, time_embed_dim, dropout, out_channels=channel_mult[1] * model_channels,
            ),
            ResBlock(
                channel_mult[1] * model_channels, time_embed_dim, dropout, out_channels=channel_mult[1] * model_channels,
            ),
            Downsample(
                channel_mult[1] * model_channels, self.conv_resample,
            ),
        )
        self.downsample_2 = TimestepEmbedSequential(
            ResBlock(
                channel_mult[1] * model_channels, time_embed_dim, dropout, out_channels=channel_mult[2] * model_channels,
            ),
            AttentionBlock(
                channel_mult[2] * model_channels, num_heads=self.num_heads,
            ),
            ResBlock(
                channel_mult[2] * model_channels, time_embed_dim, dropout, out_channels=channel_mult[2] * model_channels,
            ),
            AttentionBlock(
                channel_mult[2] * model_channels, num_heads=self.num_heads,
            ),
            Downsample(
                channel_mult[2] * model_channels, self.conv_resample,
            ),
        )
        self.downsample_3 = TimestepEmbedSequential(
            ResBlock(
                channel_mult[2] * model_channels, time_embed_dim, dropout, out_channels=channel_mult[3] * model_channels,
            ),
            AttentionBlock(
                channel_mult[3] * model_channels, num_heads=self.num_heads,
            ),
            ResBlock(
                channel_mult[3] * model_channels, time_embed_dim, dropout, out_channels=channel_mult[3] * model_channels,
            ),
            AttentionBlock(
                channel_mult[3] * model_channels, num_heads=self.num_heads,
            ),
        )

        # bottom of pyramid
        self.bottom_block = TimestepEmbedSequential(
            ResBlock(
                channel_mult[3] * model_channels, time_embed_dim, dropout, out_channels=channel_mult[3] * model_channels,
            ),
            AttentionBlock(
                channel_mult[3] * model_channels, num_heads=num_heads
            ),
            ResBlock(
                channel_mult[3] * model_channels, time_embed_dim, dropout, out_channels=channel_mult[3] * model_channels,
            ),
        )

        # upsample layers
        self.upsample_3 = TimestepEmbedSequential(
            ResBlock(
                channel_mult[3] * model_channels + channel_mult[3] * model_channels,
                time_embed_dim, dropout, out_channels=channel_mult[3] * model_channels,
            ),
            AttentionBlock(
                channel_mult[3] * model_channels, num_heads=self.num_heads,
            ),
            ResBlock(
                channel_mult[3] * model_channels,
                time_embed_dim, dropout, out_channels=channel_mult[3] * model_channels,
            ),
            AttentionBlock(
                channel_mult[3] * model_channels, num_heads=self.num_heads,
            ),
            ResBlock(
                channel_mult[3] * model_channels,
                time_embed_dim, dropout, out_channels=channel_mult[3] * model_channels,
            ),
            AttentionBlock(
                channel_mult[3] * model_channels, num_heads=self.num_heads,
            ),
        )
        self.upsample_2 = TimestepEmbedSequential(
            ResBlock(
                channel_mult[3] * model_channels + channel_mult[2] * model_channels,
                time_embed_dim, dropout, out_channels=channel_mult[2] * model_channels,
            ),
            AttentionBlock(
                channel_mult[2] * model_channels, num_heads=self.num_heads,
            ),
            ResBlock(
                channel_mult[2] * model_channels,
                time_embed_dim, dropout, out_channels=channel_mult[2] * model_channels,
            ),
            AttentionBlock(
                channel_mult[2] * model_channels, num_heads=self.num_heads,
            ),
            ResBlock(
                channel_mult[2] * model_channels,
                time_embed_dim, dropout, out_channels=channel_mult[2] * model_channels,
            ),
            AttentionBlock(
                channel_mult[2] * model_channels, num_heads=self.num_heads,
            ),
            Upsample(
                channel_mult[2] * model_channels, self.conv_resample
            ),
        )
        self.upsample_1 = TimestepEmbedSequential(
            ResBlock(
                channel_mult[2] * model_channels + channel_mult[1] * model_channels,
                time_embed_dim, dropout, out_channels=channel_mult[1] * model_channels,
            ),
            ResBlock(
                channel_mult[1] * model_channels,
                time_embed_dim, dropout, out_channels=channel_mult[1] * model_channels,
            ),
            ResBlock(
                channel_mult[1] * model_channels,
                time_embed_dim, dropout, out_channels=channel_mult[1] * model_channels,
            ),
            Upsample(
                channel_mult[1] * model_channels, self.conv_resample
            ),
        )
        self.upsample_0 = TimestepEmbedSequential(
            ResBlock(
                channel_mult[1] * model_channels + channel_mult[0] * model_channels,
                time_embed_dim, dropout, out_channels=channel_mult[0] * model_channels,
            ),
            ResBlock(
                channel_mult[0] * model_channels,
                time_embed_dim, dropout, out_channels=channel_mult[0] * model_channels,
            ),
            ResBlock(
                channel_mult[0] * model_channels,
                time_embed_dim, dropout, out_channels=channel_mult[0] * model_channels,
            ),
            Upsample(
                channel_mult[0] * model_channels, self.conv_resample
            ),
        )

        self.out = nn.Sequential(
            normalization(model_channels),
            nn.SiLU(),
            zero_module(nn.Conv2d(model_channels, self.out_channels, 3, padding=1)),
        )

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.upsample_0.parameters()).dtype

    def forward(self, x, timesteps):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x C x ...] Tensor of outputs.
        """

        hs = []
        emb = self.time_embed(timesteps)

        h = x.type(self.inner_dtype)

        h = self.init_conv(h, emb)

        h = self.downsample_0(h, emb)
        hs.append(h)
        h = self.downsample_1(h, emb)
        hs.append(h)
        h = self.downsample_2(h, emb)
        hs.append(h)
        h = self.downsample_3(h, emb)
        hs.append(h)

        h = self.bottom_block(h, emb)

        cat_in = th.cat([h, hs.pop()], dim=1)
        h = self.upsample_3(cat_in, emb)
        cat_in = th.cat([h, hs.pop()], dim=1)
        h = self.upsample_2(cat_in, emb)
        cat_in = th.cat([h, hs.pop()], dim=1)
        h = self.upsample_1(cat_in, emb)
        cat_in = th.cat([h, hs.pop()], dim=1)
        h = self.upsample_0(cat_in, emb)

        h = h.type(x.dtype)
        out = self.out(h)
        return out

    def get_feature_vectors(self, x, timesteps):
        """
        Apply the model and return all of the intermediate tensors.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: a dict with the following keys:
                 - 'down': a list of hidden state tensors from downsampling.
                 - 'middle': the tensor of the output of the lowest-resolution
                             block in the model.
                 - 'up': a list of hidden state tensors from upsampling.
        """
        hs = []
        emb = self.time_embed(timesteps)
        result = dict(down=[], up=[])
        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
            result["down"].append(h.type(x.dtype))
        h = self.middle_block(h, emb)
        result["middle"] = h.type(x.dtype)
        for module in self.output_blocks:
            cat_in = th.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
            result["up"].append(h.type(x.dtype))
        return result


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(TimestepBlock, nn.Sequential):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    """

    def __init__(self, channels, use_conv):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    """

    def __init__(self, channels, use_conv):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        if use_conv:
            self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
        else:
            self.op = nn.AvgPool2d()  # noqa

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    """

    def __init__(
            self,
            channels,
            emb_channels,
            dropout,
            out_channels=None,
            use_conv=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        h = h + emb_out
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        self.norm = normalization(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.attention = QKVAttention()
        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))

    def forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
        h = self.attention(qkv)
        h = h.reshape(b, -1, h.shape[-1])
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention.
    """

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (C * 3) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x C x T] tensor after attention.
        """
        ch = qkv.shape[1] // 3
        q, k, v = th.split(qkv, ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        return th.einsum("bts,bcs->bct", weight, v)


class SinTimestepEmbedding(nn.Module):
    def __init__(self, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.

        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        """
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, timesteps):
        """
        :param timesteps: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :return: an [N x dim] Tensor of positional embeddings.
        """
        half = self.dim // 2
        freqs = th.exp(
            -math.log(self.max_period) * th.arange(start=0, end=half, dtype=th.float32, device=timesteps.device) / half
        )
        args = timesteps[:, None].float() * freqs[None]
        embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
        if self.dim % 2:
            embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
