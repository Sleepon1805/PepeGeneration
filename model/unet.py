from pytorch_lightning import LightningModule

from config import cfg
from model.modules import *


class UNetModel(LightningModule):
    """
    https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/unet.py

    The full UNet model with attention and timestep embedding.
    :param model_channels: base channel count for the model.
    :param num_res_blocks: number of residual blocks per downsample.
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
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            num_heads=1,
    ):
        super().__init__()
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads

        self.in_channels = 3
        self.out_channels = 3
        self.time_embed_dim = self.model_channels * 4

        self.time_embed = TimeEmbedding(self.model_channels, self.time_embed_dim)

        # downsample layers
        self.init_conv = nn.Conv2d(self.in_channels, self.model_channels, 3, padding=1)

        self.downsample_0 = DownsampleLayer(
            self.model_channels, self.time_embed_dim, self.channel_mult[0] * self.model_channels,
            self.dropout, self.num_heads, self.conv_resample,
            use_attention=False, use_downsample=True,
        )
        self.downsample_1 = DownsampleLayer(
            self.channel_mult[0] * self.model_channels, self.time_embed_dim, self.channel_mult[1] * self.model_channels,
            self.dropout, self.num_heads, self.conv_resample,
            use_attention=False, use_downsample=True,
        )
        self.downsample_2 = DownsampleLayer(
            self.channel_mult[1] * self.model_channels, self.time_embed_dim, self.channel_mult[2] * self.model_channels,
            self.dropout, self.num_heads, self.conv_resample,
            use_attention=True, use_downsample=True,
        )
        self.downsample_3 = DownsampleLayer(
            self.channel_mult[2] * self.model_channels, self.time_embed_dim, self.channel_mult[3] * self.model_channels,
            self.dropout, self.num_heads, self.conv_resample,
            use_attention=True, use_downsample=False,
        )

        # bottom of pyramid
        self.bottom_block = PyramidBottomLayer(self.channel_mult[3] * self.model_channels, self.time_embed_dim,
                                               self.dropout, self.num_heads)

        # upsample layers
        self.upsample_3 = UpsampleLayer(
            self.channel_mult[3] * self.model_channels, self.time_embed_dim, self.channel_mult[3] * self.model_channels,
            self.dropout, self.num_heads, self.conv_resample,
            use_attention=True, use_upsample=False,
        )
        self.upsample_2 = UpsampleLayer(
            self.channel_mult[3] * self.model_channels, self.time_embed_dim, self.channel_mult[2] * self.model_channels,
            self.dropout, self.num_heads, self.conv_resample,
            use_attention=True, use_upsample=True,
        )
        self.upsample_1 = UpsampleLayer(
            self.channel_mult[2] * self.model_channels, self.time_embed_dim, self.channel_mult[1] * self.model_channels,
            self.dropout, self.num_heads, self.conv_resample,
            use_attention=False, use_upsample=True,
        )
        self.upsample_0 = UpsampleLayer(
            self.channel_mult[1] * self.model_channels, self.time_embed_dim, self.channel_mult[0] * self.model_channels,
            self.dropout, self.num_heads, self.conv_resample,
            use_attention=False, use_upsample=False,
        )

        self.out = nn.Sequential(
            NormalizationLayer(self.channel_mult[0] * self.model_channels),
            nn.SiLU(),
            zero_module(nn.Conv2d(self.model_channels, self.out_channels, 3, padding=1)),
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


class TimeEmbedding(nn.Module):
    def __init__(self, model_channels, time_embed_dim):
        super().__init__()

        self.time_embedding = SinTimestepEmbedding(model_channels)
        self.linear_0 = nn.Linear(model_channels, time_embed_dim)
        self.silu = nn.SiLU()
        self.linear_1 = nn.Linear(model_channels, time_embed_dim)

    def forward(self, x):
        x = self.time_embedding(x)
        x = self.linear_0(x)
        x = self.silu(x)
        x = self.linear_1(x)
        return x


class DownsampleLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, num_heads, conv_resample,
                 use_attention: bool, use_downsample: bool):
        super().__init__()
        self.use_attention = use_attention
        self.use_downsample = use_downsample

        self.res_0 = ResBlock(
            in_channels, hidden_channels, dropout, out_channels,
        )
        self.res_1 = ResBlock(
            out_channels, hidden_channels, dropout, out_channels,
        )
        if self.use_attention:
            self.attention_0 = AttentionBlock(
                out_channels, num_heads=num_heads,
            )
            self.attention_1 = AttentionBlock(
                out_channels, num_heads=num_heads,
            )
        if self.use_downsample:
            self.downsample = Downsample(
                out_channels, conv_resample,
            )

    def forward(self, x, emb):
        x = self.res_0(x, emb)
        if self.use_attention:
            x = self.attention_0(x)
        x = self.res_1(x, emb)
        if self.use_attention:
            x = self.attention_1(x)
        if self.use_downsample:
            x = self.downsample(x)
        return x


class PyramidBottomLayer(nn.Module):
    def __init__(self, channels, hidden_channels, dropout, num_heads):
        super().__init__()
        self.res_0 = ResBlock(
            channels, hidden_channels, dropout, out_channels=channels,
        )
        self.attention = AttentionBlock(
            channels, num_heads=num_heads
        )
        self.res_1 = ResBlock(
            channels, hidden_channels, dropout, out_channels=channels,
        )

    def forward(self, x, emb):
        x = self.res_0(x, emb)
        x = self.attention_0(x)
        x = self.res_1(x, emb)
        return x


class UpsampleLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, num_heads, conv_resample,
                 use_attention: bool, use_upsample: bool):
        super().__init__()
        self.use_attention = use_attention
        self.use_upsample = use_upsample

        self.res_0 = ResBlock(
            in_channels + out_channels, hidden_channels, dropout, out_channels,
        )
        self.res_1 = ResBlock(
            out_channels, hidden_channels, dropout, out_channels,
        )
        self.res_2 = ResBlock(
            out_channels, hidden_channels, dropout, out_channels,
        )
        if self.use_attention:
            self.attention_0 = AttentionBlock(
                out_channels, num_heads=num_heads,
            )
            self.attention_1 = AttentionBlock(
                out_channels, num_heads=num_heads,
            )
            self.attention_2 = AttentionBlock(
                out_channels, num_heads=num_heads,
            )
        if self.use_upsample:
            self.upsample = Upsample(
                out_channels, conv_resample
            )

    def forward(self, x, emb):
        x = self.res_0(x, emb)
        if self.use_attention:
            x = self.attention_0(x)
        x = self.res_1(x, emb)
        if self.use_attention:
            x = self.attention_1(x)
        x = self.res_2(x, emb)
        if self.use_attention:
            x = self.attention_2(x)
        if self.use_upsample:
            x = self.upsample(x)
        return x
