from pytorch_lightning import LightningModule

from model.modules import *


class UNetModel(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.init_channels = config.init_channels
        self.channel_mult = config.channel_mult
        self.conv_resample = config.conv_resample
        self.num_heads = config.num_heads
        self.dropout = config.dropout

        self.in_channels = 3
        self.model_channels = [mult * self.init_channels for mult in self.channel_mult]
        self.out_channels = 3
        self.time_embed_dim = self.init_channels * 4

        self.time_embed = TimeEmbedding(self.init_channels, self.time_embed_dim)

        # downsample layers
        self.init_conv = nn.Conv2d(self.in_channels, self.init_channels, 3, padding=1)

        self.downsample_0 = DownsampleLayer(
            self.init_channels, self.time_embed_dim, self.model_channels[0],
            self.dropout, self.num_heads, self.conv_resample,
            use_attention=False, use_downsample=True,
        )
        self.downsample_1 = DownsampleLayer(
            self.model_channels[0], self.time_embed_dim, self.model_channels[1],
            self.dropout, self.num_heads, self.conv_resample,
            use_attention=False, use_downsample=True,
        )
        self.downsample_2 = DownsampleLayer(
            self.model_channels[1], self.time_embed_dim, self.model_channels[2],
            self.dropout, self.num_heads, self.conv_resample,
            use_attention=True, use_downsample=True,
        )
        self.downsample_3 = DownsampleLayer(
            self.model_channels[2], self.time_embed_dim, self.model_channels[3],
            self.dropout, self.num_heads, self.conv_resample,
            use_attention=True, use_downsample=False,
        )

        # bottom of pyramid
        self.bottom_block = PyramidBottomLayer(self.model_channels[3], self.time_embed_dim,
                                               self.dropout, self.num_heads)

        # upsample layers
        self.upsample_3 = UpsampleLayer(
            self.model_channels[3], self.time_embed_dim, self.model_channels[3],
            self.dropout, self.num_heads, self.conv_resample,
            use_attention=True, use_upsample=False,
        )
        self.upsample_2 = UpsampleLayer(
            self.model_channels[3], self.time_embed_dim, self.model_channels[2],
            self.dropout, self.num_heads, self.conv_resample,
            use_attention=True, use_upsample=True,
        )
        self.upsample_1 = UpsampleLayer(
            self.model_channels[2], self.time_embed_dim, self.model_channels[1],
            self.dropout, self.num_heads, self.conv_resample,
            use_attention=False, use_upsample=True,
        )
        self.upsample_0 = UpsampleLayer(
            self.model_channels[1], self.time_embed_dim, self.model_channels[0],
            self.dropout, self.num_heads, self.conv_resample,
            use_attention=False, use_upsample=True,
        )

        self.out = nn.Sequential(
            NormalizationLayer(self.model_channels[0]),
            nn.SiLU(),
            zero_module(nn.Conv2d(self.init_channels, self.out_channels, 3, padding=1)),
        )

    def forward(self, x, timesteps):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x C x ...] Tensor of outputs.
        """

        hs = []
        emb = self.time_embed(timesteps)

        x = self.init_conv(x)

        down_0 = self.downsample_0(x, emb)
        down_1 = self.downsample_1(down_0, emb)
        down_2 = self.downsample_2(down_1, emb)
        down_3 = self.downsample_3(down_2, emb)

        bottom = self.bottom_block(down_3, emb)

        up_3 = self.upsample_3(bottom, down_3, emb)
        up_2 = self.upsample_2(up_3, down_2, emb)
        up_1 = self.upsample_1(up_2, down_1, emb)
        up_0 = self.upsample_0(up_1, down_0, emb)

        out = self.out(up_0)
        return out


class TimeEmbedding(nn.Module):
    def __init__(self, model_channels, time_embed_dim):
        super().__init__()

        self.time_embedding = SinTimestepEmbedding(model_channels)
        self.linear_0 = nn.Linear(model_channels, time_embed_dim)
        self.silu = nn.SiLU()
        self.linear_1 = nn.Linear(time_embed_dim, time_embed_dim)

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
        x = self.attention(x)
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

    def forward(self, x, sc_x, emb):
        x = th.cat([x, sc_x], dim=1)
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
