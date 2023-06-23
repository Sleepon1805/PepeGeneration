import torch
import torch.nn as nn
from lightning import LightningModule

from model.modules import (ResBlock, Upsample, Downsample, AttentionBlock, SinTimestepEmbedding, NormalizationLayer)
from config import Config


class UNetModel(LightningModule):
    def __init__(self, config: Config):
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
            self.model_channels[3], self.time_embed_dim, self.model_channels[2],
            sc_channels=(self.model_channels[3], self.model_channels[3], self.model_channels[3], self.model_channels[2]),
            dropout=self.dropout, num_heads=self.num_heads, conv_resample=self.conv_resample,
            use_attention=True, use_upsample=True,
        )
        self.upsample_2 = UpsampleLayer(
            self.model_channels[2], self.time_embed_dim, self.model_channels[1],
            sc_channels=(self.model_channels[2], self.model_channels[2], self.model_channels[2], self.model_channels[1]),
            dropout=self.dropout, num_heads=self.num_heads, conv_resample=self.conv_resample,
            use_attention=True, use_upsample=True,
        )
        self.upsample_1 = UpsampleLayer(
            self.model_channels[1], self.time_embed_dim, self.model_channels[0],
            sc_channels=(self.model_channels[1], self.model_channels[1], self.model_channels[1], self.model_channels[0]),
            dropout=self.dropout, num_heads=self.num_heads, conv_resample=self.conv_resample,
            use_attention=False, use_upsample=True,
        )
        self.upsample_0 = UpsampleLayer(
            self.model_channels[0], self.time_embed_dim, self.init_channels,
            sc_channels=(self.model_channels[0], self.model_channels[0], self.model_channels[0], self.init_channels),
            dropout=self.dropout, num_heads=self.num_heads, conv_resample=self.conv_resample,
            use_attention=False, use_upsample=False,
        )

        self.out = nn.Sequential(
            NormalizationLayer(self.init_channels),
            nn.SiLU(),
            nn.Conv2d(self.init_channels, self.init_channels // 2, 3, padding=1),
            nn.Conv2d(self.init_channels // 2, self.out_channels, 3, padding=1),
        )

    def forward(self, x, timesteps):
        """
        Apply the model to an input batch.

        :param x: input - torch.Tensor with shape (B, C_color, H, W)
        :param timesteps: a 1-D batch of timesteps - torch.Tensor with shape (B)
        :return: output - torch.Tensor with shape (B, C_color, H, W)
        """

        emb = self.time_embed(timesteps)

        x = self.init_conv(x)

        down_0 = self.downsample_0(x, emb)
        down_1 = self.downsample_1(down_0[-1], emb)
        down_2 = self.downsample_2(down_1[-1], emb)
        down_3 = self.downsample_3(down_2[-1], emb)

        bottom = self.bottom_block(down_3[-1], emb)

        all_scs = (x,) + down_0 + down_1 + down_2 + down_3

        up_3 = self.upsample_3(bottom, all_scs[12:16], emb)
        up_2 = self.upsample_2(up_3, all_scs[8:12], emb)
        up_1 = self.upsample_1(up_2, all_scs[4:8], emb)
        up_0 = self.upsample_0(up_1, all_scs[0:4], emb)

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
        """
        :param x: input - torch.Tensor with shape (B,)
        :return: time embedding - torch.Tensor with shape (B, C_time)
        """
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
        if self.use_downsample:
            self.downsample = Downsample(
                out_channels, conv_resample,
            )

    def forward(self, x, emb):
        """
        :param x: input - torch.Tensor with shape (B, C_in, H, W)
        :param emb: time embedding - torch.Tensor with shape (B, C_time)
        :return: outputs - (torch.Tensor with shape (B, C_out, H, W),
                            torch.Tensor with shape (B, C_out, H, W),
                            torch.Tensor with shape (B, C_out, H[//2], W[//2]))
        """
        x_0 = self.res_0(x, emb)
        if self.use_attention:
            x_0 = self.attention_0(x_0)

        x_1 = self.res_1(x_0, emb)
        if self.use_attention:
            x_1 = self.attention_1(x_1)

        x_2 = self.res_1(x_1, emb)
        if self.use_attention:
            x_2 = self.attention_1(x_2)

        if self.use_downsample:
            x_3 = self.downsample(x_2)
            return x_0, x_1, x_2, x_3
        else:
            return x_0, x_1, x_2


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
        """
        :param x: input - torch.Tensor with shape (B, C_in, H, W)
        :param emb: time embedding - torch.Tensor with shape (B, C_time)
        :return: output - torch.Tensor with shape (B, C_in, H, W)
        """
        x = self.res_0(x, emb)
        x = self.attention(x)
        x = self.res_1(x, emb)
        return x


class UpsampleLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, sc_channels,
                 dropout, num_heads, conv_resample,
                 use_attention: bool, use_upsample: bool):
        super().__init__()
        self.use_attention = use_attention
        self.use_upsample = use_upsample

        self.res_0 = ResBlock(
            in_channels + sc_channels[0], hidden_channels, dropout, out_channels,
        )
        self.res_1 = ResBlock(
            out_channels + sc_channels[1], hidden_channels, dropout, out_channels,
        )
        self.res_2 = ResBlock(
            out_channels + sc_channels[2], hidden_channels, dropout, out_channels,
        )
        self.res_3 = ResBlock(
            out_channels + sc_channels[3], hidden_channels, dropout, out_channels,
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
            self.attention_3 = AttentionBlock(
                out_channels, num_heads=num_heads,
            )
        if self.use_upsample:
            self.upsample = Upsample(
                out_channels, conv_resample
            )

    def forward(self, x, shortcuts, emb):
        """
        :param x: input - torch.Tensor with shape (B, C_in_x, H, W)
        :param shortcuts: shortcut - torch.Tensor with shape (B, C_in_sc, H, W)
        :param emb: time embedding - torch.Tensor with shape (B, C_time)
        :return: output - torch.Tensor with shape (B, C_out, H[*2], W[*2])
        """
        sc_0, sc_1, sc_2, sc_3 = shortcuts

        x = torch.cat([x, sc_3], dim=1)
        x = self.res_0(x, emb)
        if self.use_attention:
            x = self.attention_0(x)

        x = torch.cat([x, sc_2], dim=1)
        x = self.res_1(x, emb)
        if self.use_attention:
            x = self.attention_1(x)

        x = torch.cat([x, sc_1], dim=1)
        x = self.res_2(x, emb)
        if self.use_attention:
            x = self.attention_2(x)

        x = torch.cat([x, sc_0], dim=1)
        x = self.res_3(x, emb)
        if self.use_attention:
            x = self.attention_3(x)

        if self.use_upsample:
            x = self.upsample(x)

        return x


if __name__ == '__main__':
    cfg = Config()
    model = UNetModel(cfg)
    example_input_x = torch.Tensor(cfg.batch_size, 3, cfg.image_size, cfg.image_size)
    example_input_t = torch.ones(cfg.batch_size)
    model = torch.compile(model)
    print(model(example_input_x, example_input_t))
