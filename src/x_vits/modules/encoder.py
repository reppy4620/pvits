import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm

from x_vits.layers import WaveNet
from x_vits.modules.transformer import TransformerBlock, VITSTransformerBlock
from x_vits.utils.model import length_to_mask


class TransformerTextEncoder(nn.Module):
    def __init__(self, num_vocab, channels, num_layers, num_heads, dropout, context_channels=0):
        super().__init__()
        self.scale = math.sqrt(channels)
        self.embedding = nn.Embedding(num_vocab, channels)
        nn.init.normal_(self.embedding.weight, 0.0, channels**-0.5)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(channels, num_heads, dropout, context_channels=context_channels)
                for _ in range(num_layers)
            ]
        )
        self.out_layer = nn.Linear(channels, channels)

    def forward(self, x, x_lengths, context=None, context_lengths=None):
        x = self.embedding(x) * self.scale
        mask = length_to_mask(x_lengths).unsqueeze(-1).type_as(x)
        attn_mask = (mask * mask.transpose(1, 2)).unsqueeze(1)
        if context is not None:
            context_mask = length_to_mask(context_lengths).unsqueeze(-1).type_as(x)
            xattn_mask = (mask * context_mask.transpose(1, 2)).unsqueeze(1)
        else:
            context_mask = None
            xattn_mask = None
        x = x * mask
        for layer in self.layers:
            x = layer(
                x,
                mask,
                attn_mask,
                context=context,
                context_attn_mask=xattn_mask,
            )
        x = x * mask
        x = self.out_layer(x) * mask
        x, mask = x.transpose(1, 2), mask.transpose(1, 2)
        return x, mask


class VITSTextEncoder(nn.Module):
    def __init__(self, num_vocab, channels, num_layers, num_heads, dropout):
        super().__init__()
        self.scale = math.sqrt(channels)
        self.embedding = nn.Embedding(num_vocab, channels)
        nn.init.normal_(self.embedding.weight, 0.0, channels**-0.5)
        self.layers = nn.ModuleList(
            [VITSTransformerBlock(channels, num_heads, dropout) for _ in range(num_layers)]
        )
        self.out_layer = nn.Conv1d(channels, channels, 1)

    def forward(self, x, x_lengths):
        x = self.embedding(x) * self.scale
        x = x.transpose(1, 2)
        mask = length_to_mask(x_lengths).unsqueeze(1).type_as(x)
        attn_mask = mask.unsqueeze(-1) * mask.unsqueeze(1)
        x = x * mask
        for layer in self.layers:
            x = layer(x, mask, attn_mask)
        x = x * mask
        x = self.out_layer(x) * mask
        return x, mask


class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        channels,
        out_channels,
        kernel_size,
        dilation_rate,
        num_layers,
        cond_channels=0,
    ):
        super().__init__()
        self.out_channels = out_channels

        self.pre = nn.Conv1d(in_channels, channels, 1)
        self.enc = WaveNet(
            channels=channels,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            dropout_p=0,
            num_layers=num_layers,
            cond_channels=cond_channels,
        )
        self.post = nn.Conv1d(channels, out_channels * 2, 1)

    def forward(self, x, mask, cond=None):
        x = self.pre(x) * mask
        x = self.enc(x, mask, cond=cond)
        stats = self.post(x) * mask
        m, logs = stats.split(self.out_channels, dim=1)
        z = m + torch.exp(logs) * torch.randn_like(m)
        return z * mask, m, logs


class StyleEncoder(nn.Module):
    def __init__(self, dim_in=16, style_dim=192, max_conv_dim=192, slope=0.2, repeat_num=4):
        super().__init__()
        self.style_dim = style_dim
        _dim_in = dim_in
        blocks = []
        for _ in range(repeat_num):
            dim_out = min(_dim_in * 2, max_conv_dim)
            blocks += [StyleEncoderLayer(_dim_in, dim_out)]
            _dim_in = dim_out
        self.net = nn.Sequential(
            *[
                spectral_norm(nn.Conv2d(1, dim_in, 3, 1, 1)),
                *blocks,
                nn.LeakyReLU(slope, inplace=True),
                spectral_norm(nn.Conv2d(dim_out, dim_out, 5, 1, 0)),
                nn.AdaptiveAvgPool2d(1),
                nn.LeakyReLU(slope, inplace=True),
                nn.Flatten(),
                nn.Linear(dim_out, style_dim),
            ],
        )

    @torch.autocast(device_type="cuda", enabled=False)
    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        o = self.net(x)
        return o


class LearnedDownSample(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.conv = spectral_norm(
            nn.Conv2d(
                dim_in,
                dim_in,
                kernel_size=(3, 3),
                stride=(2, 2),
                groups=dim_in,
                padding=1,
            )
        )

    def forward(self, x):
        return self.conv(x)


class DownSample(nn.Module):
    def forward(self, x):
        if x.shape[-1] % 2 != 0:
            x = torch.cat([x, x[..., -1].unsqueeze(-1)], dim=-1)
        return F.avg_pool2d(x, 2)


class StyleEncoderLayer(nn.Module):
    def __init__(self, dim_in, dim_out, slope=0.2):
        super().__init__()
        self.act = nn.LeakyReLU(slope)
        self.downsample = DownSample()
        self.downsample_res = LearnedDownSample(dim_in)
        self.conv1 = spectral_norm(nn.Conv2d(dim_in, dim_in, 3, 1, 1))
        self.conv2 = spectral_norm(nn.Conv2d(dim_in, dim_out, 3, 1, 1))
        self.learned_sc = dim_in != dim_out
        if self.learned_sc:
            self.conv1x1 = spectral_norm(nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False))

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = self.downsample(x)
        return x

    def _residual(self, x):
        x = self.act(x)
        x = self.conv1(x)
        x = self.downsample_res(x)
        x = self.act(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance
