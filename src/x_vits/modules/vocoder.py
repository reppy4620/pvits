import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm

from x_vits.layers import AMPBlock, AntiAliasActivation, LearnablePQMF, SourceModuleHnNSF


class BigVGAN(nn.Module):
    def __init__(
        self,
        in_channel,
        upsample_initial_channel,
        upsample_rates,
        upsample_kernel_sizes,
        resblock_kernel_sizes,
        resblock_dilations,
        sample_rate,
        hop_length,
        harmonic_num,
        cond_channels=0,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.cond_channels = cond_channels

        self.conv_pre = weight_norm(
            nn.Conv1d(in_channel, upsample_initial_channel, kernel_size=7, stride=1, padding=3)
        )
        self.upsamples = nn.ModuleList()
        self.noise_convs = nn.ModuleList()
        if self.cond_channels > 0:
            self.cond_layers = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.upsamples.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        kernel_size=k,
                        stride=u,
                        padding=u // 2 + u % 2,
                        output_padding=u % 2,
                    )
                )
            )
            stride_f0 = np.prod(upsample_rates[i:])
            self.noise_convs.append(
                nn.Conv1d(
                    3,
                    upsample_initial_channel // (2**i),
                    kernel_size=stride_f0 * 2,
                    stride=stride_f0,
                    padding=stride_f0 // 2,
                )
            )
            if self.cond_channels > 0:
                self.cond_layers.append(
                    nn.Linear(cond_channels, upsample_initial_channel // (2**i))
                )

        self.amps = nn.ModuleList()
        for i in range(len(self.upsamples)):
            channel = upsample_initial_channel // (2 ** (i + 1))
            self.amps.append(
                nn.ModuleList(
                    [
                        AMPBlock(channel, kernel_size=k, dilations=d)
                        for k, d in zip(resblock_kernel_sizes, resblock_dilations)
                    ]
                )
            )
        self.act_post = AntiAliasActivation(channel)

        self.conv_post = weight_norm(
            nn.Conv1d(
                channel,
                1,
                kernel_size=7,
                padding=3,
            )
        )

        self.f0_upsample = nn.Upsample(scale_factor=hop_length)
        self.m_source = SourceModuleHnNSF(sample_rate, harmonic_num)

    def forward(self, x, f0, cond=None):
        f0 = self.f0_upsample(f0).transpose(-1, -2)
        har_source, _, uv = self.m_source(f0)
        har_source = har_source.transpose(-1, -2)
        uv = uv.transpose(-1, -2)
        source = torch.cat([har_source, uv, torch.randn_like(har_source)], dim=1)

        x = self.conv_pre(x)
        for i, (up, amp, noise_conv) in enumerate(zip(self.upsamples, self.amps, self.noise_convs)):
            x_source = noise_conv(source)
            if self.cond_channels > 0:
                x = x + self.cond_layers[i](cond).unsqueeze(-1)
            x = x + x_source
            x = up(x)
            xs = 0
            for layer in amp:
                xs += layer(x)
            x = xs / self.num_kernels
        x = self.act_post(x)
        y = self.conv_post(x).tanh()
        return y

    def remove_weight_norm(self):
        remove_weight_norm(self.conv_pre)
        for up in self.upsamples:
            remove_weight_norm(up)
        for amp in self.amps:
            amp.remove_weight_norm()
        remove_weight_norm(self.conv_post)


class XVocoder(nn.Module):
    def __init__(
        self,
        in_channel,
        upsample_initial_channel,
        upsample_rates,
        upsample_kernel_sizes,
        resblock_kernel_sizes,
        resblock_dilations,
        sample_rate,
        hop_length,
        harmonic_num,
        cond_channels=0,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.cond_channels = cond_channels

        self.pqmf = LearnablePQMF()
        subbands = self.pqmf.subbands

        self.conv_pre = weight_norm(
            nn.Conv1d(in_channel, upsample_initial_channel, kernel_size=7, stride=1, padding=3)
        )
        self.upsamples = nn.ModuleList()
        self.noise_convs = nn.ModuleList()
        if self.cond_channels > 0:
            self.cond_layers = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.upsamples.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        kernel_size=k,
                        stride=u,
                        padding=u // 2 + u % 2,
                        output_padding=u % 2,
                    )
                )
            )
            stride_f0 = subbands * np.prod(upsample_rates[i:])
            self.noise_convs.append(
                nn.Conv1d(
                    3,
                    upsample_initial_channel // (2**i),
                    kernel_size=stride_f0 * 2,
                    stride=stride_f0,
                    padding=stride_f0 // 2,
                )
            )
            if self.cond_channels > 0:
                self.cond_layers.append(
                    nn.Linear(cond_channels, upsample_initial_channel // (2**i))
                )

        self.amps = nn.ModuleList()
        for i in range(len(self.upsamples)):
            channel = upsample_initial_channel // (2 ** (i + 1))
            self.amps.append(
                nn.ModuleList(
                    [
                        AMPBlock(channel, kernel_size=k, dilations=d)
                        for k, d in zip(resblock_kernel_sizes, resblock_dilations)
                    ]
                )
            )
        self.act_post = AntiAliasActivation(channel)

        self.conv_post = weight_norm(
            nn.Conv1d(
                channel,
                subbands,
                kernel_size=7,
                padding=3,
            )
        )

        self.f0_upsample = nn.Upsample(scale_factor=hop_length)
        self.m_source = SourceModuleHnNSF(sample_rate, harmonic_num)

    def forward(self, x, f0, cond=None):
        f0 = self.f0_upsample(f0).transpose(-1, -2)
        har_source, _, uv = self.m_source(f0)
        har_source = har_source.transpose(-1, -2)
        uv = uv.transpose(-1, -2)
        source = torch.cat([har_source, uv, torch.randn_like(har_source)], dim=1)

        x = self.conv_pre(x)
        for i, (up, amp, noise_conv) in enumerate(zip(self.upsamples, self.amps, self.noise_convs)):
            x_source = noise_conv(source)
            if self.cond_channels > 0:
                x = x + self.cond_layers[i](cond).unsqueeze(-1)
            x = x + x_source
            x = up(x)
            xs = 0
            for layer in amp:
                xs += layer(x)
            x = xs / self.num_kernels
        x = self.act_post(x)
        y_mb = self.conv_post(x)
        y = self.pqmf.synthesis(y_mb)
        return y, y_mb

    def remove_weight_norm(self):
        remove_weight_norm(self.conv_pre)
        for up in self.upsamples:
            remove_weight_norm(up)
        for amp in self.amps:
            amp.remove_weight_norm()
        remove_weight_norm(self.conv_post)
