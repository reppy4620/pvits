import torch.nn as nn

from x_vits.layers import ChannelFirstAdaLayerNorm, ChannelFirstLayerNorm, PositionalEncoding


class ResidualLayer(nn.Module):
    def __init__(self, channels, kernel_size, dropout, cond_channels):
        super().__init__()
        self.cond_channels = cond_channels
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.act = nn.GELU()
        if self.cond_channels > 0:
            self.norm = ChannelFirstAdaLayerNorm(channels, cond_channels)
        else:
            self.norm = ChannelFirstLayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask, cond=None):
        res = x
        x = self.conv(x)
        x = self.act(x)
        x = self.dropout(x)
        if self.cond_channels > 0:
            x = self.norm(res + x, cond)
        else:
            x = self.norm(res + x)
        return x * mask


class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, dropout, num_layers, cond_channels):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ResidualLayer(
                    channels,
                    kernel_size,
                    dropout,
                    cond_channels,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, mask, cond=None):
        x = x * mask
        for layer in self.layers:
            x = layer(x, mask, cond=cond)
        return x


class FramePriorNetwork(nn.Module):
    def __init__(self, channels, kernel_size, dropout, num_layers, cond_channels=0):
        super().__init__()
        self.pos_emb = PositionalEncoding(channels, dropout)
        self.norm = ChannelFirstLayerNorm(channels)
        self.net = ResidualBlock(
            channels,
            kernel_size,
            dropout,
            num_layers,
            cond_channels,
        )
        self.conv = nn.Conv1d(channels, channels * 2, kernel_size=1)

    def forward(self, x, mask, cond=None):
        x = self.pos_emb(x.transpose(1, 2)).transpose(1, 2)
        x = self.norm(x)
        x = self.net(x, mask, cond=cond)
        stats = self.conv(x) * mask
        m_p, logs_p = stats.chunk(2, dim=1)
        return x, m_p, logs_p
