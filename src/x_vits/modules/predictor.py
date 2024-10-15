import torch.nn as nn

from x_vits.layers import ChannelFirstLayerNorm


class BasicLayer(nn.Module):
    def __init__(self, channels, kernel_size, dropout):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.act = nn.ReLU()
        self.norm = ChannelFirstLayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.conv(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.dropout(x)
        return x * mask


class VariancePredictor(nn.Module):
    def __init__(
        self,
        channels,
        out_channels,
        kernel_size,
        dropout,
        num_layers,
        cond_channels=0,
    ):
        super().__init__()
        self.cond_channels = cond_channels
        self.layers = nn.ModuleList(
            [BasicLayer(channels, kernel_size, dropout) for _ in range(num_layers)]
        )
        self.conv = nn.Conv1d(channels, out_channels, kernel_size=1)

        if cond_channels > 0:
            self.cond_layer = nn.Linear(cond_channels, channels)

    def forward(self, x, mask, cond=None):
        x = x * mask
        if self.cond_channels > 0:
            x = x + self.cond_layer(cond).unsqueeze(-1)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.conv(x) * mask
        return x
