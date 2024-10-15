import torch
import torch.nn as nn

from x_vits.layers import WaveNet


class VolumePreservingFlow(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size,
        dilation_rate,
        num_layers,
        num_flows=4,
        cond_channels=0,
    ):
        super().__init__()

        self.flows = nn.ModuleList()
        for i in range(num_flows):
            self.flows += [
                ResidualCouplingLayer(
                    channels,
                    kernel_size,
                    dilation_rate,
                    num_layers,
                    cond_channels=cond_channels,
                ),
                Flip(),
            ]

    def forward(self, x, mask, cond=None):
        for flow in self.flows:
            x = flow(x, mask, cond)
        return x

    def backward(self, x, mask, cond=None):
        for flow in reversed(self.flows):
            x = flow.reverse(x, mask, cond)
        return x


class FlowLayer(nn.Module):
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self, *args, **kwargs):
        raise NotImplementedError


class Flip(FlowLayer):
    def forward(self, x, *args, **kwargs):
        x = torch.flip(x, [1])
        return x

    def backward(self, x, *args, **kwargs):
        return self(x)


class ResidualCouplingLayer(FlowLayer):
    def __init__(
        self,
        channels,
        kernel_size,
        dilation_rate,
        num_layers,
        dropout_p=0,
        cond_channels=0,
    ):
        super().__init__()
        assert channels % 2 == 0
        self.channels = channels
        self.half_channels = channels // 2

        self.pre = nn.Conv1d(self.half_channels, channels, 1)
        self.enc = WaveNet(
            channels,
            kernel_size,
            dilation_rate,
            dropout_p,
            num_layers,
            cond_channels=cond_channels,
        )
        self.post = nn.Conv1d(channels, self.half_channels, 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def _calc_stats(self, x, mask, cond):
        x0, x1 = x.split(self.half_channels, dim=1)
        h = self.pre(x0) * mask
        h = self.enc(h, mask, cond=cond)
        m = self.post(h) * mask
        logs = torch.zeros_like(m)
        return x0, x1, m, logs

    def forward(self, x, mask, cond=None, *args, **kwargs):
        x0, x1, m, logs = self._calc_stats(x, mask, cond)
        x1 = (m + x1 * torch.exp(logs)) * mask
        x = torch.cat([x0, x1], dim=1)
        return x

    def backward(self, x, mask, cond=None, *args, **kwargs):
        x0, x1, m, logs = self._calc_stats(x, mask, cond)
        x1 = (x1 - m) * torch.exp(-logs) * mask
        x = torch.cat([x0, x1], 1)
        return x
