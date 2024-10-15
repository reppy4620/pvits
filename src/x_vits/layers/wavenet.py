import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm


class WaveNet(nn.Module):
    def __init__(
        self, channels, kernel_size, dilation_rate, dropout_p, num_layers, cond_channels=0
    ):
        super().__init__()
        self.channels = channels
        self.num_layers = num_layers
        self.cond_channels = cond_channels

        self.gate_layers = nn.ModuleList()
        self.res_layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_p)
        if cond_channels > 0:
            self.cond_layer = nn.Linear(cond_channels, channels * 2 * num_layers)
        for i in range(1, num_layers + 1):
            dilation = dilation_rate**i
            self.gate_layers.append(
                weight_norm(
                    nn.Conv1d(
                        in_channels=channels,
                        out_channels=channels * 2,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        padding=dilation * (kernel_size - 1) // 2,
                    )
                )
            )

            is_last_layer = i == num_layers
            self.res_layers.append(
                weight_norm(
                    nn.Conv1d(
                        in_channels=channels,
                        out_channels=channels if is_last_layer else channels * 2,
                        kernel_size=1,
                    )
                )
            )

    def forward(self, x, mask, cond=None, **kwargs):
        if cond is not None:
            conds = self.cond_layer(cond).unsqueeze(-1).split(self.channels * 2, dim=1)

        o = 0
        for i, (gate_layer, res_layer) in enumerate(zip(self.gate_layers, self.res_layers)):
            xx = gate_layer(x)
            if self.cond_channels > 0:
                xx = xx + conds[i]
            x1, x2 = xx.split(self.channels, dim=1)
            xx = x1.tanh() * x2.sigmoid()
            xx = self.dropout(xx)
            xx = res_layer(xx)

            is_last_layer = i == self.num_layers - 1
            if is_last_layer:
                o = o + xx
            else:
                x1, x2 = xx.split(self.channels, dim=1)
                x = (x + x1) * mask
                o = o + x2
        o = o * mask
        return o
