import torch.nn as nn


class ChannelFirstLayerNorm(nn.LayerNorm):
    def forward(self, x):
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class ChannelFirstRMSNorm(nn.RMSNorm):
    def forward(self, x):
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class AdaLayerNorm(nn.Module):
    def __init__(self, channels, cond_channels, eps=1e-5, dim=1):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps, elementwise_affine=False)

        self.gamma = nn.Linear(cond_channels, channels)
        self.beta = nn.Linear(cond_channels, channels)

    def forward(self, x, cond):
        gamma, beta = self.gamma(cond)[:, None], self.beta(cond)[:, None]
        return self.norm(x) * (1 + gamma) + beta


class ChannelFirstAdaLayerNorm(nn.Module):
    def __init__(self, channels, cond_channels, eps=1e-5, dim=1):
        super().__init__()
        self.norm = ChannelFirstLayerNorm(channels, eps=eps, elementwise_affine=False)

        self.gamma = nn.Linear(cond_channels, channels)
        self.beta = nn.Linear(cond_channels, channels)

    def forward(self, x, cond):
        gamma, beta = self.gamma(cond)[..., None], self.beta(cond)[..., None]
        return self.norm(x) * (1 + gamma) + beta


class AdaRMSNorm(nn.Module):
    def __init__(self, channels, cond_channels, eps=1e-5, dim=1):
        super().__init__()
        self.norm = nn.RMSNorm(channels, eps=eps, elementwise_affine=False)

        self.gamma = nn.Linear(cond_channels, channels)

    def forward(self, x, cond):
        gamma = self.gamma(cond)[:, None]
        return self.norm(x) * (1 + gamma)


class ChannelFirstAdaRMSNorm(nn.Module):
    def __init__(self, channels, cond_channels, eps=1e-5, dim=1):
        super().__init__()
        self.norm = ChannelFirstRMSNorm(channels, eps=eps, elementwise_affine=False)

        self.gamma = nn.Linear(cond_channels, channels)

    def forward(self, x, cond):
        gamma = self.gamma(cond)[..., None]
        return self.norm(x) * (1 + gamma)
