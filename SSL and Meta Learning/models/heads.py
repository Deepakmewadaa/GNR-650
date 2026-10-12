"""Projection heads (SSL) and the linear classification head."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPHead(nn.Module):
    """SimCLR projection g(.): Linear-BN-ReLU-...-Linear (Chen et al. 2020). The representation
    used downstream is taken *before* this head (it discards augmentation-related information)."""

    def __init__(self, in_dim, hidden=2048, out_dim=128, n_layers=2, last_bn=False):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(d, hidden, bias=False), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True)]
            d = hidden
        layers.append(nn.Linear(d, out_dim, bias=not last_bn))
        if last_bn:
            layers.append(nn.BatchNorm1d(out_dim, affine=False))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DINOHead(nn.Module):
    """DINO head (Caron et al. 2021): 3-layer MLP -> L2-normalised bottleneck -> weight-normalised
    prototype layer. Weight norm is implemented manually (no torch.nn.utils.parametrize dependency)."""

    def __init__(self, in_dim, out_dim=4096, hidden=2048, bottleneck=256, norm_last_layer=True):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, bottleneck))
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
        self.v = nn.Parameter(torch.randn(out_dim, bottleneck) * 0.02)
        self.g = nn.Parameter(torch.ones(out_dim, 1), requires_grad=not norm_last_layer)

    def forward(self, x):
        x = F.normalize(self.mlp(x), dim=-1)
        w = self.g * F.normalize(self.v, dim=1)
        return F.linear(x, w)


class LinearHead(nn.Module):
    """Single nn.Linear classifier on frozen features. Feature standardisation used during training is
    folded into (W, b) at export time, so the deployed head is exactly one Linear layer."""

    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)
