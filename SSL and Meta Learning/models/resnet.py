"""ResNet-18-style encoder written from scratch (He et al. 2016), random init only.

Design notes (viva):
- BasicBlock x [2,2,2,2]; stage widths [w, 2w, 4w, embed_dim] so the embedding size is configurable
  independently of the width (default w=64, embed_dim=512 == standard ResNet-18).
- stem 'small' (3x3 s1, no max-pool) for <=64px images keeps spatial resolution; 'standard'
  (7x7 s2 + max-pool) for larger images keeps the compute small at 128px (final map 4x4);
  'medium' (3x3 s1 + max-pool, /2) is an ablation option (more detail, ~4x compute -> fewer SSL epochs).
- Zero-init of the last BN gamma in every residual branch (Goyal et al. 2017): each block starts as
  identity, which stabilises large-batch SSL training.
- Output = global-average-pooled features (the representation used by the probe and OOD scores).
"""
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        idt = x if self.short is None else self.short(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + idt)


class ResNetEncoder(nn.Module):
    def __init__(self, in_ch=3, width=64, embed_dim=512, layers=(2, 2, 2, 2), stem="standard"):
        super().__init__()
        self.cfg = dict(in_ch=in_ch, width=width, embed_dim=embed_dim, layers=list(layers), stem=stem)
        if stem == "small":       # /1: for <=64px inputs
            self.stem = nn.Sequential(nn.Conv2d(in_ch, width, 3, 1, 1, bias=False), nn.BatchNorm2d(width),
                                      nn.ReLU(inplace=True))
        elif stem == "medium":    # /2: 4x the compute of 'standard' but keeps 2x more spatial detail
            self.stem = nn.Sequential(nn.Conv2d(in_ch, width, 3, 1, 1, bias=False), nn.BatchNorm2d(width),
                                      nn.ReLU(inplace=True), nn.MaxPool2d(3, 2, 1))
        else:
            self.stem = nn.Sequential(nn.Conv2d(in_ch, width, 7, 2, 3, bias=False), nn.BatchNorm2d(width),
                                      nn.ReLU(inplace=True), nn.MaxPool2d(3, 2, 1))
        widths = [width, 2 * width, 4 * width, embed_dim]
        stages, cin = [], width
        for i, (w, n) in enumerate(zip(widths, layers)):
            blocks = []
            for j in range(n):
                blocks.append(BasicBlock(cin, w, 2 if (j == 0 and i > 0) else 1))
                cin = w
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = embed_dim
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.zeros_(m.bn2.weight)

    def forward(self, x):
        x = self.stages(self.stem(x))
        return torch.flatten(self.pool(x), 1)


def resolve_model_cfg(mcfg, image_size, channels):
    stem = mcfg.get("stem", "auto")
    if stem in (None, "auto"):
        stem = "small" if image_size <= 64 else "standard"
    return dict(in_ch=int(channels), width=int(mcfg.get("width", 64)), embed_dim=int(mcfg.get("embed_dim", 512)),
                layers=list(mcfg.get("layers", [2, 2, 2, 2])), stem=stem)


def build_encoder(model_cfg: dict) -> ResNetEncoder:
    return ResNetEncoder(**model_cfg)
