"""Small residual conditional GAN. Every learned component starts randomly."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm
from .text import TextEncoder


class ConditionalNorm(nn.Module):
    def __init__(self, channels, cond_dim):
        super().__init__()
        self.norm = nn.BatchNorm2d(channels, affine=False)
        self.affine = nn.Linear(cond_dim, channels * 2)

    def forward(self, x, text):
        scale, bias = self.affine(text).chunk(2, dim=1)
        return self.norm(x) * (1 + scale[:, :, None, None]) + bias[:, :, None, None]


class UpBlock(nn.Module):
    def __init__(self, cin, cout, cond):
        super().__init__()
        self.norm1, self.norm2 = ConditionalNorm(cin, cond), ConditionalNorm(cout, cond)
        self.conv1, self.conv2 = nn.Conv2d(cin, cout, 3, padding=1), nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1)

    def forward(self, x, cond):
        h = F.interpolate(F.relu(self.norm1(x, cond)), scale_factor=2, mode='nearest')
        h = self.conv2(F.relu(self.norm2(self.conv1(h), cond)))
        return (h + self.skip(F.interpolate(x, scale_factor=2, mode='nearest'))) / math.sqrt(2)


class Generator(nn.Module):
    def __init__(self, z_dim=96, cond=128, base=32):
        super().__init__()
        self.z_dim = z_dim
        self.fc = nn.Linear(z_dim + cond, base * 8 * 4 * 4)
        channels = [base*8, base*4, base*2, base, base]
        self.blocks = nn.ModuleList([UpBlock(a, b, cond) for a, b in zip(channels, channels[1:])])
        self.output = nn.Sequential(nn.BatchNorm2d(base), nn.ReLU(), nn.Conv2d(base, 3, 3, padding=1), nn.Tanh())

    def forward(self, z, text):
        x = self.fc(torch.cat([z, text], dim=1)).reshape(z.shape[0], -1, 4, 4)
        for block in self.blocks:
            x = block(x, text)
        return self.output(x)


class DownBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.c1 = spectral_norm(nn.Conv2d(cin, cout, 3, padding=1))
        self.c2 = spectral_norm(nn.Conv2d(cout, cout, 3, padding=1))
        self.skip = spectral_norm(nn.Conv2d(cin, cout, 1))

    def forward(self, x):
        h = self.c2(F.leaky_relu(self.c1(F.leaky_relu(x, .2)), .2))
        return (F.avg_pool2d(h, 2) + F.avg_pool2d(self.skip(x), 2)) / math.sqrt(2)


class Discriminator(nn.Module):
    def __init__(self, breeds=120, cond=128, base=32):
        super().__init__()
        self.blocks = nn.Sequential(DownBlock(3, base), DownBlock(base, base*2),
                                    DownBlock(base*2, base*4), DownBlock(base*4, base*8))
        dim = base * 8
        self.score = spectral_norm(nn.Linear(dim, 1))
        self.project = spectral_norm(nn.Linear(cond, dim, bias=False))
        self.classify = spectral_norm(nn.Linear(dim, breeds))

    def forward(self, image, text):
        features = F.leaky_relu(self.blocks(image), .2).sum((2, 3))
        score = self.score(features).squeeze(1) + (features * self.project(text)).sum(1) / math.sqrt(features.shape[1])
        return score, self.classify(features), features


class TextToImage(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.text_encoder = TextEncoder()
        self.generator = Generator(base=base)
        self.base = base

    def forward(self, tokens, z):
        return self.generator(z, self.text_encoder(tokens))


def parameter_counts(model, discriminator=None):
    count = lambda m: sum(p.numel() for p in m.parameters())
    result = dict(generator=count(model.generator), text_encoder=count(model.text_encoder),
                  deployed_total=count(model))
    if discriminator is not None:
        result.update(discriminator=count(discriminator), training_total=count(model)+count(discriminator))
    if max(result.values()) >= 50_000_000:
        raise ValueError('Parameter budget exceeded')
    return result
