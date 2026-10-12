"""Batched GPU augmentations written from scratch in torch (no torchvision / kornia).

Every op samples its own random parameters per image, so a batch of B images gets B different
augmentations in a handful of kernel launches. Input: uint8 (B, C, S, S) on the target device.
Output: float32 normalised (B, C, out, out).

Ops (SimCLR/BYOL recipe): random-resized-crop + horizontal flip (one affine grid_sample),
colour jitter (brightness, contrast, saturation, hue; fixed order), random grayscale,
Gaussian blur (per-image sigma, grouped separable conv), solarisation.
"""
import math

import torch
import torch.nn.functional as F

DEFAULT_AUG = {
    "crop_scale": [0.08, 1.0], "crop_ratio": [0.75, 1.3333], "flip_p": 0.5,
    "jitter_p": 0.8, "brightness": 0.4, "contrast": 0.4, "saturation": 0.2, "hue": 0.1,
    "gray_p": 0.2, "blur_p": 0.5, "blur_sigma": [0.1, 2.0], "solarize_p": 0.0,
}

# Mild augmentation used for probe feature banks and the supervised baseline's "weak" option.
MILD_AUG = {
    "crop_scale": [0.35, 1.0], "crop_ratio": [0.75, 1.3333], "flip_p": 0.5,
    "jitter_p": 0.0, "brightness": 0.0, "contrast": 0.0, "saturation": 0.0, "hue": 0.0,
    "gray_p": 0.0, "blur_p": 0.0, "blur_sigma": [0.1, 2.0], "solarize_p": 0.0,
}


def _u(n, lo, hi, device):
    return torch.rand(n, device=device) * (hi - lo) + lo


def _gray(x):
    if x.shape[1] == 1:
        return x
    return (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])


def rgb_to_hsv(x, eps=1e-8):
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    maxc, _ = x.max(dim=1)
    minc, _ = x.min(dim=1)
    v = maxc
    delta = maxc - minc
    s = delta / (maxc + eps)
    dc = delta + eps
    rc, gc, bc = (maxc - r) / dc, (maxc - g) / dc, (maxc - b) / dc
    h = torch.where(maxc == r, bc - gc, torch.where(maxc == g, 2.0 + rc - bc, 4.0 + gc - rc))
    h = torch.where(delta > 0, (h / 6.0) % 1.0, torch.zeros_like(h))
    return torch.stack([h, s, v], dim=1)


def hsv_to_rgb(x):
    h, s, v = x[:, 0], x[:, 1], x[:, 2]
    i = torch.floor(h * 6.0)
    f = h * 6.0 - i
    i = i.long() % 6
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    r = torch.stack([v, q, p, p, t, v], 1).gather(1, i.unsqueeze(1)).squeeze(1)
    g = torch.stack([t, v, v, q, p, p], 1).gather(1, i.unsqueeze(1)).squeeze(1)
    b = torch.stack([p, p, t, v, v, q], 1).gather(1, i.unsqueeze(1)).squeeze(1)
    return torch.stack([r, g, b], dim=1)


def random_resized_crop(x, out_size, scale, ratio, flip_p):
    """Per-image random crop (area fraction ~ U(scale), log aspect ~ U(log ratio)) + flip as ONE
    affine transform evaluated with bilinear grid_sample."""
    B, dev = x.shape[0], x.device
    area = _u(B, scale[0], scale[1], dev)
    logr = _u(B, math.log(ratio[0]), math.log(ratio[1]), dev)
    r = torch.exp(logr)
    w = torch.sqrt(area * r).clamp(max=1.0)       # crop width as a fraction of the image side
    h = torch.sqrt(area / r).clamp(max=1.0)
    cx = (torch.rand(B, device=dev) * 2 - 1) * (1 - w)  # centre in [-1+w, 1-w] (normalised coords)
    cy = (torch.rand(B, device=dev) * 2 - 1) * (1 - h)
    flip = torch.where(torch.rand(B, device=dev) < flip_p, -1.0, 1.0)
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0] = w * flip
    theta[:, 0, 2] = cx
    theta[:, 1, 1] = h
    theta[:, 1, 2] = cy
    grid = F.affine_grid(theta, (B, x.shape[1], out_size, out_size), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def color_jitter(x, p, brightness, contrast, saturation, hue):
    B, dev = x.shape[0], x.device
    apply = (torch.rand(B, device=dev) < p).view(B, 1, 1, 1)
    y = x
    if brightness > 0:
        y = (y * _u(B, 1 - brightness, 1 + brightness, dev).view(B, 1, 1, 1)).clamp(0, 1)
    if contrast > 0:
        m = _gray(y).mean(dim=(1, 2, 3), keepdim=True)
        y = ((y - m) * _u(B, 1 - contrast, 1 + contrast, dev).view(B, 1, 1, 1) + m).clamp(0, 1)
    if x.shape[1] == 3:
        if saturation > 0:
            g = _gray(y)
            y = ((y - g) * _u(B, 1 - saturation, 1 + saturation, dev).view(B, 1, 1, 1) + g).clamp(0, 1)
        if hue > 0:
            hsv = rgb_to_hsv(y)
            hsv[:, 0] = (hsv[:, 0] + _u(B, -hue, hue, dev).view(B, 1, 1)) % 1.0
            y = hsv_to_rgb(hsv).clamp(0, 1)
    return torch.where(apply, y, x)


def random_grayscale(x, p):
    if x.shape[1] == 1 or p <= 0:
        return x
    apply = (torch.rand(x.shape[0], device=x.device) < p).view(-1, 1, 1, 1)
    return torch.where(apply, _gray(x).expand_as(x), x)


def gaussian_blur(x, p, sigma_range, rel_kernel=0.1):
    """Kernel size ~10% of the image side (SimCLR); sigma range given for 224px is rescaled to the
    actual image size so blur strength is resolution independent."""
    if p <= 0:
        return x
    B, C, H, W = x.shape
    dev = x.device
    scale = H / 224.0
    k = max(3, int(rel_kernel * H) // 2 * 2 + 1)
    sig = _u(B, sigma_range[0] * scale, sigma_range[1] * scale, dev).clamp_min(1e-3)
    r = torch.arange(k, device=dev, dtype=x.dtype) - k // 2
    ker = torch.exp(-(r.view(1, -1) ** 2) / (2 * sig.view(-1, 1) ** 2))
    ker = ker / ker.sum(1, keepdim=True)                     # (B, k)
    ker = ker.repeat_interleave(C, dim=0)                    # (B*C, k)
    pad = k // 2
    y = x.reshape(1, B * C, H, W)
    y = F.conv2d(F.pad(y, (pad, pad, 0, 0), mode="reflect"), ker.view(B * C, 1, 1, k), groups=B * C)
    y = F.conv2d(F.pad(y, (0, 0, pad, pad), mode="reflect"), ker.view(B * C, 1, k, 1), groups=B * C)
    y = y.view(B, C, H, W)
    apply = (torch.rand(B, device=dev) < p).view(B, 1, 1, 1)
    return torch.where(apply, y, x)


def solarize(x, p, thr=0.5):
    if p <= 0:
        return x
    apply = (torch.rand(x.shape[0], device=x.device) < p).view(-1, 1, 1, 1)
    return torch.where(apply & (x >= thr), 1.0 - x, x)


class GPUAugment:
    def __init__(self, params: dict, mean, std, out_size):
        self.p = dict(DEFAULT_AUG)
        self.p.update(params or {})
        self.mean = torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1)
        self.out_size = int(out_size)

    @torch.no_grad()
    def __call__(self, x_uint8, out_size=None):
        # Augmentations are called inside the training step's autocast region; force fp32 so the HSV hue
        # round-trip and blur are not computed in bf16/fp16.
        if x_uint8.device.type == "cuda":
            with torch.autocast(device_type="cuda", enabled=False):
                return self._apply(x_uint8, out_size)
        return self._apply(x_uint8, out_size)

    def _apply(self, x_uint8, out_size=None):
        p = self.p
        x = x_uint8.float().div_(255.0)
        x = random_resized_crop(x, out_size or self.out_size, p["crop_scale"], p["crop_ratio"], p["flip_p"])
        x = x.clamp(0, 1)
        if p["jitter_p"] > 0:
            x = color_jitter(x, p["jitter_p"], p["brightness"], p["contrast"], p["saturation"], p["hue"])
        x = random_grayscale(x, p["gray_p"])
        x = gaussian_blur(x, p["blur_p"], p["blur_sigma"])
        x = solarize(x, p["solarize_p"])
        dev = x.device
        return (x - self.mean.to(dev)) / self.std.to(dev)


def normalize_uint8(x_uint8, mean, std):
    """Deterministic eval preprocessing (identical to test.py): scale to [0,1] and standardise."""
    dev = x_uint8.device
    m = torch.as_tensor(mean, dtype=torch.float32, device=dev).view(1, -1, 1, 1)
    s = torch.as_tensor(std, dtype=torch.float32, device=dev).view(1, -1, 1, 1)
    return (x_uint8.float() / 255.0 - m) / s
