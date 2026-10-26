"""Barlow Twins (Zbontar et al., ICML 2021): Self-Supervised Learning via Redundancy Reduction.

Computes the empirical cross-correlation matrix between the representations of two augmented views
and forces it towards the identity matrix:
  - Invariance term: diagonal elements -> 1
  - Redundancy reduction term: off-diagonal elements -> 0 (decorrelates feature dimensions)
"""
import torch
import torch.nn as nn

from models.heads import MLPHead
from ssl_methods.augment import GPUAugment
from ssl_methods.base import SSLMethod


def barlow_twins_loss(z1, z2, lambd=0.0051):
    """Computes Barlow Twins cross-correlation loss between two view projections (N, D).

    Normalized along the batch dimension in fp32 for numerical stability under AMP.
    Returns (loss, on_diag, off_diag, corr_diag_mean).
    """
    z1 = z1.float()
    z2 = z2.float()
    n, d = z1.shape

    # 1. Zero-mean and unit-variance normalization along the batch dimension
    z1_norm = (z1 - z1.mean(0)) / (z1.std(0).clamp_min(1e-4))
    z2_norm = (z2 - z2.mean(0)) / (z2.std(0).clamp_min(1e-4))

    # 2. Empirical cross-correlation matrix between feature dimensions: (D, D)
    c = (z1_norm.T @ z2_norm) / n

    # 3. Invariance: diagonal elements should equal 1.0
    diag = torch.diagonal(c)
    on_diag = (diag - 1.0).pow(2).sum()

    # 4. Redundancy reduction: off-diagonal elements should equal 0.0
    off_diag = c.pow(2).sum() - diag.pow(2).sum()

    loss = on_diag + lambd * off_diag
    return loss, on_diag, off_diag, diag.mean()


class BarlowTwins(SSLMethod):
    name = "barlow_twins"

    def __init__(self, encoder, cfg, meta):
        super().__init__()
        bc = cfg.get("barlow_twins", {})
        self.encoder = encoder
        self.lambd = float(bc.get("lambd", 0.0051))

        # Standard Barlow Twins projection head: 3-layer MLP
        proj_dim = int(bc.get("proj_dim", 2048))
        proj_hidden = int(bc.get("proj_hidden", 2048))
        proj_layers = int(bc.get("proj_layers", 3))
        self.proj = MLPHead(encoder.out_dim, hidden=proj_hidden, out_dim=proj_dim,
                            n_layers=proj_layers, last_bn=False)

        # GPU augmentations matching the pipeline
        self.aug = GPUAugment(cfg.get("aug", {}), meta["mean"], meta["std"], meta["image_size"])
        v2 = dict(cfg.get("aug", {}))
        v2.update(bc.get("view2_overrides", {}) or {})
        self.aug2 = GPUAugment(v2, meta["mean"], meta["std"], meta["image_size"])

    def training_step(self, x):
        v1, v2 = self.aug(x), self.aug2(x)
        h = self.enc(torch.cat([v1, v2], 0))
        z = self.proj(h)
        n = x.shape[0]
        loss, on_diag, off_diag, diag_mean = barlow_twins_loss(z[:n], z[n:], self.lambd)
        logs = {
            "on_diag": on_diag.detach(),
            "off_diag": off_diag.detach(),
            "corr": diag_mean.detach()
        }
        return loss, logs


# --------------------------------------------------------------------------
# Auto-registration hook: allows ssl.method=barlow_twins without modifying base.py
# --------------------------------------------------------------------------
import ssl_methods.base as _base

_orig_build = getattr(_base, "build_ssl_method", None)
if _orig_build is not None:
    def _patched_build(cfg, encoder, meta):
        name = cfg.get("ssl", {}).get("method", "").lower()
        if name in ("barlow", "barlow_twins", "bt"):
            return BarlowTwins(encoder, cfg, meta)
        return _orig_build(cfg, encoder, meta)
    _base.build_ssl_method = _patched_build
