"""VICReg (Bardes, Ponce, LeCun, ICLR 2022): Variance-Invariance-Covariance Regularization.

Prevents collapse via three explicit, intuitive criteria:
  - Invariance: MSE between representation pairs pulling views together.
  - Variance: forces std along batch >= 1.0 (prevents all vectors collapsing to a constant).
  - Covariance: penalizes off-diagonal covariance to decorrelate feature dimensions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.heads import MLPHead
from ssl_methods.augment import GPUAugment
from ssl_methods.base import SSLMethod


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vicreg_loss(z1, z2, sim_coeff=25.0, var_coeff=25.0, cov_coeff=1.0):
    """Computes VICReg loss between projections z1, z2 of shape (N, D)."""
    z1 = z1.float()
    z2 = z2.float()
    n, d = z1.shape

    # 1. Invariance loss: Mean Squared Error pulling views together
    sim_loss = F.mse_loss(z1, z2)

    # 2. Variance loss: keep standard deviation >= 1.0 along the batch
    std_z1 = torch.sqrt(z1.var(dim=0) + 1e-4)
    std_z2 = torch.sqrt(z2.var(dim=0) + 1e-4)
    var_loss = torch.mean(F.relu(1.0 - std_z1)) + torch.mean(F.relu(1.0 - std_z2))

    # 3. Covariance loss: decorrelate dimensions (off-diagonal terms -> 0)
    z1_c = z1 - z1.mean(dim=0)
    z2_c = z2 - z2.mean(dim=0)
    cov_z1 = (z1_c.T @ z1_c) / (n - 1)
    cov_z2 = (z2_c.T @ z2_c) / (n - 1)
    cov_loss = (off_diagonal(cov_z1).pow(2).sum() / d) + (off_diagonal(cov_z2).pow(2).sum() / d)

    loss = sim_coeff * sim_loss + var_coeff * var_loss + cov_coeff * cov_loss
    return loss, sim_loss, var_loss, cov_loss


class VICReg(SSLMethod):
    name = "vicreg"

    def __init__(self, encoder, cfg, meta):
        super().__init__()
        vc = cfg.get("vicreg", {})
        self.encoder = encoder
        self.sim_coeff = float(vc.get("sim_coeff", 25.0))
        self.var_coeff = float(vc.get("var_coeff", 25.0))
        self.cov_coeff = float(vc.get("cov_coeff", 1.0))

        # Standard 3-layer MLP projector
        proj_dim = int(vc.get("proj_dim", 2048))
        proj_hidden = int(vc.get("proj_hidden", 2048))
        proj_layers = int(vc.get("proj_layers", 3))
        self.proj = MLPHead(encoder.out_dim, hidden=proj_hidden, out_dim=proj_dim,
                            n_layers=proj_layers, last_bn=False)

        # GPU augmentations matching the pipeline
        self.aug = GPUAugment(cfg.get("aug", {}), meta["mean"], meta["std"], meta["image_size"])
        v2 = dict(cfg.get("aug", {}))
        v2.update(vc.get("view2_overrides", {}) or {})
        self.aug2 = GPUAugment(v2, meta["mean"], meta["std"], meta["image_size"])

    def training_step(self, x):
        v1, v2 = self.aug(x), self.aug2(x)
        h = self.enc(torch.cat([v1, v2], 0))
        z = self.proj(h)
        n = x.shape[0]
        loss, sim_loss, var_loss, cov_loss = vicreg_loss(
            z[:n], z[n:], self.sim_coeff, self.var_coeff, self.cov_coeff
        )
        logs = {
            "sim_loss": sim_loss.detach(),
            "var_loss": var_loss.detach(),
            "cov_loss": cov_loss.detach()
        }
        return loss, logs


# --------------------------------------------------------------------------
# Auto-registration hook: allows ssl.method=vicreg without modifying base.py
# --------------------------------------------------------------------------
import ssl_methods.base as _base

_orig_build = getattr(_base, "build_ssl_method", None)
if _orig_build is not None:
    def _patched_build(cfg, encoder, meta):
        name = cfg.get("ssl", {}).get("method", "").lower()
        if name in ("vicreg", "vic"):
            return VICReg(encoder, cfg, meta)
        if name in ("barlow", "barlow_twins", "bt"):
            from ssl_methods.barlow_twins import BarlowTwins
            return BarlowTwins(encoder, cfg, meta)
        return _orig_build(cfg, encoder, meta)
    _base.build_ssl_method = _patched_build
