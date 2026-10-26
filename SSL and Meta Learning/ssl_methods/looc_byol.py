"""LooC-BYOL: Decoupled Invariance Bootstrap Your Own Latent.

Combines the two top-performing principles from our empirical benchmarks:
1. Feature-Predictive Self-Supervision (from BYOL, Grill et al., NeurIPS 2020)
   - Online vs. EMA Target encoder with Cosine Distance MSE (no negative pairs).
2. Decoupled Invariance Projections (from LooC, Xiao et al., ICLR 2021)
   - Decouples Spatial/Shape invariance from Color/Texture invariance.

Architecture:
- Head 1 (Spatial / Geometry Head):
    Online Projector & Predictor + EMA Target Projector.
    Trained on spatial crops (global [0.20, 1.0] vs local [0.10, 0.50]) with ZERO color distortion.
    Forces the backbone to learn invariant geometric shapes and object part-to-whole consistency.
- Head 2 (Color / Texture Head):
    Online Projector & Predictor + EMA Target Projector.
    Trained on full-frame views with heavy color jitter, grayscale, and solarization (ZERO cropping).
    Forces the backbone to learn lighting and texture constancy without sacrificing color sensitivity.

Both heads update the shared ResNet-34 online backbone simultaneously.
At deployment / evaluation, the heads are discarded, yielding a ResNet-34 encoder with
unprecedented fine-grained shape AND color sensitivity.
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.heads import MLPHead
from ssl_methods.augment import GPUAugment
from ssl_methods.base import SSLMethod
from utils.optim import cosine_value


def byol_loss(p, z):
    """Normalized mean squared error between online predictor p and target projection z.
    p, z: (N, D). Equivalent to 2.0 - 2.0 * cosine_similarity(p, z).
    """
    p_norm = F.normalize(p.float(), dim=-1, p=2)
    z_norm = F.normalize(z.float(), dim=-1, p=2)
    return 2.0 - 2.0 * (p_norm * z_norm).sum(dim=-1).mean()


class LooCBYOL(SSLMethod):
    name = "looc_byol"

    def __init__(self, encoder, cfg, meta):
        super().__init__()
        bc = cfg.get("looc_byol", cfg.get("byol", {}))
        self.encoder = encoder

        proj_dim = int(bc.get("proj_dim", 256))
        proj_hidden = int(bc.get("proj_hidden", 2048))
        pred_hidden = int(bc.get("pred_hidden", 2048))
        proj_layers = int(bc.get("proj_layers", 2))
        self.color_weight = float(bc.get("color_weight", 0.60))

        # 1. Dual Online Projectors & Predictors
        self.projector_geo = MLPHead(encoder.out_dim, hidden=proj_hidden, out_dim=proj_dim,
                                     n_layers=proj_layers, last_bn=False)
        self.predictor_geo = MLPHead(proj_dim, hidden=pred_hidden, out_dim=proj_dim,
                                     n_layers=2, last_bn=False)

        self.projector_col = MLPHead(encoder.out_dim, hidden=proj_hidden, out_dim=proj_dim,
                                     n_layers=proj_layers, last_bn=False)
        self.predictor_col = MLPHead(proj_dim, hidden=pred_hidden, out_dim=proj_dim,
                                     n_layers=2, last_bn=False)

        # 2. Target Networks (EMA of online encoder and projectors; no gradients)
        self.target_encoder = copy.deepcopy(encoder)
        self.target_projector_geo = copy.deepcopy(self.projector_geo)
        self.target_projector_col = copy.deepcopy(self.projector_col)

        target_params = (list(self.target_encoder.parameters()) +
                         list(self.target_projector_geo.parameters()) +
                         list(self.target_projector_col.parameters()))
        for p in target_params:
            p.requires_grad = False

        # EMA momentum
        self.m_start = float(bc.get("momentum_target", 0.996))
        self.eval_target = bool(bc.get("eval_target", True))

        # 3. Augmentation Pipeline A: Pure Spatial Multi-Crop (Zero Color Distortion)
        S = meta["image_size"]
        geo_params1 = {
            "crop_scale": [0.20, 1.0],
            "crop_ratio": [0.75, 1.3333],
            "flip_p": 0.5,
            "jitter_p": 0.0,
            "gray_p": 0.0,
            "blur_p": 0.0,
            "solarize_p": 0.0,
        }
        geo_params2 = {
            "crop_scale": [0.10, 0.50],  # local patch crop
            "crop_ratio": [0.75, 1.3333],
            "flip_p": 0.5,
            "jitter_p": 0.0,
            "gray_p": 0.0,
            "blur_p": 0.0,
            "solarize_p": 0.0,
        }
        self.aug_geo1 = GPUAugment(geo_params1, meta["mean"], meta["std"], S)
        self.aug_geo2 = GPUAugment(geo_params2, meta["mean"], meta["std"], S)

        # 4. Augmentation Pipeline B: Pure Color & Lighting Invariance (Full-Frame)
        col_params1 = {
            "crop_scale": [0.95, 1.0],
            "crop_ratio": [1.0, 1.0],
            "flip_p": 0.0,
            "jitter_p": 0.85,
            "brightness": 0.35,
            "contrast": 0.35,
            "saturation": 0.20,
            "hue": 0.08,
            "gray_p": 0.20,
            "blur_p": 0.0,
            "solarize_p": 0.20,
        }
        col_params2 = {
            "crop_scale": [0.95, 1.0],
            "crop_ratio": [1.0, 1.0],
            "flip_p": 0.0,
            "jitter_p": 0.85,
            "brightness": 0.35,
            "contrast": 0.35,
            "saturation": 0.20,
            "hue": 0.08,
            "gray_p": 0.20,
            "blur_p": 0.15,
            "solarize_p": 0.0,
        }
        self.aug_col1 = GPUAugment(col_params1, meta["mean"], meta["std"], S)
        self.aug_col2 = GPUAugment(col_params2, meta["mean"], meta["std"], S)

    def training_step(self, x):
        n = x.shape[0]

        # 1. Generate 4 decoupled views
        v1_geo = self.aug_geo1(x)  # global clean crop
        v2_geo = self.aug_geo2(x)  # local clean patch
        v1_col = self.aug_col1(x)  # full frame color jitter 1
        v2_col = self.aug_col2(x)  # full frame color jitter 2

        # 2. Forward all 4 views together through online encoder (single conv pass)
        v_all = torch.cat([v1_geo, v2_geo, v1_col, v2_col], dim=0)  # (4N, C, H, W)
        h_all = self.enc(v_all)                                      # (4N, 512)

        h1_geo, h2_geo = h_all[:n], h_all[n:2*n]
        h1_col, h2_col = h_all[2*n:3*n], h_all[3*n:]

        # Online projection & prediction
        p1_geo = self.predictor_geo(self.projector_geo(h1_geo))
        p2_geo = self.predictor_geo(self.projector_geo(h2_geo))
        p1_col = self.predictor_col(self.projector_col(h1_col))
        p2_col = self.predictor_col(self.projector_col(h2_col))

        # 3. Target forward (no gradients)
        with torch.no_grad():
            h_t_all = self.target_encoder(v_all)
            h1_t_geo, h2_t_geo = h_t_all[:n], h_t_all[n:2*n]
            h1_t_col, h2_t_col = h_t_all[2*n:3*n], h_t_all[3*n:]

            z1_t_geo = self.target_projector_geo(h1_t_geo)
            z2_t_geo = self.target_projector_geo(h2_t_geo)
            z1_t_col = self.target_projector_col(h1_t_col)
            z2_t_col = self.target_projector_col(h2_t_col)

        # 4. Symmetrized Losses
        # Spatial loss: global predicts local & local predicts global
        loss_geo = 0.5 * (byol_loss(p1_geo, z2_t_geo) + byol_loss(p2_geo, z1_t_geo))
        # Color loss: color 1 predicts color 2 & vice versa
        loss_col = 0.5 * (byol_loss(p1_col, z2_t_col) + byol_loss(p2_col, z1_t_col))

        total_loss = loss_geo + self.color_weight * loss_col

        with torch.no_grad():
            sim_geo = (F.normalize(p1_geo.float(), dim=-1) * F.normalize(z2_t_geo.float(), dim=-1)).sum(dim=-1).mean()
            sim_col = (F.normalize(p1_col.float(), dim=-1) * F.normalize(z2_t_col.float(), dim=-1)).sum(dim=-1).mean()

        logs = {
            "loss": total_loss.detach(),
            "loss_geo": loss_geo.detach(),
            "loss_col": loss_col.detach(),
            "sim_geo": sim_geo.detach(),
            "sim_col": sim_col.detach(),
        }
        return total_loss, logs

    @torch.no_grad()
    def after_step(self, step, total_steps):
        # Cosine ramp of EMA momentum from m_start to 1.0
        m = cosine_value(step, total_steps, self.m_start, 1.0)

        # Update target encoder
        for ps, pt in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            pt.mul_(m).add_(ps.detach(), alpha=1.0 - m)
        for bs, bt in zip(self.encoder.buffers(), self.target_encoder.buffers()):
            bt.copy_(bs)

        # Update target geometry projector
        for ps, pt in zip(self.projector_geo.parameters(), self.target_projector_geo.parameters()):
            pt.mul_(m).add_(ps.detach(), alpha=1.0 - m)
        for bs, bt in zip(self.projector_geo.buffers(), self.target_projector_geo.buffers()):
            bt.copy_(bs)

        # Update target color projector
        for ps, pt in zip(self.projector_col.parameters(), self.target_projector_col.parameters()):
            pt.mul_(m).add_(ps.detach(), alpha=1.0 - m)
        for bs, bt in zip(self.projector_col.buffers(), self.target_projector_col.buffers()):
            bt.copy_(bs)

    def eval_encoder(self):
        # Target encoder maintains moving-average parameters (smoother representations)
        return self.target_encoder if self.eval_target else self.encoder

    def trainable_named_parameters(self):
        return [(n, p) for n, p in self.named_parameters()
                if p.requires_grad and not n.startswith("target_")]


# --------------------------------------------------------------------------
# Auto-registration hook: allows ssl.method=looc_byol without base.py edits
# --------------------------------------------------------------------------
import ssl_methods.base as _base

_orig_build = getattr(_base, "build_ssl_method", None)
if _orig_build is not None:
    def _patched_build(cfg, encoder, meta):
        name = cfg.get("ssl", {}).get("method", "").lower()
        if name in ("looc_byol", "loocbyol"):
            return LooCBYOL(encoder, cfg, meta)
        return _orig_build(cfg, encoder, meta)

    _base.build_ssl_method = _patched_build
