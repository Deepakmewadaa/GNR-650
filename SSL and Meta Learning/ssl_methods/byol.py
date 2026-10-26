"""BYOL (Grill et al., NeurIPS 2020): Bootstrap Your Own Latent.

Self-supervised representation learning without negative pairs:
  - Online network: encoder -> MLP projector -> MLP predictor.
  - Target network: EMA of online encoder -> EMA of online projector (no predictor).
  - Objective: normalized MSE (cosine distance) between online prediction and target projection.
  - Symmetrized across two augmented views.
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
    p, z: (N, D).
    Equivalent to 2.0 - 2.0 * cosine_similarity(p, z).
    """
    p_norm = F.normalize(p.float(), dim=-1, p=2)
    z_norm = F.normalize(z.float(), dim=-1, p=2)
    return 2.0 - 2.0 * (p_norm * z_norm).sum(dim=-1).mean()


class BYOL(SSLMethod):
    name = "byol"

    def __init__(self, encoder, cfg, meta):
        super().__init__()
        bc = cfg.get("byol", {})
        self.encoder = encoder

        # Projections & Predictor dimensions
        proj_dim = int(bc.get("proj_dim", 256))
        proj_hidden = int(bc.get("proj_hidden", 2048))
        pred_hidden = int(bc.get("pred_hidden", 2048))
        proj_layers = int(bc.get("proj_layers", 2))

        # Online projector and predictor
        self.projector = MLPHead(encoder.out_dim, hidden=proj_hidden, out_dim=proj_dim,
                                 n_layers=proj_layers, last_bn=False)
        self.predictor = MLPHead(proj_dim, hidden=pred_hidden, out_dim=proj_dim,
                                 n_layers=2, last_bn=False)

        # Target network: EMA of online encoder and projector (no gradients)
        self.target_encoder = copy.deepcopy(encoder)
        self.target_projector = copy.deepcopy(self.projector)
        for p in list(self.target_encoder.parameters()) + list(self.target_projector.parameters()):
            p.requires_grad = False

        # EMA momentum parameters
        self.m_start = float(bc.get("momentum_target", 0.996))
        self.eval_target = bool(bc.get("eval_target", True))

        # Augmentation setups for views
        self.n_views = int(bc.get("n_views", 2))
        base_aug = dict(cfg.get("aug", {}))
        v1_cfg = dict(base_aug)
        v1_cfg.update(bc.get("view1_overrides", {}) or {})
        v2_cfg = dict(base_aug)
        # BYOL default view 2: lower blur probability and mild solarization
        v2_defaults = {"blur_p": 0.1, "solarize_p": 0.2}
        v2_defaults.update(bc.get("view2_overrides", {}) or {})
        v2_cfg.update(v2_defaults)

        S = meta["image_size"]
        self.aug1 = GPUAugment(v1_cfg, meta["mean"], meta["std"], S)
        self.aug2 = GPUAugment(v2_cfg, meta["mean"], meta["std"], S)

        if self.n_views == 4:
            local_scale = list(bc.get("local_crop_scale", [0.12, 0.60]))
            v3_cfg = dict(v1_cfg)
            v3_cfg["crop_scale"] = local_scale
            v4_cfg = dict(v2_cfg)
            v4_cfg["crop_scale"] = local_scale
            self.aug3 = GPUAugment(v3_cfg, meta["mean"], meta["std"], S)
            self.aug4 = GPUAugment(v4_cfg, meta["mean"], meta["std"], S)

    def training_step(self, x):
        n = x.shape[0]

        if self.n_views == 4:
            # 4-View Multi-Crop BYOL: 2 global views + 2 local crops
            v1 = self.aug1(x)
            v2 = self.aug2(x)
            v3 = self.aug3(x)
            v4 = self.aug4(x)

            # Online network encodes all 4 views (global + local)
            v_all = torch.cat([v1, v2, v3, v4], dim=0)
            h_all = self.enc(v_all)
            z_all = self.projector(h_all)
            p_all = self.predictor(z_all)
            p1, p2, p3, p4 = p_all[:n], p_all[n:2*n], p_all[2*n:3*n], p_all[3*n:]

            # Target network only encodes the 2 global views (stable representation targets)
            with torch.no_grad():
                v_target = torch.cat([v1, v2], dim=0)
                h_t_all = self.target_encoder(v_target)
                z_t_all = self.target_projector(h_t_all)
                z1_t, z2_t = z_t_all[:n], z_t_all[n:]

            # Symmetrized global loss + local-to-global predictions
            loss_global = 0.5 * (byol_loss(p1, z2_t) + byol_loss(p2, z1_t))
            loss_local = 0.25 * (byol_loss(p3, z1_t) + byol_loss(p3, z2_t) +
                                 byol_loss(p4, z1_t) + byol_loss(p4, z2_t))
            loss = 0.5 * loss_global + 0.5 * loss_local

            with torch.no_grad():
                sim1 = (F.normalize(p1.float(), dim=-1) * F.normalize(z2_t.float(), dim=-1)).sum(dim=-1).mean()
                sim2 = (F.normalize(p2.float(), dim=-1) * F.normalize(z1_t.float(), dim=-1)).sum(dim=-1).mean()
                cos_sim = 0.5 * (sim1 + sim2)

            logs = {"cos_sim": cos_sim.detach()}
            return loss, logs

        # Standard 2-view BYOL
        # 1. Generate two views on GPU
        v1 = self.aug1(x)
        v2 = self.aug2(x)

        # 2. Online forward on concatenated views: [v1; v2]
        v_all = torch.cat([v1, v2], dim=0)
        h_all = self.enc(v_all)
        z_all = self.projector(h_all)
        p_all = self.predictor(z_all)
        p1, p2 = p_all[:n], p_all[n:]

        # 3. Target forward (no gradients)
        with torch.no_grad():
            h_t_all = self.target_encoder(v_all)
            z_t_all = self.target_projector(h_t_all)
            z1_t, z2_t = z_t_all[:n], z_t_all[n:]

        # 4. Symmetrized BYOL loss
        loss1 = byol_loss(p1, z2_t)
        loss2 = byol_loss(p2, z1_t)
        loss = 0.5 * (loss1 + loss2)

        # Metric: cosine similarity between online predictions and target representations
        with torch.no_grad():
            sim1 = (F.normalize(p1.float(), dim=-1) * F.normalize(z2_t.float(), dim=-1)).sum(dim=-1).mean()
            sim2 = (F.normalize(p2.float(), dim=-1) * F.normalize(z1_t.float(), dim=-1)).sum(dim=-1).mean()
            cos_sim = 0.5 * (sim1 + sim2)

        logs = {"cos_sim": cos_sim.detach()}
        return loss, logs

    @torch.no_grad()
    def after_step(self, step, total_steps):
        # Cosine ramp of EMA momentum from m_start to 1.0
        m = cosine_value(step, total_steps, self.m_start, 1.0)
        for ps, pt in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            pt.mul_(m).add_(ps.detach(), alpha=1.0 - m)
        for ps, pt in zip(self.projector.parameters(), self.target_projector.parameters()):
            pt.mul_(m).add_(ps.detach(), alpha=1.0 - m)
        for bs, bt in zip(self.encoder.buffers(), self.target_encoder.buffers()):
            bt.copy_(bs)
        for bs, bt in zip(self.projector.buffers(), self.target_projector.buffers()):
            bt.copy_(bs)

    def eval_encoder(self):
        # Target encoder maintains moving-average parameters (smoother representations)
        return self.target_encoder if self.eval_target else self.encoder

    def trainable_named_parameters(self):
        return [(n, p) for n, p in self.named_parameters()
                if p.requires_grad and not n.startswith("target_")]


# --------------------------------------------------------------------------
# Auto-registration hook: allows ssl.method=byol when imported
# --------------------------------------------------------------------------
import ssl_methods.base as _base

_orig_build = getattr(_base, "build_ssl_method", None)
if _orig_build is not None:
    def _patched_build(cfg, encoder, meta):
        name = cfg.get("ssl", {}).get("method", "").lower()
        if name in ("byol",):
            return BYOL(encoder, cfg, meta)
        return _orig_build(cfg, encoder, meta)
    _base.build_ssl_method = _patched_build
