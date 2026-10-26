"""SwAV (Caron et al., NeurIPS 2020): Swapping Assignments between Views.

Online clustering of visual representations using optimal transport (Sinkhorn-Knopp).
Maps representations to a set of prototype vectors, and computes swapped cross-entropy loss:
view 1 features predict the cluster assignment of view 2, and vice versa.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.heads import MLPHead
from ssl_methods.augment import GPUAugment
from ssl_methods.base import SSLMethod


@torch.no_grad()
def distributed_sinkhorn(scores, eps=0.05, niters=3):
    """Computes soft cluster assignments Q from dot-product scores (B, K) using Sinkhorn-Knopp."""
    # Numerically stable exp by subtracting row max
    s = scores / eps
    s = s - torch.max(s, dim=1, keepdim=True)[0]
    Q = torch.exp(s).t()  # (K, B)
    B = Q.shape[1]
    K = Q.shape[0]

    sum_Q = torch.sum(Q)
    if torch.isnan(sum_Q) or sum_Q == 0:
        return F.softmax(scores / eps, dim=1)

    Q /= sum_Q

    for _ in range(niters):
        # Normalize rows (prototypes) so each gets 1/K total mass
        sum_rows = torch.sum(Q, dim=1, keepdim=True)
        Q /= (sum_rows + 1e-10)
        Q /= K

        # Normalize columns (samples) so each gets 1/B total mass
        sum_cols = torch.sum(Q, dim=0, keepdim=True)
        Q /= (sum_cols + 1e-10)
        Q /= B

    Q *= B  # Each sample's column sums to 1
    return Q.t()  # (B, K)


class SwAV(SSLMethod):
    name = "swav"

    def __init__(self, encoder, cfg, meta):
        super().__init__()
        sc = cfg.get("swav", {})
        self.encoder = encoder

        # Hyperparameters
        self.n_prototypes = int(sc.get("n_prototypes", 128))
        self.temperature = float(sc.get("temperature", 0.1))
        self.sinkhorn_eps = float(sc.get("sinkhorn_eps", 0.05))
        self.sinkhorn_iters = int(sc.get("sinkhorn_iters", 3))

        proj_dim = int(sc.get("proj_dim", 256))
        proj_hidden = int(sc.get("proj_hidden", 2048))
        proj_layers = int(sc.get("proj_layers", 3))

        # Projector: MLP (in_dim -> hidden -> proj_dim)
        self.projector = MLPHead(encoder.out_dim, hidden=proj_hidden, out_dim=proj_dim,
                                 n_layers=proj_layers, last_bn=True)

        # Prototypes: (n_prototypes, proj_dim)
        self.prototypes = nn.Linear(proj_dim, self.n_prototypes, bias=False)
        self.normalize_prototypes()

        # Augmentation setups for two views (supporting asymmetric global-to-local crops)
        v1_cfg = dict(cfg.get("aug", {}))
        v1_cfg.update(sc.get("view1_overrides", {}) or {})
        v2_cfg = dict(cfg.get("aug", {}))
        v2_cfg.update(sc.get("view2_overrides", {}) or {})
        self.aug1 = GPUAugment(v1_cfg, meta["mean"], meta["std"], meta["image_size"])
        self.aug2 = GPUAugment(v2_cfg, meta["mean"], meta["std"], meta["image_size"])

    @torch.no_grad()
    def normalize_prototypes(self):
        """Projects prototype vectors onto the unit sphere."""
        w = self.prototypes.weight.data
        w = F.normalize(w, dim=1, p=2)
        self.prototypes.weight.data.copy_(w)

    def eval_encoder(self):
        return self.encoder

    def training_step(self, x):
        # Ensure prototypes are unit-normalized
        self.normalize_prototypes()

        n = x.shape[0]
        v1 = self.aug1(x)
        v2 = self.aug2(x)

        # Forward encoder and projector for both views
        v_all = torch.cat([v1, v2], dim=0)
        h_all = self.enc(v_all)
        z_all = self.projector(h_all)
        # Normalize projected features to unit sphere
        z_all = F.normalize(z_all, dim=-1, p=2)

        # Dot-product similarity scores to prototypes: (2N, K)
        scores_all = self.prototypes(z_all)
        scores1 = scores_all[:n]
        scores2 = scores_all[n:]

        # Optimal transport assignments via Sinkhorn-Knopp
        q1 = distributed_sinkhorn(scores1, eps=self.sinkhorn_eps, niters=self.sinkhorn_iters)
        q2 = distributed_sinkhorn(scores2, eps=self.sinkhorn_eps, niters=self.sinkhorn_iters)

        # Swapped prediction cross-entropy loss
        log_p1 = F.log_softmax(scores1 / self.temperature, dim=1)
        log_p2 = F.log_softmax(scores2 / self.temperature, dim=1)

        loss1 = -torch.mean(torch.sum(q2 * log_p1, dim=1))
        loss2 = -torch.mean(torch.sum(q1 * log_p2, dim=1))
        loss = 0.5 * (loss1 + loss2)

        logs = {
            "swav_loss": loss.detach(),
            "loss1": loss1.detach(),
            "loss2": loss2.detach(),
        }
        return loss, logs


# --------------------------------------------------------------------------
# Auto-registration hook: allows ssl.method=swav without requiring base.py edits
# --------------------------------------------------------------------------
import ssl_methods.base as _base

_orig_build = getattr(_base, "build_ssl_method", None)
if _orig_build is not None:
    def _patched_build(cfg, encoder, meta):
        name = cfg.get("ssl", {}).get("method", "").lower()
        if name in ("swav",):
            return SwAV(encoder, cfg, meta)
        return _orig_build(cfg, encoder, meta)

    _base.build_ssl_method = _patched_build
