"""Representation-collapse diagnostics.

- feature std of L2-normalised features (SimSiam, Chen & He 2021): for a non-collapsed d-dim
  embedding the per-dimension std is ~ 1/sqrt(d); complete collapse -> 0.
- effective rank (RankMe, Garrido et al. 2023): exp(entropy of normalised singular values).
  1 = rank-1 collapse, d = perfectly isotropic.
"""
import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def collapse_stats(feats: torch.Tensor):
    f = feats.float()
    d = f.shape[1]
    z = F.normalize(f, dim=1)
    std = z.std(dim=0).mean().item()
    fc = f - f.mean(0, keepdim=True)
    try:
        s = torch.linalg.svdvals(fc)
    except Exception:
        s = torch.linalg.svdvals(fc.cpu())
    p = s / s.sum().clamp_min(1e-12)
    erank = math.exp(-(p * torch.log(p.clamp_min(1e-12))).sum().item())
    return {"feat_std": std, "feat_std_ref": 1.0 / math.sqrt(d), "eff_rank": erank, "dim": d,
            "eff_rank_frac": erank / min(d, f.shape[0])}


def collapse_warning(stats, min_std_frac=0.2, min_erank=8):
    msgs = []
    if stats["feat_std"] < min_std_frac * stats["feat_std_ref"]:
        msgs.append(f"normalised feature std {stats['feat_std']:.4f} << 1/sqrt(d)={stats['feat_std_ref']:.4f}")
    if stats["eff_rank"] < min_erank:
        msgs.append(f"effective rank {stats['eff_rank']:.1f} < {min_erank}")
    return msgs
