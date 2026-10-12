"""Noise-robust classification losses (all from scratch).

ce         : standard cross entropy
ls         : label smoothing (Szegedy et al. 2016) - softens targets, limits over-confidence on wrong labels
sce        : symmetric CE = alpha*CE + beta*RCE (Wang et al. 2019); RCE uses log(0) := A = -4
gce        : generalised CE  (1 - p_y^q)/q (Zhang & Sabuncu 2018); q->0 = CE, q=1 = MAE (robust)
bootstrap  : soft bootstrapping, target = beta*y + (1-beta)*p (Reed et al. 2015)
"""
import torch
import torch.nn.functional as F


def make_loss(name, num_classes, params=None):
    p = params or {}
    name = name.lower()

    if name == "ce":
        return lambda logits, y: F.cross_entropy(logits, y, reduction="none")

    if name == "ls":
        eps = float(p.get("ls_eps", 0.1))
        return lambda logits, y: F.cross_entropy(logits, y, reduction="none", label_smoothing=eps)

    if name == "sce":
        alpha, beta, A = float(p.get("sce_alpha", 0.1)), float(p.get("sce_beta", 1.0)), float(p.get("sce_A", -4.0))

        def sce(logits, y):
            ce = F.cross_entropy(logits, y, reduction="none")
            prob = F.softmax(logits.float(), 1).clamp(min=1e-7, max=1.0)
            oh = F.one_hot(y, num_classes).float()
            # RCE = -sum_k p_k log q_k with q = one-hot; log 0 is clamped to A
            rce = -(prob * torch.where(oh > 0, torch.zeros_like(oh), torch.full_like(oh, A))).sum(1)
            return alpha * ce + beta * rce
        return sce

    if name == "gce":
        q = float(p.get("gce_q", 0.7))

        def gce(logits, y):
            py = F.softmax(logits.float(), 1).gather(1, y.view(-1, 1)).squeeze(1).clamp_min(1e-7)
            return (1.0 - py ** q) / q
        return gce

    if name == "bootstrap":
        beta = float(p.get("boot_beta", 0.8))

        def boot(logits, y):
            logp = F.log_softmax(logits.float(), 1)
            target = beta * F.one_hot(y, num_classes).float() + (1 - beta) * logp.exp().detach()
            return -(target * logp).sum(1)
        return boot

    raise ValueError(f"unknown loss '{name}'")


LOSSES = ("ce", "ls", "sce", "gce", "bootstrap")
