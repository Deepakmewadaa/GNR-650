"""Optimisers and LR schedules: LARS (You et al. 2017, as used by SimCLR), SGD, AdamW; warmup+cosine."""
import math

import torch


class LARS(torch.optim.Optimizer):
    """LARS with SimCLR conventions: weight decay added to the gradient *before* the trust ratio;
    biases and normalisation params (ndim<=1) are excluded from both decay and adaptation."""

    def __init__(self, params, lr, momentum=0.9, weight_decay=0.0, eta=0.001):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay, eta=eta,
                                      lars_exclude=False))

    @torch.no_grad()
    def step(self, closure=None):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                d = p.grad
                if not g["lars_exclude"]:
                    if g["weight_decay"]:
                        d = d.add(p, alpha=g["weight_decay"])
                    pn, dn = torch.norm(p), torch.norm(d)
                    trust = torch.where((pn > 0) & (dn > 0), g["eta"] * pn / dn, torch.ones_like(pn))
                    d = d.mul(trust)
                st = self.state[p]
                if "mu" not in st:
                    st["mu"] = torch.clone(d).detach()
                else:
                    st["mu"].mul_(g["momentum"]).add_(d)
                p.add_(st["mu"], alpha=-g["lr"])


def param_groups(named_params, weight_decay):
    """Two groups: weights (decayed, LARS-adapted) and biases/BN/1-d params (no decay, no LARS)."""
    decay, no_decay = [], []
    for n, p in named_params:
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay, "lars_exclude": False},
            {"params": no_decay, "weight_decay": 0.0, "lars_exclude": True}]


def build_optimizer(named_params, name, lr, weight_decay, momentum=0.9, betas=(0.9, 0.999)):
    groups = param_groups(named_params, weight_decay)
    name = name.lower()
    if name == "lars":
        return LARS(groups, lr=lr, momentum=momentum, eta=0.001)
    for g in groups:
        g.pop("lars_exclude", None)
    if name == "sgd":
        return torch.optim.SGD(groups, lr=lr, momentum=momentum, nesterov=True)
    if name == "adamw":
        return torch.optim.AdamW(groups, lr=lr, betas=betas)
    raise ValueError(f"unknown optimizer {name}")


def cosine_lr(step, total_steps, warmup_steps, base_lr, final_lr=0.0):
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    t = min(1.0, t)
    return final_lr + 0.5 * (base_lr - final_lr) * (1 + math.cos(math.pi * t))


def cosine_value(step, total_steps, start, end):
    """Cosine ramp from start to end (used for DINO teacher momentum)."""
    t = min(1.0, step / max(1, total_steps))
    return end - (end - start) * 0.5 * (1 + math.cos(math.pi * t))


def set_lr(opt, lr):
    for g in opt.param_groups:
        g["lr"] = lr
