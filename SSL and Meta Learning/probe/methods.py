"""Noisy-label probe methods = optional cleaner + loss, written as '<cleaner>+<loss>' or '<loss>'.

Shared by train_probe.py (method comparison) and fit_ood.py (final fit with the selected method),
so the deployed head is trained exactly like the head that was evaluated.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from probe.cleaning import run_cleaner
from probe.linear import out_of_fold_probs, train_linear

DEFAULT_METHODS = ["ce", "ls", "sce", "gce", "bootstrap", "knn+ce", "proto+ce", "smallloss+ce",
                   "cl+ce", "consensus+ce", "consensus+gce"]

FEWSHOT_LOSSES = ("proto", "protonet", "simpleshot", "ncm", "ensemble", "proto_ensemble")


def build_fewshot_head(method, support_feats, support_labels, num_classes, pcfg, seed, device, wd=None,
                       banks_all=None):
    """Closed-form few-shot and prototype classifier heads (ProtoNet, SimpleShot, Proto-Adam Ensemble).
    Returns an nn.Linear(d, num_classes) evaluated on raw features.
    """
    K = num_classes
    d = support_feats.shape[1]
    tau = float(pcfg.get("fewshot_tau", 0.07))
    support_feats = support_feats.float()
    support_labels = np.asarray(support_labels)

    if method in ("proto", "protonet"):
        z = F.normalize(support_feats, dim=1)
        protos = torch.zeros(K, d)
        for c in range(K):
            mask = (support_labels == c)
            if mask.sum() > 0:
                protos[c] = F.normalize(z[mask].mean(0), dim=0)
            else:
                protos[c] = F.normalize(torch.randn(d), dim=0)
        head = nn.Linear(d, K)
        head.weight.data.copy_(protos / tau)
        head.bias.data.zero_()
        return head.eval()

    elif method in ("simpleshot", "ncm"):
        global_mean = support_feats.mean(0)
        f_centered = support_feats - global_mean.unsqueeze(0)
        z = F.normalize(f_centered, dim=1)
        protos = torch.zeros(K, d)
        for c in range(K):
            mask = (support_labels == c)
            if mask.sum() > 0:
                protos[c] = F.normalize(z[mask].mean(0), dim=0)
            else:
                protos[c] = F.normalize(torch.randn(d), dim=0)
        W = protos / tau
        b = -torch.matmul(protos, global_mean) / tau
        head = nn.Linear(d, K)
        head.weight.data.copy_(W)
        head.bias.data.copy_(b)
        return head.eval()

    elif method in ("ensemble", "proto_ensemble"):
        # Convex combination of ProtoNet (global cluster structure) + Adam linear head (boundary discrimination)
        head_proto = build_fewshot_head("proto", support_feats, support_labels, K, pcfg, seed, device, wd)
        banks_for_adam = banks_all if banks_all is not None else support_feats.unsqueeze(0)
        head_adam = train_linear(banks_for_adam, support_labels, K, pcfg, "ls", seed, device, wd=wd)

        W1, b1 = head_proto.weight.data, head_proto.bias.data
        W2, b2 = head_adam.weight.data, head_adam.bias.data
        norm1 = torch.norm(W1) + 1e-7
        norm2 = torch.norm(W2) + 1e-7
        W2_scaled = W2 * (norm1 / norm2)
        b2_scaled = b2 * (norm1 / norm2) if torch.norm(b2) > 0 else b2

        alpha = float(pcfg.get("ensemble_alpha", 0.5))
        W = alpha * W1 + (1.0 - alpha) * W2_scaled
        b = alpha * b1 + (1.0 - alpha) * b2_scaled
        head = nn.Linear(d, K)
        head.weight.data.copy_(W)
        head.bias.data.copy_(b)
        return head.eval()

    raise ValueError(f"Unknown few-shot method '{method}'")


def parse_method(name):
    if "+" in name:
        c, l = name.split("+", 1)
        return c, l
    return None, name


def fit_method(name, banks, labels, num_classes, pcfg, seed, device, wd=None, oof_cache=None):
    """banks: (V, N, d) features of the TRAINING samples only; labels: (N,) noisy labels.
    Returns (head nn.Linear on raw features, keep_mask over the N samples, info dict)."""
    labels = np.asarray(labels)
    cleaner, loss = parse_method(name)
    info = {"method": name}
    keep = np.ones(len(labels), bool)
    if cleaner is not None:
        key = ("oof", seed)
        if oof_cache is not None and key in oof_cache:
            probs = oof_cache[key]
        else:
            probs = out_of_fold_probs(banks, labels, num_classes, pcfg, pcfg.get("cleaning_folds", 5), seed,
                                      device, wd)
            if oof_cache is not None:
                oof_cache[key] = probs
        keep, cinfo = run_cleaner(cleaner, banks[0], labels, num_classes, probs, pcfg.get("cleaning", {}))
        info.update(cinfo)

    kept_idx = np.where(keep)[0]
    if loss in FEWSHOT_LOSSES:
        head = build_fewshot_head(loss, banks[0, kept_idx], labels[kept_idx], num_classes, pcfg, seed, device,
                                 wd=wd, banks_all=banks[:, kept_idx])
    else:
        head = train_linear(banks, labels, num_classes, pcfg, loss, seed, device, wd=wd,
                            sample_idx=kept_idx)
    return head, keep, info


def estimate_noise_rate(banks, labels, num_classes, pcfg, seed, device, wd=None):
    from probe.cleaning import confident_learning
    probs = out_of_fold_probs(banks, labels, num_classes, pcfg, pcfg.get("cleaning_folds", 5), seed, device, wd)
    _, eta, Q = confident_learning(probs, labels, num_classes)
    return eta, Q, probs


def to_t(x):
    return torch.as_tensor(np.asarray(x), dtype=torch.long)
