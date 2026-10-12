"""Noisy-label probe methods = optional cleaner + loss, written as '<cleaner>+<loss>' or '<loss>'.

Shared by train_probe.py (method comparison) and fit_ood.py (final fit with the selected method),
so the deployed head is trained exactly like the head that was evaluated.
"""
import numpy as np
import torch

from probe.cleaning import run_cleaner
from probe.linear import out_of_fold_probs, train_linear

DEFAULT_METHODS = ["ce", "ls", "sce", "gce", "bootstrap", "knn+ce", "proto+ce", "smallloss+ce",
                   "cl+ce", "consensus+ce", "consensus+gce"]


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
    head = train_linear(banks, labels, num_classes, pcfg, loss, seed, device, wd=wd,
                        sample_idx=np.where(keep)[0])
    return head, keep, info


def estimate_noise_rate(banks, labels, num_classes, pcfg, seed, device, wd=None):
    from probe.cleaning import confident_learning
    probs = out_of_fold_probs(banks, labels, num_classes, pcfg, pcfg.get("cleaning_folds", 5), seed, device, wd)
    _, eta, Q = confident_learning(probs, labels, num_classes)
    return eta, Q, probs


def to_t(x):
    return torch.as_tensor(np.asarray(x), dtype=torch.long)
