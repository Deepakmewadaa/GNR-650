"""Stratified splits with fixed seeds (numpy only -> identical on every machine)."""
import numpy as np


def stratified_split(labels, val_frac=0.2, seed=0, min_val=1):
    """Per-class random split. Classes with a single sample go to train."""
    labels = np.asarray(labels)
    rng = np.random.RandomState(seed)
    tr, va = [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n_val = int(round(val_frac * len(idx)))
        n_val = max(min_val, n_val) if len(idx) >= 2 else 0
        n_val = min(n_val, len(idx) - 1)
        va.append(idx[:n_val])
        tr.append(idx[n_val:])
    return np.sort(np.concatenate(tr)), np.sort(np.concatenate(va))


def kfold_stratified(labels, n_folds=5, seed=0):
    """Return list of (train_idx, val_idx) covering every sample exactly once in a val fold."""
    labels = np.asarray(labels)
    rng = np.random.RandomState(seed)
    fold_of = np.empty(len(labels), dtype=np.int64)
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        fold_of[idx] = (np.arange(len(idx)) + rng.randint(n_folds)) % n_folds
    out = []
    for f in range(n_folds):
        va = np.where(fold_of == f)[0]
        tr = np.where(fold_of != f)[0]
        out.append((tr, va))
    return out


def holdout_classes(num_classes, frac, seed):
    """Pick pseudo-unknown classes for the leave-class-out protocol."""
    rng = np.random.RandomState(1000 + seed)
    n = int(round(frac * num_classes))
    n = min(max(1, n), max(1, num_classes - 2))
    return np.sort(rng.choice(num_classes, n, replace=False))
