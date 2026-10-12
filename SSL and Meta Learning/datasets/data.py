"""High-level data loading used by every training script.

`load_data(cfg, paths)` discovers folders, decides image size / channels from the real data (unless
forced, e.g. by an encoder checkpoint so all stages agree), builds the uint8 caches and returns a
DataBundle with numpy uint8 arrays + labels + metadata.
"""
import os
from dataclasses import dataclass, field

import numpy as np

from datasets.cache import build_cache
from datasets.discover import find_data_dirs, list_labelled, list_unlabelled
from datasets.io import probe_image

GRAY_MODES = {"L", "1", "I", "I;16", "I;16B", "I;16L", "I;16N", "F", "LA"}


@dataclass
class DataBundle:
    unl: np.ndarray = None            # (N_u, C, S, S) uint8 (memmap)
    lab: np.ndarray = None            # (N_l, C, S, S) uint8 (memmap)
    labels: np.ndarray = None         # (N_l,) int64 noisy labels
    lab_paths: list = field(default_factory=list)
    unl_paths: list = field(default_factory=list)
    class_names: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)


def _sample(lst, n, seed=0):
    if len(lst) <= n:
        return list(lst)
    rng = np.random.RandomState(seed)
    return [lst[i] for i in sorted(rng.choice(len(lst), n, replace=False))]


def scan_stats(paths, n=300):
    sizes, modes = [], {}
    for p in _sample(paths, n):
        try:
            w, h, m = probe_image(p)
            sizes.append((w, h))
            modes[m] = modes.get(m, 0) + 1
        except Exception:
            modes["<unreadable>"] = modes.get("<unreadable>", 0) + 1
    return sizes, modes


def decide_size_channels(dcfg, paths):
    sizes, modes = scan_stats(paths)
    n_gray = sum(v for k, v in modes.items() if k in GRAY_MODES)
    n_tot = max(1, sum(v for k, v in modes.items() if k != "<unreadable>"))
    ch = dcfg.get("channels", "auto")
    channels = int(ch) if ch not in (None, "auto") else (1 if n_gray == n_tot else 3)
    native = int(np.median([min(w, h) for w, h in sizes])) if sizes else 64
    sz = dcfg.get("image_size", "auto")
    if sz in (None, "auto"):
        size = min(int(dcfg.get("max_image_size", 128)), native)
        size = max(16, size - size % 8)   # multiple of 8 keeps the ResNet stride pattern clean
    else:
        size = int(sz)
    return size, channels, {"native_min_side_median": native, "modes": modes,
                            "n_scanned": len(sizes)}


def _limit_labelled(paths, labels, max_per_class, seed=0):
    if not max_per_class:
        return paths, labels
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels)
    keep = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        keep += sorted(rng.permutation(idx)[:max_per_class].tolist())
    keep = sorted(keep)
    return [paths[i] for i in keep], labels[keep].tolist()


def load_data(cfg, out, need_unlabelled=True, need_labelled=True, force_meta=None, verbose=True):
    d = cfg["data"]
    lab_dir, unl_dir = find_data_dirs(d.get("root"), d.get("labelled_dir"), d.get("unlabelled_dir"), verbose)
    if need_labelled and not lab_dir:
        raise SystemExit("ERROR: labelled data folder not found. Put the dataset in ./data or ./dataset "
                         "(folders named like 'labelled_data' and 'unlabelled_data') or pass --data-root / "
                         "--labelled-dir.")
    if need_unlabelled and not unl_dir:
        raise SystemExit("ERROR: unlabelled data folder not found (see --data-root / --unlabelled-dir).")

    b = DataBundle()
    lab_paths, labels, classes, rule = list_labelled(lab_dir) if lab_dir else ([], [], [], None)
    lab_paths, labels = _limit_labelled(lab_paths, labels, d.get("max_labelled_per_class"))
    unl_paths = list_unlabelled(unl_dir) if (unl_dir and need_unlabelled) else []
    if d.get("max_unlabelled") and len(unl_paths) > int(d["max_unlabelled"]):
        unl_paths = _sample(unl_paths, int(d["max_unlabelled"]), seed=0)

    if force_meta:
        size, channels = int(force_meta["image_size"]), int(force_meta["channels"])
        scan = {"forced_from_checkpoint": True}
    else:
        size, channels, scan = decide_size_channels(d, unl_paths or lab_paths)
    workers = int(d.get("cache_workers", min(16, os.cpu_count() or 4)))

    if need_unlabelled and unl_paths:
        arr, ok = build_cache(unl_paths, size, channels, out["cache"], "unlabelled", workers, verbose)
        b.unl = arr if ok.all() else arr[np.where(ok)[0]]
        b.unl_paths = [p for p, o in zip(unl_paths, ok) if o]
    if need_labelled and lab_paths:
        arr, ok = build_cache(lab_paths, size, channels, out["cache"], "labelled", workers, verbose)
        idx = np.where(ok)[0]
        b.lab = arr if ok.all() else arr[idx]
        b.labels = np.asarray(labels, dtype=np.int64)[idx]
        b.lab_paths = [lab_paths[i] for i in idx]
    b.class_names = classes

    if force_meta and "mean" in force_meta:
        mean, std = force_meta["mean"], force_meta["std"]
    else:
        src = b.unl if b.unl is not None else b.lab
        mean, std = channel_stats(src)
    b.meta = {
        "image_size": size, "channels": channels, "mean": mean, "std": std,
        "num_classes": len(classes), "class_names": classes, "class_order_rule": rule,
        "n_unlabelled": 0 if b.unl is None else int(b.unl.shape[0]),
        "n_labelled": 0 if b.lab is None else int(b.lab.shape[0]),
        "labelled_dir": lab_dir, "unlabelled_dir": unl_dir, "scan": scan,
    }
    if verbose:
        print(f"[data] K={b.meta['num_classes']} labelled={b.meta['n_labelled']} "
              f"unlabelled={b.meta['n_unlabelled']} size={size} channels={channels} "
              f"mean={[round(x, 3) for x in mean]} std={[round(x, 3) for x in std]}")
    return b


def channel_stats(arr, n=5000):
    if arr is None or len(arr) == 0:
        return [0.5], [0.25]
    idx = np.sort(np.random.RandomState(0).choice(len(arr), min(n, len(arr)), replace=False))
    x = np.asarray(arr[idx], dtype=np.float32) / 255.0
    mean = x.mean(axis=(0, 2, 3))
    std = x.std(axis=(0, 2, 3)) + 1e-6
    return [float(v) for v in mean], [float(v) for v in std]


def find_duplicates(lab_paths, unl_paths):
    """Indices of unlabelled files whose bytes are identical to some labelled file (size pre-filter + md5).
    On the real data ALL labelled images are also in the unlabelled folder."""
    import hashlib

    def md5(p):
        h = hashlib.md5()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    by_size = {}
    for i, p in enumerate(unl_paths):
        try:
            by_size.setdefault(os.path.getsize(p), []).append(i)
        except OSError:
            pass
    lab_hashes = {}
    for p in lab_paths:
        try:
            s = os.path.getsize(p)
        except OSError:
            continue
        if s in by_size:
            lab_hashes.setdefault(s, set()).add(md5(p))
    dup = []
    for s, idxs in by_size.items():
        if s in lab_hashes:
            dup += [i for i in idxs if md5(unl_paths[i]) in lab_hashes[s]]
    return np.array(sorted(dup), dtype=np.int64)


def ssl_partition(unl_paths, lab_paths, dcfg, verbose=True):
    """Split the unlabelled pool into
       ssl_idx     : images used for SSL training,
       holdout_idx : a fixed random 'calibration holdout' NEVER seen by SSL (behaves like the unseen test set):
                     used as the background bank of the knn_ratio OOD score and to estimate the
                     known/unknown mixture for the threshold,
       dup_idx     : copies of labelled images; excluded from SSL (exclude_labelled_from_ssl) so the labelled
                     images are also unseen by the encoder -> probe/OOD calibration matches test conditions.
    Deterministic given the file lists (fixed seed), so every stage recomputes the same partition."""
    n = len(unl_paths)
    dup = find_duplicates(lab_paths, unl_paths) if (lab_paths and dcfg.get("exclude_labelled_from_ssl", True)) \
        else np.zeros(0, np.int64)
    rest = np.setdiff1d(np.arange(n), dup)
    n_hold = int(min(int(dcfg.get("calib_holdout", 2500)), float(dcfg.get("calib_holdout_max_frac", 0.2)) * len(rest)))
    rng = np.random.RandomState(int(dcfg.get("calib_holdout_seed", 12345)))
    hold = np.sort(rng.choice(rest, n_hold, replace=False)) if n_hold > 0 else np.zeros(0, np.int64)
    ssl_idx = np.setdiff1d(rest, hold)
    if verbose:
        print(f"[data] unlabelled partition: {len(ssl_idx)} for SSL, {len(hold)} calibration holdout (never seen "
              f"by SSL), {len(dup)} labelled duplicates excluded from SSL")
    return {"ssl_idx": ssl_idx, "holdout_idx": hold, "dup_idx": dup}
