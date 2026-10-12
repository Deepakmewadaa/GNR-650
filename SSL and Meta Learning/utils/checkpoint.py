"""Checkpoint I/O: atomic saves, safe loads, state-dict hashing."""
import hashlib
import os

import torch


def save_atomic(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)   # never leaves a half-written checkpoint behind on crash/preemption


def load_ckpt(path, map_location="cpu"):
    """Prefer weights_only=True (safe, default in torch>=2.6); fall back for our own older/resume files."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:          # very old torch without weights_only
        return torch.load(path, map_location=map_location)
    except Exception:
        return torch.load(path, map_location=map_location, weights_only=False)


def state_dict_hash(sd):
    h = hashlib.sha256()
    for k in sorted(sd):
        v = sd[k]
        h.update(k.encode())
        if isinstance(v, torch.Tensor):
            h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def file_size_mb(path):
    return os.path.getsize(path) / 1024 ** 2
