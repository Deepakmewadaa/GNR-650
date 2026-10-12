"""Feature extraction with a frozen encoder (eval mode -> BatchNorm uses running statistics).

The SAME function computes features for probe training, OOD fitting and (via models/final_model.py)
test-time inference, so preprocessing and optional flip-TTA are identical everywhere.
"""
import numpy as np
import torch

from ssl_methods.augment import GPUAugment, normalize_uint8
from utils.device import autocast


@torch.no_grad()
def encode_uint8(encoder, x_uint8, mean, std, tta_flip=True, amp=None, device=None):
    """x_uint8: (B,C,S,S) uint8 tensor on `device`. Returns float32 (B, d)."""
    x = normalize_uint8(x_uint8, mean, std)
    with autocast(x.device, amp):
        f = encoder(x).float()
        if tta_flip:
            f = 0.5 * (f + encoder(torch.flip(x, dims=[3])).float())
    return f


@torch.no_grad()
def extract_features(encoder, images, mean, std, device, batch_size=256, tta_flip=True, amp=None,
                     aug_params=None, seed=0):
    """images: numpy uint8 (N,C,S,S) (memmap ok) or uint8 tensor. Returns float32 CPU tensor (N, d).
    If aug_params is given, a random augmented view (no TTA) is encoded instead (feature banks)."""
    encoder.eval()
    feats = []
    aug = GPUAugment(aug_params, mean, std, images.shape[-1]) if aug_params else None
    if aug is not None:
        torch.manual_seed(seed)   # reproducible augmented feature banks
    for i in range(0, len(images), batch_size):
        xb = images[i:i + batch_size]
        xb = torch.from_numpy(np.array(xb)) if isinstance(xb, np.ndarray) else xb
        xb = xb.to(device, non_blocking=True)
        if aug is None:
            f = encode_uint8(encoder, xb, mean, std, tta_flip, amp)
        else:
            with autocast(xb.device, amp):
                f = encoder(aug(xb)).float()
        feats.append(f.cpu())
    return torch.cat(feats, 0) if feats else torch.zeros(0, encoder.out_dim)


def extract_banks(encoder, images, meta, device, n_aug_views, aug_params, batch_size=256, tta_flip=True,
                  amp=None, seed=0):
    """Returns float32 tensor (1 + n_aug_views, N, d): view 0 is the clean (TTA) feature, the rest are
    mildly augmented views used as cheap data augmentation when training the linear head."""
    views = [extract_features(encoder, images, meta["mean"], meta["std"], device, batch_size, tta_flip, amp)]
    for v in range(n_aug_views):
        views.append(extract_features(encoder, images, meta["mean"], meta["std"], device, batch_size,
                                      False, amp, aug_params=aug_params, seed=seed * 100 + v + 1))
    return torch.stack(views, 0)


def load_or_extract_banks(cfg, out, encoder, enc_ck, data, device, amp=None):
    """Feature banks for all labelled images, cached per (encoder hash, #views, TTA) in runs/<name>/probe/
    so train_probe.py and fit_ood.py use bit-identical features."""
    import os
    from utils.checkpoint import load_ckpt, save_atomic
    from ssl_methods.augment import MILD_AUG
    pc = cfg["probe"]
    nv, tta = int(pc.get("n_aug_views", 8)), bool(pc.get("tta_flip", True))
    path = os.path.join(out["probe"], f"banks_{enc_ck['hash']}_v{nv}_tta{int(tta)}_n{len(data.labels)}.pt")
    if os.path.isfile(path):
        b = load_ckpt(path)
        if b.get("paths") == list(data.lab_paths):
            print(f"[probe] using cached feature banks {os.path.basename(path)}")
            return b["banks"]
    meta = enc_ck["data_meta"]
    print(f"[probe] extracting features: {len(data.labels)} images x (1 clean{' +flipTTA' if tta else ''} + {nv} aug views)")
    banks = extract_banks(encoder, data.lab, meta, device, nv, MILD_AUG, pc.get("feature_batch_size", 256),
                          tta, amp, seed=0)
    save_atomic({"banks": banks, "paths": list(data.lab_paths), "labels": torch.as_tensor(data.labels)}, path)
    return banks
