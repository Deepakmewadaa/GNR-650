"""SSL training engine pieces shared by train_ssl.py, scripts/doctor.py and scripts/run_ablations.py."""
import time

import numpy as np
import torch
import torch.nn as nn

from models.resnet import build_encoder, resolve_model_cfg
from probe.features import extract_features
from probe.knn import knn_accuracy
from probe.linear import accuracy, train_linear
from ssl_methods.base import build_ssl_method
from utils.diagnostics import collapse_stats
from utils.device import autocast


def make_method(cfg, meta, device):
    model_cfg = resolve_model_cfg(cfg["model"], meta["image_size"], meta["channels"])
    encoder = build_encoder(model_cfg)
    method = build_ssl_method(cfg, encoder, meta).to(device)
    if device.type == "cuda" and cfg["ssl"].get("channels_last", True):
        method = method.to(memory_format=torch.channels_last)
    set_compile(method, cfg["ssl"].get("compile", "auto") in (True, "auto") and device.type == "cuda"
                and method.name == "simclr")      # DINO's multi-resolution crops would trigger recompiles
    return method, model_cfg


def set_compile(method, on, log=print):
    """Compile only the training forward of the encoder (~1.2-1.5x faster -> more SSL epochs in the budget).
    Errors surface lazily at the first call; callers fall back to eager mode via set_compile(method, False)."""
    if on and hasattr(torch, "compile"):
        try:
            object.__setattr__(method, "_compiled", torch.compile(method.encoder))
            log("[ssl] torch.compile enabled for the encoder (first steps include compilation time)")
            return True
        except Exception as e:  # noqa: BLE001
            log(f"[ssl] torch.compile unavailable ({type(e).__name__}: {e}); using eager mode")
    object.__setattr__(method, "_compiled", None)
    return False


def is_oom(e):
    return isinstance(e, getattr(torch.cuda, "OutOfMemoryError", ())) or "out of memory" in str(e).lower()


def train_step(method, x, opt, scaler, amp_dtype, grad_clip, device, step=0, total=1):
    """One optimisation step. Returns (loss tensor, logs) or (None, logs) if the loss was non-finite
    (the step is skipped instead of corrupting the weights)."""
    with autocast(device, amp_dtype):
        if method.name == "dino":
            loss, logs = method.training_step(x, step, total)
        else:
            loss, logs = method.training_step(x)
    if not torch.isfinite(loss):
        opt.zero_grad(set_to_none=True)
        return None, logs
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    gn = nn.utils.clip_grad_norm_([p for _, p in method.trainable_named_parameters()], grad_clip) if grad_clip else None
    method.before_optimizer_step()
    scaler.step(opt)
    scaler.update()
    method.after_step(step, total)
    if gn is not None:
        logs["grad_norm"] = gn.detach()
    return loss.detach(), logs


def try_batch(method, sample_fn, bs, amp_dtype, device):
    """Forward+backward at batch size bs without updating weights. Returns peak memory (GB) or raises."""
    method.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    x = sample_fn(bs)
    with autocast(device, amp_dtype):
        if method.name == "dino":
            loss, _ = method.training_step(x, 0, 1)
        else:
            loss, _ = method.training_step(x)
    loss.backward()
    for p in method.parameters():
        p.grad = None
    if device.type == "cuda":
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 1024 ** 3
    return 0.0


def find_batch_size(method, sample_fn, bs, min_bs, amp_dtype, device, log=print):
    """Halve the batch size on CUDA OOM (LR is scaled linearly with batch size by the caller)."""
    while True:
        try:
            mem = try_batch(method, sample_fn, bs, amp_dtype, device)
            log(f"[ssl] batch size {bs} OK (peak {mem:.2f} GB)")
            return bs, mem
        except RuntimeError as e:
            if not is_oom(e) or bs // 2 < min_bs:
                raise
            for p in method.parameters():
                p.grad = None
            torch.cuda.empty_cache()
            log(f"[ssl] OOM at batch size {bs} -> trying {bs // 2}")
            bs //= 2


@torch.no_grad()
def monitor_eval(encoder, lab_images, labels, split, meta, cfg, device, amp_dtype, quick_probe=True):
    """kNN accuracy + quick linear probe on a stratified held-out slice of the LABELLED data.
    NOTE: these labels are noisy, so absolute numbers underestimate clean accuracy; they are used only
    to compare checkpoints of the same run (noise affects all of them equally)."""
    t0 = time.time()
    was = encoder.training
    encoder.eval()
    f = extract_features(encoder, lab_images, meta["mean"], meta["std"], device,
                         cfg["probe"].get("feature_batch_size", 256), tta_flip=False, amp=amp_dtype)
    encoder.train(was)
    tr, va = split
    y = torch.as_tensor(labels)
    K = int(meta["num_classes"])
    out = {"knn_acc": knn_accuracy(f[tr], y[tr], f[va], y[va], K, cfg["ssl"].get("knn_k", 20),
                                   cfg["ssl"].get("knn_T", 0.07))}
    if quick_probe:
        with torch.enable_grad():
            head = train_linear(f[tr].unsqueeze(0), y[tr], K, {"epochs": 50, "lr": 1e-2, "weight_decay": 1e-3,
                                                             "batch_size": 256}, "ce", 0, device)
        out["probe_acc"] = accuracy(head, f[va], y[va])
    cs = collapse_stats(f)
    out.update({"feat_std": cs["feat_std"], "feat_std_ref": cs["feat_std_ref"], "eff_rank": cs["eff_rank"]})
    out["eval_time_s"] = round(time.time() - t0, 2)
    return out


def make_sampler(U, device):
    """U: uint8 tensor (N,C,S,S) on GPU or CPU. Returns sample_fn(bs) -> uint8 batch on device."""
    N = U.shape[0]

    def sample(bs, idx=None):
        if idx is None:
            idx = torch.randint(0, N, (bs,))
        if U.device != device:
            return U[idx.cpu()].to(device, non_blocking=True)
        return U[idx.to(U.device)]
    return sample


def to_numpy_loaded(arr):
    return np.ascontiguousarray(np.asarray(arr))
