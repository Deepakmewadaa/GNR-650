"""Device / autocast helpers shared by training scripts and test.py."""
import contextlib

import torch


def get_device(pref="auto"):
    pref = (pref or "auto").lower()
    if pref == "cpu":
        return torch.device("cpu")
    if pref in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    if pref == "cuda":
        print("[warn] CUDA requested but not available -> CPU")
        return torch.device("cpu")
    if pref == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_dtype(device, pref="auto"):
    """bf16 if supported (no loss scaling needed), else fp16 on CUDA; None (fp32) on CPU/MPS."""
    if device.type != "cuda" or pref in (None, "none", "fp32", False):
        return None
    if pref == "fp16":
        return torch.float16
    if pref in ("auto", "bf16") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def autocast(device, dtype):
    if dtype is None or device.type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def make_scaler(device, dtype):
    """GradScaler only needed for fp16."""
    enabled = device.type == "cuda" and dtype == torch.float16
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):   # older torch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def gpu_info():
    if not torch.cuda.is_available():
        return None
    p = torch.cuda.get_device_properties(0)
    return {"name": p.name, "total_mem_gb": round(p.total_memory / 1024 ** 3, 2),
            "capability": f"{p.major}.{p.minor}", "bf16": torch.cuda.is_bf16_supported(),
            "count": torch.cuda.device_count()}


def setup_backends():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
