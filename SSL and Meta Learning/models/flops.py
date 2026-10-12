"""Parameter and MAC/FLOP counting with our own forward hooks (no thop/fvcore/ptflops).

Convention: MACs counted for Conv2d and Linear (the dominant cost); FLOPs = 2 x MACs.
Elementwise ops (BN, ReLU, pooling, residual adds) are reported separately as `other_ops`
(1 op per output element for BN/ReLU/add-like layers, kernel-area ops per output for pooling)
so the convention is explicit in the report.
"""
import torch
import torch.nn as nn


def count_params(model, trainable_only=True):
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))


@torch.no_grad()
def count_macs(model: nn.Module, input_shape, device="cpu"):
    """input_shape includes batch dim (use batch 1 for per-image numbers)."""
    macs = {"conv": 0, "linear": 0}
    other = {"bn": 0, "act": 0, "pool": 0}
    hooks = []

    def conv_hook(m, inp, out):
        k = m.kernel_size[0] * m.kernel_size[1]
        macs["conv"] += out.numel() * (m.in_channels // m.groups) * k

    def lin_hook(m, inp, out):
        macs["linear"] += out.numel() * m.in_features

    def bn_hook(m, inp, out):
        other["bn"] += out.numel()

    def act_hook(m, inp, out):
        other["act"] += out.numel()

    def pool_hook(m, inp, out):
        if isinstance(m, nn.AdaptiveAvgPool2d):
            other["pool"] += inp[0].numel()
        else:
            ks = m.kernel_size
            other["pool"] += out.numel() * (ks * ks if isinstance(ks, int) else ks[0] * ks[1])

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(lin_hook))
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            hooks.append(m.register_forward_hook(bn_hook))
        elif isinstance(m, (nn.ReLU, nn.GELU)):
            hooks.append(m.register_forward_hook(act_hook))
        elif isinstance(m, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
            hooks.append(m.register_forward_hook(pool_hook))
    modes = {m: m.training for m in model.modules()}   # restore EVERY submodule's own mode afterwards
    model.eval()
    try:
        model(torch.zeros(*input_shape, device=device))
    finally:
        for h in hooks:
            h.remove()
        for m, t in modes.items():
            m.training = t
    total = macs["conv"] + macs["linear"]
    return {"macs": total, "flops": 2 * total, "macs_conv": macs["conv"], "macs_linear": macs["linear"],
            "other_ops": sum(other.values()), "other_detail": other, "input_shape": list(input_shape)}


def human(n):
    for unit, d in (("G", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= d:
            return f"{n / d:.3f}{unit}"
    return str(n)
