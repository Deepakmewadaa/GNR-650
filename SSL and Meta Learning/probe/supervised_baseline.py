"""REQUIRED baseline: the same ResNet encoder + single linear head trained END-TO-END from random init
on the same noisy labels, same split, same optimiser family (AdamW-type adaptive optimiser with
warmup+cosine), same augmentation family as the probe's feature banks (mild crop+flip by default;
'ssl' uses the full SSL augmentation). Evaluation uses the same deterministic preprocessing + flip TTA.
"""
import time

import numpy as np
import torch
import torch.nn as nn

from models.heads import LinearHead
from models.resnet import build_encoder
from probe.features import encode_uint8
from probe.losses import make_loss
from ssl_methods.augment import MILD_AUG, GPUAugment
from utils.device import autocast, make_scaler
from utils.optim import build_optimizer, cosine_lr, set_lr


class EndToEnd(nn.Module):
    def __init__(self, model_cfg, num_classes):
        super().__init__()
        self.encoder = build_encoder(model_cfg)
        self.head = LinearHead(self.encoder.out_dim, num_classes)

    def forward(self, x):
        return self.head(self.encoder(x))


@torch.no_grad()
def evaluate(model, images, labels, meta, device, amp, tta_flip=True, bs=256):
    model.eval()
    correct, n = 0, 0
    for i in range(0, len(images), bs):
        xb = torch.from_numpy(np.array(images[i:i + bs])).to(device)
        f = encode_uint8(model.encoder, xb, meta["mean"], meta["std"], tta_flip, amp)
        pred = model.head(f).argmax(1).cpu()
        correct += (pred == torch.as_tensor(labels[i:i + bs])).sum().item()
        n += len(pred)
    model.train()
    return correct / max(1, n)


def train_supervised(images, labels, train_idx, val_idx, meta, model_cfg, bcfg, ssl_aug, loss_name, seed,
                     device, amp_dtype, time_limit_s=None, log=print):
    torch.manual_seed(seed)
    np.random.seed(seed)
    K = int(meta["num_classes"])
    model = EndToEnd(model_cfg, K).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    aug_mode = bcfg.get("aug", "mild")
    aug_params = dict(ssl_aug) if aug_mode == "ssl" else dict(MILD_AUG)
    aug = GPUAugment(aug_params, meta["mean"], meta["std"], meta["image_size"])
    X = torch.from_numpy(np.ascontiguousarray(images[np.sort(train_idx)])).to(device)
    y = torch.as_tensor(np.asarray(labels)[np.sort(train_idx)], dtype=torch.long, device=device)
    Xv, yv = images[np.sort(val_idx)], np.asarray(labels)[np.sort(val_idx)]
    N = len(y)
    bs = int(min(bcfg.get("batch_size", 128), N))
    epochs = int(bcfg.get("epochs", 200))
    steps_per_epoch = max(1, N // bs)
    total = epochs * steps_per_epoch
    warm = int(bcfg.get("warmup_epochs", 10)) * steps_per_epoch
    lr = float(bcfg.get("lr", 1e-3)) * bs / 256 if bcfg.get("scale_lr", False) else float(bcfg.get("lr", 1e-3))
    opt = build_optimizer(model.named_parameters(), bcfg.get("optimizer", "adamw"), lr,
                          float(bcfg.get("weight_decay", 0.05)))
    scaler = make_scaler(device, amp_dtype)
    loss_fn = make_loss(loss_name, K, bcfg.get("loss_params", {}))
    eval_every = int(bcfg.get("eval_every", max(1, epochs // 10)))
    hist, step, t0 = [], 0, time.time()
    g = torch.Generator(device="cpu").manual_seed(seed)
    stopped_early = False
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(N, generator=g).to(device)
        tl, tc, tn = 0.0, 0, 0
        for i in range(steps_per_epoch):
            idx = perm[i * bs:(i + 1) * bs]
            set_lr(opt, cosine_lr(step, total, warm, lr))
            xb = aug(X[idx])
            if device.type == "cuda":
                xb = xb.contiguous(memory_format=torch.channels_last)
            with autocast(device, amp_dtype):
                logits = model(xb)
            loss = loss_fn(logits.float(), y[idx]).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), float(bcfg.get("grad_clip", 5.0)))
            scaler.step(opt)
            scaler.update()
            step += 1
            tl += loss.item() * len(idx)
            tc += (logits.argmax(1) == y[idx]).sum().item()
            tn += len(idx)
        if (ep + 1) % eval_every == 0 or ep == epochs - 1:
            va = evaluate(model, Xv, yv, meta, device, amp_dtype)
            hist.append({"epoch": ep + 1, "train_loss": tl / tn, "train_acc_noisy": tc / tn, "val_acc_noisy": va,
                         "time_s": round(time.time() - t0, 1)})
            log(f"  [baseline seed={seed} loss={loss_name}] ep {ep + 1}/{epochs} loss {tl / tn:.3f} "
                f"train_acc {tc / tn:.3f} val_acc(noisy) {va:.3f}")
        if time_limit_s and time.time() - t0 > time_limit_s:
            log(f"  [baseline] time limit reached at epoch {ep + 1}/{epochs}; evaluating and stopping")
            stopped_early = True
            if not hist or hist[-1]["epoch"] != ep + 1:
                va = evaluate(model, Xv, yv, meta, device, amp_dtype)
                hist.append({"epoch": ep + 1, "train_loss": tl / tn, "train_acc_noisy": tc / tn,
                             "val_acc_noisy": va, "time_s": round(time.time() - t0, 1)})
            break
    final = hist[-1]["val_acc_noisy"]
    best = max(h["val_acc_noisy"] for h in hist)
    return {"final_val_acc": final, "best_val_acc": best, "history": hist, "epochs_run": hist[-1]["epoch"],
            "epochs_planned": epochs, "stopped_early_by_time": stopped_early, "aug": aug_mode,
            "optimizer": bcfg.get("optimizer", "adamw"), "lr": lr, "batch_size": bs,
            "train_time_s": round(time.time() - t0, 1)}
