#!/usr/bin/env python
"""Environment sanity check + SSL throughput measurement + time-budget planning.

Checks: Python/torch/CUDA versions, GPU + memory, bf16, driver, disk space, data discovery, image decode
speed (-> cache build time), a REAL 2-step SSL dry run on a few real images (incl. OOM batch back-off), and
measured training throughput -> estimated epoch time and how many epochs fit in the time budget.
Writes results/<run>/doctor.json. Never trains a real model.
"""
import argparse
import os
import platform
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from datasets.discover import find_data_dirs, list_labelled, list_unlabelled  # noqa: E402
from datasets.io import load_image  # noqa: E402
from ssl_methods.engine import find_batch_size, is_oom, make_method, make_sampler, set_compile, train_step  # noqa: E402
from utils.config import add_common_args, load_config  # noqa: E402
from utils.device import amp_dtype as get_amp, get_device, gpu_info, make_scaler, setup_backends  # noqa: E402
from utils.logutil import save_json  # noqa: E402
from utils.optim import build_optimizer  # noqa: E402
from utils.paths import out_paths  # noqa: E402
from utils.timing import fmt_secs  # noqa: E402

OK, BAD = "OK ", "!! "


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--n-images", type=int, default=256, help="real images to load for the dry run")
    ap.add_argument("--timed-steps", type=int, default=10)
    a = ap.parse_args()
    cfg = load_config(a)
    out = out_paths(a.out_root, a.run_name)
    rep, problems = {}, []
    budget_h = float(cfg.get("time_budget_hours", 6))

    print("=" * 80 + "\nENVIRONMENT")
    rep["python"] = sys.version.split()[0]
    rep["platform"] = platform.platform()
    rep["torch"] = torch.__version__
    rep["torch_cuda_build"] = torch.version.cuda
    rep["cudnn"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    rep["gpu"] = gpu_info()
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20)
        rep["nvidia_smi"] = smi.stdout.strip()
    except Exception:  # noqa: BLE001
        rep["nvidia_smi"] = None
    print(f"  python {rep['python']} | torch {rep['torch']} (CUDA build {rep['torch_cuda_build']}, cudnn {rep['cudnn']})")
    print(f"  GPU: {rep['gpu']} | nvidia-smi: {rep['nvidia_smi']}")
    if rep["gpu"] is None:
        problems.append("No CUDA GPU visible to torch: training would run on CPU (far too slow for the real run).")
    rep["cpu_count"] = os.cpu_count()
    try:
        rep["ram_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3, 1)
    except (ValueError, OSError, AttributeError):
        rep["ram_gb"] = None
    os.makedirs(out["out_root"], exist_ok=True)
    du = shutil.disk_usage(out["out_root"])
    rep["disk_free_gb"] = round(du.free / 1024 ** 3, 1)
    print(f"  CPUs {rep['cpu_count']} | RAM {rep['ram_gb']} GB | free disk {rep['disk_free_gb']} GB at {out['out_root']}")

    print("=" * 80 + "\nDATA")
    d = cfg["data"]
    lab, unl = find_data_dirs(d.get("root"), d.get("labelled_dir"), d.get("unlabelled_dir"))
    if not lab or not unl:
        problems.append("Dataset folders not found (need labelled + unlabelled). Put them in ./data or pass --data-root.")
        save_json({**rep, "problems": problems}, os.path.join(out["results"], "doctor.json"))
        print("\n".join(BAD + p for p in problems))
        sys.exit(1)
    lp, ll, classes, _ = list_labelled(lab)
    up = list_unlabelled(unl)
    rep["data"] = {"K": len(classes), "n_labelled": len(lp), "n_unlabelled": len(up)}
    print(f"  {OK}K={len(classes)} labelled={len(lp)} unlabelled={len(up)}")

    # resolution decision identical to load_data()
    from datasets.data import decide_size_channels
    size, ch, scan = decide_size_channels(d, up)
    rep["image_size"], rep["channels"] = size, ch
    cache_gb = (len(up) + len(lp)) * ch * size * size / 1e9
    rep["cache_gb"] = round(cache_gb, 2)
    if du.free / 1e9 < cache_gb * 2 + 2:
        problems.append(f"Low disk space: need ~{cache_gb * 2 + 2:.1f} GB for caches/checkpoints.")
    print(f"  image_size={size} channels={ch} (native shorter side median {scan['native_min_side_median']}); "
          f"uint8 cache ~{cache_gb:.2f} GB")

    rng = np.random.RandomState(0)
    pick = [up[i] for i in rng.choice(len(up), min(a.n_images, len(up)), replace=False)]
    t0 = time.time()
    imgs = [load_image(p, size, ch) for p in pick[:64]]
    dec = (time.time() - t0) / max(1, len(imgs))
    workers = int(d.get("cache_workers", min(16, os.cpu_count() or 4)))
    est_cache = (len(up) + len(lp)) * dec / max(1, min(workers, os.cpu_count() or 1))
    rep["decode_s_per_image"] = round(dec, 4)
    rep["est_cache_build_s"] = round(est_cache)
    imgs += [load_image(p, size, ch) for p in pick[64:]]
    print(f"  decode {dec * 1000:.1f} ms/img -> cache build ~{fmt_secs(est_cache)} with {workers} workers (one-time)")

    print("=" * 80 + "\nSSL DRY RUN (2 real steps) + THROUGHPUT")
    device = get_device(a.device)
    setup_backends()
    amp = get_amp(device, cfg["ssl"].get("amp", "auto"))
    from datasets.data import channel_stats
    arr = np.stack(imgs)
    mean, std = channel_stats(arr)
    meta = {"image_size": size, "channels": ch, "mean": mean, "std": std, "num_classes": len(classes)}
    method, model_cfg = make_method(cfg, meta, device)
    U = torch.from_numpy(arr).to(device)
    sample = make_sampler(U, device)
    bs = int(cfg["ssl"]["batch_size"])
    if device.type == "cuda":
        try:
            bs, mem = find_batch_size(method, sample, bs, int(cfg["ssl"].get("min_batch_size", 64)), amp, device)
        except Exception as e:  # noqa: BLE001
            if method._compiled is None or is_oom(e):
                raise
            print(f"  {BAD}torch.compile failed ({type(e).__name__}: {e}); train_ssl.py will fall back to eager")
            problems.append("torch.compile failed here; training will run in eager mode (slower).")
            set_compile(method, False)
            bs, mem = find_batch_size(method, sample, bs, int(cfg["ssl"].get("min_batch_size", 64)), amp, device)
        rep["peak_mem_gb_at_batch"] = round(mem, 2)
        rep["compiled"] = method._compiled is not None
    else:
        bs = min(bs, 16)
        print(f"  (CPU: dry run with batch {bs} only)")
    rep["batch_size"] = bs
    opt = build_optimizer(method.trainable_named_parameters(), cfg["ssl"].get("optimizer", "lars"),
                          float(cfg["ssl"]["lr"]) * bs / 256, float(cfg["ssl"].get("weight_decay", 1e-6)))
    scaler = make_scaler(device, amp)
    losses = []
    for s in range(2):
        loss, _ = train_step(method, sample(bs), opt, scaler, amp, 3.0, device, s, 100)
        losses.append(None if loss is None else float(loss))
    print(f"  {OK if all(l is not None for l in losses) else BAD}2-step losses: {losses}")
    if any(l is None for l in losses):
        problems.append("Non-finite loss in the dry run.")
    steps = a.timed_steps if device.type == "cuda" else 2
    for s in range(2):
        train_step(method, sample(bs), opt, scaler, amp, 3.0, device, s, 100)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for s in range(steps):
        train_step(method, sample(bs), opt, scaler, amp, 3.0, device, s, 100)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / steps
    ips = bs / dt
    from datasets.data import ssl_partition
    part = ssl_partition(up, lp, d, verbose=True)
    n_unl = len(part["ssl_idx"])
    rep["ssl_images"] = n_unl
    epoch_s = (n_unl // bs) * dt
    ssl_budget = budget_h * 3600 * float(cfg["ssl"].get("budget_fraction", 0.8))
    n_ep = int(ssl_budget // max(epoch_s * 1.05, 1e-6))   # +5% for periodic evaluation
    rep.update({"step_time_s": round(dt, 4), "images_per_s": round(ips, 1), "epoch_time_s": round(epoch_s, 1),
                "ssl_budget_s": ssl_budget, "epochs_in_budget": n_ep, "time_budget_hours": budget_h,
                "ssl_method": cfg["ssl"]["method"], "model_cfg": model_cfg})
    print(f"  {ips:.0f} images/s ({dt * 1000:.0f} ms/step at batch {bs}) -> epoch over {n_unl} images ~{fmt_secs(epoch_s)}")
    print(f"  SSL budget {fmt_secs(ssl_budget)} ({cfg['ssl'].get('budget_fraction', 0.8):.0%} of {budget_h}h) "
          f"-> ~{n_ep} epochs fit (train_ssl.py measures this itself with epochs=auto; "
          f"clipped to [{cfg['ssl'].get('min_epochs')}, {cfg['ssl'].get('max_epochs')}])")
    if n_ep < 50:
        problems.append(f"Only ~{n_ep} SSL epochs fit in the budget: consider --config small, a larger --budget, "
                        f"or a faster GPU. (<50 epochs of SimCLR gives weak features.)")
    rest = (1 - float(cfg["ssl"].get("budget_fraction", 0.8))) * budget_h * 3600
    print(f"  remaining stages (probe, baseline, OOD, export) get ~{fmt_secs(rest)}; the baseline is time-capped "
          f"at {cfg['baseline'].get('budget_fraction', 0.1):.0%} of the budget.")
    rep["problems"] = problems
    save_json(rep, os.path.join(out["results"], "doctor.json"))
    print("=" * 80)
    if problems:
        print("PROBLEMS / WARNINGS:\n" + "\n".join(BAD + p for p in problems))
    else:
        print(OK + "all checks passed")
    print(f"wrote {os.path.join(out['results'], 'doctor.json')}")


if __name__ == "__main__":
    main()
