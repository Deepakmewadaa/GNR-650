#!/usr/bin/env python
"""Stage 3 (required baseline): same encoder architecture + single linear head trained END-TO-END from
random init on the noisy labels, using exactly the probe's train/val splits (stratified, seeds from
probe.seeds) and evaluation protocol. Compared against the frozen-SSL linear probe.

Outputs: runs/<name>/baseline/*, results/<name>/baseline_results.json (+ .md comparison table).
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from datasets.data import load_data  # noqa: E402
from datasets.splits import stratified_split  # noqa: E402
from models.encoder_io import default_encoder_path  # noqa: E402
from models.resnet import resolve_model_cfg  # noqa: E402
from probe.methods import parse_method  # noqa: E402
from probe.supervised_baseline import train_supervised  # noqa: E402
from utils.checkpoint import load_ckpt  # noqa: E402
from utils.config import add_common_args, load_config  # noqa: E402
from utils.device import amp_dtype as get_amp, get_device, gpu_info, setup_backends  # noqa: E402
from utils.logutil import Tee, append_experiment_log, load_json, save_json  # noqa: E402
from utils.paths import ensure_dir, out_paths  # noqa: E402


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--epochs", type=int, default=None, help="override baseline.epochs")
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--losses", nargs="*", default=None, help="e.g. ce gce (or 'best')")
    return ap.parse_args()


def main():
    args = parse()
    cfg = load_config(args)
    bc = cfg["baseline"]
    if args.epochs:
        bc["epochs"] = args.epochs
    if args.seeds:
        bc["seeds"] = args.seeds
    if args.losses:
        bc["losses"] = args.losses
    out = out_paths(args.out_root, args.run_name)
    ensure_dir(out["baseline"])
    tee = Tee(os.path.join(out["baseline"], "baseline.log"))
    device = get_device(args.device)
    setup_backends()
    amp = get_amp(device, cfg["ssl"].get("amp", "auto"))

    # Use the SSL run's resolved data meta / architecture when available so both arms match exactly.
    enc_path = default_encoder_path(out, cfg["export"].get("which_encoder", "best"))
    force = None
    if os.path.isfile(enc_path):
        ck = load_ckpt(enc_path)
        force, model_cfg = ck["data_meta"], ck["model_cfg"]
        print(f"[baseline] architecture + preprocessing taken from {enc_path}")
    data = load_data(cfg, out, need_unlabelled=False, need_labelled=True, force_meta=force)
    meta = data.meta
    if force is None:
        model_cfg = resolve_model_cfg(cfg["model"], meta["image_size"], meta["channels"])
    labels = np.asarray(data.labels)
    images = np.array(data.lab)

    probe_res = load_json(os.path.join(out["results"], "probe_results.json"))
    losses = []
    for l in bc["losses"]:
        if l == "best":
            if probe_res is None:
                print("[baseline] probe results not found -> skipping the 'best' loss")
                continue
            l = parse_method(probe_res["best_method"])[1]
        if l not in losses:
            losses.append(l)
    seeds = list(bc["seeds"])
    budget = float(cfg.get("time_budget_hours", 6)) * 3600 * float(bc.get("budget_fraction", 0.1))
    per_run = budget / max(1, len(losses) * len(seeds))
    print(f"[baseline] losses={losses} seeds={seeds} epochs={bc['epochs']} per-run time cap {per_run / 60:.1f} min "
          f"model={model_cfg} device={device}")
    vf = cfg["probe"]["val_frac"]
    results = {}
    t0 = time.time()
    for loss in losses:
        runs = []
        for s in seeds:
            tr, va = stratified_split(labels, vf, seed=s)       # identical split to train_probe.py
            r = train_supervised(images, labels, tr, va, meta, model_cfg, bc, cfg["aug"], loss, s, device, amp,
                                 time_limit_s=per_run)
            r["seed"] = s
            runs.append(r)
            print(f"[baseline] loss={loss} seed={s}: final val acc(noisy) {r['final_val_acc']:.4f} "
                  f"(best during training {r['best_val_acc']:.4f}, {r['epochs_run']} epochs, {r['train_time_s']}s)")
        fv = [r["final_val_acc"] for r in runs]
        bv = [r["best_val_acc"] for r in runs]
        results[loss] = {"final_val_acc": {"mean": float(np.mean(fv)), "std": float(np.std(fv)), "values": fv},
                         "best_val_acc": {"mean": float(np.mean(bv)), "std": float(np.std(bv)), "values": bv},
                         "runs": runs}

    comparison = []
    if probe_res:
        for m in ["ce", probe_res["best_method"]]:
            if m in probe_res["methods"]:
                v = probe_res["methods"][m]["val_acc"]
                comparison.append({"model": f"SSL encoder (frozen) + linear probe [{m}]", "val_acc_mean": v["mean"],
                                   "val_acc_std": v["std"]})
    for loss, r in results.items():
        comparison.append({"model": f"Supervised end-to-end from scratch [{loss}] (final epoch)",
                           "val_acc_mean": r["final_val_acc"]["mean"], "val_acc_std": r["final_val_acc"]["std"]})
    differences = {
        "trainable_parts": "baseline trains encoder+head end-to-end; probe trains only the linear head on frozen SSL features",
        "optimizer": f"baseline {bc.get('optimizer', 'adamw')} (decoupled wd={bc.get('weight_decay')}, lr={bc.get('lr')}) "
                     f"vs probe Adam (L2 wd from grid, lr={cfg['probe'].get('lr')}) - same adaptive-moment family",
        "epochs": f"baseline {bc['epochs']} epochs (time cap {per_run / 60:.1f} min/run) vs probe {cfg['probe']['epochs']} epochs",
        "augmentation": f"baseline online '{bc.get('aug', 'mild')}' GPU augmentation; probe uses {cfg['probe']['n_aug_views']} "
                        f"pre-computed mild-augmentation feature views (same crop/flip family)",
        "unlabelled_data": "baseline sees only the labelled train split; the SSL encoder also saw the 25k unlabelled images",
        "early_stopping": "none for either (final epoch reported); best-epoch baseline numbers are reported separately "
                          "and are optimistic because the epoch is chosen on the val split",
    }
    stopped = any(r["stopped_early_by_time"] for v in results.values() for r in v["runs"])
    if stopped:
        differences["time_cap"] = "WARNING: at least one baseline run hit the time cap before finishing its epochs"
    res = {"losses": losses, "seeds": seeds, "val_frac": vf, "model_cfg": model_cfg, "baseline_cfg": bc,
           "results": results, "comparison": comparison, "differences_from_probe": differences,
           "gpu": gpu_info(), "time_s": round(time.time() - t0, 1),
           "note": "val accuracies use NOISY labels; same splits as train_probe.py"}
    save_json(res, os.path.join(out["baseline"], "results.json"))
    save_json(res, os.path.join(out["results"], "baseline_results.json"))
    md = ["| model | val acc (noisy labels), mean ± std over seeds |", "|---|---|"]
    md += [f"| {c['model']} | {c['val_acc_mean']:.4f} ± {c['val_acc_std']:.4f} |" for c in comparison]
    with open(os.path.join(out["results"], "baseline_results.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    append_experiment_log(out["experiment_log"], f"train_supervised_baseline run '{args.run_name}'",
                          {"baseline": bc, "model": model_cfg},
                          {l: r["final_val_acc"] for l, r in results.items()},
                          status="WARN" if stopped else "OK")
    tee.close()


if __name__ == "__main__":
    main()
