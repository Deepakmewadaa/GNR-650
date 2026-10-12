#!/usr/bin/env python
"""Stage 1: self-supervised pre-training of the encoder on the UNLABELLED images (SimCLR by default).

Resumable: re-running the same command continues from runs/<name>/ssl/last.pt (use --fresh to restart).
Every `ssl.eval_every` epochs: kNN accuracy + quick linear probe on a stratified held-out slice of the
labelled data (noisy labels!) + collapse diagnostics. The random-init encoder is evaluated first as a
reference; a loud warning is printed at the end if SSL did not clearly beat it.

Outputs (runs/<name>/ssl/): last.pt (resume state), encoder_best.pt, encoder_last.pt, log.csv,
train.log, summary.json; results/<name>/ssl_summary.json (small, committed).
"""
import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from datasets.data import load_data, ssl_partition  # noqa: E402
from datasets.splits import stratified_split  # noqa: E402
from ssl_methods.engine import (find_batch_size, is_oom, make_method, make_sampler, monitor_eval,  # noqa: E402
                                set_compile, train_step)
from utils.checkpoint import load_ckpt, save_atomic  # noqa: E402
from utils.config import add_common_args, load_config  # noqa: E402
from utils.device import amp_dtype as get_amp, get_device, gpu_info, make_scaler, setup_backends  # noqa: E402
from utils.diagnostics import collapse_warning  # noqa: E402
from utils.logutil import CSVLogger, Tee, append_experiment_log, save_json  # noqa: E402
from utils.optim import build_optimizer, cosine_lr, set_lr  # noqa: E402
from utils.paths import ensure_dir, out_paths  # noqa: E402
from utils.seed import rng_state, seed_everything, set_rng_state  # noqa: E402
from utils.timing import fmt_secs  # noqa: E402


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--epochs", default=None, help="override ssl.epochs (int or 'auto')")
    ap.add_argument("--fresh", action="store_true", help="ignore an existing last.pt and start over")
    ap.add_argument("--steps-per-epoch", type=int, default=None, help="cap steps/epoch (smoke tests only)")
    ap.add_argument("--tag", default="ssl", help="label used in the experiment log")
    ap.add_argument("--no-exp-log", action="store_true", help="do not append to docs/experiment_log.md")
    return ap.parse_args()


def encoder_ckpt(method, model_cfg, meta, cfg, epoch, monitor):
    enc = method.eval_encoder()
    return {"encoder": {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()},
            "model_cfg": model_cfg, "data_meta": meta_for_ckpt(meta), "epoch": int(epoch),
            "monitor": monitor, "ssl_method": cfg["ssl"]["method"], "config": cfg}


def meta_for_ckpt(meta):
    keep = ("image_size", "channels", "mean", "std", "num_classes", "class_names", "class_order_rule",
            "n_unlabelled", "n_labelled", "partition")
    return {k: meta[k] for k in keep if k in meta}


def main():
    args = parse()
    cfg = load_config(args)
    if args.epochs is not None:
        cfg["ssl"]["epochs"] = args.epochs if args.epochs == "auto" else int(args.epochs)
    out = out_paths(args.out_root, args.run_name)
    ensure_dir(out["ssl"])
    tee = Tee(os.path.join(out["ssl"], "train.log"))
    sc = cfg["ssl"]
    device = get_device(args.device)
    setup_backends()
    seed_everything(cfg.get("seed", 0))
    amp = get_amp(device, sc.get("amp", "auto"))
    print(f"[ssl] device={device} gpu={gpu_info()} amp={amp} torch={torch.__version__}")

    data = load_data(cfg, out, need_unlabelled=True, need_labelled=True)
    meta = data.meta
    if meta["n_unlabelled"] == 0:
        raise SystemExit("ERROR: no unlabelled images found")
    method, model_cfg = make_method(cfg, meta, device)
    n_params = sum(p.numel() for p in method.eval_encoder().parameters())
    print(f"[ssl] method={sc['method']} encoder={model_cfg} params={n_params / 1e6:.2f}M")

    # ---------------- SSL image set: unlabelled minus (labelled duplicates + calibration holdout)
    part = ssl_partition(data.unl_paths, data.lab_paths, cfg["data"])
    meta["partition"] = {k: int(len(v)) for k, v in part.items()}
    # ---------------- data placement: uint8 cache on GPU if it fits (augmentations run on GPU)
    U = torch.from_numpy(np.asarray(data.unl)[part["ssl_idx"]])   # fancy index -> writable RAM copy
    on_gpu = sc.get("data_on_gpu", "auto")
    if device.type == "cuda" and (on_gpu is True or (on_gpu == "auto" and U.numel() <
                                                     0.25 * torch.cuda.get_device_properties(0).total_memory)):
        U = U.to(device)
        print(f"[ssl] unlabelled cache on GPU ({U.numel() / 1e9:.2f} GB)")
    elif device.type == "cuda":
        try:
            U = U.pin_memory()
        except RuntimeError:
            pass
    sample = make_sampler(U, device)
    N = U.shape[0]
    lab_np = np.ascontiguousarray(np.asarray(data.lab))
    split = stratified_split(data.labels, cfg["data"].get("monitor_val_frac", 0.2), seed=0)

    # ---------------- resume or fresh start
    last_path = os.path.join(out["ssl"], "last.pt")
    state = None
    if os.path.isfile(last_path) and not args.fresh:
        state = load_ckpt(last_path)
        print(f"[ssl] RESUMING from {last_path} (epoch {state['epoch']}/{state['total_epochs']})")
        if state.get("model_cfg") != model_cfg:
            raise SystemExit(f"ERROR: config changed since the run started ({state.get('model_cfg')} vs "
                             f"{model_cfg}). Use a new --run-name or --fresh.")
        method.load_state_dict(state["method"])
        bs = int(state["batch_size"])
    else:
        bs = int(min(sc["batch_size"], N))
        if device.type == "cuda":
            try:
                bs, _ = find_batch_size(method, sample, bs, int(sc.get("min_batch_size", 64)), amp, device)
            except Exception as e:  # noqa: BLE001
                if method._compiled is None or is_oom(e):
                    raise
                print(f"[ssl] WARNING: compiled forward failed ({type(e).__name__}: {e}); falling back to eager")
                set_compile(method, False)
                bs, _ = find_batch_size(method, sample, bs, int(sc.get("min_batch_size", 64)), amp, device)
    base_lr = float(sc["lr"]) * bs / 256.0
    opt = build_optimizer(method.trainable_named_parameters(), sc.get("optimizer", "lars"), base_lr,
                          float(sc.get("weight_decay", 1e-6)), float(sc.get("momentum", 0.9)))
    scaler = make_scaler(device, amp)
    steps_per_epoch = max(1, N // bs)
    if args.steps_per_epoch:
        steps_per_epoch = min(steps_per_epoch, args.steps_per_epoch)
    warm_steps = int(sc.get("warmup_epochs", 10)) * steps_per_epoch
    budget_s = float(cfg.get("time_budget_hours", 6.0)) * 3600 * float(sc.get("budget_fraction", 0.8))
    csv = CSVLogger(os.path.join(out["ssl"], "log.csv"))

    if state is not None:
        opt.load_state_dict(state["opt"])
        if state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        start_epoch, total_epochs = int(state["epoch"]), state["total_epochs"]
        if args.epochs not in (None, "auto") and int(args.epochs) != total_epochs:
            print(f"[ssl] WARNING: changing total epochs on resume {total_epochs} -> {args.epochs} "
                  f"(cosine schedule is stretched from here on)")
            total_epochs = int(args.epochs)
        best, baseline = state["best"], state["baseline"]
        elapsed_prev, history = float(state.get("elapsed", 0.0)), state.get("history", [])
        eval_cost = float(state.get("eval_cost", 0.0))
        set_rng_state(state.get("rng", {}))
    else:
        start_epoch, elapsed_prev, history = 0, 0.0, []
        print("[ssl] evaluating RANDOM-INIT encoder (reference baseline)...")
        baseline = monitor_eval(method.eval_encoder(), lab_np, data.labels, split, meta, cfg, device, amp)
        eval_cost = baseline["eval_time_s"]
        print(f"[ssl] random-init: kNN {baseline['knn_acc']:.4f}  probe {baseline.get('probe_acc', float('nan')):.4f} "
              f" eff_rank {baseline['eff_rank']:.1f}")
        best = {"knn_acc": -1.0, "epoch": 0}
        ep_cfg = sc.get("epochs", "auto")
        total_epochs = None if ep_cfg in (None, "auto") else int(ep_cfg)
        if total_epochs is None and device.type != "cuda":
            total_epochs = int(sc.get("default_epochs", 200))
            print(f"[ssl] epochs=auto but no GPU to time -> default_epochs={total_epochs}")

    print(f"[ssl] N={N} batch={bs} steps/epoch={steps_per_epoch} base_lr={base_lr:.4f} "
          f"budget={fmt_secs(budget_s)} epochs={'auto' if total_epochs is None else total_epochs}")
    t_run = time.time()
    epoch = start_epoch
    consecutive_bad = 0
    method.train()
    while total_epochs is None or epoch < total_epochs:
        method.on_epoch_start(epoch, total_epochs or 1)
        t_ep = time.time()
        perm = torch.randperm(N)
        sums, nlog, skipped = {}, 0, 0
        cur_total = total_epochs or max(int(sc.get("default_epochs", 200)), int(sc.get("warmup_epochs", 10)) + 1)
        total_steps = cur_total * steps_per_epoch
        for i in range(steps_per_epoch):
            step = epoch * steps_per_epoch + i
            lr = cosine_lr(step, total_steps, warm_steps, base_lr, base_lr * float(sc.get("final_lr_frac", 0.0)))
            set_lr(opt, lr)
            x = sample(bs, perm[i * bs:(i + 1) * bs])
            try:
                loss, logs = train_step(method, x, opt, scaler, amp, float(sc.get("grad_clip", 0) or 0), device,
                                        step, total_steps)
            except Exception as e:  # noqa: BLE001 - compile problems can surface on the first real step
                if method._compiled is None or is_oom(e) or i > 0 or epoch != start_epoch:
                    raise
                print(f"[ssl] WARNING: compiled step failed ({type(e).__name__}: {e}); retrying in eager mode")
                set_compile(method, False)
                opt.zero_grad(set_to_none=True)
                loss, logs = train_step(method, x, opt, scaler, amp, float(sc.get("grad_clip", 0) or 0), device,
                                        step, total_steps)
            if loss is None:
                skipped += 1
                consecutive_bad += 1
                if consecutive_bad >= 20:
                    raise SystemExit("ERROR: 20 consecutive non-finite losses; lower ssl.lr or set ssl.amp=fp32. "
                                     "The last good state is in last.pt.")
                continue
            consecutive_bad = 0
            sums["loss"] = sums.get("loss", 0.0) + loss
            for k, v in logs.items():
                sums[k] = sums.get(k, 0.0) + v
            nlog += 1
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t_ep
        epoch += 1
        row = {k: round(float(v) / max(1, nlog), 5) for k, v in sums.items()}
        row.update({"epoch": epoch, "lr": round(lr, 6), "epoch_time_s": round(dt, 2),
                    "imgs_per_s": round(steps_per_epoch * bs / dt, 1), "skipped_steps": skipped})
        if device.type == "cuda":
            row["gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)

        decide_at = max(1, min(int(sc.get("timing_epoch", 2)), int(sc.get("warmup_epochs", 10))))
        if total_epochs is None and epoch >= decide_at:   # epoch count from a measured, warmed-up epoch
            per_epoch = dt + eval_cost / max(1, int(sc.get("eval_every", 10)))
            fit = int(budget_s // per_epoch)
            total_epochs = int(np.clip(fit, int(sc.get("min_epochs", 20)), int(sc.get("max_epochs", 800))))
            total_epochs = max(total_epochs, int(sc.get("warmup_epochs", 10)) + 1)
            print(f"[ssl] epoch {epoch} time {dt:.1f}s (+{eval_cost / max(1, int(sc.get('eval_every', 10))):.1f}s eval amortised)"
                  f" -> {fit} epochs fit in {fmt_secs(budget_s)}; using total_epochs={total_epochs}")
            if fit < int(sc.get("min_epochs", 20)):
                print(f"[ssl] WARNING: budget allows only {fit} epochs; running min_epochs={total_epochs}. "
                      f"Consider the 'small' preset.")

        if epoch % int(sc.get("eval_every", 10)) == 0 or epoch == total_epochs:
            mon = monitor_eval(method.eval_encoder(), lab_np, data.labels, split, meta, cfg, device, amp)
            eval_cost = mon["eval_time_s"]
            row.update({k: round(v, 5) if isinstance(v, float) else v for k, v in mon.items()})
            for w in collapse_warning(mon):
                print(f"[ssl] COLLAPSE WARNING: {w}")
            if mon["knn_acc"] >= best["knn_acc"]:
                best = {"knn_acc": mon["knn_acc"], "epoch": epoch, "probe_acc": mon.get("probe_acc")}
                save_atomic(encoder_ckpt(method, model_cfg, meta, cfg, epoch, mon),
                            os.path.join(out["ssl"], "encoder_best.pt"))
        elapsed = elapsed_prev + (time.time() - t_run)
        t_run, elapsed_prev = time.time(), elapsed
        te = total_epochs or cur_total          # total not decided yet during the timing epochs
        remaining = (te - epoch) * elapsed / max(1, epoch)
        row["elapsed_s"] = round(elapsed, 1)
        row["eta_s"] = round(remaining, 1)
        csv.log(row)
        history.append(row)
        msg = (f"[ssl] ep {epoch}/{total_epochs or 'auto'} loss {row.get('loss', float('nan')):.4f} lr {lr:.4f} "
               f"{row['imgs_per_s']:.0f} img/s ({dt:.1f}s) ETA {fmt_secs(remaining)}")
        if "knn_acc" in row:
            msg += f" | kNN {row['knn_acc']:.4f} probe {row.get('probe_acc', float('nan')):.4f} " \
                   f"erank {row['eff_rank']:.1f} std {row['feat_std']:.4f}"
        print(msg, flush=True)

        if epoch % int(sc.get("save_every", 1)) == 0 or epoch == total_epochs:
            save_atomic({"method": method.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                         "epoch": epoch, "total_epochs": total_epochs, "batch_size": bs, "best": best,
                         "baseline": baseline, "elapsed": elapsed, "history": history, "rng": rng_state(),
                         "model_cfg": model_cfg, "eval_cost": eval_cost}, last_path)
        if elapsed > 1.3 * budget_s and epoch < te:
            print(f"[ssl] WARNING: exceeded 1.3x the SSL time budget ({fmt_secs(elapsed)}); stopping early at "
                  f"epoch {epoch}/{total_epochs}. LR schedule did not finish.")
            break

    # ---------------- wrap-up
    final = monitor_eval(method.eval_encoder(), lab_np, data.labels, split, meta, cfg, device, amp)
    save_atomic(encoder_ckpt(method, model_cfg, meta, cfg, epoch, final), os.path.join(out["ssl"], "encoder_last.pt"))
    if not os.path.isfile(os.path.join(out["ssl"], "encoder_best.pt")) or final["knn_acc"] >= best["knn_acc"]:
        best = {"knn_acc": final["knn_acc"], "epoch": epoch, "probe_acc": final.get("probe_acc")}
        save_atomic(encoder_ckpt(method, model_cfg, meta, cfg, epoch, final), os.path.join(out["ssl"], "encoder_best.pt"))
    warnings = []
    margin = max(0.03, 0.25 * baseline["knn_acc"])
    if final["knn_acc"] < baseline["knn_acc"] + margin:
        warnings.append(f"kNN accuracy {final['knn_acc']:.4f} is NOT clearly above the random-init baseline "
                        f"{baseline['knn_acc']:.4f} (needed +{margin:.3f}). SSL may have failed.")
    warnings += [f"collapse: {w}" for w in collapse_warning(final)]
    summary = {"method": sc["method"], "epochs_done": epoch, "epochs_planned": total_epochs, "batch_size": bs,
               "base_lr": base_lr, "random_init": baseline, "final": final, "best": best,
               "model_cfg": model_cfg, "encoder_params": n_params, "data_meta": meta_for_ckpt(meta),
               "train_time_s": round(elapsed_prev, 1), "gpu": gpu_info(), "warnings": warnings,
               "note": "kNN/probe accuracies use NOISY labelled hold-out labels (monitoring only)."}
    save_json(summary, os.path.join(out["ssl"], "summary.json"))
    save_json({k: v for k, v in summary.items()}, os.path.join(out["results"], "ssl_summary.json"))
    for w in warnings:
        print("\n" + "!" * 100 + f"\n!!! WARNING: {w}\n" + "!" * 100)
    print(f"[ssl] done. final kNN {final['knn_acc']:.4f} (random {baseline['knn_acc']:.4f}); best epoch "
          f"{best['epoch']} kNN {best['knn_acc']:.4f}. Encoders in {out['ssl']}")
    if not args.no_exp_log:
        append_experiment_log(out["experiment_log"], f"{args.tag}: train_ssl run '{args.run_name}'",
                              {"ssl": sc, "model": model_cfg, "aug": cfg.get("aug"), "simclr": cfg.get("simclr")},
                              {"final_knn": final["knn_acc"], "final_probe": final.get("probe_acc"),
                               "random_knn": baseline["knn_acc"], "random_probe": baseline.get("probe_acc"),
                               "eff_rank": final["eff_rank"], "epochs": epoch, "time_s": round(elapsed_prev, 1)},
                              status="WARN" if warnings else "OK", notes="; ".join(warnings))
    tee.close()


if __name__ == "__main__":
    main()
