#!/usr/bin/env python
"""Stage 4: unknown-class detection.

1. Leave-class-out evaluation of every OOD score (msp, maxlogit, energy, knn, mahalanobis, cosine,
   knn_ratio, ensemble) over ood.n_folds fixed-seed folds -> selects the score.
2. Fit the FINAL linear head (selected noisy-label method from train_probe.py) and OOD statistics on
   ALL labelled data; known calibration scores from 5-fold cross-fitting.
3. Threshold: maximise the estimated (K+1)-accuracy on the unseen unlabelled calibration holdout
   (threshold_mode=target), or the balanced LCO percentile (threshold_mode=lco). See ood/protocol.py.
Outputs: runs/<name>/ood/ood_final.pt, results/<name>/ood_results.json (+ .md) and plots.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from datasets.data import load_data, ssl_partition  # noqa: E402
from models.encoder_io import default_encoder_path, load_encoder  # noqa: E402
from ood.plots import plot_roc, plot_score_hist, plot_target, plot_threshold_sensitivity  # noqa: E402
from ood.protocol import aggregate, final_fit, leave_class_out, select_score  # noqa: E402
from probe.features import extract_features, load_or_extract_banks  # noqa: E402
from utils.checkpoint import load_ckpt, save_atomic  # noqa: E402
from utils.config import add_common_args, load_config  # noqa: E402
from utils.device import amp_dtype as get_amp, get_device, setup_backends  # noqa: E402
from utils.logutil import Tee, append_experiment_log, load_json, save_json  # noqa: E402
from utils.paths import ensure_dir, out_paths  # noqa: E402
from utils.seed import seed_everything  # noqa: E402


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--encoder", default=None)
    ap.add_argument("--method", default=None, help="override the probe method (default: best from train_probe)")
    ap.add_argument("--score", default=None, help="force an OOD score instead of automatic selection")
    ap.add_argument("--percentile", type=float, default=None, help="force the threshold percentile")
    return ap.parse_args()


def holdout_features(cfg, out, encoder, enc_ck, data, device, amp):
    """Features of the unseen unlabelled calibration holdout (cached per encoder hash)."""
    part = ssl_partition(data.unl_paths, data.lab_paths, cfg["data"])
    idx = part["holdout_idx"]
    if len(idx) == 0:
        print("[ood] calibration holdout is empty -> knn_ratio / target threshold disabled")
        return None
    path = os.path.join(out["ood"], f"holdout_feats_{enc_ck['hash']}_n{len(idx)}.pt")
    paths = [data.unl_paths[i] for i in idx]
    if os.path.isfile(path):
        c = load_ckpt(path)
        if c.get("paths") == paths:
            return c["feats"]
    meta = enc_ck["data_meta"]
    imgs = np.asarray(data.unl)[idx]
    print(f"[ood] extracting features of {len(idx)} unseen calibration-holdout images")
    f = extract_features(encoder, imgs, meta["mean"], meta["std"], device, cfg["probe"].get("feature_batch_size", 256),
                         cfg["probe"].get("tta_flip", True), amp)
    save_atomic({"feats": f, "paths": paths}, path)
    return f


def main():
    args = parse()
    cfg = load_config(args)
    oc, pc = cfg["ood"], cfg["probe"]
    out = out_paths(args.out_root, args.run_name)
    ensure_dir(out["ood"])
    tee = Tee(os.path.join(out["ood"], "fit_ood.log"))
    device = get_device(args.device)
    setup_backends()
    seed_everything(cfg.get("seed", 0))
    amp = get_amp(device, cfg["ssl"].get("amp", "auto"))
    t0 = time.time()

    enc_path = args.encoder or default_encoder_path(out, cfg["export"].get("which_encoder", "best"))
    encoder, ck = load_encoder(enc_path, device)
    meta = ck["data_meta"]
    use_holdout = int(cfg["data"].get("calib_holdout", 0)) > 0
    data = load_data(cfg, out, need_unlabelled=use_holdout, need_labelled=True, force_meta=meta)
    K = len(data.class_names)
    labels = np.asarray(data.labels)
    banks = load_or_extract_banks(cfg, out, encoder, ck, data, device, amp)
    bg = holdout_features(cfg, out, encoder, ck, data, device, amp) if use_holdout else None

    probe_res = load_json(os.path.join(out["results"], "probe_results.json"))
    if probe_res is None:
        print("[ood] WARNING: probe results not found; using method 'ce' and config weight decay")
    method = args.method or (probe_res["best_method"] if probe_res else "ce")
    wd = float(probe_res["weight_decay"]) if probe_res else float(pc.get("weight_decay", 1e-3))
    eta = float(probe_res.get("noise_rate_estimate", 0.0)) if probe_res else 0.0
    print(f"[ood] probe method={method} wd={wd:g} K={K} N={len(labels)}")

    folds, pcts = leave_class_out(banks, labels, K, pc, oc, method, wd, device, bg_feats=bg)
    objective = oc.get("objective", "balanced_acc")
    agg = aggregate(folds, pcts, objective)
    cands = [c for c in oc.get("candidates", list(agg)) if c in agg]
    score, pct = select_score(agg, cands, objective)
    if args.score:
        score = args.score
        pct = agg[score]["best_percentile"]
    if args.percentile is not None:
        pct = float(args.percentile)
    print("\n[ood] leave-class-out summary (mean over folds; known=positive):")
    print(f"  {'score':12s} {'AUROC':>7s} {'AUPR-in':>8s} {'FPR95':>7s} {'best p':>7s} {'bal.acc':>8s} "
          f"{'known':>7s} {'unk.rec':>8s}")
    for n in agg:
        a = agg[n]
        print(f"  {n:12s} {a['summary']['auroc']['mean']:7.4f} {a['summary']['aupr_in']['mean']:8.4f} "
              f"{a['summary']['fpr95']['mean']:7.4f} {a['best_percentile']:7g} {a['at_best']['balanced_acc']['mean']:8.4f} "
              f"{a['at_best']['known_acc']['mean']:7.4f} {a['at_best']['unknown_recall']['mean']:8.4f}"
              f"{'   <== selected' if n == score else ''}")

    pi_score = max(cands, key=lambda n: agg[n]["summary"]["auroc"]["mean"])
    fin = final_fit(banks, labels, K, pc, oc, method, wd, device, score, pct, bg_feats=bg, eta=eta,
                    pi_score=pi_score)
    if args.percentile is not None:     # explicit override wins over both rules
        from ood.metrics import threshold_from_percentile
        fin["threshold"] = threshold_from_percentile(fin["oof_known_scores"][score], args.percentile)
        fin["percentile"], fin["rule"] = float(args.percentile), "manual"
    pct = fin["percentile"]
    head = fin["head"]
    ood_ck = {"head": {"weight": head.weight.detach().clone(), "bias": head.bias.detach().clone()},
              "ood": {"score": score, "threshold": float(fin["threshold"]), "percentile": float(pct),
                      "rule": fin["rule"], "scorer": fin["scorer"].state_dict()},
              "method": method, "weight_decay": wd, "num_classes": K, "class_names": data.class_names,
              "encoder_path": os.path.relpath(enc_path, out["out_root"]), "encoder_hash": ck["hash"],
              "kept_indices": np.where(fin["keep"])[0].tolist()}
    save_atomic(ood_ck, os.path.join(out["ood"], "ood_final.pt"))
    save_json({"folds": folds, "percentiles": pcts.tolist()}, os.path.join(out["ood"], "lco_folds.json"))

    figs = {}
    fd = os.path.join(out["results"], "figures")
    figs["roc"] = plot_roc(folds[0], os.path.join(fd, "ood_roc_fold0.png"))
    figs["threshold"] = plot_threshold_sensitivity(agg, pcts, score, os.path.join(fd, "ood_threshold_sensitivity.png"))
    figs["hist"] = plot_score_hist(folds[0], score, agg[score]["best_percentile"],
                                   os.path.join(fd, "ood_score_hist_fold0.png"))
    if fin["target"]:
        figs["target"] = plot_target(fin["target"], score, os.path.join(fd, "ood_target_threshold.png"))
    figs = {k: os.path.relpath(v, out["results"]) for k, v in figs.items() if v}

    res = {"method": method, "weight_decay": wd, "K": K, "objective": objective, "selected_score": score,
           "selected_percentile": pct, "threshold_rule": fin["rule"], "final_threshold": fin["threshold"],
           "lco_percentile": agg[score]["best_percentile"], "candidates": cands,
           "calibration_holdout_size": 0 if bg is None else int(len(bg)),
           "target": {n: {k: v for k, v in t.items()} for n, t in fin["target"].items()},
           "pool_reject_rate_final": fin["pool_reject_rate_final"],
           "label_noise_estimate_used": eta,
           "lco": {"n_folds": len(folds), "holdout_frac": oc.get("holdout_frac"),
                   "folds": [{k: v for k, v in f.items() if k != "scores"} for f in folds]},
           "aggregate": agg, "percentiles": pcts.tolist(), "final": {"train_acc_noisy": fin["train_acc_noisy"],
                                                                     "kept_frac": float(fin["keep"].mean()),
                                                                     "cleaning": fin["cleaning"]},
           "figures": figs, "time_s": round(time.time() - t0, 1),
           "rule": "score = best LCO balanced accuracy; threshold = p-th percentile of cross-fitted known scores, "
                   "p maximising the estimated (K+1)-accuracy on the unseen unlabelled calibration holdout "
                   "(target) or the LCO balanced accuracy (lco); no test data used"}
    save_json(res, os.path.join(out["ood"], "results.json"))
    save_json(res, os.path.join(out["results"], "ood_results.json"))
    md = ["| score | AUROC | AUPR-in | FPR@95TPR | best pct | balanced acc | known acc | unknown recall |",
          "|---|---|---|---|---|---|---|---|"]
    for n, a in agg.items():
        f = lambda d: f"{d['mean']:.4f} ± {d['std']:.4f}"  # noqa: E731
        md.append(f"| {'**' + n + '**' if n == score else n} | {f(a['summary']['auroc'])} | {f(a['summary']['aupr_in'])} | "
                  f"{f(a['summary']['fpr95'])} | {a['best_percentile']:g} | {f(a['at_best']['balanced_acc'])} | "
                  f"{f(a['at_best']['known_acc'])} | {f(a['at_best']['unknown_recall'])} |")
    with open(os.path.join(out["results"], "ood_results.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    if fin["target"]:
        t = fin["target"][score]
        print(f"[ood] unseen-pool analysis for {score}: est. known fraction {t['pi_known']:.3f}, "
              f"est. (K+1)-acc at p={t['best_percentile']:g}: {t['best_expected_acc']:.4f}")
    print(f"\n[ood] selected score={score} percentile={pct:g} ({fin['rule']}) -> final threshold {fin['threshold']:.5f}")
    append_experiment_log(out["experiment_log"], f"fit_ood run '{args.run_name}'", {"ood": oc, "method": method},
                          {"selected": score, "percentile": pct,
                           "auroc": agg[score]["summary"]["auroc"], "balanced_acc": agg[score]["at_best"]["balanced_acc"]})
    tee.close()


if __name__ == "__main__":
    main()
