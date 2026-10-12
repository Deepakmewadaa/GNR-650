#!/usr/bin/env python
"""Small, budget-aware SSL ablations. Every run (including failed / underperforming ones) is appended to
docs/experiment_log.md with its config, metrics and timestamp, and summarised in
results/<run>/ablations.json + ablations.md. This produces the report's required "failed experiment"
from REAL runs - nothing is ever fabricated.

Each variant = short train_ssl.py run (same epochs for all variants) with config overrides, evaluated by
the built-in monitor (kNN + quick linear probe on the noisy labelled hold-out, collapse diagnostics).

  python scripts/run_ablations.py --config base --ablation-budget-hours 1.5
  python scripts/run_ablations.py --variants ref weak_aug temp_0.5 --epochs 30
"""
import argparse
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from utils.config import add_common_args, load_config  # noqa: E402
from utils.logutil import append_experiment_log, load_json, save_json  # noqa: E402
from utils.paths import out_paths  # noqa: E402

VARIANTS = {
    "ref": ([], "reference: default augmentation, temperature 0.2, 512-d"),
    "weak_aug": (["aug.crop_scale=[0.8,1.0]", "aug.jitter_p=0.0", "aug.gray_p=0.0", "aug.blur_p=0.0"],
                 "deliberately weak: minimal augmentation (large crops + flip only) - expected to fail"),
    "no_color": (["aug.jitter_p=0.0", "aug.gray_p=0.0"], "crops+blur but no colour distortion"),
    "temp_0.5": (["simclr.temperature=0.5"], "NT-Xent temperature 0.5"),
    "temp_0.1": (["simclr.temperature=0.1"], "NT-Xent temperature 0.1"),
    "embed_256": (["model.embed_dim=256"], "256-d embedding (last stage width)"),
    "stem_medium": (["model.stem=medium"], "3x3/1 + max-pool stem (2x finer feature maps, ~4x compute)"),
    "crop_0.2": (["aug.crop_scale=[0.2,1.0]"], "less aggressive random-resized-crop (min scale 0.2)"),
    "epochs_x2": ([], "reference with twice the epochs"),
    "dino": (["ssl.method=dino", "ssl.optimizer=sgd", "ssl.lr=0.03", "ssl.weight_decay=1.0e-4", "ssl.batch_size=256",
              "dino.global_scale=[0.14,1.0]", "dino.local_scale=[0.05,0.14]"], "DINO instead of SimCLR"),
}
DEFAULT = ["ref", "weak_aug", "temp_0.5", "embed_256", "epochs_x2"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--variants", nargs="*", default=DEFAULT, help=f"subset of {list(VARIANTS)}")
    ap.add_argument("--ablation-budget-hours", type=float, default=1.5, help="total time for all variants")
    ap.add_argument("--epochs", type=int, default=None, help="epochs per variant (default: from budget + doctor)")
    a = ap.parse_args()
    cfg = load_config(a)
    out = out_paths(a.out_root, a.run_name)
    unknown = [v for v in a.variants if v not in VARIANTS]
    if unknown:
        sys.exit(f"unknown variants {unknown}; choose from {list(VARIANTS)}")
    doc = load_json(os.path.join(out["results"], "doctor.json"), {})
    units = sum(2 if v == "epochs_x2" else 1 for v in a.variants)
    if a.epochs:
        ep = a.epochs
    elif doc.get("epoch_time_s"):
        ep = int(a.ablation_budget_hours * 3600 / units / (doc["epoch_time_s"] * 1.1))
        ep = max(5, min(ep, 100))
    else:
        ep = 20
        print("[ablate] no doctor.json -> 20 epochs per variant (run scripts/doctor.py for budget-aware epochs)")
    print(f"[ablate] variants={a.variants} epochs/variant={ep} (x2 for epochs_x2)")
    py = sys.executable
    rows = []
    for v in a.variants:
        sets, desc = VARIANTS[v]
        e = 2 * ep if v == "epochs_x2" else ep
        name = f"abl_{v}_e{e}"
        cmd = [py, os.path.join(REPO, "train_ssl.py"), "--config", a.config, "--run-name", name, "--epochs", str(e),
               "--device", a.device, "--tag", f"ablation:{v}", "--no-exp-log",
               "--set", "ssl.eval_every=5", *sets, *a.set]
        for flag, val in (("--out-root", a.out_root), ("--data-root", a.data_root)):
            if val:
                cmd += [flag, val]
        print(f"\n[ablate] === {v}: {desc}\n+ {' '.join(cmd)}")
        t0 = time.time()
        rc = subprocess.call(cmd)
        dt = time.time() - t0
        summ = load_json(os.path.join(out["out_root"], "runs", name, "ssl", "summary.json"))
        row = {"variant": v, "description": desc, "overrides": sets, "epochs": e, "returncode": rc,
               "time_s": round(dt, 1)}
        if rc != 0 or summ is None:
            row["status"] = "FAILED"
        else:
            row.update({"knn_acc": summ["final"]["knn_acc"], "probe_acc": summ["final"].get("probe_acc"),
                        "eff_rank": summ["final"]["eff_rank"], "feat_std": summ["final"]["feat_std"],
                        "random_knn": summ["random_init"]["knn_acc"], "warnings": summ["warnings"]})
            row["status"] = "WARN" if summ["warnings"] else "OK"
        rows.append(row)
    ref = next((r for r in rows if r["variant"] == "ref" and "knn_acc" in r), None)
    for r in rows:
        if ref and "knn_acc" in r and r is not ref and r["knn_acc"] < ref["knn_acc"] - 0.02:
            r["status"] = "UNDERPERFORMING" if r["status"] == "OK" else r["status"]
        append_experiment_log(out["experiment_log"], f"ablation '{r['variant']}' ({r['description']})",
                              {"base_config": a.config, "overrides": r["overrides"], "epochs": r["epochs"]},
                              {k: r.get(k) for k in ("knn_acc", "probe_acc", "eff_rank", "feat_std", "random_knn",
                                                     "time_s", "returncode")},
                              status=r["status"],
                              notes=("vs ref kNN %.4f" % ref["knn_acc"]) if ref and r is not ref else "")
    save_json({"epochs_per_variant": ep, "rows": rows}, os.path.join(out["results"], "ablations.json"))
    md = ["| variant | description | epochs | kNN acc | probe acc | eff. rank | status |", "|---|---|---|---|---|---|---|"]
    f = lambda x: "-" if x is None else f"{x:.4f}"  # noqa: E731
    for r in rows:
        md.append(f"| {r['variant']} | {r['description']} | {r['epochs']} | {f(r.get('knn_acc'))} | "
                  f"{f(r.get('probe_acc'))} | {f(r.get('eff_rank'))} | {r['status']} |")
    md.append("\nAccuracies: monitor hold-out of the labelled set, NOISY labels; same epochs for all variants.")
    with open(os.path.join(out["results"], "ablations.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(json.dumps({r["variant"]: r["status"] for r in rows}))


if __name__ == "__main__":
    main()
