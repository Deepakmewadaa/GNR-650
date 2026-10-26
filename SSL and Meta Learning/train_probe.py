#!/usr/bin/env python
"""Stage 2: linear probing of the FROZEN SSL encoder on the noisy labelled set.

For each seed s (default 0,1,2): stratified train/val split with seed s (the supervised baseline uses
the very same splits), then every noisy-label method (cleaner + loss) is trained on train and scored on
val. Val labels are themselves noisy: under roughly symmetric noise, noisy-val accuracy is a monotone
function of clean accuracy, so it still ranks methods correctly (absolute values are pessimistic).

Also: weight-decay selection (CE, seed 0), a random-init-encoder reference probe, the selected method,
cleaned sample indices over ALL labelled data and the estimated noise rate (confident learning).
Outputs: runs/<name>/probe/*, results/<name>/probe_results.json + probe_results.md
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from datasets.data import load_data  # noqa: E402
from datasets.splits import stratified_split  # noqa: E402
from models.encoder_io import default_encoder_path, load_encoder  # noqa: E402
from models.resnet import build_encoder  # noqa: E402
from probe.features import extract_features, load_or_extract_banks  # noqa: E402
from probe.linear import accuracy, train_linear  # noqa: E402
from probe.methods import estimate_noise_rate, fit_method  # noqa: E402
from utils.config import add_common_args, load_config  # noqa: E402
from utils.device import amp_dtype as get_amp, get_device, setup_backends  # noqa: E402
from utils.logutil import Tee, append_experiment_log, save_json  # noqa: E402
from utils.paths import ensure_dir, out_paths  # noqa: E402
from utils.seed import seed_everything  # noqa: E402


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--encoder", default=None, help="encoder checkpoint (default runs/<name>/ssl/encoder_best.pt)")
    ap.add_argument("--methods", nargs="*", default=None, help="override probe.methods")
    ap.add_argument("--seeds", nargs="*", type=int, default=None, help="override probe.seeds")
    return ap.parse_args()


def ms(vals):
    v = np.asarray(vals, float)
    return {"mean": float(np.mean(v)), "std": float(np.std(v)), "values": [float(x) for x in v]}


def main():
    args = parse()
    cfg = load_config(args)
    pc = cfg["probe"]
    if args.methods:
        pc["methods"] = args.methods
    if args.seeds:
        pc["seeds"] = args.seeds
    out = out_paths(args.out_root, args.run_name)
    ensure_dir(out["probe"])
    tee = Tee(os.path.join(out["probe"], "probe.log"))
    device = get_device(args.device)
    setup_backends()
    seed_everything(cfg.get("seed", 0))
    amp = get_amp(device, cfg["ssl"].get("amp", "auto"))
    t0 = time.time()

    enc_path = args.encoder or default_encoder_path(out, cfg["export"].get("which_encoder", "best"))
    encoder, ck = load_encoder(enc_path, device)
    meta = ck["data_meta"]
    print(f"[probe] encoder {enc_path} (epoch {ck.get('epoch')}, hash {ck['hash']})")
    data = load_data(cfg, out, need_unlabelled=False, need_labelled=True, force_meta=meta)
    if meta["n_labelled"] == 0 or data.labels is None:
        raise SystemExit(
            "ERROR: no labelled images found in dataset/labelled_data/.\n"
            "Upload your labelled data to the server first, then re-run:\n"
            "  bash run_all.sh --config base --budget 95 --only probe,baseline,ood,export,eval,report"
        )
    if data.class_names != meta["class_names"]:
        raise SystemExit("ERROR: labelled class folders differ from those seen during SSL training")
    K = len(data.class_names)
    labels = np.asarray(data.labels)
    banks = load_or_extract_banks(cfg, out, encoder, ck, data, device, amp)
    print(f"[probe] banks {tuple(banks.shape)}  K={K}  N={len(labels)}")

    # ---- weight decay selection (CE, mean over ALL seed splits: one 260-image split is too noisy)
    wd_scores = {}
    for wd in pc.get("wd_grid", [pc.get("weight_decay", 1e-3)]):
        accs = []
        for s in pc["seeds"]:
            tr0, va0 = stratified_split(labels, pc["val_frac"], seed=s)
            h = train_linear(banks[:, torch.as_tensor(tr0)], labels[tr0], K, pc, "ce", s, device, wd=wd)
            accs.append(accuracy(h, banks[0, torch.as_tensor(va0)], labels[va0]))
        wd_scores[str(wd)] = float(np.mean(accs))
        print(f"[probe] wd={wd:g}: mean val acc(noisy) over seeds {wd_scores[str(wd)]:.4f}")
    wd = float(max(wd_scores, key=lambda k: (round(wd_scores[k], 4), float(k))))
    print(f"[probe] selected weight_decay={wd:g}")

    # ---- reference: same probe on a RANDOM-INIT encoder (shows what SSL adds)
    torch.manual_seed(0)
    rnd = build_encoder(ck["model_cfg"]).to(device).eval()
    f_rnd = extract_features(rnd, data.lab, meta["mean"], meta["std"], device, 256, pc.get("tta_flip", True), amp)
    rnd_acc = []
    for s in pc["seeds"]:
        tr, va = stratified_split(labels, pc["val_frac"], seed=s)
        h = train_linear(f_rnd[tr], labels[tr], K, pc, "ce", s, device, wd=wd)
        rnd_acc.append(accuracy(h, f_rnd[va], labels[va]))
    print(f"[probe] random-init encoder + CE probe: {np.mean(rnd_acc):.4f} ± {np.std(rnd_acc):.4f}")

    # ---- all methods x seeds
    per = {m: {"val_acc": [], "train_acc": [], "kept_frac": [], "noise_rate_cl": []} for m in pc["methods"]}
    for s in pc["seeds"]:
        tr, va = stratified_split(labels, pc["val_frac"], seed=s)
        trt, vat = torch.as_tensor(tr), torch.as_tensor(va)
        oof_cache = {}
        for m in pc["methods"]:
            ts = time.time()
            head, keep, info = fit_method(m, banks[:, trt], labels[tr], K, pc, s, device, wd=wd, oof_cache=oof_cache)
            va_acc = accuracy(head, banks[0, vat], labels[va])
            per[m]["val_acc"].append(va_acc)
            per[m]["train_acc"].append(accuracy(head, banks[0, trt], labels[tr]))
            per[m]["kept_frac"].append(float(keep.mean()))
            if "noise_rate_cl" in info:
                per[m]["noise_rate_cl"].append(info["noise_rate_cl"])
            print(f"[probe] seed {s} {m:15s} val acc(noisy) {va_acc:.4f}  kept {keep.mean():.3f}  "
                  f"({time.time() - ts:.1f}s)")
    table = {m: {k: ms(v) for k, v in d.items() if v} for m, d in per.items()}
    order = list(pc["methods"])
    best = max(order, key=lambda m: (round(table[m]["val_acc"]["mean"], 4), -order.index(m)))
    print(f"[probe] BEST method: {best}  val acc(noisy) {table[best]['val_acc']['mean']:.4f} ± "
          f"{table[best]['val_acc']['std']:.4f}")

    # ---- noise-rate estimate + cleaned indices on ALL labelled data with the chosen method
    eta, Q, _ = estimate_noise_rate(banks, labels, K, pc, 0, device, wd)
    _, keep_all, info_all = fit_method(best, banks, labels, K, pc, 0, device, wd=wd)
    cleaned = {"method": best, "kept_indices": np.where(keep_all)[0].tolist(),
               "removed_paths": [data.lab_paths[i] for i in np.where(~keep_all)[0]],
               "kept_frac": float(keep_all.mean()), "noise_rate_estimate_cl": eta}
    save_json(cleaned, os.path.join(out["probe"], "cleaned_indices.json"))
    print(f"[probe] estimated label-noise rate (confident learning, all data): {eta:.3f}; "
          f"{best} keeps {keep_all.mean():.3f}")

    res = {"encoder": os.path.relpath(enc_path, out["out_root"]), "encoder_hash": ck["hash"],
           "encoder_epoch": ck.get("epoch"), "K": K, "n_labelled": int(len(labels)),
           "val_frac": pc["val_frac"], "seeds": pc["seeds"], "weight_decay": wd, "wd_scores": wd_scores,
           "random_init_probe_ce": ms(rnd_acc), "methods": table, "best_method": best,
           "noise_rate_estimate": eta, "best_method_kept_frac_all": float(keep_all.mean()),
           "probe_cfg": {k: v for k, v in pc.items()},
           "note": "val accuracies are measured against NOISY labels",
           "time_s": round(time.time() - t0, 1)}
    save_json(res, os.path.join(out["probe"], "results.json"))
    save_json(res, os.path.join(out["results"], "probe_results.json"))
    md = ["| method | val acc (noisy labels) | train acc | kept frac |", "|---|---|---|---|"]
    md.append(f"| random-init encoder + CE (reference) | {np.mean(rnd_acc):.4f} ± {np.std(rnd_acc):.4f} | | |")
    for m in order:
        t = table[m]
        md.append(f"| {'**' + m + '**' if m == best else m} | {t['val_acc']['mean']:.4f} ± {t['val_acc']['std']:.4f} | "
                  f"{t['train_acc']['mean']:.4f} | {t['kept_frac']['mean']:.3f} |")
    md.append(f"\nSeeds {pc['seeds']}, val_frac {pc['val_frac']}, weight decay {wd:g}. "
              f"Estimated noise rate (confident learning): {eta:.3f}.")
    with open(os.path.join(out["results"], "probe_results.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    append_experiment_log(out["experiment_log"], f"train_probe run '{args.run_name}'",
                          {"probe": pc, "encoder": res["encoder"]},
                          {"best_method": best, "best_val_acc": table[best]["val_acc"],
                           "ce_val_acc": table.get("ce", {}).get("val_acc"), "random_init_ce": ms(rnd_acc),
                           "noise_rate_estimate": eta})
    tee.close()


if __name__ == "__main__":
    main()
