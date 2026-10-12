#!/usr/bin/env python
"""Dataset inspection (read-only): counts, K, class balance, resolution / channel-mode distribution,
unreadable files, labelled-vs-unlabelled duplicates and a crude content-similarity check, sample grids,
and a recommended preset (small/base, image size, batch size) from the numbers + detected GPU memory.

Writes results/<run>/data_inspection.json and results/<run>/figures/sample_grid_*.png.
"""
import argparse
import hashlib
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from datasets.discover import find_data_dirs, list_labelled, list_unlabelled  # noqa: E402
from datasets.io import load_image, probe_image  # noqa: E402
from utils.logutil import save_json  # noqa: E402
from utils.paths import out_paths  # noqa: E402


def sample(lst, n, seed=0):
    if len(lst) <= n:
        return list(lst)
    idx = np.sort(np.random.RandomState(seed).choice(len(lst), n, replace=False))
    return [lst[i] for i in idx]


def scan(paths, n):
    sizes, modes, bad = [], Counter(), []
    for p in sample(paths, n):
        try:
            w, h, m = probe_image(p)
            sizes.append((w, h))
            modes[m] += 1
        except Exception as e:  # noqa: BLE001
            bad.append((p, str(e)))
    if not sizes:
        return {"n_scanned": 0, "unreadable": bad}
    ms = np.array([min(s) for s in sizes])
    Ms = np.array([max(s) for s in sizes])
    return {"n_scanned": len(sizes), "modes": dict(modes), "unreadable_headers": len(bad),
            "min_side": {"min": int(ms.min()), "p5": float(np.percentile(ms, 5)), "median": float(np.median(ms)),
                         "p95": float(np.percentile(ms, 95)), "max": int(ms.max())},
            "max_side_median": float(np.median(Ms)),
            "top_resolutions": [[f"{w}x{h}", c] for (w, h), c in Counter(sizes).most_common(8)],
            "frac_min_side_below_64": float((ms < 64).mean()), "frac_min_side_below_128": float((ms < 128).mean())}


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def duplicates(lab_paths, unl_paths):
    """Labelled files whose exact bytes also appear in the unlabelled folder (size pre-filter, then md5)."""
    by_size = {}
    for p in unl_paths:
        try:
            by_size.setdefault(os.path.getsize(p), []).append(p)
        except OSError:
            pass
    dup, same_name = 0, 0
    unl_names = {os.path.basename(p) for p in unl_paths}
    for p in lab_paths:
        same_name += os.path.basename(p) in unl_names
        cands = by_size.get(os.path.getsize(p), [])
        if cands:
            h = md5(p)
            if any(md5(c) == h for c in cands[:5]):
                dup += 1
    return {"labelled_files_with_identical_bytes_in_unlabelled": dup, "labelled_names_also_in_unlabelled": same_name,
            "n_labelled": len(lab_paths)}


def thumbs(paths, s=16):
    out = []
    for p in paths:
        try:
            a = load_image(p, s, 3).astype(np.float32).reshape(-1)
            a = a - a.mean()
            out.append(a / (np.linalg.norm(a) + 1e-6))
        except Exception:  # noqa: BLE001
            pass
    return np.stack(out) if out else np.zeros((0, 3 * s * s))


def similarity_check(lab_paths, unl_paths, n_lab=400, n_unl=1500):
    """Crude proxy (16x16 colour thumbnails, cosine): how close unlabelled images are to the labelled set,
    compared with how close labelled images are to each other. Only indicates overlap; SSL kNN is the real test."""
    L = thumbs(sample(lab_paths, n_lab, 1))
    U = thumbs(sample(unl_paths, n_unl, 2))
    if len(L) < 3 or len(U) < 3:
        return {}
    sLL = L @ L.T
    np.fill_diagonal(sLL, -1)
    sLL[sLL > 0.999] = -1     # ignore exact duplicates
    sUL = U @ L.T
    sUL[sUL > 0.999] = -1
    return {"labelled_to_labelled_nn_sim_median": float(np.median(sLL.max(1))),
            "unlabelled_to_labelled_nn_sim_median": float(np.median(sUL.max(1))),
            "note": "similar medians => unlabelled pool has content like the labelled classes (crude 16px proxy)"}


def grid(paths_by_row, out, cell=96):
    rows = len(paths_by_row)
    cols = max(len(r) for r in paths_by_row) if rows else 0
    if not rows or not cols:
        return None
    g = Image.new("RGB", (cols * cell, rows * cell), (255, 255, 255))
    for i, r in enumerate(paths_by_row):
        for j, p in enumerate(r):
            try:
                a = load_image(p, cell, 3).transpose(1, 2, 0)
                g.paste(Image.fromarray(np.ascontiguousarray(a)), (j * cell, i * cell))
            except Exception:  # noqa: BLE001
                pass
    os.makedirs(os.path.dirname(out), exist_ok=True)
    g.save(out)
    return out


def recommend(lab_scan, n_unl, gpu_mem_gb):
    med = lab_scan.get("min_side", {}).get("median", 128) if lab_scan else 128
    if gpu_mem_gb is None:
        preset, bs, why = "small", 64, "no CUDA GPU detected (CPU training is not realistic; use a GPU machine)"
    elif gpu_mem_gb < 12:
        preset, bs, why = "small", 256, f"GPU has {gpu_mem_gb:.0f} GB (< 12 GB)"
    else:
        preset, bs, why = "base", 512 if gpu_mem_gb >= 20 else 256, f"GPU has {gpu_mem_gb:.0f} GB"
    size = int(min(128 if preset == "base" else 96, med))
    size -= size % 8
    cache_gb = n_unl * 3 * size * size / 1e9
    return {"preset": preset, "image_size": size, "ssl_batch_size": bs, "reason": why,
            "unlabelled_cache_gb": round(cache_gb, 2),
            "command": f"bash run_all.sh --config {preset} --budget 6" +
                       ("" if bs == {"base": 512, "small": 256}[preset] else f" --set ssl.batch_size={bs}")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--labelled-dir", default=None)
    ap.add_argument("--unlabelled-dir", default=None)
    ap.add_argument("--run-name", default="main")
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--max-scan", type=int, default=3000, help="images per set to scan headers of")
    ap.add_argument("--no-dup-check", action="store_true", help="skip byte-duplicate check (reads labelled files)")
    a = ap.parse_args()
    out = out_paths(a.out_root, a.run_name)
    lab, unl = find_data_dirs(a.data_root, a.labelled_dir, a.unlabelled_dir)
    if not lab and not unl:
        sys.exit("ERROR: no dataset found. Put it at ./data or ./dataset (labelled_data/, unlabelled_data/) "
                 "or pass --data-root.")
    rep = {"labelled_dir": lab, "unlabelled_dir": unl}
    lab_paths, labels, classes, rule = list_labelled(lab) if lab else ([], [], [], None)
    unl_paths = list_unlabelled(unl) if unl else []
    counts = Counter(labels)
    cc = np.array([counts[i] for i in range(len(classes))]) if classes else np.zeros(0)
    rep.update({"K": len(classes), "class_order_rule": rule, "classes_first_last": classes[:3] + ["..."] + classes[-3:],
                "n_labelled": len(lab_paths), "n_unlabelled": len(unl_paths)})
    if len(cc):
        rep["per_class"] = {"min": int(cc.min()), "median": float(np.median(cc)), "max": int(cc.max()),
                            "imbalance_ratio": float(cc.max() / max(1, cc.min())),
                            "smallest_classes": [[classes[i], int(cc[i])] for i in np.argsort(cc)[:5]]}
        rep["empty_classes"] = [classes[i] for i in np.where(cc == 0)[0]]
    print(f"[inspect] K={rep['K']} labelled={len(lab_paths)} unlabelled={len(unl_paths)} class order: {rule}")
    if len(cc):
        print(f"[inspect] images/class: min {cc.min()} median {np.median(cc):.0f} max {cc.max()}")
    rep["labelled_scan"] = scan(lab_paths, a.max_scan) if lab_paths else {}
    rep["unlabelled_scan"] = scan(unl_paths, a.max_scan) if unl_paths else {}
    for k in ("labelled_scan", "unlabelled_scan"):
        s = rep[k]
        if s.get("n_scanned"):
            print(f"[inspect] {k}: modes {s['modes']} | shorter side median {s['min_side']['median']:.0f} "
                  f"(min {s['min_side']['min']}, max {s['min_side']['max']}) | top {s['top_resolutions'][:3]}")
    dec_bad = []
    for p in sample(lab_paths, 200, 3) + sample(unl_paths, 200, 4):
        try:
            load_image(p, 64, 3)
        except Exception as e:  # noqa: BLE001
            dec_bad.append([p, str(e)])
    rep["full_decode_failures_in_400_sampled"] = dec_bad
    print(f"[inspect] full-decode check on 400 sampled images: {len(dec_bad)} failures")
    if lab_paths and unl_paths and not a.no_dup_check:
        rep["duplicates"] = duplicates(lab_paths, unl_paths)
        d = rep["duplicates"]
        print(f"[inspect] {d['labelled_files_with_identical_bytes_in_unlabelled']}/{d['n_labelled']} labelled images "
              f"also appear byte-identical in the unlabelled folder (SSL sees them without labels - allowed, documented)")
        rep["similarity"] = similarity_check(lab_paths, unl_paths)
        if rep["similarity"]:
            s = rep["similarity"]
            print(f"[inspect] crude 16px NN cosine: unlabelled->labelled {s['unlabelled_to_labelled_nn_sim_median']:.3f} "
                  f"vs labelled->labelled {s['labelled_to_labelled_nn_sim_median']:.3f}")
    gpu = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            rep["gpu"] = {"name": torch.cuda.get_device_name(0), "mem_gb": round(gpu, 1)}
    except Exception:  # noqa: BLE001
        pass
    rep["recommendation"] = recommend(rep["labelled_scan"], len(unl_paths), gpu)
    figs = os.path.join(out["results"], "figures")
    by_cls = {}
    for p, l in zip(lab_paths, labels):
        by_cls.setdefault(l, []).append(p)
    rows = [by_cls[c][:8] for c in sorted(by_cls)[:12]]
    rep["sample_grid_labelled"] = grid(rows, os.path.join(figs, "sample_grid_labelled.png"))
    u = sample(unl_paths, 48, 5)
    rep["sample_grid_unlabelled"] = grid([u[i:i + 8] for i in range(0, len(u), 8)],
                                         os.path.join(figs, "sample_grid_unlabelled.png"))
    save_json(rep, os.path.join(out["results"], "data_inspection.json"))
    r = rep["recommendation"]
    print(f"[inspect] RECOMMENDED: preset={r['preset']} image_size={r['image_size']} ssl batch={r['ssl_batch_size']} "
          f"({r['reason']}); unlabelled cache ~{r['unlabelled_cache_gb']} GB\n[inspect]   {r['command']}")
    print(f"[inspect] wrote {os.path.join(out['results'], 'data_inspection.json')} and sample grids in {figs}")


if __name__ == "__main__":
    main()
