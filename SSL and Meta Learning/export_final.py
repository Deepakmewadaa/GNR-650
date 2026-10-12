#!/usr/bin/env python
"""Stage 5: package encoder + linear head + OOD statistics + threshold + metadata into ONE file,
checkpoints/final.pt (< 100 MB, plain torch.load / weights_only compatible), then self-test it through
the exact test.py code path and print (never run) the git commands to commit it.
"""
import argparse
import csv
import datetime
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from models.final_model import FORMAT_VERSION, FinalModel  # noqa: E402
from models.flops import count_macs, count_params, human  # noqa: E402
from ood.scores import OODScorer  # noqa: E402
from utils.checkpoint import file_size_mb, load_ckpt, save_atomic, state_dict_hash  # noqa: E402
from utils.config import add_common_args, load_config  # noqa: E402
from utils.logutil import save_json  # noqa: E402
from utils.paths import REPO_ROOT, out_paths  # noqa: E402

GIT_HINT = """
Next steps (run these yourself; this script never runs git):
  git add checkpoints/final.pt results/ docs/
  git commit -m "Add trained final checkpoint and results"
  git push
"""


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--output", default=None, help="default: <out-root>/checkpoints/final.pt")
    ap.add_argument("--skip-selftest", action="store_true")
    return ap.parse_args()


def build_final(cfg, out):
    ood_path = os.path.join(out["ood"], "ood_final.pt")
    if not os.path.isfile(ood_path):
        raise SystemExit(f"ERROR: {ood_path} not found - run fit_ood.py first")
    o = load_ckpt(ood_path)
    enc_path = os.path.join(out["out_root"], o["encoder_path"])
    enc = load_ckpt(enc_path)
    if state_dict_hash(enc["encoder"]) != o["encoder_hash"]:
        raise SystemExit("ERROR: encoder changed since fit_ood.py ran (hash mismatch) - rerun fit_ood.py")
    dtype = torch.float16 if cfg["export"].get("encoder_dtype", "float32") == "float16" else torch.float32
    enc_sd = {k: (v.to(dtype) if v.is_floating_point() else v).contiguous() for k, v in enc["encoder"].items()}
    meta = enc["data_meta"]
    scorer = OODScorer.from_state_dict(o["ood"]["scorer"])
    score = o["ood"]["score"]
    final = {
        "format_version": FORMAT_VERSION,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "num_classes": int(o["num_classes"]),
        "class_names": list(o["class_names"]),
        "image_size": int(meta["image_size"]), "channels": int(meta["channels"]),
        "mean": [float(x) for x in meta["mean"]], "std": [float(x) for x in meta["std"]],
        "tta_flip": bool(cfg["probe"].get("tta_flip", True)),
        "model_cfg": enc["model_cfg"],
        "encoder": enc_sd,
        "head": {"weight": o["head"]["weight"].float().contiguous(), "bias": o["head"]["bias"].float().contiguous()},
        "ood": {"score": score, "threshold": float(o["ood"]["threshold"]), "percentile": float(o["ood"]["percentile"]),
                "scorer": scorer.state_dict(only=score)},
        "probe_method": o["method"],
        "provenance": {"ssl_method": enc.get("ssl_method"), "encoder_epoch": int(enc.get("epoch", -1)),
                       "encoder_hash": o["encoder_hash"], "encoder_monitor": enc.get("monitor", {}),
                       "n_unlabelled": meta.get("n_unlabelled"), "n_labelled": meta.get("n_labelled")},
        "config": cfg,
    }
    h = state_dict_hash({**{f"enc.{k}": v for k, v in enc_sd.items()}, "head.w": final["head"]["weight"],
                         "head.b": final["head"]["bias"], "thr": torch.tensor([final["ood"]["threshold"]])})
    final["hash"] = h
    return final


def model_stats(model):
    S, C = model.image_size, model.channels
    enc_p = count_params(model.encoder, trainable_only=False)
    head_p = count_params(model.head, trainable_only=False)
    m = count_macs(torch.nn.Sequential(model.encoder, model.head), (1, C, S, S))
    passes = 2 if model.tta_flip else 1
    return {"params_encoder": enc_p, "params_head": head_p, "params_total": enc_p + head_p,
            "macs_per_forward": m["macs"], "flops_per_forward": m["flops"], "macs_conv": m["macs_conv"],
            "macs_linear": m["macs_linear"], "other_elementwise_ops": m["other_ops"],
            "forward_passes_per_image": passes, "macs_per_image": m["macs"] * passes,
            "flops_per_image": m["flops"] * passes, "input_shape": [1, C, S, S],
            "convention": "MACs from Conv2d+Linear; FLOPs = 2*MACs; flip-TTA doubles per-image cost; "
                          "OOD scoring cost (kNN/Mahalanobis on a 512-d vector) is negligible"}


def benchmark(ck, n_iter=10, bs=128):
    """Model-only throughput (uint8 batch already in memory -> labels), on CUDA if available else CPU.
    End-to-end test.py throughput (incl. PNG decoding) is measured by run_all.sh's eval stage."""
    import time
    from utils.device import amp_dtype
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = FinalModel(ck).to(dev)
    amp = amp_dtype(dev, "auto")
    bs = bs if dev.type == "cuda" else 32
    x = torch.randint(0, 256, (bs, m.channels, m.image_size, m.image_size), dtype=torch.uint8, device=dev)
    for _ in range(3):
        m.predict(x, amp)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iter):
        m.predict(x, amp)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    ips = n_iter * bs / (time.time() - t0)
    name = torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"
    print(f"[export] model-only inference throughput on {name}: {ips:.0f} images/s (batch {bs}, amp={amp})")
    return {"device": name, "images_per_s_model_only": round(ips, 1), "batch": bs, "amp": str(amp)}


def self_test(final_path, model, data_dirs):
    """Run test.py in a subprocess on a few labelled images (or generated ones) and compare with in-process
    predictions; also check the exact output format."""
    from datasets.io import load_image
    tmp = tempfile.mkdtemp(prefix="gnr650_selftest_")
    try:
        imgs = []
        lab = data_dirs.get("labelled_dir")
        if lab and os.path.isdir(lab):
            from datasets.discover import list_labelled
            p, _, _, _ = list_labelled(lab)
            imgs = [p[i] for i in np.linspace(0, len(p) - 1, min(12, len(p))).astype(int)]
        if not imgs:
            from PIL import Image
            rng = np.random.RandomState(0)
            for i in range(6):
                fp = os.path.join(tmp, f"rand_{i}.png")
                Image.fromarray(rng.randint(0, 255, (48 + 8 * i, 40 + 4 * i, 3), dtype=np.uint8)).save(fp)
                imgs.append(fp)
        bad = os.path.join(tmp, "corrupt.png")
        with open(bad, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\nnot really a png")
        rows = [(str(i), p) for i, p in enumerate(imgs + [bad])]
        tcsv = os.path.join(tmp, "test.csv")
        with open(tcsv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "file_path"])
            w.writerows(rows)
        outdir = os.path.join(tmp, "out_not_yet_created")
        cmd = [sys.executable, os.path.join(REPO_ROOT, "test.py"), "--test-csv", tcsv, "--output-path", outdir,
               "--checkpoint", final_path]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp, env={**os.environ, "GNR650_NO_REEXEC": "1"})
        print(r.stdout[-2000:])
        if r.returncode != 0:
            print(r.stderr[-3000:])
            raise SystemExit("SELF-TEST FAILED: test.py returned non-zero")
        with open(os.path.join(outdir, "predictions.csv"), newline="") as f:
            text = f.read()
        lines = text.splitlines()
        assert lines[0] == "id,file_path,label", f"bad header {lines[0]!r}"
        got = list(csv.reader(lines[1:]))
        assert [g[0] for g in got] == [r_[0] for r_ in rows], "ids/order mismatch"
        assert [g[1] for g in got] == [r_[1] for r_ in rows], "file_path mismatch"
        labs = [int(g[2]) for g in got]
        assert all(0 <= l <= model.K for l in labs), "label out of range"
        assert labs[-1] == model.K, "corrupt image must be predicted as Unknown (K)"
        x = torch.from_numpy(np.stack([load_image(p, model.image_size, model.channels) for p in imgs]))
        pred, _, _ = model.predict(x, None)
        mism = int((pred.numpy() != np.asarray(labs[:-1])).sum())
        print(f"[export] self-test: {len(rows)} rows OK, header/ids/order exact, corrupt->K, "
              f"in-process vs test.py mismatches: {mism} (CPU fp32 both; must be 0)")
        if mism:
            raise SystemExit("SELF-TEST FAILED: test.py predictions differ from in-process predictions")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    args = parse()
    cfg = load_config(args)
    out = out_paths(args.out_root, args.run_name)
    final = build_final(cfg, out)
    path = args.output or out["final_ckpt"]
    save_atomic(final, path)
    size = file_size_mb(path)
    print(f"[export] wrote {path} ({size:.2f} MB, hash {final['hash']})")
    limit = float(cfg["export"].get("max_size_mb", 100))
    if size > limit:
        print(f"[export] WARNING: checkpoint is {size:.1f} MB > {limit} MB (GitHub hard limit is 100 MB per file).\n"
              f"         Options: set export.encoder_dtype=float16, or use Git LFS:\n"
              f"           git lfs install && git lfs track 'checkpoints/*.pt' && git add .gitattributes")
    ck = torch.load(path, map_location="cpu", weights_only=True)    # must load with the safe loader
    model = FinalModel(ck).eval()
    stats = model_stats(model)
    stats.update({"checkpoint_mb": round(size, 3), "hash": final["hash"], "score": final["ood"]["score"],
                  "threshold": final["ood"]["threshold"], "percentile": final["ood"]["percentile"],
                  "probe_method": final["probe_method"], "K": final["num_classes"],
                  "image_size": final["image_size"], "channels": final["channels"]})
    print(f"[export] params: encoder {stats['params_encoder']:,} + head {stats['params_head']:,} = "
          f"{stats['params_total']:,} | MACs/forward {human(stats['macs_per_forward'])} | "
          f"FLOPs/image (with flip TTA) {human(stats['flops_per_image'])}")
    stats["benchmark"] = benchmark(ck)
    if not args.skip_selftest:
        d = cfg["data"]
        from datasets.discover import find_data_dirs
        lab, _ = find_data_dirs(d.get("root"), d.get("labelled_dir"), d.get("unlabelled_dir"), verbose=False)
        self_test(path, model, {"labelled_dir": lab})
    save_json(stats, os.path.join(out["results"], "model_stats.json"))
    if os.path.abspath(path) == os.path.join(REPO_ROOT, "checkpoints", "final.pt"):
        print(GIT_HINT)


if __name__ == "__main__":
    main()
