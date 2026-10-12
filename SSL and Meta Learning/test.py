#!/usr/bin/env python
"""Grader entry point.

    python test.py --test-csv /abs/path/test.csv --output-path /abs/output/dir

Reads a CSV with columns id,file_path; writes <output-path>/predictions.csv with header exactly
`id,file_path,label` (same row order and ids; label in 0..K-1 for known classes, K = Unknown).
The checkpoint is located relative to THIS file (checkpoints/final.pt), never the working directory.

Robustness: re-executes itself with the repo's .venv python if setup.bash created one (the grader does not
activate it); CPU fallback; AMP on GPU; DataLoader worker fallback; unreadable/missing images -> warning +
label K; handles greyscale / RGBA / palette / 16-bit PNGs and arbitrary sizes.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _use_repo_env():
    """setup.bash installs into HERE/.venv (or HERE/.pydeps as a fallback). Use it even if not activated."""
    venv = os.path.join(HERE, ".venv")
    py = os.path.join(venv, "bin", "python")
    if os.name == "nt":
        py = os.path.join(venv, "Scripts", "python.exe")
    in_venv = os.path.realpath(sys.prefix) == os.path.realpath(venv)
    if os.path.isfile(py) and not in_venv and not os.environ.get("GNR650_NO_REEXEC"):
        import subprocess
        try:   # only switch if the venv interpreter can actually import torch
            ok = subprocess.run([py, "-c", "import torch, numpy, PIL"], capture_output=True, timeout=300).returncode == 0
        except Exception:  # noqa: BLE001
            ok = False
        os.environ["GNR650_NO_REEXEC"] = "1"   # decided once; DataLoader workers inherit this
        if ok:
            try:
                os.execv(py, [py, os.path.abspath(__file__)] + sys.argv[1:])
            except OSError as e:
                print(f"[test] WARNING: could not re-exec with {py} ({e}); using {sys.executable}", file=sys.stderr)
        else:
            print(f"[test] WARNING: {py} cannot import torch; using current interpreter {sys.executable}",
                  file=sys.stderr)
    deps = os.path.join(HERE, ".pydeps")
    if os.path.isdir(deps) and deps not in sys.path:
        sys.path.insert(0, deps)
    if HERE not in sys.path:
        sys.path.insert(0, HERE)


_use_repo_env()

import argparse  # noqa: E402
import csv  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

DEFAULT_CKPT = os.path.join(HERE, "checkpoints", "final.pt")


def log(msg):
    print(f"[test] {msg}", flush=True)


def read_test_csv(path):
    """Returns (ids, file_paths) as the exact strings from the CSV, in order."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.reader(f) if any(c.strip() for c in r)]
    if not rows:
        raise SystemExit(f"ERROR: test CSV {path} is empty")
    header = [h.strip().lower() for h in rows[0]]
    if "id" in header and "file_path" in header:
        i_id, i_fp = header.index("id"), header.index("file_path")
        body = rows[1:]
    else:
        alt = [h.replace(" ", "_").replace("-", "_") for h in header]
        if "file_path" in alt or "filepath" in alt or "path" in alt:
            i_fp = alt.index("file_path") if "file_path" in alt else (alt.index("filepath") if "filepath" in alt
                                                                      else alt.index("path"))
            i_id = alt.index("id") if "id" in alt else 0
            body = rows[1:]
        else:   # no recognisable header: assume id,file_path columns and keep the first row as data
            log(f"WARNING: CSV header {rows[0]} not recognised; assuming columns id,file_path")
            i_id, i_fp = 0, 1
            body = rows
    ids = [r[i_id].strip() if len(r) > i_id else str(k) for k, r in enumerate(body)]
    fps = [r[i_fp].strip() if len(r) > i_fp else "" for r in body]
    return ids, fps


def resolve(fp, csv_dir):
    if os.path.isabs(fp) or os.path.exists(fp):
        return fp
    alt = os.path.join(csv_dir, fp)
    return alt if os.path.exists(alt) else fp


class TestImages(torch.utils.data.Dataset):
    def __init__(self, paths, size, channels):
        self.paths, self.size, self.channels = paths, size, channels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from datasets.io import load_image
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                a = load_image(self.paths[i], self.size, self.channels)
            return torch.from_numpy(a), True, i, ""
        except Exception as e:  # noqa: BLE001 - never crash on one bad image
            return torch.zeros(self.channels, self.size, self.size, dtype=torch.uint8), False, i, \
                f"{type(e).__name__}: {e}"


def write_predictions(path, ids, fps, labels):
    """Exact format: header id,file_path,label; one row per input row; integer labels; no index."""
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["id", "file_path", "label"])
        for i, p, l in zip(ids, fps, labels):
            w.writerow([i, p, int(l)])
    os.replace(tmp, path)


def run(model, loader, n, device, amp, K):
    labels = np.full(n, K, dtype=np.int64)          # default: Unknown (also used for unreadable images)
    bad = []
    for x, ok, idx, err in loader:
        for j in np.where(~ok.numpy())[0]:
            bad.append((int(idx[j]), err[j]))
        if ok.any():
            xs = x[ok].to(device, non_blocking=True)
            pred, _, _ = model.predict(xs, amp)
            labels[idx[ok].numpy()] = pred.cpu().numpy()
    return labels, bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-csv", required=True, help="absolute path to CSV with columns id,file_path")
    ap.add_argument("--output-path", required=True, help="output DIRECTORY; predictions.csv is written inside")
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT, help="default: <repo>/checkpoints/final.pt")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=-1, help="-1 = auto")
    ap.add_argument("--device", default="auto", help="auto|cuda|cpu")
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    if not os.path.isfile(args.checkpoint):
        print(f"ERROR: model checkpoint not found at {args.checkpoint}\n"
              f"       Run training first (bash run_all.sh, which ends with export_final.py) and commit "
              f"checkpoints/final.pt.", file=sys.stderr)
        sys.exit(2)
    if not os.path.isfile(args.test_csv):
        print(f"ERROR: test CSV not found: {args.test_csv}", file=sys.stderr)
        sys.exit(2)
    os.makedirs(args.output_path, exist_ok=True)
    out_csv = os.path.join(args.output_path, "predictions.csv")

    from models.final_model import load_final
    from utils.device import amp_dtype

    ids, fps = read_test_csv(args.test_csv)
    csv_dir = os.path.dirname(os.path.abspath(args.test_csv))
    paths = [resolve(p, csv_dir) for p in fps]
    log(f"python {sys.version.split()[0]} ({sys.executable}) | torch {torch.__version__} | {len(ids)} rows in "
        f"{args.test_csv}")

    device = torch.device("cuda" if (args.device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu")
    model, ck = load_final(args.checkpoint, "cpu")
    K = model.K
    amp = None if args.no_amp else amp_dtype(device, "auto")
    if device.type == "cuda":
        try:   # catches e.g. "no kernel image is available" on unsupported GPU/CUDA combos
            model.to(device)
            with torch.no_grad():
                model.predict(torch.zeros(2, model.channels, model.image_size, model.image_size,
                                          dtype=torch.uint8, device=device), amp)
            torch.backends.cudnn.benchmark = True
        except Exception as e:  # noqa: BLE001
            log(f"WARNING: CUDA failed ({type(e).__name__}: {e}); falling back to CPU")
            device, amp = torch.device("cpu"), None
            model.to(device)
    log(f"device={device} amp={amp} K={K} (label {K} = Unknown) image_size={model.image_size} "
        f"score={model.score_name} threshold={model.threshold:.5f} ckpt_hash={ck.get('hash')}")

    ds = TestImages(paths, model.image_size, model.channels)
    nw = args.num_workers if args.num_workers >= 0 else min(8, os.cpu_count() or 1)
    labels, bad = None, []
    for workers in ([nw, 0] if nw > 0 else [0]):
        try:
            loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=workers,
                                                 pin_memory=device.type == "cuda",
                                                 timeout=600 if workers > 0 else 0)
            labels, bad = run(model, loader, len(ds), device, amp, K)
            break
        except Exception as e:  # noqa: BLE001
            if workers == 0:
                raise
            log(f"WARNING: DataLoader with {workers} workers failed ({type(e).__name__}: {e}); retrying with 0")
    for i, err in bad[:20]:
        log(f"WARNING: could not read image row {i} ({fps[i]}): {err} -> predicted Unknown ({K})")
    if len(bad) > 20:
        log(f"WARNING: ... {len(bad) - 20} more unreadable images")

    write_predictions(out_csv, ids, fps, labels)
    dt = time.time() - t0
    n_unk = int((labels == K).sum())
    log(f"wrote {out_csv}: {len(labels)} rows | unknown {n_unk} ({n_unk / max(1, len(labels)):.1%}) | "
        f"unreadable {len(bad)}")
    log(f"total time {dt:.1f}s | {len(labels) / max(dt, 1e-9):.1f} images/s")


if __name__ == "__main__":
    main()
