"""Repo-relative path helpers. Nothing here may contain machine-specific absolute paths."""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def out_paths(out_root, run_name):
    """All pipeline outputs live under `out_root` so synthetic runs can be redirected outside the repo."""
    out_root = os.path.abspath(out_root or REPO_ROOT)
    p = {
        "out_root": out_root,
        "cache": os.path.join(out_root, "cache"),
        "run": os.path.join(out_root, "runs", run_name),
        "ssl": os.path.join(out_root, "runs", run_name, "ssl"),
        "probe": os.path.join(out_root, "runs", run_name, "probe"),
        "baseline": os.path.join(out_root, "runs", run_name, "baseline"),
        "ood": os.path.join(out_root, "runs", run_name, "ood"),
        "results": os.path.join(out_root, "results", run_name),   # small, committed
        "docs": os.path.join(out_root, "docs"),
        "checkpoints": os.path.join(out_root, "checkpoints"),
        "final_ckpt": os.path.join(out_root, "checkpoints", "final.pt"),
        "experiment_log": os.path.join(out_root, "docs", "experiment_log.md"),
    }
    return p


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path
