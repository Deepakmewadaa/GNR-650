"""Matplotlib plots for the OOD stage (optional dependency: if matplotlib is missing, plots are skipped)."""
import os

import numpy as np


def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as e:  # noqa: BLE001
        print(f"[plots] matplotlib unavailable ({e}); skipping plots")
        return None


def roc_curve(s_known, s_unknown):
    s = np.concatenate([s_known, s_unknown])
    y = np.r_[np.ones(len(s_known)), np.zeros(len(s_unknown))]
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tpr = np.r_[0, np.cumsum(y) / max(1, y.sum())]
    fpr = np.r_[0, np.cumsum(1 - y) / max(1, (1 - y).sum())]
    return fpr, tpr


def plot_roc(fold, path, title="LCO fold 0"):
    plt = _plt()
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, d in fold["scores"].items():
        fpr, tpr = roc_curve(np.asarray(d["raw"]["known"]), np.asarray(d["raw"]["unknown"]))
        ax.plot(fpr, tpr, label=f"{name} (AUROC {d['summary']['auroc']:.3f})", lw=1.5)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("FPR (unknown accepted as known)")
    ax.set_ylabel("TPR (known accepted)")
    ax.set_title(f"ROC, known-vs-pseudo-unknown ({title})")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_threshold_sensitivity(agg, pcts, selected, path):
    plt = _plt()
    if plt is None:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for name, a in agg.items():
        m, s = np.asarray(a["curve_mean"]["balanced_acc"]), np.asarray(a["curve_std"]["balanced_acc"])
        ax.plot(pcts, m, lw=2.2 if name == selected else 1.2, label=name)
        if name == selected:
            ax.fill_between(pcts, m - s, m + s, alpha=0.2)
    ax.set_xlabel("percentile of known calibration scores used as threshold (% knowns rejected)")
    ax.set_ylabel("balanced open-set accuracy")
    ax.set_title("Threshold sensitivity, all scores (mean over LCO folds)")
    ax.legend(fontsize=8)
    ax = axes[1]
    a = agg[selected]
    for metric in ("known_acc", "unknown_recall", "balanced_acc", "kplus1_acc"):
        m, s = np.asarray(a["curve_mean"][metric]), np.asarray(a["curve_std"][metric])
        ax.plot(pcts, m, label=metric)
        ax.fill_between(pcts, m - s, m + s, alpha=0.15)
    ax.axvline(a["best_percentile"], color="k", ls="--", lw=1, label=f"chosen p={a['best_percentile']:g}")
    ax.set_xlabel("percentile threshold")
    ax.set_title(f"Selected score: {selected}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_score_hist(fold, selected, thr_pct, path):
    plt = _plt()
    if plt is None:
        return None
    d = fold["scores"][selected]["raw"]
    k, u = np.asarray(d["known"]), np.asarray(d["unknown"])
    fig, ax = plt.subplots(figsize=(5.5, 4))
    bins = np.linspace(min(k.min(), u.min()), max(k.max(), u.max()), 40)
    ax.hist(k, bins, alpha=0.6, label="known (val)", density=True)
    ax.hist(u, bins, alpha=0.6, label="pseudo-unknown", density=True)
    ax.axvline(np.quantile(k, thr_pct / 100), color="k", ls="--", label=f"{thr_pct:g}th pct of known")
    ax.set_title(f"{selected} score distribution (LCO fold 0)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_target(target, selected, path):
    """Estimated known accuracy / unknown recall / expected (K+1)-accuracy on the unseen calibration pool."""
    plt = _plt()
    if plt is None:
        return None
    t = target[selected]
    p = [r["percentile"] for r in t["curve"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for key, lab in (("known_acc_est", "known acc (est.)"), ("unknown_recall_est", "unknown recall (est.)"),
                     ("expected_acc", "expected (K+1) acc"), ("pool_reject_rate", "pool reject rate")):
        ax.plot(p, [r[key] for r in t["curve"]], label=lab)
    ax.axvline(t["best_percentile"], color="k", ls="--", lw=1, label=f"chosen p={t['best_percentile']:g}")
    ax.set_xlabel("percentile of known calibration scores (% knowns rejected)")
    ax.set_title(f"{selected}: est. known fraction of unseen pool = {t['pi_known']:.2f}")
    ax.legend(fontsize=8)
    ax = axes[1]
    for name, tt in target.items():
        ax.plot([r["percentile"] for r in tt["curve"]], [r["expected_acc"] for r in tt["curve"]],
                lw=2.2 if name == selected else 1.1, label=f"{name} (pi={tt['pi_known']:.2f})")
    ax.set_title("estimated (K+1)-accuracy on the unseen pool, all scores")
    ax.set_xlabel("percentile threshold")
    ax.legend(fontsize=7)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
