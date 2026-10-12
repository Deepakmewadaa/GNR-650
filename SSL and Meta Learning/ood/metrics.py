"""OOD + open-set metrics implemented with numpy (scikit-learn is only used in tests as a cross-check).
Known samples are the POSITIVE class; scores are higher for known."""
import numpy as np


def auroc(s_known, s_unknown):
    """Mann-Whitney U statistic with average ranks for ties == ROC AUC."""
    s_known, s_unknown = np.asarray(s_known, float), np.asarray(s_unknown, float)
    n1, n0 = len(s_known), len(s_unknown)
    if n1 == 0 or n0 == 0:
        return float("nan")
    allv = np.concatenate([s_known, s_unknown])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv), float)
    sv = allv[order]
    i = 0
    while i < len(sv):                         # average ranks for ties
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def average_precision(scores, is_pos):
    """AP = sum_k P@k * 1[pos at k] / n_pos  (step-wise, as in sklearn)."""
    scores, is_pos = np.asarray(scores, float), np.asarray(is_pos, bool)
    if is_pos.sum() == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    s, p = scores[order], is_pos[order]
    tp = np.cumsum(p)
    # handle ties: evaluate precision only at the last index of each distinct score
    last = np.r_[s[1:] != s[:-1], True]
    prec = tp / np.arange(1, len(s) + 1)
    rec = tp / p.sum()
    prec, rec = prec[last], rec[last]
    rec_prev = np.r_[0.0, rec[:-1]]
    return float(np.sum((rec - rec_prev) * prec))


def aupr_in(s_known, s_unknown):
    s = np.concatenate([s_known, s_unknown])
    return average_precision(s, np.r_[np.ones(len(s_known)), np.zeros(len(s_unknown))])


def aupr_out(s_known, s_unknown):
    s = -np.concatenate([s_known, s_unknown])
    return average_precision(s, np.r_[np.zeros(len(s_known)), np.ones(len(s_unknown))])


def fpr_at_tpr(s_known, s_unknown, tpr=0.95):
    """Fraction of unknowns accepted when the threshold keeps `tpr` of knowns."""
    s_known, s_unknown = np.asarray(s_known, float), np.asarray(s_unknown, float)
    if len(s_known) == 0 or len(s_unknown) == 0:
        return float("nan")
    thr = np.quantile(s_known, 1 - tpr)
    return float((s_unknown >= thr).mean())


def threshold_from_percentile(known_scores, pct):
    """Threshold that rejects ~pct% of the KNOWN calibration scores. Expressed as a percentile so the
    rule transfers from fold models (fewer classes) to the final model (all classes)."""
    return float(np.quantile(np.asarray(known_scores, float), pct / 100.0))


def open_set_metrics(pred_known, y_known, s_known, s_unknown, thr):
    """pred_known: closed-set argmax for known samples; returns known/unknown/balanced/(K+1) metrics."""
    pred_known, y_known = np.asarray(pred_known), np.asarray(y_known)
    acc_known_mask = np.asarray(s_known) >= thr
    known_acc = float(np.mean((pred_known == y_known) & acc_known_mask)) if len(y_known) else float("nan")
    unk_rec = float(np.mean(np.asarray(s_unknown) < thr)) if len(s_unknown) else float("nan")
    n_k, n_u = len(y_known), len(s_unknown)
    kp1 = (known_acc * n_k + unk_rec * n_u) / max(1, n_k + n_u)
    return {"known_acc": known_acc, "unknown_recall": unk_rec, "balanced_acc": 0.5 * (known_acc + unk_rec),
            "kplus1_acc": float(kp1), "known_reject_rate": float(1 - acc_known_mask.mean()) if n_k else float("nan"),
            "closed_set_acc": float(np.mean(pred_known == y_known)) if n_k else float("nan")}


def ood_summary(s_known, s_unknown):
    return {"auroc": auroc(s_known, s_unknown), "aupr_in": aupr_in(s_known, s_unknown),
            "aupr_out": aupr_out(s_known, s_unknown), "fpr95": fpr_at_tpr(s_known, s_unknown, 0.95)}
