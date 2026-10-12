"""Label-noise sample selection / cleaning on frozen SSL features.

Each cleaner returns a boolean `keep` mask over the given samples. A per-class floor guarantees that
no class is emptied (keeps the highest-agreement samples), which matters with ~20-30 images/class.

- knn       : kNN label consensus — keep if the similarity-weighted vote of the k nearest neighbours
              (cosine, self excluded) agrees with the given label (Bahri et al. 2020 'Deep k-NN for noisy labels').
- proto     : class-prototype consistency — keep if the nearest class prototype (mean of L2-normalised
              features, re-estimated once on kept samples) is the given label.
- cl        : confident learning (Northcutt et al. 2021) on out-of-fold probabilities: per-class
              thresholds t_j = mean p_j over samples labelled j; off-diagonal entries of the confident
              joint are flagged. Also yields the noise-rate estimate.
- smallloss : keep the (1 - noise_rate) fraction of each class with the smallest out-of-fold CE loss
              (small-loss trick, Han et al. 2018 Co-teaching).
- consensus : flagged by at least 2 of {knn, proto, cl}.
"""
import numpy as np
import torch
import torch.nn.functional as F

from probe.knn import knn_predict


def _floor(keep, score, labels, min_keep_frac=0.3, min_keep=2):
    keep = keep.copy()
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        need = int(max(min_keep, np.ceil(min_keep_frac * len(idx))))
        need = min(need, len(idx))
        if keep[idx].sum() < need:
            top = idx[np.argsort(-score[idx])[:need]]
            keep[top] = True
    return keep


def knn_clean(feats, labels, K, k=10, T=0.07):
    labels_t = torch.as_tensor(labels)
    pred, votes = knn_predict(feats, labels_t, feats, K, k=k, T=T, exclude_self=True)
    agree = votes.gather(1, labels_t.view(-1, 1)).squeeze(1).cpu().numpy()
    return (pred.cpu().numpy() == labels), agree


def proto_clean(feats, labels, K, iters=2):
    z = F.normalize(feats.float(), dim=1)
    keep = np.ones(len(labels), bool)
    y = torch.as_tensor(labels)
    for _ in range(iters):
        protos = torch.zeros(K, z.shape[1])
        for c in range(K):
            m = (y == c) & torch.as_tensor(keep)
            if m.sum() == 0:
                m = (y == c)
            if m.sum() > 0:
                protos[c] = F.normalize(z[m].mean(0), dim=0)
        sim = z @ protos.t()
        keep = (sim.argmax(1) == y).numpy()
    score = sim.gather(1, y.view(-1, 1)).squeeze(1).numpy()
    return keep, score


def confident_learning(probs, labels, K):
    """Returns (flagged_mask, noise_rate_estimate, confident_joint (K,K) normalised)."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels)
    t = np.array([probs[labels == j, j].mean() if np.any(labels == j) else 1.0 for j in range(K)])
    above = probs >= t[None, :]
    masked = np.where(above, probs, -1.0)
    has = above.any(1)
    yhat = masked.argmax(1)
    C = np.zeros((K, K))
    for i in np.where(has)[0]:
        C[labels[i], yhat[i]] += 1
    flagged = has & (yhat != labels)
    # calibrate rows to the observed class counts (CL eq. 2), then joint Q = C / N
    counts = np.bincount(labels, minlength=K).astype(float)
    row = C.sum(1, keepdims=True)
    Cc = np.divide(C * counts[:, None], row, out=np.zeros_like(C), where=row > 0)
    Q = Cc / max(1.0, Cc.sum())
    noise_rate = float(1.0 - np.trace(Q))
    return flagged, noise_rate, Q


def small_loss_select(probs, labels, keep_ratio):
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    loss = -np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1))
    keep = np.zeros(len(labels), bool)
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        n = max(1, int(round(keep_ratio * len(idx))))
        keep[idx[np.argsort(loss[idx])[:n]]] = True
    return keep, -loss


def run_cleaner(name, feats, labels, K, oof_probs, ccfg):
    """feats: clean (view-0) features of the training samples; oof_probs from out_of_fold_probs.
    Returns (keep_mask, info)."""
    labels = np.asarray(labels)
    k = int(ccfg.get("knn_k", 10))
    floor = float(ccfg.get("min_keep_frac", 0.3))
    _, eta, _ = confident_learning(oof_probs, labels, K)
    info = {"noise_rate_cl": eta}
    if name == "knn":
        keep, score = knn_clean(feats, labels, K, k)
    elif name == "proto":
        keep, score = proto_clean(feats, labels, K)
    elif name == "cl":
        flagged, _, _ = confident_learning(oof_probs, labels, K)
        keep = ~flagged
        score = np.asarray(oof_probs)[np.arange(len(labels)), labels]
    elif name == "smallloss":
        ratio = float(np.clip(1.0 - eta, ccfg.get("smallloss_min_keep", 0.5), 0.95))
        keep, score = small_loss_select(oof_probs, labels, ratio)
        info["keep_ratio"] = ratio
    elif name == "consensus":
        k1, s1 = knn_clean(feats, labels, K, k)
        k2, _ = proto_clean(feats, labels, K)
        f3, _, _ = confident_learning(oof_probs, labels, K)
        votes = (~k1).astype(int) + (~k2).astype(int) + f3.astype(int)
        keep = votes < 2
        score = s1 - votes
    else:
        raise ValueError(f"unknown cleaner '{name}'")
    keep = _floor(keep, np.asarray(score, dtype=float), labels, floor)
    info["kept_frac"] = float(keep.mean())
    info["removed"] = int((~keep).sum())
    return keep, info
