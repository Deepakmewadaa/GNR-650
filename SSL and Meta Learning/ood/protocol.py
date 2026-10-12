"""Choosing the OOD score and the rejection threshold WITHOUT any test data, plus the final fit.

(1) Leave-class-out (LCO) protocol -> SELECTS THE SCORE. Fold f (fixed seed):
    hold out ~holdout_frac of the labelled classes as pseudo-unknown; split the remaining (known) samples
    stratified into train/val; fit the selected noisy-label probe method + all OOD statistics on known-train;
    score known-val and every pseudo-unknown sample. Objective: balanced open-set accuracy
    0.5*(known accuracy incl. rejections) + 0.5*(unknown recall) at the best percentile threshold.

(2) THRESHOLD (threshold_mode='target', default when the calibration holdout exists):
    The calibration holdout H is a random part of the unlabelled pool that the encoder never saw, so it is
    distributed like the (unseen) test set: a mixture of known-class images (fraction pi) and unknown-class
    images. With known calibration scores s_K (5-fold cross-fitted on all labelled data) and holdout scores
    s_H from the SAME fold models:
      pi  ~= P_H(s >= t) / P_K(s >= t) for t in the upper tail of s_K, where unknowns are rare
             (mixture-proportion estimation, Scott 2015 / Ramaswamy et al. 2016; median over tail levels);
      UR(t) = (P_H(s < t) - pi * P_K(s < t)) / (1 - pi)                (estimated unknown recall)
      KA(t) = P_K(correct & s >= t) / (1 - eta)                         (known acc; noisy-label corrected)
      t*    = argmax  pi * KA(t) + (1 - pi) * UR(t)                     (expected (K+1)-way accuracy)
    The threshold is stored as a percentile p of the known calibration scores (transfers to the final model).
    threshold_mode='lco' instead uses the balanced LCO percentile (no mixture assumption).
"""
import numpy as np
import torch

from datasets.splits import holdout_classes, kfold_stratified, stratified_split
from ood.metrics import ood_summary, open_set_metrics, threshold_from_percentile
from ood.scores import OODScorer
from probe.methods import fit_method


def _scores(scorer, head, feats, ens_stats_from=None):
    """All available scores for `feats`. If ens_stats_from is given (dict of base-score arrays from KNOWN
    calibration data) the ensemble is calibrated on it; otherwise on these scores themselves."""
    base = scorer.available(include_ensemble=False)
    logits = head(feats.float())
    s = scorer.score_all(feats, logits, base)
    if scorer.active_members():
        scorer.set_calibration(ens_stats_from if ens_stats_from is not None else s)
        s["ensemble"] = scorer.score("ensemble", feats, logits).cpu()
    return logits, s


def leave_class_out(banks, labels, K, pcfg, ocfg, method, wd, device, bg_feats=None, log=print):
    labels = np.asarray(labels)
    pcts = np.asarray(ocfg.get("percentiles", list(range(0, 51))), float)
    folds = []
    for f in range(int(ocfg.get("n_folds", 5))):
        held = holdout_classes(K, float(ocfg.get("holdout_frac", 0.2)), f)
        known_cls = np.setdiff1d(np.arange(K), held)
        remap = -np.ones(K, int)
        remap[known_cls] = np.arange(len(known_cls))
        kn = np.where(np.isin(labels, known_cls))[0]
        un = np.where(np.isin(labels, held))[0]
        yk = remap[labels[kn]]
        tr, va = stratified_split(yk, float(ocfg.get("val_frac", 0.2)), seed=f)
        Kk = len(known_cls)
        tr_t = torch.as_tensor(kn[tr])
        head, keep, info = fit_method(method, banks[:, tr_t], yk[tr], Kk, pcfg, f, device, wd=wd)
        ftr = banks[0, tr_t]
        scorer = OODScorer(ocfg).fit(ftr[torch.as_tensor(np.where(keep)[0])], yk[tr][keep], Kk, bank_feats=ftr,
                                     bg_feats=bg_feats)
        fk, fu = banks[0, torch.as_tensor(kn[va])], banks[0, torch.as_tensor(un)]
        lk, sk = _scores(scorer, head, fk)
        base = {n: v for n, v in sk.items() if n != "ensemble"}
        _, su = _scores(scorer, head, fu, ens_stats_from=base)     # ensemble normalised with KNOWN stats
        pred = lk.argmax(1).cpu().numpy()
        res = {"fold": f, "held_out_classes": held.tolist(), "n_known_train": int(len(tr)),
               "n_known_val": int(len(va)), "n_unknown": int(len(un)), "cleaning": info,
               "closed_set_acc_noisy": float(np.mean(pred == yk[va])), "scores": {}}
        for name in sk:
            s_k, s_u = sk[name].numpy(), su[name].numpy()
            curve = [open_set_metrics(pred, yk[va], s_k, s_u, threshold_from_percentile(s_k, p)) for p in pcts]
            res["scores"][name] = {"summary": ood_summary(s_k, s_u), "curve": curve,
                                   "raw": {"known": s_k.tolist(), "unknown": s_u.tolist()}}
        log(f"  [LCO fold {f}] held-out {len(held)} classes | closed-set acc(noisy) "
            f"{res['closed_set_acc_noisy']:.3f} | AUROC " +
            " ".join(f"{n}={res['scores'][n]['summary']['auroc']:.3f}" for n in sk))
        folds.append(res)
    return folds, pcts


def aggregate(folds, pcts, objective="balanced_acc"):
    agg = {}
    for name in folds[0]["scores"]:
        curves = {m: np.array([[c[m] for c in fr["scores"][name]["curve"]] for fr in folds])
                  for m in ("known_acc", "unknown_recall", "balanced_acc", "kplus1_acc", "known_reject_rate")}
        mean_obj = curves[objective].mean(0)
        bi = int(np.nanargmax(mean_obj))
        summ = {k: [fr["scores"][name]["summary"][k] for fr in folds] for k in ("auroc", "aupr_in", "aupr_out", "fpr95")}
        agg[name] = {
            "best_percentile": float(pcts[bi]),
            "at_best": {m: {"mean": float(np.nanmean(v[:, bi])), "std": float(np.nanstd(v[:, bi]))}
                        for m, v in curves.items()},
            "summary": {k: {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v))} for k, v in summ.items()},
            "curve_mean": {m: v.mean(0).tolist() for m, v in curves.items()},
            "curve_std": {m: v.std(0).tolist() for m, v in curves.items()},
        }
    return agg


def select_score(agg, candidates, objective="balanced_acc"):
    """Highest mean objective at its own best percentile; ties broken by mean AUROC."""
    best = max(candidates, key=lambda n: (round(agg[n]["at_best"][objective]["mean"], 4),
                                          agg[n]["summary"]["auroc"]["mean"]))
    return best, agg[best]["best_percentile"]


# ---------------------------------------------------------------- mixture-based threshold on the holdout
def estimate_known_fraction(s_known, s_pool, qs=(0.7, 0.75, 0.8, 0.85, 0.9)):
    """pi = fraction of known-class images in the pool. For thresholds in the upper tail of the known score
    distribution the pool's tail mass is (almost) only knowns, so tail_pool / tail_known -> pi."""
    s_known, s_pool = np.asarray(s_known, float), np.asarray(s_pool, float)
    ratios = []
    for q in qs:
        t = np.quantile(s_known, q)
        tk = np.mean(s_known >= t)
        tp = np.mean(s_pool >= t)
        ratios.append(float(tp / max(tk, 1e-9)))
    pi = float(np.clip(np.median(ratios), 0.02, 1.0))
    return pi, [[float(q), r] for q, r in zip(qs, ratios)]


def target_curve(s_known, correct, s_pool, pi, eta, pcts):
    s_known, s_pool = np.asarray(s_known, float), np.asarray(s_pool, float)
    correct = np.asarray(correct, bool)
    rows = []
    for p in pcts:
        t = threshold_from_percentile(s_known, p)
        acc_mask = s_known >= t
        ka = min(np.mean(correct & acc_mask) / max(1e-6, 1 - eta), float(np.mean(acc_mask)))
        fk, fp = float(np.mean(s_known < t)), float(np.mean(s_pool < t))
        ur = float(np.clip((fp - pi * fk) / (1 - pi), 0, 1)) if pi < 0.999 else 0.0
        rows.append({"percentile": float(p), "known_acc_est": float(ka), "unknown_recall_est": ur,
                     "pool_reject_rate": fp})
    best_ur = 0.0
    for r in rows:                         # estimated unknown recall must be non-decreasing in the threshold
        best_ur = max(best_ur, r["unknown_recall_est"])
        r["unknown_recall_est"] = best_ur
        r["expected_acc"] = pi * r["known_acc_est"] + (1 - pi) * best_ur
    bi = int(np.argmax([r["expected_acc"] for r in rows]))
    return {"pi_known": pi, "curve": rows, "best_percentile": rows[bi]["percentile"],
            "best_expected_acc": rows[bi]["expected_acc"], "pool_reject_rate_at_best": rows[bi]["pool_reject_rate"]}


def final_fit(banks, labels, K, pcfg, ocfg, method, wd, device, score_name, lco_pct, bg_feats=None, eta=0.0,
              pi_score=None, log=print):
    """pi_score: score used to estimate the known fraction of the unseen pool (default: best LCO AUROC);
    the fraction is a property of the pool, so one estimate from the best-separating score is used for all
    curves (per-score estimates are logged as a consistency check)."""
    """Fit head + OOD stats on ALL labelled data. Calibration (known scores, holdout scores, correctness) comes
    from 5-fold cross-fitting so no labelled sample is scored by a model that saw it."""
    labels = np.asarray(labels)
    head, keep, info = fit_method(method, banks, labels, K, pcfg, 0, device, wd=wd)
    f0 = banks[0]
    scorer = OODScorer(ocfg).fit(f0[torch.as_tensor(np.where(keep)[0])], labels[keep], K, bank_feats=f0,
                                 bg_feats=bg_feats)
    base_names = scorer.available(include_ensemble=False)
    oof = {n: np.zeros(len(labels)) for n in base_names}
    correct = np.zeros(len(labels), bool)
    pool = {n: [] for n in base_names}
    for i, (tr, va) in enumerate(kfold_stratified(labels, int(ocfg.get("calib_folds", 5)), seed=123)):
        tr_t, va_t = torch.as_tensor(tr), torch.as_tensor(va)
        h_f, keep_f, _ = fit_method(method, banks[:, tr_t], labels[tr], K, pcfg, 100 + i, device, wd=wd)
        ftr = f0[tr_t]
        sc_f = OODScorer(ocfg).fit(ftr[torch.as_tensor(np.where(keep_f)[0])], labels[tr][keep_f], K, bank_feats=ftr,
                                   bg_feats=bg_feats)
        lva = h_f(f0[va_t])
        s = sc_f.score_all(f0[va_t], lva, base_names)
        correct[va] = lva.argmax(1).numpy() == labels[va]
        for n in base_names:
            oof[n][va] = s[n].numpy()
        if bg_feats is not None:
            sp = sc_f.score_all(bg_feats, h_f(bg_feats.float()), base_names, bg_self=True)
            for n in base_names:
                pool[n].append(sp[n].numpy())
    scorer.set_calibration({n: torch.as_tensor(v) for n, v in oof.items()})
    stats = scorer.st.get("ens_stats", {})
    if stats:
        oof["ensemble"] = np.mean([(oof[m] - stats[m][0]) / stats[m][1] for m in stats], axis=0)
    target = {}
    if bg_feats is not None:
        pool = {n: np.concatenate(v) for n, v in pool.items()}
        if stats:
            pool["ensemble"] = np.mean([(pool[m] - stats[m][0]) / stats[m][1] for m in stats], axis=0)
        eta_c = float(np.clip(eta, 0.0, float(ocfg.get("max_label_noise_correction", 0.4))))
        pcts_t = np.asarray(ocfg.get("target_percentiles", list(range(0, 91, 5))), float)
        qs = ocfg.get("target_pi_quantiles", (0.7, 0.8, 0.9))
        own = {n: estimate_known_fraction(oof[n], pool[n], qs) for n in oof}
        src = pi_score if pi_score in own else score_name
        pi_glob = own[src][0]
        log(f"  [target] known fraction of the unseen pool estimated with '{src}': {pi_glob:.3f} "
            f"(per-score estimates: " + ", ".join(f"{n}={v[0]:.2f}" for n, v in own.items()) + ")")
        for n in oof:
            pi = pi_glob if ocfg.get("pi_source", "best_auroc") == "best_auroc" else own[n][0]
            target[n] = target_curve(oof[n], correct, pool[n], pi, eta_c, pcts_t)
            target[n]["pi_ratios"] = own[n][1]
            target[n]["pi_own_estimate"] = own[n][0]
            target[n]["pi_source"] = src
            target[n]["eta_correction"] = eta_c
            log(f"  [target] {n:12s} known fraction used {pi:.3f} | best pct "
                f"{target[n]['best_percentile']:4g} | est. (K+1)-acc {target[n]['best_expected_acc']:.4f} | "
                f"pool reject rate {target[n]['pool_reject_rate_at_best']:.3f}")
    mode = ocfg.get("threshold_mode", "target")
    if mode == "target" and score_name in target:
        pct, rule = target[score_name]["best_percentile"], "target"
    else:
        pct, rule = float(lco_pct), "lco"
        if mode == "target":
            log("  [target] no calibration holdout available -> falling back to the LCO percentile")
    thr = threshold_from_percentile(oof[score_name], pct)
    train_acc = float((head(f0).argmax(1).numpy() == labels).mean())
    pool_rej_final = None
    if bg_feats is not None:      # diagnostic: rejection rate of the FINAL model on the unseen pool
        s_pool_final = scorer.score(score_name, bg_feats, head(bg_feats.float()), bg_self=True).numpy()
        pool_rej_final = float(np.mean(s_pool_final < thr))
    log(f"[final] method={method} kept {keep.mean():.3f} | score={score_name} pct={pct:g} ({rule}) thr={thr:.5f} | "
        f"train acc(noisy) {train_acc:.3f}" + (f" | final-model reject rate on unseen pool {pool_rej_final:.3f}"
                                               if pool_rej_final is not None else ""))
    return {"head": head, "scorer": scorer, "keep": keep, "threshold": thr, "percentile": pct, "rule": rule,
            "score": score_name, "cleaning": info, "oof_known_scores": {k: v.tolist() for k, v in oof.items()},
            "oof_closed_set_acc_noisy": float(correct.mean()), "target": target,
            "pool_reject_rate_final": pool_rej_final, "train_acc_noisy": train_acc}
