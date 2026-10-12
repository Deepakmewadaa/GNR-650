"""Unseen-pool threshold machinery: knn_ratio, mixture-proportion estimate, expected-accuracy curve,
and the SSL/holdout partition."""
import os

import numpy as np
import torch

from datasets.data import ssl_partition
from ood.metrics import auroc
from ood.protocol import estimate_known_fraction, target_curve
from ood.scores import OODScorer


def test_known_fraction_recovered():
    rng = np.random.RandomState(0)
    s_known = rng.randn(1300) + 3.0
    for pi in (0.3, 0.5, 0.8):
        n = 2500
        nk = int(pi * n)
        pool = np.r_[rng.randn(nk) + 3.0, rng.randn(n - nk) - 1.0]
        est, _ = estimate_known_fraction(s_known, pool)
        assert abs(est - pi) < 0.08, (pi, est)    # inherent sampling noise ~ +-0.05 at these sizes


def test_target_curve_prefers_more_rejection_when_many_unknowns():
    rng = np.random.RandomState(1)
    s_known = rng.randn(1000) + 2.0
    correct = rng.rand(1000) < 0.8
    pools = {pi: np.r_[rng.randn(int(pi * 2000)) + 2.0, rng.randn(2000 - int(pi * 2000))] for pi in (0.9, 0.3)}
    pcts = list(range(0, 91, 5))
    p_hi = target_curve(s_known, correct, pools[0.9], 0.9, 0.0, pcts)["best_percentile"]
    p_lo = target_curve(s_known, correct, pools[0.3], 0.3, 0.0, pcts)["best_percentile"]
    assert p_lo > p_hi        # fewer knowns in the pool -> reject more aggressively


def test_unknown_recall_estimate_monotone_and_bounded():
    rng = np.random.RandomState(2)
    s_known = rng.randn(500) + 1
    pool = np.r_[rng.randn(500) + 1, rng.randn(500) - 1]
    t = target_curve(s_known, np.ones(500, bool), pool, 0.5, 0.2, list(range(0, 91, 10)))
    ur = [r["unknown_recall_est"] for r in t["curve"]]
    assert all(0 <= u <= 1 for u in ur) and all(b >= a for a, b in zip(ur, ur[1:]))


def test_knn_ratio_separates_pool_only_classes():
    g = torch.Generator().manual_seed(0)
    d = 32
    centres = torch.randn(6, d, generator=g) * 3
    lab_y = torch.arange(3).repeat_interleave(40)                 # classes 0-2 labelled (known)
    lab = centres[lab_y] + torch.randn(120, d, generator=g)
    pool_y = torch.arange(6).repeat_interleave(40)                # pool: known 0-2 AND unknown 3-5
    pool = centres[pool_y] + torch.randn(240, d, generator=g)
    sc = OODScorer({"knn_k": 5, "bg_k": 5}).fit(lab, lab_y, 3, bg_feats=pool)
    test_y = torch.arange(6).repeat_interleave(20)
    test = centres[test_y] + torch.randn(120, d, generator=g)
    s = sc.score("knn_ratio", test, torch.zeros(120, 3)).numpy()
    assert auroc(s[test_y.numpy() < 3], s[test_y.numpy() >= 3]) > 0.95
    # scoring the background itself: without leave-one-out an image is its own nearest neighbour (sim 1),
    # which inflates the background similarity and pushes knn_ratio down
    sc_nodup = OODScorer({"knn_k": 5, "bg_k": 5, "dup_sim": 2.0}).fit(lab, lab_y, 3, bg_feats=pool)
    loo = sc_nodup.score("knn_ratio", pool, torch.zeros(240, 3), bg_self=True)
    naive = sc_nodup.score("knn_ratio", pool, torch.zeros(240, 3), bg_self=False)
    assert loo.mean() > naive.mean() + 1e-3
    # the identical-file rule alone gives the same answer as explicit leave-one-out for exact copies
    dup_rule = sc.score("knn_ratio", pool, torch.zeros(240, 3), bg_self=False)
    assert torch.allclose(dup_rule, sc.score("knn_ratio", pool, torch.zeros(240, 3), bg_self=True), atol=1e-6)


def test_partition_deterministic_disjoint(tmp_path):
    lab, unl = [], []
    for i in range(30):
        p = tmp_path / f"u{i:03d}.png"
        p.write_bytes(os.urandom(50 + i))
        unl.append(str(p))
    for i in range(5):                       # labelled files = byte copies of some unlabelled files
        q = tmp_path / f"l{i}.png"
        q.write_bytes(open(unl[i * 3], "rb").read())
        lab.append(str(q))
    cfg = {"calib_holdout": 6, "calib_holdout_max_frac": 0.5}
    a = ssl_partition(unl, lab, cfg, verbose=False)
    b = ssl_partition(unl, lab, cfg, verbose=False)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert sorted(a["dup_idx"].tolist()) == [0, 3, 6, 9, 12]
    assert len(a["holdout_idx"]) == 6
    allk = np.concatenate([a["ssl_idx"], a["holdout_idx"], a["dup_idx"]])
    assert len(allk) == 30 and len(np.unique(allk)) == 30
