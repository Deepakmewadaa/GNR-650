import numpy as np
import pytest
import torch

from ood.metrics import auroc, average_precision, fpr_at_tpr, ood_summary
from ood.scores import ALL_SCORES, OODScorer
from probe.linear import train_linear


def _clusters(n_per=60, K=4, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    centres = torch.randn(K, d, generator=g) * 4
    x = torch.cat([centres[k] + torch.randn(n_per, d, generator=g) for k in range(K)])
    y = torch.arange(K).repeat_interleave(n_per)
    return x, y, centres


def test_far_points_score_lower_feature_scores():
    x, y, c = _clusters()
    head = train_linear(x.unsqueeze(0), y, 4, {"epochs": 60, "lr": 1e-2, "batch_size": 64}, "ce", 0, "cpu")
    sc = OODScorer({"knn_k": 3}).fit(x, y, 4)
    xin = c[torch.arange(4).repeat_interleave(20)] + torch.randn(80, 16)   # fresh in-distribution samples
    xfar = torch.randn(80, 16) * 4          # same overall scale, random directions -> far from every cluster
    sc.set_calibration(sc.score_all(xin, head(xin), sc.available(include_ensemble=False)))
    for name in ("knn", "mahalanobis", "cosine", "ensemble"):
        s_in = sc.score(name, xin, head(xin)).numpy()
        s_far = sc.score(name, xfar, head(xfar)).numpy()
        assert auroc(s_in, s_far) > 0.9, name


def test_logit_scores_shapes_and_order():
    logits = torch.tensor([[10.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    sc = OODScorer()
    for name in ("msp", "maxlogit", "energy"):
        s = sc.score(name, torch.zeros(2, 4), logits)
        assert s.shape == (2,) and s[0] > s[1]


def test_state_dict_roundtrip():
    x, y, _ = _clusters()
    head = train_linear(x.unsqueeze(0), y, 4, {"epochs": 5}, "ce", 0, "cpu")
    sc = OODScorer().fit(x, y, 4)
    sc = OODScorer().fit(x, y, 4, bg_feats=x[::2] + 0.1)
    sc.set_calibration(sc.score_all(x, head(x), sc.available(include_ensemble=False)))
    assert set(sc.available()) == set(ALL_SCORES)
    for only in (None, "knn", "mahalanobis", "knn_ratio", "ensemble"):
        sc2 = OODScorer.from_state_dict(sc.state_dict(only=only))
        names = ALL_SCORES if only is None else [only]
        for n in names:
            assert torch.allclose(sc.score(n, x, head(x)), sc2.score(n, x, head(x))), (only, n)


def test_auroc_known_cases():
    assert auroc([2, 3, 4], [0, 1]) == 1.0
    assert auroc([0, 1], [2, 3, 4]) == 0.0
    assert auroc([1, 1], [1, 1]) == 0.5
    assert fpr_at_tpr(np.arange(100), np.array([200.0])) == 1.0
    assert fpr_at_tpr(np.arange(100) + 100, np.arange(50)) == 0.0


def test_against_sklearn_if_available():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.RandomState(0)
    k, u = rng.randn(200) + 1, rng.randn(150)
    k[:20] = np.round(k[:20], 1)
    u[:20] = np.round(u[:20], 1)
    y = np.r_[np.ones(200), np.zeros(150)]
    s = np.r_[k, u]
    assert abs(auroc(k, u) - sk.roc_auc_score(y, s)) < 1e-9
    assert abs(average_precision(s, y) - sk.average_precision_score(y, s)) < 1e-9


def test_summary_keys():
    d = ood_summary(np.random.rand(50) + 1, np.random.rand(40))
    assert set(d) == {"auroc", "aupr_in", "aupr_out", "fpr95"} and d["auroc"] == 1.0
