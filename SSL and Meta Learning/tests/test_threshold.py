import numpy as np

from ood.metrics import open_set_metrics, threshold_from_percentile
from ood.protocol import aggregate, select_score


def test_percentile_rejects_expected_fraction():
    s = np.arange(1000, dtype=float)
    for p in (0, 5, 10, 25, 50):
        thr = threshold_from_percentile(s, p)
        rej = np.mean(s < thr)
        assert abs(rej - p / 100) <= 0.002


def test_open_set_metrics_toy():
    pred = np.array([0, 1, 1, 2])
    y = np.array([0, 1, 2, 2])
    s_known = np.array([0.9, 0.8, 0.7, 0.1])     # last known sample rejected at thr 0.5
    s_unk = np.array([0.2, 0.6])                 # one unknown caught, one accepted
    m = open_set_metrics(pred, y, s_known, s_unk, 0.5)
    assert m["known_acc"] == 0.5                 # samples 0,1 correct+accepted; 2 wrong; 3 rejected
    assert m["unknown_recall"] == 0.5
    assert m["balanced_acc"] == 0.5
    assert abs(m["kplus1_acc"] - (0.5 * 4 + 0.5 * 2) / 6) < 1e-12
    assert m["closed_set_acc"] == 0.75


def _fake_fold(shift):
    rng = np.random.RandomState(0)
    pcts = np.array([0, 5, 10, 20])
    k = rng.randn(100) + shift
    u = rng.randn(100)
    pred = np.zeros(100, int)
    y = np.zeros(100, int)
    curve = [open_set_metrics(pred, y, k, u, threshold_from_percentile(k, p)) for p in pcts]
    return {"scores": {"good": {"summary": {"auroc": 0.9, "aupr_in": 0, "aupr_out": 0, "fpr95": 0}, "curve": curve},
                       "bad": {"summary": {"auroc": 0.5, "aupr_in": 0, "aupr_out": 0, "fpr95": 0},
                               "curve": [open_set_metrics(pred, y, u, u, threshold_from_percentile(u, p))
                                         for p in pcts]}}}, pcts


def test_aggregate_and_select():
    f, pcts = _fake_fold(3.0)
    agg = aggregate([f, f], pcts)
    best, p = select_score(agg, ["good", "bad"])
    assert best == "good" and p in pcts and p > 0
