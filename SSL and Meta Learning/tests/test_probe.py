import numpy as np
import torch

from probe.cleaning import confident_learning, run_cleaner
from probe.linear import fold_standardisation, out_of_fold_probs, train_linear
from probe.losses import LOSSES, make_loss


def test_losses_finite_and_differentiable():
    logits = torch.randn(8, 5, requires_grad=True)
    y = torch.randint(0, 5, (8,))
    for name in LOSSES:
        l = make_loss(name, 5)(logits, y)
        assert l.shape == (8,) and torch.isfinite(l).all()
        l.mean().backward()
        assert torch.isfinite(logits.grad).all()
        logits.grad = None


def test_fold_standardisation_equivalent():
    lin = torch.nn.Linear(6, 3)
    x = torch.randn(10, 6)
    mu, sd = x.mean(0), x.std(0)
    folded = fold_standardisation(lin, mu, sd)
    assert torch.allclose(lin((x - mu) / sd), folded(x), atol=1e-5)


def _noisy(K=4, n=40, d=8, noise=0.2, seed=0):
    rng = np.random.RandomState(seed)
    c = rng.randn(K, d) * 5
    y = np.repeat(np.arange(K), n)
    x = c[y] + rng.randn(len(y), d)
    yn = y.copy()
    flip = rng.rand(len(y)) < noise
    yn[flip] = (y[flip] + rng.randint(1, K, flip.sum())) % K
    return torch.tensor(x, dtype=torch.float32), y, yn, flip


def test_confident_learning_finds_flips_and_rate():
    x, y, yn, flip = _noisy()
    pc = {"epochs": 50, "lr": 1e-2, "batch_size": 64}
    probs = out_of_fold_probs(x.unsqueeze(0), yn, 4, pc, 4, 0)
    flagged, eta, _ = confident_learning(probs, yn, 4)
    recall = (flagged & flip).sum() / flip.sum()
    precision = (flagged & flip).sum() / max(1, flagged.sum())
    assert recall > 0.8 and precision > 0.8
    assert abs(eta - flip.mean()) < 0.1


def test_cleaners_improve_label_purity():
    x, y, yn, flip = _noisy()
    pc = {"epochs": 50, "lr": 1e-2, "batch_size": 64}
    probs = out_of_fold_probs(x.unsqueeze(0), yn, 4, pc, 4, 0)
    for name in ("knn", "proto", "cl", "smallloss", "consensus"):
        keep, info = run_cleaner(name, x, yn, 4, probs, {"knn_k": 5})
        assert (yn[keep] == y[keep]).mean() > (yn == y).mean(), name
        for c in range(4):
            assert keep[yn == c].sum() >= 0.3 * (yn == c).sum() - 1      # per-class floor


def test_linear_learns_separable():
    x, y, _, _ = _noisy(noise=0.0)
    h = train_linear(x.unsqueeze(0), y, 4, {"epochs": 40, "lr": 1e-2}, "ce", 0, "cpu")
    assert (h(x).argmax(1).numpy() == y).mean() > 0.95
