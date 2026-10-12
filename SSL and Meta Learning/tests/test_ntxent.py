import math

import torch
import torch.nn.functional as F

from ssl_methods.simclr import nt_xent


def naive_nt_xent(z1, z2, tau):
    """Literal loop implementation of the SimCLR loss l(i,j) averaged over all 2N anchors."""
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    n = z1.shape[0]
    total = 0.0
    for i in range(2 * n):
        j = (i + n) % (2 * n)
        num = math.exp(float(z[i] @ z[j]) / tau)
        den = sum(math.exp(float(z[i] @ z[k]) / tau) for k in range(2 * n) if k != i)
        total += -math.log(num / den)
    return total / (2 * n)


def test_matches_naive():
    torch.manual_seed(0)
    z1, z2 = torch.randn(5, 7), torch.randn(5, 7)
    for tau in (0.1, 0.5, 1.0):
        loss, _ = nt_xent(z1, z2, tau)
        assert abs(loss.item() - naive_nt_xent(z1, z2, tau)) < 1e-4


def test_identical_views_low_loss_and_acc():
    torch.manual_seed(1)
    z = torch.randn(16, 32)
    loss_same, acc = nt_xent(z, z.clone(), 0.1)
    loss_rand, _ = nt_xent(z, torch.randn(16, 32), 0.1)
    assert loss_same < loss_rand and acc.item() == 1.0


def test_gradients_flow():
    z1 = torch.randn(4, 8, requires_grad=True)
    z2 = torch.randn(4, 8, requires_grad=True)
    loss, _ = nt_xent(z1, z2, 0.2)
    loss.backward()
    assert torch.isfinite(z1.grad).all() and z1.grad.abs().sum() > 0
