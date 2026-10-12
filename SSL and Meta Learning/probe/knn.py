"""Weighted kNN classifier on cosine similarity (Wu et al. 2018 / DINO evaluation protocol)."""
import torch
import torch.nn.functional as F


@torch.no_grad()
def knn_predict(train_f, train_y, test_f, num_classes, k=20, T=0.07, exclude_self=False, chunk=1024):
    """Returns (pred (M,), class-vote scores (M,K)). exclude_self=True when test_f is train_f."""
    tr = F.normalize(train_f.float(), dim=1)
    te = F.normalize(test_f.float(), dim=1)
    k = max(1, min(k, tr.shape[0] - (1 if exclude_self else 0)))
    preds, votes_all = [], []
    for i in range(0, te.shape[0], chunk):
        sim = te[i:i + chunk] @ tr.t()
        if exclude_self:
            rows = torch.arange(sim.shape[0], device=sim.device)
            sim[rows, rows + i] = -2.0
        s, idx = sim.topk(k, dim=1)
        w = (s / T).exp()
        votes = torch.zeros(sim.shape[0], num_classes, device=sim.device)
        votes.scatter_add_(1, train_y.to(sim.device)[idx], w)
        votes = votes / votes.sum(1, keepdim=True).clamp_min(1e-12)
        preds.append(votes.argmax(1))
        votes_all.append(votes)
    return torch.cat(preds), torch.cat(votes_all)


def knn_accuracy(train_f, train_y, test_f, test_y, num_classes, k=20, T=0.07):
    pred, _ = knn_predict(train_f, train_y, test_f, num_classes, k, T)
    return (pred.cpu() == test_y.cpu()).float().mean().item()
