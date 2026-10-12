"""Linear head training on frozen features.

Features are standardised with training statistics during optimisation; the affine standardisation
is folded into (W, b) afterwards so the deployed head is a single nn.Linear on raw encoder features:
    W' = W / sd,   b' = b - W' @ mu
"""
import numpy as np
import torch
import torch.nn as nn

from probe.losses import make_loss
from utils.optim import cosine_lr


def fold_standardisation(lin: nn.Linear, mu, sd):
    W = lin.weight.data / sd.view(1, -1)
    b = lin.bias.data - W @ mu
    out = nn.Linear(W.shape[1], W.shape[0])
    out.weight.data.copy_(W)
    out.bias.data.copy_(b)
    return out.requires_grad_(False)   # frozen from here on (scoring code calls .numpy() freely)


def train_linear(banks, labels, num_classes, pcfg, loss_name="ce", seed=0, device="cpu", wd=None,
                 epochs=None, sample_idx=None):
    """banks: (V, N, d) float tensor (view 0 = clean features). labels: (N,) long.
    sample_idx: subset of rows to train on (e.g. cleaned samples). Returns an nn.Linear on raw features."""
    if banks.dim() == 2:
        banks = banks.unsqueeze(0)
    labels = torch.as_tensor(labels, dtype=torch.long)
    if sample_idx is not None:
        sample_idx = torch.as_tensor(np.asarray(sample_idx), dtype=torch.long)
        banks, labels = banks[:, sample_idx], labels[sample_idx]
    V, N, d = banks.shape
    g = torch.Generator().manual_seed(int(seed))
    torch.manual_seed(int(seed))
    mu = banks[0].mean(0)
    sd = banks[0].std(0).clamp_min(1e-4)
    Xs = ((banks - mu) / sd).to(device)
    y = labels.to(device)
    lin = nn.Linear(d, num_classes).to(device)
    nn.init.zeros_(lin.weight)   # zero init: deterministic start, logistic-regression style
    nn.init.zeros_(lin.bias)
    wd = float(pcfg.get("weight_decay", 1e-3) if wd is None else wd)
    lr = float(pcfg.get("lr", 1e-3))
    epochs = int(pcfg.get("epochs", 100) if epochs is None else epochs)
    bs = int(min(pcfg.get("batch_size", 256), N))
    opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=wd)
    loss_fn = make_loss(loss_name, num_classes, pcfg.get("loss_params", {}))
    steps_per_epoch = max(1, int(np.ceil(N / bs)))
    total = epochs * steps_per_epoch
    step = 0
    lin.train()
    for ep in range(epochs):
        perm = torch.randperm(N, generator=g).to(device)
        view = torch.randint(0, V, (N,), generator=g).to(device)   # random feature view per sample
        for i in range(0, N, bs):
            idx = perm[i:i + bs]
            xb = Xs[view[idx], idx]
            for gr in opt.param_groups:
                gr["lr"] = cosine_lr(step, total, max(1, total // 20), lr)
            loss = loss_fn(lin(xb), y[idx]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
    lin = lin.cpu()
    return fold_standardisation(lin, mu, sd).eval()


@torch.no_grad()
def predict_logits(head: nn.Linear, feats):
    return head(feats.float())


def accuracy(head, feats, labels):
    if len(labels) == 0:
        return float("nan")
    pred = predict_logits(head, feats).argmax(1)
    return (pred == torch.as_tensor(labels)).float().mean().item()


def out_of_fold_probs(banks, labels, num_classes, pcfg, n_folds=5, seed=0, device="cpu", wd=None):
    """Cross-validated predicted probabilities (each sample predicted by a head that never saw it).
    Needed for confident learning / small-loss selection: in-sample predictions of a 512-d linear model
    on ~1k points can memorise noisy labels."""
    from datasets.splits import kfold_stratified
    labels = np.asarray(labels)
    probs = torch.zeros(len(labels), num_classes)
    for tr, va in kfold_stratified(labels, n_folds, seed):
        tr_t, va_t = torch.as_tensor(tr), torch.as_tensor(va)
        head = train_linear(banks[:, tr_t], labels[tr], num_classes, pcfg, "ce", seed, device, wd=wd,
                            epochs=pcfg.get("cleaning_epochs", pcfg.get("epochs", 100)))
        probs[va_t] = torch.softmax(predict_logits(head, banks[0, va_t]), 1)
    return probs
