"""SimCLR (Chen et al., ICML 2020): two strongly augmented views per image, MLP projection head,
NT-Xent (normalised temperature-scaled cross entropy) over the 2N views of a batch."""
import torch
import torch.nn.functional as F

from models.heads import MLPHead
from ssl_methods.augment import GPUAugment
from ssl_methods.base import SSLMethod


def nt_xent(z1, z2, temperature):
    """z1, z2: (N, d) projections of the two views. Positives are (i, i+N); the other 2N-2 views in
    the batch are negatives. Computed in fp32 for numerical safety under AMP.
    Returns (loss, positive-pair top-1 accuracy)."""
    z = F.normalize(torch.cat([z1, z2], 0).float(), dim=1)
    n = z1.shape[0]
    sim = z @ z.t() / temperature
    sim.fill_diagonal_(float("-inf"))           # a view is never its own positive/negative
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    loss = F.cross_entropy(sim, target)
    acc = (sim.argmax(1) == target).float().mean()
    return loss, acc


class SimCLR(SSLMethod):
    name = "simclr"

    def __init__(self, encoder, cfg, meta):
        super().__init__()
        sc = cfg.get("simclr", {})
        self.encoder = encoder
        self.proj = MLPHead(encoder.out_dim, sc.get("proj_hidden", 2048), sc.get("proj_dim", 128),
                            n_layers=sc.get("proj_layers", 2))
        self.temperature = float(sc.get("temperature", 0.2))
        self.aug = GPUAugment(cfg.get("aug", {}), meta["mean"], meta["std"], meta["image_size"])
        # optional asymmetric view-2 overrides (BYOL-style), e.g. {"solarize_p": 0.2}
        v2 = dict(cfg.get("aug", {}))
        v2.update(sc.get("view2_overrides", {}) or {})
        self.aug2 = GPUAugment(v2, meta["mean"], meta["std"], meta["image_size"])

    def training_step(self, x):
        v1, v2 = self.aug(x), self.aug2(x)
        h = self.enc(torch.cat([v1, v2], 0))
        z = self.proj(h)
        n = x.shape[0]
        loss, acc = nt_xent(z[:n], z[n:], self.temperature)
        return loss, {"pos_acc": acc.detach()}   # tensor: avoids a GPU sync every step
