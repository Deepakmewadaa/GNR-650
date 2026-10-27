"""OOD scores implemented from scratch (higher = more in-distribution).

logit-based (use the linear head):
  msp       : max softmax probability (Hendrycks & Gimpel 2017)
  maxlogit  : max logit (Hendrycks et al. 2022)
  energy    : T * logsumexp(logits / T)  (negative free energy; Liu et al. 2020)
feature-based (use frozen encoder features):
  knn       : - distance to the k-th nearest training feature, L2-normalised (Sun et al. 2022)
  mahalanobis: - min_c (z-mu_c)^T Sigma^-1 (z-mu_c), class means + SHARED covariance with shrinkage
               (Lee et al. 2018; shrinkage towards scaled identity a la Ledoit-Wolf, needed because
               N~1k samples vs d=512 dims makes the sample covariance ill-conditioned)
  cosine    : max_c cos(z, prototype_c)
  knn_ratio : mean top-k cosine similarity to the LABELLED bank minus mean top-k similarity to a BACKGROUND
              bank of unlabelled images never seen by SSL (a kNN density-ratio p_known(x)/p_pool(x), as in
              PU learning). Unknown-class test images have same-class neighbours in the unlabelled pool but
              not in the labelled set -> strongly negative; known-class images have neighbours in both -> ~0.
              Background neighbours with similarity > dup_sim are ignored (the same image is never its own
              evidence, e.g. when the holdout itself is scored).
ensemble    : mean of z-scored member scores; z-statistics come from KNOWN calibration data only.

All fitted statistics are plain tensors/floats so they can live inside final.pt (weights_only load).
"""
import torch
import torch.nn.functional as F

ALL_SCORES = ["msp", "maxlogit", "energy", "knn", "mahalanobis", "cosine", "knn_ratio", "ensemble"]
NEEDS = {"msp": [], "maxlogit": [], "energy": [], "knn": ["bank"], "cosine": ["protos", "present"],
         "mahalanobis": ["means", "prec", "present"], "knn_ratio": ["bank", "bg"]}


def _topk_mean_sim(z, bank, k, drop_above=None, self_is_bank=False, chunk=2048):
    """self_is_bank: z[i] IS bank[i] (scoring the bank itself) -> exclude exactly that entry (leave-one-out)."""
    k = max(1, min(k, bank.shape[0] - (1 if self_is_bank else 0)))
    out = []
    for i in range(0, z.shape[0], chunk):
        sim = z[i:i + chunk] @ bank.t()
        if self_is_bank:
            rows = torch.arange(sim.shape[0], device=sim.device)
            sim[rows, rows + i] = -2.0
        if drop_above is not None:
            sim = sim.masked_fill(sim > drop_above, -2.0)
        out.append(sim.topk(k, dim=1).values.mean(1))
    return torch.cat(out)


class OODScorer:
    def __init__(self, params=None):
        p = params or {}
        self.k = int(p.get("knn_k", 5))
        self.T = float(p.get("energy_T", 1.0))
        self.shrink = float(p.get("maha_shrinkage", 0.1))
        self.maha_norm = bool(p.get("maha_normalize", True))
        self.members = list(p.get("ensemble_members", ["knn", "mahalanobis", "energy", "cosine", "knn_ratio"]))
        self.bg_k = int(p.get("bg_k", 5))
        self.dup_sim = float(p.get("dup_sim", 0.9999))
        self.st = {}

    def available(self, include_ensemble=True):
        names = [n for n in ALL_SCORES if n != "ensemble" and all(k in self.st for k in NEEDS[n])]
        if include_ensemble and [m for m in self.members if m in names]:
            names.append("ensemble")
        return names

    def active_members(self):
        av = self.available(include_ensemble=False)
        return [m for m in self.members if m in av]

    # ------------------------------------------------------------------ fit
    @torch.no_grad()
    def fit(self, feats, labels, num_classes, bank_feats=None, bg_feats=None):
        """feats/labels: (cleaned) known training samples for class-conditional stats.
        bank_feats: label-free kNN bank (defaults to feats). bg_feats: unlabelled background bank (knn_ratio)."""
        feats = feats.float()
        labels = torch.as_tensor(labels, dtype=torch.long)
        z = F.normalize(feats, dim=1)
        bank = F.normalize((bank_feats if bank_feats is not None else feats).float(), dim=1)
        K, d = num_classes, feats.shape[1]
        protos = torch.zeros(K, d)
        mx = z if self.maha_norm else feats
        means = torch.zeros(K, d)
        for c in range(K):
            m = labels == c
            if m.any():
                protos[c] = F.normalize(z[m].mean(0), dim=0)
                means[c] = mx[m].mean(0)
        present = torch.tensor([bool((labels == c).any()) for c in range(K)])
        centred = mx - means[labels]
        cov = centred.t() @ centred / max(1, len(labels) - 1)
        tr = torch.trace(cov) / d
        cov = (1 - self.shrink) * cov + self.shrink * tr * torch.eye(d)
        prec = torch.linalg.inv(cov.double()).float().contiguous()   # inv() may return column-major memory
        self.st = {"bank": bank.contiguous(), "protos": protos, "means": means, "prec": prec,
                   "present": present, "num_classes": K}
        if bg_feats is not None and len(bg_feats):
            self.st["bg"] = F.normalize(bg_feats.float(), dim=1).contiguous()
        return self

    def set_calibration(self, scores_by_name):
        """z-score statistics for the ensemble from KNOWN calibration scores (out-of-fold / val)."""
        zs = {}
        for m in self.active_members():
            s = torch.as_tensor(scores_by_name[m]).float()
            zs[m] = (float(s.mean()), float(s.std().clamp_min(1e-8)))
        self.st["ens_stats"] = zs

    # ------------------------------------------------------------------ score
    @torch.no_grad()
    def score(self, name, feats, logits, bg_self=False):
        """bg_self=True when `feats` are exactly the background bank (in order): leave-one-out for knn_ratio."""
        feats = feats.float()
        logits = logits.float()
        if name == "msp":
            return F.softmax(logits, 1).max(1).values
        if name == "maxlogit":
            return logits.max(1).values
        if name == "energy":
            return self.T * torch.logsumexp(logits / self.T, 1)
        if name == "knn":
            z = F.normalize(feats, dim=1)
            bank = self.st["bank"].to(z.device)
            k = min(self.k, bank.shape[0])
            out = []
            for i in range(0, z.shape[0], 2048):
                sim = z[i:i + 2048] @ bank.t()
                kth = sim.topk(k, dim=1).values[:, -1]
                out.append(-torch.sqrt((2 - 2 * kth).clamp_min(0)))   # L2 distance on the unit sphere
            return torch.cat(out)
        if name == "mahalanobis":
            x = F.normalize(feats, dim=1) if self.maha_norm else feats
            means = self.st["means"].to(x.device)
            prec = self.st["prec"].to(x.device)
            present = self.st["present"].to(x.device)
            # (x-mu)^T P (x-mu) = x'Px - 2 x'P mu + mu'P mu, vectorised over classes
            xP = x @ prec
            d = (xP * x).sum(1, keepdim=True) - 2 * xP @ means.t() + ((means @ prec) * means).sum(1)[None]
            d = d.masked_fill(~present[None], float("inf"))
            return -d.min(1).values
        if name == "cosine":
            z = F.normalize(feats, dim=1)
            sim = z @ self.st["protos"].to(z.device).t()
            sim = sim.masked_fill(~self.st["present"].to(z.device)[None], -2.0)
            return sim.max(1).values
        if name == "knn_ratio":
            z = F.normalize(feats, dim=1)
            s_lab = _topk_mean_sim(z, self.st["bank"].to(z.device), self.k)
            s_bg = _topk_mean_sim(z, self.st["bg"].to(z.device), self.bg_k, drop_above=self.dup_sim,
                                  self_is_bank=bg_self)
            return s_lab - s_bg
        if name == "ensemble":
            if "ens_stats" not in self.st:
                return self.score("energy", feats, logits, bg_self)
            ms = list(self.st["ens_stats"])
            tot = 0
            for m in ms:
                mu, sd = self.st["ens_stats"][m]
                tot = tot + (self.score(m, feats, logits, bg_self) - mu) / sd
            return tot / len(ms)
        raise ValueError(f"unknown OOD score '{name}'")

    def score_all(self, feats, logits, names=None, bg_self=False):
        names = names or [n for n in self.available() if n != "ensemble" or "ens_stats" in self.st]
        return {n: self.score(n, feats, logits, bg_self).cpu() for n in names}

    # ------------------------------------------------------------------ (de)serialise
    def state_dict(self, only=None):
        """only: score name to keep just the statistics that score needs (smaller checkpoint)."""
        keys = set(self.st)
        if only is not None:
            ms = list(self.st.get("ens_stats", {})) if only == "ensemble" else [only]
            keys = {"num_classes"} | {k for m in ms for k in NEEDS[m]} | ({"ens_stats"} if only == "ensemble" else set())
        st = {k: (v.contiguous() if torch.is_tensor(v) else v) for k, v in self.st.items() if k in keys}
        if "ens_stats" in st:
            st["ens_stats"] = {m: [float(a), float(b)] for m, (a, b) in st["ens_stats"].items()}
        return {"params": {"knn_k": self.k, "energy_T": self.T, "maha_shrinkage": self.shrink,
                           "maha_normalize": self.maha_norm, "ensemble_members": self.members,
                           "bg_k": self.bg_k, "dup_sim": self.dup_sim},
                "stats": st}

    @classmethod
    def from_state_dict(cls, sd):
        o = cls(sd.get("params", {}))
        o.st = dict(sd.get("stats", sd.get("st", {})))
        if "ens_stats" in o.st:
            o.st["ens_stats"] = {m: (float(v[0]), float(v[1])) for m, v in o.st["ens_stats"].items()}
        return o
