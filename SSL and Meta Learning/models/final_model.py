"""The deployed model: encoder + linear head + OOD scorer + threshold, packaged in ONE checkpoint.

Used by test.py and by export_final.py's self-test, so both exercise exactly the same code path.
The checkpoint contains only tensors and python builtins -> loads with torch.load(weights_only=True).
"""
import torch
import torch.nn as nn

from models.resnet import build_encoder
from ood.scores import OODScorer
from probe.features import encode_uint8

FORMAT_VERSION = 1
REQUIRED_KEYS = ("format_version", "num_classes", "image_size", "channels", "mean", "std", "model_cfg",
                 "encoder", "head", "ood")


class FinalModel(nn.Module):
    def __init__(self, ck):
        super().__init__()
        missing = [k for k in REQUIRED_KEYS if k not in ck]
        if missing:
            raise ValueError(f"checkpoint is missing keys {missing}")
        self.K = int(ck["num_classes"])
        self.image_size = int(ck["image_size"])
        self.channels = int(ck["channels"])
        self.mean, self.std = list(ck["mean"]), list(ck["std"])
        self.tta_flip = bool(ck.get("tta_flip", True))
        self.tta_multiscale = bool(ck.get("tta_multiscale", False))
        self.encoder = build_encoder(ck["model_cfg"])
        self.encoder.load_state_dict({k: v.float() for k, v in ck["encoder"].items()})
        W, b = ck["head"]["weight"].float(), ck["head"]["bias"].float()
        self.head = nn.Linear(W.shape[1], W.shape[0])
        self.head.weight.data.copy_(W)
        self.head.bias.data.copy_(b)
        o = ck["ood"]
        self.score_name = o["score"]
        self.threshold = float(o["threshold"])
        self.scorer = OODScorer.from_state_dict(o["scorer"])
        if W.shape[0] != self.K:
            raise ValueError(f"head has {W.shape[0]} outputs but num_classes={self.K}")
        self.eval()

    def to(self, device):
        super().to(device)
        for k, v in list(self.scorer.st.items()):
            if torch.is_tensor(v):
                self.scorer.st[k] = v.to(device)
        return self

    @torch.no_grad()
    def predict(self, x_uint8, amp=None):
        """x_uint8: (B,C,S,S) uint8 on the model device. Returns (labels in 0..K, ood score, closed-set pred)."""
        f = encode_uint8(self.encoder, x_uint8, self.mean, self.std, self.tta_flip, amp, tta_multiscale=self.tta_multiscale)
        logits = self.head(f)
        closed = logits.argmax(1)
        s = self.scorer.score(self.score_name, f, logits)
        labels = torch.where(s >= self.threshold, closed, torch.full_like(closed, self.K))
        return labels, s, closed


def load_final(path, device="cpu"):
    from utils.checkpoint import load_ckpt
    ck = load_ckpt(path, map_location="cpu")
    if int(ck.get("format_version", -1)) != FORMAT_VERSION:
        print(f"[warn] checkpoint format_version {ck.get('format_version')} != {FORMAT_VERSION}")
    m = FinalModel(ck).to(device)
    return m, ck
