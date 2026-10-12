"""Load SSL encoder checkpoints written by train_ssl.py."""
import os

from models.resnet import build_encoder
from utils.checkpoint import load_ckpt, state_dict_hash


def default_encoder_path(out, which="best"):
    p = os.path.join(out["ssl"], f"encoder_{which}.pt")
    if not os.path.isfile(p) and which == "best":
        p = os.path.join(out["ssl"], "encoder_last.pt")
    return p


def load_encoder(path, device="cpu"):
    if not os.path.isfile(path):
        raise SystemExit(f"ERROR: encoder checkpoint not found: {path}\n       Run train_ssl.py first "
                         f"(or pass --encoder).")
    ck = load_ckpt(path)
    enc = build_encoder(ck["model_cfg"])
    enc.load_state_dict(ck["encoder"])
    enc.to(device).eval()
    for p in enc.parameters():
        p.requires_grad = False     # frozen encoder for probing / OOD
    ck["hash"] = state_dict_hash(ck["encoder"])
    return enc, ck
