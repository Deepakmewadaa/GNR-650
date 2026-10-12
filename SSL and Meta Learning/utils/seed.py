import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def rng_state():
    st = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st):
    try:
        random.setstate(st["python"])
        np.random.set_state(st["numpy"])
        torch.set_rng_state(st["torch"])
        if "cuda" in st and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(st["cuda"])
    except Exception as e:  # RNG state is a nicety on resume, never fatal
        print(f"[warn] could not restore RNG state: {e}")
