"""Common interface every SSL method implements, so train_ssl.py is method-agnostic."""
import torch.nn as nn


class SSLMethod(nn.Module):
    """Subclasses own the encoder + any heads / teacher networks.

    training_step(x_uint8) -> (loss, logs)   x_uint8: (B, C, S, S) uint8 batch on device; the method
                                             creates its own augmented views on the GPU.
    after_step(step, total_steps)            optional per-step hook (e.g. EMA teacher update)
    eval_encoder()                           backbone whose pooled features are evaluated/exported
    """

    name = "base"
    _compiled = None      # optional torch.compile'd encoder (set with object.__setattr__ -> not a submodule)

    def enc(self, x):
        """Training-time encoder forward (compiled if available). State dicts always use self.encoder."""
        return self._compiled(x) if self._compiled is not None else self.encoder(x)

    def training_step(self, x_uint8):
        raise NotImplementedError

    def before_optimizer_step(self):
        pass

    def after_step(self, step, total_steps):
        pass

    def on_epoch_start(self, epoch, total_epochs):
        pass

    def eval_encoder(self):
        return self.encoder

    def trainable_named_parameters(self):
        return [(n, p) for n, p in self.named_parameters() if p.requires_grad]


def build_ssl_method(cfg, encoder, meta):
    name = cfg["ssl"]["method"].lower()
    if name == "simclr":
        from ssl_methods.simclr import SimCLR
        return SimCLR(encoder, cfg, meta)
    if name == "dino":
        from ssl_methods.dino import DINO
        return DINO(encoder, cfg, meta)
    if name in ("barlow", "barlow_twins", "bt"):
        from ssl_methods.barlow_twins import BarlowTwins
        return BarlowTwins(encoder, cfg, meta)
    if name in ("byol",):
        from ssl_methods.byol import BYOL
        return BYOL(encoder, cfg, meta)
    if name in ("looc_byol", "loocbyol"):
        from ssl_methods.looc_byol import LooCBYOL
        return LooCBYOL(encoder, cfg, meta)
    if name in ("swav",):
        from ssl_methods.swav import SwAV
        return SwAV(encoder, cfg, meta)
    if name in ("looc_swav", "loocswav", "looc"):
        from ssl_methods.looc_swav import LooCSwAV
        return LooCSwAV(encoder, cfg, meta)
    if name in ("vicreg", "vic"):
        from ssl_methods.vicreg import VICReg
        return VICReg(encoder, cfg, meta)
    raise ValueError(f"unknown SSL method '{name}' (choose simclr|dino|byol|barlow_twins|swav|vicreg)")
