"""DINO (Caron et al., ICCV 2021): self-distillation with no labels, implemented from scratch.

Student (backbone + DINO head) sees 2 global + n local crops; the EMA teacher sees the 2 global crops.
Loss = cross-entropy between sharpened, centred teacher distributions and student distributions for
every (teacher-global, other-student-view) pair. Centering + sharpening prevent collapse.
"""
import copy

import torch
import torch.nn.functional as F

from models.heads import DINOHead
from ssl_methods.augment import GPUAugment
from ssl_methods.base import SSLMethod
from utils.optim import cosine_value


class DINO(SSLMethod):
    name = "dino"

    def __init__(self, encoder, cfg, meta):
        super().__init__()
        dc = cfg.get("dino", {})
        self.encoder = encoder
        self.head = DINOHead(encoder.out_dim, dc.get("out_dim", 4096), dc.get("hidden", 2048),
                             dc.get("bottleneck", 256))
        self.t_encoder = copy.deepcopy(encoder)
        self.t_head = copy.deepcopy(self.head)
        for p in list(self.t_encoder.parameters()) + list(self.t_head.parameters()):
            p.requires_grad = False
        self.register_buffer("center", torch.zeros(1, dc.get("out_dim", 4096)))
        self.student_temp = float(dc.get("student_temp", 0.1))
        self.t_temp_start = float(dc.get("teacher_temp_start", 0.04))
        self.t_temp_end = float(dc.get("teacher_temp", 0.07))
        self.t_temp_warmup_frac = float(dc.get("teacher_temp_warmup_frac", 0.1))
        self.center_m = float(dc.get("center_momentum", 0.9))
        self.m_start = float(dc.get("momentum_teacher", 0.996))
        self.n_local = int(dc.get("n_local", 6))
        S = meta["image_size"]
        self.local_size = int(dc.get("local_size", max(16, round(S * 96 / 224 / 8) * 8)))
        base = dict(cfg.get("aug", {}))
        g1 = dict(base, crop_scale=dc.get("global_scale", [0.25, 1.0]), blur_p=1.0, solarize_p=0.0)
        g2 = dict(base, crop_scale=dc.get("global_scale", [0.25, 1.0]), blur_p=0.1, solarize_p=0.2)
        lo = dict(base, crop_scale=dc.get("local_scale", [0.05, 0.25]), blur_p=0.5, solarize_p=0.0)
        self.aug_g1 = GPUAugment(g1, meta["mean"], meta["std"], S)
        self.aug_g2 = GPUAugment(g2, meta["mean"], meta["std"], S)
        self.aug_l = GPUAugment(lo, meta["mean"], meta["std"], self.local_size)
        self._t_temp = self.t_temp_start
        self.freeze_last_frac = float(dc.get("freeze_last_layer_frac", 0.01))
        self._progress = 0.0

    def on_epoch_start(self, epoch, total_epochs):
        self._progress = epoch / max(1, total_epochs)

    def _teacher_temp(self, step, total):
        warm = max(1, int(self.t_temp_warmup_frac * total))
        if step < warm:
            return self.t_temp_start + (self.t_temp_end - self.t_temp_start) * step / warm
        return self.t_temp_end

    def training_step(self, x, step=0, total=1):
        g = [self.aug_g1(x), self.aug_g2(x)]
        loc = [self.aug_l(x) for _ in range(self.n_local)]
        with torch.no_grad():
            t_out = self.t_head(self.t_encoder(torch.cat(g, 0))).float()
        s_out = [self.head(self.encoder(torch.cat(g, 0)))]
        if loc:
            s_out.append(self.head(self.encoder(torch.cat(loc, 0))))
        s_out = torch.cat(s_out, 0).float()
        n = x.shape[0]
        s_chunks = (s_out / self.student_temp).chunk(2 + len(loc))
        tt = self._t_temp
        t_probs = F.softmax((t_out - self.center) / tt, dim=-1).chunk(2)
        loss, terms = 0.0, 0
        for iq, q in enumerate(t_probs):
            for iv, v in enumerate(s_chunks):
                if iv == iq:
                    continue
                loss = loss + torch.sum(-q * F.log_softmax(v, dim=-1), dim=-1).mean()
                terms += 1
        loss = loss / terms
        with torch.no_grad():
            bc = t_out.mean(0, keepdim=True)
            self.center.mul_(self.center_m).add_(bc, alpha=1 - self.center_m)
            ent = -(t_probs[0] * torch.log(t_probs[0].clamp_min(1e-12))).sum(-1).mean()
        return loss, {"teacher_entropy": ent.detach()}

    def before_optimizer_step(self):
        # Freeze the last (prototype) layer during the first iterations (DINO trick for stability).
        if self._progress < self.freeze_last_frac:
            self.head.v.grad = None

    @torch.no_grad()
    def after_step(self, step, total_steps):
        m = cosine_value(step, total_steps, self.m_start, 1.0)
        for ps, pt in zip(self.encoder.parameters(), self.t_encoder.parameters()):
            pt.mul_(m).add_(ps.detach(), alpha=1 - m)
        for ps, pt in zip(self.head.parameters(), self.t_head.parameters()):
            pt.mul_(m).add_(ps.detach(), alpha=1 - m)
        for bs, bt in zip(self.encoder.buffers(), self.t_encoder.buffers()):
            bt.copy_(bs)          # BN running stats copied (teacher has no gradient-based BN updates)
        self._t_temp = self._teacher_temp(step + 1, total_steps)

    def eval_encoder(self):
        return self.t_encoder     # teacher is usually the better representation in DINO

    def trainable_named_parameters(self):
        return [(n, p) for n, p in self.named_parameters()
                if p.requires_grad and not n.startswith("t_")]
