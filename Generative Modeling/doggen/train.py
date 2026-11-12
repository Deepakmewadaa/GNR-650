import argparse
import copy
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from .common import seed_all, device_for, sha256, write_json
from .data import Dogs
from .model import TextToImage, Discriminator, parameter_counts
from .text import tokenize
from .losses import augment, discriminator_loss, generator_loss


def save_checkpoint(path, state):
    tmp = Path(str(path)+'.tmp')
    torch.save(state, tmp)
    tmp.replace(path)


def train(args):
    seed_all(args.seed)
    torch.set_num_threads(args.threads)
    device = device_for(args.device)
    dataset = Dogs(args.manifest)
    if args.batch_size < 2:
        raise ValueError('Batch size must be >=2 for mismatched text')
    sampler_rng = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=args.workers, pin_memory=device.type=='cuda', generator=sampler_rng)
    if len(loader) == 0:
        raise ValueError('Training dataset smaller than batch size')
    model = TextToImage(base=args.base).to(device)
    disc = Discriminator(len(dataset.breeds), base=args.base).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    counts = parameter_counts(model, disc)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/'parameters.json', counts)
    print(json.dumps({'device':str(device), **counts}), flush=True)
    opt_g = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0., .9))
    opt_d = torch.optim.Adam(disc.parameters(), lr=args.lr, betas=(0., .9))
    amp = args.amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    start, manifest_hash = 0, sha256(args.manifest)
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if ckpt.get('format_version') != 2:
            raise ValueError('Checkpoint text encoder format mismatch')
        if ckpt['manifest_sha256'] != manifest_hash or ckpt['base'] != args.base:
            raise ValueError('Resume dataset or architecture mismatch')
        for obj, key in [(model,'model'), (disc,'discriminator'), (ema,'ema'), (opt_g,'optimizer_g'), (opt_d,'optimizer_d'), (scaler,'scaler')]:
            obj.load_state_dict(ckpt[key])
        start = ckpt['step']
        random.setstate(ckpt['python_rng'])
        np.random.set_state(ckpt['numpy_rng'])
        torch.set_rng_state(ckpt['torch_rng'].cpu())
        if device.type == 'cuda' and ckpt.get('cuda_rng'):
            torch.cuda.set_rng_state_all([r.cpu() for r in ckpt['cuda_rng']])
        sampler_rng.set_state(ckpt['sampler_rng'].cpu())
    iterator, began = iter(loader), time.perf_counter()
    fixed_tokens = tokenize([dataset.rows[i % len(dataset)]['caption'] for i in range(16)], device)
    fixed_z = torch.randn(16, model.generator.z_dim, device=device)
    for step in range(start+1, args.steps+1):
        try:
            real, captions, labels = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            real, captions, labels = next(iterator)
        real, labels = real.to(device), labels.to(device)
        tokens = tokenize(captions, device)
        batch = real.shape[0]
        model.train()
        disc.train().requires_grad_(True)
        opt_d.zero_grad(set_to_none=True)
        with torch.autocast(device.type, enabled=amp):
            with torch.no_grad():
                cond = model.text_encoder(tokens)
                fake = model.generator(torch.randn(batch, model.generator.z_dim, device=device), cond)
            real_aug = augment(real)
            real_score, classes, _ = disc(real_aug, cond)
            real_accuracy = classes.detach().argmax(1).eq(labels).float().mean()
            fake_score, _, _ = disc(augment(fake), cond)
            wrong_score, _, _ = disc(real_aug, cond.roll(1, 0))
            mask = torch.tensor([captions[i] != captions[i-1] for i in range(batch)], device=device)
            loss_d = discriminator_loss(real_score, fake_score, wrong_score, labels, classes, mask)
        scaler.scale(loss_d).backward()
        scaler.step(opt_d)
        disc.eval().requires_grad_(False)
        opt_g.zero_grad(set_to_none=True)
        with torch.autocast(device.type, enabled=amp):
            cond = model.text_encoder(tokens)
            fake = model.generator(torch.randn(batch, model.generator.z_dim, device=device), cond)
            score, classes, _ = disc(augment(fake), cond)
            # Ground text in real visual features as well as adversarial feedback.
            with torch.no_grad():
                _, _, real_features = disc(real, cond.detach())
            alignment = 1 - F.cosine_similarity(disc.project(cond), real_features.detach()).mean()
            loss_g = generator_loss(score, classes, labels) + .2 * alignment
        scaler.scale(loss_g).backward()
        scaler.unscale_(opt_g)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
        scaler.step(opt_g)
        scaler.update()
        with torch.no_grad():
            for avg, value in zip(ema.parameters(), model.parameters()):
                avg.lerp_(value, .001 if step > 100 else .1)
            for avg, value in zip(ema.buffers(), model.buffers()):
                avg.copy_(value)
        if step % args.log_every == 0 or step == args.steps:
            record = dict(step=step, loss_g=float(loss_g.detach()), loss_d=float(loss_d.detach()),
                          discriminator_gap=float((real_score.mean()-fake_score.mean()).detach()),
                          real_breed_accuracy=float(real_accuracy),
                          fake_spatial_std=float(fake.detach().float().std(dim=(2,3)).mean()),
                          elapsed_seconds=time.perf_counter()-began)
            if not np.isfinite(record['loss_g']+record['loss_d']):
                raise RuntimeError('Nonfinite loss; refusing to save invalid training state')
            print(json.dumps(record), flush=True)
            with (out/'training.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(record)+'\n')
        if step % args.save_every == 0 or step == args.steps:
            state = dict(format_version=2, step=step, base=args.base, seed=args.seed,
                         model=model.state_dict(), ema=ema.state_dict(), discriminator=disc.state_dict(),
                         optimizer_g=opt_g.state_dict(), optimizer_d=opt_d.state_dict(), scaler=scaler.state_dict(),
                         manifest_sha256=manifest_hash, breeds=dataset.breeds, parameters=counts,
                         discriminator_pooling="sum",
                         python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                         torch_rng=torch.get_rng_state(), sampler_rng=sampler_rng.get_state(),
                         cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None,
                         training_args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
            save_checkpoint(out/'latest.pt', state)
            with torch.inference_mode():
                save_image(ema(fixed_tokens, fixed_z).float().cpu(), out/'samples.png', nrow=4, normalize=True, value_range=(-1,1))
    return out/'latest.pt'


def main():
    p = argparse.ArgumentParser(description='Train all model weights from random initialization')
    p.add_argument('--manifest', type=Path, default=Path('data/curated/manifest.jsonl'))
    p.add_argument('--output', type=Path, default=Path('checkpoints'))
    p.add_argument('--steps', type=int, default=100000)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--base', type=int, default=32)
    p.add_argument('--lr', type=float, default=.0002)
    p.add_argument('--seed', type=int, default=650)
    p.add_argument('--device', default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--resume', type=Path)
    p.add_argument('--log-every', type=int, default=100)
    p.add_argument('--save-every', type=int, default=1000)
    args = p.parse_args()
    train(args)


if __name__ == '__main__':
    main()
