"""Decode every image once into a uint8 .npy (N, C, S, S) under cache/ (git-ignored).

25k images at 128px RGB = 1.2 GB -> fits in GPU memory, so augmentation runs fully on the GPU and
training is never bottlenecked by PNG decoding.
"""
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

from datasets.io import LOADER_VERSION, load_image


def _key(paths, size, channels):
    h = hashlib.sha1(f"{LOADER_VERSION}|{size}|{channels}|{len(paths)}".encode())
    for p in paths:
        try:
            st = os.path.getsize(p)
        except OSError:
            st = -1
        h.update(f"{p}|{st}\n".encode())
    return h.hexdigest()[:12]


def _work(args):
    i, path, size, channels = args
    try:
        return i, load_image(path, size, channels), None
    except Exception as e:  # noqa: BLE001 - any decode error -> mark sample invalid
        return i, None, f"{type(e).__name__}: {e}"


def build_cache(paths, size, channels, cache_dir, tag, num_workers=8, verbose=True):
    """Returns (array[N,C,S,S] uint8 (memmap), ok_mask[N] bool)."""
    os.makedirs(cache_dir, exist_ok=True)
    key = _key(paths, size, channels)
    npy = os.path.join(cache_dir, f"{tag}_{size}px_c{channels}_{key}.npy")
    meta = npy[:-4] + ".json"
    if os.path.isfile(npy) and os.path.isfile(meta):
        with open(meta) as f:
            m = json.load(f)
        arr = np.load(npy, mmap_mode="r")
        if arr.shape[0] == len(paths):
            if verbose:
                print(f"[cache] hit {os.path.basename(npy)} ({arr.shape[0]} imgs, {len(m['failed'])} failed)")
            ok = np.ones(len(paths), bool)
            ok[[f[0] for f in m["failed"]]] = False
            return arr, ok
    if verbose:
        print(f"[cache] building {tag}: {len(paths)} images -> {size}x{size}x{channels} "
              f"({len(paths) * channels * size * size / 1e9:.2f} GB) with {num_workers} workers")
    tmp = npy + ".partial.npy"
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(len(paths), channels, size, size))
    failed = []
    jobs = [(i, p, size, channels) for i, p in enumerate(paths)]
    t0 = time.time()

    def consume(it):
        for n, (i, a, err) in enumerate(it):
            if a is None:
                failed.append((i, paths[i], err))
                out[i] = 0
            else:
                out[i] = a
            if verbose and (n + 1) % 2000 == 0:
                r = (n + 1) / (time.time() - t0)
                print(f"[cache]   {n + 1}/{len(paths)}  {r:.0f} img/s  eta {(len(paths) - n - 1) / r:.0f}s")
                sys.stdout.flush()

    done = False
    if num_workers and num_workers > 0 and len(paths) > 64:
        try:
            ctx = mp.get_context("fork") if sys.platform.startswith("linux") else mp.get_context()
            with ctx.Pool(num_workers) as pool:
                consume(pool.imap(_work, jobs, chunksize=32))
            done = True
        except Exception as e:  # noqa: BLE001
            print(f"[cache] worker pool failed ({e}); falling back to a single process")
            failed.clear()
    if not done:
        consume(map(_work, jobs))
    out.flush()
    del out
    os.replace(tmp, npy)
    with open(meta, "w") as f:
        json.dump({"paths": paths, "failed": failed, "size": size, "channels": channels,
                   "loader": LOADER_VERSION}, f)
    if failed and verbose:
        print(f"[cache] WARNING {len(failed)} unreadable images skipped, e.g. {failed[:3]}")
    if verbose:
        print(f"[cache] done in {time.time() - t0:.0f}s -> {npy}")
    arr = np.load(npy, mmap_mode="r")
    ok = np.ones(len(paths), bool)
    ok[[f[0] for f in failed]] = False
    return arr, ok
