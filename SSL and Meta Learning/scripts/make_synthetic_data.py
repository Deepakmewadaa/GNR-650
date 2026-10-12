#!/usr/bin/env python
"""Create a tiny synthetic dataset in the SAME layout as the real one, for CPU smoke tests.

<out>/dataset/labelled_data/class_XX/*.png   (K known classes, ~10% label noise)
<out>/dataset/unlabelled_data/*.png          (known + unknown-class images)
<out>/test_images/*.png, <out>/test.csv (id,file_path), <out>/test_gt.csv (id,file_path,label; K = unknown)

Images are coloured/patterned shapes at varying small resolutions, saved in mixed PNG modes
(RGB, L, RGBA, P, 16-bit) to exercise the robust loader. NEVER point --out at the real data folders.
"""
import argparse
import csv
import os
import shutil
import sys

import numpy as np
from PIL import Image

SIZES = [24, 32, 40, 48, 64, 80]
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PALETTE = [(220, 40, 40), (40, 200, 60), (40, 70, 230), (230, 210, 40), (210, 40, 210), (40, 210, 210),
           (240, 140, 20), (240, 240, 240), (120, 60, 20), (90, 90, 90)]


def pattern(cls, size, rng):
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    f = rng.uniform(3, 6)
    ph = rng.uniform(0, 2 * np.pi)
    kind = cls % 8
    if kind == 0:
        m = np.sin(2 * np.pi * f * yy + ph) > 0
    elif kind == 1:
        m = np.sin(2 * np.pi * f * xx + ph) > 0
    elif kind == 2:
        m = (np.floor(xx * f) + np.floor(yy * f)) % 2 == 0
    elif kind == 3:
        cx, cy, r = rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7), rng.uniform(0.15, 0.3)
        m = (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2
    elif kind == 4:
        m = np.sin(2 * np.pi * f * (xx + yy) + ph) > 0
    elif kind == 5:
        m = np.sin(2 * np.pi * f * (xx - yy) + ph) > 0
    elif kind == 6:
        cx, cy = rng.uniform(0.4, 0.6), rng.uniform(0.4, 0.6)
        m = np.sin(2 * np.pi * f * np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) * 2 + ph) > 0
    else:
        w = rng.uniform(0.08, 0.15)
        m = (np.abs(xx - 0.5) < w) | (np.abs(yy - 0.5) < w)
    col = np.array(PALETTE[cls % len(PALETTE)], np.float32)
    bg = rng.uniform(0, 60, 3).astype(np.float32)
    img = np.where(m[..., None], col, bg) + rng.normal(0, 12, (size, size, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


def save(img, path, rng, mode_mix=True):
    im = Image.fromarray(np.ascontiguousarray(img))
    r = rng.rand() if mode_mix else 1.0
    if r < 0.08:
        im = im.convert("L")
    elif r < 0.14:
        im = im.convert("RGBA")
    elif r < 0.18:
        im = im.convert("P", palette=Image.ADAPTIVE)
    elif r < 0.21:
        a = np.asarray(im.convert("L"), np.uint16) * 257       # 16-bit greyscale PNG
        im = Image.fromarray(a)
    im.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output folder (outside the repo, or a git-ignored path)")
    ap.add_argument("--known", type=int, default=6)
    ap.add_argument("--unknown", type=int, default=2)
    ap.add_argument("--per-class", type=int, default=30)
    ap.add_argument("--unlabelled-per-class", type=int, default=50)
    ap.add_argument("--test-per-class", type=int, default=8)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-copy-labelled", action="store_true",
                    help="do NOT also place the labelled files in the unlabelled folder (the real data does)")
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    for forbidden in ("data", "dataset"):
        fp = os.path.join(REPO, forbidden)
        if out == fp or out.startswith(fp + os.sep):
            sys.exit(f"refusing to write into the real data folder {fp}")
    rng = np.random.RandomState(a.seed)
    K, U = a.known, a.unknown
    lab = os.path.join(out, "dataset", "labelled_data")
    unl = os.path.join(out, "dataset", "unlabelled_data")
    tst = os.path.join(out, "test_images")
    for d in [unl, tst] + [os.path.join(lab, f"class_{c:02d}") for c in range(K)]:
        os.makedirs(d, exist_ok=True)
    n = 0
    flipped = 0
    for c in range(K):
        for i in range(a.per_class):
            size = int(rng.choice(SIZES))
            img = pattern(c, size, rng)
            tgt = c
            if rng.rand() < a.noise:   # symmetric label noise
                tgt = int(rng.choice([k for k in range(K) if k != c]))
                flipped += 1
            lp = os.path.join(lab, f"class_{tgt:02d}", f"{n:05d}.png")
            save(img, lp, rng)
            if not a.no_copy_labelled:     # real data: labelled files are byte-copies inside unlabelled_data
                shutil.copyfile(lp, os.path.join(unl, f"{n:05d}.png"))
            n += 1
    for c in range(K + U):
        for i in range(a.unlabelled_per_class):
            size = int(rng.choice(SIZES))      # same distribution as labelled (real data: labelled ⊂ pool)
            save(pattern(c, size, rng), os.path.join(unl, f"{n:05d}.png"), rng)
            n += 1
    rows = []
    for c in range(K + U):
        for i in range(a.test_per_class):
            size = int(rng.choice([28, 32, 50, 64]))
            p = os.path.join(tst, f"test_{len(rows):05d}.png")
            save(pattern(c, size, rng), p, rng)
            rows.append((len(rows), p, c if c < K else K))
    order = rng.permutation(len(rows))
    rows = [(i, rows[j][1], rows[j][2]) for i, j in enumerate(order)]
    with open(os.path.join(out, "test.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "file_path"])
        w.writerows([(i, p) for i, p, _ in rows])
    with open(os.path.join(out, "test_gt.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "file_path", "label"])
        w.writerows(rows)
    print(f"synthetic dataset at {out}: K={K} (+{U} unknown), {K * a.per_class} labelled "
          f"({flipped} flipped labels), {(K + U) * a.unlabelled_per_class} unlabelled, {len(rows)} test")


if __name__ == "__main__":
    main()
