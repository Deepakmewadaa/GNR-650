"""Pinned acquisition, validation, captioning and deterministic curation.

No pretrained model is used. Captions combine source breed labels with measured
scene attributes; they do not infer pose or identify background objects.
"""
import argparse
import ast
import csv
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile
import time
import warnings

import numpy as np
from PIL import Image, ImageOps, ImageStat
import requests
from scipy.fft import dctn
from tqdm import tqdm
from .common import sha256, write_json

REVISION = '00c2ba455d1c85a01f14c6c5e1ec8524eeb33a1f'
URL = f'https://huggingface.co/datasets/Alanox/stanford-dogs/resolve/{REVISION}/images.tar.gz'
METADATA_URL = f'https://huggingface.co/datasets/Alanox/stanford-dogs/resolve/{REVISION}/metadata.csv'
METADATA_SHA = '788ba75b66065a990ff7dd9fd1525cd153e2e94ee76d2ea32dc651b084d60297'
ARCHIVE_SHA = '572e8ab5097aac3f7cbf18fd7d40b752169ba188e89b12b5b9bf682157d52333'
SOURCE = 'http://vision.stanford.edu/aditya86/ImageNetDogs/'
USAGE = ('Stanford Dogs / ImageNet: non-commercial research and educational use; '
         'individual photographer rights retained; no blanket MIT image license asserted')
PALETTE = {'black': (25,25,25), 'white': (235,235,235), 'gray': (130,130,130),
           'brown': (115,75,45), 'tan': (195,155,105), 'green': (65,115,45),
           'blue': (65,115,175), 'red': (160,50,50), 'yellow': (205,190,65)}


def acquire(path, url=URL, expected_sha=ARCHIVE_SHA):
    """Stream with HTTP range resume, timeout, backoff and pinned SHA verification."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and sha256(path) == expected_sha:
        return
    part = path.with_suffix('.partial')
    for attempt in range(5):
        try:
            offset = part.stat().st_size if part.exists() else 0
            with requests.get(url, headers={'Range': f'bytes={offset}-'} if offset else {},
                              stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                resume = offset and response.status_code == 206
                with part.open('ab' if resume else 'wb') as out:
                    for chunk in response.iter_content(1024 * 1024):
                        out.write(chunk)
            if sha256(part) != expected_sha:
                part.unlink()
                raise ValueError('Archive SHA256 mismatch')
            part.replace(path)
            return
        except (requests.RequestException, ValueError):
            if attempt == 4:
                raise
            time.sleep(min(2 ** attempt, 16))


def phash(image):
    small = np.asarray(image.convert('L').resize((32, 32)), dtype=np.float32)
    coeff = dctn(small, norm='ortho')[:8, :8].reshape(-1)[1:]
    bits = coeff > np.median(coeff)
    return sum(int(v) << i for i, v in enumerate(bits))


class HashTree:
    """Exact Hamming-radius lookup using a BK tree (no quadratic all-pairs pass)."""
    def __init__(self):
        self.root = None

    def match(self, value, radius):
        pending = [self.root] if self.root else []
        while pending:
            node = pending.pop()
            dist = (value ^ node[0]).bit_count()
            if dist <= radius:
                return node[1]
            pending.extend(child for edge, child in node[2].items()
                           if dist - radius <= edge <= dist + radius)
        return None

    def add(self, value, ident):
        if self.root is None:
            self.root = [value, ident, {}]
            return
        node = self.root
        while True:
            dist = (value ^ node[0]).bit_count()
            if dist not in node[2]:
                node[2][dist] = [value, ident, {}]
                return
            node = node[2][dist]


def caption(image, breed, boxes):
    rgb = np.asarray(image.resize((32, 32)), dtype=np.float32).reshape(-1, 3)
    names = list(PALETTE)
    centers = np.array(list(PALETTE.values()))
    assignments = ((rgb[:, None] - centers) ** 2).sum(-1).argmin(-1)
    most = np.bincount(assignments, minlength=len(names)).argsort()[::-1][:2]
    light = float(rgb.mean())
    illumination = 'bright' if light > 170 else 'dark' if light < 80 else 'moderately lit'
    colors = f'{names[most[0]]} and {names[most[1]]}'
    box = max(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
    x = (box[0]+box[2])/2/image.width
    position = 'left' if x < .4 else 'right' if x > .6 else 'center'
    framing = 'close framing' if (box[2]-box[0])*(box[3]-box[1])/(image.width*image.height) > .45 else 'wider framing'
    return (f'A photograph of a {breed} dog near the {position} of the image, with {framing}, '
            f'in a {illumination} scene with predominantly {colors} colors.'), {'scene_colors': [names[i] for i in most],
                                                      'scene_brightness': illumination}


def curate(archive, output, minimum=10000, seed=650, radius=4):
    output = Path(output)
    (output / 'images').mkdir(parents=True, exist_ok=True)
    metadata_path = Path(archive).parent / "metadata.csv"
    if sha256(metadata_path) != METADATA_SHA:
        raise ValueError("Metadata hash mismatch")
    with metadata_path.open(newline="", encoding="utf-8") as f:
        metadata = {r["name"]: r for r in csv.DictReader(f)}
    rows, rejected = [], []
    exact, pixels, tree = {}, {}, HashTree()
    warnings.simplefilter('error', Image.DecompressionBombWarning)
    with tarfile.open(archive, 'r:gz') as tar:
        members = sorted((m for m in tar.getmembers() if m.isfile() and
                          m.name.lower().endswith(('.jpg', '.jpeg', '.png'))), key=lambda m: m.name)
        for member in tqdm(members, desc='Curating'):
            reason = None
            try:
                # No archive member is extracted to disk or interpreted as a local path.
                source_name = Path(member.name).name
                if source_name not in metadata:
                    raise ValueError('missing source breed annotation')
                annotation = metadata[source_name]
                breed = annotation['target'].replace('_', ' ').lower()
                boxes = ast.literal_eval(annotation['annotations'])
                if not boxes or any(len(b) != 4 for b in boxes):
                    raise ValueError('invalid dog annotation')
                raw = tar.extractfile(member).read()
                content = hashlib.sha256(raw).hexdigest()
                if content in exact:
                    reason = 'exact duplicate:' + exact[content]
                if reason is None:
                    with Image.open(io.BytesIO(raw)) as probe:
                        probe.verify()
                    with Image.open(io.BytesIO(raw)) as src:
                        image = ImageOps.exif_transpose(src).convert('RGB')
                    width, height = image.size
                    if min(width, height) < 64 or max(width, height) / min(width, height) > 4:
                        raise ValueError('size/aspect filter')
                    if sum(ImageStat.Stat(image.resize((32, 32))).stddev) < 12:
                        raise ValueError('nearly uniform image')
                    perceptual = phash(image)
                    duplicate = tree.match(perceptual, radius)
                    if duplicate:
                        reason = 'near duplicate:' + duplicate
                if reason is None:
                    processed = ImageOps.pad(image, (64, 64), method=Image.Resampling.LANCZOS,
                                             color=tuple(int(v) for v in np.asarray(image.resize((1,1)))[0,0]))
                    pixel_hash = hashlib.sha256(processed.tobytes()).hexdigest()
                    if pixel_hash in pixels:
                        reason = 'processed duplicate:' + pixels[pixel_hash]
                if reason:
                    rejected.append({'source_member': member.name, 'reason': reason})
                    continue
                ident = content[:24]
                path = 'images/' + ident + '.png'
                # Deterministic output; validate existing file on restart, repair if needed.
                dest = output / path
                if not dest.exists():
                    processed.save(dest)
                else:
                    try:
                        with Image.open(dest) as saved:
                            valid = saved.size == (64,64) and saved.convert('RGB').tobytes() == processed.tobytes()
                    except OSError:
                        valid = False
                    if not valid:
                        processed.save(dest)
                text, attributes = caption(image, breed, boxes)
                row = dict(id=ident, source=SOURCE, acquisition_url=URL,
                           source_member=member.name, source_revision=REVISION,
                           metadata_url=METADATA_URL, metadata_sha256=METADATA_SHA, bounding_boxes=boxes,
                           source_image_id=Path(member.name).stem, image_path=path,
                           original_width=width, original_height=height, width=64, height=64,
                           caption=text, caption_method='breed-label+measured-scene-palette-v1',
                           breed=breed, license=USAGE, content_hash=content,
                           pixel_hash=pixel_hash, processed_hash=sha256(dest), phash=f'{perceptual:016x}',
                           **attributes)
                rows.append(row)
                exact[content], pixels[pixel_hash] = ident, ident
                tree.add(perceptual, ident)
            except (OSError, ValueError, SyntaxError, Image.DecompressionBombWarning,
                    Image.DecompressionBombError) as exc:
                rejected.append({'source_member': member.name, 'reason': str(exc)})
    # Hash ordering within breed gives stable stratified splits, independent of OS iteration.
    groups = defaultdict(list)
    for row in rows:
        groups[row['breed']].append(row)
    for group in groups.values():
        group.sort(key=lambda r: hashlib.sha256(f"{seed}:{r['id']}".encode()).digest())
        nval = max(1, round(len(group) * .1))
        for i, row in enumerate(group):
            row['split'] = 'validation' if i < nval else 'test' if i < 2*nval else 'train'
    rows.sort(key=lambda r: r['id'])
    for name, records in [('manifest.jsonl', rows), ('rejections.jsonl', rejected)]:
        tmp = output / (name + '.tmp')
        with tmp.open('w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record) + '\n')
        tmp.replace(output / name)
    summary = dict(candidates=len(members), accepted=len(rows), rejected=len(rejected),
                   splits=dict(Counter(r['split'] for r in rows)), breeds=len(groups),
                   minimum_required=minimum, meets_minimum=len(rows) >= minimum,
                   archive_sha256=sha256(archive), manifest_sha256=sha256(output/'manifest.jsonl'),
                   seed=seed, phash_radius=radius, resolution=64)
    write_json(output/'summary.json', summary)
    print(json.dumps(summary, indent=2))
    if len(rows) < minimum:
        raise RuntimeError(f'Only {len(rows)} usable images; required {minimum}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, default=Path('data/raw/images.tar.gz'))
    p.add_argument('--output', type=Path, default=Path('data/curated'))
    p.add_argument('--minimum', type=int, default=10000)
    p.add_argument('--seed', type=int, default=650)
    p.add_argument('--phash-radius', type=int, default=4)
    p.add_argument('--skip-download', action='store_true')
    args = p.parse_args()
    if not args.skip_download:
        acquire(args.archive)
        acquire(args.archive.parent / 'metadata.csv', METADATA_URL, METADATA_SHA)
    if sha256(args.archive) != ARCHIVE_SHA:
        raise ValueError('Input is not the pinned source archive')
    curate(args.archive, args.output, args.minimum, args.seed, args.phash_radius)


if __name__ == '__main__':
    main()
