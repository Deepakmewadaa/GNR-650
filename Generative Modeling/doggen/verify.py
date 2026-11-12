"""Independent full-file dataset audit. Fails closed on any invalid requirement."""
import argparse
from collections import Counter
from pathlib import Path
import hashlib
import numpy as np
from PIL import Image
from .common import read_manifest, sha256, write_json
from .curate import HashTree


def audit(manifest, minimum=10000, radius=4):
    rows = read_manifest(manifest)
    if len(rows) < minimum:
        raise ValueError(f'{len(rows)} < {minimum}')
    tree, seen_ids, seen_hashes, seen_pixels = HashTree(), set(), set(), set()
    for row in rows:
        ident = row['id']
        assert ident not in seen_ids, f'duplicate id {ident}'
        assert row['content_hash'] not in seen_hashes, f'duplicate content {ident}'
        path = Path(manifest).parent/row['image_path']
        assert sha256(path) == row['processed_hash'], f'file hash {ident}'
        with Image.open(path) as im:
            im.load()
            assert im.size == (64,64) and im.mode=='RGB'
            pixel = hashlib.sha256(im.tobytes()).hexdigest()
        assert pixel == row['pixel_hash'] and pixel not in seen_pixels
        value = int(row['phash'],16)
        assert tree.match(value,radius) is None, f'near duplicate {ident}'
        tree.add(value,ident)
        assert row['caption'].startswith('A photograph of a ') and row['breed'] in row['caption']
        assert row['split'] in {'train','validation','test'}
        assert row['source'] and row['acquisition_url'] and row['license']
        seen_ids.add(ident)
        seen_hashes.add(row['content_hash'])
        seen_pixels.add(pixel)
    result = dict(verified_images=len(rows), splits=dict(Counter(r['split'] for r in rows)),
                  manifest_sha256=sha256(manifest), phash_radius=radius, passed=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=Path('data/curated/manifest.jsonl'))
    p.add_argument('--minimum',type=int,default=10000)
    args=p.parse_args()
    result=audit(args.manifest,args.minimum)
    write_json('reports/dataset-audit.json',result)
    print(result)


if __name__=='__main__':
    main()
