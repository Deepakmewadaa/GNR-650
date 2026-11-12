"""Assignment entrypoint: python test.py --test-prompts ABS.csv --output-dir ABS."""
import argparse
import csv
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent


def main():
    # setup.bash cannot activate its parent's shell. Re-exec the project environment.
    local_python = ROOT / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if local_python.exists() and Path(sys.prefix).resolve() != (ROOT/'.venv').resolve():
        os.execv(str(local_python), [str(local_python), str(Path(__file__).resolve()), *sys.argv[1:]])
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--test-prompts', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, default=ROOT/'checkpoints/latest.pt')
    p.add_argument('--device', default='auto')
    p.add_argument('--seed', type=int, default=650)
    args = p.parse_args()
    with args.test_prompts.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        if not reader.fieldnames or not {'id', 'prompt'}.issubset(reader.fieldnames):
            p.error('CSV must contain id,prompt columns')
        rows = list(reader)
    ids = set()
    for row in rows:
        ident = row['id'].strip()
        if not re.fullmatch(r'[A-Za-z0-9_-]+', ident) or ident in ids:
            p.error('Each id must be unique and contain only letters, digits, underscores or hyphens')
        if not row.get('prompt', '').strip():
            p.error('Every row requires a nonempty prompt')
        ids.add(ident)
    from doggen.inference import Predictor
    predictor = Predictor(args.checkpoint, args.device)
    predictor.warmup()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(rows):
        predictor.generate(row['prompt'], args.seed+i).save(args.output_dir/f"image_{row['id'].strip()}.png")
    print(f'Generated {len(rows)} images using checkpoint step {predictor.steps} on {predictor.device}')


if __name__ == '__main__':
    main()
