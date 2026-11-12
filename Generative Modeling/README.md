# DogGen: portable text-to-image GAN from scratch

A small, executable starting point for GNR650 Assignment 3. The domain is dog
photographs. Dataset acquisition and curation, random-initialized training,
checkpointed inference, parameter verification and held-out evaluation are implemented.
Read [docs/STATUS.md](docs/STATUS.md) before treating this as a finished submission.
The included step-2,510 checkpoint is a **development checkpoint with poor image
quality**, not a ready-to-submit model. The actual curated dataset is complete;
converged training and evaluation-GPU compliance are not established.

## Windows training (PowerShell)

Install 64-bit Python 3.11 and a recent NVIDIA driver supporting CUDA 13.0.
Run from this directory:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
.\.venv\Scripts\python.exe -m doggen.curate
.\.venv\Scripts\python.exe -m doggen.verify
.\.venv\Scripts\python.exe -m doggen.train --device cuda --amp --batch-size 32 --steps 100000
```

The setup creates `.venv`; activation is optional. Never copy the Mac `.venv` to
Windows: virtual environments contain platform-specific binaries and absolute paths.
Copy the source, `data/curated` and `checkpoints`, then recreate the environment.
Use `setup.ps1 -CPU` on a machine without NVIDIA. CPU training is much slower.
Windows data workers default to zero to avoid multiprocessing surprises; use
`--workers 2` if helpful. Lower `--batch-size` to 16 or 8 if VRAM is insufficient.

Resume training (steps means the **total** target step number):

```powershell
.\.venv\Scripts\python.exe -m doggen.train --device cuda --amp --batch-size 32 --steps 150000 --resume checkpoints/latest.pt
```

Keep `--base`, manifest and optimizer/training settings consistent on resume.
Optimizer, EMA, scaler, seed and RNG states are saved. The shuffled loader restarts
at a new iterator on resume; this is a continuation, not bitwise replay of an
interrupted minibatch stream. MPS RNG replay is also not guaranteed.

## Linux evaluation contract

```bash
bash setup.bash
python test.py --test-prompts /absolute/prompts.csv --output-dir /absolute/generated
```

`setup.bash` creates `.venv` and installs pinned dependencies; on NVIDIA machines it
uses PyTorch's official CUDA 13.0 wheels. `test.py` automatically re-executes in the
project venv because a shell script cannot activate its parent shell. It loads
`checkpoints/latest.pt` once and generates **one image per forward pass**. It does
not train, download model weights or fall back to random images. Missing checkpoints
fail with an actionable error. Only load checkpoints you trust (PyTorch training
checkpoint files contain pickled state).

Input is UTF-8 CSV with `id,prompt` headers. Quote prompts containing commas:

```csv
id,prompt
5,"A photograph of a golden retriever dog, in a bright green scene."
```

Output: `image_5.png`, 64x64 RGB. IDs must be unique safe filename components.
Arguments also include `--checkpoint`, `--device cpu|cuda|mps|auto` and `--seed`.
Paths may contain spaces and need not be relative to the repository.

## Dataset

```bash
python -m doggen.curate
python -m doggen.verify
```

Use the venv Python above, or activate it first. Acquisition uses a pinned Stanford
Dogs mirror (20,580 **candidates**, not an assumed final count). It verifies SHA256
for archive and metadata, retries with backoff, resumes interrupted downloads and
reuses valid cached files. Originals remain in `data/raw`; archive members are read
in memory, never extracted to arbitrary filesystem paths. Allow about 1 GB disk.

Curation validates decoding and size/aspect ratio, filters nearly uniform images,
requires source dog/breed annotations, and rejects source-byte duplicates,
perceptual-hash near duplicates (63 AC DCT bits, Hamming radius 4, BK tree) and
identical processed pixels. A deterministic sorted pass retains one representative
before any split. Padding preserves the full photograph at 64x64. Source annotations
supply breed and main-dog framing/position; measured scene palette and brightness
supply additional grounded caption details. **No pretrained captioner is used.**
These are automatic, limited descriptions: color describes the entire scene, not
necessarily fur; annotation boxes do not prove pose or scene identity. They can
support breed, framing, position, color and brightness prompts. Unsupported actions
such as “jumping over a bench” will not become reliable from these captions alone.

Outputs:

- `data/curated/manifest.jsonl`: one usable image per line, all metadata and paths.
- `data/curated/images/*.png`: actual preprocessed training images.
- `data/curated/rejections.jsonl`: source member and reason for every rejection.
- `data/curated/summary.json`: measured candidate/accepted/rejected and split counts.
- `reports/dataset-audit.json`: independent full-file integrity and uniqueness check.

Splits are stratified by breed, approximately 80/10/10 train/validation/test, sorted
by SHA256(seed + image ID). Original benchmark splits are deliberately replaced for
this generative experiment. No accepted pair can be within the configured perceptual
radius, including across splits. Perceptual hashes are a heuristic, not proof that
all possible edits of a photograph have been found. Avoid relaxing the threshold
just to meet the count. The CLI fails if fewer than 10,000 usable rows remain.
Rerunning regenerates deterministic metadata, verifies/reuses processed images and
repairs corrupted ones. It does not redownload validated source files.

Use is restricted to the assignment's non-commercial research/educational context.
Source photographers retain rights. A mirror's MIT metadata tag does **not** grant
a blanket MIT license to the photographs. See [docs/DATA.md](docs/DATA.md).

## Architecture and training

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) explains the selection and losses.
All weights are random initialized; a byte tokenizer and learned GRU with masked mean/final-state pooling handle novel
strings without pretrained text embeddings. A residual generator applies learned
text feature modulation at every resolution. The projection discriminator receives
matched and mismatched captions and an auxiliary breed head. Hinge losses, spectral
normalization, differentiable color/translation augmentation, gradient clipping and
EMA provide a compact training baseline.

```bash
python -m doggen.parameters
python -m doggen.train --device cuda --amp --steps 100000 --batch-size 32
```

Exact counts at default `--base 32`:

| Component | Parameters |
|---|---:|
| Generator | 1,756,259 |
| Text encoder | 91,264 |
| Deployed total | 1,847,523 |
| Discriminator, including conditioning projection and breed head | 1,279,705 |
| Entire training architecture | 3,127,228 |

Counts include every parameter even when EMA is frozen for deployment. EMA is a
copy, not an additional active generative network. `--base` changes model width;
programmatic counting rejects architectures reaching 50M. Resolution stays 64x64.
Checkpoints save full training state, data manifest hash, vocabulary convention,
architecture width, breed order and training arguments. Nothing is loaded from a
pretrained generative or text model. Do not expect useful images after a smoke run.

## Evaluation and tests

```bash
python -m doggen.benchmark --device cuda --runs 100
python -m pip install -r requirements-eval.txt
python -m doggen.evaluate --device cuda --split validation
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Benchmark runs batch one with synchronization and includes text tokenization,
forward pass, transfer to CPU and PNG encoding. Reports mean/p50/p95/max, cold
first-image time, load time, hardware and peak allocated VRAM. A warmed maximum
<=0.8 seconds is the development target. Disk I/O and checkpoint loading are outside
the per-image timer and explicitly reported/excluded. Only a target-machine run
can establish compliance; local CPU/MPS timings do not prove RTX 5060 performance.

FID and IS use pretrained Inception; CLIP Score uses pretrained CLIP **only in the
separate evaluation command**. These weights are never available to generation or
training. First evaluation needs internet to cache them. Use validation for model
selection, freeze choices, then test once. Report exact count and seed. A low-sample
FID (fewer than 2049 images for 2048-dimensional features) is explicitly flagged
and should not be used as a reliable final quality claim. Test and validation are
about 2k images each, so report sample-size limitations even for the whole split.
`--limit 100` is a diagnostic, not a final metric. No fabricated scores are provided.

## Files for humans and agents

- `doggen/curate.py`: pinned download, validation, captions, deduplication, splits.
- `doggen/text.py`, `model.py`, `losses.py`: tokenizer, architecture and objectives.
- `doggen/train.py`, `data.py`: training and manifest consumer.
- `doggen/inference.py`, `test.py`: single-prompt deployment and assignment CLI.
- `doggen/verify.py`, `parameters.py`, `benchmark.py`, `evaluate.py`: audits/metrics.
- `tests/test_core.py`: gradient/conditioning, hash lookup, tokenizer and CLI checks.
- `docs/STATUS.md`: actual run evidence and outstanding submission requirements.

The assignment also requires a CVPR 2027 paper (main body <=8 pages). This repository
contains methodology and evidence for that report, not a claim that a finalized
CVPR-formatted paper has been submitted. Preserve honest separation between
implemented functionality, smoke-run evidence and trained-model quality.
