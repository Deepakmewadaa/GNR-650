# CLAUDE.md — working rules for this repo (GNR650 Assignment 2: SSL + noisy linear probe + OOD)

Re-read this file and PROGRESS.md after any context compaction.

## Workflow rules
- **NO git writes.** Never run git init/add/commit/remote/push/clone/checkout/reset/stash or anything that
  changes repo state or touches a remote. Read-only `git status`, `git diff`, `git log` are fine.
  At the end only *print* the git commands for the user.
- Do NOT run real training on this machine. The real dataset (in `./dataset/`, also searched in `./data/`)
  is used only for inspection and tiny smoke tests. Never write into the dataset folders.
- Synthetic data / synthetic runs / synthetic checkpoints live OUTSIDE the repo (scratchpad / temp dir) or
  in git-ignored folders. `checkpoints/final.pt` must NOT exist when work is handed over.
- Never create or overwrite `grade.py` (supplied by instructors). Our own scorer is `scripts/local_eval.py`.

## Hard constraints (violation = zero / heavy penalty)
1. No pretrained models/weights/external checkpoints. Everything from random init.
2. No high-level model repos (torchvision.models, timm, HF) and no SSL/OOD libraries (lightly, solo-learn,
   pytorch-ood, ...). No torchvision at all (no transforms either). No thop/fvcore/ptflops.
   Allowed: torch, numpy, pillow, pyyaml, tqdm, matplotlib (plots only), scikit-learn (metrics only, optional),
   pytest (dev only).
3. Only the provided dataset. No external data.
4. Hidden test set never used for any selection; all validation from provided data.
5. Grader runs exactly: `bash setup.bash`; `python test.py --test-csv <abs csv> --output-path <abs dir>`;
   `python grade.py --output-path <dir>`.
6. test.py: reads CSV (id,file_path); creates output dir; loads checkpoint relative to the script file;
   writes `<out>/predictions.csv` with header exactly `id,file_path,label`, same order/ids, no index column;
   label int in {0..K-1} known, K = Unknown (K from checkpoint). Robust to grayscale/RGBA/palette/16-bit,
   bad/missing files (warn + predict K), CPU fallback, AMP on GPU, worker fallback, clear error if the
   checkpoint is missing. Must not assume the venv is activated (re-execs with .venv python if present).
7. Inference must be a small fraction of 1 hour. Log total time and images/sec.
8. Single checkpoint `checkpoints/final.pt` < 100 MB, plain `torch.load(map_location='cpu')`-loadable,
   containing encoder, head, OOD stats, threshold, K, image size/channels/normalization, config, version/hash.
9. setup.bash non-interactive, venv (fallback: current env / --target dir), pinned requirements, CUDA 13.x.
10. No absolute machine-specific paths in committed code; nothing at test time depends on ./data or
    git-ignored files (except the committed checkpoint).

## Layout
configs/ datasets/ models/ ssl_methods/ probe/ ood/ utils/ scripts/ tests/ docs/ + top-level
train_ssl.py train_probe.py train_supervised_baseline.py fit_ood.py export_final.py test.py
setup.bash requirements.txt run_all.sh README.md PROGRESS.md SOURCES.md.
All pipeline outputs go under `--out-root` (default = repo root): runs/ (ignored), cache/ (ignored),
results/<run>/ (committed), docs/ (committed), checkpoints/final.pt (committed).
