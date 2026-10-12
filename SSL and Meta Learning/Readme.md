# GNR650 Assignment 2 — SSL + noisy linear probing + unknown-class detection

From-scratch PyTorch pipeline (no pretrained weights, no torchvision / timm / HF, no SSL/OOD libraries):

1. **SimCLR** self-supervised pre-training of a ResNet-18-style encoder on `unlabelled_data` (DINO optional).
2. **Linear probe** (one `nn.Linear`) on the frozen encoder, trained on the noisy `labelled_data` with
   noise-robust losses / sample cleaning. Compared with an **end-to-end supervised baseline**.
3. **Unknown detection**: an OOD score + threshold chosen by a leave-class-out protocol. Rejected samples get
   label **K**.

Everything is packaged in one file, `checkpoints/final.pt`, used by `test.py`.

---

## Grader commands (exactly as in the assignment)

```bash
bash setup.bash
python test.py --test-csv /abs/path/test.csv --output-path /abs/path/output_dir
python grade.py --output-path /abs/path/output_dir      # provided by the instructors
```

- `test.py` writes `<output_dir>/predictions.csv` with header `id,file_path,label`, in the same row order as the
  input CSV. Labels 0..K-1 are known classes; label K means Unknown.
- **No venv activation needed.** `setup.bash` installs into `./.venv`, and `test.py` re-executes itself with
  that interpreter automatically. If a venv can't be created, `setup.bash` installs into `./.pydeps`, which
  `test.py` adds to `sys.path`.
- `test.py` finds the checkpoint relative to its own location, so the working directory does not matter.
- It runs on GPU (AMP) when available, otherwise CPU. Unreadable or missing images are logged and predicted
  as K.

---

## TEAMMATE QUICKSTART (training on the GPU machine)

```bash
git clone <repo-url> && cd <repo>
bash setup.bash                                # creates .venv with torch 2.12.1 (+cu130 on CUDA 13 drivers)
# put the dataset at ./data/  ->  data/labelled_data/class_XX/*.png  and  data/unlabelled_data/*.png
#   (./dataset/ also works, folder names are matched flexibly; or pass --data-root /path/to/dataset)
.venv/bin/python scripts/inspect_data.py       # stats + sample grids + recommended preset
.venv/bin/python scripts/doctor.py             # GPU/CUDA check, 2-step dry run, throughput, epochs that fit
bash run_all.sh --config base --budget 6       # the whole pipeline (use --config small if doctor says so)
.venv/bin/python scripts/run_ablations.py --ablation-budget-hours 1.5   # optional, for the report
.venv/bin/python scripts/make_report.py        # refresh docs/report.md with the new ablations
git add checkpoints/final.pt results/ docs/
git commit -m "Trained model + results"
git push
```

- `run_all.sh` runs these stages: `inspect → doctor → ssl → probe → baseline → ood → export → eval → report`.
  - It is **resumable**: re-running the same command skips finished stages, and `train_ssl.py` resumes from
    `runs/main/ssl/last.pt` after an interruption.
  - `--skip baseline` / `--only ood,export` select stages; `--force` redoes them; `--set key=value` overrides
    any config entry (e.g. `--set ssl.batch_size=256`).
- **Leaderboard tip:** if there is spare GPU time *before* the main run, run
  `.venv/bin/python scripts/run_ablations.py --variants ref temp_0.5 stem_medium crop_0.2 --ablation-budget-hours 2`.
  If a variant clearly beats `ref` on kNN/probe accuracy, launch the main run with that override, e.g.
  `bash run_all.sh --set model.stem=medium`.
- Check the `fit_ood` log. The per-score "known fraction of the unseen pool" estimates should roughly agree. The
  threshold is chosen to maximise the estimated (K+1)-accuracy on a never-seen part of the unlabelled pool (see
  `docs/decisions.md`).
- If it fails, it stops and names the log. **Send back:**
  - `logs/<stage>.log`;
  - `results/main/doctor.json`;
  - `runs/main/ssl/log.csv` (if the SSL stage started);
  - the output of `nvidia-smi` and `.venv/bin/python -c "import torch;print(torch.__version__, torch.version.cuda)"`.
- Expected artefacts after a successful run:
  - `checkpoints/final.pt` (~45 MB; must be < 100 MB);
  - `results/main/*.json|md|png`;
  - `docs/report.md`;
  - `docs/experiment_log.md`.

## Repository layout

| path | content |
|---|---|
| `configs/` | `base` (default), `small` (weak GPUs), `dino`, `tiny` (CPU smoke tests only) |
| `datasets/` | folder discovery, robust image reading (shared with `test.py`), uint8 cache, stratified splits |
| `models/` | ResNet encoder, heads, hook-based MAC/FLOP counter, the packaged `FinalModel` |
| `ssl_methods/` | GPU augmentations, SimCLR, DINO, the training engine |
| `probe/` | feature banks, losses, label cleaning, linear head, supervised baseline |
| `ood/` | OOD scores, metrics, leave-class-out protocol, plots |
| `utils/` | config, seeding, logging, checkpoints, optimisers (LARS), diagnostics |
| `train_ssl.py` `train_probe.py` `train_supervised_baseline.py` `fit_ood.py` `export_final.py` | pipeline stages |
| `test.py` | grader entry point |
| `scripts/` | `inspect_data`, `doctor`, `run_ablations`, `local_eval`, `make_report`, `make_synthetic_data` |
| `tests/` | pytest unit tests |
| `docs/` | `decisions.md` (all defaults + rationale), `report.md` (generated), `experiment_log.md` (auto) |

All stages accept `--help`, `--config`, `--set key=value`, `--run-name`, `--data-root`, `--out-root` and
`--time-budget-hours`.

## Verifying without a GPU (synthetic data)

```bash
python scripts/make_synthetic_data.py --out /tmp/syn
bash run_all.sh --config tiny --data-root /tmp/syn/dataset --run-name syn --device cpu --skip doctor,inspect
pip install pytest && python -m pytest -q tests
```

Do this in a scratch copy of the repo: it writes `checkpoints/final.pt`, which must never be a synthetic model
in the real repo. Alternatively, pass `--out-root /tmp/synrun` to each stage instead of using `run_all.sh`.

Read `docs/decisions.md` for every default and why, and `SOURCES.md` for references.
