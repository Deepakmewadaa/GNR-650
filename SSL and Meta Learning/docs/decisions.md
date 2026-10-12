# Design decisions (defaults chosen without tuning on real data)

Everything here was decided before any real training run. The code was only exercised on CPU with synthetic
data and on a handful of real images. Items marked **[check]** should be revisited once real results exist.

## Data
- **Location.** The real dataset ships as `dataset/labelled_data` + `dataset/unlabelled_data`, while the PDF
  writes `labelled data` / `unlabelled data` (with spaces). Discovery (`datasets/discover.py`) matches folder names
  case-insensitively after stripping spaces/underscores/hyphens (`unlabel` → unlabelled, `label` → labelled). It
  searches `--data-root`, `$GNR650_DATA`, `./data`, `./dataset` and one level below. Explicit
  `--labelled-dir/--unlabelled-dir` always win.
- **Observed facts (inspection only).** K = 50 (`class_00`…`class_49`), 1,300 labelled images (18–34 per class),
  25,000 unlabelled PNGs, all RGB, variable size (median 500×375, shorter side 48–2736 px). All 1,300 labelled
  files are byte-identical copies of files in the unlabelled folder. The unlabelled pool also contains classes
  that are not labelled (e.g. many dog breeds), so it is a natural source of "unknown"-like content.
- **Class index order.** If every class-folder name ends in an integer and those integers are unique, classes
  are ordered numerically; otherwise lexicographically (the torchvision ImageFolder rule). For zero-padded names
  the two rules agree, so `class_07` → label 7. **[check]** this matches the hidden test labels (it is the only
  sensible mapping).
- **Preprocessing (identical for training and `test.py`, one function in `datasets/io.py`).** Decode → EXIF
  orientation → channel normalisation → resize the shorter side to S (PIL bicubic, antialiased) → centre crop
  S×S → uint8. Special modes:
  - 16-bit/32-bit greyscale is rescaled to 0–255, not clipped.
  - Palette images are converted via RGBA when they carry transparency.
  - Transparency is composited onto white.
  - Greyscale is replicated to RGB, because the dataset is RGB (channels detected at runtime).
  - Truncated files are decoded partially with a warning.
- **Resolution.** S = native median shorter side capped at 128 (`data.max_image_size`) and rounded to a
  multiple of 8 → 128 for this dataset (96 for the `small` preset). Everything is cached once as uint8
  (25k × 3 × 128² ≈ 1.2 GB), kept on the GPU when it fits in 25 % of memory, and augmented on the GPU.
- **What the encoder may see (accuracy-critical).** The test images are almost certainly unseen: the unlabelled
  IDs are a 25k half of 0–49,998, and the labelled images sit inside that half. SSL encoders measurably memorise
  their training images, so every set used for calibration must be unseen too. `datasets/data.py: ssl_partition`
  therefore removes two groups from SSL:
  - the 1,300 labelled byte-duplicates (`data.exclude_labelled_from_ssl`);
  - a fixed random **calibration holdout** of 2,500 unlabelled images (`data.calib_holdout`, seed 12345).

  SSL trains on the remaining **21,200** images. The holdout is the background bank of the `knn_ratio` score and
  the target distribution for the threshold. Cost: ~15 % fewer SSL images. Benefit: unbiased probe/OOD
  calibration and access to the real unknown classes. **[check]** if training time allows, ablate
  `data.calib_holdout=0 data.exclude_labelled_from_ssl=false` (then `threshold_mode` falls back to `lco`).
- **Splits.** All splits are stratified per class with fixed numpy seeds (`datasets/splits.py`).
  - SSL monitoring uses a fixed 80/20 split of the labelled set (seed 0).
  - Probe and baseline use seed s ∈ {0,1,2} → the same 80/20 split for both arms.
  - Leave-class-out folds use seeds 0–4.

## Backbone
- ResNet-18-style CNN from scratch (`models/resnet.py`): BasicBlocks [2,2,2,2], widths [64,128,256,512], BN,
  GAP, 11.2 M parameters, ~0.6 GMACs/forward at 128 px. The stem is standard (7×7/2 + max-pool) above 64 px and
  small (3×3/1, no pool) at 64 px or below. The last BN of every residual branch is zero-initialised.
  - *Why:* a well-understood CNN that trains stably from scratch on ~25k images. It is small enough for
    hundreds of SimCLR epochs in ~4 GPU-hours and for test inference that is far below the 1-hour limit.
    ViTs need much more data/epochs to beat CNNs at this scale.
- Presets:
  - `base`: w = 64, 512-d, 128 px, SSL batch 512.
  - `small`: w = 32, 256-d, 96 px, batch 256, for GPUs under 12 GB or short budgets.
  - `tiny`: CPU smoke tests only.
- `embed_dim` sets the last-stage width, so it can be ablated independently of `width`.

## SSL (Component 1)
- **SimCLR** is the default; **DINO** is implemented as an option (`--config dino` or `ssl.method=dino`).
  MAE/JEPA are deliberately not implemented. SimCLR has the fewest moving parts that can silently fail; DINO's
  centering/sharpening/EMA are more fragile without tuning.
- SimCLR hyper-parameters follow the SimCLR paper / common ImageNet-100 settings:
  - NT-Xent temperature 0.2 (0.5 and 0.1 are ablated);
  - projector 512 → 2048 → 128;
  - LARS (trust coefficient 0.001, biases and BN excluded from LARS and from weight decay), lr 0.3 × B/256,
    weight decay 1e-6, momentum 0.9;
  - 10 warm-up epochs, then cosine to 0;
  - bf16 autocast (fp16 + GradScaler if bf16 is unsupported), gradient clipping at 3.0, channels-last.
- **Augmentations** (own GPU implementation, `ssl_methods/augment.py`):
  - random-resized-crop with scale 0.08–1 and ratio 3/4–4/3, plus flip p = 0.5;
  - colour jitter (0.4, 0.4, 0.4, 0.1) with p = 0.8, applied in a fixed order (torchvision randomises the
    order; immaterial);
  - grayscale p = 0.2;
  - Gaussian blur p = 0.5, kernel ≈ 10 % of the side, σ ∈ [0.1, 2.0] × S/224;
  - no solarisation for SimCLR.

  Crops whose sampled side would exceed the image are clamped instead of rejection-sampled.
- **Epochs.** `ssl.epochs: auto`:
  - After epoch 2 (epoch 1 includes cuDNN autotuning and compilation), `train_ssl.py` measures the epoch time
    (plus amortised evaluation cost) and sets the number of epochs that fit in 80 % of `--time-budget-hours`
    (default 6 h), clipped to [20, 800]. The value is stored in the resume state, so resumes keep the same
    schedule.
  - `torch.compile` of the encoder is on by default on CUDA (`ssl.compile: auto`), for ~1.2–1.5× more epochs. Any
    compile error falls back to eager mode automatically.
  - A safety stop triggers at 1.3× the SSL budget.
  - On a CUDA OOM at the start, the batch size is halved (down to 64) and the LR is scaled linearly.
- **Monitoring.** Every 10 epochs the script runs weighted kNN (k = 20, T = 0.07), a quick linear probe on a
  20 % stratified hold-out of the labelled set (noisy labels), and collapse diagnostics (std of normalised
  features vs 1/√d, RankMe effective rank). The random-init encoder is evaluated first as the reference.
- **Exported encoder.** `export.which_encoder: last` — the end of the cosine schedule. Picking the "best" epoch
  on a 260-image hold-out with noisy labels would be selection on noise (`best` remains available).

## Linear probe under label noise (Component 2)
- The encoder is frozen and in eval mode. Features use the clean view (horizontal-flip TTA, averaged), plus 8
  mildly augmented views (crop 0.35–1 + flip) as feature-level augmentation; one random view per sample per
  epoch.
- **Head.** A single `nn.Linear` trained on standardised features. The standardisation is folded into W, b
  afterwards, so the deployed head is one Linear on raw features. Training details:
  - Adam, lr 1e-2, 100 epochs, batch 256, cosine schedule with 5 % warm-up, zero init;
  - L2 weight decay picked from {1e-4, 1e-3, 1e-2, 5e-2} by mean CE validation accuracy over all 3 seed
    splits, then shared by all methods.
  - *Change during development:* lr 1e-3 left heads unconverged and made every loss identical (see the report's
    "failed decisions").
- **Methods (11):**
  - losses: CE, label smoothing (ε = 0.1), symmetric CE (α = 0.1, β = 1, A = −4), generalised CE (q = 0.7),
    soft bootstrapping (β = 0.8);
  - cleaning + CE: kNN consensus (k = 10), prototype consistency, small-loss (keep ratio 1 − η̂, clipped to
    [0.5, 0.95]), confident learning (confident-joint off-diagonal on 5-fold out-of-fold probabilities), and
    consensus (≥ 2 of kNN / prototype / CL);
  - consensus + GCE.

  Every cleaner keeps at least 30 % of each class.
- **Selection.** The method with the highest mean validation accuracy over 3 seeds; ties go to the method listed
  earlier (simpler). Validation labels are noisy. Under roughly symmetric noise the noisy accuracy is a
  monotone transform of the clean accuracy, so the ranking is preserved, but absolute values are pessimistic.
  **[check]** whether the winning margin exceeds the seed std; if not, prefer `ce` or `gce` for simplicity.
- **Noise rate.** Estimated with confident learning on all labelled data (calibrated confident joint).

## Supervised baseline (required)
- Same architecture and preprocessing (read from the SSL encoder checkpoint), the same per-seed splits and the
  same flip-TTA evaluation, single linear head, trained end to end from random init on the noisy labels.
  Training details:
  - AdamW (same adaptive-moment family as the probe's Adam), lr 1e-3, weight decay 0.05;
  - 200 epochs, batch 128, 10 warm-up epochs + cosine;
  - the same *mild* augmentation family as the probe's feature banks (`baseline.aug: ssl` uses the full SSL
    augmentation instead).
- Reported at the **final epoch** (no early stopping), like the probe. The best-epoch number is reported
  separately and labelled optimistic.
- Losses: CE plus the loss of the selected probe method (`baseline.losses: [ce, best]`).
- Time cap: 10 % of the budget, split across runs. The expected need is ~10 min, so the cap rarely binds. Every difference between the arms is written into
  `baseline_results.json` under `differences_from_probe`.

## Unknown-class detection (Component 3)
- **Scores** (higher = known), from scratch in `ood/scores.py`:
  - MSP, max-logit, energy (T = 1);
  - kNN: distance to the 5th neighbour, L2-normalised, over a bank of all labelled features;
  - Mahalanobis: L2-normalised features, class means from cleaned samples, shared covariance with 10 %
    shrinkage towards scaled identity — needed because N ≈ 1.3k vs d = 512;
  - cosine-to-prototype;
  - **knn_ratio**: mean top-5 cosine similarity to the labelled bank minus mean top-5 similarity to the unseen
    calibration holdout. This is a kNN density ratio, p_known/p_pool, as in PU learning. Unknown-class images have
    same-class neighbours in the pool but none among the labelled images. When the holdout scores itself, its own
    entry is excluded by index; neighbours with similarity > 0.9999 are treated as the same file. That only
    matters if a test image were itself in the pool;
  - z-score ensemble of {kNN, Mahalanobis, energy, cosine, knn_ratio}, normalised with known-only calibration
    scores.
- **Leave-class-out protocol** (`ood/protocol.py`): 5 folds with fixed seeds, each holding out 20 % of the
  classes as pseudo-unknown. Within each fold:
  - the known classes are split 80/20;
  - the selected probe method and all OOD statistics are fitted on known-train;
  - known-val and the held-out-class images are scored.
- **Score selection.** The score with the best mean LCO balanced open-set accuracy
  ½·(known accuracy incl. rejections) + ½·(unknown recall), at its best percentile; ties are broken by AUROC.
- **Threshold rule (`ood.threshold_mode: target`, default).** The threshold maximises the *estimated (K+1)-way
  accuracy on the unseen calibration holdout*:
  - π (known fraction of the pool) is estimated by mixture-proportion estimation: the median over known-score
    tail levels q ∈ [0.7, 0.9] of P_pool(s ≥ t_q) / P_known(s ≥ t_q). It is computed with the score that has the
    best LCO AUROC, because π is a property of the pool.
  - Unknown recall is estimated as UR(t) = (P_pool(s<t) − π·P_known(s<t))/(1−π), kept monotone.
  - Known accuracy KA(t) comes from cross-fitted predictions, divided by (1 − η̂) with η̂ capped at 0.4, because
    noisy labels underestimate KA.
  - The threshold maximises π·KA + (1−π)·UR. Known and holdout scores come from the same 5 cross-fitted fold
    models; the result is stored as a percentile of the known scores.
  - Validated on synthetic data with ground truth: estimated π 0.593 vs true 0.600, and estimated (K+1)-accuracy
    0.813 vs measured 0.800. It beat the balanced-LCO threshold (0.763).
  - **Assumption:** the hidden test set is distributed like the unlabelled pool (same ID space). If not,
    `--set ood.threshold_mode=lco` gives the mix-independent balanced percentile, which is also reported.
  - *Why balanced:* the unknown fraction of the hidden test set is unknown. The balanced point does not depend
    on it and does not reward the trivial "all known" or "all unknown" answers.
  - *Why a percentile:* it transfers from fold models (40 classes, smaller banks) to the final model (50 classes,
    all data).
  - **[check]** if the leaderboard metric is plain (K+1)-way accuracy and the test has few unknowns, a smaller p
    can do better. `ood.objective: kplus1_acc` and `fit_ood.py --percentile` exist for that.
- **Final model.** The selected probe method is fitted on all labelled data. Class-conditional statistics use
  the cleaned samples; the kNN bank uses all samples (it is label-free); the background bank is the whole
  holdout. The final threshold is the chosen percentile of known scores obtained by 5-fold cross-fitting, so no
  sample scores itself. The final model's rejection rate on the holdout is logged as a cross-check.

## Packaging and inference
- `checkpoints/final.pt`: one file that loads with `torch.load(weights_only=True)`. Contents:
  - fp32 encoder (≈ 45 MB for the base preset);
  - linear head;
  - OOD statistics for the selected score only;
  - threshold and percentile;
  - K and class names;
  - image size, channels, mean/std, flip-TTA flag;
  - model config, the full training config, provenance, a format version and a content hash.

  `export.encoder_dtype: float16` halves the size if ever needed.
- `test.py`:
  - checkpoint path relative to the script;
  - re-execs itself with `./.venv/bin/python` if that interpreter can import torch (the grader does not
    activate the venv); a `.pydeps` fallback is used otherwise;
  - CUDA with a tiny warm-up forward, falling back to CPU on any CUDA error (e.g. "no kernel image");
  - bf16 autocast on GPU;
  - DataLoader with up to 8 workers, retrying with 0 workers on failure;
  - unreadable or missing image → warning + label K;
  - relative paths in the CSV are resolved against the CSV's directory; output keeps the CSV strings verbatim;
  - an atomic write of `predictions.csv`.
- Expected inference cost: ≈ 1.2 GMACs/image with flip TTA. Throughput is dominated by PNG decoding: roughly
  6 ms/image/core, so 25k images take about 30 s with 8 workers. The 1-hour limit has very large headroom.

## Environment / CUDA 13 (verified 2026-10-12)
- The official PyTorch install pages list CUDA 13.0 (`cu130`) and 13.2 (`cu132`) wheel indexes for 2.12.x and
  2.13.0.
- `https://download.pytorch.org/whl/cu130/torch/` contains `torch-2.12.1+cu130` for cp310–cp314
  (manylinux_2_28 x86_64), and 2.13.0+cu130 as well.
- Pin: **torch 2.12.1** (a patch release, more mature than 2.13.0) for Python ≥ 3.10; torch 2.8.0 for Python 3.9.
- `setup.bash` reads the driver's CUDA version from `nvidia-smi`:
  - 13.x or newer → cu130 index, then default PyPI, then cu126;
  - 12.x → cu126, then PyPI;
  - no GPU → cu130, then PyPI, then CPU.

  NVIDIA drivers are backward compatible, so any CUDA 12.x build also runs on a CUDA 13 driver. That is why
  PyPI is a safe fallback.
- Other pins (wheels checked for cp39–cp314 as applicable): numpy 2.3.4 (≥ 3.11), 2.2.6 (3.10), 2.0.2 (3.9);
  pillow 11.3.0; pyyaml 6.0.3; matplotlib 3.10.7 / 3.9.4. No scikit-learn, no torchvision.
- **Not verifiable here:** an actual install on a CUDA 13 Linux machine (this machine is macOS, CPU only).
  `setup.bash` was verified end to end in a clean venv on macOS (Python 3.9 → torch 2.8.0 path).

## Budget split (`--time-budget-hours`, default 6)
- SSL: 80 % (epochs auto-fitted).
- Supervised baseline: ≤ 10 % (hard time cap).
- Probe, OOD, export and evaluation: minutes.
- `scripts/run_ablations.py` has its own `--ablation-budget-hours` (default 1.5) and is run separately.

## Things to double-check after the real run
1. The SSL kNN accuracy is clearly above the random-init reference, and no collapse warnings were printed.
1b. `fit_ood.py`: the per-score π estimates roughly agree (±0.1), and the target and LCO percentiles are not
   wildly different. If π estimates disagree a lot, the mixture assumption is shaky; consider
   `--set ood.threshold_mode=lco`.
2. The best probe method's margin over CE vs the seed std.
3. LCO: the selected score's AUROC, and the shape of the threshold-sensitivity curve. A flat curve means p is
   not critical.
4. The fraction of the local check set predicted as Unknown. On known training images it should be about p %.
5. `checkpoints/final.pt` is < 100 MB (≈ 45 MB expected for base) and `export_final.py`'s self-test passed.
6. `results/main/model_stats.json` throughput, and the `logs/eval.log` images/s.
