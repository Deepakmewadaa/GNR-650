# GNR650: Self-Supervised Learning & Meta-Learning for Noisy Open-Set Recognition

Implementation of a from-scratch self-supervised representation learning and meta-learning pipeline for GNR650 Assignment 2. The pipeline trains a ResNet-34 encoder via multi-view self-supervised pre-training (BYOL / SwAV / SimCLR) on unlabelled data, fits a noise-robust linear probe on noisy labelled classes, and calibrates an open-set unknown detector packaged into a single standalone checkpoint (`checkpoints/final.pt`, < 100 MB).

---

## 🚀 Quick Start & Environment Setup

### 1. Automated Setup (Recommended)
Run the automated environment installer, which detects your CUDA driver version, creates `.venv`, and installs pinned PyTorch and dependencies:
```bash
bash setup.bash
```

*(Note: `test.py` automatically detects and uses `.venv` or `.pydeps` without requiring manual virtual environment activation).*

### 2. Fast Setup with `uv` or Standard `pip`
```bash
# Using uv (fastest)
uv venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt

# Or using standard pip
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 🧪 Testing & Evaluation (Official Grader Submission Command)

To evaluate the packaged checkpoint on an unseen test dataset and generate the predictions CSV required by the grader:

### Official Submission Command
```bash
python test.py --test-csv /path/to/test.csv --output-path /path/to/output_dir
```

- **Input:** A CSV file containing columns `id,file_path`.
- **Output:** Writes `<output_dir>/predictions.csv` with columns `id,file_path,label` preserving the input order. Labels `0` to `49` denote known classes; label `50` denotes Unknown (OOD).
- **Inference Details:** Uses 6-view test-time augmentation (TTA across multi-scale crops and horizontal flips) and loads weights directly from `checkpoints/final.pt`. Auto-detects CUDA/MPS/CPU.

### Local Evaluation & Verification
To test prediction formatting and score against a local ground truth CSV:
```bash
python scripts/local_eval.py score --pred /path/to/output_dir/predictions.csv --gt /path/to/gt.csv
```

---

## 🏋️ Training the Pipeline

### Data Directory Layout
Place the dataset in `./data` (or provide `--data-root /path/to/data`):
```text
data/
├── labelled_data/
│   ├── class_00/
│   │   ├── img_001.png
│   │   └── ...
│   └── ...
└── unlabelled_data/
    ├── unlabelled_0001.png
    └── ...
```

### One-Command Full Pipeline Launcher
To run the complete pipeline (`inspect` → `doctor` → `ssl` → `probe` → `baseline` → `ood` → `export`):
```bash
bash run_all.sh --config base --data-root ./data --budget 6
```

### Step-by-Step Execution

#### 1. Self-Supervised Pre-Training (SSL)
Pre-train the ResNet-34 encoder on unlabelled data:
```bash
# BYOL Pre-training (4-View Multi-Crop, 160x160 resolution)
python train_ssl.py --config configs/base.yaml --set ssl.method=byol --epochs 400

# Alternative: SwAV Pre-training
python train_ssl.py --config configs/base.yaml --set ssl.method=swav --epochs 200

# Alternative: SimCLR Pre-training
python train_ssl.py --config configs/base.yaml --set ssl.method=simclr --epochs 200
```

#### 2. Noise-Robust Downstream Linear Probing
Train the classification head on the noisy labelled set using the frozen pre-trained encoder:
```bash
# Small-Loss sample selection + Label Smoothing (epsilon = 0.28)
python train_probe.py --config configs/base.yaml \
    --encoder runs/main/ssl/encoder_best.pt \
    --methods ls \
    --set probe.loss_params.ls.epsilon=0.28 probe.cleaner=smallloss probe.cleaner_params.smallloss.keep=0.55
```

#### 3. Supervised Baseline (Comparative Reference)
Train an end-to-end supervised ResNet-34 directly on the noisy labelled training data:
```bash
python train_supervised_baseline.py --config configs/base.yaml
```

#### 4. Unknown Class (OOD) Calibration
Calibrate open-set unknown detection scores and decision threshold via leave-class-out validation:
```bash
python fit_ood.py --config configs/base.yaml \
    --encoder runs/main/ssl/encoder_best.pt \
    --score ensemble
```

#### 5. Package Standalone Model
Export the complete model (encoder, classification head, OOD statistics, and metadata) into a single self-contained checkpoint:
```bash
python export_final.py --output checkpoints/final.pt
```

---

## 📊 Logging & Experiment Tracking

- **Console & Metric Logs:** Each pipeline stage logs to `logs/<stage>.log` and `runs/<run_name>/`.
- **Diagnostic Metrics:** kNN accuracy, representation rank, and alignment/uniformity diagnostics are tracked across pre-training epochs in `runs/<run_name>/ssl/log.csv`.
- **Results & Summaries:** Numerical tables, probe scores, and OOD performance summaries are saved to `results/<run_name>/`.
- **Exported Weights:** The final submission artifact is saved to `checkpoints/final.pt` (~85 MB, well within the 100 MB limit).

---

## 🔬 Core Experiments & Ablations

### Component 1: SSL Paradigm & Pre-Training Resolution
```bash
# Resolution 128x128 vs 160x160
python train_ssl.py --config configs/base.yaml --set data.max_image_size=128 ssl.method=byol
python train_ssl.py --config configs/base.yaml --set data.max_image_size=160 ssl.method=byol

# Multi-crop augmentations (2 Global 160x160 + 2 Local 96x96)
python train_ssl.py --config configs/base.yaml --set ssl.method=byol byol.n_local=2
```

### Component 2: Noise-Robust Probing Strategies
```bash
# Standard Cross-Entropy with Weight Decay Sweep
python train_probe.py --config configs/base.yaml --methods ce --seeds 0 1 2

# Mahalanobis Nearest Class Mean (NCM) Probing
python train_probe.py --config configs/base.yaml --methods ncm_mahalanobis

# Small-Loss Selection + Tuned Label Smoothing
python train_probe.py --config configs/base.yaml --methods ls \
    --set probe.loss_params.ls.epsilon=0.28 probe.cleaner=smallloss
```

### Component 3: Open-Set Scoring Functions
```bash
# Evaluate individual OOD scoring functions
python fit_ood.py --config configs/base.yaml --score energy
python fit_ood.py --config configs/base.yaml --score mahalanobis
python fit_ood.py --config configs/base.yaml --score ensemble
```

---

## 📁 Repository Structure

```text
├── checkpoints/              # Exported final model checkpoint (final.pt < 100 MB)
├── configs/                  # Preset configuration files (base.yaml, small.yaml, tiny.yaml)
├── datasets/                 # Image loaders, stratified splits, and caching utilities
├── models/                   # From-scratch ResNet-34 encoder and FinalModel wrapper
├── ssl_methods/              # Self-supervised algorithms (BYOL, SwAV, SimCLR) & augmentations
├── probe/                    # Linear probing heads, loss functions, and label cleaners
├── ood/                      # Open-set scoring methods (Energy, Mahalanobis, Ensemble)
├── scripts/                  # Data inspection, diagnostic tests, and local evaluation tools
├── run_all.sh                # End-to-end pipeline execution script
├── setup.bash                # Automated environment setup script
├── test.py                   # Grader evaluation script (test-csv -> predictions.csv)
├── train_ssl.py              # SSL stage pre-training runner
├── train_probe.py            # Downstream linear probing runner
├── train_supervised_baseline.py  # Supervised comparison baseline
├── fit_ood.py                # OOD calibration and threshold selection
└── export_final.py           # Packaging script for checkpoints/final.pt
```
