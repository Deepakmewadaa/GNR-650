# GNR650: Vision Transformer from Scratch

Implementation of a Vision Transformer (ViT) classifier built from scratch using NumPy/CuPy with FOVI (KNN foveation patch tokenization) for the GNR650 Vision Transformer Programming Assignment.

---

## 🚀 Quick Start on Any System (Linux / Mac / Windows / GPU Servers)

### 1. Clone the Repository
```bash
git clone https://github.com/Deepakmewadaa/GNR-650.git
cd "ViT from scratch"
```

### 2. Fast Environment Setup with `uv` (Recommended)
This repository includes full support for [`uv`](https://github.com/astral-sh/uv), the ultra-fast Python package manager.

```bash
# If you have uv installed:
uv venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt

# Or run the automated setup script:
bash setup.bash
```

*(If you do not have `uv`, standard `python3 -m venv .venv` and `pip install -r requirements.txt` work out of the box).*

---

## 🏋️ Training the Model

### One-Command Training Launcher (Linux / Mac / WSL)
```bash
bash run_train.sh --train-path "/path/to/dataset/train"
```

### On Windows
```cmd
run_train.bat --train-path "C:\path\to\dataset\train"
```

### Direct Python Execution
```bash
python train.py --train-path "/path/to/dataset/train" \
    --val-split 0.2 \
    --epochs 20 \
    --batch-size 32 \
    --patch-size 16 \
    --d-model 128 \
    --d-ff 512 \
    --num-heads 4 \
    --num-blocks 10 \
    --lr 1e-3 \
    --dropout 0.2 \
    --weight-decay 5e-4 \
    --label-smoothing 0.1 \
    --tokenizer fovi \
    --exp-name vit_baseline
```

---

## 🧪 Testing & Evaluation (Assignment Submission Command)

To evaluate a trained checkpoint on any test dataset and output all 6 mandatory assignment metrics:

### Linux / Mac / WSL
```bash
bash run_test.sh --test-path "/path/to/dataset/test"
```

### Direct Python Evaluation
```bash
python test.py --test-path "/path/to/dataset/test" 
```

---

## 📊 Automatic Logging & Experiment Tracking

* Every training run automatically creates a full console & metric log in `logs/train_<timestamp>_<exp_name>.log`.
* Every evaluation run writes to `logs/test_<timestamp>.log`.
* Detailed numerical histories (train/val loss and accuracy curves per epoch) are saved to `checkpoints/<exp_name>_log.json` for plotting in your report.
* The best model weights across all epochs are automatically saved to `checkpoints/best_model.npz`.

---

## 🔬 Core Assignment Experiments

### Core Component 2: Model Scaling
```bash
# Baseline: 4 blocks, 4 heads, d_model=64
python train.py --train-path <path> --num-blocks 4 --num-heads 4 --d-model 64 --d-ff 128 --exp-name scale_b4_d64

# Scaled: 10 blocks, 4 heads, d_model=128
python train.py --train-path <path> --num-blocks 10 --num-heads 4 --d-model 128 --d-ff 512 --exp-name scale_b10_d128
```

### Core Component 3: Data Scaling
```bash
# 10% Uniform Data
python train.py --train-path <path> --data-fraction 0.10 --sampling uniform --exp-name data_scale_10_uniform

# 25% Uniform Data
python train.py --train-path <path> --data-fraction 0.25 --sampling uniform --exp-name data_scale_25_uniform

# 50% Uniform Data
python train.py --train-path <path> --data-fraction 0.50 --sampling uniform --exp-name data_scale_50_uniform

# 100% Uniform Data
python train.py --train-path <path> --data-fraction 1.00 --sampling uniform --exp-name data_scale_100_uniform
```
