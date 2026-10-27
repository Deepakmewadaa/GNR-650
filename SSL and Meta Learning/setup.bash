#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

ENV_NAME="gnr_assignment_env"

echo "============================================================"
echo " Setting up Conda environment for GNR650 Assignment 2 (SSL)"
echo "============================================================"

# 1. Force create/update Conda environment with Python 3.11
if ! conda env list | grep -q "$ENV_NAME"; then
    echo "[INFO] Creating Conda environment '$ENV_NAME' with Python 3.11..."
    conda create -n "$ENV_NAME" python=3.11 -y
else
    echo "[INFO] Conda environment '$ENV_NAME' already exists."
fi

# 2. Locate the precise Python interpreter inside the Conda environment
CONDA_ENV_PATH=$(conda env list | grep -E "^${ENV_NAME}[[:space:]]" | awk '{print $NF}')
VPY="$CONDA_ENV_PATH/bin/python"

if [ ! -f "$VPY" ]; then
    echo "[ERROR] Conda Python interpreter not found at $VPY!" >&2
    exit 1
fi

# 3. Add PyTorch CUDA index if NVIDIA GPU is present
INDEX_ARGS=()
if command -v nvidia-smi &> /dev/null; then
    echo "[INFO] NVIDIA GPU detected. Adding PyTorch CUDA wheel repository..."
    INDEX_ARGS=(--extra-index-url "https://download.pytorch.org/whl/cu130" --extra-index-url "https://download.pytorch.org/whl/cu126")
fi

# 4. Install dependencies directly into the targeted Conda Python binary path
if command -v uv &> /dev/null; then
    echo "[INFO] Using uv to install dependencies into Conda env..."
    uv pip install --python "$VPY" "${INDEX_ARGS[@]}" -r requirements.txt
else
    echo "[INFO] Upgrading pip and installing dependencies via standard pip..."
    "$VPY" -m pip install --upgrade pip --quiet
    "$VPY" -m pip install "${INDEX_ARGS[@]}" -r requirements.txt || "$VPY" -m pip install torch numpy pillow pyyaml
fi

# 5. Create required directories
mkdir -p logs checkpoints

# 6. Verify environment inside the correct binary context
echo "[INFO] Verifying environment..."
"$VPY" -c "
import torch, numpy, PIL, yaml
print(f'[OK] PyTorch {torch.__version__} | CUDA available: {torch.cuda.is_available()}')
print(f'[OK] NumPy version: {numpy.__version__}')
if torch.cuda.is_available():
    print(f'[OK] GPU: {torch.cuda.get_device_name(0)}')
"
echo "============================================================"
echo " Setup completed successfully. Your environment is ready!"
echo "============================================================"