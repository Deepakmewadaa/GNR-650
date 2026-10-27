#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Define the custom environment name variable
ENV_NAME="gnr_assignment_env"

echo "============================================================"
echo " Setting up environment for GNR650 Assignment 2 (SSL)"
echo "============================================================"

# 1. Create virtual environment using the custom name if missing
if [ ! -d "$ENV_NAME" ]; then
    echo "[INFO] Creating virtual environment at $ENV_NAME..."
    if command -v uv &> /dev/null; then
        uv venv "$ENV_NAME"
    else
        python3 -m venv "$ENV_NAME" || python -m venv "$ENV_NAME"
    fi
fi

# 2. Locate virtualenv Python interpreter
VPY="$ENV_NAME/bin/python"
[ -f "$ENV_NAME/Scripts/python.exe" ] && VPY="$ENV_NAME/Scripts/python.exe"

if [ ! -f "$VPY" ]; then
    echo "[ERROR] Python interpreter not found in $ENV_NAME!" >&2
    exit 1
fi

# 3. Add PyTorch CUDA index if NVIDIA GPU is present
INDEX_ARGS=()
if command -v nvidia-smi &> /dev/null; then
    echo "[INFO] NVIDIA GPU detected. Adding PyTorch CUDA wheel repository..."
    INDEX_ARGS=(--extra-index-url "https://download.pytorch.org/whl/cu130" --extra-index-url "https://download.pytorch.org/whl/cu126")
fi

# 4. Install dependencies into the custom environment
if command -v uv &> /dev/null; then
    echo "[INFO] Using uv to install dependencies into $ENV_NAME..."
    uv pip install --python "$VPY" "${INDEX_ARGS[@]}" -r requirements.txt
else
    echo "[INFO] Upgrading pip and installing dependencies..."
    "$VPY" -m pip install --upgrade pip --quiet
    "$VPY" -m pip install "${INDEX_ARGS[@]}" -r requirements.txt || "$VPY" -m pip install torch numpy pillow pyyaml
fi

# 5. Create required directories
mkdir -p logs checkpoints

# 6. Verify environment
echo "[INFO] Verifying environment..."
"$VPY" -c "
import torch, numpy, PIL, yaml
print(f'[OK] PyTorch {torch.__version__} | CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'[OK] GPU: {torch.cuda.get_device_name(0)}')
"
echo "============================================================"
echo " Setup completed successfully. (venv: $ENV_NAME)"
echo "============================================================"