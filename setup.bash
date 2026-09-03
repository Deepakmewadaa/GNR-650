#!/usr/bin/env bash
set -e

echo "============================================================"
echo " Setting up environment for GNR650 Vision Transformer"
echo "============================================================"

if command -v uv &> /dev/null; then
    echo "[INFO] Using uv to install dependencies..."
    uv pip install -r requirements.txt
else
    echo "[INFO] Using pip to install dependencies..."
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
fi

mkdir -p logs checkpoints
echo "Setup completed successfully."
