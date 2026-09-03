#!/usr/bin/env bash
set -e

echo "============================================================"
echo " GNR650 Vision Transformer Training Pipeline"
echo "============================================================"

# Setup Virtual Environment using UV or pip
if command -v uv &> /dev/null; then
    echo "[INFO] Detected UV package manager."
    if [ ! -d ".venv" ]; then
        echo "[INFO] Creating virtual environment (.venv)..."
        uv venv
    fi
    source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate 2>/dev/null || true
    echo "[INFO] Syncing dependencies with UV..."
    uv pip install -r requirements.txt
else
    echo "[INFO] UV not found. Using standard python3 venv / pip..."
    if [ ! -d ".venv" ]; then
        python3 -m venv .venv 2>/dev/null || python -m venv .venv 2>/dev/null || true
    fi
    source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate 2>/dev/null || true
    pip install -r requirements.txt
fi

mkdir -p logs checkpoints

echo "[INFO] Launching train.py with arguments: $@"
python train.py "$@"
