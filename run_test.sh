#!/usr/bin/env bash
set -e

echo "============================================================"
echo " GNR650 Vision Transformer Test Evaluation"
echo "============================================================"

if [ -d ".venv" ]; then
    source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate 2>/dev/null || true
fi

mkdir -p logs

echo "[INFO] Launching test.py with arguments: $@"
python test.py "$@"
