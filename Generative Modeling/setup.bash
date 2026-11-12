#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ -x .venv/bin/python ]; then
  PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
elif command -v python3.11 >/dev/null 2>&1; then
  PYTHON_BIN="${PYTHON_BIN:-python3.11}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi
"$PYTHON_BIN" -c 'import sys; assert sys.version_info >= (3,11), "Python 3.11+ required"'
if [ ! -x .venv/bin/python ]; then
  "$PYTHON_BIN" -m venv .venv
fi
.venv/bin/python -m ensurepip --upgrade
.venv/bin/python -m pip install --upgrade pip
if command -v nvidia-smi >/dev/null 2>&1; then
  .venv/bin/python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu130
fi
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m doggen.parameters
.venv/bin/python -c 'import torch; print("PyTorch:",torch.__version__,"CUDA:",torch.cuda.is_available(),"runtime:",torch.version.cuda)'
echo 'Setup complete. test.py automatically uses this virtual environment.'
