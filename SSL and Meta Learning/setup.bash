#!/usr/bin/env bash
# Non-interactive environment setup for GNR650 Assignment 2.
#   bash setup.bash
# Creates ./.venv (falls back to installing into ./.pydeps for the current interpreter if a venv cannot be
# created), installs pinned requirements with a CUDA-13-compatible PyTorch build, and verifies imports.
# It does NOT train anything and does NOT download data. test.py finds .venv/.pydeps on its own, so the
# venv never needs to be activated.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT" || exit 1
export PIP_NO_INPUT=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONNOUSERSITE=1

log()  { echo "[setup] $*"; }
warn() { echo "[setup] WARNING: $*" >&2; }

# ---------------------------------------------------------------- 1. choose a Python (>=3.10 preferred)
PY=""
for c in python3.12 python3.11 python3.13 python3.10 python3.14 python3 python; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null; then
    PY="$(command -v "$c")"; break
  fi
done
if [ -z "$PY" ]; then echo "[setup] ERROR: no python >= 3.9 found" >&2; exit 1; fi
PYV="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
log "using $PY (Python $PYV)"
TORCH_PIN="$("$PY" -c 'import sys; print("2.12.1" if sys.version_info >= (3, 10) else "2.8.0")')"

# ---------------------------------------------------------------- 2. venv, or fall back to ./.pydeps
MODE=venv
VPY="$ROOT/.venv/bin/python"
if [ ! -x "$VPY" ]; then
  "$PY" -m venv "$ROOT/.venv" >/dev/null 2>&1 || "$PY" -m virtualenv "$ROOT/.venv" >/dev/null 2>&1 || true
fi
if [ -x "$VPY" ] && "$VPY" -m pip --version >/dev/null 2>&1; then
  log "virtualenv: $ROOT/.venv"
  PIP=("$VPY" -m pip)
  "$VPY" -m pip install --upgrade pip setuptools wheel >/dev/null 2>&1 || warn "pip upgrade failed (continuing)"
else
  MODE=target
  rm -rf "$ROOT/.venv"
  warn "could not create a venv (python3-venv missing?). Installing into $ROOT/.pydeps for $PY instead."
  "$PY" -m pip --version >/dev/null 2>&1 || "$PY" -m ensurepip --user >/dev/null 2>&1 || true
  if ! "$PY" -m pip --version >/dev/null 2>&1; then echo "[setup] ERROR: pip is not available for $PY" >&2; exit 1; fi
  PIP=("$PY" -m pip)
  VPY="$PY"
  mkdir -p "$ROOT/.pydeps"
fi

pip_install() {   # pip_install <args...> ; honours MODE
  if [ "$MODE" = target ]; then "${PIP[@]}" install --upgrade --target "$ROOT/.pydeps" "$@"; else "${PIP[@]}" install "$@"; fi
}
have_torch() {
  if [ "$MODE" = target ]; then PYTHONPATH="$ROOT/.pydeps" "$VPY" -c "import torch, sys; sys.exit(0 if torch.__version__.split('+')[0] == '$TORCH_PIN' else 1)" >/dev/null 2>&1
  else "$VPY" -c "import torch, sys; sys.exit(0 if torch.__version__.split('+')[0] == '$TORCH_PIN' else 1)" >/dev/null 2>&1; fi
}

# ---------------------------------------------------------------- 3. PyTorch matched to the driver
# The driver's max CUDA version (nvidia-smi) picks the wheel index: 13.x -> cu130, 12.x -> cu126.
# Drivers are backward compatible, so the default PyPI wheel is a valid fallback on CUDA 13 machines.
DRV_CUDA=""
if command -v nvidia-smi >/dev/null 2>&1; then
  DRV_CUDA="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9][0-9]*\.[0-9]*\).*/\1/p' | head -1)"
fi
log "driver CUDA version: ${DRV_CUDA:-none detected}; OS: $(uname -s) $(uname -m)"
IDX_LIST=()
case "$(uname -s)" in
  Linux)
    case "${DRV_CUDA%%.*}" in
      12) IDX_LIST=(https://download.pytorch.org/whl/cu126 default) ;;
      "") IDX_LIST=(https://download.pytorch.org/whl/cu130 default https://download.pytorch.org/whl/cpu) ;;
      *)  IDX_LIST=(https://download.pytorch.org/whl/cu130 default https://download.pytorch.org/whl/cu126) ;;
    esac ;;
  *) IDX_LIST=(default) ;;
esac
if have_torch; then
  log "torch $TORCH_PIN already installed - skipping"
else
  OK=0
  for idx in "${IDX_LIST[@]}"; do
    if [ "$idx" = default ]; then
      log "installing torch==$TORCH_PIN from PyPI (default build)"
      pip_install "torch==$TORCH_PIN" && OK=1
    else
      log "installing torch==$TORCH_PIN from $idx"
      pip_install "torch==$TORCH_PIN" --index-url "$idx" && OK=1
    fi
    [ $OK = 1 ] && break
    warn "torch install from $idx failed; trying next option"
  done
  if [ $OK != 1 ]; then echo "[setup] ERROR: could not install torch==$TORCH_PIN" >&2; exit 1; fi
fi

# ---------------------------------------------------------------- 4. remaining pinned requirements
# torch is excluded here: it is already installed from the CUDA-matched index above (and --target mode
# would otherwise re-download a default build over it).
REQ_NO_TORCH="$(mktemp)"
grep -v -i '^ *torch' "$ROOT/requirements.txt" > "$REQ_NO_TORCH"
if ! pip_install -r "$REQ_NO_TORCH"; then
  warn "pinned install failed (no wheel for Python $PYV?); installing core packages UNPINNED"
  pip_install numpy pillow pyyaml || { echo "[setup] ERROR: failed to install numpy/pillow/pyyaml" >&2; exit 1; }
  pip_install matplotlib || warn "matplotlib not installed (only needed for training-side plots)"
fi
rm -f "$REQ_NO_TORCH"

# ---------------------------------------------------------------- 5. verify
CHECK='import torch, numpy, PIL, yaml
print("[setup] torch", torch.__version__, "| numpy", numpy.__version__, "| pillow", PIL.__version__)
print("[setup] CUDA available:", torch.cuda.is_available(), "| torch CUDA build:", torch.version.cuda)
if torch.cuda.is_available():
    print("[setup] GPU:", torch.cuda.get_device_name(0))
    x = torch.randn(256, 256, device="cuda"); (x @ x).sum().item()
    print("[setup] CUDA matmul OK")'
if [ "$MODE" = target ]; then PYTHONPATH="$ROOT/.pydeps" "$VPY" -c "$CHECK"; else "$VPY" -c "$CHECK"; fi || {
  echo "[setup] ERROR: verification import failed" >&2; exit 1; }
if [ ! -f "$ROOT/checkpoints/final.pt" ]; then
  warn "checkpoints/final.pt not found - test.py will refuse to run until training + export_final.py are done"
fi
log "done (mode=$MODE). Run: python test.py --test-csv <abs csv> --output-path <abs dir>  (no activation needed)"
