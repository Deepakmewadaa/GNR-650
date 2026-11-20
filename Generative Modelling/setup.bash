#!/usr/bin/env bash
# ==============================================================================
# Setup script for GNR650 Generative Modeling Assignment
# Environment: Linux, CUDA 13.x, GPU (8GB VRAM)
# ==============================================================================

set -e # Exit immediately if a command exits with a non-zero status

echo "=================================================================="
echo "          GNR650 GENERATIVE MODELING: SYSTEM SETUP                "
echo "=================================================================="

# 1. Verify Python version
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi
echo "[1/4] Using Python executable: $($PYTHON_CMD --version)"

# 2. Install dependencies
echo "[2/4] Installing / verifying required Python packages..."
$PYTHON_CMD -m pip install -r requirements.txt --quiet --no-warn-script-location

# 3. Create required directory structure
echo "[3/4] Ensuring workspace directories exist..."
mkdir -p checkpoints
mkdir -p dataset
mkdir -p outputs

# 4. Run parameter constraint check
echo "[4/4] Verifying model parameter constraint (< 50M)..."
$PYTHON_CMD utils/count_params.py

echo "=================================================================="
echo "   [SUCCESS] Setup complete! Ready for evaluation via test.py.    "
echo "=================================================================="
