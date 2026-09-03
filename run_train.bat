@echo off
echo ============================================================
echo  GNR650 Vision Transformer Training Pipeline
echo ============================================================

where uv >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [INFO] Detected UV package manager.
    if not exist ".venv" (
        echo [INFO] Creating virtual environment (.venv)...
        uv venv
    )
    call .venv\Scripts\activate.bat
    echo [INFO] Syncing dependencies with UV...
    uv pip install -r requirements.txt
) else (
    echo [INFO] UV not found. Using standard python...
    if not exist ".venv" (
        python -m venv .venv
    )
    call .venv\Scripts\activate.bat
    pip install -r requirements.txt
)

if not exist "logs" mkdir logs
if not exist "checkpoints" mkdir checkpoints

echo [INFO] Starting training...
python train.py %*
