@echo off
echo ============================================================
echo  GNR650 Vision Transformer Test Evaluation
echo ============================================================

if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

if not exist "logs" mkdir logs

echo [INFO] Starting evaluation...
python test.py %*
