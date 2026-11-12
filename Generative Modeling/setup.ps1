param([switch]$CPU)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
function Check-Exit { if ($LASTEXITCODE -ne 0) { throw "Command failed: $LASTEXITCODE" } }
if (!(Test-Path '.venv\Scripts\python.exe')) {
    py -3.11 -m venv .venv
    Check-Exit
}
$PythonExe = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
& $PythonExe -m pip install --upgrade pip
Check-Exit
if (!$CPU) {
    & $PythonExe -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu130
    Check-Exit
}
& $PythonExe -m pip install -r requirements.txt
Check-Exit
& $PythonExe -m doggen.parameters
Check-Exit
& $PythonExe -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)'
Check-Exit
Write-Host 'Use .venv\Scripts\python.exe for training; activation is optional.'
