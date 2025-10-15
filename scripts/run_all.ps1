#requires -Version 5.1
param(
  [int]$Limit = 100
)

$ErrorActionPreference = 'Stop'
Set-Location "$PSScriptRoot\.."

Write-Host "Activating venv..."
if (Test-Path .\.venv\Scripts\Activate.ps1) {
  . .\.venv\Scripts\Activate.ps1
}

Write-Host "Installing dependencies..."
pip install -q -r requirements.txt

Write-Host "Bootstrapping databases..."
python - <<'PY'
from src.utils.db import bootstrap_sqlite, bootstrap_duckdb
bootstrap_sqlite(); bootstrap_duckdb()
print('DB OK')
PY

Write-Host "Running pipeline demo..."
python .\main.py --demo --limit $Limit

Write-Host "Running tests..."
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
python -m pytest tests -q

Write-Host "✅✅✅ ALL TESTS PASSED — SYSTEM READY FOR BUILD 🚀" -ForegroundColor Green
