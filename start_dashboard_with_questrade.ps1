# Start Trading Dashboard with Questrade API Integration
# This script sets required environment variables and starts the dashboard

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "TRADING DASHBOARD WITH QUESTRADE API" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Set Questrade credentials (NEW TOKEN with ALL OAuth scopes!)
$env:QUESTRADE_REFRESH_TOKEN = "_lflHLgLAv2o5bQzjmn7HuHKdQetjkkS0"
$env:TRADING_MODE = "demo"
$env:QUESTRADE_ALLOW_TRADING = "false"
$env:QUESTRADE_PRACTICE_MODE = "true"

Write-Host "Environment configured:" -ForegroundColor Green
Write-Host "  - Questrade Token: SET" -ForegroundColor White
Write-Host "  - Trading Mode: DEMO" -ForegroundColor White
Write-Host "  - Allow Trading: FALSE (read-only)" -ForegroundColor White
Write-Host "  - Practice Mode: TRUE`n" -ForegroundColor White

Write-Host "Starting dashboard..." -ForegroundColor Yellow
Write-Host "Dashboard will be available at: http://127.0.0.1:8051/`n" -ForegroundColor White

# Start the dashboard
python interactive_trading_dashboard.py

