# AI Trading Logs Viewer - PowerShell Script
# Combines all AI log files into one view

Write-Host "AI Trading Logs - Unified View" -ForegroundColor Green
Write-Host "=" * 60

# Function to display log file contents
function Show-LogFile {
    param(
        [string]$LogName,
        [string]$FilePath,
        [int]$Lines = 10
    )
    
    Write-Host "`n$LogName" -ForegroundColor Yellow
    Write-Host "-" * 40
    
    if (Test-Path $FilePath) {
        try {
            $content = Get-Content $FilePath -Tail $Lines -ErrorAction Stop
            if ($content) {
                $content | ForEach-Object { Write-Host "  $_" }
            } else {
                Write-Host "  No entries yet..." -ForegroundColor Gray
            }
        } catch {
            Write-Host "  Error reading file: $_" -ForegroundColor Red
        }
    } else {
        Write-Host "  File not found" -ForegroundColor Gray
    }
}

# Display all log files
Show-LogFile "AI ACTIVITY LOG" "logs/ai_activity.log" 5
Show-LogFile "AI TRADES LOG" "logs/ai_trades.log" 5  
Show-LogFile "AI SIGNALS LOG" "logs/ai_signals.log" 5
Show-LogFile "AI DECISIONS LOG" "logs/ai_decisions.log" 5

Write-Host "`n" + "=" * 60
Write-Host "To refresh, run this script again" -ForegroundColor Cyan
Write-Host "Or use: Get-Content logs/ai_activity.log -Tail 10 -Wait" -ForegroundColor Cyan
