@echo off
echo Starting TradingBOT Production System...
echo.
echo Choose your option:
echo 1. Start Final Trading API (Recommended)
echo 2. Start Dashboard Only
echo 3. Start Both API and Dashboard
echo 4. Run System Health Check
echo.
set /p choice="Enter choice (1-4): "

if "%choice%"=="1" (
    start_final_api.bat
) else if "%choice%"=="2" (
    start_clean_dashboard_final.bat
) else if "%choice%"=="3" (
    start "" start_final_api.bat
    timeout /t 5
    start_clean_dashboard_final.bat
) else if "%choice%"=="4" (
    python Tests/system_health_check.py
) else (
    echo Invalid choice
    pause
)
