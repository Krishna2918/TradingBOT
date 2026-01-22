@echo off
echo ========================================
echo    AI Trading Bot - Dashboard Launcher
echo ========================================
echo.
echo Select a dashboard to start:
echo.
echo 1. Main Trading Dashboard (Port 8052)
echo 2. Agentic AI Dashboard (Port 8001) 
echo 3. Risk Management Dashboard (Port 8053)
echo 4. Comprehensive Analysis Dashboard
echo 5. Exit
echo.
set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" (
    echo Starting Main Trading Dashboard...
    cd Final_dashboards
    python interactive_clean_dashboard_final.py
)
if "%choice%"=="2" (
    echo Starting Agentic AI Dashboard...
    cd Final_dashboards
    python interactive_agentic_ai_dashboard.py
)
if "%choice%"=="3" (
    echo Starting Risk Management Dashboard...
    cd Final_dashboards
    python risk_dashboard.py
)
if "%choice%"=="4" (
    echo Starting Comprehensive Analysis Dashboard...
    cd Final_dashboards
    python comprehensive_dashboard.py
)
if "%choice%"=="5" (
    echo Goodbye!
    exit
)

pause