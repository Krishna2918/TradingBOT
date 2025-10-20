@echo off
echo ============================================================
echo 🚀 AI TRADING SYSTEM - FINAL STARTUP
echo ============================================================
echo.

echo 📋 Running comprehensive health check...
python system_health_check.py
if %errorlevel% neq 0 (
    echo ❌ Health check failed! Please review the issues above.
    pause
    exit /b 1
)

echo.
echo ✅ Health check passed! Starting AI trading dashboard...
echo.

echo 🌐 Opening dashboard in your default browser...
start "" "http://localhost:8050"

echo.
echo 🚀 Starting AI trading dashboard...
python interactive_real_dashboard.py

echo.
echo 🎉 AI Trading System is now running!
echo 📊 Dashboard: http://localhost:8050
echo.
pause
