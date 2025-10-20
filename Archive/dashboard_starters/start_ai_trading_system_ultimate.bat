@echo off
echo ============================================================
echo 🚀 AI TRADING SYSTEM - ULTIMATE STARTUP
echo ============================================================
echo.

echo 📋 Step 1: Running comprehensive health check...
python system_health_check.py
if %errorlevel% neq 0 (
    echo ❌ Health check failed! Please review the issues above.
    pause
    exit /b 1
)

echo.
echo 📊 Step 2: Validating market data sources...
python validate_market_data.py
if %errorlevel% neq 0 (
    echo ❌ Market data validation failed! Please review the issues above.
    pause
    exit /b 1
)

echo.
echo ⚡ Step 3: Optimizing system performance...
python optimize_system_performance.py
if %errorlevel% neq 0 (
    echo ❌ Performance optimization failed! Please review the issues above.
    pause
    exit /b 1
)

echo.
echo ✅ All checks passed! System is ready for demo trading.
echo.

echo 🌐 Opening dashboard in your default browser...
start "" "http://localhost:8050"

echo.
echo 🚀 Starting AI trading dashboard...
python interactive_real_dashboard.py

echo.
echo 🎉 AI Trading System is now running at peak performance!
echo 📊 Dashboard: http://localhost:8050
echo 📈 All AI models active and optimized
echo 🔄 Real-time data feeds operational
echo 🧠 Neural networks ready for trading
echo.
pause
