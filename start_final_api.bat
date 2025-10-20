@echo off
echo.
echo ================================================================================
echo                    🚀 FINAL TRADING API - STARTUP SCRIPT
echo ================================================================================
echo.
echo Starting the Complete TradingBOT System API...
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ ERROR: Python is not installed or not in PATH
    echo Please install Python 3.11+ and try again
    pause
    exit /b 1
)

REM Check if required packages are installed
echo 📦 Checking dependencies...
python -c "import fastapi, uvicorn, pandas, numpy" >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ ERROR: Required packages not installed
    echo Installing dependencies...
    pip install fastapi uvicorn pandas numpy
    if %errorlevel% neq 0 (
        echo ❌ ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
)

REM Create logs directory
if not exist "logs" mkdir logs

REM Start the API
echo.
echo 🚀 Starting Final Trading API...
echo.
echo 📊 Features Available:
echo    ✅ AI Trading System (MasterOrchestrator + Maximum Power AI)
echo    ✅ Real-time Market Data (Yahoo Finance + Questrade)
echo    ✅ Portfolio Management (Live/Demo modes)
echo    ✅ Risk Management (Advanced risk metrics)
echo    ✅ Order Execution (Paper + Live trading)
echo    ✅ Performance Analytics (Comprehensive reporting)
echo    ✅ System Monitoring (Health checks + metrics)
echo    ✅ Session Management (State persistence)
echo    ✅ Dashboard Integration (Real-time updates)
echo    ✅ Advanced Logging (AI decisions + system events)
echo    ✅ WebSocket Support (Real-time updates)
echo.
echo 🌐 Access Points:
echo    📚 API Documentation: http://localhost:8000/docs
echo    📖 Alternative Docs: http://localhost:8000/redoc
echo    🏠 Root Page: http://localhost:8000/
echo    🔌 WebSocket: ws://localhost:8000/ws
echo.
echo 🔧 Starting server...
echo.

REM Open browser to API docs
start "" "http://localhost:8000/docs"

REM Start the API server
python final_trading_api.py

echo.
echo API server stopped.
pause
