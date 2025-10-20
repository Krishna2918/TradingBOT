@echo off
echo ========================================
echo    TradingBOT Dashboard System
echo ========================================
echo.

echo Starting Final Trading API...
start "Final Trading API" cmd /k "python final_trading_api.py"
echo Waiting for API to initialize...
timeout /t 5 /nobreak >nul

echo Starting Trading Dashboard...
start "Trading Dashboard" cmd /k "python interactive_agentic_ai_dashboard.py"
echo Waiting for Dashboard to initialize...
timeout /t 3 /nobreak >nul

echo Opening Dashboard in Browser...
start "" "http://localhost:8001/"

echo.
echo ========================================
echo    System Started Successfully!
echo ========================================
echo.
echo Dashboard: http://localhost:8001/
echo API Docs:  http://localhost:8000/docs
echo.
echo Press any key to exit...
pause >nul
