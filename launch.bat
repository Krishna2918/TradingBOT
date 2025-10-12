@echo off
REM AI Trading Bot - Interactive Dashboard Launcher
REM Double-click this file to start

echo.
echo ========================================
echo    AI Trading Bot - Starting...
echo ========================================
echo.

REM Change to script directory
cd /d "%~dp0"

REM Check if dashboard is already running
netstat -ano | findstr :8051 >nul 2>&1
if %errorlevel% equ 0 (
    echo [INFO] Dashboard is already running!
    echo [INFO] Opening browser...
    start http://127.0.0.1:8051
    echo.
    echo [SUCCESS] Browser opened to existing dashboard
    echo [INFO] Dashboard at: http://127.0.0.1:8051
    pause
    exit /b
)

echo [STARTING] Interactive Dashboard...
echo [INFO] This may take 10-15 seconds...
echo.

REM Activate virtual environment if it exists
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

REM Start Python dashboard
start "Trading Dashboard" python ai_trading_dashboard.py

REM Wait for server to start (longer wait)
echo [WAIT] Starting server...
timeout /t 10 /nobreak >nul

REM Check if server started
netstat -ano | findstr :8051 >nul 2>&1
if %errorlevel% equ 0 (
    echo [SUCCESS] Dashboard started successfully!
    start http://127.0.0.1:8051
    echo [INFO] Browser opened to: http://127.0.0.1:8051
) else (
    echo [ERROR] Dashboard failed to start
    echo [INFO] Check the "Trading Dashboard" window for errors
)

echo.
echo [INFO] Keep the "Trading Dashboard" window open
echo [INFO] Close it to stop the dashboard
echo.
pause
