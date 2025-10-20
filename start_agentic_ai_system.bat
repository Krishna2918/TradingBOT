@echo off
echo ========================================
echo   Agentic AI Trading System Startup
echo ========================================
echo.

echo Starting the complete Agentic AI Trading System...
echo.

echo [1/3] Starting Final Trading API (Port 8000)...
start "Final Trading API" cmd /k "python final_trading_api.py"
timeout /t 5 /nobreak > nul

echo [2/3] Starting Agentic AI Dashboard (Port 8001)...
start "Agentic AI Dashboard" cmd /k "python interactive_agentic_ai_dashboard.py"
timeout /t 3 /nobreak > nul

echo [3/3] Opening Dashboard in Browser...
timeout /t 2 /nobreak > nul
start "" "http://localhost:8001/"

echo.
echo ========================================
echo   System Started Successfully!
echo ========================================
echo.
echo Access Points:
echo - Final Trading API: http://localhost:8000
echo - API Documentation: http://localhost:8000/docs
echo - Agentic AI Dashboard: http://localhost:8001
echo - Agent Status: http://localhost:8000/api/agents/status
echo - Resource Manager: http://localhost:8000/api/agents/resource-manager/status
echo.
echo Features Available:
echo - 7 Production-Ready Agents
echo - Real-time Monitoring
echo - Resource Management
echo - Market Analysis
echo - Learning & Adaptation
echo - Portfolio Optimization
echo - Risk Management
echo - System Health Monitoring
echo.
echo Press any key to exit...
pause > nul
