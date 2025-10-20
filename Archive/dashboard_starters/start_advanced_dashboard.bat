@echo off
echo Starting Advanced AI Trading Dashboard...
echo.
echo Features:
echo   - Comprehensive AI activity logging
echo   - Real-time component status tracking
echo   - Detailed trading decision analytics
echo   - Performance metrics monitoring
echo   - Organized error tracking
echo.
echo Opening dashboard in browser...
start "" "http://localhost:8058"
echo.
echo Starting dashboard server...
python interactive_advanced_dashboard.py
