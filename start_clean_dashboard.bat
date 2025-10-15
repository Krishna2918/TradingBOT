@echo off
echo ================================================================================
echo CLEAN AI Trading Dashboard - Starting...
echo ================================================================================
echo.
echo Opening dashboard in browser...
start "" "http://localhost:8060"
echo.
echo Starting dashboard server...
python interactive_clean_dashboard.py
