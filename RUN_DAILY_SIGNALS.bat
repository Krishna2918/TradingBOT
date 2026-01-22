@echo off
cls
echo ========================================
echo   DAILY TRADING SIGNALS GENERATOR
echo   AI-Powered Buy/Sell Recommendations
echo ========================================
echo.
echo This will analyze your watchlist and generate
echo high-confidence trading signals using the
echo LSTM + GRU ensemble.
echo.
echo Minimum confidence: 60%%
echo Requires: Both models agree
echo.
pause

cd /d "%~dp0"

python daily_trading_signals.py

echo.
echo ========================================
echo Check signals/ folder for reports!
echo ========================================
echo.

pause
