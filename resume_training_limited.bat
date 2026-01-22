@echo off
REM Resume 50-hour training with 40% resource limits

echo ============================================================
echo RESUMING 50-HOUR TRAINING (40%% RESOURCE LIMITS)
echo ============================================================
echo.

python train_50h_limited_resources.py --resume

echo.
echo Training session ended.
pause
