@echo off
REM Quick resume script for 50-hour training
REM Run this if training gets interrupted or crashes

echo ========================================
echo RESUMING 50-HOUR TRAINING FROM CHECKPOINT
echo ========================================
echo.

python train_50h_continuous.py --resume

echo.
echo Training session ended.
pause
