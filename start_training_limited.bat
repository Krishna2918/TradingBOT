@echo off
REM Start 50-hour training with 40% resource limits
REM Uses maximum 40% RAM and 40% GPU, leaving 60% for other processes

echo ============================================================
echo 50-HOUR TRAINING WITH 40%% RESOURCE LIMITS
echo ============================================================
echo.
echo Resource Limits:
echo   - RAM: Maximum 40%% usage (~12.6GB out of 31.4GB)
echo   - GPU: Maximum 40%% usage (~4.8GB out of 12GB)
echo   - Leaves 60%% resources free for other processes
echo.
echo This will run for 50 hours non-stop.
echo Press Ctrl+C to interrupt (progress will be saved)
echo To resume: run resume_training_limited.bat
echo.
echo Starting in 5 seconds...
timeout /t 5

python train_50h_limited_resources.py --hours 50 --checkpoint-interval 30

echo.
echo Training session ended.
pause
