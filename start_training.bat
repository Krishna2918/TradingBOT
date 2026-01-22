@echo off
REM Start fresh 50-hour training run
REM This will train all AI models continuously for 50 hours

echo ========================================
echo STARTING 50-HOUR CONTINUOUS AI TRAINING
echo ========================================
echo.
echo This will run for 50 hours non-stop.
echo Press Ctrl+C to interrupt (progress will be saved)
echo To resume later: run resume_training.bat
echo.
echo Starting in 5 seconds...
timeout /t 5

python train_50h_continuous.py --hours 50 --checkpoint-interval 30

echo.
echo Training session ended.
pause
