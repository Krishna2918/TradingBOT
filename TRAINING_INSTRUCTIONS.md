# 50-Hour Continuous AI Training

## Quick Start

### Start Fresh Training (50 hours)
```bash
python train_50h_continuous.py
```

Or use the batch file:
```bash
start_training.bat
```

### Resume from Checkpoint (if interrupted)
```bash
python train_50h_continuous.py --resume
```

Or use the batch file:
```bash
resume_training.bat
```

## Features

✓ **Automatic Checkpointing**: Saves progress every 30 minutes
✓ **Crash Recovery**: Resume exactly where you left off
✓ **Multi-Model Training**: Trains all AI models sequentially
✓ **GPU Management**: Intelligent GPU memory management
✓ **Comprehensive Logging**: Detailed logs in `training_logs/`
✓ **Progress Tracking**: Real-time ETA and progress updates
✓ **System Monitoring**: CPU, RAM, GPU usage tracking

## Models Trained

The script will train these models in order:

1. **LSTM Model** (Production) - 10 hours estimated
2. **Transformer Model** - 12 hours estimated
3. **GRU-Transformer Hybrid** - 10 hours estimated
4. **PPO Agent** (Reinforcement Learning) - 8 hours estimated
5. **DQN Agent** (Reinforcement Learning) - 8 hours estimated
6. **Aggressive LSTM** - 6 hours estimated

Total: ~54 hours of training distributed across 50 hours

## Command Line Options

```bash
# Start fresh 50-hour training
python train_50h_continuous.py

# Resume from last checkpoint
python train_50h_continuous.py --resume

# Custom training duration
python train_50h_continuous.py --hours 100

# Change checkpoint frequency (in minutes)
python train_50h_continuous.py --checkpoint-interval 60

# Parallel training (experimental)
python train_50h_continuous.py --parallel
```

## What Happens During Training

1. **Initialization**
   - Creates checkpoint directory: `continuous_training_checkpoints/`
   - Creates log directory: `training_logs/`
   - Checks GPU availability
   - Loads previous state if resuming

2. **Training Loop**
   - Trains each model sequentially
   - Saves checkpoint every 30 minutes
   - Monitors system resources
   - Logs all output to file
   - Tracks progress and ETA

3. **Automatic Checkpointing**
   - Saves training state every 30 minutes
   - Records which model is currently training
   - Tracks elapsed time
   - Stores performance metrics

4. **Completion**
   - Generates training summary
   - Shows final statistics
   - Saves all checkpoints

## Recovery from Crashes

If training stops unexpectedly:

1. **System crash / power loss**:
   ```bash
   python train_50h_continuous.py --resume
   ```

2. **Manual interruption (Ctrl+C)**:
   ```bash
   python train_50h_continuous.py --resume
   ```

3. **Check status**:
   ```bash
   # View the state file
   type continuous_training_checkpoints\training_state.json

   # View latest log
   dir training_logs /o-d
   ```

## Monitoring Training

### View Real-time Logs
```bash
# Windows
type training_logs\training_50h_*.log

# Or use PowerShell to tail
Get-Content training_logs\training_50h_*.log -Wait -Tail 50
```

### Check Training State
```bash
type continuous_training_checkpoints\training_state.json
```

The state file contains:
- Current model being trained
- Elapsed time
- Models completed
- Performance metrics
- System information

### Monitor GPU Usage
```bash
# If you have nvidia-smi
nvidia-smi -l 5
```

## Directory Structure

```
TradingBOT/
├── train_50h_continuous.py          # Main training script
├── start_training.bat                # Quick start (Windows)
├── resume_training.bat               # Quick resume (Windows)
├── TRAINING_INSTRUCTIONS.md          # This file
├── continuous_training_checkpoints/  # Auto-created
│   └── training_state.json          # Current state
├── training_logs/                    # Auto-created
│   └── training_50h_*.log           # Detailed logs
├── checkpoints/                      # Model checkpoints
│   ├── transformer/
│   └── lstm/
└── models/                           # Saved models
```

## Troubleshooting

### Training Not Starting

**Check Python version**:
```bash
python --version
# Should be Python 3.8+
```

**Check PyTorch installation**:
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

**Install dependencies**:
```bash
pip install torch pandas numpy tqdm psutil
```

### Out of Memory Errors

**Reduce batch size** (edit model training scripts):
- LSTM: Reduce batch_size in train_lstm_production.py
- Transformer: Reduce batch_size in train_transformer_production.py

**Enable gradient checkpointing**:
- Already enabled in most scripts

**Clear GPU cache**:
```bash
python -c "import torch; torch.cuda.empty_cache()"
```

### Training Too Slow

**Check GPU usage**:
```bash
nvidia-smi
```

**Verify CUDA is being used**:
Check logs for "Using device: cuda"

**Close other applications**:
- Close browsers, games, other GPU programs
- Free up system RAM

### Resume Not Working

**State file corrupted**:
```bash
# Delete state file and start fresh
del continuous_training_checkpoints\training_state.json
python train_50h_continuous.py
```

**Wrong directory**:
Make sure you're in the TradingBOT directory:
```bash
cd C:\Users\Coding\Desktop\GRID\projects\TradingBOT
```

## Best Practices

1. **Before Starting**:
   - Close unnecessary applications
   - Ensure stable power supply (use UPS if available)
   - Check available disk space (need ~50GB free)
   - Verify GPU drivers are updated

2. **During Training**:
   - Don't run other GPU-intensive programs
   - Monitor first few hours to ensure stability
   - Check logs periodically for errors

3. **Power Management**:
   - Disable sleep mode
   - Disable hibernation
   - Set power plan to "High Performance"

```bash
# Windows: Disable sleep (run as admin)
powercfg -change -standby-timeout-ac 0
powercfg -change -hibernate-timeout-ac 0
```

4. **Thermal Management**:
   - Ensure good ventilation
   - Monitor GPU temperature (should stay under 85°C)
   - Clean dust from fans if needed

## Performance Expectations

- **LSTM Training**: ~1000 samples/sec on RTX GPU
- **Transformer Training**: ~500 samples/sec on RTX GPU
- **Memory Usage**: 4-8GB GPU memory per model
- **Disk Space**: ~50GB for logs and checkpoints

## Email Notifications (Optional)

To get notified when training completes or crashes, edit `train_50h_continuous.py` and add email configuration:

```python
# Add at top of file
import smtplib
from email.message import EmailMessage

# Add this function
def send_notification(subject, body):
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = 'your-email@gmail.com'
    msg['To'] = 'your-email@gmail.com'
    msg.set_content(body)

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login('your-email@gmail.com', 'your-app-password')
        smtp.send_message(msg)
```

## Support

For issues or questions:
1. Check logs in `training_logs/`
2. Check state in `continuous_training_checkpoints/training_state.json`
3. Review this documentation
4. Check individual model training scripts

## Performance Metrics

After training completes, check:
- `training_logs/training_50h_*.log` - Full training log
- `continuous_training_checkpoints/training_state.json` - Final state
- `models/` - Saved model files
- `checkpoints/` - Training checkpoints

## Next Steps After Training

1. **Evaluate Models**:
   ```bash
   python evaluate_models.py
   ```

2. **Backtest Results**:
   ```bash
   python backtest_models.py
   ```

3. **Deploy Best Model**:
   ```bash
   python deploy_production.py
   ```

Happy Training! 🚀
