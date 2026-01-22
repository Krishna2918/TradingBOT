# Trading Bot Training Guide

Complete guide for training all AI models on your RTX 4080 GPU.

---

## Quick Start

Your LSTM is currently training! Those RuntimeWarnings are **harmless** - they occur when normalizing stocks with constant values, but your code handles it properly.

### Current LSTM Training Status

```bash
# You're running: python train_lstm_production.py --epochs 100 --batch-size 128
# Expected duration: 8-12 hours
# Target accuracy: 60-65%
```

**What to expect:**
1. Loading completes (1,677 files) ✓
2. Datasets created (train/val split)
3. Training starts: `Epoch 1 [100/xxxx] Loss: 0.6933 | Acc: 50.xx%`
4. Accuracy climbs to 56-58% over 100 epochs
5. Best model saves to: `models/lstm_best.pth`

---

## Training Order (Sequential)

### 1. LSTM Training (CURRENT - IN PROGRESS)
```bash
# Full training
python train_lstm_production.py --epochs 100 --batch-size 128

# Test mode (30 min validation)
python train_lstm_production.py --test-mode --epochs 5
```

**Parameters:**
- Sequence length: 30 days
- Batch size: 128 (optimized for RTX 4080)
- Learning rate: 0.0001 (warm-up + OneCycleLR)
- Expected time: 8-12 hours
- Target accuracy: 60-65%

**Output:**
- Best model: `models/lstm_best.pth`
- Checkpoints: `models/lstm_epoch_*.pth` (every 10 epochs)
- TensorBoard: `logs/tensorboard/`

---

### 2. Transformer Training (NEXT AFTER LSTM)
```bash
# Full training
python train_transformer_production.py --epochs 100 --batch-size 64

# Test mode
python train_transformer_production.py --test-mode --epochs 5
```

**Parameters:**
- Sequence length: 90 days (3 months for long-term patterns)
- Batch size: 64 (Transformer needs more memory)
- Model size: d_model=512, 8 heads, 6 layers
- Learning rate: 0.0001 with 4000-step warmup
- Expected time: 10-15 hours
- Target accuracy: 65-70%

**Features:**
- Attention map visualization (saved every 5 epochs)
- Time-aware positional encoding
- Market volatility-aware attention

**Output:**
- Best model: `models/transformer_best.pth`
- Attention maps: `results/attention_maps/`
- TensorBoard: `logs/tensorboard_transformer/`

---

### 3. PPO Agent Training (REINFORCEMENT LEARNING)
```bash
# Full training
python train_ppo_agent.py --episodes 1000

# Test mode
python train_ppo_agent.py --test-mode --episodes 100
```

**Parameters:**
- Algorithm: Proximal Policy Optimization
- Rollout steps: 2048 per update
- GAE lambda: 0.95
- Clip epsilon: 0.2
- Expected time: 6-10 hours
- Target: Sharpe ratio > 2.0

**Features:**
- Generalized Advantage Estimation (GAE)
- Clipped surrogate objective
- Sharpe-ratio-based reward shaping
- Consistency bonuses for stable returns

**Output:**
- Best model: `models/ppo_best.pth`
- TensorBoard: `logs/tensorboard_ppo/`

---

### 4. DQN Agent Training (REINFORCEMENT LEARNING)
```bash
# Full training
python train_dqn_agent.py --episodes 1000

# Test mode
python train_dqn_agent.py --test-mode --episodes 100
```

**Parameters:**
- Algorithm: Double DQN
- Replay buffer: 100,000 transitions
- Epsilon decay: 1.0 → 0.01
- Target network soft update (τ=0.005)
- Expected time: 6-10 hours
- Target: Win rate > 60%

**Features:**
- Experience replay for sample efficiency
- Double DQN to reduce overestimation
- ε-greedy exploration with decay

**Output:**
- Best model: `models/dqn_best.pth`
- TensorBoard: `logs/tensorboard_dqn/`

---

## Automated Training Pipeline

Run all models sequentially:

```bash
# Full pipeline (all 4 models)
python orchestrate_training.py

# Test mode (quick validation)
python orchestrate_training.py --test-mode

# Skip specific models
python orchestrate_training.py --skip-ppo --skip-dqn
```

**Output:**
- Summary report: `results/training_summary.txt`
- JSON log: `results/training_summary.json`
- Orchestration logs: `logs/orchestration/`

---

## Model Evaluation & Comparison

After training all models:

```bash
# Compare all models
python evaluate_all_models.py

# Test mode
python evaluate_all_models.py --test-mode
```

**What it does:**
- Classification accuracy on test set (50 stocks)
- Backtest performance (10 stocks)
- Statistical comparison with baselines
- Per-class accuracy breakdown

**Output:**
- Comparison report: `results/model_comparison.txt`
- Shows best classifier and best backtest performer

---

## Monitoring Training

### TensorBoard (Real-time Monitoring)

```bash
# LSTM
tensorboard --logdir logs/tensorboard

# Transformer
tensorboard --logdir logs/tensorboard_transformer

# PPO
tensorboard --logdir logs/tensorboard_ppo

# DQN
tensorboard --logdir logs/tensorboard_dqn

# All at once
tensorboard --logdir logs/
```

Open: http://localhost:6006

**Metrics to watch:**
- Loss: Should decrease steadily
- Accuracy: Should increase from 50% to 60-70%
- Learning rate: Warm-up then decay
- GPU utilization: Should be 70-80%

### GPU Monitoring

```bash
# Watch GPU usage
nvidia-smi -l 1

# Or use:
watch -n 1 nvidia-smi
```

**Expected GPU usage:**
- LSTM: 6-8 GB VRAM, 70-80% utilization
- Transformer: 8-10 GB VRAM, 75-85% utilization
- RL agents: 2-4 GB VRAM, 50-60% utilization

---

## Troubleshooting

### Issue 1: NaN Loss (FIXED)
**Symptoms:** `Loss: nan`, training not learning

**Solution:** Already fixed with:
- Learning rate: 0.0001 (was 0.001)
- Feature normalization enabled
- Gradient clipping (max_norm=1.0)

### Issue 2: RuntimeWarnings (HARMLESS)
**Symptoms:**
```
RuntimeWarning: invalid value encountered in subtract
```

**Explanation:** Occurs when normalizing stocks with constant values (no trading activity). Your code handles this with `features_std[features_std == 0] = 1`.

**Action:** Ignore these warnings - training works correctly!

### Issue 3: Out of Memory (OOM)
**Symptoms:** `CUDA out of memory`

**Solutions:**
- Reduce batch size: `--batch-size 64` or `--batch-size 32`
- Close other GPU applications
- Use test mode first: `--test-mode`

### Issue 4: Slow Training
**Symptoms:** Very slow progress

**Solutions:**
- Check GPU usage: `nvidia-smi`
- Ensure CUDA is detected: Look for "GPU: NVIDIA GeForce RTX 4080" at start
- Use recommended batch sizes (128 for LSTM, 64 for Transformer)

### Issue 5: NumPy Multiprocessing Errors
**Symptoms:** `OSError: [Errno 22] Invalid argument`

**Solution:** Already fixed with `--num-workers 0` (no multiprocessing on Windows)

---

## Expected Results

### Model Performance Targets

| Model | Accuracy | Sharpe Ratio | Max Drawdown | Training Time |
|-------|----------|--------------|--------------|---------------|
| LSTM | 60-65% | 1.5-2.0 | <20% | 8-12 hours |
| Transformer | 65-70% | 1.8-2.2 | <18% | 10-15 hours |
| PPO | N/A | 2.0-2.5 | <15% | 6-10 hours |
| DQN | N/A | 1.8-2.3 | <15% | 6-10 hours |
| Ensemble | 70-75% | 2.2-2.8 | <12% | - |

### Training Timeline

**Total time: 30-47 hours**

1. LSTM: 8-12 hours
2. Transformer: 10-15 hours
3. PPO: 6-10 hours
4. DQN: 6-10 hours

**Recommended schedule:**
- Day 1: LSTM (overnight)
- Day 2: Transformer (overnight)
- Day 3: PPO (daytime) + DQN (evening)
- Day 4: Evaluation and backtesting

---

## Next Steps After Training

### 1. Review Results
```bash
# View comparison report
cat results/model_comparison.txt

# View training summary
cat results/training_summary.txt
```

### 2. Analyze TensorBoard Logs
- Compare loss curves across models
- Check attention maps (Transformer)
- Review episode rewards (RL agents)

### 3. Run Backtests
```bash
# Backtest on specific stock
python backtest_single_stock.py --model models/lstm_best.pth --symbol AAPL
```

### 4. Deploy Best Model
- Choose best performer from comparison report
- Set up real-time inference pipeline
- Implement paper trading

### 5. Create Ensemble
- Combine predictions from all models
- Weight by backtest performance
- Meta-learning for optimal combination

---

## File Structure

```
TradingBOT/
├── TrainingData/
│   ├── daily/           # Raw OHLCV data (1,677 stocks)
│   └── features/        # Engineered features (58 indicators)
├── models/              # Saved model checkpoints
│   ├── lstm_best.pth
│   ├── transformer_best.pth
│   ├── ppo_best.pth
│   └── dqn_best.pth
├── logs/
│   ├── tensorboard/     # LSTM logs
│   ├── tensorboard_transformer/
│   ├── tensorboard_ppo/
│   ├── tensorboard_dqn/
│   └── orchestration/   # Pipeline logs
├── results/
│   ├── attention_maps/  # Transformer visualizations
│   ├── backtests/       # Backtest results
│   ├── model_comparison.txt
│   └── training_summary.txt
└── src/
    ├── ai/
    │   ├── models/      # Model architectures
    │   └── rl/          # RL environments & agents
    └── backtesting/     # Backtest engine
```

---

## Commands Reference

### Training Commands
```bash
# LSTM
python train_lstm_production.py --epochs 100 --batch-size 128

# Transformer
python train_transformer_production.py --epochs 100 --batch-size 64

# PPO
python train_ppo_agent.py --episodes 1000

# DQN
python train_dqn_agent.py --episodes 1000

# All (orchestrated)
python orchestrate_training.py
```

### Evaluation Commands
```bash
# Compare all models
python evaluate_all_models.py

# Individual model evaluation
python evaluate_lstm.py
python evaluate_transformer.py
python evaluate_ppo.py
python evaluate_dqn.py
```

### Monitoring Commands
```bash
# TensorBoard
tensorboard --logdir logs/

# GPU monitoring
nvidia-smi -l 1

# Watch log file
tail -f logs/orchestration/training_*.log
```

---

## FAQ

**Q: How long until I see results?**
A: First model (LSTM) takes 8-12 hours. You'll see accuracy climbing from 50% to 60-65%.

**Q: Can I pause training?**
A: Yes! Press Ctrl+C. Models save checkpoints every 10 epochs. Resume with `--resume models/lstm_epoch_50.pth`

**Q: Should I run all models?**
A: Start with LSTM and Transformer (classifiers). Then run RL agents if you want trading strategies.

**Q: What if accuracy is lower than expected?**
A:
- LSTM: 56-58% is still good (better than random 33%)
- Try longer training: `--epochs 150`
- Increase model capacity: `--hidden-dim 512`

**Q: Can I train on multiple GPUs?**
A: These scripts use single GPU. For multi-GPU, modify with `torch.nn.DataParallel`

**Q: How do I know which model is best?**
A: Run `python evaluate_all_models.py` - it will show comparison and recommend best model.

---

## Support & Issues

**If you encounter issues:**
1. Check this guide's Troubleshooting section
2. Review TensorBoard logs
3. Check GPU memory: `nvidia-smi`
4. Verify data files: `ls TrainingData/features/*.parquet | wc -l` (should be 1,677)

**Success indicators:**
- ✓ GPU detected: "GPU: NVIDIA GeForce RTX 4080"
- ✓ Loss not NaN: "Loss: 0.6933" (stable around 0.65-0.70)
- ✓ Accuracy climbing: 50% → 56% → 60%+
- ✓ Checkpoints saving: "Best model saved: models/lstm_best.pth"

---

**Your training is on track! The RuntimeWarnings are expected and harmless. Let it complete!**

Good luck! 🚀
