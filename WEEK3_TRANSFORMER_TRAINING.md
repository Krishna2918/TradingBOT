# WEEK 3 - TRANSFORMER MODEL TRAINING
## Target: 65-70% Accuracy (vs LSTM 60-65%)

**Timeline**: Week 3 (7 days)
**Prerequisites**: LSTM retrained on 1,695 stocks (Week 2 complete)
**Current Status**: Transformer NOT trained (only architecture exists)
**Target**: 65-70% accuracy, beat LSTM by 5-10%

---

## 🎯 WHY TRANSFORMER?

### LSTM vs Transformer

**LSTM Strengths**:
- Good at sequential patterns
- Memory of past events
- Works well with 30-60 day windows

**LSTM Weaknesses**:
- Short memory (vanishing gradients)
- Sequential processing (slow)
- Can't capture long-term dependencies (>60 days)

**Transformer Strengths**:
- Attention mechanism (sees entire history)
- Parallel processing (fast training)
- Long-term dependencies (100+ days)
- Better at complex patterns

**Why Transformer beats LSTM**:
- Stock patterns often span months (earnings cycles, seasonal trends)
- Attention mechanism finds what matters (e.g., last earnings report more important than random day)
- Parallel training = faster experiments

**Expected improvement**: +5-10% accuracy (60% → 65-70%)

---

## 📊 TRANSFORMER ARCHITECTURE

### Model Configuration

```python
{
  "d_model": 512,           # Embedding dimension
  "nhead": 8,               # Number of attention heads
  "num_encoder_layers": 6,  # Depth of encoder
  "dim_feedforward": 2048,  # Hidden layer size
  "dropout": 0.1,           # Regularization
  "sequence_length": 90,    # Days to look back (3 months)
  "batch_size": 32,
  "learning_rate": 0.0001,
  "epochs": 100,
  "warmup_steps": 4000      # Learning rate warm-up
}
```

### Why These Numbers?

- **d_model=512**: Standard for Transformer, balances capacity vs speed
- **nhead=8**: Multi-head attention splits into 8 parallel attention mechanisms
- **num_encoder_layers=6**: Depth for complex pattern learning
- **sequence_length=90**: 3 months captures earnings cycles, seasonal patterns
- **warmup_steps=4000**: Gradually increase LR (Transformer training trick)

---

## 📅 WEEK 3 DAILY PLAN

### Day 1 (Monday) - Architecture Setup

**Morning**:
1. Review existing Transformer code (`src/ai/models/transformer_model.py`)
2. Update architecture for 1,695 stocks
3. Test forward pass (ensure no errors)

**Afternoon**:
4. Create data loader for Transformer (90-day sequences)
5. Test data pipeline end-to-end
6. Create training script (`transformer_trainer.py`)

**Commands**:
```bash
# Test architecture
python -c "from src.ai.models.transformer_model import TransformerModel; \
           model = TransformerModel(input_dim=95, d_model=512, nhead=8); \
           print('Architecture OK')"

# Test data loader
python test_transformer_dataloader.py
```

### Day 2 (Tuesday) - Initial Training

**Morning**:
1. Start training run #1 (baseline configuration)
2. Monitor training (loss, accuracy, attention weights)

**Afternoon**:
3. Analyze initial results
4. Adjust hyperparameters if needed
5. Start training run #2

**Commands**:
```bash
# Training run #1
python transformer_trainer.py \
  --data TrainingData/features/*.parquet \
  --output models/transformer_1695_v1.pth \
  --d-model 512 \
  --nhead 8 \
  --layers 6 \
  --sequence-length 90 \
  --batch-size 32 \
  --learning-rate 0.0001 \
  --epochs 100
```

**Expected training time**: 10-15 hours

### Day 3 (Wednesday) - Attention Analysis

**Morning**:
1. Visualize attention weights (what does model focus on?)
2. Analyze which features get most attention
3. Identify patterns (e.g., model focuses on earnings dates, Fed announcements)

**Afternoon**:
4. Feature engineering based on attention insights
5. Add high-attention features, remove low-attention features
6. Start training run #3 with improved features

**Tools**:
```bash
# Visualize attention
python visualize_transformer_attention.py \
  --model models/transformer_1695_v1.pth \
  --stock AAPL \
  --date 2024-01-15 \
  --output results/attention_maps/
```

### Day 4 (Thursday) - Hyperparameter Tuning

**Morning**:
1. Grid search on key hyperparameters:
   - d_model: [256, 512, 1024]
   - nhead: [4, 8, 16]
   - layers: [4, 6, 8]
   - sequence_length: [60, 90, 120]

**Afternoon**:
2. Run best 3-5 configurations
3. Compare results
4. Select best model

### Day 5 (Friday) - Final Training & Evaluation

**Morning**:
1. Train final Transformer with best configuration
2. Train on full train+validation set

**Afternoon**:
3. Evaluate on test set
4. Compare with LSTM:
   - Accuracy: Transformer vs LSTM
   - Sharpe: Transformer vs LSTM
   - Per-sector: Which model wins where?

**Commands**:
```bash
# Evaluation
python evaluate_transformer_1695.py \
  --model models/transformer_1695_final.pth \
  --lstm-model models/lstm_1695_final.pth \
  --test-data TrainingData/features/test/*.parquet \
  --output results/transformer_vs_lstm.json
```

### Day 6-7 (Weekend) - Ensemble Preparation

**Saturday**:
1. Create ensemble combining LSTM + Transformer
2. Simple averaging: 50% LSTM + 50% Transformer
3. Weighted averaging: Optimize weights (e.g., 40% LSTM + 60% Transformer)

**Sunday**:
4. Evaluate ensemble performance
5. Compare: Ensemble vs LSTM vs Transformer
6. Document results, plan Week 4 (RL agents)

---

## 🔍 ATTENTION MECHANISM EXPLAINED

### What is Attention?

Traditional LSTM:
```
Day 1 → Day 2 → Day 3 → ... → Day 90 → Prediction
         ↓        ↓              ↓
      Forgets  Forgets      Remembers
```

Transformer with Attention:
```
Day 1 ──┐
Day 2 ──┤
Day 3 ──┤
...     ├──→ Attention ──→ Prediction
Day 88──┤      Weights
Day 89──┤       ↓
Day 90──┘    Which days
             matter most?
```

### Example: Earnings Attention

Stock: AAPL
Date: 2024-11-02 (day after earnings)

Attention weights might look like:
```
Oct 1:  0.01  (not important)
Oct 2:  0.01
...
Oct 31: 0.35  (earnings announcement day!)
Nov 1:  0.20  (reaction day)
Nov 2:  0.10  (current day)
Other days: 0.33 total
```

**Insight**: Model learns earnings dates are critical!

### Multi-Head Attention (8 heads)

Each head focuses on different patterns:
- **Head 1**: Earnings dates
- **Head 2**: Fed announcement dates
- **Head 3**: Volume spikes
- **Head 4**: Price gaps
- **Head 5**: Sector rotation
- **Head 6**: Technical breakouts
- **Head 7**: Volatility spikes
- **Head 8**: Correlation with SPY

This is why Transformer outperforms LSTM - it can focus on multiple patterns simultaneously!

---

## 📊 EXPECTED RESULTS

### Performance Comparison

| Metric | LSTM (Week 2) | Transformer (Week 3) | Improvement |
|--------|---------------|----------------------|-------------|
| **Accuracy** | 60-65% | 65-70% | +5-10% |
| **Precision** | 62-67% | 67-72% | +5-8% |
| **Recall** | 58-63% | 63-68% | +5-8% |
| **Sharpe Ratio** | 1.5-1.8 | 1.7-2.0 | +13-22% |
| **Max Drawdown** | 12-15% | 10-13% | -17-20% |
| **Win Rate** | 55-58% | 57-60% | +4-7% |
| **Training Time** | 8-12 hours | 10-15 hours | +20-25% |

### Why Transformer Beats LSTM

1. **Long-term dependencies**: Captures patterns over 90+ days (earnings cycles, seasonal trends)
2. **Attention mechanism**: Focuses on important events (earnings, Fed, etc.)
3. **Parallel processing**: Faster training, more experiments
4. **Better generalization**: Multi-head attention learns diverse patterns

### When LSTM Might Win

- **Very short-term** (<5 days): LSTM better at immediate momentum
- **High-frequency**: Intraday trading where long-term doesn't matter
- **Simple patterns**: If patterns are purely sequential (rare in stocks)

**For most cases (swing trading, 1-30 day holds): Transformer wins**

---

## 🛠️ FILES TO CREATE

### Week 3 Scripts

1. **transformer_trainer.py** (Day 1)
   - Training pipeline for Transformer
   - Learning rate scheduler with warm-up
   - Checkpointing, early stopping

2. **test_transformer_dataloader.py** (Day 1)
   - Test data pipeline
   - Verify 90-day sequences created correctly
   - Check batch shapes, no data leakage

3. **visualize_transformer_attention.py** (Day 3)
   - Visualize attention weights
   - Identify which features/days model focuses on
   - Output heatmaps, interactive plots

4. **evaluate_transformer_1695.py** (Day 5)
   - Comprehensive evaluation
   - Compare with LSTM
   - Per-sector, per-market-condition analysis

5. **ensemble_simple.py** (Day 6)
   - Simple LSTM + Transformer ensemble
   - Averaging, weighted averaging
   - Optimization of ensemble weights

### Modified Existing Files

6. **src/ai/models/transformer_model.py**
   - Update for 1,695 stocks
   - Add positional encoding
   - Add attention visualization hooks

---

## 🚨 POTENTIAL ISSUES & SOLUTIONS

### Issue 1: Training Even Slower Than LSTM

**Symptoms**: Epoch time >2 hours, expected completion >5 days

**Solutions**:
1. **Reduce sequence length**: 90 → 60 days
2. **Reduce d_model**: 512 → 256
3. **Reduce layers**: 6 → 4
4. **Use GPU**: Transformers NEED GPU (10-50x faster)

**GPU Options**:
- **Free**: Google Colab (limited to 12 hours/session)
- **Paid**: Lambda Labs ($0.50/hr for A100), AWS, GCP
- **Local**: Buy RTX 4070 ($600) or rent

### Issue 2: Out of Memory (OOM)

**Symptoms**: Training crashes with "CUDA out of memory"

**Solutions**:
1. **Reduce batch size**: 32 → 16 → 8
2. **Reduce sequence length**: 90 → 60
3. **Reduce d_model**: 512 → 256
4. **Gradient checkpointing**: Trade compute for memory

### Issue 3: Not Beating LSTM

**Symptoms**: Transformer accuracy 62%, LSTM 63%

**Solutions**:
1. **Increase sequence length**: 90 → 120 days (capture more context)
2. **Add positional encoding**: Improve time-awareness
3. **Tune learning rate**: Use warm-up, cosine annealing
4. **Feature engineering**: Add features that benefit from long-term context

### Issue 4: Overfitting

**Symptoms**: Train 75%, Validation 60%

**Solutions**:
1. **Increase dropout**: 0.1 → 0.2 → 0.3
2. **Add label smoothing**: Reduce overconfidence
3. **Data augmentation**: Add noise, time shifts
4. **Reduce capacity**: Fewer layers or smaller d_model

---

## 💡 PRO TIPS

### Tip 1: Transformers Need GPUs
Don't train Transformer on CPU. It will take days/weeks. Rent GPU for $10-20 total.

### Tip 2: Warm-up Learning Rate
Transformers are sensitive to LR. Use warm-up: Start low (1e-7), increase to peak (1e-4) over 4000 steps.

### Tip 3: Visualize Attention
Attention weights tell you what model learned. If attention is random, model isn't learning.

### Tip 4: Ensemble is King
LSTM + Transformer ensemble often beats both individual models. Plan for ensemble.

### Tip 5: Compare Apples to Apples
Test both models on exact same test set, same metrics. Fair comparison.

---

## 🎯 SUCCESS CRITERIA

### Minimum Viable Model
- ✅ Accuracy ≥ 63% (beats LSTM 60%)
- ✅ Sharpe ratio ≥ 1.6
- ✅ No severe overfitting (train-val gap <10%)

### Target Model
- ✅ Accuracy ≥ 65% (beats LSTM by 5%)
- ✅ Sharpe ratio ≥ 1.7
- ✅ Win rate ≥ 57%
- ✅ Beats LSTM on at least 70% of stocks

### Stretch Goal
- 🎯 Accuracy ≥ 70% (beats LSTM by 10%)
- 🎯 Sharpe ratio ≥ 2.0
- 🎯 Win rate ≥ 60%
- 🎯 Beats LSTM on 80%+ of stocks

---

## 📈 ARCHITECTURE VISUALIZATION

```
Input (90 days × 95 features)
    ↓
Embedding Layer (d_model=512)
    ↓
Positional Encoding (sine/cosine)
    ↓
┌─────────────────────────────┐
│  Encoder Layer 1            │
│  ┌──────────────────────┐   │
│  │  Multi-Head Attention │   │
│  │  (8 heads)           │   │
│  └──────────────────────┘   │
│           ↓                  │
│  ┌──────────────────────┐   │
│  │  Feed Forward        │   │
│  │  (2048 hidden)       │   │
│  └──────────────────────┘   │
└─────────────────────────────┘
    ↓
  (Repeat 6 times)
    ↓
Global Average Pooling
    ↓
Linear Layer (512 → 3)
    ↓
Softmax
    ↓
Output: [Buy, Hold, Sell]
```

---

## 🔄 WEEK 3 CHECKLIST

### Day 1 Checklist
- [ ] Review existing Transformer code
- [ ] Update architecture for 1,695 stocks
- [ ] Test forward pass
- [ ] Create data loader (90-day sequences)
- [ ] Create training script

### Day 2 Checklist
- [ ] Start training run #1
- [ ] Monitor training progress
- [ ] Analyze initial results
- [ ] Start training run #2 (if needed)

### Day 3 Checklist
- [ ] Visualize attention weights
- [ ] Identify high-attention features
- [ ] Feature engineering based on attention
- [ ] Start training run #3

### Day 4 Checklist
- [ ] Hyperparameter grid search
- [ ] Run best 3-5 configurations
- [ ] Select best model

### Day 5 Checklist
- [ ] Train final Transformer
- [ ] Evaluate on test set
- [ ] Compare with LSTM
- [ ] Document results

### Day 6-7 Checklist
- [ ] Create simple ensemble (LSTM + Transformer)
- [ ] Evaluate ensemble performance
- [ ] Compare: Ensemble vs LSTM vs Transformer
- [ ] Plan Week 4 (RL agents)

---

## 🚀 NEXT: WEEK 4 - RL AGENTS

After Transformer training, Week 4 focuses on:

1. **PPO Agent**: Learn dynamic position sizing
2. **DQN Agent**: Learn entry/exit timing
3. **RL Ensemble**: Combine LSTM + Transformer + RL
4. **Risk Management**: Automatic stop-loss, take-profit

**Expected improvement**: +3-5% accuracy (70% → 73-75% with RL)

---

*Week 3 Transformer Training Plan v1.0 - October 28, 2025*
