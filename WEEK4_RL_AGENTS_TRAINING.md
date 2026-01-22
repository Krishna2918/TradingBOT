# WEEK 4 - REINFORCEMENT LEARNING AGENTS
## Dynamic Position Sizing & Advanced Decision Making

**Timeline**: Week 4 (7 days)
**Prerequisites**: LSTM (60-65%) + Transformer (65-70%) trained
**Current Status**: RL agents NOT trained (only architecture exists)
**Target**: 73-75% accuracy with RL-enhanced ensemble

---

## 🎯 WHY REINFORCEMENT LEARNING?

### What RL Adds to the System

**LSTM + Transformer** can predict:
- "AAPL will go UP tomorrow" (65-70% accuracy)

**RL Agents** add:
- "HOW MUCH should I invest?" (position sizing)
- "WHEN exactly to enter/exit?" (timing)
- "How to manage RISK?" (stop-loss, take-profit)
- "How to handle DRAWDOWNS?" (reduce exposure in volatile markets)

###  Problem: Prediction ≠ Profit

**Scenario 1: Same prediction, different outcomes**

LSTM/Transformer says: "AAPL will go up 2% tomorrow"

**Trader A** (no RL):
- Invests 100% of capital
- AAPL gaps down 1% at open (stop-loss triggered)
- Result: -1% loss despite correct prediction

**Trader B** (with RL):
- RL agent says "volatility high, invest only 30%"
- AAPL gaps down 1%, but position is small
- AAPL then goes up 3% (prediction correct)
- Result: +0.9% profit (30% × 3%)

**RL makes the difference between correct prediction and actual profit.**

---

## 🤖 RL AGENTS ARCHITECTURE

### Agent 1: PPO (Proximal Policy Optimization)

**Purpose**: Learn optimal position sizing

**Input**:
- LSTM prediction: [Buy=0.7, Hold=0.2, Sell=0.1]
- Transformer prediction: [Buy=0.8, Hold=0.15, Sell=0.05]
- Market features: volatility, volume, sector strength, SPY trend
- Portfolio state: current positions, cash, P&L, drawdown

**Output**:
- Position size: 0% to 100% of capital
- Confidence: 0.0 to 1.0

**Example**:
```python
Input:
  LSTM: 70% buy
  Transformer: 80% buy
  Volatility: High (VIX=25)
  Current drawdown: 5%

Output:
  Position size: 25% (reduced due to high volatility)
  Confidence: 0.75
```

### Agent 2: DQN (Deep Q-Network)

**Purpose**: Learn optimal entry/exit timing

**Input**:
- LSTM prediction stream (last 10 predictions)
- Transformer prediction stream (last 10 predictions)
- Price action: current price, recent moves, support/resistance
- Technical indicators: RSI, MACD, Bollinger Bands

**Output (Discrete Actions)**:
- **Action 0**: Wait (do nothing)
- **Action 1**: Enter long (buy)
- **Action 2**: Exit long (sell)
- **Action 3**: Tighten stop-loss
- **Action 4**: Take partial profit (sell 50%)

**Example**:
```python
Situation:
  LSTM: 65% buy (weak signal)
  Price: Near resistance level
  RSI: 70 (overbought)

DQN Decision: Action 0 (Wait)
Reason: Signal weak + overbought + resistance

---

Situation:
  LSTM: 70% buy (strong signal)
  Transformer: 75% buy (strong signal)
  Price: Breakout above resistance
  Volume: 2x average

DQN Decision: Action 1 (Enter long)
Reason: Strong signal + breakout + high volume
```

---

## 🏗️ RL TRAINING ENVIRONMENT

### Custom Trading Environment

```python
class TradingEnvironment(gym.Env):
    """
    Simulates real trading for RL agent training
    """

    def __init__(self, data, lstm_model, transformer_model):
        self.data = data  # 1,695 stocks, 20 years
        self.lstm = lstm_model
        self.transformer = transformer_model

        # Action space
        self.action_space = spaces.Box(
            low=0, high=1,  # Position size 0-100%
            shape=(1,)
        )

        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(150,)  # 150 features
        )

        # Trading parameters
        self.initial_capital = 100000
        self.commission = 0.001  # 0.1% per trade
        self.slippage = 0.0005   # 0.05% slippage

    def step(self, action):
        """
        Execute one trading step
        """
        # 1. Get predictions from LSTM/Transformer
        lstm_pred = self.lstm.predict(self.current_data)
        transformer_pred = self.transformer.predict(self.current_data)

        # 2. Execute action (position size from agent)
        position_size = action[0]
        shares = int(self.capital * position_size / self.current_price)

        # 3. Apply commission + slippage
        cost = shares * self.current_price * (1 + self.commission + self.slippage)

        # 4. Move to next day
        next_price = self.data[self.step_idx + 1]['close']

        # 5. Calculate reward
        pnl = shares * (next_price - self.current_price) - cost
        reward = pnl / self.capital  # Percentage return

        # 6. Apply penalties for risk
        if self.volatility > 0.03:  # High volatility
            reward *= 0.8  # 20% penalty

        if self.drawdown > 0.1:  # 10% drawdown
            reward *= 0.7  # 30% penalty

        return next_state, reward, done, info

    def reset(self):
        """
        Reset to start of episode
        """
        self.capital = self.initial_capital
        self.positions = []
        self.step_idx = 0
        return self.get_observation()
```

### Reward Function Design

**Simple reward** (percentage return):
```python
reward = (portfolio_value_today - portfolio_value_yesterday) / portfolio_value_yesterday
```

**Problem**: Encourages high risk

**Better reward** (risk-adjusted):
```python
# Sharpe ratio as reward
returns = portfolio_returns[-252:]  # Last year
reward = returns.mean() / returns.std() * np.sqrt(252)

# Penalty for drawdown
if current_drawdown > 0.1:
    reward *= 0.5

# Penalty for high volatility
if portfolio_volatility > 0.3:
    reward *= 0.7

# Bonus for consistent wins
if win_streak > 5:
    reward *= 1.2
```

---

## 📅 WEEK 4 DAILY PLAN

### Day 1 (Monday) - Environment Setup

**Morning**:
1. Create trading environment (`TradingEnvironment`)
2. Test environment with random actions
3. Verify rewards calculated correctly

**Afternoon**:
4. Create observation builder (150 features)
5. Test observation space
6. Integrate LSTM + Transformer predictions

**Commands**:
```bash
# Test environment
python test_trading_environment.py

# Verify observations
python verify_observations.py
```

### Day 2 (Tuesday) - PPO Training

**Morning**:
1. Configure PPO hyperparameters
2. Start PPO training (position sizing agent)
3. Monitor training (reward curve, position size distribution)

**Afternoon**:
4. Analyze initial PPO behavior
5. Adjust reward function if needed
6. Continue training

**Commands**:
```bash
# Train PPO
python train_ppo_agent.py \
  --env TradingEnvironment \
  --lstm models/lstm_1695_final.pth \
  --transformer models/transformer_1695_final.pth \
  --total-timesteps 1000000 \
  --learning-rate 0.0003 \
  --output models/ppo_agent_v1.zip
```

**Expected training time**: 6-10 hours

### Day 3 (Wednesday) - DQN Training

**Morning**:
1. Configure DQN hyperparameters
2. Start DQN training (entry/exit timing agent)
3. Monitor training (Q-values, action distribution)

**Afternoon**:
4. Analyze DQN behavior
5. Adjust epsilon (exploration vs exploitation)
6. Continue training

**Commands**:
```bash
# Train DQN
python train_dqn_agent.py \
  --env TradingEnvironment \
  --lstm models/lstm_1695_final.pth \
  --transformer models/transformer_1695_final.pth \
  --total-timesteps 500000 \
  --learning-rate 0.0001 \
  --buffer-size 100000 \
  --exploration-fraction 0.2 \
  --output models/dqn_agent_v1.zip
```

**Expected training time**: 4-8 hours

### Day 4 (Thursday) - Integration & Testing

**Morning**:
1. Integrate PPO + DQN + LSTM + Transformer
2. Create unified decision system
3. Test on validation set

**Afternoon**:
4. Analyze integrated system performance
5. Compare with baseline (LSTM + Transformer only)
6. Tune integration weights

### Day 5 (Friday) - Backtesting

**Morning**:
1. Comprehensive backtest on 20 years
2. Test on all 1,695 stocks
3. Analyze per-stock, per-sector, per-period

**Afternoon**:
4. Calculate all metrics
5. Compare: RL-enhanced vs Non-RL
6. Validate improvement

**Commands**:
```bash
# Backtest RL system
python backtest_rl_system.py \
  --lstm models/lstm_1695_final.pth \
  --transformer models/transformer_1695_final.pth \
  --ppo models/ppo_agent_final.zip \
  --dqn models/dqn_agent_final.zip \
  --data TrainingData/features/*.parquet \
  --start-date 2000-01-01 \
  --end-date 2025-10-28 \
  --initial-capital 100000 \
  --output results/rl_system_backtest.json
```

### Day 6-7 (Weekend) - Optimization & Documentation

**Saturday**:
1. Hyperparameter optimization for PPO + DQN
2. Test different reward functions
3. A/B testing: Which RL configuration is best?

**Sunday**:
4. Documentation (RL agent specifications, training logs)
5. Create performance dashboard
6. Plan Week 5 (Final ensemble + portfolio optimization)

---

## 📊 EXPECTED RESULTS

### Performance Improvement

| Metric | LSTM+Transformer | +RL Agents | Improvement |
|--------|------------------|------------|-------------|
| **Accuracy** | 65-70% | 73-75% | +8-10% |
| **Sharpe Ratio** | 1.7-2.0 | 2.2-2.5 | +29-47% |
| **Max Drawdown** | 10-13% | 7-10% | -30-38% |
| **Win Rate** | 57-60% | 61-64% | +7-11% |
| **Profit Factor** | 1.5-1.7 | 1.9-2.2 | +27-44% |
| **Calmar Ratio** | 1.5-1.8 | 2.0-2.5 | +33-67% |

### Why RL Improves Performance

**1. Dynamic Position Sizing**:
- High confidence + low volatility → Large position (80-100%)
- Low confidence + high volatility → Small position (10-30%)
- Result: Better risk-adjusted returns

**2. Optimal Timing**:
- Wait for confirmation (avoid false signals)
- Enter on breakouts (catch trends early)
- Exit before reversals (avoid giving back profits)
- Result: Higher win rate, better entries/exits

**3. Risk Management**:
- Reduce exposure in drawdowns (preserve capital)
- Increase exposure in winning streaks (maximize profits)
- Auto-adjust stop-loss based on volatility
- Result: Lower drawdown, smoother equity curve

**4. Adaptability**:
- Learn from mistakes (trial and error)
- Adapt to changing market conditions
- Discover non-obvious strategies
- Result: More robust system

---

## 🛠️ FILES TO CREATE

### Week 4 Scripts

1. **trading_environment.py** (Day 1)
   - Custom OpenAI Gym environment
   - Simulates realistic trading
   - Integrates LSTM + Transformer predictions

2. **train_ppo_agent.py** (Day 2)
   - PPO training pipeline
   - Position sizing agent
   - Monitoring, checkpointing

3. **train_dqn_agent.py** (Day 3)
   - DQN training pipeline
   - Entry/exit timing agent
   - Replay buffer, target network

4. **integrate_rl_system.py** (Day 4)
   - Unified decision system
   - PPO + DQN + LSTM + Transformer
   - Integration logic

5. **backtest_rl_system.py** (Day 5)
   - Comprehensive backtesting
   - 20 years, 1,695 stocks
   - All metrics

### Modified Existing Files

6. **src/ai/agents/ppo_agent.py**
   - Update for 1,695 stocks
   - Add reward shaping
   - Add monitoring hooks

7. **src/ai/agents/dqn_agent.py**
   - Update for 1,695 stocks
   - Add experience replay
   - Add target network

---

## 🚨 POTENTIAL ISSUES & SOLUTIONS

### Issue 1: RL Not Improving Performance

**Symptoms**: RL-enhanced accuracy 66%, Non-RL 67%

**Solutions**:
1. **Reward function**: May be poorly designed, try different rewards
2. **Training time**: May need more timesteps (1M → 5M)
3. **Observation space**: May be missing key features
4. **Hyperparameters**: Tune learning rate, discount factor

### Issue 2: RL Agent Too Conservative

**Symptoms**: Position sizes always <20%, missing profits

**Solutions**:
1. **Reduce risk penalty**: Current reward over-penalizes risk
2. **Adjust reward**: Increase return weight, decrease risk weight
3. **Change exploration**: More exploration (higher epsilon)

### Issue 3: RL Agent Too Aggressive

**Symptoms**: Position sizes always >80%, high drawdown

**Solutions**:
1. **Increase risk penalty**: Current reward under-penalizes risk
2. **Add drawdown penalty**: Harsh penalty for >15% drawdown
3. **Add volatility penalty**: Reduce size in volatile markets

### Issue 4: Training Unstable

**Symptoms**: Reward curve oscillates wildly, never converges

**Solutions**:
1. **Reduce learning rate**: 0.0003 → 0.0001
2. **Increase batch size**: 64 → 128 → 256
3. **Clip gradients**: Prevent exploding gradients
4. **Use PPO instead of DQN**: PPO more stable

---

## 💡 PRO TIPS

### Tip 1: Start with PPO, not DQN
PPO is more stable and easier to tune. DQN can be tricky for beginners.

### Tip 2: Reward Shaping is Critical
RL is 80% reward function design, 20% algorithm. Spend time on reward function.

### Tip 3: Visualize Agent Behavior
Plot position sizes, actions, Q-values. If behavior looks random, agent isn't learning.

### Tip 4: Use Stable-Baselines3
Don't implement RL from scratch. Use stable-baselines3 library (battle-tested).

### Tip 5: Train on Subset First
Test on 100 stocks first (1 hour), then scale to 1,695 stocks (10 hours).

---

## 🎯 SUCCESS CRITERIA

### Minimum Viable Model
- ✅ RL-enhanced accuracy ≥ 68% (beats non-RL 65-70%)
- ✅ Sharpe ratio ≥ 2.0
- ✅ Max drawdown ≤ 12%
- ✅ Profit factor ≥ 1.7

### Target Model
- ✅ RL-enhanced accuracy ≥ 73% (+8% over non-RL)
- ✅ Sharpe ratio ≥ 2.2
- ✅ Max drawdown ≤ 10%
- ✅ Profit factor ≥ 1.9

### Stretch Goal
- 🎯 RL-enhanced accuracy ≥ 75% (+10% over non-RL)
- 🎯 Sharpe ratio ≥ 2.5
- 🎯 Max drawdown ≤ 7%
- 🎯 Profit factor ≥ 2.2

---

## 🧠 RL ALGORITHMS EXPLAINED

### PPO (Proximal Policy Optimization)

**How it works**:
1. Collect experience (state, action, reward) for N steps
2. Calculate advantage (how much better action was vs expected)
3. Update policy to increase probability of good actions
4. Use "clipping" to prevent too-large updates (proximal = nearby)

**Why PPO?**
- Stable (won't suddenly forget what it learned)
- Sample efficient (learns from less data)
- Easy to tune (fewer hyperparameters)
- Industry standard (used by OpenAI, DeepMind)

**Best for**: Continuous actions (position sizing 0-100%)

### DQN (Deep Q-Network)

**How it works**:
1. Learn Q-function: Q(state, action) = expected future reward
2. Choose action with highest Q-value: argmax Q(state, a)
3. Use experience replay (learn from past experiences)
4. Use target network (stable training target)

**Why DQN?**
- Good for discrete actions (buy, hold, sell)
- Can learn complex strategies
- Off-policy (learn from any data, not just current policy)

**Best for**: Discrete actions (wait, enter, exit, etc.)

### A3C (Asynchronous Advantage Actor-Critic)

**Alternative to PPO/DQN** (if you want to explore):
- Trains multiple agents in parallel (faster)
- Good for multi-stock trading
- More complex to implement

---

## 🔄 WEEK 4 CHECKLIST

### Day 1 Checklist
- [ ] Create trading environment
- [ ] Test environment with random actions
- [ ] Verify reward calculation
- [ ] Create observation builder (150 features)
- [ ] Integrate LSTM + Transformer

### Day 2 Checklist
- [ ] Configure PPO hyperparameters
- [ ] Start PPO training
- [ ] Monitor reward curve
- [ ] Analyze position size distribution

### Day 3 Checklist
- [ ] Configure DQN hyperparameters
- [ ] Start DQN training
- [ ] Monitor Q-values
- [ ] Analyze action distribution

### Day 4 Checklist
- [ ] Integrate PPO + DQN + LSTM + Transformer
- [ ] Test integrated system
- [ ] Compare with baseline
- [ ] Tune integration weights

### Day 5 Checklist
- [ ] Comprehensive backtest (20 years, 1,695 stocks)
- [ ] Calculate all metrics
- [ ] Compare RL-enhanced vs Non-RL
- [ ] Validate improvement ≥+8%

### Day 6-7 Checklist
- [ ] Hyperparameter optimization
- [ ] Test different reward functions
- [ ] A/B testing
- [ ] Documentation
- [ ] Performance dashboard
- [ ] Plan Week 5

---

## 🚀 NEXT: WEEK 5 - FINAL ENSEMBLE

After RL training, Week 5 focuses on:

1. **Meta-Ensemble**: Combine all models (LSTM, Transformer, PPO, DQN)
2. **Portfolio Optimization**: Multi-stock portfolio with correlation analysis
3. **Risk Management**: Advanced stop-loss, position limits, drawdown control
4. **Production Deployment**: Dockerize, create API, monitoring

**Expected final accuracy**: 75-78% (ensemble of all models)

---

*Week 4 RL Agents Training Plan v1.0 - October 28, 2025*
