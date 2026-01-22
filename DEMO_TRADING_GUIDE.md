# 🎮 Demo Trading Mode - Complete Guide

## ✅ What You Now Have

A **complete practice/demo trading system** with:
- ✅ **Real Canadian market data** (live prices from TSX via Yahoo Finance)
- ✅ **AI-controlled trading** (5 strategies running automatically)
- ✅ **$50,000 CAD starting capital** (simulated)
- ✅ **1-week trial period**
- ✅ **Real-time dashboard** (Groww-style UI)
- ✅ **Full risk management** (stop loss, take profit, position sizing)

---

## 🚀 Quick Start

### Start Demo System:
```bash
python src/dashboard/demo_dashboard.py
```

Then open: **http://localhost:8051**

---

## 📊 System Architecture

### 1. **Real Market Data**
- **Source**: Yahoo Finance API (free, real-time)
- **Markets**: TSX/TSXV Canadian stocks
- **Symbols Tracked**:
  - Large caps: RY.TO, TD.TO, SHOP.TO, CNR.TO, ENB.TO, etc.
  - Mid caps: WCN.TO, AEM.TO, QSR.TO, ATD.TO, NTR.TO
- **Update Frequency**: Every 5 seconds

### 2. **AI Trading Strategies** (All Active)

| Strategy | Capital | Max Positions | Holding Period |
|----------|---------|---------------|----------------|
| **Momentum Scalping 2.0** | 20% ($10K) | 5 | 15-60 min |
| **News-Volatility** | 20% ($10K) | 3 | 30-120 min |
| **Gamma/OI Squeeze** | 15% ($7.5K) | 2 | 60-240 min |
| **Arbitrage/Latency** | 15% ($7.5K) | 4 | 5-30 min |
| **AI/ML Pattern Discovery** | 30% ($15K) | 6 | 30-180 min |

### 3. **Risk Management**
- **Max Position Size**: 10% of capital per trade
- **Stop Loss**: -3% per trade
- **Take Profit**: +5% per trade
- **Daily Loss Limit**: -5% (stops trading for the day)
- **Total Loss Limit**: -15% (ends demo early)
- **Anti-Martingale**: Increase position size after wins (max 2x)

### 4. **Order Execution Simulation**
- **Slippage**: 0.1% per trade
- **Commission**: $0 (most Canadian brokers now)
- **Fill Rate**: 95% (5% partial fills)
- **Execution Delay**: 1-3 seconds

---

## 🎯 What the AI Does

### Automated Actions:
1. **Fetches real Canadian market prices** every 5 seconds
2. **Analyzes market conditions** using 5 strategies
3. **Generates buy/sell signals** based on AI models
4. **Executes trades automatically** in demo account
5. **Monitors positions** for stop loss/take profit
6. **Rebalances portfolio** based on strategy performance
7. **Allocates 20% of profits to ETFs** (VCN.TO, XAW.TO, ZAG.TO, VDY.TO)

### Signal Generation:
- Each strategy analyzes market independently
- Signals generated based on:
  - Price momentum & volume
  - News sentiment & volatility spikes
  - Options open interest & gamma exposure
  - Cross-exchange price differences
  - ML pattern recognition

---

## 📱 Dashboard Features

### Real-Time Metrics:
- **Total Portfolio Value** - Live account value
- **Available Cash** - Remaining buying power
- **Total Trades** - Number of trades executed
- **Open Positions** - Current holdings
- **P&L** - Profit/loss ($ and %)

### Charts & Tables:
- **Portfolio Performance Chart** - Value over time
- **Current Positions** - All open positions with P&L
- **Recent Trades** - Last 10 trades with details
- **Demo Status** - Days remaining, market status

### Auto-Refresh:
- Updates every 5 seconds automatically
- No manual refresh needed

---

## 📁 Configuration Files

### `config/demo_config.yaml`
Complete demo settings:
- Starting capital
- Strategy allocations
- Risk limits
- Market symbols
- Trading hours

### `config/risk_config.yaml`
Risk management rules (also used in live trading)

### `config/trading_config.yaml`
Market hours and instrument specs

---

## 📝 Logs & Reports

### Real-Time Logs:
```
logs/demo_trading_YYYYMMDD_HHMMSS.log
```

Contains:
- Every trade executed
- Every signal generated
- Price updates
- P&L changes
- Risk events (stop loss, take profit)

### End-of-Demo Report:
Generated automatically after 7 days:
```
logs/demo_report_YYYYMMDD_HHMMSS.txt
```

Contains:
- Final P&L
- Total trades
- Win rate
- Best/worst trades
- Strategy performance
- Recommendation for live trading

---

## 🎮 How to Use

### Day 1: Start Demo
```bash
python src/dashboard/demo_dashboard.py
```

Open http://localhost:8051 and watch the AI trade!

### Days 2-6: Monitor Progress
- Check dashboard daily
- Review trade history
- Monitor P&L
- Let AI do its thing

### Day 7: Review Results
- Final report generated
- Evaluate performance
- Decide if ready for live trading

---

## 🔄 Demo Workflow

```
┌─────────────────────────────────────────┐
│  1. Fetch Real Market Data (TSX/Yahoo) │
│     ↓                                   │
│  2. AI Strategies Analyze Market       │
│     ↓                                   │
│  3. Generate Buy/Sell Signals          │
│     ↓                                   │
│  4. Execute Trades (Simulated)         │
│     ↓                                   │
│  5. Monitor Positions (Stop Loss/TP)   │
│     ↓                                   │
│  6. Update Dashboard                    │
│     ↓                                   │
│  7. Wait 5 seconds → Repeat            │
└─────────────────────────────────────────┘
```

---

## 💡 Key Differences: Demo vs Live

| Feature | Demo Mode | Live Mode |
|---------|-----------|-----------|
| **Market Data** | ✅ Real (TSX) | ✅ Real (TSX) |
| **Prices** | ✅ Real-time | ✅ Real-time |
| **Order Execution** | ❌ Simulated | ✅ Real (Manual via Questrade) |
| **Money at Risk** | ❌ None ($0) | ✅ Yes (Real CAD) |
| **AI Control** | ✅ Full Auto | ⚠️ Signals only (manual execution) |
| **Capital** | 💵 $50K demo | 💵 Your actual capital |
| **Slippage** | ⚙️ Simulated (0.1%) | 📊 Real market conditions |
| **Purpose** | 🎓 Learning & Testing | 💰 Real trading |

---

## 🎯 After Demo Ends

### If Performance is Good (>2% return, >55% win rate):
1. Review final report
2. Analyze which strategies performed best
3. Consider starting with smaller live capital
4. Remember: Live trading = manual execution (Questrade compliance)

### If Performance is Poor (<2% return or high drawdown):
1. Review what went wrong
2. Adjust strategy parameters
3. Run another demo with tweaks
4. Don't risk real money until consistent

---

## 🛠️ Customization

### Change Starting Capital:
Edit `config/demo_config.yaml`:
```yaml
starting_capital:
  total: 100000  # Change to $100K
```

### Adjust Strategy Allocations:
Edit `config/demo_config.yaml`:
```yaml
momentum_scalping:
  capital_allocation: 0.30  # Increase to 30%
```

### Add More Symbols:
Edit `config/demo_config.yaml`:
```yaml
symbols:
  large_caps:
    - "WEED.TO"  # Add Canopy Growth
    - "AC.TO"    # Add Air Canada
```

### Change Risk Limits:
Edit `config/demo_config.yaml`:
```yaml
risk_management:
  max_position_size: 0.15  # Increase to 15%
  stop_loss_percent: 0.05  # Widen to 5%
```

---

## 📊 Example Demo Session

```
Day 1: Start with $50,000 CAD
  09:35 - AI BUY 50 RY.TO @ $130.50 (Momentum strategy)
  10:12 - AI BUY 75 SHOP.TO @ $102.30 (AI/ML strategy)
  11:45 - AI SELL 50 RY.TO @ $132.80 (+$115 profit)
  14:20 - AI BUY 100 TD.TO @ $88.25 (News-Vol strategy)
  16:00 - Market close
  
  End of Day 1: $50,115 (+$115, +0.23%)

Day 2-6: Continue trading...

Day 7: Final Results
  Final Value: $52,450
  Total P&L: +$2,450 (+4.9%)
  Total Trades: 127
  Win Rate: 58%
  Best Strategy: AI/ML Patterns (+$1,200)
  
  ✅ Recommendation: Performance meets criteria for live trading
```

---

## 🚨 Important Notes

### This is NOT Live Trading:
- No real money at risk
- Orders are simulated
- Perfect for learning and testing
- No broker integration required

### Questrade Compliance:
- When you go live, **you must execute trades manually**
- Bot will generate signals
- You review and execute via Questrade platform
- This is a **regulatory requirement** for retail accounts

### Data Accuracy:
- Prices are real but may have small delays (~15 seconds)
- Yahoo Finance is free but not tick-level data
- Good enough for demo and most strategies

---

## 🎉 Benefits of Demo Mode

✅ **Risk-Free Learning** - $0 real money at risk
✅ **Real Market Conditions** - Actual TSX prices
✅ **Strategy Testing** - See which strategies work
✅ **Confidence Building** - Get comfortable with AI trading
✅ **Performance Validation** - Prove profitability before going live
✅ **Parameter Tuning** - Optimize settings safely

---

## 🌐 Access Demo Dashboard

**URL**: http://localhost:8051

**Features**:
- Real-time portfolio value
- Live trade feed
- Position monitoring
- P&L tracking
- Groww-style UI
- Auto-refreshing

---

## 📞 Support

For issues or questions:
1. Check logs in `logs/demo_trading_*.log`
2. Review configuration in `config/demo_config.yaml`
3. Ensure Yahoo Finance API is accessible
4. Verify internet connection for market data

---

## 🎯 Ready to Start!

```bash
python src/dashboard/demo_dashboard.py
```

**Open**: http://localhost:8051

**Watch the AI trade in real-time with real Canadian market data!** 🚀📈🤖

