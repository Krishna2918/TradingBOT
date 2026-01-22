# 🚀 **PRODUCTION RUNBOOK - HYBRID CONTROL PLANE**

## **One-Pager Ops SOP for Engineers**

---

## 📋 **System Overview**

**Hybrid Architecture**: GPT-5 API (≤3 calls/day) + Local Stack (RTX 4080, 32GB RAM)
- **Local Reasoner**: Qwen2.5-14B-Instruct (4-bit GGUF via Ollama)
- **Meta-Ensemble**: Deterministic fusion with hard risk clamps
- **Control Plane**: Escalation triggers and GPT-5 coordination

---

## 🎯 **Escalation Matrix**

### **GPT-5 Call Triggers (Any TRUE → Call GPT-5)**

| **Trigger** | **Threshold** | **Call Type** | **JSON Fields** |
|-------------|---------------|---------------|-----------------|
| **Risk** | 5-day DD > 8% OR daily loss > 2.5% | REWARD_ENGINEER | `max_drawdown_5d`, `daily_pnl`, `pnl_decomposition` |
| **Regime** | Vol z-score \|z\| > 2.0 AND corr < 0.4 | REGIME_AUDITOR | `volatility_zscore`, `correlation_breakdown`, `sector_dispersion` |
| **Model Decay** | Accuracy -7pp OR Sharpe < 0.6 | REWARD_ENGINEER | `ensemble_accuracy`, `sharpe_ratio`, `feature_performance` |
| **Event Heat** | Put/Call OI ratio > 1.6 | REGIME_AUDITOR | `put_call_ratio`, `news_sentiment`, `macro_events` |
| **Kill Switch** | Daily P&L < -2.5% | SANITY_JUDGE | `kill_switch_reason`, `recent_trades`, `portfolio_state` |

---

## 🔧 **Exact JSON Fields for GPT-5 Calls**

### **REGIME_AUDITOR**
```json
{
  "timestamp": "2025-10-05T10:00:00Z",
  "portfolio_state": {
    "net_liquidity": 100000.0,
    "daily_pnl": -0.025,
    "volatility_zscore": 2.3,
    "correlation_breakdown": 0.35
  },
  "sector_dispersion": {
    "technology": 0.15,
    "financials": 0.12,
    "energy": 0.18
  },
  "bucket_usage": {
    "penny": 0.015,
    "fno": 0.03,
    "core": 0.85
  }
}
```

**Expected Response:**
```json
{
  "regime_label": "volatile",
  "blend_weights_delta": {"short": 0.1, "mid": -0.1, "rl": 0.0},
  "bucket_caps_delta": {"penny": -0.01, "fno": -0.02, "core": 0.0},
  "do_not_trade_list": ["SYMBOL1", "SYMBOL2"]
}
```

### **REWARD_ENGINEER**
```json
{
  "timestamp": "2025-10-05T10:00:00Z",
  "pnl_decomposition": {
    "core": 0.02,
    "penny": -0.01,
    "fno": 0.005,
    "sip": 0.001
  },
  "turnover_stats": {
    "daily_turnover": 0.05,
    "weekly_turnover": 0.15,
    "monthly_turnover": 0.35
  },
  "feature_performance": {
    "rsi": 0.65,
    "macd": 0.58,
    "volume": 0.72,
    "news": 0.45
  }
}
```

**Expected Response:**
```json
{
  "reward_coefficients": {"returns": 0.6, "turnover": 0.2, "drawdown": 0.2},
  "turnover_penalties": {"penalty_rate": 0.01},
  "disabled_features": ["feature1", "feature2", "feature3"]
}
```

### **SANITY_JUDGE**
```json
{
  "timestamp": "2025-10-05T10:00:00Z",
  "kill_switch_reason": "Daily P&L -2.5% below threshold",
  "recent_trades": [
    {"symbol": "TD.TO", "action": "BUY", "size": 0.01, "pnl": -0.005}
  ],
  "portfolio_state": {
    "net_liquidity": 100000.0,
    "daily_pnl": -0.03
  }
}
```

**Expected Response:**
```json
{
  "stay_demo": true,
  "live_changes": {
    "max_position_pct": 0.015,
    "stop_multipliers": {"atr": 1.8, "trailing": 1.2}
  }
}
```

---

## 🎚️ **Meta-Ensemble Formula**

### **Deterministic Fusion Logic**
```python
# Base score calculation
base = 0.4*p_short + 0.3*p_mid + 0.3*sigmoid(PPO_size_hint)

# Risk adjustments
adjust = - penalty_if_vol_spike - penalty_if_news_neg - penalty_if_low_liquidity

# Final score
score = clamp(base + adjust, 0, 1)

# Decision thresholds
if score >= 0.67: action = "BUY"
elif score <= 0.33: action = "SELL"  
else: action = "HOLD"

# Position sizing
position_size = bucket_limit * (2 * |score - 0.5|)
```

### **Risk Adjustment Penalties**
- **Vol Spike**: -0.15 (if vol z-score > 2.0)
- **Negative News**: -0.10 (if sentiment < -0.3)
- **Low Liquidity**: -0.05 (if liquidity < 0.5)
- **Correlation Breakdown**: -0.08 (if correlation < 0.4)

---

## 🛡️ **Hard Risk Clamps**

### **4-Bucket Capital Limits**
- **Penny**: 2% (TSXV, min ADV 200k CAD, price ≥ 0.25 CAD)
- **F&O**: 5% (delta-equivalent, margin-aware)
- **Core**: 90% (diversified longs)
- **SIP**: 1% of daily profit (auto-moved EOD)

### **Per-Name Position Limits**
- **Core Stocks**: 1.5% max
- **Penny Stocks**: 0.4% max  
- **F&O**: 0.7% max (delta-equivalent)

### **Stop Loss & Take Profit**
- **Initial Stop**: 1.8×ATR
- **Trailing Stop**: 1.2×ATR
- **Take Profit**: +2.5×ATR

### **Kill Switch Rules**
- **Daily Loss**: -2.5% → Flatten all positions
- **Quarantine**: 3 kill-switch days in 10 → 7-day demo quarantine
- **Override**: GPT-5 must explicitly clear quarantine

---

## 📊 **Dashboard KPIs**

### **Top Bar Metrics**
- **Mode**: Live/Demo indicator
- **Net Liquidity**: Current portfolio value
- **Day P&L**: Today's profit/loss percentage
- **DD%**: Current drawdown percentage
- **Vol Regime**: Normal/High/Extreme badge

### **Risk Panel**
- **Bucket Usage**: Penny/F&O/Core/SIP percentages
- **Per-Name Caps**: Current vs max position sizes
- **Open Risk**: Total exposure and risk metrics
- **Kill Switch**: Armed/Disarmed status

### **Learning Panel**
- **Feature Toggles**: Enabled/disabled features
- **Reward Mix**: Current reward coefficients
- **Policy Version**: Timestamp of last update
- **GPT-5 Calls**: Today's usage (X/3)

### **Alerts**
- **"Escalation to GPT-5: Regime Auditor (1/3)"**
- **"Kill-switch armed—flattened; Demo tomorrow unless override"**
- **"Demo quarantine active—7 days remaining"**

---

## 🔄 **Go-Live Runbook (7 Days)**

### **Day 1-2: Demo Setup**
- [ ] Ingest historical data, train models
- [ ] Validate walk-forward backtest
- [ ] Set base reward coefficients and bucket caps
- [ ] Dry-run order generation

### **Day 3-4: Demo Testing**
- [ ] Open-to-close paper trades
- [ ] Verify logs, stops, bucket math
- [ ] Test escalation triggers
- [ ] Force kill-switch test

### **Day 5: Stress Testing**
- [ ] Replay volatility spike scenarios
- [ ] Confirm kill-switch activation
- [ ] Test GPT-5 escalation calls
- [ ] Validate risk clamps

### **Day 6: Live Micro**
- [ ] Enable micro-size (25% of caps)
- [ ] No shorts or options
- [ ] Monitor all risk metrics
- [ ] Verify execution engine

### **Day 7: Live Limited**
- [ ] Enable full core bucket
- [ ] Keep penny & F&O at half caps
- [ ] Full monitoring and alerting
- [ ] Daily performance review

### **Promotion Criteria**
- [ ] All risk gates hold for 10 consecutive sessions
- [ ] Sharpe ratio > 0.8
- [ ] Max drawdown < 8%
- [ ] No kill-switch activations
- [ ] GPT-5 escalations < 1 per day average

---

## 🚨 **Failure Mode Responses**

### **Data Stall**
- **Action**: Hold state, cancel new orders, keep stops active
- **Alert**: "Data feed interrupted—holding positions"

### **Local Reasoner Offline**
- **Action**: Fall back to pure RL, freeze feature toggles
- **Alert**: "Local reasoner offline—using RL-only mode"

### **RL Model Missing**
- **Action**: Meta-ensemble runs predictors only, halve position sizes
- **Alert**: "RL models unavailable—reduced position sizing"

### **GPT-5 API Failure**
- **Action**: Use local reasoner fallback, log escalation attempts
- **Alert**: "GPT-5 unavailable—using local reasoner"

### **Three Kill-Switch Days in 10**
- **Action**: 7-day demo quarantine, require GPT-5 override
- **Alert**: "Demo quarantine activated—GPT-5 override required"

---

## 📝 **Logging Requirements**

### **Immutable Trade Log**
```json
{
  "timestamp": "2025-10-05T10:00:00Z",
  "mode": "demo",
  "symbol": "TD.TO",
  "direction": "BUY",
  "size_pre": 0.015,
  "size_post_risk": 0.012,
  "entry_price": 100.50,
  "exit_price": null,
  "reason_tags": ["lstm_bullish", "news_positive", "vol_normal"]
}
```

### **Config Diffs (Git-Style)**
- Track all changes to reward coefficients
- Log bucket cap adjustments
- Record feature toggles
- Monitor blacklist updates

### **Experiment IDs**
- Each policy epoch stamps all trades
- Enable rollback to previous versions
- Track performance by policy version

### **Post-Mortems**
- Auto-generated nightly by local LLM
- GPT-5 adds deltas on escalation days
- Include root cause analysis and recommendations

---

## 🎯 **Success Metrics**

### **Performance Gates**
- **Sharpe Ratio**: > 0.9 (out-of-sample)
- **Hit Rate**: > 52% (liquid names only)
- **Max Drawdown**: < 10%
- **PBO**: Low (probability of backtest overfitting)

### **Operational Gates**
- **Uptime**: > 99.5%
- **Latency**: < 100ms (order generation)
- **GPT-5 Calls**: < 3 per day average
- **Kill Switch**: < 1 activation per month

### **Risk Gates**
- **Bucket Compliance**: 100% adherence
- **Position Limits**: No breaches
- **Stop Losses**: 100% execution
- **Correlation**: Maintain diversification

---

## 🔧 **Quick Commands**

### **Start System**
```bash
# Start Ollama with Qwen2.5-14B
ollama pull qwen2.5:14b-instruct

# Start trading system
python interactive_trading_dashboard.py
```

### **Monitor Status**
```bash
# Check GPT-5 call usage
curl http://localhost:8051/api/gpt5-status

# View risk metrics
curl http://localhost:8051/api/risk-status

# Check model performance
curl http://localhost:8051/api/model-stats
```

### **Emergency Override**
```bash
# Force demo mode
curl -X POST http://localhost:8051/api/force-demo

# Clear quarantine (requires GPT-5)
curl -X POST http://localhost:8051/api/clear-quarantine
```

---

## 📞 **Escalation Contacts**

- **System Issues**: Engineering Team
- **Risk Breaches**: Risk Management
- **GPT-5 API**: OpenAI Support
- **Market Data**: Data Provider Support

---

**This runbook provides complete operational guidance for the hybrid control plane system. All thresholds, JSON formats, and procedures are production-ready.**
