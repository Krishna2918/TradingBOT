# Quick Start Guide - Enhanced Trading Bot

## ⚡ Start in 3 Steps

### 1. Set Environment Variables
```powershell
$env:QUESTRADE_REFRESH_TOKEN='A9NHZNXFJcpbIAS9htG7jeUIG-Wk6Fhc0'
$env:QUESTRADE_PRACTICE_MODE='true'
$env:QUESTRADE_ALLOW_TRADING='false'
```

### 2. Start Dashboard
```powershell
python demo_realtime_dashboard.py
```

### 3. Open Browser
Navigate to: **http://127.0.0.1:8050**

---

## 🎯 What You Get

### Dashboard Features
- **Portfolio KPIs**: Value, P&L, cash, returns
- **Holdings Table**: Real-time positions with P&L
- **Recent Trades**: Last 20 trades with details
- **AI Signals**: Current trading opportunities
- **Performance Chart**: Portfolio value over time
- **Logs**: Detailed AI decision reasoning

### AI Features
- **497 stocks** monitored (full TSX/TSXV)
- **Smart Scanner**: 4-tier priority system
- **Intelligence Fusion**: 6 data sources combined
- **Auto-Trading**: No intervention needed
- **Real Quotes**: Questrade + Yahoo Finance

---

## 📊 Expected Behavior

### First 30 Seconds
```
✅ Smart Scanner initialized: 497 stocks
✅ Tier 2 (60 blue chips) scanning
✅ Intelligence sources ready
📊 Analyzing first batch...
```

### First 2 Minutes
```
🔍 Scanning stocks across tiers
💡 Generating multi-source signals
🟢 BUY RY.TO @ confidence=72.5%
   • Insider buying ($500K, CFO)
   • News bullish (dividend increase)
   • Social positive (25 mentions)
   • 3 sources aligned
✅ DEMO BUY: 15 RY.TO @ $125.50
```

### First 10 Minutes
```
📈 Hot stock detected: SHOP.TO
   → Volume spike: 4.2x average
   → Price move: +6.3%
   → PROMOTED to Tier 1 (Hot List)
   
🔥 Tier 1 stocks: [SHOP.TO, HUT.TO, ENB.TO]
💰 5 trades executed
📊 P&L: +$142.50 (+1.43%)
```

---

## ⚙️ Settings

### Force Market Open
- **Default**: ON
- **What it does**: Allows trading 24/7 for demo testing
- **Data source**: Real quotes from Questrade/Yahoo
- **Note**: No fake prices - always uses real market data

### Starting Capital
- **Min**: $1 (no restrictions)
- **Max**: Unlimited
- **Default**: $10,000 (if empty)
- **Tip**: Start with $10k-$50k for realistic testing

---

## 🧠 Intelligence Sources

All sources currently in **demo mode** (simulated data):

| Source | Weight | Status | What It Tracks |
|--------|--------|--------|----------------|
| Insider Trades | 30% | 🟡 Demo | CEO/CFO buy/sell activity |
| News Sentiment | 25% | 🟡 Demo | Breaking news, earnings |
| Social Sentiment | 15% | 🟡 Demo | Reddit, Twitter, StockTwits |
| Weather/Commodity | 10% | 🟡 Demo | Oil, gas, gold, weather |
| Whale Activity | 10% | 🟡 Demo | Warren Buffett, CPP, ETFs |
| Macro Alignment | 10% | 🟡 Demo | Rate hikes, GDP, inflation |

**To activate real data**: Add API keys to config files (see `IMPLEMENTATION_COMPLETE.md`)

---

## 🎮 Dashboard Controls

### Main Controls
- **Start Demo**: Initialize trading with starting capital
- **Pause**: Temporarily halt trading (quotes still update)
- **Force Market Open**: Toggle 24/7 trading (default ON)
- **Open Logs**: View detailed AI decision logs in new tab

### Auto-Actions
- **Live Interval**: Updates holdings/trades every 5 seconds
- **Logs Interval**: Refreshes logs every 5 seconds
- **Training Interval**: Auto-trains on trades every 60 seconds

---

## 📈 Performance Metrics

### What To Watch
1. **Total P&L**: Overall profit/loss
2. **P&L %**: Return on capital
3. **Win Rate**: % of profitable trades
4. **Confidence**: AI signal strength (50-95%)
5. **Hot List**: Tier 1 stocks being monitored closely

### Healthy Indicators
- ✅ Confidence >70% for most trades
- ✅ P&L trending upward
- ✅ 5-15 trades per hour
- ✅ Hot List cycling (promotions/demotions)

### Warning Signs
- ⚠️ Confidence <50% consistently
- ⚠️ P&L declining rapidly
- ⚠️ No trades for >30 minutes
- ⚠️ Hot List empty (no market activity)

---

## 🔧 Troubleshooting

### Dashboard Won't Start
```powershell
# Kill any existing Python processes
taskkill /F /IM python.exe

# Restart dashboard
python demo_realtime_dashboard.py
```

### No Trades Happening
1. Check starting capital >$0
2. Verify "Force Market Open" is ON
3. Check logs for "TRADING DISABLED" warnings
4. Ensure Questrade token is valid

### Quotes Not Updating
1. Check Questrade token is valid
2. Verify internet connection
3. Look for "rate limit" warnings in logs
4. Restart dashboard if stuck

### Browser Shows Old Version
```
Hard refresh: Ctrl + Shift + R (Chrome/Edge)
Or clear browser cache
```

---

## 📝 Logs Explained

### Signal Generation
```
🟢 SIGNAL: BUY SHOP.TO @ confidence=85.0%
   Strength: 0.542 | Position size: 5.0%
   Target: $102.35 | Stop: $90.75
   Sources: 4 positive, 1 negative
   Reasoning:
      • Insider buying ($2M, CEO)
      • News bullish (earnings beat, score=0.75)
      • Social positive (450 mentions, score=0.68)
      • Whale accumulating (net=$10M, score=0.85)
      • ⚠️ Macro environment headwind (score=-0.20)
      • ✅ 4 sources aligned (strong consensus)
```

### Trade Execution
```
💰 DEMO BUY: 10 SHOP.TO @ $95.42
   Cost: $954.20
   Capital remaining: $9,045.80
   Reason: Multi-source signal (4 positive sources)
```

### Smart Scanner Activity
```
🔥 PROMOTED 2 stocks to Tier 1 (Hot List)
   MMED.TO: volume_spike_5.2x + price_move_8.3%
   HUT.TO: volume_spike_3.8x + sentiment_0.75
   
❄️ DEMOTED 1 stock from Tier 1
   ENB.TO: cooled_off time_in_tier=2.3h
```

---

## 🚀 Next Steps

1. **Let it run for 1 hour** - Observe behavior, P&L, hot list
2. **Check logs** - Understand AI reasoning
3. **Note patterns** - Which sources are most accurate?
4. **Tune if needed** - Adjust confidence thresholds in code
5. **Add real APIs** - Upgrade from demo mode to real intelligence

---

## 📞 Support

Check these files for more details:
- `IMPLEMENTATION_COMPLETE.md` - Full technical documentation
- `plan.md` - Original implementation plan
- `logs/ai_trades.log` - Detailed trade logs

---

**System Status**: ✅ FULLY OPERATIONAL
**Dashboard**: http://127.0.0.1:8050
**Mode**: Demo (real data, practice money)

