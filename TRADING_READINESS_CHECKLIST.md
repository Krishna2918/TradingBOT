# TradingBOT - Canadian Market Trading Readiness Checklist

## Pre-Trading Verification

### 1. API Keys Required
Set these environment variables before running:

```bash
# Required APIs
export ALPHA_VANTAGE_API_KEY="your_key_here"
export FINNHUB_API_KEY="your_key_here"
export NEWS_API_KEY="your_key_here"

# Optional (for enhanced features)
export REDDIT_CLIENT_ID="your_client_id"
export REDDIT_CLIENT_SECRET="your_secret"

# Questrade (if using broker integration)
# Follow QUESTRADE_SETUP_GUIDE.md for OAuth setup
```

### 2. System Components Status

| Component | Status | Location |
|-----------|--------|----------|
| **Data Collection** | ✅ Ready | `src/data_collection/` |
| **AI Models** | ✅ Ready | `src/ai/` (107 files) |
| **Trading Strategies** | ✅ Ready | `src/trading/strategies/` (5 strategies) |
| **Risk Management** | ✅ Ready | `src/risk_management/` |
| **Execution Engine** | ✅ Ready | `src/execution/` |
| **Portfolio Optimization** | ✅ Ready | `src/portfolio_optimization/` |
| **Currency/FX Tracker** | ✅ Ready | `src/data_services/currency_tracker.py` |
| **Pre-Market Scheduler** | ✅ Ready | `src/workflows/premarket_scheduler.py` |
| **Event Calendar** | ✅ Ready | `src/event_awareness/` |
| **Sentiment Analysis** | ✅ Ready | `src/ai/natural_language_processing/` |

### 3. Market Factor Coverage

**Total: 76 Factors**
- ✅ Fully Covered: 47 (62%)
- ⚠️ Partially Covered: 29 (38%)
- ❌ Missing: 0 (0%)

See `FACTOR_COVERAGE_MATRIX.md` for detailed breakdown.

---

## Pre-Market Checklist (Before 9:30 AM ET)

### Automated (run via pre-market scheduler):
```python
from src.workflows.premarket_scheduler import run_premarket_collection

# Run all pre-market data collection
result = run_premarket_collection(demo_mode=False)
print(result['data']['premarket_summary']['quick_view'])
```

### Manual Verification:
- [ ] Check overnight US futures (S&P 500, Nasdaq, Crude Oil)
- [ ] Review USD/CAD exchange rate trend
- [ ] Check for Bank of Canada announcements
- [ ] Review any earnings releases for held positions
- [ ] Check social sentiment for watchlist stocks
- [ ] Verify API connections are working
- [ ] Review yesterday's portfolio P&L

---

## Quick Start Commands

### 1. Run Pre-Market Analysis
```bash
cd /path/to/TradingBOT
python -m src.workflows.premarket_scheduler
```

### 2. Run Full System Test (Demo Mode)
```python
from src.demo.demo_mode import run_demo_trading
run_demo_trading()
```

### 3. Generate Trading Signals
```python
from src.trading.strategies import (
    MomentumBreakoutStrategy,
    MeanReversionStrategy,
    VWAPCrossoverStrategy,
    OpeningRangeBreakoutStrategy,
    RSIDivergenceStrategy
)

# Initialize strategies
momentum = MomentumBreakoutStrategy(config={})
signals = momentum.generate_signals(market_data)
```

### 4. Check Currency Impact
```python
from src.data_services.currency_tracker import CurrencyTracker, USPreMarketTracker

currency = CurrencyTracker(demo_mode=False)
print(currency.get_usdcad())

premarket = USPreMarketTracker(demo_mode=False)
print(premarket.predict_tsx_gap())
```

---

## Canadian Market Hours

| Session | Time (ET) | Notes |
|---------|-----------|-------|
| Pre-Market | 7:00 - 9:30 AM | Limited liquidity |
| Regular Trading | 9:30 AM - 4:00 PM | Main session |
| Post-Market | 4:00 - 5:00 PM | Limited liquidity |

### Key Times:
- **6:00 AM ET**: Pre-market data collection starts
- **8:30 AM ET**: US economic data releases
- **9:30 AM ET**: TSX opens
- **10:00 AM ET**: Bank of Canada announcements (when scheduled)
- **4:00 PM ET**: TSX closes

---

## Symbol Universe (Default Watchlist)

### Major Banks
- RY.TO - Royal Bank of Canada
- TD.TO - Toronto-Dominion Bank
- BNS.TO - Bank of Nova Scotia
- BMO.TO - Bank of Montreal
- CM.TO - CIBC

### Energy
- CNQ.TO - Canadian Natural Resources
- SU.TO - Suncor Energy
- ENB.TO - Enbridge
- TRP.TO - TC Energy

### Mining
- ABX.TO - Barrick Gold
- WPM.TO - Wheaton Precious Metals
- FNV.TO - Franco-Nevada

### Technology
- SHOP.TO - Shopify
- CSU.TO - Constellation Software

### ETFs
- XIU.TO - iShares S&P/TSX 60
- XIC.TO - iShares Core S&P/TSX
- VFV.TO - Vanguard S&P 500

---

## Important Notes

### Questrade API Limitations
⚠️ **Retail accounts CANNOT place trades programmatically** via Questrade API.

The system uses Questrade for:
- Real-time market data
- Account monitoring
- Position tracking

**Actual trade execution requires manual approval** through the Questrade platform.

### Risk Management
- Kill switches enabled by default
- Maximum position size: 10% of portfolio
- Daily loss limit: 2% of portfolio
- Stop-loss on all positions

### Data Quality
- Minimum 100 data points required for analysis
- Maximum 5% missing data allowed
- OHLC validation enabled
- Volume verification required

---

## Troubleshooting

### API Connection Issues
```bash
# Test Alpha Vantage
curl "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=RY.TO&interval=5min&apikey=YOUR_KEY"

# Test Finnhub
curl "https://finnhub.io/api/v1/quote?symbol=RY.TO&token=YOUR_KEY"
```

### Missing Dependencies
```bash
pip install yfinance pandas numpy scipy torch scikit-learn nltk vaderSentiment textblob
```

### Log Files
- System logs: `src/logs/`
- Trading logs: `data/trading_logs/`
- Pre-market reports: `data/premarket/`

---

*Last Updated: 2026-01-29*
*Branch: claude/complete-project-01VJxiUSmYeY7hayncQTLQnQ*
