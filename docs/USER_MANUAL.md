# Trading Bot User Manual

## Table of Contents

1. [Getting Started](#getting-started)
2. [System Overview](#system-overview)
3. [Configuration](#configuration)
4. [Trading Modes](#trading-modes)
5. [Dashboard Usage](#dashboard-usage)
6. [Monitoring & Analytics](#monitoring--analytics)
7. [Risk Management](#risk-management)
8. [Troubleshooting](#troubleshooting)
9. [Best Practices](#best-practices)
10. [FAQ](#faq)

---

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Windows 10/11 or Linux
- Minimum 8GB RAM
- 10GB free disk space
- Internet connection for market data

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd TradingBOT
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**:
   ```bash
   cp config/settings.py.example config/settings.py
   # Edit config/settings.py with your settings
   ```

4. **Initialize the system**:
   ```bash
   python scripts/setup.py
   ```

### First Run

1. **Start in DEMO mode** (recommended for first-time users):
   ```bash
   python main.py --mode DEMO
   ```

2. **Access the dashboard**:
   - Open your browser to `http://localhost:8080`
   - Default credentials: `admin` / `password`

3. **Verify system status**:
   - Check the System Status panel
   - Ensure all components show "OK"

---

## System Overview

### Architecture

The Trading Bot consists of several interconnected components:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Pipeline │    │   AI Engine     │    │  Trading Engine │
│                 │    │                 │    │                 │
│ • Market Data   │───▶│ • ML Models     │───▶│ • Risk Mgmt     │
│ • API Budgets   │    │ • Calibration   │    │ • Position Mgmt │
│ • Caching       │    │ • Ensemble      │    │ • Execution     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Monitoring    │
                    │                 │
                    │ • System Health │
                    │ • Performance   │
                    │ • Analytics     │
                    └─────────────────┘
```

### Key Features

- **Multi-Model AI**: Ensemble of machine learning models for predictions
- **Risk Management**: Advanced position sizing and drawdown control
- **Regime Detection**: Market condition awareness
- **Real-time Monitoring**: Comprehensive system health tracking
- **Dual Mode Operation**: Live and demo trading modes
- **API Budget Management**: Intelligent rate limiting and caching

---

## Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
# Trading Mode
TRADING_MODE=DEMO  # or LIVE

# Questrade API (for LIVE mode)
QUESTRADE_ACCESS_TOKEN=your_token_here
QUESTRADE_REFRESH_TOKEN=your_refresh_token_here
QUESTRADE_API_URL=https://api01.iq.questrade.com

# Database
DATABASE_PATH=data/trading_demo.db

# Monitoring
ENABLE_MONITORING=true
LOG_LEVEL=INFO

# Risk Management
MAX_DAILY_DRAWDOWN=0.05
MAX_PORTFOLIO_RISK=0.20
MAX_POSITIONS=10
```

### Configuration Files

#### `config/trading_config.yaml`
```yaml
risk_management:
  max_daily_drawdown: 0.05
  max_portfolio_risk: 0.20
  max_positions: 10
  kelly_fraction_limit: 0.25

trading:
  default_position_size: 1000
  min_confidence_threshold: 0.6
  max_position_size: 10000

api_budgets:
  questrade:
    daily_limit: 1000
    qps_limit: 10
  yahoo_finance:
    daily_limit: 2000
    qps_limit: 5
```

#### `config/ai_ensemble_config.yaml`
```yaml
models:
  - name: "trend_following"
    weight: 0.3
    enabled: true
  - name: "mean_reversion"
    weight: 0.3
    enabled: true
  - name: "momentum"
    weight: 0.4
    enabled: true

calibration:
  window_size_days: 30
  min_trades_for_calibration: 10
  alpha_prior: 2.0
  beta_prior: 2.0
```

---

## Trading Modes

### DEMO Mode

**Purpose**: Safe testing environment with simulated trading

**Features**:
- No real money at risk
- Simulated market data
- Full system functionality
- Performance tracking
- Risk management testing

**Usage**:
```bash
python main.py --mode DEMO
```

**Configuration**:
- Uses `data/trading_demo.db`
- Simulated account balance: $100,000
- All features enabled

### LIVE Mode

**Purpose**: Real trading with actual money

**Requirements**:
- Valid Questrade API credentials
- Sufficient account balance
- Risk management settings configured
- Thorough testing in DEMO mode

**Usage**:
```bash
python main.py --mode LIVE
```

**Safety Features**:
- Position size limits
- Daily drawdown limits
- Emergency stop functionality
- Real-time monitoring

**⚠️ Warning**: LIVE mode trades with real money. Ensure you understand the risks and have tested thoroughly in DEMO mode.

---

## Dashboard Usage

### Main Dashboard

Access at `http://localhost:8080`

#### System Status Panel
- **Green**: All systems operational
- **Yellow**: Minor issues detected
- **Red**: Critical issues requiring attention

#### Trading Overview
- Current positions
- Portfolio value
- Daily P&L
- Risk metrics

#### AI Performance
- Model predictions
- Confidence levels
- Calibration status
- Ensemble weights

#### Market Data
- Real-time prices
- Market regime
- Volatility indicators
- Volume analysis

### Navigation

1. **Dashboard**: Main overview
2. **Positions**: Current holdings
3. **Analytics**: Performance analysis
4. **Settings**: Configuration
5. **Logs**: System logs
6. **Help**: Documentation

### Key Metrics

#### Portfolio Metrics
- **Total Value**: Current portfolio value
- **Daily P&L**: Today's profit/loss
- **Total Return**: Overall performance
   - **Sharpe Ratio**: Risk-adjusted returns

#### Risk Metrics
- **Daily Drawdown**: Current drawdown
- **Max Drawdown**: Historical maximum
- **Portfolio Risk**: Overall risk level
- **Position Risk**: Individual position risk

#### AI Metrics
- **Model Confidence**: Average confidence
- **Calibration Quality**: Prediction accuracy
- **Regime Detection**: Current market regime
- **Ensemble Weights**: Model importance

---

## Monitoring & Analytics

### Real-time Monitoring

#### System Health
- CPU usage
- Memory usage
- Disk space
- Network connectivity
- Database status

#### Trading Performance
- Position performance
- Trade execution
- Slippage analysis
- Commission tracking

#### AI Performance
- Model accuracy
- Prediction confidence
- Calibration quality
- Ensemble performance

### Analytics Dashboard

#### Performance Analysis
- **Returns**: Daily, weekly, monthly returns
- **Risk Metrics**: Volatility, Sharpe ratio, max drawdown
- **Trade Analysis**: Win rate, average win/loss
- **Correlation**: Market correlation analysis

#### Risk Analysis
- **Drawdown Analysis**: Historical drawdowns
- **VaR**: Value at Risk calculations
- **Stress Testing**: Scenario analysis
- **Correlation Risk**: Position correlations

#### AI Analysis
- **Model Performance**: Individual model accuracy
- **Ensemble Analysis**: Combined performance
- **Calibration**: Prediction vs actual outcomes
- **Regime Analysis**: Performance by market regime

### Alerts and Notifications

#### System Alerts
- High CPU/memory usage
- Database connection issues
- API rate limit warnings
- Error rate thresholds

#### Trading Alerts
- Large position changes
- Drawdown warnings
- Risk limit breaches
- Unusual market conditions

#### AI Alerts
- Low model confidence
- Calibration degradation
- Regime changes
- Ensemble weight shifts

---

## Risk Management

### Position Sizing

#### Kelly Criterion
The system uses the Kelly criterion for optimal position sizing:

```
Position Size = (bp - q) / b
```

Where:
- `b` = odds received on the wager
- `p` = probability of winning
- `q` = probability of losing (1-p)

#### Drawdown-Aware Sizing
Position sizes are adjusted based on current drawdown:

```
Adjusted Size = Base Size × (1 - Current Drawdown / Max Drawdown)
```

### Risk Limits

#### Daily Limits
- **Max Daily Drawdown**: 5% (configurable)
- **Max Daily Loss**: $1,000 (configurable)
- **Max Trades per Day**: 50 (configurable)

#### Portfolio Limits
- **Max Portfolio Risk**: 20% (configurable)
- **Max Positions**: 10 (configurable)
- **Max Position Size**: $10,000 (configurable)

#### Individual Position Limits
- **Max Position Risk**: 2% of portfolio
- **Min Confidence**: 60% for new positions
- **Max Correlation**: 0.7 with existing positions

### Stop Loss and Take Profit

#### ATR-Based Brackets
- **Stop Loss**: 2× ATR below entry
- **Take Profit**: 1.5× ATR above entry
- **Trailing Stop**: Enabled for profitable positions

#### Regime-Aware Adjustments
- **Trending Markets**: Wider stops, higher targets
- **Ranging Markets**: Tighter stops, lower targets
- **High Volatility**: Increased ATR multipliers

---

## Troubleshooting

### Common Issues

#### 1. System Won't Start

**Symptoms**: Error on startup, system not responding

**Solutions**:
1. Check Python version (3.11+ required)
2. Verify all dependencies installed
3. Check configuration files
4. Review error logs

```bash
# Check Python version
python --version

# Install dependencies
pip install -r requirements.txt

# Check logs
tail -f logs/system.log
```

#### 2. Database Connection Issues

**Symptoms**: Database errors, data not loading

**Solutions**:
1. Check database file permissions
2. Verify database path in config
3. Recreate database if corrupted

```bash
# Check database file
ls -la data/trading_demo.db

# Recreate database
python scripts/setup.py --recreate-db
```

#### 3. API Connection Problems

**Symptoms**: Market data not updating, API errors

**Solutions**:
1. Check internet connection
2. Verify API credentials
3. Check rate limits
4. Review API budget settings

```bash
# Test API connection
python scripts/test_api_connection.py

# Check API budgets
python scripts/check_api_budgets.py
```

#### 4. High Memory Usage

**Symptoms**: System slow, memory warnings

**Solutions**:
1. Reduce data retention period
2. Clear cache
3. Restart system
4. Check for memory leaks

```bash
# Clear cache
python scripts/clear_cache.py

# Check memory usage
python scripts/check_memory.py
```

#### 5. Poor AI Performance

**Symptoms**: Low accuracy, poor predictions

**Solutions**:
1. Check model calibration
2. Review training data
3. Adjust ensemble weights
4. Update models

```bash
# Check calibration
python scripts/check_calibration.py

# Retrain models
python scripts/retrain_models.py
```

### Error Codes

#### Database Errors
- **DB001**: Connection failed
- **DB002**: Query timeout
- **DB003**: Schema mismatch
- **DB004**: Data corruption

#### API Errors
- **API001**: Authentication failed
- **API002**: Rate limit exceeded
- **API003**: Invalid request
- **API004**: Service unavailable

#### Trading Errors
- **TRD001**: Insufficient funds
- **TRD002**: Invalid order
- **TRD003**: Market closed
- **TRD004**: Position limit exceeded

#### AI Errors
- **AI001**: Model not found
- **AI002**: Calibration failed
- **AI003**: Prediction error
- **AI004**: Ensemble error

### Log Analysis

#### Log Locations
- **System Logs**: `logs/system.log`
- **Trading Logs**: `logs/trading.log`
- **AI Logs**: `logs/ai.log`
- **Error Logs**: `logs/errors.log`

#### Log Levels
- **DEBUG**: Detailed information
- **INFO**: General information
- **WARNING**: Warning messages
- **ERROR**: Error conditions
- **CRITICAL**: Critical errors

#### Log Analysis Tools
```bash
# View recent errors
grep "ERROR" logs/system.log | tail -20

# Monitor real-time logs
tail -f logs/system.log

# Search for specific errors
grep "DB001" logs/system.log
```

---

## Best Practices

### System Setup

1. **Start with DEMO Mode**: Always test in demo mode first
2. **Configure Risk Limits**: Set appropriate risk parameters
3. **Monitor System Health**: Check system status regularly
4. **Backup Configuration**: Keep configuration backups

### Trading

1. **Diversify Positions**: Don't put all money in one position
2. **Monitor Drawdown**: Watch daily and total drawdown
3. **Review Performance**: Regular performance analysis
4. **Adjust Settings**: Fine-tune based on performance

### Risk Management

1. **Set Stop Losses**: Always use stop losses
2. **Limit Position Size**: Don't risk too much per trade
3. **Monitor Correlations**: Avoid highly correlated positions
4. **Regular Reviews**: Weekly risk assessment

### Maintenance

1. **Regular Updates**: Keep system updated
2. **Monitor Logs**: Check logs for issues
3. **Clean Data**: Regular data cleanup
4. **Performance Tuning**: Optimize based on usage

---

## FAQ

### General Questions

**Q: How much money do I need to start?**
A: The system works with any amount, but we recommend at least $1,000 for meaningful position sizing.

**Q: Can I run this on a VPS?**
A: Yes, the system can run on a VPS with sufficient resources (8GB RAM, 4 CPU cores).

**Q: Is this suitable for beginners?**
A: The system is designed for users with some trading knowledge. Start with DEMO mode to learn.

**Q: How often should I check the system?**
A: Daily monitoring is recommended, but the system can run autonomously with proper alerts.

### Technical Questions

**Q: What programming language is this written in?**
A: Python 3.11+ with various libraries for data analysis and machine learning.

**Q: Can I modify the AI models?**
A: Yes, the system is open-source and customizable. However, test changes thoroughly.

**Q: How do I add new data sources?**
A: New data sources can be added by implementing the appropriate interface in the data pipeline.

**Q: Can I run multiple instances?**
A: Yes, but ensure they use different database files and don't conflict.

### Trading Questions

**Q: What markets does this support?**
A: Currently supports US and Canadian markets through Questrade API.

**Q: Can I trade options or futures?**
A: Currently supports stocks only. Options and futures support is planned for future versions.

**Q: How does the AI make decisions?**
A: The AI uses ensemble of machine learning models with confidence calibration and regime awareness.

**Q: What's the expected performance?**
A: Performance varies by market conditions. Historical backtesting shows positive returns, but past performance doesn't guarantee future results.

### Risk Questions

**Q: What's the maximum loss I can have?**
A: The system has multiple risk controls, but you can still lose money. Never risk more than you can afford to lose.

**Q: How does the system handle market crashes?**
A: The system has drawdown controls and can detect high volatility regimes to reduce risk.

**Q: Can I override risk controls?**
A: Risk controls can be adjusted in configuration, but this is not recommended.

**Q: What happens if the system goes down?**
A: The system has fail-safes and can be restarted. Monitor system health and have backup plans.

---

## Support

### Getting Help

1. **Documentation**: Check this manual and API reference
2. **Logs**: Review system logs for error details
3. **Community**: Join the user community for discussions
4. **Support**: Contact support for technical issues

### Reporting Issues

When reporting issues, include:
- System configuration
- Error messages
- Log files
- Steps to reproduce
- Expected vs actual behavior

### Contributing

The system is open-source. Contributions are welcome:
- Bug fixes
- Feature enhancements
- Documentation improvements
- Testing

---

**Document Version**: 1.0.0
**Last Updated**: 2025-10-13
**Compatibility**: Trading Bot v1.0.0+