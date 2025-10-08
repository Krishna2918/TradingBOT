# Upgraded Ultra-Aggressive Daily Doubling Trading Bot (Canada)

A production-grade, low-latency trading bot designed for Canadian markets with advanced risk management, multiple intraday strategies, and real-time monitoring capabilities.

## 🚀 Features

- **5 Intraday Strategies**: Momentum Scalping 2.0, News-Volatility, Gamma/OI Squeeze, Arbitrage/Latency, AI/ML Pattern Discovery
- **Advanced Risk Management**: Dynamic capital allocation, anti-Martingale recovery, kill switches, cool-down mechanisms
- **Automatic ETF Allocation**: 20% of profits automatically invested in diversified ETFs
- **Real-time Monitoring**: Live dashboards, alerts, and automated recovery systems
- **Low-latency Architecture**: Toronto VPS optimized for TSX/TSXV execution
- **Open-source Stack**: Built with Python, Redis, InfluxDB, and Grafana

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Feeds    │    │  Risk Manager   │    │   Strategies    │
│                 │    │                 │    │                 │
│ • TSX/TSXV      │───▶│ • Capital       │───▶│ • Momentum      │
│ • News/Economic │    │ • Leverage      │    │ • News-Vol      │
│ • Bank of Canada│    │ • Kill Switches │    │ • Gamma/OI      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Execution     │    │   Monitoring    │
                       │                 │    │                 │
                       │ • Order Router  │    │ • Dashboards    │
                       │ • Broker APIs   │    │ • Alerts        │
                       │ • Slippage      │    │ • Recovery      │
                       └─────────────────┘    └─────────────────┘
```

## 📋 Prerequisites

- **OS**: Ubuntu 20.04+ or CentOS 8+
- **Python**: 3.11+
- **Memory**: 8GB+ RAM
- **Storage**: 100GB+ SSD
- **Network**: Low-latency connection to Toronto (preferably colocated)

## 🛠️ Installation

### Quick Setup (Automated)

```bash
# Clone repository
git clone https://github.com/yourusername/trading-bot.git
cd trading-bot

# Run automated setup
chmod +x scripts/setup.sh
./scripts/setup.sh
```

### Manual Setup

```bash
# 1. Install system dependencies
sudo apt update
sudo apt install python3.11 python3.11-venv redis-server influxdb2

# 2. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Configure services
sudo systemctl start redis
sudo systemctl start influxdb
```

## ⚙️ Configuration

### 1. Trading Configuration (`config/trading_config.yaml`)

```yaml
trading:
  market_hours:
    pre_market: "07:00"
    open: "09:30"
    close: "16:00"
    timezone: "America/Toronto"
  
  strategies:
    momentum_scalping:
      enabled: true
      allocation: 0.25
      max_leverage: 2.0
```

### 2. Risk Configuration (`config/risk_config.yaml`)

```yaml
risk:
  capital:
    total_capital: 100000  # $100K CAD
    active_capital: 80000   # 80%
    safety_reserve: 20000   # 20%
  
  limits:
    daily_loss_limit: 0.08   # 8%
    max_drawdown: 0.15      # 15%
```

### 3. Broker Configuration (`config/broker_config.yaml`)

```yaml
brokers:
  primary:
    name: "questrade"
    api_key: "YOUR_API_KEY"
    api_secret: "YOUR_API_SECRET"
```

## 🧪 Testing

### Run All Tests

```bash
# Activate virtual environment
source venv/bin/activate

# Run test suite
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/unit/ -v
python -m pytest tests/integration/ -v
```

### Test Setup

```bash
# Verify all components
python scripts/test_setup.py
```

## 🚀 Usage

### Start Trading Bot

```bash
# Start as service
sudo systemctl start trading-bot

# Check status
sudo systemctl status trading-bot

# View logs
tail -f logs/trading_bot.log
```

### Manual Start

```bash
# Activate environment
source venv/bin/activate

# Start bot
python src/main.py
```

## 📊 Monitoring

### Grafana Dashboards

- **Risk Dashboard**: Capital allocation, leverage, drawdown
- **Performance Dashboard**: P&L, hit rate, slippage
- **System Dashboard**: Latency, uptime, alerts

### Alerts

- **Email**: Daily reports, risk alerts
- **Telegram**: Real-time notifications
- **SMS**: Emergency alerts (optional)

## 🛡️ Risk Management

### Capital Allocation

- **Active Capital**: 80% for trading strategies
- **Safety Reserve**: 20% for emergency situations
- **ETF Allocation**: 20% of profits automatically invested in ETFs
- **Dynamic Sizing**: Position size adjusted based on performance

### Risk Limits

- **Daily Loss**: Maximum 8% of total capital
- **Drawdown**: Maximum 15% peak-to-trough
- **Consecutive Losses**: Cool-down after 3 losses

### Kill Switches

- **Automatic**: Triggered by risk limit breaches
- **Manual**: Admin override capability
- **Recovery**: Automated system restoration

### ETF Allocation

- **Automatic Investment**: 20% of profits automatically invested in ETFs
- **Diversified Portfolio**: VTI, VEA, VWO, BND, VXUS for global diversification
- **Minimum Threshold**: Only allocates when profits exceed $1,000 CAD
- **Weekly Rebalancing**: Automatic portfolio rebalancing
- **Tax Efficiency**: Long-term wealth building through index funds

## 🔧 Development

### Project Structure

```
trading-bot/
├── src/                    # Source code
│   ├── strategies/        # Trading strategies
│   ├── risk_management/   # Risk controls
│   ├── data_pipeline/     # Data collection
│   ├── execution/         # Order execution
│   └── monitoring/        # Monitoring & alerts
├── tests/                 # Test suite
├── config/                # Configuration files
├── scripts/               # Setup & utility scripts
└── docs/                  # Documentation
```

### Adding New Strategies

1. Create strategy module in `src/strategies/`
2. Implement required interface methods
3. Add configuration in `config/trading_config.yaml`
4. Write unit tests
5. Update monitoring dashboards

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

## 📈 Performance Metrics

### Expected Performance

- **Execution Latency**: < 100ms end-to-end
- **Data Feed Latency**: < 50ms
- **Order Fill Rate**: > 95%
- **System Uptime**: > 99.5%

### Risk Metrics

- **Daily P&L**: Target +2% to +5%
- **Hit Rate**: > 60%
- **Max Drawdown**: < 5%
- **Sharpe Ratio**: > 2.0

## 🚨 Troubleshooting

### Common Issues

1. **Redis Connection Failed**
   ```bash
   sudo systemctl restart redis
   redis-cli ping
   ```

2. **InfluxDB Connection Failed**
   ```bash
   sudo systemctl restart influxdb
   curl http://localhost:8086/health
   ```

3. **Python Import Errors**
   ```bash
   source venv/bin/activate
   pip install -r requirements.txt
   ```

### Log Analysis

```bash
# View real-time logs
tail -f logs/trading_bot.log

# Search for errors
grep "ERROR" logs/trading_bot.log

# Monitor specific components
grep "risk_management" logs/trading_bot.log
```

## 📚 Documentation

- **Architecture**: `docs/architecture.md`
- **Deployment**: `docs/deployment.md`
- **API Reference**: `docs/api.md`
- **Troubleshooting**: `docs/troubleshooting.md`

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

**Trading involves substantial risk of loss and is not suitable for all investors. Past performance does not guarantee future results. This software is provided for educational and research purposes only. Use at your own risk.**

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/trading-bot/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/trading-bot/discussions)
- **Wiki**: [Project Wiki](https://github.com/yourusername/trading-bot/wiki)

---

**Built with ❤️ for Canadian Markets**

