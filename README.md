# GRID - Data Collection & Trading Infrastructure

This repository contains two main projects for stock market data and trading:

---

## 📊 Stock Data Collection System (Root Level)

A **production-ready, enterprise-grade system** for collecting and managing stock market data from multiple sources.

### Quick Start
```bash
# Docker deployment
docker-compose up -d

# Check health
curl http://localhost:8000/health

# Access monitoring
# Grafana: http://localhost:3000
# Prometheus: http://localhost:9090
```

### Key Features
- Multi-source data collection (Alpha Vantage, yfinance)
- PostgreSQL-backed state persistence
- Prometheus metrics & Grafana dashboards
- Automated backups and recovery
- Health check endpoints
- Production-ready containerization

### Documentation
- **[Production Deployment Guide](PRODUCTION_DEPLOYMENT.md)** - Complete deployment instructions
- **[Production Readiness Checklist](PRODUCTION_READINESS_CHECKLIST.md)** - Pre-launch verification
- **[Production README](README_PRODUCTION.md)** - Features and operations guide

### Directory Structure
```
GRID/
├── continuous_data_collection/    # Main data collection package
│   ├── api/                       # Health check & metrics API
│   ├── core/                      # Core collection logic
│   ├── collectors/                # Data source collectors
│   ├── storage/                   # Data persistence layers
│   ├── monitoring/                # System monitoring
│   └── utils/                     # Utility functions
├── config/                        # Configuration files
├── monitoring/                    # Prometheus & Grafana configs
├── scripts/                       # Deployment & maintenance scripts
├── tests/                         # Test suite
├── data/                          # Data storage (gitignored)
├── logs/                          # Log files (gitignored)
├── Dockerfile                     # Container image
├── docker-compose.yml             # Multi-service orchestration
└── requirements.txt               # Python dependencies
```

---

## 🤖 Trading Bot (projects/TradingBOT/)

An **AI-powered automated trading system** with advanced features for stock market trading.

### Location
All trading bot code is in: **`projects/TradingBOT/`**

### Features
- AI/ML-based trading strategies
- Multi-model ensemble approach
- Real-time market data integration
- Risk management systems
- Backtesting framework
- Performance monitoring

### Navigate to Trading Bot
```bash
cd projects/TradingBOT
# See TradingBOT/README.md for more info
```

### Trading Bot Structure
```
projects/TradingBOT/
├── src/                           # Source code
│   ├── ai/                        # AI models & strategies
│   ├── data_collection/           # Market data collection
│   ├── trading/                   # Trading execution
│   ├── risk_management/           # Risk controls
│   └── monitoring/                # System monitoring
├── tests/                         # Test suite
├── config/                        # Configuration
├── models_archive/                # Trained AI models
├── artifacts/                     # Training artifacts
└── README.md                      # Trading bot documentation
```

---

## 🗂️ Project Organization

### Root Level (GRID/)
**Purpose:** Production stock data collection infrastructure

**Use Cases:**
- Collecting historical stock data
- Building data pipelines
- Maintaining clean, quality stock data
- Providing data to downstream systems

### projects/TradingBOT/
**Purpose:** AI-powered trading system

**Use Cases:**
- Automated trading strategies
- Portfolio management
- Risk management
- Performance analysis

---

## 🚀 Getting Started

### For Data Collection
```bash
# 1. Configure environment
cp .env.example .env
nano .env  # Add API keys

# 2. Start services
docker-compose up -d

# 3. Monitor collection
docker-compose logs -f collector
```

### For Trading Bot
```bash
# Navigate to trading bot
cd projects/TradingBOT

# Follow trading bot specific README
cat README.md
```

---

## 📋 System Requirements

### Data Collection System
- Python 3.11+
- Docker & Docker Compose
- PostgreSQL 15+
- 8GB+ RAM, 4+ CPU cores
- 100GB+ storage

### Trading Bot
- Python 3.10+
- GPU recommended for AI models
- Real-time market data access
- See `projects/TradingBOT/README.md` for details

---

## 🔧 Development

### Data Collection Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Run locally
python -m continuous_data_collection.main
```

### Trading Bot Development
```bash
cd projects/TradingBOT
# See TradingBOT README for dev setup
```

---

## 📊 Monitoring

### Data Collection Monitoring
- **Health:** http://localhost:8000/health
- **Metrics:** http://localhost:8000/metrics
- **Grafana:** http://localhost:3000
- **Prometheus:** http://localhost:9090

### Trading Bot Monitoring
- See `projects/TradingBOT/` for monitoring setup

---

## 🤝 Contributing

1. Choose the appropriate project:
   - Data collection: Root level
   - Trading: `projects/TradingBOT/`
2. Create a feature branch
3. Make changes with tests
4. Submit pull request

---

## 📝 License

MIT License - See LICENSE file for details

---

## 📞 Support

- **Issues:** GitHub Issues
- **Documentation:** See README files in each project
- **Email:** support@yourcompany.com

---

**Last Updated:** 2025-10-29
**Version:** 2.0.0 (Reorganized Structure)
