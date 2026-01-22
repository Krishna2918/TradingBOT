# Project Structure and Organization

**Last Updated:** 2025-10-29

This document describes the organization of the GRID repository after restructuring.

---

## 🗂️ Repository Structure

```
GRID/
├── README.md                              # Main repository README
├── PROJECT_STRUCTURE.md                   # This file
│
├── Stock Data Collection System (Root)    # Production data collection
│   ├── continuous_data_collection/        # Main package
│   │   ├── api/                           # Health & metrics endpoints
│   │   ├── core/                          # Core collection engine
│   │   ├── collectors/                    # Data source collectors
│   │   ├── storage/                       # Persistence layers
│   │   ├── monitoring/                    # System monitoring
│   │   └── utils/                         # Utility functions
│   │
│   ├── config/                            # Configuration files
│   │   └── production.yaml                # Production config
│   │
│   ├── monitoring/                        # Monitoring infrastructure
│   │   ├── prometheus.yml                 # Metrics collection
│   │   ├── alertmanager.yml               # Alert routing
│   │   ├── alert_rules.yml                # Alert definitions
│   │   └── grafana/                       # Dashboard configs
│   │
│   ├── scripts/                           # Deployment & maintenance
│   │   ├── init-db.sql                    # Database setup
│   │   ├── stock-collector.service        # Systemd service
│   │   ├── install-service.sh             # Installation script
│   │   ├── backup.sh                      # Backup automation
│   │   └── restore.sh                     # Restore procedures
│   │
│   ├── tests/                             # Test suite
│   │   └── integration/                   # Integration tests
│   │
│   ├── data/                              # Data storage (gitignored)
│   ├── logs/                              # Log files (gitignored)
│   │
│   ├── Dockerfile                         # Container image
│   ├── docker-compose.yml                 # Service orchestration
│   ├── .dockerignore                      # Docker build exclusions
│   ├── .env.example                       # Environment template
│   ├── requirements.txt                   # Python dependencies
│   │
│   └── Documentation/
│       ├── PRODUCTION_DEPLOYMENT.md       # Deployment guide
│       ├── PRODUCTION_READINESS_CHECKLIST.md
│       ├── README_PRODUCTION.md           # Production features
│       ├── DEPLOYMENT_GUIDE.md            # General deployment
│       ├── OPERATIONAL_PROCEDURES.md      # Operations manual
│       ├── PERFORMANCE_OPTIMIZATION_GUIDE.md
│       ├── SYSTEM_REQUIREMENTS.md         # System specs
│       └── TROUBLESHOOTING_GUIDE.md       # Problem solving
│
└── projects/
    └── TradingBOT/                        # AI Trading System
        ├── src/                           # Trading bot source
        │   ├── ai/                        # AI models & strategies
        │   ├── data_collection/           # Market data collection
        │   ├── trading/                   # Trade execution
        │   ├── risk_management/           # Risk controls
        │   ├── monitoring/                # System monitoring
        │   └── ...
        │
        ├── tests/                         # Trading bot tests
        ├── config/                        # Trading configurations
        ├── artifacts/                     # Training artifacts
        ├── checkpoints/                   # Model checkpoints
        ├── models_archive/                # Trained models (moved from root)
        │
        ├── alerts.db                      # Trading alerts (moved from root)
        ├── AI_TRAINING_REPORT.md          # Training reports
        ├── feature_manifest_*.json        # Feature definitions (moved from root)
        │
        └── Documentation/
            ├── AGENTIC_AI_IMPLEMENTATION_*.md
            ├── AI_SYSTEM_VALIDATION_REPORT.md
            ├── API_KEYS_AND_SERVICES_STATUS.md
            ├── DASHBOARD_INTEGRATION_COMPLETE.md
            ├── DEMO_TRADING_GUIDE.md
            └── README.md                  # Trading bot README
```

---

## 📦 What Was Moved

### Files Moved from GRID Root → projects/TradingBOT/

1. **`models/`** → **`models_archive/`**
   - Trained AI models for trading
   - Feature manifests
   - Scaler statistics
   - LSTM model checkpoints

2. **`alerts.db`** → **`alerts.db`**
   - Trading alert database

3. **`feature_manifest_20251027_115407.json`**
   - Feature engineering manifests

4. **`feature_manifest_20251027_115437.json`**
   - Feature engineering manifests

5. **`AI_TRAINING_REPORT.md`** (deleted from root)
   - Older version removed (newer version already in TradingBOT)

6. **`production_monitoring_dashboard.py`** (deleted)
   - Empty file removed

---

## 🎯 Project Purposes

### Stock Data Collection System (Root Level)

**Purpose:** Production infrastructure for collecting and managing stock market data

**Responsibilities:**
- Collect historical stock data from multiple sources
- Maintain data quality and consistency
- Provide clean, structured data storage
- Offer data to downstream systems via APIs
- Monitor collection health and performance

**Key Technologies:**
- Python 3.11
- PostgreSQL (state persistence)
- Redis (caching)
- Prometheus + Grafana (monitoring)
- Docker (containerization)

**Deployment:**
- Production-ready containerization
- Systemd service for bare metal
- Health check endpoints
- Automated backups
- Alert management

---

### Trading Bot (projects/TradingBOT/)

**Purpose:** AI-powered automated trading system

**Responsibilities:**
- Execute trading strategies using AI/ML models
- Manage portfolio and positions
- Implement risk management controls
- Backtest strategies
- Monitor trading performance
- Generate trade signals

**Key Technologies:**
- Python 3.10+
- PyTorch / TensorFlow (AI models)
- Real-time market data feeds
- Trading APIs (Questrade, etc.)

**Features:**
- Multi-model ensemble approach
- Adaptive confidence scoring
- News sentiment analysis
- Technical indicator analysis
- Risk-adjusted position sizing

---

## 🔄 Integration Points

The two systems can work together:

1. **Data Flow:**
   - Stock Data Collection System → Provides clean historical data
   - Trading Bot → Consumes data for model training and backtesting

2. **Shared Resources:**
   - Can share PostgreSQL database
   - Can share Redis cache
   - Can share monitoring infrastructure

3. **Independent Operation:**
   - Each system can run standalone
   - No hard dependencies between systems
   - Separate deployment lifecycles

---

## 🚀 Getting Started

### For Data Collection
```bash
# Stay in GRID root
cd /path/to/GRID

# Configure and deploy
cp .env.example .env
docker-compose up -d
```

### For Trading Bot
```bash
# Navigate to trading bot
cd /path/to/GRID/projects/TradingBOT

# Follow TradingBOT README
cat README.md
```

---

## 📋 Directory Ownership

| Directory/File | Purpose | Owner |
|---------------|---------|-------|
| `/` (root) | Stock data collection | Data Engineering Team |
| `continuous_data_collection/` | Collection engine | Data Engineering Team |
| `monitoring/` | Observability | DevOps Team |
| `scripts/` | Deployment automation | DevOps Team |
| `projects/TradingBOT/` | Trading system | Trading/Quant Team |
| `projects/TradingBOT/src/ai/` | AI models | ML Engineering Team |

---

## 🔒 Access Control

### Stock Data Collection
- **Read Access:** All teams
- **Write Access:** Data Engineering, DevOps
- **Deploy Access:** DevOps

### Trading Bot
- **Read Access:** Trading, Quant, ML teams
- **Write Access:** Trading, Quant, ML teams
- **Deploy Access:** Trading team leads

---

## 📊 Monitoring

### Stock Data Collection
- **Health:** http://localhost:8000/health
- **Metrics:** http://localhost:8000/metrics
- **Grafana:** http://localhost:3000
- **Logs:** `docker-compose logs -f collector`

### Trading Bot
- See `projects/TradingBOT/README.md` for monitoring details

---

## 🔧 Maintenance

### Stock Data Collection
- **Backups:** Automated daily via cron
- **Updates:** Rolling updates via Docker
- **Monitoring:** 24/7 via Prometheus/Grafana
- **Alerts:** Email + Slack

### Trading Bot
- See `projects/TradingBOT/` for maintenance procedures

---

## 📝 Documentation Index

### Root Level Docs (Data Collection)
- `README.md` - Main overview
- `PROJECT_STRUCTURE.md` - This file
- `PRODUCTION_DEPLOYMENT.md` - Deploy guide
- `PRODUCTION_READINESS_CHECKLIST.md` - Pre-launch
- `README_PRODUCTION.md` - Production features
- `TROUBLESHOOTING_GUIDE.md` - Problem solving

### Trading Bot Docs
- `projects/TradingBOT/README.md` - Trading bot overview
- `projects/TradingBOT/AGENTIC_AI_IMPLEMENTATION_*.md` - AI implementation
- `projects/TradingBOT/DEMO_TRADING_GUIDE.md` - Demo guide

---

## 🎯 Next Steps

1. **Review** this structure and familiarize yourself with locations
2. **Navigate** to the appropriate directory for your work
3. **Follow** the README in that directory
4. **Maintain** this separation in future development

---

**Questions?** See README files in each project directory or contact the team leads.
