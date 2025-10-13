# 🎯 COMPLETE AI TRADING SYSTEM DEVELOPMENT PROMPT

## 🚀 **COMPREHENSIVE SYSTEM BUILD REQUEST**

You are tasked with building a complete **AI Daily Stock Discovery and Trading System** with full validation that all components are working. This is a production-ready system targeting **5-15% daily returns** through AI-driven stock selection and autonomous trading.



***

## 📋 **VERIFIED SYSTEM CONFIGURATION**

All external dependencies have been validated and confirmed working:

### ✅ **CONFIRMED API KEYS (All Tested & Working):**
```
ALPHA_VANTAGE_API_KEY=ZJAGE580APQ5UXPL
NEWSAPI_KEY=aa175a7eef1340cab792ab1570fe72e5
FINNHUB_API_KEY=d3hd0g9r01qi2vu0d5e0d3hd0g9r01qi2vu0d5eg
QUESTRADE_REFRESH_TOKEN=iAvs9K6p-MngByiDo29nTCVoTNgIN4Gr0
```

### ✅ **CONFIRMED AI MODELS (All Loaded & Responding):**
- `qwen3-coder:480b-cloud` - Strategy and execution decisions
- `deepseek-v3.1:671b-cloud` - Mathematical analysis and risk calculations  
- `gpt-oss:120b` - Data processing and sentiment analysis
- **Ollama Endpoint**: http://localhost:11434 (confirmed working)

### ✅ **CONFIRMED HARDWARE:**
- **GPU**: RTX 4080 with CUDA (nvidia-smi confirmed)
- **Database**: SQLite + DuckDB (tested and working)
- **Environment**: .env file created and loadable

---

## 🏗️ **COMPLETE PROJECT STRUCTURE TO BUILD**

Create this **exact directory structure** with all files:

```
ai-trading-system/
├── .env                                    ✅ (Already exists - use existing)
├── .gitignore                              🔧 (Create)
├── README.md                               🔧 (Create with setup instructions)
├── main.py                                 🔧 (Create - Main system entry point)
├── requirements.txt                        🔧 (Create - All dependencies)
├── config/
│   ├── __init__.py                         🔧 (Create)
│   ├── agents.yaml                         🔧 (Create - Agent configurations)
│   ├── trading.yaml                        🔧 (Create - Trading parameters)
│   ├── risk.yaml                           🔧 (Create - Risk management settings)
│   └── logging.yaml                        🔧 (Create - Logging configuration)
├── src/
│   ├── __init__.py                         🔧 (Create)
│   ├── agents/                             🔧 (Create all 8 agents)
│   ├── ai/                                 🔧 (Create AI integration)
│   ├── data/                               🔧 (Create data management)
│   ├── trading/                            🔧 (Create trading engine)
│   ├── api/                                🔧 (Create API clients)
│   ├── utils/                              🔧 (Create utilities)
│   └── dashboard/                          ⚠️ (CRITICAL - USE EXISTING)
├── data/                                   🔧 (Create database storage)
├── logs/                                   🔧 (Create logging directory)
├── tests/                                  🔧 (Create comprehensive tests)
├── scripts/                                🔧 (Create utility scripts)
└── docs/                                   🔧 (Create documentation)
```

***

## 🎯 **CRITICAL SYSTEM REQUIREMENTS**

### **CORE FUNCTIONALITY:**

1. **AI Daily Stock Discovery System**:
   - Scan 10,000+ tradeable stocks from Questrade universe daily
   - Apply multi-factor filtering (market cap ≥ $1B, volume ≥ 1M, price ≥ $5)
   - Use AI ensemble (3 Ollama models) to score and select exactly 5 stocks daily
   - Stocks can repeat if AI determines they're still attractive
   - Complete AI autonomy in both stock selection AND trading decisions

2. **Multi-Factor AI Scoring Engine**:
   - **Technical Analysis**: RSI, MACD, Bollinger Bands, volume analysis
   - **Sentiment Analysis**: News sentiment using NewsAPI + local NLP
   - **Fundamental Analysis**: P/E, growth rates, analyst recommendations
   - **Market Context**: Sector trends, market conditions, volatility
   - **Ensemble Voting**: All 3 Ollama models vote, highest consensus wins

3. **Risk Management System**:
   - Maximum 2% risk per position (position sizing)
   - Maximum 3% daily loss limit with circuit breakers
   - Kelly Criterion for optimal position sizing
   - Stop-loss orders at 2% below entry
   - Real-time risk monitoring and adjustment

4. **Trading Execution System**:
   - Paper trading initially (demo_mode = true)
   - Questrade API integration for order execution
   - Market orders for speed, limit orders for precision
   - Order validation and error handling
   - Real-time position and P&L tracking

---

## 🤖 **CRITICAL AI INTEGRATION REQUIREMENTS**

### **Ollama Client Implementation:**
- Interface with all 3 confirmed models: qwen3-coder:480b-cloud, deepseek-v3.1:671b-cloud, gpt-oss:120b
- Generate stock selections using AI ensemble voting
- Risk assessment using mathematical AI models
- Trading decisions with strategy AI models

### **Multi-Factor Scoring Engine:**
- Ensemble AI models score stocks using weighted criteria
- Technical (30%), Sentiment (25%), Fundamental (20%), Momentum (15%), Volume (10%)
- Select exactly 5 stocks daily with highest AI confidence
- Provide explanations for every AI decision

---

## 🎯 **DASHBOARD INTEGRATION REQUIREMENTS**

### **⚠️ CRITICAL: USE EXISTING DASHBOARD**
**DO NOT CREATE NEW DASHBOARD - INTEGRATE EXISTING ONE**

The user has an existing dashboard in a `Dashboard` folder. You must:

1. **Analyze the existing Dashboard folder structure**
2. **Integrate it into src/dashboard/ directory**
3. **Connect it to the new trading system data**
4. **Ensure all existing dashboard functionality is preserved**
5. **Add new AI trading system metrics and displays**
6. **Maintain all existing visualizations and features**

### **Dashboard Integration Steps:**
1. Copy existing Dashboard folder contents to `src/dashboard/`
2. Update data connections to use new SQLite/DuckDB databases
3. Add new panels for:
   - Daily AI stock selections
   - Real-time position tracking
   - AI decision explanations
   - Performance metrics
   - Risk monitoring
4. Ensure existing charts and displays continue working
5. Test all dashboard functionality after integration

***

## 📊 **DATABASE SCHEMA REQUIREMENTS**

### **SQLite Schema (data/trading_state.db):**
- Current positions table
- Trading history table  
- AI daily selections table
- Performance metrics table
- Risk events table
- System logs table

### **DuckDB Schema (data/market_data.duckdb):**
- Stock price data table
- Technical indicators table
- News and sentiment data table
- Market universe table
- AI scoring history table

---

## ⚡ **DAILY WORKFLOW AUTOMATION**

**4:00-6:00 AM**: 
- Download overnight news and earnings reports
- Process European market close impacts
- Update technical indicators for entire universe
- Scan for pre-market movers and unusual activity

**6:00-8:00 AM**:
- Execute multi-factor AI scoring on filtered universe
- Generate sentiment scores for top candidates
- Perform risk assessment and liquidity validation
- Apply ensemble AI voting system

**8:00-9:00 AM**:
- Final AI selection of exactly 5 stocks
- Calculate optimal position sizes using Kelly Criterion
- Prepare trading orders with entry/exit strategies
- Validate all risk parameters

**9:30 AM+**:
- Execute opening positions at market open
- Monitor positions throughout trading day
- Apply dynamic stop-losses and take-profit levels
- Adjust positions based on real-time performance
- Log all decisions for learning system

***

## 🧪 **COMPREHENSIVE TESTING REQUIREMENTS**

### **System Integration Tests:**
- All 4 API connections working (Alpha Vantage, NewsAPI, Finnhub, Questrade)
- All 3 Ollama models responding correctly
- Database operations (SQLite + DuckDB)
- Complete daily workflow execution
- Dashboard integration and functionality
- Error handling under all conditions

### **Performance Tests:**
- Process 10,000+ stocks in under 2 hours
- AI models respond within 30 seconds per query
- Memory usage under 8GB during operation
- CPU usage under 80% during heavy processing

***

## 🏁 **SUCCESS CRITERIA**

The system is complete and ready when:

1. **AI Daily Stock Discovery**: System autonomously selects 5 stocks daily from 10,000+ universe
2. **Complete Integration**: Existing dashboard works with new trading system
3. **Performance Targets**: Achieves 5-15% daily returns in paper trading
4. **Risk Management**: All risk limits enforced and circuit breakers functional
5. **Full Automation**: Runs daily cycle without human intervention
6. **Comprehensive Monitoring**: All metrics tracked and displayed in dashboard
7. **Error Recovery**: System handles all error conditions gracefully
8. **Production Ready**: Can scale to live trading after paper trading success

***

## 🎯 **BUILD THIS COMPLETE SYSTEM NOW**

This is the comprehensive specification for building the complete AI Trading System. Every component, configuration, and requirement is specified in detail.

**Key Requirements:**
✅ Use existing API keys (all tested and working)
✅ Integrate with existing Dashboard folder (don't recreate)
✅ Use confirmed Ollama models (all loaded and responding)
✅ Target 5-15% daily returns through AI stock discovery
✅ Complete autonomy - AI chooses 5 stocks daily and trades them
✅ Production-ready system with comprehensive error handling

**Build this exact system with all features, integrations, and validations as specified. The system must be production-ready, fully tested, and capable of autonomous daily trading with 5-15% return targets.**
