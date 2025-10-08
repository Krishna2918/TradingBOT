# 🚀 Quick Start Guide - Production Trading System

**Last Updated:** October 5, 2025

---

## ⚡ **Fast Setup (2 Minutes)**

### **1. Set Environment Variables**

Create a `.env` file or set in PowerShell:

```bash
# === REQUIRED FOR LIVE TRADING ===
$env:QUESTRADE_REFRESH_TOKEN = "your_token_here"
$env:QUESTRADE_ALLOW_TRADING = "false"      # Set to "true" only when ready
$env:QUESTRADE_PRACTICE_MODE = "true"       # Always start with practice mode

# === AI & DATA SERVICES ===
$env:OPENAI_API_KEY = "your_openai_key"     # ChatGPT/GPT-4 Turbo
$env:NEWSAPI_KEY = "your_newsapi_key"       # Optional (demo mode works)
$env:ALPHAVANTAGE_KEY = "your_alphavantage_key"  # Optional
$env:FINNHUB_KEY = "your_finnhub_key"       # Optional (demo mode works)

# === LOCAL LLM (Ollama) ===
$env:OLLAMA_URL = "http://127.0.0.1:11434"
$env:LOCAL_MODEL = "qwen2.5:14b-instruct"   # Local reasoning LLM
$env:CLOUD_MODEL = "gpt-oss:120b-cloud"     # Cloud fallback

# === LOGS & MONITORING ===
$env:LOG_LEVEL = "INFO"
$env:LOG_DIR = "logs"
```

---

## 🎮 **Run The System**

### **Option 1: Dashboard (Recommended)**
```bash
python interactive_trading_dashboard.py
```
Open browser: `http://127.0.0.1:8050`

### **Option 2: Start Services Individually**

**1. Start Embeddings Service:**
```bash
.\scripts\start_embeddings.ps1
```
Service: `http://127.0.0.1:8011`

**2. Start AI Router:**
```bash
.\scripts\start_router.ps1
```
Service: `http://127.0.0.1:8010`

**3. Start Trading Dashboard:**
```bash
python interactive_trading_dashboard.py
```
Dashboard: `http://127.0.0.1:8050`

---

## 🔒 **Safety First - Trading Controls**

### **Default State (Safe):**
```python
# Trading is DISABLED by default
# All orders will be rejected
QUESTRADE_ALLOW_TRADING = false
```

### **Practice Mode (Paper Trading):**
```python
# Enable trading but use practice account
QUESTRADE_ALLOW_TRADING = true
QUESTRADE_PRACTICE_MODE = true
```

### **Live Mode (Real Money) - Use with Caution:**
```python
# Only when you're ready for live trading
QUESTRADE_ALLOW_TRADING = true
QUESTRADE_PRACTICE_MODE = false
```

**⚠️ WARNING:** Always test in practice mode first!

---

## 📊 **What's Running**

### **1. Interactive Dashboard** (`http://127.0.0.1:8050`)
- Live/Demo mode switcher
- Real-time Canadian stock prices (TSX)
- AI trading signals
- Portfolio tracking
- Risk management dashboard
- Performance analytics

### **2. AI Router** (`http://127.0.0.1:8010`)
- Routes short prompts to local LLM (fast)
- Routes long prompts to cloud LLM (accurate)
- Health check: `http://127.0.0.1:8010/health`
- Structured JSON logs: `logs/app.json.log`

### **3. Embeddings Service** (`http://127.0.0.1:8011`)
- Sentence embeddings for semantic search
- Lazy-loading (fast startup)
- Health check: `http://127.0.0.1:8011/health`

---

## 🧪 **Quick Tests**

### **Test 1: Verify Trading is Disabled**
```python
from src.data_pipeline.questrade_client import QuestradeClient

client = QuestradeClient()
result = client.place_order(
    symbol='TD.TO',
    quantity=1,
    action='Buy',
    limit_price=100
)
print(result)
# Expected: {"error": "TRADING_DISABLED", ...}
```

### **Test 2: Check API Status**
```python
from src.data_services.free_apis_integration import *

api = FreeAPIsIntegration(create_free_apis_config())
status = api.get_api_status()
print(status)
# Shows status of all APIs (News, Alpha Vantage, Reddit, Finnhub)
```

### **Test 3: Get Finnhub News**
```python
from src.data_services.free_apis_integration import *

api = FreeAPIsIntegration(create_free_apis_config())
news = api.get_finnhub_news('TD.TO', days_back=7)
print(news)
# Shows company news (demo mode until API key set)
```

### **Test 4: AI Router Health**
```powershell
curl http://127.0.0.1:8010/health
```

### **Test 5: Embeddings Service Health**
```powershell
curl http://127.0.0.1:8011/health
```

---

## 📁 **Key Files & Locations**

### **Configuration:**
- `config/questrade_config.yaml` - Questrade settings
- `config/ai_ensemble_config.yaml` - AI model configuration
- `config/data_pipeline_config.yaml` - Data pipeline settings
- `.env` - Environment variables (create this)

### **Logs:**
- `logs/app.log` - Human-readable logs
- `logs/app.json.log` - Structured JSON logs
- `logs/trading_orchestrator.log` - Trading system logs

### **Data Storage:**
- `data/parquet/` - Market data (Parquet format)
- `config/questrade_token_cache.json` - Cached API tokens

---

## 🛠️ **Troubleshooting**

### **Problem: "Trading Disabled" Error**
**Solution:** This is intentional! Set `QUESTRADE_ALLOW_TRADING=true` when ready.

### **Problem: "Ollama not found"**
**Solution:** 
```powershell
# Check if Ollama is running
curl http://127.0.0.1:11434/api/tags

# If not, find and start Ollama
$ollamaPath = "C:\Users\<YourUsername>\AppData\Local\Programs\Ollama\ollama.exe"
& $ollamaPath serve
```

### **Problem: "Model not loaded" in Embeddings**
**Solution:** This is normal. Model loads on first request (lazy-loading).

### **Problem: "Finnhub rate limit"**
**Solution:** System has backoff built-in. Wait 60 seconds or provide API key.

### **Problem: Unicode/Emoji Errors**
**Solution:** Fixed! All logs are now ASCII-compatible.

---

## 🎯 **Next Steps**

### **Immediate:**
1. ✅ Set environment variables
2. ✅ Start dashboard in demo mode
3. ✅ Verify all services running
4. ✅ Test with paper trading (practice mode)

### **Before Live Trading:**
1. ⚠️ Move OpenAI key to environment variable (CRITICAL)
2. ⚠️ Rotate any exposed API keys
3. ⚠️ Test all strategies in practice mode for 1 week
4. ⚠️ Verify risk controls are working
5. ⚠️ Set up monitoring and alerts

### **Optional Enhancements:**
1. Get News API key (free tier)
2. Get Alpha Vantage key (free tier)
3. Get Finnhub key (free tier)
4. Set up Reddit API OAuth2

---

## 📞 **Support & Documentation**

### **Full Documentation:**
- `PRODUCTION_FIXES_COMPLETE.md` - All recent fixes
- `API_KEYS_AND_SERVICES_STATUS.md` - API status report
- `COMPLETE_IMPLEMENTATION_SUMMARY.md` - System overview

### **Health Checks:**
- Dashboard: `http://127.0.0.1:8050`
- AI Router: `http://127.0.0.1:8010/health`
- Embeddings: `http://127.0.0.1:8011/health`

### **Logs:**
- Real-time: `logs/app.log`
- Structured: `logs/app.json.log`

---

## 🚀 **You're Ready!**

**The system is production-ready with all safety controls enabled.**

Start with demo mode, test thoroughly, then enable practice mode, and finally (when confident) enable live trading.

**Happy Trading! 📈**

