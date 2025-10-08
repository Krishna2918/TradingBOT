# 🔑 FREE API Keys Setup Guide

## Date: October 5, 2025

---

## 🎯 Overview

This guide shows you how to get **FREE API keys** for enhanced trading data:

1. **News API** - Market news sentiment analysis
2. **Alpha Vantage** - Technical indicators (SMA, RSI, MACD)
3. **Reddit API** - Social sentiment from Reddit

---

## 📰 1. News API (FREE)

### **What it provides:**
- Real-time financial news
- Sentiment analysis for stocks
- Market-moving news alerts

### **How to get it:**
1. Go to [newsapi.org](https://newsapi.org)
2. Click **"Get API Key"**
3. Sign up with email (FREE)
4. Copy your API key

### **Free tier limits:**
- ✅ 1,000 requests per day
- ✅ 30 requests per minute
- ✅ Perfect for trading bot

### **Setup:**
```yaml
# In config/data_pipeline_config.yaml
api_keys:
  newsapi: "YOUR_NEWS_API_KEY_HERE"  # Replace "demo"
```

---

## 📊 2. Alpha Vantage (FREE)

### **What it provides:**
- Technical indicators (SMA, RSI, MACD)
- Real-time stock data
- Historical price data

### **How to get it:**
1. Go to [alphavantage.co](https://www.alphavantage.co/support/#api-key)
2. Click **"Get Free API Key"**
3. Fill out the form (FREE)
4. Copy your API key

### **Free tier limits:**
- ✅ 500 requests per day
- ✅ 5 requests per minute
- ✅ Great for technical analysis

### **Setup:**
```yaml
# In config/data_pipeline_config.yaml
api_keys:
  alpha_vantage: "YOUR_ALPHA_VANTAGE_KEY_HERE"  # Replace "demo"
```

---

## 🔴 3. Reddit API (FREE)

### **What it provides:**
- Social sentiment from r/wallstreetbets, r/stocks
- Market discussion analysis
- Crowd sentiment indicators

### **How to get it:**
1. Go to [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps)
2. Click **"Create App"** or **"Create Another App"**
3. Fill out the form:
   - **Name**: TradingBot
   - **App type**: script
   - **Description**: AI Trading Bot
4. Copy **Client ID** and **Client Secret**

### **Free tier limits:**
- ✅ 60 requests per minute
- ✅ 3,600 requests per hour
- ✅ Perfect for sentiment analysis

### **Setup:**
```yaml
# In config/data_pipeline_config.yaml
api_keys:
  reddit:
    client_id: "YOUR_REDDIT_CLIENT_ID_HERE"      # Replace "demo"
    client_secret: "YOUR_REDDIT_CLIENT_SECRET_HERE"  # Replace "demo"
    user_agent: "TradingBot/1.0"
```

---

## 🚀 Quick Setup

### **Step 1: Get your API keys**
- News API: [newsapi.org](https://newsapi.org) (2 minutes)
- Alpha Vantage: [alphavantage.co](https://www.alphavantage.co/support/#api-key) (2 minutes)
- Reddit API: [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps) (3 minutes)

### **Step 2: Update configuration**
Edit `config/data_pipeline_config.yaml`:

```yaml
api_keys:
  # News API (FREE)
  newsapi: "YOUR_NEWS_API_KEY_HERE"
  
  # Alpha Vantage (FREE)
  alpha_vantage: "YOUR_ALPHA_VANTAGE_KEY_HERE"
  
  # Reddit API (FREE)
  reddit:
    client_id: "YOUR_REDDIT_CLIENT_ID_HERE"
    client_secret: "YOUR_REDDIT_CLIENT_SECRET_HERE"
    user_agent: "TradingBot/1.0"
```

### **Step 3: Test the integration**
```bash
python -c "
from src.data_services import FreeAPIsIntegration, create_free_apis_config
config = create_free_apis_config()
api = FreeAPIsIntegration(config)
print('API Status:', api.get_api_status())
"
```

---

## 💡 Benefits of Real API Keys

### **With Demo Keys:**
- ❌ Random/fake data
- ❌ No real market insights
- ❌ Limited functionality

### **With Real API Keys:**
- ✅ **Real news sentiment** from financial news
- ✅ **Real technical indicators** (SMA, RSI, MACD)
- ✅ **Real social sentiment** from Reddit
- ✅ **Enhanced AI decision-making**
- ✅ **Better trading performance**

---

## 🔧 Integration Status

### **Current Status:**
- ✅ **Yahoo Finance**: Working (no API key needed)
- ⚠️ **News API**: Demo mode (needs API key)
- ⚠️ **Alpha Vantage**: Demo mode (needs API key)
- ⚠️ **Reddit API**: Demo mode (needs API key)

### **After Setup:**
- ✅ **Yahoo Finance**: Working
- ✅ **News API**: Real news sentiment
- ✅ **Alpha Vantage**: Real technical indicators
- ✅ **Reddit API**: Real social sentiment

---

## 📈 What You'll Get

### **Enhanced AI Trading:**
1. **News Sentiment Analysis**
   - Positive/negative news impact
   - Market-moving news detection
   - Sector-specific sentiment

2. **Technical Indicators**
   - Moving averages (SMA)
   - Relative Strength Index (RSI)
   - MACD signals
   - Trend analysis

3. **Social Sentiment**
   - Reddit discussion sentiment
   - Crowd psychology indicators
   - Market fear/greed analysis

### **Better Trading Decisions:**
- AI considers news sentiment
- Technical analysis integration
- Social sentiment factors
- Multi-dimensional decision making

---

## 🎯 Next Steps

1. **Get your FREE API keys** (5 minutes total)
2. **Update the configuration file**
3. **Restart the trading bot**
4. **Watch enhanced AI performance**

The AI will now have access to:
- Real financial news
- Technical indicators
- Social sentiment
- Multi-source decision making

**Total cost: $0 (all services are FREE!)**
