# 🤖 AI Ensemble Setup Guide
## Grok + Kimi K2 + Claude Integration

### ✅ What You Now Have

A **multi-AI ensemble trading system** that combines:
- ✅ **Grok AI** - Real-time market analysis and creative insights
- ✅ **Kimi K2** - Technical analysis and pattern recognition  
- ✅ **Claude AI** - Risk management and logical reasoning
- ✅ **Weighted Consensus** - Combines all AI opinions for better decisions
- ✅ **Canadian Market Focus** - Specialized prompts for TSX/Canadian stocks

---

## 🚀 Quick Setup

### 1. Get API Keys

#### **Grok AI (X/Twitter)**
1. Go to [x.ai](https://x.ai)
2. Sign up with your X/Twitter account
3. Get API key from developer portal
4. Add to config: `config/ai_ensemble_config.yaml`

#### **Kimi K2 (Moonshot AI)**
1. Go to [platform.moonshot.cn](https://platform.moonshot.cn)
2. Create account and verify
3. Get API key from dashboard
4. Add to config: `config/ai_ensemble_config.yaml`

#### **Claude AI (Anthropic)**
1. Go to [console.anthropic.com](https://console.anthropic.com)
2. Create account and verify
3. Get API key from API section
4. Add to config: `config/ai_ensemble_config.yaml`

### 2. Update Configuration

Edit `config/ai_ensemble_config.yaml`:

```yaml
grok:
  api_key: "xai-your-actual-grok-api-key-here"
  
kimi:
  api_key: "sk-your-actual-kimi-api-key-here"
  
claude:
  api_key: "sk-ant-your-actual-claude-api-key-here"
```

### 3. Start Enhanced Demo

```bash
python src/dashboard/demo_dashboard.py
```

---

## 🧠 How the AI Ensemble Works

### **Decision Making Process**

```
┌─────────────────────────────────────────────────────────┐
│  1. Market Data → All 3 AIs                            │
│     ↓                                                   │
│  2. Grok: Real-time sentiment & creative analysis      │
│     ↓                                                   │
│  3. Kimi: Technical indicators & pattern recognition   │
│     ↓                                                   │
│  4. Claude: Risk management & portfolio analysis       │
│     ↓                                                   │
│  5. Weighted Consensus (35% Grok, 30% Kimi, 35% Claude)│
│     ↓                                                   │
│  6. Trading Decision (Buy/Sell/Hold)                   │
└─────────────────────────────────────────────────────────┘
```

### **AI Specializations**

| AI | Strength | Weight | Focus |
|----|----------|--------|-------|
| **Grok** | Real-time, Creative | 35% | Market sentiment, unconventional insights |
| **Kimi** | Technical Analysis | 30% | Patterns, indicators, mathematical precision |
| **Claude** | Risk Management | 35% | Portfolio risk, logical reasoning |

---

## 📊 AI Analysis Types

### **Grok AI Analysis**
```json
{
  "signal": "buy",
  "confidence": 0.85,
  "reason": "Strong bullish sentiment in Canadian banking sector with RY.TO showing unusual volume patterns",
  "recommendations": ["RY.TO", "TD.TO"],
  "market_insights": "Oil price correlation breaking down, suggesting sector rotation"
}
```

### **Kimi AI Analysis**
```json
{
  "signal": "buy", 
  "confidence": 0.78,
  "reason": "RSI oversold at 28, MACD showing bullish divergence, price above 20-day MA",
  "technical_indicators": {
    "rsi": "28 (oversold)",
    "macd": "Bullish crossover imminent",
    "bollinger": "Price at lower band, potential bounce"
  },
  "recommendations": ["SHOP.TO"],
  "risk_assessment": "Low risk entry with tight stop loss"
}
```

### **Claude AI Analysis**
```json
{
  "signal": "hold",
  "confidence": 0.65,
  "reason": "Portfolio already 15% concentrated in banking sector, adding more would increase risk",
  "risk_assessment": {
    "portfolio_risk": "Medium - good diversification",
    "position_risk": "Banking sector concentration high",
    "market_risk": "Moderate volatility expected"
  },
  "recommendations": {
    "position_sizing": "Reduce position size to 5% max",
    "stop_loss": "Set at -3% from entry",
    "risk_limits": "Maintain sector diversification"
  }
}
```

### **Ensemble Decision**
```json
{
  "signal": "buy",
  "confidence": 0.76,
  "reason": "Grok: Bullish sentiment | Kimi: Technical buy signal | Claude: Acceptable risk",
  "consensus": "weak_buy",
  "individual_signals": {
    "grok": "buy",
    "kimi": "buy", 
    "claude": "hold"
  }
}
```

---

## 🎯 AI-Powered Features

### **Enhanced Signal Generation**
- **Multi-AI Consensus**: 2+ AIs must agree for high-confidence signals
- **Confidence Scoring**: Weighted average of all AI confidences
- **Risk-Adjusted Decisions**: Claude's risk analysis influences final decision
- **Canadian Market Context**: All AIs get Canadian-specific prompts

### **Real-Time Analysis**
- **Market Sentiment**: Grok analyzes real-time market mood
- **Technical Patterns**: Kimi identifies chart patterns and indicators
- **Risk Assessment**: Claude monitors portfolio risk continuously
- **Consensus Building**: Ensemble combines all insights

### **Adaptive Learning**
- **Performance Tracking**: Monitors which AI performs best
- **Weight Adjustment**: Automatically adjusts AI weights weekly
- **Strategy Optimization**: Learns which AI works best for each strategy
- **Feedback Loop**: Improves based on trade outcomes

---

## 📈 Performance Benefits

### **Before AI Ensemble**
- Single strategy decisions
- Limited market perspective
- Manual risk assessment
- Basic technical analysis

### **After AI Ensemble**
- Multi-perspective analysis
- Real-time market sentiment
- Advanced risk management
- Consensus-driven decisions
- Continuous learning and adaptation

---

## 🔧 Configuration Options

### **AI Weights (Adjustable)**
```yaml
weights:
  grok: 0.35      # Increase for more creative/real-time analysis
  kimi: 0.30      # Increase for more technical analysis
  claude: 0.35    # Increase for more risk management
```

### **Confidence Thresholds**
```yaml
decision_making:
  consensus_threshold: 0.6  # Minimum agreement for action
  confidence_threshold: 0.7  # Minimum confidence for execution
```

### **Strategy Integration**
```yaml
strategy_integration:
  momentum_scalping:
    ai_weight: 0.4  # 40% AI, 60% technical
    primary_ai: "kimi"  # Technical focus
  
  news_volatility:
    ai_weight: 0.6  # 60% AI, 40% technical  
    primary_ai: "grok"  # Sentiment focus
  
  ai_ml_patterns:
    ai_weight: 0.8  # 80% AI, 20% technical
    primary_ai: "ensemble"  # All AIs
```

---

## 🚨 Fallback Behavior

### **API Failures**
- **Grok fails**: Continue with Kimi + Claude
- **Kimi fails**: Continue with Grok + Claude  
- **Claude fails**: Continue with Grok + Kimi
- **All fail**: Use technical indicators only

### **No Consensus**
- **Strong consensus** (3/3 agree): Execute with full confidence
- **Weak consensus** (2/3 agree): Execute with reduced position size
- **No consensus** (1/3 or 0/3): Hold position, wait for better signal

---

## 📊 Dashboard Integration

The demo dashboard now shows:
- **AI Status**: Which AIs are active/available
- **Consensus Level**: Strong/weak/no consensus indicators
- **Individual AI Signals**: What each AI recommended
- **Ensemble Decision**: Final weighted decision
- **AI Performance**: Historical accuracy of each AI

---

## 💰 Cost Considerations

### **API Usage Estimates**
- **Grok**: ~$0.01 per analysis (estimated)
- **Kimi**: ~$0.005 per analysis (estimated)  
- **Claude**: ~$0.02 per analysis (estimated)
- **Total**: ~$0.035 per analysis

### **Daily Usage**
- **Analyses per day**: ~288 (every 5 minutes during market hours)
- **Daily cost**: ~$10 CAD
- **Monthly cost**: ~$200 CAD

### **Cost Optimization**
- Reduce analysis frequency during low-volatility periods
- Use AI ensemble only for high-confidence signals
- Implement smart caching for similar market conditions

---

## 🎮 Demo Mode Benefits

### **Risk-Free Testing**
- Test AI ensemble without API costs
- Validate AI decision quality
- Optimize weights and thresholds
- Build confidence before live trading

### **Performance Validation**
- Compare AI vs non-AI performance
- Identify best AI combinations
- Fine-tune confidence thresholds
- Optimize strategy integration

---

## 🚀 Getting Started

### **Step 1: Get API Keys**
1. Sign up for Grok, Kimi, and Claude
2. Get API keys from each platform
3. Add keys to `config/ai_ensemble_config.yaml`

### **Step 2: Test Integration**
```bash
python src/ai/ai_ensemble.py
```

### **Step 3: Start Enhanced Demo**
```bash
python src/dashboard/demo_dashboard.py
```

### **Step 4: Monitor AI Performance**
- Watch consensus building in real-time
- Monitor individual AI recommendations
- Track ensemble decision accuracy
- Adjust weights based on performance

---

## 🎯 Expected Results

With the AI ensemble, you should see:
- **Higher signal quality** (more accurate buy/sell decisions)
- **Better risk management** (fewer large losses)
- **Improved consistency** (more stable performance)
- **Enhanced insights** (deeper market understanding)
- **Adaptive learning** (continuously improving decisions)

---

## 📞 Support

### **API Issues**
- Check API key validity
- Verify account limits
- Monitor rate limiting
- Check network connectivity

### **Performance Issues**
- Adjust AI weights
- Modify confidence thresholds
- Optimize analysis frequency
- Review fallback behavior

---

## 🎉 Ready to Enhance Your Trading!

The AI ensemble transforms your trading bot from a simple rule-based system to an intelligent, multi-perspective decision maker that combines the strengths of three leading AI models.

**Start with demo mode to test the AI ensemble risk-free, then scale up to live trading with confidence!** 🚀🤖📈
