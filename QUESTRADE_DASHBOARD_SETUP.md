# 🚀 Questrade Dashboard - Real Data Integration

## ✅ What's Been Implemented

You now have a **live dashboard** that connects to your **real Questrade account** and displays:

### 📊 Real-Time Data Display
- ✅ **Total Portfolio Value** - Live account balance
- ✅ **Available Cash** - Current buying power in CAD
- ✅ **Open Positions** - Number and value of holdings
- ✅ **Position Details** - Complete holdings table with:
  - Symbol & quantity
  - Average purchase price
  - Current market price
  - Total market value
  - Profit/Loss ($ and %)
- ✅ **Auto-Refresh** - Updates every 5 seconds

### 🔐 Questrade API Integration

#### Configuration Files
- **`config/questrade_config.yaml`** - API credentials and settings
  - Your API refresh token: `ZZEyvxRFv0gG8lcaOPAHg7wy7Bmu-Yn60`
  - Rate limiting (1 req/sec)
  - Read-only permissions
  - Compliance settings

#### API Client
- **`src/data_pipeline/questrade_client.py`** - Full API integration
  - OAuth 2.0 authentication
  - Token refresh automation
  - Portfolio data fetching
  - Real-time quotes
  - Position tracking
  - Account balances

#### Dashboard
- **`src/dashboard/questrade_dashboard.py`** - Live Groww-style UI
  - Real Questrade data display
  - Auto-updating metrics
  - Holdings table
  - Connection status monitoring

### 🌐 How to Access

1. **Start Dashboard:**
   ```bash
   python start_questrade_dashboard.py
   ```

2. **Open Browser:**
   - URL: http://localhost:8050
   - The dashboard will automatically authenticate with Questrade
   - Data refreshes every 5 seconds

3. **What You'll See:**
   - If you have **$0 invested**: Dashboard shows $0.00 and "No positions"
   - If you have **positions**: Shows real portfolio value, holdings, P&L
   - **Connection status** at the top shows authentication state

## 📝 Important Compliance Notes

### ⚠️ Questrade API Limitations (Retail Accounts)

**You CANNOT place trades programmatically via Questrade API**

The dashboard and bot operate in **READ-ONLY mode**:
- ✅ View account balances
- ✅ View positions
- ✅ Get real-time market data
- ✅ Generate AI trading signals
- ❌ **Cannot execute trades automatically**

### How Trading Works with Questrade

1. **Bot Generates Signals** - AI strategies analyze market and create buy/sell recommendations
2. **You Review Signals** - Dashboard displays AI recommendations with confidence scores
3. **Manual Execution** - You place trades manually through:
   - Questrade web platform
   - Questrade mobile app
   - Questrade desktop software

This ensures **full compliance** with Questrade's retail trading rules.

## 🔧 Configuration

### Update Your Refresh Token

If your Questrade token expires, update it in:
```yaml
# config/questrade_config.yaml
questrade:
  api:
    refresh_token: "YOUR_NEW_TOKEN_HERE"
```

### Capital Settings

Your current configuration shows $0 CAD:
```yaml
# config/risk_config.yaml
risk:
  capital:
    total_capital: 0  # Update this to match your real capital
```

**To update your capital:**
1. Edit `config/risk_config.yaml`
2. Set `total_capital` to your actual investment amount
3. The bot will use this for risk management calculations

## 📊 Dashboard Features

### Current Display (with $0 capital)
```
┌─────────────────────────────────────────────────┐
│ 🤖 Trading Bot Dashboard            LIVE 06:19 │
├─────────────────────────────────────────────────┤
│ ✅ Connected to Questrade • Last updated: ...   │
├─────────────────────────────────────────────────┤
│ 💰 $0.00          💵 $0.00                     │
│ Total Value       Available Cash               │
│                                                 │
│ 📈 0 positions    🤖 READ-ONLY                 │
│ $0.00             Manual Trading Only           │
├─────────────────────────────────────────────────┤
│ Portfolio Holdings                              │
│                                                 │
│ No positions currently held                     │
└─────────────────────────────────────────────────┘
```

### With Real Positions (Example)
```
┌─────────────────────────────────────────────────────────────┐
│ Portfolio Holdings                                          │
├────────┬─────┬──────────┬─────────┬───────────┬────────────┤
│ Symbol │ Qty │ Avg Price│ Current │ Mkt Value │    P&L     │
├────────┼─────┼──────────┼─────────┼───────────┼────────────┤
│ RY.TO  │ 50  │ $125.50  │ $132.80 │  $6,640   │ +$365 (+5.8%)
│ TD.TO  │ 40  │  $88.25  │  $92.10 │  $3,684   │ +$154 (+4.4%)
│ SHOP.TO│ 25  │  $95.60  │ $102.30 │  $2,558   │ +$168 (+7.0%)
└────────┴─────┴──────────┴─────────┴───────────┴────────────┘
```

## 🧪 Testing the Integration

### Test Questrade Connection
```bash
python src/data_pipeline/questrade_client.py
```

This will:
1. Authenticate with Questrade
2. Fetch your accounts
3. Display portfolio summary
4. Show all positions

## 📁 File Structure

```
TradingBOT/
├── config/
│   ├── questrade_config.yaml          # Questrade API credentials
│   └── risk_config.yaml                # Capital allocation (updated to $0)
├── src/
│   ├── data_pipeline/
│   │   └── questrade_client.py         # Questrade API integration
│   └── dashboard/
│       ├── app.py                      # Original dashboard (mock data)
│       └── questrade_dashboard.py      # New! Real Questrade data
├── start_dashboard.py                  # Launches original dashboard
└── start_questrade_dashboard.py        # New! Launches Questrade dashboard
```

## 🎯 Next Steps

### Option 1: Paper Trading Mode
If you want to test the bot without real money:
1. Keep `total_capital: 0` in config
2. Bot will generate signals
3. Use demo/paper trading account
4. Track performance manually

### Option 2: Live Trading (Manual Execution)
1. Update `total_capital` in `config/risk_config.yaml`
2. Bot generates AI signals based on 5 strategies
3. You review signals in dashboard
4. Execute trades manually via Questrade platform
5. Dashboard tracks your real portfolio

### Option 3: Track Existing Portfolio
If you already have positions in Questrade:
1. Dashboard will automatically display them
2. Bot will analyze your holdings
3. Generate optimization suggestions
4. Show real-time P&L

## 🔄 Workflow

1. **Morning Setup**
   ```bash
   python start_questrade_dashboard.py
   ```

2. **Monitor Dashboard**
   - Open http://localhost:8050
   - View real portfolio data
   - Check AI trading signals
   - Review strategy recommendations

3. **Execute Trades (Manual)**
   - Based on bot signals
   - Through Questrade platform
   - Bot tracks results automatically

4. **End of Day**
   - Dashboard shows daily P&L
   - ETF allocation calculated (20% of profits)
   - Performance metrics updated

## 🛠️ Troubleshooting

### "Authentication Failed"
- Check refresh token in `config/questrade_config.yaml`
- Token may have expired - get new one from Questrade API portal
- Verify internet connection

### "No positions" but you have holdings
- Wait for data refresh (5 seconds)
- Check Questrade API status
- Verify account_id is correct

### Dashboard shows old data
- Check connection status banner
- Refresh browser (Ctrl+F5)
- Restart dashboard if needed

## 📞 Support

For Questrade API issues:
- Questrade API Documentation: https://www.questrade.com/api/documentation
- API Support: Contact Questrade developer support

## 🎉 Summary

✅ **Dashboard connected to real Questrade data**
✅ **Shows $0.00 if you have no positions** (as requested)
✅ **Auto-updates every 5 seconds**
✅ **Compliance-friendly (read-only mode)**
✅ **Ready to track live portfolio when you add funds**

**Access now:** http://localhost:8050

