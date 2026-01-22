# 🔑 Questrade API Setup Guide

**Last Updated:** October 5, 2025

---

## 📋 **Quick Setup Steps**

### **Step 1: Get Your Questrade Refresh Token**

1. **Log into Questrade:**
   - Go to [my.questrade.com](https://my.questrade.com)
   - Sign in with your credentials

2. **Navigate to API Settings:**
   - Click on your name (top right)
   - Select **"Account Management"**
   - Go to **"App Hub"** or **"API Access"**

3. **Generate Refresh Token:**
   - Click **"Generate Token"** or **"Create New Token"**
   - Select **"Personal Use"**
   - Note: The token will be displayed **ONLY ONCE**
   - Copy the entire token (it's long, ~100+ characters)

4. **Save Token Securely:**
   ```text
   Example format: 
   abc123def456ghi789jkl012mno345pqr678stu901vwx234yz567890...
   ```

---

### **Step 2: Set Environment Variable**

**Option A: PowerShell (Current Session Only)**
```powershell
$env:QUESTRADE_REFRESH_TOKEN = "paste_your_token_here"
```

**Option B: PowerShell (Permanent - User Level)**
```powershell
[System.Environment]::SetEnvironmentVariable(
    "QUESTRADE_REFRESH_TOKEN",
    "paste_your_token_here",
    [System.EnvironmentVariableTarget]::User
)
```

**Option C: Create `.env` file** (Recommended)
```bash
# Create a .env file in the project root
QUESTRADE_REFRESH_TOKEN=paste_your_token_here
QUESTRADE_ALLOW_TRADING=false
QUESTRADE_PRACTICE_MODE=true
```

Then load it using:
```powershell
# Install dotenv if needed
pip install python-dotenv

# Or manually load in PowerShell
Get-Content .env | ForEach-Object {
    if ($_ -match '^(.+?)=(.+)$') {
        [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2])
    }
}
```

---

### **Step 3: Verify Connection**

Run the test script:
```powershell
python test_questrade_account.py
```

**Expected Output:**
```
============================================================
  QUESTRADE API CONNECTION TEST
============================================================

Refresh Token: abc123def4...stu901vwx2 (masked)
Token Length: 150 characters

Initializing Questrade client...
SUCCESS: Client initialized

============================================================
  TEST 1: AUTHENTICATION & SERVER INFO
============================================================

API Server: https://api01.iq.questrade.com
Access Token: eyJhbGciO...xyz123 (masked)
Token Expiry: 2025-10-05 16:30:00
SUCCESS: Authentication successful

... (account info, balances, positions, quotes)
```

---

## 🔒 **Security Best Practices**

### **DO:**
✅ Store refresh token in environment variables  
✅ Use `.env` file (add to `.gitignore`)  
✅ Keep trading disabled until ready  
✅ Start with practice mode  
✅ Rotate tokens periodically  

### **DON'T:**
❌ Commit tokens to Git  
❌ Share tokens publicly  
❌ Store in plain text files (without `.gitignore`)  
❌ Enable live trading without testing  

---

## 🛡️ **Trading Safety Controls**

### **Default Settings (Safest):**
```powershell
$env:QUESTRADE_ALLOW_TRADING = "false"  # Trading disabled
$env:QUESTRADE_PRACTICE_MODE = "true"   # Practice account
```

### **Practice Trading:**
```powershell
$env:QUESTRADE_ALLOW_TRADING = "true"   # Enable trading
$env:QUESTRADE_PRACTICE_MODE = "true"   # Use practice account
```

### **Live Trading (Use with Caution):**
```powershell
$env:QUESTRADE_ALLOW_TRADING = "true"   # Enable trading
$env:QUESTRADE_PRACTICE_MODE = "false"  # Use real account
```

---

## 📊 **What the Test Will Show**

### **1. Authentication Status**
- API server URL
- Token validity
- Connection status

### **2. Account Information**
- Account number(s)
- Account type (Individual, TFSA, RRSP, etc.)
- Account status
- Primary account indicator

### **3. Account Balances**
- Total cash
- Market value
- Total equity
- Buying power
- Maintenance excess
- Per-currency breakdown (CAD/USD)

### **4. Current Positions**
- Stock symbols
- Quantity held
- Average entry price
- Current price
- Market value
- Open P&L (profit/loss)

### **5. Market Quotes**
- Real-time prices for TD, RY, SHOP
- Bid/ask prices
- Trading volume
- Live TSX data

### **6. Safety Verification**
- Trading enabled/disabled status
- Practice mode status
- Safety controls active

---

## 🔧 **Troubleshooting**

### **Problem: "Environment variable not set"**
**Solution:**
```powershell
# Set it now
$env:QUESTRADE_REFRESH_TOKEN = "your_token_here"

# Verify it's set
echo $env:QUESTRADE_REFRESH_TOKEN
```

### **Problem: "401 Unauthorized"**
**Causes:**
- Token expired
- Token invalid
- Wrong token format

**Solution:**
1. Generate new refresh token from Questrade
2. Update environment variable
3. Delete token cache: `config/questrade_token_cache.json`
4. Run test again

### **Problem: "Account ID not found"**
**Solution:**
- This is normal on first run
- Client will fetch and cache account ID
- Subsequent requests will use cached ID

### **Problem: "No positions found"**
**Solution:**
- This is normal if you have no open positions
- Test is working correctly

### **Problem: "Rate limit exceeded"**
**Solution:**
- Wait 60 seconds
- Client has built-in rate limiting (1 req/sec)
- Should not occur under normal use

---

## 📁 **File Locations**

### **Config Files:**
- `config/questrade_config.yaml` - Configuration
- `config/questrade_token_cache.json` - Cached tokens (auto-generated)

### **Test Files:**
- `test_questrade_account.py` - Connection test script

### **Logs:**
- Check console output for errors
- Client logs API calls automatically

---

## 🎯 **Quick Checklist**

Before running the test:

- [ ] Have Questrade account access
- [ ] Generated refresh token from Questrade portal
- [ ] Set `QUESTRADE_REFRESH_TOKEN` environment variable
- [ ] Trading controls set to safe defaults
- [ ] Ready to verify connection

---

## 💡 **Next Steps After Successful Test**

### **1. Immediate:**
- ✅ Verify account information is correct
- ✅ Check balances match Questrade portal
- ✅ Confirm positions are accurate
- ✅ Test market quotes are real-time

### **2. Integration:**
- Connect to trading dashboard
- Enable real-time data feeds
- Configure AI trading signals
- Set up risk management

### **3. Before Live Trading:**
- Test thoroughly in practice mode
- Verify all safety controls
- Monitor system for 24-48 hours
- Review all trade signals

---

## 🚨 **Important Notes**

### **Questrade API Limitations:**
1. **Retail accounts** have read-only access
2. **No programmatic trading** for retail users
3. **Market data only** - quotes, account info
4. **Manual trading** required through Questrade platform

### **What This System Does:**
- ✅ Fetches real-time market data
- ✅ Monitors account balances
- ✅ Tracks positions
- ✅ Generates AI trading signals
- ❌ Does NOT execute trades (requires manual execution)

### **Token Lifespan:**
- **Refresh token:** Does not expire (until revoked)
- **Access token:** 30 minutes (auto-refreshed by client)
- **Cache:** Stores tokens locally for efficiency

---

## 📞 **Need Help?**

### **Documentation:**
- [Questrade API Docs](https://www.questrade.com/api/documentation)
- [API Registration](https://www.questrade.com/api/home)

### **Common Issues:**
- Check refresh token is correct (no spaces, full length)
- Ensure token hasn't been revoked
- Verify network connection
- Check Questrade API status

---

## ✅ **Ready to Test!**

**Run this command:**
```powershell
python test_questrade_account.py
```

**You should see:**
- ✅ Successful authentication
- ✅ Account information
- ✅ Current balances
- ✅ Active positions
- ✅ Real-time market quotes

**If successful, you're ready to integrate with the trading system!** 🚀

