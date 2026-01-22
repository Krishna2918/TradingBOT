# AI Trading System - Troubleshooting Guide

## Overview

This troubleshooting guide provides step-by-step solutions for common issues encountered while using the AI Trading System. Follow the procedures in order of complexity, starting with simple solutions before attempting more advanced troubleshooting.

## Quick Diagnostic Checklist

Before diving into specific issues, run through this quick checklist:

- [ ] Is the system running and accessible?
- [ ] Are you connected to the internet?
- [ ] Is the market open (for live trading)?
- [ ] Do you have sufficient account balance?
- [ ] Are there any system alerts or errors visible?
- [ ] Have you tried refreshing the browser page?

## Common Issues and Solutions

### 1. System Access Issues

#### Problem: Cannot Access Dashboard
**Symptoms:**
- Browser shows "Connection Refused" or "Page Not Found"
- Dashboard fails to load
- Login page not accessible

**Solutions:**

**Step 1: Check System Status**
```bash
# Check if the system is running
ps aux | grep python
netstat -tulpn | grep 8050
```

**Step 2: Restart System**
```bash
# Stop the system
sudo systemctl stop tradingbot
# or
pkill -f "python.*main.py"

# Start the system
sudo systemctl start tradingbot
# or
python src/main.py
```

**Step 3: Check Port Availability**
```bash
# Check if port 8050 is available
lsof -i :8050
netstat -tulpn | grep 8050
```

**Step 4: Check Firewall Settings**
```bash
# Check firewall status
sudo ufw status
sudo iptables -L

# Allow port 8050 if needed
sudo ufw allow 8050
```

#### Problem: Login Issues
**Symptoms:**
- Login credentials not accepted
- Session expires immediately
- Authentication errors

**Solutions:**

**Step 1: Verify Credentials**
- Check username and password
- Ensure caps lock is off
- Try resetting password

**Step 2: Clear Browser Data**
- Clear cookies and cache
- Try incognito/private mode
- Try different browser

**Step 3: Check System Logs**
```bash
# Check authentication logs
tail -f logs/tradingbot.log | grep -i auth
journalctl -u tradingbot | grep -i auth
```

### 2. Trading Issues

#### Problem: AI Trading Not Starting
**Symptoms:**
- "Start AI Trading" button not responding
- System shows "Trading Stopped" status
- No trading activity despite being enabled

**Solutions:**

**Step 1: Check System Status**
1. Navigate to **Monitoring** → **System Health**
2. Verify all components are green
3. Check for any error messages

**Step 2: Verify Market Hours**
```python
# Check market hours
from datetime import datetime
import pytz

def is_market_open():
    now = datetime.now(pytz.timezone('America/New_York'))
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    if now.weekday() < 5 and market_open <= now <= market_close:
        return True
    return False

print(f"Market open: {is_market_open()}")
```

**Step 3: Check Account Balance**
1. Navigate to **Trading** → **Portfolio**
2. Verify sufficient cash balance
3. Check for any account restrictions

**Step 4: Review System Logs**
```bash
# Check trading logs
tail -f logs/tradingbot.log | grep -i trading
grep -i "trading.*error" logs/tradingbot.log
```

**Step 5: Restart Trading Engine**
```python
# Restart trading components
from src.workflows.trading_cycle import get_trading_cycle

trading_cycle = get_trading_cycle()
trading_cycle.stop()
trading_cycle.start()
```

#### Problem: Orders Not Executing
**Symptoms:**
- Orders submitted but remain in "Submitted" status
- No fills despite market being open
- Orders rejected by broker

**Solutions:**

**Step 1: Check Order Parameters**
1. Verify stock symbol is correct
2. Check order type and price
3. Ensure quantity is valid
4. Verify time in force settings

**Step 2: Check Broker Connection**
```python
# Test broker connectivity
from src.execution.execution_engine import get_execution_engine

execution_engine = get_execution_engine()
status = execution_engine.check_connection()
print(f"Broker connection: {status}")
```

**Step 3: Check Market Data**
1. Navigate to **Trading** → **Market Data**
2. Verify real-time quotes are updating
3. Check for data feed issues

**Step 4: Review Order Logs**
```bash
# Check order execution logs
grep -i "order.*execution" logs/tradingbot.log
grep -i "broker.*error" logs/tradingbot.log
```

#### Problem: Positions Not Updating
**Symptoms:**
- Position values not reflecting current market prices
- P&L calculations appear incorrect
- Positions stuck in old state

**Solutions:**

**Step 1: Refresh Market Data**
```python
# Force market data refresh
from src.data_pipeline.questrade_client import QuestradeClient

client = QuestradeClient()
client.refresh_quotes()
```

**Step 2: Check Database Connection**
```python
# Test database connectivity
from src.config.database import get_connection

with get_connection() as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM positions")
    count = cursor.fetchone()[0]
    print(f"Positions in database: {count}")
```

**Step 3: Recalculate Positions**
```python
# Recalculate position values
from src.trading.positions import get_position_manager

position_manager = get_position_manager()
position_manager.recalculate_all_positions()
```

### 3. AI and Analytics Issues

#### Problem: AI Models Not Responding
**Symptoms:**
- AI analysis taking too long
- "AI Model Unavailable" errors
- Low confidence scores

**Solutions:**

**Step 1: Check AI Model Status**
```python
# Check AI model availability
from src.ai.multi_model import get_multi_model_manager

manager = get_multi_model_manager()
status = manager.check_model_availability()
print(f"Model status: {status}")
```

**Step 2: Restart AI Models**
```python
# Restart AI models
from src.ai.multi_model import get_multi_model_manager

manager = get_multi_model_manager()
manager.restart_models()
```

**Step 3: Check Ollama Server**
```bash
# Check Ollama server status
curl http://localhost:11434/api/tags
ps aux | grep ollama
```

**Step 4: Verify Model Performance**
```python
# Test model performance
from src.ai.multi_model import get_multi_model_manager

manager = get_multi_model_manager()
performance = manager.get_model_performance()
print(f"Model performance: {performance}")
```

#### Problem: Analytics Not Loading
**Symptoms:**
- Performance charts not displaying
- Analytics dashboard blank
- Historical data missing

**Solutions:**

**Step 1: Check Data Availability**
```python
# Check historical data
from src.config.database import get_connection

with get_connection() as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM trades")
    trade_count = cursor.fetchone()[0]
    print(f"Trades in database: {trade_count}")
```

**Step 2: Regenerate Analytics**
```python
# Regenerate analytics data
from src.analytics.performance_analytics import PerformanceAnalytics

analytics = PerformanceAnalytics()
analytics.regenerate_all_metrics()
```

**Step 3: Check Chart Rendering**
1. Open browser developer tools (F12)
2. Check for JavaScript errors
3. Verify chart libraries are loading
4. Try refreshing the page

### 4. Performance Issues

#### Problem: Slow System Response
**Symptoms:**
- Dashboard loading slowly
- Delayed order execution
- High CPU/memory usage

**Solutions:**

**Step 1: Check System Resources**
```bash
# Check system resources
htop
iotop
df -h
free -h
```

**Step 2: Optimize Database**
```python
# Optimize database performance
from src.config.database import get_connection

with get_connection() as conn:
    conn.execute("VACUUM")
    conn.execute("ANALYZE")
    conn.execute("PRAGMA optimize")
```

**Step 3: Clear Cache**
```python
# Clear system cache
from src.monitoring.system_monitor import SystemMonitor

monitor = SystemMonitor()
monitor.clear_cache()
```

**Step 4: Restart Services**
```bash
# Restart system services
sudo systemctl restart tradingbot
sudo systemctl restart redis
```

#### Problem: High Memory Usage
**Symptoms:**
- System running out of memory
- Out of memory errors
- System becoming unresponsive

**Solutions:**

**Step 1: Check Memory Usage**
```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head -10
```

**Step 2: Optimize Memory Settings**
```python
# Configure memory limits
import gc
gc.set_threshold(1000, 10, 10)
gc.collect()
```

**Step 3: Restart System**
```bash
# Restart system to free memory
sudo systemctl restart tradingbot
```

### 5. Data Issues

#### Problem: Missing Market Data
**Symptoms:**
- Stock prices not updating
- Historical data gaps
- API connection errors

**Solutions:**

**Step 1: Check API Connections**
```python
# Test API connectivity
import requests

apis = {
    'Questrade': 'https://api.questrade.com/v1/time',
    'Yahoo Finance': 'https://query1.finance.yahoo.com/v8/finance/chart/AAPL'
}

for name, url in apis.items():
    try:
        response = requests.get(url, timeout=10)
        print(f"{name}: {response.status_code}")
    except Exception as e:
        print(f"{name}: Error - {e}")
```

**Step 2: Refresh Data Feeds**
```python
# Refresh all data feeds
from src.data_pipeline.comprehensive_data_pipeline import ComprehensiveDataPipeline

pipeline = ComprehensiveDataPipeline()
pipeline.refresh_all_data()
```

**Step 3: Check API Rate Limits**
```bash
# Check API usage logs
grep -i "rate.*limit" logs/tradingbot.log
grep -i "api.*quota" logs/tradingbot.log
```

#### Problem: Database Corruption
**Symptoms:**
- Database errors
- Missing data
- System crashes

**Solutions:**

**Step 1: Check Database Integrity**
```bash
# Check database integrity
sqlite3 data/trading_demo.db "PRAGMA integrity_check;"
sqlite3 data/trading_demo.db "PRAGMA quick_check;"
```

**Step 2: Repair Database**
```bash
# Repair database
sqlite3 data/trading_demo.db "VACUUM;"
sqlite3 data/trading_demo.db "REINDEX;"
```

**Step 3: Restore from Backup**
```bash
# Restore from backup
cp data/trading_demo_backup_20250113.db data/trading_demo.db
```

### 6. Security Issues

#### Problem: Authentication Failures
**Symptoms:**
- Login attempts failing
- Session timeouts
- Access denied errors

**Solutions:**

**Step 1: Check Authentication Logs**
```bash
# Check authentication logs
grep -i "auth.*fail" logs/tradingbot.log
grep -i "login.*error" logs/tradingbot.log
```

**Step 2: Reset Authentication**
```python
# Reset authentication tokens
from src.config.mode_manager import get_mode_manager

mode_manager = get_mode_manager()
mode_manager.reset_authentication()
```

**Step 3: Check API Keys**
```python
# Verify API keys
import os

required_keys = [
    'QUESTRADE_CLIENT_ID',
    'QUESTRADE_CLIENT_SECRET',
    'YAHOO_FINANCE_API_KEY'
]

for key in required_keys:
    value = os.getenv(key)
    if value:
        print(f"{key}: Set")
    else:
        print(f"{key}: Missing")
```

## Advanced Troubleshooting

### System Diagnostics

#### Comprehensive System Check
```python
# Run comprehensive system diagnostics
from src.monitoring.system_monitor import SystemMonitor

monitor = SystemMonitor()
diagnostics = monitor.run_comprehensive_diagnostics()
print(diagnostics)
```

#### Performance Profiling
```python
# Profile system performance
import cProfile
import pstats

def profile_system():
    # Run system operations
    pass

cProfile.run('profile_system()', 'profile_output.prof')
stats = pstats.Stats('profile_output.prof')
stats.sort_stats('cumulative').print_stats(10)
```

### Log Analysis

#### Error Pattern Analysis
```bash
# Analyze error patterns
grep -i "error" logs/tradingbot.log | sort | uniq -c | sort -nr
grep -i "exception" logs/tradingbot.log | sort | uniq -c | sort -nr
```

#### Performance Analysis
```bash
# Analyze performance logs
grep -i "slow" logs/tradingbot.log
grep -i "timeout" logs/tradingbot.log
grep -i "latency" logs/tradingbot.log
```

### Network Troubleshooting

#### Network Connectivity
```bash
# Test network connectivity
ping google.com
nslookup api.questrade.com
traceroute api.questrade.com
```

#### Port Testing
```bash
# Test port connectivity
telnet api.questrade.com 443
nc -zv api.questrade.com 443
```

## Emergency Procedures

### System Recovery

#### Complete System Restart
```bash
# Emergency system restart
sudo systemctl stop tradingbot
sudo systemctl stop redis
sudo systemctl start redis
sudo systemctl start tradingbot
```

#### Data Recovery
```bash
# Emergency data recovery
cp data/trading_demo_backup_latest.db data/trading_demo.db
sudo systemctl restart tradingbot
```

#### Configuration Reset
```bash
# Reset configuration to defaults
cp config/default_config.json src/config/mode_config.json
sudo systemctl restart tradingbot
```

### Emergency Trading Stop

#### Immediate Trading Halt
```python
# Emergency stop all trading
from src.workflows.trading_cycle import get_trading_cycle

trading_cycle = get_trading_cycle()
trading_cycle.emergency_stop()
```

#### Close All Positions
```python
# Emergency close all positions
from src.trading.positions import get_position_manager

position_manager = get_position_manager()
position_manager.emergency_close_all()
```

## Prevention and Maintenance

### Regular Maintenance

#### Daily Checks
- [ ] Verify system health status
- [ ] Check for error alerts
- [ ] Review trading performance
- [ ] Monitor resource usage

#### Weekly Maintenance
- [ ] Review system logs
- [ ] Check database integrity
- [ ] Update system components
- [ ] Backup configuration files

#### Monthly Maintenance
- [ ] Full system backup
- [ ] Performance optimization
- [ ] Security audit
- [ ] Update documentation

### Monitoring Setup

#### Automated Monitoring
```python
# Setup automated monitoring
from src.monitoring.system_monitor import SystemMonitor

monitor = SystemMonitor()
monitor.setup_automated_checks()
```

#### Alert Configuration
```python
# Configure monitoring alerts
from src.monitoring.alert_system import AlertSystem

alert_system = AlertSystem()
alert_system.configure_alerts({
    'cpu_threshold': 80,
    'memory_threshold': 85,
    'error_rate_threshold': 0.05
})
```

## Getting Help

### Support Channels
- **Email Support**: support@tradingbot.com
- **Phone Support**: +1-800-TRADING
- **Live Chat**: Available during business hours
- **Documentation**: https://docs.tradingbot.com

### Information to Provide
When contacting support, please provide:
1. Description of the issue
2. Steps to reproduce
3. Error messages (if any)
4. System logs
5. Browser and system information
6. Screenshots (if applicable)

### Log Collection
```bash
# Collect system logs for support
tar -czf support_logs_$(date +%Y%m%d_%H%M%S).tar.gz logs/ config/ data/
```

## Conclusion

This troubleshooting guide provides comprehensive solutions for common issues. If you encounter problems not covered in this guide, please contact the support team with detailed information about the issue.

Remember to always backup your data before attempting advanced troubleshooting procedures.
