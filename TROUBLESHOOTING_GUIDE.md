# Troubleshooting Guide - Continuous Data Collection System

## Table of Contents

1. [Quick Diagnostics](#quick-diagnostics)
2. [Common Issues](#common-issues)
3. [API-Related Problems](#api-related-problems)
4. [Network and Connectivity Issues](#network-and-connectivity-issues)
5. [Performance Problems](#performance-problems)
6. [Data Quality Issues](#data-quality-issues)
7. [System Resource Problems](#system-resource-problems)
8. [State and Recovery Issues](#state-and-recovery-issues)
9. [Configuration Problems](#configuration-problems)
10. [Emergency Procedures](#emergency-procedures)
11. [Log Analysis](#log-analysis)
12. [Advanced Debugging](#advanced-debugging)

## Quick Diagnostics

### System Health Check

Run this quick diagnostic to identify immediate issues:

```bash
# Quick system status check
python check_progress.py --health-check

# Expected output:
# ✓ System Status: Running
# ✓ API Connectivity: OK
# ✓ Disk Space: 85% available
# ✓ Memory Usage: 45%
# ✓ Collection Progress: 234/1400 (16.7%)
```

### Diagnostic Toolkit

Use the built-in diagnostic toolkit for comprehensive analysis:

```bash
# Run comprehensive diagnostics
python diagnostic_toolkit.py --full-scan

# Run specific diagnostic categories
python diagnostic_toolkit.py --network
python diagnostic_toolkit.py --performance
python diagnostic_toolkit.py --data-quality
python diagnostic_toolkit.py --system-resources
```

### Log Quick Check

```bash
# Check for recent errors
tail -50 logs/continuous_collection.log | grep -i error

# Check system status
grep "System Status" logs/continuous_collection.log | tail -5

# Check recent progress
grep "Progress:" logs/continuous_collection.log | tail -10
```

## Common Issues

### Issue 1: System Not Starting

**Symptoms:**
- `python start_collection.py` fails immediately
- "System failed to initialize" error
- No log entries created

**Diagnosis:**
```bash
# Check configuration
python config_validator.py

# Check system requirements
python system_requirements_checker.py

# Check Python environment
python -c "import continuous_data_collection; print('Import successful')"
```

**Solutions:**

1. **Configuration Issues:**
```bash
# Verify configuration file exists
ls -la config/production.yaml

# Check configuration syntax
python -c "
import yaml
with open('config/production.yaml', 'r') as f:
    config = yaml.safe_load(f)
print('Configuration is valid')
"
```

2. **Missing Dependencies:**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check specific imports
python -c "
import pandas, numpy, aiohttp, asyncio
print('Core dependencies available')
"
```

3. **Permission Issues:**
```bash
# Check directory permissions
ls -la data/ logs/ backups/

# Fix permissions if needed
chmod 755 data logs backups
chmod 644 config/production.yaml
```

### Issue 2: Collection Stops Unexpectedly

**Symptoms:**
- System was running but stopped
- Last log entry shows normal operation
- Process not found in system

**Diagnosis:**
```bash
# Check if process is still running
ps aux | grep python | grep continuous

# Check system logs for crashes
dmesg | grep -i "killed\|oom\|segfault" | tail -10

# Check available resources
free -h
df -h
```

**Solutions:**

1. **Out of Memory:**
```bash
# Check memory usage history
grep "Memory usage" logs/continuous_collection.log | tail -20

# Reduce worker count
# Edit config/production.yaml:
# system.max_workers: 4  # Reduce from 8
```

2. **Disk Space Full:**
```bash
# Clean up temporary files
rm -rf temp/*
find logs/ -name "*.log.*" -mtime +7 -delete

# Compress old logs
gzip logs/*.log.1 logs/*.log.2
```

3. **System Crash Recovery:**
```bash
# Run emergency recovery
python emergency_recovery.py

# Check state integrity
python -c "
from continuous_data_collection.core.state_manager import StateManager
import asyncio

async def check_state():
    state_manager = StateManager()
    is_valid = await state_manager.verify_state_integrity()
    print(f'State integrity: {\"Valid\" if is_valid else \"Corrupted\"}')

asyncio.run(check_state())
"
```

### Issue 3: Very Slow Collection Speed

**Symptoms:**
- Collection speed <30 stocks/minute
- High response times in logs
- System appears to be waiting frequently

**Diagnosis:**
```bash
# Check current performance
python performance_analyzer.py --real-time

# Monitor system resources
top -p $(pgrep -f continuous_collection)

# Check network latency
ping -c 5 www.alphavantage.co
ping -c 5 query1.finance.yahoo.com
```

**Solutions:**

1. **Network Issues:**
```bash
# Test API connectivity
curl -w "@curl-format.txt" -o /dev/null -s "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=AAPL&apikey=demo"

# Create curl-format.txt:
cat > curl-format.txt << EOF
     time_namelookup:  %{time_namelookup}\n
        time_connect:  %{time_connect}\n
     time_appconnect:  %{time_appconnect}\n
    time_pretransfer:  %{time_pretransfer}\n
       time_redirect:  %{time_redirect}\n
  time_starttransfer:  %{time_starttransfer}\n
                     ----------\n
          time_total:  %{time_total}\n
EOF
```

2. **Rate Limiting:**
```bash
# Check API key usage
python -c "
from continuous_data_collection.core.api_manager import APIManager
manager = APIManager()
stats = manager.get_usage_stats()
for key_id, usage in stats.items():
    print(f'Key {key_id}: {usage} requests/minute')
"

# Reduce rate limit if hitting limits
# Edit config/production.yaml:
# apis.alpha_vantage.rate_limit: 70  # Reduce from 75
```

3. **System Resource Constraints:**
```bash
# Check I/O wait
iostat -x 1 5

# Check CPU usage per core
mpstat -P ALL 1 5

# Optimize worker count
python -c "
import psutil
cpu_count = psutil.cpu_count()
memory_gb = psutil.virtual_memory().total / (1024**3)
optimal_workers = min(cpu_count, int(memory_gb / 2), 12)
print(f'Recommended workers: {optimal_workers}')
"
```

## API-Related Problems

### Alpha Vantage API Issues

**Issue: Rate Limit Exceeded (HTTP 429)**

**Symptoms:**
```
API Error: Rate limit exceeded for key 1
HTTP 429: Too Many Requests
Switching to next API key
```

**Solutions:**

1. **Verify API Key Limits:**
```bash
# Test each API key individually
python -c "
import asyncio
from continuous_data_collection.collectors.alpha_vantage_client import AlphaVantageClient

async def test_keys():
    client = AlphaVantageClient()
    for i, key in enumerate(client.api_keys):
        client.current_key_index = i
        result = await client.get_daily_data('AAPL')
        print(f'Key {i+1}: {\"✓\" if result.success else \"✗\"} - {result.error_message or \"OK\"}')

asyncio.run(test_keys())
"
```

2. **Adjust Rate Limiting:**
```yaml
# config/production.yaml
apis:
  alpha_vantage:
    rate_limit: 70        # Reduce from 75 for safety margin
    burst_allowance: 3    # Allow small bursts
    backoff_multiplier: 2.0
```

3. **Monitor API Usage:**
```bash
# Real-time API usage monitoring
watch -n 10 'python -c "
from continuous_data_collection.core.api_manager import APIManager
manager = APIManager()
stats = manager.get_usage_stats()
for key_id, usage in stats.items():
    print(f\"Key {key_id}: {usage} requests/minute\")
"'
```

**Issue: Invalid API Key (HTTP 401)**

**Symptoms:**
```
API Error: Invalid API key
HTTP 401: Unauthorized
Authentication failed for Alpha Vantage
```

**Solutions:**

1. **Verify API Keys:**
```bash
# Test API key manually
curl "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=AAPL&apikey=YOUR_API_KEY"

# Should return JSON data, not error message
```

2. **Check Environment Variables:**
```bash
# Verify environment variables are set
echo $ALPHA_VANTAGE_API_KEY_1
echo $ALPHA_VANTAGE_API_KEY_2
echo $ALPHA_VANTAGE_API_KEY_3
echo $ALPHA_VANTAGE_API_KEY_4

# Check .env file
cat .env | grep ALPHA_VANTAGE
```

3. **Update API Keys:**
```bash
# Update .env file with new keys
nano .env

# Restart system to reload keys
python stop_collection.py
python start_collection.py
```

### yfinance Fallback Issues

**Issue: yfinance Data Incomplete**

**Symptoms:**
```
yfinance fallback successful but data quality low
Insufficient historical data from yfinance
Data validation failed for yfinance source
```

**Solutions:**

1. **Check yfinance Data Quality:**
```python
# Test yfinance data quality
import yfinance as yf
import pandas as pd

def test_yfinance_quality(symbol):
    ticker = yf.Ticker(symbol)
    data = ticker.history(period="max")
    
    print(f"Symbol: {symbol}")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    print(f"Years of data: {(data.index.max() - data.index.min()).days / 365.25:.1f}")
    print(f"Total records: {len(data)}")
    print(f"Missing values: {data.isnull().sum().sum()}")
    
    return data

# Test with sample stocks
for symbol in ['AAPL', 'MSFT', 'GOOGL']:
    test_yfinance_quality(symbol)
    print("-" * 40)
```

2. **Adjust Data Quality Thresholds:**
```yaml
# config/production.yaml
data_validation:
  min_years_required: 8      # Reduce from 10 for yfinance
  min_years_preferred: 15    # Reduce from 20
  max_missing_dates_pct: 0.1 # Allow 10% missing dates
```

## Network and Connectivity Issues

### DNS Resolution Problems

**Symptoms:**
```
DNS resolution failed for www.alphavantage.co
getaddrinfo failed
Network unreachable
```

**Diagnosis:**
```bash
# Test DNS resolution
nslookup www.alphavantage.co
nslookup query1.finance.yahoo.com

# Test with different DNS servers
nslookup www.alphavantage.co 8.8.8.8
nslookup www.alphavantage.co 1.1.1.1
```

**Solutions:**

1. **Configure Alternative DNS:**
```bash
# Temporarily use Google DNS
echo "nameserver 8.8.8.8" | sudo tee /etc/resolv.conf.backup
echo "nameserver 1.1.1.1" | sudo tee -a /etc/resolv.conf.backup
sudo cp /etc/resolv.conf.backup /etc/resolv.conf
```

2. **Add DNS Caching:**
```bash
# Install local DNS cache (Ubuntu)
sudo apt install systemd-resolved
sudo systemctl enable systemd-resolved
sudo systemctl start systemd-resolved
```

### SSL/TLS Certificate Issues

**Symptoms:**
```
SSL certificate verification failed
CERTIFICATE_VERIFY_FAILED
SSL: WRONG_VERSION_NUMBER
```

**Solutions:**

1. **Update CA Certificates:**
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install ca-certificates

# CentOS/RHEL
sudo yum update ca-certificates
```

2. **Check System Time:**
```bash
# Verify system time is correct
date
timedatectl status

# Sync time if needed
sudo ntpdate -s time.nist.gov
```

### Proxy and Firewall Issues

**Symptoms:**
```
Connection timeout
Connection refused
Proxy authentication required
```

**Solutions:**

1. **Configure Proxy Settings:**
```bash
# Set proxy environment variables
export HTTP_PROXY=http://proxy.company.com:8080
export HTTPS_PROXY=http://proxy.company.com:8080
export NO_PROXY=localhost,127.0.0.1

# For authenticated proxy
export HTTP_PROXY=http://username:password@proxy.company.com:8080
```

2. **Test Direct Connectivity:**
```bash
# Test without proxy
unset HTTP_PROXY HTTPS_PROXY
curl -I https://www.alphavantage.co/

# Test with proxy
export HTTP_PROXY=http://proxy.company.com:8080
curl -I https://www.alphavantage.co/
```

## Performance Problems

### High Memory Usage

**Symptoms:**
```
Memory usage: 95%
System swapping detected
Out of memory errors
```

**Diagnosis:**
```bash
# Check memory usage by process
ps aux --sort=-%mem | head -10

# Check swap usage
swapon --show
free -h

# Monitor memory usage over time
vmstat 5 12
```

**Solutions:**

1. **Reduce Worker Count:**
```yaml
# config/production.yaml
system:
  max_workers: 4        # Reduce from 8
  batch_size: 10        # Reduce from 20
```

2. **Optimize Memory Settings:**
```python
# Add to config/production.yaml
memory_management:
  gc_threshold: 1000    # Force garbage collection
  max_buffer_size: 50   # Limit buffer sizes
  clear_cache_interval: 300  # Clear caches every 5 minutes
```

3. **Enable Memory Monitoring:**
```bash
# Monitor memory usage continuously
python -c "
import psutil
import time

while True:
    mem = psutil.virtual_memory()
    print(f'Memory: {mem.percent:.1f}% used, {mem.available / (1024**3):.1f} GB available')
    time.sleep(60)
" &
```

### High CPU Usage

**Symptoms:**
```
CPU usage consistently >90%
System becomes unresponsive
High load average
```

**Diagnosis:**
```bash
# Check CPU usage per core
mpstat -P ALL 1 5

# Check load average
uptime

# Identify CPU-intensive processes
top -o %CPU
```

**Solutions:**

1. **Optimize Worker Configuration:**
```yaml
# config/production.yaml
system:
  max_workers: 6        # Don't exceed CPU core count
  cpu_affinity: true    # Enable CPU affinity
  nice_level: 10        # Lower process priority
```

2. **Implement CPU Throttling:**
```python
# Add CPU usage monitoring
import psutil
import time

def monitor_cpu_usage():
    while True:
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 85:
            print(f"High CPU usage: {cpu_percent}%")
            # Implement throttling logic
            time.sleep(2)  # Brief pause
```

### Disk I/O Bottlenecks

**Symptoms:**
```
High I/O wait times
Slow file operations
Disk queue length >10
```

**Diagnosis:**
```bash
# Check I/O statistics
iostat -x 1 5

# Check disk usage
df -h
du -sh data/ logs/ backups/

# Monitor I/O in real-time
iotop -o
```

**Solutions:**

1. **Optimize Storage Configuration:**
```yaml
# config/production.yaml
storage:
  compression: "lz4"     # Faster compression
  write_buffer_size: 64  # Larger write buffers
  sync_interval: 30      # Less frequent syncing
```

2. **Move to Faster Storage:**
```bash
# Check if using SSD
lsblk -d -o name,rota
# rota=1 means HDD, rota=0 means SSD

# Consider moving data to SSD mount point
sudo mkdir /mnt/ssd_data
sudo mount /dev/nvme0n1p1 /mnt/ssd_data
ln -sf /mnt/ssd_data ./data
```

## Data Quality Issues

### High Rejection Rate

**Symptoms:**
```
Data quality score: 0.45 (below threshold)
Rejection rate: 35%
Many stocks failing validation
```

**Diagnosis:**
```bash
# Check data quality report
python data_quality_reporter.py --detailed

# Analyze rejection reasons
grep "Data rejected" logs/data_quality.log | tail -20

# Check specific failed stocks
python -c "
from continuous_data_collection.core.state_manager import StateManager
import asyncio

async def check_failures():
    state_manager = StateManager()
    state = await state_manager.load_state()
    print('Top 10 failed stocks:')
    for symbol, count in list(state.failed_stocks.items())[:10]:
        print(f'  {symbol}: {count} failures')

asyncio.run(check_failures())
"
```

**Solutions:**

1. **Adjust Quality Thresholds:**
```yaml
# config/production.yaml
data_validation:
  min_quality_score: 0.6    # Reduce from 0.8
  min_years_required: 8     # Reduce from 10
  max_missing_dates_pct: 0.15  # Allow more missing dates
```

2. **Implement Graduated Quality Standards:**
```python
# Implement tiered quality standards
quality_tiers = {
    'premium': {'min_years': 20, 'min_score': 0.9},
    'standard': {'min_years': 15, 'min_score': 0.7},
    'acceptable': {'min_years': 10, 'min_score': 0.5}
}
```

### Missing Historical Data

**Symptoms:**
```
Insufficient historical data: 5.2 years
Required minimum: 10 years
Data starts from 2019-01-01
```

**Solutions:**

1. **Check Alternative Data Sources:**
```python
# Test multiple data sources for historical coverage
import yfinance as yf
from datetime import datetime, timedelta

def check_data_coverage(symbol):
    sources = {
        'yfinance': yf.Ticker(symbol).history(period="max"),
        # Add other sources as needed
    }
    
    for source_name, data in sources.items():
        if not data.empty:
            years = (data.index.max() - data.index.min()).days / 365.25
            print(f"{source_name}: {years:.1f} years ({data.index.min()} to {data.index.max()})")

# Test problematic stocks
for symbol in ['NEWER_STOCK', 'IPO_STOCK']:
    print(f"\n{symbol}:")
    check_data_coverage(symbol)
```

2. **Implement Flexible Date Requirements:**
```yaml
# config/production.yaml
data_validation:
  flexible_date_requirements: true
  min_years_by_category:
    large_cap: 15
    mid_cap: 10
    small_cap: 5
    recent_ipo: 2
```

## System Resource Problems

### Disk Space Issues

**Symptoms:**
```
No space left on device
Disk usage: 98%
Failed to write data file
```

**Immediate Actions:**
```bash
# Check disk usage
df -h

# Find large files
find . -type f -size +100M -exec ls -lh {} \;

# Clean up immediately
rm -rf temp/*
find logs/ -name "*.log.*" -mtime +3 -delete
```

**Long-term Solutions:**

1. **Implement Automatic Cleanup:**
```python
# Add to system configuration
cleanup_policy = {
    'temp_files_max_age_hours': 24,
    'log_files_max_age_days': 7,
    'backup_files_max_count': 5,
    'compress_old_logs': True
}
```

2. **Set Up Disk Monitoring:**
```bash
# Create disk monitoring script
cat > monitor_disk.sh << 'EOF'
#!/bin/bash
THRESHOLD=90
USAGE=$(df / | awk 'NR==2 {print $5}' | sed 's/%//')

if [ $USAGE -gt $THRESHOLD ]; then
    echo "Disk usage is ${USAGE}% - cleaning up..."
    find temp/ -type f -mtime +1 -delete
    find logs/ -name "*.log.*" -mtime +7 -delete
fi
EOF

chmod +x monitor_disk.sh

# Add to crontab
echo "0 */6 * * * /path/to/monitor_disk.sh" | crontab -
```

### File Handle Exhaustion

**Symptoms:**
```
Too many open files
OSError: [Errno 24] Too many open files
File handle limit exceeded
```

**Solutions:**

1. **Increase File Handle Limits:**
```bash
# Check current limits
ulimit -n

# Increase temporarily
ulimit -n 65536

# Increase permanently
echo "* soft nofile 65536" | sudo tee -a /etc/security/limits.conf
echo "* hard nofile 65536" | sudo tee -a /etc/security/limits.conf
```

2. **Fix File Handle Leaks:**
```python
# Add proper file handle management
import contextlib

@contextlib.contextmanager
def managed_file_operations():
    """Ensure proper file handle cleanup."""
    open_files = []
    try:
        yield open_files
    finally:
        for f in open_files:
            if not f.closed:
                f.close()
```

## State and Recovery Issues

### State Corruption

**Symptoms:**
```
State file corrupted
JSON decode error
State validation failed
```

**Diagnosis:**
```bash
# Check state file integrity
python -c "
import json
try:
    with open('backups/state/system_state.json', 'r') as f:
        state = json.load(f)
    print('State file is valid JSON')
except json.JSONDecodeError as e:
    print(f'State file corrupted: {e}')
"
```

**Recovery:**

1. **Restore from Backup:**
```bash
# List available backups
ls -la backups/state/

# Restore from most recent backup
python emergency_recovery.py --restore-state --backup-id latest

# Verify restored state
python -c "
from continuous_data_collection.core.state_manager import StateManager
import asyncio

async def verify_state():
    state_manager = StateManager()
    is_valid = await state_manager.verify_state_integrity()
    print(f'State integrity: {\"Valid\" if is_valid else \"Corrupted\"}')

asyncio.run(verify_state())
"
```

2. **Rebuild State from Data:**
```bash
# Rebuild state from collected data files
python emergency_recovery.py --rebuild-state-from-data

# This will:
# - Scan data directory for collected files
# - Rebuild completed_stocks list
# - Reset failed_stocks counters
# - Recalculate progress statistics
```

### Recovery Point Issues

**Symptoms:**
```
Cannot determine resume point
Duplicate collection detected
Progress calculation incorrect
```

**Solutions:**

1. **Manual Progress Verification:**
```python
# Verify progress against actual data
import os
import glob

def verify_progress():
    data_files = glob.glob('data/raw/*.parquet')
    collected_symbols = set()
    
    for file_path in data_files:
        filename = os.path.basename(file_path)
        symbol = filename.replace('.parquet', '')
        collected_symbols.add(symbol)
    
    print(f"Files found: {len(data_files)}")
    print(f"Unique symbols: {len(collected_symbols)}")
    
    # Compare with state
    from continuous_data_collection.core.state_manager import StateManager
    import asyncio
    
    async def compare_state():
        state_manager = StateManager()
        state = await state_manager.load_state()
        
        state_completed = set(state.completed_stocks)
        file_symbols = collected_symbols
        
        print(f"State completed: {len(state_completed)}")
        print(f"File symbols: {len(file_symbols)}")
        
        missing_in_state = file_symbols - state_completed
        missing_files = state_completed - file_symbols
        
        if missing_in_state:
            print(f"Files exist but not in state: {len(missing_in_state)}")
        if missing_files:
            print(f"In state but no files: {len(missing_files)}")
    
    asyncio.run(compare_state())

verify_progress()
```

2. **Synchronize State with Reality:**
```bash
# Synchronize state with actual data files
python emergency_recovery.py --sync-state-with-files
```

## Configuration Problems

### YAML Syntax Errors

**Symptoms:**
```
YAML parsing error
Configuration validation failed
Invalid configuration format
```

**Diagnosis:**
```bash
# Validate YAML syntax
python -c "
import yaml
try:
    with open('config/production.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print('YAML syntax is valid')
except yaml.YAMLError as e:
    print(f'YAML syntax error: {e}')
"
```

**Solutions:**

1. **Fix Common YAML Issues:**
```bash
# Check for common issues
grep -n ":" config/production.yaml | grep -v " "  # Missing spaces after colons
grep -n "^[[:space:]]*-" config/production.yaml   # Check list formatting
```

2. **Use Configuration Template:**
```bash
# Restore from template
cp config/production.yaml.template config/production.yaml
# Then edit with your specific values
```

### Environment Variable Issues

**Symptoms:**
```
API key not found
Environment variable not set
Configuration value missing
```

**Solutions:**

1. **Verify Environment Variables:**
```bash
# Check all required environment variables
python -c "
import os
required_vars = [
    'ALPHA_VANTAGE_API_KEY_1',
    'ALPHA_VANTAGE_API_KEY_2',
    'ALPHA_VANTAGE_API_KEY_3',
    'ALPHA_VANTAGE_API_KEY_4',
    'ENVIRONMENT'
]

for var in required_vars:
    value = os.getenv(var)
    status = '✓' if value else '✗'
    print(f'{var}: {status}')
"
```

2. **Load Environment File:**
```bash
# Ensure .env file is loaded
source .env  # For bash
# OR
set -a; source .env; set +a  # For more compatibility
```

## Emergency Procedures

### Complete System Recovery

**When to Use:** System completely unresponsive, multiple component failures

**Steps:**

1. **Stop All Processes:**
```bash
# Force stop all related processes
pkill -f continuous_collection
pkill -f python.*start_collection

# Verify no processes running
ps aux | grep continuous
```

2. **Backup Current State:**
```bash
# Create emergency backup
mkdir -p emergency_backup_$(date +%Y%m%d_%H%M%S)
cp -r backups/state emergency_backup_$(date +%Y%m%d_%H%M%S)/
cp -r logs emergency_backup_$(date +%Y%m%d_%H%M%S)/
cp config/production.yaml emergency_backup_$(date +%Y%m%d_%H%M%S)/
```

3. **Run Emergency Recovery:**
```bash
# Full emergency recovery
python emergency_recovery.py --full-recovery

# This will:
# - Verify system integrity
# - Restore from best available backup
# - Rebuild corrupted state files
# - Validate configuration
# - Test API connectivity
```

4. **Restart System:**
```bash
# Start with verbose logging
python start_collection.py --verbose --log-level DEBUG

# Monitor startup
tail -f logs/continuous_collection.log
```

### Data Corruption Recovery

**When to Use:** Data files corrupted, validation failures

**Steps:**

1. **Identify Corrupted Files:**
```bash
# Check all data files
python -c "
import os
import pandas as pd
import glob

corrupted_files = []
data_files = glob.glob('data/raw/*.parquet')

for file_path in data_files:
    try:
        df = pd.read_parquet(file_path)
        if df.empty or len(df) < 100:  # Basic validation
            corrupted_files.append(file_path)
    except Exception as e:
        corrupted_files.append(file_path)
        print(f'Corrupted: {file_path} - {e}')

print(f'Found {len(corrupted_files)} corrupted files')
"
```

2. **Quarantine Corrupted Files:**
```bash
# Move corrupted files to quarantine
mkdir -p data/quarantine
# Move files identified above
```

3. **Rebuild from Source:**
```bash
# Re-collect corrupted stocks
python -c "
# Extract symbols from corrupted files
corrupted_symbols = []
# Add logic to extract symbols from file names
print(f'Need to re-collect: {corrupted_symbols}')
"

# Add to failed stocks for re-collection
python emergency_recovery.py --requeue-stocks AAPL,MSFT,GOOGL
```

## Log Analysis

### Log File Structure

```
logs/
├── continuous_collection.log      # Main system log
├── api_requests.log              # API request/response details
├── data_quality.log              # Data validation results
├── errors.log                    # Error-specific entries
├── performance.log               # Performance metrics
└── debug.log                     # Debug information
```

### Useful Log Analysis Commands

**Error Analysis:**
```bash
# Find all errors in the last hour
find logs/ -name "*.log" -mmin -60 -exec grep -H "ERROR" {} \;

# Count errors by type
grep "ERROR" logs/*.log | cut -d: -f3 | sort | uniq -c | sort -nr

# Find API-specific errors
grep -E "(HTTP [45][0-9][0-9]|API Error)" logs/api_requests.log | tail -20
```

**Performance Analysis:**
```bash
# Track throughput over time
grep "stocks/minute" logs/continuous_collection.log | tail -20

# Find slow operations
grep -E "(took [0-9]+\.[0-9]+ seconds)" logs/*.log | awk '$NF > 10' | tail -10

# Memory usage trends
grep "Memory usage" logs/continuous_collection.log | tail -20
```

**Progress Tracking:**
```bash
# Collection progress over time
grep "Progress:" logs/continuous_collection.log | tail -20

# Success rate trends
grep "Success rate" logs/continuous_collection.log | tail -10

# ETA calculations
grep "ETA:" logs/continuous_collection.log | tail -10
```

### Log Rotation and Management

**Set up log rotation:**
```bash
# Create logrotate configuration
sudo tee /etc/logrotate.d/continuous-collection << EOF
/path/to/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 644 user group
    postrotate
        # Signal application to reopen log files if needed
        pkill -USR1 -f continuous_collection || true
    endscript
}
EOF
```

## Advanced Debugging

### Python Debugging

**Enable Debug Mode:**
```bash
# Start with debug logging
python start_collection.py --log-level DEBUG

# Enable Python debugging
export PYTHONPATH=$PYTHONPATH:.
export PYTHONDEBUG=1
python -u start_collection.py
```

**Memory Profiling:**
```python
# Add memory profiling
import tracemalloc
import psutil
import gc

def profile_memory():
    tracemalloc.start()
    
    # Your code here
    
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
    print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")
    
    # Force garbage collection
    gc.collect()
    
    tracemalloc.stop()
```

**Performance Profiling:**
```python
# Add performance profiling
import cProfile
import pstats

def profile_performance():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Your code here
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions
```

### Network Debugging

**Capture Network Traffic:**
```bash
# Monitor HTTP requests
sudo tcpdump -i any -A -s 0 'host www.alphavantage.co or host query1.finance.yahoo.com'

# Use mitmproxy for detailed HTTP analysis
pip install mitmproxy
mitmdump -s capture_script.py
```

**Test API Endpoints:**
```bash
# Comprehensive API testing
curl -v -w "@curl-format.txt" \
  "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=AAPL&apikey=demo"

# Test with different user agents
curl -H "User-Agent: Mozilla/5.0" \
  "https://query1.finance.yahoo.com/v8/finance/chart/AAPL"
```

This comprehensive troubleshooting guide should help you diagnose and resolve most issues that may arise with the continuous data collection system. Always start with the quick diagnostics and work through the specific issue categories as needed.