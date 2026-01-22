# Continuous Data Collection System - Deployment Guide

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Environment Setup](#environment-setup)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Deployment Steps](#deployment-steps)
6. [Verification](#verification)
7. [Troubleshooting](#troubleshooting)
8. [Performance Tuning](#performance-tuning)

## System Requirements

### Hardware Requirements

**Minimum Requirements:**
- CPU: 4 cores, 2.5 GHz
- RAM: 8 GB
- Storage: 100 GB free space (SSD recommended)
- Network: Stable internet connection (10 Mbps+)

**Recommended Requirements:**
- CPU: 8+ cores, 3.0+ GHz
- RAM: 16+ GB
- Storage: 500+ GB free space (NVMe SSD)
- Network: High-speed internet (50+ Mbps)

### Software Requirements

**Operating System:**
- Linux (Ubuntu 20.04+ LTS recommended)
- Windows 10/11 (with WSL2 for optimal performance)
- macOS 11+ (Intel or Apple Silicon)

**Python Environment:**
- Python 3.8+ (Python 3.9+ recommended)
- pip 21.0+
- virtualenv or conda

**System Dependencies:**
- Git 2.25+
- curl or wget
- SQLite 3.31+
- OpenSSL 1.1.1+

### Network Requirements

**Outbound Connections:**
- Alpha Vantage API: `www.alphavantage.co` (HTTPS/443)
- Yahoo Finance: `query1.finance.yahoo.com` (HTTPS/443)
- Python Package Index: `pypi.org` (HTTPS/443)

**Firewall Configuration:**
- Allow outbound HTTPS (port 443)
- Allow outbound HTTP (port 80) for redirects
- No inbound ports required

## Environment Setup

### 1. System Preparation

**Ubuntu/Debian:**
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y python3 python3-pip python3-venv git curl sqlite3 build-essential

# Install Python development headers
sudo apt install -y python3-dev libffi-dev libssl-dev
```

**CentOS/RHEL:**
```bash
# Update system packages
sudo yum update -y

# Install system dependencies
sudo yum install -y python3 python3-pip git curl sqlite gcc gcc-c++ make

# Install development tools
sudo yum groupinstall -y "Development Tools"
```

**Windows (PowerShell as Administrator):**
```powershell
# Install Chocolatey (if not already installed)
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))

# Install dependencies
choco install python git sqlite -y
```

### 2. Python Environment Setup

**Create Virtual Environment:**
```bash
# Create project directory
mkdir -p /opt/continuous-data-collection
cd /opt/continuous-data-collection

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/macOS
# OR
venv\Scripts\activate     # Windows
```

**Verify Python Installation:**
```bash
python --version  # Should be 3.8+
pip --version     # Should be 21.0+
```

## Installation

### 1. Clone Repository

```bash
# Clone the repository
git clone <repository-url> .

# Verify project structure
ls -la
# Should show: continuous_data_collection/, config/, tests/, etc.
```

### 2. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt

# Verify installation
python -c "import continuous_data_collection; print('Installation successful')"
```

### 3. Install System Scripts

```bash
# Make scripts executable
chmod +x start_collection.py
chmod +x stop_collection.py
chmod +x check_progress.py
chmod +x emergency_recovery.py

# Create symbolic links (optional, for system-wide access)
sudo ln -sf $(pwd)/start_collection.py /usr/local/bin/start-collection
sudo ln -sf $(pwd)/stop_collection.py /usr/local/bin/stop-collection
sudo ln -sf $(pwd)/check_progress.py /usr/local/bin/check-progress
```

## Configuration

### 1. API Keys Setup

**Alpha Vantage API Keys:**
```bash
# Create environment file
cp config/production.yaml.template config/production.yaml

# Edit configuration file
nano config/production.yaml
```

**Required API Keys:**
- Obtain 4 Alpha Vantage API keys from: https://www.alphavantage.co/support/#api-key
- Each key allows 75 requests per minute
- Total capacity: 300 requests per minute

**Environment Variables:**
```bash
# Create .env file
cat > .env << EOF
ALPHA_VANTAGE_API_KEY_1=your_first_api_key
ALPHA_VANTAGE_API_KEY_2=your_second_api_key
ALPHA_VANTAGE_API_KEY_3=your_third_api_key
ALPHA_VANTAGE_API_KEY_4=your_fourth_api_key
ENVIRONMENT=production
LOG_LEVEL=INFO
EOF

# Secure the environment file
chmod 600 .env
```

### 2. Configuration File Setup

**Production Configuration (`config/production.yaml`):**
```yaml
# System Configuration
system:
  environment: "production"
  log_level: "INFO"
  max_workers: 8
  batch_size: 20
  
# Data Collection
collection:
  target_throughput: 65  # stocks per minute
  retry_limit: 5
  timeout_seconds: 30
  
# Storage
storage:
  data_directory: "./data"
  backup_directory: "./backups"
  compression: "snappy"
  
# Monitoring
monitoring:
  progress_save_interval: 10
  health_check_interval: 60
  alert_thresholds:
    failure_rate: 0.1
    throughput_min: 30
    
# API Configuration
apis:
  alpha_vantage:
    rate_limit: 75  # per minute per key
    timeout: 30
  yfinance:
    timeout: 15
    retry_delay: 1
```

### 3. Directory Structure Setup

```bash
# Create required directories
mkdir -p data/{raw,processed,failed}
mkdir -p logs
mkdir -p backups/state
mkdir -p temp

# Set appropriate permissions
chmod 755 data logs backups temp
chmod 700 backups/state  # Sensitive state data
```

### 4. Stock Lists Setup

**Download Stock Lists:**
```bash
# Create stock lists directory
mkdir -p data/stock_lists

# S&P 500 stocks (highest priority)
curl -o data/stock_lists/sp500.txt "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"

# Convert CSV to simple text list (symbols only)
python3 -c "
import pandas as pd
df = pd.read_csv('data/stock_lists/sp500.txt')
with open('data/stock_lists/sp500_symbols.txt', 'w') as f:
    for symbol in df['Symbol']:
        f.write(f'{symbol}\n')
"
```

**Manual Stock Lists (if needed):**
```bash
# Create custom stock lists
cat > data/stock_lists/priority_stocks.txt << EOF
AAPL
MSFT
GOOGL
AMZN
TSLA
EOF
```

## Deployment Steps

### 1. Pre-Deployment Validation

**System Requirements Check:**
```bash
# Run system requirements checker
python system_requirements_checker.py

# Expected output: All checks should pass
# - Python version: ✓
# - Dependencies: ✓
# - Disk space: ✓
# - Network connectivity: ✓
```

**Configuration Validation:**
```bash
# Validate configuration
python config_validator.py

# Expected output: Configuration validation successful
```

**API Connectivity Test:**
```bash
# Test API connections
python -c "
from continuous_data_collection.collectors.alpha_vantage_client import AlphaVantageClient
from continuous_data_collection.collectors.yfinance_client import YfinanceClient
import asyncio

async def test_apis():
    av_client = AlphaVantageClient()
    yf_client = YfinanceClient()
    
    # Test Alpha Vantage
    av_result = await av_client.get_daily_data('AAPL')
    print(f'Alpha Vantage: {\"✓\" if av_result.success else \"✗\"}')
    
    # Test yfinance
    yf_result = await yf_client.get_daily_data('AAPL')
    print(f'yfinance: {\"✓\" if yf_result.success else \"✗\"}')

asyncio.run(test_apis())
"
```

### 2. Initial Deployment

**Deploy System:**
```bash
# Run deployment setup script
python deployment_setup.py

# This script will:
# - Validate all configurations
# - Create necessary directories
# - Initialize state files
# - Set up logging
# - Verify API connectivity
```

**Initialize System State:**
```bash
# Initialize empty state
python -c "
from continuous_data_collection.core.state_manager import StateManager
import asyncio

async def init_state():
    state_manager = StateManager()
    await state_manager.initialize_empty_state()
    print('State initialized successfully')

asyncio.run(init_state())
"
```

### 3. Test Deployment

**Run System Tests:**
```bash
# Run comprehensive test suite
python run_tests.py all

# Expected: All tests should pass
# If any tests fail, review logs and fix issues before proceeding
```

**Dry Run Test:**
```bash
# Test with a small subset of stocks
python -c "
import asyncio
from continuous_data_collection.core.continuous_collector import ContinuousCollector

async def dry_run():
    collector = ContinuousCollector()
    # Test with just 5 stocks
    await collector.start_collection(max_stocks=5, dry_run=True)

asyncio.run(dry_run())
"
```

### 4. Production Deployment

**Start System:**
```bash
# Start the collection system
python start_collection.py

# Monitor initial startup
tail -f logs/continuous_collection.log
```

**Verify System Status:**
```bash
# Check system status
python check_progress.py

# Expected output:
# System Status: Running
# Progress: X/1400 stocks collected
# Success Rate: XX%
# ETA: X hours remaining
```

## Verification

### 1. System Health Checks

**Basic Health Check:**
```bash
# Run health check
python -c "
from continuous_data_collection.monitoring.health_monitor import HealthMonitor
import asyncio

async def health_check():
    monitor = HealthMonitor()
    status = await monitor.check_system_health()
    print(f'System Health: {status.overall_status}')
    for component, health in status.component_health.items():
        print(f'  {component}: {health}')

asyncio.run(health_check())
"
```

**Resource Monitoring:**
```bash
# Monitor system resources
python system_health_dashboard.py --check-once

# Expected output:
# CPU Usage: <80%
# Memory Usage: <80%
# Disk Usage: <90%
# Network: Connected
```

### 2. Data Quality Verification

**Check Collected Data:**
```bash
# Verify data quality
python data_quality_reporter.py

# Expected output:
# Total files: X
# Average data quality score: >0.8
# Files with 20+ years: X%
# Files with 10+ years: X%
```

**Sample Data Inspection:**
```bash
# Inspect sample collected data
python -c "
import pandas as pd
import os

data_dir = 'data/raw'
files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
if files:
    sample_file = files[0]
    df = pd.read_parquet(os.path.join(data_dir, sample_file))
    print(f'Sample file: {sample_file}')
    print(f'Date range: {df.index.min()} to {df.index.max()}')
    print(f'Records: {len(df)}')
    print(f'Columns: {list(df.columns)}')
else:
    print('No data files found yet')
"
```

### 3. Performance Verification

**Throughput Check:**
```bash
# Check collection throughput
python performance_analyzer.py --real-time

# Expected output:
# Current throughput: 60-70 stocks/minute
# Average response time: <5 seconds
# Success rate: >90%
```

**Progress Tracking:**
```bash
# Monitor progress over time
watch -n 30 'python check_progress.py --brief'

# Should show steady progress
```

## Troubleshooting

### Common Issues

#### 1. API Rate Limiting

**Symptoms:**
- HTTP 429 errors in logs
- Reduced throughput
- "Rate limit exceeded" messages

**Solutions:**
```bash
# Check API key usage
python -c "
from continuous_data_collection.core.api_manager import APIManager
manager = APIManager()
print(manager.get_usage_stats())
"

# Verify API key rotation
grep "API key rotation" logs/continuous_collection.log

# Adjust rate limiting if needed
# Edit config/production.yaml:
# apis.alpha_vantage.rate_limit: 70  # Reduce from 75
```

#### 2. Network Connectivity Issues

**Symptoms:**
- Connection timeout errors
- DNS resolution failures
- Intermittent API failures

**Solutions:**
```bash
# Test network connectivity
curl -I https://www.alphavantage.co/
curl -I https://query1.finance.yahoo.com/

# Check DNS resolution
nslookup www.alphavantage.co
nslookup query1.finance.yahoo.com

# Test with different DNS servers
echo "nameserver 8.8.8.8" | sudo tee /etc/resolv.conf.backup
```

#### 3. Disk Space Issues

**Symptoms:**
- "No space left on device" errors
- System slowdown
- Failed data writes

**Solutions:**
```bash
# Check disk usage
df -h

# Clean up temporary files
python -c "
import os, glob
temp_files = glob.glob('temp/*')
for f in temp_files:
    os.remove(f)
print(f'Cleaned {len(temp_files)} temporary files')
"

# Compress old log files
gzip logs/*.log.1 logs/*.log.2
```

#### 4. Memory Issues

**Symptoms:**
- Out of memory errors
- System swapping
- Slow performance

**Solutions:**
```bash
# Check memory usage
free -h
ps aux | grep python | head -10

# Reduce worker count
# Edit config/production.yaml:
# system.max_workers: 4  # Reduce from 8

# Restart system with new configuration
python stop_collection.py
python start_collection.py
```

#### 5. Data Quality Issues

**Symptoms:**
- High rejection rate
- Poor data quality scores
- Missing historical data

**Solutions:**
```bash
# Check data quality report
python data_quality_reporter.py --detailed

# Review failed collections
python -c "
from continuous_data_collection.core.state_manager import StateManager
import asyncio

async def check_failures():
    state_manager = StateManager()
    state = await state_manager.load_state()
    print(f'Failed stocks: {len(state.failed_stocks)}')
    for symbol, count in list(state.failed_stocks.items())[:10]:
        print(f'  {symbol}: {count} failures')

asyncio.run(check_failures())
"

# Retry failed stocks
python -c "
from continuous_data_collection.core.continuous_collector import ContinuousCollector
import asyncio

async def retry_failed():
    collector = ContinuousCollector()
    await collector.retry_failed_stocks()

asyncio.run(retry_failed())
"
```

### Emergency Procedures

#### System Recovery

**If system crashes:**
```bash
# Check system status
python check_progress.py

# If system is not running, check logs
tail -100 logs/continuous_collection.log

# Run emergency recovery
python emergency_recovery.py

# Restart system
python start_collection.py
```

**If state corruption detected:**
```bash
# Backup current state
cp -r backups/state backups/state_backup_$(date +%Y%m%d_%H%M%S)

# Run state recovery
python emergency_recovery.py --restore-state

# Verify state integrity
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

### Log Analysis

**Key Log Files:**
- `logs/continuous_collection.log`: Main system log
- `logs/api_requests.log`: API request/response log
- `logs/data_quality.log`: Data validation log
- `logs/errors.log`: Error-specific log

**Useful Log Commands:**
```bash
# Monitor real-time logs
tail -f logs/continuous_collection.log

# Search for errors
grep -i error logs/*.log

# Check API errors
grep "API error" logs/api_requests.log | tail -20

# Monitor progress
grep "Progress:" logs/continuous_collection.log | tail -10

# Check data quality issues
grep "Data quality" logs/data_quality.log | tail -20
```

## Performance Tuning

### System Optimization

#### 1. Worker Pool Tuning

**Optimal Worker Count:**
```python
# Calculate optimal workers based on system resources
import psutil

cpu_count = psutil.cpu_count()
memory_gb = psutil.virtual_memory().total / (1024**3)

# Conservative approach
optimal_workers = min(cpu_count, int(memory_gb / 2), 12)
print(f"Recommended workers: {optimal_workers}")
```

**Configuration Update:**
```yaml
# config/production.yaml
system:
  max_workers: 8  # Adjust based on calculation above
  batch_size: 20  # Increase for better throughput
```

#### 2. Rate Limiting Optimization

**API Key Utilization:**
```bash
# Monitor API key usage distribution
python -c "
from continuous_data_collection.core.api_manager import APIManager
manager = APIManager()
stats = manager.get_usage_stats()
for key_id, usage in stats.items():
    print(f'Key {key_id}: {usage} requests/minute')
"
```

**Rate Limit Tuning:**
```yaml
# config/production.yaml
apis:
  alpha_vantage:
    rate_limit: 73  # Slightly below 75 for safety margin
    burst_allowance: 5  # Allow short bursts
```

#### 3. Storage Optimization

**Compression Settings:**
```yaml
# config/production.yaml
storage:
  compression: "snappy"  # Fast compression
  compression_level: 6   # Balance speed vs size
```

**File System Optimization:**
```bash
# For Linux systems with ext4
sudo tune2fs -o journal_data_writeback /dev/your_disk
sudo mount -o remount,noatime /path/to/data/directory
```

#### 4. Network Optimization

**Connection Pooling:**
```yaml
# config/production.yaml
network:
  connection_pool_size: 20
  keep_alive_timeout: 30
  max_retries: 3
```

**DNS Caching:**
```bash
# Install and configure local DNS cache
sudo apt install systemd-resolved
sudo systemctl enable systemd-resolved
sudo systemctl start systemd-resolved
```

### Monitoring and Alerting

#### Performance Metrics

**Key Metrics to Monitor:**
- Collection throughput (stocks/minute)
- API response times
- Success/failure rates
- System resource usage
- Data quality scores

**Monitoring Setup:**
```bash
# Set up continuous monitoring
python system_health_dashboard.py --daemon

# Configure alerts
python -c "
from continuous_data_collection.monitoring.alert_system import AlertSystem
alert_system = AlertSystem()
alert_system.configure_alerts({
    'throughput_min': 30,
    'failure_rate_max': 0.15,
    'response_time_max': 10.0,
    'cpu_usage_max': 0.85,
    'memory_usage_max': 0.85
})
"
```

#### Performance Baselines

**Establish Baselines:**
```bash
# Run performance baseline test
python performance_analyzer.py --baseline --duration 3600

# Expected baselines:
# - Throughput: 60-70 stocks/minute
# - API response time: 2-5 seconds
# - Success rate: >95%
# - CPU usage: <70%
# - Memory usage: <60%
```

### Scaling Considerations

#### Horizontal Scaling

**Multi-Instance Deployment:**
```bash
# For very large stock universes (>5000 stocks)
# Deploy multiple instances with different stock lists

# Instance 1: S&P 500
python start_collection.py --stock-list data/stock_lists/sp500_symbols.txt

# Instance 2: S&P 400
python start_collection.py --stock-list data/stock_lists/sp400_symbols.txt --port 8001

# Instance 3: Remaining stocks
python start_collection.py --stock-list data/stock_lists/remaining_symbols.txt --port 8002
```

#### Vertical Scaling

**Resource Scaling:**
```yaml
# config/production.yaml for high-end systems
system:
  max_workers: 16      # For 16+ core systems
  batch_size: 50       # Larger batches
  memory_limit: "8GB"  # Explicit memory limit

storage:
  buffer_size: "100MB" # Larger I/O buffers
  write_batch_size: 1000
```

This deployment guide provides comprehensive instructions for setting up and running the continuous data collection system in production environments. Follow the steps carefully and refer to the troubleshooting section for any issues encountered during deployment.