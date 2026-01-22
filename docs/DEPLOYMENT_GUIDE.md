# AI Trading System - Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the AI Trading System across different environments, from local development to production deployment.

## Prerequisites

### System Requirements
- **Operating System**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+)
- **Python**: Version 3.11 or higher
- **Memory**: Minimum 8GB RAM (16GB recommended for production)
- **Storage**: Minimum 10GB free space (50GB recommended for production)
- **Network**: Stable internet connection for API access

### Required Software
- **Python 3.11+**: [Download from python.org](https://www.python.org/downloads/)
- **Git**: [Download from git-scm.com](https://git-scm.com/downloads)
- **Docker** (Optional): [Download from docker.com](https://www.docker.com/products/docker-desktop)

## Environment Setup

### 1. Development Environment

#### Clone Repository
```bash
git clone <repository-url>
cd TradingBOT
```

#### Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

#### Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Environment Variables
Create `.env` file in project root:
```env
# API Keys
QUESTRADE_CLIENT_ID=your_questrade_client_id
QUESTRADE_CLIENT_SECRET=your_questrade_client_secret
QUESTRADE_ACCESS_TOKEN=your_questrade_access_token
YAHOO_FINANCE_API_KEY=your_yahoo_api_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
FINNHUB_API_KEY=your_finnhub_key

# Database Configuration
DATABASE_URL=sqlite:///data/trading_demo.db
REDIS_URL=redis://localhost:6379

# System Configuration
LOG_LEVEL=INFO
MODE=DEMO
MAX_POSITIONS=10
RISK_LIMIT_PERCENT=2.0

# Monitoring
ENABLE_MONITORING=true
ALERT_EMAIL=your_email@example.com
SLACK_WEBHOOK_URL=your_slack_webhook_url
```

### 2. Staging Environment

#### Server Setup
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev
sudo apt install redis-server postgresql-client

# CentOS/RHEL
sudo yum install python3.11 python3.11-venv python3.11-devel
sudo yum install redis postgresql-client
```

#### Application Deployment
```bash
# Create application user
sudo useradd -m -s /bin/bash tradingbot
sudo su - tradingbot

# Clone and setup
git clone <repository-url>
cd TradingBOT
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### System Service
Create `/etc/systemd/system/tradingbot.service`:
```ini
[Unit]
Description=AI Trading System
After=network.target redis.service

[Service]
Type=simple
User=tradingbot
WorkingDirectory=/home/tradingbot/TradingBOT
Environment=PATH=/home/tradingbot/TradingBOT/venv/bin
ExecStart=/home/tradingbot/TradingBOT/venv/bin/python src/main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start service:
```bash
sudo systemctl enable tradingbot
sudo systemctl start tradingbot
```

### 3. Production Environment

#### Docker Deployment

##### Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory
RUN mkdir -p data logs

# Set permissions
RUN chmod +x src/main.py

# Expose port
EXPOSE 8050

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8050/health || exit 1

# Run application
CMD ["python", "src/main.py"]
```

##### Docker Compose
```yaml
version: '3.8'

services:
  tradingbot:
    build: .
    ports:
      - "8050:8050"
    environment:
      - MODE=DEMO
      - DATABASE_URL=sqlite:///data/trading_demo.db
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      - redis
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - tradingbot
    restart: unless-stopped

volumes:
  redis_data:
```

#### Cloud Deployment

##### AWS Deployment
```bash
# Install AWS CLI
pip install awscli

# Configure AWS credentials
aws configure

# Deploy using AWS ECS
aws ecs create-cluster --cluster-name tradingbot-cluster
aws ecs register-task-definition --cli-input-json file://task-definition.json
aws ecs create-service --cluster tradingbot-cluster --service-name tradingbot-service --task-definition tradingbot-task
```

##### Google Cloud Platform
```bash
# Install gcloud CLI
curl https://sdk.cloud.google.com | bash

# Deploy using Cloud Run
gcloud run deploy tradingbot \
    --source . \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated
```

##### Azure Deployment
```bash
# Install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Deploy using Container Instances
az container create \
    --resource-group tradingbot-rg \
    --name tradingbot-container \
    --image your-registry/tradingbot:latest \
    --ports 8050
```

## Configuration Management

### Mode Configuration

#### Demo Mode Configuration
```json
{
  "mode": "DEMO",
  "database": {
    "path": "data/trading_demo.db",
    "backup_interval": 3600
  },
  "trading": {
    "max_positions": 10,
    "position_size_percent": 10.0,
    "risk_limit_percent": 2.0
  },
  "ai": {
    "models": ["qwen3-coder:480b-cloud", "deepseek-v3.1:671b-cloud"],
    "confidence_threshold": 0.7
  }
}
```

#### Live Mode Configuration
```json
{
  "mode": "LIVE",
  "database": {
    "path": "data/trading_live.db",
    "backup_interval": 1800
  },
  "trading": {
    "max_positions": 5,
    "position_size_percent": 5.0,
    "risk_limit_percent": 1.0
  },
  "ai": {
    "models": ["gpt-oss:120b"],
    "confidence_threshold": 0.8
  }
}
```

### Security Configuration

#### API Key Management
```python
# Use environment variables for sensitive data
import os
from cryptography.fernet import Fernet

class SecureConfig:
    def __init__(self):
        self.encryption_key = os.getenv('ENCRYPTION_KEY')
        self.cipher = Fernet(self.encryption_key)
    
    def encrypt_api_key(self, api_key):
        return self.cipher.encrypt(api_key.encode())
    
    def decrypt_api_key(self, encrypted_key):
        return self.cipher.decrypt(encrypted_key).decode()
```

#### SSL/TLS Configuration
```nginx
server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
    
    location / {
        proxy_pass http://localhost:8050;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Database Setup

### SQLite Database
```bash
# Initialize database
python -c "from src.config.database import get_database_manager; get_database_manager().initialize_database()"

# Create backup
cp data/trading_demo.db data/trading_demo_backup_$(date +%Y%m%d_%H%M%S).db
```

### PostgreSQL (Production)
```sql
-- Create database
CREATE DATABASE tradingbot;
CREATE USER tradingbot_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE tradingbot TO tradingbot_user;

-- Create tables
\c tradingbot
\i src/config/schema.sql
```

### Database Migration
```python
# Migration script
from src.config.database import DatabaseManager

def migrate_database():
    db_manager = DatabaseManager()
    
    # Backup existing data
    db_manager.backup_database()
    
    # Run migrations
    db_manager.run_migrations()
    
    # Verify migration
    db_manager.verify_schema()
```

## Monitoring Setup

### System Monitoring
```python
# Enable monitoring
from src.monitoring.system_monitor import SystemMonitor

monitor = SystemMonitor()
monitor.start_monitoring()

# Configure alerts
monitor.configure_alerts({
    'cpu_threshold': 80,
    'memory_threshold': 85,
    'disk_threshold': 90,
    'error_rate_threshold': 5
})
```

### Log Management
```python
# Configure logging
import logging
from logging.handlers import RotatingFileHandler

# Create rotating file handler
handler = RotatingFileHandler(
    'logs/tradingbot.log',
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)

# Configure formatter
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)

# Add handler to root logger
logging.getLogger().addHandler(handler)
```

## Backup and Recovery

### Automated Backup
```bash
#!/bin/bash
# backup.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/tradingbot"
DB_PATH="data/trading_demo.db"

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup database
cp $DB_PATH $BACKUP_DIR/trading_demo_$DATE.db

# Backup configuration
cp -r config/ $BACKUP_DIR/config_$DATE/

# Backup logs
tar -czf $BACKUP_DIR/logs_$DATE.tar.gz logs/

# Cleanup old backups (keep 30 days)
find $BACKUP_DIR -name "*.db" -mtime +30 -delete
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete
```

### Recovery Procedure
```bash
#!/bin/bash
# restore.sh

BACKUP_FILE=$1
BACKUP_DIR="/backups/tradingbot"

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file>"
    exit 1
fi

# Stop application
systemctl stop tradingbot

# Restore database
cp $BACKUP_DIR/$BACKUP_FILE data/trading_demo.db

# Restore configuration
tar -xzf $BACKUP_DIR/config_$BACKUP_FILE.tar.gz

# Start application
systemctl start tradingbot
```

## Performance Optimization

### System Tuning
```bash
# Increase file descriptor limits
echo "* soft nofile 65536" >> /etc/security/limits.conf
echo "* hard nofile 65536" >> /etc/security/limits.conf

# Optimize kernel parameters
echo "net.core.somaxconn = 65536" >> /etc/sysctl.conf
echo "net.ipv4.tcp_max_syn_backlog = 65536" >> /etc/sysctl.conf
sysctl -p
```

### Application Optimization
```python
# Configure connection pooling
import sqlite3
from sqlite3 import Connection

def get_optimized_connection():
    conn = sqlite3.connect('data/trading_demo.db')
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA synchronous=NORMAL')
    conn.execute('PRAGMA cache_size=10000')
    conn.execute('PRAGMA temp_store=MEMORY')
    return conn
```

## Troubleshooting

### Common Issues

#### Database Connection Issues
```bash
# Check database file permissions
ls -la data/trading_demo.db

# Repair database
sqlite3 data/trading_demo.db "PRAGMA integrity_check;"
sqlite3 data/trading_demo.db "VACUUM;"
```

#### API Connection Issues
```python
# Test API connectivity
import requests

def test_api_connectivity():
    apis = {
        'Questrade': 'https://api.questrade.com/v1/time',
        'Yahoo Finance': 'https://query1.finance.yahoo.com/v8/finance/chart/AAPL',
        'Alpha Vantage': 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=AAPL&interval=1min&apikey=demo'
    }
    
    for name, url in apis.items():
        try:
            response = requests.get(url, timeout=10)
            print(f"{name}: {response.status_code}")
        except Exception as e:
            print(f"{name}: Error - {e}")
```

#### Performance Issues
```bash
# Monitor system resources
htop
iotop
netstat -tulpn

# Check application logs
tail -f logs/tradingbot.log
journalctl -u tradingbot -f
```

### Emergency Procedures

#### System Recovery
```bash
# Emergency stop
systemctl stop tradingbot
pkill -f "python.*main.py"

# Safe restart
systemctl start tradingbot
systemctl status tradingbot
```

#### Data Recovery
```bash
# Restore from latest backup
./restore.sh trading_demo_$(date +%Y%m%d)_*.db

# Verify data integrity
python -c "from src.config.database import get_database_manager; get_database_manager().verify_schema()"
```

## Security Checklist

### Pre-Deployment Security
- [ ] All API keys stored in environment variables
- [ ] Database credentials secured
- [ ] SSL/TLS certificates installed
- [ ] Firewall rules configured
- [ ] Access logs enabled
- [ ] Backup encryption enabled

### Post-Deployment Security
- [ ] Security monitoring enabled
- [ ] Regular security updates scheduled
- [ ] Access control implemented
- [ ] Audit logging verified
- [ ] Penetration testing completed
- [ ] Incident response plan tested

## Conclusion

This deployment guide provides comprehensive instructions for deploying the AI Trading System across different environments. Follow the security checklist and best practices to ensure a secure and reliable deployment.

For additional support, refer to the troubleshooting section or contact the development team.
