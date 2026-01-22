# Production Deployment Guide
## Stock Data Collection System

This guide provides comprehensive instructions for deploying the Stock Data Collection System in production environments.

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Pre-Deployment Checklist](#pre-deployment-checklist)
3. [Deployment Options](#deployment-options)
4. [Docker Deployment](#docker-deployment)
5. [Bare Metal / VM Deployment](#bare-metal--vm-deployment)
6. [Configuration](#configuration)
7. [Monitoring and Alerting](#monitoring-and-alerting)
8. [Backup and Recovery](#backup-and-recovery)
9. [Scaling](#scaling)
10. [Troubleshooting](#troubleshooting)
11. [Security Best Practices](#security-best-practices)

---

## System Requirements

### Minimum Requirements
- **CPU**: 4 cores (8 recommended)
- **RAM**: 8 GB (16 GB recommended)
- **Storage**: 100 GB SSD (500 GB recommended for long-term data)
- **Network**: Stable internet connection with 10 Mbps+ bandwidth
- **OS**: Linux (Ubuntu 22.04 LTS or CentOS 8+), Docker-compatible environment

### Software Dependencies
- Python 3.11+
- PostgreSQL 15+
- Redis 7+ (for caching and rate limiting)
- Docker 24+ and Docker Compose (for containerized deployment)

---

## Pre-Deployment Checklist

- [ ] Obtain valid Alpha Vantage API keys (minimum 2 recommended)
- [ ] Provision server/VM with adequate resources
- [ ] Set up PostgreSQL database
- [ ] Configure firewall rules (allow ports 8000, 5432, 6379)
- [ ] Set up monitoring infrastructure (Prometheus, Grafana)
- [ ] Configure backup storage location
- [ ] Set up alerting channels (email, Slack, PagerDuty)
- [ ] Review and customize production.yaml configuration
- [ ] Set up log aggregation (optional but recommended)

---

## Deployment Options

### Option 1: Docker Deployment (Recommended)
Best for: Easy deployment, portability, isolated environments

### Option 2: Bare Metal / VM Deployment
Best for: Maximum performance, direct hardware access

### Option 3: Kubernetes Deployment
Best for: High availability, auto-scaling, multi-node deployments

---

## Docker Deployment

### 1. Clone Repository
```bash
git clone https://github.com/yourorg/stock-data-collector.git
cd stock-data-collector
```

### 2. Configure Environment
```bash
# Copy example environment file
cp .env.example .env

# Edit with your API keys and settings
nano .env
```

Required environment variables:
```bash
# API Keys
ALPHA_VANTAGE_API_KEY_1=your_key_here
ALPHA_VANTAGE_API_KEY_2=your_key_here
ALPHA_VANTAGE_API_KEY_3=your_key_here
ALPHA_VANTAGE_API_KEY_4=your_key_here

# Database
DB_PASSWORD=secure_password_here
DATABASE_URL=postgresql://postgres:${DB_PASSWORD}@postgres:5432/stockdata

# Monitoring
GRAFANA_PASSWORD=secure_grafana_password

# Environment
ENVIRONMENT=production
LOG_LEVEL=INFO
```

### 3. Build and Start Services
```bash
# Build images
docker-compose build

# Start all services
docker-compose up -d

# Verify services are running
docker-compose ps
```

### 4. Initialize Database
```bash
# Database is auto-initialized by init-db.sql
# Verify tables were created
docker-compose exec postgres psql -U postgres -d stockdata -c "\dt"
```

### 5. Monitor Logs
```bash
# View collector logs
docker-compose logs -f collector

# View all service logs
docker-compose logs -f
```

### 6. Access Monitoring Dashboards
- **Grafana**: http://localhost:3000 (admin/your_grafana_password)
- **Prometheus**: http://localhost:9090
- **Health Check**: http://localhost:8000/health

---

## Bare Metal / VM Deployment

### 1. Create Service User
```bash
sudo useradd -r -s /bin/false -d /opt/stock-collector stockcollector
```

### 2. Install System Dependencies
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip postgresql-15 redis-server

# CentOS/RHEL
sudo yum install -y python311 python311-pip postgresql15-server redis
```

### 3. Set Up PostgreSQL
```bash
# Initialize database
sudo postgresql-setup --initdb

# Start service
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Create database and user
sudo -u postgres psql << EOF
CREATE DATABASE stockdata;
CREATE USER stockcollector WITH ENCRYPTED PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE stockdata TO stockcollector;
\q
EOF

# Run initialization script
sudo -u postgres psql -d stockdata -f scripts/init-db.sql
```

### 4. Install Application
```bash
# Copy repository to installation directory
sudo mkdir -p /opt/stock-collector
sudo cp -r . /opt/stock-collector/
cd /opt/stock-collector

# Create Python virtual environment
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Configure Environment
```bash
# Create environment file
sudo mkdir -p /etc/stock-collector
sudo nano /etc/stock-collector/env
```

Add configuration (same as Docker .env)

### 6. Install Systemd Service
```bash
# Install service
cd scripts
sudo bash install-service.sh

# Start service
sudo systemctl start stock-collector
sudo systemctl enable stock-collector

# Check status
sudo systemctl status stock-collector
```

### 7. Set Up Log Rotation
```bash
sudo nano /etc/logrotate.d/stock-collector
```

Add:
```
/var/log/stock-collector/*.log {
    daily
    rotate 30
    compress
    delaycompress
    notifempty
    create 0640 stockcollector stockcollector
    sharedscripts
    postrotate
        systemctl reload stock-collector > /dev/null 2>&1 || true
    endscript
}
```

---

## Configuration

### Production Configuration File
Edit `config/production.yaml`:

```yaml
# Key settings to customize:
collection:
  target_stocks_count: 1400  # Number of stocks to collect
  batch_size: 20             # Stocks per batch
  max_workers: 8             # Concurrent workers

monitoring:
  health_check_interval: 60  # Seconds between health checks
  log_level: INFO            # DEBUG, INFO, WARNING, ERROR

resilience:
  enable_graceful_shutdown: true
  shutdown_timeout: 60       # Seconds to wait for graceful shutdown
```

### Database Connection
Configure in environment or YAML:
```yaml
database:
  connection_url: postgresql://user:password@host:5432/dbname
  pool_size: 10
  connection_timeout: 30
```

---

## Monitoring and Alerting

### Health Check Endpoints

| Endpoint | Purpose | Response |
|----------|---------|----------|
| `/health` | Overall health | 200 (healthy) / 503 (unhealthy) |
| `/health/live` | Liveness probe | 200 (alive) / 503 (dead) |
| `/health/ready` | Readiness probe | 200 (ready) / 503 (not ready) |
| `/metrics` | Prometheus metrics | Metrics in text format |
| `/status` | Detailed status | JSON with full system state |

### Grafana Dashboards
1. Access Grafana at http://localhost:3000
2. Login with admin credentials
3. Import dashboard from `monitoring/grafana/dashboards/`
4. Configure alert rules in Prometheus

### Setting Up Alerts

#### Email Alerts
Configure in `monitoring/alertmanager.yml`:
```yaml
receivers:
  - name: 'email'
    email_configs:
      - to: 'alerts@yourcompany.com'
        from: 'monitoring@yourcompany.com'
        smarthost: 'smtp.gmail.com:587'
        auth_username: 'your_email@gmail.com'
        auth_password: 'your_app_password'
```

#### Slack Alerts
```yaml
receivers:
  - name: 'slack'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL'
        channel: '#alerts'
        title: 'Stock Collector Alert'
```

---

## Backup and Recovery

### Automated Backups

#### Set Up Cron Job
```bash
# Edit crontab
sudo crontab -e

# Add daily backup at 2 AM
0 2 * * * /opt/stock-collector/scripts/backup.sh >> /var/log/stock-collector-backup.log 2>&1
```

#### Backup Script
The `scripts/backup.sh` script backs up:
- PostgreSQL database
- Data files (Parquet files)
- Configuration files
- System state

Backups are stored in `/backup/stock-collector/` with 30-day retention.

### Manual Backup
```bash
# Run backup manually
sudo /opt/stock-collector/scripts/backup.sh
```

### Restore from Backup
```bash
# List available backups
ls -lh /backup/stock-collector/

# Restore specific backup
sudo /opt/stock-collector/scripts/restore.sh /backup/stock-collector/backup_20241028_020000.tar.gz
```

---

## Scaling

### Vertical Scaling
Increase resources on single machine:
```yaml
# In production.yaml
worker_pool:
  max_workers: 16  # Increase workers

collection:
  batch_size: 50   # Increase batch size
```

### Horizontal Scaling
Run multiple instances with shared database:
```bash
# Instance 1: Collect symbols A-M
STOCK_FILTER="A-M" docker-compose up collector1

# Instance 2: Collect symbols N-Z
STOCK_FILTER="N-Z" docker-compose up collector2
```

---

## Troubleshooting

### Common Issues

#### 1. High Memory Usage
**Symptoms**: System running out of memory
**Solutions**:
- Reduce `max_workers` in configuration
- Decrease `batch_size`
- Increase system RAM
- Enable swap space

#### 2. API Rate Limiting
**Symptoms**: Many failed requests, "rate limit exceeded" errors
**Solutions**:
- Add more API keys
- Reduce `requests_per_minute` in config
- Implement exponential backoff

#### 3. Database Connection Issues
**Symptoms**: "connection refused", "too many connections"
**Solutions**:
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Check connection limit
sudo -u postgres psql -c "SHOW max_connections;"

# Increase if needed
sudo nano /etc/postgresql/15/main/postgresql.conf
# Set max_connections = 200
sudo systemctl restart postgresql
```

#### 4. Service Won't Start
**Symptoms**: systemctl start fails
**Solutions**:
```bash
# Check logs
sudo journalctl -u stock-collector -n 50 --no-pager

# Check configuration
/opt/stock-collector/venv/bin/python -m continuous_data_collection.main --validate-config

# Check permissions
sudo chown -R stockcollector:stockcollector /opt/stock-collector
```

### Debug Mode
Enable detailed logging:
```bash
# Set LOG_LEVEL=DEBUG in /etc/stock-collector/env
LOG_LEVEL=DEBUG

# Restart service
sudo systemctl restart stock-collector

# Watch logs
sudo journalctl -u stock-collector -f
```

---

## Security Best Practices

### 1. API Key Protection
- Store keys in environment variables or secrets manager
- Never commit keys to version control
- Rotate keys periodically (every 90 days)
- Use separate keys for dev/staging/production

### 2. Database Security
```bash
# Use strong passwords
# Restrict database access to localhost
sudo nano /etc/postgresql/15/main/pg_hba.conf
# Add: host stockdata stockcollector 127.0.0.1/32 md5

# Enable SSL connections
ssl = on
ssl_cert_file = '/path/to/cert.pem'
ssl_key_file = '/path/to/key.pem'
```

### 3. Firewall Configuration
```bash
# Allow only necessary ports
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp      # SSH
sudo ufw allow 8000/tcp    # Health checks (restrict to monitoring IPs)
sudo ufw enable
```

### 4. Service Hardening
The systemd service includes security hardening:
- Runs as non-root user
- Limited system access
- Memory limits
- CPU quotas
- Restricted system calls

### 5. Regular Updates
```bash
# Update system packages
sudo apt update && sudo apt upgrade

# Update Python dependencies
source /opt/stock-collector/venv/bin/activate
pip install --upgrade -r requirements.txt

# Restart service
sudo systemctl restart stock-collector
```

---

## Maintenance Tasks

### Daily
- [ ] Check service status: `systemctl status stock-collector`
- [ ] Review error logs: `journalctl -u stock-collector | grep ERROR`
- [ ] Monitor dashboard for anomalies

### Weekly
- [ ] Review collection metrics in Grafana
- [ ] Check disk space: `df -h`
- [ ] Review backup success
- [ ] Check database size: `du -sh /var/lib/postgresql/`

### Monthly
- [ ] Review and clean old logs
- [ ] Update dependencies
- [ ] Test backup restoration
- [ ] Review API usage and costs
- [ ] Rotate API keys if needed

---

## Support and Resources

- **Documentation**: https://docs.yourcompany.com/stock-collector
- **Issues**: https://github.com/yourorg/stock-data-collector/issues
- **Monitoring**: http://monitoring.yourcompany.com
- **On-Call**: PagerDuty rotation

---

## Quick Reference

### Common Commands
```bash
# Start/Stop/Restart
sudo systemctl start stock-collector
sudo systemctl stop stock-collector
sudo systemctl restart stock-collector

# Logs
sudo journalctl -u stock-collector -f
sudo journalctl -u stock-collector -n 100

# Status
sudo systemctl status stock-collector
curl http://localhost:8000/health

# Backup
sudo /opt/stock-collector/scripts/backup.sh

# Restore
sudo /opt/stock-collector/scripts/restore.sh <backup_file>
```

---

**Last Updated**: 2025-10-28
**Version**: 1.0.0
