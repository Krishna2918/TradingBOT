# Stock Data Collection System - Production Ready

A production-grade, enterprise-ready system for collecting and managing stock market data from multiple sources with comprehensive monitoring, reliability, and scalability features.

---

## 🚀 Features

### Core Capabilities
- **Multi-Source Data Collection**: Alpha Vantage API with yfinance fallback
- **Concurrent Processing**: Configurable worker pools with auto-scaling
- **State Persistence**: PostgreSQL-backed state management with automatic recovery
- **Priority Queue Management**: Intelligent task prioritization and batching
- **Data Quality Control**: Automated validation and quality scoring

### Production-Grade Features
- **Health Checks**: Kubernetes-compatible liveness and readiness probes
- **Metrics & Monitoring**: Prometheus metrics with Grafana dashboards
- **Alerting**: Multi-channel alerts (email, Slack, PagerDuty)
- **Graceful Shutdown**: Clean termination with state preservation
- **Automatic Backups**: Scheduled backups with configurable retention
- **Database Persistence**: PostgreSQL for reliable state and metrics storage
- **Resource Management**: Memory limits, CPU quotas, connection pooling
- **Security Hardening**: Non-root execution, restricted system calls, encrypted secrets

### Deployment Options
- **Docker Deployment**: Containerized with Docker Compose orchestration
- **Bare Metal**: Systemd service with full system integration
- **Kubernetes**: Ready for K8s with health probes and resource limits

---

## 📋 Quick Start

### Docker Deployment (Recommended)

```bash
# 1. Clone repository
git clone https://github.com/yourorg/stock-data-collector.git
cd stock-data-collector

# 2. Configure environment
cp .env.example .env
nano .env  # Add your API keys

# 3. Start services
docker-compose up -d

# 4. Check health
curl http://localhost:8000/health

# 5. View logs
docker-compose logs -f collector

# 6. Access monitoring
# Grafana: http://localhost:3000
# Prometheus: http://localhost:9090
```

### Bare Metal / VM Deployment

```bash
# 1. Run installation script
cd scripts
sudo bash install-service.sh

# 2. Configure API keys
sudo nano /etc/stock-collector/env

# 3. Start service
sudo systemctl start stock-collector

# 4. Check status
sudo systemctl status stock-collector
sudo journalctl -u stock-collector -f
```

---

## 📊 Monitoring & Observability

### Health Check Endpoints

| Endpoint | Purpose | Response |
|----------|---------|----------|
| `/health` | Overall health | 200 (healthy) / 503 (unhealthy) |
| `/health/live` | Liveness probe | 200 (alive) / 503 (dead) |
| `/health/ready` | Readiness probe | 200 (ready) / 503 (not ready) |
| `/metrics` | Prometheus metrics | Metrics in text format |
| `/status` | Detailed status | JSON with full system state |

### Grafana Dashboards

Access Grafana at `http://localhost:3000` with default credentials (configure in `.env`).

**Pre-configured Dashboards**:
- **System Overview**: Collection progress, success rates, throughput
- **Resource Usage**: CPU, memory, database connections
- **API Usage**: Rate limiting, error rates, response times
- **Alerts**: Active alerts and alert history

### Prometheus Metrics

Key metrics exposed:
- `stock_collector_running`: System running status
- `stock_collector_completed_total`: Total completed stocks
- `stock_collector_failed_total`: Total failed stocks
- `stock_collector_pending`: Current pending stocks
- `stock_collector_workers_active`: Active worker count
- `stock_collector_cpu_percent`: CPU usage
- `stock_collector_memory_bytes`: Memory usage

---

## 🔒 Security

### Best Practices Implemented
- **Non-root execution**: Service runs as dedicated user
- **Secret management**: API keys in environment variables
- **Database encryption**: SSL/TLS for database connections
- **Restricted permissions**: Minimal file system access
- **Security hardening**: Systemd security features enabled
- **Input validation**: All user inputs validated
- **Rate limiting**: API rate limits enforced

### Configuration
```bash
# Secure environment file
sudo chmod 600 /etc/stock-collector/env

# Verify service isolation
systemctl show stock-collector | grep -E '(User|Group|NoNewPrivileges|ProtectSystem)'
```

---

## 💾 Backup & Recovery

### Automated Backups

```bash
# Configure cron for daily backups
sudo crontab -e
# Add: 0 2 * * * /opt/stock-collector/scripts/backup.sh

# Manual backup
sudo /opt/stock-collector/scripts/backup.sh

# List backups
ls -lh /backup/stock-collector/
```

### Restore from Backup

```bash
# Restore specific backup
sudo /opt/stock-collector/scripts/restore.sh /backup/stock-collector/backup_20241028_020000.tar.gz

# Service will automatically restart after restore
```

**Backup includes**:
- PostgreSQL database (all tables)
- Parquet data files
- System state
- Configuration files

**Retention**: 30 days (configurable)

---

## 🎯 Performance & Scaling

### Vertical Scaling
```yaml
# Edit config/production.yaml
worker_pool:
  max_workers: 16  # Increase workers
collection:
  batch_size: 50   # Increase batch size
```

### Horizontal Scaling
Run multiple instances with shared database:
```bash
# Instance 1: Symbols A-M
STOCK_FILTER="A-M" docker-compose up collector1

# Instance 2: Symbols N-Z
STOCK_FILTER="N-Z" docker-compose up collector2
```

### Resource Requirements

| Deployment Size | Stocks | Workers | RAM | CPU | Storage |
|----------------|--------|---------|-----|-----|---------|
| Small | 500 | 4 | 4 GB | 2 cores | 100 GB |
| Medium | 1,400 | 8 | 8 GB | 4 cores | 250 GB |
| Large | 5,000+ | 16 | 16 GB | 8 cores | 1 TB |

---

## 🔧 Configuration

### Production Configuration
Edit `config/production.yaml`:

```yaml
# Core settings
collection:
  target_stocks_count: 1400
  batch_size: 20
  max_workers: 8

# Database
database:
  enabled: true
  connection_pool_size: 10

# Monitoring
monitoring:
  health_check_interval: 60
  enable_metrics: true
  metrics_port: 8000

# Alerts
alerts:
  enabled: true
  channels: [email, slack]
```

### Environment Variables
```bash
# /etc/stock-collector/env or .env file
ENVIRONMENT=production
LOG_LEVEL=INFO
ALPHA_VANTAGE_API_KEY_1=your_key_here
DATABASE_URL=postgresql://user:pass@localhost:5432/stockdata
```

---

## 📈 Operational Procedures

### Daily Operations
```bash
# Check service status
sudo systemctl status stock-collector

# View real-time logs
sudo journalctl -u stock-collector -f

# Check health
curl http://localhost:8000/health | jq

# View metrics
curl http://localhost:8000/metrics
```

### Common Tasks

**Restart Service**
```bash
sudo systemctl restart stock-collector
```

**Update Configuration**
```bash
sudo nano /etc/stock-collector/env
sudo systemctl restart stock-collector
```

**View Collection Progress**
```bash
# Via API
curl http://localhost:8000/status | jq '.progress'

# Via database
psql -U postgres -d stockdata -c "SELECT status, COUNT(*) FROM stock_status GROUP BY status;"
```

---

## 🐛 Troubleshooting

### Service Won't Start
```bash
# Check logs
sudo journalctl -u stock-collector -n 100 --no-pager

# Verify configuration
/opt/stock-collector/venv/bin/python -m continuous_data_collection.main --validate-config

# Check permissions
sudo chown -R stockcollector:stockcollector /opt/stock-collector
```

### High Memory Usage
```yaml
# Reduce workers in config/production.yaml
worker_pool:
  max_workers: 4
collection:
  batch_size: 10
```

### API Rate Limiting
```yaml
# Add more API keys or reduce rate
api:
  alpha_vantage_keys:
    - key1
    - key2
    - key3
  rate_limit_per_minute: 3
```

### Database Connection Issues
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Test connection
psql -U postgres -d stockdata -c "SELECT 1;"

# Check connections
psql -U postgres -c "SELECT count(*) FROM pg_stat_activity;"
```

---

## 📚 Documentation

- **[Production Deployment Guide](PRODUCTION_DEPLOYMENT.md)**: Complete deployment instructions
- **[Production Readiness Checklist](PRODUCTION_READINESS_CHECKLIST.md)**: Pre-launch verification
- **[API Documentation](docs/API.md)**: Health check and monitoring endpoints
- **[Configuration Reference](docs/CONFIGURATION.md)**: All configuration options
- **[Troubleshooting Guide](TROUBLESHOOTING_GUIDE.md)**: Common issues and solutions

---

## 🧪 Testing

### Run Integration Tests
```bash
# Set test environment
export TEST_HOST=localhost
export TEST_PORT=8000
export DATABASE_URL=postgresql://postgres:password@localhost:5432/stockdata

# Run tests
pytest tests/integration/ -v
```

### Load Testing
```bash
# Install hey
go install github.com/rakyll/hey@latest

# Load test health endpoint
hey -n 1000 -c 10 http://localhost:8000/health
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 🆘 Support

- **Issues**: https://github.com/yourorg/stock-data-collector/issues
- **Discussions**: https://github.com/yourorg/stock-data-collector/discussions
- **Email**: support@yourcompany.com

---

## 🎉 Acknowledgments

- Alpha Vantage for stock data API
- yfinance library for Yahoo Finance data
- Prometheus and Grafana for monitoring
- PostgreSQL for reliable data persistence

---

**Version**: 1.0.0
**Status**: Production Ready ✅
**Last Updated**: 2025-10-28
