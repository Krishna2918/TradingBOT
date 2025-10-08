#!/bin/bash

# Trading Bot Deployment Script
# Deploys the complete trading bot system

set -e

echo "🚀 Starting Trading Bot Deployment..."

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo "❌ This script should not be run as root"
   exit 1
fi

# Check if we're in the right directory
if [ ! -f "src/main.py" ]; then
    echo "❌ Please run this script from the trading-bot root directory"
    exit 1
fi

# Check Python version
echo "🐍 Checking Python version..."
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.11"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.11+ required, found $python_version"
    exit 1
fi

echo "✅ Python version check passed"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "🐍 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Run system tests
echo "🧪 Running system tests..."
python scripts/test_complete_system.py

if [ $? -ne 0 ]; then
    echo "❌ System tests failed. Deployment aborted."
    exit 1
fi

# Create systemd service
echo "🔧 Creating systemd service..."
sudo tee /etc/systemd/system/trading-bot.service > /dev/null <<EOF
[Unit]
Description=Canadian Market Trading Bot
After=network.target redis.service influxdb.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment=PATH=$(pwd)/venv/bin
ExecStart=$(pwd)/venv/bin/python src/main.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd
sudo systemctl daemon-reload

# Enable service
sudo systemctl enable trading-bot

# Set up log rotation
echo "📝 Setting up log rotation..."
sudo tee /etc/logrotate.d/trading-bot > /dev/null <<EOF
$(pwd)/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 $USER $USER
    postrotate
        systemctl reload trading-bot
    endscript
}
EOF

# Set up monitoring cron job
echo "⏰ Setting up monitoring cron job..."
(crontab -l 2>/dev/null; echo "*/5 * * * * cd $(pwd) && python scripts/health_check.py") | crontab -

# Create health check script
echo "🏥 Creating health check script..."
cat > scripts/health_check.py << 'EOF'
#!/usr/bin/env python3
"""
Health Check Script
Monitors trading bot health and restarts if needed
"""

import subprocess
import sys
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_service_status():
    """Check if trading bot service is running"""
    try:
        result = subprocess.run(['systemctl', 'is-active', 'trading-bot'], 
                              capture_output=True, text=True)
        return result.stdout.strip() == 'active'
    except Exception as e:
        logger.error(f"Failed to check service status: {e}")
        return False

def restart_service():
    """Restart trading bot service"""
    try:
        subprocess.run(['sudo', 'systemctl', 'restart', 'trading-bot'], check=True)
        logger.info("Trading bot service restarted")
        return True
    except Exception as e:
        logger.error(f"Failed to restart service: {e}")
        return False

def main():
    """Main health check function"""
    if not check_service_status():
        logger.warning("Trading bot service is not running, attempting restart...")
        if restart_service():
            logger.info("Service restart successful")
        else:
            logger.error("Service restart failed")
            sys.exit(1)
    else:
        logger.info("Trading bot service is healthy")

if __name__ == "__main__":
    main()
EOF

chmod +x scripts/health_check.py

# Set permissions
echo "🔐 Setting permissions..."
chmod +x scripts/*.py
chmod +x scripts/*.sh

# Create backup script
echo "💾 Creating backup script..."
cat > scripts/backup.sh << 'EOF'
#!/bin/bash
# Backup script for trading bot

BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup configuration
cp -r config/ "$BACKUP_DIR/"

# Backup logs
cp -r logs/ "$BACKUP_DIR/"

# Backup data
cp -r data/ "$BACKUP_DIR/"

echo "Backup created: $BACKUP_DIR"
EOF

chmod +x scripts/backup.sh

# Create startup script
echo "🚀 Creating startup script..."
cat > start_trading_bot.sh << 'EOF'
#!/bin/bash
# Start trading bot

echo "🚀 Starting Canadian Market Trading Bot..."

# Activate virtual environment
source venv/bin/activate

# Start the bot
python src/main.py
EOF

chmod +x start_trading_bot.sh

# Final system check
echo "🔍 Running final system check..."
python scripts/test_complete_system.py

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 DEPLOYMENT COMPLETED SUCCESSFULLY!"
    echo ""
    echo "📋 Next Steps:"
    echo "1. Review configuration files in config/"
    echo "2. Update broker API credentials in config/broker_config.yaml"
    echo "3. Start the service: sudo systemctl start trading-bot"
    echo "4. Check status: sudo systemctl status trading-bot"
    echo "5. View logs: journalctl -u trading-bot -f"
    echo ""
    echo "🔧 Management Commands:"
    echo "• Start: sudo systemctl start trading-bot"
    echo "• Stop: sudo systemctl stop trading-bot"
    echo "• Restart: sudo systemctl restart trading-bot"
    echo "• Status: sudo systemctl status trading-bot"
    echo "• Logs: journalctl -u trading-bot -f"
    echo ""
    echo "📊 Monitoring:"
    echo "• Health check runs every 5 minutes"
    echo "• Logs are rotated daily"
    echo "• Backups can be created with: ./scripts/backup.sh"
    echo ""
    echo "🚀 Your Canadian Market Trading Bot is ready!"
else
    echo "❌ Final system check failed. Please review the issues."
    exit 1
fi
