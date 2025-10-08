#!/bin/bash

# Trading Bot Setup Script
# Week 1-2: Foundation Setup

set -e

echo "🚀 Starting Trading Bot Setup..."

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo "❌ This script should not be run as root"
   exit 1
fi

# Update system packages
echo "📦 Updating system packages..."
sudo apt update
sudo apt upgrade -y

# Install system dependencies
echo "🔧 Installing system dependencies..."
sudo apt install -y python3.11 python3.11-venv python3.11-pip
sudo apt install -y redis-server influxdb2 postgresql-client
sudo apt install -y build-essential libssl-dev libffi-dev
sudo apt install -y git curl wget unzip

# Create project directory
echo "📁 Creating project structure..."
mkdir -p ~/trading-bot
cd ~/trading-bot

# Create virtual environment
echo "🐍 Setting up Python virtual environment..."
python3.11 -m venv venv
source venv/bin/activate

# Upgrade pip and install dependencies
echo "📚 Installing Python dependencies..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Create project structure
echo "🏗️ Creating project directories..."
mkdir -p {src,tests,config,scripts,docs,logs,data}
mkdir -p src/{strategies,risk_management,data_pipeline,execution,monitoring}
mkdir -p src/strategies/{momentum_scalping,news_volatility,gamma_oi,arbitrage,ai_ml}
mkdir -p src/risk_management/{capital_allocation,leverage_governance,kill_switches}
mkdir -p src/data_pipeline/{collectors,processors,storage}
mkdir -p src/execution/{order_router,broker_adapters,slippage_mitigation}
mkdir -p src/monitoring/{alerts,dashboards,recovery}
mkdir -p tests/{unit,integration,performance}
mkdir -p config/{trading,risk,broker,monitoring}

# Configure Redis
echo "🔴 Configuring Redis..."
sudo systemctl start redis
sudo systemctl enable redis

# Configure InfluxDB
echo "📊 Configuring InfluxDB..."
sudo systemctl start influxdb
sudo systemctl enable influxdb

# Wait for InfluxDB to start
sleep 5

# Create database and user
echo "CREATE DATABASE trading_metrics" | sudo -u influxdb influx
echo "CREATE USER trading_user WITH PASSWORD 'MOCK_PASSWORD_123'" | sudo -u influxdb influx
echo "GRANT ALL ON trading_metrics TO trading_user" | sudo -u influxdb influx

# Set up logging
echo "📝 Setting up logging..."
mkdir -p logs
touch logs/trading_bot.log
chmod 644 logs/trading_bot.log

# Create systemd service file
echo "🔧 Creating systemd service..."
sudo tee /etc/systemd/system/trading-bot.service > /dev/null <<EOF
[Unit]
Description=Trading Bot Service
After=network.target redis.service influxdb.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$HOME/trading-bot
Environment=PATH=$HOME/trading-bot/venv/bin
ExecStart=$HOME/trading-bot/venv/bin/python src/main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd and enable service
sudo systemctl daemon-reload
sudo systemctl enable trading-bot

# Set up cron jobs
echo "⏰ Setting up cron jobs..."
(crontab -l 2>/dev/null; echo "0 8 * * 1-5 cd $HOME/trading-bot && ./scripts/daily_start.sh") | crontab -
(crontab -l 2>/dev/null; echo "0 16 * * 1-5 cd $HOME/trading-bot && ./scripts/daily_end.sh") | crontab -

# Set permissions
echo "🔐 Setting permissions..."
chmod +x scripts/*.sh
chmod +x scripts/*.py

echo "✅ Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Review and update configuration files in config/"
echo "2. Run tests: python -m pytest tests/"
echo "3. Start the service: sudo systemctl start trading-bot"
echo "4. Check status: sudo systemctl status trading-bot"
echo ""
echo "📚 Documentation available in docs/"
echo "🐛 Logs available in logs/"

