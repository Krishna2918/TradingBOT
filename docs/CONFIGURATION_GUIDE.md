# AI Trading System - Configuration Guide

## Overview

This guide provides comprehensive configuration instructions for the AI Trading System, covering all system parameters, mode-specific settings, and optimization options.

## Configuration Files

### Main Configuration Files
- `src/config/mode_config.json` - Mode-specific configurations
- `.env` - Environment variables and sensitive data
- `config/trading_config.yaml` - Trading parameters
- `config/ai_config.yaml` - AI model configurations
- `config/risk_config.yaml` - Risk management settings

## Environment Variables

### Required Environment Variables
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

### Optional Environment Variables
```env
# Performance Tuning
MAX_WORKERS=4
CACHE_TTL=300
BATCH_SIZE=100

# Security
ENCRYPTION_KEY=your_encryption_key
JWT_SECRET=your_jwt_secret

# External Services
SENTRY_DSN=your_sentry_dsn
PROMETHEUS_ENDPOINT=http://localhost:9090
```

## Mode Configuration

### Demo Mode Configuration
```json
{
  "mode": "DEMO",
  "database": {
    "path": "data/trading_demo.db",
    "backup_interval": 3600,
    "max_connections": 10
  },
  "trading": {
    "max_positions": 10,
    "position_size_percent": 10.0,
    "risk_limit_percent": 2.0,
    "min_position_size": 100.0,
    "max_position_size": 10000.0
  },
  "ai": {
    "models": [
      "qwen3-coder:480b-cloud",
      "deepseek-v3.1:671b-cloud"
    ],
    "confidence_threshold": 0.7,
    "decision_timeout": 30
  },
  "monitoring": {
    "log_level": "INFO",
    "enable_alerts": true,
    "alert_thresholds": {
      "error_rate": 0.05,
      "response_time": 5000
    }
  }
}
```

### Live Mode Configuration
```json
{
  "mode": "LIVE",
  "database": {
    "path": "data/trading_live.db",
    "backup_interval": 1800,
    "max_connections": 20
  },
  "trading": {
    "max_positions": 5,
    "position_size_percent": 5.0,
    "risk_limit_percent": 1.0,
    "min_position_size": 500.0,
    "max_position_size": 5000.0
  },
  "ai": {
    "models": [
      "gpt-oss:120b"
    ],
    "confidence_threshold": 0.8,
    "decision_timeout": 60
  },
  "monitoring": {
    "log_level": "WARNING",
    "enable_alerts": true,
    "alert_thresholds": {
      "error_rate": 0.01,
      "response_time": 2000
    }
  }
}
```

## Trading Configuration

### Trading Parameters
```yaml
# config/trading_config.yaml
trading:
  # Position Management
  max_positions: 10
  position_size_percent: 10.0
  min_position_size: 100.0
  max_position_size: 10000.0
  
  # Risk Management
  risk_limit_percent: 2.0
  stop_loss_percent: 5.0
  take_profit_percent: 10.0
  
  # Order Management
  order_types:
    - market
    - limit
    - stop
    - stop_limit
  
  time_in_force:
    - day
    - gtc
    - ioc
    - fok
  
  # Execution Settings
  slippage_tolerance: 0.1
  max_slippage: 0.5
  execution_timeout: 30
  
  # Market Hours
  market_hours:
    start: "09:30"
    end: "16:00"
    timezone: "America/New_York"
  
  # Trading Days
  trading_days:
    - monday
    - tuesday
    - wednesday
    - thursday
    - friday
```

### Stock Universe Configuration
```yaml
# config/stock_universe.yaml
stock_universe:
  # TSX Stocks
  tsx:
    enabled: true
    max_stocks: 1000
    min_market_cap: 100000000
    min_volume: 100000
    sectors:
      - technology
      - healthcare
      - financial
      - energy
      - materials
  
  # TSXV Stocks
  tsxv:
    enabled: true
    max_stocks: 1000
    min_market_cap: 10000000
    min_volume: 50000
    sectors:
      - mining
      - technology
      - healthcare
  
  # US Stocks
  us:
    enabled: false
    max_stocks: 500
    min_market_cap: 1000000000
    min_volume: 1000000
    exchanges:
      - NYSE
      - NASDAQ
```

## AI Configuration

### AI Model Settings
```yaml
# config/ai_config.yaml
ai:
  # Model Selection
  models:
    primary:
      id: "qwen3-coder:480b-cloud"
      name: "Qwen3 Coder"
      type: "coding"
      weight: 0.4
    
    secondary:
      id: "deepseek-v3.1:671b-cloud"
      name: "DeepSeek V3.1"
      type: "analysis"
      weight: 0.3
    
    tertiary:
      id: "gpt-oss:120b"
      name: "GPT-OSS"
      type: "general"
      weight: 0.3
  
  # Decision Making
  decision_thresholds:
    buy_confidence: 0.7
    sell_confidence: 0.6
    hold_confidence: 0.5
  
  # Analysis Parameters
  analysis:
    lookback_period: 30
    technical_indicators:
      - sma_20
      - sma_50
      - rsi
      - macd
      - bollinger_bands
    
    fundamental_metrics:
      - pe_ratio
      - pb_ratio
      - debt_to_equity
      - roe
      - revenue_growth
  
  # Learning Parameters
  learning:
    batch_size: 32
    learning_rate: 0.001
    epochs: 100
    validation_split: 0.2
    
  # Performance Monitoring
  performance:
    accuracy_threshold: 0.6
    latency_threshold: 5000
    retrain_threshold: 0.05
```

### Ollama Configuration
```yaml
# config/ollama_config.yaml
ollama:
  # Server Configuration
  server:
    host: "localhost"
    port: 11434
    timeout: 60
  
  # Model Management
  models:
    auto_download: true
    update_interval: 86400
    max_models: 5
  
  # Performance Settings
  performance:
    max_concurrent_requests: 3
    request_timeout: 30
    retry_attempts: 3
    retry_delay: 5
```

## Risk Management Configuration

### Risk Parameters
```yaml
# config/risk_config.yaml
risk:
  # Portfolio Risk
  portfolio:
    max_portfolio_risk: 2.0
    max_sector_exposure: 30.0
    max_single_stock_exposure: 10.0
    max_correlation: 0.7
  
  # Position Risk
  position:
    max_position_size: 10000.0
    min_position_size: 100.0
    max_position_risk: 1.0
    stop_loss_percent: 5.0
    take_profit_percent: 10.0
  
  # VaR Configuration
  var:
    confidence_level: 0.95
    time_horizon: 1
    calculation_method: "historical"
    lookback_period: 252
  
  # Stress Testing
  stress_testing:
    enabled: true
    scenarios:
      - market_crash: -20.0
      - sector_crash: -15.0
      - volatility_spike: 2.0
      - liquidity_crisis: -10.0
  
  # Risk Limits
  limits:
    max_daily_loss: 1000.0
    max_weekly_loss: 5000.0
    max_monthly_loss: 20000.0
    max_drawdown: 10.0
```

## Monitoring Configuration

### System Monitoring
```yaml
# config/monitoring_config.yaml
monitoring:
  # System Health
  system:
    cpu_threshold: 80.0
    memory_threshold: 85.0
    disk_threshold: 90.0
    network_threshold: 1000.0
  
  # Application Monitoring
  application:
    response_time_threshold: 5000
    error_rate_threshold: 0.05
    throughput_threshold: 100
  
  # Trading Monitoring
  trading:
    order_execution_time: 30
    position_update_delay: 5
    market_data_latency: 1000
  
  # Alerting
  alerts:
    email:
      enabled: true
      recipients:
        - admin@example.com
        - trader@example.com
    
    slack:
      enabled: true
      webhook_url: "https://hooks.slack.com/services/..."
      channel: "#trading-alerts"
    
    sms:
      enabled: false
      provider: "twilio"
      recipients:
        - "+1234567890"
  
  # Logging
  logging:
    level: "INFO"
    format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    max_file_size: 10485760
    backup_count: 5
    rotation: "daily"
```

## Performance Configuration

### Performance Tuning
```yaml
# config/performance_config.yaml
performance:
  # Database Optimization
  database:
    connection_pool_size: 20
    max_overflow: 30
    pool_timeout: 30
    pool_recycle: 3600
  
  # Caching
  cache:
    redis:
      host: "localhost"
      port: 6379
      db: 0
      max_connections: 10
    
    ttl:
      market_data: 60
      ai_analysis: 300
      portfolio_data: 30
  
  # Async Processing
  async:
    max_workers: 4
    queue_size: 1000
    timeout: 30
  
  # API Rate Limiting
  rate_limiting:
    questrade:
      requests_per_minute: 60
      burst_limit: 10
    
    yahoo_finance:
      requests_per_minute: 100
      burst_limit: 20
    
    alpha_vantage:
      requests_per_minute: 5
      burst_limit: 2
```

## Security Configuration

### Security Settings
```yaml
# config/security_config.yaml
security:
  # API Security
  api:
    rate_limiting:
      enabled: true
      requests_per_minute: 100
      burst_limit: 20
    
    authentication:
      jwt_secret: "your_jwt_secret"
      token_expiry: 3600
      refresh_token_expiry: 86400
  
  # Data Protection
  data_protection:
    encryption:
      enabled: true
      algorithm: "AES-256-GCM"
      key_rotation_interval: 86400
    
    pii_detection:
      enabled: true
      patterns:
        - email
        - phone
        - ssn
        - credit_card
  
  # Audit Logging
  audit:
    enabled: true
    log_level: "INFO"
    retention_days: 365
    sensitive_fields:
      - password
      - api_key
      - token
```

## Adaptive Configuration

### Adaptive Parameters
```yaml
# config/adaptive_config.yaml
adaptive:
  # Parameter Optimization
  optimization:
    enabled: true
    method: "bayesian"
    objective: "sharpe_ratio"
    constraints:
      - max_drawdown < 0.1
      - win_rate > 0.5
    
    parameters:
      position_size_percent:
        min: 1.0
        max: 20.0
        step: 0.5
      
      risk_limit_percent:
        min: 0.5
        max: 5.0
        step: 0.1
      
      confidence_threshold:
        min: 0.5
        max: 0.9
        step: 0.05
  
  # Learning Configuration
  learning:
    enabled: true
    algorithm: "reinforcement_learning"
    reward_function: "sharpe_ratio"
    learning_rate: 0.001
    exploration_rate: 0.1
    
    training:
      episodes: 1000
      batch_size: 32
      update_frequency: 100
  
  # Performance Tracking
  performance:
    metrics:
      - sharpe_ratio
      - max_drawdown
      - win_rate
      - profit_factor
    
    evaluation_period: 30
    reoptimization_threshold: 0.05
```

## Configuration Validation

### Configuration Validation Script
```python
# config/validate_config.py
import yaml
import json
from jsonschema import validate, ValidationError

def validate_configuration():
    """Validate all configuration files"""
    
    # Load configuration files
    with open('config/trading_config.yaml', 'r') as f:
        trading_config = yaml.safe_load(f)
    
    with open('config/ai_config.yaml', 'r') as f:
        ai_config = yaml.safe_load(f)
    
    with open('config/risk_config.yaml', 'r') as f:
        risk_config = yaml.safe_load(f)
    
    # Define schemas
    trading_schema = {
        "type": "object",
        "properties": {
            "trading": {
                "type": "object",
                "properties": {
                    "max_positions": {"type": "integer", "minimum": 1, "maximum": 100},
                    "position_size_percent": {"type": "number", "minimum": 0.1, "maximum": 100.0},
                    "risk_limit_percent": {"type": "number", "minimum": 0.1, "maximum": 10.0}
                },
                "required": ["max_positions", "position_size_percent", "risk_limit_percent"]
            }
        },
        "required": ["trading"]
    }
    
    # Validate configurations
    try:
        validate(trading_config, trading_schema)
        print("✓ Trading configuration is valid")
    except ValidationError as e:
        print(f"✗ Trading configuration error: {e.message}")
    
    # Additional validation logic...
    validate_risk_parameters(risk_config)
    validate_ai_parameters(ai_config)

def validate_risk_parameters(config):
    """Validate risk management parameters"""
    risk_config = config.get('risk', {})
    
    # Check portfolio risk limits
    portfolio_risk = risk_config.get('portfolio', {}).get('max_portfolio_risk', 0)
    if portfolio_risk > 5.0:
        print("⚠️  Warning: Portfolio risk limit is very high")
    
    # Check position limits
    max_position = risk_config.get('position', {}).get('max_position_size', 0)
    if max_position > 50000:
        print("⚠️  Warning: Maximum position size is very large")

if __name__ == "__main__":
    validate_configuration()
```

## Configuration Management

### Dynamic Configuration Updates
```python
# config/config_manager.py
import json
import yaml
from typing import Dict, Any
from src.config.mode_manager import get_current_mode

class ConfigurationManager:
    def __init__(self):
        self.config_cache = {}
        self.load_all_configs()
    
    def load_all_configs(self):
        """Load all configuration files"""
        config_files = [
            'config/trading_config.yaml',
            'config/ai_config.yaml',
            'config/risk_config.yaml',
            'config/monitoring_config.yaml'
        ]
        
        for config_file in config_files:
            with open(config_file, 'r') as f:
                config_name = config_file.split('/')[-1].replace('.yaml', '')
                self.config_cache[config_name] = yaml.safe_load(f)
    
    def get_config(self, config_name: str, key: str = None) -> Any:
        """Get configuration value"""
        config = self.config_cache.get(config_name, {})
        
        if key is None:
            return config
        
        keys = key.split('.')
        value = config
        for k in keys:
            value = value.get(k)
            if value is None:
                break
        
        return value
    
    def update_config(self, config_name: str, key: str, value: Any):
        """Update configuration value"""
        config = self.config_cache.get(config_name, {})
        
        keys = key.split('.')
        current = config
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
        
        # Save to file
        config_file = f'config/{config_name}.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def get_mode_specific_config(self, key: str) -> Any:
        """Get mode-specific configuration"""
        mode = get_current_mode()
        mode_config_file = f'src/config/mode_config.json'
        
        with open(mode_config_file, 'r') as f:
            mode_configs = json.load(f)
        
        return mode_configs.get(mode, {}).get(key)
```

## Best Practices

### Configuration Best Practices
1. **Environment Separation**: Use different configurations for development, staging, and production
2. **Sensitive Data**: Store sensitive information in environment variables, not configuration files
3. **Validation**: Always validate configuration parameters before use
4. **Documentation**: Document all configuration parameters and their effects
5. **Version Control**: Keep configuration files in version control, but exclude sensitive data
6. **Backup**: Regularly backup configuration files
7. **Testing**: Test configuration changes in a safe environment first

### Security Best Practices
1. **API Keys**: Rotate API keys regularly
2. **Access Control**: Limit access to configuration files
3. **Encryption**: Encrypt sensitive configuration data
4. **Audit**: Log all configuration changes
5. **Validation**: Validate all configuration inputs

## Troubleshooting

### Common Configuration Issues

#### Invalid Configuration Values
```bash
# Check configuration syntax
python -c "import yaml; yaml.safe_load(open('config/trading_config.yaml'))"

# Validate configuration
python config/validate_config.py
```

#### Missing Environment Variables
```bash
# Check environment variables
env | grep -E "(QUESTRADE|YAHOO|ALPHA)"

# Load environment file
source .env
```

#### Configuration Loading Errors
```python
# Debug configuration loading
from src.config.mode_manager import get_mode_manager

mode_manager = get_mode_manager()
print(f"Current mode: {mode_manager.get_current_mode()}")
print(f"Mode config: {mode_manager.get_mode_config()}")
```

## Conclusion

This configuration guide provides comprehensive instructions for setting up and managing the AI Trading System. Follow the best practices and validation procedures to ensure optimal system performance and security.

For additional configuration options or troubleshooting, refer to the system documentation or contact the development team.
