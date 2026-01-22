"""
Configuration management for the Portfolio Optimization Engine.

This module provides centralized configuration management to ensure
resource efficiency and prevent API over-usage across all components.
"""

import os
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class APILimits:
    """API rate limiting configuration"""
    alpha_vantage_calls_per_minute: int = 74  # Respect existing rate limit
    max_concurrent_requests: int = 5
    request_timeout: int = 30
    retry_attempts: int = 3
    backoff_factor: float = 1.5


@dataclass
class OptimizationSettings:
    """Core optimization configuration"""
    default_method: str = 'mean_variance'
    max_portfolio_size: int = 200  # Updated for 200 companies
    optimization_timeout: int = 15  # Reduced for smaller portfolio
    rebalance_threshold: float = 0.05  # 5% deviation
    correlation_update_interval: int = 300  # 5 minutes
    factor_update_interval: int = 300  # 5 minutes
    risk_aversion: float = 3.0
    transaction_cost_threshold: float = 0.005  # 0.5%
    historical_years: int = 25  # Maximum historical data years


@dataclass
class RiskConstraints:
    """Risk management constraints"""
    max_position_size: float = 0.05  # 5% per stock
    max_sector_concentration: float = 0.25  # 25% per sector
    max_leverage: float = 1.0  # No leverage by default
    max_correlation: float = 0.8
    min_diversification_score: float = 0.6
    max_drawdown_threshold: float = 0.15  # 15%
    var_confidence_level: float = 0.95
    
    # Existing 4-bucket allocation
    penny_stocks_allocation: float = 0.02  # 2%
    futures_options_allocation: float = 0.05  # 5%
    core_allocation: float = 0.90  # 90%
    sip_allocation: float = 0.01  # 1%


@dataclass
class PerformanceSettings:
    """Performance and monitoring configuration"""
    cache_size: int = 2000  # Increased for 200 companies
    memory_limit_mb: int = 8192  # 8GB working memory from 32GB total
    parallel_workers: int = 8  # Utilize more cores for 200 companies
    enable_caching: bool = True
    log_level: str = 'INFO'
    metrics_retention_days: int = 365
    compression_enabled: bool = True  # Enable data compression
    chunk_size: int = 10000  # For processing large datasets


@dataclass
class DatabaseConfig:
    """Database connection configuration"""
    host: str = 'localhost'
    port: int = 5432
    database: str = 'trading_bot'
    username: str = 'trader'
    password: str = ''
    connection_pool_size: int = 10
    connection_timeout: int = 30


class ConfigManager:
    """
    Centralized configuration manager for the Portfolio Optimization Engine.
    
    Ensures consistent settings across all components and prevents resource
    conflicts with existing trading system components.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or self._get_default_config_path()
        self._config = self._load_config()
        
    def _get_default_config_path(self) -> str:
        """Get default configuration file path"""
        return os.path.join(
            os.path.dirname(__file__), 
            '..', '..', '..', 
            'config', 
            'portfolio_optimization.json'
        )
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load config file {self.config_file}: {e}")
                print("Using default configuration")
        
        return self._create_default_config()
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration"""
        return {
            'api_limits': asdict(APILimits()),
            'optimization': asdict(OptimizationSettings()),
            'risk_constraints': asdict(RiskConstraints()),
            'performance': asdict(PerformanceSettings()),
            'database': asdict(DatabaseConfig())
        }
    
    def save_config(self) -> None:
        """Save current configuration to file"""
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(self._config, f, indent=2)
    
    @property
    def api_limits(self) -> APILimits:
        """Get API limits configuration"""
        return APILimits(**self._config.get('api_limits', {}))
    
    @property
    def optimization(self) -> OptimizationSettings:
        """Get optimization settings"""
        return OptimizationSettings(**self._config.get('optimization', {}))
    
    @property
    def risk_constraints(self) -> RiskConstraints:
        """Get risk constraints"""
        return RiskConstraints(**self._config.get('risk_constraints', {}))
    
    @property
    def performance(self) -> PerformanceSettings:
        """Get performance settings"""
        return PerformanceSettings(**self._config.get('performance', {}))
    
    @property
    def database(self) -> DatabaseConfig:
        """Get database configuration"""
        return DatabaseConfig(**self._config.get('database', {}))
    
    def update_setting(self, section: str, key: str, value: Any) -> None:
        """Update a specific configuration setting"""
        if section not in self._config:
            self._config[section] = {}
        self._config[section][key] = value
    
    def get_setting(self, section: str, key: str, default: Any = None) -> Any:
        """Get a specific configuration setting"""
        return self._config.get(section, {}).get(key, default)


# Global configuration instance
config = ConfigManager()


def get_config() -> ConfigManager:
    """Get the global configuration instance"""
    return config


def reload_config(config_file: Optional[str] = None) -> ConfigManager:
    """Reload configuration from file"""
    global config
    config = ConfigManager(config_file)
    return config