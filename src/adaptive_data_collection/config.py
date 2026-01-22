"""
Configuration management for the Adaptive Data Collection System.
"""

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
import yaml
from pathlib import Path


@dataclass
class CollectionConfig:
    """Configuration for the data collection system."""
    
    # API Configuration
    alpha_vantage_api_key: str
    alpha_vantage_rpm: int = 74
    
    # Collection Settings
    years_to_collect: int = 25
    max_retries: int = 999
    retry_backoff_base: float = 2.0
    
    # Storage Configuration
    output_base_dir: str = "data"
    log_level: str = "INFO"
    log_file: str = "logs/data_jobs.log"
    
    # Safety Settings
    min_disk_space_gb: float = 1.0
    max_memory_usage_gb: float = 8.0
    
    # Symbol Configuration
    us_symbols_file: str = "lists/us_100.txt"
    
    @classmethod
    def from_env(cls) -> "CollectionConfig":
        """Create configuration from environment variables."""
        return cls(
            alpha_vantage_api_key=os.getenv("AV_API_KEY", ""),
            alpha_vantage_rpm=int(os.getenv("AV_RPM", "74")),
            years_to_collect=int(os.getenv("YEARS_TO_COLLECT", "25")),
            max_retries=int(os.getenv("MAX_RETRIES", "999")),
            retry_backoff_base=float(os.getenv("RETRY_BACKOFF_BASE", "2.0")),
            output_base_dir=os.getenv("OUTPUT_BASE_DIR", "data"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_file=os.getenv("LOG_FILE", "logs/data_jobs.log"),
            min_disk_space_gb=float(os.getenv("MIN_DISK_SPACE_GB", "1.0")),
            max_memory_usage_gb=float(os.getenv("MAX_MEMORY_USAGE_GB", "8.0")),
            us_symbols_file=os.getenv("US_SYMBOLS_FILE", "lists/us_100.txt")
        )
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "CollectionConfig":
        """Create configuration from YAML file."""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Merge with environment variables (env takes precedence)
        env_config = cls.from_env()
        
        # Update with YAML values where env vars are not set
        for key, value in config_data.items():
            if hasattr(env_config, key) and getattr(env_config, key) == getattr(cls(), key, None):
                setattr(env_config, key, value)
        
        return env_config
    
    def validate(self) -> None:
        """Validate configuration values."""
        if not self.alpha_vantage_api_key:
            raise ValueError("Alpha Vantage API key is required (set AV_API_KEY environment variable)")
        
        if self.alpha_vantage_rpm <= 0:
            raise ValueError("Alpha Vantage RPM must be positive")
        
        if self.years_to_collect <= 0:
            raise ValueError("Years to collect must be positive")
        
        if self.max_retries < 0:
            raise ValueError("Max retries cannot be negative")
        
        if not Path(self.us_symbols_file).exists():
            raise FileNotFoundError(f"US symbols file not found: {self.us_symbols_file}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "alpha_vantage_api_key": "***REDACTED***",  # Don't expose API key
            "alpha_vantage_rpm": self.alpha_vantage_rpm,
            "years_to_collect": self.years_to_collect,
            "max_retries": self.max_retries,
            "retry_backoff_base": self.retry_backoff_base,
            "output_base_dir": self.output_base_dir,
            "log_level": self.log_level,
            "log_file": self.log_file,
            "min_disk_space_gb": self.min_disk_space_gb,
            "max_memory_usage_gb": self.max_memory_usage_gb,
            "us_symbols_file": self.us_symbols_file
        }