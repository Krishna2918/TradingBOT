"""
Configuration Loader for Feature Consistency System

This module provides utilities to load, validate, and apply configuration
with environment-specific overrides and runtime parameter support.
"""

import json
import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime

from .config_validation import FeatureConsistencyConfigValidator, load_and_validate_config

logger = logging.getLogger(__name__)

@dataclass
class FeatureConsistencyConfig:
    """
    Feature Consistency Configuration dataclass with all configurable parameters.
    
    This matches the requirements from the spec for configurable parameters.
    """
    # Core processing parameters
    warmup_trim_days: int = 200
    nan_drop_threshold_per_symbol: float = 0.05
    global_feature_keep_ratio: float = 0.95
    min_symbol_feature_coverage: float = 0.90
    use_missingness_mask: bool = True
    
    # Imputation settings
    imputation_strategy: str = "zero"  # "zero", "mean", "median"
    imputation_value: float = 0.0
    
    # File paths
    manifest_path: str = "models/feature_manifest.json"
    config_backup_path: str = "models/feature_consistency_config.json"
    
    # Logging settings
    log_level: str = "INFO"
    detailed_logging: bool = True
    
    # Advanced settings
    enforce_feature_manifest: bool = True
    batch_processing_enabled: bool = False
    memory_optimization: bool = True
    parallel_processing: bool = False
    max_symbols_per_batch: int = 50
    
    # Monitoring settings
    enable_metrics_collection: bool = True
    enable_drift_detection: bool = True
    alert_on_high_exclusion_rate: bool = True
    exclusion_rate_threshold: float = 0.20
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        if not 0 <= self.warmup_trim_days <= 500:
            raise ValueError(f"warmup_trim_days must be between 0 and 500, got {self.warmup_trim_days}")
        
        if not 0.01 <= self.nan_drop_threshold_per_symbol <= 0.50:
            raise ValueError(f"nan_drop_threshold_per_symbol must be between 0.01 and 0.50, got {self.nan_drop_threshold_per_symbol}")
        
        if not 0.50 <= self.global_feature_keep_ratio <= 1.00:
            raise ValueError(f"global_feature_keep_ratio must be between 0.50 and 1.00, got {self.global_feature_keep_ratio}")
        
        if not 0.50 <= self.min_symbol_feature_coverage <= 1.00:
            raise ValueError(f"min_symbol_feature_coverage must be between 0.50 and 1.00, got {self.min_symbol_feature_coverage}")
        
        if self.imputation_strategy not in ["zero", "mean", "median"]:
            raise ValueError(f"imputation_strategy must be one of ['zero', 'mean', 'median'], got {self.imputation_strategy}")
        
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            raise ValueError(f"log_level must be one of ['DEBUG', 'INFO', 'WARNING', 'ERROR'], got {self.log_level}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'FeatureConsistencyConfig':
        """Create configuration from dictionary."""
        # Filter out unknown keys
        valid_keys = {field.name for field in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        
        return cls(**filtered_dict)
    
    def save_to_file(self, file_path: Union[str, Path]):
        """Save configuration to JSON file."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_data = {
            '_description': 'Feature Consistency System Configuration',
            '_version': '1.0',
            '_created': datetime.now().isoformat(),
            **self.to_dict()
        }
        
        with open(file_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"Configuration saved to {file_path}")


class ConfigurationManager:
    """
    Manages configuration loading, validation, and environment-specific overrides.
    """
    
    def __init__(self, base_config_dir: Union[str, Path] = "config"):
        """
        Initialize configuration manager.
        
        Args:
            base_config_dir: Base directory for configuration files
        """
        self.base_config_dir = Path(base_config_dir)
        self.validator = FeatureConsistencyConfigValidator()
        self.current_config: Optional[FeatureConsistencyConfig] = None
        self.current_environment: Optional[str] = None
    
    def load_config(
        self, 
        config_path: Optional[Union[str, Path]] = None,
        environment: Optional[str] = None,
        validate: bool = True
    ) -> FeatureConsistencyConfig:
        """
        Load configuration from file or environment.
        
        Args:
            config_path: Specific configuration file path
            environment: Environment name (development, testing, production)
            validate: Whether to validate configuration
            
        Returns:
            FeatureConsistencyConfig instance
        """
        if config_path:
            config_path = Path(config_path)
        elif environment:
            config_path = self.base_config_dir / "templates" / f"feature_consistency_{environment}.json"
        else:
            # Try to detect environment from environment variables
            environment = self._detect_environment()
            config_path = self.base_config_dir / "templates" / f"feature_consistency_{environment}.json"
        
        if not config_path.exists():
            logger.warning(f"Configuration file not found: {config_path}")
            logger.info("Using default configuration")
            config = FeatureConsistencyConfig()
        else:
            logger.info(f"Loading configuration from: {config_path}")
            
            if validate:
                config_dict, validation_result = load_and_validate_config(config_path, environment)
                
                if not validation_result.is_valid:
                    logger.error("Configuration validation failed:")
                    self.validator.print_validation_report(validation_result, str(config_path))
                    raise ValueError("Invalid configuration file")
                
                if validation_result.warnings:
                    logger.warning("Configuration validation warnings:")
                    for warning in validation_result.warnings:
                        logger.warning(f"{warning.field_name}: {warning.error_message}")
                
                config = FeatureConsistencyConfig.from_dict(config_dict)
            else:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                
                # Extract values from nested structure
                config_dict = self._flatten_config(config_data)
                config = FeatureConsistencyConfig.from_dict(config_dict)
        
        # Apply environment-specific overrides
        if environment:
            config = self._apply_environment_overrides(config, environment)
        
        # Apply runtime overrides from environment variables
        config = self._apply_runtime_overrides(config)
        
        self.current_config = config
        self.current_environment = environment
        
        logger.info(f"Configuration loaded successfully for environment: {environment or 'default'}")
        return config
    
    def _detect_environment(self) -> str:
        """Detect environment from environment variables or defaults."""
        env = os.getenv('FEATURE_CONSISTENCY_ENV', os.getenv('ENVIRONMENT', 'development')).lower()
        
        if env in ['prod', 'production']:
            return 'production'
        elif env in ['test', 'testing']:
            return 'testing'
        else:
            return 'development'
    
    def _flatten_config(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten nested configuration structure."""
        flat_config = {}
        
        for key, value in config_data.items():
            if key.startswith('_'):  # Skip metadata
                continue
            
            if isinstance(value, dict) and 'value' in value:
                flat_config[key] = value['value']
            elif not isinstance(value, dict):
                flat_config[key] = value
        
        return flat_config
    
    def _apply_environment_overrides(self, config: FeatureConsistencyConfig, environment: str) -> FeatureConsistencyConfig:
        """Apply environment-specific configuration overrides."""
        config_dict = config.to_dict()
        
        if environment == 'production':
            # Production-specific overrides
            config_dict.update({
                'log_level': 'INFO',
                'detailed_logging': True,
                'enable_metrics_collection': True,
                'enable_drift_detection': True,
                'alert_on_high_exclusion_rate': True,
                'memory_optimization': True
            })
        elif environment == 'development':
            # Development-specific overrides
            config_dict.update({
                'log_level': 'DEBUG',
                'detailed_logging': True,
                'warmup_trim_days': min(config_dict['warmup_trim_days'], 100),  # Faster iteration
                'enable_metrics_collection': False
            })
        elif environment == 'testing':
            # Testing-specific overrides
            config_dict.update({
                'log_level': 'DEBUG',
                'detailed_logging': True,
                'warmup_trim_days': min(config_dict['warmup_trim_days'], 50),  # Fast tests
                'enable_metrics_collection': False,
                'enable_drift_detection': False
            })
        
        return FeatureConsistencyConfig.from_dict(config_dict)
    
    def _apply_runtime_overrides(self, config: FeatureConsistencyConfig) -> FeatureConsistencyConfig:
        """Apply runtime overrides from environment variables."""
        config_dict = config.to_dict()
        
        # Map of environment variable names to config field names
        env_var_mapping = {
            'FC_WARMUP_TRIM_DAYS': 'warmup_trim_days',
            'FC_NAN_THRESHOLD': 'nan_drop_threshold_per_symbol',
            'FC_GLOBAL_KEEP_RATIO': 'global_feature_keep_ratio',
            'FC_MIN_COVERAGE': 'min_symbol_feature_coverage',
            'FC_USE_MISSINGNESS_MASK': 'use_missingness_mask',
            'FC_IMPUTATION_STRATEGY': 'imputation_strategy',
            'FC_LOG_LEVEL': 'log_level',
            'FC_DETAILED_LOGGING': 'detailed_logging',
            'FC_MANIFEST_PATH': 'manifest_path'
        }
        
        for env_var, config_field in env_var_mapping.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Convert to appropriate type
                if config_field in ['warmup_trim_days', 'max_symbols_per_batch']:
                    config_dict[config_field] = int(env_value)
                elif config_field in ['nan_drop_threshold_per_symbol', 'global_feature_keep_ratio', 
                                    'min_symbol_feature_coverage', 'imputation_value', 'exclusion_rate_threshold']:
                    config_dict[config_field] = float(env_value)
                elif config_field in ['use_missingness_mask', 'detailed_logging', 'enforce_feature_manifest',
                                    'batch_processing_enabled', 'memory_optimization', 'parallel_processing',
                                    'enable_metrics_collection', 'enable_drift_detection', 'alert_on_high_exclusion_rate']:
                    config_dict[config_field] = env_value.lower() in ['true', '1', 'yes', 'on']
                else:
                    config_dict[config_field] = env_value
                
                logger.info(f"Applied runtime override: {config_field} = {config_dict[config_field]} (from {env_var})")
        
        return FeatureConsistencyConfig.from_dict(config_dict)
    
    def validate_current_config(self) -> bool:
        """Validate the currently loaded configuration."""
        if not self.current_config:
            logger.error("No configuration loaded")
            return False
        
        try:
            self.current_config._validate_config()
            logger.info("Configuration validation passed")
            return True
        except ValueError as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def save_current_config(self, file_path: Union[str, Path]):
        """Save the current configuration to file."""
        if not self.current_config:
            raise ValueError("No configuration loaded")
        
        self.current_config.save_to_file(file_path)
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of the current configuration."""
        if not self.current_config:
            return {"status": "No configuration loaded"}
        
        return {
            "environment": self.current_environment,
            "warmup_trim_days": self.current_config.warmup_trim_days,
            "nan_drop_threshold_per_symbol": self.current_config.nan_drop_threshold_per_symbol,
            "global_feature_keep_ratio": self.current_config.global_feature_keep_ratio,
            "min_symbol_feature_coverage": self.current_config.min_symbol_feature_coverage,
            "use_missingness_mask": self.current_config.use_missingness_mask,
            "imputation_strategy": self.current_config.imputation_strategy,
            "log_level": self.current_config.log_level,
            "manifest_path": self.current_config.manifest_path
        }


# Convenience functions for easy usage
def load_config_for_environment(environment: str = None) -> FeatureConsistencyConfig:
    """
    Load configuration for a specific environment.
    
    Args:
        environment: Environment name (development, testing, production)
        
    Returns:
        FeatureConsistencyConfig instance
    """
    manager = ConfigurationManager()
    return manager.load_config(environment=environment)


def load_config_from_file(config_path: Union[str, Path]) -> FeatureConsistencyConfig:
    """
    Load configuration from a specific file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        FeatureConsistencyConfig instance
    """
    manager = ConfigurationManager()
    return manager.load_config(config_path=config_path)


def create_default_config(environment: str = 'development') -> FeatureConsistencyConfig:
    """
    Create a default configuration for the specified environment.
    
    Args:
        environment: Environment name
        
    Returns:
        FeatureConsistencyConfig instance with environment-appropriate defaults
    """
    config = FeatureConsistencyConfig()
    manager = ConfigurationManager()
    return manager._apply_environment_overrides(config, environment)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        env = sys.argv[1]
        config = load_config_for_environment(env)
        print(f"Loaded configuration for {env} environment:")
        print(json.dumps(config.to_dict(), indent=2))
    else:
        # Test configuration loading
        for env in ['development', 'testing', 'production']:
            try:
                config = load_config_for_environment(env)
                print(f"✓ {env} configuration loaded successfully")
            except Exception as e:
                print(f"✗ {env} configuration failed: {e}")