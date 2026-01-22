"""
Simple Configuration Loader for Feature Consistency System

This module provides a straightforward way to load and validate configuration
files for the feature consistency system.
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class FeatureConsistencyConfig:
    """
    Feature Consistency Configuration dataclass with all configurable parameters.
    
    This matches the requirements from the spec for configurable parameters.
    """
    # Core processing parameters (Requirements 5.1-5.4)
    warmup_trim_days: int = 200
    nan_drop_threshold_per_symbol: float = 0.05
    global_feature_keep_ratio: float = 0.95
    min_symbol_feature_coverage: float = 0.90
    use_missingness_mask: bool = True  # Requirement 5.5
    
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
        return {
            'warmup_trim_days': self.warmup_trim_days,
            'nan_drop_threshold_per_symbol': self.nan_drop_threshold_per_symbol,
            'global_feature_keep_ratio': self.global_feature_keep_ratio,
            'min_symbol_feature_coverage': self.min_symbol_feature_coverage,
            'use_missingness_mask': self.use_missingness_mask,
            'imputation_strategy': self.imputation_strategy,
            'imputation_value': self.imputation_value,
            'manifest_path': self.manifest_path,
            'config_backup_path': self.config_backup_path,
            'log_level': self.log_level,
            'detailed_logging': self.detailed_logging,
            'enforce_feature_manifest': self.enforce_feature_manifest,
            'batch_processing_enabled': self.batch_processing_enabled,
            'memory_optimization': self.memory_optimization,
            'parallel_processing': self.parallel_processing,
            'max_symbols_per_batch': self.max_symbols_per_batch,
            'enable_metrics_collection': self.enable_metrics_collection,
            'enable_drift_detection': self.enable_drift_detection,
            'alert_on_high_exclusion_rate': self.alert_on_high_exclusion_rate,
            'exclusion_rate_threshold': self.exclusion_rate_threshold
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'FeatureConsistencyConfig':
        """Create configuration from dictionary."""
        # Filter out unknown keys and create instance
        valid_keys = {
            'warmup_trim_days', 'nan_drop_threshold_per_symbol', 'global_feature_keep_ratio',
            'min_symbol_feature_coverage', 'use_missingness_mask', 'imputation_strategy',
            'imputation_value', 'manifest_path', 'config_backup_path', 'log_level',
            'detailed_logging', 'enforce_feature_manifest', 'batch_processing_enabled',
            'memory_optimization', 'parallel_processing', 'max_symbols_per_batch',
            'enable_metrics_collection', 'enable_drift_detection', 'alert_on_high_exclusion_rate',
            'exclusion_rate_threshold'
        }
        
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        
        return cls(**filtered_dict)


def load_config_from_file(config_path: Union[str, Path]) -> FeatureConsistencyConfig:
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        FeatureConsistencyConfig instance
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        logger.warning(f"Configuration file not found: {config_path}")
        logger.info("Using default configuration")
        return FeatureConsistencyConfig()
    
    logger.info(f"Loading configuration from: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Extract values from nested structure if needed
        config_dict = _flatten_config(config_data)
        
        # Apply environment-specific overrides
        environment = config_data.get('_environment')
        if environment:
            config_dict = _apply_environment_overrides(config_dict, environment)
        
        # Apply runtime overrides from environment variables
        config_dict = _apply_runtime_overrides(config_dict)
        
        config = FeatureConsistencyConfig.from_dict(config_dict)
        
        logger.info(f"Configuration loaded successfully for environment: {environment or 'default'}")
        return config
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in configuration file: {e}")
        raise ValueError(f"Invalid JSON in configuration file: {e}")
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise


def load_config_for_environment(environment: str = None) -> FeatureConsistencyConfig:
    """
    Load configuration for a specific environment.
    
    Args:
        environment: Environment name (development, testing, production)
        
    Returns:
        FeatureConsistencyConfig instance
    """
    if not environment:
        environment = _detect_environment()
    
    config_path = Path("config/templates") / f"feature_consistency_{environment}.json"
    
    return load_config_from_file(config_path)


def _detect_environment() -> str:
    """Detect environment from environment variables or defaults."""
    env = os.getenv('FEATURE_CONSISTENCY_ENV', os.getenv('ENVIRONMENT', 'development')).lower()
    
    if env in ['prod', 'production']:
        return 'production'
    elif env in ['test', 'testing']:
        return 'testing'
    else:
        return 'development'


def _flatten_config(config_data: Dict[str, Any]) -> Dict[str, Any]:
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


def _apply_environment_overrides(config_dict: Dict[str, Any], environment: str) -> Dict[str, Any]:
    """Apply environment-specific configuration overrides."""
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
            'warmup_trim_days': min(config_dict.get('warmup_trim_days', 200), 100),  # Faster iteration
            'enable_metrics_collection': False
        })
    elif environment == 'testing':
        # Testing-specific overrides
        config_dict.update({
            'log_level': 'DEBUG',
            'detailed_logging': True,
            'warmup_trim_days': min(config_dict.get('warmup_trim_days', 200), 50),  # Fast tests
            'enable_metrics_collection': False,
            'enable_drift_detection': False
        })
    
    return config_dict


def _apply_runtime_overrides(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Apply runtime overrides from environment variables."""
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
    
    return config_dict


def create_default_config(environment: str = 'development') -> FeatureConsistencyConfig:
    """
    Create a default configuration for the specified environment.
    
    Args:
        environment: Environment name
        
    Returns:
        FeatureConsistencyConfig instance with environment-appropriate defaults
    """
    config_dict = {}
    config_dict = _apply_environment_overrides(config_dict, environment)
    
    # Fill in any missing values with defaults
    default_config = FeatureConsistencyConfig()
    default_dict = default_config.to_dict()
    
    for key, value in default_dict.items():
        if key not in config_dict:
            config_dict[key] = value
    
    return FeatureConsistencyConfig.from_dict(config_dict)


# Convenience function for backward compatibility
def load_feature_consistency_config(config_path: str = None, environment: str = None) -> FeatureConsistencyConfig:
    """
    Load feature consistency configuration.
    
    Args:
        config_path: Optional path to specific configuration file
        environment: Optional environment name
        
    Returns:
        FeatureConsistencyConfig instance
    """
    if config_path:
        return load_config_from_file(config_path)
    else:
        return load_config_for_environment(environment)


def convert_to_feature_consistency_config(simple_config: FeatureConsistencyConfig):
    """
    Convert simple config to the existing FeatureConsistencyConfig format.
    
    This function bridges the gap between the new configuration system and
    the existing feature consistency implementation.
    """
    try:
        # Import the existing config class
        from .feature_consistency import FeatureConsistencyConfig as ExistingConfig
        
        # Create instance with matching parameters
        existing_config = ExistingConfig()
        existing_config.warmup_trim_days = simple_config.warmup_trim_days
        existing_config.nan_drop_threshold_per_symbol = simple_config.nan_drop_threshold_per_symbol
        existing_config.global_feature_keep_ratio = simple_config.global_feature_keep_ratio
        existing_config.min_symbol_feature_coverage = simple_config.min_symbol_feature_coverage
        existing_config.use_missingness_mask = simple_config.use_missingness_mask
        existing_config.imputation_strategy = simple_config.imputation_strategy
        existing_config.imputation_value = simple_config.imputation_value
        existing_config.manifest_path = simple_config.manifest_path
        existing_config.config_backup_path = simple_config.config_backup_path
        existing_config.log_level = simple_config.log_level
        existing_config.detailed_logging = simple_config.detailed_logging
        
        return existing_config
        
    except ImportError:
        # If the existing config class is not available, return the simple config
        logger.warning("Could not import existing FeatureConsistencyConfig, using simple config")
        return simple_config


def load_config_for_feature_consistency_manager(environment: str = None):
    """
    Load configuration specifically for use with FeatureConsistencyManager.
    
    This function loads configuration and converts it to the format expected
    by the existing FeatureConsistencyManager implementation.
    """
    simple_config = load_config_for_environment(environment)
    return convert_to_feature_consistency_config(simple_config)


if __name__ == "__main__":
    # Test configuration loading
    import sys
    
    if len(sys.argv) > 1:
        env = sys.argv[1]
        config = load_config_for_environment(env)
        print(f"Loaded configuration for {env} environment:")
        print(json.dumps(config.to_dict(), indent=2))
    else:
        # Test all environments
        for env in ['development', 'testing', 'production']:
            try:
                config = load_config_for_environment(env)
                print(f"✓ {env} configuration loaded successfully")
                print(f"  - Warmup trim days: {config.warmup_trim_days}")
                print(f"  - NaN threshold: {config.nan_drop_threshold_per_symbol}")
                print(f"  - Global keep ratio: {config.global_feature_keep_ratio}")
                print(f"  - Log level: {config.log_level}")
                print()
            except Exception as e:
                print(f"✗ {env} configuration failed: {e}")
                print()