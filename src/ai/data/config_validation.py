"""
Configuration Validation Module for Feature Consistency System

This module provides comprehensive validation for feature consistency configuration
files with helpful error messages and environment-specific parameter overrides.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import os
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class ValidationError:
    """Represents a configuration validation error."""
    field_name: str
    current_value: Any
    error_message: str
    severity: str = "error"  # "error", "warning", "info"
    suggestion: Optional[str] = None

@dataclass
class ValidationResult:
    """Results from configuration validation."""
    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    info_messages: List[ValidationError] = field(default_factory=list)
    validated_config: Optional[Dict[str, Any]] = None
    
    def add_error(self, field_name: str, current_value: Any, message: str, suggestion: str = None):
        """Add a validation error."""
        self.errors.append(ValidationError(field_name, current_value, message, "error", suggestion))
        self.is_valid = False
    
    def add_warning(self, field_name: str, current_value: Any, message: str, suggestion: str = None):
        """Add a validation warning."""
        self.warnings.append(ValidationError(field_name, current_value, message, "warning", suggestion))
    
    def add_info(self, field_name: str, current_value: Any, message: str, suggestion: str = None):
        """Add a validation info message."""
        self.info_messages.append(ValidationError(field_name, current_value, message, "info", suggestion))
    
    def get_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        return {
            'is_valid': self.is_valid,
            'error_count': len(self.errors),
            'warning_count': len(self.warnings),
            'info_count': len(self.info_messages),
            'total_issues': len(self.errors) + len(self.warnings)
        }

class FeatureConsistencyConfigValidator:
    """
    Comprehensive validator for feature consistency configuration files.
    
    Provides validation with helpful error messages and environment-specific
    parameter override support.
    """
    
    def __init__(self):
        """Initialize the validator with default rules."""
        self.validation_rules = {
            'warmup_trim_days': {
                'type': int,
                'min': 0,
                'max': 500,
                'required': True,
                'description': 'Number of days to trim from start for indicator warm-up'
            },
            'nan_drop_threshold_per_symbol': {
                'type': float,
                'min': 0.01,
                'max': 0.50,
                'required': True,
                'description': 'Maximum NaN ratio allowed per symbol'
            },
            'global_feature_keep_ratio': {
                'type': float,
                'min': 0.50,
                'max': 1.00,
                'required': True,
                'description': 'Minimum symbol coverage required to keep feature globally'
            },
            'min_symbol_feature_coverage': {
                'type': float,
                'min': 0.50,
                'max': 1.00,
                'required': True,
                'description': 'Minimum feature coverage required to include symbol'
            },
            'use_missingness_mask': {
                'type': bool,
                'required': True,
                'description': 'Whether to create _isnan columns for missing values'
            },
            'imputation_strategy': {
                'type': str,
                'options': ['zero', 'mean', 'median'],
                'required': True,
                'description': 'Strategy for final imputation'
            },
            'imputation_value': {
                'type': (int, float),
                'required': False,
                'description': 'Value to use for imputation when strategy is zero'
            },
            'manifest_path': {
                'type': str,
                'required': True,
                'description': 'Path to save/load feature manifest JSON file'
            },
            'log_level': {
                'type': str,
                'options': ['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                'required': False,
                'default': 'INFO',
                'description': 'Logging level'
            },
            'detailed_logging': {
                'type': bool,
                'required': False,
                'default': True,
                'description': 'Enable detailed logging of processing decisions'
            }
        }
    
    def validate_config(self, config: Dict[str, Any], environment: str = None) -> ValidationResult:
        """
        Validate a configuration dictionary.
        
        Args:
            config: Configuration dictionary to validate
            environment: Environment name for context-specific validation
            
        Returns:
            ValidationResult with validation status and detailed feedback
        """
        result = ValidationResult(is_valid=True)
        validated_config = {}
        
        # Extract values from nested structure if needed
        flat_config = self._flatten_config(config)
        
        # Validate each required field
        for field_name, rules in self.validation_rules.items():
            value = flat_config.get(field_name)
            
            # Check if required field is missing
            if rules.get('required', False) and value is None:
                if 'default' in rules:
                    value = rules['default']
                    result.add_info(field_name, value, f"Using default value: {value}")
                else:
                    result.add_error(
                        field_name, 
                        None, 
                        f"Required field '{field_name}' is missing",
                        f"Add '{field_name}' to configuration"
                    )
                    continue
            
            if value is not None:
                # Validate type
                expected_type = rules.get('type')
                if expected_type and not isinstance(value, expected_type):
                    result.add_error(
                        field_name,
                        value,
                        f"Expected {expected_type.__name__}, got {type(value).__name__}",
                        f"Change {field_name} to {expected_type.__name__} type"
                    )
                    continue
                
                # Validate range for numeric values
                if isinstance(value, (int, float)):
                    min_val = rules.get('min')
                    max_val = rules.get('max')
                    
                    if min_val is not None and value < min_val:
                        result.add_error(
                            field_name,
                            value,
                            f"Value {value} is below minimum {min_val}",
                            f"Set {field_name} to at least {min_val}"
                        )
                        continue
                    
                    if max_val is not None and value > max_val:
                        result.add_error(
                            field_name,
                            value,
                            f"Value {value} is above maximum {max_val}",
                            f"Set {field_name} to at most {max_val}"
                        )
                        continue
                
                # Validate options for string values
                if isinstance(value, str):
                    options = rules.get('options')
                    if options and value not in options:
                        result.add_error(
                            field_name,
                            value,
                            f"Invalid option '{value}'. Must be one of: {', '.join(options)}",
                            f"Set {field_name} to one of: {', '.join(options)}"
                        )
                        continue
                
                validated_config[field_name] = value
        
        # Environment-specific validation
        if environment:
            self._validate_environment_specific(validated_config, environment, result)
        
        # Cross-field validation
        self._validate_cross_field_rules(validated_config, result)
        
        if result.is_valid:
            result.validated_config = validated_config
        
        return result
    
    def _flatten_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Flatten nested configuration structure to extract actual values.
        
        Handles both simple key-value pairs and nested structures with 'value' keys.
        """
        flat_config = {}
        
        for key, value in config.items():
            if key.startswith('_'):  # Skip metadata fields
                continue
                
            if isinstance(value, dict) and 'value' in value:
                # Extract value from nested structure
                flat_config[key] = value['value']
            elif not isinstance(value, dict) or key in ['validation_rules', 'testing_settings', 'performance_settings', 'monitoring_settings']:
                # Skip complex nested objects that aren't configuration values
                if key not in ['validation_rules', 'testing_settings', 'performance_settings', 'monitoring_settings', 'test_data_generation']:
                    flat_config[key] = value
        
        return flat_config
    
    def _validate_environment_specific(self, config: Dict[str, Any], environment: str, result: ValidationResult):
        """Validate environment-specific configuration rules."""
        
        if environment == 'production':
            # Production should have stricter thresholds
            if config.get('nan_drop_threshold_per_symbol', 0) > 0.10:
                result.add_warning(
                    'nan_drop_threshold_per_symbol',
                    config['nan_drop_threshold_per_symbol'],
                    'Production environments typically use stricter NaN thresholds (≤ 0.10)',
                    'Consider reducing nan_drop_threshold_per_symbol for production'
                )
            
            if config.get('global_feature_keep_ratio', 0) < 0.90:
                result.add_warning(
                    'global_feature_keep_ratio',
                    config['global_feature_keep_ratio'],
                    'Production environments typically require higher feature stability (≥ 0.90)',
                    'Consider increasing global_feature_keep_ratio for production'
                )
            
            if config.get('log_level') == 'DEBUG':
                result.add_warning(
                    'log_level',
                    config['log_level'],
                    'DEBUG logging may impact production performance',
                    'Consider using INFO or WARNING for production'
                )
        
        elif environment == 'development':
            # Development can be more lenient but warn about extreme values
            if config.get('warmup_trim_days', 0) > 300:
                result.add_info(
                    'warmup_trim_days',
                    config['warmup_trim_days'],
                    'Large warmup periods may slow development iteration',
                    'Consider reducing warmup_trim_days for faster development'
                )
        
        elif environment == 'testing':
            # Testing should have fast execution settings
            if config.get('warmup_trim_days', 0) > 100:
                result.add_warning(
                    'warmup_trim_days',
                    config['warmup_trim_days'],
                    'Large warmup periods may slow test execution',
                    'Consider reducing warmup_trim_days for faster tests'
                )
    
    def _validate_cross_field_rules(self, config: Dict[str, Any], result: ValidationResult):
        """Validate rules that depend on multiple configuration fields."""
        
        # Ensure global threshold is not lower than symbol threshold
        global_ratio = config.get('global_feature_keep_ratio')
        symbol_ratio = config.get('min_symbol_feature_coverage')
        
        if global_ratio and symbol_ratio and global_ratio < symbol_ratio:
            result.add_warning(
                'global_feature_keep_ratio',
                global_ratio,
                f'Global feature ratio ({global_ratio}) is lower than symbol coverage ratio ({symbol_ratio})',
                'Consider setting global_feature_keep_ratio ≥ min_symbol_feature_coverage'
            )
        
        # Validate imputation strategy consistency
        strategy = config.get('imputation_strategy')
        imputation_value = config.get('imputation_value')
        
        if strategy == 'zero' and imputation_value is not None and imputation_value != 0.0:
            result.add_warning(
                'imputation_value',
                imputation_value,
                f'Imputation value ({imputation_value}) conflicts with zero strategy',
                'Set imputation_value to 0.0 when using zero strategy'
            )
    
    def validate_config_file(self, config_path: Union[str, Path]) -> ValidationResult:
        """
        Validate a configuration file.
        
        Args:
            config_path: Path to configuration JSON file
            
        Returns:
            ValidationResult with validation status and detailed feedback
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            result = ValidationResult(is_valid=False)
            result.add_error('file', str(config_path), 'Configuration file does not exist')
            return result
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            result = ValidationResult(is_valid=False)
            result.add_error('file', str(config_path), f'Invalid JSON: {e}')
            return result
        except Exception as e:
            result = ValidationResult(is_valid=False)
            result.add_error('file', str(config_path), f'Error reading file: {e}')
            return result
        
        # Extract environment from config or filename
        environment = config.get('_environment')
        if not environment:
            if 'production' in str(config_path):
                environment = 'production'
            elif 'development' in str(config_path):
                environment = 'development'
            elif 'testing' in str(config_path):
                environment = 'testing'
        
        return self.validate_config(config, environment)
    
    def generate_config_template(self, environment: str = 'development') -> Dict[str, Any]:
        """
        Generate a configuration template for the specified environment.
        
        Args:
            environment: Target environment (development, testing, production)
            
        Returns:
            Configuration template dictionary
        """
        template = {
            '_description': f'Feature Consistency System Configuration - {environment.title()} Environment',
            '_version': '1.0',
            '_environment': environment,
            '_created': datetime.now().isoformat()
        }
        
        # Environment-specific defaults
        env_defaults = {
            'development': {
                'warmup_trim_days': 100,
                'nan_drop_threshold_per_symbol': 0.10,
                'global_feature_keep_ratio': 0.90,
                'min_symbol_feature_coverage': 0.80,
                'log_level': 'DEBUG'
            },
            'testing': {
                'warmup_trim_days': 50,
                'nan_drop_threshold_per_symbol': 0.15,
                'global_feature_keep_ratio': 0.80,
                'min_symbol_feature_coverage': 0.70,
                'log_level': 'DEBUG'
            },
            'production': {
                'warmup_trim_days': 200,
                'nan_drop_threshold_per_symbol': 0.05,
                'global_feature_keep_ratio': 0.95,
                'min_symbol_feature_coverage': 0.90,
                'log_level': 'INFO'
            }
        }
        
        defaults = env_defaults.get(environment, env_defaults['development'])
        
        # Add configuration fields with metadata
        for field_name, rules in self.validation_rules.items():
            value = defaults.get(field_name)
            if value is None:
                if field_name == 'use_missingness_mask':
                    value = True
                elif field_name == 'imputation_strategy':
                    value = 'zero'
                elif field_name == 'imputation_value':
                    value = 0.0
                elif field_name == 'manifest_path':
                    value = f'models/feature_manifest_{environment}.json'
                elif field_name == 'detailed_logging':
                    value = True
                else:
                    continue
            
            template[field_name] = {
                'value': value,
                'description': rules['description']
            }
            
            # Add validation metadata
            if 'min' in rules and 'max' in rules:
                template[field_name]['min'] = rules['min']
                template[field_name]['max'] = rules['max']
            
            if 'options' in rules:
                template[field_name]['options'] = rules['options']
        
        return template
    
    def print_validation_report(self, result: ValidationResult, config_path: str = None):
        """
        Print a formatted validation report.
        
        Args:
            result: ValidationResult to report
            config_path: Optional path to configuration file being validated
        """
        print("\n" + "="*60)
        print("FEATURE CONSISTENCY CONFIGURATION VALIDATION REPORT")
        print("="*60)
        
        if config_path:
            print(f"Configuration File: {config_path}")
        
        summary = result.get_summary()
        print(f"Validation Status: {'✓ VALID' if result.is_valid else '✗ INVALID'}")
        print(f"Errors: {summary['error_count']}")
        print(f"Warnings: {summary['warning_count']}")
        print(f"Info Messages: {summary['info_count']}")
        
        # Print errors
        if result.errors:
            print("\n" + "-"*40)
            print("ERRORS:")
            print("-"*40)
            for error in result.errors:
                print(f"❌ {error.field_name}: {error.error_message}")
                if error.current_value is not None:
                    print(f"   Current value: {error.current_value}")
                if error.suggestion:
                    print(f"   Suggestion: {error.suggestion}")
                print()
        
        # Print warnings
        if result.warnings:
            print("\n" + "-"*40)
            print("WARNINGS:")
            print("-"*40)
            for warning in result.warnings:
                print(f"⚠️  {warning.field_name}: {warning.error_message}")
                if warning.current_value is not None:
                    print(f"   Current value: {warning.current_value}")
                if warning.suggestion:
                    print(f"   Suggestion: {warning.suggestion}")
                print()
        
        # Print info messages
        if result.info_messages:
            print("\n" + "-"*40)
            print("INFO:")
            print("-"*40)
            for info in result.info_messages:
                print(f"ℹ️  {info.field_name}: {info.error_message}")
                if info.current_value is not None:
                    print(f"   Value: {info.current_value}")
                print()
        
        print("="*60)


def load_and_validate_config(config_path: Union[str, Path], environment: str = None) -> Tuple[Dict[str, Any], ValidationResult]:
    """
    Load and validate a configuration file.
    
    Args:
        config_path: Path to configuration file
        environment: Optional environment override
        
    Returns:
        Tuple of (config_dict, validation_result)
    """
    validator = FeatureConsistencyConfigValidator()
    result = validator.validate_config_file(config_path)
    
    if result.is_valid and result.validated_config:
        return result.validated_config, result
    else:
        return {}, result


def create_environment_configs(output_dir: Union[str, Path] = "config/templates"):
    """
    Create configuration templates for all environments.
    
    Args:
        output_dir: Directory to save configuration templates
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    validator = FeatureConsistencyConfigValidator()
    
    environments = ['development', 'testing', 'production']
    
    for env in environments:
        template = validator.generate_config_template(env)
        config_path = output_dir / f"feature_consistency_{env}.json"
        
        with open(config_path, 'w') as f:
            json.dump(template, f, indent=2)
        
        print(f"Created {env} configuration template: {config_path}")


if __name__ == "__main__":
    # Example usage and testing
    import sys
    
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        validator = FeatureConsistencyConfigValidator()
        result = validator.validate_config_file(config_file)
        validator.print_validation_report(result, config_file)
    else:
        # Create example templates
        create_environment_configs()
        print("\nConfiguration templates created successfully!")
        print("Use: python config_validation.py <config_file> to validate a configuration")