#!/usr/bin/env python3
"""
Simple Feature Consistency Configuration Validation Script

This script provides basic validation for feature consistency configuration files.
"""

import argparse
import json
import sys
from pathlib import Path

# Add the src directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai.data.simple_config_loader import load_config_from_file, load_config_for_environment


def validate_config_file(config_path):
    """Validate a specific configuration file."""
    print(f"Validating configuration file: {config_path}")
    print("-" * 50)
    
    try:
        config = load_config_from_file(config_path)
        
        print("✓ Configuration loaded successfully")
        print(f"✓ All validation checks passed")
        
        # Show key configuration values
        print(f"\nKey Configuration Values:")
        print(f"  Warmup trim days: {config.warmup_trim_days}")
        print(f"  NaN threshold per symbol: {config.nan_drop_threshold_per_symbol}")
        print(f"  Global feature keep ratio: {config.global_feature_keep_ratio}")
        print(f"  Min symbol feature coverage: {config.min_symbol_feature_coverage}")
        print(f"  Use missingness mask: {config.use_missingness_mask}")
        print(f"  Imputation strategy: {config.imputation_strategy}")
        print(f"  Log level: {config.log_level}")
        print(f"  Manifest path: {config.manifest_path}")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration validation failed: {e}")
        return False


def validate_environment_config(environment):
    """Validate configuration for a specific environment."""
    print(f"Validating {environment} environment configuration")
    print("-" * 50)
    
    try:
        config = load_config_for_environment(environment)
        
        print("✓ Configuration loaded successfully")
        print(f"✓ All validation checks passed")
        
        # Environment-specific validation
        if environment == 'production':
            warnings = []
            if config.nan_drop_threshold_per_symbol > 0.10:
                warnings.append(f"NaN threshold ({config.nan_drop_threshold_per_symbol}) is high for production")
            if config.global_feature_keep_ratio < 0.90:
                warnings.append(f"Global keep ratio ({config.global_feature_keep_ratio}) is low for production")
            if config.log_level == 'DEBUG':
                warnings.append("DEBUG logging may impact production performance")
            
            if warnings:
                print(f"\n⚠️  Production Environment Warnings:")
                for warning in warnings:
                    print(f"  - {warning}")
        
        elif environment == 'testing':
            if config.warmup_trim_days > 100:
                print(f"\n⚠️  Testing Environment Warning:")
                print(f"  - Warmup period ({config.warmup_trim_days}) may slow test execution")
        
        # Show key configuration values
        print(f"\nKey Configuration Values:")
        print(f"  Warmup trim days: {config.warmup_trim_days}")
        print(f"  NaN threshold per symbol: {config.nan_drop_threshold_per_symbol}")
        print(f"  Global feature keep ratio: {config.global_feature_keep_ratio}")
        print(f"  Min symbol feature coverage: {config.min_symbol_feature_coverage}")
        print(f"  Use missingness mask: {config.use_missingness_mask}")
        print(f"  Imputation strategy: {config.imputation_strategy}")
        print(f"  Log level: {config.log_level}")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration validation failed: {e}")
        return False


def validate_all_environments():
    """Validate configurations for all environments."""
    print("Validating all environment configurations")
    print("=" * 60)
    
    environments = ['development', 'testing', 'production']
    all_valid = True
    
    for env in environments:
        print(f"\n{env.upper()} ENVIRONMENT:")
        is_valid = validate_environment_config(env)
        all_valid = all_valid and is_valid
        
        if not is_valid:
            print(f"✗ {env} environment configuration is invalid")
        else:
            print(f"✓ {env} environment configuration is valid")
    
    print(f"\n{'='*60}")
    if all_valid:
        print("✓ All environment configurations are valid!")
    else:
        print("✗ Some environment configurations have issues")
    
    return all_valid


def show_config_summary(environment):
    """Show a summary of configuration for an environment."""
    try:
        config = load_config_for_environment(environment)
        
        print(f"\nFeature Consistency Configuration Summary - {environment.title()}")
        print("=" * 60)
        
        print(f"\nCore Processing Parameters:")
        print(f"  Warmup trim days: {config.warmup_trim_days}")
        print(f"  NaN threshold per symbol: {config.nan_drop_threshold_per_symbol}")
        print(f"  Global feature keep ratio: {config.global_feature_keep_ratio}")
        print(f"  Min symbol feature coverage: {config.min_symbol_feature_coverage}")
        
        print(f"\nImputation Settings:")
        print(f"  Use missingness mask: {config.use_missingness_mask}")
        print(f"  Imputation strategy: {config.imputation_strategy}")
        print(f"  Imputation value: {config.imputation_value}")
        
        print(f"\nFile Paths:")
        print(f"  Manifest path: {config.manifest_path}")
        print(f"  Config backup path: {config.config_backup_path}")
        
        print(f"\nLogging:")
        print(f"  Log level: {config.log_level}")
        print(f"  Detailed logging: {config.detailed_logging}")
        
        print(f"\nAdvanced Settings:")
        print(f"  Enforce feature manifest: {config.enforce_feature_manifest}")
        print(f"  Memory optimization: {config.memory_optimization}")
        print(f"  Max symbols per batch: {config.max_symbols_per_batch}")
        
        print(f"\nMonitoring:")
        print(f"  Enable metrics collection: {config.enable_metrics_collection}")
        print(f"  Enable drift detection: {config.enable_drift_detection}")
        print(f"  Alert on high exclusion rate: {config.alert_on_high_exclusion_rate}")
        print(f"  Exclusion rate threshold: {config.exclusion_rate_threshold}")
        
    except Exception as e:
        print(f"Error loading configuration: {e}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Feature Consistency Configuration Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate a specific configuration file
  python validate_feature_config.py --file config/templates/feature_consistency_production.json
  
  # Validate configuration for an environment
  python validate_feature_config.py --env production
  
  # Validate all environments
  python validate_feature_config.py --all
  
  # Show configuration summary
  python validate_feature_config.py --show development
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--file', help='Validate a specific configuration file')
    group.add_argument('--env', choices=['development', 'testing', 'production'],
                      help='Validate configuration for an environment')
    group.add_argument('--all', action='store_true', help='Validate all environments')
    group.add_argument('--show', choices=['development', 'testing', 'production'],
                      help='Show configuration summary for an environment')
    
    args = parser.parse_args()
    
    try:
        if args.file:
            success = validate_config_file(args.file)
        elif args.env:
            success = validate_environment_config(args.env)
        elif args.all:
            success = validate_all_environments()
        elif args.show:
            show_config_summary(args.show)
            success = True
        
        if not success:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()