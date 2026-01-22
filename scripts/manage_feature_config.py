#!/usr/bin/env python3
"""
Feature Consistency Configuration Management CLI

This script provides command-line utilities for managing feature consistency
configuration files, validation, and environment-specific setup.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Add the src directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai.data.config_validation import FeatureConsistencyConfigValidator, create_environment_configs
from ai.data.config_loader import ConfigurationManager, load_config_for_environment


def validate_config_command(args):
    """Validate a configuration file."""
    validator = FeatureConsistencyConfigValidator()
    result = validator.validate_config_file(args.config_file)
    
    validator.print_validation_report(result, args.config_file)
    
    if not result.is_valid:
        sys.exit(1)
    
    print(f"\n✓ Configuration file '{args.config_file}' is valid!")


def create_templates_command(args):
    """Create configuration templates for all environments."""
    output_dir = Path(args.output_dir)
    
    print(f"Creating configuration templates in: {output_dir}")
    create_environment_configs(output_dir)
    
    print("\n✓ Configuration templates created successfully!")
    print(f"Templates created in: {output_dir}")
    
    # List created files
    for env in ['development', 'testing', 'production']:
        template_path = output_dir / f"feature_consistency_{env}.json"
        if template_path.exists():
            print(f"  - {template_path}")


def show_config_command(args):
    """Show configuration for an environment."""
    try:
        config = load_config_for_environment(args.environment)
        
        print(f"\nFeature Consistency Configuration - {args.environment.title()} Environment")
        print("=" * 60)
        
        config_dict = config.to_dict()
        
        # Group related settings
        groups = {
            "Core Processing": [
                'warmup_trim_days', 'nan_drop_threshold_per_symbol', 
                'global_feature_keep_ratio', 'min_symbol_feature_coverage'
            ],
            "Imputation Settings": [
                'use_missingness_mask', 'imputation_strategy', 'imputation_value'
            ],
            "File Paths": [
                'manifest_path', 'config_backup_path'
            ],
            "Logging": [
                'log_level', 'detailed_logging'
            ],
            "Advanced Settings": [
                'enforce_feature_manifest', 'batch_processing_enabled', 
                'memory_optimization', 'parallel_processing', 'max_symbols_per_batch'
            ],
            "Monitoring": [
                'enable_metrics_collection', 'enable_drift_detection',
                'alert_on_high_exclusion_rate', 'exclusion_rate_threshold'
            ]
        }
        
        for group_name, fields in groups.items():
            print(f"\n{group_name}:")
            print("-" * len(group_name))
            for field in fields:
                if field in config_dict:
                    value = config_dict[field]
                    print(f"  {field}: {value}")
        
        if args.json:
            print(f"\nJSON Configuration:")
            print("-" * 20)
            print(json.dumps(config_dict, indent=2))
            
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)


def compare_configs_command(args):
    """Compare configurations between environments."""
    try:
        config1 = load_config_for_environment(args.env1)
        config2 = load_config_for_environment(args.env2)
        
        dict1 = config1.to_dict()
        dict2 = config2.to_dict()
        
        print(f"\nConfiguration Comparison: {args.env1} vs {args.env2}")
        print("=" * 60)
        
        # Find differences
        all_keys = set(dict1.keys()) | set(dict2.keys())
        differences = []
        
        for key in sorted(all_keys):
            val1 = dict1.get(key, "NOT SET")
            val2 = dict2.get(key, "NOT SET")
            
            if val1 != val2:
                differences.append((key, val1, val2))
        
        if differences:
            print(f"\nFound {len(differences)} differences:")
            print("-" * 40)
            for key, val1, val2 in differences:
                print(f"{key}:")
                print(f"  {args.env1}: {val1}")
                print(f"  {args.env2}: {val2}")
                print()
        else:
            print("\n✓ No differences found between configurations")
            
    except Exception as e:
        print(f"Error comparing configurations: {e}")
        sys.exit(1)


def test_config_command(args):
    """Test configuration loading and validation."""
    print("Testing Feature Consistency Configuration System")
    print("=" * 50)
    
    # Test environment detection
    manager = ConfigurationManager()
    detected_env = manager._detect_environment()
    print(f"Detected environment: {detected_env}")
    
    # Test loading each environment
    environments = ['development', 'testing', 'production']
    
    for env in environments:
        try:
            print(f"\nTesting {env} environment...")
            config = load_config_for_environment(env)
            
            # Basic validation
            config._validate_config()
            
            print(f"✓ {env} configuration loaded and validated successfully")
            
            # Show key settings
            print(f"  - Warmup trim days: {config.warmup_trim_days}")
            print(f"  - NaN threshold: {config.nan_drop_threshold_per_symbol}")
            print(f"  - Global keep ratio: {config.global_feature_keep_ratio}")
            print(f"  - Log level: {config.log_level}")
            
        except Exception as e:
            print(f"✗ {env} configuration failed: {e}")
    
    print(f"\n✓ Configuration system test completed")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Feature Consistency Configuration Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate a configuration file
  python manage_feature_config.py validate config/feature_consistency_production.json
  
  # Create configuration templates
  python manage_feature_config.py create-templates --output-dir config/templates
  
  # Show configuration for an environment
  python manage_feature_config.py show production
  
  # Compare configurations between environments
  python manage_feature_config.py compare development production
  
  # Test the configuration system
  python manage_feature_config.py test
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate a configuration file')
    validate_parser.add_argument('config_file', help='Path to configuration file')
    validate_parser.set_defaults(func=validate_config_command)
    
    # Create templates command
    create_parser = subparsers.add_parser('create-templates', help='Create configuration templates')
    create_parser.add_argument('--output-dir', default='config/templates', 
                              help='Output directory for templates (default: config/templates)')
    create_parser.set_defaults(func=create_templates_command)
    
    # Show config command
    show_parser = subparsers.add_parser('show', help='Show configuration for an environment')
    show_parser.add_argument('environment', choices=['development', 'testing', 'production'],
                            help='Environment to show configuration for')
    show_parser.add_argument('--json', action='store_true', help='Also output as JSON')
    show_parser.set_defaults(func=show_config_command)
    
    # Compare configs command
    compare_parser = subparsers.add_parser('compare', help='Compare configurations between environments')
    compare_parser.add_argument('env1', choices=['development', 'testing', 'production'],
                               help='First environment')
    compare_parser.add_argument('env2', choices=['development', 'testing', 'production'],
                               help='Second environment')
    compare_parser.set_defaults(func=compare_configs_command)
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test configuration system')
    test_parser.set_defaults(func=test_config_command)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()