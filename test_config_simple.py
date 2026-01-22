#!/usr/bin/env python3
"""
Simple test of configuration validation functionality.
"""

import json
import sys
from pathlib import Path

# Test the existing configuration templates
def test_existing_configs():
    """Test the existing configuration templates."""
    print("Testing existing configuration templates...")
    
    config_dir = Path("config/templates")
    
    for env in ['development', 'testing', 'production']:
        config_file = config_dir / f"feature_consistency_{env}.json"
        
        if config_file.exists():
            print(f"\n✓ Found {env} configuration: {config_file}")
            
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                print(f"  - Environment: {config.get('_environment', 'unknown')}")
                print(f"  - Version: {config.get('_version', 'unknown')}")
                
                # Extract key settings
                warmup_days = config.get('warmup_trim_days', {}).get('value', 'N/A')
                nan_threshold = config.get('nan_drop_threshold_per_symbol', {}).get('value', 'N/A')
                global_ratio = config.get('global_feature_keep_ratio', {}).get('value', 'N/A')
                
                print(f"  - Warmup trim days: {warmup_days}")
                print(f"  - NaN threshold: {nan_threshold}")
                print(f"  - Global keep ratio: {global_ratio}")
                
            except Exception as e:
                print(f"  ✗ Error reading config: {e}")
        else:
            print(f"\n✗ Missing {env} configuration: {config_file}")

def validate_config_structure(config_file):
    """Validate basic configuration structure."""
    print(f"\nValidating configuration structure: {config_file}")
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Check required fields
        required_fields = [
            'warmup_trim_days',
            'nan_drop_threshold_per_symbol', 
            'global_feature_keep_ratio',
            'min_symbol_feature_coverage',
            'use_missingness_mask',
            'imputation_strategy'
        ]
        
        missing_fields = []
        invalid_values = []
        
        for field in required_fields:
            if field not in config:
                missing_fields.append(field)
            else:
                field_config = config[field]
                if isinstance(field_config, dict) and 'value' in field_config:
                    value = field_config['value']
                    
                    # Basic validation
                    if field == 'warmup_trim_days' and (not isinstance(value, int) or value < 0):
                        invalid_values.append(f"{field}: {value} (should be non-negative integer)")
                    elif field == 'nan_drop_threshold_per_symbol' and (not isinstance(value, (int, float)) or not 0.01 <= value <= 0.50):
                        invalid_values.append(f"{field}: {value} (should be between 0.01 and 0.50)")
                    elif field == 'global_feature_keep_ratio' and (not isinstance(value, (int, float)) or not 0.50 <= value <= 1.00):
                        invalid_values.append(f"{field}: {value} (should be between 0.50 and 1.00)")
                    elif field == 'min_symbol_feature_coverage' and (not isinstance(value, (int, float)) or not 0.50 <= value <= 1.00):
                        invalid_values.append(f"{field}: {value} (should be between 0.50 and 1.00)")
                    elif field == 'use_missingness_mask' and not isinstance(value, bool):
                        invalid_values.append(f"{field}: {value} (should be boolean)")
                    elif field == 'imputation_strategy' and value not in ['zero', 'mean', 'median']:
                        invalid_values.append(f"{field}: {value} (should be one of: zero, mean, median)")
        
        # Report results
        if not missing_fields and not invalid_values:
            print("  ✓ Configuration structure is valid")
            return True
        else:
            if missing_fields:
                print(f"  ✗ Missing fields: {', '.join(missing_fields)}")
            if invalid_values:
                print(f"  ✗ Invalid values:")
                for invalid in invalid_values:
                    print(f"    - {invalid}")
            return False
            
    except Exception as e:
        print(f"  ✗ Error validating config: {e}")
        return False

def main():
    """Main test function."""
    print("Feature Consistency Configuration Test")
    print("=" * 50)
    
    # Test existing configurations
    test_existing_configs()
    
    # Validate each configuration
    config_dir = Path("config/templates")
    all_valid = True
    
    for env in ['development', 'testing', 'production']:
        config_file = config_dir / f"feature_consistency_{env}.json"
        if config_file.exists():
            is_valid = validate_config_structure(config_file)
            all_valid = all_valid and is_valid
    
    print(f"\n{'='*50}")
    if all_valid:
        print("✓ All configuration files are valid!")
    else:
        print("✗ Some configuration files have issues")
        sys.exit(1)

if __name__ == "__main__":
    main()