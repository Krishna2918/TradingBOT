#!/usr/bin/env python3
"""
Integration test for Feature Consistency Configuration System

This test verifies that the configuration system integrates properly with
the existing feature consistency implementation.
"""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ai.data.simple_config_loader import load_config_for_environment, load_config_for_feature_consistency_manager, FeatureConsistencyConfig
from ai.data.feature_consistency import FeatureConsistencyManager


def test_config_integration():
    """Test that configuration integrates with FeatureConsistencyManager."""
    print("Testing Feature Consistency Configuration Integration")
    print("=" * 60)
    
    # Test each environment
    environments = ['development', 'testing', 'production']
    
    for env in environments:
        print(f"\nTesting {env} environment...")
        
        try:
            # Load configuration for display
            config = load_config_for_environment(env)
            print(f"  ✓ Configuration loaded for {env}")
            
            # Load configuration for FeatureConsistencyManager
            manager_config = load_config_for_feature_consistency_manager(env)
            
            # Create FeatureConsistencyManager with config
            manager = FeatureConsistencyManager(manager_config)
            print(f"  ✓ FeatureConsistencyManager created with {env} config")
            
            # Verify configuration values are applied
            assert manager.config.warmup_trim_days == config.warmup_trim_days
            assert manager.config.nan_drop_threshold_per_symbol == config.nan_drop_threshold_per_symbol
            assert manager.config.global_feature_keep_ratio == config.global_feature_keep_ratio
            assert manager.config.min_symbol_feature_coverage == config.min_symbol_feature_coverage
            assert manager.config.use_missingness_mask == config.use_missingness_mask
            print(f"  ✓ Configuration values correctly applied")
            
            # Test configuration-specific behavior
            if env == 'development':
                assert config.warmup_trim_days <= 100, "Development should have reduced warmup for faster iteration"
                assert config.log_level == 'DEBUG', "Development should use DEBUG logging"
            elif env == 'testing':
                assert config.warmup_trim_days <= 50, "Testing should have minimal warmup for fast tests"
                assert config.log_level == 'DEBUG', "Testing should use DEBUG logging"
            elif env == 'production':
                assert config.warmup_trim_days == 200, "Production should use full warmup period"
                assert config.log_level == 'INFO', "Production should use INFO logging"
                assert config.nan_drop_threshold_per_symbol <= 0.05, "Production should have strict NaN threshold"
            
            print(f"  ✓ Environment-specific settings validated")
            
        except Exception as e:
            print(f"  ✗ Error testing {env}: {e}")
            return False
    
    print(f"\n{'='*60}")
    print("✓ All configuration integration tests passed!")
    return True


def test_runtime_overrides():
    """Test runtime configuration overrides via environment variables."""
    print("\nTesting Runtime Configuration Overrides")
    print("-" * 40)
    
    import os
    
    # Set some environment variables
    original_values = {}
    test_env_vars = {
        'FC_WARMUP_TRIM_DAYS': '150',
        'FC_NAN_THRESHOLD': '0.08',
        'FC_LOG_LEVEL': 'WARNING'
    }
    
    try:
        # Save original values and set test values
        for var, value in test_env_vars.items():
            original_values[var] = os.getenv(var)
            os.environ[var] = value
        
        # Load configuration (should pick up environment variables)
        config = load_config_for_environment('development')
        
        # Verify overrides were applied
        assert config.warmup_trim_days == 150, f"Expected 150, got {config.warmup_trim_days}"
        assert config.nan_drop_threshold_per_symbol == 0.08, f"Expected 0.08, got {config.nan_drop_threshold_per_symbol}"
        assert config.log_level == 'WARNING', f"Expected WARNING, got {config.log_level}"
        
        print("✓ Runtime overrides applied successfully")
        
    finally:
        # Restore original environment variables
        for var, original_value in original_values.items():
            if original_value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = original_value
    
    return True


def test_config_validation():
    """Test configuration validation."""
    print("\nTesting Configuration Validation")
    print("-" * 40)
    
    # Test valid configuration
    try:
        valid_config = FeatureConsistencyConfig(
            warmup_trim_days=200,
            nan_drop_threshold_per_symbol=0.05,
            global_feature_keep_ratio=0.95,
            min_symbol_feature_coverage=0.90
        )
        print("✓ Valid configuration accepted")
    except Exception as e:
        print(f"✗ Valid configuration rejected: {e}")
        return False
    
    # Test invalid configurations
    invalid_configs = [
        {'warmup_trim_days': -1, 'error': 'negative warmup days'},
        {'warmup_trim_days': 600, 'error': 'excessive warmup days'},
        {'nan_drop_threshold_per_symbol': 0.0, 'error': 'too low NaN threshold'},
        {'nan_drop_threshold_per_symbol': 0.6, 'error': 'too high NaN threshold'},
        {'global_feature_keep_ratio': 0.4, 'error': 'too low global ratio'},
        {'global_feature_keep_ratio': 1.1, 'error': 'too high global ratio'},
        {'imputation_strategy': 'invalid', 'error': 'invalid imputation strategy'},
        {'log_level': 'INVALID', 'error': 'invalid log level'}
    ]
    
    for invalid_config in invalid_configs:
        error_desc = invalid_config.pop('error')
        try:
            FeatureConsistencyConfig(**invalid_config)
            print(f"✗ Invalid configuration accepted: {error_desc}")
            return False
        except ValueError:
            print(f"✓ Invalid configuration rejected: {error_desc}")
        except Exception as e:
            print(f"✗ Unexpected error for {error_desc}: {e}")
            return False
    
    return True


def main():
    """Run all integration tests."""
    print("Feature Consistency Configuration Integration Tests")
    print("=" * 60)
    
    tests = [
        test_config_integration,
        test_runtime_overrides,
        test_config_validation
    ]
    
    all_passed = True
    
    for test in tests:
        try:
            result = test()
            all_passed = all_passed and result
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with error: {e}")
            all_passed = False
    
    print(f"\n{'='*60}")
    if all_passed:
        print("✓ All integration tests passed!")
        return 0
    else:
        print("✗ Some integration tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())