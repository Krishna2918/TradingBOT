# Feature Consistency Configuration System

This directory contains configuration templates and documentation for the Feature Consistency System used in the trading bot's LSTM training pipeline.

## Overview

The Feature Consistency System ensures that all symbols have identical feature sets and handles missing data robustly. This prevents training instability caused by feature drift where different symbols end up with different numbers of features.

## Configuration Files

### Templates

The `templates/` directory contains environment-specific configuration templates:

- `feature_consistency_development.json` - Development environment settings
- `feature_consistency_testing.json` - Testing environment settings  
- `feature_consistency_production.json` - Production environment settings

### Configuration Parameters

#### Core Processing Parameters

- **`warmup_trim_days`** (int): Number of days to trim from the start of each symbol's data to remove technical indicator warm-up periods. Default: 200 for production, 100 for development, 50 for testing.

- **`nan_drop_threshold_per_symbol`** (float): Maximum NaN ratio allowed per symbol before dropping features. Range: 0.01-0.50. Default: 0.05 for production, 0.10 for development, 0.15 for testing.

- **`global_feature_keep_ratio`** (float): Minimum fraction of symbols that must have a feature for it to be included in the global feature whitelist. Range: 0.50-1.00. Default: 0.95 for production, 0.90 for development, 0.80 for testing.

- **`min_symbol_feature_coverage`** (float): Minimum fraction of whitelist features a symbol must have to be included in training. Range: 0.50-1.00. Default: 0.90 for production, 0.80 for development, 0.70 for testing.

#### Imputation Settings

- **`use_missingness_mask`** (bool): Whether to create binary `_isnan` columns indicating original missing values before imputation. Default: true.

- **`imputation_strategy`** (string): Strategy for filling remaining missing values. Options: "zero", "mean", "median". Default: "zero".

- **`imputation_value`** (float): Value to use when imputation_strategy is "zero". Default: 0.0.

#### File Paths

- **`manifest_path`** (string): Path where the feature manifest JSON file is saved/loaded. Default: "models/feature_manifest.json".

- **`config_backup_path`** (string): Path where configuration backups are saved. Default: "models/feature_consistency_config.json".

#### Logging

- **`log_level`** (string): Logging verbosity level. Options: "DEBUG", "INFO", "WARNING", "ERROR". Default: "INFO" for production, "DEBUG" for development/testing.

- **`detailed_logging`** (bool): Enable detailed logging of processing decisions. Default: true.

## Environment-Specific Settings

### Development Environment
- Reduced warmup period (100 days) for faster iteration
- More lenient thresholds to include more data for experimentation
- DEBUG logging enabled
- Metrics collection disabled

### Testing Environment  
- Minimal warmup period (50 days) for fast test execution
- Very lenient thresholds to test edge cases
- DEBUG logging enabled
- All monitoring features disabled

### Production Environment
- Full warmup period (200 days) for stable indicators
- Strict thresholds for data quality
- INFO logging for performance
- All monitoring and alerting enabled

## Usage

### Loading Configuration

```python
from ai.data.simple_config_loader import load_config_for_environment

# Load configuration for specific environment
config = load_config_for_environment('production')

# Use with FeatureConsistencyManager
from ai.data.simple_config_loader import load_config_for_feature_consistency_manager
from ai.data.feature_consistency import FeatureConsistencyManager

config = load_config_for_feature_consistency_manager('production')
manager = FeatureConsistencyManager(config)
```

### Runtime Overrides

Configuration can be overridden at runtime using environment variables:

```bash
# Override warmup period
export FC_WARMUP_TRIM_DAYS=150

# Override NaN threshold
export FC_NAN_THRESHOLD=0.08

# Override log level
export FC_LOG_LEVEL=WARNING

# Override manifest path
export FC_MANIFEST_PATH=/custom/path/manifest.json
```

### Validation

Use the validation script to check configuration files:

```bash
# Validate all environments
python scripts/validate_feature_config.py --all

# Validate specific environment
python scripts/validate_feature_config.py --env production

# Validate specific file
python scripts/validate_feature_config.py --file config/templates/feature_consistency_production.json

# Show configuration summary
python scripts/validate_feature_config.py --show production
```

## Configuration File Format

Configuration files use a nested JSON structure with metadata:

```json
{
  "_description": "Feature Consistency System Configuration - Production Environment",
  "_version": "1.0",
  "_environment": "production",
  "_created": "2025-10-27T12:00:00.000000",
  
  "warmup_trim_days": {
    "value": 200,
    "description": "Number of days to trim from start to remove indicator warm-up period",
    "min": 0,
    "max": 500,
    "environment_note": "Standard production warmup period for stable indicators"
  },
  
  "nan_drop_threshold_per_symbol": {
    "value": 0.05,
    "description": "Maximum NaN ratio allowed per symbol (0.05 = 5%)",
    "min": 0.01,
    "max": 0.50,
    "environment_note": "Strict threshold for production data quality"
  }
}
```

## Best Practices

### Development
- Use reduced warmup periods to speed up iteration
- Enable detailed logging for debugging
- Use more lenient thresholds to include more data for experimentation

### Testing
- Use minimal warmup periods for fast test execution
- Test with various threshold combinations
- Validate configuration loading and validation logic

### Production
- Use full warmup periods for stable technical indicators
- Use strict thresholds to ensure data quality
- Enable monitoring and alerting
- Regularly backup configurations

## Troubleshooting

### Common Issues

1. **Configuration file not found**: Check that the file exists in the expected location and has the correct name format.

2. **Invalid parameter values**: Use the validation script to check for out-of-range values or invalid options.

3. **Environment variable overrides not working**: Ensure environment variables use the correct prefix (`FC_`) and are set before running the application.

4. **Import errors**: Make sure the `src` directory is in your Python path when importing configuration modules.

### Validation Errors

The validation system provides detailed error messages with suggestions:

- **Range errors**: Parameter values outside allowed ranges
- **Type errors**: Incorrect parameter types (e.g., string instead of number)
- **Option errors**: Invalid choices for enumerated parameters
- **Cross-field errors**: Inconsistencies between related parameters

### Getting Help

Run the validation script with `--help` for usage information:

```bash
python scripts/validate_feature_config.py --help
```

For integration testing:

```bash
python test_config_integration.py
```

For basic configuration testing:

```bash
python test_config_simple.py
```