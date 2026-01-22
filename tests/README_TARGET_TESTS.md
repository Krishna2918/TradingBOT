# Target Labeling Surgical Fixes Test Suite

This directory contains comprehensive tests for the target labeling surgical fixes implementation. The test suite validates the complete pipeline from raw price data to encoded targets, ensuring data integrity, preventing leakage, and validating error handling.

## Test Structure

### Unit Tests (`tests/unit/`)

#### `test_ensure_direction_1d.py`
Tests the core `ensure_direction_1d` function with:
- **Forward return calculation accuracy**: Verifies correct calculation using tomorrow's close vs today's close
- **Various neutral band values**: Tests with 0.002, 0.003, 0.004, 0.005 bands
- **Preservation of existing targets**: Ensures existing `direction_1d` columns are not modified
- **Error handling**: Tests missing close column scenarios
- **Edge cases**: Constant prices, extreme volatility, single rows, empty DataFrames
- **Boundary conditions**: Exact neutral band threshold testing
- **NaN handling**: Proper handling of missing price data

#### `test_validation_system.py`
Tests the validation functions with:
- **Symbol validation**: `validate_symbol_targets` with valid/invalid inputs
- **Global validation**: `validate_global_targets` with missing classes
- **Error propagation**: Accurate error messages and proper exception handling
- **Custom requirements**: Testing with different required class sets
- **Performance**: Large dataset validation
- **Realistic distributions**: Trading-like data patterns

### Integration Tests (`tests/integration/`)

#### `test_target_pipeline.py`
Tests the complete end-to-end pipeline:
- **Single symbol pipeline**: Complete flow from raw data to encoded targets
- **Multiple symbols pipeline**: Cross-symbol consistency and validation
- **Data leakage verification**: Ensures no future information leaks into targets
- **Various data patterns**: Trending, sideways, volatile, low-volatility data
- **Error handling**: Recovery from problematic data
- **Performance**: Large dataset processing
- **Consistency**: Reproducible results across runs
- **Edge cases**: Single rows, constant prices, extreme movements

## Key Test Coverage

### Requirements Validation
- **Requirement 1.1-1.5**: Centralized target builder functionality
- **Requirement 2.1-2.5**: Symbol loading integration
- **Requirement 3.2-3.5**: Validation system error handling
- **Requirement 4.1-4.5**: Configurable neutral bands
- **Requirement 5.1-5.5**: Comprehensive logging and validation
- **Requirement 6.1-6.5**: Target encoding separation

### Critical Features Tested
1. **Data Leakage Prevention**: Forward returns use only future data (t+1 close vs t close)
2. **Consistent Target Creation**: Same logic across all symbols and runs
3. **Error Handling**: Hard failures instead of silent fallbacks
4. **Validation Pipeline**: Symbol-level and global-level validation
5. **Neutral Band Configuration**: Proper classification with different thresholds
6. **Edge Case Handling**: Robust behavior with unusual data patterns

## Running the Tests

### Run All Target Tests
```bash
python run_target_tests.py
```

### Run Individual Test Suites
```bash
# Unit tests only
python -m unittest discover tests/unit -v

# Integration tests only
python tests/integration/test_target_pipeline.py

# Specific test file
python tests/unit/test_ensure_direction_1d.py
```

### Expected Output
- **41 total tests**: 31 unit tests + 10 integration tests
- **All tests should pass** for a healthy implementation
- **Logging output**: Shows target distributions and validation messages
- **Performance metrics**: Large dataset tests complete quickly

## Test Data Patterns

The tests use various data patterns to ensure robustness:

1. **Realistic Random Walks**: Simulated stock price movements
2. **Trending Data**: Consistent upward/downward movements
3. **Sideways Markets**: Range-bound price action
4. **High Volatility**: Large price swings
5. **Low Volatility**: Minimal price movements
6. **Edge Cases**: Constant prices, extreme movements, single data points

## Validation Criteria

### Target Distribution Validation
- **FLAT class percentage**: Should be 15-50% for healthy neutral bands
- **Class balance**: All three classes (-1, 0, 1) should be present
- **Value constraints**: Only -1, 0, 1 allowed in raw targets
- **Encoding validation**: Proper mapping to 0, 1, 2 for neural networks

### Data Integrity Checks
- **No data leakage**: Forward returns calculated correctly
- **Consistent results**: Same input produces same output
- **Error propagation**: Meaningful error messages with context
- **Performance**: Handles large datasets efficiently

## Troubleshooting

### Common Test Failures
1. **Missing close column**: Ensure test data has 'close' column
2. **Invalid target values**: Check for values outside {-1, 0, 1}
3. **Missing target classes**: Verify all classes present in combined data
4. **FLAT class percentage**: Adjust neutral band if FLAT class too low/high

### Test Environment
- **Python version**: Compatible with Python 3.7+
- **Dependencies**: pandas, numpy (no external test frameworks required)
- **Data requirements**: Synthetic data generated within tests
- **Performance**: Tests complete in under 1 second on modern hardware

## Maintenance

### Adding New Tests
1. Follow existing naming conventions (`test_*`)
2. Use descriptive test method names
3. Include docstrings explaining test purpose
4. Test both success and failure cases
5. Use realistic data patterns

### Updating Tests
- Update tests when requirements change
- Maintain backward compatibility where possible
- Document breaking changes in test behavior
- Ensure all edge cases remain covered

This test suite provides comprehensive validation of the target labeling surgical fixes, ensuring the system is robust, leak-free, and production-ready.