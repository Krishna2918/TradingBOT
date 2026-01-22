#!/usr/bin/env python3
"""
Test script for Task 7 integration - Verify the complete feature consistency system integration.

This script tests:
1. Tensor shape validation system
2. Comprehensive logging throughout pipeline
3. Integration with existing training pipeline
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path

# Import the feature consistency system
from src.ai.data.feature_consistency import FeatureConsistencyManager, FeatureConsistencyConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_data():
    """Create sample test data for validation"""
    logger.info("Creating test data...")
    
    # Create sample data for 3 symbols
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    test_data = {}
    
    np.random.seed(42)  # For reproducible results
    
    for symbol in symbols:
        # Create 1000 rows of sample data
        n_rows = 1000
        
        # Essential columns
        dates = pd.date_range('2020-01-01', periods=n_rows, freq='D')
        
        # Create base features (consistent across symbols)
        base_features = [
            'close', 'volume', 'rsi_14', 'sma_20', 'ema_12', 'macd', 'bb_upper', 'bb_lower',
            'atr_14', 'momentum_10', 'roc_12', 'stoch_k', 'stoch_d', 'williams_r'
        ]
        
        data = {
            'symbol': symbol,
            'date': dates,
            'target': np.random.choice([0, 1, 2], n_rows),  # 0=down, 1=neutral, 2=up
            'direction_1d': np.random.choice([0, 1], n_rows),  # Binary direction
        }
        
        # Add base features with some random variation
        for feature in base_features:
            data[feature] = np.random.randn(n_rows) * 10 + 100
            
            # Add some NaN values to test missingness handling
            if np.random.random() < 0.3:  # 30% chance of having some NaN values
                nan_indices = np.random.choice(n_rows, size=int(n_rows * 0.02), replace=False)
                data[feature][nan_indices] = np.nan
        
        # Add symbol-specific features to test feature consistency
        if symbol == 'AAPL':
            data['apple_specific'] = np.random.randn(n_rows)
        elif symbol == 'GOOGL':
            data['google_specific'] = np.random.randn(n_rows)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        df.set_index('date', inplace=True)
        
        test_data[symbol] = df
        logger.info(f"Created test data for {symbol}: {df.shape}")
    
    return test_data

def test_tensor_shape_validation():
    """Test the tensor shape validation system (Task 7.1)"""
    logger.info("=" * 60)
    logger.info("TESTING TENSOR SHAPE VALIDATION SYSTEM")
    logger.info("=" * 60)
    
    # Create configuration
    config = FeatureConsistencyConfig(
        warmup_trim_days=50,  # Smaller for test data
        nan_drop_threshold_per_symbol=0.05,
        global_feature_keep_ratio=0.95,
        min_symbol_feature_coverage=0.90,
        use_missingness_mask=True,
        detailed_logging=True
    )
    
    # Initialize manager
    manager = FeatureConsistencyManager(config)
    
    # Create test data
    test_data = create_test_data()
    
    # Process with complete pipeline
    logger.info("Processing test data with complete pipeline...")
    processed_symbols = manager.process_symbols_with_complete_pipeline(test_data)
    
    if not processed_symbols:
        logger.error("No symbols processed - test failed")
        return False
    
    # Test tensor shape validation
    logger.info("Testing tensor shape validation...")
    is_valid, validation_report = manager.validate_tensor_shapes(processed_symbols)
    
    if is_valid:
        logger.info("✓ Tensor shape validation PASSED")
        logger.info(f"  Validated {len(processed_symbols)} symbols")
        logger.info(f"  Uniform column count: {validation_report['shape_consistency'].get('uniform_column_count', 'N/A')}")
        logger.info(f"  Uniform feature count: {validation_report['feature_consistency'].get('uniform_base_feature_count', 'N/A')}")
        return True
    else:
        logger.error("✗ Tensor shape validation FAILED")
        logger.error(f"  Errors: {len(validation_report['errors'])}")
        for error in validation_report['errors'][:3]:
            logger.error(f"    - {error}")
        return False

def test_comprehensive_logging():
    """Test comprehensive logging throughout pipeline (Task 7.2)"""
    logger.info("=" * 60)
    logger.info("TESTING COMPREHENSIVE LOGGING SYSTEM")
    logger.info("=" * 60)
    
    # Create configuration
    config = FeatureConsistencyConfig(
        warmup_trim_days=50,
        detailed_logging=True  # Enable detailed logging
    )
    
    # Initialize manager
    manager = FeatureConsistencyManager(config)
    
    # Create test data
    test_data = create_test_data()
    
    # Process with complete pipeline (this should generate comprehensive logs)
    processed_symbols = manager.process_symbols_with_complete_pipeline(test_data)
    
    if not processed_symbols:
        logger.error("No symbols processed - logging test failed")
        return False
    
    # Test comprehensive pipeline summary logging
    logger.info("Testing comprehensive pipeline summary logging...")
    
    # Create some excluded symbols for testing
    excluded_symbols = [
        ('TEST1', 0.85, 'insufficient_coverage_0.85_below_0.90'),
        ('TEST2', 0.0, 'processing_error_test')
    ]
    
    # Test comprehensive logging
    manager.log_comprehensive_pipeline_summary(processed_symbols, excluded_symbols)
    
    # Test feature drift detection
    if manager.global_analysis_result:
        logger.info("Testing feature drift detection...")
        drift_analysis = manager.detect_and_log_feature_drift(
            manager.global_analysis_result.stable_features
        )
        
        if drift_analysis['analysis_result'] in ['completed', 'no_previous_manifest']:
            logger.info("✓ Feature drift detection completed")
        else:
            logger.warning("⚠ Feature drift detection had issues")
    
    # Test detailed exclusion logging
    logger.info("Testing detailed exclusion logging...")
    manager.log_detailed_exclusion_decisions(excluded_symbols)
    
    logger.info("✓ Comprehensive logging system tested successfully")
    return True

def test_pipeline_integration():
    """Test integration with existing training pipeline (Task 7.3)"""
    logger.info("=" * 60)
    logger.info("TESTING PIPELINE INTEGRATION")
    logger.info("=" * 60)
    
    # Test that the system can be imported and used as expected
    try:
        from src.ai.data.feature_consistency import FeatureConsistencyManager, FeatureConsistencyConfig
        logger.info("✓ Feature consistency system imports successfully")
    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        return False
    
    # Test configuration loading
    try:
        config = FeatureConsistencyConfig()
        manager = FeatureConsistencyManager(config)
        logger.info("✓ Feature consistency system initializes successfully")
    except Exception as e:
        logger.error(f"✗ Initialization failed: {e}")
        return False
    
    # Test that the system produces consistent output format
    test_data = create_test_data()
    
    try:
        processed_symbols = manager.process_symbols_with_complete_pipeline(test_data)
        
        if not processed_symbols:
            logger.error("✗ Pipeline produced no output")
            return False
        
        # Verify output format is compatible with existing training code
        for symbol, df in processed_symbols.items():
            # Check required columns exist
            required_cols = ['target', 'direction_1d']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.error(f"✗ Missing required columns in {symbol}: {missing_cols}")
                return False
            
            # Check data types
            if not pd.api.types.is_numeric_dtype(df['target']):
                logger.error(f"✗ Target column is not numeric in {symbol}")
                return False
            
            # Check for NaN values in feature columns
            essential_columns = {'symbol', 'target', 'date', 'timestamp', 'direction_1d', 'close'}
            feature_cols = [col for col in df.columns if col not in essential_columns and not col.endswith('_isnan')]
            
            nan_counts = df[feature_cols].isnull().sum().sum()
            if nan_counts > 0:
                logger.warning(f"⚠ Found {nan_counts} NaN values in feature columns for {symbol}")
        
        logger.info("✓ Pipeline integration test passed")
        logger.info(f"  Processed {len(processed_symbols)} symbols successfully")
        logger.info(f"  Output format is compatible with existing training code")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Pipeline integration test failed: {e}")
        return False

def main():
    """Run all integration tests for Task 7"""
    logger.info("=" * 80)
    logger.info("TASK 7 INTEGRATION TESTS")
    logger.info("=" * 80)
    
    test_results = {}
    
    # Test 7.1: Tensor shape validation system
    logger.info("\n" + "=" * 40)
    logger.info("TEST 7.1: TENSOR SHAPE VALIDATION")
    logger.info("=" * 40)
    test_results['7.1'] = test_tensor_shape_validation()
    
    # Test 7.2: Comprehensive logging
    logger.info("\n" + "=" * 40)
    logger.info("TEST 7.2: COMPREHENSIVE LOGGING")
    logger.info("=" * 40)
    test_results['7.2'] = test_comprehensive_logging()
    
    # Test 7.3: Pipeline integration
    logger.info("\n" + "=" * 40)
    logger.info("TEST 7.3: PIPELINE INTEGRATION")
    logger.info("=" * 40)
    test_results['7.3'] = test_pipeline_integration()
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TASK 7 INTEGRATION TEST RESULTS")
    logger.info("=" * 80)
    
    all_passed = True
    for test_name, result in test_results.items():
        status = "PASSED" if result else "FAILED"
        logger.info(f"  Task {test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        logger.info("\n🎉 ALL TASK 7 INTEGRATION TESTS PASSED!")
        logger.info("The feature consistency system is fully integrated and working correctly.")
    else:
        logger.error("\n❌ SOME TASK 7 INTEGRATION TESTS FAILED!")
        logger.error("Please review the errors above and fix the issues.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)