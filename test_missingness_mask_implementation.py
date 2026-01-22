#!/usr/bin/env python3
"""
Test script for the MissingnessMaskGenerator implementation.

This script tests the core functionality of the missingness mask and imputation system
to ensure it meets the requirements specified in task 6.
"""

import sys
import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ai.data.feature_consistency import (
    FeatureConsistencyConfig, 
    MissingnessMaskGenerator,
    FeatureConsistencyManager
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_data():
    """Create test DataFrame with missing values."""
    np.random.seed(42)
    
    # Create sample data with missing values
    n_rows = 1000
    data = {
        'symbol': ['AAPL'] * n_rows,
        'date': pd.date_range('2023-01-01', periods=n_rows),
        'close': np.random.randn(n_rows) * 10 + 100,
        'target': np.random.randn(n_rows),
        'direction_1d': np.random.choice([0, 1], n_rows),
        
        # Feature columns with various missing patterns
        'rsi_14': np.random.randn(n_rows) * 20 + 50,
        'sma_20': np.random.randn(n_rows) * 5 + 100,
        'ema_12': np.random.randn(n_rows) * 8 + 95,
        'volume': np.random.randint(1000000, 10000000, n_rows),
        'macd': np.random.randn(n_rows) * 2,
    }
    
    df = pd.DataFrame(data)
    
    # Introduce missing values in different patterns
    # RSI: 10% missing randomly
    missing_indices = np.random.choice(n_rows, int(0.1 * n_rows), replace=False)
    df.loc[missing_indices, 'rsi_14'] = np.nan
    
    # SMA: 5% missing in chunks
    chunk_start = 100
    chunk_end = 150
    df.loc[chunk_start:chunk_end, 'sma_20'] = np.nan
    
    # EMA: 15% missing randomly
    missing_indices = np.random.choice(n_rows, int(0.15 * n_rows), replace=False)
    df.loc[missing_indices, 'ema_12'] = np.nan
    
    # Volume: 2% missing
    missing_indices = np.random.choice(n_rows, int(0.02 * n_rows), replace=False)
    df.loc[missing_indices, 'volume'] = np.nan
    
    # MACD: 8% missing
    missing_indices = np.random.choice(n_rows, int(0.08 * n_rows), replace=False)
    df.loc[missing_indices, 'macd'] = np.nan
    
    return df

def test_missingness_mask_creation():
    """Test subtask 6.1: Implement MissingnessMaskGenerator class."""
    logger.info("=" * 60)
    logger.info("Testing Subtask 6.1: MissingnessMaskGenerator class")
    logger.info("=" * 60)
    
    # Test different configurations
    configs_to_test = [
        {"use_missingness_mask": True, "imputation_strategy": "zero", "imputation_value": 0.0},
        {"use_missingness_mask": True, "imputation_strategy": "mean", "imputation_value": 0.0},
        {"use_missingness_mask": True, "imputation_strategy": "median", "imputation_value": 0.0},
        {"use_missingness_mask": False, "imputation_strategy": "zero", "imputation_value": -999.0},
    ]
    
    test_df = create_test_data()
    logger.info(f"Created test data with shape: {test_df.shape}")
    
    # Log initial missing value counts
    feature_cols = ['rsi_14', 'sma_20', 'ema_12', 'volume', 'macd']
    logger.info("Initial missing value counts:")
    for col in feature_cols:
        missing_count = test_df[col].isna().sum()
        logger.info(f"  {col}: {missing_count} ({missing_count/len(test_df):.1%})")
    
    for i, config_dict in enumerate(configs_to_test, 1):
        logger.info(f"\n--- Test Configuration {i}: {config_dict} ---")
        
        # Create config and generator
        config = FeatureConsistencyConfig(**config_dict)
        generator = MissingnessMaskGenerator(config)
        
        # Test create_missingness_masks
        df_with_masks = generator.create_missingness_masks(test_df.copy(), feature_cols)
        
        # Verify masks were created correctly
        if config.use_missingness_mask:
            expected_mask_cols = [f"{col}_isnan" for col in feature_cols]
            actual_mask_cols = [col for col in df_with_masks.columns if col.endswith('_isnan')]
            
            assert len(actual_mask_cols) == len(expected_mask_cols), f"Expected {len(expected_mask_cols)} mask columns, got {len(actual_mask_cols)}"
            logger.info(f"✓ Created {len(actual_mask_cols)} missingness mask columns")
            
            # Verify mask values are correct
            for col in feature_cols:
                mask_col = f"{col}_isnan"
                original_missing = test_df[col].isna()
                mask_values = df_with_masks[mask_col].astype(bool)
                assert (original_missing == mask_values).all(), f"Mask for {col} doesn't match original missing values"
            
            logger.info("✓ All mask values are correct")
        else:
            mask_cols = [col for col in df_with_masks.columns if col.endswith('_isnan')]
            assert len(mask_cols) == 0, f"Expected no mask columns when disabled, got {len(mask_cols)}"
            logger.info("✓ No mask columns created when disabled")
        
        # Test apply_final_imputation
        df_imputed = generator.apply_final_imputation(df_with_masks, feature_cols)
        
        # Verify no missing values remain in feature columns
        for col in feature_cols:
            remaining_missing = df_imputed[col].isna().sum()
            assert remaining_missing == 0, f"Column {col} still has {remaining_missing} missing values after imputation"
        
        logger.info("✓ All missing values successfully imputed")
        
        # Test complete workflow
        df_complete = generator.process_symbol_with_missingness_handling(test_df.copy(), "TEST", feature_cols)
        
        # Verify complete workflow
        for col in feature_cols:
            remaining_missing = df_complete[col].isna().sum()
            assert remaining_missing == 0, f"Column {col} still has {remaining_missing} missing values after complete workflow"
        
        logger.info("✓ Complete missingness handling workflow successful")
    
    logger.info("\n✅ Subtask 6.1 tests passed!")

def test_pipeline_integration():
    """Test subtask 6.2: Integrate missingness masks into processing pipeline."""
    logger.info("=" * 60)
    logger.info("Testing Subtask 6.2: Pipeline integration")
    logger.info("=" * 60)
    
    # Create test data for multiple symbols
    test_symbols = {}
    for symbol in ['AAPL', 'GOOGL', 'MSFT']:
        df = create_test_data()
        df['symbol'] = symbol
        test_symbols[symbol] = df
    
    logger.info(f"Created test data for {len(test_symbols)} symbols")
    
    # Test with FeatureConsistencyManager
    config = FeatureConsistencyConfig(
        use_missingness_mask=True,
        imputation_strategy="mean",
        warmup_trim_days=50,  # Smaller for testing
        global_feature_keep_ratio=0.8,  # More lenient for testing
        min_symbol_feature_coverage=0.7
    )
    
    manager = FeatureConsistencyManager(config)
    
    # Test the complete pipeline
    processed_symbols = manager.process_symbols_with_complete_pipeline(test_symbols)
    
    logger.info(f"Pipeline processed {len(processed_symbols)} symbols")
    
    # Verify results
    for symbol, df in processed_symbols.items():
        # Check for missingness mask columns
        mask_cols = [col for col in df.columns if col.endswith('_isnan')]
        logger.info(f"{symbol}: {len(mask_cols)} mask columns created")
        
        # Check for no missing values in feature columns
        essential_columns = {'symbol', 'target', 'date', 'timestamp', 'direction_1d', 'close'}
        feature_cols = [col for col in df.columns if col not in essential_columns and not col.endswith('_isnan')]
        
        total_missing = 0
        for col in feature_cols:
            missing_count = df[col].isna().sum()
            total_missing += missing_count
        
        assert total_missing == 0, f"{symbol}: {total_missing} missing values remain in feature columns"
        logger.info(f"{symbol}: ✓ No missing values in {len(feature_cols)} feature columns")
    
    logger.info("\n✅ Subtask 6.2 tests passed!")

def test_configurable_handling():
    """Test subtask 6.3: Add configurable missingness handling."""
    logger.info("=" * 60)
    logger.info("Testing Subtask 6.3: Configurable missingness handling")
    logger.info("=" * 60)
    
    test_df = create_test_data()
    feature_cols = ['rsi_14', 'sma_20', 'ema_12', 'volume', 'macd']
    
    # Test different imputation strategies
    strategies = ["zero", "mean", "median"]
    
    for strategy in strategies:
        logger.info(f"\n--- Testing strategy: {strategy} ---")
        
        config = FeatureConsistencyConfig(
            use_missingness_mask=True,
            imputation_strategy=strategy,
            imputation_value=42.0  # Custom fallback value
        )
        
        generator = MissingnessMaskGenerator(config)
        
        # Test strategy validation
        assert generator.validate_imputation_strategy(strategy), f"Strategy {strategy} should be valid"
        
        # Test getting available strategies
        available = generator.get_available_imputation_strategies()
        assert strategy in available, f"Strategy {strategy} should be in available strategies"
        
        # Test setting strategy
        generator.set_imputation_strategy(strategy)
        assert generator.config.imputation_strategy == strategy, f"Strategy not set correctly"
        
        # Test imputation summary
        summary = generator.get_imputation_summary(test_df, "TEST")
        assert summary['strategy'] == strategy, f"Summary should show strategy {strategy}"
        assert summary['features_to_process'] == len(feature_cols), f"Should process {len(feature_cols)} features"
        
        logger.info(f"✓ Strategy {strategy} configuration working correctly")
        
        # Test actual imputation
        df_processed = generator.process_symbol_with_missingness_handling(test_df.copy(), "TEST", feature_cols)
        
        # Verify no missing values
        for col in feature_cols:
            missing_count = df_processed[col].isna().sum()
            assert missing_count == 0, f"Column {col} has {missing_count} missing values after {strategy} imputation"
        
        logger.info(f"✓ Strategy {strategy} imputation successful")
    
    # Test invalid strategy
    try:
        invalid_config = FeatureConsistencyConfig(imputation_strategy="invalid")
        invalid_config.validate()
        assert False, "Should have raised ValueError for invalid strategy"
    except ValueError as e:
        logger.info(f"✓ Invalid strategy correctly rejected: {e}")
    
    logger.info("\n✅ Subtask 6.3 tests passed!")

def main():
    """Run all tests."""
    logger.info("Starting MissingnessMaskGenerator implementation tests")
    
    try:
        # Test each subtask
        test_missingness_mask_creation()
        test_pipeline_integration()
        test_configurable_handling()
        
        logger.info("\n" + "=" * 70)
        logger.info("🎉 ALL TESTS PASSED! 🎉")
        logger.info("Task 6: Build missingness mask and imputation system - COMPLETE")
        logger.info("=" * 70)
        
        return True
        
    except Exception as e:
        logger.error(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)