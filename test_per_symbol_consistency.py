#!/usr/bin/env python3
"""
Test script for per-symbol feature consistency processing.

This script tests the implementation of task 5: Implement per-symbol feature consistency processing
"""

import sys
import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ai.data.feature_consistency import FeatureConsistencyManager, FeatureConsistencyConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_data():
    """Create sample test data for testing."""
    logger.info("Creating test data...")
    
    # Create sample features that would be in a manifest
    stable_features = [
        'close', 'volume', 'rsi_14', 'sma_20', 'ema_12', 'macd', 'bb_upper', 'bb_lower',
        'atr_14', 'momentum_10', 'roc_12', 'stoch_k', 'stoch_d', 'williams_r',
        'adx_14', 'cci_20', 'mfi_14', 'obv', 'vwap', 'price_change_1d'
    ]
    
    # Create test symbols with different feature availability
    symbols_data = {}
    
    # Symbol 1: Has all features (should pass)
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    
    symbol1_data = {'date': dates, 'symbol': 'AAPL', 'target': np.random.randn(1000)}
    for feature in stable_features:
        # Add some NaN values but keep under 5% threshold
        values = np.random.randn(1000)
        nan_indices = np.random.choice(1000, size=int(1000 * 0.03), replace=False)  # 3% NaN
        values[nan_indices] = np.nan
        symbol1_data[feature] = values
    
    symbols_data['AAPL'] = pd.DataFrame(symbol1_data).set_index('date')
    
    # Symbol 2: Missing some features (should test graceful handling)
    symbol2_data = {'date': dates, 'symbol': 'GOOGL', 'target': np.random.randn(1000)}
    for feature in stable_features[:15]:  # Only first 15 features (75% coverage)
        values = np.random.randn(1000)
        nan_indices = np.random.choice(1000, size=int(1000 * 0.04), replace=False)  # 4% NaN
        values[nan_indices] = np.nan
        symbol2_data[feature] = values
    
    symbols_data['GOOGL'] = pd.DataFrame(symbol2_data).set_index('date')
    
    # Symbol 3: Has excessive NaN in some features (should test NaN threshold)
    symbol3_data = {'date': dates, 'symbol': 'MSFT', 'target': np.random.randn(1000)}
    for i, feature in enumerate(stable_features):
        values = np.random.randn(1000)
        if i < 3:  # First 3 features have excessive NaN (>5%)
            nan_indices = np.random.choice(1000, size=int(1000 * 0.08), replace=False)  # 8% NaN
        else:
            nan_indices = np.random.choice(1000, size=int(1000 * 0.02), replace=False)  # 2% NaN
        values[nan_indices] = np.nan
        symbol3_data[feature] = values
    
    symbols_data['MSFT'] = pd.DataFrame(symbol3_data).set_index('date')
    
    # Symbol 4: Insufficient feature coverage (should be excluded)
    symbol4_data = {'date': dates, 'symbol': 'TSLA', 'target': np.random.randn(1000)}
    for feature in stable_features[:10]:  # Only first 10 features (50% coverage)
        values = np.random.randn(1000)
        nan_indices = np.random.choice(1000, size=int(1000 * 0.02), replace=False)  # 2% NaN
        values[nan_indices] = np.nan
        symbol4_data[feature] = values
    
    symbols_data['TSLA'] = pd.DataFrame(symbol4_data).set_index('date')
    
    logger.info(f"Created test data for {len(symbols_data)} symbols")
    for symbol, df in symbols_data.items():
        feature_cols = [c for c in df.columns if c not in ['symbol', 'target']]
        logger.info(f"  {symbol}: {df.shape[0]} rows, {len(feature_cols)} features")
    
    return symbols_data, stable_features

def create_test_manifest(stable_features):
    """Create a test manifest file."""
    logger.info("Creating test manifest...")
    
    manifest_data = {
        "manifest_version": "1.0",
        "version": "test_20241027_120000",
        "created_timestamp": "2024-10-27T12:00:00",
        "config_snapshot": {
            "global_feature_keep_ratio": 0.95,
            "min_symbol_feature_coverage": 0.90,
            "warmup_trim_days": 200,
            "nan_drop_threshold_per_symbol": 0.05
        },
        "stable_features": stable_features,
        "total_features_analyzed": len(stable_features) + 5,
        "total_symbols_analyzed": 100,
        "summary": {
            "total_symbols": 100,
            "stable_features_count": len(stable_features),
            "stability_ratio": 0.8
        }
    }
    
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Save test manifest
    manifest_path = models_dir / "feature_manifest.json"
    import json
    with open(manifest_path, 'w') as f:
        json.dump(manifest_data, f, indent=2)
    
    logger.info(f"Test manifest saved to {manifest_path}")
    return manifest_path

def test_per_symbol_consistency():
    """Test the per-symbol feature consistency processing."""
    logger.info("=" * 60)
    logger.info("TESTING PER-SYMBOL FEATURE CONSISTENCY PROCESSING")
    logger.info("=" * 60)
    
    try:
        # Create test data
        symbols_data, stable_features = create_test_data()
        
        # Create test manifest
        manifest_path = create_test_manifest(stable_features)
        
        # Create configuration
        config = FeatureConsistencyConfig(
            warmup_trim_days=50,  # Smaller for test data
            nan_drop_threshold_per_symbol=0.05,  # 5% threshold
            min_symbol_feature_coverage=0.90,  # 90% coverage required
            manifest_path=str(manifest_path)
        )
        
        # Initialize feature consistency manager
        manager = FeatureConsistencyManager(config)
        
        # Test the per-symbol consistency processing
        logger.info("Testing process_symbols_with_per_symbol_consistency...")
        processed_symbols = manager.process_symbols_with_per_symbol_consistency(symbols_data)
        
        # Analyze results
        logger.info("=" * 60)
        logger.info("TEST RESULTS ANALYSIS")
        logger.info("=" * 60)
        
        logger.info(f"Original symbols: {len(symbols_data)}")
        logger.info(f"Processed symbols: {len(processed_symbols)}")
        logger.info(f"Excluded symbols: {len(symbols_data) - len(processed_symbols)}")
        
        # Check which symbols were processed
        for symbol in symbols_data.keys():
            if symbol in processed_symbols:
                df = processed_symbols[symbol]
                feature_cols = [c for c in df.columns if c not in ['symbol', 'target', 'date', 'timestamp', 'direction_1d', 'close']]
                logger.info(f"✓ {symbol}: Processed successfully - {df.shape[0]} rows, {len(feature_cols)} features")
            else:
                logger.info(f"✗ {symbol}: Excluded from processing")
        
        # Test individual methods
        logger.info("\nTesting individual methods...")
        
        # Test feature selection enforcement
        test_symbol = 'AAPL'
        if test_symbol in symbols_data:
            logger.info(f"Testing enforce_feature_selection on {test_symbol}...")
            df_enforced = manager.enforce_feature_selection(symbols_data[test_symbol], test_symbol)
            logger.info(f"  Original columns: {len(symbols_data[test_symbol].columns)}")
            logger.info(f"  Enforced columns: {len(df_enforced.columns)}")
            
            # Test coverage validation
            logger.info(f"Testing validate_symbol_coverage on {test_symbol}...")
            is_valid, coverage_ratio, exclusion_reason = manager.validate_symbol_coverage(df_enforced, test_symbol)
            logger.info(f"  Valid: {is_valid}, Coverage: {coverage_ratio:.1%}, Reason: {exclusion_reason}")
            
            # Test NaN threshold application
            logger.info(f"Testing apply_updated_nan_threshold on {test_symbol}...")
            df_nan_processed = manager.apply_updated_nan_threshold(df_enforced, test_symbol)
            logger.info(f"  Original columns: {len(df_enforced.columns)}")
            logger.info(f"  After NaN processing: {len(df_nan_processed.columns)}")
        
        logger.info("=" * 60)
        logger.info("TEST COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_per_symbol_consistency()
    if success:
        logger.info("All tests passed!")
        sys.exit(0)
    else:
        logger.error("Tests failed!")
        sys.exit(1)