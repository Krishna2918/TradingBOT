#!/usr/bin/env python3
"""
Test core target labeling functionality
"""

import sys
import os
import pandas as pd
import numpy as np

# Add the project path to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'projects', 'TradingBOT'))

from src.ai.data.targets import ensure_direction_1d, validate_symbol_targets, validate_global_targets, encode_targets

def test_core_functionality():
    """Test the complete target labeling pipeline"""
    print("Testing core target labeling functionality...")
    
    # Create test data with realistic price movements
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    # Generate realistic price data with trends
    base_price = 100
    returns = np.random.normal(0, 0.02, 100)  # 2% daily volatility
    prices = [base_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    df = pd.DataFrame({
        'date': dates,
        'close': prices
    })
    
    print(f"Created test data: {len(df)} rows")
    print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    # Test 1: Basic target creation with different neutral bands
    neutral_bands = [0.002, 0.003, 0.004, 0.005]  # 0.2% to 0.5%
    
    for band in neutral_bands:
        print(f"\n--- Testing neutral band ±{band*100:.1f}% ---")
        
        # Create targets
        df_test = ensure_direction_1d(df.copy(), neutral_band=band, symbol="TEST")
        
        # Validate targets
        validate_symbol_targets(df_test, "TEST")
        
        # Check distribution
        targets = df_test['direction_1d'].dropna()
        vals, counts = np.unique(targets, return_counts=True)
        total = len(targets)
        
        for val, count in zip(vals, counts):
            pct = (count / total) * 100
            class_name = {-1: "DOWN", 0: "FLAT", 1: "UP"}.get(val, f"Class_{val}")
            print(f"  {class_name}: {count} ({pct:.1f}%)")
    
    # Test 2: Global validation with multiple symbols
    print(f"\n--- Testing global validation ---")
    
    # Create multiple symbol datasets
    symbols_data = []
    for i, symbol in enumerate(['AAPL', 'MSFT', 'GOOGL']):
        # Slightly different price patterns for each symbol
        symbol_prices = [100 + i*10]
        for ret in returns[1:]:
            symbol_prices.append(symbol_prices[-1] * (1 + ret * (1 + i*0.1)))
        
        symbol_df = pd.DataFrame({
            'date': dates,
            'close': symbol_prices,
            'symbol': symbol
        })
        
        # Create targets
        symbol_df = ensure_direction_1d(symbol_df, neutral_band=0.004, symbol=symbol)
        validate_symbol_targets(symbol_df, symbol)
        symbols_data.append(symbol_df)
    
    # Combine and validate globally
    combined_df = pd.concat(symbols_data, ignore_index=True)
    validate_global_targets(combined_df)
    
    # Test 3: Target encoding
    print(f"\n--- Testing target encoding ---")
    
    raw_targets = combined_df['direction_1d'].values
    encoded_targets = encode_targets(raw_targets)
    
    print(f"Raw targets range: {set(np.unique(raw_targets[~np.isnan(raw_targets)]))}")
    print(f"Encoded targets range: {set(np.unique(encoded_targets[~np.isnan(encoded_targets)]))}")
    
    # Verify encoding: -1->0, 0->1, 1->2
    for raw_val, encoded_val in [(-1, 0), (0, 1), (1, 2)]:
        mask = raw_targets == raw_val
        if np.any(mask):
            assert np.all(encoded_targets[mask] == encoded_val), f"Encoding failed for {raw_val}->{encoded_val}"
    
    print("✅ Target encoding test passed")
    
    # Test 4: Error handling
    print(f"\n--- Testing error handling ---")
    
    # Test missing close column
    df_no_close = df.drop('close', axis=1)
    try:
        ensure_direction_1d(df_no_close, symbol="TEST")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✅ Correctly caught missing close column: {e}")
    
    # Test invalid target values
    df_invalid = df.copy()
    df_invalid['direction_1d'] = [2, 3, -2, 0, 1] * 20  # Invalid values
    
    try:
        validate_symbol_targets(df_invalid, "TEST")
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        print(f"✅ Correctly caught invalid targets: {e}")
    
    print(f"\n🎉 All core functionality tests passed!")
    print(f"✅ Target creation with configurable neutral bands")
    print(f"✅ Per-symbol validation")
    print(f"✅ Global validation")
    print(f"✅ Target encoding")
    print(f"✅ Error handling")
    
    return True

if __name__ == "__main__":
    test_core_functionality()