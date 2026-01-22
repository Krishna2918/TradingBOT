"""
Integration tests for complete target creation pipeline.

Tests end-to-end flow from raw data to encoded targets, multiple symbols,
various data patterns, and verification of no data leakage.
"""

import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import tempfile
import os

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from ai.data.targets import (
    ensure_direction_1d, 
    validate_symbol_targets, 
    validate_global_targets,
    encode_targets
)


class TestTargetPipeline(unittest.TestCase):
    """Integration test suite for complete target creation pipeline."""
    
    def setUp(self):
        """Set up test data for integration tests."""
        # Create realistic price data for multiple symbols
        np.random.seed(42)  # For reproducible tests
        
        self.symbols = ['AAPL', 'MSFT', 'GOOGL']
        self.symbol_data = {}
        
        for symbol in self.symbols:
            # Generate realistic price series
            prices = [100.0]
            for _ in range(99):
                # Random walk with slight upward bias
                change = np.random.normal(0.001, 0.02)  # 0.1% mean, 2% std
                prices.append(prices[-1] * (1 + change))
            
            # Create DataFrame with additional columns
            self.symbol_data[symbol] = pd.DataFrame({
                'close': prices,
                'volume': np.random.randint(1000, 10000, 100),
                'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
                'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
                'symbol': symbol
            })
    
    def test_end_to_end_single_symbol_pipeline(self):
        """Test complete pipeline for a single symbol."""
        symbol = 'AAPL'
        df = self.symbol_data[symbol].copy()
        
        # Step 1: Create targets
        df_with_targets = ensure_direction_1d(df, neutral_band=0.004, symbol=symbol)
        
        # Verify targets were created
        self.assertIn('direction_1d', df_with_targets.columns)
        
        # Step 2: Validate symbol targets
        validate_symbol_targets(df_with_targets, symbol)
        
        # Step 3: Validate global targets (single symbol case)
        validate_global_targets(df_with_targets)
        
        # Step 4: Encode targets
        targets = df_with_targets['direction_1d'].dropna().values
        encoded_targets = encode_targets(targets)
        
        # Verify encoding
        self.assertTrue(all(t in [0, 1, 2] for t in encoded_targets))
        self.assertEqual(len(encoded_targets), len(targets))
        
        # Verify encoding mapping: -1->0, 0->1, 1->2
        for original, encoded in zip(targets, encoded_targets):
            expected = original + 1
            self.assertEqual(encoded, expected)
    
    def test_end_to_end_multiple_symbols_pipeline(self):
        """Test complete pipeline with multiple symbols."""
        all_data = []
        neutral_band = 0.005
        
        # Step 1: Process each symbol individually
        for symbol in self.symbols:
            df = self.symbol_data[symbol].copy()
            
            # Create targets
            df_with_targets = ensure_direction_1d(df, neutral_band=neutral_band, symbol=symbol)
            
            # Validate individual symbol
            validate_symbol_targets(df_with_targets, symbol)
            
            all_data.append(df_with_targets)
        
        # Step 2: Combine all symbols
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Step 3: Global validation
        validate_global_targets(combined_df)
        
        # Step 4: Encode targets
        targets = combined_df['direction_1d'].dropna().values
        encoded_targets = encode_targets(targets)
        
        # Verify results
        self.assertTrue(len(encoded_targets) > 0)
        self.assertTrue(all(t in [0, 1, 2] for t in encoded_targets))
        
        # Verify all symbols contributed data
        symbols_in_combined = set(combined_df['symbol'].unique())
        self.assertEqual(symbols_in_combined, set(self.symbols))
    
    def test_pipeline_with_various_data_patterns(self):
        """Test pipeline with different data patterns."""
        test_patterns = {
            'trending_up': self._create_trending_data(trend=0.002),
            'trending_down': self._create_trending_data(trend=-0.002),
            'sideways': self._create_sideways_data(),
            'volatile': self._create_volatile_data(),
            'low_volatility': self._create_low_volatility_data()
        }
        
        for pattern_name, df in test_patterns.items():
            with self.subTest(pattern=pattern_name):
                # Run complete pipeline
                df_with_targets = ensure_direction_1d(df, neutral_band=0.004, symbol=pattern_name)
                validate_symbol_targets(df_with_targets, pattern_name)
                
                targets = df_with_targets['direction_1d'].dropna().values
                if len(targets) > 0:
                    encoded_targets = encode_targets(targets)
                    self.assertTrue(all(t in [0, 1, 2] for t in encoded_targets))
    
    def test_no_data_leakage_verification(self):
        """Test that forward returns use only future data."""
        # Create test data with known future information
        df = pd.DataFrame({
            'close': [100.0, 102.0, 98.0, 104.0, 96.0],  # Known price sequence
            'future_info': [1, 1, -1, 1, -1]  # This should NOT influence targets
        })
        
        # Create targets
        df_with_targets = ensure_direction_1d(df, neutral_band=0.01, symbol="LEAKAGE_TEST")
        
        # Manually calculate expected forward returns
        expected_returns = [
            (102.0 / 100.0) - 1.0,  # 0.02 (2%)
            (98.0 / 102.0) - 1.0,   # -0.0392 (-3.92%)
            (104.0 / 98.0) - 1.0,   # 0.0612 (6.12%)
            (96.0 / 104.0) - 1.0,   # -0.0769 (-7.69%)
        ]
        
        # Verify targets match forward returns, not future_info
        targets = df_with_targets['direction_1d'].values[:4]
        for i, (ret, target) in enumerate(zip(expected_returns, targets)):
            if ret > 0.01:
                expected_target = 1
            elif ret < -0.01:
                expected_target = -1
            else:
                expected_target = 0
            
            self.assertEqual(target, expected_target, 
                           f"Row {i}: target {target} doesn't match forward return {ret}")
    
    def test_pipeline_preserves_existing_targets(self):
        """Test that pipeline preserves existing direction_1d columns."""
        # Create data with existing targets
        original_targets = [-1, 0, 1, -1, 0]
        df = pd.DataFrame({
            'close': [100.0, 101.0, 102.0, 103.0, 104.0],
            'direction_1d': original_targets,
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
        
        # Run pipeline
        df_with_targets = ensure_direction_1d(df, neutral_band=0.004, symbol="EXISTING_TEST")
        validate_symbol_targets(df_with_targets, "EXISTING_TEST")
        
        # Verify original targets preserved
        self.assertEqual(df_with_targets['direction_1d'].tolist(), original_targets)
    
    def test_pipeline_error_handling_and_recovery(self):
        """Test pipeline error handling with problematic data."""
        # Test with missing close column
        bad_df = pd.DataFrame({
            'volume': [1000, 1100, 1200],
            'high': [101, 102, 103]
        })
        
        with self.assertRaises(ValueError):
            ensure_direction_1d(bad_df, symbol="BAD_DATA")
        
        # Test with invalid existing targets
        invalid_df = pd.DataFrame({
            'close': [100.0, 101.0, 102.0],
            'direction_1d': [-1, 5, 1]  # 5 is invalid
        })
        
        # Should fail during target creation (validates existing targets)
        with self.assertRaises(ValueError):
            ensure_direction_1d(invalid_df, symbol="INVALID_TEST")
    
    def test_pipeline_with_different_neutral_bands(self):
        """Test pipeline with various neutral band configurations."""
        df = self.symbol_data['AAPL'].copy()
        bands = [0.002, 0.003, 0.004, 0.005, 0.006]
        
        results = {}
        for band in bands:
            df_with_targets = ensure_direction_1d(df.copy(), neutral_band=band, symbol=f"BAND_{band}")
            validate_symbol_targets(df_with_targets, f"BAND_{band}")
            
            targets = df_with_targets['direction_1d'].dropna()
            flat_pct = (targets == 0).sum() / len(targets) * 100
            results[band] = flat_pct
        
        # Verify that wider bands result in more FLAT classifications
        band_list = sorted(results.keys())
        for i in range(len(band_list) - 1):
            self.assertLessEqual(results[band_list[i]], results[band_list[i + 1]],
                               f"FLAT percentage should increase with band width: {results}")
    
    def test_pipeline_consistency_across_runs(self):
        """Test that pipeline produces consistent results across multiple runs."""
        df = self.symbol_data['AAPL'].copy()
        
        # Run pipeline multiple times
        results = []
        for run in range(3):
            df_with_targets = ensure_direction_1d(df.copy(), neutral_band=0.004, symbol=f"CONSISTENCY_{run}")
            validate_symbol_targets(df_with_targets, f"CONSISTENCY_{run}")
            
            targets = df_with_targets['direction_1d'].values
            results.append(targets.tolist())
        
        # All runs should produce identical results
        for i in range(1, len(results)):
            self.assertEqual(results[0], results[i], f"Run {i} produced different results")
    
    def test_large_dataset_pipeline_performance(self):
        """Test pipeline performance with large dataset."""
        # Create large dataset
        size = 5000
        prices = [100.0]
        for _ in range(size - 1):
            change = np.random.normal(0, 0.01)
            prices.append(prices[-1] * (1 + change))
        
        large_df = pd.DataFrame({
            'close': prices,
            'volume': np.random.randint(1000, 10000, size),
            'symbol': 'LARGE_TEST'
        })
        
        # Run pipeline (should complete without timeout)
        df_with_targets = ensure_direction_1d(large_df, neutral_band=0.004, symbol="LARGE_TEST")
        validate_symbol_targets(df_with_targets, "LARGE_TEST")
        validate_global_targets(df_with_targets)
        
        targets = df_with_targets['direction_1d'].dropna().values
        encoded_targets = encode_targets(targets)
        
        # Verify results
        self.assertEqual(len(encoded_targets), len(targets))
        self.assertTrue(all(t in [0, 1, 2] for t in encoded_targets))
    
    def test_pipeline_with_edge_case_data(self):
        """Test pipeline with edge case data patterns."""
        edge_cases = {
            'single_row': pd.DataFrame({'close': [100.0]}),
            'two_rows': pd.DataFrame({'close': [100.0, 101.0]}),
            'constant_prices': pd.DataFrame({'close': [100.0] * 10}),
            'extreme_volatility': pd.DataFrame({'close': [100.0, 200.0, 50.0, 150.0, 25.0]})
        }
        
        for case_name, df in edge_cases.items():
            with self.subTest(case=case_name):
                df_with_targets = ensure_direction_1d(df, neutral_band=0.004, symbol=case_name)
                validate_symbol_targets(df_with_targets, case_name)
                
                if len(df_with_targets) > 0:
                    targets = df_with_targets['direction_1d'].dropna().values
                    if len(targets) > 0:
                        encoded_targets = encode_targets(targets)
                        self.assertTrue(all(t in [0, 1, 2] for t in encoded_targets))
    
    def _create_trending_data(self, trend=0.001, size=50):
        """Create trending price data."""
        prices = [100.0]
        for _ in range(size - 1):
            change = trend + np.random.normal(0, 0.005)
            prices.append(prices[-1] * (1 + change))
        
        return pd.DataFrame({
            'close': prices,
            'volume': np.random.randint(1000, 5000, size)
        })
    
    def _create_sideways_data(self, size=50):
        """Create sideways (range-bound) price data."""
        prices = []
        base_price = 100.0
        for _ in range(size):
            # Oscillate around base price
            change = np.random.normal(0, 0.003)
            price = base_price * (1 + change)
            prices.append(max(95.0, min(105.0, price)))  # Keep in range
        
        return pd.DataFrame({
            'close': prices,
            'volume': np.random.randint(1000, 5000, size)
        })
    
    def _create_volatile_data(self, size=50):
        """Create highly volatile price data."""
        prices = [100.0]
        for _ in range(size - 1):
            change = np.random.normal(0, 0.03)  # High volatility
            prices.append(prices[-1] * (1 + change))
        
        return pd.DataFrame({
            'close': prices,
            'volume': np.random.randint(1000, 5000, size)
        })
    
    def _create_low_volatility_data(self, size=50):
        """Create low volatility price data."""
        prices = [100.0]
        for _ in range(size - 1):
            change = np.random.normal(0, 0.001)  # Very low volatility
            prices.append(prices[-1] * (1 + change))
        
        return pd.DataFrame({
            'close': prices,
            'volume': np.random.randint(1000, 5000, size)
        })


if __name__ == "__main__":
    unittest.main()