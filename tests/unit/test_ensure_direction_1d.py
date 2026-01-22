"""
Unit tests for ensure_direction_1d function.

Tests forward return calculation accuracy, various neutral band values,
preservation of existing direction_1d columns, error handling, and edge cases.
"""

import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from ai.data.targets import ensure_direction_1d


class TestEnsureDirection1d(unittest.TestCase):
    """Test suite for ensure_direction_1d function."""
    
    def test_forward_return_calculation_accuracy(self):
        """Test that forward returns are calculated correctly."""
        # Create test data with known price movements
        df = pd.DataFrame({
            'close': [100.0, 102.0, 99.0, 101.0, 98.0]  # +2%, -2.94%, +2.02%, -2.97%
        })
        
        result = ensure_direction_1d(df, neutral_band=0.01, symbol="TEST")
        
        # Calculate expected forward returns manually
        expected_returns = [
            (102.0 / 100.0) - 1.0,  # 0.02 (2%)
            (99.0 / 102.0) - 1.0,   # -0.0294 (-2.94%)
            (101.0 / 99.0) - 1.0,   # 0.0202 (2.02%)
            (98.0 / 101.0) - 1.0,   # -0.0297 (-2.97%)
            np.nan  # Last row has no forward return
        ]
        
        # Check that direction_1d was created
        self.assertIn('direction_1d', result.columns)
        
        # Check classifications with 1% neutral band
        expected_labels = [1, -1, 1, -1, 0]  # Last row gets 0 due to NaN handling
        
        # Compare non-NaN values
        actual_labels = result['direction_1d'].values
        for i in range(4):  # Skip last row
            self.assertEqual(actual_labels[i], expected_labels[i], f"Row {i}: expected {expected_labels[i]}, got {actual_labels[i]}")
    
    def test_various_neutral_band_values(self):
        """Test different neutral band values (0.002, 0.003, 0.004, 0.005)."""
        # Create test data with specific return patterns
        df = pd.DataFrame({
            'close': [100.0, 100.1, 100.3, 100.6, 101.0]  # 0.1%, 0.2%, 0.3%, 0.4%
        })
        
        test_bands = [0.002, 0.003, 0.004, 0.005]  # 0.2%, 0.3%, 0.4%, 0.5%
        
        for band in test_bands:
            result = ensure_direction_1d(df.copy(), neutral_band=band, symbol=f"TEST_{band}")
            
            # Calculate expected returns
            returns = [
                (100.1 / 100.0) - 1.0,  # 0.001 (0.1%)
                (100.3 / 100.1) - 1.0,  # ~0.002 (0.2%)
                (100.6 / 100.3) - 1.0,  # ~0.003 (0.3%)
                (101.0 / 100.6) - 1.0,  # ~0.004 (0.4%)
            ]
            
            labels = result['direction_1d'].values[:4]  # Skip last row
            
            # Check each return against the band
            for i, ret in enumerate(returns):
                if ret > band:
                    expected = 1
                elif ret < -band:
                    expected = -1
                else:
                    expected = 0
                
                self.assertEqual(labels[i], expected, f"Band {band}, row {i}: expected {expected}, got {labels[i]}")
    
    def test_preservation_of_existing_direction_1d(self):
        """Test that existing direction_1d columns are preserved without modification."""
        # Create DataFrame with existing direction_1d
        original_targets = [-1, 0, 1, -1, 0]
        df = pd.DataFrame({
            'close': [100.0, 101.0, 102.0, 103.0, 104.0],
            'direction_1d': original_targets
        })
        
        result = ensure_direction_1d(df, neutral_band=0.004, symbol="EXISTING_TEST")
        
        # Should preserve original values exactly
        self.assertEqual(result['direction_1d'].tolist(), original_targets)
        
        # DataFrame should be unchanged except possibly for copying
        pd.testing.assert_frame_equal(result, df)
    
    def test_error_handling_missing_close_column(self):
        """Test error handling when close column is missing."""
        # DataFrame without close column
        df = pd.DataFrame({
            'volume': [1000, 1100, 1200],
            'high': [101, 102, 103]
        })
        
        with self.assertRaises(ValueError) as context:
            ensure_direction_1d(df, symbol="NO_CLOSE_TEST")
        self.assertIn("Missing close column 'close' needed to build targets", str(context.exception))
        
        # Test with custom close column name
        df_custom = pd.DataFrame({
            'price': [100.0, 101.0, 102.0],
            'volume': [1000, 1100, 1200]
        })
        
        with self.assertRaises(ValueError) as context:
            ensure_direction_1d(df_custom, symbol="CUSTOM_CLOSE_TEST")
        self.assertIn("Missing close column 'close' needed to build targets", str(context.exception))
        
        # Should work with correct custom column name
        result = ensure_direction_1d(df_custom, close_col='price', symbol="CUSTOM_CLOSE_TEST")
        self.assertIn('direction_1d', result.columns)
    
    def test_edge_case_constant_prices(self):
        """Test edge case with constant prices (no movement)."""
        # All prices the same
        df = pd.DataFrame({
            'close': [100.0, 100.0, 100.0, 100.0, 100.0]
        })
        
        result = ensure_direction_1d(df, neutral_band=0.004, symbol="CONSTANT_TEST")
        
        # All returns should be 0, so all labels should be 0 (FLAT)
        labels = result['direction_1d'].dropna().values
        self.assertTrue(all(label == 0 for label in labels), f"Expected all FLAT (0), got {labels}")
    
    def test_edge_case_extreme_volatility(self):
        """Test edge case with extreme price volatility."""
        # Extreme price movements
        df = pd.DataFrame({
            'close': [100.0, 150.0, 50.0, 200.0, 25.0]  # +50%, -66.7%, +300%, -87.5%
        })
        
        result = ensure_direction_1d(df, neutral_band=0.004, symbol="VOLATILE_TEST")
        
        # Calculate expected returns
        returns = [
            (150.0 / 100.0) - 1.0,  # 0.5 (50%)
            (50.0 / 150.0) - 1.0,   # -0.667 (-66.7%)
            (200.0 / 50.0) - 1.0,   # 3.0 (300%)
            (25.0 / 200.0) - 1.0,   # -0.875 (-87.5%)
        ]
        
        expected_labels = [1, -1, 1, -1]  # All should be outside neutral band
        actual_labels = result['direction_1d'].values[:4]
        
        self.assertEqual(actual_labels.tolist(), expected_labels, f"Expected {expected_labels}, got {actual_labels.tolist()}")
    
    def test_single_row_dataframe(self):
        """Test with single row DataFrame."""
        df = pd.DataFrame({
            'close': [100.0]
        })
        
        result = ensure_direction_1d(df, neutral_band=0.004, symbol="SINGLE_ROW_TEST")
        
        # Should have direction_1d column
        self.assertIn('direction_1d', result.columns)
        
        # Single row should get 0 (no forward return possible)
        self.assertEqual(result['direction_1d'].iloc[0], 0)
    
    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame({
            'close': []
        })
        
        result = ensure_direction_1d(df, neutral_band=0.004, symbol="EMPTY_TEST")
        
        # Should have direction_1d column but empty
        self.assertIn('direction_1d', result.columns)
        self.assertEqual(len(result), 0)
    
    def test_neutral_band_boundary_conditions(self):
        """Test exact boundary conditions for neutral band classification."""
        # Create data with returns exactly at band boundaries
        neutral_band = 0.004
        df = pd.DataFrame({
            'close': [
                100.0,
                100.0 * (1 + neutral_band),      # Exactly +0.4%
                100.0 * (1 + neutral_band) * (1 - neutral_band),  # Exactly -0.4%
                100.0 * (1 + neutral_band) * (1 - neutral_band) * (1 + neutral_band + 0.0001),  # Slightly above
                100.0  # End value
            ]
        })
        
        result = ensure_direction_1d(df, neutral_band=neutral_band, symbol="BOUNDARY_TEST")
        
        labels = result['direction_1d'].values
        
        # First return: exactly +neutral_band should be UP (1)
        self.assertEqual(labels[0], 1, f"Exact positive boundary should be UP, got {labels[0]}")
        
        # Second return: exactly -neutral_band should be DOWN (-1)
        self.assertEqual(labels[1], -1, f"Exact negative boundary should be DOWN, got {labels[1]}")
        
        # Third return: slightly above +neutral_band should be UP (1)
        self.assertEqual(labels[2], 1, f"Above positive boundary should be UP, got {labels[2]}")
    
    def test_nan_handling_in_close_prices(self):
        """Test handling of NaN values in close prices."""
        df = pd.DataFrame({
            'close': [100.0, np.nan, 102.0, 103.0, np.nan]
        })
        
        result = ensure_direction_1d(df, neutral_band=0.004, symbol="NAN_TEST")
        
        # Should have direction_1d column
        self.assertIn('direction_1d', result.columns)
        
        # Check that NaN close prices result in appropriate handling
        # (implementation may vary, but should not crash)
        self.assertEqual(len(result), len(df))
    
    def test_return_type_and_structure(self):
        """Test that function returns correct type and preserves DataFrame structure."""
        df = pd.DataFrame({
            'close': [100.0, 101.0, 102.0],
            'volume': [1000, 1100, 1200],
            'other_col': ['A', 'B', 'C']
        })
        
        result = ensure_direction_1d(df, neutral_band=0.004, symbol="STRUCTURE_TEST")
        
        # Should return DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        
        # Should preserve all original columns
        for col in df.columns:
            self.assertIn(col, result.columns)
        
        # Should add direction_1d column
        self.assertIn('direction_1d', result.columns)
        
        # Should preserve row count
        self.assertEqual(len(result), len(df))
        
        # Should preserve index
        pd.testing.assert_index_equal(result.index, df.index)
    
    def test_different_neutral_band_effects_on_flat_percentage(self):
        """Test that different neutral bands affect FLAT class percentage as expected."""
        # Create data with various small movements
        np.random.seed(42)  # For reproducible results
        prices = [100.0]
        for _ in range(99):
            # Small random movements around ±0.5%
            change = np.random.normal(0, 0.003)  # Mean 0, std 0.3%
            prices.append(prices[-1] * (1 + change))
        
        df = pd.DataFrame({'close': prices})
        
        bands = [0.001, 0.003, 0.005]  # 0.1%, 0.3%, 0.5%
        flat_percentages = []
        
        for band in bands:
            result = ensure_direction_1d(df.copy(), neutral_band=band, symbol=f"FLAT_TEST_{band}")
            flat_count = (result['direction_1d'] == 0).sum()
            flat_pct = (flat_count / len(result)) * 100
            flat_percentages.append(flat_pct)
        
        # Wider bands should result in higher FLAT percentages
        self.assertTrue(flat_percentages[0] < flat_percentages[1] < flat_percentages[2], 
                       f"FLAT percentages should increase with band width: {flat_percentages}")


if __name__ == "__main__":
    unittest.main()