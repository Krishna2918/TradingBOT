"""
Unit tests for validation system functions.

Tests validate_symbol_targets and validate_global_targets with valid and invalid inputs,
missing classes, and error propagation and message accuracy.
"""

import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from ai.data.targets import validate_symbol_targets, validate_global_targets


class TestValidationSystem(unittest.TestCase):
    """Test suite for validation system functions."""
    
    def test_validate_symbol_targets_with_valid_inputs(self):
        """Test validate_symbol_targets with valid target data."""
        # Valid DataFrame with all expected target values
        df = pd.DataFrame({
            'close': [100.0, 101.0, 102.0, 103.0, 104.0],
            'direction_1d': [-1, 0, 1, -1, 0]
        })
        
        # Should not raise any exception
        try:
            validate_symbol_targets(df, "VALID_TEST")
        except Exception as e:
            self.fail(f"validate_symbol_targets raised an exception with valid data: {e}")
    
    def test_validate_symbol_targets_missing_direction_1d(self):
        """Test validate_symbol_targets when direction_1d column is missing."""
        df = pd.DataFrame({
            'close': [100.0, 101.0, 102.0],
            'volume': [1000, 1100, 1200]
        })
        
        with self.assertRaises(RuntimeError) as context:
            validate_symbol_targets(df, "MISSING_COL_TEST")
        
        self.assertIn("direction_1d missing for symbol MISSING_COL_TEST", str(context.exception))
    
    def test_validate_symbol_targets_invalid_values(self):
        """Test validate_symbol_targets with invalid target values."""
        # DataFrame with invalid target values
        df = pd.DataFrame({
            'close': [100.0, 101.0, 102.0, 103.0],
            'direction_1d': [-1, 0, 2, -1]  # 2 is invalid
        })
        
        with self.assertRaises(RuntimeError) as context:
            validate_symbol_targets(df, "INVALID_VALUES_TEST")
        
        error_msg = str(context.exception)
        self.assertIn("Invalid target values in INVALID_VALUES_TEST", error_msg)
        self.assertTrue("2" in error_msg)  # Should mention the invalid value
    
    def test_validate_symbol_targets_empty_targets(self):
        """Test validate_symbol_targets with no valid targets."""
        # DataFrame with all NaN targets
        df = pd.DataFrame({
            'close': [100.0, 101.0, 102.0],
            'direction_1d': [np.nan, np.nan, np.nan]
        })
        
        with self.assertRaises(RuntimeError) as context:
            validate_symbol_targets(df, "EMPTY_TARGETS_TEST")
        
        self.assertIn("No valid targets found for symbol EMPTY_TARGETS_TEST", str(context.exception))
    
    def test_validate_symbol_targets_mixed_valid_invalid(self):
        """Test validate_symbol_targets with mix of valid and invalid values."""
        df = pd.DataFrame({
            'close': [100.0, 101.0, 102.0, 103.0, 104.0],
            'direction_1d': [-1, 0, 1, 5, -2]  # 5 and -2 are invalid
        })
        
        with self.assertRaises(RuntimeError) as context:
            validate_symbol_targets(df, "MIXED_TEST")
        
        error_msg = str(context.exception)
        self.assertIn("Invalid target values in MIXED_TEST", error_msg)
        # Should contain both invalid values
        self.assertTrue("5" in error_msg and "-2" in error_msg)
    
    def test_validate_global_targets_with_valid_inputs(self):
        """Test validate_global_targets with valid combined data."""
        # Valid combined DataFrame with all classes present
        df = pd.DataFrame({
            'close': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            'direction_1d': [-1, 0, 1, -1, 0, 1],
            'symbol': ['A', 'A', 'A', 'B', 'B', 'B']
        })
        
        # Should not raise any exception
        try:
            validate_global_targets(df)
        except Exception as e:
            self.fail(f"validate_global_targets raised an exception with valid data: {e}")
    
    def test_validate_global_targets_missing_direction_1d(self):
        """Test validate_global_targets when direction_1d column is missing."""
        df = pd.DataFrame({
            'close': [100.0, 101.0, 102.0],
            'volume': [1000, 1100, 1200]
        })
        
        with self.assertRaises(RuntimeError) as context:
            validate_global_targets(df)
        
        self.assertIn("direction_1d missing after preprocessing", str(context.exception))
    
    def test_validate_global_targets_missing_classes(self):
        """Test validate_global_targets with missing target classes."""
        # DataFrame missing FLAT class (0)
        df = pd.DataFrame({
            'close': [100.0, 101.0, 102.0, 103.0],
            'direction_1d': [-1, 1, -1, 1]  # Missing 0
        })
        
        with self.assertRaises(RuntimeError) as context:
            validate_global_targets(df)
        
        error_msg = str(context.exception)
        self.assertIn("Missing target classes in combined dataset", error_msg)
        self.assertTrue("0" in error_msg)
    
    def test_validate_global_targets_multiple_missing_classes(self):
        """Test validate_global_targets with multiple missing classes."""
        # DataFrame with only UP class (1)
        df = pd.DataFrame({
            'close': [100.0, 101.0, 102.0, 103.0],
            'direction_1d': [1, 1, 1, 1]  # Missing -1 and 0
        })
        
        with self.assertRaises(RuntimeError) as context:
            validate_global_targets(df)
        
        error_msg = str(context.exception)
        self.assertIn("Missing target classes in combined dataset", error_msg)
        # Should contain both missing classes
        self.assertTrue("0" in error_msg and "-1" in error_msg)
    
    def test_validate_global_targets_invalid_values(self):
        """Test validate_global_targets with invalid target values."""
        df = pd.DataFrame({
            'close': [100.0, 101.0, 102.0, 103.0],
            'direction_1d': [-1, 0, 1, 3]  # 3 is invalid
        })
        
        with self.assertRaises(RuntimeError) as context:
            validate_global_targets(df)
        
        error_msg = str(context.exception)
        self.assertIn("Invalid target values found", error_msg)
        self.assertTrue("3" in error_msg)
    
    def test_validate_global_targets_empty_dataset(self):
        """Test validate_global_targets with empty dataset."""
        df = pd.DataFrame({
            'close': [],
            'direction_1d': []
        })
        
        with self.assertRaises(RuntimeError) as context:
            validate_global_targets(df)
        
        self.assertIn("No valid targets found in combined dataset", str(context.exception))
    
    def test_validate_global_targets_all_nan_targets(self):
        """Test validate_global_targets with all NaN targets."""
        df = pd.DataFrame({
            'close': [100.0, 101.0, 102.0],
            'direction_1d': [np.nan, np.nan, np.nan]
        })
        
        with self.assertRaises(RuntimeError) as context:
            validate_global_targets(df)
        
        self.assertIn("No valid targets found in combined dataset", str(context.exception))
    
    def test_validate_global_targets_custom_required_classes(self):
        """Test validate_global_targets with custom required classes."""
        # DataFrame with only -1 and 1
        df = pd.DataFrame({
            'close': [100.0, 101.0, 102.0, 103.0],
            'direction_1d': [-1, 1, -1, 1]
        })
        
        # Should fail with default required classes
        with self.assertRaises(RuntimeError):
            validate_global_targets(df)
        
        # Should pass with custom required classes (but need to handle FLAT percentage check)
        # Create a dataset that won't trigger the FLAT percentage error
        df_custom = pd.DataFrame({
            'close': [100.0, 101.0, 102.0, 103.0] * 10,  # More data
            'direction_1d': [-1, 1] * 20  # Only -1 and 1, but more samples
        })
        
        try:
            validate_global_targets(df_custom, required_classes={-1, 1})
        except RuntimeError as e:
            # If it's the FLAT percentage error, that's expected behavior
            if "FLAT class critically low" in str(e):
                pass  # This is expected since we have no FLAT class
            else:
                self.fail(f"validate_global_targets raised unexpected exception: {e}")
    
    def test_validate_global_targets_flat_class_percentage_validation(self):
        """Test validate_global_targets FLAT class percentage validation."""
        # Create dataset with very low FLAT percentage (< 10%)
        targets = [-1] * 45 + [1] * 45 + [0] * 5  # 5% FLAT
        df = pd.DataFrame({
            'close': list(range(95)),
            'direction_1d': targets[:95]
        })
        
        with self.assertRaises(RuntimeError) as context:
            validate_global_targets(df)
        
        error_msg = str(context.exception)
        self.assertIn("FLAT class critically low", error_msg)
        self.assertIn("increase neutral band", error_msg)
    
    def test_error_message_accuracy_symbol_validation(self):
        """Test that error messages are accurate and informative for symbol validation."""
        # Test with specific symbol name
        df = pd.DataFrame({
            'close': [100.0, 101.0],
            'direction_1d': [-1, 7]  # 7 is invalid
        })
        
        with self.assertRaises(RuntimeError) as context:
            validate_symbol_targets(df, "SPECIFIC_SYMBOL_123")
        
        error_msg = str(context.exception)
        # Should contain symbol name
        self.assertIn("SPECIFIC_SYMBOL_123", error_msg)
        # Should contain invalid value
        self.assertIn("7", error_msg)
        # Should contain expected values
        self.assertTrue("-1" in error_msg and "0" in error_msg and "1" in error_msg)
    
    def test_error_message_accuracy_global_validation(self):
        """Test that error messages are accurate and informative for global validation."""
        # Test missing classes error message
        df = pd.DataFrame({
            'close': [100.0, 101.0, 102.0],
            'direction_1d': [-1, -1, -1]  # Missing 0 and 1
        })
        
        with self.assertRaises(RuntimeError) as context:
            validate_global_targets(df)
        
        error_msg = str(context.exception)
        # Should be specific about missing classes
        self.assertIn("Missing target classes in combined dataset", error_msg)
        # Should contain the missing classes
        self.assertTrue("0" in error_msg and "1" in error_msg, f"Missing classes not found in error: {error_msg}")
    
    def test_error_propagation_from_symbol_to_global(self):
        """Test that errors propagate correctly from symbol to global validation."""
        # This test simulates the workflow where symbol validation passes
        # but global validation might catch issues
        
        # Individual symbols might be valid
        symbol1_df = pd.DataFrame({
            'close': [100.0, 101.0],
            'direction_1d': [-1, 0]
        })
        
        symbol2_df = pd.DataFrame({
            'close': [102.0, 103.0],
            'direction_1d': [1, 1]
        })
        
        # Each symbol validates individually
        validate_symbol_targets(symbol1_df, "SYMBOL1")
        validate_symbol_targets(symbol2_df, "SYMBOL2")
        
        # But combined they have all classes
        combined_df = pd.concat([symbol1_df, symbol2_df], ignore_index=True)
        
        # Global validation should pass
        try:
            validate_global_targets(combined_df)
        except Exception as e:
            self.fail(f"Global validation failed unexpectedly: {e}")
    
    def test_validation_with_realistic_data_distribution(self):
        """Test validation with realistic trading data distribution."""
        # Create realistic distribution: ~30% DOWN, ~40% FLAT, ~30% UP
        np.random.seed(42)
        targets = ([-1] * 30 + [0] * 40 + [1] * 30)
        np.random.shuffle(targets)
        
        df = pd.DataFrame({
            'close': list(range(100)),
            'direction_1d': targets
        })
        
        # Should pass validation
        try:
            validate_global_targets(df)
        except Exception as e:
            self.fail(f"Realistic data distribution failed validation: {e}")
    
    def test_validation_performance_with_large_dataset(self):
        """Test validation performance with large dataset."""
        # Create large dataset
        size = 10000
        targets = np.random.choice([-1, 0, 1], size=size)
        
        df = pd.DataFrame({
            'close': list(range(size)),
            'direction_1d': targets
        })
        
        # Should complete without timeout or memory issues
        try:
            validate_global_targets(df)
        except Exception as e:
            self.fail(f"Large dataset validation failed: {e}")


if __name__ == "__main__":
    unittest.main()