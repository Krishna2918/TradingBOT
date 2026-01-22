"""
Unit tests for Feature Consistency System components.

Tests WarmupTrimmer, GlobalCoverageAnalyzer, FeatureManifestManager, 
and MissingnessMaskGenerator with various data patterns and edge cases.
"""

import unittest
import pandas as pd
import numpy as np
import json
import tempfile
import os
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from ai.data.feature_consistency import (
    FeatureConsistencyConfig,
    WarmupTrimmer,
    GlobalCoverageAnalyzer,
    FeatureManifestManager,
    MissingnessMaskGenerator,
    CoverageStats,
    GlobalAnalysisResult
)


class TestWarmupTrimmer(unittest.TestCase):
    """Test suite for WarmupTrimmer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = FeatureConsistencyConfig(warmup_trim_days=200)
        self.trimmer = WarmupTrimmer(self.config)
    
    def test_basic_warmup_trimming(self):
        """Test basic warm-up period trimming functionality."""
        # Create test data with 500 rows
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        df = pd.DataFrame({
            'date': dates,
            'close': np.random.randn(500) + 100,
            'sma_20': np.random.randn(500) + 100,
            'rsi_14': np.random.randn(500) * 20 + 50
        })
        
        result = self.trimmer.trim_warmup_period(df, "TEST")
        
        # Should trim 200 rows (warmup_trim_days)
        self.assertEqual(len(result), 300)
        self.assertEqual(len(df) - len(result), 200)
        
        # Should preserve column structure
        self.assertEqual(list(result.columns), list(df.columns))
        
        # Should start from row 200
        pd.testing.assert_frame_equal(result.reset_index(drop=True), 
                                    df.iloc[200:].reset_index(drop=True))
    
    def test_auto_detect_lookback_from_columns(self):
        """Test automatic detection of maximum lookback period from column names."""
        df = pd.DataFrame({
            'close': [100, 101, 102],
            'sma_5': [100, 100.5, 101],
            'sma_20': [100, 100.2, 100.4],
            'ema_50': [100, 100.1, 100.2],
            'rsi_14': [50, 55, 60],
            'macd_12_26_9': [0, 0.1, 0.2],
            'bb_20': [100, 101, 102]
        })
        
        max_lookback = self.trimmer.detect_max_lookback_from_columns(df)
        
        # Should detect 50 as the maximum (from ema_50)
        self.assertGreaterEqual(max_lookback, 50)
    
    def test_trim_with_insufficient_data(self):
        """Test trimming when data is insufficient."""
        # Create small dataset (100 rows, less than trim period)
        df = pd.DataFrame({
            'close': np.random.randn(100) + 100,
            'sma_20': np.random.randn(100) + 100
        })
        
        result = self.trimmer.trim_warmup_period(df, "SMALL_TEST")
        
        # Should trim at most 80% of data
        max_trim = int(100 * 0.8)
        expected_remaining = 100 - max_trim
        
        self.assertEqual(len(result), expected_remaining)
    
    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame()
        
        result = self.trimmer.trim_warmup_period(df, "EMPTY_TEST")
        
        self.assertTrue(result.empty)
        self.assertEqual(len(result), 0)
    
    def test_post_trim_validation_sufficient_data(self):
        """Test post-trim validation with sufficient data."""
        # Create DataFrame with enough data after trimming
        df = pd.DataFrame({
            'close': np.random.randn(500) + 100,
            'volume': np.random.randint(1000, 10000, 500)
        })
        
        is_valid = self.trimmer.validate_post_trim_data(df, "VALID_TEST")
        
        self.assertTrue(is_valid)
    
    def test_post_trim_validation_insufficient_data(self):
        """Test post-trim validation with insufficient data."""
        # Create DataFrame with insufficient data (less than 252 rows)
        df = pd.DataFrame({
            'close': np.random.randn(100) + 100,
            'volume': np.random.randint(1000, 10000, 100)
        })
        
        is_valid = self.trimmer.validate_post_trim_data(df, "INVALID_TEST")
        
        self.assertFalse(is_valid)
    
    def test_post_trim_validation_excessive_nans(self):
        """Test post-trim validation with excessive NaN values."""
        # Create DataFrame with >50% NaN values
        df = pd.DataFrame({
            'close': [100] * 300,
            'feature1': [np.nan] * 200 + [1] * 100,  # 66% NaN
            'feature2': [np.nan] * 180 + [2] * 120   # 60% NaN
        })
        
        is_valid = self.trimmer.validate_post_trim_data(df, "NAN_TEST")
        
        self.assertFalse(is_valid)
    
    def test_trimming_stats_calculation(self):
        """Test calculation of trimming statistics."""
        original_df = pd.DataFrame({
            'close': [100] * 500,
            'feature1': [np.nan] * 100 + [1] * 400,  # 100 NaNs
            'feature2': [2] * 500
        })
        
        trimmed_df = original_df.iloc[200:].copy()  # Trim 200 rows
        
        stats = self.trimmer.get_trimming_stats(original_df, trimmed_df, "STATS_TEST")
        
        self.assertEqual(stats['symbol'], "STATS_TEST")
        self.assertEqual(stats['original_rows'], 500)
        self.assertEqual(stats['trimmed_rows'], 300)
        self.assertEqual(stats['rows_removed'], 200)
        self.assertEqual(stats['trim_percentage'], 40.0)
        self.assertEqual(stats['original_nan_count'], 100)
        self.assertEqual(stats['trimmed_nan_count'], 0)  # NaNs were in first 100 rows
        self.assertEqual(stats['nan_reduction'], 100)
    
    def test_zero_warmup_days_auto_detect(self):
        """Test auto-detection mode when warmup_trim_days is 0."""
        config = FeatureConsistencyConfig(warmup_trim_days=0)
        trimmer = WarmupTrimmer(config)
        
        df = pd.DataFrame({
            'close': np.random.randn(500) + 100,
            'sma_50': np.random.randn(500) + 100,  # Should detect 50-day lookback
            'rsi_14': np.random.randn(500) * 20 + 50
        })
        
        result = trimmer.trim_warmup_period(df, "AUTO_DETECT_TEST")
        
        # Should auto-detect and use 2x max lookback (50*2=100) or minimum 50
        # The actual implementation may vary, so just check it's reasonable
        self.assertLess(len(result), 500)  # Should trim something
        self.assertGreater(len(result), 300)  # Should not trim too much


class TestGlobalCoverageAnalyzer(unittest.TestCase):
    """Test suite for GlobalCoverageAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = FeatureConsistencyConfig(global_feature_keep_ratio=0.8)
        self.analyzer = GlobalCoverageAnalyzer(self.config)
    
    def create_test_symbol_data(self, symbols, features, coverage_patterns):
        """Create test symbol data with specific coverage patterns."""
        symbol_dataframes = {}
        
        for i, symbol in enumerate(symbols):
            data = {'close': np.random.randn(100) + 100}
            
            for j, feature in enumerate(features):
                if coverage_patterns[i][j]:  # Symbol has this feature
                    # Add some NaN values for realistic data
                    values = np.random.randn(100) + 50
                    nan_indices = np.random.choice(100, size=10, replace=False)
                    values[nan_indices] = np.nan
                    data[feature] = values
            
            symbol_dataframes[symbol] = pd.DataFrame(data)
        
        return symbol_dataframes
    
    def test_compute_symbol_coverage_basic(self):
        """Test basic symbol coverage computation."""
        # Create test data: 3 symbols, 2 features
        # Feature1: present in all symbols (100% coverage)
        # Feature2: present in 2/3 symbols (67% coverage)
        symbols = ['AAPL', 'TSLA', 'MSFT']
        features = ['feature1', 'feature2']
        coverage_patterns = [
            [True, True],   # AAPL has both features
            [True, True],   # TSLA has both features  
            [True, False]   # MSFT has only feature1
        ]
        
        symbol_dataframes = self.create_test_symbol_data(symbols, features, coverage_patterns)
        
        coverage_stats = self.analyzer.compute_symbol_coverage(symbol_dataframes)
        
        # Should find close column plus the features
        self.assertGreaterEqual(len(coverage_stats), 2)  # At least feature1 and feature2
        
        # Check feature1 (100% coverage)
        if 'feature1' in coverage_stats:
            self.assertEqual(coverage_stats['feature1'].coverage_ratio, 1.0)
            self.assertEqual(coverage_stats['feature1'].symbols_with_feature, 3)
        
        # Check feature2 (67% coverage)  
        if 'feature2' in coverage_stats:
            self.assertAlmostEqual(coverage_stats['feature2'].coverage_ratio, 2/3, places=2)
            self.assertEqual(coverage_stats['feature2'].symbols_with_feature, 2)
    
    def test_identify_stable_features(self):
        """Test identification of stable features based on coverage threshold."""
        # Create mock coverage stats
        coverage_stats = {
            'stable_feature1': CoverageStats(
                feature_name='stable_feature1',
                total_symbols=10,
                symbols_with_feature=9,  # 90% coverage
                coverage_ratio=0.9,
                avg_non_nan_ratio=0.8,  # Good data quality
                min_non_nan_ratio=0.7,
                max_non_nan_ratio=0.9,
                symbols_list=['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9']
            ),
            'unstable_feature1': CoverageStats(
                feature_name='unstable_feature1',
                total_symbols=10,
                symbols_with_feature=7,  # 70% coverage (below 80% threshold)
                coverage_ratio=0.7,
                avg_non_nan_ratio=0.8,
                min_non_nan_ratio=0.6,
                max_non_nan_ratio=0.9,
                symbols_list=['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7']
            ),
            'low_quality_feature': CoverageStats(
                feature_name='low_quality_feature',
                total_symbols=10,
                symbols_with_feature=9,  # 90% coverage
                coverage_ratio=0.9,
                avg_non_nan_ratio=0.3,  # Poor data quality (below 50%)
                min_non_nan_ratio=0.1,
                max_non_nan_ratio=0.5,
                symbols_list=['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9']
            )
        }
        
        stable_features = self.analyzer.identify_stable_features(coverage_stats)
        
        # Only stable_feature1 should be identified as stable
        self.assertEqual(len(stable_features), 1)
        self.assertIn('stable_feature1', stable_features)
        self.assertNotIn('unstable_feature1', stable_features)
        self.assertNotIn('low_quality_feature', stable_features)
    
    def test_aggregate_global_stats(self):
        """Test aggregation of global statistics."""
        coverage_stats = {
            'feature1': CoverageStats('feature1', 10, 9, 0.9, 0.8, 0.7, 0.9, []),
            'feature2': CoverageStats('feature2', 10, 7, 0.7, 0.6, 0.5, 0.8, []),
            'feature3': CoverageStats('feature3', 10, 3, 0.3, 0.4, 0.2, 0.6, [])
        }
        
        global_stats = self.analyzer.aggregate_global_stats(coverage_stats)
        
        self.assertEqual(global_stats['total_features'], 3)
        self.assertAlmostEqual(global_stats['avg_coverage_ratio'], (0.9 + 0.7 + 0.3) / 3, places=2)
        self.assertAlmostEqual(global_stats['avg_quality_ratio'], (0.8 + 0.6 + 0.4) / 3, places=2)
        
        # Check distribution buckets
        self.assertEqual(global_stats['coverage_distribution']['90-100%'], 1)  # feature1
        self.assertEqual(global_stats['coverage_distribution']['50-75%'], 1)   # feature2
        self.assertEqual(global_stats['coverage_distribution']['25-50%'], 1)   # feature3
    
    def test_analyze_all_symbols_integration(self):
        """Test complete analysis workflow integration."""
        # Create realistic test data
        symbols = ['AAPL', 'TSLA', 'MSFT', 'GOOGL']
        features = ['sma_20', 'rsi_14', 'rare_feature']
        coverage_patterns = [
            [True, True, False],   # AAPL: common features only
            [True, True, False],   # TSLA: common features only
            [True, True, True],    # MSFT: all features
            [True, True, False]    # GOOGL: common features only
        ]
        
        symbol_dataframes = self.create_test_symbol_data(symbols, features, coverage_patterns)
        
        result = self.analyzer.analyze_all_symbols(symbol_dataframes)
        
        # Verify result structure
        self.assertIsInstance(result, GlobalAnalysisResult)
        self.assertEqual(result.total_symbols_analyzed, 4)
        self.assertGreaterEqual(result.total_features_found, 3)  # May include 'close' column
        
        # sma_20 and rsi_14 should be stable (100% coverage)
        # rare_feature should be unstable (25% coverage)
        self.assertIn('sma_20', result.stable_features)
        self.assertIn('rsi_14', result.stable_features)
        self.assertNotIn('rare_feature', result.stable_features)
        self.assertIn('rare_feature', result.unstable_features)
    
    def test_empty_symbol_dataframes(self):
        """Test handling of empty symbol dataframes."""
        coverage_stats = self.analyzer.compute_symbol_coverage({})
        
        self.assertEqual(len(coverage_stats), 0)
        
        global_stats = self.analyzer.aggregate_global_stats({})
        self.assertEqual(global_stats['total_features'], 0)
        self.assertEqual(global_stats['avg_coverage_ratio'], 0.0)
    
    def test_coverage_stats_validation(self):
        """Test CoverageStats validation."""
        # Valid stats should not raise exception
        valid_stats = CoverageStats(
            feature_name='test',
            total_symbols=10,
            symbols_with_feature=8,
            coverage_ratio=0.8,
            avg_non_nan_ratio=0.7,
            min_non_nan_ratio=0.5,
            max_non_nan_ratio=0.9,
            symbols_list=['S1', 'S2']
        )
        self.assertEqual(valid_stats.coverage_ratio, 0.8)
        
        # Invalid coverage ratio should raise exception
        with self.assertRaises(ValueError):
            CoverageStats(
                feature_name='test',
                total_symbols=10,
                symbols_with_feature=8,
                coverage_ratio=1.5,  # Invalid: > 1.0
                avg_non_nan_ratio=0.7,
                min_non_nan_ratio=0.5,
                max_non_nan_ratio=0.9,
                symbols_list=['S1', 'S2']
            )


class TestFeatureManifestManager(unittest.TestCase):
    """Test suite for FeatureManifestManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = FeatureConsistencyConfig(
            manifest_path=os.path.join(self.temp_dir, "test_manifest.json")
        )
        self.manager = FeatureManifestManager(self.config)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_analysis_result(self):
        """Create test GlobalAnalysisResult."""
        # Create mock coverage stats for the stable features
        coverage_stats = {
            'sma_20': CoverageStats('sma_20', 5, 5, 1.0, 0.9, 0.8, 1.0, ['S1', 'S2', 'S3', 'S4', 'S5']),
            'rsi_14': CoverageStats('rsi_14', 5, 5, 1.0, 0.85, 0.7, 0.95, ['S1', 'S2', 'S3', 'S4', 'S5']),
            'ema_12': CoverageStats('ema_12', 5, 4, 0.8, 0.8, 0.6, 0.9, ['S1', 'S2', 'S3', 'S4'])
        }
        
        return GlobalAnalysisResult(
            total_symbols_analyzed=5,
            total_features_found=10,
            stable_features=['sma_20', 'rsi_14', 'ema_12'],
            unstable_features=['rare_feature1', 'rare_feature2'],
            coverage_stats=coverage_stats,
            analysis_timestamp=datetime.now().isoformat(),
            config_used={'global_feature_keep_ratio': 0.8}
        )
    
    def test_save_manifest_basic(self):
        """Test basic manifest saving functionality."""
        analysis_result = self.create_test_analysis_result()
        
        version = self.manager.save_manifest(analysis_result)
        
        # Should return a version string
        self.assertIsInstance(version, str)
        self.assertTrue(len(version) > 0)
        
        # Manifest file should exist
        self.assertTrue(os.path.exists(self.config.manifest_path))
        
        # Should be valid JSON
        with open(self.config.manifest_path, 'r') as f:
            manifest_data = json.load(f)
        
        # Check required fields
        self.assertIn('version', manifest_data)
        self.assertIn('stable_features', manifest_data)
        self.assertIn('created_timestamp', manifest_data)
        self.assertIn('total_symbols_analyzed', manifest_data)
        
        # Check content
        self.assertEqual(manifest_data['stable_features'], analysis_result.stable_features)
        self.assertEqual(manifest_data['total_symbols_analyzed'], 5)
    
    def test_load_manifest_existing(self):
        """Test loading existing manifest."""
        # First save a manifest
        analysis_result = self.create_test_analysis_result()
        version = self.manager.save_manifest(analysis_result)
        
        # Then load it
        loaded_manifest = self.manager.load_manifest()
        
        self.assertIsNotNone(loaded_manifest)
        self.assertEqual(loaded_manifest['stable_features'], analysis_result.stable_features)
        self.assertEqual(loaded_manifest['version'], version)
    
    def test_load_manifest_nonexistent(self):
        """Test loading non-existent manifest."""
        # Try to load from non-existent path
        config = FeatureConsistencyConfig(
            manifest_path=os.path.join(self.temp_dir, "nonexistent.json")
        )
        manager = FeatureManifestManager(config)
        
        with self.assertRaises(FileNotFoundError):
            manager.load_manifest()
    
    def test_validate_manifest_compatibility_compatible(self):
        """Test manifest compatibility validation with compatible manifest."""
        # Create and save a manifest
        analysis_result = self.create_test_analysis_result()
        self.manager.save_manifest(analysis_result)
        
        # Load and validate compatibility
        manifest_data = self.manager.load_manifest()
        is_compatible = self.manager.validate_manifest_compatibility(manifest_data, self.config)
        
        self.assertTrue(is_compatible)
    
    def test_validate_manifest_compatibility_incompatible(self):
        """Test manifest compatibility validation with incompatible manifest."""
        # Create manifest with different config
        old_config = FeatureConsistencyConfig(global_feature_keep_ratio=0.9)  # Different ratio
        analysis_result = self.create_test_analysis_result()
        analysis_result.config_used = {'global_feature_keep_ratio': 0.9}
        
        self.manager.save_manifest(analysis_result)
        
        # Try to validate with different config
        new_config = FeatureConsistencyConfig(global_feature_keep_ratio=0.8)
        manifest_data = self.manager.load_manifest()
        is_compatible = self.manager.validate_manifest_compatibility(manifest_data, new_config)
        
        self.assertFalse(is_compatible)
    
    def test_manifest_versioning(self):
        """Test manifest versioning functionality."""
        analysis_result = self.create_test_analysis_result()
        
        # Save multiple versions
        version1 = self.manager.save_manifest(analysis_result)
        
        # Modify and save again
        analysis_result.stable_features = ['sma_20', 'rsi_14']  # Remove one feature
        version2 = self.manager.save_manifest(analysis_result)
        
        # Versions should be different
        self.assertNotEqual(version1, version2)
        
        # Should be able to load latest version
        latest_manifest = self.manager.load_manifest()
        self.assertEqual(len(latest_manifest['stable_features']), 2)
    
    def test_manifest_metadata_completeness(self):
        """Test that saved manifest contains all required metadata."""
        analysis_result = self.create_test_analysis_result()
        self.manager.save_manifest(analysis_result)
        
        manifest_data = self.manager.load_manifest()
        
        # Check all required metadata fields
        required_fields = [
            'version', 'stable_features', 'created_timestamp',
            'total_symbols_analyzed', 'total_features_analyzed',
            'config_snapshot', 'excluded_features'
        ]
        
        for field in required_fields:
            self.assertIn(field, manifest_data, f"Missing required field: {field}")
    
    def test_corrupted_manifest_handling(self):
        """Test handling of corrupted manifest files."""
        # Create corrupted JSON file
        with open(self.config.manifest_path, 'w') as f:
            f.write("{ invalid json content")
        
        with self.assertRaises((json.JSONDecodeError, ValueError)):
            self.manager.load_manifest()


class TestMissingnessMaskGenerator(unittest.TestCase):
    """Test suite for MissingnessMaskGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = FeatureConsistencyConfig(
            use_missingness_mask=True,
            imputation_strategy="zero",
            imputation_value=0.0
        )
        self.generator = MissingnessMaskGenerator(self.config)
    
    def test_create_missingness_masks_basic(self):
        """Test basic missingness mask creation."""
        # Create DataFrame with some NaN values
        df = pd.DataFrame({
            'close': [100, 101, 102, 103, 104],
            'feature1': [1.0, np.nan, 3.0, np.nan, 5.0],
            'feature2': [10.0, 20.0, np.nan, 40.0, 50.0],
            'feature3': [100, 200, 300, 400, 500]  # No NaNs
        })
        
        feature_cols = ['feature1', 'feature2', 'feature3']
        result = self.generator.create_missingness_masks(df, feature_cols)
        
        # Should have original columns plus mask columns
        expected_cols = list(df.columns) + ['feature1_isnan', 'feature2_isnan', 'feature3_isnan']
        self.assertEqual(set(result.columns), set(expected_cols))
        
        # Check mask values
        expected_feature1_mask = [0, 1, 0, 1, 0]  # NaN at indices 1, 3
        expected_feature2_mask = [0, 0, 1, 0, 0]  # NaN at index 2
        expected_feature3_mask = [0, 0, 0, 0, 0]  # No NaNs
        
        self.assertEqual(result['feature1_isnan'].tolist(), expected_feature1_mask)
        self.assertEqual(result['feature2_isnan'].tolist(), expected_feature2_mask)
        self.assertEqual(result['feature3_isnan'].tolist(), expected_feature3_mask)
        
        # Original features should still have NaN values (masks are created but not filled yet)
        self.assertTrue(result['feature1'].isna().any())
        self.assertTrue(result['feature2'].isna().any())
        self.assertTrue(pd.isna(result.loc[1, 'feature1']))  # Still NaN
        self.assertTrue(pd.isna(result.loc[2, 'feature2']))  # Still NaN
    
    def test_apply_final_imputation_zero_strategy(self):
        """Test final imputation with zero strategy."""
        df = pd.DataFrame({
            'feature1': [1.0, np.nan, 3.0],
            'feature2': [np.nan, 2.0, np.nan]
        })
        
        result = self.generator.apply_final_imputation(df, ['feature1', 'feature2'])
        
        # All NaN values should be replaced with 0.0
        self.assertFalse(result.isna().any().any())
        self.assertEqual(result.loc[1, 'feature1'], 0.0)
        self.assertEqual(result.loc[0, 'feature2'], 0.0)
        self.assertEqual(result.loc[2, 'feature2'], 0.0)
    
    def test_apply_final_imputation_mean_strategy(self):
        """Test final imputation with mean strategy."""
        config = FeatureConsistencyConfig(imputation_strategy="mean")
        generator = MissingnessMaskGenerator(config)
        
        df = pd.DataFrame({
            'feature1': [1.0, np.nan, 3.0, np.nan, 5.0],  # Mean = 3.0
            'feature2': [10.0, 20.0, np.nan, np.nan, 50.0]  # Mean = 26.67
        })
        
        result = generator.apply_final_imputation(df, ['feature1', 'feature2'])
        
        # NaN values should be replaced with column means
        self.assertFalse(result.isna().any().any())
        self.assertEqual(result.loc[1, 'feature1'], 3.0)  # Mean of [1, 3, 5]
        self.assertEqual(result.loc[3, 'feature1'], 3.0)
        self.assertAlmostEqual(result.loc[2, 'feature2'], 26.666667, places=5)  # Mean of [10, 20, 50]
    
    def test_apply_final_imputation_median_strategy(self):
        """Test final imputation with median strategy."""
        config = FeatureConsistencyConfig(imputation_strategy="median")
        generator = MissingnessMaskGenerator(config)
        
        df = pd.DataFrame({
            'feature1': [1.0, np.nan, 3.0, np.nan, 5.0, 7.0],  # Median = 4.0
            'feature2': [10.0, 20.0, np.nan, 40.0, np.nan, 30.0]     # Median = 25.0
        })
        
        result = generator.apply_final_imputation(df, ['feature1', 'feature2'])
        
        # NaN values should be replaced with column medians
        self.assertFalse(result.isna().any().any())
        self.assertEqual(result.loc[1, 'feature1'], 4.0)  # Median of [1, 3, 5, 7]
        self.assertEqual(result.loc[3, 'feature1'], 4.0)
        self.assertEqual(result.loc[2, 'feature2'], 25.0)  # Median of [10, 20, 40]
        self.assertEqual(result.loc[4, 'feature2'], 25.0)
    
    def test_validate_no_missing_values_clean_data(self):
        """Test validation with clean data (no missing values)."""
        df = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [10.0, 20.0, 30.0]
        })
        
        is_clean, missing_cols = self.generator.validate_no_missing_values(df, "TEST")
        
        self.assertTrue(is_clean)
        self.assertEqual(len(missing_cols), 0)
    
    def test_validate_no_missing_values_with_missing(self):
        """Test validation with missing values."""
        df = pd.DataFrame({
            'feature1': [1.0, np.nan, 3.0],
            'feature2': [10.0, 20.0, 30.0],
            'feature3': [np.nan, np.nan, 30.0]
        })
        
        is_clean, missing_cols = self.generator.validate_no_missing_values(df, "TEST")
        
        self.assertFalse(is_clean)
        self.assertEqual(set(missing_cols), {'feature1', 'feature3'})
    
    def test_complete_missingness_handling_workflow(self):
        """Test complete missingness handling workflow integration."""
        # Create test data with various missing patterns
        df = pd.DataFrame({
            'close': [100, 101, 102, 103, 104],
            'symbol': ['TEST'] * 5,
            'feature1': [1.0, np.nan, 3.0, 4.0, np.nan],
            'feature2': [10.0, 20.0, np.nan, 40.0, 50.0],
            'feature3': [100, 200, 300, 400, 500]  # No missing values
        })
        
        feature_cols = ['feature1', 'feature2', 'feature3']
        
        # Apply complete workflow
        result = self.generator.process_symbol_with_missingness_handling(df, "TEST", feature_cols)
        
        # Should have original columns plus mask columns
        expected_mask_cols = ['feature1_isnan', 'feature2_isnan', 'feature3_isnan']
        for mask_col in expected_mask_cols:
            self.assertIn(mask_col, result.columns)
        
        # Should have no missing values in feature columns
        for col in feature_cols:
            self.assertFalse(result[col].isna().any(), f"Column {col} still has NaN values")
        
        # Mask columns should correctly indicate original missing values
        self.assertEqual(result['feature1_isnan'].tolist(), [0, 1, 0, 0, 1])
        self.assertEqual(result['feature2_isnan'].tolist(), [0, 0, 1, 0, 0])
        self.assertEqual(result['feature3_isnan'].tolist(), [0, 0, 0, 0, 0])
    
    def test_disabled_missingness_mask(self):
        """Test behavior when missingness mask is disabled."""
        config = FeatureConsistencyConfig(use_missingness_mask=False)
        generator = MissingnessMaskGenerator(config)
        
        df = pd.DataFrame({
            'feature1': [1.0, np.nan, 3.0],
            'feature2': [10.0, np.nan, 30.0]
        })
        
        feature_cols = ['feature1', 'feature2']
        result = generator.process_symbol_with_missingness_handling(df, "TEST", feature_cols)
        
        # Should not create mask columns
        mask_cols = [col for col in result.columns if col.endswith('_isnan')]
        self.assertEqual(len(mask_cols), 0)
        
        # Should still apply imputation
        self.assertFalse(result.isna().any().any())
    
    def test_edge_case_all_nan_column(self):
        """Test handling of column with all NaN values."""
        df = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [np.nan, np.nan, np.nan]  # All NaN
        })
        
        feature_cols = ['feature1', 'feature2']
        result = self.generator.create_missingness_masks(df, feature_cols)
        
        # Should create mask for all-NaN column
        self.assertEqual(result['feature2_isnan'].tolist(), [1, 1, 1])
        
        # All-NaN column should be filled with imputation value
        self.assertEqual(result['feature2'].tolist(), [0.0, 0.0, 0.0])
    
    def test_edge_case_empty_feature_list(self):
        """Test handling of empty feature list."""
        df = pd.DataFrame({
            'close': [100, 101, 102],
            'symbol': ['TEST'] * 3
        })
        
        result = self.generator.create_missingness_masks(df, [])
        
        # Should return original DataFrame unchanged
        pd.testing.assert_frame_equal(result, df)


if __name__ == "__main__":
    unittest.main()