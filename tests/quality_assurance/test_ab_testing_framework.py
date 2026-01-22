"""
A/B Testing Framework for Target Creation Methods.

This module provides comprehensive A/B testing capabilities to:
- Compare old vs new target creation methods for identical results
- Test different neutral band values and measure impact on model performance
- Validate macro-F1, precision/recall, and PnL metrics
"""

import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import warnings

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from ai.data.targets import ensure_direction_1d, encode_targets

logger = logging.getLogger(__name__)


@dataclass
class ABTestResult:
    """Results from A/B testing comparison."""
    test_name: str
    method_a_name: str
    method_b_name: str
    identical_results: bool
    differences_count: int
    total_samples: int
    method_a_metrics: Dict[str, float]
    method_b_metrics: Dict[str, float]
    neutral_band_a: Optional[float] = None
    neutral_band_b: Optional[float] = None
    performance_difference: Optional[Dict[str, float]] = None


@dataclass
class PerformanceMetrics:
    """Performance metrics for target creation method."""
    macro_f1: float
    precision: float
    recall: float
    class_distribution: Dict[str, float]
    flat_percentage: float
    total_samples: int


class ABTestingFramework:
    """Framework for A/B testing target creation methods."""
    
    def __init__(self, random_seed: int = 42):
        """Initialize A/B testing framework."""
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self.test_results: List[ABTestResult] = []
    
    def compare_target_methods(self, 
                             df: pd.DataFrame,
                             method_a_func: callable,
                             method_b_func: callable,
                             method_a_name: str = "Method A",
                             method_b_name: str = "Method B",
                             **kwargs) -> ABTestResult:
        """
        Compare two target creation methods for identical results.
        
        Args:
            df: Input DataFrame with price data
            method_a_func: First target creation method
            method_b_func: Second target creation method
            method_a_name: Name for first method
            method_b_name: Name for second method
            **kwargs: Additional arguments for methods
            
        Returns:
            ABTestResult with comparison results
        """
        # Apply both methods
        df_a = method_a_func(df.copy(), **kwargs)
        df_b = method_b_func(df.copy(), **kwargs)
        
        # Extract targets
        targets_a = df_a['direction_1d'].values
        targets_b = df_b['direction_1d'].values
        
        # Compare results
        identical = np.array_equal(targets_a, targets_b, equal_nan=True)
        differences = np.sum(targets_a != targets_b)
        total_samples = len(targets_a)
        
        # Calculate metrics for both methods
        metrics_a = self._calculate_metrics(targets_a)
        metrics_b = self._calculate_metrics(targets_b)
        
        result = ABTestResult(
            test_name=f"{method_a_name}_vs_{method_b_name}",
            method_a_name=method_a_name,
            method_b_name=method_b_name,
            identical_results=identical,
            differences_count=differences,
            total_samples=total_samples,
            method_a_metrics=metrics_a,
            method_b_metrics=metrics_b
        )
        
        self.test_results.append(result)
        return result
    
    def test_neutral_band_impact(self, 
                               df: pd.DataFrame,
                               neutral_bands: List[float],
                               symbol: str = "TEST") -> List[ABTestResult]:
        """
        Test different neutral band values and measure impact.
        
        Args:
            df: Input DataFrame with price data
            neutral_bands: List of neutral band values to test
            symbol: Symbol name for logging
            
        Returns:
            List of ABTestResult comparing each band to baseline
        """
        results = []
        baseline_band = neutral_bands[0]
        
        # Create baseline
        baseline_df = ensure_direction_1d(df.copy(), neutral_band=baseline_band, symbol=f"{symbol}_baseline")
        baseline_targets = baseline_df['direction_1d'].values
        baseline_metrics = self._calculate_metrics(baseline_targets)
        
        for band in neutral_bands[1:]:
            # Test band
            test_df = ensure_direction_1d(df.copy(), neutral_band=band, symbol=f"{symbol}_band_{band}")
            test_targets = test_df['direction_1d'].values
            test_metrics = self._calculate_metrics(test_targets)
            
            # Compare
            identical = np.array_equal(baseline_targets, test_targets, equal_nan=True)
            differences = np.sum(baseline_targets != test_targets)
            
            # Calculate performance difference
            perf_diff = {
                'macro_f1_diff': test_metrics['macro_f1'] - baseline_metrics['macro_f1'],
                'precision_diff': test_metrics['precision'] - baseline_metrics['precision'],
                'recall_diff': test_metrics['recall'] - baseline_metrics['recall'],
                'flat_pct_diff': test_metrics['flat_percentage'] - baseline_metrics['flat_percentage']
            }
            
            result = ABTestResult(
                test_name=f"neutral_band_{baseline_band}_vs_{band}",
                method_a_name=f"Band_{baseline_band}",
                method_b_name=f"Band_{band}",
                identical_results=identical,
                differences_count=differences,
                total_samples=len(baseline_targets),
                method_a_metrics=baseline_metrics,
                method_b_metrics=test_metrics,
                neutral_band_a=baseline_band,
                neutral_band_b=band,
                performance_difference=perf_diff
            )
            
            results.append(result)
            self.test_results.append(result)
        
        return results
    
    def validate_macro_f1_precision_recall(self, 
                                         targets_true: np.ndarray,
                                         targets_pred: np.ndarray,
                                         method_name: str = "Method") -> Dict[str, float]:
        """
        Validate macro-F1, precision, and recall metrics.
        
        Args:
            targets_true: True target labels
            targets_pred: Predicted target labels
            method_name: Name of method being validated
            
        Returns:
            Dictionary with calculated metrics
        """
        # Remove NaN values
        mask = ~(np.isnan(targets_true) | np.isnan(targets_pred))
        true_clean = targets_true[mask]
        pred_clean = targets_pred[mask]
        
        if len(true_clean) == 0:
            logger.warning(f"{method_name}: No valid samples for metric calculation")
            return {
                'macro_f1': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'samples': 0
            }
        
        # Calculate metrics
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress sklearn warnings for missing classes
            
            macro_f1 = f1_score(true_clean, pred_clean, average='macro', zero_division=0)
            precision = precision_score(true_clean, pred_clean, average='macro', zero_division=0)
            recall = recall_score(true_clean, pred_clean, average='macro', zero_division=0)
        
        metrics = {
            'macro_f1': macro_f1,
            'precision': precision,
            'recall': recall,
            'samples': len(true_clean)
        }
        
        logger.info(f"{method_name} metrics: F1={macro_f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
        
        return metrics
    
    def simulate_pnl_metrics(self, 
                           targets: np.ndarray,
                           returns: np.ndarray,
                           method_name: str = "Method") -> Dict[str, float]:
        """
        Simulate PnL metrics based on target predictions.
        
        Args:
            targets: Target predictions (-1, 0, 1)
            returns: Actual forward returns
            method_name: Name of method being tested
            
        Returns:
            Dictionary with PnL metrics
        """
        # Remove NaN values
        mask = ~(np.isnan(targets) | np.isnan(returns))
        targets_clean = targets[mask]
        returns_clean = returns[mask]
        
        if len(targets_clean) == 0:
            return {
                'total_pnl': 0.0,
                'sharpe_ratio': 0.0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'trades': 0
            }
        
        # Calculate PnL assuming we trade in direction of prediction
        # Only trade on non-FLAT predictions
        trade_mask = targets_clean != 0
        trade_targets = targets_clean[trade_mask]
        trade_returns = returns_clean[trade_mask]
        
        if len(trade_targets) == 0:
            return {
                'total_pnl': 0.0,
                'sharpe_ratio': 0.0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'trades': 0
            }
        
        # PnL = prediction * actual_return (positive if correct direction)
        pnl_per_trade = trade_targets * trade_returns
        
        total_pnl = np.sum(pnl_per_trade)
        sharpe_ratio = np.mean(pnl_per_trade) / np.std(pnl_per_trade) if np.std(pnl_per_trade) > 0 else 0.0
        
        wins = pnl_per_trade > 0
        win_rate = np.mean(wins) if len(wins) > 0 else 0.0
        avg_win = np.mean(pnl_per_trade[wins]) if np.any(wins) else 0.0
        avg_loss = np.mean(pnl_per_trade[~wins]) if np.any(~wins) else 0.0
        
        metrics = {
            'total_pnl': total_pnl,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'trades': len(trade_targets)
        }
        
        logger.info(f"{method_name} PnL: Total={total_pnl:.4f}, Sharpe={sharpe_ratio:.4f}, WinRate={win_rate:.2%}")
        
        return metrics
    
    def run_comprehensive_ab_test(self, 
                                df: pd.DataFrame,
                                neutral_bands: List[float] = None,
                                symbol: str = "TEST") -> Dict[str, Any]:
        """
        Run comprehensive A/B test suite.
        
        Args:
            df: Input DataFrame with price data
            neutral_bands: List of neutral bands to test
            symbol: Symbol name for logging
            
        Returns:
            Dictionary with comprehensive test results
        """
        if neutral_bands is None:
            neutral_bands = [0.002, 0.003, 0.004, 0.005, 0.006]
        
        logger.info(f"Running comprehensive A/B test for {symbol}")
        
        results = {
            'symbol': symbol,
            'neutral_band_tests': [],
            'consistency_tests': [],
            'performance_comparison': {},
            'recommendations': []
        }
        
        # Test different neutral bands
        band_results = self.test_neutral_band_impact(df, neutral_bands, symbol)
        results['neutral_band_tests'] = band_results
        
        # Test consistency across multiple runs
        consistency_results = self._test_consistency(df, neutral_bands[2], symbol)  # Use middle band
        results['consistency_tests'] = consistency_results
        
        # Performance comparison
        perf_comparison = self._compare_band_performance(df, neutral_bands, symbol)
        results['performance_comparison'] = perf_comparison
        
        # Generate recommendations
        recommendations = self._generate_recommendations(band_results, perf_comparison)
        results['recommendations'] = recommendations
        
        logger.info(f"Comprehensive A/B test completed for {symbol}")
        return results
    
    def _calculate_metrics(self, targets: np.ndarray) -> Dict[str, float]:
        """Calculate metrics for target array."""
        targets_clean = targets[~np.isnan(targets)]
        
        if len(targets_clean) == 0:
            return {
                'macro_f1': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'flat_percentage': 0.0,
                'total_samples': 0
            }
        
        # For target creation, we don't have true vs pred, so calculate distribution metrics
        unique_vals, counts = np.unique(targets_clean, return_counts=True)
        total = len(targets_clean)
        
        flat_count = counts[unique_vals == 0][0] if 0 in unique_vals else 0
        flat_percentage = (flat_count / total) * 100
        
        # For A/B testing, we use distribution balance as proxy for quality
        # Ideal distribution might be roughly balanced
        class_balance = np.std(counts / total)  # Lower std = more balanced
        
        return {
            'macro_f1': 1.0 - class_balance,  # Proxy metric: higher balance = higher score
            'precision': 1.0 - class_balance,  # Proxy metric
            'recall': 1.0 - class_balance,  # Proxy metric
            'flat_percentage': flat_percentage,
            'total_samples': total,
            'class_balance_std': class_balance
        }
    
    def _test_consistency(self, df: pd.DataFrame, neutral_band: float, symbol: str) -> List[Dict[str, Any]]:
        """Test consistency across multiple runs."""
        results = []
        baseline_targets = None
        
        for run in range(3):
            df_test = ensure_direction_1d(df.copy(), neutral_band=neutral_band, symbol=f"{symbol}_run_{run}")
            targets = df_test['direction_1d'].values
            
            if baseline_targets is None:
                baseline_targets = targets
            else:
                identical = np.array_equal(baseline_targets, targets, equal_nan=True)
                differences = np.sum(baseline_targets != targets)
                
                results.append({
                    'run': run,
                    'identical_to_baseline': identical,
                    'differences': differences,
                    'total_samples': len(targets)
                })
        
        return results
    
    def _compare_band_performance(self, df: pd.DataFrame, neutral_bands: List[float], symbol: str) -> Dict[str, Any]:
        """Compare performance across different bands."""
        performance = {}
        
        # Calculate forward returns for PnL simulation
        forward_returns = df['close'].shift(-1) / df['close'] - 1.0
        
        for band in neutral_bands:
            df_test = ensure_direction_1d(df.copy(), neutral_band=band, symbol=f"{symbol}_perf_{band}")
            targets = df_test['direction_1d'].values
            
            # Calculate metrics
            metrics = self._calculate_metrics(targets)
            pnl_metrics = self.simulate_pnl_metrics(targets, forward_returns.values, f"Band_{band}")
            
            performance[f"band_{band}"] = {
                'neutral_band': band,
                'target_metrics': metrics,
                'pnl_metrics': pnl_metrics
            }
        
        return performance
    
    def _generate_recommendations(self, band_results: List[ABTestResult], perf_comparison: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Find best performing band
        best_band = None
        best_score = -float('inf')
        
        for band_key, perf in perf_comparison.items():
            if band_key.startswith('band_'):
                score = perf['pnl_metrics']['sharpe_ratio']
                if score > best_score:
                    best_score = score
                    best_band = perf['neutral_band']
        
        if best_band:
            recommendations.append(f"Optimal neutral band: ±{best_band*100:.1f}% (Sharpe: {best_score:.3f})")
        
        # Check FLAT percentage recommendations
        for band_key, perf in perf_comparison.items():
            if band_key.startswith('band_'):
                flat_pct = perf['target_metrics']['flat_percentage']
                band = perf['neutral_band']
                
                if flat_pct < 15:
                    recommendations.append(f"Band ±{band*100:.1f}%: FLAT class too low ({flat_pct:.1f}%) - consider wider band")
                elif flat_pct > 50:
                    recommendations.append(f"Band ±{band*100:.1f}%: FLAT class too high ({flat_pct:.1f}%) - consider narrower band")
        
        # Check for consistency issues
        for result in band_results:
            if not result.identical_results and result.differences_count > 0:
                recommendations.append(f"Inconsistency detected between {result.method_a_name} and {result.method_b_name}")
        
        return recommendations
    
    def generate_report(self) -> str:
        """Generate comprehensive A/B testing report."""
        report = ["=" * 60]
        report.append("A/B TESTING FRAMEWORK REPORT")
        report.append("=" * 60)
        report.append("")
        
        if not self.test_results:
            report.append("No test results available.")
            return "\n".join(report)
        
        # Summary
        total_tests = len(self.test_results)
        identical_results = sum(1 for r in self.test_results if r.identical_results)
        
        report.append(f"Total Tests Run: {total_tests}")
        report.append(f"Identical Results: {identical_results}/{total_tests}")
        report.append("")
        
        # Detailed results
        for result in self.test_results:
            report.append(f"Test: {result.test_name}")
            report.append(f"  Methods: {result.method_a_name} vs {result.method_b_name}")
            report.append(f"  Identical: {result.identical_results}")
            report.append(f"  Differences: {result.differences_count}/{result.total_samples}")
            
            if result.neutral_band_a and result.neutral_band_b:
                report.append(f"  Neutral Bands: {result.neutral_band_a} vs {result.neutral_band_b}")
            
            if result.performance_difference:
                report.append("  Performance Differences:")
                for metric, diff in result.performance_difference.items():
                    report.append(f"    {metric}: {diff:+.4f}")
            
            report.append("")
        
        return "\n".join(report)


class TestABTestingFramework(unittest.TestCase):
    """Test suite for A/B testing framework."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Create realistic test data
        prices = [100.0]
        for _ in range(99):
            change = np.random.normal(0.001, 0.02)
            prices.append(prices[-1] * (1 + change))
        
        self.test_df = pd.DataFrame({
            'close': prices,
            'volume': np.random.randint(1000, 5000, 100)
        })
        
        self.framework = ABTestingFramework(random_seed=42)
    
    def test_compare_identical_methods(self):
        """Test comparison of identical target creation methods."""
        # Compare same method with same parameters
        result = self.framework.compare_target_methods(
            self.test_df,
            lambda df, **kwargs: ensure_direction_1d(df, neutral_band=0.004, symbol="A"),
            lambda df, **kwargs: ensure_direction_1d(df, neutral_band=0.004, symbol="B"),
            "Method_A",
            "Method_B"
        )
        
        self.assertTrue(result.identical_results)
        self.assertEqual(result.differences_count, 0)
    
    def test_compare_different_neutral_bands(self):
        """Test comparison of methods with different neutral bands."""
        result = self.framework.compare_target_methods(
            self.test_df,
            lambda df, **kwargs: ensure_direction_1d(df, neutral_band=0.002, symbol="A"),
            lambda df, **kwargs: ensure_direction_1d(df, neutral_band=0.006, symbol="B"),
            "Band_0.002",
            "Band_0.006"
        )
        
        # Different bands should produce different results
        self.assertFalse(result.identical_results)
        self.assertGreater(result.differences_count, 0)
    
    def test_neutral_band_impact_testing(self):
        """Test neutral band impact analysis."""
        bands = [0.003, 0.004, 0.005]
        results = self.framework.test_neutral_band_impact(self.test_df, bands, "TEST")
        
        self.assertEqual(len(results), 2)  # Comparing to baseline
        
        for result in results:
            self.assertIsNotNone(result.performance_difference)
            self.assertIn('flat_pct_diff', result.performance_difference)
    
    def test_macro_f1_precision_recall_validation(self):
        """Test metric validation functionality."""
        # Create mock true and predicted targets
        true_targets = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1])
        pred_targets = np.array([-1, 0, 1, -1, 1, 1, 0, 0, 1])  # Some differences
        
        metrics = self.framework.validate_macro_f1_precision_recall(
            true_targets, pred_targets, "Test_Method"
        )
        
        self.assertIn('macro_f1', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertGreater(metrics['samples'], 0)
    
    def test_pnl_metrics_simulation(self):
        """Test PnL metrics simulation."""
        targets = np.array([1, -1, 0, 1, -1])  # Mix of predictions
        returns = np.array([0.02, -0.01, 0.005, 0.01, -0.015])  # Corresponding returns
        
        pnl_metrics = self.framework.simulate_pnl_metrics(targets, returns, "Test_PnL")
        
        self.assertIn('total_pnl', pnl_metrics)
        self.assertIn('sharpe_ratio', pnl_metrics)
        self.assertIn('win_rate', pnl_metrics)
        self.assertGreater(pnl_metrics['trades'], 0)
    
    def test_comprehensive_ab_test(self):
        """Test comprehensive A/B test suite."""
        results = self.framework.run_comprehensive_ab_test(
            self.test_df, 
            neutral_bands=[0.003, 0.004, 0.005],
            symbol="COMPREHENSIVE_TEST"
        )
        
        self.assertIn('neutral_band_tests', results)
        self.assertIn('consistency_tests', results)
        self.assertIn('performance_comparison', results)
        self.assertIn('recommendations', results)
        
        self.assertGreater(len(results['neutral_band_tests']), 0)
        self.assertIsInstance(results['recommendations'], list)
    
    def test_report_generation(self):
        """Test report generation functionality."""
        # Run some tests first
        self.framework.compare_target_methods(
            self.test_df,
            lambda df, **kwargs: ensure_direction_1d(df, neutral_band=0.004, symbol="A"),
            lambda df, **kwargs: ensure_direction_1d(df, neutral_band=0.004, symbol="B"),
            "Method_A",
            "Method_B"
        )
        
        report = self.framework.generate_report()
        
        self.assertIn("A/B TESTING FRAMEWORK REPORT", report)
        self.assertIn("Total Tests Run:", report)
        self.assertIn("Method_A vs Method_B", report)


if __name__ == "__main__":
    unittest.main()