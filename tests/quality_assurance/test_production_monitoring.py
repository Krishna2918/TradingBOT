"""
Tests for Production Monitoring System.

Tests target distribution monitoring, alerting, dashboard output,
and guardrails for automatic reversion.
"""

import unittest
import pandas as pd
import numpy as np
import sys
import tempfile
import json
import os
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from ai.monitoring.target_quality_monitor import (
    TargetQualityMonitor,
    TargetQualityGuardrails,
    QualityThresholds,
    TargetQualityMetrics,
    monitor_target_creation_with_logging,
    setup_production_monitoring,
    create_quality_alerts_summary
)
from ai.data.targets import ensure_direction_1d


class TestTargetQualityMonitor(unittest.TestCase):
    """Test suite for target quality monitoring."""
    
    def setUp(self):
        """Set up test data and monitor."""
        np.random.seed(42)
        
        # Create test data with known characteristics
        self.good_data = self._create_test_data(
            size=200, 
            volatility=0.02, 
            trend=0.001
        )
        
        self.poor_data = self._create_test_data(
            size=50,  # Too few samples
            volatility=0.001,  # Too low volatility (will create imbalanced classes)
            trend=0.0
        )
        
        # Create temporary files for testing
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.temp_dir, "test_monitor.log")
        self.dashboard_file = os.path.join(self.temp_dir, "test_dashboard.json")
        
        # Create monitor with test configuration
        self.thresholds = QualityThresholds(
            min_flat_percentage=15.0,
            max_flat_percentage=50.0,
            min_samples=100,
            min_quality_score=70.0
        )
        
        self.monitor = TargetQualityMonitor(
            log_file=self.log_file,
            dashboard_output=self.dashboard_file,
            thresholds=self.thresholds
        )
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_monitor_good_quality_targets(self):
        """Test monitoring of good quality targets."""
        # Create targets
        df_with_targets = ensure_direction_1d(
            self.good_data, 
            neutral_band=0.004, 
            symbol="GOOD_TEST"
        )
        
        # Monitor
        metrics = self.monitor.monitor_target_creation(
            df_with_targets, 
            "GOOD_TEST", 
            0.004
        )
        
        # Verify metrics
        self.assertEqual(metrics.symbol, "GOOD_TEST")
        self.assertGreater(metrics.total_samples, 100)
        self.assertGreater(metrics.quality_score, 70.0)
        self.assertEqual(len(metrics.alerts), 0)  # Should have no alerts
        
        # Verify distribution
        total_pct = metrics.down_percentage + metrics.flat_percentage + metrics.up_percentage
        self.assertAlmostEqual(total_pct, 100.0, places=1)
    
    def test_monitor_poor_quality_targets(self):
        """Test monitoring of poor quality targets."""
        # Create targets with poor data
        df_with_targets = ensure_direction_1d(
            self.poor_data, 
            neutral_band=0.004, 
            symbol="POOR_TEST"
        )
        
        # Monitor
        metrics = self.monitor.monitor_target_creation(
            df_with_targets, 
            "POOR_TEST", 
            0.004
        )
        
        # Verify metrics indicate poor quality
        self.assertEqual(metrics.symbol, "POOR_TEST")
        self.assertLess(metrics.total_samples, 100)  # Should trigger low samples alert
        self.assertGreater(len(metrics.alerts), 0)  # Should have alerts
        
        # Check for specific alerts
        alert_text = " ".join(metrics.alerts)
        self.assertIn("LOW_SAMPLES", alert_text)
    
    def test_target_distribution_monitoring(self):
        """Test target distribution monitoring and alerting."""
        # Create data that will result in imbalanced classes
        imbalanced_data = pd.DataFrame({
            'close': [100.0] * 50 + [100.1] * 50  # Mostly flat movement
        })
        
        df_with_targets = ensure_direction_1d(
            imbalanced_data, 
            neutral_band=0.0001,  # Very narrow band
            symbol="IMBALANCED_TEST"
        )
        
        metrics = self.monitor.monitor_target_creation(
            df_with_targets, 
            "IMBALANCED_TEST", 
            0.0001
        )
        
        # Should detect class imbalance
        alert_text = " ".join(metrics.alerts)
        self.assertTrue(
            "CLASS_IMBALANCE" in alert_text or "FLAT_TOO_HIGH" in alert_text,
            f"Expected imbalance alert, got: {metrics.alerts}"
        )
    
    def test_dashboard_output_creation(self):
        """Test dashboard JSON output creation."""
        df_with_targets = ensure_direction_1d(
            self.good_data, 
            neutral_band=0.004, 
            symbol="DASHBOARD_TEST"
        )
        
        metrics = self.monitor.monitor_target_creation(
            df_with_targets, 
            "DASHBOARD_TEST", 
            0.004
        )
        
        # Verify dashboard file was created
        self.assertTrue(os.path.exists(self.dashboard_file))
        
        # Verify dashboard content
        with open(self.dashboard_file, 'r') as f:
            dashboard_data = json.load(f)
        
        self.assertIn('last_updated', dashboard_data)
        self.assertIn('metrics', dashboard_data)
        self.assertEqual(dashboard_data['metrics']['symbol'], "DASHBOARD_TEST")
    
    def test_quality_degradation_detection(self):
        """Test quality degradation detection."""
        # Create several metrics with declining quality
        for i in range(5):
            # Create progressively worse data
            data = self._create_test_data(
                size=150 - i*20,  # Decreasing samples
                volatility=0.02 - i*0.003,  # Decreasing volatility
                trend=0.001
            )
            
            df_with_targets = ensure_direction_1d(
                data, 
                neutral_band=0.004, 
                symbol="DEGRADATION_TEST"
            )
            
            self.monitor.monitor_target_creation(
                df_with_targets, 
                "DEGRADATION_TEST", 
                0.004
            )
        
        # Check for degradation
        latest_metrics = self.monitor.metrics_history[-1]
        degradation_result = self.monitor.check_quality_degradation(
            latest_metrics, 
            lookback_periods=3
        )
        
        # Should detect degradation if quality dropped significantly
        if degradation_result['degradation_detected']:
            self.assertIn('recommendation', degradation_result)
            self.assertGreater(len(degradation_result['reasons']), 0)
    
    def test_global_target_monitoring(self):
        """Test global target monitoring across multiple symbols."""
        # Create data for multiple symbols
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        all_data = []
        
        for symbol in symbols:
            data = self._create_test_data(size=100, volatility=0.02, trend=0.001)
            data['symbol'] = symbol
            
            df_with_targets = ensure_direction_1d(
                data, 
                neutral_band=0.004, 
                symbol=symbol
            )
            
            all_data.append(df_with_targets)
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Monitor global targets
        global_metrics = self.monitor.monitor_global_targets(
            combined_df, 
            symbols, 
            0.004
        )
        
        # Verify global metrics
        self.assertTrue(global_metrics.symbol.startswith("GLOBAL_"))
        self.assertGreater(global_metrics.total_samples, 250)  # Should have data from all symbols
    
    def test_get_dashboard_data(self):
        """Test dashboard data retrieval."""
        # Create some test metrics
        for i in range(3):
            data = self._create_test_data(size=150, volatility=0.02, trend=0.001)
            df_with_targets = ensure_direction_1d(
                data, 
                neutral_band=0.004, 
                symbol=f"TEST_{i}"
            )
            
            self.monitor.monitor_target_creation(
                df_with_targets, 
                f"TEST_{i}", 
                0.004
            )
        
        # Get dashboard data
        dashboard_data = self.monitor.get_dashboard_data(hours_back=1)
        
        self.assertEqual(dashboard_data['status'], 'success')
        self.assertIn('summary', dashboard_data)
        self.assertIn('latest_metrics', dashboard_data)
        self.assertGreater(dashboard_data['summary']['total_symbols'], 0)
    
    def _create_test_data(self, size: int, volatility: float, trend: float) -> pd.DataFrame:
        """Create test price data with specified characteristics."""
        prices = [100.0]
        
        for _ in range(size - 1):
            change = trend + np.random.normal(0, volatility)
            prices.append(prices[-1] * (1 + change))
        
        return pd.DataFrame({
            'close': prices,
            'volume': np.random.randint(1000, 5000, size)
        })


class TestTargetQualityGuardrails(unittest.TestCase):
    """Test suite for target quality guardrails."""
    
    def setUp(self):
        """Set up test guardrails."""
        self.monitor = TargetQualityMonitor()
        self.guardrails = TargetQualityGuardrails(
            monitor=self.monitor,
            reversion_threshold=60.0,
            consecutive_failures=2
        )
    
    def test_guardrails_no_reversion_needed(self):
        """Test guardrails when quality is good."""
        # Create good quality metrics
        good_metrics = TargetQualityMetrics(
            timestamp=datetime.now().isoformat(),
            symbol="GOOD_TEST",
            total_samples=200,
            down_count=60,
            flat_count=80,
            up_count=60,
            down_percentage=30.0,
            flat_percentage=40.0,
            up_percentage=30.0,
            neutral_band_used=0.004,
            class_balance_score=0.9,
            missing_classes=[],
            quality_score=85.0,
            alerts=[]
        )
        
        config = {'neutral_band': 0.004}
        
        result = self.guardrails.check_and_revert(good_metrics, config)
        
        self.assertFalse(result['should_revert'])
        self.assertEqual(self.guardrails.failure_count, 0)
    
    def test_guardrails_reversion_after_consecutive_failures(self):
        """Test guardrails trigger reversion after consecutive failures."""
        # Create poor quality metrics
        poor_metrics = TargetQualityMetrics(
            timestamp=datetime.now().isoformat(),
            symbol="POOR_TEST",
            total_samples=50,
            down_count=5,
            flat_count=40,
            up_count=5,
            down_percentage=10.0,
            flat_percentage=80.0,
            up_percentage=10.0,
            neutral_band_used=0.008,
            class_balance_score=0.3,
            missing_classes=[],
            quality_score=45.0,  # Below threshold
            alerts=["LOW_QUALITY", "FLAT_TOO_HIGH"]
        )
        
        config = {'neutral_band': 0.008}
        
        # First failure
        result1 = self.guardrails.check_and_revert(poor_metrics, config)
        self.assertFalse(result1['should_revert'])
        self.assertEqual(self.guardrails.failure_count, 1)
        
        # Second failure - should trigger reversion
        result2 = self.guardrails.check_and_revert(poor_metrics, config)
        self.assertTrue(result2['should_revert'])
        self.assertIn('consecutive periods', result2['reason'])
        self.assertEqual(self.guardrails.failure_count, 0)  # Reset after reversion
    
    def test_get_reversion_recommendations(self):
        """Test reversion recommendations."""
        # Add some historical metrics to monitor
        for i in range(5):
            quality_score = 80.0 - i * 5  # Declining quality
            metrics = TargetQualityMetrics(
                timestamp=(datetime.now() - timedelta(hours=i)).isoformat(),
                symbol="REVERT_TEST",
                total_samples=200,
                down_count=60,
                flat_count=80,
                up_count=60,
                down_percentage=30.0,
                flat_percentage=40.0,
                up_percentage=30.0,
                neutral_band_used=0.004,
                class_balance_score=0.8,
                missing_classes=[],
                quality_score=quality_score,
                alerts=[]
            )
            self.monitor.metrics_history.append(metrics)
        
        recommendations = self.guardrails.get_reversion_recommendations("REVERT_TEST")
        
        self.assertIn('recommendation', recommendations)
        if recommendations['recommendation'] == 'revert_to_best':
            self.assertIn('best_config', recommendations)


class TestIntegrationFunctions(unittest.TestCase):
    """Test integration functions for production monitoring."""
    
    def test_monitor_target_creation_with_logging(self):
        """Test convenience function for monitoring with logging."""
        # Create test data
        data = pd.DataFrame({
            'close': [100.0, 102.0, 98.0, 104.0, 96.0],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
        
        df_with_targets = ensure_direction_1d(data, neutral_band=0.01, symbol="INTEGRATION_TEST")
        
        # Monitor with logging
        metrics = monitor_target_creation_with_logging(
            df_with_targets, 
            "INTEGRATION_TEST", 
            0.01
        )
        
        self.assertEqual(metrics.symbol, "INTEGRATION_TEST")
        self.assertGreater(metrics.total_samples, 0)
    
    def test_setup_production_monitoring(self):
        """Test production monitoring setup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "prod_monitor.log")
            dashboard_file = os.path.join(temp_dir, "prod_dashboard.json")
            
            monitor = setup_production_monitoring(log_file, dashboard_file)
            
            self.assertIsInstance(monitor, TargetQualityMonitor)
            self.assertEqual(monitor.log_file, log_file)
            self.assertEqual(monitor.dashboard_output, dashboard_file)
    
    def test_create_quality_alerts_summary(self):
        """Test quality alerts summary creation."""
        monitor = TargetQualityMonitor()
        
        # Add some test metrics with alerts
        metrics = TargetQualityMetrics(
            timestamp=datetime.now().isoformat(),
            symbol="ALERT_TEST",
            total_samples=50,
            down_count=10,
            flat_count=30,
            up_count=10,
            down_percentage=20.0,
            flat_percentage=60.0,
            up_percentage=20.0,
            neutral_band_used=0.004,
            class_balance_score=0.6,
            missing_classes=[],
            quality_score=65.0,
            alerts=["LOW_SAMPLES", "FLAT_TOO_HIGH"]
        )
        
        monitor.metrics_history.append(metrics)
        
        summary = create_quality_alerts_summary(monitor, hours_back=1)
        
        self.assertIn("TARGET QUALITY MONITORING SUMMARY", summary)
        self.assertIn("ALERT_TEST", summary)
        self.assertIn("LOW_SAMPLES", summary)


if __name__ == "__main__":
    unittest.main()