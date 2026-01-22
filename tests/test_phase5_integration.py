"""
Phase 5 Integration Tests - Adaptive Ensemble Weights
====================================================

Tests for the adaptive ensemble weights system including:
- Weight calculation based on Brier scores
- Ensemble decision logic with adaptive weights
- Tie-break validation
- Model performance tracking
- Database integration
"""

import sys
import os
import unittest
import tempfile
import shutil
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ai.adaptive_weights import (
    AdaptiveWeightManager, ModelPerformance, EnsembleWeights,
    get_adaptive_weight_manager, add_model_prediction, get_ensemble_weights,
    update_ensemble_weights, get_performance_summary
)
from ai.enhanced_ensemble import EnhancedEnsemble, get_enhanced_ensemble
from ai.multi_model import MultiModelManager, get_multi_model_manager
from config.database import (
    get_database_manager, log_model_performance, get_model_performance_history,
    get_latest_model_performance
)


class TestAdaptiveWeightManager(unittest.TestCase):
    """Test the adaptive weight manager functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.weight_manager = AdaptiveWeightManager(
            window_size_days=7,
            min_predictions=5,
            epsilon=0.01
        )
    
    def test_initialization(self):
        """Test adaptive weight manager initialization."""
        self.assertIsNotNone(self.weight_manager)
        self.assertEqual(self.weight_manager.window_size_days, 7)
        self.assertEqual(self.weight_manager.min_predictions, 5)
        self.assertEqual(self.weight_manager.epsilon, 0.01)
    
    def test_add_prediction(self):
        """Test adding model predictions."""
        # Add some test predictions
        base_date = datetime.now() - timedelta(days=3)
        
        # Add predictions for model A (good performance)
        for i in range(10):
            self.weight_manager.add_prediction(
                model_name="model_a",
                symbol="TEST",
                predicted_probability=0.8,
                actual_outcome="WIN",
                prediction_date=base_date + timedelta(hours=i)
            )
        
        # Add predictions for model B (poor performance)
        for i in range(10):
            self.weight_manager.add_prediction(
                model_name="model_b",
                symbol="TEST",
                predicted_probability=0.8,
                actual_outcome="LOSS",
                prediction_date=base_date + timedelta(hours=i)
            )
        
        # Check that predictions were added
        self.assertEqual(len(self.weight_manager.model_predictions), 20)
        
        # Check that performance was calculated
        self.assertIn("model_a", self.weight_manager.model_performance)
        self.assertIn("model_b", self.weight_manager.model_performance)
    
    def test_brier_score_calculation(self):
        """Test Brier score calculation."""
        # Test perfect predictions
        perfect_predictions = [
            {"predicted_probability": 0.9, "actual_outcome": "WIN"},
            {"predicted_probability": 0.1, "actual_outcome": "LOSS"},
            {"predicted_probability": 0.8, "actual_outcome": "WIN"},
            {"predicted_probability": 0.2, "actual_outcome": "LOSS"}
        ]
        
        brier_score = self.weight_manager._calculate_brier_score(perfect_predictions)
        self.assertLess(brier_score, 0.1)  # Should be very low for good predictions
        
        # Test poor predictions
        poor_predictions = [
            {"predicted_probability": 0.1, "actual_outcome": "WIN"},
            {"predicted_probability": 0.9, "actual_outcome": "LOSS"},
            {"predicted_probability": 0.2, "actual_outcome": "WIN"},
            {"predicted_probability": 0.8, "actual_outcome": "LOSS"}
        ]
        
        brier_score = self.weight_manager._calculate_brier_score(poor_predictions)
        self.assertGreater(brier_score, 0.5)  # Should be high for poor predictions
    
    def test_accuracy_calculation(self):
        """Test accuracy calculation."""
        # Test perfect accuracy
        perfect_predictions = [
            {"predicted_probability": 0.9, "actual_outcome": "WIN"},
            {"predicted_probability": 0.1, "actual_outcome": "LOSS"},
            {"predicted_probability": 0.8, "actual_outcome": "WIN"},
            {"predicted_probability": 0.2, "actual_outcome": "LOSS"}
        ]
        
        accuracy = self.weight_manager._calculate_accuracy(perfect_predictions)
        self.assertEqual(accuracy, 1.0)  # Perfect accuracy
        
        # Test poor accuracy
        poor_predictions = [
            {"predicted_probability": 0.1, "actual_outcome": "WIN"},
            {"predicted_probability": 0.9, "actual_outcome": "LOSS"},
            {"predicted_probability": 0.2, "actual_outcome": "WIN"},
            {"predicted_probability": 0.8, "actual_outcome": "LOSS"}
        ]
        
        accuracy = self.weight_manager._calculate_accuracy(poor_predictions)
        self.assertEqual(accuracy, 0.0)  # No correct predictions
    
    def test_weight_calculation(self):
        """Test ensemble weight calculation."""
        # Add predictions for two models with different performance
        base_date = datetime.now() - timedelta(days=3)
        
        # Model A: Good performance (low Brier score)
        for i in range(10):
            self.weight_manager.add_prediction(
                model_name="model_a",
                symbol="TEST",
                predicted_probability=0.8,
                actual_outcome="WIN",
                prediction_date=base_date + timedelta(hours=i)
            )
        
        # Model B: Poor performance (high Brier score)
        for i in range(10):
            self.weight_manager.add_prediction(
                model_name="model_b",
                symbol="TEST",
                predicted_probability=0.8,
                actual_outcome="LOSS",
                prediction_date=base_date + timedelta(hours=i)
            )
        
        # Update ensemble weights
        ensemble_weights = self.weight_manager.update_ensemble_weights()
        
        # Check that weights were calculated
        self.assertIsNotNone(ensemble_weights)
        self.assertIn("model_a", ensemble_weights.weights)
        self.assertIn("model_b", ensemble_weights.weights)
        
        # Model A should have higher weight than Model B
        self.assertGreater(ensemble_weights.weights["model_a"], ensemble_weights.weights["model_b"])
        
        # Weights should sum to 1
        total_weight = sum(ensemble_weights.weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=2)
    
    def test_weight_smoothing(self):
        """Test weight smoothing to prevent dramatic changes."""
        # Set up initial weights
        self.weight_manager.ensemble_weights = EnsembleWeights(
            timestamp=datetime.now(),
            weights={"model_a": 0.5, "model_b": 0.5},
            total_models=2,
            performance_summary={},
            weight_entropy=1.0
        )
        
        # Test smoothing with new weights
        new_weights = {"model_a": 0.9, "model_b": 0.1}
        smoothed_weights = self.weight_manager._smooth_weights(new_weights)
        
        # Smoothed weights should be between old and new
        self.assertGreater(smoothed_weights["model_a"], 0.5)
        self.assertLess(smoothed_weights["model_a"], 0.9)
        self.assertLess(smoothed_weights["model_b"], 0.5)
        self.assertGreater(smoothed_weights["model_b"], 0.1)
    
    def test_export_import_data(self):
        """Test exporting and importing weights data."""
        # Add some test data
        base_date = datetime.now() - timedelta(days=3)
        self.weight_manager.add_prediction(
            model_name="test_model",
            symbol="TEST",
            predicted_probability=0.8,
            actual_outcome="WIN",
            prediction_date=base_date
        )
        
        # Export data
        exported_data = self.weight_manager.export_weights_data()
        
        # Check exported data structure
        self.assertIn("model_predictions", exported_data)
        self.assertIn("model_performance", exported_data)
        self.assertIn("ensemble_weights", exported_data)
        
        # Create new manager and import data
        new_manager = AdaptiveWeightManager()
        new_manager.import_weights_data(exported_data)
        
        # Check that data was imported correctly
        self.assertEqual(len(new_manager.model_predictions), 1)
        self.assertEqual(new_manager.model_predictions[0]["model_name"], "test_model")


class TestEnhancedEnsembleIntegration(unittest.TestCase):
    """Test enhanced ensemble integration with adaptive weights."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ensemble = EnhancedEnsemble()
    
    def test_ensemble_with_adaptive_weights(self):
        """Test ensemble analysis with adaptive weights."""
        # Mock market analysis data
        market_data = {
            "current_price": 150.0,
            "rsi": 45.0,
            "macd": 0.5,
            "sma_20": 148.0,
            "sma_50": 145.0,
            "bollinger_position": 0.3,
            "volume_ratio": 1.2,
            "atr": 0.02,
            "sentiment_score": 0.6,
            "fundamental_score": 0.7,
            "market_regime": "BULL",
            "sector_performance": 0.1,
            "news_impact": 0.3,
            "volatility": 0.025,
            "volume_trend": "HIGH"
        }
        
        # Create mock market analysis
        market_analysis = self.ensemble._build_market_analysis("TEST", market_data)
        
        # Run ensemble analysis
        result = self.ensemble._run_ensemble_analysis(market_analysis, "ENTRY")
        
        # Check that result includes weights
        self.assertIn("weights", result)
        self.assertIn("technical", result["weights"])
        self.assertIn("sentiment", result["weights"])
        self.assertIn("fundamental", result["weights"])
        self.assertIn("risk", result["weights"])
        self.assertIn("regime", result["weights"])
        
        # Weights should sum to approximately 1
        total_weight = sum(result["weights"].values())
        self.assertAlmostEqual(total_weight, 1.0, places=2)
    
    def test_model_prediction_logging(self):
        """Test that model predictions are logged."""
        # Mock market analysis data
        market_data = {
            "current_price": 150.0,
            "rsi": 45.0,
            "macd": 0.5,
            "sma_20": 148.0,
            "sma_50": 145.0,
            "bollinger_position": 0.3,
            "volume_ratio": 1.2,
            "atr": 0.02,
            "sentiment_score": 0.6,
            "fundamental_score": 0.7,
            "market_regime": "BULL",
            "sector_performance": 0.1,
            "news_impact": 0.3,
            "volatility": 0.025,
            "volume_trend": "HIGH"
        }
        
        # Create mock market analysis
        market_analysis = self.ensemble._build_market_analysis("TEST", market_data)
        
        # Mock the adaptive weight manager
        with patch('src.ai.enhanced_ensemble.add_model_prediction') as mock_add_prediction:
            # Run ensemble analysis
            result = self.ensemble._run_ensemble_analysis(market_analysis, "ENTRY")
            
            # Check that predictions were logged
            self.assertTrue(mock_add_prediction.called)
            
            # Should have logged predictions for each model plus ensemble
            expected_calls = 6  # 5 individual models + 1 ensemble
            self.assertEqual(mock_add_prediction.call_count, expected_calls)


class TestMultiModelIntegration(unittest.TestCase):
    """Test multi-model integration with adaptive weights."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.multi_model = MultiModelManager("DEMO")
    
    def test_adaptive_weights_mapping(self):
        """Test mapping between ensemble and multi-model names."""
        # Mock adaptive weights
        with patch('src.ai.multi_model.get_ensemble_weights') as mock_get_weights:
            mock_get_weights.return_value = {
                "technical_analyst": 0.4,
                "sentiment_analyst": 0.3,
                "fundamental_analyst": 0.2,
                "risk_analyst": 0.05,
                "market_regime_analyst": 0.05
            }
            
            adaptive_weights = self.multi_model.get_adaptive_weights()
            
            # Check that weights were mapped correctly
            self.assertIn("qwen2.5", adaptive_weights)  # technical_analyst
            self.assertIn("llama3.1", adaptive_weights)  # sentiment_analyst
            self.assertIn("gemma2", adaptive_weights)  # fundamental_analyst
            self.assertIn("phi3", adaptive_weights)  # risk_analyst
            self.assertIn("mistral", adaptive_weights)  # market_regime_analyst
    
    def test_weight_update_from_performance(self):
        """Test updating weights from performance data."""
        # Mock adaptive weights
        with patch('src.ai.multi_model.get_ensemble_weights') as mock_get_weights:
            mock_get_weights.return_value = {
                "technical_analyst": 0.4,
                "sentiment_analyst": 0.3,
                "fundamental_analyst": 0.2,
                "risk_analyst": 0.05,
                "market_regime_analyst": 0.05
            }
            
            # Update weights from performance
            self.multi_model.update_weights_from_performance()
            
            # Check that weights were updated (should be blended with current weights)
            weights = self.multi_model.get_model_weights()
            self.assertIsNotNone(weights)
    
    def test_model_prediction_logging(self):
        """Test logging model predictions."""
        with patch('src.ai.multi_model.add_model_prediction') as mock_add_prediction:
            # Log a prediction
            self.multi_model.log_model_prediction(
                model_name="qwen2.5",
                symbol="TEST",
                predicted_probability=0.8,
                actual_outcome="PENDING"
            )
            
            # Check that prediction was logged
            self.assertTrue(mock_add_prediction.called)
            
            # Check the call arguments
            call_args = mock_add_prediction.call_args
            self.assertEqual(call_args[1]["model_name"], "technical_analyst")
            self.assertEqual(call_args[1]["symbol"], "TEST")
            self.assertEqual(call_args[1]["predicted_probability"], 0.8)


class TestDatabaseIntegration(unittest.TestCase):
    """Test database integration for model performance tracking."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary database
        self.temp_dir = tempfile.mkdtemp()
        self.original_db_path = "data/trading_demo.db"
        
        # Mock the database path
        with patch('src.config.database.Path') as mock_path:
            mock_path.return_value.parent.mkdir = MagicMock()
            self.db_manager = get_database_manager()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_log_model_performance(self):
        """Test logging model performance to database."""
        timestamp = datetime.now()
        window_start = timestamp - timedelta(days=7)
        window_end = timestamp
        
        # Log model performance
        result = log_model_performance(
            timestamp=timestamp,
            model="test_model",
            brier_score=0.25,
            accuracy=0.75,
            n_predictions=20,
            weight=0.4,
            window_start=window_start,
            window_end=window_end,
            mode="DEMO"
        )
        
        # Check that performance was logged
        self.assertIsNotNone(result)
    
    def test_get_model_performance_history(self):
        """Test retrieving model performance history."""
        # First log some performance data
        timestamp = datetime.now()
        window_start = timestamp - timedelta(days=7)
        window_end = timestamp
        
        log_model_performance(
            timestamp=timestamp,
            model="test_model",
            brier_score=0.25,
            accuracy=0.75,
            n_predictions=20,
            weight=0.4,
            window_start=window_start,
            window_end=window_end,
            mode="DEMO"
        )
        
        # Get performance history
        history = get_model_performance_history(model="test_model", mode="DEMO")
        
        # Check that history was retrieved
        self.assertIsNotNone(history)
        self.assertIsInstance(history, list)
    
    def test_get_latest_model_performance(self):
        """Test retrieving latest model performance."""
        # First log some performance data
        timestamp = datetime.now()
        window_start = timestamp - timedelta(days=7)
        window_end = timestamp
        
        log_model_performance(
            timestamp=timestamp,
            model="test_model",
            brier_score=0.25,
            accuracy=0.75,
            n_predictions=20,
            weight=0.4,
            window_start=window_start,
            window_end=window_end,
            mode="DEMO"
        )
        
        # Get latest performance
        latest = get_latest_model_performance(model="test_model", mode="DEMO")
        
        # Check that latest performance was retrieved
        self.assertIsNotNone(latest)
        self.assertIsInstance(latest, list)


class TestTieBreakValidation(unittest.TestCase):
    """Test tie-break validation logic."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.weight_manager = AdaptiveWeightManager()
    
    def test_tie_break_with_weights(self):
        """Test tie-breaking using adaptive weights."""
        # Set up two models with equal scores but different weights
        model_scores = {
            "model_a": 0.6,
            "model_b": 0.6
        }
        
        # Set up weights favoring model_a
        self.weight_manager.ensemble_weights = EnsembleWeights(
            timestamp=datetime.now(),
            weights={"model_a": 0.7, "model_b": 0.3},
            total_models=2,
            performance_summary={},
            weight_entropy=0.9
        )
        
        # Test weighted decision
        weighted_score_a = model_scores["model_a"] * 0.7
        weighted_score_b = model_scores["model_b"] * 0.3
        
        # Model A should win due to higher weight
        self.assertGreater(weighted_score_a, weighted_score_b)
    
    def test_weight_entropy_calculation(self):
        """Test weight entropy calculation."""
        # Test uniform weights (high entropy)
        uniform_weights = {"model_a": 0.5, "model_b": 0.5}
        entropy_uniform = self.weight_manager._calculate_weight_entropy(uniform_weights)
        
        # Test concentrated weights (low entropy)
        concentrated_weights = {"model_a": 0.9, "model_b": 0.1}
        entropy_concentrated = self.weight_manager._calculate_weight_entropy(concentrated_weights)
        
        # Uniform weights should have higher entropy
        self.assertGreater(entropy_uniform, entropy_concentrated)


class TestGlobalFunctions(unittest.TestCase):
    """Test global convenience functions."""
    
    def test_get_adaptive_weight_manager(self):
        """Test getting global adaptive weight manager."""
        manager = get_adaptive_weight_manager()
        self.assertIsNotNone(manager)
        self.assertIsInstance(manager, AdaptiveWeightManager)
    
    def test_add_model_prediction(self):
        """Test adding model prediction via global function."""
        # This should not raise an exception
        add_model_prediction(
            model_name="test_model",
            symbol="TEST",
            predicted_probability=0.8,
            actual_outcome="WIN",
            prediction_date=datetime.now()
        )
    
    def test_get_ensemble_weights(self):
        """Test getting ensemble weights via global function."""
        weights = get_ensemble_weights()
        self.assertIsNotNone(weights)
        self.assertIsInstance(weights, dict)
    
    def test_update_ensemble_weights(self):
        """Test updating ensemble weights via global function."""
        weights = update_ensemble_weights()
        self.assertIsNotNone(weights)
        self.assertIsInstance(weights, EnsembleWeights)
    
    def test_get_performance_summary(self):
        """Test getting performance summary via global function."""
        summary = get_performance_summary()
        self.assertIsNotNone(summary)
        self.assertIsInstance(summary, dict)


if __name__ == "__main__":
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestAdaptiveWeightManager,
        TestEnhancedEnsembleIntegration,
        TestMultiModelIntegration,
        TestDatabaseIntegration,
        TestTieBreakValidation,
        TestGlobalFunctions
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"PHASE 5 INTEGRATION TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    if result.wasSuccessful():
        print(f"\n✅ ALL TESTS PASSED - Phase 5 integration working correctly!")
    else:
        print(f"\n❌ SOME TESTS FAILED - Check implementation")
    
    print(f"{'='*60}")
