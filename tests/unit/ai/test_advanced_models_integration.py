"""
Unit tests for Advanced Models Integration and Optimization.

This module contains comprehensive unit tests for all advanced model integration
components including model integration, performance optimization, feature pipeline,
prediction ensemble, model monitoring, and adaptive systems.
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import warnings

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore")

# Import advanced model components
from src.ai.advanced_models.model_integration import AdvancedModelIntegration, ModelEnsemble
from src.ai.advanced_models.performance_optimizer import PerformanceOptimizer, ModelOptimizer
from src.ai.advanced_models.feature_pipeline import AdvancedFeaturePipeline, FeatureOptimizer
from src.ai.advanced_models.prediction_ensemble import PredictionEnsemble, EnsembleManager
from src.ai.advanced_models.model_monitoring import ModelMonitoring, PerformanceTracker
from src.ai.advanced_models.adaptive_system import AdaptiveSystem, SystemOptimizer


class TestAdvancedModelIntegration(unittest.TestCase):
    """Test cases for AdvancedModelIntegration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.integration = AdvancedModelIntegration(
            enable_deep_learning=False,
            enable_time_series=False,
            enable_reinforcement_learning=False,
            enable_nlp=False
        )
        
        # Mock data
        self.sample_data = {
            'market_data': pd.DataFrame({
                'open': [100, 101, 102],
                'high': [105, 106, 107],
                'low': [95, 96, 97],
                'close': [102, 103, 104],
                'volume': [1000, 1100, 1200]
            }),
            'time_series_data': pd.DataFrame({
                'value': [1, 2, 3, 4, 5],
                'timestamp': pd.date_range('2023-01-01', periods=5)
            }),
            'market_state': {'price': 100, 'volume': 1000},
            'text_data': 'The market is bullish today'
        }
    
    def test_initialization(self):
        """Test integration initialization."""
        self.assertEqual(self.integration.model_name, "advanced_model_integration")
        self.assertFalse(self.integration.enable_deep_learning)
        self.assertFalse(self.integration.enable_time_series)
        self.assertFalse(self.integration.enable_reinforcement_learning)
        self.assertFalse(self.integration.enable_nlp)
        self.assertEqual(self.integration.max_concurrent_models, 4)
    
    def test_predict_sync(self):
        """Test synchronous prediction."""
        result = self.integration.predict_sync(self.sample_data)
        
        self.assertIsInstance(result, dict)
        self.assertIn('timestamp', result)
    
    def test_ensemble_predict(self):
        """Test ensemble prediction."""
        result = self.integration.ensemble_predict(
            self.sample_data,
            ensemble_method="weighted_average"
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('ensemble_method', result)
        self.assertIn('individual_predictions', result)
    
    def test_get_performance_metrics(self):
        """Test performance metrics retrieval."""
        metrics = self.integration.get_performance_metrics()
        
        self.assertIn('total_predictions', metrics)
        self.assertIn('successful_predictions', metrics)
        self.assertIn('average_execution_time', metrics)
        self.assertIn('enabled_features', metrics)
    
    def test_get_model_status(self):
        """Test model status retrieval."""
        status = self.integration.get_model_status()
        
        self.assertIn('integration_name', status)
        self.assertIn('available_managers', status)
        self.assertIn('manager_status', status)
    
    def test_health_check(self):
        """Test health check functionality."""
        health = self.integration.health_check()
        
        self.assertIn('overall_health', health)
        self.assertIn('timestamp', health)
        self.assertIn('component_health', health)


class TestModelEnsemble(unittest.TestCase):
    """Test cases for ModelEnsemble."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ensemble = ModelEnsemble()
        
        # Mock predictions
        self.sample_predictions = {
            'model1': {'prediction': 0.8, 'confidence': 0.9},
            'model2': {'prediction': 0.7, 'confidence': 0.8},
            'model3': {'prediction': 0.9, 'confidence': 0.7}
        }
    
    def test_initialization(self):
        """Test ensemble initialization."""
        self.assertEqual(self.ensemble.ensemble_name, "model_ensemble")
        self.assertIsInstance(self.ensemble.ensemble_methods, list)
        self.assertEqual(len(self.ensemble.ensemble_history), 0)
    
    def test_create_ensemble_weighted_average(self):
        """Test weighted average ensemble creation."""
        result = self.ensemble.create_ensemble(
            self.sample_predictions,
            method="weighted_average"
        )
        
        self.assertIn('ensemble_method', result)
        self.assertIn('ensemble_value', result)
        self.assertIn('weights_used', result)
        self.assertEqual(result['ensemble_method'], 'weighted_average')
    
    def test_create_ensemble_majority_vote(self):
        """Test majority vote ensemble creation."""
        categorical_predictions = {
            'model1': {'prediction': 'positive'},
            'model2': {'prediction': 'positive'},
            'model3': {'prediction': 'negative'}
        }
        
        result = self.ensemble.create_ensemble(
            categorical_predictions,
            method="majority_vote"
        )
        
        self.assertIn('ensemble_method', result)
        self.assertIn('majority_prediction', result)
        self.assertIn('vote_counts', result)
        self.assertEqual(result['ensemble_method'], 'majority_vote')
    
    def test_create_ensemble_stacking(self):
        """Test stacking ensemble creation."""
        result = self.ensemble.create_ensemble(
            self.sample_predictions,
            method="stacking"
        )
        
        self.assertIn('ensemble_method', result)
        self.assertEqual(result['ensemble_method'], 'stacking')
    
    def test_create_ensemble_bayesian(self):
        """Test Bayesian model averaging ensemble creation."""
        result = self.ensemble.create_ensemble(
            self.sample_predictions,
            method="bayesian_model_averaging"
        )
        
        self.assertIn('ensemble_method', result)
        self.assertEqual(result['ensemble_method'], 'bayesian_model_averaging')
    
    def test_get_ensemble_statistics(self):
        """Test ensemble statistics retrieval."""
        # Create some ensemble predictions first
        self.ensemble.create_ensemble(self.sample_predictions)
        
        stats = self.ensemble.get_ensemble_statistics()
        
        self.assertIn('ensemble_name', stats)
        self.assertIn('total_ensembles', stats)
        self.assertIn('successful_ensembles', stats)
        self.assertIn('success_rate', stats)


class TestPerformanceOptimizer(unittest.TestCase):
    """Test cases for PerformanceOptimizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.optimizer = PerformanceOptimizer()
    
    def test_initialization(self):
        """Test optimizer initialization."""
        self.assertEqual(self.optimizer.optimizer_name, "performance_optimizer")
        self.assertEqual(self.optimizer.memory_threshold, 0.8)
        self.assertEqual(self.optimizer.gpu_memory_threshold, 0.8)
        self.assertEqual(self.optimizer.cpu_threshold, 0.8)
        self.assertEqual(self.optimizer.optimization_interval, 60)
    
    def test_optimize_system(self):
        """Test system optimization."""
        result = self.optimizer.optimize_system(['memory_cleanup'])
        
        self.assertIn('optimization_strategies', result)
        self.assertIn('results', result)
        self.assertIn('total_execution_time', result)
    
    def test_get_system_status(self):
        """Test system status retrieval."""
        status = self.optimizer.get_system_status()
        
        self.assertIn('timestamp', status)
        self.assertIn('monitoring_active', status)
        self.assertIn('current_stats', status)
        self.assertIn('average_stats', status)
    
    def test_get_performance_metrics(self):
        """Test performance metrics retrieval."""
        metrics = self.optimizer.get_performance_metrics()
        
        self.assertIn('total_optimizations', metrics)
        self.assertIn('memory_optimizations', metrics)
        self.assertIn('gpu_optimizations', metrics)
        self.assertIn('monitoring_active', metrics)


class TestModelOptimizer(unittest.TestCase):
    """Test cases for ModelOptimizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.optimizer = ModelOptimizer()
        self.mock_model = Mock()
    
    def test_initialization(self):
        """Test optimizer initialization."""
        self.assertEqual(self.optimizer.model_name, "model_optimizer")
        self.assertIsInstance(self.optimizer.optimization_targets, list)
        self.assertEqual(len(self.optimizer.optimization_history), 0)
    
    def test_optimize_model_quantization(self):
        """Test model quantization optimization."""
        result = self.optimizer.optimize_model(
            self.mock_model,
            optimization_type="quantization",
            target_metric="inference_speed"
        )
        
        self.assertIn('model_name', result)
        self.assertIn('optimization_type', result)
        self.assertIn('target_metric', result)
        self.assertIn('execution_time', result)
    
    def test_optimize_model_pruning(self):
        """Test model pruning optimization."""
        result = self.optimizer.optimize_model(
            self.mock_model,
            optimization_type="pruning",
            target_metric="model_size"
        )
        
        self.assertIn('optimization_type', result)
        self.assertEqual(result['optimization_type'], 'pruning')
    
    def test_get_optimization_statistics(self):
        """Test optimization statistics retrieval."""
        # Perform some optimizations first
        self.optimizer.optimize_model(self.mock_model, "quantization")
        
        stats = self.optimizer.get_optimization_statistics()
        
        self.assertIn('total_optimizations', stats)
        self.assertIn('successful_optimizations', stats)
        self.assertIn('success_rate', stats)
        self.assertIn('optimization_type_counts', stats)


class TestAdvancedFeaturePipeline(unittest.TestCase):
    """Test cases for AdvancedFeaturePipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.pipeline = AdvancedFeaturePipeline()
        
        # Sample data
        self.sample_data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [105, 106, 107, 108, 109],
            'low': [95, 96, 97, 98, 99],
            'close': [102, 103, 104, 105, 106],
            'volume': [1000, 1100, 1200, 1300, 1400],
            'date': pd.date_range('2023-01-01', periods=5),
            'text': ['Bullish market', 'Strong earnings', 'Market rally', 'Positive sentiment', 'Growth expected']
        })
    
    def test_initialization(self):
        """Test pipeline initialization."""
        self.assertEqual(self.pipeline.pipeline_name, "advanced_feature_pipeline")
        self.assertTrue(self.pipeline.enable_technical_indicators)
        self.assertTrue(self.pipeline.enable_statistical_features)
        self.assertTrue(self.pipeline.enable_nlp_features)
        self.assertTrue(self.pipeline.enable_time_series_features)
    
    def test_create_features_technical(self):
        """Test technical feature creation."""
        result = self.pipeline.create_features(
            self.sample_data,
            feature_types=['technical']
        )
        
        self.assertIn('features', result)
        self.assertIn('technical', result['features'])
        self.assertIn('feature_metadata', result)
    
    def test_create_features_statistical(self):
        """Test statistical feature creation."""
        result = self.pipeline.create_features(
            self.sample_data,
            feature_types=['statistical']
        )
        
        self.assertIn('features', result)
        self.assertIn('statistical', result['features'])
    
    def test_create_features_time_series(self):
        """Test time series feature creation."""
        result = self.pipeline.create_features(
            self.sample_data,
            feature_types=['time_series']
        )
        
        self.assertIn('features', result)
        self.assertIn('time_series', result['features'])
    
    def test_create_features_nlp(self):
        """Test NLP feature creation."""
        result = self.pipeline.create_features(
            self.sample_data,
            feature_types=['nlp']
        )
        
        self.assertIn('features', result)
        self.assertIn('nlp', result['features'])
    
    def test_create_features_combined(self):
        """Test combined feature creation."""
        result = self.pipeline.create_features(
            self.sample_data,
            feature_types=['technical', 'statistical', 'time_series']
        )
        
        self.assertIn('features', result)
        self.assertIn('combined_features', result)
        self.assertIn('combined_shape', result)
    
    def test_get_feature_statistics(self):
        """Test feature statistics retrieval."""
        # Create some features first
        self.pipeline.create_features(self.sample_data)
        
        stats = self.pipeline.get_feature_statistics()
        
        self.assertIn('pipeline_name', stats)
        self.assertIn('total_features_created', stats)
        self.assertIn('total_pipelines_executed', stats)
        self.assertIn('enabled_features', stats)


class TestFeatureOptimizer(unittest.TestCase):
    """Test cases for FeatureOptimizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.optimizer = FeatureOptimizer()
        
        # Sample features
        self.sample_features = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10],  # Highly correlated with feature1
            'feature3': [0, 0, 0, 0, 0],  # Low variance
            'feature4': [1, 3, 2, 4, 1],
            'feature5': [5, 4, 3, 2, 1]
        })
        
        self.sample_target = pd.Series([1, 2, 3, 4, 5])
    
    def test_initialization(self):
        """Test optimizer initialization."""
        self.assertEqual(self.optimizer.optimizer_name, "feature_optimizer")
        self.assertIsInstance(self.optimizer.optimization_methods, list)
        self.assertEqual(len(self.optimizer.optimization_history), 0)
    
    def test_optimize_features_correlation_removal(self):
        """Test correlation-based feature optimization."""
        result = self.optimizer.optimize_features(
            self.sample_features,
            optimization_methods=['correlation_removal']
        )
        
        self.assertIn('optimization_strategies', result)
        self.assertIn('optimized_features', result)
        self.assertIn('features_removed', result)
    
    def test_optimize_features_variance_threshold(self):
        """Test variance-based feature optimization."""
        result = self.optimizer.optimize_features(
            self.sample_features,
            optimization_methods=['variance_threshold']
        )
        
        self.assertIn('optimization_strategies', result)
        self.assertIn('optimized_features', result)
    
    def test_optimize_features_mutual_information(self):
        """Test mutual information-based feature optimization."""
        result = self.optimizer.optimize_features(
            self.sample_features,
            target=self.sample_target,
            optimization_methods=['mutual_information']
        )
        
        self.assertIn('optimization_strategies', result)
        self.assertIn('optimized_features', result)
    
    def test_get_optimization_statistics(self):
        """Test optimization statistics retrieval."""
        # Perform some optimizations first
        self.optimizer.optimize_features(self.sample_features)
        
        stats = self.optimizer.get_optimization_statistics()
        
        self.assertIn('optimizer_name', stats)
        self.assertIn('total_optimizations', stats)
        self.assertIn('method_usage_counts', stats)
        self.assertIn('total_features_removed', stats)


class TestPredictionEnsemble(unittest.TestCase):
    """Test cases for PredictionEnsemble."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ensemble = PredictionEnsemble()
        
        # Mock models
        self.mock_model1 = Mock()
        self.mock_model1.predict.return_value = 0.8
        
        self.mock_model2 = Mock()
        self.mock_model2.predict.return_value = 0.7
        
        self.mock_model3 = Mock()
        self.mock_model3.predict.return_value = 0.9
    
    def test_initialization(self):
        """Test ensemble initialization."""
        self.assertEqual(self.ensemble.ensemble_name, "prediction_ensemble")
        self.assertIsInstance(self.ensemble.ensemble_methods, list)
        self.assertEqual(len(self.ensemble.models), 0)
    
    def test_add_model(self):
        """Test adding models to ensemble."""
        self.ensemble.add_model('model1', self.mock_model1, initial_weight=1.0)
        self.ensemble.add_model('model2', self.mock_model2, initial_weight=0.8)
        
        self.assertEqual(len(self.ensemble.models), 2)
        self.assertIn('model1', self.ensemble.models)
        self.assertIn('model2', self.ensemble.models)
    
    def test_remove_model(self):
        """Test removing models from ensemble."""
        self.ensemble.add_model('model1', self.mock_model1)
        self.ensemble.add_model('model2', self.mock_model2)
        
        self.ensemble.remove_model('model1')
        
        self.assertEqual(len(self.ensemble.models), 1)
        self.assertNotIn('model1', self.ensemble.models)
        self.assertIn('model2', self.ensemble.models)
    
    def test_update_model_weight(self):
        """Test updating model weights."""
        self.ensemble.add_model('model1', self.mock_model1, initial_weight=1.0)
        
        self.ensemble.update_model_weight('model1', 1.5, 'direct')
        
        self.assertEqual(self.ensemble.model_weights['model1'], 1.5)
    
    def test_predict_ensemble_weighted_average(self):
        """Test weighted average ensemble prediction."""
        self.ensemble.add_model('model1', self.mock_model1, initial_weight=1.0)
        self.ensemble.add_model('model2', self.mock_model2, initial_weight=0.8)
        
        X = np.array([[1, 2, 3]])
        result = self.ensemble.predict_ensemble(X, method='weighted_average')
        
        self.assertIn('ensemble_name', result)
        self.assertIn('method', result)
        self.assertIn('ensemble_prediction', result)
        self.assertIn('confidence', result)
    
    def test_predict_ensemble_majority_vote(self):
        """Test majority vote ensemble prediction."""
        # Mock models for classification
        mock_classifier1 = Mock()
        mock_classifier1.predict.return_value = 'positive'
        
        mock_classifier2 = Mock()
        mock_classifier2.predict.return_value = 'positive'
        
        mock_classifier3 = Mock()
        mock_classifier3.predict.return_value = 'negative'
        
        self.ensemble.add_model('model1', mock_classifier1)
        self.ensemble.add_model('model2', mock_classifier2)
        self.ensemble.add_model('model3', mock_classifier3)
        
        X = np.array([[1, 2, 3]])
        result = self.ensemble.predict_ensemble(X, method='majority_vote')
        
        self.assertIn('ensemble_name', result)
        self.assertIn('method', result)
        self.assertIn('ensemble_prediction', result)
    
    def test_update_model_performance(self):
        """Test updating model performance."""
        self.ensemble.add_model('model1', self.mock_model1)
        
        self.ensemble.update_model_performance('model1', 1.0, 0.9, 'accuracy')
        
        self.assertIn('model1', self.ensemble.model_performance)
        self.assertEqual(len(self.ensemble.model_performance['model1']), 1)
    
    def test_get_ensemble_statistics(self):
        """Test ensemble statistics retrieval."""
        self.ensemble.add_model('model1', self.mock_model1)
        
        stats = self.ensemble.get_ensemble_statistics()
        
        self.assertIn('ensemble_name', stats)
        self.assertIn('total_models', stats)
        self.assertIn('model_names', stats)
        self.assertIn('model_weights', stats)


class TestEnsembleManager(unittest.TestCase):
    """Test cases for EnsembleManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.manager = EnsembleManager()
    
    def test_initialization(self):
        """Test manager initialization."""
        self.assertEqual(self.manager.manager_name, "ensemble_manager")
        self.assertEqual(len(self.manager.ensembles), 0)
    
    def test_create_ensemble(self):
        """Test creating ensembles."""
        ensemble = self.manager.create_ensemble('test_ensemble')
        
        self.assertIsInstance(ensemble, PredictionEnsemble)
        self.assertEqual(ensemble.ensemble_name, 'test_ensemble')
        self.assertIn('test_ensemble', self.manager.ensembles)
    
    def test_get_ensemble(self):
        """Test getting ensembles."""
        self.manager.create_ensemble('test_ensemble')
        
        ensemble = self.manager.get_ensemble('test_ensemble')
        
        self.assertIsInstance(ensemble, PredictionEnsemble)
        self.assertEqual(ensemble.ensemble_name, 'test_ensemble')
    
    def test_remove_ensemble(self):
        """Test removing ensembles."""
        self.manager.create_ensemble('test_ensemble')
        
        self.manager.remove_ensemble('test_ensemble')
        
        self.assertNotIn('test_ensemble', self.manager.ensembles)
    
    def test_list_ensembles(self):
        """Test listing ensembles."""
        self.manager.create_ensemble('ensemble1')
        self.manager.create_ensemble('ensemble2')
        
        ensembles = self.manager.list_ensembles()
        
        self.assertEqual(len(ensembles), 2)
        self.assertIn('ensemble1', ensembles)
        self.assertIn('ensemble2', ensembles)
    
    def test_get_manager_statistics(self):
        """Test manager statistics retrieval."""
        self.manager.create_ensemble('test_ensemble')
        
        stats = self.manager.get_manager_statistics()
        
        self.assertIn('manager_name', stats)
        self.assertIn('total_ensembles', stats)
        self.assertIn('ensemble_names', stats)
        self.assertIn('ensemble_details', stats)


class TestModelMonitoring(unittest.TestCase):
    """Test cases for ModelMonitoring."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.monitoring = ModelMonitoring()
        self.mock_model = Mock()
    
    def test_initialization(self):
        """Test monitoring initialization."""
        self.assertEqual(self.monitoring.monitoring_name, "model_monitoring")
        self.assertEqual(self.monitoring.monitoring_interval, 60)
        self.assertIsInstance(self.monitoring.alert_thresholds, dict)
        self.assertFalse(self.monitoring.monitoring_active)
    
    def test_register_model(self):
        """Test model registration."""
        self.monitoring.register_model('test_model', self.mock_model, 'regression')
        
        self.assertIn('test_model', self.monitoring.monitored_models)
        self.assertEqual(self.monitoring.monitored_models['test_model']['type'], 'regression')
    
    def test_unregister_model(self):
        """Test model unregistration."""
        self.monitoring.register_model('test_model', self.mock_model)
        
        self.monitoring.unregister_model('test_model')
        
        self.assertNotIn('test_model', self.monitoring.monitored_models)
    
    def test_log_prediction(self):
        """Test prediction logging."""
        self.monitoring.register_model('test_model', self.mock_model)
        
        self.monitoring.log_prediction('test_model', 0.8, actual=0.9, latency=0.1)
        
        self.assertIn('test_model', self.monitoring.performance_metrics)
        self.assertEqual(len(self.monitoring.performance_metrics['test_model']['predictions']), 1)
    
    def test_get_model_performance(self):
        """Test model performance retrieval."""
        self.monitoring.register_model('test_model', self.mock_model)
        
        # Log some predictions
        self.monitoring.log_prediction('test_model', 0.8, actual=0.9, latency=0.1)
        self.monitoring.log_prediction('test_model', 0.7, actual=0.8, latency=0.2)
        
        performance = self.monitoring.get_model_performance('test_model')
        
        self.assertIn('model_name', performance)
        self.assertIn('total_predictions', performance)
        self.assertIn('accuracy', performance)
        self.assertIn('latency', performance)
    
    def test_get_all_performance(self):
        """Test all model performance retrieval."""
        self.monitoring.register_model('test_model', self.mock_model)
        
        all_performance = self.monitoring.get_all_performance()
        
        self.assertIn('test_model', all_performance)
        self.assertIsInstance(all_performance['test_model'], dict)
    
    def test_get_recent_alerts(self):
        """Test recent alerts retrieval."""
        alerts = self.monitoring.get_recent_alerts(hours=24)
        
        self.assertIsInstance(alerts, list)
    
    def test_get_monitoring_statistics(self):
        """Test monitoring statistics retrieval."""
        stats = self.monitoring.get_monitoring_statistics()
        
        self.assertIn('monitoring_name', stats)
        self.assertIn('monitoring_active', stats)
        self.assertIn('total_models', stats)
        self.assertIn('total_alerts', stats)


class TestPerformanceTracker(unittest.TestCase):
    """Test cases for PerformanceTracker."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tracker = PerformanceTracker()
    
    def test_initialization(self):
        """Test tracker initialization."""
        self.assertEqual(self.tracker.tracker_name, "performance_tracker")
        self.assertEqual(len(self.tracker.performance_history), 0)
        self.assertEqual(len(self.tracker.model_performance), 0)
    
    def test_track_prediction(self):
        """Test prediction tracking."""
        self.tracker.track_prediction('test_model', 0.8, actual=0.9, latency=0.1)
        
        self.assertEqual(len(self.tracker.performance_history), 1)
        self.assertIn('test_model', self.tracker.model_performance)
    
    def test_get_model_statistics(self):
        """Test model statistics retrieval."""
        self.tracker.track_prediction('test_model', 0.8, actual=0.9, latency=0.1)
        self.tracker.track_prediction('test_model', 0.7, actual=0.8, latency=0.2)
        
        stats = self.tracker.get_model_statistics('test_model')
        
        self.assertIn('model_name', stats)
        self.assertIn('total_predictions', stats)
        self.assertIn('accuracy', stats)
        self.assertIn('average_latency', stats)
    
    def test_get_all_statistics(self):
        """Test all model statistics retrieval."""
        self.tracker.track_prediction('test_model', 0.8, actual=0.9, latency=0.1)
        
        all_stats = self.tracker.get_all_statistics()
        
        self.assertIn('test_model', all_stats)
        self.assertIsInstance(all_stats['test_model'], dict)
    
    def test_get_performance_trend(self):
        """Test performance trend retrieval."""
        # Track multiple predictions
        for i in range(20):
            self.tracker.track_prediction('test_model', 0.8 + i*0.01, actual=0.9 + i*0.01, latency=0.1)
        
        trend = self.tracker.get_performance_trend('test_model', hours=24, metric='accuracy')
        
        self.assertIsInstance(trend, list)
    
    def test_get_tracker_statistics(self):
        """Test tracker statistics retrieval."""
        self.tracker.track_prediction('test_model', 0.8, actual=0.9, latency=0.1)
        
        stats = self.tracker.get_tracker_statistics()
        
        self.assertIn('tracker_name', stats)
        self.assertIn('total_records', stats)
        self.assertIn('total_models', stats)
        self.assertIn('model_names', stats)


class TestAdaptiveSystem(unittest.TestCase):
    """Test cases for AdaptiveSystem."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.adaptive_system = AdaptiveSystem()
        self.mock_component = Mock()
    
    def test_initialization(self):
        """Test adaptive system initialization."""
        self.assertEqual(self.adaptive_system.system_name, "adaptive_system")
        self.assertEqual(self.adaptive_system.adaptation_interval, 300)
        self.assertEqual(self.adaptive_system.learning_rate, 0.1)
        self.assertIsInstance(self.adaptive_system.adaptation_strategies, list)
        self.assertFalse(self.adaptive_system.adaptation_active)
    
    def test_register_component(self):
        """Test component registration."""
        self.adaptive_system.register_component('test_component', self.mock_component)
        
        self.assertIn('test_component', self.adaptive_system.adaptable_components)
        self.assertEqual(self.adaptive_system.adaptable_components['test_component']['component'], self.mock_component)
    
    def test_unregister_component(self):
        """Test component unregistration."""
        self.adaptive_system.register_component('test_component', self.mock_component)
        
        self.adaptive_system.unregister_component('test_component')
        
        self.assertNotIn('test_component', self.adaptive_system.adaptable_components)
    
    def test_log_performance(self):
        """Test performance logging."""
        self.adaptive_system.register_component('test_component', self.mock_component)
        
        self.adaptive_system.log_performance('test_component', 0.8)
        
        self.assertEqual(len(self.adaptive_system.performance_history), 1)
        self.assertEqual(len(self.adaptive_system.adaptable_components['test_component']['performance_history']), 1)
    
    def test_get_adaptation_statistics(self):
        """Test adaptation statistics retrieval."""
        stats = self.adaptive_system.get_adaptation_statistics()
        
        self.assertIn('system_name', stats)
        self.assertIn('adaptation_active', stats)
        self.assertIn('adaptation_interval', stats)
        self.assertIn('learning_rate', stats)
        self.assertIn('total_components', stats)
    
    def test_get_component_performance(self):
        """Test component performance retrieval."""
        self.adaptive_system.register_component('test_component', self.mock_component)
        
        # Log some performance
        self.adaptive_system.log_performance('test_component', 0.8)
        self.adaptive_system.log_performance('test_component', 0.9)
        
        performance = self.adaptive_system.get_component_performance('test_component')
        
        self.assertIn('component_name', performance)
        self.assertIn('performance_records', performance)
        self.assertIn('current_performance', performance)
        self.assertIn('average_performance', performance)


class TestSystemOptimizer(unittest.TestCase):
    """Test cases for SystemOptimizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.optimizer = SystemOptimizer()
        self.adaptive_system = AdaptiveSystem()
    
    def test_initialization(self):
        """Test optimizer initialization."""
        self.assertEqual(self.optimizer.optimizer_name, "system_optimizer")
        self.assertEqual(len(self.optimizer.adaptive_systems), 0)
    
    def test_register_adaptive_system(self):
        """Test adaptive system registration."""
        self.optimizer.register_adaptive_system('test_system', self.adaptive_system)
        
        self.assertIn('test_system', self.optimizer.adaptive_systems)
        self.assertEqual(self.optimizer.adaptive_systems['test_system'], self.adaptive_system)
    
    def test_unregister_adaptive_system(self):
        """Test adaptive system unregistration."""
        self.optimizer.register_adaptive_system('test_system', self.adaptive_system)
        
        self.optimizer.unregister_adaptive_system('test_system')
        
        self.assertNotIn('test_system', self.optimizer.adaptive_systems)
    
    def test_get_system_statistics(self):
        """Test system statistics retrieval."""
        self.optimizer.register_adaptive_system('test_system', self.adaptive_system)
        
        stats = self.optimizer.get_system_statistics()
        
        self.assertIn('optimizer_name', stats)
        self.assertIn('total_systems', stats)
        self.assertIn('system_names', stats)
        self.assertIn('system_statistics', stats)


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestAdvancedModelIntegration,
        TestModelEnsemble,
        TestPerformanceOptimizer,
        TestModelOptimizer,
        TestAdvancedFeaturePipeline,
        TestFeatureOptimizer,
        TestPredictionEnsemble,
        TestEnsembleManager,
        TestModelMonitoring,
        TestPerformanceTracker,
        TestAdaptiveSystem,
        TestSystemOptimizer
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*50}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")

