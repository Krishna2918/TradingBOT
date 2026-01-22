"""
Test Multi-Model Core Infrastructure

This script tests the core multi-model infrastructure components:
- Multi-Model Orchestrator
- Model Registry
- Configuration System
- Integration Module
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from ai.multi_model_orchestrator import MultiModelOrchestrator, MultiModelConfig
from ai.model_registry import ModelRegistry, ModelMetadata
from ai.multi_model_config import ConfigurationManager, LSTMConfig, XGBoostConfig
from ai.multi_model_integration import MultiModelSystem, quick_train_models

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_data(n_samples: int = 1000, n_features: int = 20) -> pd.DataFrame:
    """Create sample trading data for testing"""
    np.random.seed(42)
    
    # Generate features
    features = {}
    for i in range(n_features):
        features[f'feature_{i}'] = np.random.randn(n_samples)
    
    # Generate target (direction prediction)
    target_probs = 1 / (1 + np.exp(-features['feature_0'] - features['feature_1']))
    target = np.random.binomial(1, target_probs)
    
    # Convert to 3-class problem (down, neutral, up)
    direction = np.where(target_probs < 0.33, 0, np.where(target_probs < 0.67, 1, 2))
    
    data = pd.DataFrame(features)
    data['direction_1d'] = direction
    
    return data

def test_configuration_system():
    """Test the configuration system"""
    logger.info("Testing Configuration System...")
    
    try:
        # Test configuration manager
        config_manager = ConfigurationManager("test_config")
        
        # Create and save a test configuration
        config = MultiModelConfig(
            enabled_models=['lstm', 'xgboost'],
            parallel_training=False
        )
        
        config.validate()
        config_path = config_manager.save_config(config, "test_config.yaml")
        logger.info(f"Saved test configuration to {config_path}")
        
        # Load configuration back
        loaded_config = config_manager.load_config("test_config.yaml")
        logger.info(f"Loaded configuration with models: {loaded_config.enabled_models}")
        
        # Create template configurations
        config_manager.create_template_configs()
        logger.info("Created template configurations")
        
        logger.info("✓ Configuration system test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Configuration system test failed: {e}")
        return False

def test_model_registry():
    """Test the model registry"""
    logger.info("Testing Model Registry...")
    
    try:
        # Initialize registry
        registry = ModelRegistry("test_models/registry")
        
        # Create sample metadata
        metadata = ModelMetadata(
            model_id="test_lstm_001",
            model_type="lstm",
            version="1.0.0",
            training_date=pd.Timestamp.now(),
            performance_metrics={'test_accuracy': 0.65, 'val_accuracy': 0.67},
            hyperparameters={'hidden_size': 128, 'num_layers': 2},
            feature_importance=None,
            training_data_hash="abc123",
            model_size_mb=0.0,
            inference_latency_ms=50.0,
            training_time_seconds=120.0,
            validation_split=0.2,
            test_split=0.2,
            feature_columns=['feature_0', 'feature_1'],
            target_column='direction_1d'
        )
        
        # Register a dummy model
        dummy_model = {"type": "lstm", "weights": [1, 2, 3]}
        model_path = registry.register_model(dummy_model, metadata)
        logger.info(f"Registered model at {model_path}")
        
        # List models
        models_df = registry.list_models()
        logger.info(f"Found {len(models_df)} models in registry")
        
        # Get registry stats
        stats = registry.get_registry_stats()
        logger.info(f"Registry stats: {stats}")
        
        logger.info("✓ Model registry test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Model registry test failed: {e}")
        return False

def test_orchestrator():
    """Test the multi-model orchestrator"""
    logger.info("Testing Multi-Model Orchestrator...")
    
    try:
        # Create configuration
        config = MultiModelConfig(
            enabled_models=['lstm', 'xgboost'],
            parallel_training=False,
            max_concurrent_models=2,
            model_save_path="test_models/orchestrator",
            results_save_path="test_results/orchestrator"
        )
        
        # Initialize orchestrator
        orchestrator = MultiModelOrchestrator(config)
        
        # Create sample data
        data = create_sample_data(500, 10)
        logger.info(f"Created sample data with shape {data.shape}")
        
        # Test training (this will use placeholder implementations)
        import asyncio
        results = asyncio.run(orchestrator.train_all_models(data))
        
        logger.info(f"Training completed for {len(results)} models")
        
        # Test model comparison
        comparison = orchestrator.compare_models()
        logger.info(f"Model comparison shape: {comparison.shape}")
        
        # Test best model selection
        best_model = orchestrator.get_best_model()
        if best_model:
            logger.info(f"Best model: {best_model.model_type} with accuracy {best_model.test_metrics.get('accuracy', 'N/A')}")
        
        # Cleanup
        orchestrator.cleanup()
        
        logger.info("✓ Multi-Model Orchestrator test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Multi-Model Orchestrator test failed: {e}")
        return False

def test_integration_system():
    """Test the integrated multi-model system"""
    logger.info("Testing Multi-Model Integration System...")
    
    try:
        # Create sample data
        data = create_sample_data(300, 8)
        
        # Test quick training function
        results = quick_train_models(data, models=['xgboost'], target_column='direction_1d')
        logger.info(f"Quick training completed for {len(results)} models")
        
        # Test full system
        system = MultiModelSystem()
        
        # Get system status
        status = system.get_system_status()
        logger.info(f"System status: {status}")
        
        # Create configuration templates
        system.create_configuration_templates()
        
        # Cleanup
        system.cleanup()
        
        logger.info("✓ Multi-Model Integration System test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Multi-Model Integration System test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("Starting Multi-Model Core Infrastructure Tests")
    logger.info("=" * 60)
    
    tests = [
        test_configuration_system,
        test_model_registry,
        test_orchestrator,
        test_integration_system
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            logger.error(f"Test {test.__name__} failed with exception: {e}")
        
        logger.info("-" * 40)
    
    logger.info("=" * 60)
    logger.info(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Core infrastructure is working correctly.")
    else:
        logger.warning(f"⚠️  {total - passed} tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)