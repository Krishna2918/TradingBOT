"""
Test MarketTransformer Integration with Multi-Model System

This script tests the integration of MarketTransformer with the existing
multi-model training orchestrator.
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import asyncio

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from ai.multi_model_orchestrator import MultiModelOrchestrator
from ai.multi_model_config import MultiModelConfig, TransformerConfig
from ai.multi_model_integration import MultiModelSystem

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_data(n_samples: int = 500, n_features: int = 15) -> pd.DataFrame:
    """Create sample trading data for testing"""
    np.random.seed(42)
    
    # Generate features
    features = {}
    for i in range(n_features):
        features[f'feature_{i}'] = np.random.randn(n_samples)
    
    # Generate target (direction prediction)
    target_probs = 1 / (1 + np.exp(-features['feature_0'] - features['feature_1']))
    direction = np.where(target_probs < 0.33, 0, np.where(target_probs < 0.67, 1, 2))
    
    data = pd.DataFrame(features)
    data['direction_1d'] = direction
    
    return data

def test_transformer_orchestrator_integration():
    """Test MarketTransformer integration with orchestrator"""
    logger.info("Testing MarketTransformer integration with orchestrator...")
    
    try:
        # Create configuration with transformer enabled
        config = MultiModelConfig(
            enabled_models=['transformer'],
            parallel_training=False,
            transformer_config=TransformerConfig(
                d_model=64,
                nhead=2,
                num_layers=1,
                batch_size=16,
                epochs=5,
                learning_rate=0.001,
                sequence_length=50
            ),
            model_save_path="test_models/transformer_integration",
            results_save_path="test_results/transformer_integration"
        )
        
        # Create orchestrator
        orchestrator = MultiModelOrchestrator(config)
        
        # Create sample data
        data = create_sample_data(200, 10)
        logger.info(f"Created sample data with shape {data.shape}")
        
        # Train transformer model
        results = asyncio.run(orchestrator.train_all_models(data))
        
        # Check results
        assert 'transformer' in results, "Transformer model not found in results"
        
        transformer_result = results['transformer']
        logger.info(f"Transformer training completed:")
        logger.info(f"  - Model ID: {transformer_result.model_id}")
        logger.info(f"  - Validation accuracy: {transformer_result.validation_metrics.get('accuracy', 'N/A')}")
        logger.info(f"  - Test accuracy: {transformer_result.test_metrics.get('accuracy', 'N/A')}")
        logger.info(f"  - Model path: {transformer_result.model_path}")
        logger.info(f"  - Training logs: {len(transformer_result.training_logs)} entries")
        
        # Cleanup
        orchestrator.cleanup()
        
        logger.info("✓ MarketTransformer orchestrator integration test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ MarketTransformer orchestrator integration test failed: {e}")
        return False

def test_transformer_multi_model_system():
    """Test MarketTransformer with full multi-model system"""
    logger.info("Testing MarketTransformer with full multi-model system...")
    
    try:
        # Create sample data
        data = create_sample_data(150, 8)
        
        # Test with multi-model system
        system = MultiModelSystem()
        
        # Update config to include transformer
        system.config.enabled_models = ['transformer']
        system.config.transformer_config = TransformerConfig(
            d_model=32,
            nhead=2,
            num_layers=1,
            batch_size=8,
            epochs=3,
            sequence_length=30
        )
        
        # Recreate orchestrator with new config
        system.orchestrator = MultiModelOrchestrator(system.config)
        
        # Train models
        results = system.train_models(data, register_models=True)
        
        # Check results
        assert 'transformer' in results, "Transformer not found in training results"
        
        transformer_result = results['transformer']
        logger.info(f"Multi-model system training completed:")
        logger.info(f"  - Model registered: {transformer_result.model_id}")
        logger.info(f"  - Performance: {transformer_result.validation_metrics}")
        
        # Test model comparison
        comparison = system.compare_models()
        logger.info(f"  - Model comparison shape: {comparison.shape}")
        
        # Test best model selection
        best_model = system.get_best_model()
        if best_model:
            logger.info(f"  - Best model: {best_model.model_type}")
        
        # Get system status
        status = system.get_system_status()
        logger.info(f"  - System status: {status['registry']['total_models']} models in registry")
        
        # Cleanup
        system.cleanup()
        
        logger.info("✓ MarketTransformer multi-model system test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ MarketTransformer multi-model system test failed: {e}")
        return False

def test_transformer_configuration_validation():
    """Test transformer configuration validation"""
    logger.info("Testing transformer configuration validation...")
    
    try:
        # Test valid configuration
        valid_config = TransformerConfig(
            d_model=128,
            nhead=4,
            num_layers=2,
            batch_size=32,
            epochs=50,
            learning_rate=0.0001
        )
        
        assert valid_config.validate(), "Valid configuration should pass validation"
        logger.info("✓ Valid configuration passed validation")
        
        # Test invalid configuration (d_model not divisible by nhead)
        try:
            invalid_config = TransformerConfig(
                d_model=127,  # Not divisible by nhead=4
                nhead=4,
                num_layers=2
            )
            invalid_config.validate()
            logger.error("Invalid configuration should have failed validation")
            return False
        except ValueError as e:
            logger.info(f"✓ Invalid configuration correctly rejected: {e}")
        
        logger.info("✓ Transformer configuration validation test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Transformer configuration validation test failed: {e}")
        return False

def main():
    """Run all integration tests"""
    logger.info("Starting MarketTransformer Integration Tests")
    logger.info("=" * 60)
    
    tests = [
        test_transformer_orchestrator_integration,
        test_transformer_multi_model_system,
        test_transformer_configuration_validation
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
    logger.info(f"Integration Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All integration tests passed! MarketTransformer is ready.")
    else:
        logger.warning(f"⚠️  {total - passed} tests failed. Please check the integration.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)