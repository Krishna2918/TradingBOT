"""
Test TransformerTrainer Implementation

This script tests the TransformerTrainer class to ensure it works correctly
with the MarketTransformer and provides comprehensive training capabilities.
"""

import sys
import logging
from pathlib import Path
import torch
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from ai.models.transformer_trainer import (
    TransformerTrainer, 
    TransformerTrainingConfig, 
    TransformerDataset
)
from ai.models.market_transformer import create_market_transformer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_data(n_samples: int = 1000, n_features: int = 15) -> tuple:
    """Create sample trading data for testing"""
    np.random.seed(42)
    
    # Generate features
    features = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # Generate targets (3-class classification)
    target_probs = 1 / (1 + np.exp(-features[:, 0] - features[:, 1]))
    targets = np.where(target_probs < 0.33, 0, np.where(target_probs < 0.67, 1, 2))
    
    # Generate time features (day, month, hour, regime)
    time_features = np.random.randint(0, 7, (n_samples, 4))
    time_features[:, 1] = np.random.randint(0, 12, n_samples)  # Month
    time_features[:, 2] = np.random.randint(0, 24, n_samples)  # Hour
    time_features[:, 3] = np.random.randint(0, 4, n_samples)   # Regime
    
    # Generate volatility
    volatility = np.random.rand(n_samples, 1) * 0.5 + 0.1
    
    return features, targets, time_features.astype(np.int32), volatility.astype(np.float32)

def test_transformer_dataset():
    """Test TransformerDataset creation and functionality"""
    logger.info("Testing TransformerDataset...")
    
    try:
        # Create sample data
        features, targets, time_features, volatility = create_sample_data(500, 10)
        
        # Create dataset
        dataset = TransformerDataset(
            features=features,
            targets=targets,
            sequence_length=50,
            time_features=time_features,
            volatility=volatility
        )
        
        # Test dataset properties
        assert len(dataset) > 0, "Dataset should not be empty"
        
        # Test data loading
        sample = dataset[0]
        assert 'features' in sample, "Sample should contain features"
        assert 'targets' in sample, "Sample should contain targets"
        assert 'time_features' in sample, "Sample should contain time_features"
        assert 'volatility' in sample, "Sample should contain volatility"
        
        # Test shapes
        assert sample['features'].shape == (50, 10), f"Expected features shape (50, 10), got {sample['features'].shape}"
        assert sample['targets'].shape == (1,), f"Expected targets shape (1,), got {sample['targets'].shape}"
        assert sample['time_features'].shape == (50, 4), f"Expected time_features shape (50, 4), got {sample['time_features'].shape}"
        assert sample['volatility'].shape == (50, 1), f"Expected volatility shape (50, 1), got {sample['volatility'].shape}"
        
        logger.info(f"Dataset created successfully with {len(dataset)} samples")
        logger.info("✓ TransformerDataset test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ TransformerDataset test failed: {e}")
        return False

def test_transformer_trainer_setup():
    """Test TransformerTrainer setup and configuration"""
    logger.info("Testing TransformerTrainer setup...")
    
    try:
        # Create training configuration
        config = TransformerTrainingConfig(
            batch_size=8,
            max_epochs=2,
            learning_rate=1e-4,
            mixed_precision=False,  # Disable for testing
            gradient_accumulation_steps=2,
            validation_split=0.2,
            log_frequency=10,
            save_checkpoints=False,  # Disable for testing
            memory_optimization=False  # Disable for testing
        )
        
        # Create trainer
        trainer = TransformerTrainer(config)
        
        # Test model setup
        model_config = {
            'input_dim': 10,
            'd_model': 64,
            'num_heads': 2,
            'num_layers': 1,
            'max_seq_length': 50,
            'num_classes': 3,
            'dropout': 0.1
        }
        
        model = trainer.setup_model(model_config)
        
        # Verify model setup
        assert trainer.model is not None, "Model should be initialized"
        assert trainer.optimizer is not None, "Optimizer should be initialized"
        assert trainer.scheduler is not None, "Scheduler should be initialized"
        
        param_count = model.count_parameters()
        logger.info(f"Model setup successful with {param_count} parameters")
        
        logger.info("✓ TransformerTrainer setup test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ TransformerTrainer setup test failed: {e}")
        return False

def test_transformer_trainer_data_loaders():
    """Test data loader creation"""
    logger.info("Testing TransformerTrainer data loaders...")
    
    try:
        # Create configuration
        config = TransformerTrainingConfig(
            batch_size=4,
            validation_split=0.3,
            model_config={
                'max_seq_length': 30
            },
            num_workers=0,  # Disable multiprocessing for testing
            pin_memory=False
        )
        
        trainer = TransformerTrainer(config)
        
        # Create sample data
        features, targets, time_features, volatility = create_sample_data(200, 8)
        
        # Create data loaders
        train_loader, val_loader = trainer.create_data_loaders(
            features, targets, time_features, volatility
        )
        
        # Test data loaders
        assert len(train_loader) > 0, "Train loader should not be empty"
        assert len(val_loader) > 0, "Validation loader should not be empty"
        
        # Test batch loading
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        
        # Verify batch structure
        for batch, name in [(train_batch, 'train'), (val_batch, 'val')]:
            assert 'features' in batch, f"{name} batch should contain features"
            assert 'targets' in batch, f"{name} batch should contain targets"
            assert batch['features'].shape[0] <= config.batch_size, f"{name} batch size should be <= {config.batch_size}"
        
        logger.info(f"Data loaders created: train_batches={len(train_loader)}, val_batches={len(val_loader)}")
        logger.info("✓ TransformerTrainer data loaders test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ TransformerTrainer data loaders test failed: {e}")
        return False

def test_transformer_trainer_training():
    """Test actual training functionality"""
    logger.info("Testing TransformerTrainer training...")
    
    try:
        # Create minimal configuration for fast testing
        config = TransformerTrainingConfig(
            batch_size=4,
            max_epochs=2,
            learning_rate=1e-3,
            mixed_precision=False,
            gradient_accumulation_steps=1,
            validation_split=0.2,
            validation_frequency=1,
            log_frequency=5,
            save_checkpoints=False,
            memory_optimization=False,
            num_workers=0,
            pin_memory=False,
            early_stopping_patience=10,
            model_config={'max_seq_length': 10}  # Set smaller sequence length
        )
        
        trainer = TransformerTrainer(config)
        
        # Create sample data
        features, targets, time_features, volatility = create_sample_data(300, 6)
        
        # Model configuration
        model_config = {
            'input_dim': 6,
            'd_model': 32,
            'num_heads': 2,
            'num_layers': 1,
            'max_seq_length': 10,  # Smaller sequence length for testing
            'num_classes': 3,
            'dropout': 0.1
        }
        
        # Train model
        results = trainer.train(
            features=features,
            targets=targets,
            time_features=time_features,
            volatility=volatility,
            model_config=model_config
        )
        
        # Verify training results
        assert 'training_metrics' in results, "Results should contain training_metrics"
        assert 'total_training_time' in results, "Results should contain total_training_time"
        assert len(results['training_metrics']) > 0, "Should have training metrics"
        
        # Check training metrics
        final_metrics = results['training_metrics'][-1]
        assert 'train_loss' in final_metrics, "Should have train_loss"
        assert 'train_accuracy' in final_metrics, "Should have train_accuracy"
        assert 'val_loss' in final_metrics, "Should have val_loss"
        assert 'val_accuracy' in final_metrics, "Should have val_accuracy"
        
        logger.info(f"Training completed successfully:")
        logger.info(f"  - Epochs: {len(results['training_metrics'])}")
        logger.info(f"  - Final train loss: {final_metrics['train_loss']:.4f}")
        logger.info(f"  - Final train accuracy: {final_metrics['train_accuracy']:.4f}")
        logger.info(f"  - Final val loss: {final_metrics['val_loss']:.4f}")
        logger.info(f"  - Final val accuracy: {final_metrics['val_accuracy']:.4f}")
        logger.info(f"  - Total time: {results['total_training_time']:.2f}s")
        
        # Test training summary
        summary = trainer.get_training_summary()
        assert 'total_epochs' in summary, "Summary should contain total_epochs"
        assert 'model_parameters' in summary, "Summary should contain model_parameters"
        
        logger.info("✓ TransformerTrainer training test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ TransformerTrainer training test failed: {e}")
        return False

def test_transformer_trainer_features():
    """Test additional TransformerTrainer features"""
    logger.info("Testing TransformerTrainer additional features...")
    
    try:
        # Create configuration with advanced features
        config = TransformerTrainingConfig(
            batch_size=4,
            max_epochs=1,
            mixed_precision=torch.cuda.is_available(),  # Enable if CUDA available
            gradient_accumulation_steps=2,
            use_warmup=True,
            warmup_steps=10,
            scheduler_type='cosine',
            early_stopping_patience=5,
            save_checkpoints=False,  # Disable for testing
            memory_optimization=torch.cuda.is_available()
        )
        
        trainer = TransformerTrainer(config)
        
        # Test configuration
        assert trainer.config.mixed_precision == torch.cuda.is_available(), "Mixed precision should match CUDA availability"
        assert trainer.device.type in ['cuda', 'cpu'], "Device should be cuda or cpu"
        
        # Test memory usage tracking
        memory_usage = trainer._get_memory_usage()
        assert isinstance(memory_usage, float), "Memory usage should be a float"
        assert memory_usage >= 0, "Memory usage should be non-negative"
        
        logger.info(f"Device: {trainer.device}")
        logger.info(f"Mixed precision: {config.mixed_precision}")
        logger.info(f"Memory usage: {memory_usage:.2f}GB")
        
        logger.info("✓ TransformerTrainer features test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ TransformerTrainer features test failed: {e}")
        return False

def main():
    """Run all TransformerTrainer tests"""
    logger.info("Starting TransformerTrainer Tests")
    logger.info("=" * 60)
    
    tests = [
        test_transformer_dataset,
        test_transformer_trainer_setup,
        test_transformer_trainer_data_loaders,
        test_transformer_trainer_training,
        test_transformer_trainer_features
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
    logger.info(f"TransformerTrainer Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All TransformerTrainer tests passed! Ready for integration.")
    else:
        logger.warning(f"⚠️  {total - passed} tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)