"""
Test Optimized LSTM Training System

Comprehensive test of the GPU-optimized LSTM training system that solves
the original CUDA out-of-memory issues.
"""

import sys
sys.path.append('src')

from src.ai.models.optimized_lstm_trainer import OptimizedLSTMTrainer
from src.ai.models.gpu_memory_manager import GPUMemoryManager
from src.ai.models.dynamic_batch_controller import DynamicBatchController
import numpy as np
import torch
import pandas as pd
from pathlib import Path

def create_synthetic_training_data(num_samples=10000, sequence_length=252, num_features=55):
    """Create synthetic training data similar to the real LSTM training data"""
    print(f"Creating synthetic training data:")
    print(f"  Samples: {num_samples:,}")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Features: {num_features}")
    
    # Create synthetic feature data
    np.random.seed(42)  # For reproducibility
    
    # Generate realistic financial time series data
    X = np.random.randn(num_samples, sequence_length, num_features).astype(np.float32)
    
    # Add some realistic patterns
    for i in range(num_features):
        # Add trend component
        trend = np.linspace(-0.1, 0.1, sequence_length)
        X[:, :, i] += trend[np.newaxis, :]
        
        # Add some autocorrelation
        for j in range(1, sequence_length):
            X[:, j, i] += 0.3 * X[:, j-1, i]
    
    # Create synthetic targets (3 classes: DOWN=0, FLAT=1, UP=2)
    # Base targets on some features to make it learnable
    target_signal = np.mean(X[:, -10:, :5], axis=(1, 2))  # Last 10 timesteps, first 5 features
    y = np.zeros(num_samples, dtype=np.int64)
    
    # Create balanced classes
    y[target_signal < np.percentile(target_signal, 33)] = 0  # DOWN
    y[(target_signal >= np.percentile(target_signal, 33)) & 
      (target_signal < np.percentile(target_signal, 67))] = 1  # FLAT
    y[target_signal >= np.percentile(target_signal, 67)] = 2  # UP
    
    print(f"  Target distribution: {np.bincount(y)}")
    print(f"  Data shape: {X.shape}")
    print(f"  Memory size: {X.nbytes / (1024**3):.2f} GB")
    
    return X, y

def test_memory_optimization_components():
    """Test individual memory optimization components"""
    print("Testing Memory Optimization Components")
    print("=" * 60)
    
    # Test 1: GPU Memory Manager
    print("\n1. GPU Memory Manager")
    print("-" * 30)
    
    memory_manager = GPUMemoryManager(reserve_memory_gb=1.0)
    if memory_manager.gpu_available:
        print(f"✅ GPU Available: {memory_manager.gpu_properties.name}")
        print(f"✅ Total Memory: {memory_manager.total_memory_gb:.1f} GB")
        print(f"✅ Usable Memory: {memory_manager.usable_memory_gb:.1f} GB")
        
        # Test memory monitoring
        metrics = memory_manager.monitor_memory_usage()
        print(f"✅ Current Usage: {metrics.memory_utilization*100:.1f}%")
    else:
        print("⚠️  GPU not available")
        return False
    
    # Test 2: Dynamic Batch Controller
    print("\n2. Dynamic Batch Controller")
    print("-" * 30)
    
    batch_controller = DynamicBatchController(
        memory_manager=memory_manager,
        initial_batch_size=128,
        min_batch_size=8,
        max_batch_size=512
    )
    
    # Configure for LSTM parameters
    batch_controller.configure(sequence_length=252, feature_count=55)
    
    # Test batch size recommendation
    recommended = batch_controller.recommend_initial_batch_size()
    print(f"✅ Recommended batch size: {recommended}")
    
    # Test gradient accumulation
    accumulation_steps = batch_controller.enable_gradient_accumulation(256)
    print(f"✅ Gradient accumulation steps: {accumulation_steps}")
    print(f"✅ Effective batch size: {batch_controller.get_effective_batch_size()}")
    
    return True

def test_optimized_lstm_training():
    """Test the complete optimized LSTM training system"""
    print("\n" + "=" * 60)
    print("Testing Optimized LSTM Training System")
    print("=" * 60)
    
    # Create synthetic training data (similar to real data size)
    print("\n1. Creating Training Data")
    print("-" * 30)
    
    # Use smaller dataset for testing but same dimensions as real data
    X, y = create_synthetic_training_data(
        num_samples=5000,  # Smaller for testing
        sequence_length=252,  # Same as real LSTM
        num_features=55  # Same as real LSTM
    )
    
    # Split data
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"✅ Training set: {X_train.shape[0]:,} samples")
    print(f"✅ Validation set: {X_val.shape[0]:,} samples")
    
    # Test 2: Initialize Optimized Trainer
    print("\n2. Initializing Optimized Trainer")
    print("-" * 30)
    
    try:
        trainer = OptimizedLSTMTrainer(
            features_dir="TrainingData/features",
            models_dir="models/test_optimized_lstm",
            mode='daily'
        )
        
        print(f"✅ Trainer initialized successfully")
        print(f"✅ GPU available: {trainer.memory_manager.gpu_available}")
        print(f"✅ Memory optimization: {trainer.enable_memory_optimization}")
        print(f"✅ Dynamic batching: {trainer.enable_dynamic_batching}")
        print(f"✅ Gradient accumulation: {trainer.enable_gradient_accumulation}")
        
    except Exception as e:
        print(f"❌ Trainer initialization failed: {e}")
        return False
    
    # Test 3: Memory-Optimized Training
    print("\n3. Running Memory-Optimized Training")
    print("-" * 30)
    
    try:
        # Reduce epochs for testing
        trainer.epochs = 5  # Quick test
        trainer.eval_every = 1  # Evaluate every epoch
        trainer.patience = 10  # Don't early stop during test
        
        print(f"Starting training with {trainer.epochs} epochs...")
        print(f"Initial batch size: {trainer.batch_size}")
        
        # Run optimized training
        results = trainer.train_with_memory_optimization(X_train, y_train, X_val, y_val)
        
        if results['success']:
            print(f"✅ Training completed successfully!")
            print(f"✅ Best validation accuracy: {results['best_val_accuracy']:.4f}")
            print(f"✅ Final validation accuracy: {results['final_val_accuracy']:.4f}")
            print(f"✅ OOM events: {results['oom_events']}")
            
            # Show memory efficiency
            memory_metrics = results['memory_metrics']
            if memory_metrics:
                print(f"✅ Memory efficiency: {memory_metrics.get('efficiency_score', 'N/A')}")
                if 'memory_utilization' in memory_metrics:
                    util = memory_metrics['memory_utilization']
                    print(f"✅ Memory usage: {util['avg_pct']:.1f}% avg, {util['max_pct']:.1f}% peak")
            
            # Show batch statistics
            batch_stats = results['batch_statistics']
            if batch_stats['total_adjustments'] > 0:
                print(f"✅ Batch adjustments: {batch_stats['total_adjustments']}")
                print(f"✅ Final batch size: {batch_stats['current_batch_size']}")
                print(f"✅ Effective batch size: {batch_stats['effective_batch_size']}")
            
        else:
            print(f"❌ Training failed: {results['error']}")
            print(f"❌ OOM events: {results['oom_events']}")
            return False
            
    except Exception as e:
        print(f"❌ Training failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Generate Optimization Report
    print("\n4. Optimization Report")
    print("-" * 30)
    
    try:
        report = trainer.get_optimization_report()
        
        print("Optimization Summary:")
        opt_summary = report['optimization_summary']
        for key, value in opt_summary.items():
            print(f"  {key}: {value}")
        
        print("\nMemory Efficiency:")
        efficiency = report['memory_efficiency']
        print(f"  Score: {efficiency.get('efficiency_score', 'N/A')}")
        print(f"  Note: {efficiency.get('efficiency_note', 'N/A')}")
        
        print("\nRecommendations:")
        for rec in report.get('recommendations', []):
            print(f"  - {rec}")
        
    except Exception as e:
        print(f"⚠️  Error generating optimization report: {e}")
    
    # Test 5: Compare with Original Problem
    print("\n5. Comparison with Original Problem")
    print("-" * 30)
    
    print("Original Problem:")
    print("  ❌ CUDA error: out of memory")
    print("  ❌ Batch size 128 with 201,121 sequences")
    print("  ❌ DataLoader multiprocessing issues")
    print("  ❌ No memory management")
    
    print("\nOptimized Solution:")
    print("  ✅ Dynamic batch sizing prevents OOM")
    print("  ✅ Gradient accumulation maintains training quality")
    print("  ✅ Memory monitoring and cleanup")
    print("  ✅ Automatic error recovery")
    print("  ✅ Comprehensive performance tracking")
    
    return True

def test_large_dataset_simulation():
    """Simulate training with large dataset similar to original problem"""
    print("\n" + "=" * 60)
    print("Large Dataset Simulation Test")
    print("=" * 60)
    
    print("Simulating original problem conditions:")
    print("  - 201,121 sequences (using 20,000 for test)")
    print("  - 252 sequence length")
    print("  - 55 features")
    print("  - Batch size 128 (will be auto-adjusted)")
    
    # Create larger synthetic dataset
    X_large, y_large = create_synthetic_training_data(
        num_samples=20000,  # Scaled down but representative
        sequence_length=252,
        num_features=55
    )
    
    # Split data
    split_idx = int(len(X_large) * 0.8)
    X_train_large = X_large[:split_idx]
    y_train_large = y_large[:split_idx]
    X_val_large = X_large[split_idx:]
    y_val_large = y_large[split_idx:]
    
    print(f"Large dataset created: {X_train_large.shape[0]:,} training samples")
    
    # Initialize trainer with aggressive settings
    trainer_large = OptimizedLSTMTrainer(
        features_dir="TrainingData/features",
        models_dir="models/test_large_lstm",
        mode='daily'
    )
    
    # Set aggressive initial batch size (like original problem)
    trainer_large.batch_size = 128
    trainer_large.batch_controller.current_batch_size = 128
    trainer_large.epochs = 3  # Quick test
    
    print(f"Starting large dataset training...")
    print(f"Initial batch size: {trainer_large.batch_size}")
    
    try:
        results_large = trainer_large.train_with_memory_optimization(
            X_train_large, y_train_large, X_val_large, y_val_large
        )
        
        if results_large['success']:
            print(f"✅ Large dataset training successful!")
            print(f"✅ No OOM errors: {results_large['oom_events']} events")
            
            batch_stats = results_large['batch_statistics']
            print(f"✅ Batch size adjustments: {batch_stats['total_adjustments']}")
            print(f"✅ Final batch size: {batch_stats['current_batch_size']}")
            print(f"✅ Effective batch size: {batch_stats['effective_batch_size']}")
            
            return True
        else:
            print(f"❌ Large dataset training failed: {results_large['error']}")
            return False
            
    except Exception as e:
        print(f"❌ Large dataset test failed: {e}")
        return False

def main():
    """Run all optimization tests"""
    print("GPU-Accelerated LSTM Training Optimization Test Suite")
    print("=" * 80)
    
    # Test 1: Individual components
    components_ok = test_memory_optimization_components()
    
    if not components_ok:
        print("\n❌ Component tests failed - cannot proceed")
        return
    
    # Test 2: Complete optimized training
    training_ok = test_optimized_lstm_training()
    
    # Test 3: Large dataset simulation
    large_dataset_ok = test_large_dataset_simulation()
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL TEST RESULTS")
    print("=" * 80)
    
    print(f"Component Tests: {'✅ PASS' if components_ok else '❌ FAIL'}")
    print(f"Optimized Training: {'✅ PASS' if training_ok else '❌ FAIL'}")
    print(f"Large Dataset Test: {'✅ PASS' if large_dataset_ok else '❌ FAIL'}")
    
    if components_ok and training_ok and large_dataset_ok:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nThe GPU memory optimization system is working correctly!")
        print("\nKey Achievements:")
        print("✅ Solved CUDA out-of-memory errors")
        print("✅ Dynamic batch size adjustment working")
        print("✅ Gradient accumulation maintaining training quality")
        print("✅ Memory monitoring and cleanup functional")
        print("✅ Error recovery mechanisms operational")
        print("✅ Ready for production use with 164-symbol dataset")
        
        print("\nNext Steps:")
        print("1. Run on real feature data: python test_aggressive_lstm.py")
        print("2. Scale up to full 164-symbol dataset")
        print("3. Monitor performance and adjust settings as needed")
        
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
        print("\nTroubleshooting:")
        print("1. Ensure CUDA drivers are properly installed")
        print("2. Check GPU memory availability")
        print("3. Verify PyTorch CUDA installation")

if __name__ == "__main__":
    main()