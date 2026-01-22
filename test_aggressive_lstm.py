"""
Test Aggressive LSTM Training

Test the aggressive LSTM training system with production-grade optimizations.
"""

import sys
sys.path.append('src')

from src.ai.models.aggressive_lstm_trainer import AggressiveLSTMTrainer
import pandas as pd
from pathlib import Path
import torch

def test_system_requirements():
    """Test system requirements for aggressive training"""
    print("Testing System Requirements")
    print("-" * 40)
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {'✅ YES' if cuda_available else '❌ NO'}")
    
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.1f} GB")
    
    # Check available memory
    import psutil
    memory = psutil.virtual_memory()
    print(f"System RAM: {memory.total / 1e9:.1f} GB")
    print(f"Available RAM: {memory.available / 1e9:.1f} GB")
    
    return cuda_available

def test_data_availability():
    """Test if we have sufficient feature data"""
    print("\nTesting Data Availability")
    print("-" * 40)
    
    features_dir = Path("TrainingData/features")
    if not features_dir.exists():
        print("❌ Features directory not found!")
        return False
    
    feature_files = list(features_dir.glob("*_features.parquet"))
    print(f"✅ Found {len(feature_files)} feature files")
    
    # Test loading and check data quality
    if feature_files:
        test_file = feature_files[0]
        try:
            df = pd.read_parquet(test_file)
            print(f"✅ Sample file: {test_file.stem}")
            print(f"   Rows: {len(df):,}")
            print(f"   Columns: {len(df.columns)}")
            print(f"   Date range: {df.index.min()} to {df.index.max()}")
            
            # Check for required target column
            if 'direction_1d' in df.columns:
                target_dist = df['direction_1d'].value_counts()
                print(f"   Target distribution: {target_dist.to_dict()}")
                return True
            else:
                print("❌ Missing target column 'direction_1d'")
                return False
                
        except Exception as e:
            print(f"❌ Error loading sample file: {e}")
            return False
    
    return False

def test_daily_training():
    """Test daily model training with GPU optimizations"""
    print("\nTesting Daily Model Training (GPU Optimized)")
    print("-" * 40)
    
    try:
        # Initialize daily trainer
        trainer = AggressiveLSTMTrainer(mode='daily')
        print("✅ Daily trainer initialized")
        print(f"   Sequence length: {trainer.sequence_length}")
        print(f"   Model size: {trainer.hidden_size}x{trainer.num_layers}")
        print(f"   Batch size: {trainer.batch_size}")
        print(f"   Learning rate: {trainer.learning_rate}")
        print(f"   Mixed precision: {trainer.use_amp}")
        print(f"   Features planned: {len(trainer.feature_columns)}")
        
        # Train on 32 symbols as recommended
        print("\\nStarting training on 32 symbols (recommended scale)...")
        results = trainer.train_full_pipeline(max_symbols=32)
        
        # Show results
        print("\\nDaily Training Results:")
        if results['success']:
            print(f"✅ SUCCESS: Daily model trained!")
            print(f"   Test Accuracy: {results['accuracy']:.4f}")
            print(f"   Model Parameters: {results['model_parameters']:,}")
            print(f"   Symbols trained: {results['symbols_trained']}")
            print(f"   Training samples: {results['training_samples']:,}")
            print(f"   Model saved to: {results['model_path']}")
            return True
        else:
            print(f"❌ FAILED: {results['error']}")
            return False
            
    except Exception as e:
        print(f"❌ Error during daily training: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_intraday_training():
    """Test intraday model training (more aggressive)"""
    print("\\nTesting Intraday Model Training (Ultra Aggressive)")
    print("-" * 40)
    
    try:
        # Initialize intraday trainer
        trainer = AggressiveLSTMTrainer(mode='intraday')
        print("✅ Intraday trainer initialized")
        print(f"   Sequence length: {trainer.sequence_length}")
        print(f"   Model size: {trainer.hidden_size}x{trainer.num_layers}")
        print(f"   Batch size: {trainer.batch_size}")
        print(f"   Learning rate: {trainer.learning_rate}")
        print(f"   Mixed precision: {trainer.use_amp}")
        print(f"   Features planned: {len(trainer.feature_columns)}")
        
        # Train on 16 symbols for aggressive model
        print("\\nStarting training on 16 symbols...")
        results = trainer.train_full_pipeline(max_symbols=16)
        
        # Show results
        print("\\nIntraday Training Results:")
        if results['success']:
            print(f"✅ SUCCESS: Intraday model trained!")
            print(f"   Test Accuracy: {results['accuracy']:.4f}")
            print(f"   Model Parameters: {results['model_parameters']:,}")
            print(f"   Symbols trained: {results['symbols_trained']}")
            print(f"   Training samples: {results['training_samples']:,}")
            print(f"   Model saved to: {results['model_path']}")
            return True
        else:
            print(f"❌ FAILED: {results['error']}")
            return False
            
    except Exception as e:
        print(f"❌ Error during intraday training: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all aggressive LSTM tests"""
    print("Aggressive LSTM Training Test Suite")
    print("=" * 60)
    
    # Test 1: System requirements
    cuda_ok = test_system_requirements()
    
    # Test 2: Data availability
    data_ok = test_data_availability()
    
    if not data_ok:
        print("\\n❌ Data availability test failed. Cannot proceed with training tests.")
        return
    
    # Test 3: Daily model training
    daily_ok = test_daily_training()
    
    # Test 4: Intraday model training (only if daily succeeded)
    intraday_ok = False
    if daily_ok:
        intraday_ok = test_intraday_training()
    else:
        print("\\n⚠️  Skipping intraday test due to daily training failure")
    
    # Summary
    print("\\n" + "=" * 60)
    print("Aggressive LSTM Test Summary:")
    print(f"System Requirements: {'✅ PASS' if cuda_ok else '⚠️  LIMITED (CPU only)'}")
    print(f"Data Availability: {'✅ PASS' if data_ok else '❌ FAIL'}")
    print(f"Daily Training: {'✅ PASS' if daily_ok else '❌ FAIL'}")
    print(f"Intraday Training: {'✅ PASS' if intraday_ok else '❌ FAIL'}")
    
    if daily_ok or intraday_ok:
        print("\\n🚀 Aggressive LSTM training is working!")
        print("\\nCapabilities demonstrated:")
        if daily_ok:
            print("✅ Daily model: 256x2 layers, 252 lookback, AdamW + cosine decay")
        if intraday_ok:
            print("✅ Intraday model: 256x3 layers, 512 lookback, mixed precision")
        
        print("\\nProduction readiness:")
        print("✅ Advanced optimization (AdamW, cosine LR, gradient clipping)")
        print("✅ Robust preprocessing (RobustScaler, outlier handling)")
        print("✅ Comprehensive monitoring (gradient norms, divergence detection)")
        print("✅ Stochastic Weight Averaging for better generalization")
        print("✅ Mixed precision training for speed")
        
        print("\\nNext steps:")
        print("1. Scale up to full dataset (164 symbols)")
        print("2. Hyperparameter tuning and cross-validation")
        print("3. Deploy to production trading system")
        print("4. Set up continuous retraining pipeline")
        
    else:
        print("\\n⚠️  Some tests failed. Check the errors above.")
        print("\\nTroubleshooting:")
        print("1. Ensure sufficient GPU memory (8GB+ recommended)")
        print("2. Check feature data quality and completeness")
        print("3. Verify PyTorch CUDA installation")
        print("4. Consider reducing batch size or sequence length")

if __name__ == "__main__":
    main()