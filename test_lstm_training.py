"""
Test LSTM Training

Test the LSTM training process on a small subset of symbols.
"""

import sys
sys.path.append('src')

from src.ai.models.lstm_trainer import LSTMTrainer
import pandas as pd
from pathlib import Path

def test_data_availability():
    """Test if we have feature data available"""
    print("Testing Data Availability")
    print("-" * 30)
    
    features_dir = Path("TrainingData/features")
    if not features_dir.exists():
        print("❌ Features directory not found!")
        return False
    
    feature_files = list(features_dir.glob("*_features.parquet"))
    print(f"✅ Found {len(feature_files)} feature files")
    
    # Test loading one file
    if feature_files:
        test_file = feature_files[0]
        try:
            df = pd.read_parquet(test_file)
            print(f"✅ Successfully loaded {test_file.stem}: {len(df)} rows, {len(df.columns)} columns")
            print(f"   Date range: {df.index.min()} to {df.index.max()}")
            return True
        except Exception as e:
            print(f"❌ Error loading {test_file}: {e}")
            return False
    
    return False

def test_lstm_training():
    """Test LSTM training on a few symbols"""
    print("\nTesting LSTM Training")
    print("-" * 30)
    
    try:
        # Initialize trainer
        trainer = LSTMTrainer()
        print("✅ LSTM Trainer initialized")
        
        # Test on first 3 symbols (small test)
        print("Starting training on 3 symbols...")
        results = trainer.train_full_pipeline(max_symbols=3)
        
        # Show results
        print("\nTraining Results:")
        if results['success']:
            print(f"✅ SUCCESS: LSTM model trained!")
            print(f"   Test Accuracy: {results['accuracy']:.4f}")
            print(f"   Symbols trained: {results['symbols_trained']}")
            print(f"   Training samples: {results['training_samples']}")
            print(f"   Model saved to: {results['model_path']}")
            return True
        else:
            print(f"❌ FAILED: {results['error']}")
            return False
            
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("LSTM Training Test Suite")
    print("=" * 50)
    
    # Test 1: Data availability
    data_ok = test_data_availability()
    
    if not data_ok:
        print("\n❌ Data availability test failed. Cannot proceed with training test.")
        return
    
    # Test 2: LSTM training
    training_ok = test_lstm_training()
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"Data Availability: {'✅ PASS' if data_ok else '❌ FAIL'}")
    print(f"LSTM Training: {'✅ PASS' if training_ok else '❌ FAIL'}")
    
    if data_ok and training_ok:
        print("\n🎉 All tests passed! LSTM training is working correctly.")
        print("\nNext steps:")
        print("1. Train on more symbols if results look good")
        print("2. Tune hyperparameters if needed")
        print("3. Proceed to GRU model training")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()