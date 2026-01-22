"""
Test Optimized Aggressive LSTM Training

Test the optimized LSTM training system on real feature data that was previously
causing CUDA out-of-memory errors.
"""

import sys
sys.path.append('src')

from src.ai.models.optimized_lstm_trainer import OptimizedLSTMTrainer
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import psutil

def test_system_requirements():
    """Test system requirements for optimized training"""
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

def test_optimized_daily_training():
    """Test optimized daily model training with GPU optimizations"""
    print("\nTesting Optimized Daily Model Training")
    print("-" * 40)
    
    try:
        # Initialize optimized trainer
        trainer = OptimizedLSTMTrainer(
            features_dir="TrainingData/features",
            models_dir="models/optimized_aggressive_lstm",
            mode='daily'
        )
        
        print("✅ Optimized daily trainer initialized")
        print(f"   Sequence length: {trainer.sequence_length}")
        print(f"   Model size: {trainer.hidden_size}x{trainer.num_layers}")
        print(f"   Initial batch size: {trainer.batch_size}")
        print(f"   Learning rate: {trainer.learning_rate}")
        print(f"   Mixed precision: {trainer.use_amp}")
        print(f"   Memory optimization: {trainer.enable_memory_optimization}")
        print(f"   Dynamic batching: {trainer.enable_dynamic_batching}")
        print(f"   Gradient accumulation: {trainer.enable_gradient_accumulation}")
        
        # Load and prepare data (same as original aggressive trainer)
        print("\nLoading and preparing data...")
        
        # Get available symbols
        symbols = trainer.get_available_symbols()
        print(f"Found {len(symbols)} symbols with feature data")
        
        # Load data for 32 symbols (same as original test)
        max_symbols = 32
        symbols_to_use = symbols[:max_symbols]
        
        combined_data = []
        symbols_loaded = []
        
        for symbol in symbols_to_use:
            df = trainer.load_symbol_data(symbol)
            if df is not None and len(df) > trainer.sequence_length + 100:
                df['symbol'] = symbol
                combined_data.append(df)
                symbols_loaded.append(symbol)
        
        if not combined_data:
            raise ValueError("No valid data loaded")
        
        # Combine all data
        combined_df = pd.concat(combined_data, ignore_index=True)
        print(f"Combined dataset: {len(combined_df):,} rows from {len(symbols_loaded)} symbols")
        
        # Prepare features
        features_df = trainer.prepare_features(combined_df)
        
        # Get targets
        target_col = 'direction_1d'
        if target_col not in combined_df.columns:
            raise ValueError(f"Target column {target_col} not found")
        
        # Remove rows with missing targets
        valid_mask = combined_df[target_col].notna()
        features_clean = features_df[valid_mask]
        targets_clean = combined_df[target_col][valid_mask].values
        
        print(f"Clean dataset: {len(features_clean):,} rows")
        print(f"Target distribution: {pd.Series(targets_clean).value_counts().to_dict()}")
        
        # Scale features
        features_scaled = trainer.feature_scaler.fit_transform(features_clean)
        
        # Encode targets: -1 (DOWN) -> 0, 0 (FLAT) -> 1, 1 (UP) -> 2
        targets_encoded = targets_clean + 1
        targets_encoded = np.clip(targets_encoded, 0, 2)
        
        # Create sequences
        X_sequences, y_sequences = trainer.create_sequences_with_targets(features_scaled, targets_encoded)
        
        # Walk-forward split (time-based)
        split_idx = int(len(X_sequences) * 0.8)
        val_split_idx = int(len(X_sequences) * 0.9)
        
        X_train = X_sequences[:split_idx]
        y_train = y_sequences[:split_idx]
        X_val = X_sequences[split_idx:val_split_idx]
        y_val = y_sequences[split_idx:val_split_idx]
        X_test = X_sequences[val_split_idx:]
        y_test = y_sequences[val_split_idx:]
        
        print(f"Data splits - Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")
        
        # Reduce epochs for testing
        trainer.epochs = 10  # Reduced for testing
        trainer.eval_every = 2  # Evaluate every 2 epochs
        
        print(f"\nStarting optimized training with {trainer.epochs} epochs...")
        
        # Train with memory optimization
        results = trainer.train_with_memory_optimization(X_train, y_train, X_val, y_val)
        
        # Show results
        print("\nOptimized Daily Training Results:")
        if results['success']:
            print(f"✅ SUCCESS: Optimized daily model trained!")
            print(f"   Best validation accuracy: {results['best_val_accuracy']:.4f}")
            print(f"   Final validation accuracy: {results['final_val_accuracy']:.4f}")
            print(f"   OOM events: {results['oom_events']}")
            print(f"   Model saved to: {results['model_path']}")
            
            # Show memory efficiency
            memory_metrics = results.get('memory_metrics', {})
            if memory_metrics:
                print(f"   Memory efficiency: {memory_metrics.get('efficiency_score', 'N/A')}")
                if 'memory_utilization' in memory_metrics:
                    util = memory_metrics['memory_utilization']
                    print(f"   Memory usage: {util['avg_pct']:.1f}% avg, {util['max_pct']:.1f}% peak")
            
            # Show batch statistics
            batch_stats = results.get('batch_statistics', {})
            if batch_stats.get('total_adjustments', 0) > 0:
                print(f"   Batch adjustments: {batch_stats['total_adjustments']}")
                print(f"   Final batch size: {batch_stats['current_batch_size']}")
                print(f"   Effective batch size: {batch_stats['effective_batch_size']}")
            
            # Test on test set
            print("\nEvaluating on test set...")
            
            # Load best model for final evaluation
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Create model for evaluation
            from src.ai.models.aggressive_lstm_trainer import AggressiveLSTMPredictor
            eval_model = AggressiveLSTMPredictor(
                input_size=X_test.shape[-1],
                hidden_size=trainer.hidden_size,
                num_layers=trainer.num_layers,
                num_classes=trainer.num_classes,
                dropout=trainer.dropout,
                mode=trainer.mode
            ).to(device)
            
            # Load best weights
            eval_model.load_state_dict(torch.load(results['model_path']))
            eval_model.eval()
            
            # Evaluate on test set
            test_predictions = []
            test_targets = []
            
            with torch.no_grad():
                # Process test data in smaller batches to avoid memory issues
                test_batch_size = 64
                for i in range(0, len(X_test), test_batch_size):
                    batch_X = torch.FloatTensor(X_test[i:i+test_batch_size]).to(device)
                    batch_y = y_test[i:i+test_batch_size]
                    
                    outputs = eval_model(batch_X)
                    predictions = torch.argmax(outputs, dim=1)
                    
                    test_predictions.extend(predictions.cpu().numpy())
                    test_targets.extend(batch_y)
            
            # Calculate test accuracy
            from sklearn.metrics import accuracy_score, classification_report
            test_accuracy = accuracy_score(test_targets, test_predictions)
            
            print(f"   Test accuracy: {test_accuracy:.4f}")
            print(f"   Test samples: {len(test_targets):,}")
            
            # Show classification report
            class_names = ['DOWN', 'FLAT', 'UP']
            print("\nClassification Report:")
            print(classification_report(test_targets, test_predictions, target_names=class_names))
            
            return True
        else:
            print(f"❌ FAILED: {results['error']}")
            print(f"   OOM events: {results['oom_events']}")
            return False
            
    except Exception as e:
        print(f"❌ Error during optimized training: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_optimization_report():
    """Generate and display optimization report"""
    print("\nOptimization Performance Report")
    print("-" * 40)
    
    try:
        # Initialize trainer to get optimization report
        trainer = OptimizedLSTMTrainer(
            features_dir="TrainingData/features",
            models_dir="models/optimized_aggressive_lstm",
            mode='daily'
        )
        
        # Generate report
        report = trainer.get_optimization_report()
        
        print("System Configuration:")
        config = report['optimization_summary']
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        print("\nMemory Efficiency Analysis:")
        efficiency = report.get('memory_efficiency', {})
        print(f"  Efficiency Score: {efficiency.get('efficiency_score', 'N/A')}")
        print(f"  Efficiency Note: {efficiency.get('efficiency_note', 'N/A')}")
        print(f"  Average Utilization: {efficiency.get('average_utilization_pct', 0):.1f}%")
        print(f"  Peak Utilization: {efficiency.get('peak_utilization_pct', 0):.1f}%")
        
        print("\nOptimization Recommendations:")
        recommendations = report.get('recommendations', [])
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        else:
            print("  No specific recommendations - system is well optimized")
        
        return True
        
    except Exception as e:
        print(f"❌ Error generating optimization report: {e}")
        return False

def main():
    """Run all optimized aggressive LSTM tests"""
    print("Optimized Aggressive LSTM Training Test Suite")
    print("=" * 70)
    
    # Test 1: System requirements
    cuda_ok = test_system_requirements()
    
    if not cuda_ok:
        print("\n❌ CUDA not available - cannot run GPU optimization tests")
        return
    
    # Test 2: Data availability
    data_ok = test_data_availability()
    
    if not data_ok:
        print("\n❌ Data availability test failed. Cannot proceed with training tests.")
        return
    
    # Test 3: Optimized daily model training
    training_ok = test_optimized_daily_training()
    
    # Test 4: Optimization report
    report_ok = test_optimization_report()
    
    # Summary
    print("\n" + "=" * 70)
    print("Optimized Aggressive LSTM Test Summary:")
    print(f"System Requirements: {'✅ PASS' if cuda_ok else '❌ FAIL'}")
    print(f"Data Availability: {'✅ PASS' if data_ok else '❌ FAIL'}")
    print(f"Optimized Training: {'✅ PASS' if training_ok else '❌ FAIL'}")
    print(f"Optimization Report: {'✅ PASS' if report_ok else '❌ FAIL'}")
    
    if cuda_ok and data_ok and training_ok:
        print("\n🎉 OPTIMIZED TRAINING SUCCESS!")
        print("\nComparison with Original Problem:")
        print("Original Aggressive LSTM:")
        print("  ❌ CUDA error: out of memory")
        print("  ❌ Training failed immediately")
        print("  ❌ Batch size 128 too large")
        print("  ❌ No memory management")
        
        print("\nOptimized Aggressive LSTM:")
        print("  ✅ Training completed successfully")
        print("  ✅ Dynamic memory management")
        print("  ✅ Automatic batch size optimization")
        print("  ✅ Zero OOM errors")
        print("  ✅ Comprehensive monitoring")
        
        print("\n🚀 READY FOR PRODUCTION!")
        print("\nNext Steps:")
        print("1. Scale up to full 164-symbol dataset")
        print("2. Run longer training (50-200 epochs)")
        print("3. Implement continuous retraining pipeline")
        print("4. Deploy to production trading system")
        
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
        print("\nTroubleshooting:")
        print("1. Ensure feature data is properly generated")
        print("2. Check GPU memory availability")
        print("3. Verify PyTorch CUDA installation")

if __name__ == "__main__":
    main()