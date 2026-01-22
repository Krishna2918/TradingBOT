"""
Optimized MarketTransformer Training Script

This script trains MarketTransformer models with proper GPU utilization and resource management,
using the same optimized approach as the LSTM trainer.
"""

import sys
import logging
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from datetime import datetime
import json
import psutil
import gc

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from ai.models.transformer_trainer import TransformerTrainer, TransformerTrainingConfig
from ai.models.market_transformer import create_market_transformer
from ai.model_registry import ModelRegistry, ModelMetadata
from ai.multi_model_config import TransformerConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/transformer_training_optimized.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_system_resources():
    """Check and log system resources"""
    # CPU info
    cpu_count = psutil.cpu_count()
    cpu_percent = psutil.cpu_percent(interval=1)
    
    # Memory info
    memory = psutil.virtual_memory()
    memory_gb = memory.total / (1024**3)
    memory_available_gb = memory.available / (1024**3)
    
    # GPU info
    gpu_available = torch.cuda.is_available()
    gpu_info = {}
    if gpu_available:
        gpu_info = {
            'name': torch.cuda.get_device_name(0),
            'memory_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3),
            'memory_allocated_gb': torch.cuda.memory_allocated(0) / (1024**3),
            'memory_reserved_gb': torch.cuda.memory_reserved(0) / (1024**3)
        }
    
    logger.info("System Resources:")
    logger.info(f"  CPU: {cpu_count} cores, {cpu_percent}% usage")
    logger.info(f"  RAM: {memory_gb:.1f} GB total, {memory_available_gb:.1f} GB available")
    if gpu_available:
        logger.info(f"  GPU: {gpu_info['name']}")
        logger.info(f"  GPU Memory: {gpu_info['memory_gb']:.1f} GB total")
        logger.info(f"  GPU Allocated: {gpu_info['memory_allocated_gb']:.2f} GB")
        logger.info(f"  GPU Reserved: {gpu_info['memory_reserved_gb']:.2f} GB")
    else:
        logger.warning("  GPU: Not available")
    
    return {
        'cpu_count': cpu_count,
        'memory_gb': memory_gb,
        'memory_available_gb': memory_available_gb,
        'gpu_available': gpu_available,
        'gpu_info': gpu_info
    }

def load_real_trading_data_optimized(data_path: str = "TrainingData/features", max_symbols: int = 3) -> tuple:
    """
    Load real trading data with optimized memory usage
    
    Args:
        data_path: Path to feature parquet files
        max_symbols: Maximum number of symbols to use
        
    Returns:
        Tuple of (features, targets, time_features, volatility, feature_names)
    """
    logger.info(f"Loading trading data from {data_path} (max {max_symbols} symbols)")
    
    # Import the same data loading function used by LSTM
    sys.path.append('src')
    from ai.data.targets import ensure_direction_1d, validate_global_targets, encode_targets
    
    features_path = Path(data_path)
    if not features_path.exists():
        raise FileNotFoundError(f"Features directory not found: {data_path}")
    
    # Get available feature files
    feature_files = list(features_path.glob("*_features.parquet"))
    if not feature_files:
        raise FileNotFoundError(f"No feature files found in {data_path}")
    
    logger.info(f"Found {len(feature_files)} feature files")
    
    # Select symbols to use (prioritize liquid symbols)
    priority_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'SPY', 'QQQ']
    available_symbols = [f.stem.replace('_features', '') for f in feature_files]
    
    symbols = []
    for symbol in priority_symbols:
        if symbol in available_symbols:
            symbols.append(symbol)
            if len(symbols) >= max_symbols:
                break
    
    # Fill remaining slots with other symbols
    if len(symbols) < max_symbols:
        for symbol in available_symbols:
            if symbol not in symbols:
                symbols.append(symbol)
                if len(symbols) >= max_symbols:
                    break
    
    logger.info(f"Selected symbols for training: {symbols}")
    
    # Load and process data from all symbols
    all_data = []
    
    for symbol in symbols:
        feature_file = features_path / f"{symbol}_features.parquet"
        if not feature_file.exists():
            logger.warning(f"Feature file not found for {symbol}, skipping")
            continue
        
        try:
            df = pd.read_parquet(feature_file)
            logger.info(f"Loaded {symbol}: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Basic validation
            if df.empty or df.shape[0] < 500:  # Need more data for transformer
                logger.warning(f"Insufficient data for {symbol}, skipping")
                continue
            
            # Ensure consistent direction_1d targets
            df = ensure_direction_1d(df, close_col="close", neutral_band=0.005, symbol=symbol)
            
            # Add symbol identifier
            df['symbol'] = symbol
            all_data.append(df)
            
        except Exception as e:
            logger.error(f"Error loading {symbol}: {e}")
            continue
    
    if not all_data:
        raise ValueError("No valid data loaded from any symbol - check data directory and files")
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"Combined dataset: {combined_df.shape[0]} rows, {combined_df.shape[1]} columns")
    
    # Validate global targets
    validate_global_targets(combined_df)
    
    # Encode targets (-1,0,1 -> 0,1,2)
    combined_df['target'] = encode_targets(combined_df['direction_1d'].values)
    
    # Prepare features
    feature_columns = [col for col in combined_df.columns 
                      if col not in ['symbol', 'target', 'date', 'timestamp', 'direction_1d']]
    

    
    # For transformer, we need to return flat data, not sequences
    # The TransformerDataset will create sequences from flat data
    
    # Flatten the sequences back to individual samples
    X_flat = []
    y_flat = []
    time_flat = []
    vol_flat = []
    
    for i, symbol in enumerate(symbols):
        symbol_data = combined_df[combined_df['symbol'] == symbol].copy()
        if len(symbol_data) < 500:  # Skip if insufficient data
            continue
        
        # Sort by index to ensure temporal order
        symbol_data = symbol_data.sort_index()
        
        # Handle NaN values
        symbol_data[feature_columns] = symbol_data[feature_columns].replace([np.inf, -np.inf], np.nan)
        symbol_data[feature_columns] = symbol_data[feature_columns].ffill().bfill()
        symbol_data[feature_columns] = symbol_data[feature_columns].fillna(0.0)
        
        # Extract features and targets
        features = symbol_data[feature_columns].values
        targets = symbol_data['target'].values
        
        # Create time features from index
        if hasattr(symbol_data.index, 'to_pydatetime'):
            dates = pd.to_datetime(symbol_data.index)
        else:
            dates = pd.date_range('2020-01-01', periods=len(symbol_data), freq='D')
        
        time_features_symbol = np.column_stack([
            dates.dayofweek.values,
            dates.month.values - 1,  # 0-11
            dates.hour.values,
            np.random.randint(0, 4, len(dates))  # Market regime
        ]).astype(np.int32)
        
        # Calculate volatility from close prices
        if 'close' in symbol_data.columns:
            returns = symbol_data['close'].pct_change().fillna(0)
            volatility_symbol = returns.rolling(20).std().fillna(0.1).values.reshape(-1, 1).astype(np.float32)
        else:
            volatility_symbol = np.random.rand(len(features), 1).astype(np.float32) * 0.5 + 0.1
        
        # Add to flat arrays
        X_flat.append(features)
        y_flat.append(targets)
        time_flat.append(time_features_symbol)
        vol_flat.append(volatility_symbol)
    
    # Concatenate all symbols
    X = np.concatenate(X_flat, axis=0).astype(np.float32)
    y = np.concatenate(y_flat, axis=0).astype(np.int64)
    time_features = np.concatenate(time_flat, axis=0).astype(np.int32)
    volatility = np.concatenate(vol_flat, axis=0).astype(np.float32)
    
    logger.info(f"Created flat data: X={X.shape}, y={y.shape}, time={time_features.shape}, vol={volatility.shape}")
    
    # Handle NaN values
    if np.isnan(X).any() or np.isnan(y).any():
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X, y = X[mask], y[mask]
        time_features = time_features[mask]
        volatility = volatility[mask]
        logger.warning(f"Filtered samples with NaN values, remaining: {len(X)}")
    
    # Ensure we have data after filtering
    assert len(X) > 0, "No samples after NaN filter"
    
    # Fast vectorized normalization
    logger.info("Normalizing features...")
    X = X.astype(np.float32)
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Normalize features
    mean_vals = np.nanmean(X, axis=0)
    std_vals = np.nanstd(X, axis=0)
    std_vals = np.clip(std_vals, 1e-6, None)
    
    # Normalize all features at once
    X = (X - mean_vals) / std_vals
    
    # Validate class distribution
    num_classes = len(np.unique(y))
    assert num_classes > 1, f"Need multiple classes for classification, got {num_classes}"
    
    logger.info(f"Final flat data shapes:")
    logger.info(f"  Features: {X.shape}")
    logger.info(f"  Targets: {y.shape}")
    logger.info(f"  Time features: {time_features.shape}")
    logger.info(f"  Volatility: {volatility.shape}")
    logger.info(f"  Number of classes: {num_classes}")
    logger.info(f"  Target distribution: {np.bincount(y)}")
    
    return X, y, time_features, volatility, feature_columns

def create_optimized_config(system_resources: dict) -> TransformerTrainingConfig:
    """Create optimized training configuration based on system resources"""
    
    gpu_available = system_resources['gpu_available']
    memory_gb = system_resources['memory_available_gb']
    
    if gpu_available:
        gpu_memory_gb = system_resources['gpu_info']['memory_gb']
        logger.info(f"Optimizing for GPU with {gpu_memory_gb:.1f} GB memory")
        
        # Conservative settings for transformer
        if gpu_memory_gb >= 16:
            batch_size = 16
            num_workers = 2
        elif gpu_memory_gb >= 8:
            batch_size = 8
            num_workers = 1
        else:
            batch_size = 4
            num_workers = 0
    else:
        logger.warning("No GPU available, using CPU with minimal settings")
        batch_size = 4
        num_workers = 0
    
    # Force num_workers = 0 on Windows to avoid multiprocessing issues
    import platform
    if platform.system() == 'Windows':
        num_workers = 0
        logger.info("Windows detected: disabling DataLoader multiprocessing")
    
    # Use only 80% of available resources
    target_memory_usage = min(memory_gb * 0.8, 8.0)  # Cap at 8GB
    
    return TransformerTrainingConfig(
        model_config={},
        
        # Conservative training parameters
        batch_size=batch_size,
        max_epochs=15,  # Reduced for faster training
        learning_rate=5e-5,  # Lower learning rate for stability
        weight_decay=1e-4,
        
        # Mixed precision only if GPU available
        mixed_precision=gpu_available,
        gradient_accumulation_steps=2,
        max_grad_norm=0.5,  # Lower for stability
        
        # Learning rate scheduling
        use_warmup=True,
        warmup_steps=200,
        scheduler_type='cosine',
        
        # Aggressive early stopping
        early_stopping_patience=3,
        early_stopping_min_delta=1e-4,
        
        # Validation
        validation_split=0.2,
        validation_frequency=1,
        
        # Checkpointing
        save_checkpoints=True,
        checkpoint_frequency=3,
        keep_best_checkpoint=True,
        
        # Logging
        log_frequency=25,
        save_attention_maps=False,  # Disable to save memory
        
        # Memory optimization
        memory_optimization=True,
        pin_memory=gpu_available and memory_gb > 4,
        num_workers=num_workers
    )

def create_model_config(n_features: int, sequence_length: int = 126) -> dict:
    """Create optimized MarketTransformer model configuration"""
    
    return {
        'input_dim': n_features,
        'd_model': 128,  # Reduced from 256
        'num_heads': 4,  # Reduced from 8
        'num_layers': 3,  # Reduced from 6
        'd_ff': 256,     # Reduced from 1024
        'max_seq_length': sequence_length,
        'num_classes': 3,
        'dropout': 0.1
    }

def train_optimized_transformer(
    features: np.ndarray,
    targets: np.ndarray,
    time_features: np.ndarray,
    volatility: np.ndarray,
    feature_names: list,
    model_name: str = "optimized_transformer",
    system_resources: dict = None
) -> dict:
    """
    Train an optimized MarketTransformer model
    
    Args:
        features: Feature array
        targets: Target array
        time_features: Time features
        volatility: Volatility data
        feature_names: List of feature names
        model_name: Name for the trained model
        system_resources: System resource information
        
    Returns:
        Training results dictionary
    """
    logger.info(f"Starting optimized transformer training for {model_name}")
    
    # Create configurations
    training_config = create_optimized_config(system_resources)
    model_config = create_model_config(features.shape[1], 126)  # features.shape[1] is n_features, 126 is seq_len
    training_config.model_config = model_config
    
    # Initialize trainer
    trainer = TransformerTrainer(training_config)
    
    # Log training setup
    logger.info(f"Training setup:")
    logger.info(f"  Data shape: {features.shape}")
    logger.info(f"  Sequence length: {model_config['max_seq_length']}")
    logger.info(f"  Model d_model: {model_config['d_model']}")
    logger.info(f"  Model layers: {model_config['num_layers']}")
    logger.info(f"  Batch size: {training_config.batch_size}")
    logger.info(f"  Max epochs: {training_config.max_epochs}")
    logger.info(f"  Learning rate: {training_config.learning_rate}")
    logger.info(f"  Mixed precision: {training_config.mixed_precision}")
    logger.info(f"  GPU available: {torch.cuda.is_available()}")
    
    # Clear GPU cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Train the model
    start_time = datetime.now()
    
    try:
        results = trainer.train(
            features=features,
            targets=targets,
            time_features=time_features,
            volatility=volatility,
            model_config=model_config
        )
        
        training_time = datetime.now() - start_time
        logger.info(f"Training completed in {training_time}")
        
        # Get training summary
        summary = trainer.get_training_summary()
        
        # Save training plots
        trainer.save_training_plots(f"plots/{model_name}")
        
        # Prepare results
        training_results = {
            'model_name': model_name,
            'training_config': training_config.__dict__,
            'model_config': model_config,
            'training_summary': summary,
            'training_time_str': str(training_time),
            'feature_names': feature_names,
            'data_shape': features.shape,
            'sequence_length': model_config['max_seq_length'],
            'system_resources': system_resources
        }
        
        # Save results
        results_path = Path(f"results/{model_name}")
        results_path.mkdir(parents=True, exist_ok=True)
        
        with open(results_path / "training_results.json", 'w') as f:
            json.dump(training_results, f, indent=2, default=str)
        
        logger.info(f"Training results saved to {results_path}")
        
        return {
            'trainer': trainer,
            'results': training_results,
            'model': trainer.model
        }
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Optimized MarketTransformer Training')
    parser.add_argument('--data-path', type=str, default='TrainingData/features',
                       help='Path to training data')
    parser.add_argument('--model-name', type=str, default='optimized_transformer',
                       help='Name for the trained model')
    parser.add_argument('--max-symbols', type=int, default=3,
                       help='Maximum number of symbols to use')
    
    args = parser.parse_args()
    
    logger.info("Starting Optimized MarketTransformer Training")
    logger.info("=" * 60)
    
    try:
        # Check system resources
        system_resources = check_system_resources()
        
        # Create necessary directories
        Path("logs").mkdir(exist_ok=True)
        Path("plots").mkdir(exist_ok=True)
        Path("results").mkdir(exist_ok=True)
        
        # Load data with memory optimization
        logger.info("Loading training data...")
        features, targets, time_features, volatility, feature_names = load_real_trading_data_optimized(
            args.data_path, args.max_symbols
        )
        
        logger.info(f"Data loaded successfully:")
        logger.info(f"  Features: {features.shape}")
        logger.info(f"  Targets: {targets.shape}")
        logger.info(f"  Time features: {time_features.shape}")
        logger.info(f"  Volatility: {volatility.shape}")
        logger.info(f"  Target distribution: {np.bincount(targets)}")
        
        # Train model
        logger.info("Starting model training...")
        training_output = train_optimized_transformer(
            features=features,
            targets=targets,
            time_features=time_features,
            volatility=volatility,
            feature_names=feature_names,
            model_name=args.model_name,
            system_resources=system_resources
        )
        
        # Print training summary
        summary = training_output['results']['training_summary']
        logger.info("Training Summary:")
        logger.info(f"  Total epochs: {summary['total_epochs']}")
        logger.info(f"  Total steps: {summary['total_steps']}")
        logger.info(f"  Best validation loss: {summary['best_val_loss']:.4f}")
        logger.info(f"  Final train accuracy: {summary['final_train_accuracy']:.4f}")
        logger.info(f"  Final validation accuracy: {summary['final_val_accuracy']:.4f}")
        logger.info(f"  Total training time: {summary['total_training_time']:.2f}s")
        logger.info(f"  Peak memory usage: {summary['peak_memory_usage_gb']:.2f}GB")
        logger.info(f"  Model parameters: {summary['model_parameters']:,}")
        
        logger.info("=" * 60)
        logger.info("🎉 Optimized MarketTransformer Training Completed Successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)