"""
Full Scale MarketTransformer Training Script

This script demonstrates production-ready training of MarketTransformer models
using the TransformerTrainer with real financial data and comprehensive features.
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
        logging.FileHandler('logs/transformer_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_real_trading_data(data_path: str = "TrainingData/features") -> tuple:
    """
    Load real trading data for training using the same optimized approach as LSTM training
    
    Args:
        data_path: Path to feature parquet files
        
    Returns:
        Tuple of (features, targets, time_features, volatility, feature_names)
    """
    logger.info(f"Loading trading data from {data_path}")
    
    # Import the same data loading function used by LSTM - NO FALLBACK TO SYNTHETIC
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
    
    # Select symbols to use (same as LSTM training)
    priority_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'SPY', 'QQQ']
    available_symbols = [f.stem.replace('_features', '') for f in feature_files]
    
    symbols = []
    max_symbols = 5  # Start with 5 symbols for transformer training
    
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
            if df.empty or df.shape[0] < 100:
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
    
    # Create sequences for transformer (similar to LSTM but adapted for transformer)
    sequence_length = 252  # 1 year of trading days
    X_sequences = []
    y_sequences = []
    time_sequences = []
    vol_sequences = []
    
    # Group by symbol to maintain temporal order
    for symbol in symbols:
        symbol_data = combined_df[combined_df['symbol'] == symbol].copy()
        if len(symbol_data) < sequence_length + 1:
            logger.warning(f"Insufficient data for sequences in {symbol}")
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
        
        # Create time features from index (assuming it's datetime)
        if hasattr(symbol_data.index, 'to_pydatetime'):
            dates = pd.to_datetime(symbol_data.index)
        else:
            # Create synthetic dates if index is not datetime
            dates = pd.date_range('2020-01-01', periods=len(symbol_data), freq='D')
        
        time_features = np.column_stack([
            dates.dayofweek.values,
            dates.month.values - 1,  # 0-11
            dates.hour.values,
            np.random.randint(0, 4, len(dates))  # Market regime (synthetic)
        ]).astype(np.int32)
        
        # Calculate volatility from close prices
        if 'close' in symbol_data.columns:
            returns = symbol_data['close'].pct_change().fillna(0)
            volatility = returns.rolling(20).std().fillna(0.1).values.reshape(-1, 1).astype(np.float32)
        else:
            volatility = np.random.rand(len(features), 1).astype(np.float32) * 0.5 + 0.1
        
        # Create sequences
        for i in range(len(features) - sequence_length):
            X_sequences.append(features[i:i+sequence_length])
            y_sequences.append(targets[i+sequence_length])
            time_sequences.append(time_features[i:i+sequence_length])
            vol_sequences.append(volatility[i:i+sequence_length])
    
    if not X_sequences:
        raise ValueError("No sequences could be created from the data - check sequence length and data size")
    
    # Convert to numpy arrays
    X = np.array(X_sequences, dtype=np.float32)
    y = np.array(y_sequences, dtype=np.int64)
    time_features = np.array(time_sequences, dtype=np.int32)
    volatility = np.array(vol_sequences, dtype=np.float32)
    
    logger.info(f"Created sequences: X={X.shape}, y={y.shape}, time={time_features.shape}, vol={volatility.shape}")
    
    # Handle NaN values
    if np.isnan(X).any() or np.isnan(y).any():
        mask = ~np.isnan(X).any(axis=(1, 2)) & ~np.isnan(y)
        X, y = X[mask], y[mask]
        time_features = time_features[mask]
        volatility = volatility[mask]
        logger.warning(f"Filtered sequences with NaN values, remaining: {len(X)}")
    
    # Ensure we have data after filtering
    assert len(X) > 0, "No sequences after NaN filter"
    
    # Fast vectorized normalization (same as LSTM trainer)
    logger.info("Normalizing features...")
    X = X.astype(np.float32)
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Vectorized column-wise normalization
    X_reshaped = X.reshape(-1, X.shape[2])  # (samples*time, features)
    mean_vals = np.nanmean(X_reshaped, axis=0)
    std_vals = np.nanstd(X_reshaped, axis=0)
    std_vals = np.clip(std_vals, 1e-6, None)  # Prevent division by zero
    
    # Normalize all features at once
    X_reshaped = (X_reshaped - mean_vals) / std_vals
    X = X_reshaped.reshape(X.shape)  # Back to original shape
    
    # Validate class distribution
    num_classes = len(np.unique(y))
    assert num_classes > 1, f"Need multiple classes for classification, got {num_classes}"
    
    logger.info(f"Final data shapes:")
    logger.info(f"  Features: {X.shape}")
    logger.info(f"  Targets: {y.shape}")
    logger.info(f"  Time features: {time_features.shape}")
    logger.info(f"  Volatility: {volatility.shape}")
    logger.info(f"  Number of classes: {num_classes}")
    logger.info(f"  Target distribution: {np.bincount(y)}")
    
    return X, y, time_features, volatility, feature_columns

def create_synthetic_data(n_samples: int = 10000, n_features: int = 50) -> tuple:
    """
    Create synthetic financial data for training
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        
    Returns:
        Tuple of (features, targets, time_features, volatility, feature_names)
    """
    logger.info(f"Creating synthetic data: {n_samples} samples, {n_features} features")
    
    np.random.seed(42)
    
    # Generate realistic financial features
    features = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # Add some structure to make it more realistic
    for i in range(1, n_features):
        features[:, i] += 0.3 * features[:, i-1]  # Add some correlation
    
    # Add trend and seasonality
    trend = np.linspace(-0.5, 0.5, n_samples)
    seasonality = 0.2 * np.sin(2 * np.pi * np.arange(n_samples) / 252)  # Yearly cycle
    features[:, 0] += trend + seasonality
    
    # Generate targets based on features with some noise
    signal = (features[:, 0] + features[:, 1] + features[:, 2]) / 3
    noise = np.random.randn(n_samples) * 0.5
    combined_signal = signal + noise
    
    # Convert to 3-class classification (down, neutral, up)
    targets = np.digitize(combined_signal, bins=np.percentile(combined_signal, [33, 67]))
    targets = np.clip(targets, 0, 2)
    
    # Generate time features
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    time_features = np.column_stack([
        dates.dayofweek.values,
        dates.month.values - 1,
        dates.hour.values,
        np.random.randint(0, 4, n_samples)  # Market regime
    ]).astype(np.int32)
    
    # Generate volatility
    base_vol = 0.2
    vol_process = np.random.randn(n_samples) * 0.05
    volatility = np.maximum(base_vol + np.cumsum(vol_process) * 0.01, 0.05)
    volatility = volatility.reshape(-1, 1).astype(np.float32)
    
    # Feature names
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    return features, targets, time_features, volatility, feature_names

def create_production_config() -> TransformerTrainingConfig:
    """Create production-ready training configuration with optimized resource usage"""
    
    # Check available GPU memory to optimize settings
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU available: {gpu_memory_gb:.1f} GB memory")
        
        # Adjust batch size based on GPU memory (conservative for transformers)
        if gpu_memory_gb >= 16:
            batch_size = 32  # Conservative for transformer
            num_workers = 4
        elif gpu_memory_gb >= 8:
            batch_size = 16
            num_workers = 2
        else:
            batch_size = 8
            num_workers = 1
    else:
        logger.warning("No GPU available, using CPU with reduced settings")
        batch_size = 8
        num_workers = 1
    
    return TransformerTrainingConfig(
        # Model configuration will be set separately
        model_config={},
        
        # Training parameters - optimized for resource usage
        batch_size=batch_size,
        max_epochs=20,  # Reduced from 100 for faster training
        learning_rate=1e-4,
        weight_decay=1e-4,
        
        # Mixed precision and optimization - use only 80% of resources
        mixed_precision=gpu_available,  # Only use if GPU available
        gradient_accumulation_steps=2,  # Reduced from 4
        max_grad_norm=1.0,
        
        # Learning rate scheduling
        use_warmup=True,
        warmup_steps=500,  # Reduced from 2000
        scheduler_type='cosine',
        
        # Early stopping - more aggressive
        early_stopping_patience=5,  # Reduced from 15
        early_stopping_min_delta=1e-4,
        
        # Validation
        validation_split=0.2,
        validation_frequency=1,
        
        # Checkpointing
        save_checkpoints=True,
        checkpoint_frequency=5,  # More frequent saves
        keep_best_checkpoint=True,
        
        # Logging
        log_frequency=50,  # More frequent logging
        save_attention_maps=False,  # Disable to save memory
        
        # Memory optimization - conservative settings
        memory_optimization=True,
        pin_memory=gpu_available and gpu_memory_gb > 4,  # Only if enough memory
        num_workers=num_workers
    )

def create_model_config(n_features: int, sequence_length: int = 252) -> dict:
    """Create MarketTransformer model configuration"""
    
    return {
        'input_dim': n_features,
        'd_model': 256,
        'num_heads': 8,
        'num_layers': 6,
        'd_ff': 1024,
        'max_seq_length': sequence_length,
        'num_classes': 3,
        'dropout': 0.1
    }

def train_full_scale_transformer(
    features: np.ndarray,
    targets: np.ndarray,
    time_features: np.ndarray,
    volatility: np.ndarray,
    feature_names: list,
    model_name: str = "production_transformer",
    sequence_length: int = 252
) -> dict:
    """
    Train a full-scale MarketTransformer model
    
    Args:
        features: Feature array
        targets: Target array
        time_features: Time features
        volatility: Volatility data
        feature_names: List of feature names
        model_name: Name for the trained model
        sequence_length: Sequence length for training
        
    Returns:
        Training results dictionary
    """
    logger.info(f"Starting full-scale training for {model_name}")
    
    # Create configurations
    training_config = create_production_config()
    model_config = create_model_config(features.shape[1], sequence_length)
    training_config.model_config = model_config
    
    # Initialize trainer
    trainer = TransformerTrainer(training_config)
    
    # Log training setup
    logger.info(f"Training setup:")
    logger.info(f"  - Data shape: {features.shape}")
    logger.info(f"  - Sequence length: {sequence_length}")
    logger.info(f"  - Model parameters: ~{model_config['d_model'] * model_config['num_layers'] * 1000} params")
    logger.info(f"  - Batch size: {training_config.batch_size}")
    logger.info(f"  - Max epochs: {training_config.max_epochs}")
    logger.info(f"  - Learning rate: {training_config.learning_rate}")
    logger.info(f"  - Mixed precision: {training_config.mixed_precision}")
    
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
            'sequence_length': sequence_length
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
        raise

def register_trained_model(
    trainer: TransformerTrainer,
    training_results: dict,
    model_registry: ModelRegistry
) -> str:
    """
    Register the trained model in the model registry
    
    Args:
        trainer: Trained TransformerTrainer
        training_results: Training results dictionary
        model_registry: Model registry instance
        
    Returns:
        Model registration path
    """
    logger.info("Registering trained model in registry")
    
    # Create model metadata
    summary = training_results['training_summary']
    
    metadata = ModelMetadata(
        model_id=training_results['model_name'],
        model_type="transformer",
        version="1.0.0",
        training_date=datetime.now(),
        performance_metrics={
            'final_train_accuracy': summary['final_train_accuracy'],
            'final_val_accuracy': summary['final_val_accuracy'],
            'final_train_loss': summary['final_train_loss'],
            'final_val_loss': summary['final_val_loss'],
            'best_val_loss': summary['best_val_loss']
        },
        hyperparameters=training_results['model_config'],
        feature_importance=None,  # Could be added with attention analysis
        training_data_hash="full_scale_training",
        model_size_mb=0.0,  # Will be calculated during registration
        inference_latency_ms=0.0,  # Could be benchmarked
        training_time_seconds=summary['total_training_time'],
        validation_split=0.2,
        test_split=0.0,
        feature_columns=training_results['feature_names'],
        target_column='direction_1d',
        sequence_length=training_results['sequence_length'],
        architecture_config=training_results['model_config']
    )
    
    # Register model
    model_path = model_registry.register_model(trainer.model, metadata)
    
    logger.info(f"Model registered at: {model_path}")
    return model_path

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Full Scale MarketTransformer Training')
    parser.add_argument('--data-path', type=str, default='TrainingData/features',
                       help='Path to training data')
    parser.add_argument('--model-name', type=str, default='production_transformer',
                       help='Name for the trained model')
    parser.add_argument('--sequence-length', type=int, default=252,
                       help='Sequence length for training')
    parser.add_argument('--register-model', action='store_true',
                       help='Register trained model in registry')
    
    args = parser.parse_args()
    
    logger.info("Starting Full Scale MarketTransformer Training")
    logger.info("=" * 60)
    
    try:
        # Load data
        logger.info("Loading training data...")
        features, targets, time_features, volatility, feature_names = load_real_trading_data(args.data_path)
        
        logger.info(f"Data loaded successfully:")
        logger.info(f"  - Features: {features.shape}")
        logger.info(f"  - Targets: {targets.shape}")
        logger.info(f"  - Time features: {time_features.shape}")
        logger.info(f"  - Volatility: {volatility.shape}")
        logger.info(f"  - Target distribution: {np.bincount(targets)}")
        
        # Update sequence length based on actual data
        actual_sequence_length = features.shape[1] if len(features.shape) > 2 else args.sequence_length
        
        # Train model
        logger.info("Starting model training...")
        training_output = train_full_scale_transformer(
            features=features,
            targets=targets,
            time_features=time_features,
            volatility=volatility,
            feature_names=feature_names,
            model_name=args.model_name,
            sequence_length=actual_sequence_length
        )
        
        # Print training summary
        summary = training_output['results']['training_summary']
        logger.info("Training Summary:")
        logger.info(f"  - Total epochs: {summary['total_epochs']}")
        logger.info(f"  - Total steps: {summary['total_steps']}")
        logger.info(f"  - Best validation loss: {summary['best_val_loss']:.4f}")
        logger.info(f"  - Final train accuracy: {summary['final_train_accuracy']:.4f}")
        logger.info(f"  - Final validation accuracy: {summary['final_val_accuracy']:.4f}")
        logger.info(f"  - Total training time: {summary['total_training_time']:.2f}s")
        logger.info(f"  - Peak memory usage: {summary['peak_memory_usage_gb']:.2f}GB")
        logger.info(f"  - Model parameters: {summary['model_parameters']:,}")
        
        # Register model if requested
        if args.register_model:
            model_registry = ModelRegistry("models/registry")
            model_path = register_trained_model(
                training_output['trainer'],
                training_output['results'],
                model_registry
            )
            logger.info(f"Model registered successfully at: {model_path}")
        
        logger.info("=" * 60)
        logger.info("🎉 Full Scale MarketTransformer Training Completed Successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)