"""
Train on ALL Available Market Data

Trains LSTM model on:
- US Stocks: ~1,681 symbols
- Indian Stocks: ~577 symbols (.NS suffix)
- Canadian Stocks: ~21 symbols

Total: ~2,258 symbols, ~11 million rows, 26 years of data
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ai.training.training_pipeline import TrainingPipeline, TrainingConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_all_symbols() -> dict:
    """Get all available symbols from TrainingData and PastData directories."""
    training_path = Path("TrainingData/daily")
    past_path = Path("PastData/daily")

    symbols = {
        'us': [],
        'indian': [],
        'canadian': []
    }

    # Process TrainingData
    if training_path.exists():
        for file in training_path.glob("*.parquet"):
            name = file.stem
            # Remove _daily suffix
            if name.endswith("_daily"):
                name = name[:-6]

            if ".NS" in name:
                symbols['indian'].append(name)
            elif ".TO" in name:
                symbols['canadian'].append(name)
            else:
                symbols['us'].append(name)

    # Process PastData (Canadian stocks)
    if past_path.exists():
        for file in past_path.glob("*.parquet"):
            name = file.stem
            if name.endswith("_daily"):
                name = name[:-6]
            if name not in symbols['canadian']:
                symbols['canadian'].append(name)

    return symbols


def main():
    print("=" * 70)
    print("TRADINGBOT - MULTI-MARKET TRAINING PIPELINE")
    print("=" * 70)
    print()

    # Get all available symbols
    symbols_by_market = get_all_symbols()

    us_count = len(symbols_by_market['us'])
    indian_count = len(symbols_by_market['indian'])
    canadian_count = len(symbols_by_market['canadian'])
    total_symbols = us_count + indian_count + canadian_count

    print("Available Training Data:")
    print(f"  - US Stocks:       {us_count:,} symbols")
    print(f"  - Indian Stocks:   {indian_count:,} symbols")
    print(f"  - Canadian Stocks: {canadian_count:,} symbols")
    print(f"  - TOTAL:           {total_symbols:,} symbols")
    print()

    # Combine all symbols
    all_symbols = (
        symbols_by_market['us'] +
        symbols_by_market['indian'] +
        symbols_by_market['canadian']
    )

    print(f"Device: {'CUDA (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU Memory: {gpu_mem:.1f} GB")
    print()

    # Configure training for large dataset
    config = TrainingConfig(
        # Larger batch size for efficiency (if GPU has enough memory)
        batch_size=128 if torch.cuda.is_available() else 32,

        # More epochs since we have more data
        epochs=50,

        # Slightly higher learning rate for larger batches
        learning_rate=0.001,

        # Early stopping
        early_stopping_patience=15,

        # Sequence length for LSTM
        sequence_length=60,

        # Model architecture
        hidden_size=256,  # Larger model for more data
        num_layers=3,     # More layers
        dropout=0.3,

        # Mixed precision for speed
        use_mixed_precision=True if torch.cuda.is_available() else False,

        # Gradient clipping
        gradient_clip_value=1.0,

        # Learning rate scheduler
        lr_scheduler="cosine",
        warmup_epochs=5,

        # Paths
        model_dir="models",
        checkpoint_dir="models/checkpoints"
    )

    print("Training Configuration:")
    print(f"  - Batch Size:      {config.batch_size}")
    print(f"  - Max Epochs:      {config.epochs}")
    print(f"  - Learning Rate:   {config.learning_rate}")
    print(f"  - Hidden Size:     {config.hidden_size}")
    print(f"  - LSTM Layers:     {config.num_layers}")
    print(f"  - Sequence Length: {config.sequence_length}")
    print(f"  - Mixed Precision: {config.use_mixed_precision}")
    print()

    # Initialize pipeline
    pipeline = TrainingPipeline(
        config=config,
        use_snapshots=False,  # Disable for speed
        use_registry=False    # Disable for speed
    )

    # Start training
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    start_time = datetime.now()

    results = pipeline.run_full_training(
        symbols=all_symbols,
        description=f"Multi-market training: {us_count} US + {indian_count} Indian + {canadian_count} Canadian stocks"
    )

    training_time = (datetime.now() - start_time).total_seconds()

    # Print results
    print()
    print("=" * 70)
    print("TRAINING RESULTS")
    print("=" * 70)
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Training Time: {training_time/60:.1f} minutes")
    print()

    for model_name, result in results.items():
        status = "SUCCESS" if result.success else "FAILED"
        print(f"{model_name.upper()}:")
        print(f"  Status:        {status}")

        if result.success:
            print(f"  Test Accuracy: {result.accuracy:.4f} ({result.accuracy*100:.2f}%)")
            print(f"  Test Loss:     {result.test_loss:.4f}")
            print(f"  Val Loss:      {result.val_loss:.4f}")
            print(f"  Train Loss:    {result.train_loss:.4f}")
            print(f"  Epochs:        {result.epochs_trained}")
            print(f"  Train Time:    {result.training_time:.1f}s")
            print(f"  Model Path:    {result.model_path}")

            # Per-class accuracy
            if 'class_0_acc' in result.metrics:
                print(f"\n  Per-Class Accuracy:")
                print(f"    Down (0):    {result.metrics.get('class_0_acc', 0):.4f}")
                print(f"    Neutral (1): {result.metrics.get('class_1_acc', 0):.4f}")
                print(f"    Up (2):      {result.metrics.get('class_2_acc', 0):.4f}")
        else:
            print(f"  Error: {result.error_message}")
        print()

    print("=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
