"""
Start training with optimized memory settings
This script uses the fixed train_lstm_production.py with:
- Limited sequences per file (50 instead of unlimited)
- NaN handling improvements
- Memory cleanup and garbage collection
- Disabled persistent workers
"""

import subprocess
import sys
from pathlib import Path

def start_training(test_mode=False):
    """Start optimized LSTM training"""

    # Base command
    cmd = [
        sys.executable,
        'train_lstm_production.py',
        '--data', 'TrainingData/features',
        '--output-dir', 'models',
        '--epochs', '100',
        '--batch-size', '128',  # Reduced from 256 for stability
        '--learning-rate', '0.0001',
        '--max-sequences-per-file', '50',  # NEW: Limits memory usage
        '--stride', '5',  # Sample every 5 days
        '--num-workers', '0',  # No multiprocessing (avoids NumPy errors)
        '--early-stopping-patience', '10'
    ]

    if test_mode:
        cmd.append('--test-mode')
        print("="*70)
        print("STARTING TEST MODE TRAINING")
        print("="*70)
        print("- Using 10% of data")
        print("- 5 epochs")
        print("- Max 50 sequences per file")
        print("="*70)
    else:
        print("="*70)
        print("STARTING FULL TRAINING (OPTIMIZED)")
        print("="*70)
        print("- Max 50 sequences per file (memory optimization)")
        print("- Batch size: 128")
        print("- NaN handling improvements")
        print("- Memory cleanup enabled")
        print("="*70)

    # Run training
    try:
        result = subprocess.run(cmd, check=True)
        print("\n" + "="*70)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("="*70)
        return 0
    except subprocess.CalledProcessError as e:
        print("\n" + "="*70)
        print(f"TRAINING FAILED: {e}")
        print("="*70)
        return 1
    except KeyboardInterrupt:
        print("\n" + "="*70)
        print("TRAINING INTERRUPTED BY USER")
        print("="*70)
        return 2

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Start optimized LSTM training')
    parser.add_argument('--test', action='store_true', help='Run in test mode (10% data, 5 epochs)')
    args = parser.parse_args()

    sys.exit(start_training(test_mode=args.test))
