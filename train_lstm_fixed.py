#!/usr/bin/env python3
"""
Quick fix for LSTM training with proper settings to avoid NaN loss
"""

import subprocess
import sys

def main():
    # Run with safer parameters
    cmd = [
        sys.executable, "train_lstm_production.py",
        "--epochs", "50",
        "--batch-size", "64",  # Smaller batch size
        "--learning-rate", "0.0001",  # Much lower learning rate
        "--weight-decay", "0.01",  # Higher weight decay for regularization
        "--test-mode"  # Use test mode for faster debugging
    ]
    
    print("Running LSTM training with safe parameters:")
    print(" ".join(cmd))
    print()
    
    subprocess.run(cmd)

if __name__ == "__main__":
    main()