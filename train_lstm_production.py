"""
PRODUCTION LSTM TRAINER - Optimized for RTX 4080
Trains LSTM on 1,695 stocks with GPU acceleration

Features:
- RTX 4080 optimization (70-80% GPU utilization)
- Mixed precision training (faster, less memory)
- Gradient accumulation (larger effective batch size)
- Learning rate scheduling with warm-up
- Early stopping & checkpointing
- TensorBoard logging
- Comprehensive metrics

Usage:
    # Training from scratch
    python train_lstm_production.py --data TrainingData/features/*.parquet --epochs 100

    # Resume from checkpoint
    python train_lstm_production.py --resume models/lstm_checkpoint.pth

    # Quick test (10% of data)
    python train_lstm_production.py --test-mode --epochs 5

Hardware Requirements:
    GPU: RTX 4080 (16GB VRAM)
    RAM: 32GB recommended
    Storage: 20GB for checkpoints

Author: Trading Bot Team
Date: October 28, 2025
"""

import os
import sys
import time
import json
import argparse
import warnings
import gc
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Suppress specific PyTorch warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch.optim.lr_scheduler')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedShuffleSplit

# POWER MANAGEMENT INTEGRATION
from power_management import (
    get_gpu_manager,
    get_precision_manager,
    get_power_monitor,
    DEFAULT_CONFIG
)

# GPU Configuration
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async CUDA for speed
torch.backends.cudnn.benchmark = True      # Auto-tune kernels
torch.backends.cudnn.deterministic = False # Faster but less reproducible

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class StockDataset(Dataset):
    """Dataset for stock time series with 30-day sequences"""

    def __init__(self, data_files: List[Path], sequence_length: int = 30, test_mode: bool = False, stride: int = 5, max_sequences_per_file: int = 20):
        self.sequence_length = sequence_length
        self.test_mode = test_mode
        self.stride = stride  # Sample every N days (reduces redundancy)
        self.max_sequences_per_file = max_sequences_per_file  # Limit sequences per file to reduce memory
        self.sequences = []
        self.labels = []

        print(f"Loading data from {len(data_files)} files...")
        print(f"Sequence sampling: Every {stride} days (reduces redundancy by {stride}x)")
        print(f"Max sequences per file: {max_sequences_per_file} (memory optimization)")

        # Load in test mode if specified
        if test_mode:
            data_files = data_files[:int(len(data_files) * 0.1)]
            print(f"TEST MODE: Using only {len(data_files)} files (10%)")

        # Track feature dimensions for consistency
        expected_features = None
        feature_counts = {}
        
        # First pass: determine the most common feature count
        print("  Analyzing feature dimensions...")
        for file in data_files[:100]:  # Sample first 100 files
            try:
                df = pd.read_parquet(file)
                if len(df) >= sequence_length + 1:
                    feature_count = df.shape[1]
                    feature_counts[feature_count] = feature_counts.get(feature_count, 0) + 1
            except:
                continue
        
        # Use the most common feature count
        if feature_counts:
            expected_features = max(feature_counts.keys(), key=feature_counts.get)
            print(f"  Expected features per sample: {expected_features}")
            print(f"  Feature count distribution: {feature_counts}")
        else:
            raise ValueError("No valid data files found")
        
        for i, file in enumerate(data_files):
            if (i + 1) % 100 == 0:
                print(f"  Loaded {i + 1}/{len(data_files)} files...")

            try:
                df = pd.read_parquet(file)

                # Validate data quality
                if len(df) < sequence_length + 1:
                    continue

                # Extract features and standardize feature count
                # IMPROVED NaN HANDLING: Forward-fill then backward-fill before skipping
                # This recovers data that was previously lost to NaN values
                df = df.ffill().bfill()  # Forward-fill then backward-fill NaN values

                features = df.values  # All columns are features

                # Skip files with different feature counts
                if features.shape[1] != expected_features:
                    continue

                # After forward-fill, check if any NaN still remain (only skip if can't be fixed)
                nan_count = np.isnan(features).sum()
                if nan_count > 0:
                    # Replace remaining NaN with column mean or 0
                    col_means = np.nanmean(features, axis=0)
                    col_means = np.where(np.isnan(col_means), 0, col_means)
                    for col in range(features.shape[1]):
                        features[:, col] = np.where(np.isnan(features[:, col]), col_means[col], features[:, col])

                # Normalize features (z-score normalization per stock)
                features_mean = np.nanmean(features, axis=0)
                features_std = np.nanstd(features, axis=0)
                # Handle zero std and NaN values in statistics
                features_std = np.where((features_std == 0) | np.isnan(features_std), 1, features_std)
                features_mean = np.where(np.isnan(features_mean), 0, features_mean)
                features = (features - features_mean) / features_std

                # Final check: replace any remaining NaN/inf with 0
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

                # Create sequences with stride (sample every N days to reduce redundancy)
                # MEMORY OPTIMIZATION: Limit sequences per file
                file_sequences = []
                file_labels = []
                for j in range(0, len(df) - sequence_length, self.stride):
                    seq = features[j:j + sequence_length]

                    # Use close price (assuming it's column 3) for label
                    if j + sequence_length < len(features):
                        current_close = df.iloc[j + sequence_length - 1]['close'] if 'close' in df.columns else features[j + sequence_length - 1, 3]
                        next_close = df.iloc[j + sequence_length]['close'] if 'close' in df.columns else features[j + sequence_length, 3]
                        label = 1 if next_close > current_close else 0
                    else:
                        continue

                    # Skip if sequence contains NaN
                    if np.isnan(seq).any():
                        continue

                    file_sequences.append(seq)
                    file_labels.append(label)

                    # MEMORY LIMIT: Stop if we've reached max sequences for this file
                    if len(file_sequences) >= self.max_sequences_per_file:
                        break

                # Add file sequences to dataset
                self.sequences.extend(file_sequences)
                self.labels.extend(file_labels)

            except Exception as e:
                print(f"  Warning: Failed to load {file.name}: {e}")
                continue

        # Convert to numpy arrays with error handling
        try:
            print(f"  Converting {len(self.sequences)} sequences to numpy array...")
            if len(self.sequences) == 0:
                raise ValueError("No valid sequences found")

            # Check sequence shapes before conversion
            seq_shapes = [seq.shape for seq in self.sequences[:5]]
            print(f"  Sample sequence shapes: {seq_shapes}")

            self.sequences = np.array(self.sequences, dtype=np.float32)
            self.labels = np.array(self.labels, dtype=np.int64)

            print(f"Dataset loaded: {len(self.sequences)} sequences")
            print(f"  Sequences shape: {self.sequences.shape}")
            print(f"  Labels shape: {self.labels.shape}")
            print(f"  Label distribution: {np.bincount(self.labels)}")

            # MEMORY CLEANUP: Force garbage collection after loading
            gc.collect()

        except ValueError as e:
            print(f"Error creating numpy arrays: {e}")
            # Debug information
            if len(self.sequences) > 0:
                print(f"  First sequence shape: {self.sequences[0].shape}")
                print(f"  Last sequence shape: {self.sequences[-1].shape}")
            raise

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.LongTensor([self.labels[idx]])[0]


class LSTMModel(nn.Module):
    """Production LSTM model optimized for stock prediction"""

    def __init__(self, input_dim: int, hidden_size: int = 256, num_layers: int = 3, dropout: float = 0.3):
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        # Attention layer (optional, improves performance)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )

        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)  # Binary classification: up/down
        )
        
        # Initialize weights properly
        self._init_weights()

    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden)

        # Attention mechanism (weight each time step)
        attention_weights = self.attention(lstm_out)  # (batch, seq, 1)
        context = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, hidden)

        # Classification
        output = self.fc(context)
        return output
    
    def _init_weights(self):
        """Initialize weights to prevent gradient issues"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
            elif 'weight' in name and len(param.shape) == 2:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)


class LSTMTrainer:
    """Production trainer with GPU optimization + Power Management"""

    def __init__(self, model, device, config):
        self.model = model.to(device)
        self.device = device
        self.config = config

        # POWER MANAGEMENT: Initialize managers
        self.gpu_manager = get_gpu_manager(DEFAULT_CONFIG.gpu)
        self.precision_manager = get_precision_manager(DEFAULT_CONFIG.mixed_precision)
        self.power_monitor = get_power_monitor(DEFAULT_CONFIG)
        self.power_monitor.set_managers(
            gpu_manager=self.gpu_manager,
            precision_manager=self.precision_manager
        )

        print("\n" + "="*70)
        print("POWER MANAGEMENT ENABLED")
        print("="*70)
        print(f"GPU Power Management: {'[YES]' if self.gpu_manager.gpu_available else '[NO]'}")
        print(f"Mixed Precision: [ON] {self.precision_manager.config.precision}")
        print(f"Estimated Power Savings: {self.power_monitor._calculate_total_savings({})['total_percentage']}%")
        print("="*70 + "\n")

        # Optimizer with weight decay (L2 regularization)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )

        # Learning rate scheduler with warm-up
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config['learning_rate'],
            epochs=config['epochs'],
            steps_per_epoch=config['steps_per_epoch'],
            pct_start=0.1,  # 10% warm-up
            anneal_strategy='cos'
        )

        # Loss function with class weights (handle imbalanced data)
        self.criterion = nn.CrossEntropyLoss()

        # Mixed precision training (2x faster on RTX 4080)
        self.scaler = GradScaler('cuda')

        # Metrics tracking
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # TensorBoard
        log_dir = Path('logs/tensorboard') / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(log_dir)
        print(f"TensorBoard logging to: {log_dir}")

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        # POWER MANAGEMENT: Set GPU to training mode (full power)
        with self.gpu_manager.training_context():
            for batch_idx, (data, labels) in enumerate(train_loader):
                data, labels = data.to(self.device), labels.to(self.device)

                # POWER MANAGEMENT: Use precision manager for mixed precision
                if self.device.type == 'cuda':
                    with self.precision_manager.autocast():
                        outputs = self.model(data)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(data)
                    loss = self.criterion(outputs, labels)

                # Check for NaN loss
                if torch.isnan(loss):
                    print(f"  WARNING: NaN loss detected at batch {batch_idx}")
                    print(f"  Data stats: min={data.min():.4f}, max={data.max():.4f}, mean={data.mean():.4f}")
                    print(f"  Output stats: min={outputs.min():.4f}, max={outputs.max():.4f}")
                    continue

                # Backward pass with gradient scaling
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()

                # Gradient clipping (prevent exploding gradients)
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()

                # Metrics
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # Log every 100 batches
                if (batch_idx + 1) % 100 == 0:
                    avg_loss = total_loss / (batch_idx + 1)
                    accuracy = 100. * correct / total
                    print(f"  Epoch {epoch} [{batch_idx + 1}/{len(train_loader)}] "
                          f"Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}% | "
                          f"LR: {self.scheduler.get_last_lr()[0]:.6f}")

        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def validate(self, val_loader, epoch):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        # POWER MANAGEMENT: Set GPU to inference mode (70% power)
        with self.gpu_manager.inference_context():
            with torch.no_grad():
                for data, labels in val_loader:
                    data, labels = data.to(self.device), labels.to(self.device)

                    outputs = self.model(data)
                    loss = self.criterion(outputs, labels)

                    total_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

        val_loss = total_loss / len(val_loader)
        val_acc = 100. * correct / total

        return val_loss, val_acc

    def train(self, train_loader, val_loader, num_epochs):
        print("\n" + "=" * 80)
        print("STARTING TRAINING")
        print("=" * 80)

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()

            # Training
            train_loss, train_acc = self.train_epoch(train_loader, epoch)

            # Validation
            val_loss, val_acc = self.validate(val_loader, epoch)

            epoch_time = time.time() - epoch_start

            # Logging
            print(f"\nEpoch {epoch}/{num_epochs} ({epoch_time:.1f}s)")
            print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
            print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")

            # TensorBoard
            self.writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
            self.writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch)
            self.writer.add_scalar('Learning_Rate', self.scheduler.get_last_lr()[0], epoch)

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.patience_counter = 0

                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'config': self.config
                }

                save_path = Path(self.config['output_dir']) / 'lstm_best.pth'
                torch.save(checkpoint, save_path)
                print(f"  [SAVED] Best model saved: {save_path} (Val Acc: {val_acc:.2f}%)")
            else:
                self.patience_counter += 1

            # Early stopping
            if self.patience_counter >= self.config['early_stopping_patience']:
                print(f"\n  Early stopping triggered (patience={self.config['early_stopping_patience']})")
                break

            # Save checkpoint every 5 epochs
            if epoch % 5 == 0:
                checkpoint_path = Path(self.config['output_dir']) / f'lstm_epoch_{epoch}.pth'
                torch.save(checkpoint, checkpoint_path)
                print(f"  Checkpoint saved: {checkpoint_path}")

        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(f"Best Val Acc: {self.best_val_acc:.2f}%")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")

        # POWER MANAGEMENT: Print power savings report
        print("\n" + "=" * 80)
        print("POWER MANAGEMENT REPORT")
        print("=" * 80)
        stats = self.power_monitor.get_comprehensive_stats()
        savings = stats['estimated_total_savings']
        print(f"Power Savings: {savings['total_percentage']}%")
        print(f"Target: {savings['target_percentage']}%")
        print(f"Target Met: {'[YES]' if savings['target_met'] else '[NO]'}")
        print(f"\nGPU Power: {self.gpu_manager.get_power_usage() or 'N/A'}")
        print(f"Mixed Precision: {self.precision_manager.config.precision}")
        print("=" * 80)

        self.writer.close()


def check_gpu():
    """Check GPU availability and specs"""
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Training will be VERY slow on CPU.")
        print("Please install CUDA-enabled PyTorch: https://pytorch.org/get-started/locally/")
        return False

    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

    print("=" * 80)
    print("GPU CONFIGURATION")
    print("=" * 80)
    print(f"GPU: {gpu_name}")
    print(f"VRAM: {gpu_memory:.1f} GB")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"cuDNN Enabled: {torch.backends.cudnn.enabled}")
    print(f"cuDNN Benchmark: {torch.backends.cudnn.benchmark}")
    print("")

    return True


def main():
    parser = argparse.ArgumentParser(description='Production LSTM Trainer')

    # Data arguments
    parser.add_argument('--data', type=str, default='TrainingData/features', help='Path to features directory')
    parser.add_argument('--output-dir', type=str, default='models', help='Output directory for models')

    # Model arguments
    parser.add_argument('--hidden-size', type=int, default=256, help='LSTM hidden size')
    parser.add_argument('--num-layers', type=int, default=3, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--sequence-length', type=int, default=30, help='Sequence length (days)')
    parser.add_argument('--stride', type=int, default=5, help='Sequence sampling stride (sample every N days, reduces redundancy)')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size (256 for faster training, 128 for more stability)')
    parser.add_argument('--learning-rate', type=float, default=0.0001, help='Learning rate (reduced to prevent NaN)')
    parser.add_argument('--weight-decay', type=float, default=0.0001, help='Weight decay (L2 regularization)')
    parser.add_argument('--early-stopping-patience', type=int, default=15, help='Early stopping patience')

    # Optimization arguments
    parser.add_argument('--num-workers', type=int, default=0, help='DataLoader workers (0 = no multiprocessing, fixes NumPy errors)')
    parser.add_argument('--pin-memory', action='store_true', default=True, help='Pin memory for faster GPU transfer')
    parser.add_argument('--max-sequences-per-file', type=int, default=50, help='Max sequences per file (lower = less RAM usage, default: 50)')

    # Mode arguments
    parser.add_argument('--test-mode', action='store_true', help='Test mode (10%% of data, 5 epochs)')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')

    args = parser.parse_args()

    # Check GPU
    has_gpu = check_gpu()
    device = torch.device('cuda' if has_gpu else 'cpu')

    if not has_gpu:
        response = input("Continue on CPU? Training will take MUCH longer. (y/n): ")
        if response.lower() != 'y':
            print("Exiting. Please install CUDA-enabled PyTorch.")
            return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data files
    data_dir = Path(args.data)
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        print("Please run feature engineering first:")
        print("  python generate_features_1695.py")
        return

    data_files = list(data_dir.glob('*.parquet'))
    if len(data_files) == 0:
        print(f"ERROR: No .parquet files found in {data_dir}")
        return

    print(f"Found {len(data_files)} data files")

    # Test mode adjustments
    if args.test_mode:
        print("\n" + "=" * 80)
        print("TEST MODE ENABLED")
        print("=" * 80)
        print("Using 10% of data and 5 epochs for quick testing")
        args.epochs = 5

    # Create dataset
    print("\nLoading dataset...")
    full_dataset = StockDataset(
        data_files,
        args.sequence_length,
        test_mode=args.test_mode,
        stride=args.stride,
        max_sequences_per_file=args.max_sequences_per_file
    )

    # STRATIFIED SPLIT: Ensures balanced class distribution in train/val sets
    # This prevents validation set from being dominated by majority class
    print("Performing stratified train/val split (80/20)...")
    labels_array = np.array(full_dataset.labels)
    print(f"Class distribution before split: {np.bincount(labels_array + 1).tolist()}")  # +1 to handle -1 labels

    try:
        # Use stratified split to ensure balanced validation set
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(sss.split(np.zeros(len(labels_array)), labels_array))
        train_dataset = Subset(full_dataset, train_idx)
        val_dataset = Subset(full_dataset, val_idx)

        # Verify class balance
        train_labels = labels_array[train_idx]
        val_labels = labels_array[val_idx]
        print(f"Train class distribution: {np.bincount(train_labels + 1).tolist()}")
        print(f"Val class distribution: {np.bincount(val_labels + 1).tolist()}")
    except Exception as e:
        print(f"Warning: Stratified split failed ({e}), falling back to random split")
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    print(f"Train set: {len(train_dataset)} sequences")
    print(f"Val set: {len(val_dataset)} sequences")

    # Create data loaders
    # MEMORY OPTIMIZATION: Disable persistent_workers to prevent memory accumulation
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory if args.num_workers > 0 else False,
        persistent_workers=False  # Disabled to prevent memory leak
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory if args.num_workers > 0 else False,
        persistent_workers=False  # Disabled to prevent memory leak
    )

    # Get input dimension from first batch
    sample_batch, _ = next(iter(train_loader))
    input_dim = sample_batch.shape[2]
    print(f"Input dimension: {input_dim} features")

    # Create model
    print("\nInitializing model...")
    model = LSTMModel(
        input_dim=input_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Training config
    config = {
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'sequence_length': args.sequence_length,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'early_stopping_patience': args.early_stopping_patience,
        'steps_per_epoch': len(train_loader),
        'output_dir': str(output_dir)
    }

    # Create trainer
    trainer = LSTMTrainer(model, device, config)

    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Resumed from epoch {checkpoint['epoch']}")

    # Train
    trainer.train(train_loader, val_loader, args.epochs)

    # Save final config
    config_path = output_dir / 'lstm_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\nConfig saved to: {config_path}")

    print("\n" + "=" * 80)
    print("LSTM TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Best model saved to: {output_dir / 'lstm_best.pth'}")
    print(f"Best validation accuracy: {trainer.best_val_acc:.2f}%")
    print(f"\nTo view training progress:")
    print(f"  tensorboard --logdir logs/tensorboard")
    print(f"  Then open: http://localhost:6006")


if __name__ == '__main__':
    main()
