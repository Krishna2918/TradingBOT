"""
PRODUCTION TRANSFORMER TRAINER - Optimized for RTX 4080
Trains Market Transformer on 1,677 stocks with GPU acceleration

Features:
- RTX 4080 optimization (70-80% GPU utilization)
- Mixed precision training (faster, less memory)
- 90-day sequences for long-term pattern recognition
- Learning rate warm-up (4000 steps)
- Attention map visualization
- Early stopping & checkpointing
- TensorBoard logging
- Comprehensive metrics

Usage:
    # Training from scratch
    python train_transformer_production.py --data TrainingData/features/*.parquet --epochs 100

    # Resume from checkpoint
    python train_transformer_production.py --resume models/transformer_checkpoint.pth

    # Quick test (10% of data)
    python train_transformer_production.py --test-mode --epochs 5

Hardware Requirements:
    GPU: RTX 4080 (16GB VRAM)
    RAM: 32GB recommended
    Storage: 25GB for checkpoints

Author: Trading Bot Team
Date: October 28, 2025
"""

import os
import sys
import time
import json
import argparse
import warnings
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

# Import custom model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.ai.models.market_transformer import MarketTransformer

# GPU Configuration
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async CUDA for speed
torch.backends.cudnn.benchmark = True      # Auto-tune kernels
torch.backends.cudnn.deterministic = False # Faster but less reproducible

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class TransformerDataset(Dataset):
    """Dataset for stock time series with 90-day sequences (3 months)"""

    def __init__(self, data_files: List[Path], sequence_length: int = 90, test_mode: bool = False):
        self.sequence_length = sequence_length
        self.test_mode = test_mode
        self.sequences = []
        self.labels = []
        self.time_features = []  # Day, month, hour, regime
        self.volatility = []  # For volatility-aware attention

        print(f"Loading data from {len(data_files)} files...")

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
            print("  ERROR: Could not determine feature count!")
            return

        # Second pass: load sequences
        print("  Loading sequences...")
        loaded_files = 0
        skipped_files = 0
        total_sequences = 0

        for file_idx, file in enumerate(data_files):
            if (file_idx + 1) % 100 == 0:
                print(f"    Progress: {file_idx + 1}/{len(data_files)} files, "
                      f"{total_sequences} sequences created")

            try:
                df = pd.read_parquet(file)

                # Skip if insufficient data
                if len(df) < sequence_length + 1:
                    skipped_files += 1
                    continue

                # Skip if feature count doesn't match
                if df.shape[1] != expected_features:
                    skipped_files += 1
                    continue

                # Extract values
                features = df.values  # shape: (time_steps, features)

                # Check for NaN or Inf
                if np.isnan(features).any() or np.isinf(features).any():
                    skipped_files += 1
                    continue

                # Normalize features per stock (z-score normalization)
                features_mean = np.mean(features, axis=0)
                features_std = np.std(features, axis=0)
                features_std[features_std == 0] = 1  # Avoid division by zero
                features = (features - features_mean) / features_std

                # Create overlapping sequences (sliding window)
                for i in range(len(features) - sequence_length):
                    # Input sequence (90 days)
                    sequence = features[i:i + sequence_length]

                    # Label: next day's price movement
                    # Assuming first column is close price
                    current_price = features[i + sequence_length - 1, 0]
                    next_price = features[i + sequence_length, 0]

                    # 3-class classification: Down (0), Neutral (1), Up (2)
                    price_change = (next_price - current_price) / (abs(current_price) + 1e-8)
                    if price_change < -0.01:  # Down >1%
                        label = 0
                    elif price_change > 0.01:  # Up >1%
                        label = 2
                    else:  # Neutral
                        label = 1

                    # Extract time features (if available in index)
                    if hasattr(df.index, 'dayofweek'):
                        day_of_week = df.index[i + sequence_length - 1].dayofweek
                        month = df.index[i + sequence_length - 1].month - 1  # 0-11
                        hour = 12  # Assume daily data (market close)
                    else:
                        # Defaults if no datetime index
                        day_of_week = 2  # Wednesday
                        month = 6  # July
                        hour = 12

                    # Simple market regime based on recent volatility
                    recent_volatility = np.std(sequence[-20:, 0])  # Last 20 days
                    if recent_volatility < 0.5:
                        regime = 0  # Low volatility
                    elif recent_volatility < 1.0:
                        regime = 1  # Medium volatility
                    elif recent_volatility < 1.5:
                        regime = 2  # High volatility
                    else:
                        regime = 3  # Extreme volatility

                    # Time features for each time step in sequence
                    time_feat = np.array([[day_of_week, month, hour, regime]] * sequence_length)

                    # Volatility for each time step (rolling 20-day std)
                    vol = []
                    for j in range(sequence_length):
                        if j < 20:
                            vol.append(np.std(sequence[:j+1, 0]))
                        else:
                            vol.append(np.std(sequence[j-19:j+1, 0]))
                    vol = np.array(vol).reshape(-1, 1)

                    self.sequences.append(sequence.astype(np.float32))
                    self.labels.append(label)
                    self.time_features.append(time_feat.astype(np.int32))
                    self.volatility.append(vol.astype(np.float32))
                    total_sequences += 1

                loaded_files += 1

            except Exception as e:
                skipped_files += 1
                if file_idx < 10:  # Print first few errors
                    print(f"    Error loading {file.name}: {e}")
                continue

        print(f"\nDataset Summary:")
        print(f"  Files loaded: {loaded_files}")
        print(f"  Files skipped: {skipped_files}")
        print(f"  Total sequences: {len(self.sequences)}")
        print(f"  Sequence shape: ({sequence_length}, {expected_features})")

        # Class distribution
        labels_array = np.array(self.labels)
        print(f"  Label distribution:")
        print(f"    Down (0): {(labels_array == 0).sum()} ({100 * (labels_array == 0).mean():.1f}%)")
        print(f"    Neutral (1): {(labels_array == 1).sum()} ({100 * (labels_array == 1).mean():.1f}%)")
        print(f"    Up (2): {(labels_array == 2).sum()} ({100 * (labels_array == 2).mean():.1f}%)")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.float32),
            torch.tensor(self.time_features[idx], dtype=torch.long),
            torch.tensor(self.volatility[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )


class TransformerTrainer:
    """Production trainer with GPU optimization and warm-up scheduling"""

    def __init__(self, model, device, config):
        self.model = model.to(device)
        self.device = device
        self.config = config

        # Optimizer with weight decay (L2 regularization)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.98),  # Better for Transformer
            eps=1e-9
        )

        # Learning rate scheduler with warm-up (critical for Transformer!)
        self.warmup_steps = config.get('warmup_steps', 4000)
        self.current_step = 0

        # We'll manually control LR for warm-up
        self.base_lr = config['learning_rate']
        self.d_model = config.get('d_model', 512)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Mixed precision training (2x faster on RTX 4080)
        self.scaler = GradScaler()

        # Metrics tracking
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # Attention map storage (for visualization)
        self.attention_maps = []

        # TensorBoard
        log_dir = Path('logs/tensorboard_transformer') / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(log_dir)
        print(f"TensorBoard logging to: {log_dir}")

    def get_lr(self):
        """Get learning rate with warm-up (Transformer training trick)"""
        # LR = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
        step = max(self.current_step, 1)
        lr = self.base_lr * min(step ** (-0.5), step * self.warmup_steps ** (-1.5))
        return lr

    def update_lr(self):
        """Update learning rate for all param groups"""
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, time_feat, volatility, labels) in enumerate(train_loader):
            data = data.to(self.device)
            time_feat = time_feat.to(self.device)
            volatility = volatility.to(self.device)
            labels = labels.to(self.device)

            # Update learning rate with warm-up
            self.current_step += 1
            current_lr = self.update_lr()

            # Mixed precision training
            with autocast('cuda'):
                # Forward pass (returns dict with 'logits', 'attention_weights', 'hidden_states')
                outputs = self.model(data, time_feat, volatility)
                logits = outputs['logits']
                loss = self.criterion(logits, labels)

            # Check for NaN loss
            if torch.isnan(loss):
                print(f"  WARNING: NaN loss detected at batch {batch_idx}")
                print(f"  Data stats: min={data.min():.4f}, max={data.max():.4f}, mean={data.mean():.4f}")
                print(f"  Logits stats: min={logits.min():.4f}, max={logits.max():.4f}")
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

            # Metrics
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Log every 100 batches
            if (batch_idx + 1) % 100 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                accuracy = 100. * correct / total
                print(f"  Epoch {epoch} [{batch_idx + 1}/{len(train_loader)}] "
                      f"Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}% | "
                      f"LR: {current_lr:.8f} | Step: {self.current_step}")

        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def validate(self, val_loader, epoch, save_attention=False):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        # Store attention maps for first batch if requested
        stored_attention = False

        with torch.no_grad():
            for batch_idx, (data, time_feat, volatility, labels) in enumerate(val_loader):
                data = data.to(self.device)
                time_feat = time_feat.to(self.device)
                volatility = volatility.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(data, time_feat, volatility)
                logits = outputs['logits']
                loss = self.criterion(logits, labels)

                total_loss += loss.item()
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # Save attention maps from first batch
                if save_attention and not stored_attention and batch_idx == 0:
                    self.attention_maps = outputs['attention_weights']
                    stored_attention = True

        val_loss = total_loss / len(val_loader)
        val_acc = 100. * correct / total

        return val_loss, val_acc

    def visualize_attention(self, epoch):
        """Visualize attention maps"""
        if not self.attention_maps:
            return

        # Average attention across all layers and heads
        # attention_maps is a list of tensors: [num_layers] each with shape [batch, heads, seq_len, seq_len]
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # Create attention visualization directory
            attn_dir = Path('results/attention_maps')
            attn_dir.mkdir(parents=True, exist_ok=True)

            # Process first sample from batch
            for layer_idx, attn_weights in enumerate(self.attention_maps):
                # Average across batch and heads: [seq_len, seq_len]
                avg_attn = attn_weights[0].mean(dim=0).cpu().numpy()

                # Create heatmap
                plt.figure(figsize=(12, 10))
                sns.heatmap(avg_attn, cmap='viridis', cbar=True)
                plt.title(f'Attention Map - Layer {layer_idx + 1} - Epoch {epoch}')
                plt.xlabel('Key Position (Time Step)')
                plt.ylabel('Query Position (Time Step)')

                # Save figure
                save_path = attn_dir / f'epoch_{epoch}_layer_{layer_idx + 1}.png'
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()

            print(f"  Attention maps saved to: {attn_dir}")

        except Exception as e:
            print(f"  Warning: Could not visualize attention: {e}")

    def train(self, train_loader, val_loader, num_epochs):
        print("\n" + "=" * 80)
        print("STARTING TRANSFORMER TRAINING")
        print("=" * 80)

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()

            # Training
            train_loss, train_acc = self.train_epoch(train_loader, epoch)

            # Validation (save attention every 5 epochs)
            save_attn = (epoch % 5 == 0)
            val_loss, val_acc = self.validate(val_loader, epoch, save_attention=save_attn)

            epoch_time = time.time() - epoch_start

            # Logging
            print(f"\nEpoch {epoch}/{num_epochs} ({epoch_time:.1f}s)")
            print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
            print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")

            # Visualize attention maps
            if save_attn:
                self.visualize_attention(epoch)

            # TensorBoard
            self.writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
            self.writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch)
            self.writer.add_scalar('Learning_Rate', self.get_lr(), epoch)

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.patience_counter = 0

                # Save checkpoint
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'config': self.config,
                    'current_step': self.current_step
                }

                save_path = Path('models') / 'transformer_best.pth'
                save_path.parent.mkdir(exist_ok=True)
                torch.save(checkpoint, save_path)
                print(f"  [SAVED] Best model saved: {save_path} (Acc: {val_acc:.2f}%)")

            else:
                self.patience_counter += 1

            # Early stopping
            if self.patience_counter >= self.config['patience']:
                print(f"\nEarly stopping triggered! No improvement for {self.config['patience']} epochs.")
                break

            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                checkpoint_path = Path('models') / f'transformer_epoch_{epoch}.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'current_step': self.current_step
                }, checkpoint_path)
                print(f"  Checkpoint saved: {checkpoint_path}")

        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print(f"Best Validation Accuracy: {self.best_val_acc:.2f}%")
        print(f"Best Validation Loss: {self.best_val_loss:.4f}")
        print("=" * 80)

        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train Transformer on stock data')

    # Data arguments
    parser.add_argument('--data', type=str, default='TrainingData/features/*.parquet',
                       help='Path pattern to feature files')
    parser.add_argument('--test-mode', action='store_true',
                       help='Use only 10%% of data for quick testing')

    # Model arguments
    parser.add_argument('--d-model', type=int, default=512,
                       help='Model dimension (embedding size)')
    parser.add_argument('--num-heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=6,
                       help='Number of transformer layers')
    parser.add_argument('--d-ff', type=int, default=2048,
                       help='Feed-forward dimension')
    parser.add_argument('--sequence-length', type=int, default=90,
                       help='Sequence length (days to look back)')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size (Transformer benefits from larger batches)')
    parser.add_argument('--learning-rate', type=float, default=0.0001,
                       help='Base learning rate (will use warm-up)')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay (L2 regularization)')
    parser.add_argument('--warmup-steps', type=int, default=4000,
                       help='Number of warm-up steps for LR scheduler')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    parser.add_argument('--num-workers', type=int, default=0,
                       help='DataLoader workers (0 = no multiprocessing)')

    # Checkpoint arguments
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')

    args = parser.parse_args()

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Load data files
    from glob import glob
    data_files = [Path(f) for f in glob(args.data)]

    if len(data_files) == 0:
        print(f"ERROR: No data files found matching pattern: {args.data}")
        return

    print(f"Found {len(data_files)} data files")

    # Create datasets
    print("\n" + "=" * 80)
    print("CREATING DATASETS")
    print("=" * 80)

    # Split data: 70% train, 15% val, 15% test
    np.random.shuffle(data_files)
    train_size = int(0.7 * len(data_files))
    val_size = int(0.15 * len(data_files))

    train_files = data_files[:train_size]
    val_files = data_files[train_size:train_size + val_size]

    print(f"\nTrain files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")

    # Create datasets
    train_dataset = TransformerDataset(train_files, args.sequence_length, args.test_mode)
    val_dataset = TransformerDataset(val_files, args.sequence_length, args.test_mode)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Get input dimension from first batch
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch[0].shape[2]  # [batch, seq_len, features]
    print(f"\nInput dimension: {input_dim}")

    # Create model
    print("\n" + "=" * 80)
    print("CREATING MODEL")
    print("=" * 80)

    model = MarketTransformer(
        input_dim=input_dim,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        max_seq_length=args.sequence_length,
        num_classes=3,
        dropout=args.dropout
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024**2:.1f} MB")

    # Training configuration
    config = {
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'warmup_steps': args.warmup_steps,
        'epochs': args.epochs,
        'patience': args.patience,
        'd_model': args.d_model,
        'batch_size': args.batch_size
    }

    # Create trainer
    trainer = TransformerTrainer(model, device, config)

    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.current_step = checkpoint.get('current_step', 0)
        print(f"Resumed from epoch {checkpoint['epoch']}")
        print(f"Previous best validation accuracy: {checkpoint['val_acc']:.2f}%")

    # Start training
    trainer.train(train_loader, val_loader, args.epochs)

    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"Best Validation Accuracy: {trainer.best_val_acc:.2f}%")
    print(f"Best Validation Loss: {trainer.best_val_loss:.4f}")
    print(f"Model saved to: models/transformer_best.pth")
    print("\nNext steps:")
    print("  1. Evaluate model: python evaluate_transformer.py")
    print("  2. Compare with LSTM: python evaluate_all_models.py")
    print("  3. Visualize attention: Check results/attention_maps/")
    print("  4. TensorBoard: tensorboard --logdir logs/tensorboard_transformer")
    print("=" * 80)


if __name__ == '__main__':
    main()
