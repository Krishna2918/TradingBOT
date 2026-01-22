"""
IMPROVED LSTM TRAINER - Fixed for Production Quality
=====================================================

Critical Fixes Applied:
1. 3-Class Classification (Up/Down/Flat) with 0.5% threshold
2. Focal Loss for class imbalance handling
3. Stronger regularization (dropout, weight decay, label smoothing)
4. Class-weighted sampling for balanced training
5. Improved NaN handling
6. Per-class metrics logging
7. Class collapse detection

Target: 70%+ validation accuracy

Usage:
    python train_lstm_improved.py --data TrainingData/features --epochs 100

Author: Trading Bot Team
Date: December 2025
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
from collections import Counter

warnings.filterwarnings('ignore', category=UserWarning, module='torch.optim.lr_scheduler')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix

# POWER MANAGEMENT INTEGRATION
try:
    from power_management import (
        get_gpu_manager,
        get_precision_manager,
        get_power_monitor,
        DEFAULT_CONFIG
    )
    HAS_POWER_MANAGEMENT = True
except ImportError:
    HAS_POWER_MANAGEMENT = False
    print("Warning: Power management not available")

# GPU Configuration
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)


# =============================================================================
# FOCAL LOSS - Handles Class Imbalance
# =============================================================================
class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Down-weights easy examples, focuses on hard ones.

    Paper: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    """
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0,
                 weight: Optional[torch.Tensor] = None,
                 label_smoothing: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Apply label smoothing
        num_classes = inputs.size(1)

        # Create smoothed targets
        with torch.no_grad():
            smooth_targets = torch.zeros_like(inputs)
            smooth_targets.fill_(self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        # Compute cross-entropy with smoothed targets
        log_probs = F.log_softmax(inputs, dim=1)
        ce_loss = -(smooth_targets * log_probs).sum(dim=1)

        # Compute focal weight
        probs = torch.softmax(inputs, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = self.alpha * (1 - pt) ** self.gamma

        # Apply class weights if provided
        if self.weight is not None:
            class_weight = self.weight.gather(0, targets)
            focal_weight = focal_weight * class_weight

        focal_loss = focal_weight * ce_loss
        return focal_loss.mean()


# =============================================================================
# DATASET WITH 3-CLASS LABELS
# =============================================================================
class ImprovedStockDataset(Dataset):
    """
    Dataset with 3-class classification:
    - Class 0: DOWN (return < -0.5%)
    - Class 1: FLAT (return between -0.5% and +0.5%)
    - Class 2: UP (return > +0.5%)
    """

    # Direction threshold (0.5% = 0.005)
    DIRECTION_THRESHOLD = 0.005

    def __init__(self, data_files: List[Path], sequence_length: int = 60,
                 test_mode: bool = False, stride: int = 3,
                 max_sequences_per_file: int = 100):
        self.sequence_length = sequence_length
        self.test_mode = test_mode
        self.stride = stride
        self.max_sequences_per_file = max_sequences_per_file
        self.sequences = []
        self.labels = []

        print(f"\n{'='*70}")
        print("IMPROVED DATASET LOADING")
        print(f"{'='*70}")
        print(f"Files to load: {len(data_files)}")
        print(f"Sequence length: {sequence_length} days")
        print(f"Stride: {stride} (sample every {stride} days)")
        print(f"Max sequences/file: {max_sequences_per_file}")
        print(f"Direction threshold: +/-{self.DIRECTION_THRESHOLD*100:.1f}%")
        print(f"Classification: 3-class (Down/Flat/Up)")

        if test_mode:
            data_files = data_files[:int(len(data_files) * 0.1)]
            print(f"TEST MODE: Using {len(data_files)} files (10%)")

        # Analyze feature dimensions
        expected_features = self._analyze_features(data_files)

        # Load data
        self._load_data(data_files, expected_features)

        # Convert to numpy
        self._finalize()

    def _analyze_features(self, data_files: List[Path]) -> int:
        """Determine expected feature count"""
        feature_counts = {}
        for file in data_files[:100]:
            try:
                df = pd.read_parquet(file)
                if len(df) >= self.sequence_length + 1:
                    feature_counts[df.shape[1]] = feature_counts.get(df.shape[1], 0) + 1
            except:
                continue

        if not feature_counts:
            raise ValueError("No valid data files found")

        expected = max(feature_counts.keys(), key=feature_counts.get)
        print(f"Expected features: {expected}")
        print(f"Feature distribution: {feature_counts}")
        return expected

    def _load_data(self, data_files: List[Path], expected_features: int):
        """Load and process all data files"""
        for i, file in enumerate(data_files):
            if (i + 1) % 100 == 0:
                print(f"  Loaded {i + 1}/{len(data_files)} files...")

            try:
                df = pd.read_parquet(file)

                if len(df) < self.sequence_length + 5:
                    continue

                # Improved NaN handling
                df = df.ffill().bfill()

                features = df.values

                if features.shape[1] != expected_features:
                    continue

                # Handle remaining NaN
                if np.isnan(features).any():
                    col_means = np.nanmean(features, axis=0)
                    col_means = np.where(np.isnan(col_means), 0, col_means)
                    for col in range(features.shape[1]):
                        features[:, col] = np.where(
                            np.isnan(features[:, col]),
                            col_means[col],
                            features[:, col]
                        )

                # Z-score normalization
                features_mean = np.nanmean(features, axis=0)
                features_std = np.nanstd(features, axis=0)
                features_std = np.where((features_std == 0) | np.isnan(features_std), 1, features_std)
                features_mean = np.where(np.isnan(features_mean), 0, features_mean)
                features = (features - features_mean) / features_std
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

                # Create sequences with 3-class labels
                file_sequences = []
                file_labels = []

                for j in range(0, len(df) - self.sequence_length - 1, self.stride):
                    seq = features[j:j + self.sequence_length]

                    if np.isnan(seq).any():
                        continue

                    # Get current and next close prices
                    close_col = 3 if 'close' not in df.columns else df.columns.get_loc('close')
                    current_close = df.iloc[j + self.sequence_length - 1, close_col]
                    next_close = df.iloc[j + self.sequence_length, close_col]

                    # Calculate return
                    if current_close > 0:
                        ret = (next_close - current_close) / current_close
                    else:
                        continue

                    # 3-CLASS CLASSIFICATION
                    if ret > self.DIRECTION_THRESHOLD:
                        label = 2  # UP
                    elif ret < -self.DIRECTION_THRESHOLD:
                        label = 0  # DOWN
                    else:
                        label = 1  # FLAT

                    file_sequences.append(seq)
                    file_labels.append(label)

                    if len(file_sequences) >= self.max_sequences_per_file:
                        break

                self.sequences.extend(file_sequences)
                self.labels.extend(file_labels)

            except Exception as e:
                print(f"  Warning: Failed to load {file.name}: {e}")
                continue

    def _finalize(self):
        """Convert to numpy arrays and report statistics"""
        if len(self.sequences) == 0:
            raise ValueError("No valid sequences found")

        print(f"\nConverting {len(self.sequences)} sequences...")
        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)

        # Report class distribution
        class_counts = Counter(self.labels)
        print(f"\n{'='*70}")
        print("DATASET STATISTICS")
        print(f"{'='*70}")
        print(f"Total sequences: {len(self.sequences)}")
        print(f"Sequence shape: {self.sequences.shape}")
        print(f"\nClass distribution:")
        print(f"  DOWN (class 0): {class_counts.get(0, 0):,} ({100*class_counts.get(0, 0)/len(self.labels):.1f}%)")
        print(f"  FLAT (class 1): {class_counts.get(1, 0):,} ({100*class_counts.get(1, 0)/len(self.labels):.1f}%)")
        print(f"  UP   (class 2): {class_counts.get(2, 0):,} ({100*class_counts.get(2, 0)/len(self.labels):.1f}%)")

        gc.collect()

    def get_sample_weights(self) -> np.ndarray:
        """Calculate sample weights for balanced sampling"""
        class_counts = np.bincount(self.labels, minlength=3)
        class_weights = 1.0 / (class_counts + 1e-8)
        sample_weights = class_weights[self.labels]
        return sample_weights

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.LongTensor([self.labels[idx]])[0]


# =============================================================================
# MULTI-HEAD TEMPORAL ATTENTION
# =============================================================================
class MultiHeadTemporalAttention(nn.Module):
    """Multi-head self-attention for temporal sequences"""

    def __init__(self, input_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"

        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.out_proj = nn.Linear(input_dim, input_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Linear projections
        Q = self.query(x)  # (batch, seq, dim)
        K = self.key(x)
        V = self.value(x)

        # Reshape to (batch, heads, seq, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.out_proj(attn_output)

        return output, attn_weights


# =============================================================================
# IMPROVED LSTM MODEL
# =============================================================================
class ImprovedLSTMModel(nn.Module):
    """
    LSTM with stronger regularization for 3-class classification
    """

    def __init__(self, input_dim: int, hidden_size: int = 256,
                 num_layers: int = 3, dropout: float = 0.5, num_classes: int = 3):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        # Input projection with LayerNorm
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )

        # LSTM layers
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # Bidirectional for better context
        )

        # Multi-head attention mechanism
        self.attention = MultiHeadTemporalAttention(
            input_dim=hidden_size * 2,  # bidirectional
            num_heads=4,
            dropout=dropout * 0.5
        )
        self.attention_norm = nn.LayerNorm(hidden_size * 2)

        # Output layers with strong regularization
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)  # 3 classes
        )

        self._init_weights()

    def forward(self, x):
        # Input projection
        x = self.input_proj(x)

        # LSTM
        lstm_out, _ = self.lstm(x)

        # Multi-head attention with residual connection
        attn_out, _ = self.attention(lstm_out)
        attn_out = self.attention_norm(attn_out + lstm_out)  # Residual connection
        context = attn_out.mean(dim=1)  # Mean pooling over sequence

        # Classification
        output = self.fc(context)
        return output

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
            elif 'weight' in name and len(param.shape) == 2:
                nn.init.xavier_uniform_(param.data)


# =============================================================================
# IMPROVED TRAINER
# =============================================================================
class ImprovedTrainer:
    """Trainer with class collapse detection and per-class metrics"""

    def __init__(self, model, device, config, class_weights: Optional[torch.Tensor] = None):
        self.model = model.to(device)
        self.device = device
        self.config = config

        # Power management
        if HAS_POWER_MANAGEMENT:
            self.gpu_manager = get_gpu_manager(DEFAULT_CONFIG.gpu)
            self.precision_manager = get_precision_manager(DEFAULT_CONFIG.mixed_precision)
            self.power_monitor = get_power_monitor(DEFAULT_CONFIG)
            self.power_monitor.set_managers(
                gpu_manager=self.gpu_manager,
                precision_manager=self.precision_manager
            )
        else:
            self.gpu_manager = None
            self.precision_manager = None

        # Optimizer with stronger weight decay
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.999)
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config['learning_rate'],
            epochs=config['epochs'],
            steps_per_epoch=config['steps_per_epoch'],
            pct_start=0.1,
            anneal_strategy='cos'
        )

        # Focal Loss with class weights
        self.criterion = FocalLoss(
            alpha=1.0,
            gamma=2.0,
            weight=class_weights.to(device) if class_weights is not None else None,
            label_smoothing=config.get('label_smoothing', 0.1)
        )

        # Mixed precision
        self.scaler = GradScaler('cuda')

        # Metrics
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # TensorBoard
        log_dir = Path('logs/tensorboard_improved') / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(log_dir)
        print(f"TensorBoard: {log_dir}")

    def check_class_collapse(self, predictions: np.ndarray) -> bool:
        """Detect if model predicts mostly one class"""
        unique, counts = np.unique(predictions, return_counts=True)
        max_ratio = counts.max() / len(predictions)

        if max_ratio > 0.8:
            dominant = unique[counts.argmax()]
            print(f"\n  WARNING: CLASS COLLAPSE DETECTED: {max_ratio*100:.1f}% predictions are class {dominant}")
            return True
        return False

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(self.device), labels.to(self.device)

            # Forward pass with mixed precision
            if self.device.type == 'cuda':
                with autocast('cuda'):
                    outputs = self.model(data)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)

            if torch.isnan(loss):
                print(f"  WARNING: NaN loss at batch {batch_idx}")
                continue

            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            # Metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if (batch_idx + 1) % 50 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                accuracy = 100. * correct / total
                print(f"  Epoch {epoch} [{batch_idx + 1}/{len(train_loader)}] "
                      f"Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}% | "
                      f"LR: {self.scheduler.get_last_lr()[0]:.6f}")

        # Check for class collapse
        self.check_class_collapse(np.array(all_preds))

        return total_loss / len(train_loader), 100. * correct / total

    def validate(self, val_loader, epoch):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = total_loss / len(val_loader)
        val_acc = 100. * correct / total

        # Log per-class metrics
        self._log_class_metrics(all_labels, all_preds, epoch)

        return val_loss, val_acc

    def _log_class_metrics(self, y_true, y_pred, epoch):
        """Log detailed per-class metrics"""
        report = classification_report(y_true, y_pred,
                                       target_names=['DOWN', 'FLAT', 'UP'],
                                       output_dict=True,
                                       zero_division=0)

        print(f"\n  Per-class metrics (Epoch {epoch}):")
        for cls in ['DOWN', 'FLAT', 'UP']:
            if cls in report:
                f1 = report[cls]['f1-score']
                prec = report[cls]['precision']
                rec = report[cls]['recall']
                print(f"    {cls}: F1={f1:.3f} | Precision={prec:.3f} | Recall={rec:.3f}")

                self.writer.add_scalar(f'F1/{cls}', f1, epoch)
                self.writer.add_scalar(f'Precision/{cls}', prec, epoch)
                self.writer.add_scalar(f'Recall/{cls}', rec, epoch)

        # Log confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print(f"\n  Confusion Matrix:")
        print(f"           Pred DOWN  Pred FLAT  Pred UP")
        print(f"    DOWN      {cm[0,0]:5d}      {cm[0,1]:5d}     {cm[0,2]:5d}")
        print(f"    FLAT      {cm[1,0]:5d}      {cm[1,1]:5d}     {cm[1,2]:5d}")
        print(f"    UP        {cm[2,0]:5d}      {cm[2,1]:5d}     {cm[2,2]:5d}")

    def train(self, train_loader, val_loader, num_epochs):
        print("\n" + "=" * 80)
        print("STARTING IMPROVED TRAINING")
        print("=" * 80)
        print(f"Epochs: {num_epochs}")
        print(f"Using: Focal Loss + Class Weights + Label Smoothing")
        print(f"Regularization: Dropout={self.config['dropout']}, Weight Decay={self.config['weight_decay']}")

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()

            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            val_loss, val_acc = self.validate(val_loader, epoch)

            epoch_time = time.time() - epoch_start

            print(f"\nEpoch {epoch}/{num_epochs} ({epoch_time:.1f}s)")
            print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
            print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")

            # TensorBoard
            self.writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
            self.writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch)

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.patience_counter = 0

                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'config': self.config,
                    'num_classes': 3
                }

                save_path = Path(self.config['output_dir']) / 'lstm_improved_best.pth'
                torch.save(checkpoint, save_path)
                print(f"  [SAVED] Best model saved: {save_path} (Val Acc: {val_acc:.2f}%)")
            else:
                self.patience_counter += 1

            # Early stopping
            if self.patience_counter >= self.config['early_stopping_patience']:
                print(f"\n  Early stopping (patience={self.config['early_stopping_patience']})")
                break

            # Checkpoint
            if epoch % 10 == 0:
                checkpoint_path = Path(self.config['output_dir']) / f'lstm_improved_epoch_{epoch}.pth'
                torch.save(checkpoint, checkpoint_path)
                print(f"  Checkpoint: {checkpoint_path}")

        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(f"Best Val Accuracy: {self.best_val_acc:.2f}%")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")

        self.writer.close()


def check_gpu():
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Training will be slow.")
        return False

    print("=" * 80)
    print("GPU CONFIGURATION")
    print("=" * 80)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Improved LSTM Trainer')

    # Data
    parser.add_argument('--data', type=str, default='TrainingData/features')
    parser.add_argument('--output-dir', type=str, default='models')

    # Model
    parser.add_argument('--hidden-size', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.5)  # Increased
    parser.add_argument('--sequence-length', type=int, default=60)
    parser.add_argument('--stride', type=int, default=3)

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=0.0005)
    parser.add_argument('--weight-decay', type=float, default=0.002)  # Increased for better regularization
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--early-stopping-patience', type=int, default=30)

    # Optimization
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--max-sequences-per-file', type=int, default=100)
    parser.add_argument('--use-weighted-sampler', action='store_true', default=True)

    # Mode
    parser.add_argument('--test-mode', action='store_true')

    args = parser.parse_args()

    # GPU check
    has_gpu = check_gpu()
    device = torch.device('cuda' if has_gpu else 'cpu')

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data files
    data_dir = Path(args.data)
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        return

    data_files = list(data_dir.glob('*.parquet'))
    if len(data_files) == 0:
        print(f"ERROR: No .parquet files found")
        return

    print(f"\nFound {len(data_files)} data files")

    if args.test_mode:
        print("\nTEST MODE: 10% data, 10 epochs")
        args.epochs = 10

    # Create dataset
    print("\nLoading dataset...")
    full_dataset = ImprovedStockDataset(
        data_files,
        args.sequence_length,
        test_mode=args.test_mode,
        stride=args.stride,
        max_sequences_per_file=args.max_sequences_per_file
    )

    # Stratified split
    print("\nPerforming stratified split (70/30)...")
    labels_array = np.array(full_dataset.labels)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, val_idx = next(sss.split(np.zeros(len(labels_array)), labels_array))

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # Calculate class weights
    train_labels = labels_array[train_idx]
    class_counts = np.bincount(train_labels, minlength=3)
    class_weights = torch.FloatTensor(len(train_labels) / (3 * class_counts + 1e-8))
    print(f"\nClass weights: DOWN={class_weights[0]:.2f}, FLAT={class_weights[1]:.2f}, UP={class_weights[2]:.2f}")

    # Weighted sampler for balanced batches
    if args.use_weighted_sampler:
        sample_weights = full_dataset.get_sample_weights()[train_idx]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        shuffle = False
        print("Using WeightedRandomSampler for balanced training")
    else:
        sampler = None
        shuffle = True

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False
    )

    # Get input dimension
    sample_batch, _ = next(iter(train_loader))
    input_dim = sample_batch.shape[2]
    print(f"\nInput dimension: {input_dim} features")

    # Create model
    print("\nInitializing model...")
    model = ImprovedLSTMModel(
        input_dim=input_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_classes=3
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Config
    config = {
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'sequence_length': args.sequence_length,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'label_smoothing': args.label_smoothing,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'early_stopping_patience': args.early_stopping_patience,
        'steps_per_epoch': len(train_loader),
        'output_dir': str(output_dir),
        'num_classes': 3,
        'direction_threshold': ImprovedStockDataset.DIRECTION_THRESHOLD
    }

    # Train - Use WeightedRandomSampler with adjusted weights
    # Penalize FLAT (middle ground) and boost UP to prevent lazy predictions
    if args.use_weighted_sampler:
        # Adjusted weights: DOWN=1.2, FLAT=0.8 (penalized), UP=1.2 (symmetric boost for directional classes)
        adjusted_weights = torch.FloatTensor([1.2, 0.8, 1.2])  # Symmetric boost for DOWN/UP, penalize FLAT
        trainer = ImprovedTrainer(model, device, config, adjusted_weights)
        print("Using WeightedRandomSampler + adjusted weights (DOWN=1.2, FLAT=0.8, UP=1.2)")
    else:
        # No sampler, use full class weights in loss function
        trainer = ImprovedTrainer(model, device, config, class_weights)
        print("Using class weights in loss function (no sampler)")
    trainer.train(train_loader, val_loader, args.epochs)

    # Save config
    config_path = output_dir / 'lstm_improved_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 80)
    print("IMPROVED LSTM TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Best model: {output_dir / 'lstm_improved_best.pth'}")
    print(f"Best accuracy: {trainer.best_val_acc:.2f}%")
    print(f"\nTensorBoard: tensorboard --logdir logs/tensorboard_improved")


if __name__ == '__main__':
    main()
