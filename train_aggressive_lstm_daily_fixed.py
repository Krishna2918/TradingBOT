"""
10-Hour Aggressive LSTM Daily Training - Fixed Version
======================================================

Trains Aggressive LSTM (Daily Mode) with:
- 200 epoch target
- 252-day lookback
- Uses ACTUAL available features (not hardcoded)
- Automatic checkpointing
- 40% resource usage
- Graceful termination
"""

import sys
import os
import time
import signal
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, classification_report, f1_score

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/aggressive_lstm_daily_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class GracefulKiller:
    """Handle graceful shutdown"""
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        logger.warning(f"Received signal {signum}, saving checkpoint and exiting...")
        self.kill_now = True


class AggressiveLSTMModel(nn.Module):
    """Aggressive LSTM with 256x2 architecture for daily predictions"""

    def __init__(self, input_size, hidden_size=256, num_layers=2, num_classes=3, dropout=0.1):
        super(AggressiveLSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[(n//4):(n//2)].fill_(1)

    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        last_output = self.layer_norm(last_output)
        last_output = self.dropout(last_output)
        output = self.classifier(last_output)
        return output


def configure_resources(resource_limit=0.4):
    """Configure 40% resource usage"""
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(resource_limit, device=0)
        logger.info(f"GPU memory limited to {resource_limit*100}%")

    num_cpus = os.cpu_count() or 1
    limited_threads = max(1, int(num_cpus * resource_limit))
    torch.set_num_threads(limited_threads)
    logger.info(f"CPU threads limited to {limited_threads}")


def load_data(features_dir: Path, max_symbols=None):
    """Load training data with ACTUAL available features - ALL SYMBOLS"""
    logger.info(f"Loading data from {features_dir}")

    feature_files = list(features_dir.glob("*_features.parquet"))
    logger.info(f"Found {len(feature_files)} feature files")

    # Load a sample to see available columns
    sample_df = pd.read_parquet(feature_files[0])
    available_features = [col for col in sample_df.columns
                         if col not in ['date', 'timestamp', 'symbol', 'direction_1d', 'target']]

    logger.info(f"Available features ({len(available_features)}): {available_features[:10]}...")

    # Load ALL symbols (or max_symbols if specified)
    files_to_load = feature_files if max_symbols is None else feature_files[:max_symbols]
    all_data = []
    for i, file in enumerate(files_to_load):

        df = pd.read_parquet(file)
        symbol = file.stem.replace('_features', '')

        # Add direction_1d if not present
        if 'direction_1d' not in df.columns:
            if 'close' in df.columns:
                df['direction_1d'] = np.sign(df['close'].pct_change().shift(-1))
                df['direction_1d'] = df['direction_1d'].fillna(0).astype(int)

        df['symbol'] = symbol
        all_data.append(df)
        logger.info(f"Loaded {symbol}: {len(df)} rows")

    combined_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"Combined: {len(combined_df)} rows from {len(all_data)} symbols")

    return combined_df, available_features


def create_sequences(features, targets, sequence_length=252):
    """Create sequences for training"""
    X_sequences = []
    y_sequences = []

    for i in range(sequence_length, len(features)):
        X_sequences.append(features[i-sequence_length:i])
        y_sequences.append(targets[i])

    return np.array(X_sequences), np.array(y_sequences)


def train_aggressive_lstm_daily():
    """Train Aggressive LSTM Daily for 10 hours"""
    logger.info("=" * 80)
    logger.info("10-HOUR AGGRESSIVE LSTM DAILY TRAINING")
    logger.info("=" * 80)

    # Configuration - WINNING FORMULA (Proven Strategy)
    total_hours = 10.0
    resource_limit = 0.4
    sequence_length = 30  # SHORT sequences like winning LSTM!
    hidden_size = 256  # Proven size from winning model
    num_layers = 2  # Simple architecture wins
    batch_size = 256  # Large batches like winner (512 was too big for our data)
    learning_rate = 5e-4  # Conservative initial LR
    target_epochs = 2000  # Many cycles of winning formula

    configure_resources(resource_limit)
    killer = GracefulKiller()

    # Paths
    features_dir = Path("TrainingData/features")
    checkpoint_dir = Path("models/aggressive_lstm_daily_checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load data - ALL SYMBOLS for comprehensive training
    combined_df, available_features = load_data(features_dir, max_symbols=None)  # Use ALL 1,681 stocks

    # Prepare features and targets
    features_df = combined_df[available_features].copy()

    # Handle missing values
    features_df = features_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    features_df = features_df.replace([np.inf, -np.inf], 0)

    # Scale features
    scaler = RobustScaler()
    features_scaled = scaler.fit_transform(features_df)

    # Prepare targets
    targets = combined_df['direction_1d'].values
    targets_encoded = targets + 1  # -1,0,1 -> 0,1,2
    targets_encoded = np.clip(targets_encoded, 0, 2)

    logger.info(f"Target distribution: {np.bincount(targets_encoded)}")

    # Create sequences
    X_sequences, y_sequences = create_sequences(features_scaled, targets_encoded, sequence_length)
    logger.info(f"Created {len(X_sequences)} sequences of shape {X_sequences.shape}")

    # Split data (80/10/10)
    split_train = int(len(X_sequences) * 0.8)
    split_val = int(len(X_sequences) * 0.9)

    X_train = X_sequences[:split_train]
    y_train = y_sequences[:split_train]
    X_val = X_sequences[split_train:split_val]
    y_val = y_sequences[split_train:split_val]

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}")

    # DON'T load all to GPU at once - use CPU tensors in DataLoader
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Keep data on CPU for DataLoader to stream
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)

    # DataLoaders with pin_memory for efficient GPU transfer
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)

    # Initialize model
    input_size = X_train.shape[-1]
    model = AggressiveLSTMModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=3,
        dropout=0.1
    ).to(device)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Input size: {input_size}")
    logger.info(f"Sequence length: {sequence_length}")

    # Class weights
    unique, counts = np.unique(y_train, return_counts=True)
    class_weights = 1.0 / np.clip(counts, 1, None)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    # Optimizer and loss - AGGRESSIVE settings
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.05)  # More smoothing

    # Advanced scheduler with warmup
    warmup_epochs = 50
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        total_steps=target_epochs,
        pct_start=warmup_epochs/target_epochs,
        anneal_strategy='cos'
    )
    scaler_amp = GradScaler()

    # Load checkpoint if exists
    best_checkpoint = checkpoint_dir / "best_model.pth"
    start_epoch = 0
    best_val_acc = 0.0

    if best_checkpoint.exists():
        logger.info(f"Loading checkpoint from {best_checkpoint}")
        checkpoint = torch.load(best_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        logger.info(f"Resumed from epoch {start_epoch}, best val acc: {best_val_acc:.4f}")

    # Training loop - ENHANCED WITH ADVANCED TECHNIQUES
    start_time = time.time()
    end_time = start_time + (total_hours * 3600)

    logger.info(f"Training until: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Starting from epoch {start_epoch}")
    logger.info(f"WINNING FORMULA MODE: 2000 epochs, 256x2 model, 20 symbols, 30-day lookback")

    # Stochastic Weight Averaging
    swa_model = None
    swa_start = 1800  # Start SWA at epoch 1800

    epoch = start_epoch
    patience_counter = 0
    patience_limit = 100  # Very patient for 2000 epochs

    while time.time() < end_time and epoch < target_epochs and not killer.kill_now:
        epoch += 1

        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        grad_norms = []

        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            if killer.kill_now or time.time() >= end_time:
                break

            # Move batch to GPU
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            # Mixup augmentation for harder training (randomly mix samples)
            if np.random.rand() < 0.3 and epoch > 100:  # 30% chance after warmup
                lam = np.random.beta(0.2, 0.2)
                index = torch.randperm(batch_x.size(0)).to(device)
                mixed_x = lam * batch_x + (1 - lam) * batch_x[index]
                y_a, y_b = batch_y, batch_y[index]

                with autocast():
                    outputs = model(mixed_x)
                    loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
            else:
                with autocast():
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)

            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grad_norms.append(grad_norm.item())
            scaler_amp.step(optimizer)
            scaler_amp.update()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                # Move batch to GPU
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                with autocast():
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()

                val_preds.extend(predicted.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        macro_f1 = f1_score(val_targets, val_preds, average='macro', zero_division=0)

        elapsed = time.time() - start_time
        remaining = end_time - time.time()
        avg_grad_norm = np.mean(grad_norms) if grad_norms else 0.0

        logger.info(
            f"Epoch {epoch}/{target_epochs} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, F1: {macro_f1:.4f}, "
            f"Grad: {avg_grad_norm:.3f}, Elapsed: {elapsed/3600:.2f}h, Remaining: {remaining/3600:.2f}h"
        )

        # Stochastic Weight Averaging (for epochs 1800+)
        if epoch >= swa_start:
            if swa_model is None:
                swa_model = torch.optim.swa_utils.AveragedModel(model)
                logger.info(f"Started SWA at epoch {epoch}")
            swa_model.update_parameters(model)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'val_loss': val_loss,
                'f1_score': macro_f1
            }
            torch.save(checkpoint, best_checkpoint)
            logger.info(f"Saved best model! Val Acc: {val_acc:.4f}, F1: {macro_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                logger.info(f"Early stopping at epoch {epoch} - no improvement for {patience_limit} epochs")
                break

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc
            }, checkpoint_path)
            logger.info(f"Saved checkpoint at epoch {epoch}")

        scheduler.step()

    # Apply SWA if used
    if swa_model is not None:
        logger.info("Applying Stochastic Weight Averaging to final model...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        # Save SWA model
        swa_checkpoint = checkpoint_dir / f"swa_model.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': swa_model.module.state_dict(),
            'best_val_acc': best_val_acc,
            'scaler': scaler
        }, swa_checkpoint)
        logger.info(f"SWA model saved to {swa_checkpoint}")

    # Final save
    final_checkpoint = checkpoint_dir / f"final_epoch_{epoch}.pth"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_acc': best_val_acc,
        'scaler': scaler
    }, final_checkpoint)

    total_elapsed = time.time() - start_time
    logger.info("=" * 80)
    logger.info("ENHANCED AGGRESSIVE TRAINING COMPLETED")
    logger.info(f"Total time: {total_elapsed/3600:.2f} hours")
    logger.info(f"Epochs completed: {epoch - start_epoch}")
    logger.info(f"Best val accuracy: {best_val_acc:.4f}")
    logger.info(f"Model: 512x3 with {sequence_length}-day lookback on 50 symbols")
    logger.info("=" * 80)


if __name__ == "__main__":
    try:
        train_aggressive_lstm_daily()
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)
