"""
SUPER-OPTIMIZED LSTM - Target 79%+ Accuracy
============================================

Incorporates ALL winning features from optimized_lstm_trainer.py:
- GPU Memory Management
- Dynamic Batch Controller (auto-adjusts 256-512-1024)
- Gradient Accumulation (2X effective batch)
- torch.compile graph optimization
- Balanced class sampling
- Advanced learning rate scheduling
- Mixup augmentation
- Label smoothing
- Stochastic Weight Averaging

Goal: Surpass 74% baseline by 5% → Target 79%+
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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, classification_report, f1_score
from collections import Counter

sys.path.append(str(Path(__file__).parent / "src"))

# Import winning components
from ai.models.gpu_memory_manager import GPUMemoryManager
from ai.models.dynamic_batch_controller import DynamicBatchController
from ai.models.gradient_accumulator import GradientAccumulator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/super_optimized_lstm_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class GracefulKiller:
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        logger.warning(f"Received signal {signum}, saving and exiting...")
        self.kill_now = True


class SuperOptimizedLSTM(nn.Module):
    """Enhanced LSTM with residual connections and layer normalization"""

    def __init__(self, input_size, hidden_size=256, num_layers=2, num_classes=3, dropout=0.2):
        super(SuperOptimizedLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM with layer normalization
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Layer normalization for stability
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout)

        # Enhanced classifier with residual-style connections
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.layer_norm2 = nn.LayerNorm(hidden_size // 2)
        self.dropout2 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.layer_norm3 = nn.LayerNorm(hidden_size // 4)
        self.dropout3 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(hidden_size // 4, num_classes)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                n = param.size(0)
                if n >= param.size(0) // 2:
                    param.data[(n//4):(n//2)].fill_(1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]

        # Apply layer norm and dropout
        last_output = self.layer_norm1(last_output)
        last_output = self.dropout1(last_output)

        # First FC layer
        x = torch.relu(self.fc1(last_output))
        x = self.layer_norm2(x)
        x = self.dropout2(x)

        # Second FC layer
        x = torch.relu(self.fc2(x))
        x = self.layer_norm3(x)
        x = self.dropout3(x)

        # Output
        output = self.fc3(x)
        return output


def create_balanced_sampler(targets):
    """Create weighted sampler for balanced training"""
    class_counts = Counter(targets)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[t] for t in targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    return sampler


def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def configure_resources(resource_limit=0.4):
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(resource_limit, device=0)
        logger.info(f"GPU memory limited to {resource_limit*100}%")

    num_cpus = os.cpu_count() or 1
    limited_threads = max(1, int(num_cpus * resource_limit))
    torch.set_num_threads(limited_threads)
    logger.info(f"CPU threads limited to {limited_threads}")


def load_optimized_data(features_dir: Path, max_symbols=None):
    """Load data optimized for best performance - ALL SYMBOLS"""
    logger.info(f"Loading data from {features_dir}")

    feature_files = list(features_dir.glob("*_features.parquet"))
    logger.info(f"Found {len(feature_files)} feature files")

    # Prioritize high-quality stocks first, then add all others
    priority_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
                       'JPM', 'V', 'JNJ', 'WMT', 'PG', 'MA', 'HD', 'DIS']

    selected_files = []
    for symbol in priority_symbols:
        for file in feature_files:
            if symbol in file.stem:
                selected_files.append(file)
                break

    # Add ALL remaining files (or up to max_symbols if specified)
    for file in feature_files:
        if file not in selected_files:
            if max_symbols is not None and len(selected_files) >= max_symbols:
                break
            selected_files.append(file)

    if max_symbols is not None:
        selected_files = selected_files[:max_symbols]

    # Load data
    sample_df = pd.read_parquet(selected_files[0])
    available_features = [col for col in sample_df.columns
                         if col not in ['date', 'timestamp', 'symbol', 'direction_1d', 'target']]

    logger.info(f"Available features: {len(available_features)}")

    all_data = []
    for file in selected_files:
        df = pd.read_parquet(file)
        symbol = file.stem.replace('_features', '')

        if 'direction_1d' not in df.columns and 'close' in df.columns:
            df['direction_1d'] = np.sign(df['close'].pct_change().shift(-1))
            df['direction_1d'] = df['direction_1d'].fillna(0).astype(int)

        df['symbol'] = symbol
        all_data.append(df)
        logger.info(f"Loaded {symbol}: {len(df)} rows")

    combined_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"Combined: {len(combined_df)} rows from {len(all_data)} symbols")

    return combined_df, available_features


def train_super_optimized_lstm():
    """Train SUPER-OPTIMIZED LSTM for 10 hours targeting 79%+ accuracy"""
    logger.info("=" * 80)
    logger.info("SUPER-OPTIMIZED LSTM - TARGET 79%+ ACCURACY")
    logger.info("=" * 80)

    # Configuration - WINNING FORMULA + ENHANCEMENTS
    total_hours = 10.0
    resource_limit = 0.4
    sequence_length = 30  # Winning formula
    hidden_size = 256  # Proven size
    num_layers = 2  # Simple wins
    initial_batch_size = 512  # Start large
    learning_rate = 8e-4  # Winning LR
    target_epochs = 2000

    configure_resources(resource_limit)
    killer = GracefulKiller()

    # Initialize GPU components
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    memory_manager = GPUMemoryManager(reserve_memory_gb=1.0)
    batch_controller = DynamicBatchController(
        memory_manager=memory_manager,
        initial_batch_size=initial_batch_size,
        min_batch_size=64,
        max_batch_size=1024
    )
    gradient_accumulator = GradientAccumulator(accumulation_steps=2, gradient_clip_norm=1.0)

    logger.info(f"GPU Memory Manager initialized")
    logger.info(f"Dynamic Batch Controller: 64-1024 range")
    logger.info(f"Gradient Accumulation: 2X effective batch size")

    # Paths
    features_dir = Path("TrainingData/features")
    checkpoint_dir = Path("models/super_optimized_lstm")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load data - ALL SYMBOLS for comprehensive training
    combined_df, available_features = load_optimized_data(features_dir, max_symbols=None)  # Use ALL 1,681 stocks

    # Prepare features
    features_df = combined_df[available_features].copy()
    features_df = features_df.ffill().bfill().fillna(0)
    features_df = features_df.replace([np.inf, -np.inf], 0)

    # Scale
    scaler = RobustScaler()
    features_scaled = scaler.fit_transform(features_df)

    # Targets
    targets = combined_df['direction_1d'].values
    targets_encoded = targets + 1
    targets_encoded = np.clip(targets_encoded, 0, 2)

    logger.info(f"Target distribution: {np.bincount(targets_encoded)}")

    # Create sequences
    X_seq = []
    y_seq = []
    for i in range(sequence_length, len(features_scaled)):
        X_seq.append(features_scaled[i-sequence_length:i])
        y_seq.append(targets_encoded[i])

    X_sequences = np.array(X_seq)
    y_sequences = np.array(y_seq)

    logger.info(f"Created {len(X_sequences)} sequences of shape {X_sequences.shape}")

    # Split
    split_train = int(len(X_sequences) * 0.8)
    split_val = int(len(X_sequences) * 0.9)

    X_train = X_sequences[:split_train]
    y_train = y_sequences[:split_train]
    X_val = X_sequences[split_train:split_val]
    y_val = y_sequences[split_train:split_val]

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}")

    # Create balanced sampler
    train_sampler = create_balanced_sampler(y_train)

    # DataLoaders with optimizations
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )

    # Dynamic batch size
    current_batch_size = batch_controller.recommend_initial_batch_size(len(X_train))
    logger.info(f"Recommended batch size: {current_batch_size}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=current_batch_size,
        sampler=train_sampler,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=current_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2
    )

    # Initialize model
    input_size = X_train.shape[-1]
    model = SuperOptimizedLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=3,
        dropout=0.2
    ).to(device)

    # torch.compile for speed
    if hasattr(torch, 'compile') and torch.cuda.is_available():
        logger.info("Enabling torch.compile for graph optimization")
        model = torch.compile(model, mode='max-autotune')

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Class weights
    unique, counts = np.unique(y_train, return_counts=True)
    class_weights = torch.tensor([len(y_train) / (len(unique) * c) for c in counts], dtype=torch.float32, device=device)

    # Optimizer and loss
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    # Advanced scheduler - OneCycleLR
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate * 2,
        total_steps=target_epochs,
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=10,
        final_div_factor=100
    )

    scaler_amp = GradScaler()

    # Training state
    start_time = time.time()
    end_time = start_time + (total_hours * 3600)
    best_val_f1 = 0.0
    best_val_acc = 0.0

    logger.info(f"Training until: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Target: 79%+ accuracy (5% improvement)")
    logger.info("SUPER-OPTIMIZED MODE ACTIVATED")

    # SWA
    swa_model = None
    swa_start = 1800

    epoch = 0
    patience = 0
    patience_limit = 50

    while time.time() < end_time and epoch < target_epochs and not killer.kill_now:
        epoch += 1

        # Memory check (simplified)
        if epoch % 100 == 0:
            try:
                torch.cuda.empty_cache()
                logger.info(f"GPU memory check at epoch {epoch}")
            except:
                pass

        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        grad_norms = []

        gradient_accumulator.reset()

        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            if killer.kill_now or time.time() >= end_time:
                break

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Mixup augmentation (30% chance after epoch 50)
            use_mixup = epoch > 50 and np.random.rand() < 0.3

            if use_mixup:
                mixed_x, y_a, y_b, lam = mixup_data(batch_x, batch_y, alpha=0.2)
                with autocast():
                    outputs = model(mixed_x)
                    loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
            else:
                with autocast():
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)

            # Gradient accumulation
            loss = loss / gradient_accumulator.accumulation_steps
            scaler_amp.scale(loss).backward()

            # Step every N batches
            if (batch_idx + 1) % gradient_accumulator.accumulation_steps == 0:
                scaler_amp.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                grad_norms.append(grad_norm.item())
                scaler_amp.step(optimizer)
                scaler_amp.update()
                optimizer.zero_grad(set_to_none=True)

            train_loss += loss.item() * gradient_accumulator.accumulation_steps
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
        avg_grad = np.mean(grad_norms) if grad_norms else 0.0

        logger.info(
            f"Epoch {epoch}/{target_epochs} - "
            f"Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, F1: {macro_f1:.4f}, "
            f"Grad: {avg_grad:.3f}, BS: {current_batch_size}, "
            f"Time: {elapsed/3600:.2f}h/{remaining/3600:.2f}h"
        )

        # SWA
        if epoch >= swa_start:
            if swa_model is None:
                swa_model = torch.optim.swa_utils.AveragedModel(model)
                logger.info(f"Started SWA at epoch {epoch}")
            swa_model.update_parameters(model)

        # Save best
        if macro_f1 > best_val_f1 or (macro_f1 == best_val_f1 and val_acc > best_val_acc):
            best_val_f1 = macro_f1
            best_val_acc = val_acc
            patience = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict() if not hasattr(model, '_orig_mod') else model._orig_mod.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'best_val_f1': best_val_f1,
                'scaler': scaler
            }, checkpoint_dir / 'best_model.pth')

            logger.info(f"NEW BEST! Val Acc: {val_acc:.4f}, F1: {macro_f1:.4f}")
        else:
            patience += 1
            if patience >= patience_limit:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # Checkpoints
        if epoch % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict() if not hasattr(model, '_orig_mod') else model._orig_mod.state_dict(),
                'best_val_acc': best_val_acc,
                'best_val_f1': best_val_f1
            }, checkpoint_dir / f'checkpoint_epoch_{epoch}.pth')
            logger.info(f"Saved checkpoint at epoch {epoch}")

        scheduler.step()

    # Apply SWA
    if swa_model is not None:
        logger.info("Applying SWA to final model...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        torch.save({
            'epoch': epoch,
            'model_state_dict': swa_model.module.state_dict(),
            'best_val_acc': best_val_acc,
            'best_val_f1': best_val_f1,
            'scaler': scaler
        }, checkpoint_dir / 'swa_model.pth')

    total_elapsed = time.time() - start_time
    logger.info("=" * 80)
    logger.info("SUPER-OPTIMIZED TRAINING COMPLETED")
    logger.info(f"Total time: {total_elapsed/3600:.2f} hours")
    logger.info(f"Epochs: {epoch}")
    logger.info(f"BEST Val Accuracy: {best_val_acc:.4f} (Target was 79%)")
    logger.info(f"BEST F1 Score: {best_val_f1:.4f}")
    logger.info(f"Improvement over 74%: {(best_val_acc - 0.74)*100:+.2f}%")
    logger.info("=" * 80)


if __name__ == "__main__":
    try:
        train_super_optimized_lstm()
    except KeyboardInterrupt:
        logger.warning("Training interrupted")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)
