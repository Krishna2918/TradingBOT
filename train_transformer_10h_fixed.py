"""
10-Hour Transformer Training - Fixed Memory Bug
================================================

This script trains the MarketTransformer model for 10 hours with:
- Fixed memory allocation bug
- Automatic checkpointing every 10 epochs
- Graceful termination
- 40% resource usage
- Resume from existing checkpoint
"""

import sys
import os
import time
import signal
from pathlib import Path
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging

sys.path.append(str(Path(__file__).parent / "src"))

from ai.models.market_transformer import MarketTransformer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/transformer_10h_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class FixedTransformerDataset(Dataset):
    """Fixed dataset that handles pre-sequenced data correctly"""

    def __init__(self, features, targets, time_features=None, volatility=None):
        """
        Args:
            features: Already sequenced [n_samples, seq_len, n_features]
            targets: Target labels [n_samples]
            time_features: Time features [n_samples, seq_len, 4]
            volatility: Volatility [n_samples, seq_len, 1]
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.LongTensor(targets)
        self.time_features = torch.LongTensor(time_features) if time_features is not None else None
        self.volatility = torch.FloatTensor(volatility) if volatility is not None else None

        logger.info(f"Dataset created: {len(self)} samples, shape {self.features.shape}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        item = {
            'features': self.features[idx],
            'targets': self.targets[idx]
        }

        if self.time_features is not None:
            item['time_features'] = self.time_features[idx]

        if self.volatility is not None:
            item['volatility'] = self.volatility[idx]

        return item


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


def load_training_data():
    """Load pre-processed training data"""
    logger.info("Loading training data...")

    # Import the data loading function
    from train_market_transformer_full_scale import load_real_trading_data

    features, targets, time_features, volatility, feature_names = load_real_trading_data()

    logger.info(f"Data loaded: Features {features.shape}, Targets {targets.shape}")
    return features, targets, time_features, volatility, feature_names


def configure_resources(resource_limit=0.4):
    """Configure 40% resource usage"""
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(resource_limit, device=0)
        logger.info(f"GPU memory limited to {resource_limit*100}%")

    num_cpus = os.cpu_count() or 1
    limited_threads = max(1, int(num_cpus * resource_limit))
    torch.set_num_threads(limited_threads)
    logger.info(f"CPU threads limited to {limited_threads}")


def train_transformer_10h():
    """Train Transformer for 10 hours"""
    logger.info("=" * 80)
    logger.info("10-HOUR TRANSFORMER TRAINING SESSION")
    logger.info("=" * 80)

    # Configuration
    total_hours = 10.0
    resource_limit = 0.4
    checkpoint_dir = Path("models/transformer_checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Configure resources
    configure_resources(resource_limit)

    # Setup graceful shutdown
    killer = GracefulKiller()

    # Load data
    features, targets, time_features, volatility, feature_names = load_training_data()

    # Split data
    n_samples = len(features)
    n_train = int(n_samples * 0.8)
    n_val = int(n_samples * 0.9)

    X_train = features[:n_train]
    y_train = targets[:n_train]
    time_train = time_features[:n_train] if time_features is not None else None
    vol_train = volatility[:n_train] if volatility is not None else None

    X_val = features[n_train:n_val]
    y_val = targets[n_train:n_val]
    time_val = time_features[n_train:n_val] if time_features is not None else None
    vol_val = volatility[n_train:n_val] if volatility is not None else None

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}")

    # Create datasets with FIXED memory handling
    train_dataset = FixedTransformerDataset(X_train, y_train, time_train, vol_train)
    val_dataset = FixedTransformerDataset(X_val, y_val, time_val, vol_val)

    # DataLoaders with reduced batch size for stability
    batch_size = 4  # Even smaller batch size to prevent crashes
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_config = {
        'input_dim': features.shape[2],
        'd_model': 256,
        'num_heads': 8,
        'num_layers': 6,
        'd_ff': 1024,
        'max_seq_length': features.shape[1],
        'num_classes': 3,
        'dropout': 0.1
    }

    model = MarketTransformer(**model_config).to(device)

    # Check for existing checkpoint
    best_checkpoint = checkpoint_dir / "best_model.pth"
    start_epoch = 0
    best_val_loss = float('inf')

    if best_checkpoint.exists():
        logger.info(f"Loading checkpoint from {best_checkpoint}")
        checkpoint = torch.load(best_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        logger.info(f"Resumed from epoch {start_epoch}, best val loss: {best_val_loss:.4f}")

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    start_time = time.time()
    end_time = start_time + (total_hours * 3600)

    logger.info(f"Training until: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Starting from epoch {start_epoch}")

    epoch = start_epoch
    max_epochs = 1000  # High limit, will stop based on time

    while time.time() < end_time and epoch < max_epochs and not killer.kill_now:
        epoch += 1
        remaining_time = end_time - time.time()

        if remaining_time < 300:  # Less than 5 minutes
            logger.info("Less than 5 minutes remaining, stopping training")
            break

        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            if killer.kill_now or time.time() >= end_time:
                break

            features_batch = batch['features'].to(device)
            targets_batch = batch['targets'].to(device)
            time_batch = batch.get('time_features')
            vol_batch = batch.get('volatility')

            if time_batch is not None:
                time_batch = time_batch.to(device)
            if vol_batch is not None:
                vol_batch = vol_batch.to(device)

            optimizer.zero_grad()

            # Forward pass
            output = model(features_batch, time_batch, vol_batch)
            loss = criterion(output['logits'], targets_batch)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(output['logits'], 1)
            train_total += targets_batch.size(0)
            train_correct += (predicted == targets_batch).sum().item()

            if batch_idx % 100 == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                features_batch = batch['features'].to(device)
                targets_batch = batch['targets'].to(device)
                time_batch = batch.get('time_features')
                vol_batch = batch.get('volatility')

                if time_batch is not None:
                    time_batch = time_batch.to(device)
                if vol_batch is not None:
                    vol_batch = vol_batch.to(device)

                output = model(features_batch, time_batch, vol_batch)
                loss = criterion(output['logits'], targets_batch)

                val_loss += loss.item()
                _, predicted = torch.max(output['logits'], 1)
                val_total += targets_batch.size(0)
                val_correct += (predicted == targets_batch).sum().item()

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        epoch_time = time.time() - epoch_start
        elapsed_total = time.time() - start_time

        logger.info(f"Epoch {epoch}/{max_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                   f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Time: {epoch_time:.1f}s, "
                   f"Elapsed: {elapsed_total/3600:.2f}h")

        # Save checkpoint every 10 epochs or if best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'val_acc': val_acc,
                'train_acc': train_acc
            }
            torch.save(checkpoint, best_checkpoint)
            logger.info(f"Saved best model! Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if epoch % 10 == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss
            }
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint at epoch {epoch}")

        scheduler.step()

    # Final save
    final_checkpoint = checkpoint_dir / f"final_epoch_{epoch}.pth"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss
    }, final_checkpoint)

    total_elapsed = time.time() - start_time
    logger.info("=" * 80)
    logger.info("TRAINING COMPLETED")
    logger.info(f"Total time: {total_elapsed/3600:.2f} hours")
    logger.info(f"Epochs completed: {epoch - start_epoch}")
    logger.info(f"Best val loss: {best_val_loss:.4f}")
    logger.info("=" * 80)


if __name__ == "__main__":
    try:
        train_transformer_10h()
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)
