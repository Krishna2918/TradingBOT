"""
10-Hour LSTM Training - NO EARLY STOPPING
Uses 40% GPU for FULL 10 hours
Based on winning formula from train_lstm_production.py
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
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/lstm_10h_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=2, num_classes=3, dropout=0.2):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout if num_layers > 1 else 0, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


def train_lstm_10h():
    logger.info("=" * 80)
    logger.info("10-HOUR LSTM TRAINING - NO EARLY STOPPING")
    logger.info("=" * 80)

    # Configuration
    total_hours = 10.0
    sequence_length = 30
    batch_size = 256
    learning_rate = 8e-4

    # GPU setup - 40% usage
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.4, device=0)
        logger.info("GPU memory limited to 40%")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load data
    logger.info("Loading data...")
    features_dir = Path("TrainingData/features")
    feature_files = list(features_dir.glob("*_features.parquet"))[:30]

    all_data = []
    for file in feature_files:
        df = pd.read_parquet(file)
        if len(df) < 100:
            continue

        if 'direction_1d' not in df.columns and 'close' in df.columns:
            df['direction_1d'] = np.sign(df['close'].pct_change().shift(-1)).fillna(0).astype(int)

        all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"Loaded {len(combined_df)} rows from {len(all_data)} symbols")

    # Features
    feature_cols = [c for c in combined_df.columns if c not in ['date', 'timestamp', 'symbol', 'direction_1d', 'target']]
    features_df = combined_df[feature_cols].ffill().bfill().fillna(0).replace([np.inf, -np.inf], 0)

    scaler = RobustScaler()
    features_scaled = scaler.fit_transform(features_df)

    targets = combined_df['direction_1d'].values + 1
    targets = np.clip(targets, 0, 2)

    # Sequences
    X_seq = [features_scaled[i-sequence_length:i] for i in range(sequence_length, len(features_scaled))]
    y_seq = [targets[i] for i in range(sequence_length, len(targets))]

    X = np.array(X_seq)
    y = np.array(y_seq)

    # Split
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}")
    logger.info(f"Sequence shape: {X_train.shape}")

    # DataLoaders
    train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))

    # Balanced sampling
    class_counts = Counter(y_train)
    weights = [1.0 / class_counts[y] for y in y_train]
    sampler = WeightedRandomSampler(weights, len(weights))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, pin_memory=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True)

    # Model
    model = SimpleLSTM(X_train.shape[-1], 256, 2, 3, 0.2).to(device)
    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000, eta_min=1e-6)
    scaler_amp = GradScaler()

    # Training loop - NO EARLY STOPPING
    start_time = time.time()
    end_time = start_time + (total_hours * 3600)
    best_f1 = 0.0
    epoch = 0

    checkpoint_dir = Path("models/lstm_10h_checkpoint")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Training until: {datetime.fromtimestamp(end_time).strftime('%H:%M:%S')}")
    logger.info("NO EARLY STOPPING - Will use full 10 hours")

    while time.time() < end_time:
        epoch += 1

        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            if time.time() >= end_time:
                break

            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()

            with autocast():
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()

        # Validate
        model.eval()
        val_loss = 0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                with autocast():
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = train_correct / train_total
        val_acc = accuracy_score(val_targets, val_preds)
        f1 = f1_score(val_targets, val_preds, average='macro', zero_division=0)

        elapsed = time.time() - start_time
        remaining = end_time - time.time()

        logger.info(
            f"Epoch {epoch} - TrLoss: {train_loss:.4f}, TrAcc: {train_acc:.4f}, "
            f"ValLoss: {val_loss:.4f}, ValAcc: {val_acc:.4f}, F1: {f1:.4f}, "
            f"Time: {elapsed/3600:.2f}h/{remaining/3600:.2f}h"
        )

        # Save best
        if f1 > best_f1:
            best_f1 = f1
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'val_acc': val_acc,
                'f1': f1
            }, checkpoint_dir / 'best_model.pth')
            logger.info(f"NEW BEST: {val_acc:.4f} accuracy, {f1:.4f} F1")

        # Checkpoint every 100 epochs
        if epoch % 100 == 0:
            torch.save({'epoch': epoch, 'model': model.state_dict()}, checkpoint_dir / f'epoch_{epoch}.pth')

        scheduler.step()

    logger.info("=" * 80)
    logger.info(f"TRAINING COMPLETE - {epoch} epochs in {(time.time()-start_time)/3600:.2f} hours")
    logger.info(f"Best F1: {best_f1:.4f}")
    logger.info("=" * 80)


if __name__ == "__main__":
    train_lstm_10h()
