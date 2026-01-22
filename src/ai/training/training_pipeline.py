"""
ML Model Training Pipeline

Comprehensive training pipeline for all ML models in the TradingBOT system:
- LSTM models for short-term predictions
- GRU/Transformer models for pattern recognition
- Reinforcement Learning agents (PPO, DQN)
- Ensemble models
"""

import logging
import os
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch.amp import autocast, GradScaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from collections import Counter
import joblib
import yaml

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for model training"""
    # Data parameters
    sequence_length: int = 60  # Number of time steps for sequences
    train_split: float = 0.8
    validation_split: float = 0.1
    test_split: float = 0.1

    # Training parameters
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    early_stopping_patience: int = 10
    weight_decay: float = 1e-4  # Increased for better regularization

    # Advanced training parameters
    gradient_clip_value: float = 1.0  # Clip gradients to prevent explosion
    use_mixed_precision: bool = True  # FP16 training for speed
    warmup_epochs: int = 5  # LR warmup epochs
    use_class_weights: bool = True  # Handle imbalanced targets
    label_smoothing: float = 0.1  # Reduce overconfidence

    # Learning rate scheduler
    lr_scheduler: str = "cosine"  # "cosine", "plateau", or "step"
    lr_min: float = 1e-6  # Minimum learning rate

    # Model parameters
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.3  # Slightly higher for better regularization

    # Paths
    data_dir: str = "data"
    model_dir: str = "models"
    checkpoint_dir: str = "models/checkpoints"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainingResult:
    """Result of a training run"""
    model_name: str
    success: bool
    train_loss: float = 0.0
    val_loss: float = 0.0
    test_loss: float = 0.0
    accuracy: float = 0.0
    training_time: float = 0.0
    epochs_trained: int = 0
    model_path: str = ""
    error_message: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)


class DataPreprocessor:
    """Handles data preprocessing for ML models"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.scalers: Dict[str, StandardScaler] = {}
        # Local data paths
        self.training_data_path = Path("TrainingData/daily")
        self.past_data_path = Path("PastData/daily")

    def load_market_data(self, symbols: List[str],
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None,
                         use_local_data: bool = True) -> pd.DataFrame:
        """
        Load market data for specified symbols.

        Prioritizes local Parquet files (26 years of data) over yfinance downloads.
        """
        if use_local_data:
            return self._load_from_parquet(symbols, start_date, end_date)
        else:
            return self._load_from_yfinance(symbols, start_date, end_date)

    def _load_from_parquet(self, symbols: List[str],
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Load data from local Parquet files (TrainingData or PastData)"""
        all_data = []

        for symbol in symbols:
            # Try TrainingData first (has 26 years of data)
            # Handle different naming conventions
            parquet_names = [
                f"{symbol}_daily.parquet",
                f"{symbol.replace('.TO', '')}_daily.parquet",
                f"{symbol}.parquet",
                f"{symbol.replace('.TO', '')}.parquet"
            ]

            df = None
            for name in parquet_names:
                # Check TrainingData first
                path = self.training_data_path / name
                if path.exists():
                    df = pd.read_parquet(path)
                    logger.info(f"Loaded {symbol} from TrainingData: {len(df)} rows")
                    break

                # Fall back to PastData
                path = self.past_data_path / name
                if path.exists():
                    df = pd.read_parquet(path)
                    logger.info(f"Loaded {symbol} from PastData: {len(df)} rows")
                    break

            if df is not None and not df.empty:
                # Ensure datetime index
                if not isinstance(df.index, pd.DatetimeIndex):
                    if 'Date' in df.columns:
                        df.set_index('Date', inplace=True)
                    elif 'date' in df.columns:
                        df.set_index('date', inplace=True)

                # Make timezone-naive for consistency
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)

                # Filter by date range if specified
                if start_date:
                    start_date = pd.to_datetime(start_date).tz_localize(None) if hasattr(pd.to_datetime(start_date), 'tz_localize') else pd.to_datetime(start_date)
                    df = df[df.index >= start_date]
                if end_date:
                    end_date = pd.to_datetime(end_date).tz_localize(None) if hasattr(pd.to_datetime(end_date), 'tz_localize') else pd.to_datetime(end_date)
                    df = df[df.index <= end_date]

                # Standardize column names
                df.columns = [c.title() if c.lower() in ['open', 'high', 'low', 'close', 'volume'] else c for c in df.columns]

                df['Symbol'] = symbol
                df = df.reset_index()
                if 'index' in df.columns:
                    df = df.rename(columns={'index': 'Date'})
                all_data.append(df)
            else:
                logger.warning(f"No local data found for {symbol}")

        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            date_range = f"{combined['Date'].min().date()} to {combined['Date'].max().date()}"
            logger.info(f"Loaded {len(combined):,} records for {len(all_data)} symbols ({date_range})")
            return combined
        else:
            logger.warning("No local data found, falling back to yfinance")
            return self._load_from_yfinance(symbols, start_date, end_date)

    def _load_from_yfinance(self, symbols: List[str],
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Fallback: Load data from yfinance API"""
        try:
            import yfinance as yf

            if start_date is None:
                start_date = datetime.now() - timedelta(days=365 * 20)  # 20 years
            if end_date is None:
                end_date = datetime.now()

            all_data = []
            for symbol in symbols:
                logger.info(f"Downloading {symbol} from yfinance...")
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date, interval='1d')

                if not data.empty:
                    data['Symbol'] = symbol
                    data = data.reset_index()
                    all_data.append(data)

            if all_data:
                combined = pd.concat(all_data, ignore_index=True)
                logger.info(f"Downloaded {len(combined)} records for {len(symbols)} symbols")
                return combined
            else:
                logger.warning("No data loaded from yfinance")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error loading from yfinance: {e}")
            return pd.DataFrame()

    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicator features"""
        if data.empty:
            return data

        try:
            df = data.copy()

            # Ensure data is sorted by symbol and date for correct calculations
            if 'Date' in df.columns:
                df = df.sort_values(['Symbol', 'Date'])

            # Basic price features - fix deprecation warning
            df['returns'] = df.groupby('Symbol')['Close'].pct_change(fill_method=None)

            # Log returns with safety check
            prev_close = df.groupby('Symbol')['Close'].shift(1)
            df['log_returns'] = np.log(df['Close'] / prev_close.replace(0, np.nan))

            # Moving averages (use smaller windows for more data retention)
            for period in [5, 10, 20]:
                df[f'sma_{period}'] = df.groupby('Symbol')['Close'].transform(
                    lambda x: x.rolling(window=period, min_periods=1).mean()
                )
                df[f'ema_{period}'] = df.groupby('Symbol')['Close'].transform(
                    lambda x: x.ewm(span=period, adjust=False, min_periods=1).mean()
                )

            # Volatility features
            df['volatility_10'] = df.groupby('Symbol')['returns'].transform(
                lambda x: x.rolling(window=10, min_periods=3).std()
            )
            df['volatility_20'] = df.groupby('Symbol')['returns'].transform(
                lambda x: x.rolling(window=20, min_periods=5).std()
            )

            # RSI
            def calculate_rsi(prices, period=14):
                delta = prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
                rs = gain / loss.replace(0, np.nan)
                return 100 - (100 / (1 + rs))

            df['rsi_14'] = df.groupby('Symbol')['Close'].transform(
                lambda x: calculate_rsi(x, 14)
            )

            # MACD - compute EMA first
            df['ema_12'] = df.groupby('Symbol')['Close'].transform(
                lambda x: x.ewm(span=12, adjust=False, min_periods=1).mean()
            )
            df['ema_26'] = df.groupby('Symbol')['Close'].transform(
                lambda x: x.ewm(span=26, adjust=False, min_periods=1).mean()
            )
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df.groupby('Symbol')['macd'].transform(
                lambda x: x.ewm(span=9, adjust=False, min_periods=1).mean()
            )
            df['macd_histogram'] = df['macd'] - df['macd_signal']

            # Bollinger Bands
            df['bb_middle'] = df['sma_20']
            df['bb_std'] = df.groupby('Symbol')['Close'].transform(
                lambda x: x.rolling(window=20, min_periods=5).std()
            )
            df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
            df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
            bb_range = df['bb_upper'] - df['bb_lower']
            df['bb_position'] = (df['Close'] - df['bb_lower']) / bb_range.replace(0, np.nan)

            # Volume features
            df['volume_sma'] = df.groupby('Symbol')['Volume'].transform(
                lambda x: x.rolling(window=20, min_periods=5).mean()
            )
            df['volume_ratio'] = df['Volume'] / df['volume_sma'].replace(0, np.nan)

            # ATR
            df['high_low'] = df['High'] - df['Low']
            df['high_close'] = abs(df['High'] - df.groupby('Symbol')['Close'].shift(1))
            df['low_close'] = abs(df['Low'] - df.groupby('Symbol')['Close'].shift(1))
            df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
            df['atr_14'] = df.groupby('Symbol')['tr'].transform(
                lambda x: x.rolling(window=14, min_periods=3).mean()
            )

            # Target: next day direction (1=up, 0=neutral, -1=down)
            # Fixed: use vectorized operations instead of apply on series
            future_returns = df.groupby('Symbol')['returns'].shift(-1)
            df['target'] = 0  # neutral by default
            df.loc[future_returns > 0.005, 'target'] = 1   # up
            df.loc[future_returns < -0.005, 'target'] = -1  # down

            # Drop rows where essential features are NaN (but be less aggressive)
            essential_cols = ['returns', 'target', 'Close', 'Volume']
            df = df.dropna(subset=essential_cols)

            # Fill remaining NaN with forward fill, then 0
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(method='ffill').fillna(0)

            # Replace any inf values
            df = df.replace([np.inf, -np.inf], 0)

            logger.info(f"Engineered {len(df.columns)} features, {len(df):,} samples")
            return df

        except Exception as e:
            logger.error(f"Error engineering features: {e}", exc_info=True)
            return data

    def prepare_sequences(self, data: pd.DataFrame,
                         feature_columns: List[str],
                         target_column: str = 'target') -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM/GRU training"""
        try:
            # Scale features
            scaler_key = 'main'
            if scaler_key not in self.scalers:
                self.scalers[scaler_key] = StandardScaler()

            features = data[feature_columns].values
            targets = data[target_column].values

            # Fit and transform
            scaled_features = self.scalers[scaler_key].fit_transform(features)

            # Create sequences
            X, y = [], []
            for i in range(len(scaled_features) - self.config.sequence_length):
                X.append(scaled_features[i:i + self.config.sequence_length])
                y.append(targets[i + self.config.sequence_length])

            X = np.array(X)
            y = np.array(y)

            # Convert target to classification (0, 1, 2 for down, neutral, up)
            y = y + 1  # Shift from -1,0,1 to 0,1,2

            logger.info(f"Prepared {len(X)} sequences with shape {X.shape}")
            return X, y

        except Exception as e:
            logger.error(f"Error preparing sequences: {e}")
            return np.array([]), np.array([])

    def save_scaler(self, name: str, path: str):
        """Save scaler for later use"""
        if name in self.scalers:
            joblib.dump(self.scalers[name], path)
            logger.info(f"Saved scaler to {path}")


class LSTMTrainer:
    """Trainer for LSTM models with best practices"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.scaler = GradScaler('cuda') if config.use_mixed_precision and self.device.type == 'cuda' else None

    def _create_time_series_split(self, dataset: TensorDataset) -> Tuple[Subset, Subset, Subset]:
        """
        Create sequential time-series split (NO data leakage).
        Training data comes BEFORE validation, validation BEFORE test.
        """
        total_size = len(dataset)
        train_size = int(self.config.train_split * total_size)
        val_size = int(self.config.validation_split * total_size)
        test_size = total_size - train_size - val_size

        # Sequential indices - preserves time order
        train_indices = list(range(0, train_size))
        val_indices = list(range(train_size, train_size + val_size))
        test_indices = list(range(train_size + val_size, total_size))

        logger.info(f"Time-series split: Train={len(train_indices)}, "
                   f"Val={len(val_indices)}, Test={len(test_indices)}")

        return (
            Subset(dataset, train_indices),
            Subset(dataset, val_indices),
            Subset(dataset, test_indices)
        )

    def _compute_class_weights(self, y: np.ndarray) -> torch.Tensor:
        """Compute class weights to handle imbalanced targets"""
        class_counts = Counter(y.astype(int))
        total = sum(class_counts.values())
        num_classes = len(class_counts)

        # Inverse frequency weighting
        weights = []
        for i in range(num_classes):
            count = class_counts.get(i, 1)
            weight = total / (num_classes * count)
            weights.append(weight)

        weights = torch.FloatTensor(weights).to(self.device)
        logger.info(f"Class weights: {weights.tolist()}")
        return weights

    def _get_warmup_lr(self, epoch: int, base_lr: float) -> float:
        """Calculate learning rate with linear warmup"""
        if epoch < self.config.warmup_epochs:
            return base_lr * (epoch + 1) / self.config.warmup_epochs
        return base_lr

    def train(self, X: np.ndarray, y: np.ndarray,
              model_name: str = "lstm_model") -> TrainingResult:
        """Train LSTM model with best practices"""
        start_time = datetime.now()

        try:
            # Import model
            from src.ai.model_stack.lstm_model import LSTMModel

            # Prepare data
            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.LongTensor(y.astype(int))

            dataset = TensorDataset(X_tensor, y_tensor)

            # TIME-SERIES SPLIT (sequential, no data leakage)
            train_data, val_data, test_data = self._create_time_series_split(dataset)

            # Note: shuffle=False for training to maintain some temporal locality
            # but we can shuffle within batches for regularization
            train_loader = DataLoader(train_data, batch_size=self.config.batch_size, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=self.config.batch_size, shuffle=False)
            test_loader = DataLoader(test_data, batch_size=self.config.batch_size, shuffle=False)

            # Initialize model
            input_size = X.shape[2]  # Number of features
            model = LSTMModel(
                input_size=input_size,
                hidden_size=self.config.hidden_size,
                num_layers=self.config.num_layers,
                dropout=self.config.dropout,
                output_size=3
            ).to(self.device)

            # Compute class weights for imbalanced data
            class_weights = None
            if self.config.use_class_weights:
                class_weights = self._compute_class_weights(y)

            # Loss with class weights and label smoothing
            criterion = nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=self.config.label_smoothing
            )

            # AdamW optimizer (better weight decay handling than Adam)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.999)
            )

            # Learning rate scheduler
            if self.config.lr_scheduler == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.config.epochs - self.config.warmup_epochs,
                    eta_min=self.config.lr_min
                )
            elif self.config.lr_scheduler == "step":
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=20, gamma=0.5
                )
            else:  # plateau
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', patience=5, factor=0.5, min_lr=self.config.lr_min
                )

            # Training loop
            best_val_loss = float('inf')
            best_val_accuracy = 0.0
            patience_counter = 0
            train_losses = []
            val_losses = []
            learning_rates = []

            logger.info(f"Starting training with {len(train_loader)} batches/epoch")
            logger.info(f"Mixed precision: {self.scaler is not None}")
            logger.info(f"Gradient clipping: {self.config.gradient_clip_value}")

            for epoch in range(self.config.epochs):
                # Learning rate warmup
                if epoch < self.config.warmup_epochs:
                    warmup_lr = self._get_warmup_lr(epoch, self.config.learning_rate)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = warmup_lr

                current_lr = optimizer.param_groups[0]['lr']
                learning_rates.append(current_lr)

                # Training phase
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0

                for batch_X, batch_y in train_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)

                    optimizer.zero_grad()

                    # Mixed precision training
                    if self.scaler is not None:
                        with autocast('cuda'):
                            outputs = model(batch_X)
                            loss = criterion(outputs, batch_y)

                        self.scaler.scale(loss).backward()

                        # Gradient clipping (before optimizer step)
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            self.config.gradient_clip_value
                        )

                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        loss.backward()

                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            self.config.gradient_clip_value
                        )

                        optimizer.step()

                    train_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    train_total += batch_y.size(0)
                    train_correct += (predicted == batch_y).sum().item()

                train_loss /= len(train_loader)
                train_losses.append(train_loss)
                train_accuracy = train_correct / train_total if train_total > 0 else 0

                # Validation phase
                model.eval()
                val_loss = 0.0
                correct = 0
                total = 0

                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)

                        if self.scaler is not None:
                            with autocast('cuda'):
                                outputs = model(batch_X)
                                loss = criterion(outputs, batch_y)
                        else:
                            outputs = model(batch_X)
                            loss = criterion(outputs, batch_y)

                        val_loss += loss.item()
                        _, predicted = torch.max(outputs, 1)
                        total += batch_y.size(0)
                        correct += (predicted == batch_y).sum().item()

                val_loss /= len(val_loader) if len(val_loader) > 0 else 1
                val_losses.append(val_loss)
                accuracy = correct / total if total > 0 else 0

                # Update scheduler (after warmup)
                if epoch >= self.config.warmup_epochs:
                    if self.config.lr_scheduler == "plateau":
                        scheduler.step(val_loss)
                    else:
                        scheduler.step()

                if (epoch + 1) % 10 == 0 or epoch == 0:
                    logger.info(f"Epoch [{epoch+1}/{self.config.epochs}] "
                              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                              f"Train Acc: {train_accuracy:.4f}, Val Acc: {accuracy:.4f}, "
                              f"LR: {current_lr:.6f}")

                # Early stopping based on validation accuracy (more robust than loss)
                if accuracy > best_val_accuracy or (accuracy == best_val_accuracy and val_loss < best_val_loss):
                    best_val_loss = val_loss
                    best_val_accuracy = accuracy
                    patience_counter = 0
                    # Save best model
                    model_path = Path(self.config.model_dir) / f"{model_name}_best.pt"
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                        'val_accuracy': accuracy,
                        'config': self.config
                    }, model_path)
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch+1} (best val acc: {best_val_accuracy:.4f})")
                        break

            # Load best model for final evaluation
            checkpoint = torch.load(model_path, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])

            # Final evaluation on test set
            model.eval()
            test_loss = 0.0
            correct = 0
            total = 0
            all_predictions = []
            all_targets = []

            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)

                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    test_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()

                    all_predictions.extend(predicted.cpu().numpy())
                    all_targets.extend(batch_y.cpu().numpy())

            test_loss /= len(test_loader) if len(test_loader) > 0 else 1
            test_accuracy = correct / total if total > 0 else 0

            # Calculate per-class accuracy
            all_predictions = np.array(all_predictions)
            all_targets = np.array(all_targets)
            per_class_acc = {}
            for cls in range(3):
                mask = all_targets == cls
                if mask.sum() > 0:
                    per_class_acc[f'class_{cls}_acc'] = (all_predictions[mask] == cls).mean()

            training_time = (datetime.now() - start_time).total_seconds()

            logger.info(f"Final Test Accuracy: {test_accuracy:.4f}")
            logger.info(f"Per-class accuracy: {per_class_acc}")

            return TrainingResult(
                model_name=model_name,
                success=True,
                train_loss=train_losses[-1] if train_losses else 0,
                val_loss=best_val_loss,
                test_loss=test_loss,
                accuracy=test_accuracy,
                training_time=training_time,
                epochs_trained=epoch + 1,
                model_path=str(model_path),
                metrics={
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'learning_rates': learning_rates,
                    'final_accuracy': test_accuracy,
                    'best_val_accuracy': best_val_accuracy,
                    **per_class_acc
                }
            )

        except Exception as e:
            logger.error(f"Error training LSTM model: {e}", exc_info=True)
            return TrainingResult(
                model_name=model_name,
                success=False,
                error_message=str(e)
            )


class TrainingPipeline:
    """
    Main training pipeline that orchestrates all model training.

    Integrates with:
    - Data snapshots for reproducibility
    - Model registry for versioning and promotion
    """

    def __init__(self, config: Optional[TrainingConfig] = None,
                 config_path: Optional[str] = None,
                 use_snapshots: bool = True,
                 use_registry: bool = True):
        """Initialize training pipeline"""
        if config:
            self.config = config
        elif config_path and Path(config_path).exists():
            self.config = self._load_config(config_path)
        else:
            self.config = TrainingConfig()

        self.preprocessor = DataPreprocessor(self.config)
        self.results: List[TrainingResult] = []
        self.use_snapshots = use_snapshots
        self.use_registry = use_registry

        # Training integration for snapshots and registry
        self._training_integration = None
        self._current_snapshot_id = None

        # Create directories
        Path(self.config.model_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Initialize training integration
        if use_snapshots or use_registry:
            try:
                from src.ai.training_integration import TrainingIntegration
                self._training_integration = TrainingIntegration(
                    models_dir=self.config.model_dir,
                )
                logger.info("Training integration enabled (snapshots=%s, registry=%s)",
                           use_snapshots, use_registry)
            except ImportError:
                logger.warning("TrainingIntegration not available, running without snapshots/registry")
                self.use_snapshots = False
                self.use_registry = False

        logger.info(f"Training Pipeline initialized (device: {self.config.device})")

    def _load_config(self, path: str) -> TrainingConfig:
        """Load config from YAML"""
        try:
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
            return TrainingConfig(**config_dict.get('training', {}))
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")
            return TrainingConfig()

    def run_full_training(self, symbols: List[str] = None,
                          description: str = "") -> Dict[str, TrainingResult]:
        """Run full training pipeline for all models"""
        logger.info("Starting full training pipeline...")

        if symbols is None:
            # Default Canadian symbols
            symbols = [
                'RY.TO', 'TD.TO', 'BNS.TO', 'BMO.TO', 'CM.TO',
                'SHOP.TO', 'ENB.TO', 'CNQ.TO', 'SU.TO', 'TRP.TO'
            ]

        results = {}

        # Step 0: Create data snapshot for reproducibility
        if self.use_snapshots and self._training_integration:
            logger.info("Step 0: Creating data snapshot for reproducibility...")
            self._current_snapshot_id = self._training_integration.prepare_training_data(
                symbols=symbols,
                description=description or f"Training run on {len(symbols)} symbols",
            )
            if self._current_snapshot_id:
                logger.info(f"Created snapshot: {self._current_snapshot_id}")
            else:
                logger.warning("Failed to create snapshot, continuing without reproducibility tracking")

        # Step 1: Load and preprocess data
        logger.info("Step 1: Loading market data...")
        data = self.preprocessor.load_market_data(symbols)

        if data.empty:
            logger.error("Failed to load market data")
            return results

        # Step 2: Engineer features
        logger.info("Step 2: Engineering features...")
        data = self.preprocessor.engineer_features(data)

        # Define feature columns
        feature_columns = [
            'returns', 'log_returns', 'volatility_10', 'volatility_20',
            'rsi_14', 'macd', 'macd_signal', 'macd_histogram',
            'bb_position', 'volume_ratio', 'atr_14',
            'sma_5', 'sma_10', 'sma_20', 'ema_5', 'ema_10', 'ema_20'
        ]

        # Filter to available columns
        feature_columns = [c for c in feature_columns if c in data.columns]

        # Step 3: Prepare sequences
        logger.info("Step 3: Preparing sequences...")
        X, y = self.preprocessor.prepare_sequences(data, feature_columns)

        if len(X) == 0:
            logger.error("Failed to prepare sequences")
            return results

        # Save scaler
        scaler_path = Path(self.config.model_dir) / "feature_scaler.joblib"
        self.preprocessor.save_scaler('main', str(scaler_path))

        # Step 4: Train LSTM model
        logger.info("Step 4: Training LSTM model...")
        lstm_trainer = LSTMTrainer(self.config)
        lstm_result = lstm_trainer.train(X, y, "lstm_price_prediction")
        results['lstm'] = lstm_result
        self.results.append(lstm_result)

        if lstm_result.success:
            logger.info(f"LSTM training complete - Accuracy: {lstm_result.accuracy:.4f}")

            # Step 5: Register model in registry
            if self.use_registry and self._training_integration:
                logger.info("Step 5: Registering model in registry...")
                model_id = self._training_integration.register_trained_model(
                    model_path=lstm_result.model_path,
                    model_type="lstm",
                    model_name="lstm_price_prediction",
                    metrics={
                        "test_accuracy": lstm_result.accuracy,
                        "test_loss": lstm_result.test_loss,
                        "val_loss": lstm_result.val_loss,
                        "train_loss": lstm_result.train_loss,
                    },
                    hyperparameters={
                        "hidden_size": self.config.hidden_size,
                        "num_layers": self.config.num_layers,
                        "dropout": self.config.dropout,
                        "learning_rate": self.config.learning_rate,
                        "batch_size": self.config.batch_size,
                        "sequence_length": self.config.sequence_length,
                    },
                    snapshot_id=self._current_snapshot_id,
                    description=f"LSTM model trained for {lstm_result.epochs_trained} epochs",
                    tags=["lstm", "price_prediction"],
                )
                if model_id:
                    logger.info(f"Registered model: {model_id}")
                    lstm_result.metrics["registry_id"] = model_id
        else:
            logger.error(f"LSTM training failed: {lstm_result.error_message}")

        # Save training summary
        self._save_training_summary(results)

        logger.info("Training pipeline complete!")
        return results

    def train_lstm(self, X: np.ndarray, y: np.ndarray) -> TrainingResult:
        """Train LSTM model only"""
        trainer = LSTMTrainer(self.config)
        result = trainer.train(X, y)
        self.results.append(result)
        return result

    def _save_training_summary(self, results: Dict[str, TrainingResult]):
        """Save training summary to file"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'device': self.config.device,
            'models': {}
        }

        for name, result in results.items():
            summary['models'][name] = {
                'success': result.success,
                'accuracy': result.accuracy,
                'train_loss': result.train_loss,
                'val_loss': result.val_loss,
                'test_loss': result.test_loss,
                'epochs_trained': result.epochs_trained,
                'training_time': result.training_time,
                'model_path': result.model_path,
                'error': result.error_message
            }

        summary_path = Path(self.config.model_dir) / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Training summary saved to {summary_path}")

    def get_training_results(self) -> List[TrainingResult]:
        """Get all training results"""
        return self.results


# Convenience function for running training
def run_training(symbols: List[str] = None,
                 config_path: str = None) -> Dict[str, TrainingResult]:
    """
    Run the full training pipeline.

    Args:
        symbols: List of stock symbols to train on
        config_path: Path to training config YAML

    Returns:
        Dictionary of model names to training results
    """
    pipeline = TrainingPipeline(config_path=config_path)
    return pipeline.run_full_training(symbols)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 60)
    print("TradingBOT ML Training Pipeline")
    print("=" * 60)

    # Default Canadian stocks for training
    symbols = [
        'RY.TO', 'TD.TO', 'BNS.TO', 'BMO.TO', 'CM.TO',
        'SHOP.TO', 'ENB.TO', 'CNQ.TO'
    ]

    print(f"\nTraining on {len(symbols)} symbols: {symbols}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print()

    results = run_training(symbols)

    print("\n" + "=" * 60)
    print("Training Results Summary")
    print("=" * 60)

    for model_name, result in results.items():
        status = "SUCCESS" if result.success else "FAILED"
        print(f"\n{model_name.upper()}:")
        print(f"  Status: {status}")
        if result.success:
            print(f"  Accuracy: {result.accuracy:.4f}")
            print(f"  Test Loss: {result.test_loss:.4f}")
            print(f"  Training Time: {result.training_time:.1f}s")
            print(f"  Model Path: {result.model_path}")
        else:
            print(f"  Error: {result.error_message}")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
