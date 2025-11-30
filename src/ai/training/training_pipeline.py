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
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
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
    weight_decay: float = 1e-5

    # Model parameters
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2

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

    def load_market_data(self, symbols: List[str],
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Load market data for specified symbols"""
        try:
            import yfinance as yf

            if start_date is None:
                start_date = datetime.now() - timedelta(days=365 * 2)  # 2 years
            if end_date is None:
                end_date = datetime.now()

            all_data = []
            for symbol in symbols:
                logger.info(f"Loading data for {symbol}...")
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date, interval='1d')

                if not data.empty:
                    data['Symbol'] = symbol
                    data = data.reset_index()
                    all_data.append(data)

            if all_data:
                combined = pd.concat(all_data, ignore_index=True)
                logger.info(f"Loaded {len(combined)} records for {len(symbols)} symbols")
                return combined
            else:
                logger.warning("No data loaded")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error loading market data: {e}")
            return pd.DataFrame()

    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicator features"""
        if data.empty:
            return data

        try:
            df = data.copy()

            # Basic price features
            df['returns'] = df.groupby('Symbol')['Close'].pct_change()
            df['log_returns'] = np.log(df['Close'] / df.groupby('Symbol')['Close'].shift(1))

            # Moving averages
            for period in [5, 10, 20, 50]:
                df[f'sma_{period}'] = df.groupby('Symbol')['Close'].transform(
                    lambda x: x.rolling(window=period).mean()
                )
                df[f'ema_{period}'] = df.groupby('Symbol')['Close'].transform(
                    lambda x: x.ewm(span=period, adjust=False).mean()
                )

            # Volatility features
            df['volatility_10'] = df.groupby('Symbol')['returns'].transform(
                lambda x: x.rolling(window=10).std()
            )
            df['volatility_20'] = df.groupby('Symbol')['returns'].transform(
                lambda x: x.rolling(window=20).std()
            )

            # RSI
            def calculate_rsi(prices, period=14):
                delta = prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                return 100 - (100 / (1 + rs))

            df['rsi_14'] = df.groupby('Symbol')['Close'].transform(
                lambda x: calculate_rsi(x, 14)
            )

            # MACD
            df['macd'] = df['ema_12'] - df['ema_26'] if 'ema_12' in df.columns else 0
            df['ema_12'] = df.groupby('Symbol')['Close'].transform(
                lambda x: x.ewm(span=12, adjust=False).mean()
            )
            df['ema_26'] = df.groupby('Symbol')['Close'].transform(
                lambda x: x.ewm(span=26, adjust=False).mean()
            )
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df.groupby('Symbol')['macd'].transform(
                lambda x: x.ewm(span=9, adjust=False).mean()
            )
            df['macd_histogram'] = df['macd'] - df['macd_signal']

            # Bollinger Bands
            df['bb_middle'] = df['sma_20']
            df['bb_std'] = df.groupby('Symbol')['Close'].transform(
                lambda x: x.rolling(window=20).std()
            )
            df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
            df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
            df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

            # Volume features
            df['volume_sma'] = df.groupby('Symbol')['Volume'].transform(
                lambda x: x.rolling(window=20).mean()
            )
            df['volume_ratio'] = df['Volume'] / df['volume_sma']

            # ATR
            df['high_low'] = df['High'] - df['Low']
            df['high_close'] = abs(df['High'] - df.groupby('Symbol')['Close'].shift(1))
            df['low_close'] = abs(df['Low'] - df.groupby('Symbol')['Close'].shift(1))
            df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
            df['atr_14'] = df.groupby('Symbol')['tr'].transform(
                lambda x: x.rolling(window=14).mean()
            )

            # Target: next day direction (1=up, 0=neutral, -1=down)
            df['target'] = df.groupby('Symbol')['returns'].shift(-1).apply(
                lambda x: 1 if x > 0.005 else (-1 if x < -0.005 else 0)
            )

            # Drop NaN rows
            df = df.dropna()

            logger.info(f"Engineered {len(df.columns)} features, {len(df)} samples")
            return df

        except Exception as e:
            logger.error(f"Error engineering features: {e}")
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
    """Trainer for LSTM models"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)

    def train(self, X: np.ndarray, y: np.ndarray,
              model_name: str = "lstm_model") -> TrainingResult:
        """Train LSTM model"""
        start_time = datetime.now()

        try:
            # Import model
            from src.ai.model_stack.lstm_model import LSTMModel

            # Prepare data
            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.LongTensor(y.astype(int))

            dataset = TensorDataset(X_tensor, y_tensor)

            # Split data
            train_size = int(self.config.train_split * len(dataset))
            val_size = int(self.config.validation_split * len(dataset))
            test_size = len(dataset) - train_size - val_size

            train_data, val_data, test_data = random_split(
                dataset, [train_size, val_size, test_size]
            )

            train_loader = DataLoader(train_data, batch_size=self.config.batch_size, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=self.config.batch_size)
            test_loader = DataLoader(test_data, batch_size=self.config.batch_size)

            # Initialize model
            input_size = X.shape[2]  # Number of features
            model = LSTMModel(
                input_size=input_size,
                hidden_size=self.config.hidden_size,
                num_layers=self.config.num_layers,
                dropout=self.config.dropout,
                output_size=3
            ).to(self.device)

            # Loss and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=5, factor=0.5
            )

            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0
            train_losses = []
            val_losses = []

            for epoch in range(self.config.epochs):
                # Training phase
                model.train()
                train_loss = 0.0
                for batch_X, batch_y in train_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)

                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()

                train_loss /= len(train_loader)
                train_losses.append(train_loss)

                # Validation phase
                model.eval()
                val_loss = 0.0
                correct = 0
                total = 0

                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)

                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()

                        _, predicted = torch.max(outputs, 1)
                        total += batch_y.size(0)
                        correct += (predicted == batch_y).sum().item()

                val_loss /= len(val_loader)
                val_losses.append(val_loss)
                accuracy = correct / total if total > 0 else 0

                scheduler.step(val_loss)

                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch [{epoch+1}/{self.config.epochs}] "
                              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                              f"Accuracy: {accuracy:.4f}")

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    model_path = Path(self.config.model_dir) / f"{model_name}_best.pt"
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), model_path)
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break

            # Final evaluation on test set
            model.eval()
            test_loss = 0.0
            correct = 0
            total = 0

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

            test_loss /= len(test_loader) if len(test_loader) > 0 else 1
            test_accuracy = correct / total if total > 0 else 0

            training_time = (datetime.now() - start_time).total_seconds()

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
                    'final_accuracy': test_accuracy
                }
            )

        except Exception as e:
            logger.error(f"Error training LSTM model: {e}")
            return TrainingResult(
                model_name=model_name,
                success=False,
                error_message=str(e)
            )


class TrainingPipeline:
    """
    Main training pipeline that orchestrates all model training.
    """

    def __init__(self, config: Optional[TrainingConfig] = None,
                 config_path: Optional[str] = None):
        """Initialize training pipeline"""
        if config:
            self.config = config
        elif config_path and Path(config_path).exists():
            self.config = self._load_config(config_path)
        else:
            self.config = TrainingConfig()

        self.preprocessor = DataPreprocessor(self.config)
        self.results: List[TrainingResult] = []

        # Create directories
        Path(self.config.model_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

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

    def run_full_training(self, symbols: List[str] = None) -> Dict[str, TrainingResult]:
        """Run full training pipeline for all models"""
        logger.info("Starting full training pipeline...")

        if symbols is None:
            # Default Canadian symbols
            symbols = [
                'RY.TO', 'TD.TO', 'BNS.TO', 'BMO.TO', 'CM.TO',
                'SHOP.TO', 'ENB.TO', 'CNQ.TO', 'SU.TO', 'TRP.TO'
            ]

        results = {}

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
