"""
Short-term LSTM Model for 1-minute predictions
Technical Analysis + Market Microstructure
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import logging
from typing import Dict, List, Tuple, Optional
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

class MarketDataset(Dataset):
    """Dataset for market data time series"""
    
    def __init__(self, data: np.ndarray, labels: np.ndarray, sequence_length: int = 60):
        self.data = torch.FloatTensor(data)
        self.labels = torch.FloatTensor(labels)
        self.sequence_length = sequence_length
    
    def __len__(self):
        return len(self.data) - self.sequence_length
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.sequence_length]
        y = self.labels[idx + self.sequence_length]
        return x, y

class LSTMModel(nn.Module):
    """
    LSTM Model for short-term price prediction
    
    Architecture:
    - Input: Technical indicators + microstructure features
    - LSTM layers with dropout
    - Attention mechanism
    - Dense layers for prediction
    - Output: Price direction probability (up/down/neutral)
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 3  # up, down, neutral
    ):
        super(LSTMModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention layer
        self.attention = nn.Linear(hidden_size, 1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Attention mechanism
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Dense layers
        out = self.relu(self.fc1(context_vector))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        
        # Softmax for probability distribution
        out = self.softmax(out)
        
        return out

class LSTMPredictor:
    """
    LSTM Predictor for short-term (1-minute) price movements
    
    Features:
    - Technical indicators (RSI, MACD, Bollinger Bands, etc.)
    - Market microstructure (bid-ask spread, order imbalance, volume)
    - Price action (returns, volatility)
    """
    
    def __init__(
        self,
        model_path: str = "models/lstm_model.pth",
        sequence_length: int = 60,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.scaler = StandardScaler()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.feature_names = [
            # Price features
            'close', 'open', 'high', 'low', 'volume',
            'returns', 'log_returns', 'volatility',
            
            # Technical indicators
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
            'sma_5', 'sma_10', 'sma_20', 'sma_50',
            'ema_5', 'ema_10', 'ema_20',
            'atr', 'adx',
            
            # Market microstructure
            'bid_ask_spread', 'order_imbalance', 'trade_imbalance',
            'volume_imbalance', 'price_impact',
            
            # Volume features
            'volume_ma', 'volume_std', 'volume_ratio',
            'vwap', 'vwap_deviation'
        ]
        
        self.input_size = len(self.feature_names)
        
        logger.info(f" LSTM Predictor initialized with {self.input_size} features")
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features from raw data"""
        
        # Make a copy to avoid modifying original
        data = df.copy()
        
        # Calculate basic features
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        data['volatility'] = data['returns'].rolling(window=20).std()
        
        # Technical indicators (simplified - in production, use ta-lib)
        data['rsi'] = self._calculate_rsi(data['close'])
        data['macd'], data['macd_signal'], data['macd_hist'] = self._calculate_macd(data['close'])
        data['bb_upper'], data['bb_middle'], data['bb_lower'], data['bb_width'] = self._calculate_bollinger_bands(data['close'])
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            data[f'sma_{period}'] = data['close'].rolling(window=period).mean()
        
        for period in [5, 10, 20]:
            data[f'ema_{period}'] = data['close'].ewm(span=period, adjust=False).mean()
        
        # ATR
        data['atr'] = self._calculate_atr(data)
        
        # ADX (simplified)
        data['adx'] = 25.0  # Placeholder
        
        # Market microstructure (using available data)
        data['bid_ask_spread'] = (data['high'] - data['low']) / data['close']
        data['order_imbalance'] = 0.0  # Placeholder - requires order book data
        data['trade_imbalance'] = 0.0  # Placeholder
        data['volume_imbalance'] = 0.0  # Placeholder
        data['price_impact'] = abs(data['returns']) / (data['volume'] / data['volume'].mean())
        
        # Volume features
        data['volume_ma'] = data['volume'].rolling(window=20).mean()
        data['volume_std'] = data['volume'].rolling(window=20).std()
        data['volume_ratio'] = data['volume'] / data['volume_ma']
        data['vwap'] = (data['close'] * data['volume']).rolling(window=20).sum() / data['volume'].rolling(window=20).sum()
        data['vwap_deviation'] = (data['close'] - data['vwap']) / data['vwap']
        
        # Select only the features we need
        features = data[self.feature_names].values
        
        # Remove NaN values
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        width = upper - lower
        return upper, middle, lower, width
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr
    
    def prepare_labels(self, df: pd.DataFrame, threshold: float = 0.001) -> np.ndarray:
        """
        Prepare labels for training
        0: Down (< -threshold)
        1: Neutral (-threshold to +threshold)
        2: Up (> +threshold)
        """
        future_returns = df['close'].shift(-1) / df['close'] - 1
        
        labels = np.zeros(len(future_returns))
        labels[future_returns < -threshold] = 0  # Down
        labels[(future_returns >= -threshold) & (future_returns <= threshold)] = 1  # Neutral
        labels[future_returns > threshold] = 2  # Up
        
        return labels
    
    def train(
        self,
        df: pd.DataFrame,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        validation_split: float = 0.2
    ):
        """Train the LSTM model"""
        
        logger.info(" Training LSTM model...")
        
        # Prepare features and labels
        features = self.prepare_features(df)
        labels = self.prepare_labels(df)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Split data
        split_idx = int(len(features_scaled) * (1 - validation_split))
        train_features = features_scaled[:split_idx]
        train_labels = labels[:split_idx]
        val_features = features_scaled[split_idx:]
        val_labels = labels[split_idx:]
        
        # Create datasets
        train_dataset = MarketDataset(train_features, train_labels, self.sequence_length)
        val_dataset = MarketDataset(val_features, val_labels, self.sequence_length)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        self.model = LSTMModel(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.long().to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.long().to(self.device)
                    
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            # Calculate metrics
            train_loss /= len(train_loader)
            train_acc = 100 * train_correct / train_total
            val_loss /= len(val_loader)
            val_acc = 100 * val_correct / val_total
            
            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model()
                logger.info(f" Model saved with validation loss: {val_loss:.4f}")
        
        logger.info(" LSTM training completed")
    
    def predict(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Predict price direction
        
        Returns:
        {
            'direction': 'up'/'down'/'neutral',
            'confidence': 0.0-1.0,
            'probabilities': [down_prob, neutral_prob, up_prob]
        }
        """
        
        if self.model is None:
            self.load_model()
        
        if self.model is None:
            logger.warning(" No model loaded, returning neutral prediction")
            return {
                'direction': 'neutral',
                'confidence': 0.33,
                'probabilities': [0.33, 0.34, 0.33]
            }
        
        # Prepare features
        features = self.prepare_features(df)
        features_scaled = self.scaler.transform(features)
        
        # Get last sequence
        if len(features_scaled) < self.sequence_length:
            logger.warning(" Insufficient data for prediction")
            return {
                'direction': 'neutral',
                'confidence': 0.33,
                'probabilities': [0.33, 0.34, 0.33]
            }
        
        sequence = features_scaled[-self.sequence_length:]
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            output = self.model(sequence_tensor)
            probabilities = output.cpu().numpy()[0]
        
        # Get direction and confidence
        predicted_class = np.argmax(probabilities)
        confidence = float(probabilities[predicted_class])
        
        direction_map = {0: 'down', 1: 'neutral', 2: 'up'}
        direction = direction_map[predicted_class]
        
        return {
            'direction': direction,
            'confidence': confidence,
            'probabilities': probabilities.tolist()
        }
    
    def save_model(self):
        """Save model and scaler"""
        torch.save(self.model.state_dict(), self.model_path)
        joblib.dump(self.scaler, str(self.model_path).replace('.pth', '_scaler.pkl'))
        logger.info(f" Model saved to {self.model_path}")
    
    def load_model(self):
        """Load model and scaler"""
        try:
            if not self.model_path.exists():
                logger.warning(f" Model file not found: {self.model_path}")
                return False
            
            self.model = LSTMModel(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout
            ).to(self.device)
            
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
            
            scaler_path = str(self.model_path).replace('.pth', '_scaler.pkl')
            self.scaler = joblib.load(scaler_path)
            
            logger.info(f" Model loaded from {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f" Failed to load model: {e}")
            return False

