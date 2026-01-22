"""
Mid-term GRU/Transformer Model for 5-15 minute predictions
Technical Analysis + Macro Data + Options Data
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

class GRUTransformerModel(nn.Module):
    """
    Hybrid GRU-Transformer Model for mid-term price prediction
    
    Architecture:
    - GRU layers for sequential processing
    - Multi-head self-attention (Transformer)
    - Dense layers for prediction
    - Output: Price direction probability (up/down/neutral)
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_gru_layers: int = 2,
        num_attention_heads: int = 8,
        dropout: float = 0.2,
        output_size: int = 3  # up, down, neutral
    ):
        super(GRUTransformerModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_gru_layers = num_gru_layers
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_gru_layers,
            dropout=dropout if num_gru_layers > 1 else 0,
            batch_first=True
        )
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        
        # Output layers
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # GRU forward pass
        gru_out, h_n = self.gru(x)
        
        # Multi-head self-attention
        attn_out, _ = self.attention(gru_out, gru_out, gru_out)
        
        # Residual connection and layer norm
        attn_out = self.layer_norm1(gru_out + attn_out)
        
        # Feed-forward network
        ffn_out = self.ffn(attn_out)
        
        # Residual connection and layer norm
        ffn_out = self.layer_norm2(attn_out + ffn_out)
        
        # Global average pooling
        pooled = torch.mean(ffn_out, dim=1)
        
        # Dense layers
        out = self.relu(self.fc1(pooled))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        
        # Softmax for probability distribution
        out = self.softmax(out)
        
        return out

class GRUTransformerPredictor:
    """
    GRU-Transformer Predictor for mid-term (5-15 minute) price movements
    
    Features:
    - Technical indicators
    - Macro economic data
    - Options data (OI, IV, Greeks)
    - Market regime indicators
    """
    
    def __init__(
        self,
        model_path: str = "models/gru_transformer_model.pth",
        sequence_length: int = 100,
        hidden_size: int = 256,
        num_gru_layers: int = 2,
        num_attention_heads: int = 8,
        dropout: float = 0.2
    ):
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_gru_layers = num_gru_layers
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        
        self.scaler = StandardScaler()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.feature_names = [
            # Price features
            'close', 'open', 'high', 'low', 'volume',
            'returns', 'log_returns', 'volatility',
            'high_low_ratio', 'close_open_ratio',
            
            # Technical indicators
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
            'sma_20', 'sma_50', 'sma_200',
            'ema_12', 'ema_26', 'ema_50',
            'atr', 'adx', 'cci', 'stochastic',
            
            # Volume features
            'volume_ma', 'volume_ratio', 'vwap', 'vwap_deviation',
            'obv', 'mfi',
            
            # Options data
            'call_oi', 'put_oi', 'put_call_ratio',
            'iv_call', 'iv_put', 'iv_rank',
            'gamma_exposure', 'delta_exposure',
            'max_pain', 'pcr_oi', 'pcr_volume',
            
            # Macro indicators
            'interest_rate', 'vix', 'usdcad',
            'oil_price', 'gold_price',
            'market_regime', 'sector_strength',
            
            # Market breadth
            'advance_decline', 'new_highs_lows',
            'market_momentum', 'sector_rotation'
        ]
        
        self.input_size = len(self.feature_names)
        
        logger.info(f" GRU-Transformer Predictor initialized with {self.input_size} features")
    
    def prepare_features(self, df: pd.DataFrame, macro_data: Optional[Dict] = None, options_data: Optional[Dict] = None) -> np.ndarray:
        """Prepare features from raw data, macro data, and options data"""
        
        # Make a copy
        data = df.copy()
        
        # Calculate basic features
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        data['volatility'] = data['returns'].rolling(window=20).std()
        data['high_low_ratio'] = data['high'] / data['low']
        data['close_open_ratio'] = data['close'] / data['open']
        
        # Technical indicators
        data['rsi'] = self._calculate_rsi(data['close'])
        data['macd'], data['macd_signal'], data['macd_hist'] = self._calculate_macd(data['close'])
        data['bb_upper'], data['bb_middle'], data['bb_lower'], data['bb_width'] = self._calculate_bollinger_bands(data['close'])
        
        # Moving averages
        for period in [20, 50, 200]:
            data[f'sma_{period}'] = data['close'].rolling(window=period).mean()
        
        for period in [12, 26, 50]:
            data[f'ema_{period}'] = data['close'].ewm(span=period, adjust=False).mean()
        
        # ATR and ADX
        data['atr'] = self._calculate_atr(data)
        data['adx'] = 25.0  # Placeholder
        data['cci'] = self._calculate_cci(data)
        data['stochastic'] = self._calculate_stochastic(data)
        
        # Volume features
        data['volume_ma'] = data['volume'].rolling(window=20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_ma']
        data['vwap'] = (data['close'] * data['volume']).rolling(window=20).sum() / data['volume'].rolling(window=20).sum()
        data['vwap_deviation'] = (data['close'] - data['vwap']) / data['vwap']
        data['obv'] = self._calculate_obv(data)
        data['mfi'] = self._calculate_mfi(data)
        
        # Options data (if available)
        if options_data:
            data['call_oi'] = options_data.get('call_oi', 0.0)
            data['put_oi'] = options_data.get('put_oi', 0.0)
            data['put_call_ratio'] = options_data.get('put_call_ratio', 1.0)
            data['iv_call'] = options_data.get('iv_call', 0.25)
            data['iv_put'] = options_data.get('iv_put', 0.25)
            data['iv_rank'] = options_data.get('iv_rank', 50.0)
            data['gamma_exposure'] = options_data.get('gamma_exposure', 0.0)
            data['delta_exposure'] = options_data.get('delta_exposure', 0.0)
            data['max_pain'] = options_data.get('max_pain', data['close'].iloc[-1])
            data['pcr_oi'] = options_data.get('pcr_oi', 1.0)
            data['pcr_volume'] = options_data.get('pcr_volume', 1.0)
        else:
            # Default values
            data['call_oi'] = 0.0
            data['put_oi'] = 0.0
            data['put_call_ratio'] = 1.0
            data['iv_call'] = 0.25
            data['iv_put'] = 0.25
            data['iv_rank'] = 50.0
            data['gamma_exposure'] = 0.0
            data['delta_exposure'] = 0.0
            data['max_pain'] = data['close']
            data['pcr_oi'] = 1.0
            data['pcr_volume'] = 1.0
        
        # Macro indicators (if available)
        if macro_data:
            data['interest_rate'] = macro_data.get('interest_rate', 5.0)
            data['vix'] = macro_data.get('vix', 15.0)
            data['usdcad'] = macro_data.get('usdcad', 1.35)
            data['oil_price'] = macro_data.get('oil_price', 75.0)
            data['gold_price'] = macro_data.get('gold_price', 2000.0)
            data['market_regime'] = macro_data.get('market_regime', 0.0)  # -1: bearish, 0: neutral, 1: bullish
            data['sector_strength'] = macro_data.get('sector_strength', 0.0)
        else:
            # Default values
            data['interest_rate'] = 5.0
            data['vix'] = 15.0
            data['usdcad'] = 1.35
            data['oil_price'] = 75.0
            data['gold_price'] = 2000.0
            data['market_regime'] = 0.0
            data['sector_strength'] = 0.0
        
        # Market breadth
        data['advance_decline'] = 0.0  # Placeholder
        data['new_highs_lows'] = 0.0  # Placeholder
        data['market_momentum'] = data['returns'].rolling(window=20).mean()
        data['sector_rotation'] = 0.0  # Placeholder
        
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
    
    def _calculate_cci(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index"""
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: abs(x - x.mean()).mean())
        cci = (tp - sma) / (0.015 * mad)
        return cci
    
    def _calculate_stochastic(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Stochastic Oscillator"""
        low_min = df['low'].rolling(window=period).min()
        high_max = df['high'].rolling(window=period).max()
        stochastic = 100 * (df['close'] - low_min) / (high_max - low_min)
        return stochastic
    
    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume"""
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        return obv
    
    def _calculate_mfi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Money Flow Index"""
        tp = (df['high'] + df['low'] + df['close']) / 3
        mf = tp * df['volume']
        mf_pos = mf.where(tp > tp.shift(), 0).rolling(window=period).sum()
        mf_neg = mf.where(tp < tp.shift(), 0).rolling(window=period).sum()
        mfi = 100 - (100 / (1 + mf_pos / mf_neg))
        return mfi
    
    def prepare_labels(self, df: pd.DataFrame, threshold: float = 0.002) -> np.ndarray:
        """
        Prepare labels for training
        0: Down (< -threshold)
        1: Neutral (-threshold to +threshold)
        2: Up (> +threshold)
        """
        future_returns = df['close'].shift(-5) / df['close'] - 1  # 5-period ahead
        
        labels = np.zeros(len(future_returns))
        labels[future_returns < -threshold] = 0  # Down
        labels[(future_returns >= -threshold) & (future_returns <= threshold)] = 1  # Neutral
        labels[future_returns > threshold] = 2  # Up
        
        return labels
    
    def train(
        self,
        df: pd.DataFrame,
        macro_data: Optional[Dict] = None,
        options_data: Optional[Dict] = None,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        validation_split: float = 0.2
    ):
        """Train the GRU-Transformer model"""
        
        logger.info(" Training GRU-Transformer model...")
        
        # Prepare features and labels
        features = self.prepare_features(df, macro_data, options_data)
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
        from src.ai.model_stack.lstm_model import MarketDataset
        train_dataset = MarketDataset(train_features, train_labels, self.sequence_length)
        val_dataset = MarketDataset(val_features, val_labels, self.sequence_length)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        self.model = GRUTransformerModel(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_gru_layers=self.num_gru_layers,
            num_attention_heads=self.num_attention_heads,
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
        
        logger.info(" GRU-Transformer training completed")
    
    def predict(self, df: pd.DataFrame, macro_data: Optional[Dict] = None, options_data: Optional[Dict] = None) -> Dict[str, float]:
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
            logger.warning("No model loaded, returning neutral prediction")
            return {
                'direction': 'neutral',
                'confidence': 0.33,
                'probabilities': [0.33, 0.34, 0.33]
            }
        
        # Prepare features
        features = self.prepare_features(df, macro_data, options_data)
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
            
            self.model = GRUTransformerModel(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_gru_layers=self.num_gru_layers,
                num_attention_heads=self.num_attention_heads,
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

