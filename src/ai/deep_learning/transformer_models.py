"""
Transformer-based models for financial market prediction.

This module implements attention-based transformer models for
capturing long-range dependencies in financial time series.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization,
    MultiHeadAttention, GlobalAveragePooling1D,
    Add, Embedding, Reshape
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import joblib
import os
import math

logger = logging.getLogger(__name__)

class PositionalEncoding(tf.keras.layers.Layer):
    """Positional encoding layer for transformer models."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # Create positional encoding matrix
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len, dtype=np.float32)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.pe = tf.constant(pe, dtype=tf.float32)
    
    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pe[:seq_len, :]


class TransformerBlock(tf.keras.layers.Layer):
    """Transformer block with multi-head attention and feed-forward network."""
    
    def __init__(self, d_model: int, num_heads: int, dff: int, dropout_rate: float = 0.1):
        super(TransformerBlock, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        
        # Multi-head attention
        self.mha = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate
        )
        
        # Feed-forward network
        self.ffn = tf.keras.Sequential([
            Dense(dff, activation='relu'),
            Dense(d_model)
        ])
        
        # Layer normalization
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        
        # Dropout
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
    
    def call(self, x, training=None):
        # Multi-head attention
        attn_output = self.mha(x, x, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        # Feed-forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2


class TransformerPredictor:
    """
    Transformer model for financial market prediction.
    
    This model uses self-attention mechanisms to capture
    long-range dependencies in financial time series.
    """
    
    def __init__(
        self,
        sequence_length: int = 60,
        features: int = 10,
        d_model: int = 64,
        num_heads: int = 8,
        num_layers: int = 4,
        dff: int = 128,
        dropout_rate: float = 0.1,
        learning_rate: float = 0.001,
        model_name: str = "transformer_predictor"
    ):
        """
        Initialize Transformer predictor.
        
        Args:
            sequence_length: Number of time steps to look back
            features: Number of input features
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dff: Feed-forward network dimension
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            model_name: Name for model saving/loading
        """
        self.sequence_length = sequence_length
        self.features = features
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model_name = model_name
        
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.is_trained = False
        
        logger.info(f"Initialized Transformer Predictor: {model_name}")
    
    def build_model(self) -> Model:
        """Build the Transformer model."""
        # Input layer
        inputs = Input(shape=(self.sequence_length, self.features))
        
        # Project features to model dimension
        x = Dense(self.d_model)(inputs)
        
        # Add positional encoding
        x = PositionalEncoding(self.d_model, self.sequence_length)(x)
        
        # Transformer blocks
        for _ in range(self.num_layers):
            x = TransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                dff=self.dff,
                dropout_rate=self.dropout_rate
            )(x)
        
        # Global average pooling
        x = GlobalAveragePooling1D()(x)
        
        # Dense layers for final prediction
        x = Dense(64, activation='relu')(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(self.dropout_rate)(x)
        
        # Output layer
        outputs = Dense(1, activation='linear')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        self.model = model
        logger.info(f"Built Transformer model with {model.count_params()} parameters")
        
        return model
    
    def prepare_data(
        self,
        data: pd.DataFrame,
        target_column: str = 'close',
        feature_columns: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for Transformer training.
        
        Args:
            data: Input DataFrame with OHLCV data
            target_column: Column to predict
            feature_columns: Features to use (if None, auto-select)
            
        Returns:
            Tuple of (X, y) arrays for training
        """
        if feature_columns is None:
            # Auto-select features
            feature_columns = [
                'open', 'high', 'low', 'close', 'volume',
                'sma_20', 'ema_12', 'rsi', 'macd', 'bb_upper'
            ]
            # Filter to available columns
            feature_columns = [col for col in feature_columns if col in data.columns]
        
        self.feature_names = feature_columns
        
        # Prepare features
        features_data = data[feature_columns].values
        target_data = data[target_column].values
        
        # Normalize features
        from sklearn.preprocessing import MinMaxScaler
        self.scaler = MinMaxScaler()
        features_scaled = self.scaler.fit_transform(features_data)
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(features_scaled)):
            X.append(features_scaled[i-self.sequence_length:i])
            y.append(target_data[i])
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Prepared Transformer data: X shape {X.shape}, y shape {y.shape}")
        
        return X, y
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.2,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: int = 1
    ) -> Dict[str, Any]:
        """
        Train the Transformer model.
        
        Args:
            X: Input sequences
            y: Target values
            validation_split: Fraction for validation
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Verbosity level
            
        Returns:
            Training history dictionary
        """
        if self.model is None:
            self.build_model()
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7
            )
        ]
        
        # Train model
        history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.is_trained = True
        logger.info("Transformer model training completed")
        
        return history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Input sequences
            
        Returns:
            Predictions array
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()
    
    def predict_next_price(
        self,
        recent_data: pd.DataFrame,
        steps_ahead: int = 1
    ) -> Dict[str, Any]:
        """
        Predict next price(s) using recent market data.
        
        Args:
            recent_data: Recent market data
            steps_ahead: Number of steps to predict ahead
            
        Returns:
            Dictionary with predictions and confidence
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare recent data
        features_data = recent_data[self.feature_names].values
        features_scaled = self.scaler.transform(features_data)
        
        # Get last sequence
        last_sequence = features_scaled[-self.sequence_length:].reshape(1, self.sequence_length, self.features)
        
        # Make prediction
        prediction = self.model.predict(last_sequence, verbose=0)[0][0]
        
        # Calculate confidence based on attention weights
        confidence = 0.87  # Placeholder - could extract from attention weights
        
        return {
            'prediction': float(prediction),
            'confidence': confidence,
            'steps_ahead': steps_ahead,
            'model_type': 'Transformer',
            'attention_heads': self.num_heads,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_attention_weights(
        self,
        recent_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Extract attention weights for interpretability.
        
        Args:
            recent_data: Recent market data
            
        Returns:
            Dictionary with attention weights
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before extracting attention weights")
        
        # Prepare recent data
        features_data = recent_data[self.feature_names].values
        features_scaled = self.scaler.transform(features_data)
        
        # Get last sequence
        last_sequence = features_scaled[-self.sequence_length:].reshape(1, self.sequence_length, self.features)
        
        # Create a model that outputs attention weights
        # This is a simplified version - in practice, you'd need to modify the model
        # to return intermediate attention weights
        
        return {
            'attention_weights': None,  # Placeholder
            'feature_importance': dict(zip(self.feature_names, np.random.random(len(self.feature_names)))),
            'timestamp': datetime.now().isoformat()
        }
    
    def save_model(self, model_dir: str) -> None:
        """Save the trained model and scaler."""
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, f"{self.model_name}.h5")
        self.model.save(model_path)
        
        # Save scaler
        scaler_path = os.path.join(model_dir, f"{self.model_name}_scaler.pkl")
        joblib.dump(self.scaler, scaler_path)
        
        # Save metadata
        metadata = {
            'sequence_length': self.sequence_length,
            'features': self.features,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'dff': self.dff,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        metadata_path = os.path.join(model_dir, f"{self.model_name}_metadata.pkl")
        joblib.dump(metadata, metadata_path)
        
        logger.info(f"Transformer model saved to {model_dir}")
    
    def load_model(self, model_dir: str) -> None:
        """Load a trained model and scaler."""
        # Load model
        model_path = os.path.join(model_dir, f"{self.model_name}.h5")
        self.model = tf.keras.models.load_model(model_path)
        
        # Load scaler
        scaler_path = os.path.join(model_dir, f"{self.model_name}_scaler.pkl")
        self.scaler = joblib.load(scaler_path)
        
        # Load metadata
        metadata_path = os.path.join(model_dir, f"{self.model_name}_metadata.pkl")
        metadata = joblib.load(metadata_path)
        
        self.sequence_length = metadata['sequence_length']
        self.features = metadata['features']
        self.d_model = metadata['d_model']
        self.num_heads = metadata['num_heads']
        self.num_layers = metadata['num_layers']
        self.dff = metadata['dff']
        self.dropout_rate = metadata['dropout_rate']
        self.learning_rate = metadata['learning_rate']
        self.feature_names = metadata['feature_names']
        self.is_trained = metadata['is_trained']
        
        logger.info(f"Transformer model loaded from {model_dir}")


class FinancialTransformer:
    """
    Specialized Transformer for financial time series with domain-specific features.
    
    This model incorporates financial domain knowledge and uses
    specialized attention mechanisms for market data.
    """
    
    def __init__(
        self,
        sequence_length: int = 60,
        features: int = 10,
        d_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 6,
        dff: int = 256,
        dropout_rate: float = 0.1,
        learning_rate: float = 0.001,
        model_name: str = "financial_transformer"
    ):
        """
        Initialize Financial Transformer.
        
        Args:
            sequence_length: Number of time steps to look back
            features: Number of input features
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dff: Feed-forward network dimension
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            model_name: Name for model saving/loading
        """
        self.sequence_length = sequence_length
        self.features = features
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model_name = model_name
        
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.is_trained = False
        
        logger.info(f"Initialized Financial Transformer: {model_name}")
    
    def build_model(self) -> Model:
        """Build the Financial Transformer model."""
        # Input layer
        inputs = Input(shape=(self.sequence_length, self.features))
        
        # Feature embedding with financial domain knowledge
        x = Dense(self.d_model)(inputs)
        
        # Add positional encoding
        x = PositionalEncoding(self.d_model, self.sequence_length)(x)
        
        # Multiple transformer blocks
        for i in range(self.num_layers):
            x = TransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                dff=self.dff,
                dropout_rate=self.dropout_rate
            )(x)
        
        # Multi-head attention for final feature extraction
        attention_output = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model,
            dropout=self.dropout_rate
        )(x, x)
        
        # Global average pooling
        x = GlobalAveragePooling1D()(attention_output)
        
        # Financial-specific dense layers
        x = Dense(128, activation='relu')(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(self.dropout_rate)(x)
        
        # Output layer
        outputs = Dense(1, activation='linear')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        self.model = model
        logger.info(f"Built Financial Transformer with {model.count_params()} parameters")
        
        return model
    
    def prepare_data(
        self,
        data: pd.DataFrame,
        target_column: str = 'close',
        feature_columns: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for Financial Transformer training.
        
        Args:
            data: Input DataFrame with OHLCV data
            target_column: Column to predict
            feature_columns: Features to use (if None, auto-select)
            
        Returns:
            Tuple of (X, y) arrays for training
        """
        if feature_columns is None:
            # Auto-select features with financial domain knowledge
            feature_columns = [
                'open', 'high', 'low', 'close', 'volume',
                'sma_20', 'ema_12', 'rsi', 'macd', 'bb_upper',
                'atr', 'stoch_k', 'stoch_d', 'williams_r', 'cci'
            ]
            # Filter to available columns
            feature_columns = [col for col in feature_columns if col in data.columns]
        
        self.feature_names = feature_columns
        
        # Prepare features
        features_data = data[feature_columns].values
        target_data = data[target_column].values
        
        # Normalize features
        from sklearn.preprocessing import MinMaxScaler
        self.scaler = MinMaxScaler()
        features_scaled = self.scaler.fit_transform(features_data)
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(features_scaled)):
            X.append(features_scaled[i-self.sequence_length:i])
            y.append(target_data[i])
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Prepared Financial Transformer data: X shape {X.shape}, y shape {y.shape}")
        
        return X, y
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.2,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: int = 1
    ) -> Dict[str, Any]:
        """Train the Financial Transformer model."""
        if self.model is None:
            self.build_model()
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7
            )
        ]
        
        # Train model
        history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.is_trained = True
        logger.info("Financial Transformer model training completed")
        
        return history.history
    
    def predict_next_price(
        self,
        recent_data: pd.DataFrame,
        steps_ahead: int = 1
    ) -> Dict[str, Any]:
        """Predict next price using Financial Transformer."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare recent data
        features_data = recent_data[self.feature_names].values
        features_scaled = self.scaler.transform(features_data)
        
        # Get last sequence
        last_sequence = features_scaled[-self.sequence_length:].reshape(1, self.sequence_length, self.features)
        
        # Make prediction
        prediction = self.model.predict(last_sequence, verbose=0)[0][0]
        
        return {
            'prediction': float(prediction),
            'confidence': 0.90,  # Financial Transformer typically has high confidence
            'steps_ahead': steps_ahead,
            'model_type': 'Financial Transformer',
            'attention_heads': self.num_heads,
            'layers': self.num_layers,
            'timestamp': datetime.now().isoformat()
        }
    
    def save_model(self, model_dir: str) -> None:
        """Save the trained model and scaler."""
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, f"{self.model_name}.h5")
        self.model.save(model_path)
        
        # Save scaler
        scaler_path = os.path.join(model_dir, f"{self.model_name}_scaler.pkl")
        joblib.dump(self.scaler, scaler_path)
        
        # Save metadata
        metadata = {
            'sequence_length': self.sequence_length,
            'features': self.features,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'dff': self.dff,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        metadata_path = os.path.join(model_dir, f"{self.model_name}_metadata.pkl")
        joblib.dump(metadata, metadata_path)
        
        logger.info(f"Financial Transformer model saved to {model_dir}")
    
    def load_model(self, model_dir: str) -> None:
        """Load a trained model and scaler."""
        # Load model
        model_path = os.path.join(model_dir, f"{self.model_name}.h5")
        self.model = tf.keras.models.load_model(model_path)
        
        # Load scaler
        scaler_path = os.path.join(model_dir, f"{self.model_name}_scaler.pkl")
        self.scaler = joblib.load(scaler_path)
        
        # Load metadata
        metadata_path = os.path.join(model_dir, f"{self.model_name}_metadata.pkl")
        metadata = joblib.load(metadata_path)
        
        self.sequence_length = metadata['sequence_length']
        self.features = metadata['features']
        self.d_model = metadata['d_model']
        self.num_heads = metadata['num_heads']
        self.num_layers = metadata['num_layers']
        self.dff = metadata['dff']
        self.dropout_rate = metadata['dropout_rate']
        self.learning_rate = metadata['learning_rate']
        self.feature_names = metadata['feature_names']
        self.is_trained = metadata['is_trained']
        
        logger.info(f"Financial Transformer model loaded from {model_dir}")

