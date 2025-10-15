"""
LSTM-based models for financial market prediction.

This module implements various LSTM architectures for price prediction,
trend analysis, and volatility forecasting.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import joblib
import os

logger = logging.getLogger(__name__)

class LSTMPricePredictor:
    """
    LSTM model for price prediction with multiple timeframes.
    """
    
    def __init__(
        self,
        sequence_length: int = 60,
        features: int = 10,
        lstm_units: List[int] = [50, 50],
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        model_name: str = "lstm_price_predictor"
    ):
        """
        Initialize LSTM price predictor.
        
        Args:
            sequence_length: Number of time steps to look back
            features: Number of input features
            lstm_units: List of LSTM units for each layer
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            model_name: Name for model saving/loading
        """
        self.sequence_length = sequence_length
        self.features = features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model_name = model_name
        
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.is_trained = False
        
        logger.info(f"Initialized LSTM Price Predictor: {model_name}")
    
    def build_model(self) -> Sequential:
        """Build the LSTM model architecture."""
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(
            units=self.lstm_units[0],
            return_sequences=True,
            input_shape=(self.sequence_length, self.features)
        ))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))
        
        # Additional LSTM layers
        for units in self.lstm_units[1:]:
            model.add(LSTM(units, return_sequences=True))
            model.add(BatchNormalization())
            model.add(Dropout(self.dropout_rate))
        
        # Final LSTM layer (no return sequences)
        model.add(LSTM(self.lstm_units[-1], return_sequences=False))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))
        
        # Dense layers
        model.add(Dense(25, activation='relu'))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(1, activation='linear'))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        self.model = model
        logger.info(f"Built LSTM model with {model.count_params()} parameters")
        
        return model
    
    def prepare_data(
        self,
        data: pd.DataFrame,
        target_column: str = 'close',
        feature_columns: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM training.
        
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
        
        logger.info(f"Prepared data: X shape {X.shape}, y shape {y.shape}")
        
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
        Train the LSTM model.
        
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
                patience=10,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
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
        logger.info("LSTM model training completed")
        
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
        
        # Calculate confidence based on model uncertainty
        # This is a simplified confidence measure
        confidence = 0.8  # Placeholder - could be improved with ensemble or uncertainty quantification
        
        return {
            'prediction': float(prediction),
            'confidence': confidence,
            'steps_ahead': steps_ahead,
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
            'lstm_units': self.lstm_units,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        metadata_path = os.path.join(model_dir, f"{self.model_name}_metadata.pkl")
        joblib.dump(metadata, metadata_path)
        
        logger.info(f"Model saved to {model_dir}")
    
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
        self.lstm_units = metadata['lstm_units']
        self.dropout_rate = metadata['dropout_rate']
        self.learning_rate = metadata['learning_rate']
        self.feature_names = metadata['feature_names']
        self.is_trained = metadata['is_trained']
        
        logger.info(f"Model loaded from {model_dir}")


class LSTMTrendPredictor:
    """
    LSTM model for trend direction prediction (classification).
    """
    
    def __init__(
        self,
        sequence_length: int = 30,
        features: int = 8,
        lstm_units: List[int] = [32, 32],
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
        model_name: str = "lstm_trend_predictor"
    ):
        """
        Initialize LSTM trend predictor.
        
        Args:
            sequence_length: Number of time steps to look back
            features: Number of input features
            lstm_units: List of LSTM units for each layer
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            model_name: Name for model saving/loading
        """
        self.sequence_length = sequence_length
        self.features = features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model_name = model_name
        
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.is_trained = False
        
        logger.info(f"Initialized LSTM Trend Predictor: {model_name}")
    
    def build_model(self) -> Sequential:
        """Build the LSTM model for trend classification."""
        model = Sequential()
        
        # LSTM layers
        model.add(LSTM(
            units=self.lstm_units[0],
            return_sequences=True,
            input_shape=(self.sequence_length, self.features)
        ))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))
        
        for units in self.lstm_units[1:]:
            model.add(LSTM(units, return_sequences=True))
            model.add(BatchNormalization())
            model.add(Dropout(self.dropout_rate))
        
        # Final LSTM layer
        model.add(LSTM(self.lstm_units[-1], return_sequences=False))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))
        
        # Dense layers for classification
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(3, activation='softmax'))  # Up, Down, Sideways
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        logger.info(f"Built LSTM trend model with {model.count_params()} parameters")
        
        return model
    
    def prepare_data(
        self,
        data: pd.DataFrame,
        feature_columns: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for trend classification.
        
        Args:
            data: Input DataFrame with market data
            feature_columns: Features to use
            
        Returns:
            Tuple of (X, y) arrays for training
        """
        if feature_columns is None:
            feature_columns = [
                'open', 'high', 'low', 'close', 'volume',
                'rsi', 'macd', 'bb_position'
            ]
            feature_columns = [col for col in feature_columns if col in data.columns]
        
        self.feature_names = feature_columns
        
        # Prepare features
        features_data = data[feature_columns].values
        
        # Create trend labels (simplified)
        price_changes = data['close'].pct_change().shift(-1)
        trend_labels = []
        
        for change in price_changes:
            if pd.isna(change):
                trend_labels.append([0, 0, 1])  # Sideways
            elif change > 0.01:  # 1% threshold
                trend_labels.append([1, 0, 0])  # Up
            elif change < -0.01:
                trend_labels.append([0, 1, 0])  # Down
            else:
                trend_labels.append([0, 0, 1])  # Sideways
        
        trend_labels = np.array(trend_labels)
        
        # Normalize features
        from sklearn.preprocessing import MinMaxScaler
        self.scaler = MinMaxScaler()
        features_scaled = self.scaler.fit_transform(features_data)
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(features_scaled)):
            X.append(features_scaled[i-self.sequence_length:i])
            y.append(trend_labels[i])
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Prepared trend data: X shape {X.shape}, y shape {y.shape}")
        
        return X, y
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.2,
        epochs: int = 50,
        batch_size: int = 32,
        verbose: int = 1
    ) -> Dict[str, Any]:
        """Train the LSTM trend model."""
        if self.model is None:
            self.build_model()
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
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
        logger.info("LSTM trend model training completed")
        
        return history.history
    
    def predict_trend(
        self,
        recent_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Predict trend direction using recent market data.
        
        Args:
            recent_data: Recent market data
            
        Returns:
            Dictionary with trend prediction and confidence
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare recent data
        features_data = recent_data[self.feature_names].values
        features_scaled = self.scaler.transform(features_data)
        
        # Get last sequence
        last_sequence = features_scaled[-self.sequence_length:].reshape(1, self.sequence_length, self.features)
        
        # Make prediction
        prediction_probs = self.model.predict(last_sequence, verbose=0)[0]
        
        # Get trend and confidence
        trend_classes = ['UP', 'DOWN', 'SIDEWAYS']
        predicted_trend = trend_classes[np.argmax(prediction_probs)]
        confidence = float(np.max(prediction_probs))
        
        return {
            'trend': predicted_trend,
            'confidence': confidence,
            'probabilities': {
                'up': float(prediction_probs[0]),
                'down': float(prediction_probs[1]),
                'sideways': float(prediction_probs[2])
            },
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
            'lstm_units': self.lstm_units,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        metadata_path = os.path.join(model_dir, f"{self.model_name}_metadata.pkl")
        joblib.dump(metadata, metadata_path)
        
        logger.info(f"Trend model saved to {model_dir}")
    
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
        self.lstm_units = metadata['lstm_units']
        self.dropout_rate = metadata['dropout_rate']
        self.learning_rate = metadata['learning_rate']
        self.feature_names = metadata['feature_names']
        self.is_trained = metadata['is_trained']
        
        logger.info(f"Trend model loaded from {model_dir}")

