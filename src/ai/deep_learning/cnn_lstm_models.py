"""
CNN-LSTM hybrid models for financial market prediction.

This module implements convolutional neural networks combined with LSTM
for feature extraction and sequence modeling in financial time series.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization,
    Conv1D, MaxPooling1D, Flatten, Reshape,
    Input, Concatenate, GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import joblib
import os

logger = logging.getLogger(__name__)

class CNNLSTMPredictor:
    """
    CNN-LSTM hybrid model for financial market prediction.
    
    This model uses 1D convolutions to extract local patterns
    and LSTM layers to model temporal dependencies.
    """
    
    def __init__(
        self,
        sequence_length: int = 60,
        features: int = 10,
        cnn_filters: List[int] = [64, 32],
        cnn_kernel_size: int = 3,
        lstm_units: List[int] = [50, 50],
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        model_name: str = "cnn_lstm_predictor"
    ):
        """
        Initialize CNN-LSTM predictor.
        
        Args:
            sequence_length: Number of time steps to look back
            features: Number of input features
            cnn_filters: Number of filters for each CNN layer
            cnn_kernel_size: Kernel size for CNN layers
            lstm_units: List of LSTM units for each layer
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            model_name: Name for model saving/loading
        """
        self.sequence_length = sequence_length
        self.features = features
        self.cnn_filters = cnn_filters
        self.cnn_kernel_size = cnn_kernel_size
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model_name = model_name
        
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.is_trained = False
        
        logger.info(f"Initialized CNN-LSTM Predictor: {model_name}")
    
    def build_model(self) -> Model:
        """Build the CNN-LSTM hybrid model."""
        # Input layer
        input_layer = Input(shape=(self.sequence_length, self.features))
        
        # CNN layers for feature extraction
        cnn_output = input_layer
        
        for i, filters in enumerate(self.cnn_filters):
            cnn_output = Conv1D(
                filters=filters,
                kernel_size=self.cnn_kernel_size,
                activation='relu',
                padding='same'
            )(cnn_output)
            cnn_output = BatchNormalization()(cnn_output)
            cnn_output = Dropout(self.dropout_rate)(cnn_output)
            
            # Add pooling every other layer
            if i % 2 == 1:
                cnn_output = MaxPooling1D(pool_size=2)(cnn_output)
        
        # LSTM layers for sequence modeling
        lstm_output = cnn_output
        
        for i, units in enumerate(self.lstm_units):
            return_sequences = i < len(self.lstm_units) - 1
            lstm_output = LSTM(
                units=units,
                return_sequences=return_sequences,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate
            )(lstm_output)
            lstm_output = BatchNormalization()(lstm_output)
        
        # Dense layers for final prediction
        dense_output = Dense(32, activation='relu')(lstm_output)
        dense_output = Dropout(self.dropout_rate)(dense_output)
        dense_output = Dense(16, activation='relu')(dense_output)
        dense_output = Dropout(self.dropout_rate)(dense_output)
        
        # Output layer
        output = Dense(1, activation='linear')(dense_output)
        
        # Create model
        model = Model(inputs=input_layer, outputs=output)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        self.model = model
        logger.info(f"Built CNN-LSTM model with {model.count_params()} parameters")
        
        return model
    
    def prepare_data(
        self,
        data: pd.DataFrame,
        target_column: str = 'close',
        feature_columns: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for CNN-LSTM training.
        
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
        
        logger.info(f"Prepared CNN-LSTM data: X shape {X.shape}, y shape {y.shape}")
        
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
        Train the CNN-LSTM model.
        
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
        logger.info("CNN-LSTM model training completed")
        
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
        confidence = 0.85  # Placeholder - could be improved with ensemble
        
        return {
            'prediction': float(prediction),
            'confidence': confidence,
            'steps_ahead': steps_ahead,
            'model_type': 'CNN-LSTM',
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
            'cnn_filters': self.cnn_filters,
            'cnn_kernel_size': self.cnn_kernel_size,
            'lstm_units': self.lstm_units,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        metadata_path = os.path.join(model_dir, f"{self.model_name}_metadata.pkl")
        joblib.dump(metadata, metadata_path)
        
        logger.info(f"CNN-LSTM model saved to {model_dir}")
    
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
        self.cnn_filters = metadata['cnn_filters']
        self.cnn_kernel_size = metadata['cnn_kernel_size']
        self.lstm_units = metadata['lstm_units']
        self.dropout_rate = metadata['dropout_rate']
        self.learning_rate = metadata['learning_rate']
        self.feature_names = metadata['feature_names']
        self.is_trained = metadata['is_trained']
        
        logger.info(f"CNN-LSTM model loaded from {model_dir}")


class MultiScaleCNNLSTM:
    """
    Multi-scale CNN-LSTM model that processes different timeframes.
    
    This model uses multiple CNN-LSTM branches to capture patterns
    at different time scales (short-term, medium-term, long-term).
    """
    
    def __init__(
        self,
        sequence_lengths: List[int] = [20, 40, 60],
        features: int = 10,
        cnn_filters: List[int] = [32, 16],
        lstm_units: List[int] = [32, 16],
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
        model_name: str = "multi_scale_cnn_lstm"
    ):
        """
        Initialize multi-scale CNN-LSTM model.
        
        Args:
            sequence_lengths: List of sequence lengths for different scales
            features: Number of input features
            cnn_filters: Number of filters for CNN layers
            lstm_units: List of LSTM units for each layer
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            model_name: Name for model saving/loading
        """
        self.sequence_lengths = sequence_lengths
        self.features = features
        self.cnn_filters = cnn_filters
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model_name = model_name
        
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.is_trained = False
        
        logger.info(f"Initialized Multi-Scale CNN-LSTM: {model_name}")
    
    def build_model(self) -> Model:
        """Build the multi-scale CNN-LSTM model."""
        # Input layers for different scales
        inputs = []
        branches = []
        
        for seq_len in self.sequence_lengths:
            # Input layer for this scale
            input_layer = Input(shape=(seq_len, self.features), name=f'input_{seq_len}')
            inputs.append(input_layer)
            
            # CNN layers
            cnn_output = input_layer
            for filters in self.cnn_filters:
                cnn_output = Conv1D(
                    filters=filters,
                    kernel_size=3,
                    activation='relu',
                    padding='same'
                )(cnn_output)
                cnn_output = BatchNormalization()(cnn_output)
                cnn_output = Dropout(self.dropout_rate)(cnn_output)
            
            # LSTM layers
            lstm_output = cnn_output
            for i, units in enumerate(self.lstm_units):
                return_sequences = i < len(self.lstm_units) - 1
                lstm_output = LSTM(
                    units=units,
                    return_sequences=return_sequences,
                    dropout=self.dropout_rate
                )(lstm_output)
                lstm_output = BatchNormalization()(lstm_output)
            
            branches.append(lstm_output)
        
        # Concatenate all branches
        if len(branches) > 1:
            merged = Concatenate()(branches)
        else:
            merged = branches[0]
        
        # Final dense layers
        dense_output = Dense(64, activation='relu')(merged)
        dense_output = Dropout(self.dropout_rate)(dense_output)
        dense_output = Dense(32, activation='relu')(dense_output)
        dense_output = Dropout(self.dropout_rate)(dense_output)
        
        # Output layer
        output = Dense(1, activation='linear')(dense_output)
        
        # Create model
        model = Model(inputs=inputs, outputs=output)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        self.model = model
        logger.info(f"Built Multi-Scale CNN-LSTM with {model.count_params()} parameters")
        
        return model
    
    def prepare_data(
        self,
        data: pd.DataFrame,
        target_column: str = 'close',
        feature_columns: Optional[List[str]] = None
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Prepare data for multi-scale CNN-LSTM training.
        
        Args:
            data: Input DataFrame with OHLCV data
            target_column: Column to predict
            feature_columns: Features to use
            
        Returns:
            Tuple of (X_list, y) where X_list contains arrays for each scale
        """
        if feature_columns is None:
            feature_columns = [
                'open', 'high', 'low', 'close', 'volume',
                'sma_20', 'ema_12', 'rsi', 'macd', 'bb_upper'
            ]
            feature_columns = [col for col in feature_columns if col in data.columns]
        
        self.feature_names = feature_columns
        
        # Prepare features
        features_data = data[feature_columns].values
        target_data = data[target_column].values
        
        # Normalize features
        from sklearn.preprocessing import MinMaxScaler
        self.scaler = MinMaxScaler()
        features_scaled = self.scaler.fit_transform(features_data)
        
        # Create sequences for each scale
        X_list = []
        max_seq_len = max(self.sequence_lengths)
        
        for seq_len in self.sequence_lengths:
            X_scale = []
            for i in range(max_seq_len, len(features_scaled)):
                X_scale.append(features_scaled[i-seq_len:i])
            X_list.append(np.array(X_scale))
        
        # Target data (aligned with max sequence length)
        y = target_data[max_seq_len:]
        
        logger.info(f"Prepared multi-scale data: {[X.shape for X in X_list]}, y shape {y.shape}")
        
        return X_list, y
    
    def train(
        self,
        X_list: List[np.ndarray],
        y: np.ndarray,
        validation_split: float = 0.2,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: int = 1
    ) -> Dict[str, Any]:
        """Train the multi-scale CNN-LSTM model."""
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
            X_list, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.is_trained = True
        logger.info("Multi-scale CNN-LSTM model training completed")
        
        return history.history
    
    def predict_next_price(
        self,
        recent_data: pd.DataFrame,
        steps_ahead: int = 1
    ) -> Dict[str, Any]:
        """Predict next price using multi-scale analysis."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare recent data
        features_data = recent_data[self.feature_names].values
        features_scaled = self.scaler.transform(features_data)
        
        # Create input sequences for each scale
        input_sequences = []
        for seq_len in self.sequence_lengths:
            sequence = features_scaled[-seq_len:].reshape(1, seq_len, self.features)
            input_sequences.append(sequence)
        
        # Make prediction
        prediction = self.model.predict(input_sequences, verbose=0)[0][0]
        
        return {
            'prediction': float(prediction),
            'confidence': 0.88,  # Multi-scale typically has higher confidence
            'steps_ahead': steps_ahead,
            'model_type': 'Multi-Scale CNN-LSTM',
            'scales_used': self.sequence_lengths,
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
            'sequence_lengths': self.sequence_lengths,
            'features': self.features,
            'cnn_filters': self.cnn_filters,
            'lstm_units': self.lstm_units,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        metadata_path = os.path.join(model_dir, f"{self.model_name}_metadata.pkl")
        joblib.dump(metadata, metadata_path)
        
        logger.info(f"Multi-scale CNN-LSTM model saved to {model_dir}")
    
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
        
        self.sequence_lengths = metadata['sequence_lengths']
        self.features = metadata['features']
        self.cnn_filters = metadata['cnn_filters']
        self.lstm_units = metadata['lstm_units']
        self.dropout_rate = metadata['dropout_rate']
        self.learning_rate = metadata['learning_rate']
        self.feature_names = metadata['feature_names']
        self.is_trained = metadata['is_trained']
        
        logger.info(f"Multi-scale CNN-LSTM model loaded from {model_dir}")

