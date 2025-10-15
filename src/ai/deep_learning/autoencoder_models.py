"""
Autoencoder models for anomaly detection and dimensionality reduction.

This module implements various autoencoder architectures for detecting
market anomalies and reducing feature dimensionality.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Dense, LSTM, Dropout, BatchNormalization,
    Input, RepeatVector, TimeDistributed,
    Conv1D, MaxPooling1D, UpSampling1D,
    Reshape, Flatten
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

logger = logging.getLogger(__name__)

class MarketAutoencoder:
    """
    Standard autoencoder for market data anomaly detection.
    
    This model learns to reconstruct normal market patterns and
    identifies anomalies as high reconstruction errors.
    """
    
    def __init__(
        self,
        input_dim: int = 10,
        encoding_dim: int = 5,
        hidden_layers: List[int] = [8, 6],
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        model_name: str = "market_autoencoder"
    ):
        """
        Initialize Market Autoencoder.
        
        Args:
            input_dim: Number of input features
            encoding_dim: Dimension of encoded representation
            hidden_layers: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            model_name: Name for model saving/loading
        """
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model_name = model_name
        
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self.scaler = None
        self.feature_names = None
        self.is_trained = False
        self.reconstruction_threshold = None
        
        logger.info(f"Initialized Market Autoencoder: {model_name}")
    
    def build_model(self) -> Tuple[Model, Model, Model]:
        """Build the autoencoder model."""
        # Input layer
        input_layer = Input(shape=(self.input_dim,))
        
        # Encoder
        encoded = input_layer
        for units in self.hidden_layers:
            encoded = Dense(units, activation='relu')(encoded)
            encoded = BatchNormalization()(encoded)
            encoded = Dropout(self.dropout_rate)(encoded)
        
        # Encoded representation
        encoded = Dense(self.encoding_dim, activation='relu')(encoded)
        
        # Decoder
        decoded = encoded
        for units in reversed(self.hidden_layers):
            decoded = Dense(units, activation='relu')(decoded)
            decoded = BatchNormalization()(decoded)
            decoded = Dropout(self.dropout_rate)(decoded)
        
        # Output layer
        decoded = Dense(self.input_dim, activation='linear')(decoded)
        
        # Create models
        autoencoder = Model(input_layer, decoded)
        encoder = Model(input_layer, encoded)
        
        # Decoder model
        encoded_input = Input(shape=(self.encoding_dim,))
        decoder_layer = encoded_input
        for units in reversed(self.hidden_layers):
            decoder_layer = Dense(units, activation='relu')(decoder_layer)
            decoder_layer = BatchNormalization()(decoder_layer)
            decoder_layer = Dropout(self.dropout_rate)(decoder_layer)
        decoder_layer = Dense(self.input_dim, activation='linear')(decoder_layer)
        decoder = Model(encoded_input, decoder_layer)
        
        # Compile autoencoder
        autoencoder.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        self.autoencoder = autoencoder
        self.encoder = encoder
        self.decoder = decoder
        
        logger.info(f"Built Market Autoencoder with {autoencoder.count_params()} parameters")
        
        return autoencoder, encoder, decoder
    
    def prepare_data(
        self,
        data: pd.DataFrame,
        feature_columns: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Prepare data for autoencoder training.
        
        Args:
            data: Input DataFrame with market data
            feature_columns: Features to use (if None, auto-select)
            
        Returns:
            Normalized feature array
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
        
        # Normalize features
        self.scaler = MinMaxScaler()
        features_scaled = self.scaler.fit_transform(features_data)
        
        logger.info(f"Prepared autoencoder data: shape {features_scaled.shape}")
        
        return features_scaled
    
    def train(
        self,
        X: np.ndarray,
        validation_split: float = 0.2,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: int = 1
    ) -> Dict[str, Any]:
        """
        Train the autoencoder model.
        
        Args:
            X: Input features
            validation_split: Fraction for validation
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Verbosity level
            
        Returns:
            Training history dictionary
        """
        if self.autoencoder is None:
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
        
        # Train autoencoder (input = target for reconstruction)
        history = self.autoencoder.fit(
            X, X,  # Autoencoder learns to reconstruct input
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.is_trained = True
        
        # Calculate reconstruction threshold
        self._calculate_reconstruction_threshold(X)
        
        logger.info("Market Autoencoder training completed")
        
        return history.history
    
    def _calculate_reconstruction_threshold(self, X: np.ndarray, percentile: float = 95.0) -> None:
        """Calculate reconstruction error threshold for anomaly detection."""
        # Get reconstruction errors for training data
        reconstructions = self.autoencoder.predict(X, verbose=0)
        reconstruction_errors = np.mean(np.square(X - reconstructions), axis=1)
        
        # Set threshold at specified percentile
        self.reconstruction_threshold = np.percentile(reconstruction_errors, percentile)
        
        logger.info(f"Reconstruction threshold set to {self.reconstruction_threshold:.6f}")
    
    def detect_anomalies(
        self,
        data: pd.DataFrame,
        threshold_multiplier: float = 1.0
    ) -> Dict[str, Any]:
        """
        Detect anomalies in market data.
        
        Args:
            data: Input DataFrame with market data
            threshold_multiplier: Multiplier for threshold adjustment
            
        Returns:
            Dictionary with anomaly detection results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before detecting anomalies")
        
        # Prepare data
        features_data = data[self.feature_names].values
        features_scaled = self.scaler.transform(features_data)
        
        # Get reconstructions
        reconstructions = self.autoencoder.predict(features_scaled, verbose=0)
        
        # Calculate reconstruction errors
        reconstruction_errors = np.mean(np.square(features_scaled - reconstructions), axis=1)
        
        # Detect anomalies
        threshold = self.reconstruction_threshold * threshold_multiplier
        anomalies = reconstruction_errors > threshold
        
        # Get anomaly details
        anomaly_indices = np.where(anomalies)[0]
        anomaly_scores = reconstruction_errors[anomalies]
        
        return {
            'anomalies_detected': int(np.sum(anomalies)),
            'total_samples': len(data),
            'anomaly_rate': float(np.mean(anomalies)),
            'threshold': float(threshold),
            'anomaly_indices': anomaly_indices.tolist(),
            'anomaly_scores': anomaly_scores.tolist(),
            'reconstruction_errors': reconstruction_errors.tolist(),
            'timestamp': datetime.now().isoformat()
        }
    
    def encode_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Encode features to lower-dimensional representation.
        
        Args:
            data: Input DataFrame with market data
            
        Returns:
            Encoded features array
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before encoding features")
        
        # Prepare data
        features_data = data[self.feature_names].values
        features_scaled = self.scaler.transform(features_data)
        
        # Encode features
        encoded_features = self.encoder.predict(features_scaled, verbose=0)
        
        return encoded_features
    
    def save_model(self, model_dir: str) -> None:
        """Save the trained model and scaler."""
        os.makedirs(model_dir, exist_ok=True)
        
        # Save models
        autoencoder_path = os.path.join(model_dir, f"{self.model_name}_autoencoder.h5")
        self.autoencoder.save(autoencoder_path)
        
        encoder_path = os.path.join(model_dir, f"{self.model_name}_encoder.h5")
        self.encoder.save(encoder_path)
        
        decoder_path = os.path.join(model_dir, f"{self.model_name}_decoder.h5")
        self.decoder.save(decoder_path)
        
        # Save scaler
        scaler_path = os.path.join(model_dir, f"{self.model_name}_scaler.pkl")
        joblib.dump(self.scaler, scaler_path)
        
        # Save metadata
        metadata = {
            'input_dim': self.input_dim,
            'encoding_dim': self.encoding_dim,
            'hidden_layers': self.hidden_layers,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained,
            'reconstruction_threshold': self.reconstruction_threshold
        }
        
        metadata_path = os.path.join(model_dir, f"{self.model_name}_metadata.pkl")
        joblib.dump(metadata, metadata_path)
        
        logger.info(f"Market Autoencoder saved to {model_dir}")
    
    def load_model(self, model_dir: str) -> None:
        """Load a trained model and scaler."""
        # Load models
        autoencoder_path = os.path.join(model_dir, f"{self.model_name}_autoencoder.h5")
        self.autoencoder = tf.keras.models.load_model(autoencoder_path)
        
        encoder_path = os.path.join(model_dir, f"{self.model_name}_encoder.h5")
        self.encoder = tf.keras.models.load_model(encoder_path)
        
        decoder_path = os.path.join(model_dir, f"{self.model_name}_decoder.h5")
        self.decoder = tf.keras.models.load_model(decoder_path)
        
        # Load scaler
        scaler_path = os.path.join(model_dir, f"{self.model_name}_scaler.pkl")
        self.scaler = joblib.load(scaler_path)
        
        # Load metadata
        metadata_path = os.path.join(model_dir, f"{self.model_name}_metadata.pkl")
        metadata = joblib.load(metadata_path)
        
        self.input_dim = metadata['input_dim']
        self.encoding_dim = metadata['encoding_dim']
        self.hidden_layers = metadata['hidden_layers']
        self.dropout_rate = metadata['dropout_rate']
        self.learning_rate = metadata['learning_rate']
        self.feature_names = metadata['feature_names']
        self.is_trained = metadata['is_trained']
        self.reconstruction_threshold = metadata['reconstruction_threshold']
        
        logger.info(f"Market Autoencoder loaded from {model_dir}")


class LSTMAnomalyDetector:
    """
    LSTM-based autoencoder for time series anomaly detection.
    
    This model uses LSTM layers to capture temporal patterns
    and detect anomalies in time series data.
    """
    
    def __init__(
        self,
        sequence_length: int = 30,
        features: int = 10,
        lstm_units: List[int] = [32, 16],
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        model_name: str = "lstm_anomaly_detector"
    ):
        """
        Initialize LSTM Anomaly Detector.
        
        Args:
            sequence_length: Number of time steps in sequence
            features: Number of input features
            lstm_units: List of LSTM units for encoder/decoder
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
        
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self.scaler = None
        self.feature_names = None
        self.is_trained = False
        self.reconstruction_threshold = None
        
        logger.info(f"Initialized LSTM Anomaly Detector: {model_name}")
    
    def build_model(self) -> Tuple[Model, Model, Model]:
        """Build the LSTM autoencoder model."""
        # Input layer
        input_layer = Input(shape=(self.sequence_length, self.features))
        
        # Encoder LSTM layers
        encoded = input_layer
        for i, units in enumerate(self.lstm_units):
            return_sequences = i < len(self.lstm_units) - 1
            encoded = LSTM(
                units=units,
                return_sequences=return_sequences,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate
            )(encoded)
            encoded = BatchNormalization()(encoded)
        
        # Decoder LSTM layers
        decoded = RepeatVector(self.sequence_length)(encoded)
        for units in reversed(self.lstm_units):
            decoded = LSTM(
                units=units,
                return_sequences=True,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate
            )(decoded)
            decoded = BatchNormalization()(decoded)
        
        # Output layer
        decoded = TimeDistributed(Dense(self.features, activation='linear'))(decoded)
        
        # Create models
        autoencoder = Model(input_layer, decoded)
        
        # Encoder model
        encoder = Model(input_layer, encoded)
        
        # Decoder model
        encoded_input = Input(shape=(self.lstm_units[-1],))
        decoder_layer = RepeatVector(self.sequence_length)(encoded_input)
        for units in reversed(self.lstm_units):
            decoder_layer = LSTM(
                units=units,
                return_sequences=True,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate
            )(decoder_layer)
            decoder_layer = BatchNormalization()(decoder_layer)
        decoder_layer = TimeDistributed(Dense(self.features, activation='linear'))(decoder_layer)
        decoder = Model(encoded_input, decoder_layer)
        
        # Compile autoencoder
        autoencoder.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        self.autoencoder = autoencoder
        self.encoder = encoder
        self.decoder = decoder
        
        logger.info(f"Built LSTM Anomaly Detector with {autoencoder.count_params()} parameters")
        
        return autoencoder, encoder, decoder
    
    def prepare_data(
        self,
        data: pd.DataFrame,
        feature_columns: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Prepare data for LSTM autoencoder training.
        
        Args:
            data: Input DataFrame with market data
            feature_columns: Features to use (if None, auto-select)
            
        Returns:
            Sequence array for training
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
        
        # Normalize features
        self.scaler = MinMaxScaler()
        features_scaled = self.scaler.fit_transform(features_data)
        
        # Create sequences
        sequences = []
        for i in range(self.sequence_length, len(features_scaled)):
            sequences.append(features_scaled[i-self.sequence_length:i])
        
        sequences = np.array(sequences)
        
        logger.info(f"Prepared LSTM autoencoder data: shape {sequences.shape}")
        
        return sequences
    
    def train(
        self,
        X: np.ndarray,
        validation_split: float = 0.2,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: int = 1
    ) -> Dict[str, Any]:
        """Train the LSTM autoencoder model."""
        if self.autoencoder is None:
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
        
        # Train autoencoder
        history = self.autoencoder.fit(
            X, X,  # LSTM autoencoder learns to reconstruct sequences
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.is_trained = True
        
        # Calculate reconstruction threshold
        self._calculate_reconstruction_threshold(X)
        
        logger.info("LSTM Anomaly Detector training completed")
        
        return history.history
    
    def _calculate_reconstruction_threshold(self, X: np.ndarray, percentile: float = 95.0) -> None:
        """Calculate reconstruction error threshold for anomaly detection."""
        # Get reconstruction errors for training data
        reconstructions = self.autoencoder.predict(X, verbose=0)
        reconstruction_errors = np.mean(np.square(X - reconstructions), axis=(1, 2))
        
        # Set threshold at specified percentile
        self.reconstruction_threshold = np.percentile(reconstruction_errors, percentile)
        
        logger.info(f"LSTM reconstruction threshold set to {self.reconstruction_threshold:.6f}")
    
    def detect_anomalies(
        self,
        data: pd.DataFrame,
        threshold_multiplier: float = 1.0
    ) -> Dict[str, Any]:
        """Detect anomalies in time series data."""
        if not self.is_trained:
            raise ValueError("Model must be trained before detecting anomalies")
        
        # Prepare data
        features_data = data[self.feature_names].values
        features_scaled = self.scaler.transform(features_data)
        
        # Create sequences
        sequences = []
        for i in range(self.sequence_length, len(features_scaled)):
            sequences.append(features_scaled[i-self.sequence_length:i])
        
        if len(sequences) == 0:
            return {
                'anomalies_detected': 0,
                'total_samples': len(data),
                'anomaly_rate': 0.0,
                'threshold': 0.0,
                'anomaly_indices': [],
                'anomaly_scores': [],
                'reconstruction_errors': [],
                'timestamp': datetime.now().isoformat()
            }
        
        sequences = np.array(sequences)
        
        # Get reconstructions
        reconstructions = self.autoencoder.predict(sequences, verbose=0)
        
        # Calculate reconstruction errors
        reconstruction_errors = np.mean(np.square(sequences - reconstructions), axis=(1, 2))
        
        # Detect anomalies
        threshold = self.reconstruction_threshold * threshold_multiplier
        anomalies = reconstruction_errors > threshold
        
        # Get anomaly details
        anomaly_indices = np.where(anomalies)[0]
        anomaly_scores = reconstruction_errors[anomalies]
        
        return {
            'anomalies_detected': int(np.sum(anomalies)),
            'total_samples': len(sequences),
            'anomaly_rate': float(np.mean(anomalies)),
            'threshold': float(threshold),
            'anomaly_indices': anomaly_indices.tolist(),
            'anomaly_scores': anomaly_scores.tolist(),
            'reconstruction_errors': reconstruction_errors.tolist(),
            'timestamp': datetime.now().isoformat()
        }
    
    def save_model(self, model_dir: str) -> None:
        """Save the trained model and scaler."""
        os.makedirs(model_dir, exist_ok=True)
        
        # Save models
        autoencoder_path = os.path.join(model_dir, f"{self.model_name}_autoencoder.h5")
        self.autoencoder.save(autoencoder_path)
        
        encoder_path = os.path.join(model_dir, f"{self.model_name}_encoder.h5")
        self.encoder.save(encoder_path)
        
        decoder_path = os.path.join(model_dir, f"{self.model_name}_decoder.h5")
        self.decoder.save(decoder_path)
        
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
            'is_trained': self.is_trained,
            'reconstruction_threshold': self.reconstruction_threshold
        }
        
        metadata_path = os.path.join(model_dir, f"{self.model_name}_metadata.pkl")
        joblib.dump(metadata, metadata_path)
        
        logger.info(f"LSTM Anomaly Detector saved to {model_dir}")
    
    def load_model(self, model_dir: str) -> None:
        """Load a trained model and scaler."""
        # Load models
        autoencoder_path = os.path.join(model_dir, f"{self.model_name}_autoencoder.h5")
        self.autoencoder = tf.keras.models.load_model(autoencoder_path)
        
        encoder_path = os.path.join(model_dir, f"{self.model_name}_encoder.h5")
        self.encoder = tf.keras.models.load_model(encoder_path)
        
        decoder_path = os.path.join(model_dir, f"{self.model_name}_decoder.h5")
        self.decoder = tf.keras.models.load_model(decoder_path)
        
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
        self.reconstruction_threshold = metadata['reconstruction_threshold']
        
        logger.info(f"LSTM Anomaly Detector loaded from {model_dir}")


class AnomalyDetector:
    """
    Unified anomaly detection system combining multiple autoencoder models.
    
    This class provides a high-level interface for anomaly detection
    using different autoencoder architectures.
    """
    
    def __init__(
        self,
        model_types: List[str] = ['standard', 'lstm'],
        model_configs: Optional[Dict[str, Dict]] = None
    ):
        """
        Initialize unified anomaly detector.
        
        Args:
            model_types: List of autoencoder types to use
            model_configs: Configuration for each model type
        """
        self.model_types = model_types
        self.model_configs = model_configs or {}
        self.models = {}
        self.is_trained = False
        
        # Initialize models
        for model_type in model_types:
            if model_type == 'standard':
                config = self.model_configs.get('standard', {})
                self.models['standard'] = MarketAutoencoder(**config)
            elif model_type == 'lstm':
                config = self.model_configs.get('lstm', {})
                self.models['lstm'] = LSTMAnomalyDetector(**config)
        
        logger.info(f"Initialized Anomaly Detector with models: {model_types}")
    
    def train(
        self,
        data: pd.DataFrame,
        feature_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Train all anomaly detection models.
        
        Args:
            data: Input DataFrame with market data
            feature_columns: Features to use
            
        Returns:
            Training results for all models
        """
        results = {}
        
        for model_name, model in self.models.items():
            logger.info(f"Training {model_name} anomaly detector...")
            
            # Prepare data for this model
            if model_name == 'lstm':
                X = model.prepare_data(data, feature_columns)
            else:
                X = model.prepare_data(data, feature_columns)
            
            # Train model
            history = model.train(X)
            results[model_name] = history
        
        self.is_trained = True
        logger.info("All anomaly detection models trained")
        
        return results
    
    def detect_anomalies(
        self,
        data: pd.DataFrame,
        threshold_multiplier: float = 1.0,
        ensemble_method: str = 'majority'
    ) -> Dict[str, Any]:
        """
        Detect anomalies using ensemble of models.
        
        Args:
            data: Input DataFrame with market data
            threshold_multiplier: Multiplier for threshold adjustment
            ensemble_method: Method for combining results ('majority', 'average', 'max')
            
        Returns:
            Ensemble anomaly detection results
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before detecting anomalies")
        
        # Get results from all models
        model_results = {}
        for model_name, model in self.models.items():
            model_results[model_name] = model.detect_anomalies(data, threshold_multiplier)
        
        # Combine results based on ensemble method
        if ensemble_method == 'majority':
            # Majority voting
            all_anomalies = set()
            for result in model_results.values():
                all_anomalies.update(result['anomaly_indices'])
            
            # Count votes for each index
            anomaly_votes = {}
            for result in model_results.values():
                for idx in result['anomaly_indices']:
                    anomaly_votes[idx] = anomaly_votes.get(idx, 0) + 1
            
            # Anomalies detected by majority
            majority_threshold = len(self.models) // 2 + 1
            ensemble_anomalies = [idx for idx, votes in anomaly_votes.items() if votes >= majority_threshold]
            
        elif ensemble_method == 'average':
            # Average reconstruction errors
            all_indices = set()
            for result in model_results.values():
                all_indices.update(result['anomaly_indices'])
            
            # Calculate average scores
            avg_scores = {}
            for idx in all_indices:
                scores = []
                for result in model_results.values():
                    if idx in result['anomaly_indices']:
                        score_idx = result['anomaly_indices'].index(idx)
                        scores.append(result['anomaly_scores'][score_idx])
                avg_scores[idx] = np.mean(scores) if scores else 0
            
            # Threshold based on average
            avg_threshold = np.mean([result['threshold'] for result in model_results.values()])
            ensemble_anomalies = [idx for idx, score in avg_scores.items() if score > avg_threshold]
            
        else:  # max
            # Union of all anomalies
            ensemble_anomalies = set()
            for result in model_results.values():
                ensemble_anomalies.update(result['anomaly_indices'])
            ensemble_anomalies = list(ensemble_anomalies)
        
        return {
            'ensemble_anomalies': ensemble_anomalies,
            'anomalies_detected': len(ensemble_anomalies),
            'total_samples': model_results[list(model_results.keys())[0]]['total_samples'],
            'anomaly_rate': len(ensemble_anomalies) / model_results[list(model_results.keys())[0]]['total_samples'],
            'ensemble_method': ensemble_method,
            'model_results': model_results,
            'timestamp': datetime.now().isoformat()
        }
    
    def save_models(self, model_dir: str) -> None:
        """Save all trained models."""
        for model_name, model in self.models.items():
            model.save_model(os.path.join(model_dir, model_name))
        
        logger.info(f"All anomaly detection models saved to {model_dir}")
    
    def load_models(self, model_dir: str) -> None:
        """Load all trained models."""
        for model_name, model in self.models.items():
            model.load_model(os.path.join(model_dir, model_name))
        
        self.is_trained = True
        logger.info(f"All anomaly detection models loaded from {model_dir}")

