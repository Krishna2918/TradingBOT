"""
Deep Learning Model Manager for advanced ML models.

This module provides a unified interface for managing, training,
and deploying deep learning models in the trading system.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from .lstm_models import LSTMPricePredictor, LSTMTrendPredictor
from .cnn_lstm_models import CNNLSTMPredictor, MultiScaleCNNLSTM
from .transformer_models import TransformerPredictor, FinancialTransformer
from .autoencoder_models import MarketAutoencoder, LSTMAnomalyDetector, AnomalyDetector

logger = logging.getLogger(__name__)

class DeepLearningModelManager:
    """
    Manager for deep learning models in the trading system.
    
    This class provides a unified interface for managing multiple
    deep learning models, including training, inference, and deployment.
    """
    
    def __init__(
        self,
        model_dir: str = "models/deep_learning",
        config_file: Optional[str] = None
    ):
        """
        Initialize Deep Learning Model Manager.
        
        Args:
            model_dir: Directory for storing models
            config_file: Path to configuration file
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.model_configs = {}
        self.model_metadata = {}
        
        # Load configuration
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)
        else:
            self._load_default_config()
        
        logger.info(f"Initialized Deep Learning Model Manager: {model_dir}")
    
    def _load_default_config(self) -> None:
        """Load default model configurations."""
        self.model_configs = {
            'lstm_price': {
                'class': LSTMPricePredictor,
                'params': {
                    'sequence_length': 60,
                    'features': 10,
                    'lstm_units': [50, 50],
                    'dropout_rate': 0.2,
                    'learning_rate': 0.001,
                    'model_name': 'lstm_price_predictor'
                }
            },
            'lstm_trend': {
                'class': LSTMTrendPredictor,
                'params': {
                    'sequence_length': 30,
                    'features': 8,
                    'lstm_units': [32, 32],
                    'dropout_rate': 0.3,
                    'learning_rate': 0.001,
                    'model_name': 'lstm_trend_predictor'
                }
            },
            'cnn_lstm': {
                'class': CNNLSTMPredictor,
                'params': {
                    'sequence_length': 60,
                    'features': 10,
                    'cnn_filters': [64, 32],
                    'cnn_kernel_size': 3,
                    'lstm_units': [50, 50],
                    'dropout_rate': 0.2,
                    'learning_rate': 0.001,
                    'model_name': 'cnn_lstm_predictor'
                }
            },
            'multi_scale_cnn_lstm': {
                'class': MultiScaleCNNLSTM,
                'params': {
                    'sequence_lengths': [20, 40, 60],
                    'features': 10,
                    'cnn_filters': [32, 16],
                    'lstm_units': [32, 16],
                    'dropout_rate': 0.3,
                    'learning_rate': 0.001,
                    'model_name': 'multi_scale_cnn_lstm'
                }
            },
            'transformer': {
                'class': TransformerPredictor,
                'params': {
                    'sequence_length': 60,
                    'features': 10,
                    'd_model': 64,
                    'num_heads': 8,
                    'num_layers': 4,
                    'dff': 128,
                    'dropout_rate': 0.1,
                    'learning_rate': 0.001,
                    'model_name': 'transformer_predictor'
                }
            },
            'financial_transformer': {
                'class': FinancialTransformer,
                'params': {
                    'sequence_length': 60,
                    'features': 10,
                    'd_model': 128,
                    'num_heads': 8,
                    'num_layers': 6,
                    'dff': 256,
                    'dropout_rate': 0.1,
                    'learning_rate': 0.001,
                    'model_name': 'financial_transformer'
                }
            },
            'market_autoencoder': {
                'class': MarketAutoencoder,
                'params': {
                    'input_dim': 10,
                    'encoding_dim': 5,
                    'hidden_layers': [8, 6],
                    'dropout_rate': 0.2,
                    'learning_rate': 0.001,
                    'model_name': 'market_autoencoder'
                }
            },
            'lstm_anomaly_detector': {
                'class': LSTMAnomalyDetector,
                'params': {
                    'sequence_length': 30,
                    'features': 10,
                    'lstm_units': [32, 16],
                    'dropout_rate': 0.2,
                    'learning_rate': 0.001,
                    'model_name': 'lstm_anomaly_detector'
                }
            }
        }
    
    def load_config(self, config_file: str) -> None:
        """Load model configurations from file."""
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        self.model_configs = config_data.get('model_configs', {})
        logger.info(f"Loaded configuration from {config_file}")
    
    def save_config(self, config_file: str) -> None:
        """Save model configurations to file."""
        config_data = {
            'model_configs': self.model_configs,
            'metadata': self.model_metadata,
            'last_updated': datetime.now().isoformat()
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"Saved configuration to {config_file}")
    
    def create_model(self, model_name: str, **kwargs) -> Any:
        """
        Create a new model instance.
        
        Args:
            model_name: Name of the model to create
            **kwargs: Additional parameters for model creation
            
        Returns:
            Model instance
        """
        if model_name not in self.model_configs:
            raise ValueError(f"Unknown model type: {model_name}")
        
        config = self.model_configs[model_name]
        model_class = config['class']
        params = config['params'].copy()
        
        # Update parameters with kwargs
        params.update(kwargs)
        
        # Create model instance
        model = model_class(**params)
        
        # Store model
        self.models[model_name] = model
        
        # Store metadata
        self.model_metadata[model_name] = {
            'created_at': datetime.now().isoformat(),
            'model_type': model_name,
            'parameters': params,
            'is_trained': False
        }
        
        logger.info(f"Created model: {model_name}")
        
        return model
    
    def get_model(self, model_name: str) -> Any:
        """
        Get an existing model instance.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model instance
        """
        if model_name not in self.models:
            raise ValueError(f"Model not found: {model_name}")
        
        return self.models[model_name]
    
    def train_model(
        self,
        model_name: str,
        data: pd.DataFrame,
        target_column: str = 'close',
        feature_columns: Optional[List[str]] = None,
        validation_split: float = 0.2,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: int = 1
    ) -> Dict[str, Any]:
        """
        Train a model with the provided data.
        
        Args:
            model_name: Name of the model to train
            data: Training data
            target_column: Target column for prediction
            feature_columns: Features to use
            validation_split: Fraction for validation
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        if model_name not in self.models:
            raise ValueError(f"Model not found: {model_name}")
        
        model = self.models[model_name]
        
        logger.info(f"Training model: {model_name}")
        
        # Prepare data based on model type
        if hasattr(model, 'prepare_data'):
            if 'autoencoder' in model_name.lower():
                # Autoencoder models don't need target column
                X = model.prepare_data(data, feature_columns)
                history = model.train(X, validation_split, epochs, batch_size, verbose)
            else:
                # Prediction models need target column
                X, y = model.prepare_data(data, target_column, feature_columns)
                history = model.train(X, y, validation_split, epochs, batch_size, verbose)
        else:
            raise ValueError(f"Model {model_name} does not support training")
        
        # Update metadata
        self.model_metadata[model_name]['is_trained'] = True
        self.model_metadata[model_name]['trained_at'] = datetime.now().isoformat()
        self.model_metadata[model_name]['training_history'] = history
        
        logger.info(f"Model {model_name} training completed")
        
        return history
    
    def predict(
        self,
        model_name: str,
        data: pd.DataFrame,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make predictions using a trained model.
        
        Args:
            model_name: Name of the model to use
            data: Input data for prediction
            **kwargs: Additional parameters for prediction
            
        Returns:
            Prediction results
        """
        if model_name not in self.models:
            raise ValueError(f"Model not found: {model_name}")
        
        model = self.models[model_name]
        
        if not model.is_trained:
            raise ValueError(f"Model {model_name} is not trained")
        
        logger.info(f"Making predictions with model: {model_name}")
        
        # Make predictions based on model type
        if 'autoencoder' in model_name.lower() or 'anomaly' in model_name.lower():
            # Anomaly detection models
            result = model.detect_anomalies(data, **kwargs)
        elif hasattr(model, 'predict_next_price'):
            # Price prediction models
            result = model.predict_next_price(data, **kwargs)
        elif hasattr(model, 'predict_trend'):
            # Trend prediction models
            result = model.predict_trend(data, **kwargs)
        else:
            raise ValueError(f"Model {model_name} does not support prediction")
        
        # Add model metadata to result
        result['model_name'] = model_name
        result['model_type'] = self.model_metadata[model_name]['model_type']
        
        return result
    
    def get_model_performance(self, model_name: str) -> Dict[str, Any]:
        """
        Get performance metrics for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Performance metrics
        """
        if model_name not in self.model_metadata:
            raise ValueError(f"Model metadata not found: {model_name}")
        
        metadata = self.model_metadata[model_name]
        
        performance = {
            'model_name': model_name,
            'model_type': metadata['model_type'],
            'is_trained': metadata['is_trained'],
            'created_at': metadata['created_at'],
            'parameters': metadata['parameters']
        }
        
        if 'trained_at' in metadata:
            performance['trained_at'] = metadata['trained_at']
        
        if 'training_history' in metadata:
            history = metadata['training_history']
            performance['final_loss'] = history.get('loss', [0])[-1] if history.get('loss') else 0
            performance['final_val_loss'] = history.get('val_loss', [0])[-1] if history.get('val_loss') else 0
            performance['training_epochs'] = len(history.get('loss', []))
        
        return performance
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all available models.
        
        Returns:
            List of model information
        """
        models_info = []
        
        for model_name in self.model_configs.keys():
            info = {
                'name': model_name,
                'class': self.model_configs[model_name]['class'].__name__,
                'parameters': self.model_configs[model_name]['params'],
                'is_created': model_name in self.models,
                'is_trained': self.model_metadata.get(model_name, {}).get('is_trained', False)
            }
            
            if model_name in self.model_metadata:
                info.update({
                    'created_at': self.model_metadata[model_name].get('created_at'),
                    'trained_at': self.model_metadata[model_name].get('trained_at')
                })
            
            models_info.append(info)
        
        return models_info
    
    def save_model(self, model_name: str) -> None:
        """
        Save a trained model to disk.
        
        Args:
            model_name: Name of the model to save
        """
        if model_name not in self.models:
            raise ValueError(f"Model not found: {model_name}")
        
        model = self.models[model_name]
        model_path = self.model_dir / model_name
        
        # Save model
        model.save_model(str(model_path))
        
        # Save metadata
        metadata_path = model_path / f"{model_name}_manager_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.model_metadata[model_name], f, indent=2)
        
        logger.info(f"Saved model {model_name} to {model_path}")
    
    def load_model(self, model_name: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            model_name: Name of the model to load
        """
        if model_name not in self.model_configs:
            raise ValueError(f"Unknown model type: {model_name}")
        
        model_path = self.model_dir / model_name
        
        if not model_path.exists():
            raise ValueError(f"Model directory not found: {model_path}")
        
        # Create model instance
        model = self.create_model(model_name)
        
        # Load trained model
        model.load_model(str(model_path))
        
        # Load metadata
        metadata_path = model_path / f"{model_name}_manager_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.model_metadata[model_name] = json.load(f)
        
        logger.info(f"Loaded model {model_name} from {model_path}")
    
    def save_all_models(self) -> None:
        """Save all trained models to disk."""
        for model_name in self.models.keys():
            if self.models[model_name].is_trained:
                self.save_model(model_name)
        
        logger.info("Saved all trained models")
    
    def load_all_models(self) -> None:
        """Load all available models from disk."""
        for model_name in self.model_configs.keys():
            model_path = self.model_dir / model_name
            if model_path.exists():
                try:
                    self.load_model(model_name)
                except Exception as e:
                    logger.warning(f"Failed to load model {model_name}: {e}")
        
        logger.info("Loaded all available models")
    
    def delete_model(self, model_name: str) -> None:
        """
        Delete a model and its files.
        
        Args:
            model_name: Name of the model to delete
        """
        # Remove from memory
        if model_name in self.models:
            del self.models[model_name]
        
        if model_name in self.model_metadata:
            del self.model_metadata[model_name]
        
        # Remove from disk
        model_path = self.model_dir / model_name
        if model_path.exists():
            import shutil
            shutil.rmtree(model_path)
        
        logger.info(f"Deleted model: {model_name}")
    
    def get_model_status(self) -> Dict[str, Any]:
        """
        Get status of all models.
        
        Returns:
            Dictionary with model status information
        """
        status = {
            'total_models': len(self.model_configs),
            'created_models': len(self.models),
            'trained_models': sum(1 for m in self.model_metadata.values() if m.get('is_trained', False)),
            'models': {}
        }
        
        for model_name in self.model_configs.keys():
            model_status = {
                'is_created': model_name in self.models,
                'is_trained': self.model_metadata.get(model_name, {}).get('is_trained', False),
                'model_type': self.model_configs[model_name]['class'].__name__
            }
            
            if model_name in self.model_metadata:
                model_status.update({
                    'created_at': self.model_metadata[model_name].get('created_at'),
                    'trained_at': self.model_metadata[model_name].get('trained_at')
                })
            
            status['models'][model_name] = model_status
        
        return status
    
    def create_ensemble_prediction(
        self,
        data: pd.DataFrame,
        model_names: Optional[List[str]] = None,
        weights: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Create ensemble prediction using multiple models.
        
        Args:
            data: Input data for prediction
            model_names: List of model names to use (if None, use all trained models)
            weights: Weights for each model (if None, use equal weights)
            
        Returns:
            Ensemble prediction results
        """
        if model_names is None:
            # Use all trained models
            model_names = [
                name for name, metadata in self.model_metadata.items()
                if metadata.get('is_trained', False)
            ]
        
        if not model_names:
            raise ValueError("No trained models available for ensemble prediction")
        
        if weights is None:
            # Use equal weights
            weights = [1.0 / len(model_names)] * len(model_names)
        
        if len(weights) != len(model_names):
            raise ValueError("Number of weights must match number of models")
        
        # Get predictions from all models
        predictions = []
        confidences = []
        
        for model_name in model_names:
            try:
                result = self.predict(model_name, data)
                
                if 'prediction' in result:
                    predictions.append(result['prediction'])
                    confidences.append(result.get('confidence', 0.5))
                elif 'trend' in result:
                    # Convert trend to numeric
                    trend_map = {'UP': 1, 'DOWN': -1, 'SIDEWAYS': 0}
                    predictions.append(trend_map.get(result['trend'], 0))
                    confidences.append(result.get('confidence', 0.5))
                else:
                    logger.warning(f"Model {model_name} returned unexpected result format")
                    
            except Exception as e:
                logger.warning(f"Failed to get prediction from {model_name}: {e}")
        
        if not predictions:
            raise ValueError("No valid predictions obtained from models")
        
        # Calculate weighted ensemble prediction
        weighted_prediction = sum(p * w for p, w in zip(predictions, weights))
        weighted_confidence = sum(c * w for c, w in zip(confidences, weights))
        
        return {
            'ensemble_prediction': weighted_prediction,
            'ensemble_confidence': weighted_confidence,
            'individual_predictions': dict(zip(model_names, predictions)),
            'individual_confidences': dict(zip(model_names, confidences)),
            'weights': dict(zip(model_names, weights)),
            'model_count': len(model_names),
            'timestamp': datetime.now().isoformat()
        }

