"""
Time Series Model Manager for financial time series prediction.

This module provides a unified interface for managing, training,
and deploying time series models in the trading system.
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

from .arima_garch import ARIMAGARCHPredictor, GARCHVolatilityPredictor
from .prophet_models import ProphetPredictor, ProphetAnomalyDetector
from .var_models import VARPredictor, VECMPredictor
from .state_space import KalmanFilterPredictor, DynamicLinearModel
from .seasonality import SeasonalityDetector, SeasonalDecomposer

logger = logging.getLogger(__name__)

class TimeSeriesModelManager:
    """
    Manager for time series models in the trading system.
    
    This class provides a unified interface for managing multiple
    time series models, including training, inference, and deployment.
    """
    
    def __init__(
        self,
        model_dir: str = "models/time_series",
        config_file: Optional[str] = None
    ):
        """
        Initialize Time Series Model Manager.
        
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
        
        logger.info(f"Initialized Time Series Model Manager: {model_dir}")
    
    def _load_default_config(self) -> None:
        """Load default model configurations."""
        self.model_configs = {
            'arima_garch': {
                'class': ARIMAGARCHPredictor,
                'params': {
                    'arima_order': (1, 1, 1),
                    'garch_order': (1, 1),
                    'garch_model': 'GARCH',
                    'distribution': 'normal',
                    'model_name': 'arima_garch_predictor'
                }
            },
            'garch_volatility': {
                'class': GARCHVolatilityPredictor,
                'params': {
                    'garch_order': (1, 1),
                    'garch_model': 'GARCH',
                    'distribution': 'normal',
                    'model_name': 'garch_volatility_predictor'
                }
            },
            'prophet': {
                'class': ProphetPredictor,
                'params': {
                    'growth': 'linear',
                    'seasonality_mode': 'additive',
                    'yearly_seasonality': True,
                    'weekly_seasonality': True,
                    'daily_seasonality': False,
                    'model_name': 'prophet_predictor'
                }
            },
            'prophet_anomaly': {
                'class': ProphetAnomalyDetector,
                'params': {
                    'anomaly_threshold': 2.0,
                    'confidence_level': 0.95,
                    'model_name': 'prophet_anomaly_detector'
                }
            },
            'var': {
                'class': VARPredictor,
                'params': {
                    'maxlags': 15,
                    'ic': 'aic',
                    'model_name': 'var_predictor'
                }
            },
            'vecm': {
                'class': VECMPredictor,
                'params': {
                    'k_ar_diff': 1,
                    'coint_rank': None,
                    'deterministic': 'ci',
                    'model_name': 'vecm_predictor'
                }
            },
            'kalman_filter': {
                'class': KalmanFilterPredictor,
                'params': {
                    'state_dim': 2,
                    'observation_dim': 1,
                    'model_name': 'kalman_filter_predictor'
                }
            },
            'dynamic_linear': {
                'class': DynamicLinearModel,
                'params': {
                    'k_factors': 1,
                    'factor_order': 1,
                    'model_name': 'dynamic_linear_model'
                }
            },
            'seasonality_detector': {
                'class': SeasonalityDetector,
                'params': {
                    'model_name': 'seasonality_detector'
                }
            },
            'seasonal_decomposer': {
                'class': SeasonalDecomposer,
                'params': {
                    'model_name': 'seasonal_decomposer'
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
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train a model with the provided data.
        
        Args:
            model_name: Name of the model to train
            data: Training data
            **kwargs: Additional parameters for training
            
        Returns:
            Training results
        """
        if model_name not in self.models:
            raise ValueError(f"Model not found: {model_name}")
        
        model = self.models[model_name]
        
        logger.info(f"Training model: {model_name}")
        
        try:
            # Train model based on type
            if model_name in ['arima_garch', 'garch_volatility']:
                # ARIMA-GARCH models
                results = model.fit(data, **kwargs)
            elif model_name in ['prophet', 'prophet_anomaly']:
                # Prophet models
                results = model.fit(data, **kwargs)
            elif model_name in ['var', 'vecm']:
                # VAR/VECM models
                variables = kwargs.get('variables', ['close', 'volume'])
                results = model.fit(data, variables, **kwargs)
            elif model_name in ['kalman_filter', 'dynamic_linear']:
                # State space models
                results = model.fit(data, **kwargs)
            elif model_name in ['seasonality_detector', 'seasonal_decomposer']:
                # Analysis models
                results = model.analyze_seasonality(data, **kwargs) if model_name == 'seasonality_detector' else model.decompose(data, **kwargs)
            else:
                raise ValueError(f"Unknown model type for training: {model_name}")
            
            # Update metadata
            self.model_metadata[model_name]['is_trained'] = True
            self.model_metadata[model_name]['trained_at'] = datetime.now().isoformat()
            self.model_metadata[model_name]['training_results'] = results
            
            logger.info(f"Model {model_name} training completed")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to train model {model_name}: {e}")
            raise
    
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
        
        if not model.is_fitted and not model.is_trained and not model.is_analyzed and not model.is_decomposed:
            raise ValueError(f"Model {model_name} is not trained")
        
        logger.info(f"Making predictions with model: {model_name}")
        
        try:
            # Make predictions based on model type
            if model_name == 'arima_garch':
                result = model.predict_next_return(data, **kwargs)
            elif model_name == 'garch_volatility':
                result = model.predict_next_volatility(data, **kwargs)
            elif model_name == 'prophet':
                result = model.predict_next_value(data, **kwargs)
            elif model_name == 'prophet_anomaly':
                result = model.detect_anomalies(data, **kwargs)
            elif model_name == 'var':
                variables = kwargs.get('variables', ['close', 'volume'])
                result = model.predict_next_values(data, variables)
            elif model_name == 'vecm':
                variables = kwargs.get('variables', ['close', 'volume'])
                result = model.predict_next_values(data, variables)
            elif model_name == 'kalman_filter':
                result = model.predict_next_value(data, **kwargs)
            elif model_name == 'dynamic_linear':
                variables = kwargs.get('variables', ['close', 'volume'])
                result = model.predict_next_values(data, variables)
            elif model_name == 'seasonality_detector':
                result = model.get_seasonal_patterns()
            elif model_name == 'seasonal_decomposer':
                result = model.get_component_statistics()
            else:
                raise ValueError(f"Unknown model type for prediction: {model_name}")
            
            # Add model metadata to result
            result['model_name'] = model_name
            result['model_type'] = self.model_metadata[model_name]['model_type']
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to make predictions with {model_name}: {e}")
            raise
    
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
        
        if 'training_results' in metadata:
            training_results = metadata['training_results']
            # Extract performance metrics based on model type
            if 'aic' in training_results:
                performance['aic'] = training_results['aic']
            if 'bic' in training_results:
                performance['bic'] = training_results['bic']
            if 'loglikelihood' in training_results:
                performance['loglikelihood'] = training_results['loglikelihood']
        
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
            model = self.models[model_name]
            if (hasattr(model, 'is_fitted') and model.is_fitted) or \
               (hasattr(model, 'is_trained') and model.is_trained) or \
               (hasattr(model, 'is_analyzed') and model.is_analyzed) or \
               (hasattr(model, 'is_decomposed') and model.is_decomposed):
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
        Create ensemble prediction using multiple time series models.
        
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
                
                if 'predicted_value' in result:
                    predictions.append(result['predicted_value'])
                    confidences.append(0.8)  # Default confidence for time series models
                elif 'predicted_return' in result:
                    predictions.append(result['predicted_return'])
                    confidences.append(0.8)
                elif 'predicted_volatility' in result:
                    predictions.append(result['predicted_volatility'])
                    confidences.append(0.8)
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

