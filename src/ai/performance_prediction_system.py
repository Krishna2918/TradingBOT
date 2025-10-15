"""
Performance Prediction System
============================

Advanced ML-based system for predicting model performance in different market conditions.
Uses ensemble ML models to forecast accuracy, reliability, and optimal usage scenarios.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import threading
import pickle
import os

# ML imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

logger = logging.getLogger(__name__)

@dataclass
class PerformancePrediction:
    """Prediction of model performance."""
    model_name: str
    predicted_accuracy: float
    predicted_brier_score: float
    predicted_execution_time: float
    confidence_interval: Tuple[float, float]
    feature_importance: Dict[str, float]
    prediction_confidence: float
    timestamp: datetime
    conditions: Dict[str, Any]

@dataclass
class PredictionModel:
    """ML model for performance prediction."""
    model_type: str  # 'accuracy', 'brier_score', 'execution_time'
    model: Any  # sklearn model
    scaler: StandardScaler
    feature_names: List[str]
    performance_metrics: Dict[str, float]
    last_trained: datetime
    training_samples: int

class PerformancePredictionSystem:
    """
    Advanced ML system for predicting model performance.
    
    Features:
    - Multi-target prediction (accuracy, brier score, execution time)
    - Feature engineering for market conditions
    - Ensemble ML models
    - Model retraining and adaptation
    - Performance forecasting
    """
    
    def __init__(self, model_storage_path: str = "data/performance_models"):
        """
        Initialize the performance prediction system.
        
        Args:
            model_storage_path: Path to store trained models
        """
        self.system_name = "performance_prediction_system"
        self.model_storage_path = model_storage_path
        
        # Ensure storage directory exists
        os.makedirs(model_storage_path, exist_ok=True)
        
        # Prediction models
        self.prediction_models: Dict[str, PredictionModel] = {}
        self.feature_encoders: Dict[str, LabelEncoder] = {}
        
        # Training data
        self.training_data: deque = deque(maxlen=50000)
        self.feature_names: List[str] = []
        
        # System state
        self.is_trained = False
        self.last_training = None
        self.retraining_threshold = 1000  # Retrain after 1000 new samples
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize feature names
        self._initialize_feature_names()
        
        logger.info(f"Performance Prediction System initialized: {self.system_name}")
    
    def add_training_data(self, model_name: str, conditions: Dict[str, Any], 
                         accuracy: float, brier_score: float, execution_time: float) -> bool:
        """
        Add training data for model performance prediction.
        
        Args:
            model_name: Name of the model
            conditions: Market conditions
            accuracy: Actual accuracy achieved
            brier_score: Actual brier score
            execution_time: Actual execution time
            
        Returns:
            True if data added successfully
        """
        try:
            with self._lock:
                # Create feature vector
                features = self._create_feature_vector(model_name, conditions)
                
                # Add to training data
                training_sample = {
                    'model_name': model_name,
                    'conditions': conditions.copy(),
                    'features': features,
                    'accuracy': accuracy,
                    'brier_score': brier_score,
                    'execution_time': execution_time,
                    'timestamp': datetime.now()
                }
                
                self.training_data.append(training_sample)
                
                # Check if retraining is needed
                if len(self.training_data) % self.retraining_threshold == 0:
                    self._trigger_retraining()
                
                logger.debug(f"Training data added for {model_name}: accuracy={accuracy:.3f}")
                return True
                
        except Exception as e:
            logger.error(f"Error adding training data: {e}")
            return False
    
    def predict_performance(self, model_name: str, conditions: Dict[str, Any]) -> PerformancePrediction:
        """
        Predict model performance in given conditions.
        
        Args:
            model_name: Name of the model
            conditions: Market conditions
            
        Returns:
            Performance prediction
        """
        try:
            with self._lock:
                if not self.is_trained:
                    # Return default prediction if not trained
                    return self._get_default_prediction(model_name, conditions)
                
                # Create feature vector
                features = self._create_feature_vector(model_name, conditions)
                features_array = np.array(features).reshape(1, -1)
                
                # Make predictions
                predictions = {}
                confidences = {}
                feature_importance = {}
                
                for target in ['accuracy', 'brier_score', 'execution_time']:
                    if target in self.prediction_models:
                        model_info = self.prediction_models[target]
                        
                        # Scale features
                        scaled_features = model_info.scaler.transform(features_array)
                        
                        # Make prediction
                        prediction = model_info.model.predict(scaled_features)[0]
                        predictions[target] = prediction
                        
                        # Calculate confidence (simplified)
                        confidences[target] = model_info.performance_metrics.get('r2_score', 0.5)
                        
                        # Get feature importance
                        if hasattr(model_info.model, 'feature_importances_'):
                            importance_dict = dict(zip(model_info.feature_names, model_info.model.feature_importances_))
                            feature_importance.update(importance_dict)
                
                # Calculate overall confidence (ensure non-negative)
                overall_confidence = max(0.0, np.mean(list(confidences.values()))) if confidences else 0.5
                
                # Calculate confidence interval (simplified)
                accuracy_std = 0.1  # Simplified standard deviation
                confidence_interval = (
                    max(0.0, predictions.get('accuracy', 0.5) - 1.96 * accuracy_std),
                    min(1.0, predictions.get('accuracy', 0.5) + 1.96 * accuracy_std)
                )
                
                return PerformancePrediction(
                    model_name=model_name,
                    predicted_accuracy=predictions.get('accuracy', 0.5),
                    predicted_brier_score=predictions.get('brier_score', 1.0),
                    predicted_execution_time=predictions.get('execution_time', 1.0),
                    confidence_interval=confidence_interval,
                    feature_importance=feature_importance,
                    prediction_confidence=overall_confidence,
                    timestamp=datetime.now(),
                    conditions=conditions.copy()
                )
                
        except Exception as e:
            logger.error(f"Error predicting performance for {model_name}: {e}")
            return self._get_default_prediction(model_name, conditions)
    
    def train_models(self) -> bool:
        """
        Train the performance prediction models.
        
        Returns:
            True if training successful
        """
        try:
            with self._lock:
                if len(self.training_data) < 100:  # Minimum training samples
                    logger.warning("Insufficient training data for model training")
                    return False
                
                # Prepare training data
                X, y_accuracy, y_brier, y_execution = self._prepare_training_data()
                
                if X is None or len(X) == 0:
                    logger.error("Failed to prepare training data")
                    return False
                
                # Train models for each target
                targets = {
                    'accuracy': y_accuracy,
                    'brier_score': y_brier,
                    'execution_time': y_execution
                }
                
                for target_name, y_target in targets.items():
                    if y_target is None or len(y_target) == 0:
                        continue
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y_target, test_size=0.2, random_state=42
                    )
                    
                    # Scale features
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Train ensemble model
                    model = self._create_ensemble_model()
                    model.fit(X_train_scaled, y_train)
                    
                    # Evaluate model
                    y_pred = model.predict(X_test_scaled)
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    # Store model
                    prediction_model = PredictionModel(
                        model_type=target_name,
                        model=model,
                        scaler=scaler,
                        feature_names=self.feature_names.copy(),
                        performance_metrics={
                            'mse': mse,
                            'r2_score': r2,
                            'rmse': np.sqrt(mse)
                        },
                        last_trained=datetime.now(),
                        training_samples=len(X_train)
                    )
                    
                    self.prediction_models[target_name] = prediction_model
                    
                    # Save model
                    self._save_model(target_name, prediction_model)
                    
                    logger.info(f"Model trained for {target_name}: R²={r2:.3f}, RMSE={np.sqrt(mse):.3f}")
                
                self.is_trained = True
                self.last_training = datetime.now()
                
                return True
                
        except Exception as e:
            logger.error(f"Error training models: {e}")
            return False
    
    def get_prediction_statistics(self) -> Dict[str, Any]:
        """Get prediction system statistics."""
        try:
            with self._lock:
                stats = {
                    'system_name': self.system_name,
                    'is_trained': self.is_trained,
                    'last_training': self.last_training.isoformat() if self.last_training else None,
                    'training_samples': len(self.training_data),
                    'trained_models': len(self.prediction_models),
                    'model_types': list(self.prediction_models.keys()),
                    'feature_count': len(self.feature_names)
                }
                
                # Add model performance metrics
                model_metrics = {}
                for target, model_info in self.prediction_models.items():
                    model_metrics[target] = {
                        'r2_score': model_info.performance_metrics.get('r2_score', 0.0),
                        'rmse': model_info.performance_metrics.get('rmse', 0.0),
                        'training_samples': model_info.training_samples,
                        'last_trained': model_info.last_trained.isoformat()
                    }
                
                stats['model_metrics'] = model_metrics
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting prediction statistics: {e}")
            return {'error': str(e)}
    
    def _initialize_feature_names(self):
        """Initialize feature names for the prediction system."""
        # Model features
        model_features = ['model_name_encoded']
        
        # Market condition features
        condition_features = [
            'regime_trending', 'regime_ranging', 'regime_volatile', 'regime_calm',
            'volatility_low', 'volatility_medium', 'volatility_high',
            'market_phase_pre_market', 'market_phase_open', 'market_phase_mid_day',
            'market_phase_close', 'market_phase_after_hours',
            'sector_tech', 'sector_finance', 'sector_healthcare', 'sector_energy', 'sector_other'
        ]
        
        # Numeric features
        numeric_features = [
            'volatility_zscore', 'correlation', 'liquidity_score', 'news_sentiment',
            'sector_dispersion_avg', 'market_cap_log', 'volume_ratio'
        ]
        
        # Time features
        time_features = [
            'hour_of_day', 'day_of_week', 'month', 'quarter'
        ]
        
        self.feature_names = model_features + condition_features + numeric_features + time_features
    
    def _create_feature_vector(self, model_name: str, conditions: Dict[str, Any]) -> List[float]:
        """Create feature vector for prediction."""
        try:
            features = [0.0] * len(self.feature_names)
            
            # Model name encoding
            if 'model_name_encoded' in self.feature_names:
                model_idx = self.feature_names.index('model_name_encoded')
                # Simple hash-based encoding
                features[model_idx] = hash(model_name) % 1000 / 1000.0
            
            # Market condition features
            regime = conditions.get('regime', 'unknown')
            volatility_level = conditions.get('volatility_level', 'medium')
            market_phase = conditions.get('market_phase', 'mid_day')
            sector = conditions.get('sector', 'other')
            
            # Set categorical features
            feature_mapping = {
                f'regime_{regime}': 1.0,
                f'volatility_{volatility_level}': 1.0,
                f'market_phase_{market_phase}': 1.0,
                f'sector_{sector}': 1.0
            }
            
            for feature_name, value in feature_mapping.items():
                if feature_name in self.feature_names:
                    idx = self.feature_names.index(feature_name)
                    features[idx] = value
            
            # Numeric features
            numeric_mapping = {
                'volatility_zscore': conditions.get('volatility_zscore', 0.0),
                'correlation': conditions.get('correlation', 0.0),
                'liquidity_score': conditions.get('liquidity_score', 0.5),
                'news_sentiment': conditions.get('news_sentiment', 0.0),
                'sector_dispersion_avg': conditions.get('sector_dispersion_avg', 0.0),
                'market_cap_log': conditions.get('market_cap_log', 0.0),
                'volume_ratio': conditions.get('volume_ratio', 1.0)
            }
            
            for feature_name, value in numeric_mapping.items():
                if feature_name in self.feature_names:
                    idx = self.feature_names.index(feature_name)
                    features[idx] = float(value)
            
            # Time features
            now = datetime.now()
            time_mapping = {
                'hour_of_day': now.hour / 24.0,
                'day_of_week': now.weekday() / 7.0,
                'month': now.month / 12.0,
                'quarter': ((now.month - 1) // 3 + 1) / 4.0
            }
            
            for feature_name, value in time_mapping.items():
                if feature_name in self.feature_names:
                    idx = self.feature_names.index(feature_name)
                    features[idx] = value
            
            return features
            
        except Exception as e:
            logger.error(f"Error creating feature vector: {e}")
            return [0.0] * len(self.feature_names)
    
    def _prepare_training_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], 
                                            Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare training data for model training."""
        try:
            if not self.training_data:
                return None, None, None, None
            
            # Convert to DataFrame for easier processing
            data_list = []
            for sample in self.training_data:
                data_list.append({
                    'features': sample['features'],
                    'accuracy': sample['accuracy'],
                    'brier_score': sample['brier_score'],
                    'execution_time': sample['execution_time']
                })
            
            df = pd.DataFrame(data_list)
            
            # Extract features and targets
            X = np.array([features for features in df['features']])
            y_accuracy = df['accuracy'].values
            y_brier = df['brier_score'].values
            y_execution = df['execution_time'].values
            
            return X, y_accuracy, y_brier, y_execution
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return None, None, None, None
    
    def _create_ensemble_model(self):
        """Create ensemble model for prediction."""
        # Use Random Forest as the base model
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    
    def _trigger_retraining(self):
        """Trigger model retraining."""
        try:
            logger.info("Triggering model retraining...")
            success = self.train_models()
            if success:
                logger.info("Model retraining completed successfully")
            else:
                logger.warning("Model retraining failed")
        except Exception as e:
            logger.error(f"Error in retraining trigger: {e}")
    
    def _save_model(self, target_name: str, model_info: PredictionModel):
        """Save trained model to disk."""
        try:
            model_path = os.path.join(self.model_storage_path, f"{target_name}_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model_info, f)
            logger.debug(f"Model saved: {model_path}")
        except Exception as e:
            logger.error(f"Error saving model {target_name}: {e}")
    
    def _load_model(self, target_name: str) -> Optional[PredictionModel]:
        """Load trained model from disk."""
        try:
            model_path = os.path.join(self.model_storage_path, f"{target_name}_model.pkl")
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading model {target_name}: {e}")
        return None
    
    def _get_default_prediction(self, model_name: str, conditions: Dict[str, Any]) -> PerformancePrediction:
        """Get default prediction when models are not trained."""
        return PerformancePrediction(
            model_name=model_name,
            predicted_accuracy=0.5,
            predicted_brier_score=1.0,
            predicted_execution_time=1.0,
            confidence_interval=(0.3, 0.7),
            feature_importance={},
            prediction_confidence=0.0,
            timestamp=datetime.now(),
            conditions=conditions.copy()
        )

# Global performance prediction system instance
_prediction_system = None

def get_prediction_system() -> PerformancePredictionSystem:
    """Get the global performance prediction system instance."""
    global _prediction_system
    if _prediction_system is None:
        _prediction_system = PerformancePredictionSystem()
    return _prediction_system