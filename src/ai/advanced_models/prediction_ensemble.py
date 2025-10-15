"""
Prediction Ensemble System for Advanced AI Models

This module provides advanced ensemble methods for combining predictions
from multiple AI models with sophisticated weighting and selection strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import logging
from datetime import datetime, timedelta
from collections import defaultdict, deque
import warnings

# Machine learning libraries
try:
    from sklearn.ensemble import VotingClassifier, VotingRegressor
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("Scikit-learn not available. Some ensemble features will be limited.")

logger = logging.getLogger(__name__)

class PredictionEnsemble:
    """
    Advanced prediction ensemble system for combining multiple model predictions.
    """
    
    def __init__(
        self,
        ensemble_name: str = "prediction_ensemble",
        ensemble_methods: List[str] = None,
        performance_window: int = 100
    ):
        """
        Initialize prediction ensemble.
        
        Args:
            ensemble_name: Name for the ensemble
            ensemble_methods: List of ensemble methods to use
            performance_window: Window size for performance tracking
        """
        self.ensemble_name = ensemble_name
        self.ensemble_methods = ensemble_methods or [
            'weighted_average',
            'stacking',
            'bayesian_model_averaging',
            'dynamic_weighting',
            'hierarchical_ensemble'
        ]
        self.performance_window = performance_window
        
        # Model registry
        self.models = {}
        self.model_weights = {}
        self.model_performance = defaultdict(lambda: deque(maxlen=performance_window))
        
        # Ensemble history
        self.ensemble_history = deque(maxlen=1000)
        
        # Performance metrics
        self.ensemble_metrics = {
            'total_predictions': 0,
            'successful_predictions': 0,
            'ensemble_accuracy': 0.0,
            'individual_model_performance': {},
            'ensemble_method_performance': defaultdict(list)
        }
        
        logger.info(f"Prediction Ensemble initialized: {ensemble_name}")
    
    def add_model(
        self,
        model_name: str,
        model: Any,
        initial_weight: float = 1.0,
        model_type: str = 'regression'
    ) -> None:
        """
        Add a model to the ensemble.
        
        Args:
            model_name: Name for the model
            model: Model object
            initial_weight: Initial weight for the model
            model_type: Type of model ('regression' or 'classification')
        """
        self.models[model_name] = {
            'model': model,
            'type': model_type,
            'weight': initial_weight,
            'added_at': datetime.now()
        }
        self.model_weights[model_name] = initial_weight
        
        logger.info(f"Added model '{model_name}' to ensemble with weight {initial_weight}")
    
    def remove_model(self, model_name: str) -> None:
        """Remove a model from the ensemble."""
        if model_name in self.models:
            del self.models[model_name]
            del self.model_weights[model_name]
            if model_name in self.model_performance:
                del self.model_performance[model_name]
            logger.info(f"Removed model '{model_name}' from ensemble")
        else:
            logger.warning(f"Model '{model_name}' not found in ensemble")
    
    def update_model_weight(
        self,
        model_name: str,
        new_weight: float,
        update_method: str = 'direct'
    ) -> None:
        """
        Update model weight.
        
        Args:
            model_name: Name of the model
            new_weight: New weight value
            update_method: Method for updating ('direct', 'performance_based', 'adaptive')
        """
        if model_name not in self.models:
            logger.warning(f"Model '{model_name}' not found in ensemble")
            return
        
        if update_method == 'direct':
            self.model_weights[model_name] = new_weight
            self.models[model_name]['weight'] = new_weight
        
        elif update_method == 'performance_based':
            # Update weight based on recent performance
            if model_name in self.model_performance:
                recent_performance = list(self.model_performance[model_name])[-10:]  # Last 10 predictions
                if recent_performance:
                    avg_performance = np.mean(recent_performance)
                    # Weight is proportional to performance
                    self.model_weights[model_name] = max(0.1, avg_performance)
                    self.models[model_name]['weight'] = self.model_weights[model_name]
        
        elif update_method == 'adaptive':
            # Adaptive weighting based on prediction confidence and historical performance
            if model_name in self.model_performance:
                recent_performance = list(self.model_performance[model_name])[-20:]  # Last 20 predictions
                if recent_performance:
                    performance_trend = np.polyfit(range(len(recent_performance)), recent_performance, 1)[0]
                    # Increase weight if performance is improving
                    weight_adjustment = 1.0 + (performance_trend * 0.1)
                    new_weight = self.model_weights[model_name] * weight_adjustment
                    self.model_weights[model_name] = max(0.1, min(2.0, new_weight))
                    self.models[model_name]['weight'] = self.model_weights[model_name]
        
        logger.info(f"Updated model '{model_name}' weight to {self.model_weights[model_name]} using {update_method}")
    
    def predict_ensemble(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        method: str = 'weighted_average',
        return_individual: bool = False
    ) -> Dict[str, Any]:
        """
        Make ensemble prediction.
        
        Args:
            X: Input features
            method: Ensemble method to use
            return_individual: Whether to return individual model predictions
            
        Returns:
            Ensemble prediction results
        """
        start_time = datetime.now()
        
        if not self.models:
            return {'error': 'No models in ensemble'}
        
        results = {
            'ensemble_name': self.ensemble_name,
            'method': method,
            'timestamp': datetime.now().isoformat(),
            'individual_predictions': {},
            'ensemble_prediction': None,
            'confidence': 0.0,
            'model_weights': self.model_weights.copy()
        }
        
        try:
            # Get individual predictions
            individual_predictions = {}
            for model_name, model_info in self.models.items():
                try:
                    model = model_info['model']
                    prediction = self._get_model_prediction(model, X, model_info['type'])
                    individual_predictions[model_name] = prediction
                    results['individual_predictions'][model_name] = prediction
                except Exception as e:
                    logger.warning(f"Prediction failed for model '{model_name}': {e}")
                    individual_predictions[model_name] = None
            
            # Filter out failed predictions
            valid_predictions = {k: v for k, v in individual_predictions.items() if v is not None}
            
            if not valid_predictions:
                return {'error': 'All model predictions failed'}
            
            # Create ensemble prediction
            if method == 'weighted_average':
                ensemble_result = self._weighted_average_ensemble(valid_predictions)
            elif method == 'majority_vote':
                ensemble_result = self._majority_vote_ensemble(valid_predictions)
            elif method == 'stacking':
                ensemble_result = self._stacking_ensemble(valid_predictions, X)
            elif method == 'bayesian_model_averaging':
                ensemble_result = self._bayesian_model_averaging(valid_predictions)
            elif method == 'dynamic_weighting':
                ensemble_result = self._dynamic_weighting_ensemble(valid_predictions)
            elif method == 'hierarchical_ensemble':
                ensemble_result = self._hierarchical_ensemble(valid_predictions)
            else:
                return {'error': f'Unknown ensemble method: {method}'}
            
            results['ensemble_prediction'] = ensemble_result['prediction']
            results['confidence'] = ensemble_result.get('confidence', 0.0)
            results['method_details'] = ensemble_result.get('details', {})
            
            # Update performance metrics
            self.ensemble_metrics['total_predictions'] += 1
            self.ensemble_metrics['successful_predictions'] += 1
            
            # Store in history
            results['execution_time'] = (datetime.now() - start_time).total_seconds()
            self.ensemble_history.append(results)
            
            logger.info(f"Ensemble prediction completed using {method}")
            
        except Exception as e:
            logger.error(f"Ensemble prediction error: {e}")
            results['error'] = str(e)
            self.ensemble_metrics['total_predictions'] += 1
        
        return results
    
    def _get_model_prediction(
        self,
        model: Any,
        X: Union[np.ndarray, pd.DataFrame],
        model_type: str
    ) -> Optional[Union[float, int, str, np.ndarray]]:
        """Get prediction from a single model."""
        try:
            # Handle different model types
            if hasattr(model, 'predict'):
                prediction = model.predict(X)
            elif hasattr(model, 'forward'):
                # PyTorch model
                import torch
                if isinstance(X, pd.DataFrame):
                    X = torch.tensor(X.values, dtype=torch.float32)
                elif isinstance(X, np.ndarray):
                    X = torch.tensor(X, dtype=torch.float32)
                model.eval()
                with torch.no_grad():
                    prediction = model(X).numpy()
            elif callable(model):
                prediction = model(X)
            else:
                logger.warning(f"Unknown model type: {type(model)}")
                return None
            
            # Handle different prediction formats
            if isinstance(prediction, np.ndarray):
                if prediction.ndim > 1:
                    prediction = prediction.flatten()
                if len(prediction) == 1:
                    prediction = prediction[0]
            
            return prediction
            
        except Exception as e:
            logger.warning(f"Model prediction error: {e}")
            return None
    
    def _weighted_average_ensemble(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Create weighted average ensemble prediction."""
        try:
            # Extract numeric predictions
            numeric_predictions = {}
            total_weight = 0.0
            
            for model_name, prediction in predictions.items():
                numeric_value = self._extract_numeric_value(prediction)
                if numeric_value is not None:
                    weight = self.model_weights.get(model_name, 1.0)
                    numeric_predictions[model_name] = {
                        'value': numeric_value,
                        'weight': weight
                    }
                    total_weight += weight
            
            if not numeric_predictions:
                return {'prediction': 0.0, 'confidence': 0.0}
            
            # Calculate weighted average
            weighted_sum = sum(pred['value'] * pred['weight'] for pred in numeric_predictions.values())
            ensemble_prediction = weighted_sum / total_weight if total_weight > 0 else 0.0
            
            # Calculate confidence based on weight distribution
            max_weight = max(pred['weight'] for pred in numeric_predictions.values())
            confidence = min(max_weight / total_weight, 1.0) if total_weight > 0 else 0.0
            
            return {
                'prediction': ensemble_prediction,
                'confidence': confidence,
                'details': {
                    'total_weight': total_weight,
                    'individual_contributions': {
                        name: pred['value'] * pred['weight'] / total_weight
                        for name, pred in numeric_predictions.items()
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Weighted average ensemble error: {e}")
            return {'prediction': 0.0, 'confidence': 0.0, 'error': str(e)}
    
    def _stacking_ensemble(self, predictions: Dict[str, Any], X: Union[np.ndarray, pd.DataFrame]) -> Dict[str, Any]:
        """Create stacking ensemble prediction."""
        try:
            # For now, use weighted average as a simple stacking approach
            # In a full implementation, this would use a meta-learner
            return self._weighted_average_ensemble(predictions)
            
        except Exception as e:
            logger.error(f"Stacking ensemble error: {e}")
            return {'prediction': 0.0, 'confidence': 0.0, 'error': str(e)}
    
    def _bayesian_model_averaging(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Create Bayesian model averaging ensemble prediction."""
        try:
            # Simple Bayesian model averaging using model weights as priors
            # In a full implementation, this would use proper Bayesian inference
            
            numeric_predictions = {}
            total_weight = 0.0
            
            for model_name, prediction in predictions.items():
                numeric_value = self._extract_numeric_value(prediction)
                if numeric_value is not None:
                    weight = self.model_weights.get(model_name, 1.0)
                    numeric_predictions[model_name] = {
                        'value': numeric_value,
                        'weight': weight
                    }
                    total_weight += weight
            
            if not numeric_predictions:
                return {'prediction': 0.0, 'confidence': 0.0}
            
            # Bayesian weighted average
            weighted_sum = sum(pred['value'] * pred['weight'] for pred in numeric_predictions.values())
            ensemble_prediction = weighted_sum / total_weight if total_weight > 0 else 0.0
            
            # Calculate uncertainty (simplified)
            variance = sum(
                pred['weight'] * (pred['value'] - ensemble_prediction) ** 2
                for pred in numeric_predictions.values()
            ) / total_weight if total_weight > 0 else 0.0
            
            confidence = 1.0 / (1.0 + variance) if variance > 0 else 1.0
            
            return {
                'prediction': ensemble_prediction,
                'confidence': confidence,
                'details': {
                    'variance': variance,
                    'uncertainty': np.sqrt(variance) if variance > 0 else 0.0
                }
            }
            
        except Exception as e:
            logger.error(f"Bayesian model averaging error: {e}")
            return {'prediction': 0.0, 'confidence': 0.0, 'error': str(e)}
    
    def _dynamic_weighting_ensemble(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Create dynamic weighting ensemble prediction."""
        try:
            # Dynamic weighting based on recent performance
            numeric_predictions = {}
            dynamic_weights = {}
            
            for model_name, prediction in predictions.items():
                numeric_value = self._extract_numeric_value(prediction)
                if numeric_value is not None:
                    # Calculate dynamic weight based on recent performance
                    if model_name in self.model_performance:
                        recent_performance = list(self.model_performance[model_name])[-10:]
                        if recent_performance:
                            performance_weight = np.mean(recent_performance)
                        else:
                            performance_weight = 1.0
                    else:
                        performance_weight = 1.0
                    
                    base_weight = self.model_weights.get(model_name, 1.0)
                    dynamic_weight = base_weight * performance_weight
                    
                    numeric_predictions[model_name] = {
                        'value': numeric_value,
                        'weight': dynamic_weight
                    }
                    dynamic_weights[model_name] = dynamic_weight
            
            if not numeric_predictions:
                return {'prediction': 0.0, 'confidence': 0.0}
            
            # Calculate weighted average with dynamic weights
            total_weight = sum(pred['weight'] for pred in numeric_predictions.values())
            weighted_sum = sum(pred['value'] * pred['weight'] for pred in numeric_predictions.values())
            ensemble_prediction = weighted_sum / total_weight if total_weight > 0 else 0.0
            
            # Confidence based on weight consistency
            weight_variance = np.var(list(dynamic_weights.values()))
            confidence = 1.0 / (1.0 + weight_variance) if weight_variance > 0 else 1.0
            
            return {
                'prediction': ensemble_prediction,
                'confidence': confidence,
                'details': {
                    'dynamic_weights': dynamic_weights,
                    'weight_variance': weight_variance
                }
            }
            
        except Exception as e:
            logger.error(f"Dynamic weighting ensemble error: {e}")
            return {'prediction': 0.0, 'confidence': 0.0, 'error': str(e)}
    
    def _hierarchical_ensemble(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Create hierarchical ensemble prediction."""
        try:
            # Hierarchical ensemble with multiple levels
            # Level 1: Group models by type
            model_groups = defaultdict(list)
            for model_name, prediction in predictions.items():
                model_type = self.models[model_name]['type']
                numeric_value = self._extract_numeric_value(prediction)
                if numeric_value is not None:
                    model_groups[model_type].append({
                        'name': model_name,
                        'value': numeric_value,
                        'weight': self.model_weights.get(model_name, 1.0)
                    })
            
            # Level 2: Create group predictions
            group_predictions = {}
            for group_type, models in model_groups.items():
                if models:
                    total_weight = sum(model['weight'] for model in models)
                    weighted_sum = sum(model['value'] * model['weight'] for model in models)
                    group_prediction = weighted_sum / total_weight if total_weight > 0 else 0.0
                    group_predictions[group_type] = {
                        'prediction': group_prediction,
                        'weight': total_weight,
                        'model_count': len(models)
                    }
            
            # Level 3: Combine group predictions
            if group_predictions:
                total_group_weight = sum(group['weight'] for group in group_predictions.values())
                weighted_sum = sum(
                    group['prediction'] * group['weight']
                    for group in group_predictions.values()
                )
                ensemble_prediction = weighted_sum / total_group_weight if total_group_weight > 0 else 0.0
                
                # Confidence based on group diversity
                group_count = len(group_predictions)
                confidence = min(group_count / 3.0, 1.0)  # More groups = higher confidence
            else:
                ensemble_prediction = 0.0
                confidence = 0.0
            
            return {
                'prediction': ensemble_prediction,
                'confidence': confidence,
                'details': {
                    'group_predictions': group_predictions,
                    'hierarchy_levels': 3
                }
            }
            
        except Exception as e:
            logger.error(f"Hierarchical ensemble error: {e}")
            return {'prediction': 0.0, 'confidence': 0.0, 'error': str(e)}
    
    def _extract_numeric_value(self, prediction: Any) -> Optional[float]:
        """Extract numeric value from prediction."""
        try:
            if isinstance(prediction, (int, float)):
                return float(prediction)
            elif isinstance(prediction, np.ndarray):
                if prediction.size == 1:
                    return float(prediction.item())
                else:
                    return float(np.mean(prediction))
            elif isinstance(prediction, dict):
                # Try common keys
                for key in ['prediction', 'value', 'score', 'probability']:
                    if key in prediction and isinstance(prediction[key], (int, float)):
                        return float(prediction[key])
            return None
        except:
            return None
    
    def update_model_performance(
        self,
        model_name: str,
        actual_value: Union[float, int],
        predicted_value: Union[float, int],
        metric: str = 'accuracy'
    ) -> None:
        """
        Update model performance for adaptive weighting.
        
        Args:
            model_name: Name of the model
            actual_value: Actual target value
            predicted_value: Predicted value
            metric: Performance metric to use
        """
        if model_name not in self.models:
            logger.warning(f"Model '{model_name}' not found in ensemble")
            return
        
        try:
            if metric == 'accuracy':
                # For classification
                performance = 1.0 if actual_value == predicted_value else 0.0
            elif metric == 'mse':
                # For regression (inverted MSE for higher = better)
                mse = (actual_value - predicted_value) ** 2
                performance = 1.0 / (1.0 + mse)
            elif metric == 'mae':
                # For regression (inverted MAE for higher = better)
                mae = abs(actual_value - predicted_value)
                performance = 1.0 / (1.0 + mae)
            else:
                # Default: simple accuracy
                performance = 1.0 if actual_value == predicted_value else 0.0
            
            self.model_performance[model_name].append(performance)
            
            # Update ensemble metrics
            if model_name not in self.ensemble_metrics['individual_model_performance']:
                self.ensemble_metrics['individual_model_performance'][model_name] = []
            self.ensemble_metrics['individual_model_performance'][model_name].append(performance)
            
        except Exception as e:
            logger.error(f"Error updating model performance: {e}")
    
    def get_ensemble_statistics(self) -> Dict[str, Any]:
        """Get ensemble performance statistics."""
        return {
            'ensemble_name': self.ensemble_name,
            'total_models': len(self.models),
            'model_names': list(self.models.keys()),
            'model_weights': self.model_weights.copy(),
            'performance_metrics': self.ensemble_metrics.copy(),
            'model_performance_summary': {
                name: {
                    'recent_performance': list(perf)[-10:] if perf else [],
                    'average_performance': np.mean(list(perf)) if perf else 0.0,
                    'performance_count': len(perf)
                }
                for name, perf in self.model_performance.items()
            },
            'ensemble_history_size': len(self.ensemble_history)
        }


class EnsembleManager:
    """
    Manager for multiple prediction ensembles.
    """
    
    def __init__(self, manager_name: str = "ensemble_manager"):
        """
        Initialize ensemble manager.
        
        Args:
            manager_name: Name for the manager
        """
        self.manager_name = manager_name
        self.ensembles = {}
        
        logger.info(f"Ensemble Manager initialized: {manager_name}")
    
    def create_ensemble(
        self,
        ensemble_name: str,
        ensemble_methods: Optional[List[str]] = None,
        performance_window: int = 100
    ) -> PredictionEnsemble:
        """Create a new prediction ensemble."""
        ensemble = PredictionEnsemble(
            ensemble_name=ensemble_name,
            ensemble_methods=ensemble_methods,
            performance_window=performance_window
        )
        self.ensembles[ensemble_name] = ensemble
        
        logger.info(f"Created ensemble '{ensemble_name}'")
        return ensemble
    
    def get_ensemble(self, ensemble_name: str) -> Optional[PredictionEnsemble]:
        """Get an existing ensemble."""
        return self.ensembles.get(ensemble_name)
    
    def remove_ensemble(self, ensemble_name: str) -> None:
        """Remove an ensemble."""
        if ensemble_name in self.ensembles:
            del self.ensembles[ensemble_name]
            logger.info(f"Removed ensemble '{ensemble_name}'")
        else:
            logger.warning(f"Ensemble '{ensemble_name}' not found")
    
    def list_ensembles(self) -> List[str]:
        """List all ensemble names."""
        return list(self.ensembles.keys())
    
    def get_manager_statistics(self) -> Dict[str, Any]:
        """Get ensemble manager statistics."""
        return {
            'manager_name': self.manager_name,
            'total_ensembles': len(self.ensembles),
            'ensemble_names': list(self.ensembles.keys()),
            'ensemble_details': {
                name: {
                    'total_models': len(ensemble.models),
                    'model_names': list(ensemble.models.keys()),
                    'total_predictions': ensemble.ensemble_metrics['total_predictions']
                }
                for name, ensemble in self.ensembles.items()
            }
        }
    
    def _majority_vote_ensemble(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Ensemble using majority voting."""
        from datetime import datetime
        from collections import Counter
        
        try:
            # Extract numeric predictions
            numeric_predictions = []
            for model_name, prediction in predictions.items():
                numeric_value = self._extract_numeric_value(prediction)
                if numeric_value is not None:
                    numeric_predictions.append(numeric_value)
            
            if not numeric_predictions:
                return {
                    'prediction': 0.0,
                    'confidence': 0.0,
                    'error': 'No valid numeric predictions'
                }
            
            # For continuous predictions, use median as "majority"
            numeric_predictions.sort()
            n = len(numeric_predictions)
            if n % 2 == 0:
                majority_prediction = (numeric_predictions[n//2-1] + numeric_predictions[n//2]) / 2
            else:
                majority_prediction = numeric_predictions[n//2]
            
            # Calculate confidence based on agreement
            # For continuous values, use inverse of standard deviation
            import numpy as np
            std_dev = np.std(numeric_predictions)
            confidence_score = max(0.0, 1.0 - std_dev) if std_dev > 0 else 1.0
            
            # Calculate agreement percentage
            agreement_threshold = 0.1  # Within 10% of majority
            agreement_count = sum(1 for pred in numeric_predictions 
                                if abs(pred - majority_prediction) <= agreement_threshold)
            agreement_percentage = agreement_count / len(numeric_predictions)
            
            return {
                'prediction': majority_prediction,
                'confidence': confidence_score * agreement_percentage,
                'details': {
                    'total_predictions': len(numeric_predictions),
                    'agreement_percentage': agreement_percentage,
                    'standard_deviation': std_dev
                }
            }
            
        except Exception as e:
            logger.error(f"Majority vote ensemble error: {e}")
            return {
                'prediction': 0.0,
                'confidence': 0.0,
                'error': str(e)
            }

