"""Advanced Ensemble Methods for Trading Predictions"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class EnsemblePrediction:
    """Result of ensemble prediction."""
    final_prediction: float
    confidence: float
    method_used: str
    individual_predictions: Dict[str, float]
    weights: Dict[str, float]
    uncertainty: float
    timestamp: datetime

class DynamicStackingEnsemble:
    """Dynamic stacking ensemble that adapts to market conditions."""
    
    def __init__(self, base_models: List[str], meta_models: List[str] = None):
        self.base_models = base_models
        self.meta_models = meta_models or ['linear', 'ridge', 'random_forest']
        self.meta_learner = None
        self.training_data = []
        self.performance_history = {}
        self.adaptation_rate = 0.1
        
    def add_training_sample(self, predictions: Dict[str, float], actual_outcome: float, 
                          market_conditions: Dict[str, Any]):
        """Add a training sample for meta-learning."""
        sample = {
            'predictions': predictions,
            'actual_outcome': actual_outcome,
            'market_conditions': market_conditions,
            'timestamp': datetime.now()
        }
        self.training_data.append(sample)
        
        # Update performance history
        for model_name, prediction in predictions.items():
            if model_name not in self.performance_history:
                self.performance_history[model_name] = []
            
            error = abs(prediction - actual_outcome)
            self.performance_history[model_name].append({
                'error': error,
                'timestamp': datetime.now(),
                'market_conditions': market_conditions
            })
    
    def _calculate_dynamic_weights(self, predictions: Dict[str, float], 
                                 market_conditions: Dict[str, Any]) -> Dict[str, float]:
        """Calculate dynamic weights based on recent performance and market conditions."""
        weights = {}
        
        # Base weights from recent performance
        for model_name in predictions.keys():
            if model_name in self.performance_history:
                recent_errors = [p['error'] for p in self.performance_history[model_name][-10:]]
                if recent_errors:
                    avg_error = np.mean(recent_errors)
                    # Lower error = higher weight
                    weights[model_name] = 1.0 / (1.0 + avg_error)
                else:
                    weights[model_name] = 1.0
            else:
                weights[model_name] = 1.0
        
        # Adjust weights based on market conditions
        regime = market_conditions.get('regime', 'unknown')
        volatility = market_conditions.get('volatility_regime', 'medium')
        
        # Regime-specific adjustments
        if regime == 'trending':
            # Favor trend-following models
            for model_name in weights.keys():
                if 'trend' in model_name.lower() or 'momentum' in model_name.lower():
                    weights[model_name] *= 1.2
        elif regime == 'ranging':
            # Favor mean-reversion models
            for model_name in weights.keys():
                if 'mean_reversion' in model_name.lower() or 'oscillator' in model_name.lower():
                    weights[model_name] *= 1.2
        elif regime == 'volatile':
            # Favor volatility models
            for model_name in weights.keys():
                if 'volatility' in model_name.lower() or 'garch' in model_name.lower():
                    weights[model_name] *= 1.2
        
        # Volatility adjustments
        if volatility == 'high':
            # Reduce weights for models that perform poorly in high volatility
            for model_name in weights.keys():
                if 'linear' in model_name.lower() or 'simple' in model_name.lower():
                    weights[model_name] *= 0.8
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def predict(self, predictions: Dict[str, float], market_conditions: Dict[str, Any]) -> EnsemblePrediction:
        """Make ensemble prediction using dynamic stacking."""
        if not predictions:
            return EnsemblePrediction(
                final_prediction=0.5,
                confidence=0.0,
                method_used='dynamic_stacking',
                individual_predictions={},
                weights={},
                uncertainty=1.0,
                timestamp=datetime.now()
            )
        
        # Calculate dynamic weights
        weights = self._calculate_dynamic_weights(predictions, market_conditions)
        
        # Calculate weighted prediction
        weighted_sum = sum(pred * weights.get(model, 0) for model, pred in predictions.items())
        final_prediction = weighted_sum
        
        # Calculate confidence based on agreement and weights
        if len(predictions) > 1:
            pred_values = list(predictions.values())
            agreement = 1.0 - np.std(pred_values) / (np.mean(pred_values) + 1e-8)
            confidence = agreement * np.mean(list(weights.values()))
        else:
            confidence = 0.5
        
        # Calculate uncertainty
        uncertainty = 1.0 - confidence
        
        return EnsemblePrediction(
            final_prediction=final_prediction,
            confidence=confidence,
            method_used='dynamic_stacking',
            individual_predictions=predictions,
            weights=weights,
            uncertainty=uncertainty,
            timestamp=datetime.now()
        )

class HierarchicalEnsemble:
    """Hierarchical ensemble with multiple levels of combination."""
    
    def __init__(self):
        self.levels = {
            'level_1': ['technical', 'fundamental', 'sentiment'],
            'level_2': ['trend', 'mean_reversion', 'volatility'],
            'level_3': ['final']
        }
        self.level_weights = {
            'level_1': 0.4,
            'level_2': 0.4,
            'level_3': 0.2
        }
        self.performance_tracking = {}
    
    def _group_predictions_by_category(self, predictions: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Group predictions by model category."""
        grouped = {
            'technical': {},
            'fundamental': {},
            'sentiment': {},
            'trend': {},
            'mean_reversion': {},
            'volatility': {},
            'other': {}
        }
        
        for model_name, prediction in predictions.items():
            model_lower = model_name.lower()
            
            if any(keyword in model_lower for keyword in ['rsi', 'macd', 'bollinger', 'technical']):
                grouped['technical'][model_name] = prediction
            elif any(keyword in model_lower for keyword in ['fundamental', 'pe', 'pb', 'earnings']):
                grouped['fundamental'][model_name] = prediction
            elif any(keyword in model_lower for keyword in ['sentiment', 'news', 'social']):
                grouped['sentiment'][model_name] = prediction
            elif any(keyword in model_lower for keyword in ['trend', 'momentum', 'moving']):
                grouped['trend'][model_name] = prediction
            elif any(keyword in model_lower for keyword in ['mean_reversion', 'oscillator', 'range']):
                grouped['mean_reversion'][model_name] = prediction
            elif any(keyword in model_lower for keyword in ['volatility', 'garch', 'atr']):
                grouped['volatility'][model_name] = prediction
            else:
                grouped['other'][model_name] = prediction
        
        return grouped
    
    def _combine_level_predictions(self, level_predictions: Dict[str, float]) -> float:
        """Combine predictions within a level."""
        if not level_predictions:
            return 0.5
        
        # Use weighted average with equal weights for now
        return np.mean(list(level_predictions.values()))
    
    def predict(self, predictions: Dict[str, float], market_conditions: Dict[str, Any]) -> EnsemblePrediction:
        """Make hierarchical ensemble prediction."""
        if not predictions:
            return EnsemblePrediction(
                final_prediction=0.5,
                confidence=0.0,
                method_used='hierarchical',
                individual_predictions={},
                weights={},
                uncertainty=1.0,
                timestamp=datetime.now()
            )
        
        # Group predictions by category
        grouped_predictions = self._group_predictions_by_category(predictions)
        
        # Combine predictions at each level
        level_combinations = {}
        for level, categories in self.levels.items():
            level_preds = []
            for category in categories:
                if category in grouped_predictions and grouped_predictions[category]:
                    level_pred = self._combine_level_predictions(grouped_predictions[category])
                    level_preds.append(level_pred)
            
            if level_preds:
                level_combinations[level] = np.mean(level_preds)
            else:
                level_combinations[level] = 0.5
        
        # Combine levels with weights
        final_prediction = sum(
            level_combinations[level] * self.level_weights[level]
            for level in self.levels.keys()
            if level in level_combinations
        )
        
        # Calculate confidence based on level agreement
        level_values = list(level_combinations.values())
        if len(level_values) > 1:
            agreement = 1.0 - np.std(level_values) / (np.mean(level_values) + 1e-8)
            confidence = max(0.0, min(1.0, agreement))
        else:
            confidence = 0.5
        
        # Create weights dictionary for transparency
        weights = {}
        for level, pred in level_combinations.items():
            weights[f"{level}_prediction"] = pred
        
        return EnsemblePrediction(
            final_prediction=final_prediction,
            confidence=confidence,
            method_used='hierarchical',
            individual_predictions=predictions,
            weights=weights,
            uncertainty=1.0 - confidence,
            timestamp=datetime.now()
        )

class AdaptiveWeightingEnsemble:
    """Ensemble with adaptive weighting based on real-time performance."""
    
    def __init__(self, adaptation_window: int = 20):
        self.adaptation_window = adaptation_window
        self.performance_buffer = {}
        self.weights = {}
        self.initial_weight = 1.0
        self.learning_rate = 0.1
        
    def update_performance(self, model_name: str, prediction: float, actual_outcome: float):
        """Update performance tracking for a model."""
        if model_name not in self.performance_buffer:
            self.performance_buffer[model_name] = []
        
        error = abs(prediction - actual_outcome)
        self.performance_buffer[model_name].append({
            'error': error,
            'timestamp': datetime.now()
        })
        
        # Keep only recent performance data
        if len(self.performance_buffer[model_name]) > self.adaptation_window:
            self.performance_buffer[model_name] = self.performance_buffer[model_name][-self.adaptation_window:]
    
    def _calculate_adaptive_weights(self, model_names: List[str]) -> Dict[str, float]:
        """Calculate adaptive weights based on recent performance."""
        weights = {}
        
        for model_name in model_names:
            if model_name in self.performance_buffer and self.performance_buffer[model_name]:
                # Calculate recent average error
                recent_errors = [p['error'] for p in self.performance_buffer[model_name]]
                avg_error = np.mean(recent_errors)
                
                # Convert error to weight (lower error = higher weight)
                weight = 1.0 / (1.0 + avg_error)
            else:
                # Use initial weight for new models
                weight = self.initial_weight
            
            weights[model_name] = weight
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def predict(self, predictions: Dict[str, float], market_conditions: Dict[str, Any]) -> EnsemblePrediction:
        """Make adaptive ensemble prediction."""
        if not predictions:
            return EnsemblePrediction(
                final_prediction=0.5,
                confidence=0.0,
                method_used='adaptive_weighting',
                individual_predictions={},
                weights={},
                uncertainty=1.0,
                timestamp=datetime.now()
            )
        
        # Calculate adaptive weights
        weights = self._calculate_adaptive_weights(list(predictions.keys()))
        
        # Calculate weighted prediction
        weighted_sum = sum(pred * weights.get(model, 0) for model, pred in predictions.items())
        final_prediction = weighted_sum
        
        # Calculate confidence based on weight distribution and prediction agreement
        if len(predictions) > 1:
            pred_values = list(predictions.values())
            agreement = 1.0 - np.std(pred_values) / (np.mean(pred_values) + 1e-8)
            weight_confidence = np.mean(list(weights.values()))
            confidence = (agreement + weight_confidence) / 2.0
        else:
            confidence = 0.5
        
        return EnsemblePrediction(
            final_prediction=final_prediction,
            confidence=confidence,
            method_used='adaptive_weighting',
            individual_predictions=predictions,
            weights=weights,
            uncertainty=1.0 - confidence,
            timestamp=datetime.now()
        )

class AdvancedEnsembleManager:
    """Manager for all advanced ensemble methods."""
    
    def __init__(self):
        self.ensembles = {
            'dynamic_stacking': DynamicStackingEnsemble(['model1', 'model2', 'model3']),
            'hierarchical': HierarchicalEnsemble(),
            'adaptive_weighting': AdaptiveWeightingEnsemble()
        }
        self.ensemble_weights = {
            'dynamic_stacking': 0.4,
            'hierarchical': 0.3,
            'adaptive_weighting': 0.3
        }
        self.performance_tracking = {}
        
    def add_training_sample(self, predictions: Dict[str, float], actual_outcome: float,
                          market_conditions: Dict[str, Any]):
        """Add training sample to all ensembles."""
        for ensemble in self.ensembles.values():
            if hasattr(ensemble, 'add_training_sample'):
                ensemble.add_training_sample(predictions, actual_outcome, market_conditions)
    
    def update_performance(self, model_name: str, prediction: float, actual_outcome: float):
        """Update performance for adaptive weighting ensemble."""
        if 'adaptive_weighting' in self.ensembles:
            self.ensembles['adaptive_weighting'].update_performance(model_name, prediction, actual_outcome)
    
    def predict_ensemble(self, predictions: Dict[str, float], 
                        market_conditions: Dict[str, Any],
                        method: str = 'combined') -> EnsemblePrediction:
        """Make ensemble prediction using specified method."""
        if method == 'combined':
            # Get predictions from all ensembles
            ensemble_predictions = {}
            for name, ensemble in self.ensembles.items():
                result = ensemble.predict(predictions, market_conditions)
                ensemble_predictions[name] = result.final_prediction
            
            # Combine ensemble predictions
            if ensemble_predictions:
                final_prediction = sum(
                    pred * self.ensemble_weights.get(name, 0)
                    for name, pred in ensemble_predictions.items()
                )
                
                # Calculate overall confidence
                confidences = [
                    ensemble.predict(predictions, market_conditions).confidence
                    for ensemble in self.ensembles.values()
                ]
                confidence = np.mean(confidences) if confidences else 0.5
            else:
                final_prediction = 0.5
                confidence = 0.0
            
            return EnsemblePrediction(
                final_prediction=final_prediction,
                confidence=confidence,
                method_used='combined',
                individual_predictions=predictions,
                weights=self.ensemble_weights,
                uncertainty=1.0 - confidence,
                timestamp=datetime.now()
            )
        
        elif method in self.ensembles:
            return self.ensembles[method].predict(predictions, market_conditions)
        else:
            # Fallback to simple average
            if predictions:
                final_prediction = np.mean(list(predictions.values()))
                confidence = 0.5
            else:
                final_prediction = 0.5
                confidence = 0.0
            
            return EnsemblePrediction(
                final_prediction=final_prediction,
                confidence=confidence,
                method_used='simple_average',
                individual_predictions=predictions,
                weights={},
                uncertainty=1.0 - confidence,
                timestamp=datetime.now()
            )
    
    def get_ensemble_statistics(self) -> Dict[str, Any]:
        """Get statistics about ensemble performance."""
        stats = {
            'total_ensembles': len(self.ensembles),
            'ensemble_methods': list(self.ensembles.keys()),
            'ensemble_weights': self.ensemble_weights,
            'performance_tracking': {}
        }
        
        for name, ensemble in self.ensembles.items():
            if hasattr(ensemble, 'performance_history'):
                stats['performance_tracking'][name] = {
                    'total_samples': len(ensemble.performance_history),
                    'models_tracked': len(ensemble.performance_history)
                }
            elif hasattr(ensemble, 'performance_buffer'):
                stats['performance_tracking'][name] = {
                    'total_samples': sum(len(buffer) for buffer in ensemble.performance_buffer.values()),
                    'models_tracked': len(ensemble.performance_buffer)
                }
        
        return stats

# Global instance
_ensemble_manager = None

def get_ensemble_manager() -> AdvancedEnsembleManager:
    """Get the global ensemble manager instance."""
    global _ensemble_manager
    if _ensemble_manager is None:
        _ensemble_manager = AdvancedEnsembleManager()
    return _ensemble_manager