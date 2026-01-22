"""
Adaptive Ensemble Weights Module
===============================

This module implements dynamic model weighting based on rolling performance metrics,
specifically Brier scores, to adaptively weight ensemble models based on their
recent performance.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


@dataclass
class ModelPerformance:
    """Represents model performance metrics."""
    model_name: str
    timestamp: datetime
    brier_score: float
    accuracy: float
    n_predictions: int
    weight: float
    window_start: datetime
    window_end: datetime


@dataclass
class EnsembleWeights:
    """Represents ensemble weights for all models."""
    timestamp: datetime
    weights: Dict[str, float]
    total_models: int
    performance_summary: Dict[str, ModelPerformance]
    weight_entropy: float  # Measure of weight distribution


class AdaptiveWeightManager:
    """Manages adaptive ensemble weights based on model performance."""
    
    def __init__(self, window_size_days: int = 7, min_predictions: int = 10, 
                 epsilon: float = 0.01, weight_update_frequency_hours: int = 24):
        """
        Initialize adaptive weight manager.
        
        Args:
            window_size_days: Rolling window size for performance calculation
            min_predictions: Minimum predictions needed for reliable performance metrics
            epsilon: Small value to prevent division by zero in weight calculation
            weight_update_frequency_hours: How often to update weights
        """
        self.window_size_days = window_size_days
        self.min_predictions = min_predictions
        self.epsilon = epsilon
        self.weight_update_frequency_hours = weight_update_frequency_hours
        
        # Performance tracking
        self.model_predictions: List[Dict[str, Any]] = []
        self.model_performance: Dict[str, ModelPerformance] = {}
        self.ensemble_weights: Optional[EnsembleWeights] = None
        
        # Weight calculation parameters
        self.weight_smoothing_factor = 0.1  # How much to smooth weight changes
        self.max_weight_change = 0.2  # Maximum weight change per update
        
        logger.info(f"Adaptive Weight Manager initialized with {window_size_days} day window")
    
    def add_prediction(self, model_name: str, symbol: str, predicted_probability: float,
                      actual_outcome: str, prediction_date: datetime, 
                      trade_id: str = None) -> None:
        """
        Add a model prediction for performance tracking.
        
        Args:
            model_name: Name of the model
            symbol: Symbol being predicted
            predicted_probability: Model's predicted probability (0-1)
            actual_outcome: Actual outcome ("WIN", "LOSS", "PENDING")
            prediction_date: Date of the prediction
            trade_id: Optional trade ID for tracking
        """
        prediction = {
            "model_name": model_name,
            "symbol": symbol,
            "predicted_probability": predicted_probability,
            "actual_outcome": actual_outcome,
            "prediction_date": prediction_date,
            "trade_id": trade_id,
            "timestamp": datetime.now()
        }
        
        self.model_predictions.append(prediction)
        
        # Update performance for this model
        self._update_model_performance(model_name)
        
        logger.debug(f"Added prediction for {model_name}: {predicted_probability:.3f} -> {actual_outcome}")
    
    def _update_model_performance(self, model_name: str) -> None:
        """Update performance metrics for a specific model."""
        # Get predictions for this model in the current window
        cutoff_date = datetime.now() - timedelta(days=self.window_size_days)
        
        model_predictions = [
            p for p in self.model_predictions
            if p["model_name"] == model_name and 
               p["prediction_date"] >= cutoff_date and
               p["actual_outcome"] in ["WIN", "LOSS"]
        ]
        
        if len(model_predictions) < self.min_predictions:
            # Not enough predictions for reliable metrics
            if model_name in self.model_performance:
                # Keep existing performance but mark as insufficient data
                self.model_performance[model_name].n_predictions = len(model_predictions)
            return
        
        # Calculate Brier score
        brier_score = self._calculate_brier_score(model_predictions)
        
        # Calculate accuracy
        accuracy = self._calculate_accuracy(model_predictions)
        
        # Create or update performance record
        self.model_performance[model_name] = ModelPerformance(
            model_name=model_name,
            timestamp=datetime.now(),
            brier_score=brier_score,
            accuracy=accuracy,
            n_predictions=len(model_predictions),
            weight=0.0,  # Will be calculated in update_ensemble_weights
            window_start=cutoff_date,
            window_end=datetime.now()
        )
        
        logger.debug(f"Updated performance for {model_name}: Brier={brier_score:.3f}, "
                    f"Accuracy={accuracy:.3f}, N={len(model_predictions)}")
    
    def _calculate_brier_score(self, predictions: List[Dict[str, Any]]) -> float:
        """Calculate Brier score for a set of predictions."""
        if not predictions:
            return 1.0  # Worst possible score
        
        brier_scores = []
        for pred in predictions:
            predicted_prob = pred["predicted_probability"]
            actual_outcome = 1.0 if pred["actual_outcome"] == "WIN" else 0.0
            brier_score = (predicted_prob - actual_outcome) ** 2
            brier_scores.append(brier_score)
        
        return np.mean(brier_scores)
    
    def _calculate_accuracy(self, predictions: List[Dict[str, Any]]) -> float:
        """Calculate accuracy for a set of predictions."""
        if not predictions:
            return 0.0
        
        correct_predictions = 0
        for pred in predictions:
            predicted_prob = pred["predicted_probability"]
            actual_outcome = pred["actual_outcome"]
            
            # Consider prediction correct if probability > 0.5 and WIN, or < 0.5 and LOSS
            if (predicted_prob > 0.5 and actual_outcome == "WIN") or \
               (predicted_prob <= 0.5 and actual_outcome == "LOSS"):
                correct_predictions += 1
        
        return correct_predictions / len(predictions)
    
    def update_ensemble_weights(self) -> EnsembleWeights:
        """
        Update ensemble weights based on current model performance.
        
        Returns:
            EnsembleWeights with updated weights for all models
        """
        if not self.model_performance:
            logger.warning("No model performance data available for weight calculation")
            return self._create_default_weights()
        
        # Calculate weights based on inverse Brier score
        weights = {}
        total_weight = 0.0
        
        for model_name, performance in self.model_performance.items():
            if performance.n_predictions >= self.min_predictions:
                # Weight inversely proportional to Brier score
                # Lower Brier score = better performance = higher weight
                weight = 1.0 / (performance.brier_score + self.epsilon)
                weights[model_name] = weight
                total_weight += weight
            else:
                # Insufficient data - use default weight
                weights[model_name] = 1.0 / len(self.model_performance)
                total_weight += weights[model_name]
        
        # Normalize weights to sum to 1
        if total_weight > 0:
            for model_name in weights:
                weights[model_name] /= total_weight
        
        # Apply weight smoothing to prevent dramatic changes
        if self.ensemble_weights is not None:
            weights = self._smooth_weights(weights)
        
        # Calculate weight entropy (measure of distribution)
        weight_entropy = self._calculate_weight_entropy(weights)
        
        # Create ensemble weights object
        self.ensemble_weights = EnsembleWeights(
            timestamp=datetime.now(),
            weights=weights,
            total_models=len(weights),
            performance_summary=self.model_performance.copy(),
            weight_entropy=weight_entropy
        )
        
        # Update weights in performance records
        for model_name, weight in weights.items():
            if model_name in self.model_performance:
                self.model_performance[model_name].weight = weight
        
        logger.info(f"Updated ensemble weights: {weights}")
        return self.ensemble_weights
    
    def _smooth_weights(self, new_weights: Dict[str, float]) -> Dict[str, float]:
        """Apply smoothing to prevent dramatic weight changes."""
        if self.ensemble_weights is None:
            return new_weights
        
        smoothed_weights = {}
        for model_name, new_weight in new_weights.items():
            old_weight = self.ensemble_weights.weights.get(model_name, 1.0 / len(new_weights))
            
            # Calculate smoothed weight
            smoothed_weight = (self.weight_smoothing_factor * new_weight + 
                             (1 - self.weight_smoothing_factor) * old_weight)
            
            # Limit maximum change
            weight_change = abs(smoothed_weight - old_weight)
            if weight_change > self.max_weight_change:
                if smoothed_weight > old_weight:
                    smoothed_weight = old_weight + self.max_weight_change
                else:
                    smoothed_weight = old_weight - self.max_weight_change
            
            smoothed_weights[model_name] = max(0.01, smoothed_weight)  # Minimum weight
        
        # Renormalize after smoothing
        total_weight = sum(smoothed_weights.values())
        if total_weight > 0:
            for model_name in smoothed_weights:
                smoothed_weights[model_name] /= total_weight
        
        return smoothed_weights
    
    def _calculate_weight_entropy(self, weights: Dict[str, float]) -> float:
        """Calculate entropy of weight distribution."""
        if not weights:
            return 0.0
        
        entropy = 0.0
        for weight in weights.values():
            if weight > 0:
                entropy -= weight * np.log2(weight)
        
        return entropy
    
    def _create_default_weights(self) -> EnsembleWeights:
        """Create default equal weights when no performance data is available."""
        # Get list of models from predictions
        models = list(set(p["model_name"] for p in self.model_predictions))
        
        if not models:
            models = ["default_model"]  # Fallback
        
        equal_weight = 1.0 / len(models)
        weights = {model: equal_weight for model in models}
        
        return EnsembleWeights(
            timestamp=datetime.now(),
            weights=weights,
            total_models=len(models),
            performance_summary={},
            weight_entropy=self._calculate_weight_entropy(weights)
        )
    
    def get_model_weight(self, model_name: str) -> float:
        """Get current weight for a specific model."""
        if self.ensemble_weights is None:
            self.update_ensemble_weights()
        
        return self.ensemble_weights.weights.get(model_name, 1.0 / len(self.ensemble_weights.weights))
    
    def get_ensemble_weights(self) -> Dict[str, float]:
        """Get current ensemble weights for all models."""
        if self.ensemble_weights is None:
            self.update_ensemble_weights()
        
        return self.ensemble_weights.weights.copy()
    
    def get_performance_summary(self, model_name: str = None) -> Dict[str, Any]:
        """Get performance summary for a model or all models."""
        if model_name:
            if model_name in self.model_performance:
                perf = self.model_performance[model_name]
                return {
                    "model_name": model_name,
                    "brier_score": perf.brier_score,
                    "accuracy": perf.accuracy,
                    "n_predictions": perf.n_predictions,
                    "weight": perf.weight,
                    "window_days": self.window_size_days,
                    "last_updated": perf.timestamp.isoformat()
                }
            else:
                return {"error": f"No performance data for model {model_name}"}
        else:
            # Summary for all models
            summary = {
                "total_models": len(self.model_performance),
                "models": {}
            }
            
            for model_name, perf in self.model_performance.items():
                summary["models"][model_name] = {
                    "brier_score": perf.brier_score,
                    "accuracy": perf.accuracy,
                    "n_predictions": perf.n_predictions,
                    "weight": perf.weight
                }
            
            if self.ensemble_weights:
                summary["ensemble_entropy"] = self.ensemble_weights.weight_entropy
                summary["last_weight_update"] = self.ensemble_weights.timestamp.isoformat()
            
            return summary
    
    def should_update_weights(self) -> bool:
        """Check if weights should be updated based on frequency."""
        if self.ensemble_weights is None:
            return True
        
        time_since_update = datetime.now() - self.ensemble_weights.timestamp
        return time_since_update >= timedelta(hours=self.weight_update_frequency_hours)
    
    def clear_old_data(self, days_to_keep: int = 30):
        """Clear old prediction data to manage memory."""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        # Remove old predictions
        self.model_predictions = [
            p for p in self.model_predictions
            if p["prediction_date"] >= cutoff_date
        ]
        
        # Remove old performance data
        old_models = [
            model_name for model_name, perf in self.model_performance.items()
            if perf.timestamp < cutoff_date
        ]
        
        for model_name in old_models:
            del self.model_performance[model_name]
        
        logger.info(f"Cleared data older than {days_to_keep} days")
    
    def export_weights_data(self) -> Dict[str, Any]:
        """Export weights and performance data for persistence."""
        return {
            "model_predictions": [
                {
                    "model_name": p["model_name"],
                    "symbol": p["symbol"],
                    "predicted_probability": p["predicted_probability"],
                    "actual_outcome": p["actual_outcome"],
                    "prediction_date": p["prediction_date"].isoformat(),
                    "trade_id": p["trade_id"],
                    "timestamp": p["timestamp"].isoformat()
                }
                for p in self.model_predictions
            ],
            "model_performance": {
                model_name: {
                    "model_name": perf.model_name,
                    "timestamp": perf.timestamp.isoformat(),
                    "brier_score": perf.brier_score,
                    "accuracy": perf.accuracy,
                    "n_predictions": perf.n_predictions,
                    "weight": perf.weight,
                    "window_start": perf.window_start.isoformat(),
                    "window_end": perf.window_end.isoformat()
                }
                for model_name, perf in self.model_performance.items()
            },
            "ensemble_weights": {
                "timestamp": self.ensemble_weights.timestamp.isoformat(),
                "weights": self.ensemble_weights.weights,
                "total_models": self.ensemble_weights.total_models,
                "weight_entropy": self.ensemble_weights.weight_entropy
            } if self.ensemble_weights else None
        }
    
    def import_weights_data(self, data: Dict[str, Any]):
        """Import weights and performance data from persistence."""
        # Import model predictions
        self.model_predictions = []
        for pred_data in data.get("model_predictions", []):
            self.model_predictions.append({
                "model_name": pred_data["model_name"],
                "symbol": pred_data["symbol"],
                "predicted_probability": pred_data["predicted_probability"],
                "actual_outcome": pred_data["actual_outcome"],
                "prediction_date": datetime.fromisoformat(pred_data["prediction_date"]),
                "trade_id": pred_data["trade_id"],
                "timestamp": datetime.fromisoformat(pred_data["timestamp"])
            })
        
        # Import model performance
        self.model_performance = {}
        for model_name, perf_data in data.get("model_performance", {}).items():
            self.model_performance[model_name] = ModelPerformance(
                model_name=perf_data["model_name"],
                timestamp=datetime.fromisoformat(perf_data["timestamp"]),
                brier_score=perf_data["brier_score"],
                accuracy=perf_data["accuracy"],
                n_predictions=perf_data["n_predictions"],
                weight=perf_data["weight"],
                window_start=datetime.fromisoformat(perf_data["window_start"]),
                window_end=datetime.fromisoformat(perf_data["window_end"])
            )
        
        # Import ensemble weights
        if data.get("ensemble_weights"):
            weights_data = data["ensemble_weights"]
            self.ensemble_weights = EnsembleWeights(
                timestamp=datetime.fromisoformat(weights_data["timestamp"]),
                weights=weights_data["weights"],
                total_models=weights_data["total_models"],
                performance_summary=self.model_performance.copy(),
                weight_entropy=weights_data["weight_entropy"]
            )
        
        logger.info(f"Imported weights data: {len(self.model_predictions)} predictions, "
                   f"{len(self.model_performance)} model performances")


# Global adaptive weight manager instance
_adaptive_weight_manager: Optional[AdaptiveWeightManager] = None


def get_adaptive_weight_manager() -> AdaptiveWeightManager:
    """Get global adaptive weight manager instance."""
    global _adaptive_weight_manager
    if _adaptive_weight_manager is None:
        _adaptive_weight_manager = AdaptiveWeightManager()
    return _adaptive_weight_manager


# Convenience functions
def add_model_prediction(model_name: str, symbol: str, predicted_probability: float,
                        actual_outcome: str, prediction_date: datetime, 
                        trade_id: str = None) -> None:
    """Add a model prediction for performance tracking."""
    manager = get_adaptive_weight_manager()
    manager.add_prediction(model_name, symbol, predicted_probability, 
                          actual_outcome, prediction_date, trade_id)


def get_model_weight(model_name: str) -> float:
    """Get current weight for a specific model."""
    manager = get_adaptive_weight_manager()
    return manager.get_model_weight(model_name)


def get_ensemble_weights() -> Dict[str, float]:
    """Get current ensemble weights for all models."""
    manager = get_adaptive_weight_manager()
    return manager.get_ensemble_weights()


def update_ensemble_weights() -> EnsembleWeights:
    """Update ensemble weights based on current performance."""
    manager = get_adaptive_weight_manager()
    return manager.update_ensemble_weights()


def get_performance_summary(model_name: str = None) -> Dict[str, Any]:
    """Get performance summary for a model or all models."""
    manager = get_adaptive_weight_manager()
    return manager.get_performance_summary(model_name)
