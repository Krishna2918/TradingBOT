"""
Cross-Model Validator
====================

Validates predictions across multiple models to ensure reliability and detect outliers.
Implements consensus building, outlier detection, and prediction reliability assessment.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import statistics
import threading

logger = logging.getLogger(__name__)

@dataclass
class ModelPrediction:
    """Individual model prediction with metadata."""
    model_name: str
    prediction: float
    confidence: float
    timestamp: datetime
    reasoning: str
    metadata: Dict[str, Any]

@dataclass
class ValidationResult:
    """Result of cross-model validation."""
    timestamp: datetime
    predictions: List[ModelPrediction]
    consensus_prediction: float
    consensus_confidence: float
    agreement_score: float
    outlier_models: List[str]
    reliability_score: float
    validation_status: str  # 'reliable', 'warning', 'unreliable'
    recommendations: List[str]

@dataclass
class ValidationMetrics:
    """Metrics for validation performance tracking."""
    total_validations: int
    reliable_predictions: int
    warning_predictions: int
    unreliable_predictions: int
    average_agreement: float
    average_consensus_confidence: float
    outlier_detection_rate: float
    false_positive_rate: float

class CrossModelValidator:
    """
    Cross-model validation system for prediction reliability.
    
    Features:
    - Outlier detection and filtering
    - Consensus building
    - Agreement scoring
    - Reliability assessment
    - Performance tracking
    - Adaptive thresholds
    """
    
    def __init__(self, min_agreement: float = 0.7, outlier_threshold: float = 1.5):
        """
        Initialize the cross-model validator.
        
        Args:
            min_agreement: Minimum agreement score for reliable predictions
            outlier_threshold: Z-score threshold for outlier detection
        """
        self.validator_name = "cross_model_validator"
        self.min_agreement = min_agreement
        self.outlier_threshold = outlier_threshold
        
        # Validation history
        self.validation_history: deque = deque(maxlen=10000)
        self.model_performance: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Adaptive thresholds
        self.adaptive_thresholds = {
            'agreement': min_agreement,
            'outlier': outlier_threshold,
            'confidence': 0.6
        }
        
        # Validation metrics
        self.metrics = ValidationMetrics(
            total_validations=0,
            reliable_predictions=0,
            warning_predictions=0,
            unreliable_predictions=0,
            average_agreement=0.0,
            average_consensus_confidence=0.0,
            outlier_detection_rate=0.0,
            false_positive_rate=0.0
        )
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info(f"Cross-Model Validator initialized: {self.validator_name}")
    
    def validate_predictions(self, predictions: List[Dict[str, Any]]) -> ValidationResult:
        """
        Validate predictions from multiple models.
        
        Args:
            predictions: List of prediction dictionaries with model_name, prediction, confidence
            
        Returns:
            Validation result with consensus and reliability assessment
        """
        try:
            with self._lock:
                if not predictions or len(predictions) < 2:
                    return self._get_default_validation_result(predictions)
                
                # Convert to ModelPrediction objects
                model_predictions = []
                for pred in predictions:
                    if isinstance(pred, dict) and 'model_name' in pred and 'prediction' in pred:
                        model_pred = ModelPrediction(
                            model_name=pred['model_name'],
                            prediction=float(pred['prediction']),
                            confidence=float(pred.get('confidence', 0.5)),
                            timestamp=datetime.now(),
                            reasoning=pred.get('reasoning', ''),
                            metadata=pred.get('metadata', {})
                        )
                        model_predictions.append(model_pred)
                
                if len(model_predictions) < 2:
                    return self._get_default_validation_result(predictions)
                
                # Detect outliers
                outlier_models = self._detect_outliers(model_predictions)
                
                # Filter out outliers
                valid_predictions = [p for p in model_predictions if p.model_name not in outlier_models]
                
                if not valid_predictions:
                    # If all predictions are outliers, use all predictions
                    valid_predictions = model_predictions
                    outlier_models = []
                
                # Build consensus
                consensus_prediction, consensus_confidence = self._build_consensus(valid_predictions)
                
                # Calculate agreement score
                agreement_score = self._calculate_agreement_score(valid_predictions)
                
                # Calculate reliability score
                reliability_score = self._calculate_reliability_score(
                    valid_predictions, agreement_score, consensus_confidence
                )
                
                # Determine validation status
                validation_status = self._determine_validation_status(
                    agreement_score, consensus_confidence, reliability_score
                )
                
                # Generate recommendations
                recommendations = self._generate_recommendations(
                    validation_status, outlier_models, agreement_score
                )
                
                # Create validation result
                result = ValidationResult(
                    timestamp=datetime.now(),
                    predictions=model_predictions,
                    consensus_prediction=consensus_prediction,
                    consensus_confidence=consensus_confidence,
                    agreement_score=agreement_score,
                    outlier_models=outlier_models,
                    reliability_score=reliability_score,
                    validation_status=validation_status,
                    recommendations=recommendations
                )
                
                # Update metrics and history
                self._update_metrics(result)
                self.validation_history.append(result)
                
                logger.debug(f"Validation completed: {validation_status}, agreement={agreement_score:.3f}")
                return result
                
        except Exception as e:
            logger.error(f"Error validating predictions: {e}")
            return self._get_default_validation_result(predictions)
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics and metrics."""
        try:
            with self._lock:
                return {
                    'validator_name': self.validator_name,
                    'total_validations': self.metrics.total_validations,
                    'reliable_predictions': self.metrics.reliable_predictions,
                    'warning_predictions': self.metrics.warning_predictions,
                    'unreliable_predictions': self.metrics.unreliable_predictions,
                    'reliability_rate': (
                        self.metrics.reliable_predictions / max(1, self.metrics.total_validations)
                    ),
                    'average_agreement': self.metrics.average_agreement,
                    'average_consensus_confidence': self.metrics.average_consensus_confidence,
                    'outlier_detection_rate': self.metrics.outlier_detection_rate,
                    'adaptive_thresholds': self.adaptive_thresholds,
                    'model_count': len(self.model_performance)
                }
                
        except Exception as e:
            logger.error(f"Error getting validation statistics: {e}")
            return {'error': str(e)}
    
    def get_model_performance(self, model_name: str) -> Dict[str, Any]:
        """Get performance statistics for a specific model."""
        try:
            with self._lock:
                if model_name not in self.model_performance:
                    return {'error': f'No performance data for model {model_name}'}
                
                return self.model_performance[model_name].copy()
                
        except Exception as e:
            logger.error(f"Error getting model performance for {model_name}: {e}")
            return {'error': str(e)}
    
    def update_model_performance(self, model_name: str, actual_outcome: float, 
                               prediction: float, confidence: float) -> bool:
        """
        Update model performance based on actual outcomes.
        
        Args:
            model_name: Name of the model
            actual_outcome: Actual outcome value
            prediction: Model prediction
            confidence: Model confidence
            
        Returns:
            True if update successful
        """
        try:
            with self._lock:
                if model_name not in self.model_performance:
                    self.model_performance[model_name] = {
                        'total_predictions': 0,
                        'accurate_predictions': 0,
                        'outlier_count': 0,
                        'average_confidence': 0.0,
                        'average_error': 0.0,
                        'recent_performance': deque(maxlen=100)
                    }
                
                model_stats = self.model_performance[model_name]
                
                # Calculate accuracy
                error = abs(prediction - actual_outcome)
                is_accurate = error < (actual_outcome * 0.1)  # Within 10% considered accurate
                
                # Update statistics
                model_stats['total_predictions'] += 1
                if is_accurate:
                    model_stats['accurate_predictions'] += 1
                
                # Update running averages
                total = model_stats['total_predictions']
                model_stats['average_confidence'] = (
                    (model_stats['average_confidence'] * (total - 1) + confidence) / total
                )
                model_stats['average_error'] = (
                    (model_stats['average_error'] * (total - 1) + error) / total
                )
                
                # Track recent performance
                model_stats['recent_performance'].append({
                    'timestamp': datetime.now(),
                    'prediction': prediction,
                    'actual': actual_outcome,
                    'error': error,
                    'confidence': confidence,
                    'accurate': is_accurate
                })
                
                logger.debug(f"Performance updated for {model_name}: accuracy={is_accurate}")
                return True
                
        except Exception as e:
            logger.error(f"Error updating model performance for {model_name}: {e}")
            return False
    
    def _detect_outliers(self, predictions: List[ModelPrediction]) -> List[str]:
        """Detect outlier predictions using statistical methods."""
        try:
            if len(predictions) < 3:
                return []  # Need at least 3 predictions for outlier detection
            
            # Extract prediction values
            pred_values = [p.prediction for p in predictions]
            
            # Calculate z-scores
            mean_pred = statistics.mean(pred_values)
            std_pred = statistics.stdev(pred_values) if len(pred_values) > 1 else 0.01
            
            if std_pred == 0:
                return []  # No variation, no outliers
            
            outliers = []
            for prediction in predictions:
                z_score = abs((prediction.prediction - mean_pred) / std_pred)
                if z_score > self.adaptive_thresholds['outlier']:
                    outliers.append(prediction.model_name)
            
            return outliers
            
        except Exception as e:
            logger.error(f"Error detecting outliers: {e}")
            return []
    
    def _build_consensus(self, predictions: List[ModelPrediction]) -> Tuple[float, float]:
        """Build consensus prediction from valid predictions."""
        try:
            if not predictions:
                return 0.0, 0.0
            
            if len(predictions) == 1:
                return predictions[0].prediction, predictions[0].confidence
            
            # Weighted average based on confidence
            total_weight = 0.0
            weighted_sum = 0.0
            
            for pred in predictions:
                weight = pred.confidence
                weighted_sum += pred.prediction * weight
                total_weight += weight
            
            if total_weight == 0:
                # Fallback to simple average
                consensus_prediction = statistics.mean([p.prediction for p in predictions])
                consensus_confidence = statistics.mean([p.confidence for p in predictions])
            else:
                consensus_prediction = weighted_sum / total_weight
                consensus_confidence = min(1.0, total_weight / len(predictions))
            
            return consensus_prediction, consensus_confidence
            
        except Exception as e:
            logger.error(f"Error building consensus: {e}")
            return 0.0, 0.0
    
    def _calculate_agreement_score(self, predictions: List[ModelPrediction]) -> float:
        """Calculate agreement score among predictions."""
        try:
            if len(predictions) < 2:
                return 1.0  # Perfect agreement with single prediction
            
            pred_values = [p.prediction for p in predictions]
            
            # Calculate coefficient of variation (lower is better agreement)
            mean_pred = statistics.mean(pred_values)
            std_pred = statistics.stdev(pred_values) if len(pred_values) > 1 else 0.0
            
            if mean_pred == 0:
                return 1.0 if std_pred == 0 else 0.0
            
            cv = std_pred / abs(mean_pred)
            agreement_score = max(0.0, 1.0 - cv)  # Convert to 0-1 scale
            
            return agreement_score
            
        except Exception as e:
            logger.error(f"Error calculating agreement score: {e}")
            return 0.5
    
    def _calculate_reliability_score(self, predictions: List[ModelPrediction], 
                                   agreement_score: float, consensus_confidence: float) -> float:
        """Calculate overall reliability score."""
        try:
            # Base reliability on agreement and confidence
            base_reliability = (agreement_score + consensus_confidence) / 2.0
            
            # Adjust based on number of models (more models = higher reliability)
            model_count_factor = min(1.0, len(predictions) / 5.0)  # Max at 5 models
            
            # Adjust based on model performance history
            performance_factor = 1.0
            for pred in predictions:
                if pred.model_name in self.model_performance:
                    model_stats = self.model_performance[pred.model_name]
                    if model_stats['total_predictions'] > 0:
                        accuracy_rate = model_stats['accurate_predictions'] / model_stats['total_predictions']
                        performance_factor *= accuracy_rate
            
            # Combine factors
            reliability_score = base_reliability * model_count_factor * performance_factor
            return min(1.0, max(0.0, reliability_score))
            
        except Exception as e:
            logger.error(f"Error calculating reliability score: {e}")
            return 0.5
    
    def _determine_validation_status(self, agreement_score: float, 
                                   consensus_confidence: float, reliability_score: float) -> str:
        """Determine validation status based on scores."""
        try:
            if (agreement_score >= self.adaptive_thresholds['agreement'] and 
                consensus_confidence >= self.adaptive_thresholds['confidence'] and
                reliability_score >= 0.7):
                return 'reliable'
            elif (agreement_score >= 0.5 and consensus_confidence >= 0.4 and 
                  reliability_score >= 0.4):
                return 'warning'
            else:
                return 'unreliable'
                
        except Exception as e:
            logger.error(f"Error determining validation status: {e}")
            return 'warning'
    
    def _generate_recommendations(self, validation_status: str, outlier_models: List[str], 
                                agreement_score: float) -> List[str]:
        """Generate recommendations based on validation results."""
        try:
            recommendations = []
            
            if validation_status == 'reliable':
                recommendations.append("Predictions are reliable and can be used for trading decisions")
            elif validation_status == 'warning':
                recommendations.append("Use predictions with caution - consider additional validation")
                if agreement_score < 0.6:
                    recommendations.append("Low agreement between models - wait for more consensus")
            else:
                recommendations.append("Predictions are unreliable - avoid trading decisions")
                recommendations.append("Consider retraining models or using different approaches")
            
            if outlier_models:
                recommendations.append(f"Models {', '.join(outlier_models)} are outliers and should be investigated")
            
            if len(outlier_models) > 0:
                recommendations.append("Consider excluding outlier models from future predictions")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ["Unable to generate recommendations"]
    
    def _update_metrics(self, result: ValidationResult):
        """Update validation metrics based on result."""
        try:
            self.metrics.total_validations += 1
            
            if result.validation_status == 'reliable':
                self.metrics.reliable_predictions += 1
            elif result.validation_status == 'warning':
                self.metrics.warning_predictions += 1
            else:
                self.metrics.unreliable_predictions += 1
            
            # Update running averages
            total = self.metrics.total_validations
            self.metrics.average_agreement = (
                (self.metrics.average_agreement * (total - 1) + result.agreement_score) / total
            )
            self.metrics.average_consensus_confidence = (
                (self.metrics.average_consensus_confidence * (total - 1) + result.consensus_confidence) / total
            )
            
            # Update outlier detection rate
            if result.outlier_models:
                self.metrics.outlier_detection_rate = (
                    (self.metrics.outlier_detection_rate * (total - 1) + 1.0) / total
                )
            else:
                self.metrics.outlier_detection_rate = (
                    (self.metrics.outlier_detection_rate * (total - 1) + 0.0) / total
                )
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    def _get_default_validation_result(self, predictions: List[Dict[str, Any]]) -> ValidationResult:
        """Get default validation result when insufficient data."""
        return ValidationResult(
            timestamp=datetime.now(),
            predictions=[],
            consensus_prediction=0.0,
            consensus_confidence=0.0,
            agreement_score=0.0,
            outlier_models=[],
            reliability_score=0.0,
            validation_status='unreliable',
            recommendations=['Insufficient predictions for validation']
        )

# Global cross-model validator instance
_cross_model_validator = None

def get_cross_model_validator() -> CrossModelValidator:
    """Get the global cross-model validator instance."""
    global _cross_model_validator
    if _cross_model_validator is None:
        _cross_model_validator = CrossModelValidator()
    return _cross_model_validator