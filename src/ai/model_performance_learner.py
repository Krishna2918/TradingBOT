"""
Model Performance Learning System
================================

Records and learns from model performance in different market conditions.
Implements adaptive learning and performance prediction capabilities.
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
class PerformanceRecord:
    """Record of model performance in specific conditions."""
    model_name: str
    timestamp: datetime
    market_conditions: Dict[str, Any]  # regime, volatility, etc.
    prediction: float
    actual_outcome: float
    accuracy: float
    brier_score: float
    confidence: float
    execution_time: float
    context: Dict[str, Any]  # Additional context

@dataclass
class LearningInsight:
    """Insight derived from performance learning."""
    model_name: str
    insight_type: str  # 'condition_performance', 'bias_detection', 'improvement_suggestion'
    conditions: Dict[str, Any]
    insight_data: Dict[str, Any]
    confidence: float
    timestamp: datetime

@dataclass
class PerformanceProfile:
    """Performance profile for a model in specific conditions."""
    model_name: str
    conditions: Dict[str, Any]
    avg_accuracy: float
    avg_brier_score: float
    sample_count: int
    last_updated: datetime
    performance_trend: str  # 'improving', 'stable', 'declining'
    volatility: float  # Performance volatility

class ModelPerformanceLearner:
    """
    System for learning from model performance patterns.
    
    Features:
    - Performance tracking by market conditions
    - Bias detection and correction
    - Performance prediction
    - Adaptive learning recommendations
    - Performance profiling
    """
    
    def __init__(self, max_records: int = 10000, learning_window: int = 100):
        """
        Initialize the performance learning system.
        
        Args:
            max_records: Maximum number of performance records to store
            learning_window: Window size for learning analysis
        """
        self.learner_name = "model_performance_learner"
        self.max_records = max_records
        self.learning_window = learning_window
        
        # Performance storage
        self.performance_records: deque = deque(maxlen=max_records)
        self.performance_profiles: Dict[str, PerformanceProfile] = {}
        self.learning_insights: deque = deque(maxlen=1000)
        
        # Learning state
        self.learning_active = True
        self.learning_threshold = 0.05  # Minimum change to trigger learning
        self.condition_categories = {
            'regime': ['trending', 'ranging', 'volatile', 'calm'],
            'volatility_level': ['low', 'medium', 'high'],
            'market_phase': ['pre_market', 'open', 'mid_day', 'close', 'after_hours'],
            'sector': ['tech', 'finance', 'healthcare', 'energy', 'other']
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info(f"Model Performance Learner initialized: {self.learner_name}")
    
    def record_performance(self, model_name: str, accuracy: float, brier_score: float, 
                          conditions: Dict[str, Any], prediction: Optional[float] = None,
                          actual_outcome: Optional[float] = None, confidence: Optional[float] = None,
                          execution_time: Optional[float] = None, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Record model performance for learning.
        
        Args:
            model_name: Name of the model
            accuracy: Prediction accuracy (0-1)
            brier_score: Brier score (0-2, lower is better)
            conditions: Market conditions during prediction
            prediction: Model prediction value
            actual_outcome: Actual outcome value
            confidence: Model confidence level
            execution_time: Time taken for prediction
            context: Additional context information
            
        Returns:
            True if recording successful
        """
        try:
            with self._lock:
                record = PerformanceRecord(
                    model_name=model_name,
                    timestamp=datetime.now(),
                    market_conditions=conditions.copy(),
                    prediction=prediction or 0.0,
                    actual_outcome=actual_outcome or 0.0,
                    accuracy=accuracy,
                    brier_score=brier_score,
                    confidence=confidence or 0.5,
                    execution_time=execution_time or 0.0,
                    context=context or {}
                )
                
                self.performance_records.append(record)
                
                # Update performance profiles
                self._update_performance_profile(record)
                
                # Trigger learning if enough new data
                if len(self.performance_records) % self.learning_window == 0:
                    self._trigger_learning()
                
                logger.debug(f"Performance recorded for {model_name}: accuracy={accuracy:.3f}, brier={brier_score:.3f}")
                return True
                
        except Exception as e:
            logger.error(f"Error recording performance for {model_name}: {e}")
            return False
    
    def get_model_performance_summary(self, model_name: str, 
                                    time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """
        Get performance summary for a specific model.
        
        Args:
            model_name: Name of the model
            time_window: Time window for analysis (None for all time)
            
        Returns:
            Performance summary dictionary
        """
        try:
            with self._lock:
                # Filter records for the model and time window
                cutoff_time = datetime.now() - time_window if time_window else None
                model_records = [
                    record for record in self.performance_records
                    if record.model_name == model_name and 
                    (cutoff_time is None or record.timestamp >= cutoff_time)
                ]
                
                if not model_records:
                    return {
                        'model_name': model_name,
                        'total_predictions': 0,
                        'avg_accuracy': 0.0,
                        'avg_brier_score': 0.0,
                        'performance_trend': 'unknown',
                        'best_conditions': {},
                        'worst_conditions': {}
                    }
                
                # Calculate summary statistics
                accuracies = [record.accuracy for record in model_records]
                brier_scores = [record.brier_score for record in model_records]
                
                avg_accuracy = statistics.mean(accuracies)
                avg_brier_score = statistics.mean(brier_scores)
                accuracy_std = statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0
                
                # Determine performance trend
                if len(model_records) >= 10:
                    recent_accuracies = [r.accuracy for r in model_records[-10:]]
                    older_accuracies = [r.accuracy for r in model_records[-20:-10]] if len(model_records) >= 20 else []
                    
                    if older_accuracies:
                        recent_avg = statistics.mean(recent_accuracies)
                        older_avg = statistics.mean(older_accuracies)
                        if recent_avg > older_avg + self.learning_threshold:
                            trend = 'improving'
                        elif recent_avg < older_avg - self.learning_threshold:
                            trend = 'declining'
                        else:
                            trend = 'stable'
                    else:
                        trend = 'stable'
                else:
                    trend = 'insufficient_data'
                
                # Find best and worst performing conditions
                condition_performance = defaultdict(list)
                for record in model_records:
                    for condition_key, condition_value in record.market_conditions.items():
                        condition_performance[f"{condition_key}_{condition_value}"].append(record.accuracy)
                
                best_conditions = {}
                worst_conditions = {}
                
                for condition, accuracies in condition_performance.items():
                    if len(accuracies) >= 3:  # Minimum sample size
                        avg_acc = statistics.mean(accuracies)
                        if not best_conditions or avg_acc > list(best_conditions.values())[0]:
                            best_conditions = {condition: avg_acc}
                        if not worst_conditions or avg_acc < list(worst_conditions.values())[0]:
                            worst_conditions = {condition: avg_acc}
                
                return {
                    'model_name': model_name,
                    'total_predictions': len(model_records),
                    'avg_accuracy': avg_accuracy,
                    'avg_brier_score': avg_brier_score,
                    'accuracy_std': accuracy_std,
                    'performance_trend': trend,
                    'best_conditions': best_conditions,
                    'worst_conditions': worst_conditions,
                    'time_window': time_window.total_seconds() if time_window else None
                }
                
        except Exception as e:
            logger.error(f"Error getting performance summary for {model_name}: {e}")
            return {'error': str(e)}
    
    def predict_model_performance(self, model_name: str, conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict model performance in given conditions.
        
        Args:
            model_name: Name of the model
            conditions: Market conditions for prediction
            
        Returns:
            Performance prediction dictionary
        """
        try:
            with self._lock:
                # Find similar conditions in history
                similar_records = self._find_similar_conditions(model_name, conditions)
                
                if not similar_records:
                    return {
                        'model_name': model_name,
                        'predicted_accuracy': 0.5,  # Default neutral prediction
                        'predicted_brier_score': 1.0,
                        'confidence': 0.0,
                        'sample_size': 0,
                        'similarity_score': 0.0
                    }
                
                # Calculate predictions based on similar conditions
                accuracies = [record.accuracy for record in similar_records]
                brier_scores = [record.brier_score for record in similar_records]
                
                predicted_accuracy = statistics.mean(accuracies)
                predicted_brier_score = statistics.mean(brier_scores)
                
                # Calculate confidence based on sample size and consistency
                sample_size = len(similar_records)
                accuracy_std = statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0
                consistency_score = max(0.0, 1.0 - accuracy_std)
                sample_confidence = min(1.0, sample_size / 20.0)  # Max confidence at 20+ samples
                
                overall_confidence = (consistency_score + sample_confidence) / 2.0
                
                # Calculate similarity score
                similarity_score = self._calculate_condition_similarity(conditions, similar_records[0].market_conditions)
                
                return {
                    'model_name': model_name,
                    'predicted_accuracy': predicted_accuracy,
                    'predicted_brier_score': predicted_brier_score,
                    'confidence': overall_confidence,
                    'sample_size': sample_size,
                    'similarity_score': similarity_score,
                    'accuracy_range': (min(accuracies), max(accuracies)),
                    'brier_range': (min(brier_scores), max(brier_scores))
                }
                
        except Exception as e:
            logger.error(f"Error predicting performance for {model_name}: {e}")
            return {'error': str(e)}
    
    def get_learning_insights(self, model_name: Optional[str] = None) -> List[LearningInsight]:
        """
        Get learning insights for models.
        
        Args:
            model_name: Filter by specific model (None for all)
            
        Returns:
            List of learning insights
        """
        try:
            with self._lock:
                if model_name:
                    return [insight for insight in self.learning_insights if insight.model_name == model_name]
                else:
                    return list(self.learning_insights)
                    
        except Exception as e:
            logger.error(f"Error getting learning insights: {e}")
            return []
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get learning system statistics."""
        try:
            with self._lock:
                # Count records by model
                model_counts = defaultdict(int)
                for record in self.performance_records:
                    model_counts[record.model_name] += 1
                
                # Calculate overall statistics
                total_records = len(self.performance_records)
                unique_models = len(model_counts)
                total_insights = len(self.learning_insights)
                
                # Calculate average performance
                if total_records > 0:
                    avg_accuracy = statistics.mean([r.accuracy for r in self.performance_records])
                    avg_brier_score = statistics.mean([r.brier_score for r in self.performance_records])
                else:
                    avg_accuracy = 0.0
                    avg_brier_score = 0.0
                
                return {
                    'learner_name': self.learner_name,
                    'total_records': total_records,
                    'unique_models': unique_models,
                    'total_insights': total_insights,
                    'avg_accuracy': avg_accuracy,
                    'avg_brier_score': avg_brier_score,
                    'learning_active': self.learning_active,
                    'model_counts': dict(model_counts),
                    'last_activity': self.performance_records[-1].timestamp.isoformat() if self.performance_records else None
                }
                
        except Exception as e:
            logger.error(f"Error getting learning statistics: {e}")
            return {'error': str(e)}
    
    def _update_performance_profile(self, record: PerformanceRecord):
        """Update performance profile for a model."""
        try:
            # Create condition key
            condition_key = self._create_condition_key(record.market_conditions)
            profile_key = f"{record.model_name}_{condition_key}"
            
            if profile_key in self.performance_profiles:
                profile = self.performance_profiles[profile_key]
                
                # Update running averages
                total_samples = profile.sample_count + 1
                profile.avg_accuracy = (profile.avg_accuracy * profile.sample_count + record.accuracy) / total_samples
                profile.avg_brier_score = (profile.avg_brier_score * profile.sample_count + record.brier_score) / total_samples
                profile.sample_count = total_samples
                profile.last_updated = record.timestamp
                
            else:
                # Create new profile
                profile = PerformanceProfile(
                    model_name=record.model_name,
                    conditions=record.market_conditions.copy(),
                    avg_accuracy=record.accuracy,
                    avg_brier_score=record.brier_score,
                    sample_count=1,
                    last_updated=record.timestamp,
                    performance_trend='stable',
                    volatility=0.0
                )
                self.performance_profiles[profile_key] = profile
            
        except Exception as e:
            logger.error(f"Error updating performance profile: {e}")
    
    def _trigger_learning(self):
        """Trigger learning analysis when enough data is available."""
        try:
            if not self.learning_active:
                return
            
            # Analyze recent performance patterns
            recent_records = list(self.performance_records)[-self.learning_window:]
            
            # Group by model
            model_groups = defaultdict(list)
            for record in recent_records:
                model_groups[record.model_name].append(record)
            
            # Generate insights for each model
            for model_name, records in model_groups.items():
                if len(records) >= 10:  # Minimum sample size for learning
                    insights = self._analyze_performance_patterns(model_name, records)
                    for insight in insights:
                        self.learning_insights.append(insight)
            
            logger.debug(f"Learning triggered: {len(self.learning_insights)} total insights")
            
        except Exception as e:
            logger.error(f"Error in learning trigger: {e}")
    
    def _analyze_performance_patterns(self, model_name: str, records: List[PerformanceRecord]) -> List[LearningInsight]:
        """Analyze performance patterns to generate insights."""
        insights = []
        
        try:
            # Analyze condition-based performance
            condition_performance = defaultdict(list)
            for record in records:
                for condition_key, condition_value in record.market_conditions.items():
                    condition_performance[f"{condition_key}_{condition_value}"].append(record.accuracy)
            
            # Find significant performance differences
            for condition, accuracies in condition_performance.items():
                if len(accuracies) >= 5:  # Minimum sample size
                    avg_accuracy = statistics.mean(accuracies)
                    overall_avg = statistics.mean([r.accuracy for r in records])
                    
                    # Check for significant difference
                    if abs(avg_accuracy - overall_avg) > self.learning_threshold:
                        insight = LearningInsight(
                            model_name=model_name,
                            insight_type='condition_performance',
                            conditions={condition: avg_accuracy},
                            insight_data={
                                'condition_accuracy': avg_accuracy,
                                'overall_accuracy': overall_avg,
                                'difference': avg_accuracy - overall_avg,
                                'sample_size': len(accuracies)
                            },
                            confidence=min(1.0, len(accuracies) / 20.0),
                            timestamp=datetime.now()
                        )
                        insights.append(insight)
            
            # Analyze performance trend
            if len(records) >= 10:
                recent_accuracies = [r.accuracy for r in records[-5:]]
                older_accuracies = [r.accuracy for r in records[-10:-5]]
                
                recent_avg = statistics.mean(recent_accuracies)
                older_avg = statistics.mean(older_accuracies)
                
                if abs(recent_avg - older_avg) > self.learning_threshold:
                    trend_type = 'improving' if recent_avg > older_avg else 'declining'
                    insight = LearningInsight(
                        model_name=model_name,
                        insight_type='performance_trend',
                        conditions={},
                        insight_data={
                            'trend': trend_type,
                            'recent_accuracy': recent_avg,
                            'older_accuracy': older_avg,
                            'change': recent_avg - older_avg
                        },
                        confidence=0.7,
                        timestamp=datetime.now()
                    )
                    insights.append(insight)
            
        except Exception as e:
            logger.error(f"Error analyzing performance patterns for {model_name}: {e}")
        
        return insights
    
    def _find_similar_conditions(self, model_name: str, conditions: Dict[str, Any]) -> List[PerformanceRecord]:
        """Find performance records with similar conditions."""
        similar_records = []
        
        try:
            for record in self.performance_records:
                if record.model_name != model_name:
                    continue
                
                similarity = self._calculate_condition_similarity(conditions, record.market_conditions)
                if similarity > 0.7:  # Similarity threshold
                    similar_records.append(record)
            
            # Sort by similarity (most similar first)
            similar_records.sort(key=lambda r: self._calculate_condition_similarity(conditions, r.market_conditions), reverse=True)
            
        except Exception as e:
            logger.error(f"Error finding similar conditions: {e}")
        
        return similar_records
    
    def _calculate_condition_similarity(self, conditions1: Dict[str, Any], conditions2: Dict[str, Any]) -> float:
        """Calculate similarity between two condition dictionaries."""
        try:
            if not conditions1 or not conditions2:
                return 0.0
            
            # Get all unique keys
            all_keys = set(conditions1.keys()) | set(conditions2.keys())
            
            if not all_keys:
                return 1.0
            
            # Calculate similarity
            matches = 0
            for key in all_keys:
                val1 = conditions1.get(key)
                val2 = conditions2.get(key)
                
                if val1 == val2:
                    matches += 1
                elif val1 is None or val2 is None:
                    # Missing values reduce similarity
                    matches += 0.5
            
            return matches / len(all_keys)
            
        except Exception as e:
            logger.error(f"Error calculating condition similarity: {e}")
            return 0.0
    
    def _create_condition_key(self, conditions: Dict[str, Any]) -> str:
        """Create a key for condition grouping."""
        try:
            # Sort keys for consistent ordering
            sorted_items = sorted(conditions.items())
            return "_".join([f"{k}_{v}" for k, v in sorted_items])
        except Exception as e:
            logger.error(f"Error creating condition key: {e}")
            return "unknown"

# Global performance learner instance
_performance_learner = None

def get_performance_learner() -> ModelPerformanceLearner:
    """Get the global performance learner instance."""
    global _performance_learner
    if _performance_learner is None:
        _performance_learner = ModelPerformanceLearner()
    return _performance_learner