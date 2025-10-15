"""
Model Communication Hub
======================

Centralized system for inter-model communication and context sharing.
Enables models to share insights, coordinate decisions, and learn from each other.
"""

import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import threading

logger = logging.getLogger(__name__)

@dataclass
class MarketContext:
    """Market context information shared between models."""
    timestamp: datetime
    regime: str  # 'trending', 'ranging', 'volatile', 'calm'
    volatility_zscore: float
    correlation: float
    sector_dispersion: Dict[str, float]
    liquidity_score: float
    news_sentiment: float
    market_phase: str  # 'pre_market', 'open', 'mid_day', 'close', 'after_hours'

@dataclass
class ModelInsight:
    """Insight shared by a model."""
    model_name: str
    timestamp: datetime
    insight_type: str  # 'prediction', 'confidence', 'risk_assessment', 'market_analysis'
    data: Dict[str, Any]
    confidence: float
    reasoning: str

@dataclass
class ModelPerformance:
    """Model performance tracking."""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_return: float
    last_updated: datetime

class ModelCommunicationHub:
    """
    Central hub for model communication and coordination.
    
    Features:
    - Inter-model communication
    - Market context sharing
    - Performance tracking
    - Insight aggregation
    - Consensus building
    """
    
    def __init__(self, max_insights: int = 1000, max_performance_history: int = 100):
        """
        Initialize the communication hub.
        
        Args:
            max_insights: Maximum number of insights to store
            max_performance_history: Maximum performance records to keep
        """
        self.hub_name = "model_communication_hub"
        self.max_insights = max_insights
        self.max_performance_history = max_performance_history
        
        # Communication storage
        self.insights: deque = deque(maxlen=max_insights)
        self.market_context: Optional[MarketContext] = None
        self.model_performance: Dict[str, ModelPerformance] = {}
        self.performance_history: deque = deque(maxlen=max_performance_history)
        
        # Communication channels
        self.communication_channels: Dict[str, List[str]] = defaultdict(list)
        self.consensus_cache: Dict[str, Dict[str, Any]] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info(f"Model Communication Hub initialized: {self.hub_name}")
    
    def share_insight(self, model_name: str, insight_type: str, data: Dict[str, Any], 
                     confidence: float, reasoning: str) -> bool:
        """
        Share an insight with other models.
        
        Args:
            model_name: Name of the model sharing the insight
            insight_type: Type of insight (prediction, confidence, etc.)
            data: Insight data
            confidence: Confidence level (0-1)
            reasoning: Reasoning behind the insight
            
        Returns:
            True if insight was shared successfully
        """
        try:
            with self._lock:
                insight = ModelInsight(
                    model_name=model_name,
                    timestamp=datetime.now(),
                    insight_type=insight_type,
                    data=data,
                    confidence=confidence,
                    reasoning=reasoning
                )
                
                self.insights.append(insight)
                self._log_communication('insight_shared', asdict(insight))
                
                logger.debug(f"Insight shared by {model_name}: {insight_type}")
                return True
                
        except Exception as e:
            logger.error(f"Error sharing insight from {model_name}: {e}")
            return False
    
    def get_insights(self, model_name: Optional[str] = None, 
                    insight_type: Optional[str] = None,
                    time_window: Optional[timedelta] = None) -> List[ModelInsight]:
        """
        Get insights from the communication hub.
        
        Args:
            model_name: Filter by specific model (None for all)
            insight_type: Filter by insight type (None for all)
            time_window: Filter by time window (None for all)
            
        Returns:
            List of matching insights
        """
        try:
            with self._lock:
                filtered_insights = []
                cutoff_time = datetime.now() - time_window if time_window else None
                
                for insight in self.insights:
                    # Apply filters
                    if model_name and insight.model_name != model_name:
                        continue
                    if insight_type and insight.insight_type != insight_type:
                        continue
                    if cutoff_time and insight.timestamp < cutoff_time:
                        continue
                    
                    filtered_insights.append(insight)
                
                return filtered_insights
                
        except Exception as e:
            logger.error(f"Error getting insights: {e}")
            return []
    
    def update_market_context(self, context: MarketContext):
        """
        Update the shared market context.
        
        Args:
            context: New market context
        """
        try:
            with self._lock:
                self.market_context = context
                self._log_communication('market_context_update', asdict(context))
                logger.debug(f"Market context updated: {context.regime}")
                
        except Exception as e:
            logger.error(f"Error updating market context: {e}")
    
    def get_market_context(self) -> Optional[MarketContext]:
        """Get the current market context."""
        with self._lock:
            return self.market_context
    
    def update_model_performance(self, performance: ModelPerformance):
        """
        Update model performance metrics.
        
        Args:
            performance: Model performance data
        """
        try:
            with self._lock:
                self.model_performance[performance.model_name] = performance
                self.performance_history.append(performance)
                self._log_communication('performance_update', asdict(performance))
                logger.debug(f"Performance updated for {performance.model_name}")
                
        except Exception as e:
            logger.error(f"Error updating performance for {performance.model_name}: {e}")
    
    def get_model_performance(self, model_name: str) -> Optional[ModelPerformance]:
        """Get performance metrics for a specific model."""
        with self._lock:
            return self.model_performance.get(model_name)
    
    def get_top_performing_models(self, metric: str = 'accuracy', limit: int = 5) -> List[str]:
        """
        Get top performing models by specified metric.
        
        Args:
            metric: Performance metric to rank by
            limit: Maximum number of models to return
            
        Returns:
            List of model names ranked by performance
        """
        try:
            with self._lock:
                valid_models = []
                for model_name, performance in self.model_performance.items():
                    if hasattr(performance, metric):
                        value = getattr(performance, metric)
                        if value is not None:
                            valid_models.append((model_name, value))
                
                # Sort by metric value (descending)
                valid_models.sort(key=lambda x: x[1], reverse=True)
                return [model_name for model_name, _ in valid_models[:limit]]
                
        except Exception as e:
            logger.error(f"Error getting top performing models: {e}")
            return []
    
    def build_consensus(self, insight_type: str, time_window: timedelta = timedelta(minutes=5)) -> Dict[str, Any]:
        """
        Build consensus from recent insights of a specific type.
        
        Args:
            insight_type: Type of insights to build consensus from
            time_window: Time window for consensus building
            
        Returns:
            Consensus data including average confidence, agreement level, etc.
        """
        try:
            with self._lock:
                recent_insights = self.get_insights(insight_type=insight_type, time_window=time_window)
                
                if not recent_insights:
                    return {
                        'consensus_type': insight_type,
                        'agreement_level': 0.0,
                        'average_confidence': 0.0,
                        'participant_count': 0,
                        'timestamp': datetime.now().isoformat()
                    }
                
                # Calculate consensus metrics
                confidences = [insight.confidence for insight in recent_insights]
                avg_confidence = sum(confidences) / len(confidences)
                
                # Calculate agreement level (standard deviation of confidences)
                if len(confidences) > 1:
                    variance = sum((c - avg_confidence) ** 2 for c in confidences) / (len(confidences) - 1)
                    agreement_level = max(0.0, 1.0 - (variance ** 0.5))
                else:
                    agreement_level = 1.0
                
                consensus = {
                    'consensus_type': insight_type,
                    'agreement_level': agreement_level,
                    'average_confidence': avg_confidence,
                    'participant_count': len(recent_insights),
                    'participants': [insight.model_name for insight in recent_insights],
                    'timestamp': datetime.now().isoformat()
                }
                
                # Cache consensus
                cache_key = f"{insight_type}_{time_window.total_seconds()}"
                self.consensus_cache[cache_key] = consensus
                
                return consensus
                
        except Exception as e:
            logger.error(f"Error building consensus for {insight_type}: {e}")
            return {
                'consensus_type': insight_type,
                'agreement_level': 0.0,
                'average_confidence': 0.0,
                'participant_count': 0,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_communication_statistics(self) -> Dict[str, Any]:
        """Get communication hub statistics."""
        try:
            with self._lock:
                return {
                    'hub_name': self.hub_name,
                    'total_insights': len(self.insights),
                    'active_models': len(self.model_performance),
                    'market_context_available': self.market_context is not None,
                    'consensus_cache_size': len(self.consensus_cache),
                    'performance_history_size': len(self.performance_history),
                    'last_activity': self.insights[-1].timestamp.isoformat() if self.insights else None
                }
                
        except Exception as e:
            logger.error(f"Error getting communication statistics: {e}")
            return {'error': str(e)}
    
    def _log_communication(self, event_type: str, data: Dict[str, Any]):
        """Log communication events."""
        try:
            # Convert datetime objects to ISO format strings
            def convert_datetime(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                elif isinstance(obj, dict):
                    return {k: convert_datetime(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_datetime(item) for item in obj]
                else:
                    return obj
            
            converted_data = convert_datetime(data)
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'event_type': event_type,
                'data': converted_data
            }
            logger.debug(f"Communication event: {json.dumps(log_entry)}")
        except Exception as e:
            logger.error(f"Error logging communication event: {e}")

# Global communication hub instance
_communication_hub = None

def get_communication_hub() -> ModelCommunicationHub:
    """Get the global communication hub instance."""
    global _communication_hub
    if _communication_hub is None:
        _communication_hub = ModelCommunicationHub()
    return _communication_hub