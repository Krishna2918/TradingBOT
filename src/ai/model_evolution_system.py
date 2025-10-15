"""Model Evolution System for Automated Model Improvement"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import json
import os
from enum import Enum

logger = logging.getLogger(__name__)

class EvolutionTrigger(Enum):
    """Types of evolution triggers."""
    PERFORMANCE_DECLINE = "performance_decline"
    MARKET_REGIME_CHANGE = "market_regime_change"
    SCHEDULED_RETRAIN = "scheduled_retrain"
    NEW_DATA_AVAILABLE = "new_data_available"
    MANUAL_REQUEST = "manual_request"

class EvolutionStrategy(Enum):
    """Types of evolution strategies."""
    HYPERPARAMETER_TUNING = "hyperparameter_tuning"
    ARCHITECTURE_SEARCH = "architecture_search"
    FEATURE_ENGINEERING = "feature_engineering"
    ENSEMBLE_ADJUSTMENT = "ensemble_adjustment"
    FULL_RETRAIN = "full_retrain"

@dataclass
class ModelPerformance:
    """Model performance metrics."""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    timestamp: datetime
    market_conditions: Dict[str, Any]

@dataclass
class EvolutionEvent:
    """An evolution event."""
    trigger: EvolutionTrigger
    strategy: EvolutionStrategy
    model_name: str
    timestamp: datetime
    reason: str
    parameters: Dict[str, Any]
    expected_improvement: float

class PerformanceTracker:
    """Tracks model performance over time."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.performance_history = {}
        self.current_performance = {}
        
    def update_performance(self, model_name: str, performance: ModelPerformance):
        """Update performance for a model."""
        if model_name not in self.performance_history:
            self.performance_history[model_name] = []
        
        self.performance_history[model_name].append(performance)
        self.current_performance[model_name] = performance
        
        # Keep only recent performance data
        if len(self.performance_history[model_name]) > self.window_size:
            self.performance_history[model_name] = self.performance_history[model_name][-self.window_size:]
    
    def get_performance_trend(self, model_name: str, metric: str = 'accuracy') -> float:
        """Get performance trend for a model."""
        if model_name not in self.performance_history or len(self.performance_history[model_name]) < 2:
            return 0.0
        
        recent_performance = self.performance_history[model_name][-10:]  # Last 10 samples
        if len(recent_performance) < 2:
            return 0.0
        
        # Calculate trend using linear regression
        x = np.arange(len(recent_performance))
        y = [getattr(perf, metric) for perf in recent_performance]
        
        if len(y) < 2:
            return 0.0
        
        # Simple linear trend calculation
        trend = np.polyfit(x, y, 1)[0]
        return trend
    
    def detect_performance_decline(self, model_name: str, threshold: float = -0.05) -> bool:
        """Detect if model performance is declining."""
        trend = self.get_performance_trend(model_name, 'accuracy')
        return trend < threshold
    
    def get_best_performance(self, model_name: str, metric: str = 'accuracy') -> float:
        """Get best performance for a model."""
        if model_name not in self.performance_history:
            return 0.0
        
        performances = [getattr(perf, metric) for perf in self.performance_history[model_name]]
        return max(performances) if performances else 0.0

class EvolutionTriggerManager:
    """Manages evolution triggers and conditions."""
    
    def __init__(self):
        self.trigger_conditions = {
            EvolutionTrigger.PERFORMANCE_DECLINE: {
                'threshold': -0.05,
                'window_size': 10,
                'enabled': True
            },
            EvolutionTrigger.MARKET_REGIME_CHANGE: {
                'regime_change_threshold': 0.3,
                'enabled': True
            },
            EvolutionTrigger.SCHEDULED_RETRAIN: {
                'interval_hours': 24,
                'enabled': True
            },
            EvolutionTrigger.NEW_DATA_AVAILABLE: {
                'data_threshold': 1000,
                'enabled': True
            }
        }
        self.last_trigger_times = {}
        self.market_regime_history = []
        
    def check_performance_decline_trigger(self, model_name: str, 
                                        performance_tracker: PerformanceTracker) -> bool:
        """Check if performance decline should trigger evolution."""
        if not self.trigger_conditions[EvolutionTrigger.PERFORMANCE_DECLINE]['enabled']:
            return False
        
        return performance_tracker.detect_performance_decline(
            model_name, 
            self.trigger_conditions[EvolutionTrigger.PERFORMANCE_DECLINE]['threshold']
        )
    
    def check_market_regime_change_trigger(self, current_regime: str) -> bool:
        """Check if market regime change should trigger evolution."""
        if not self.trigger_conditions[EvolutionTrigger.MARKET_REGIME_CHANGE]['enabled']:
            return False
        
        self.market_regime_history.append({
            'regime': current_regime,
            'timestamp': datetime.now()
        })
        
        # Keep only recent regime history
        if len(self.market_regime_history) > 10:
            self.market_regime_history = self.market_regime_history[-10:]
        
        # Check for regime change
        if len(self.market_regime_history) >= 2:
            recent_regimes = [r['regime'] for r in self.market_regime_history[-3:]]
            if len(set(recent_regimes)) > 1:  # Multiple regimes in recent history
                return True
        
        return False
    
    def check_scheduled_retrain_trigger(self, model_name: str) -> bool:
        """Check if scheduled retrain should trigger evolution."""
        if not self.trigger_conditions[EvolutionTrigger.SCHEDULED_RETRAIN]['enabled']:
            return False
        
        interval_hours = self.trigger_conditions[EvolutionTrigger.SCHEDULED_RETRAIN]['interval_hours']
        last_trigger = self.last_trigger_times.get(model_name)
        
        if last_trigger is None:
            return True  # First time
        
        time_since_last = datetime.now() - last_trigger
        return time_since_last.total_seconds() >= interval_hours * 3600
    
    def check_new_data_trigger(self, new_data_count: int) -> bool:
        """Check if new data should trigger evolution."""
        if not self.trigger_conditions[EvolutionTrigger.NEW_DATA_AVAILABLE]['enabled']:
            return False
        
        threshold = self.trigger_conditions[EvolutionTrigger.NEW_DATA_AVAILABLE]['data_threshold']
        return new_data_count >= threshold
    
    def record_trigger(self, model_name: str, trigger: EvolutionTrigger):
        """Record that a trigger occurred."""
        self.last_trigger_times[model_name] = datetime.now()

class EvolutionStrategySelector:
    """Selects appropriate evolution strategy based on conditions."""
    
    def __init__(self):
        self.strategy_conditions = {
            EvolutionStrategy.HYPERPARAMETER_TUNING: {
                'performance_decline_threshold': -0.1,
                'max_iterations': 50
            },
            EvolutionStrategy.ARCHITECTURE_SEARCH: {
                'performance_decline_threshold': -0.2,
                'max_architectures': 10
            },
            EvolutionStrategy.FEATURE_ENGINEERING: {
                'feature_importance_threshold': 0.1,
                'max_features': 20
            },
            EvolutionStrategy.ENSEMBLE_ADJUSTMENT: {
                'ensemble_size_threshold': 3,
                'max_models': 10
            },
            EvolutionStrategy.FULL_RETRAIN: {
                'performance_decline_threshold': -0.3,
                'data_availability_threshold': 10000
            }
        }
    
    def select_strategy(self, trigger: EvolutionTrigger, model_name: str,
                       performance_tracker: PerformanceTracker,
                       market_conditions: Dict[str, Any]) -> EvolutionStrategy:
        """Select appropriate evolution strategy."""
        
        if trigger == EvolutionTrigger.PERFORMANCE_DECLINE:
            # Check severity of performance decline
            current_perf = performance_tracker.current_performance.get(model_name)
            if current_perf:
                accuracy_decline = performance_tracker.get_performance_trend(model_name, 'accuracy')
                
                if accuracy_decline < -0.3:
                    return EvolutionStrategy.FULL_RETRAIN
                elif accuracy_decline < -0.2:
                    return EvolutionStrategy.ARCHITECTURE_SEARCH
                elif accuracy_decline < -0.1:
                    return EvolutionStrategy.HYPERPARAMETER_TUNING
                else:
                    return EvolutionStrategy.FEATURE_ENGINEERING
        
        elif trigger == EvolutionTrigger.MARKET_REGIME_CHANGE:
            # Market regime change - adjust ensemble or retrain
            regime = market_conditions.get('regime', 'unknown')
            if regime in ['volatile', 'extreme']:
                return EvolutionStrategy.ENSEMBLE_ADJUSTMENT
            else:
                return EvolutionStrategy.HYPERPARAMETER_TUNING
        
        elif trigger == EvolutionTrigger.SCHEDULED_RETRAIN:
            # Scheduled retrain - use moderate strategy
            return EvolutionStrategy.HYPERPARAMETER_TUNING
        
        elif trigger == EvolutionTrigger.NEW_DATA_AVAILABLE:
            # New data available - retrain with new data
            return EvolutionStrategy.FULL_RETRAIN
        
        else:
            # Default strategy
            return EvolutionStrategy.HYPERPARAMETER_TUNING

class ModelEvolutionEngine:
    """Main engine for model evolution."""
    
    def __init__(self):
        self.performance_tracker = PerformanceTracker()
        self.trigger_manager = EvolutionTriggerManager()
        self.strategy_selector = EvolutionStrategySelector()
        self.evolution_history = []
        self.model_registry = {}
        self.evolution_callbacks = {}
        
    def register_model(self, model_name: str, model_instance: Any, 
                      evolution_callback: Callable = None):
        """Register a model for evolution."""
        self.model_registry[model_name] = {
            'instance': model_instance,
            'registered_at': datetime.now(),
            'evolution_count': 0
        }
        
        if evolution_callback:
            self.evolution_callbacks[model_name] = evolution_callback
    
    def update_performance(self, model_name: str, performance: ModelPerformance):
        """Update model performance and check for evolution triggers."""
        self.performance_tracker.update_performance(model_name, performance)
        
        # Check for evolution triggers
        evolution_event = self._check_evolution_triggers(model_name, performance.market_conditions)
        
        if evolution_event:
            self._execute_evolution(evolution_event)
    
    def _check_evolution_triggers(self, model_name: str, 
                                market_conditions: Dict[str, Any]) -> Optional[EvolutionEvent]:
        """Check if any evolution triggers are activated."""
        
        # Check performance decline
        if self.trigger_manager.check_performance_decline_trigger(model_name, self.performance_tracker):
            strategy = self.strategy_selector.select_strategy(
                EvolutionTrigger.PERFORMANCE_DECLINE, model_name, 
                self.performance_tracker, market_conditions
            )
            return EvolutionEvent(
                trigger=EvolutionTrigger.PERFORMANCE_DECLINE,
                strategy=strategy,
                model_name=model_name,
                timestamp=datetime.now(),
                reason="Performance decline detected",
                parameters={'decline_threshold': -0.05},
                expected_improvement=0.1
            )
        
        # Check market regime change
        current_regime = market_conditions.get('regime', 'unknown')
        if self.trigger_manager.check_market_regime_change_trigger(current_regime):
            strategy = self.strategy_selector.select_strategy(
                EvolutionTrigger.MARKET_REGIME_CHANGE, model_name,
                self.performance_tracker, market_conditions
            )
            return EvolutionEvent(
                trigger=EvolutionTrigger.MARKET_REGIME_CHANGE,
                strategy=strategy,
                model_name=model_name,
                timestamp=datetime.now(),
                reason="Market regime change detected",
                parameters={'new_regime': current_regime},
                expected_improvement=0.05
            )
        
        # Check scheduled retrain
        if self.trigger_manager.check_scheduled_retrain_trigger(model_name):
            strategy = self.strategy_selector.select_strategy(
                EvolutionTrigger.SCHEDULED_RETRAIN, model_name,
                self.performance_tracker, market_conditions
            )
            return EvolutionEvent(
                trigger=EvolutionTrigger.SCHEDULED_RETRAIN,
                strategy=strategy,
                model_name=model_name,
                timestamp=datetime.now(),
                reason="Scheduled retrain time reached",
                parameters={'interval_hours': 24},
                expected_improvement=0.02
            )
        
        return None
    
    def _execute_evolution(self, evolution_event: EvolutionEvent):
        """Execute model evolution."""
        logger.info(f"Executing evolution for {evolution_event.model_name}: {evolution_event.strategy.value}")
        
        # Record evolution event
        self.evolution_history.append(evolution_event)
        
        # Update model registry
        if evolution_event.model_name in self.model_registry:
            self.model_registry[evolution_event.model_name]['evolution_count'] += 1
        
        # Execute evolution callback if available
        if evolution_event.model_name in self.evolution_callbacks:
            try:
                callback = self.evolution_callbacks[evolution_event.model_name]
                result = callback(evolution_event)
                logger.info(f"Evolution callback executed for {evolution_event.model_name}: {result}")
            except Exception as e:
                logger.error(f"Evolution callback failed for {evolution_event.model_name}: {e}")
        
        # Record trigger time
        self.trigger_manager.record_trigger(evolution_event.model_name, evolution_event.trigger)
    
    def trigger_manual_evolution(self, model_name: str, strategy: EvolutionStrategy,
                               reason: str = "Manual request") -> bool:
        """Manually trigger model evolution."""
        if model_name not in self.model_registry:
            logger.warning(f"Model {model_name} not registered for evolution")
            return False
        
        evolution_event = EvolutionEvent(
            trigger=EvolutionTrigger.MANUAL_REQUEST,
            strategy=strategy,
            model_name=model_name,
            timestamp=datetime.now(),
            reason=reason,
            parameters={},
            expected_improvement=0.05
        )
        
        self._execute_evolution(evolution_event)
        return True
    
    def get_evolution_statistics(self) -> Dict[str, Any]:
        """Get evolution statistics."""
        stats = {
            'total_models_registered': len(self.model_registry),
            'total_evolution_events': len(self.evolution_history),
            'evolution_by_trigger': {},
            'evolution_by_strategy': {},
            'model_evolution_counts': {},
            'recent_evolutions': []
        }
        
        # Count evolutions by trigger
        for event in self.evolution_history:
            trigger = event.trigger.value
            stats['evolution_by_trigger'][trigger] = stats['evolution_by_trigger'].get(trigger, 0) + 1
        
        # Count evolutions by strategy
        for event in self.evolution_history:
            strategy = event.strategy.value
            stats['evolution_by_strategy'][strategy] = stats['evolution_by_strategy'].get(strategy, 0) + 1
        
        # Count evolutions per model
        for model_name, model_info in self.model_registry.items():
            stats['model_evolution_counts'][model_name] = model_info['evolution_count']
        
        # Recent evolutions
        recent_events = sorted(self.evolution_history, key=lambda x: x.timestamp, reverse=True)[:10]
        stats['recent_evolutions'] = [
            {
                'model_name': event.model_name,
                'trigger': event.trigger.value,
                'strategy': event.strategy.value,
                'timestamp': event.timestamp.isoformat(),
                'reason': event.reason
            }
            for event in recent_events
        ]
        
        return stats
    
    def get_model_evolution_history(self, model_name: str) -> List[Dict[str, Any]]:
        """Get evolution history for a specific model."""
        model_events = [event for event in self.evolution_history if event.model_name == model_name]
        
        return [
            {
                'trigger': event.trigger.value,
                'strategy': event.strategy.value,
                'timestamp': event.timestamp.isoformat(),
                'reason': event.reason,
                'parameters': event.parameters,
                'expected_improvement': event.expected_improvement
            }
            for event in model_events
        ]

# Global instance
_evolution_engine = None

def get_evolution_engine() -> ModelEvolutionEngine:
    """Get the global evolution engine instance."""
    global _evolution_engine
    if _evolution_engine is None:
        _evolution_engine = ModelEvolutionEngine()
    return _evolution_engine