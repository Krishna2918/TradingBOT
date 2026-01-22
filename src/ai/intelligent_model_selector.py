"""
Intelligent Model Selector
=========================

Dynamically selects optimal models based on market conditions, performance,
and specific requirements. Implements adaptive model selection strategies.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import statistics

logger = logging.getLogger(__name__)

class SelectionStrategy(Enum):
    """Model selection strategies."""
    PERFORMANCE_BASED = "performance_based"
    REGIME_AWARE = "regime_aware"
    CONSENSUS_BASED = "consensus_based"
    ADAPTIVE = "adaptive"
    FALLBACK = "fallback"

@dataclass
class ModelCapability:
    """Model capability profile."""
    model_name: str
    capabilities: List[str]  # ['technical_analysis', 'sentiment', 'fundamental', 'risk']
    performance_score: float
    reliability_score: float
    speed_score: float
    specialization: str  # 'trending', 'ranging', 'volatile', 'general'
    last_updated: datetime

@dataclass
class SelectionCriteria:
    """Criteria for model selection."""
    required_capabilities: List[str]
    min_performance_threshold: float
    max_models: int
    strategy: SelectionStrategy
    market_regime: Optional[str] = None
    time_constraint: Optional[float] = None  # seconds
    consensus_required: bool = False

@dataclass
class ModelSelection:
    """Result of model selection."""
    selected_models: List[str]
    selection_reasoning: str
    confidence: float
    strategy_used: SelectionStrategy
    selection_time: datetime
    criteria_met: bool

class IntelligentModelSelector:
    """
    Intelligent model selection system.
    
    Features:
    - Performance-based selection
    - Regime-aware selection
    - Consensus building
    - Adaptive strategies
    - Fallback mechanisms
    """
    
    def __init__(self):
        """Initialize the intelligent model selector."""
        self.selector_name = "intelligent_model_selector"
        self.model_capabilities: Dict[str, ModelCapability] = {}
        self.selection_history: List[ModelSelection] = []
        self.performance_tracker: Dict[str, List[float]] = {}
        
        # Selection weights
        self.performance_weight = 0.4
        self.reliability_weight = 0.3
        self.speed_weight = 0.2
        self.specialization_weight = 0.1
        
        logger.info(f"Intelligent Model Selector initialized: {self.selector_name}")
    
    def register_model(self, model_name: str, capabilities: List[str], 
                      specialization: str = "general") -> bool:
        """
        Register a model with its capabilities.
        
        Args:
            model_name: Name of the model
            capabilities: List of model capabilities
            specialization: Market specialization
            
        Returns:
            True if registration successful
        """
        try:
            capability = ModelCapability(
                model_name=model_name,
                capabilities=capabilities,
                performance_score=0.5,  # Default neutral score
                reliability_score=0.5,
                speed_score=0.5,
                specialization=specialization,
                last_updated=datetime.now()
            )
            
            self.model_capabilities[model_name] = capability
            self.performance_tracker[model_name] = []
            
            logger.info(f"Model registered: {model_name} with capabilities: {capabilities}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering model {model_name}: {e}")
            return False
    
    def update_model_performance(self, model_name: str, performance_score: float,
                               reliability_score: Optional[float] = None,
                               speed_score: Optional[float] = None) -> bool:
        """
        Update model performance metrics.
        
        Args:
            model_name: Name of the model
            performance_score: Performance score (0-1)
            reliability_score: Reliability score (0-1)
            speed_score: Speed score (0-1)
            
        Returns:
            True if update successful
        """
        try:
            if model_name not in self.model_capabilities:
                logger.warning(f"Model {model_name} not registered")
                return False
            
            # Update performance scores
            capability = self.model_capabilities[model_name]
            capability.performance_score = performance_score
            capability.last_updated = datetime.now()
            
            if reliability_score is not None:
                capability.reliability_score = reliability_score
            if speed_score is not None:
                capability.speed_score = speed_score
            
            # Track performance history
            self.performance_tracker[model_name].append(performance_score)
            if len(self.performance_tracker[model_name]) > 100:  # Keep last 100 scores
                self.performance_tracker[model_name] = self.performance_tracker[model_name][-100:]
            
            logger.debug(f"Performance updated for {model_name}: {performance_score}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating performance for {model_name}: {e}")
            return False
    
    def select_models_for_conditions(self, available_models: List[str]) -> List[str]:
        """
        Select optimal models based on current conditions.
        
        Args:
            available_models: List of available model names
            
        Returns:
            List of selected model names
        """
        try:
            # Filter to only registered models
            registered_models = [m for m in available_models if m in self.model_capabilities]
            
            if not registered_models:
                logger.warning("No registered models available")
                return []
            
            # Use adaptive strategy by default
            criteria = SelectionCriteria(
                required_capabilities=['technical_analysis'],  # Default requirement
                min_performance_threshold=0.3,
                max_models=min(3, len(registered_models)),
                strategy=SelectionStrategy.ADAPTIVE
            )
            
            selection = self._select_models(criteria, registered_models)
            return selection.selected_models
            
        except Exception as e:
            logger.error(f"Error selecting models: {e}")
            return available_models[:3] if available_models else []  # Fallback
    
    def select_models(self, criteria: SelectionCriteria) -> ModelSelection:
        """
        Select models based on specific criteria.
        
        Args:
            criteria: Selection criteria
            
        Returns:
            Model selection result
        """
        try:
            available_models = list(self.model_capabilities.keys())
            return self._select_models(criteria, available_models)
            
        except Exception as e:
            logger.error(f"Error selecting models with criteria: {e}")
            return ModelSelection(
                selected_models=[],
                selection_reasoning=f"Error: {e}",
                confidence=0.0,
                strategy_used=SelectionStrategy.FALLBACK,
                selection_time=datetime.now(),
                criteria_met=False
            )
    
    def _select_models(self, criteria: SelectionCriteria, available_models: List[str]) -> ModelSelection:
        """Internal method to select models based on criteria."""
        try:
            # Filter models by capabilities
            capable_models = []
            for model_name in available_models:
                capability = self.model_capabilities[model_name]
                if all(cap in capability.capabilities for cap in criteria.required_capabilities):
                    if capability.performance_score >= criteria.min_performance_threshold:
                        capable_models.append(model_name)
            
            if not capable_models:
                logger.warning("No models meet the criteria")
                return ModelSelection(
                    selected_models=[],
                    selection_reasoning="No models meet the criteria",
                    confidence=0.0,
                    strategy_used=SelectionStrategy.FALLBACK,
                    selection_time=datetime.now(),
                    criteria_met=False
                )
            
            # Apply selection strategy
            if criteria.strategy == SelectionStrategy.PERFORMANCE_BASED:
                selected = self._performance_based_selection(capable_models, criteria)
            elif criteria.strategy == SelectionStrategy.REGIME_AWARE:
                selected = self._regime_aware_selection(capable_models, criteria)
            elif criteria.strategy == SelectionStrategy.CONSENSUS_BASED:
                selected = self._consensus_based_selection(capable_models, criteria)
            elif criteria.strategy == SelectionStrategy.ADAPTIVE:
                selected = self._adaptive_selection(capable_models, criteria)
            else:
                selected = self._fallback_selection(capable_models, criteria)
            
            # Limit number of models
            if len(selected) > criteria.max_models:
                selected = selected[:criteria.max_models]
            
            # Create selection result
            selection = ModelSelection(
                selected_models=selected,
                selection_reasoning=f"Selected {len(selected)} models using {criteria.strategy.value} strategy",
                confidence=self._calculate_selection_confidence(selected),
                strategy_used=criteria.strategy,
                selection_time=datetime.now(),
                criteria_met=True
            )
            
            # Store selection history
            self.selection_history.append(selection)
            if len(self.selection_history) > 1000:  # Keep last 1000 selections
                self.selection_history = self.selection_history[-1000:]
            
            return selection
            
        except Exception as e:
            logger.error(f"Error in model selection: {e}")
            return ModelSelection(
                selected_models=[],
                selection_reasoning=f"Selection error: {e}",
                confidence=0.0,
                strategy_used=SelectionStrategy.FALLBACK,
                selection_time=datetime.now(),
                criteria_met=False
            )
    
    def _performance_based_selection(self, models: List[str], criteria: SelectionCriteria) -> List[str]:
        """Select models based on performance scores."""
        try:
            model_scores = []
            for model_name in models:
                capability = self.model_capabilities[model_name]
                score = capability.performance_score
                model_scores.append((model_name, score))
            
            # Sort by performance score (descending)
            model_scores.sort(key=lambda x: x[1], reverse=True)
            return [model_name for model_name, _ in model_scores]
            
        except Exception as e:
            logger.error(f"Error in performance-based selection: {e}")
            return models[:criteria.max_models]
    
    def _regime_aware_selection(self, models: List[str], criteria: SelectionCriteria) -> List[str]:
        """Select models based on market regime specialization."""
        try:
            if not criteria.market_regime:
                return self._performance_based_selection(models, criteria)
            
            # Prioritize models specialized for current regime
            regime_models = []
            general_models = []
            
            for model_name in models:
                capability = self.model_capabilities[model_name]
                if capability.specialization == criteria.market_regime:
                    regime_models.append(model_name)
                elif capability.specialization == "general":
                    general_models.append(model_name)
            
            # Combine regime-specific and general models
            selected = regime_models + general_models
            return selected[:criteria.max_models]
            
        except Exception as e:
            logger.error(f"Error in regime-aware selection: {e}")
            return models[:criteria.max_models]
    
    def _consensus_based_selection(self, models: List[str], criteria: SelectionCriteria) -> List[str]:
        """Select models that can build consensus."""
        try:
            # For consensus, we want models with good reliability scores
            model_scores = []
            for model_name in models:
                capability = self.model_capabilities[model_name]
                # Weight reliability more heavily for consensus
                score = (capability.reliability_score * 0.6 + 
                        capability.performance_score * 0.4)
                model_scores.append((model_name, score))
            
            # Sort by consensus score (descending)
            model_scores.sort(key=lambda x: x[1], reverse=True)
            return [model_name for model_name, _ in model_scores]
            
        except Exception as e:
            logger.error(f"Error in consensus-based selection: {e}")
            return models[:criteria.max_models]
    
    def _adaptive_selection(self, models: List[str], criteria: SelectionCriteria) -> List[str]:
        """Adaptive selection combining multiple factors."""
        try:
            model_scores = []
            for model_name in models:
                capability = self.model_capabilities[model_name]
                
                # Calculate composite score
                composite_score = (
                    capability.performance_score * self.performance_weight +
                    capability.reliability_score * self.reliability_weight +
                    capability.speed_score * self.speed_weight +
                    (1.0 if capability.specialization == criteria.market_regime else 0.5) * self.specialization_weight
                )
                
                model_scores.append((model_name, composite_score))
            
            # Sort by composite score (descending)
            model_scores.sort(key=lambda x: x[1], reverse=True)
            return [model_name for model_name, _ in model_scores]
            
        except Exception as e:
            logger.error(f"Error in adaptive selection: {e}")
            return models[:criteria.max_models]
    
    def _fallback_selection(self, models: List[str], criteria: SelectionCriteria) -> List[str]:
        """Fallback selection method."""
        return models[:criteria.max_models]
    
    def _calculate_selection_confidence(self, selected_models: List[str]) -> float:
        """Calculate confidence in the selection."""
        try:
            if not selected_models:
                return 0.0
            
            # Calculate average performance of selected models
            total_performance = 0.0
            for model_name in selected_models:
                if model_name in self.model_capabilities:
                    total_performance += self.model_capabilities[model_name].performance_score
            
            avg_performance = total_performance / len(selected_models)
            
            # Boost confidence if we have multiple models (diversity)
            diversity_bonus = min(0.2, (len(selected_models) - 1) * 0.1)
            
            return min(1.0, avg_performance + diversity_bonus)
            
        except Exception as e:
            logger.error(f"Error calculating selection confidence: {e}")
            return 0.5
    
    def get_selection_statistics(self) -> Dict[str, Any]:
        """Get selection statistics."""
        try:
            if not self.selection_history:
                return {
                    'total_selections': 0,
                    'average_confidence': 0.0,
                    'strategy_usage': {},
                    'most_selected_models': []
                }
            
            # Calculate statistics
            total_selections = len(self.selection_history)
            avg_confidence = statistics.mean([s.confidence for s in self.selection_history])
            
            # Strategy usage
            strategy_usage = {}
            for selection in self.selection_history:
                strategy = selection.strategy_used.value
                strategy_usage[strategy] = strategy_usage.get(strategy, 0) + 1
            
            # Most selected models
            model_counts = {}
            for selection in self.selection_history:
                for model_name in selection.selected_models:
                    model_counts[model_name] = model_counts.get(model_name, 0) + 1
            
            most_selected = sorted(model_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            return {
                'total_selections': total_selections,
                'average_confidence': avg_confidence,
                'strategy_usage': strategy_usage,
                'most_selected_models': most_selected,
                'registered_models': len(self.model_capabilities)
            }
            
        except Exception as e:
            logger.error(f"Error getting selection statistics: {e}")
            return {'error': str(e)}

# Global model selector instance
_model_selector = None

def get_model_selector() -> IntelligentModelSelector:
    """Get the global model selector instance."""
    global _model_selector
    if _model_selector is None:
        _model_selector = IntelligentModelSelector()
    return _model_selector