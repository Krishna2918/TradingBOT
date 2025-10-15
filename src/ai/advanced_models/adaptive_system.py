"""
Adaptive System for Advanced AI Models

This module provides adaptive system capabilities for automatically
adjusting model parameters, weights, and strategies based on performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import logging
from datetime import datetime, timedelta
import threading
import time
from collections import defaultdict, deque
import warnings

logger = logging.getLogger(__name__)

class AdaptiveSystem:
    """
    Adaptive system for automatically adjusting AI model parameters and strategies.
    """
    
    def __init__(
        self,
        system_name: str = "adaptive_system",
        adaptation_interval: int = 300,  # 5 minutes
        learning_rate: float = 0.1,
        adaptation_strategies: Optional[List[str]] = None
    ):
        """
        Initialize adaptive system.
        
        Args:
            system_name: Name for the adaptive system
            adaptation_interval: Interval for adaptation in seconds
            learning_rate: Learning rate for parameter updates
            adaptation_strategies: List of adaptation strategies to use
        """
        self.system_name = system_name
        self.adaptation_interval = adaptation_interval
        self.learning_rate = learning_rate
        self.adaptation_strategies = adaptation_strategies or [
            'performance_based_weighting',
            'dynamic_parameter_adjustment',
            'model_selection_optimization',
            'ensemble_rebalancing'
        ]
        
        # System components
        self.adaptable_components = {}
        self.performance_history = deque(maxlen=1000)
        self.adaptation_history = deque(maxlen=1000)
        
        # Adaptation parameters
        self.adaptation_parameters = {
            'min_performance_threshold': 0.7,
            'max_adaptation_rate': 0.2,
            'performance_window': 50,
            'stability_threshold': 0.05
        }
        
        # Control flags
        self.adaptation_active = False
        self.adaptation_thread = None
        
        # Performance metrics
        self.adaptation_metrics = {
            'total_adaptations': 0,
            'successful_adaptations': 0,
            'performance_improvements': 0,
            'average_adaptation_time': 0.0
        }
        
        logger.info(f"Adaptive System initialized: {system_name}")
    
    def register_component(
        self,
        component_name: str,
        component: Any,
        adaptation_type: str = 'weight_adjustment',
        adaptation_params: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a component for adaptation.
        
        Args:
            component_name: Name for the component
            component: Component object
            adaptation_type: Type of adaptation to apply
            adaptation_params: Parameters for adaptation
        """
        self.adaptable_components[component_name] = {
            'component': component,
            'type': adaptation_type,
            'params': adaptation_params or {},
            'registered_at': datetime.now(),
            'adaptation_count': 0,
            'last_adaptation': None,
            'performance_history': deque(maxlen=100)
        }
        
        logger.info(f"Registered component '{component_name}' for adaptation")
    
    def unregister_component(self, component_name: str) -> None:
        """Unregister a component from adaptation."""
        if component_name in self.adaptable_components:
            del self.adaptable_components[component_name]
            logger.info(f"Unregistered component '{component_name}' from adaptation")
        else:
            logger.warning(f"Component '{component_name}' not found")
    
    def start_adaptation(self) -> None:
        """Start continuous adaptation."""
        if self.adaptation_active:
            logger.warning("Adaptation is already active")
            return
        
        self.adaptation_active = True
        self.adaptation_thread = threading.Thread(target=self._adaptation_loop, daemon=True)
        self.adaptation_thread.start()
        
        logger.info("Adaptive system started")
    
    def stop_adaptation(self) -> None:
        """Stop continuous adaptation."""
        self.adaptation_active = False
        if self.adaptation_thread:
            self.adaptation_thread.join(timeout=5)
        
        logger.info("Adaptive system stopped")
    
    def _adaptation_loop(self) -> None:
        """Main adaptation loop."""
        while self.adaptation_active:
            try:
                # Perform adaptation for each component
                for component_name, component_info in self.adaptable_components.items():
                    self._adapt_component(component_name, component_info)
                
                # Sleep for adaptation interval
                time.sleep(self.adaptation_interval)
                
            except Exception as e:
                logger.error(f"Error in adaptation loop: {e}")
                time.sleep(self.adaptation_interval)
    
    def _adapt_component(self, component_name: str, component_info: Dict[str, Any]) -> None:
        """Adapt a specific component."""
        try:
            adaptation_type = component_info['type']
            component = component_info['component']
            
            # Check if adaptation is needed
            if not self._should_adapt(component_name, component_info):
                return
            
            # Perform adaptation based on type
            if adaptation_type == 'weight_adjustment':
                result = self._adapt_weights(component_name, component, component_info)
            elif adaptation_type == 'parameter_adjustment':
                result = self._adapt_parameters(component_name, component, component_info)
            elif adaptation_type == 'model_selection':
                result = self._adapt_model_selection(component_name, component, component_info)
            elif adaptation_type == 'ensemble_rebalancing':
                result = self._adapt_ensemble_rebalancing(component_name, component, component_info)
            else:
                logger.warning(f"Unknown adaptation type: {adaptation_type}")
                return
            
            # Update component info
            if result.get('success', False):
                component_info['adaptation_count'] += 1
                component_info['last_adaptation'] = datetime.now()
                
                # Update performance history
                if 'performance_change' in result:
                    component_info['performance_history'].append(result['performance_change'])
                
                # Store adaptation record
                adaptation_record = {
                    'component_name': component_name,
                    'adaptation_type': adaptation_type,
                    'result': result,
                    'timestamp': datetime.now()
                }
                self.adaptation_history.append(adaptation_record)
                
                # Update metrics
                self.adaptation_metrics['total_adaptations'] += 1
                self.adaptation_metrics['successful_adaptations'] += 1
                
                if result.get('performance_change', 0) > 0:
                    self.adaptation_metrics['performance_improvements'] += 1
                
                logger.info(f"Adapted component '{component_name}': {result.get('message', 'Success')}")
        
        except Exception as e:
            logger.error(f"Error adapting component '{component_name}': {e}")
            self.adaptation_metrics['total_adaptations'] += 1
    
    def _should_adapt(self, component_name: str, component_info: Dict[str, Any]) -> bool:
        """Check if a component should be adapted."""
        try:
            # Check if enough time has passed since last adaptation
            last_adaptation = component_info.get('last_adaptation')
            if last_adaptation:
                time_since_adaptation = datetime.now() - last_adaptation
                if time_since_adaptation.total_seconds() < self.adaptation_interval:
                    return False
            
            # Check performance history
            performance_history = component_info['performance_history']
            if len(performance_history) < 10:  # Need enough history
                return True
            
            # Check if performance is below threshold
            recent_performance = list(performance_history)[-10:]
            avg_performance = np.mean(recent_performance)
            
            if avg_performance < self.adaptation_parameters['min_performance_threshold']:
                return True
            
            # Check for performance degradation
            if len(performance_history) >= 20:
                older_performance = list(performance_history)[-20:-10]
                recent_performance = list(performance_history)[-10:]
                
                older_avg = np.mean(older_performance)
                recent_avg = np.mean(recent_performance)
                
                if older_avg - recent_avg > self.adaptation_parameters['stability_threshold']:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking adaptation need: {e}")
            return False
    
    def _adapt_weights(
        self,
        component_name: str,
        component: Any,
        component_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adapt component weights based on performance."""
        try:
            # Get current performance
            performance_history = component_info['performance_history']
            if not performance_history:
                return {'success': False, 'message': 'No performance history'}
            
            recent_performance = list(performance_history)[-10:]
            current_performance = np.mean(recent_performance)
            
            # Calculate performance trend
            if len(performance_history) >= 20:
                older_performance = list(performance_history)[-20:-10]
                performance_trend = np.mean(recent_performance) - np.mean(older_performance)
            else:
                performance_trend = 0.0
            
            # Adjust weights based on performance
            if hasattr(component, 'model_weights'):
                # For ensemble models
                old_weights = component.model_weights.copy()
                new_weights = {}
                
                for model_name, weight in old_weights.items():
                    # Adjust weight based on performance trend
                    adjustment = self.learning_rate * performance_trend
                    new_weight = max(0.1, min(2.0, weight + adjustment))
                    new_weights[model_name] = new_weight
                
                # Normalize weights
                total_weight = sum(new_weights.values())
                if total_weight > 0:
                    new_weights = {k: v / total_weight for k, v in new_weights.items()}
                    component.model_weights = new_weights
                
                return {
                    'success': True,
                    'message': f'Adjusted weights for {len(new_weights)} models',
                    'old_weights': old_weights,
                    'new_weights': new_weights,
                    'performance_change': performance_trend
                }
            
            elif hasattr(component, 'weights'):
                # For other weighted components
                old_weights = component.weights.copy() if hasattr(component.weights, 'copy') else component.weights
                
                # Simple weight adjustment
                adjustment = self.learning_rate * performance_trend
                if isinstance(old_weights, dict):
                    new_weights = {k: max(0.1, min(2.0, v + adjustment)) for k, v in old_weights.items()}
                else:
                    new_weights = max(0.1, min(2.0, old_weights + adjustment))
                
                component.weights = new_weights
                
                return {
                    'success': True,
                    'message': 'Adjusted component weights',
                    'old_weights': old_weights,
                    'new_weights': new_weights,
                    'performance_change': performance_trend
                }
            
            else:
                return {'success': False, 'message': 'Component does not support weight adjustment'}
        
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _adapt_parameters(
        self,
        component_name: str,
        component: Any,
        component_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adapt component parameters based on performance."""
        try:
            # Get current performance
            performance_history = component_info['performance_history']
            if not performance_history:
                return {'success': False, 'message': 'No performance history'}
            
            recent_performance = list(performance_history)[-10:]
            current_performance = np.mean(recent_performance)
            
            # Calculate performance trend
            if len(performance_history) >= 20:
                older_performance = list(performance_history)[-20:-10]
                performance_trend = np.mean(recent_performance) - np.mean(older_performance)
            else:
                performance_trend = 0.0
            
            # Get adaptation parameters
            adaptation_params = component_info.get('params', {})
            parameter_ranges = adaptation_params.get('parameter_ranges', {})
            
            if not parameter_ranges:
                return {'success': False, 'message': 'No parameter ranges defined'}
            
            # Adjust parameters
            parameter_changes = {}
            for param_name, (min_val, max_val) in parameter_ranges.items():
                if hasattr(component, param_name):
                    current_value = getattr(component, param_name)
                    
                    # Calculate adjustment
                    adjustment = self.learning_rate * performance_trend * (max_val - min_val)
                    new_value = max(min_val, min(max_val, current_value + adjustment))
                    
                    setattr(component, param_name, new_value)
                    parameter_changes[param_name] = {
                        'old_value': current_value,
                        'new_value': new_value,
                        'adjustment': adjustment
                    }
            
            return {
                'success': True,
                'message': f'Adjusted {len(parameter_changes)} parameters',
                'parameter_changes': parameter_changes,
                'performance_change': performance_trend
            }
        
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _adapt_model_selection(
        self,
        component_name: str,
        component: Any,
        component_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adapt model selection based on performance."""
        try:
            # This would implement model selection adaptation
            # For now, return a placeholder result
            
            return {
                'success': True,
                'message': 'Model selection adaptation (placeholder)',
                'performance_change': 0.0
            }
        
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _adapt_ensemble_rebalancing(
        self,
        component_name: str,
        component: Any,
        component_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adapt ensemble rebalancing based on performance."""
        try:
            # This would implement ensemble rebalancing
            # For now, return a placeholder result
            
            return {
                'success': True,
                'message': 'Ensemble rebalancing adaptation (placeholder)',
                'performance_change': 0.0
            }
        
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def log_performance(
        self,
        component_name: str,
        performance_score: float,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Log performance for a component.
        
        Args:
            component_name: Name of the component
            performance_score: Performance score (0-1)
            timestamp: Timestamp for the performance (default: now)
        """
        if component_name not in self.adaptable_components:
            logger.warning(f"Component '{component_name}' not registered for adaptation")
            return
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # Store performance record
        performance_record = {
            'component_name': component_name,
            'performance_score': performance_score,
            'timestamp': timestamp
        }
        self.performance_history.append(performance_record)
        
        # Update component performance history
        self.adaptable_components[component_name]['performance_history'].append(performance_score)
    
    def get_adaptation_statistics(self) -> Dict[str, Any]:
        """Get adaptation statistics."""
        return {
            'system_name': self.system_name,
            'adaptation_active': self.adaptation_active,
            'adaptation_interval': self.adaptation_interval,
            'learning_rate': self.learning_rate,
            'total_components': len(self.adaptable_components),
            'component_names': list(self.adaptable_components.keys()),
            'adaptation_metrics': self.adaptation_metrics.copy(),
            'adaptation_parameters': self.adaptation_parameters.copy(),
            'total_performance_records': len(self.performance_history),
            'total_adaptation_records': len(self.adaptation_history)
        }
    
    def get_component_performance(self, component_name: str) -> Dict[str, Any]:
        """Get performance statistics for a specific component."""
        if component_name not in self.adaptable_components:
            return {'error': f'Component {component_name} not found'}
        
        component_info = self.adaptable_components[component_name]
        performance_history = component_info['performance_history']
        
        if not performance_history:
            return {
                'component_name': component_name,
                'performance_records': 0,
                'message': 'No performance history available'
            }
        
        performance_scores = list(performance_history)
        
        return {
            'component_name': component_name,
            'performance_records': len(performance_scores),
            'current_performance': performance_scores[-1] if performance_scores else 0.0,
            'average_performance': np.mean(performance_scores),
            'performance_trend': np.mean(performance_scores[-10:]) - np.mean(performance_scores[-20:-10]) if len(performance_scores) >= 20 else 0.0,
            'adaptation_count': component_info['adaptation_count'],
            'last_adaptation': component_info['last_adaptation'].isoformat() if component_info['last_adaptation'] else None,
            'registered_at': component_info['registered_at'].isoformat()
        }


class SystemOptimizer:
    """
    System-wide optimizer for coordinating multiple adaptive components.
    """
    
    def __init__(self, optimizer_name: str = "system_optimizer"):
        """
        Initialize system optimizer.
        
        Args:
            optimizer_name: Name for the optimizer
        """
        self.optimizer_name = optimizer_name
        self.adaptive_systems = {}
        
        logger.info(f"System Optimizer initialized: {optimizer_name}")
    
    def register_adaptive_system(self, system_name: str, adaptive_system: AdaptiveSystem) -> None:
        """Register an adaptive system."""
        self.adaptive_systems[system_name] = adaptive_system
        logger.info(f"Registered adaptive system '{system_name}'")
    
    def unregister_adaptive_system(self, system_name: str) -> None:
        """Unregister an adaptive system."""
        if system_name in self.adaptive_systems:
            del self.adaptive_systems[system_name]
            logger.info(f"Unregistered adaptive system '{system_name}'")
        else:
            logger.warning(f"Adaptive system '{system_name}' not found")
    
    def start_all_systems(self) -> None:
        """Start all registered adaptive systems."""
        for system_name, system in self.adaptive_systems.items():
            try:
                system.start_adaptation()
                logger.info(f"Started adaptive system '{system_name}'")
            except Exception as e:
                logger.error(f"Error starting adaptive system '{system_name}': {e}")
    
    def stop_all_systems(self) -> None:
        """Stop all registered adaptive systems."""
        for system_name, system in self.adaptive_systems.items():
            try:
                system.stop_adaptation()
                logger.info(f"Stopped adaptive system '{system_name}'")
            except Exception as e:
                logger.error(f"Error stopping adaptive system '{system_name}': {e}")
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get statistics for all adaptive systems."""
        return {
            'optimizer_name': self.optimizer_name,
            'total_systems': len(self.adaptive_systems),
            'system_names': list(self.adaptive_systems.keys()),
            'system_statistics': {
                name: system.get_adaptation_statistics()
                for name, system in self.adaptive_systems.items()
            }
        }

