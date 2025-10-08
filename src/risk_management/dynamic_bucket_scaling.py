"""
Dynamic Bucket Scaling System
Implements dynamic capital allocation scaling based on performance and risk metrics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)

class ScalingTrigger(Enum):
    """Scaling trigger types"""
    PERFORMANCE = "performance"
    RISK = "risk"
    VOLATILITY = "volatility"
    DRAWDOWN = "drawdown"
    CORRELATION = "correlation"
    VOLUME = "volume"
    MANUAL = "manual"

@dataclass
class ScalingRule:
    """Scaling rule configuration"""
    trigger: ScalingTrigger
    condition: str  # e.g., "sharpe_ratio > 1.5"
    action: str  # e.g., "increase_penny_bucket"
    scale_factor: float  # e.g., 1.2 for 20% increase
    min_scale: float = 0.1  # Minimum scale factor
    max_scale: float = 3.0  # Maximum scale factor
    cooldown_hours: int = 24  # Hours before rule can trigger again
    enabled: bool = True

@dataclass
class ScalingEvent:
    """Scaling event record"""
    event_id: str
    timestamp: datetime
    trigger: ScalingTrigger
    rule: ScalingRule
    old_allocation: Dict[str, float]
    new_allocation: Dict[str, float]
    reason: str
    performance_metrics: Dict[str, float]

class DynamicBucketScaler:
    """Dynamic bucket scaling system"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.scaling_rules = []
        self.scaling_history = []
        self.current_allocation = {}
        self.performance_history = []
        self.risk_metrics_history = []
        self.last_scaling_time = {}
        
        # Load default scaling rules
        self._load_default_rules()
        
        logger.info("Dynamic Bucket Scaler initialized")
    
    def _load_default_rules(self):
        """Load default scaling rules"""
        try:
            default_rules = [
                # Performance-based scaling
                ScalingRule(
                    trigger=ScalingTrigger.PERFORMANCE,
                    condition="sharpe_ratio > 2.0",
                    action="increase_penny_bucket",
                    scale_factor=1.5,
                    cooldown_hours=24
                ),
                ScalingRule(
                    trigger=ScalingTrigger.PERFORMANCE,
                    condition="sharpe_ratio < 0.5",
                    action="decrease_penny_bucket",
                    scale_factor=0.7,
                    cooldown_hours=12
                ),
                
                # Risk-based scaling
                ScalingRule(
                    trigger=ScalingTrigger.RISK,
                    condition="var_95 > 0.05",
                    action="decrease_all_buckets",
                    scale_factor=0.8,
                    cooldown_hours=6
                ),
                ScalingRule(
                    trigger=ScalingTrigger.RISK,
                    condition="var_95 < 0.02",
                    action="increase_core_bucket",
                    scale_factor=1.2,
                    cooldown_hours=48
                ),
                
                # Drawdown-based scaling
                ScalingRule(
                    trigger=ScalingTrigger.DRAWDOWN,
                    condition="max_drawdown > 0.15",
                    action="emergency_scaling",
                    scale_factor=0.5,
                    cooldown_hours=1
                ),
                
                # Volatility-based scaling
                ScalingRule(
                    trigger=ScalingTrigger.VOLATILITY,
                    condition="volatility > 0.03",
                    action="decrease_penny_bucket",
                    scale_factor=0.6,
                    cooldown_hours=12
                ),
                ScalingRule(
                    trigger=ScalingTrigger.VOLATILITY,
                    condition="volatility < 0.01",
                    action="increase_fno_bucket",
                    scale_factor=1.3,
                    cooldown_hours=24
                )
            ]
            
            self.scaling_rules = default_rules
            logger.info(f"Loaded {len(default_rules)} default scaling rules")
            
        except Exception as e:
            logger.error(f"Error loading default rules: {e}")
    
    def add_scaling_rule(self, rule: ScalingRule):
        """Add a new scaling rule"""
        try:
            self.scaling_rules.append(rule)
            logger.info(f"Added scaling rule: {rule.trigger.value} - {rule.condition}")
            
        except Exception as e:
            logger.error(f"Error adding scaling rule: {e}")
    
    def remove_scaling_rule(self, rule_index: int):
        """Remove a scaling rule by index"""
        try:
            if 0 <= rule_index < len(self.scaling_rules):
                removed_rule = self.scaling_rules.pop(rule_index)
                logger.info(f"Removed scaling rule: {removed_rule.trigger.value}")
            else:
                logger.warning(f"Invalid rule index: {rule_index}")
                
        except Exception as e:
            logger.error(f"Error removing scaling rule: {e}")
    
    def update_performance_metrics(self, metrics: Dict[str, float]):
        """Update performance metrics for scaling decisions"""
        try:
            metrics['timestamp'] = datetime.now()
            self.performance_history.append(metrics)
            
            # Keep only last 1000 records
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]
            
            logger.debug(f"Updated performance metrics: {len(metrics)} metrics")
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    def update_risk_metrics(self, metrics: Dict[str, float]):
        """Update risk metrics for scaling decisions"""
        try:
            metrics['timestamp'] = datetime.now()
            self.risk_metrics_history.append(metrics)
            
            # Keep only last 1000 records
            if len(self.risk_metrics_history) > 1000:
                self.risk_metrics_history = self.risk_metrics_history[-1000:]
            
            logger.debug(f"Updated risk metrics: {len(metrics)} metrics")
            
        except Exception as e:
            logger.error(f"Error updating risk metrics: {e}")
    
    def evaluate_scaling_rules(self, current_allocation: Dict[str, float]) -> List[ScalingRule]:
        """Evaluate all scaling rules and return triggered rules"""
        try:
            triggered_rules = []
            
            # Get latest metrics
            latest_performance = self._get_latest_metrics(self.performance_history)
            latest_risk = self._get_latest_metrics(self.risk_metrics_history)
            
            # Combine metrics
            all_metrics = {**latest_performance, **latest_risk}
            
            for rule in self.scaling_rules:
                if not rule.enabled:
                    continue
                
                # Check cooldown
                if self._is_rule_in_cooldown(rule):
                    continue
                
                # Evaluate condition
                if self._evaluate_condition(rule.condition, all_metrics):
                    triggered_rules.append(rule)
                    logger.info(f"Rule triggered: {rule.trigger.value} - {rule.condition}")
            
            return triggered_rules
            
        except Exception as e:
            logger.error(f"Error evaluating scaling rules: {e}")
            return []
    
    def _get_latest_metrics(self, metrics_history: List[Dict]) -> Dict[str, float]:
        """Get latest metrics from history"""
        try:
            if not metrics_history:
                return {}
            
            latest = metrics_history[-1].copy()
            latest.pop('timestamp', None)  # Remove timestamp
            return latest
            
        except Exception as e:
            logger.error(f"Error getting latest metrics: {e}")
            return {}
    
    def _is_rule_in_cooldown(self, rule: ScalingRule) -> bool:
        """Check if rule is in cooldown period"""
        try:
            rule_key = f"{rule.trigger.value}_{rule.condition}"
            
            if rule_key not in self.last_scaling_time:
                return False
            
            last_time = self.last_scaling_time[rule_key]
            cooldown_duration = timedelta(hours=rule.cooldown_hours)
            
            return datetime.now() - last_time < cooldown_duration
            
        except Exception as e:
            logger.error(f"Error checking cooldown: {e}")
            return False
    
    def _evaluate_condition(self, condition: str, metrics: Dict[str, float]) -> bool:
        """Evaluate scaling condition"""
        try:
            # Simple condition evaluation
            # In production, this would use a more sophisticated expression evaluator
            
            # Replace metric names with values
            for metric_name, value in metrics.items():
                condition = condition.replace(metric_name, str(value))
            
            # Evaluate the condition
            try:
                result = eval(condition)
                return bool(result)
            except:
                logger.warning(f"Could not evaluate condition: {condition}")
                return False
                
        except Exception as e:
            logger.error(f"Error evaluating condition: {e}")
            return False
    
    def apply_scaling_action(self, action: str, scale_factor: float, 
                           current_allocation: Dict[str, float]) -> Dict[str, float]:
        """Apply scaling action to allocation"""
        try:
            new_allocation = current_allocation.copy()
            
            if action == "increase_penny_bucket":
                new_allocation['penny'] = min(
                    new_allocation.get('penny', 0.02) * scale_factor,
                    0.05  # Max 5% for penny stocks
                )
            elif action == "decrease_penny_bucket":
                new_allocation['penny'] = max(
                    new_allocation.get('penny', 0.02) * scale_factor,
                    0.01  # Min 1% for penny stocks
                )
            elif action == "increase_fno_bucket":
                new_allocation['fno'] = min(
                    new_allocation.get('fno', 0.05) * scale_factor,
                    0.10  # Max 10% for F&O
                )
            elif action == "decrease_fno_bucket":
                new_allocation['fno'] = max(
                    new_allocation.get('fno', 0.05) * scale_factor,
                    0.02  # Min 2% for F&O
                )
            elif action == "increase_core_bucket":
                new_allocation['core'] = min(
                    new_allocation.get('core', 0.90) * scale_factor,
                    0.95  # Max 95% for core
                )
            elif action == "decrease_core_bucket":
                new_allocation['core'] = max(
                    new_allocation.get('core', 0.90) * scale_factor,
                    0.80  # Min 80% for core
                )
            elif action == "decrease_all_buckets":
                for bucket in ['penny', 'fno', 'core', 'sip']:
                    if bucket in new_allocation:
                        new_allocation[bucket] *= scale_factor
            elif action == "emergency_scaling":
                # Emergency scaling - reduce all risky buckets
                new_allocation['penny'] = max(new_allocation.get('penny', 0.02) * 0.3, 0.005)
                new_allocation['fno'] = max(new_allocation.get('fno', 0.05) * 0.3, 0.01)
                new_allocation['core'] = min(new_allocation.get('core', 0.90) * 1.1, 0.95)
            
            # Normalize allocation to ensure it sums to 1.0
            new_allocation = self._normalize_allocation(new_allocation)
            
            return new_allocation
            
        except Exception as e:
            logger.error(f"Error applying scaling action: {e}")
            return current_allocation
    
    def _normalize_allocation(self, allocation: Dict[str, float]) -> Dict[str, float]:
        """Normalize allocation to sum to 1.0"""
        try:
            total = sum(allocation.values())
            if total == 0:
                return allocation
            
            return {bucket: weight / total for bucket, weight in allocation.items()}
            
        except Exception as e:
            logger.error(f"Error normalizing allocation: {e}")
            return allocation
    
    def execute_scaling(self, current_allocation: Dict[str, float]) -> Dict[str, Any]:
        """Execute dynamic scaling based on current conditions"""
        try:
            # Evaluate scaling rules
            triggered_rules = self.evaluate_scaling_rules(current_allocation)
            
            if not triggered_rules:
                return {
                    'scaled': False,
                    'reason': 'No rules triggered',
                    'new_allocation': current_allocation
                }
            
            # Apply scaling actions
            new_allocation = current_allocation.copy()
            scaling_events = []
            
            for rule in triggered_rules:
                # Apply scaling action
                new_allocation = self.apply_scaling_action(
                    rule.action, rule.scale_factor, new_allocation
                )
                
                # Record scaling event
                event = ScalingEvent(
                    event_id=f"scale_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    timestamp=datetime.now(),
                    trigger=rule.trigger,
                    rule=rule,
                    old_allocation=current_allocation.copy(),
                    new_allocation=new_allocation.copy(),
                    reason=f"{rule.trigger.value}: {rule.condition}",
                    performance_metrics=self._get_latest_metrics(self.performance_history)
                )
                
                scaling_events.append(event)
                self.scaling_history.append(event)
                
                # Update last scaling time
                rule_key = f"{rule.trigger.value}_{rule.condition}"
                self.last_scaling_time[rule_key] = datetime.now()
                
                logger.info(f"Applied scaling: {rule.action} with factor {rule.scale_factor}")
            
            # Update current allocation
            self.current_allocation = new_allocation
            
            return {
                'scaled': True,
                'reason': f"{len(triggered_rules)} rules triggered",
                'new_allocation': new_allocation,
                'scaling_events': scaling_events,
                'changes': self._calculate_allocation_changes(current_allocation, new_allocation)
            }
            
        except Exception as e:
            logger.error(f"Error executing scaling: {e}")
            return {
                'scaled': False,
                'reason': f'Error: {str(e)}',
                'new_allocation': current_allocation
            }
    
    def _calculate_allocation_changes(self, old_allocation: Dict[str, float], 
                                    new_allocation: Dict[str, float]) -> Dict[str, float]:
        """Calculate allocation changes"""
        try:
            changes = {}
            all_buckets = set(old_allocation.keys()) | set(new_allocation.keys())
            
            for bucket in all_buckets:
                old_value = old_allocation.get(bucket, 0.0)
                new_value = new_allocation.get(bucket, 0.0)
                changes[bucket] = new_value - old_value
            
            return changes
            
        except Exception as e:
            logger.error(f"Error calculating allocation changes: {e}")
            return {}
    
    def get_scaling_statistics(self) -> Dict:
        """Get scaling statistics"""
        try:
            if not self.scaling_history:
                return {}
            
            # Calculate statistics
            total_scaling_events = len(self.scaling_history)
            trigger_counts = {}
            action_counts = {}
            
            for event in self.scaling_history:
                trigger = event.trigger.value
                action = event.rule.action
                
                trigger_counts[trigger] = trigger_counts.get(trigger, 0) + 1
                action_counts[action] = action_counts.get(action, 0) + 1
            
            # Calculate average scaling frequency
            if total_scaling_events > 1:
                time_span = (self.scaling_history[-1].timestamp - 
                           self.scaling_history[0].timestamp).total_seconds()
                avg_frequency = total_scaling_events / (time_span / 3600)  # events per hour
            else:
                avg_frequency = 0
            
            return {
                'total_scaling_events': total_scaling_events,
                'trigger_counts': trigger_counts,
                'action_counts': action_counts,
                'average_frequency_per_hour': avg_frequency,
                'active_rules': len([r for r in self.scaling_rules if r.enabled]),
                'total_rules': len(self.scaling_rules),
                'current_allocation': self.current_allocation
            }
            
        except Exception as e:
            logger.error(f"Error getting scaling statistics: {e}")
            return {}
    
    def get_scaling_recommendations(self, current_allocation: Dict[str, float]) -> List[Dict]:
        """Get scaling recommendations based on current metrics"""
        try:
            recommendations = []
            
            # Get latest metrics
            latest_performance = self._get_latest_metrics(self.performance_history)
            latest_risk = self._get_latest_metrics(self.risk_metrics_history)
            all_metrics = {**latest_performance, **latest_risk}
            
            # Generate recommendations based on metrics
            if all_metrics.get('sharpe_ratio', 0) > 2.0:
                recommendations.append({
                    'action': 'increase_penny_bucket',
                    'reason': 'High Sharpe ratio suggests good risk-adjusted returns',
                    'priority': 'medium',
                    'scale_factor': 1.3
                })
            
            if all_metrics.get('var_95', 0) > 0.05:
                recommendations.append({
                    'action': 'decrease_all_buckets',
                    'reason': 'High VaR indicates elevated risk',
                    'priority': 'high',
                    'scale_factor': 0.8
                })
            
            if all_metrics.get('max_drawdown', 0) > 0.15:
                recommendations.append({
                    'action': 'emergency_scaling',
                    'reason': 'Large drawdown requires immediate risk reduction',
                    'priority': 'critical',
                    'scale_factor': 0.5
                })
            
            if all_metrics.get('volatility', 0) < 0.01:
                recommendations.append({
                    'action': 'increase_fno_bucket',
                    'reason': 'Low volatility allows for more aggressive strategies',
                    'priority': 'low',
                    'scale_factor': 1.2
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting scaling recommendations: {e}")
            return []
    
    def export_scaling_rules(self, filepath: str):
        """Export scaling rules to file"""
        try:
            rules_data = []
            for rule in self.scaling_rules:
                rules_data.append({
                    'trigger': rule.trigger.value,
                    'condition': rule.condition,
                    'action': rule.action,
                    'scale_factor': rule.scale_factor,
                    'min_scale': rule.min_scale,
                    'max_scale': rule.max_scale,
                    'cooldown_hours': rule.cooldown_hours,
                    'enabled': rule.enabled
                })
            
            with open(filepath, 'w') as f:
                json.dump(rules_data, f, indent=2)
            
            logger.info(f"Exported {len(rules_data)} scaling rules to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting scaling rules: {e}")
    
    def import_scaling_rules(self, filepath: str):
        """Import scaling rules from file"""
        try:
            with open(filepath, 'r') as f:
                rules_data = json.load(f)
            
            imported_rules = []
            for rule_data in rules_data:
                rule = ScalingRule(
                    trigger=ScalingTrigger(rule_data['trigger']),
                    condition=rule_data['condition'],
                    action=rule_data['action'],
                    scale_factor=rule_data['scale_factor'],
                    min_scale=rule_data.get('min_scale', 0.1),
                    max_scale=rule_data.get('max_scale', 3.0),
                    cooldown_hours=rule_data.get('cooldown_hours', 24),
                    enabled=rule_data.get('enabled', True)
                )
                imported_rules.append(rule)
            
            self.scaling_rules = imported_rules
            logger.info(f"Imported {len(imported_rules)} scaling rules from {filepath}")
            
        except Exception as e:
            logger.error(f"Error importing scaling rules: {e}")
