"""
Trigger Monitor for GPT-5 Escalation Framework
Monitors system conditions and triggers GPT-5 API calls when needed
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from enum import Enum
import asyncio
import time
from collections import deque

logger = logging.getLogger(__name__)

class TriggerType(Enum):
    """Types of escalation triggers"""
    RISK_THRESHOLD = "risk_threshold"
    REGIME_SHIFT = "regime_shift"
    MODEL_DECAY = "model_decay"
    EVENT_HEAT = "event_heat"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    VOLATILITY_SPIKE = "volatility_spike"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    MANUAL_OVERRIDE = "manual_override"

class TriggerSeverity(Enum):
    """Trigger severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class TriggerCondition:
    """Trigger condition configuration"""
    trigger_type: TriggerType
    condition: str  # e.g., "var_95 > 0.05"
    severity: TriggerSeverity
    cooldown_minutes: int = 60  # Minimum time between triggers
    enabled: bool = True
    description: str = ""

@dataclass
class TriggerEvent:
    """Trigger event record"""
    event_id: str
    timestamp: datetime
    trigger_type: TriggerType
    severity: TriggerSeverity
    condition: str
    current_value: float
    threshold_value: float
    context: Dict[str, Any]
    resolved: bool = False
    resolution_time: Optional[datetime] = None

class TriggerMonitor:
    """Monitors system conditions and triggers GPT-5 escalations"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.trigger_conditions = []
        self.trigger_history = deque(maxlen=1000)  # Keep last 1000 events
        self.active_triggers = {}
        self.last_trigger_time = {}
        self.escalation_count = 0
        self.max_daily_escalations = config.get('max_daily_escalations', 3)
        self.daily_escalation_count = 0
        self.last_reset_date = datetime.now().date()
        
        # Load default trigger conditions
        self._load_default_triggers()
        
        logger.info("Trigger Monitor initialized")
    
    def _load_default_triggers(self):
        """Load default trigger conditions"""
        try:
            default_triggers = [
                # Risk-based triggers
                TriggerCondition(
                    trigger_type=TriggerType.RISK_THRESHOLD,
                    condition="var_95 > 0.05",
                    severity=TriggerSeverity.HIGH,
                    cooldown_minutes=30,
                    description="VaR exceeds 5% threshold"
                ),
                TriggerCondition(
                    trigger_type=TriggerType.RISK_THRESHOLD,
                    condition="var_99 > 0.08",
                    severity=TriggerSeverity.CRITICAL,
                    cooldown_minutes=15,
                    description="VaR exceeds 8% threshold"
                ),
                
                # Regime shift triggers
                TriggerCondition(
                    trigger_type=TriggerType.REGIME_SHIFT,
                    condition="regime_confidence < 0.6",
                    severity=TriggerSeverity.MEDIUM,
                    cooldown_minutes=60,
                    description="Market regime uncertainty detected"
                ),
                TriggerCondition(
                    trigger_type=TriggerType.REGIME_SHIFT,
                    condition="regime_change_detected == True",
                    severity=TriggerSeverity.HIGH,
                    cooldown_minutes=45,
                    description="Market regime change detected"
                ),
                
                # Model decay triggers
                TriggerCondition(
                    trigger_type=TriggerType.MODEL_DECAY,
                    condition="model_accuracy < 0.55",
                    severity=TriggerSeverity.MEDIUM,
                    cooldown_minutes=120,
                    description="Model accuracy below 55%"
                ),
                TriggerCondition(
                    trigger_type=TriggerType.MODEL_DECAY,
                    condition="prediction_confidence < 0.4",
                    severity=TriggerSeverity.HIGH,
                    cooldown_minutes=90,
                    description="Prediction confidence below 40%"
                ),
                
                # Event heat triggers
                TriggerCondition(
                    trigger_type=TriggerType.EVENT_HEAT,
                    condition="event_heat_score > 0.8",
                    severity=TriggerSeverity.MEDIUM,
                    cooldown_minutes=60,
                    description="High event heat score detected"
                ),
                TriggerCondition(
                    trigger_type=TriggerType.EVENT_HEAT,
                    condition="macro_event_impact > 0.7",
                    severity=TriggerSeverity.HIGH,
                    cooldown_minutes=30,
                    description="High macro event impact"
                ),
                
                # Performance degradation triggers
                TriggerCondition(
                    trigger_type=TriggerType.PERFORMANCE_DEGRADATION,
                    condition="sharpe_ratio < 0.5",
                    severity=TriggerSeverity.MEDIUM,
                    cooldown_minutes=180,
                    description="Sharpe ratio below 0.5"
                ),
                TriggerCondition(
                    trigger_type=TriggerType.PERFORMANCE_DEGRADATION,
                    condition="max_drawdown > 0.15",
                    severity=TriggerSeverity.HIGH,
                    cooldown_minutes=60,
                    description="Maximum drawdown exceeds 15%"
                ),
                
                # Volatility spike triggers
                TriggerCondition(
                    trigger_type=TriggerType.VOLATILITY_SPIKE,
                    condition="volatility > 0.04",
                    severity=TriggerSeverity.MEDIUM,
                    cooldown_minutes=45,
                    description="Volatility spike detected"
                ),
                TriggerCondition(
                    trigger_type=TriggerType.VOLATILITY_SPIKE,
                    condition="volatility > 0.06",
                    severity=TriggerSeverity.HIGH,
                    cooldown_minutes=30,
                    description="Extreme volatility spike"
                ),
                
                # Correlation breakdown triggers
                TriggerCondition(
                    trigger_type=TriggerType.CORRELATION_BREAKDOWN,
                    condition="correlation_breakdown == True",
                    severity=TriggerSeverity.MEDIUM,
                    cooldown_minutes=120,
                    description="Correlation breakdown detected"
                ),
                
                # Liquidity crisis triggers
                TriggerCondition(
                    trigger_type=TriggerType.LIQUIDITY_CRISIS,
                    condition="liquidity_score < 0.3",
                    severity=TriggerSeverity.HIGH,
                    cooldown_minutes=30,
                    description="Low liquidity detected"
                )
            ]
            
            self.trigger_conditions = default_triggers
            logger.info(f"Loaded {len(default_triggers)} default trigger conditions")
            
        except Exception as e:
            logger.error(f"Error loading default triggers: {e}")
    
    def add_trigger_condition(self, condition: TriggerCondition):
        """Add a new trigger condition"""
        try:
            self.trigger_conditions.append(condition)
            logger.info(f"Added trigger condition: {condition.trigger_type.value} - {condition.condition}")
            
        except Exception as e:
            logger.error(f"Error adding trigger condition: {e}")
    
    def remove_trigger_condition(self, condition_index: int):
        """Remove a trigger condition by index"""
        try:
            if 0 <= condition_index < len(self.trigger_conditions):
                removed_condition = self.trigger_conditions.pop(condition_index)
                logger.info(f"Removed trigger condition: {removed_condition.trigger_type.value}")
            else:
                logger.warning(f"Invalid condition index: {condition_index}")
                
        except Exception as e:
            logger.error(f"Error removing trigger condition: {e}")
    
    def check_triggers(self, system_metrics: Dict[str, Any]) -> List[TriggerEvent]:
        """Check all trigger conditions against current system metrics"""
        try:
            # Reset daily escalation count if new day
            current_date = datetime.now().date()
            if current_date != self.last_reset_date:
                self.daily_escalation_count = 0
                self.last_reset_date = current_date
            
            triggered_events = []
            
            for condition in self.trigger_conditions:
                if not condition.enabled:
                    continue
                
                # Check cooldown
                if self._is_condition_in_cooldown(condition):
                    continue
                
                # Check if daily escalation limit reached
                if self.daily_escalation_count >= self.max_daily_escalations:
                    logger.warning(f"Daily escalation limit reached: {self.max_daily_escalations}")
                    break
                
                # Evaluate condition
                if self._evaluate_condition(condition.condition, system_metrics):
                    # Create trigger event
                    event = self._create_trigger_event(condition, system_metrics)
                    triggered_events.append(event)
                    
                    # Record trigger
                    self.trigger_history.append(event)
                    self.active_triggers[event.event_id] = event
                    
                    # Update last trigger time
                    self.last_trigger_time[condition.trigger_type.value] = datetime.now()
                    
                    # Increment escalation count
                    self.escalation_count += 1
                    self.daily_escalation_count += 1
                    
                    logger.warning(f"TRIGGER ACTIVATED: {condition.trigger_type.value} - {condition.condition}")
            
            return triggered_events
            
        except Exception as e:
            logger.error(f"Error checking triggers: {e}")
            return []
    
    def _is_condition_in_cooldown(self, condition: TriggerCondition) -> bool:
        """Check if condition is in cooldown period"""
        try:
            trigger_key = condition.trigger_type.value
            
            if trigger_key not in self.last_trigger_time:
                return False
            
            last_time = self.last_trigger_time[trigger_key]
            cooldown_duration = timedelta(minutes=condition.cooldown_minutes)
            
            return datetime.now() - last_time < cooldown_duration
            
        except Exception as e:
            logger.error(f"Error checking cooldown: {e}")
            return False
    
    def _evaluate_condition(self, condition: str, metrics: Dict[str, Any]) -> bool:
        """Evaluate trigger condition"""
        try:
            # Simple condition evaluation
            # In production, this would use a more sophisticated expression evaluator
            
            # Replace metric names with values
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    condition = condition.replace(metric_name, str(value))
                elif isinstance(value, bool):
                    condition = condition.replace(metric_name, str(value))
                elif isinstance(value, str):
                    condition = condition.replace(f'"{metric_name}"', f'"{value}"')
            
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
    
    def _create_trigger_event(self, condition: TriggerCondition, 
                            metrics: Dict[str, Any]) -> TriggerEvent:
        """Create a trigger event"""
        try:
            event_id = f"trigger_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.escalation_count}"
            
            # Extract current value and threshold from condition
            current_value, threshold_value = self._extract_values_from_condition(
                condition.condition, metrics
            )
            
            # Create context
            context = {
                'system_metrics': metrics,
                'escalation_count': self.escalation_count,
                'daily_escalation_count': self.daily_escalation_count,
                'active_triggers_count': len(self.active_triggers)
            }
            
            event = TriggerEvent(
                event_id=event_id,
                timestamp=datetime.now(),
                trigger_type=condition.trigger_type,
                severity=condition.severity,
                condition=condition.condition,
                current_value=current_value,
                threshold_value=threshold_value,
                context=context
            )
            
            return event
            
        except Exception as e:
            logger.error(f"Error creating trigger event: {e}")
            return None
    
    def _extract_values_from_condition(self, condition: str, metrics: Dict[str, Any]) -> Tuple[float, float]:
        """Extract current value and threshold from condition"""
        try:
            # Simple extraction for conditions like "var_95 > 0.05"
            parts = condition.split()
            if len(parts) >= 3:
                metric_name = parts[0]
                operator = parts[1]
                threshold_str = parts[2]
                
                current_value = metrics.get(metric_name, 0.0)
                threshold_value = float(threshold_str)
                
                return current_value, threshold_value
            
            return 0.0, 0.0
            
        except Exception as e:
            logger.error(f"Error extracting values from condition: {e}")
            return 0.0, 0.0
    
    def resolve_trigger(self, event_id: str, resolution_context: Dict[str, Any] = None):
        """Mark a trigger as resolved"""
        try:
            if event_id in self.active_triggers:
                event = self.active_triggers[event_id]
                event.resolved = True
                event.resolution_time = datetime.now()
                
                if resolution_context:
                    event.context['resolution'] = resolution_context
                
                # Remove from active triggers
                del self.active_triggers[event_id]
                
                logger.info(f"Trigger resolved: {event_id}")
                
        except Exception as e:
            logger.error(f"Error resolving trigger: {e}")
    
    def get_escalation_priority(self, trigger_events: List[TriggerEvent]) -> Optional[TriggerEvent]:
        """Determine which trigger should be escalated to GPT-5"""
        try:
            if not trigger_events:
                return None
            
            # Sort by severity (critical first)
            severity_order = {
                TriggerSeverity.CRITICAL: 4,
                TriggerSeverity.HIGH: 3,
                TriggerSeverity.MEDIUM: 2,
                TriggerSeverity.LOW: 1
            }
            
            # Sort events by severity and timestamp
            sorted_events = sorted(
                trigger_events,
                key=lambda x: (severity_order[x.severity], x.timestamp),
                reverse=True
            )
            
            # Return highest priority event
            return sorted_events[0]
            
        except Exception as e:
            logger.error(f"Error determining escalation priority: {e}")
            return None
    
    def should_escalate_to_gpt5(self, trigger_events: List[TriggerEvent]) -> bool:
        """Determine if triggers should be escalated to GPT-5"""
        try:
            # Check daily escalation limit
            if self.daily_escalation_count >= self.max_daily_escalations:
                return False
            
            # Check if any high or critical severity triggers
            high_severity_triggers = [
                event for event in trigger_events 
                if event.severity in [TriggerSeverity.HIGH, TriggerSeverity.CRITICAL]
            ]
            
            if high_severity_triggers:
                return True
            
            # Check if multiple medium severity triggers
            medium_triggers = [
                event for event in trigger_events 
                if event.severity == TriggerSeverity.MEDIUM
            ]
            
            if len(medium_triggers) >= 2:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error determining GPT-5 escalation: {e}")
            return False
    
    def get_trigger_statistics(self) -> Dict:
        """Get trigger monitoring statistics"""
        try:
            if not self.trigger_history:
                return {}
            
            # Calculate statistics
            total_triggers = len(self.trigger_history)
            severity_counts = {}
            trigger_type_counts = {}
            
            for event in self.trigger_history:
                severity = event.severity.value
                trigger_type = event.trigger_type.value
                
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
                trigger_type_counts[trigger_type] = trigger_type_counts.get(trigger_type, 0) + 1
            
            # Calculate resolution rate
            resolved_count = sum(1 for event in self.trigger_history if event.resolved)
            resolution_rate = resolved_count / total_triggers if total_triggers > 0 else 0
            
            # Calculate average resolution time
            resolution_times = []
            for event in self.trigger_history:
                if event.resolved and event.resolution_time:
                    resolution_time = (event.resolution_time - event.timestamp).total_seconds()
                    resolution_times.append(resolution_time)
            
            avg_resolution_time = np.mean(resolution_times) if resolution_times else 0
            
            return {
                'total_triggers': total_triggers,
                'active_triggers': len(self.active_triggers),
                'severity_counts': severity_counts,
                'trigger_type_counts': trigger_type_counts,
                'resolution_rate': resolution_rate,
                'average_resolution_time_seconds': avg_resolution_time,
                'total_escalations': self.escalation_count,
                'daily_escalations': self.daily_escalation_count,
                'max_daily_escalations': self.max_daily_escalations,
                'active_conditions': len([c for c in self.trigger_conditions if c.enabled])
            }
            
        except Exception as e:
            logger.error(f"Error getting trigger statistics: {e}")
            return {}
    
    def get_escalation_context(self, trigger_event: TriggerEvent) -> Dict[str, Any]:
        """Get context for GPT-5 escalation"""
        try:
            context = {
                'trigger_event': {
                    'event_id': trigger_event.event_id,
                    'timestamp': trigger_event.timestamp.isoformat(),
                    'trigger_type': trigger_event.trigger_type.value,
                    'severity': trigger_event.severity.value,
                    'condition': trigger_event.condition,
                    'current_value': trigger_event.current_value,
                    'threshold_value': trigger_event.threshold_value
                },
                'system_context': trigger_event.context,
                'escalation_metadata': {
                    'escalation_count': self.escalation_count,
                    'daily_escalation_count': self.daily_escalation_count,
                    'active_triggers_count': len(self.active_triggers),
                    'max_daily_escalations': self.max_daily_escalations
                }
            }
            
            return context
            
        except Exception as e:
            logger.error(f"Error getting escalation context: {e}")
            return {}
    
    def export_trigger_conditions(self, filepath: str):
        """Export trigger conditions to file"""
        try:
            import json
            
            conditions_data = []
            for condition in self.trigger_conditions:
                conditions_data.append({
                    'trigger_type': condition.trigger_type.value,
                    'condition': condition.condition,
                    'severity': condition.severity.value,
                    'cooldown_minutes': condition.cooldown_minutes,
                    'enabled': condition.enabled,
                    'description': condition.description
                })
            
            with open(filepath, 'w') as f:
                json.dump(conditions_data, f, indent=2)
            
            logger.info(f"Exported {len(conditions_data)} trigger conditions to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting trigger conditions: {e}")
    
    def import_trigger_conditions(self, filepath: str):
        """Import trigger conditions from file"""
        try:
            import json
            
            with open(filepath, 'r') as f:
                conditions_data = json.load(f)
            
            imported_conditions = []
            for condition_data in conditions_data:
                condition = TriggerCondition(
                    trigger_type=TriggerType(condition_data['trigger_type']),
                    condition=condition_data['condition'],
                    severity=TriggerSeverity(condition_data['severity']),
                    cooldown_minutes=condition_data.get('cooldown_minutes', 60),
                    enabled=condition_data.get('enabled', True),
                    description=condition_data.get('description', '')
                )
                imported_conditions.append(condition)
            
            self.trigger_conditions = imported_conditions
            logger.info(f"Imported {len(imported_conditions)} trigger conditions from {filepath}")
            
        except Exception as e:
            logger.error(f"Error importing trigger conditions: {e}")
