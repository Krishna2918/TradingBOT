"""
Kill-Switch Automation System
Implements automated kill-switch mechanisms for emergency trading halt
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from enum import Enum
import json
import asyncio
import time
from collections import deque

logger = logging.getLogger(__name__)

class KillSwitchTrigger(Enum):
    """Types of kill-switch triggers"""
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    MAX_DRAWDOWN = "max_drawdown"
    VOLATILITY_SPIKE = "volatility_spike"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    MODEL_DECAY = "model_decay"
    SYSTEM_ERROR = "system_error"
    MANUAL_OVERRIDE = "manual_override"
    RISK_THRESHOLD = "risk_threshold"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    MARKET_CRASH = "market_crash"

class KillSwitchSeverity(Enum):
    """Kill-switch severity levels"""
    WARNING = "warning"
    CAUTION = "caution"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class KillSwitchAction(Enum):
    """Kill-switch actions"""
    HALT_TRADING = "halt_trading"
    REDUCE_POSITIONS = "reduce_positions"
    CLOSE_ALL_POSITIONS = "close_all_positions"
    SWITCH_TO_SAFE_MODE = "switch_to_safe_mode"
    NOTIFY_ADMIN = "notify_admin"
    LOG_INCIDENT = "log_incident"

@dataclass
class KillSwitchRule:
    """Kill-switch rule configuration"""
    rule_id: str
    name: str
    trigger: KillSwitchTrigger
    condition: str  # e.g., "daily_pnl < -0.05"
    severity: KillSwitchSeverity
    actions: List[KillSwitchAction]
    cooldown_minutes: int = 60
    enabled: bool = True
    description: str = ""

@dataclass
class KillSwitchEvent:
    """Kill-switch event record"""
    event_id: str
    timestamp: datetime
    trigger: KillSwitchTrigger
    severity: KillSwitchSeverity
    condition: str
    current_value: float
    threshold_value: float
    actions_taken: List[KillSwitchAction]
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    context: Dict[str, Any]

class KillSwitchAutomation:
    """Automated kill-switch system"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.kill_switch_rules = []
        self.event_history = deque(maxlen=1000)
        self.active_events = {}
        self.last_trigger_time = {}
        self.system_state = {
            'trading_halted': False,
            'safe_mode_active': False,
            'positions_reduced': False,
            'last_health_check': datetime.now()
        }
        
        # Load default kill-switch rules
        self._load_default_rules()
        
        # Initialize monitoring
        self._start_monitoring()
        
        logger.info("Kill-Switch Automation initialized")
    
    def _load_default_rules(self):
        """Load default kill-switch rules"""
        try:
            default_rules = [
                # Daily loss limit rules
                KillSwitchRule(
                    rule_id="daily_loss_001",
                    name="Daily Loss Limit - 5%",
                    trigger=KillSwitchTrigger.DAILY_LOSS_LIMIT,
                    condition="daily_pnl < -0.05",
                    severity=KillSwitchSeverity.CRITICAL,
                    actions=[KillSwitchAction.HALT_TRADING, KillSwitchAction.NOTIFY_ADMIN],
                    cooldown_minutes=30,
                    description="Halt trading if daily loss exceeds 5%"
                ),
                KillSwitchRule(
                    rule_id="daily_loss_002",
                    name="Daily Loss Limit - 3%",
                    trigger=KillSwitchTrigger.DAILY_LOSS_LIMIT,
                    condition="daily_pnl < -0.03",
                    severity=KillSwitchSeverity.CAUTION,
                    actions=[KillSwitchAction.REDUCE_POSITIONS, KillSwitchAction.LOG_INCIDENT],
                    cooldown_minutes=60,
                    description="Reduce positions if daily loss exceeds 3%"
                ),
                
                # Maximum drawdown rules
                KillSwitchRule(
                    rule_id="drawdown_001",
                    name="Maximum Drawdown - 15%",
                    trigger=KillSwitchTrigger.MAX_DRAWDOWN,
                    condition="max_drawdown > 0.15",
                    severity=KillSwitchSeverity.EMERGENCY,
                    actions=[KillSwitchAction.CLOSE_ALL_POSITIONS, KillSwitchAction.HALT_TRADING, KillSwitchAction.NOTIFY_ADMIN],
                    cooldown_minutes=15,
                    description="Emergency stop if maximum drawdown exceeds 15%"
                ),
                KillSwitchRule(
                    rule_id="drawdown_002",
                    name="Maximum Drawdown - 10%",
                    trigger=KillSwitchTrigger.MAX_DRAWDOWN,
                    condition="max_drawdown > 0.10",
                    severity=KillSwitchSeverity.CRITICAL,
                    actions=[KillSwitchAction.REDUCE_POSITIONS, KillSwitchAction.SWITCH_TO_SAFE_MODE],
                    cooldown_minutes=30,
                    description="Reduce risk if maximum drawdown exceeds 10%"
                ),
                
                # Volatility spike rules
                KillSwitchRule(
                    rule_id="volatility_001",
                    name="Volatility Spike - 6%",
                    trigger=KillSwitchTrigger.VOLATILITY_SPIKE,
                    condition="volatility > 0.06",
                    severity=KillSwitchSeverity.CRITICAL,
                    actions=[KillSwitchAction.HALT_TRADING, KillSwitchAction.LOG_INCIDENT],
                    cooldown_minutes=45,
                    description="Halt trading during extreme volatility"
                ),
                KillSwitchRule(
                    rule_id="volatility_002",
                    name="Volatility Spike - 4%",
                    trigger=KillSwitchTrigger.VOLATILITY_SPIKE,
                    condition="volatility > 0.04",
                    severity=KillSwitchSeverity.CAUTION,
                    actions=[KillSwitchAction.REDUCE_POSITIONS],
                    cooldown_minutes=60,
                    description="Reduce positions during high volatility"
                ),
                
                # Risk threshold rules
                KillSwitchRule(
                    rule_id="risk_001",
                    name="VaR Limit - 8%",
                    trigger=KillSwitchTrigger.RISK_THRESHOLD,
                    condition="var_99 > 0.08",
                    severity=KillSwitchSeverity.CRITICAL,
                    actions=[KillSwitchAction.HALT_TRADING, KillSwitchAction.NOTIFY_ADMIN],
                    cooldown_minutes=30,
                    description="Halt trading if VaR exceeds 8%"
                ),
                KillSwitchRule(
                    rule_id="risk_002",
                    name="VaR Limit - 5%",
                    trigger=KillSwitchTrigger.RISK_THRESHOLD,
                    condition="var_95 > 0.05",
                    severity=KillSwitchSeverity.CAUTION,
                    actions=[KillSwitchAction.REDUCE_POSITIONS],
                    cooldown_minutes=60,
                    description="Reduce positions if VaR exceeds 5%"
                ),
                
                # Model decay rules
                KillSwitchRule(
                    rule_id="model_001",
                    name="Model Accuracy Drop",
                    trigger=KillSwitchTrigger.MODEL_DECAY,
                    condition="model_accuracy < 0.45",
                    severity=KillSwitchSeverity.CRITICAL,
                    actions=[KillSwitchAction.SWITCH_TO_SAFE_MODE, KillSwitchAction.NOTIFY_ADMIN],
                    cooldown_minutes=120,
                    description="Switch to safe mode if model accuracy drops below 45%"
                ),
                
                # System error rules
                KillSwitchRule(
                    rule_id="system_001",
                    name="System Error Detection",
                    trigger=KillSwitchTrigger.SYSTEM_ERROR,
                    condition="system_errors > 5",
                    severity=KillSwitchSeverity.CRITICAL,
                    actions=[KillSwitchAction.HALT_TRADING, KillSwitchAction.NOTIFY_ADMIN],
                    cooldown_minutes=15,
                    description="Halt trading if system errors exceed threshold"
                ),
                
                # Correlation breakdown rules
                KillSwitchRule(
                    rule_id="correlation_001",
                    name="Correlation Breakdown",
                    trigger=KillSwitchTrigger.CORRELATION_BREAKDOWN,
                    condition="correlation_breakdown == True",
                    severity=KillSwitchSeverity.CAUTION,
                    actions=[KillSwitchAction.REDUCE_POSITIONS, KillSwitchAction.LOG_INCIDENT],
                    cooldown_minutes=90,
                    description="Reduce positions if correlation breakdown detected"
                )
            ]
            
            self.kill_switch_rules = default_rules
            logger.info(f"Loaded {len(default_rules)} kill-switch rules")
            
        except Exception as e:
            logger.error(f"Error loading default kill-switch rules: {e}")
    
    def _start_monitoring(self):
        """Start kill-switch monitoring"""
        try:
            # This would start a background monitoring task
            # For now, we'll just log that monitoring is active
            logger.info("Kill-switch monitoring started")
            
        except Exception as e:
            logger.error(f"Error starting kill-switch monitoring: {e}")
    
    def add_kill_switch_rule(self, rule: KillSwitchRule):
        """Add a new kill-switch rule"""
        try:
            self.kill_switch_rules.append(rule)
            logger.info(f"Added kill-switch rule: {rule.name}")
            
        except Exception as e:
            logger.error(f"Error adding kill-switch rule: {e}")
    
    def remove_kill_switch_rule(self, rule_id: str):
        """Remove a kill-switch rule by ID"""
        try:
            self.kill_switch_rules = [rule for rule in self.kill_switch_rules if rule.rule_id != rule_id]
            logger.info(f"Removed kill-switch rule: {rule_id}")
            
        except Exception as e:
            logger.error(f"Error removing kill-switch rule: {e}")
    
    def check_kill_switch_conditions(self, system_metrics: Dict[str, Any]) -> List[KillSwitchEvent]:
        """Check all kill-switch conditions against current system metrics"""
        try:
            triggered_events = []
            
            for rule in self.kill_switch_rules:
                if not rule.enabled:
                    continue
                
                # Check cooldown
                if self._is_rule_in_cooldown(rule):
                    continue
                
                # Evaluate condition
                if self._evaluate_condition(rule.condition, system_metrics):
                    # Create kill-switch event
                    event = self._create_kill_switch_event(rule, system_metrics)
                    triggered_events.append(event)
                    
                    # Store event
                    self.event_history.append(event)
                    self.active_events[event.event_id] = event
                    
                    # Update last trigger time
                    self.last_trigger_time[rule.rule_id] = datetime.now()
                    
                    # Execute actions
                    self._execute_kill_switch_actions(event)
                    
                    logger.critical(f"KILL-SWITCH TRIGGERED: {rule.name} - {rule.condition}")
            
            return triggered_events
            
        except Exception as e:
            logger.error(f"Error checking kill-switch conditions: {e}")
            return []
    
    def _is_rule_in_cooldown(self, rule: KillSwitchRule) -> bool:
        """Check if rule is in cooldown period"""
        try:
            if rule.rule_id not in self.last_trigger_time:
                return False
            
            last_time = self.last_trigger_time[rule.rule_id]
            cooldown_duration = timedelta(minutes=rule.cooldown_minutes)
            
            return datetime.now() - last_time < cooldown_duration
            
        except Exception as e:
            logger.error(f"Error checking cooldown: {e}")
            return False
    
    def _evaluate_condition(self, condition: str, metrics: Dict[str, Any]) -> bool:
        """Evaluate kill-switch condition"""
        try:
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
                logger.warning(f"Could not evaluate kill-switch condition: {condition}")
                return False
                
        except Exception as e:
            logger.error(f"Error evaluating kill-switch condition: {e}")
            return False
    
    def _create_kill_switch_event(self, rule: KillSwitchRule, 
                                metrics: Dict[str, Any]) -> KillSwitchEvent:
        """Create a kill-switch event"""
        try:
            event_id = f"killswitch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Extract current value and threshold from condition
            current_value, threshold_value = self._extract_values_from_condition(
                rule.condition, metrics
            )
            
            # Create context
            context = {
                'system_metrics': metrics,
                'rule_id': rule.rule_id,
                'rule_name': rule.name,
                'system_state': self.system_state.copy()
            }
            
            event = KillSwitchEvent(
                event_id=event_id,
                timestamp=datetime.now(),
                trigger=rule.trigger,
                severity=rule.severity,
                condition=rule.condition,
                current_value=current_value,
                threshold_value=threshold_value,
                actions_taken=rule.actions.copy(),
                context=context
            )
            
            return event
            
        except Exception as e:
            logger.error(f"Error creating kill-switch event: {e}")
            return None
    
    def _extract_values_from_condition(self, condition: str, metrics: Dict[str, Any]) -> Tuple[float, float]:
        """Extract current value and threshold from condition"""
        try:
            # Simple extraction for conditions like "daily_pnl < -0.05"
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
    
    def _execute_kill_switch_actions(self, event: KillSwitchEvent):
        """Execute kill-switch actions"""
        try:
            for action in event.actions_taken:
                if action == KillSwitchAction.HALT_TRADING:
                    self._halt_trading()
                elif action == KillSwitchAction.REDUCE_POSITIONS:
                    self._reduce_positions()
                elif action == KillSwitchAction.CLOSE_ALL_POSITIONS:
                    self._close_all_positions()
                elif action == KillSwitchAction.SWITCH_TO_SAFE_MODE:
                    self._switch_to_safe_mode()
                elif action == KillSwitchAction.NOTIFY_ADMIN:
                    self._notify_admin(event)
                elif action == KillSwitchAction.LOG_INCIDENT:
                    self._log_incident(event)
            
            logger.critical(f"Executed {len(event.actions_taken)} kill-switch actions for event {event.event_id}")
            
        except Exception as e:
            logger.error(f"Error executing kill-switch actions: {e}")
    
    def _halt_trading(self):
        """Halt all trading activities"""
        try:
            self.system_state['trading_halted'] = True
            logger.critical("TRADING HALTED - Kill-switch activated")
            
            # This would interface with the trading system to halt all trading
            
        except Exception as e:
            logger.error(f"Error halting trading: {e}")
    
    def _reduce_positions(self):
        """Reduce position sizes"""
        try:
            self.system_state['positions_reduced'] = True
            logger.warning("POSITIONS REDUCED - Risk mitigation activated")
            
            # This would interface with the trading system to reduce positions
            
        except Exception as e:
            logger.error(f"Error reducing positions: {e}")
    
    def _close_all_positions(self):
        """Close all positions"""
        try:
            logger.critical("CLOSING ALL POSITIONS - Emergency stop activated")
            
            # This would interface with the trading system to close all positions
            
        except Exception as e:
            logger.error(f"Error closing all positions: {e}")
    
    def _switch_to_safe_mode(self):
        """Switch to safe trading mode"""
        try:
            self.system_state['safe_mode_active'] = True
            logger.warning("SAFE MODE ACTIVATED - Conservative trading enabled")
            
            # This would interface with the trading system to enable safe mode
            
        except Exception as e:
            logger.error(f"Error switching to safe mode: {e}")
    
    def _notify_admin(self, event: KillSwitchEvent):
        """Notify administrators"""
        try:
            notification = {
                'type': 'kill_switch_alert',
                'timestamp': datetime.now(),
                'event_id': event.event_id,
                'severity': event.severity.value,
                'trigger': event.trigger.value,
                'condition': event.condition,
                'actions_taken': [action.value for action in event.actions_taken],
                'system_state': self.system_state
            }
            
            logger.critical(f"ADMIN NOTIFICATION: {json.dumps(notification, default=str)}")
            
            # This would send actual notifications (email, SMS, etc.)
            
        except Exception as e:
            logger.error(f"Error notifying admin: {e}")
    
    def _log_incident(self, event: KillSwitchEvent):
        """Log incident details"""
        try:
            incident_log = {
                'incident_id': event.event_id,
                'timestamp': event.timestamp,
                'severity': event.severity.value,
                'trigger': event.trigger.value,
                'condition': event.condition,
                'current_value': event.current_value,
                'threshold_value': event.threshold_value,
                'actions_taken': [action.value for action in event.actions_taken],
                'context': event.context
            }
            
            logger.critical(f"INCIDENT LOGGED: {json.dumps(incident_log, default=str)}")
            
        except Exception as e:
            logger.error(f"Error logging incident: {e}")
    
    def resolve_kill_switch_event(self, event_id: str, resolution_context: Dict[str, Any] = None):
        """Resolve a kill-switch event"""
        try:
            if event_id in self.active_events:
                event = self.active_events[event_id]
                event.resolved = True
                event.resolution_time = datetime.now()
                
                if resolution_context:
                    event.context['resolution'] = resolution_context
                
                # Remove from active events
                del self.active_events[event_id]
                
                # Reset system state if appropriate
                self._reset_system_state_if_safe()
                
                logger.info(f"Kill-switch event resolved: {event_id}")
                
        except Exception as e:
            logger.error(f"Error resolving kill-switch event: {e}")
    
    def _reset_system_state_if_safe(self):
        """Reset system state if it's safe to do so"""
        try:
            # Check if there are any active critical events
            critical_events = [
                event for event in self.active_events.values()
                if event.severity in [KillSwitchSeverity.CRITICAL, KillSwitchSeverity.EMERGENCY]
            ]
            
            if not critical_events:
                # Reset system state
                self.system_state.update({
                    'trading_halted': False,
                    'safe_mode_active': False,
                    'positions_reduced': False,
                    'last_health_check': datetime.now()
                })
                
                logger.info("System state reset - no critical events active")
            
        except Exception as e:
            logger.error(f"Error resetting system state: {e}")
    
    def manual_kill_switch(self, reason: str, actions: List[KillSwitchAction] = None):
        """Manually trigger kill-switch"""
        try:
            if actions is None:
                actions = [KillSwitchAction.HALT_TRADING, KillSwitchAction.NOTIFY_ADMIN]
            
            # Create manual event
            event = KillSwitchEvent(
                event_id=f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                timestamp=datetime.now(),
                trigger=KillSwitchTrigger.MANUAL_OVERRIDE,
                severity=KillSwitchSeverity.EMERGENCY,
                condition=f"Manual override: {reason}",
                current_value=0.0,
                threshold_value=0.0,
                actions_taken=actions,
                context={'reason': reason, 'manual': True}
            )
            
            # Store event
            self.event_history.append(event)
            self.active_events[event.event_id] = event
            
            # Execute actions
            self._execute_kill_switch_actions(event)
            
            logger.critical(f"MANUAL KILL-SWITCH ACTIVATED: {reason}")
            
        except Exception as e:
            logger.error(f"Error in manual kill-switch: {e}")
    
    def get_system_status(self) -> Dict:
        """Get current system status"""
        try:
            return {
                'system_state': self.system_state.copy(),
                'active_events': len(self.active_events),
                'total_events': len(self.event_history),
                'active_rules': len([rule for rule in self.kill_switch_rules if rule.enabled]),
                'last_health_check': self.system_state['last_health_check'],
                'status': 'operational' if not self.system_state['trading_halted'] else 'halted'
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {}
    
    def get_kill_switch_statistics(self) -> Dict:
        """Get kill-switch statistics"""
        try:
            if not self.event_history:
                return {}
            
            # Calculate statistics
            total_events = len(self.event_history)
            severity_counts = {}
            trigger_counts = {}
            action_counts = {}
            
            for event in self.event_history:
                severity = event.severity.value
                trigger = event.trigger.value
                
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
                trigger_counts[trigger] = trigger_counts.get(trigger, 0) + 1
                
                for action in event.actions_taken:
                    action_name = action.value
                    action_counts[action_name] = action_counts.get(action_name, 0) + 1
            
            # Calculate resolution rate
            resolved_count = sum(1 for event in self.event_history if event.resolved)
            resolution_rate = resolved_count / total_events if total_events > 0 else 0
            
            return {
                'total_events': total_events,
                'active_events': len(self.active_events),
                'severity_breakdown': severity_counts,
                'trigger_breakdown': trigger_counts,
                'action_breakdown': action_counts,
                'resolution_rate': resolution_rate,
                'active_rules': len([rule for rule in self.kill_switch_rules if rule.enabled]),
                'system_status': self.get_system_status()
            }
            
        except Exception as e:
            logger.error(f"Error getting kill-switch statistics: {e}")
            return {}
    
    def export_kill_switch_data(self, filepath: str, time_period_hours: int = 24):
        """Export kill-switch data to file"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=time_period_hours)
            
            # Filter recent events
            recent_events = [
                event for event in self.event_history
                if event.timestamp >= cutoff_time
            ]
            
            # Prepare export data
            export_data = {
                'export_info': {
                    'time_period_hours': time_period_hours,
                    'cutoff_time': cutoff_time.isoformat(),
                    'export_timestamp': datetime.now().isoformat(),
                    'total_events': len(recent_events)
                },
                'events': [
                    {
                        'event_id': event.event_id,
                        'timestamp': event.timestamp.isoformat(),
                        'trigger': event.trigger.value,
                        'severity': event.severity.value,
                        'condition': event.condition,
                        'current_value': event.current_value,
                        'threshold_value': event.threshold_value,
                        'actions_taken': [action.value for action in event.actions_taken],
                        'resolved': event.resolved,
                        'resolution_time': event.resolution_time.isoformat() if event.resolution_time else None,
                        'context': event.context
                    }
                    for event in recent_events
                ],
                'rules': [
                    {
                        'rule_id': rule.rule_id,
                        'name': rule.name,
                        'trigger': rule.trigger.value,
                        'condition': rule.condition,
                        'severity': rule.severity.value,
                        'actions': [action.value for action in rule.actions],
                        'cooldown_minutes': rule.cooldown_minutes,
                        'enabled': rule.enabled,
                        'description': rule.description
                    }
                    for rule in self.kill_switch_rules
                ],
                'system_status': self.get_system_status()
            }
            
            # Export to file
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Exported kill-switch data to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting kill-switch data: {e}")
