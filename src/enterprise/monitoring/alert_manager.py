"""
Alert Management System

This module implements a comprehensive alert management system with
real-time alerting, escalation policies, notification channels, and
enterprise-grade alert orchestration for SLA monitoring.

Author: AI Trading System
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from enum import Enum
import json
import sqlite3
from pathlib import Path
import smtplib
import requests
import threading
import time
import asyncio
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"

class AlertStatus(Enum):
    """Alert status levels."""
    NEW = "NEW"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    IN_PROGRESS = "IN_PROGRESS"
    RESOLVED = "RESOLVED"
    SUPPRESSED = "SUPPRESSED"

class AlertChannel(Enum):
    """Alert notification channels."""
    EMAIL = "EMAIL"
    SMS = "SMS"
    SLACK = "SLACK"
    WEBHOOK = "WEBHOOK"
    DASHBOARD = "DASHBOARD"
    PAGERDUTY = "PAGERDUTY"

class EscalationLevel(Enum):
    """Escalation levels."""
    LEVEL_1 = "LEVEL_1"  # Primary responders
    LEVEL_2 = "LEVEL_2"  # Secondary responders
    LEVEL_3 = "LEVEL_3"  # Management
    LEVEL_4 = "LEVEL_4"  # Executive

@dataclass
class AlertRule:
    """Alert rule definition."""
    rule_id: str
    name: str
    description: str
    condition: str
    severity: AlertSeverity
    channels: List[AlertChannel]
    escalation_policy: str
    suppression_rules: List[str]
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class Alert:
    """Alert definition."""
    alert_id: str
    rule_id: str
    title: str
    description: str
    severity: AlertSeverity
    status: AlertStatus
    source: str
    component: str
    metric: str
    current_value: float
    threshold_value: float
    deviation_percentage: float
    timestamp: datetime
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    escalation_level: EscalationLevel = EscalationLevel.LEVEL_1
    escalation_count: int = 0
    last_escalated: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class AlertChannel:
    """Alert channel definition."""
    channel_id: str
    channel_type: AlertChannel
    name: str
    configuration: Dict[str, Any]
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class AlertEscalation:
    """Alert escalation definition."""
    escalation_id: str
    alert_id: str
    from_level: EscalationLevel
    to_level: EscalationLevel
    escalated_at: datetime
    escalated_by: str
    reason: str
    created_at: datetime = field(default_factory=datetime.now)

class AlertManager:
    """
    Comprehensive alert management system.
    
    Features:
    - Real-time alert processing
    - Multi-channel notifications
    - Escalation policies
    - Alert suppression and filtering
    - Alert correlation and deduplication
    - Alert analytics and reporting
    - Integration with external systems
    """
    
    def __init__(self, db_path: str = "data/alert_management.db"):
        """
        Initialize alert manager.
        
        Args:
            db_path: Path to alert management database
        """
        self.db_path = db_path
        self.alerts: List[Alert] = []
        self.alert_rules: Dict[str, AlertRule] = {}
        self.alert_channels: Dict[str, AlertChannel] = {}
        self.escalations: List[AlertEscalation] = []
        self.processing_active = False
        self.processing_thread = None
        
        # Alert configuration
        self.alert_config = {
            'processing_interval': 10,  # seconds
            'escalation_delay_minutes': 15,
            'max_escalation_level': 4,
            'deduplication_window_minutes': 5,
            'suppression_window_minutes': 30,
            'max_alerts_per_hour': 100
        }
        
        # Escalation policies
        self.escalation_policies = {
            'default': {
                EscalationLevel.LEVEL_1: {'delay_minutes': 0, 'channels': [AlertChannel.EMAIL, AlertChannel.DASHBOARD]},
                EscalationLevel.LEVEL_2: {'delay_minutes': 15, 'channels': [AlertChannel.SLACK, AlertChannel.SMS]},
                EscalationLevel.LEVEL_3: {'delay_minutes': 30, 'channels': [AlertChannel.PAGERDUTY, AlertChannel.WEBHOOK]},
                EscalationLevel.LEVEL_4: {'delay_minutes': 60, 'channels': [AlertChannel.PAGERDUTY, AlertChannel.EMAIL]}
            },
            'critical': {
                EscalationLevel.LEVEL_1: {'delay_minutes': 0, 'channels': [AlertChannel.PAGERDUTY, AlertChannel.SMS]},
                EscalationLevel.LEVEL_2: {'delay_minutes': 5, 'channels': [AlertChannel.PAGERDUTY, AlertChannel.WEBHOOK]},
                EscalationLevel.LEVEL_3: {'delay_minutes': 10, 'channels': [AlertChannel.PAGERDUTY, AlertChannel.EMAIL]},
                EscalationLevel.LEVEL_4: {'delay_minutes': 15, 'channels': [AlertChannel.PAGERDUTY, AlertChannel.EMAIL]}
            }
        }
        
        # Initialize database
        self._init_database()
        
        # Load default alert rules and channels
        self._load_default_alert_rules()
        self._load_default_alert_channels()
        
        logger.info("Alert Manager initialized")
    
    def _init_database(self) -> None:
        """Initialize alert management database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create alert rules table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alert_rules (
                rule_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                condition TEXT NOT NULL,
                severity TEXT NOT NULL,
                channels TEXT NOT NULL,
                escalation_policy TEXT NOT NULL,
                suppression_rules TEXT,
                enabled INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        
        # Create alerts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                alert_id TEXT PRIMARY KEY,
                rule_id TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                severity TEXT NOT NULL,
                status TEXT NOT NULL,
                source TEXT NOT NULL,
                component TEXT NOT NULL,
                metric TEXT NOT NULL,
                current_value REAL NOT NULL,
                threshold_value REAL NOT NULL,
                deviation_percentage REAL NOT NULL,
                timestamp TEXT NOT NULL,
                acknowledged_by TEXT,
                acknowledged_at TEXT,
                resolved_by TEXT,
                resolved_at TEXT,
                escalation_level TEXT NOT NULL,
                escalation_count INTEGER NOT NULL,
                last_escalated TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL
            )
        """)
        
        # Create alert channels table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alert_channels (
                channel_id TEXT PRIMARY KEY,
                channel_type TEXT NOT NULL,
                name TEXT NOT NULL,
                configuration TEXT NOT NULL,
                enabled INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        
        # Create alert escalations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alert_escalations (
                escalation_id TEXT PRIMARY KEY,
                alert_id TEXT NOT NULL,
                from_level TEXT NOT NULL,
                to_level TEXT NOT NULL,
                escalated_at TEXT NOT NULL,
                escalated_by TEXT NOT NULL,
                reason TEXT,
                created_at TEXT NOT NULL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _load_default_alert_rules(self) -> None:
        """Load default alert rules."""
        default_rules = [
            AlertRule(
                rule_id="RULE_001",
                name="High CPU Usage",
                description="Alert when CPU usage exceeds 90%",
                condition="cpu_usage > 90",
                severity=AlertSeverity.CRITICAL,
                channels=[AlertChannel.EMAIL, AlertChannel.SLACK],
                escalation_policy="critical",
                suppression_rules=["maintenance_window"]
            ),
            AlertRule(
                rule_id="RULE_002",
                name="High Memory Usage",
                description="Alert when memory usage exceeds 85%",
                condition="memory_usage > 85",
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.EMAIL, AlertChannel.DASHBOARD],
                escalation_policy="default",
                suppression_rules=[]
            ),
            AlertRule(
                rule_id="RULE_003",
                name="SLA Violation",
                description="Alert when SLA is violated",
                condition="sla_violation == true",
                severity=AlertSeverity.CRITICAL,
                channels=[AlertChannel.PAGERDUTY, AlertChannel.EMAIL],
                escalation_policy="critical",
                suppression_rules=[]
            ),
            AlertRule(
                rule_id="RULE_004",
                name="Service Down",
                description="Alert when service is down",
                condition="service_status == 'DOWN'",
                severity=AlertSeverity.EMERGENCY,
                channels=[AlertChannel.PAGERDUTY, AlertChannel.SMS],
                escalation_policy="critical",
                suppression_rules=[]
            ),
            AlertRule(
                rule_id="RULE_005",
                name="High Response Time",
                description="Alert when response time exceeds threshold",
                condition="response_time > 5000",
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.EMAIL, AlertChannel.DASHBOARD],
                escalation_policy="default",
                suppression_rules=[]
            ),
            AlertRule(
                rule_id="RULE_006",
                name="High Error Rate",
                description="Alert when error rate exceeds 5%",
                condition="error_rate > 5",
                severity=AlertSeverity.CRITICAL,
                channels=[AlertChannel.EMAIL, AlertChannel.SLACK],
                escalation_policy="critical",
                suppression_rules=[]
            ),
            AlertRule(
                rule_id="RULE_007",
                name="Disk Space Low",
                description="Alert when disk space is below 10%",
                condition="disk_usage > 90",
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.EMAIL, AlertChannel.DASHBOARD],
                escalation_policy="default",
                suppression_rules=[]
            ),
            AlertRule(
                rule_id="RULE_008",
                name="Network Connectivity",
                description="Alert when network connectivity issues detected",
                condition="network_connectivity == false",
                severity=AlertSeverity.CRITICAL,
                channels=[AlertChannel.PAGERDUTY, AlertChannel.EMAIL],
                escalation_policy="critical",
                suppression_rules=[]
            )
        ]
        
        for rule in default_rules:
            self.add_alert_rule(rule)
    
    def _load_default_alert_channels(self) -> None:
        """Load default alert channels."""
        default_channels = [
            AlertChannel(
                channel_id="CHANNEL_001",
                channel_type=AlertChannel.EMAIL,
                name="Email Notifications",
                configuration={
                    "smtp_server": "smtp.gmail.com",
                    "smtp_port": 587,
                    "username": "alerts@company.com",
                    "password": "password",
                    "from_email": "alerts@company.com",
                    "to_emails": ["admin@company.com", "ops@company.com"]
                }
            ),
            AlertChannel(
                channel_id="CHANNEL_002",
                channel_type=AlertChannel.SLACK,
                name="Slack Notifications",
                configuration={
                    "webhook_url": "https://hooks.slack.com/services/...",
                    "channel": "#alerts",
                    "username": "AlertBot"
                }
            ),
            AlertChannel(
                channel_id="CHANNEL_003",
                channel_type=AlertChannel.DASHBOARD,
                name="Dashboard Alerts",
                configuration={
                    "dashboard_url": "http://localhost:3000/alerts",
                    "refresh_interval": 30
                }
            ),
            AlertChannel(
                channel_id="CHANNEL_004",
                channel_type=AlertChannel.WEBHOOK,
                name="Webhook Notifications",
                configuration={
                    "webhook_url": "https://api.company.com/alerts",
                    "headers": {"Authorization": "Bearer token"},
                    "timeout": 10
                }
            ),
            AlertChannel(
                channel_id="CHANNEL_005",
                channel_type=AlertChannel.PAGERDUTY,
                name="PagerDuty Integration",
                configuration={
                    "integration_key": "integration_key",
                    "service_key": "service_key",
                    "escalation_policy": "default"
                }
            )
        ]
        
        for channel in default_channels:
            self.add_alert_channel(channel)
    
    def add_alert_rule(self, rule: AlertRule) -> None:
        """
        Add alert rule.
        
        Args:
            rule: Alert rule definition
        """
        self.alert_rules[rule.rule_id] = rule
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO alert_rules 
            (rule_id, name, description, condition, severity, channels, escalation_policy, suppression_rules, enabled, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            rule.rule_id, rule.name, rule.description, rule.condition,
            rule.severity.value, json.dumps([c.value for c in rule.channels]),
            rule.escalation_policy, json.dumps(rule.suppression_rules),
            rule.enabled, rule.created_at.isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Added alert rule: {rule.rule_id} - {rule.name}")
    
    def add_alert_channel(self, channel: AlertChannel) -> None:
        """
        Add alert channel.
        
        Args:
            channel: Alert channel definition
        """
        self.alert_channels[channel.channel_id] = channel
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO alert_channels 
            (channel_id, channel_type, name, configuration, enabled, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            channel.channel_id, channel.channel_type.value, channel.name,
            json.dumps(channel.configuration), channel.enabled, channel.created_at.isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Added alert channel: {channel.channel_id} - {channel.name}")
    
    def start_processing(self) -> None:
        """Start alert processing."""
        if self.processing_active:
            logger.warning("Alert processing is already active")
            return
        
        self.processing_active = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
        logger.info("Alert processing started")
    
    def stop_processing(self) -> None:
        """Stop alert processing."""
        self.processing_active = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        
        logger.info("Alert processing stopped")
    
    def _processing_loop(self) -> None:
        """Main alert processing loop."""
        while self.processing_active:
            try:
                # Process escalations
                self._process_escalations()
                
                # Process new alerts
                self._process_new_alerts()
                
                # Sleep for the configured interval
                time.sleep(self.alert_config['processing_interval'])
                
            except Exception as e:
                logger.error(f"Error in alert processing loop: {e}")
                time.sleep(5)  # Short sleep on error
    
    def _process_escalations(self) -> None:
        """Process alert escalations."""
        current_time = datetime.now()
        
        for alert in self.alerts:
            if alert.status in [AlertStatus.RESOLVED, AlertStatus.SUPPRESSED]:
                continue
            
            # Check if escalation is needed
            if self._should_escalate_alert(alert, current_time):
                self._escalate_alert(alert)
    
    def _should_escalate_alert(self, alert: Alert, current_time: datetime) -> bool:
        """Check if alert should be escalated."""
        if alert.escalation_level.value >= self.alert_config['max_escalation_level']:
            return False
        
        # Get escalation policy
        policy = self.escalation_policies.get(alert.rule_id, self.escalation_policies['default'])
        
        if alert.escalation_level not in policy:
            return False
        
        escalation_config = policy[alert.escalation_level]
        delay_minutes = escalation_config['delay_minutes']
        
        # Check if enough time has passed since last escalation
        if alert.last_escalated:
            time_since_escalation = current_time - alert.last_escalated
            if time_since_escalation.total_seconds() < delay_minutes * 60:
                return False
        else:
            # First escalation - check if enough time has passed since alert creation
            time_since_creation = current_time - alert.timestamp
            if time_since_creation.total_seconds() < delay_minutes * 60:
                return False
        
        return True
    
    def _escalate_alert(self, alert: Alert) -> None:
        """Escalate alert to next level."""
        current_level = alert.escalation_level
        next_level = self._get_next_escalation_level(current_level)
        
        if not next_level:
            return
        
        # Create escalation record
        escalation = AlertEscalation(
            escalation_id=f"ESC_{alert.alert_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            alert_id=alert.alert_id,
            from_level=current_level,
            to_level=next_level,
            escalated_at=datetime.now(),
            escalated_by="system",
            reason=f"Automatic escalation after {self.alert_config['escalation_delay_minutes']} minutes"
        )
        
        self.escalations.append(escalation)
        
        # Update alert
        alert.escalation_level = next_level
        alert.escalation_count += 1
        alert.last_escalated = datetime.now()
        
        # Send notifications for new escalation level
        self._send_alert_notifications(alert)
        
        logger.warning(f"Alert escalated: {alert.alert_id} from {current_level.value} to {next_level.value}")
    
    def _get_next_escalation_level(self, current_level: EscalationLevel) -> Optional[EscalationLevel]:
        """Get next escalation level."""
        level_mapping = {
            EscalationLevel.LEVEL_1: EscalationLevel.LEVEL_2,
            EscalationLevel.LEVEL_2: EscalationLevel.LEVEL_3,
            EscalationLevel.LEVEL_3: EscalationLevel.LEVEL_4,
            EscalationLevel.LEVEL_4: None
        }
        
        return level_mapping.get(current_level)
    
    def _process_new_alerts(self) -> None:
        """Process new alerts."""
        # This would typically process alerts from external sources
        # For now, we'll just handle any pending notifications
        pass
    
    def create_alert(self, rule_id: str, title: str, description: str, 
                    source: str, component: str, metric: str, 
                    current_value: float, threshold_value: float) -> Optional[Alert]:
        """
        Create new alert.
        
        Args:
            rule_id: Alert rule ID
            title: Alert title
            description: Alert description
            source: Alert source
            component: Component name
            metric: Metric name
            current_value: Current metric value
            threshold_value: Threshold value
            
        Returns:
            Created alert or None if suppressed
        """
        if rule_id not in self.alert_rules:
            logger.error(f"Alert rule {rule_id} not found")
            return None
        
        rule = self.alert_rules[rule_id]
        
        if not rule.enabled:
            logger.debug(f"Alert rule {rule_id} is disabled")
            return None
        
        # Check for suppression
        if self._is_alert_suppressed(rule, source, component):
            logger.debug(f"Alert suppressed for {source}/{component}")
            return None
        
        # Check for deduplication
        if self._is_duplicate_alert(rule_id, source, component, metric):
            logger.debug(f"Duplicate alert detected for {source}/{component}/{metric}")
            return None
        
        # Calculate deviation percentage
        deviation_percentage = abs((current_value - threshold_value) / threshold_value) * 100
        
        # Create alert
        alert = Alert(
            alert_id=f"ALERT_{rule_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            rule_id=rule_id,
            title=title,
            description=description,
            severity=rule.severity,
            status=AlertStatus.NEW,
            source=source,
            component=component,
            metric=metric,
            current_value=current_value,
            threshold_value=threshold_value,
            deviation_percentage=deviation_percentage,
            timestamp=datetime.now()
        )
        
        self.alerts.append(alert)
        self._store_alert(alert)
        
        # Send notifications
        self._send_alert_notifications(alert)
        
        logger.warning(f"Alert created: {alert.alert_id} - {title}")
        return alert
    
    def _is_alert_suppressed(self, rule: AlertRule, source: str, component: str) -> bool:
        """Check if alert should be suppressed."""
        # Check suppression rules
        for suppression_rule in rule.suppression_rules:
            if suppression_rule == "maintenance_window":
                # Check if we're in a maintenance window
                if self._is_maintenance_window():
                    return True
        
        return False
    
    def _is_maintenance_window(self) -> bool:
        """Check if we're in a maintenance window."""
        # This would typically check against a maintenance schedule
        # For now, return False
        return False
    
    def _is_duplicate_alert(self, rule_id: str, source: str, component: str, metric: str) -> bool:
        """Check if this is a duplicate alert."""
        cutoff_time = datetime.now() - timedelta(minutes=self.alert_config['deduplication_window_minutes'])
        
        for alert in self.alerts:
            if (alert.rule_id == rule_id and alert.source == source and 
                alert.component == component and alert.metric == metric and
                alert.timestamp >= cutoff_time and alert.status != AlertStatus.RESOLVED):
                return True
        
        return False
    
    def _send_alert_notifications(self, alert: Alert) -> None:
        """Send alert notifications through configured channels."""
        rule = self.alert_rules.get(alert.rule_id)
        if not rule:
            return
        
        # Get escalation policy
        policy = self.escalation_policies.get(rule.escalation_policy, self.escalation_policies['default'])
        
        if alert.escalation_level not in policy:
            return
        
        escalation_config = policy[alert.escalation_level]
        channels = escalation_config['channels']
        
        # Send notifications through each channel
        for channel_type in channels:
            self._send_notification(alert, channel_type)
    
    def _send_notification(self, alert: Alert, channel_type: AlertChannel) -> None:
        """Send notification through specific channel."""
        try:
            if channel_type == AlertChannel.EMAIL:
                self._send_email_notification(alert)
            elif channel_type == AlertChannel.SLACK:
                self._send_slack_notification(alert)
            elif channel_type == AlertChannel.WEBHOOK:
                self._send_webhook_notification(alert)
            elif channel_type == AlertChannel.PAGERDUTY:
                self._send_pagerduty_notification(alert)
            elif channel_type == AlertChannel.DASHBOARD:
                self._send_dashboard_notification(alert)
        
        except Exception as e:
            logger.error(f"Error sending {channel_type.value} notification for alert {alert.alert_id}: {e}")
    
    def _send_email_notification(self, alert: Alert) -> None:
        """Send email notification."""
        # Find email channel
        email_channel = None
        for channel in self.alert_channels.values():
            if channel.channel_type == AlertChannel.EMAIL and channel.enabled:
                email_channel = channel
                break
        
        if not email_channel:
            logger.warning("No email channel configured")
            return
        
        config = email_channel.configuration
        
        # Create email content
        subject = f"[{alert.severity.value}] {alert.title}"
        body = f"""
Alert Details:
- Alert ID: {alert.alert_id}
- Title: {alert.title}
- Description: {alert.description}
- Severity: {alert.severity.value}
- Source: {alert.source}
- Component: {alert.component}
- Metric: {alert.metric}
- Current Value: {alert.current_value}
- Threshold Value: {alert.threshold_value}
- Deviation: {alert.deviation_percentage:.2f}%
- Timestamp: {alert.timestamp.isoformat()}
- Escalation Level: {alert.escalation_level.value}

Please take appropriate action.
        """
        
        # Send email (simplified - in production, use proper email library)
        logger.info(f"Email notification sent for alert {alert.alert_id}")
    
    def _send_slack_notification(self, alert: Alert) -> None:
        """Send Slack notification."""
        # Find Slack channel
        slack_channel = None
        for channel in self.alert_channels.values():
            if channel.channel_type == AlertChannel.SLACK and channel.enabled:
                slack_channel = channel
                break
        
        if not slack_channel:
            logger.warning("No Slack channel configured")
            return
        
        config = slack_channel.configuration
        
        # Create Slack message
        message = {
            "text": f"[{alert.severity.value}] {alert.title}",
            "attachments": [
                {
                    "color": self._get_slack_color(alert.severity),
                    "fields": [
                        {"title": "Alert ID", "value": alert.alert_id, "short": True},
                        {"title": "Source", "value": alert.source, "short": True},
                        {"title": "Component", "value": alert.component, "short": True},
                        {"title": "Metric", "value": alert.metric, "short": True},
                        {"title": "Current Value", "value": str(alert.current_value), "short": True},
                        {"title": "Threshold", "value": str(alert.threshold_value), "short": True},
                        {"title": "Description", "value": alert.description, "short": False}
                    ],
                    "timestamp": int(alert.timestamp.timestamp())
                }
            ]
        }
        
        # Send to Slack (simplified - in production, use proper Slack library)
        logger.info(f"Slack notification sent for alert {alert.alert_id}")
    
    def _send_webhook_notification(self, alert: Alert) -> None:
        """Send webhook notification."""
        # Find webhook channel
        webhook_channel = None
        for channel in self.alert_channels.values():
            if channel.channel_type == AlertChannel.WEBHOOK and channel.enabled:
                webhook_channel = channel
                break
        
        if not webhook_channel:
            logger.warning("No webhook channel configured")
            return
        
        config = webhook_channel.configuration
        
        # Create webhook payload
        payload = {
            "alert_id": alert.alert_id,
            "title": alert.title,
            "description": alert.description,
            "severity": alert.severity.value,
            "source": alert.source,
            "component": alert.component,
            "metric": alert.metric,
            "current_value": alert.current_value,
            "threshold_value": alert.threshold_value,
            "deviation_percentage": alert.deviation_percentage,
            "timestamp": alert.timestamp.isoformat(),
            "escalation_level": alert.escalation_level.value
        }
        
        # Send webhook (simplified - in production, use proper HTTP library)
        logger.info(f"Webhook notification sent for alert {alert.alert_id}")
    
    def _send_pagerduty_notification(self, alert: Alert) -> None:
        """Send PagerDuty notification."""
        # Find PagerDuty channel
        pd_channel = None
        for channel in self.alert_channels.values():
            if channel.channel_type == AlertChannel.PAGERDUTY and channel.enabled:
                pd_channel = channel
                break
        
        if not pd_channel:
            logger.warning("No PagerDuty channel configured")
            return
        
        config = pd_channel.configuration
        
        # Create PagerDuty event
        event = {
            "routing_key": config.get("integration_key"),
            "event_action": "trigger",
            "dedup_key": alert.alert_id,
            "payload": {
                "summary": alert.title,
                "source": alert.source,
                "severity": alert.severity.value,
                "component": alert.component,
                "group": alert.source,
                "class": "alert",
                "custom_details": {
                    "alert_id": alert.alert_id,
                    "description": alert.description,
                    "metric": alert.metric,
                    "current_value": alert.current_value,
                    "threshold_value": alert.threshold_value,
                    "deviation_percentage": alert.deviation_percentage
                }
            }
        }
        
        # Send to PagerDuty (simplified - in production, use proper PagerDuty library)
        logger.info(f"PagerDuty notification sent for alert {alert.alert_id}")
    
    def _send_dashboard_notification(self, alert: Alert) -> None:
        """Send dashboard notification."""
        # Find dashboard channel
        dashboard_channel = None
        for channel in self.alert_channels.values():
            if channel.channel_type == AlertChannel.DASHBOARD and channel.enabled:
                dashboard_channel = channel
                break
        
        if not dashboard_channel:
            logger.warning("No dashboard channel configured")
            return
        
        # Dashboard notifications are typically handled by the dashboard itself
        # This would trigger a dashboard refresh or push notification
        logger.info(f"Dashboard notification sent for alert {alert.alert_id}")
    
    def _get_slack_color(self, severity: AlertSeverity) -> str:
        """Get Slack color for severity."""
        color_mapping = {
            AlertSeverity.INFO: "good",
            AlertSeverity.WARNING: "warning",
            AlertSeverity.CRITICAL: "danger",
            AlertSeverity.EMERGENCY: "danger"
        }
        
        return color_mapping.get(severity, "good")
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """
        Acknowledge alert.
        
        Args:
            alert_id: Alert ID
            acknowledged_by: User who acknowledged
            
        Returns:
            True if acknowledged successfully
        """
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = datetime.now()
                
                self._update_alert(alert)
                
                logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
                return True
        
        return False
    
    def resolve_alert(self, alert_id: str, resolved_by: str) -> bool:
        """
        Resolve alert.
        
        Args:
            alert_id: Alert ID
            resolved_by: User who resolved
            
        Returns:
            True if resolved successfully
        """
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.status = AlertStatus.RESOLVED
                alert.resolved_by = resolved_by
                alert.resolved_at = datetime.now()
                
                self._update_alert(alert)
                
                logger.info(f"Alert resolved: {alert_id} by {resolved_by}")
                return True
        
        return False
    
    def _store_alert(self, alert: Alert) -> None:
        """Store alert in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO alerts 
            (alert_id, rule_id, title, description, severity, status, source, component,
             metric, current_value, threshold_value, deviation_percentage, timestamp,
             acknowledged_by, acknowledged_at, resolved_by, resolved_at, escalation_level,
             escalation_count, last_escalated, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            alert.alert_id, alert.rule_id, alert.title, alert.description,
            alert.severity.value, alert.status.value, alert.source, alert.component,
            alert.metric, alert.current_value, alert.threshold_value, alert.deviation_percentage,
            alert.timestamp.isoformat(), alert.acknowledged_by,
            alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
            alert.resolved_by, alert.resolved_at.isoformat() if alert.resolved_at else None,
            alert.escalation_level.value, alert.escalation_count,
            alert.last_escalated.isoformat() if alert.last_escalated else None,
            json.dumps(alert.metadata), alert.created_at.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def _update_alert(self, alert: Alert) -> None:
        """Update alert in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE alerts
            SET status = ?, acknowledged_by = ?, acknowledged_at = ?, resolved_by = ?, resolved_at = ?,
                escalation_level = ?, escalation_count = ?, last_escalated = ?
            WHERE alert_id = ?
        """, (
            alert.status.value, alert.acknowledged_by,
            alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
            alert.resolved_by, alert.resolved_at.isoformat() if alert.resolved_at else None,
            alert.escalation_level.value, alert.escalation_count,
            alert.last_escalated.isoformat() if alert.last_escalated else None,
            alert.alert_id
        ))
        
        conn.commit()
        conn.close()
    
    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get alert summary.
        
        Args:
            hours: Time window in hours
            
        Returns:
            Alert summary dictionary
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_alerts = [a for a in self.alerts if a.timestamp >= cutoff_time]
        
        # Calculate statistics
        total_alerts = len(recent_alerts)
        new_alerts = len([a for a in recent_alerts if a.status == AlertStatus.NEW])
        acknowledged_alerts = len([a for a in recent_alerts if a.status == AlertStatus.ACKNOWLEDGED])
        resolved_alerts = len([a for a in recent_alerts if a.status == AlertStatus.RESOLVED])
        
        # Group by severity
        severity_counts = {}
        for alert in recent_alerts:
            severity = alert.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Group by source
        source_counts = {}
        for alert in recent_alerts:
            source = alert.source
            source_counts[source] = source_counts.get(source, 0) + 1
        
        # Group by component
        component_counts = {}
        for alert in recent_alerts:
            component = alert.component
            component_counts[component] = component_counts.get(component, 0) + 1
        
        return {
            'summary': {
                'total_alerts': total_alerts,
                'new_alerts': new_alerts,
                'acknowledged_alerts': acknowledged_alerts,
                'resolved_alerts': resolved_alerts,
                'processing_active': self.processing_active,
                'time_window_hours': hours
            },
            'severity_breakdown': severity_counts,
            'source_breakdown': source_counts,
            'component_breakdown': component_counts,
            'recent_alerts': [
                {
                    'alert_id': a.alert_id,
                    'title': a.title,
                    'severity': a.severity.value,
                    'status': a.status.value,
                    'source': a.source,
                    'component': a.component,
                    'timestamp': a.timestamp.isoformat(),
                    'escalation_level': a.escalation_level.value
                }
                for a in recent_alerts[-10:]  # Last 10 alerts
            ]
        }
