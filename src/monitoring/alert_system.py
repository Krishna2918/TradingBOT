"""
Alert System
============

Real-time alerting system with multiple channels including
email, Slack, and custom webhooks.
"""

import asyncio
import logging
import smtplib
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import aiohttp

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Alert channels."""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    LOG = "log"


@dataclass
class AlertRule:
    """Alert rule configuration."""
    rule_id: str
    name: str
    condition: str
    severity: AlertSeverity
    channels: List[AlertChannel]
    enabled: bool
    cooldown_minutes: int
    last_triggered: Optional[datetime]


@dataclass
class Alert:
    """Alert data structure."""
    alert_id: str
    timestamp: datetime
    severity: AlertSeverity
    title: str
    message: str
    source: str
    context: Dict[str, Any]
    channels: List[AlertChannel]
    acknowledged: bool
    resolved: bool


@dataclass
class AlertConfiguration:
    """Alert system configuration."""
    email_config: Dict[str, Any]
    slack_config: Dict[str, Any]
    webhook_config: Dict[str, Any]
    default_channels: List[AlertChannel]
    global_cooldown_minutes: int


class AlertSystem:
    """Real-time alerting with multiple channels."""
    
    def __init__(self, config: Optional[AlertConfiguration] = None):
        self.config = config or self._get_default_config()
        self.alert_rules: List[AlertRule] = []
        self.alert_history: List[Alert] = []
        self.alert_counter = 0
        self.silenced_until: Optional[datetime] = None
        self.escalation_rules: Dict[AlertSeverity, int] = {
            AlertSeverity.INFO: 0,
            AlertSeverity.WARNING: 1,
            AlertSeverity.ERROR: 2,
            AlertSeverity.CRITICAL: 3
        }
        
    def _get_default_config(self) -> AlertConfiguration:
        """Get default alert configuration."""
        return AlertConfiguration(
            email_config={
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "username": "",
                "password": "",
                "from_email": "",
                "to_emails": []
            },
            slack_config={
                "webhook_url": "",
                "channel": "#alerts",
                "username": "TradingBot"
            },
            webhook_config={
                "url": "",
                "headers": {"Content-Type": "application/json"},
                "timeout": 30
            },
            default_channels=[AlertChannel.LOG],
            global_cooldown_minutes=5
        )
    
    async def configure_alerts(self, rules: List[AlertRule]) -> bool:
        """Configure alert rules."""
        try:
            self.alert_rules = rules
            logger.info(f"Configured {len(rules)} alert rules")
            return True
            
        except Exception as e:
            logger.error(f"Failed to configure alerts: {e}")
            return False
    
    async def send_alert(self, alert: Alert) -> bool:
        """Send an alert through configured channels."""
        try:
            # Check if alerts are silenced
            if self.silenced_until and datetime.now() < self.silenced_until:
                logger.info(f"Alerts are silenced until {self.silenced_until}")
                return True
            
            # Check cooldown for this alert type
            if not await self._check_cooldown(alert):
                logger.info(f"Alert cooldown active for {alert.title}")
                return True
            
            # Send alert through each channel
            success_count = 0
            total_channels = len(alert.channels)
            
            for channel in alert.channels:
                try:
                    if channel == AlertChannel.EMAIL:
                        success = await self.send_email_alert(alert)
                    elif channel == AlertChannel.SLACK:
                        success = await self.send_slack_alert(alert)
                    elif channel == AlertChannel.WEBHOOK:
                        success = await self.send_webhook_alert(alert)
                    elif channel == AlertChannel.LOG:
                        success = await self.send_log_alert(alert)
                    else:
                        logger.warning(f"Unknown alert channel: {channel}")
                        success = False
                    
                    if success:
                        success_count += 1
                        
                except Exception as e:
                    logger.error(f"Failed to send alert via {channel}: {e}")
            
            # Store alert in history
            self.alert_history.append(alert)
            
            # Log alert result
            if success_count == total_channels:
                logger.info(f"Alert sent successfully via all {total_channels} channels")
                return True
            elif success_count > 0:
                logger.warning(f"Alert sent via {success_count}/{total_channels} channels")
                return True
            else:
                logger.error("Failed to send alert via any channel")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
            return False
    
    async def _check_cooldown(self, alert: Alert) -> bool:
        """Check if alert is within cooldown period."""
        try:
            # Check global cooldown
            if self.config.global_cooldown_minutes > 0:
                recent_alerts = [
                    a for a in self.alert_history
                    if a.title == alert.title and
                    a.timestamp >= datetime.now() - timedelta(minutes=self.config.global_cooldown_minutes)
                ]
                if recent_alerts:
                    return False
            
            # Check rule-specific cooldown
            for rule in self.alert_rules:
                if rule.name == alert.title and rule.cooldown_minutes > 0:
                    if rule.last_triggered:
                        cooldown_end = rule.last_triggered + timedelta(minutes=rule.cooldown_minutes)
                        if datetime.now() < cooldown_end:
                            return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking cooldown: {e}")
            return True
    
    async def send_email_alert(self, alert: Alert) -> bool:
        """Send email alert."""
        try:
            email_config = self.config.email_config
            
            if not email_config.get("smtp_server") or not email_config.get("to_emails"):
                logger.warning("Email configuration incomplete, skipping email alert")
                return False
            
            # Create email message
            msg = MIMEMultipart()
            msg['From'] = email_config.get("from_email", "noreply@tradingbot.com")
            msg['To'] = ", ".join(email_config["to_emails"])
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            # Create email body
            body = f"""
Alert Details:
=============
Severity: {alert.severity.value.upper()}
Title: {alert.title}
Message: {alert.message}
Source: {alert.source}
Timestamp: {alert.timestamp}
Alert ID: {alert.alert_id}

Context:
{json.dumps(alert.context, indent=2)}

This is an automated alert from the AI Trading System.
"""
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(email_config["smtp_server"], email_config.get("smtp_port", 587))
            server.starttls()
            
            if email_config.get("username") and email_config.get("password"):
                server.login(email_config["username"], email_config["password"])
            
            text = msg.as_string()
            server.sendmail(msg['From'], email_config["to_emails"], text)
            server.quit()
            
            logger.info(f"Email alert sent: {alert.title}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False
    
    async def send_slack_alert(self, alert: Alert) -> bool:
        """Send Slack alert."""
        try:
            slack_config = self.config.slack_config
            
            if not slack_config.get("webhook_url"):
                logger.warning("Slack webhook URL not configured, skipping Slack alert")
                return False
            
            # Create Slack message
            color_map = {
                AlertSeverity.INFO: "good",
                AlertSeverity.WARNING: "warning",
                AlertSeverity.ERROR: "danger",
                AlertSeverity.CRITICAL: "danger"
            }
            
            slack_message = {
                "channel": slack_config.get("channel", "#alerts"),
                "username": slack_config.get("username", "TradingBot"),
                "icon_emoji": ":robot_face:",
                "attachments": [
                    {
                        "color": color_map.get(alert.severity, "good"),
                        "title": f"[{alert.severity.value.upper()}] {alert.title}",
                        "text": alert.message,
                        "fields": [
                            {
                                "title": "Source",
                                "value": alert.source,
                                "short": True
                            },
                            {
                                "title": "Timestamp",
                                "value": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                                "short": True
                            },
                            {
                                "title": "Alert ID",
                                "value": alert.alert_id,
                                "short": True
                            }
                        ],
                        "footer": "AI Trading System",
                        "ts": int(alert.timestamp.timestamp())
                    }
                ]
            }
            
            # Add context if available
            if alert.context:
                context_text = "\n".join([f"{k}: {v}" for k, v in alert.context.items()])
                slack_message["attachments"][0]["fields"].append({
                    "title": "Context",
                    "value": context_text,
                    "short": False
                })
            
            # Send to Slack
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    slack_config["webhook_url"],
                    json=slack_message,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        logger.info(f"Slack alert sent: {alert.title}")
                        return True
                    else:
                        logger.error(f"Slack alert failed with status {response.status}")
                        return False
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False
    
    async def send_webhook_alert(self, alert: Alert) -> bool:
        """Send webhook alert."""
        try:
            webhook_config = self.config.webhook_config
            
            if not webhook_config.get("url"):
                logger.warning("Webhook URL not configured, skipping webhook alert")
                return False
            
            # Create webhook payload
            payload = {
                "alert_id": alert.alert_id,
                "timestamp": alert.timestamp.isoformat(),
                "severity": alert.severity.value,
                "title": alert.title,
                "message": alert.message,
                "source": alert.source,
                "context": alert.context,
                "acknowledged": alert.acknowledged,
                "resolved": alert.resolved
            }
            
            # Send webhook
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    webhook_config["url"],
                    json=payload,
                    headers=webhook_config.get("headers", {}),
                    timeout=aiohttp.ClientTimeout(total=webhook_config.get("timeout", 30))
                ) as response:
                    if response.status in [200, 201, 202]:
                        logger.info(f"Webhook alert sent: {alert.title}")
                        return True
                    else:
                        logger.error(f"Webhook alert failed with status {response.status}")
                        return False
            
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return False
    
    async def send_log_alert(self, alert: Alert) -> bool:
        """Send log alert."""
        try:
            # Log alert based on severity
            log_message = f"ALERT [{alert.severity.value.upper()}] {alert.title}: {alert.message}"
            
            if alert.severity == AlertSeverity.CRITICAL:
                logger.critical(log_message)
            elif alert.severity == AlertSeverity.ERROR:
                logger.error(log_message)
            elif alert.severity == AlertSeverity.WARNING:
                logger.warning(log_message)
            else:
                logger.info(log_message)
            
            # Log context if available
            if alert.context:
                logger.info(f"Alert context: {json.dumps(alert.context, indent=2)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send log alert: {e}")
            return False
    
    async def get_alert_history(self) -> List[Alert]:
        """Get alert history."""
        return self.alert_history.copy()
    
    async def silence_alerts(self, duration_minutes: int) -> bool:
        """Silence alerts for specified duration."""
        try:
            self.silenced_until = datetime.now() + timedelta(minutes=duration_minutes)
            logger.info(f"Alerts silenced for {duration_minutes} minutes until {self.silenced_until}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to silence alerts: {e}")
            return False
    
    async def escalate_alert(self, alert: Alert) -> bool:
        """Escalate an alert to higher severity."""
        try:
            # Determine escalation level
            current_level = self.escalation_rules.get(alert.severity, 0)
            
            if current_level >= 3:  # Already at maximum escalation
                logger.warning(f"Alert {alert.alert_id} is already at maximum escalation level")
                return False
            
            # Find next severity level
            severity_levels = [
                AlertSeverity.INFO,
                AlertSeverity.WARNING,
                AlertSeverity.ERROR,
                AlertSeverity.CRITICAL
            ]
            
            next_level = current_level + 1
            if next_level < len(severity_levels):
                new_severity = severity_levels[next_level]
                
                # Create escalated alert
                escalated_alert = Alert(
                    alert_id=f"{alert.alert_id}_ESCALATED",
                    timestamp=datetime.now(),
                    severity=new_severity,
                    title=f"ESCALATED: {alert.title}",
                    message=f"Escalated from {alert.severity.value}: {alert.message}",
                    source=alert.source,
                    context=alert.context,
                    channels=alert.channels,
                    acknowledged=False,
                    resolved=False
                )
                
                # Send escalated alert
                success = await self.send_alert(escalated_alert)
                
                if success:
                    logger.info(f"Alert {alert.alert_id} escalated to {new_severity.value}")
                    return True
                else:
                    logger.error(f"Failed to send escalated alert for {alert.alert_id}")
                    return False
            else:
                logger.warning(f"Alert {alert.alert_id} cannot be escalated further")
                return False
                
        except Exception as e:
            logger.error(f"Failed to escalate alert: {e}")
            return False
    
    async def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        try:
            # Find alert in history
            for alert in self.alert_history:
                if alert.alert_id == alert_id:
                    alert.acknowledged = True
                    logger.info(f"Alert {alert_id} acknowledged")
                    return True
            
            logger.warning(f"Alert {alert_id} not found in history")
            return False
            
        except Exception as e:
            logger.error(f"Failed to acknowledge alert: {e}")
            return False
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        try:
            # Find alert in history
            for alert in self.alert_history:
                if alert.alert_id == alert_id:
                    alert.resolved = True
                    logger.info(f"Alert {alert_id} resolved")
                    return True
            
            logger.warning(f"Alert {alert_id} not found in history")
            return False
            
        except Exception as e:
            logger.error(f"Failed to resolve alert: {e}")
            return False


# Global alert system instance
_alert_system: Optional[AlertSystem] = None


def get_alert_system() -> AlertSystem:
    """Get global alert system instance."""
    global _alert_system
    if _alert_system is None:
        _alert_system = AlertSystem()
    return _alert_system


async def send_alert(alert: Alert) -> bool:
    """Send an alert."""
    system = get_alert_system()
    return await system.send_alert(alert)


async def send_email_alert(alert: Alert) -> bool:
    """Send email alert."""
    system = get_alert_system()
    return await system.send_email_alert(alert)


async def send_slack_alert(alert: Alert) -> bool:
    """Send Slack alert."""
    system = get_alert_system()
    return await system.send_slack_alert(alert)


async def get_alert_history() -> List[Alert]:
    """Get alert history."""
    system = get_alert_system()
    return await system.get_alert_history()


async def silence_alerts(duration_minutes: int) -> bool:
    """Silence alerts for specified duration."""
    system = get_alert_system()
    return await system.silence_alerts(duration_minutes)


async def escalate_alert(alert: Alert) -> bool:
    """Escalate an alert."""
    system = get_alert_system()
    return await system.escalate_alert(alert)
