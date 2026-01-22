"""
Audit Trail System
==================

Complete audit trail system for compliance and tracking
all system activities, changes, and transactions.
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import hashlib
import uuid

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Audit event types."""
    ORDER_CREATION = "order_creation"
    ORDER_EXECUTION = "order_execution"
    ORDER_CANCELLATION = "order_cancellation"
    POSITION_OPEN = "position_open"
    POSITION_CLOSE = "position_close"
    POSITION_UPDATE = "position_update"
    PARAMETER_CHANGE = "parameter_change"
    CONFIGURATION_CHANGE = "configuration_change"
    USER_ACTION = "user_action"
    SYSTEM_EVENT = "system_event"
    SECURITY_EVENT = "security_event"
    ERROR_EVENT = "error_event"


class AuditSeverity(Enum):
    """Audit event severity."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Audit event data structure."""
    event_id: str
    timestamp: datetime
    event_type: AuditEventType
    severity: AuditSeverity
    user_id: Optional[str]
    session_id: Optional[str]
    source: str
    action: str
    details: Dict[str, Any]
    before_state: Optional[Dict[str, Any]]
    after_state: Optional[Dict[str, Any]]
    ip_address: Optional[str]
    user_agent: Optional[str]
    checksum: str


@dataclass
class AuditReport:
    """Audit report data structure."""
    report_id: str
    generated_at: datetime
    period_start: datetime
    period_end: datetime
    total_events: int
    events_by_type: Dict[str, int]
    events_by_severity: Dict[str, int]
    events_by_user: Dict[str, int]
    critical_events: List[AuditEvent]
    summary: str


class AuditTrail:
    """Complete audit trail for compliance."""
    
    def __init__(self):
        self.audit_events: List[AuditEvent] = []
        self.event_counter = 0
        self.retention_days = 365  # Keep audit events for 1 year
        self.max_events = 100000  # Maximum number of events to keep in memory
        
    async def log_order_creation(self, order: Dict[str, Any], mode: str) -> str:
        """Log order creation event."""
        try:
            event_id = await self._create_audit_event(
                event_type=AuditEventType.ORDER_CREATION,
                severity=AuditSeverity.MEDIUM,
                source="trading_system",
                action="create_order",
                details={
                    "order_id": order.get("order_id"),
                    "symbol": order.get("symbol"),
                    "side": order.get("side"),
                    "quantity": order.get("quantity"),
                    "price": order.get("price"),
                    "mode": mode
                },
                after_state=order
            )
            
            logger.info(f"Order creation logged: {event_id}")
            return event_id
            
        except Exception as e:
            logger.error(f"Failed to log order creation: {e}")
            return "AUDIT_LOG_FAILED"
    
    async def log_order_execution(self, order: Dict[str, Any], mode: str) -> str:
        """Log order execution event."""
        try:
            event_id = await self._create_audit_event(
                event_type=AuditEventType.ORDER_EXECUTION,
                severity=AuditSeverity.HIGH,
                source="trading_system",
                action="execute_order",
                details={
                    "order_id": order.get("order_id"),
                    "symbol": order.get("symbol"),
                    "side": order.get("side"),
                    "quantity": order.get("quantity"),
                    "execution_price": order.get("execution_price"),
                    "execution_time": order.get("execution_time"),
                    "mode": mode
                },
                after_state=order
            )
            
            logger.info(f"Order execution logged: {event_id}")
            return event_id
            
        except Exception as e:
            logger.error(f"Failed to log order execution: {e}")
            return "AUDIT_LOG_FAILED"
    
    async def log_order_cancellation(self, order: Dict[str, Any], mode: str) -> str:
        """Log order cancellation event."""
        try:
            event_id = await self._create_audit_event(
                event_type=AuditEventType.ORDER_CANCELLATION,
                severity=AuditSeverity.MEDIUM,
                source="trading_system",
                action="cancel_order",
                details={
                    "order_id": order.get("order_id"),
                    "symbol": order.get("symbol"),
                    "side": order.get("side"),
                    "quantity": order.get("quantity"),
                    "cancellation_reason": order.get("cancellation_reason"),
                    "mode": mode
                },
                before_state=order
            )
            
            logger.info(f"Order cancellation logged: {event_id}")
            return event_id
            
        except Exception as e:
            logger.error(f"Failed to log order cancellation: {e}")
            return "AUDIT_LOG_FAILED"
    
    async def log_position_change(self, position: Dict[str, Any], mode: str) -> str:
        """Log position change event."""
        try:
            event_id = await self._create_audit_event(
                event_type=AuditEventType.POSITION_UPDATE,
                severity=AuditSeverity.MEDIUM,
                source="trading_system",
                action="update_position",
                details={
                    "position_id": position.get("position_id"),
                    "symbol": position.get("symbol"),
                    "quantity": position.get("quantity"),
                    "entry_price": position.get("entry_price"),
                    "current_price": position.get("current_price"),
                    "pnl": position.get("pnl"),
                    "mode": mode
                },
                after_state=position
            )
            
            logger.info(f"Position change logged: {event_id}")
            return event_id
            
        except Exception as e:
            logger.error(f"Failed to log position change: {e}")
            return "AUDIT_LOG_FAILED"
    
    async def log_position_closure(self, position: Dict[str, Any], mode: str) -> str:
        """Log position closure event."""
        try:
            event_id = await self._create_audit_event(
                event_type=AuditEventType.POSITION_CLOSE,
                severity=AuditSeverity.HIGH,
                source="trading_system",
                action="close_position",
                details={
                    "position_id": position.get("position_id"),
                    "symbol": position.get("symbol"),
                    "quantity": position.get("quantity"),
                    "entry_price": position.get("entry_price"),
                    "exit_price": position.get("exit_price"),
                    "pnl": position.get("pnl"),
                    "hold_duration": position.get("hold_duration"),
                    "mode": mode
                },
                before_state=position
            )
            
            logger.info(f"Position closure logged: {event_id}")
            return event_id
            
        except Exception as e:
            logger.error(f"Failed to log position closure: {e}")
            return "AUDIT_LOG_FAILED"
    
    async def log_parameter_change(self, param: str, old_value: Any, new_value: Any) -> str:
        """Log parameter change event."""
        try:
            event_id = await self._create_audit_event(
                event_type=AuditEventType.PARAMETER_CHANGE,
                severity=AuditSeverity.MEDIUM,
                source="configuration_system",
                action="change_parameter",
                details={
                    "parameter": param,
                    "old_value": str(old_value),
                    "new_value": str(new_value),
                    "change_type": "parameter_update"
                },
                before_state={param: old_value},
                after_state={param: new_value}
            )
            
            logger.info(f"Parameter change logged: {event_id}")
            return event_id
            
        except Exception as e:
            logger.error(f"Failed to log parameter change: {e}")
            return "AUDIT_LOG_FAILED"
    
    async def log_configuration_change(self, config: Dict[str, Any]) -> str:
        """Log configuration change event."""
        try:
            event_id = await self._create_audit_event(
                event_type=AuditEventType.CONFIGURATION_CHANGE,
                severity=AuditSeverity.HIGH,
                source="configuration_system",
                action="update_configuration",
                details={
                    "configuration_type": config.get("type", "unknown"),
                    "changes": config.get("changes", {}),
                    "change_reason": config.get("reason", "unknown")
                },
                after_state=config
            )
            
            logger.info(f"Configuration change logged: {event_id}")
            return event_id
            
        except Exception as e:
            logger.error(f"Failed to log configuration change: {e}")
            return "AUDIT_LOG_FAILED"
    
    async def log_user_action(self, user_id: str, action: str, details: Dict[str, Any]) -> str:
        """Log user action event."""
        try:
            event_id = await self._create_audit_event(
                event_type=AuditEventType.USER_ACTION,
                severity=AuditSeverity.LOW,
                source="user_interface",
                action=action,
                details=details,
                user_id=user_id
            )
            
            logger.info(f"User action logged: {event_id}")
            return event_id
            
        except Exception as e:
            logger.error(f"Failed to log user action: {e}")
            return "AUDIT_LOG_FAILED"
    
    async def log_system_event(self, event: str, details: Dict[str, Any]) -> str:
        """Log system event."""
        try:
            event_id = await self._create_audit_event(
                event_type=AuditEventType.SYSTEM_EVENT,
                severity=AuditSeverity.LOW,
                source="system",
                action=event,
                details=details
            )
            
            logger.info(f"System event logged: {event_id}")
            return event_id
            
        except Exception as e:
            logger.error(f"Failed to log system event: {e}")
            return "AUDIT_LOG_FAILED"
    
    async def log_security_event(self, event: str, details: Dict[str, Any]) -> str:
        """Log security event."""
        try:
            event_id = await self._create_audit_event(
                event_type=AuditEventType.SECURITY_EVENT,
                severity=AuditSeverity.HIGH,
                source="security_system",
                action=event,
                details=details
            )
            
            logger.info(f"Security event logged: {event_id}")
            return event_id
            
        except Exception as e:
            logger.error(f"Failed to log security event: {e}")
            return "AUDIT_LOG_FAILED"
    
    async def log_error_event(self, error: Exception, context: Dict[str, Any]) -> str:
        """Log error event."""
        try:
            event_id = await self._create_audit_event(
                event_type=AuditEventType.ERROR_EVENT,
                severity=AuditSeverity.MEDIUM,
                source="error_handler",
                action="error_occurred",
                details={
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "context": context
                }
            )
            
            logger.info(f"Error event logged: {event_id}")
            return event_id
            
        except Exception as e:
            logger.error(f"Failed to log error event: {e}")
            return "AUDIT_LOG_FAILED"
    
    async def _create_audit_event(
        self,
        event_type: AuditEventType,
        severity: AuditSeverity,
        source: str,
        action: str,
        details: Dict[str, Any],
        before_state: Optional[Dict[str, Any]] = None,
        after_state: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> str:
        """Create an audit event."""
        try:
            # Generate unique event ID
            self.event_counter += 1
            event_id = f"AUDIT_{self.event_counter:06d}_{int(datetime.now().timestamp())}"
            
            # Create audit event
            audit_event = AuditEvent(
                event_id=event_id,
                timestamp=datetime.now(),
                event_type=event_type,
                severity=severity,
                user_id=user_id,
                session_id=session_id,
                source=source,
                action=action,
                details=details,
                before_state=before_state,
                after_state=after_state,
                ip_address=ip_address,
                user_agent=user_agent,
                checksum=""
            )
            
            # Calculate checksum for integrity
            audit_event.checksum = self._calculate_checksum(audit_event)
            
            # Store audit event
            self.audit_events.append(audit_event)
            
            # Trim events if necessary
            if len(self.audit_events) > self.max_events:
                self.audit_events = self.audit_events[-self.max_events:]
            
            return event_id
            
        except Exception as e:
            logger.error(f"Failed to create audit event: {e}")
            return "AUDIT_EVENT_FAILED"
    
    def _calculate_checksum(self, event: AuditEvent) -> str:
        """Calculate checksum for audit event integrity."""
        try:
            # Create a string representation of the event (excluding checksum)
            event_data = {
                "event_id": event.event_id,
                "timestamp": event.timestamp.isoformat(),
                "event_type": event.event_type.value,
                "severity": event.severity.value,
                "user_id": event.user_id,
                "session_id": event.session_id,
                "source": event.source,
                "action": event.action,
                "details": event.details,
                "before_state": event.before_state,
                "after_state": event.after_state,
                "ip_address": event.ip_address,
                "user_agent": event.user_agent
            }
            
            # Convert to JSON string and calculate hash
            event_json = json.dumps(event_data, sort_keys=True, default=str)
            checksum = hashlib.sha256(event_json.encode()).hexdigest()
            
            return checksum
            
        except Exception as e:
            logger.error(f"Failed to calculate checksum: {e}")
            return ""
    
    async def generate_audit_report(self, period: str, mode: str) -> str:
        """Generate audit report for specified period and mode."""
        try:
            # Calculate time range
            now = datetime.now()
            if period == "1h":
                start_time = now - timedelta(hours=1)
            elif period == "24h":
                start_time = now - timedelta(days=1)
            elif period == "7d":
                start_time = now - timedelta(days=7)
            elif period == "30d":
                start_time = now - timedelta(days=30)
            else:
                start_time = now - timedelta(hours=1)
            
            # Filter events by period and mode
            period_events = [
                event for event in self.audit_events
                if event.timestamp >= start_time and
                (mode == "ALL" or event.details.get("mode") == mode)
            ]
            
            # Generate report
            report_id = str(uuid.uuid4())
            report = f"""
Audit Report
============
Report ID: {report_id}
Generated At: {now}
Period: {start_time} to {now}
Mode: {mode}
Total Events: {len(period_events)}

Events by Type:
"""
            
            # Count events by type
            events_by_type = {}
            for event in period_events:
                event_type = event.event_type.value
                events_by_type[event_type] = events_by_type.get(event_type, 0) + 1
            
            for event_type, count in sorted(events_by_type.items()):
                report += f"- {event_type}: {count}\n"
            
            report += "\nEvents by Severity:\n"
            
            # Count events by severity
            events_by_severity = {}
            for event in period_events:
                severity = event.severity.value
                events_by_severity[severity] = events_by_severity.get(severity, 0) + 1
            
            for severity, count in sorted(events_by_severity.items()):
                report += f"- {severity}: {count}\n"
            
            # Critical events
            critical_events = [e for e in period_events if e.severity == AuditSeverity.CRITICAL]
            if critical_events:
                report += f"\nCritical Events ({len(critical_events)}):\n"
                for event in critical_events[-10:]:  # Last 10 critical events
                    report += f"- {event.timestamp}: {event.action} - {event.details}\n"
            
            # Recent events
            recent_events = period_events[-20:]  # Last 20 events
            if recent_events:
                report += f"\nRecent Events ({len(recent_events)}):\n"
                for event in recent_events:
                    report += f"- {event.timestamp}: [{event.severity.value}] {event.action} - {event.source}\n"
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating audit report: {e}")
            return f"Error generating audit report: {e}"
    
    async def validate_audit_completeness(self) -> bool:
        """Validate audit trail completeness."""
        try:
            # Check for missing events
            missing_events = []
            
            # Check for events without checksums
            events_without_checksum = [
                event for event in self.audit_events
                if not event.checksum
            ]
            
            if events_without_checksum:
                missing_events.append(f"Events without checksum: {len(events_without_checksum)}")
            
            # Check for events with invalid checksums
            invalid_checksum_events = []
            for event in self.audit_events:
                if event.checksum:
                    expected_checksum = self._calculate_checksum(event)
                    if event.checksum != expected_checksum:
                        invalid_checksum_events.append(event.event_id)
            
            if invalid_checksum_events:
                missing_events.append(f"Events with invalid checksum: {len(invalid_checksum_events)}")
            
            # Check for events older than retention period
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            old_events = [
                event for event in self.audit_events
                if event.timestamp < cutoff_date
            ]
            
            if old_events:
                missing_events.append(f"Events older than retention period: {len(old_events)}")
            
            # Log validation results
            if missing_events:
                logger.warning(f"Audit completeness issues: {missing_events}")
                return False
            else:
                logger.info("Audit trail completeness validation passed")
                return True
                
        except Exception as e:
            logger.error(f"Error validating audit completeness: {e}")
            return False
    
    async def export_audit_trail(self, format: str) -> str:
        """Export audit trail in specified format."""
        try:
            if format.lower() == "json":
                # Export as JSON
                export_data = {
                    "export_timestamp": datetime.now().isoformat(),
                    "total_events": len(self.audit_events),
                    "events": []
                }
                
                for event in self.audit_events:
                    event_data = {
                        "event_id": event.event_id,
                        "timestamp": event.timestamp.isoformat(),
                        "event_type": event.event_type.value,
                        "severity": event.severity.value,
                        "user_id": event.user_id,
                        "session_id": event.session_id,
                        "source": event.source,
                        "action": event.action,
                        "details": event.details,
                        "before_state": event.before_state,
                        "after_state": event.after_state,
                        "ip_address": event.ip_address,
                        "user_agent": event.user_agent,
                        "checksum": event.checksum
                    }
                    export_data["events"].append(event_data)
                
                return json.dumps(export_data, indent=2, default=str)
                
            elif format.lower() == "csv":
                # Export as CSV
                csv_lines = [
                    "event_id,timestamp,event_type,severity,user_id,source,action,details,checksum"
                ]
                
                for event in self.audit_events:
                    details_str = json.dumps(event.details).replace(",", ";").replace("\n", " ")
                    csv_line = f"{event.event_id},{event.timestamp.isoformat()},{event.event_type.value},{event.severity.value},{event.user_id or ''},{event.source},{event.action},{details_str},{event.checksum}"
                    csv_lines.append(csv_line)
                
                return "\n".join(csv_lines)
                
            else:
                logger.error(f"Unsupported export format: {format}")
                return f"Unsupported export format: {format}"
                
        except Exception as e:
            logger.error(f"Error exporting audit trail: {e}")
            return f"Error exporting audit trail: {e}"


# Global audit trail instance
_audit_trail: Optional[AuditTrail] = None


def get_audit_trail() -> AuditTrail:
    """Get global audit trail instance."""
    global _audit_trail
    if _audit_trail is None:
        _audit_trail = AuditTrail()
    return _audit_trail


async def log_order_creation(order: Dict[str, Any], mode: str) -> str:
    """Log order creation event."""
    trail = get_audit_trail()
    return await trail.log_order_creation(order, mode)


async def log_order_execution(order: Dict[str, Any], mode: str) -> str:
    """Log order execution event."""
    trail = get_audit_trail()
    return await trail.log_order_execution(order, mode)


async def log_order_cancellation(order: Dict[str, Any], mode: str) -> str:
    """Log order cancellation event."""
    trail = get_audit_trail()
    return await trail.log_order_cancellation(order, mode)


async def log_position_change(position: Dict[str, Any], mode: str) -> str:
    """Log position change event."""
    trail = get_audit_trail()
    return await trail.log_position_change(position, mode)


async def log_position_closure(position: Dict[str, Any], mode: str) -> str:
    """Log position closure event."""
    trail = get_audit_trail()
    return await trail.log_position_closure(position, mode)


async def log_parameter_change(param: str, old_value: Any, new_value: Any) -> str:
    """Log parameter change event."""
    trail = get_audit_trail()
    return await trail.log_parameter_change(param, old_value, new_value)


async def log_configuration_change(config: Dict[str, Any]) -> str:
    """Log configuration change event."""
    trail = get_audit_trail()
    return await trail.log_configuration_change(config)


async def generate_audit_report(period: str, mode: str) -> str:
    """Generate audit report for specified period and mode."""
    trail = get_audit_trail()
    return await trail.generate_audit_report(period, mode)


async def validate_audit_completeness() -> bool:
    """Validate audit trail completeness."""
    trail = get_audit_trail()
    return await trail.validate_audit_completeness()


async def export_audit_trail(format: str) -> str:
    """Export audit trail in specified format."""
    trail = get_audit_trail()
    return await trail.export_audit_trail(format)
