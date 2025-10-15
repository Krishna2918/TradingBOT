"""
Error Tracking and Alerting System
==================================

Comprehensive error tracking and categorization system with
alerting capabilities and error trend analysis.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import traceback
import json

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories."""
    SYSTEM = "system"
    TRADING = "trading"
    AI = "ai"
    DATABASE = "database"
    NETWORK = "network"
    API = "api"
    SECURITY = "security"
    PERFORMANCE = "performance"
    UNKNOWN = "unknown"


@dataclass
class ErrorRecord:
    """Error record data structure."""
    error_id: str
    timestamp: datetime
    error_type: str
    error_message: str
    severity: ErrorSeverity
    category: ErrorCategory
    context: Dict[str, Any]
    stack_trace: str
    resolved: bool
    resolution_time: Optional[datetime]
    resolution_notes: Optional[str]


@dataclass
class ErrorStatistics:
    """Error statistics data structure."""
    total_errors: int
    errors_by_severity: Dict[str, int]
    errors_by_category: Dict[str, int]
    errors_by_hour: Dict[int, int]
    average_resolution_time: float
    resolution_rate: float
    trend: str


@dataclass
class ErrorTrend:
    """Error trend data structure."""
    period: str
    error_count: int
    trend_direction: str
    trend_percentage: float
    critical_errors: int
    new_errors: int
    resolved_errors: int


class ErrorTracker:
    """Comprehensive error tracking and categorization."""
    
    def __init__(self):
        self.error_records: List[ErrorRecord] = []
        self.error_counter = 0
        self.alert_thresholds = {
            ErrorSeverity.CRITICAL: 1,
            ErrorSeverity.HIGH: 5,
            ErrorSeverity.MEDIUM: 10,
            ErrorSeverity.LOW: 20
        }
        self.alert_callbacks: List[callable] = []
        
    async def log_error(self, error: Exception, context: Dict[str, Any]) -> str:
        """Log an error with context."""
        try:
            # Generate unique error ID
            self.error_counter += 1
            error_id = f"ERR_{self.error_counter:06d}_{int(datetime.now().timestamp())}"
            
            # Categorize error
            category = self.categorize_error(error)
            severity = self._determine_severity(error, category)
            
            # Create error record
            error_record = ErrorRecord(
                error_id=error_id,
                timestamp=datetime.now(),
                error_type=type(error).__name__,
                error_message=str(error),
                severity=severity,
                category=category,
                context=context,
                stack_trace=traceback.format_exc(),
                resolved=False,
                resolution_time=None,
                resolution_notes=None
            )
            
            # Store error record
            self.error_records.append(error_record)
            
            # Check for alerts
            await self._check_alert_thresholds(severity)
            
            # Log error
            logger.error(f"Error logged: {error_id} - {error_record.error_message}")
            
            return error_id
            
        except Exception as e:
            logger.error(f"Failed to log error: {e}")
            return "ERROR_LOG_FAILED"
    
    def categorize_error(self, error: Exception) -> ErrorCategory:
        """Categorize an error based on its type and message."""
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # System errors
        if error_type in ["OSError", "IOError", "FileNotFoundError", "PermissionError"]:
            return ErrorCategory.SYSTEM
        
        # Trading errors
        if any(keyword in error_message for keyword in ["trade", "order", "position", "portfolio"]):
            return ErrorCategory.TRADING
        
        # AI errors
        if any(keyword in error_message for keyword in ["ai", "model", "prediction", "analysis"]):
            return ErrorCategory.AI
        
        # Database errors
        if error_type in ["sqlite3.Error", "psycopg2.Error", "pymongo.errors"] or "database" in error_message:
            return ErrorCategory.DATABASE
        
        # Network errors
        if error_type in ["ConnectionError", "TimeoutError", "requests.exceptions"] or "network" in error_message:
            return ErrorCategory.NETWORK
        
        # API errors
        if any(keyword in error_message for keyword in ["api", "http", "request", "response"]):
            return ErrorCategory.API
        
        # Security errors
        if any(keyword in error_message for keyword in ["security", "auth", "permission", "access"]):
            return ErrorCategory.SECURITY
        
        # Performance errors
        if any(keyword in error_message for keyword in ["timeout", "slow", "performance", "memory"]):
            return ErrorCategory.PERFORMANCE
        
        return ErrorCategory.UNKNOWN
    
    def _determine_severity(self, error: Exception, category: ErrorCategory) -> ErrorSeverity:
        """Determine error severity based on error type and category."""
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # Critical errors
        if error_type in ["SystemExit", "KeyboardInterrupt"] or "critical" in error_message:
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        if (error_type in ["MemoryError", "SystemError"] or 
            category in [ErrorCategory.SECURITY, ErrorCategory.TRADING] or
            "fatal" in error_message or "crash" in error_message):
            return ErrorSeverity.HIGH
        
        # Medium severity errors
        if (error_type in ["ValueError", "TypeError", "AttributeError"] or
            category in [ErrorCategory.AI, ErrorCategory.DATABASE] or
            "error" in error_message):
            return ErrorSeverity.MEDIUM
        
        # Low severity errors
        return ErrorSeverity.LOW
    
    async def _check_alert_thresholds(self, severity: ErrorSeverity) -> None:
        """Check if error count exceeds alert thresholds."""
        try:
            # Count errors of this severity in the last hour
            one_hour_ago = datetime.now() - timedelta(hours=1)
            recent_errors = [
                error for error in self.error_records
                if error.severity == severity and error.timestamp >= one_hour_ago
            ]
            
            threshold = self.alert_thresholds.get(severity, 0)
            
            if len(recent_errors) >= threshold:
                await self.trigger_alert(severity, f"Error threshold exceeded: {len(recent_errors)} {severity.value} errors in the last hour")
                
        except Exception as e:
            logger.error(f"Error checking alert thresholds: {e}")
    
    async def trigger_alert(self, severity: str, message: str) -> None:
        """Trigger an alert."""
        try:
            alert = {
                "timestamp": datetime.now(),
                "severity": severity,
                "message": message,
                "error_count": len(self.error_records)
            }
            
            # Call alert callbacks
            for callback in self.alert_callbacks:
                try:
                    await callback(alert)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")
            
            logger.warning(f"Alert triggered: {severity} - {message}")
            
        except Exception as e:
            logger.error(f"Failed to trigger alert: {e}")
    
    async def get_error_statistics(self) -> ErrorStatistics:
        """Get error statistics."""
        try:
            total_errors = len(self.error_records)
            
            # Count errors by severity
            errors_by_severity = {}
            for severity in ErrorSeverity:
                count = len([e for e in self.error_records if e.severity == severity])
                errors_by_severity[severity.value] = count
            
            # Count errors by category
            errors_by_category = {}
            for category in ErrorCategory:
                count = len([e for e in self.error_records if e.category == category])
                errors_by_category[category.value] = count
            
            # Count errors by hour
            errors_by_hour = {}
            for hour in range(24):
                count = len([
                    e for e in self.error_records
                    if e.timestamp.hour == hour
                ])
                errors_by_hour[hour] = count
            
            # Calculate average resolution time
            resolved_errors = [e for e in self.error_records if e.resolved and e.resolution_time]
            if resolved_errors:
                total_resolution_time = sum(
                    (e.resolution_time - e.timestamp).total_seconds()
                    for e in resolved_errors
                )
                average_resolution_time = total_resolution_time / len(resolved_errors)
            else:
                average_resolution_time = 0.0
            
            # Calculate resolution rate
            resolution_rate = len(resolved_errors) / total_errors if total_errors > 0 else 0.0
            
            # Determine trend
            trend = self._calculate_error_trend()
            
            return ErrorStatistics(
                total_errors=total_errors,
                errors_by_severity=errors_by_severity,
                errors_by_category=errors_by_category,
                errors_by_hour=errors_by_hour,
                average_resolution_time=average_resolution_time,
                resolution_rate=resolution_rate,
                trend=trend
            )
            
        except Exception as e:
            logger.error(f"Error getting error statistics: {e}")
            return ErrorStatistics(
                total_errors=0,
                errors_by_severity={},
                errors_by_category={},
                errors_by_hour={},
                average_resolution_time=0.0,
                resolution_rate=0.0,
                trend="unknown"
            )
    
    def _calculate_error_trend(self) -> str:
        """Calculate error trend over time."""
        try:
            if len(self.error_records) < 2:
                return "insufficient_data"
            
            # Compare last 24 hours with previous 24 hours
            now = datetime.now()
            last_24h = now - timedelta(hours=24)
            last_48h = now - timedelta(hours=48)
            
            recent_errors = [e for e in self.error_records if e.timestamp >= last_24h]
            previous_errors = [e for e in self.error_records if last_48h <= e.timestamp < last_24h]
            
            if len(previous_errors) == 0:
                return "increasing" if len(recent_errors) > 0 else "stable"
            
            trend_ratio = len(recent_errors) / len(previous_errors)
            
            if trend_ratio > 1.2:
                return "increasing"
            elif trend_ratio < 0.8:
                return "decreasing"
            else:
                return "stable"
                
        except Exception as e:
            logger.error(f"Error calculating trend: {e}")
            return "unknown"
    
    async def generate_error_report(self, period: str) -> str:
        """Generate error report for specified period."""
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
            
            # Filter errors by period
            period_errors = [e for e in self.error_records if e.timestamp >= start_time]
            
            # Generate report
            report = f"""
Error Report - {period.upper()}
============================
Period: {start_time} to {now}
Total Errors: {len(period_errors)}

Errors by Severity:
"""
            
            for severity in ErrorSeverity:
                count = len([e for e in period_errors if e.severity == severity])
                report += f"- {severity.value.upper()}: {count}\n"
            
            report += "\nErrors by Category:\n"
            for category in ErrorCategory:
                count = len([e for e in period_errors if e.category == category])
                report += f"- {category.value.upper()}: {count}\n"
            
            # Top errors
            error_types = {}
            for error in period_errors:
                error_type = error.error_type
                error_types[error_type] = error_types.get(error_type, 0) + 1
            
            top_errors = sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:5]
            
            report += "\nTop Error Types:\n"
            for error_type, count in top_errors:
                report += f"- {error_type}: {count}\n"
            
            # Recent critical errors
            critical_errors = [e for e in period_errors if e.severity == ErrorSeverity.CRITICAL]
            if critical_errors:
                report += "\nRecent Critical Errors:\n"
                for error in critical_errors[-5:]:  # Last 5 critical errors
                    report += f"- {error.timestamp}: {error.error_message}\n"
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating error report: {e}")
            return f"Error generating report: {e}"
    
    async def get_error_trends(self) -> List[ErrorTrend]:
        """Get error trends over different periods."""
        try:
            trends = []
            periods = ["1h", "24h", "7d", "30d"]
            
            for period in periods:
                # Calculate time range
                now = datetime.now()
                if period == "1h":
                    start_time = now - timedelta(hours=1)
                    prev_start_time = now - timedelta(hours=2)
                elif period == "24h":
                    start_time = now - timedelta(days=1)
                    prev_start_time = now - timedelta(days=2)
                elif period == "7d":
                    start_time = now - timedelta(days=7)
                    prev_start_time = now - timedelta(days=14)
                elif period == "30d":
                    start_time = now - timedelta(days=30)
                    prev_start_time = now - timedelta(days=60)
                
                # Count errors in current and previous periods
                current_errors = [e for e in self.error_records if e.timestamp >= start_time]
                previous_errors = [e for e in self.error_records if prev_start_time <= e.timestamp < start_time]
                
                # Calculate trend
                if len(previous_errors) == 0:
                    trend_direction = "increasing" if len(current_errors) > 0 else "stable"
                    trend_percentage = 100.0 if len(current_errors) > 0 else 0.0
                else:
                    trend_ratio = len(current_errors) / len(previous_errors)
                    if trend_ratio > 1.1:
                        trend_direction = "increasing"
                        trend_percentage = (trend_ratio - 1) * 100
                    elif trend_ratio < 0.9:
                        trend_direction = "decreasing"
                        trend_percentage = (1 - trend_ratio) * 100
                    else:
                        trend_direction = "stable"
                        trend_percentage = 0.0
                
                # Count critical errors
                critical_errors = len([e for e in current_errors if e.severity == ErrorSeverity.CRITICAL])
                
                # Count new and resolved errors
                new_errors = len([e for e in current_errors if not e.resolved])
                resolved_errors = len([e for e in current_errors if e.resolved])
                
                trend = ErrorTrend(
                    period=period,
                    error_count=len(current_errors),
                    trend_direction=trend_direction,
                    trend_percentage=trend_percentage,
                    critical_errors=critical_errors,
                    new_errors=new_errors,
                    resolved_errors=resolved_errors
                )
                
                trends.append(trend)
            
            return trends
            
        except Exception as e:
            logger.error(f"Error getting error trends: {e}")
            return []
    
    async def resolve_error(self, error_id: str, resolution_notes: str) -> bool:
        """Resolve an error."""
        try:
            # Find error record
            error_record = None
            for error in self.error_records:
                if error.error_id == error_id:
                    error_record = error
                    break
            
            if not error_record:
                logger.warning(f"Error record not found: {error_id}")
                return False
            
            # Update error record
            error_record.resolved = True
            error_record.resolution_time = datetime.now()
            error_record.resolution_notes = resolution_notes
            
            logger.info(f"Error resolved: {error_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to resolve error: {e}")
            return False
    
    def add_alert_callback(self, callback: callable) -> None:
        """Add an alert callback function."""
        self.alert_callbacks.append(callback)
    
    def remove_alert_callback(self, callback: callable) -> None:
        """Remove an alert callback function."""
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)


# Global error tracker instance
_error_tracker: Optional[ErrorTracker] = None


def get_error_tracker() -> ErrorTracker:
    """Get global error tracker instance."""
    global _error_tracker
    if _error_tracker is None:
        _error_tracker = ErrorTracker()
    return _error_tracker


async def log_error(error: Exception, context: Dict[str, Any]) -> str:
    """Log an error with context."""
    tracker = get_error_tracker()
    return await tracker.log_error(error, context)


async def get_error_statistics() -> ErrorStatistics:
    """Get error statistics."""
    tracker = get_error_tracker()
    return await tracker.get_error_statistics()


async def generate_error_report(period: str) -> str:
    """Generate error report for specified period."""
    tracker = get_error_tracker()
    return await tracker.generate_error_report(period)


async def get_error_trends() -> List[ErrorTrend]:
    """Get error trends over different periods."""
    tracker = get_error_tracker()
    return await tracker.get_error_trends()


async def resolve_error(error_id: str, resolution_notes: str) -> bool:
    """Resolve an error."""
    tracker = get_error_tracker()
    return await tracker.resolve_error(error_id, resolution_notes)
