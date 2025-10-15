"""
Comprehensive Audit Trail System

This module implements a comprehensive audit trail system for tracking
all system activities, changes, and events with detailed logging,
search capabilities, and compliance reporting.

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
import hashlib
import uuid
import threading
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class EventType(Enum):
    """Types of audit events."""
    TRADE_EXECUTION = "TRADE_EXECUTION"
    ORDER_PLACEMENT = "ORDER_PLACEMENT"
    ORDER_MODIFICATION = "ORDER_MODIFICATION"
    ORDER_CANCELLATION = "ORDER_CANCELLATION"
    POSITION_CHANGE = "POSITION_CHANGE"
    RISK_VIOLATION = "RISK_VIOLATION"
    COMPLIANCE_VIOLATION = "COMPLIANCE_VIOLATION"
    SYSTEM_CONFIGURATION = "SYSTEM_CONFIGURATION"
    USER_ACTION = "USER_ACTION"
    DATA_ACCESS = "DATA_ACCESS"
    MODEL_UPDATE = "MODEL_UPDATE"
    ALERT_GENERATION = "ALERT_GENERATION"
    REPORT_GENERATION = "REPORT_GENERATION"
    SECURITY_EVENT = "SECURITY_EVENT"
    PERFORMANCE_METRIC = "PERFORMANCE_METRIC"

class EventSeverity(Enum):
    """Event severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class EventStatus(Enum):
    """Event status."""
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

@dataclass
class AuditEvent:
    """Comprehensive audit event record."""
    event_id: str
    event_type: EventType
    severity: EventSeverity
    status: EventStatus
    timestamp: datetime
    user_id: Optional[str]
    session_id: Optional[str]
    source_system: str
    target_system: Optional[str]
    action: str
    description: str
    entity_type: Optional[str]
    entity_id: Optional[str]
    old_values: Optional[Dict[str, Any]]
    new_values: Optional[Dict[str, Any]]
    metadata: Dict[str, Any]
    ip_address: Optional[str]
    user_agent: Optional[str]
    checksum: str
    parent_event_id: Optional[str] = None
    child_event_ids: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

class AuditLogger:
    """
    High-performance audit logger with thread safety and batch processing.
    
    Features:
    - Thread-safe event logging
    - Batch processing for performance
    - Event correlation and chaining
    - Data integrity verification
    - Search and filtering capabilities
    - Compliance reporting
    """
    
    def __init__(self, db_path: str = "data/audit_trail.db", batch_size: int = 100):
        """
        Initialize audit logger.
        
        Args:
            db_path: Path to audit trail database
            batch_size: Batch size for bulk operations
        """
        self.db_path = db_path
        self.batch_size = batch_size
        self.event_buffer: List[AuditEvent] = []
        self.lock = threading.Lock()
        
        # Performance metrics
        self.metrics = {
            'events_logged': 0,
            'events_processed': 0,
            'batch_count': 0,
            'last_flush': datetime.now()
        }
        
        # Initialize database
        self._init_database()
        
        # Start background flush thread
        self._start_background_flush()
        
        logger.info("Audit Logger initialized")
    
    def _init_database(self) -> None:
        """Initialize audit trail database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create audit events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                status TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                user_id TEXT,
                session_id TEXT,
                source_system TEXT NOT NULL,
                target_system TEXT,
                action TEXT NOT NULL,
                description TEXT NOT NULL,
                entity_type TEXT,
                entity_id TEXT,
                old_values TEXT,
                new_values TEXT,
                metadata TEXT,
                ip_address TEXT,
                user_agent TEXT,
                checksum TEXT NOT NULL,
                parent_event_id TEXT,
                child_event_ids TEXT,
                tags TEXT,
                created_at TEXT NOT NULL
            )
        """)
        
        # Create indexes for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_events(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_event_type ON audit_events(event_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_user_id ON audit_events(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_entity ON audit_events(entity_type, entity_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_severity ON audit_events(severity)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_status ON audit_events(status)")
        
        # Create audit summary table for performance
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                event_type TEXT,
                severity TEXT,
                count INTEGER,
                created_at TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _start_background_flush(self) -> None:
        """Start background thread for periodic buffer flushing."""
        def flush_worker():
            while True:
                try:
                    time.sleep(30)  # Flush every 30 seconds
                    self.flush_buffer()
                except Exception as e:
                    logger.error(f"Error in audit flush worker: {e}")
        
        flush_thread = threading.Thread(target=flush_worker, daemon=True)
        flush_thread.start()
    
    def log_event(self, event: AuditEvent) -> None:
        """
        Log an audit event.
        
        Args:
            event: Audit event to log
        """
        # Generate checksum for data integrity
        event.checksum = self._generate_checksum(event)
        
        with self.lock:
            self.event_buffer.append(event)
            self.metrics['events_logged'] += 1
            
            # Flush buffer if it reaches batch size
            if len(self.event_buffer) >= self.batch_size:
                self.flush_buffer()
    
    def log_trade_execution(self, trade_data: Dict[str, Any], user_id: str = None, 
                           session_id: str = None, ip_address: str = None) -> str:
        """
        Log trade execution event.
        
        Args:
            trade_data: Trade execution data
            user_id: User ID who executed the trade
            session_id: Session ID
            ip_address: IP address
            
        Returns:
            Event ID
        """
        event_id = str(uuid.uuid4())
        
        event = AuditEvent(
            event_id=event_id,
            event_type=EventType.TRADE_EXECUTION,
            severity=EventSeverity.INFO,
            status=EventStatus.COMPLETED,
            timestamp=datetime.now(),
            user_id=user_id,
            session_id=session_id,
            source_system="TRADING_ENGINE",
            target_system="BROKER_API",
            action="EXECUTE_TRADE",
            description=f"Trade executed: {trade_data.get('symbol')} {trade_data.get('quantity')} shares at ${trade_data.get('price')}",
            entity_type="TRADE",
            entity_id=trade_data.get('trade_id'),
            old_values=None,
            new_values=trade_data,
            metadata={
                'symbol': trade_data.get('symbol'),
                'quantity': trade_data.get('quantity'),
                'price': trade_data.get('price'),
                'side': trade_data.get('side'),
                'order_type': trade_data.get('order_type'),
                'execution_time': trade_data.get('execution_time'),
                'commission': trade_data.get('commission'),
                'slippage': trade_data.get('slippage')
            },
            ip_address=ip_address,
            user_agent=None,
            checksum="",
            tags=['TRADE', 'EXECUTION', trade_data.get('symbol', 'UNKNOWN')]
        )
        
        self.log_event(event)
        return event_id
    
    def log_order_placement(self, order_data: Dict[str, Any], user_id: str = None,
                           session_id: str = None, ip_address: str = None) -> str:
        """
        Log order placement event.
        
        Args:
            order_data: Order placement data
            user_id: User ID who placed the order
            session_id: Session ID
            ip_address: IP address
            
        Returns:
            Event ID
        """
        event_id = str(uuid.uuid4())
        
        event = AuditEvent(
            event_id=event_id,
            event_type=EventType.ORDER_PLACEMENT,
            severity=EventSeverity.INFO,
            status=EventStatus.PENDING,
            timestamp=datetime.now(),
            user_id=user_id,
            session_id=session_id,
            source_system="TRADING_ENGINE",
            target_system="ORDER_MANAGEMENT",
            action="PLACE_ORDER",
            description=f"Order placed: {order_data.get('symbol')} {order_data.get('quantity')} shares {order_data.get('side')}",
            entity_type="ORDER",
            entity_id=order_data.get('order_id'),
            old_values=None,
            new_values=order_data,
            metadata={
                'symbol': order_data.get('symbol'),
                'quantity': order_data.get('quantity'),
                'side': order_data.get('side'),
                'order_type': order_data.get('order_type'),
                'limit_price': order_data.get('limit_price'),
                'stop_price': order_data.get('stop_price'),
                'time_in_force': order_data.get('time_in_force')
            },
            ip_address=ip_address,
            user_agent=None,
            checksum="",
            tags=['ORDER', 'PLACEMENT', order_data.get('symbol', 'UNKNOWN')]
        )
        
        self.log_event(event)
        return event_id
    
    def log_risk_violation(self, violation_data: Dict[str, Any], user_id: str = None) -> str:
        """
        Log risk violation event.
        
        Args:
            violation_data: Risk violation data
            user_id: User ID (if applicable)
            
        Returns:
            Event ID
        """
        event_id = str(uuid.uuid4())
        
        event = AuditEvent(
            event_id=event_id,
            event_type=EventType.RISK_VIOLATION,
            severity=EventSeverity.WARNING,
            status=EventStatus.COMPLETED,
            timestamp=datetime.now(),
            user_id=user_id,
            session_id=None,
            source_system="RISK_MANAGEMENT",
            target_system="TRADING_ENGINE",
            action="DETECT_VIOLATION",
            description=f"Risk violation detected: {violation_data.get('violation_type')}",
            entity_type="RISK_VIOLATION",
            entity_id=violation_data.get('violation_id'),
            old_values=None,
            new_values=violation_data,
            metadata={
                'violation_type': violation_data.get('violation_type'),
                'limit_type': violation_data.get('limit_type'),
                'current_value': violation_data.get('current_value'),
                'limit_value': violation_data.get('limit_value'),
                'excess_amount': violation_data.get('excess_amount'),
                'severity': violation_data.get('severity'),
                'enforcement_action': violation_data.get('enforcement_action')
            },
            ip_address=None,
            user_agent=None,
            checksum="",
            tags=['RISK', 'VIOLATION', violation_data.get('violation_type', 'UNKNOWN')]
        )
        
        self.log_event(event)
        return event_id
    
    def log_compliance_violation(self, violation_data: Dict[str, Any], user_id: str = None) -> str:
        """
        Log compliance violation event.
        
        Args:
            violation_data: Compliance violation data
            user_id: User ID (if applicable)
            
        Returns:
            Event ID
        """
        event_id = str(uuid.uuid4())
        
        event = AuditEvent(
            event_id=event_id,
            event_type=EventType.COMPLIANCE_VIOLATION,
            severity=EventSeverity.ERROR,
            status=EventStatus.COMPLETED,
            timestamp=datetime.now(),
            user_id=user_id,
            session_id=None,
            source_system="COMPLIANCE_ENGINE",
            target_system="TRADING_ENGINE",
            action="DETECT_VIOLATION",
            description=f"Compliance violation detected: {violation_data.get('violation_type')}",
            entity_type="COMPLIANCE_VIOLATION",
            entity_id=violation_data.get('violation_id'),
            old_values=None,
            new_values=violation_data,
            metadata={
                'violation_type': violation_data.get('violation_type'),
                'rule_id': violation_data.get('rule_id'),
                'rule_name': violation_data.get('rule_name'),
                'severity': violation_data.get('severity'),
                'affected_entities': violation_data.get('affected_entities'),
                'enforcement_action': violation_data.get('enforcement_action')
            },
            ip_address=None,
            user_agent=None,
            checksum="",
            tags=['COMPLIANCE', 'VIOLATION', violation_data.get('violation_type', 'UNKNOWN')]
        )
        
        self.log_event(event)
        return event_id
    
    def log_system_configuration(self, config_data: Dict[str, Any], user_id: str = None,
                                ip_address: str = None) -> str:
        """
        Log system configuration change.
        
        Args:
            config_data: Configuration change data
            user_id: User ID who made the change
            ip_address: IP address
            
        Returns:
            Event ID
        """
        event_id = str(uuid.uuid4())
        
        event = AuditEvent(
            event_id=event_id,
            event_type=EventType.SYSTEM_CONFIGURATION,
            severity=EventSeverity.INFO,
            status=EventStatus.COMPLETED,
            timestamp=datetime.now(),
            user_id=user_id,
            session_id=None,
            source_system="CONFIGURATION_MANAGER",
            target_system="SYSTEM",
            action="UPDATE_CONFIGURATION",
            description=f"System configuration updated: {config_data.get('config_type')}",
            entity_type="CONFIGURATION",
            entity_id=config_data.get('config_id'),
            old_values=config_data.get('old_values'),
            new_values=config_data.get('new_values'),
            metadata={
                'config_type': config_data.get('config_type'),
                'config_section': config_data.get('config_section'),
                'change_reason': config_data.get('change_reason'),
                'approval_required': config_data.get('approval_required', False)
            },
            ip_address=ip_address,
            user_agent=None,
            checksum="",
            tags=['CONFIGURATION', 'SYSTEM', config_data.get('config_type', 'UNKNOWN')]
        )
        
        self.log_event(event)
        return event_id
    
    def log_user_action(self, action_data: Dict[str, Any], user_id: str = None,
                       session_id: str = None, ip_address: str = None) -> str:
        """
        Log user action event.
        
        Args:
            action_data: User action data
            user_id: User ID
            session_id: Session ID
            ip_address: IP address
            
        Returns:
            Event ID
        """
        event_id = str(uuid.uuid4())
        
        event = AuditEvent(
            event_id=event_id,
            event_type=EventType.USER_ACTION,
            severity=EventSeverity.INFO,
            status=EventStatus.COMPLETED,
            timestamp=datetime.now(),
            user_id=user_id,
            session_id=session_id,
            source_system="USER_INTERFACE",
            target_system=action_data.get('target_system'),
            action=action_data.get('action'),
            description=f"User action: {action_data.get('action')}",
            entity_type=action_data.get('entity_type'),
            entity_id=action_data.get('entity_id'),
            old_values=action_data.get('old_values'),
            new_values=action_data.get('new_values'),
            metadata={
                'action_type': action_data.get('action_type'),
                'resource': action_data.get('resource'),
                'permissions': action_data.get('permissions'),
                'success': action_data.get('success', True)
            },
            ip_address=ip_address,
            user_agent=action_data.get('user_agent'),
            checksum="",
            tags=['USER', 'ACTION', action_data.get('action', 'UNKNOWN')]
        )
        
        self.log_event(event)
        return event_id
    
    def _generate_checksum(self, event: AuditEvent) -> str:
        """Generate checksum for event data integrity."""
        data_string = f"{event.event_id}{event.timestamp.isoformat()}{event.action}{event.description}"
        return hashlib.md5(data_string.encode()).hexdigest()
    
    def flush_buffer(self) -> None:
        """Flush event buffer to database."""
        if not self.event_buffer:
            return
        
        with self.lock:
            events_to_flush = self.event_buffer.copy()
            self.event_buffer.clear()
        
        if not events_to_flush:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for event in events_to_flush:
                cursor.execute("""
                    INSERT INTO audit_events 
                    (event_id, event_type, severity, status, timestamp, user_id, session_id,
                     source_system, target_system, action, description, entity_type, entity_id,
                     old_values, new_values, metadata, ip_address, user_agent, checksum,
                     parent_event_id, child_event_ids, tags, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.event_id, event.event_type.value, event.severity.value, event.status.value,
                    event.timestamp.isoformat(), event.user_id, event.session_id,
                    event.source_system, event.target_system, event.action, event.description,
                    event.entity_type, event.entity_id,
                    json.dumps(event.old_values) if event.old_values else None,
                    json.dumps(event.new_values) if event.new_values else None,
                    json.dumps(event.metadata), event.ip_address, event.user_agent, event.checksum,
                    event.parent_event_id, json.dumps(event.child_event_ids),
                    json.dumps(event.tags), event.created_at.isoformat()
                ))
            
            conn.commit()
            conn.close()
            
            self.metrics['events_processed'] += len(events_to_flush)
            self.metrics['batch_count'] += 1
            self.metrics['last_flush'] = datetime.now()
            
            logger.debug(f"Flushed {len(events_to_flush)} audit events to database")
            
        except Exception as e:
            logger.error(f"Error flushing audit events: {e}")
            # Re-add events to buffer for retry
            with self.lock:
                self.event_buffer.extend(events_to_flush)
    
    def search_events(self, filters: Dict[str, Any], limit: int = 1000) -> List[AuditEvent]:
        """
        Search audit events with filters.
        
        Args:
            filters: Search filters
            limit: Maximum number of results
            
        Returns:
            List of matching audit events
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Build query
        query = "SELECT * FROM audit_events WHERE 1=1"
        params = []
        
        if 'start_date' in filters:
            query += " AND timestamp >= ?"
            params.append(filters['start_date'].isoformat())
        
        if 'end_date' in filters:
            query += " AND timestamp <= ?"
            params.append(filters['end_date'].isoformat())
        
        if 'event_type' in filters:
            query += " AND event_type = ?"
            params.append(filters['event_type'].value)
        
        if 'severity' in filters:
            query += " AND severity = ?"
            params.append(filters['severity'].value)
        
        if 'user_id' in filters:
            query += " AND user_id = ?"
            params.append(filters['user_id'])
        
        if 'entity_type' in filters:
            query += " AND entity_type = ?"
            params.append(filters['entity_type'])
        
        if 'entity_id' in filters:
            query += " AND entity_id = ?"
            params.append(filters['entity_id'])
        
        if 'action' in filters:
            query += " AND action LIKE ?"
            params.append(f"%{filters['action']}%")
        
        if 'description' in filters:
            query += " AND description LIKE ?"
            params.append(f"%{filters['description']}%")
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        # Convert rows to AuditEvent objects
        events = []
        for row in rows:
            event = self._row_to_event(row)
            events.append(event)
        
        return events
    
    def _row_to_event(self, row: Tuple) -> AuditEvent:
        """Convert database row to AuditEvent object."""
        return AuditEvent(
            event_id=row[0],
            event_type=EventType(row[1]),
            severity=EventSeverity(row[2]),
            status=EventStatus(row[3]),
            timestamp=datetime.fromisoformat(row[4]),
            user_id=row[5],
            session_id=row[6],
            source_system=row[7],
            target_system=row[8],
            action=row[9],
            description=row[10],
            entity_type=row[11],
            entity_id=row[12],
            old_values=json.loads(row[13]) if row[13] else None,
            new_values=json.loads(row[14]) if row[14] else None,
            metadata=json.loads(row[15]) if row[15] else {},
            ip_address=row[16],
            user_agent=row[17],
            checksum=row[18],
            parent_event_id=row[19],
            child_event_ids=json.loads(row[20]) if row[20] else [],
            tags=json.loads(row[21]) if row[21] else [],
            created_at=datetime.fromisoformat(row[22])
        )
    
    def get_event_by_id(self, event_id: str) -> Optional[AuditEvent]:
        """
        Get audit event by ID.
        
        Args:
            event_id: Event ID
            
        Returns:
            Audit event or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM audit_events WHERE event_id = ?", (event_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return self._row_to_event(row)
        return None
    
    def get_events_by_entity(self, entity_type: str, entity_id: str, 
                            limit: int = 100) -> List[AuditEvent]:
        """
        Get all events for a specific entity.
        
        Args:
            entity_type: Entity type
            entity_id: Entity ID
            limit: Maximum number of results
            
        Returns:
            List of audit events
        """
        filters = {
            'entity_type': entity_type,
            'entity_id': entity_id
        }
        return self.search_events(filters, limit)
    
    def get_events_by_user(self, user_id: str, start_date: datetime = None,
                          end_date: datetime = None, limit: int = 100) -> List[AuditEvent]:
        """
        Get all events for a specific user.
        
        Args:
            user_id: User ID
            start_date: Start date filter
            end_date: End date filter
            limit: Maximum number of results
            
        Returns:
            List of audit events
        """
        filters = {'user_id': user_id}
        if start_date:
            filters['start_date'] = start_date
        if end_date:
            filters['end_date'] = end_date
        
        return self.search_events(filters, limit)
    
    def generate_audit_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Generate comprehensive audit report.
        
        Args:
            start_date: Start date for report
            end_date: End date for report
            
        Returns:
            Audit report dictionary
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get event counts by type
        cursor.execute("""
            SELECT event_type, COUNT(*) as count
            FROM audit_events
            WHERE timestamp BETWEEN ? AND ?
            GROUP BY event_type
            ORDER BY count DESC
        """, (start_date.isoformat(), end_date.isoformat()))
        
        event_type_counts = dict(cursor.fetchall())
        
        # Get event counts by severity
        cursor.execute("""
            SELECT severity, COUNT(*) as count
            FROM audit_events
            WHERE timestamp BETWEEN ? AND ?
            GROUP BY severity
            ORDER BY count DESC
        """, (start_date.isoformat(), end_date.isoformat()))
        
        severity_counts = dict(cursor.fetchall())
        
        # Get user activity
        cursor.execute("""
            SELECT user_id, COUNT(*) as count
            FROM audit_events
            WHERE timestamp BETWEEN ? AND ? AND user_id IS NOT NULL
            GROUP BY user_id
            ORDER BY count DESC
            LIMIT 10
        """, (start_date.isoformat(), end_date.isoformat()))
        
        user_activity = dict(cursor.fetchall())
        
        # Get system activity
        cursor.execute("""
            SELECT source_system, COUNT(*) as count
            FROM audit_events
            WHERE timestamp BETWEEN ? AND ?
            GROUP BY source_system
            ORDER BY count DESC
        """, (start_date.isoformat(), end_date.isoformat()))
        
        system_activity = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            'report_period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            },
            'summary': {
                'total_events': sum(event_type_counts.values()),
                'event_types': len(event_type_counts),
                'active_users': len(user_activity),
                'active_systems': len(system_activity)
            },
            'event_type_breakdown': event_type_counts,
            'severity_breakdown': severity_counts,
            'user_activity': user_activity,
            'system_activity': system_activity,
            'generated_at': datetime.now().isoformat()
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get audit logger performance metrics."""
        return {
            'events_logged': self.metrics['events_logged'],
            'events_processed': self.metrics['events_processed'],
            'batch_count': self.metrics['batch_count'],
            'last_flush': self.metrics['last_flush'].isoformat(),
            'buffer_size': len(self.event_buffer),
            'events_per_second': self.metrics['events_processed'] / max(1, (datetime.now() - self.metrics['last_flush']).total_seconds())
        }
    
    def verify_data_integrity(self, event_id: str) -> bool:
        """
        Verify data integrity for an audit event.
        
        Args:
            event_id: Event ID to verify
            
        Returns:
            True if data integrity is valid
        """
        event = self.get_event_by_id(event_id)
        if not event:
            return False
        
        # Recalculate checksum
        expected_checksum = self._generate_checksum(event)
        return event.checksum == expected_checksum
    
    def cleanup_old_events(self, retention_days: int = 365) -> int:
        """
        Clean up old audit events.
        
        Args:
            retention_days: Number of days to retain events
            
        Returns:
            Number of events deleted
        """
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM audit_events WHERE timestamp < ?", (cutoff_date.isoformat(),))
        deleted_count = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        logger.info(f"Cleaned up {deleted_count} old audit events")
        return deleted_count

class AuditTrail:
    """
    High-level audit trail interface for the trading system.
    
    This class provides a simplified interface for logging common
    trading system events and generating compliance reports.
    """
    
    def __init__(self, db_path: str = "data/audit_trail.db"):
        """
        Initialize audit trail.
        
        Args:
            db_path: Path to audit trail database
        """
        self.logger = AuditLogger(db_path)
        logger.info("Audit Trail initialized")
    
    def log_trade(self, trade_data: Dict[str, Any], user_id: str = None) -> str:
        """Log trade execution."""
        return self.logger.log_trade_execution(trade_data, user_id)
    
    def log_order(self, order_data: Dict[str, Any], user_id: str = None) -> str:
        """Log order placement."""
        return self.logger.log_order_placement(order_data, user_id)
    
    def log_risk_violation(self, violation_data: Dict[str, Any]) -> str:
        """Log risk violation."""
        return self.logger.log_risk_violation(violation_data)
    
    def log_compliance_violation(self, violation_data: Dict[str, Any]) -> str:
        """Log compliance violation."""
        return self.logger.log_compliance_violation(violation_data)
    
    def log_config_change(self, config_data: Dict[str, Any], user_id: str = None) -> str:
        """Log configuration change."""
        return self.logger.log_system_configuration(config_data, user_id)
    
    def log_user_action(self, action_data: Dict[str, Any], user_id: str = None) -> str:
        """Log user action."""
        return self.logger.log_user_action(action_data, user_id)
    
    def get_trade_history(self, symbol: str = None, user_id: str = None, 
                         start_date: datetime = None, end_date: datetime = None) -> List[AuditEvent]:
        """Get trade execution history."""
        filters = {'event_type': EventType.TRADE_EXECUTION}
        if symbol:
            filters['entity_id'] = symbol
        if user_id:
            filters['user_id'] = user_id
        if start_date:
            filters['start_date'] = start_date
        if end_date:
            filters['end_date'] = end_date
        
        return self.logger.search_events(filters)
    
    def get_violation_history(self, start_date: datetime = None, end_date: datetime = None) -> List[AuditEvent]:
        """Get violation history."""
        filters = {}
        if start_date:
            filters['start_date'] = start_date
        if end_date:
            filters['end_date'] = end_date
        
        events = []
        events.extend(self.logger.search_events({**filters, 'event_type': EventType.RISK_VIOLATION}))
        events.extend(self.logger.search_events({**filters, 'event_type': EventType.COMPLIANCE_VIOLATION}))
        
        return sorted(events, key=lambda x: x.timestamp, reverse=True)
    
    def generate_compliance_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate compliance audit report."""
        return self.logger.generate_audit_report(start_date, end_date)
    
    def flush(self) -> None:
        """Flush audit buffer."""
        self.logger.flush_buffer()
