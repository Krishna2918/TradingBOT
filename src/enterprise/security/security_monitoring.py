"""
Security Monitoring System

This module implements a comprehensive security monitoring system with
real-time threat detection, incident response, and threat intelligence
integration for enterprise security operations.

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
import re
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class SecurityEventType(Enum):
    """Types of security events."""
    AUTHENTICATION_FAILURE = "AUTHENTICATION_FAILURE"
    SUSPICIOUS_LOGIN = "SUSPICIOUS_LOGIN"
    PRIVILEGE_ESCALATION = "PRIVILEGE_ESCALATION"
    DATA_ACCESS_ANOMALY = "DATA_ACCESS_ANOMALY"
    NETWORK_ANOMALY = "NETWORK_ANOMALY"
    MALWARE_DETECTED = "MALWARE_DETECTED"
    INTRUSION_ATTEMPT = "INTRUSION_ATTEMPT"
    DATA_EXFILTRATION = "DATA_EXFILTRATION"
    SYSTEM_COMPROMISE = "SYSTEM_COMPROMISE"
    POLICY_VIOLATION = "POLICY_VIOLATION"

class ThreatLevel(Enum):
    """Threat levels."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFORMATIONAL = "INFORMATIONAL"

class IncidentStatus(Enum):
    """Incident status."""
    NEW = "NEW"
    ASSIGNED = "ASSIGNED"
    IN_PROGRESS = "IN_PROGRESS"
    RESOLVED = "RESOLVED"
    CLOSED = "CLOSED"
    FALSE_POSITIVE = "FALSE_POSITIVE"

class ThreatIntelligenceType(Enum):
    """Threat intelligence types."""
    IP_ADDRESS = "IP_ADDRESS"
    DOMAIN = "DOMAIN"
    URL = "URL"
    HASH = "HASH"
    EMAIL = "EMAIL"
    FILE = "FILE"
    CVE = "CVE"
    MALWARE = "MALWARE"

@dataclass
class SecurityEvent:
    """Security event definition."""
    event_id: str
    event_type: SecurityEventType
    threat_level: ThreatLevel
    timestamp: datetime
    source_ip: Optional[str]
    source_user: Optional[str]
    target_system: str
    description: str
    raw_data: Dict[str, Any]
    indicators: List[str]
    context: Dict[str, Any]
    correlation_id: Optional[str] = None
    processed: bool = False
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class ThreatIntelligence:
    """Threat intelligence definition."""
    ti_id: str
    indicator_type: ThreatIntelligenceType
    indicator_value: str
    threat_level: ThreatLevel
    confidence: float  # 0.0 to 1.0
    source: str
    description: str
    first_seen: datetime
    last_seen: datetime
    tags: List[str]
    metadata: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class IncidentResponse:
    """Incident response definition."""
    incident_id: str
    title: str
    description: str
    threat_level: ThreatLevel
    status: IncidentStatus
    assigned_to: Optional[str]
    created_at: datetime
    updated_at: datetime
    resolved_at: Optional[datetime]
    affected_systems: List[str]
    indicators: List[str]
    response_actions: List[str]
    lessons_learned: List[str]
    created_at: datetime = field(default_factory=datetime.now)

class SecurityMonitor:
    """
    Comprehensive security monitoring system.
    
    Features:
    - Real-time security event monitoring
    - Threat intelligence integration
    - Incident response automation
    - Security analytics and reporting
    - Correlation and detection rules
    - Alert management
    """
    
    def __init__(self, db_path: str = "data/security_monitoring.db"):
        """
        Initialize security monitor.
        
        Args:
            db_path: Path to security monitoring database
        """
        self.db_path = db_path
        self.security_events: List[SecurityEvent] = []
        self.threat_intelligence: Dict[str, ThreatIntelligence] = {}
        self.incidents: List[IncidentResponse] = []
        self.detection_rules: Dict[str, Dict[str, Any]] = {}
        
        # Monitoring parameters
        self.monitoring_params = {
            'correlation_window': 300,  # 5 minutes
            'alert_threshold': 5,  # events
            'threat_intelligence_update_interval': 3600,  # 1 hour
            'incident_auto_assignment': True
        }
        
        # Initialize database
        self._init_database()
        
        # Load default detection rules
        self._load_default_detection_rules()
        
        # Load default threat intelligence
        self._load_default_threat_intelligence()
        
        logger.info("Security Monitoring system initialized")
    
    def _init_database(self) -> None:
        """Initialize security monitoring database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create security events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS security_events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                threat_level TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                source_ip TEXT,
                source_user TEXT,
                target_system TEXT NOT NULL,
                description TEXT NOT NULL,
                raw_data TEXT,
                indicators TEXT,
                context TEXT,
                correlation_id TEXT,
                processed INTEGER,
                created_at TEXT NOT NULL
            )
        """)
        
        # Create threat intelligence table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS threat_intelligence (
                ti_id TEXT PRIMARY KEY,
                indicator_type TEXT NOT NULL,
                indicator_value TEXT NOT NULL,
                threat_level TEXT NOT NULL,
                confidence REAL NOT NULL,
                source TEXT NOT NULL,
                description TEXT,
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                tags TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL
            )
        """)
        
        # Create incidents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS incidents (
                incident_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                threat_level TEXT NOT NULL,
                status TEXT NOT NULL,
                assigned_to TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                resolved_at TEXT,
                affected_systems TEXT,
                indicators TEXT,
                response_actions TEXT,
                lessons_learned TEXT
            )
        """)
        
        # Create detection rules table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS detection_rules (
                rule_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                rule_type TEXT NOT NULL,
                conditions TEXT NOT NULL,
                actions TEXT NOT NULL,
                enabled INTEGER,
                created_at TEXT NOT NULL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _load_default_detection_rules(self) -> None:
        """Load default detection rules."""
        default_rules = [
            {
                'rule_id': 'RULE_001',
                'name': 'Multiple Failed Logins',
                'description': 'Detect multiple failed login attempts from same source',
                'rule_type': 'CORRELATION',
                'conditions': {
                    'event_type': 'AUTHENTICATION_FAILURE',
                    'time_window': 300,  # 5 minutes
                    'threshold': 5,
                    'group_by': 'source_ip'
                },
                'actions': ['CREATE_ALERT', 'BLOCK_IP'],
                'enabled': True
            },
            {
                'rule_id': 'RULE_002',
                'name': 'Suspicious Data Access',
                'description': 'Detect unusual data access patterns',
                'rule_type': 'ANOMALY',
                'conditions': {
                    'event_type': 'DATA_ACCESS_ANOMALY',
                    'time_window': 3600,  # 1 hour
                    'threshold': 3,
                    'group_by': 'source_user'
                },
                'actions': ['CREATE_ALERT', 'NOTIFY_ADMIN'],
                'enabled': True
            },
            {
                'rule_id': 'RULE_003',
                'name': 'Privilege Escalation',
                'description': 'Detect privilege escalation attempts',
                'rule_type': 'SINGLE_EVENT',
                'conditions': {
                    'event_type': 'PRIVILEGE_ESCALATION',
                    'threat_level': 'HIGH'
                },
                'actions': ['CREATE_INCIDENT', 'BLOCK_USER'],
                'enabled': True
            },
            {
                'rule_id': 'RULE_004',
                'name': 'Network Anomaly',
                'description': 'Detect network anomalies and intrusions',
                'rule_type': 'CORRELATION',
                'conditions': {
                    'event_type': 'NETWORK_ANOMALY',
                    'time_window': 600,  # 10 minutes
                    'threshold': 10,
                    'group_by': 'source_ip'
                },
                'actions': ['CREATE_ALERT', 'BLOCK_IP'],
                'enabled': True
            },
            {
                'rule_id': 'RULE_005',
                'name': 'Malware Detection',
                'description': 'Detect malware and malicious files',
                'rule_type': 'SINGLE_EVENT',
                'conditions': {
                    'event_type': 'MALWARE_DETECTED',
                    'threat_level': 'CRITICAL'
                },
                'actions': ['CREATE_INCIDENT', 'QUARANTINE_FILE'],
                'enabled': True
            }
        ]
        
        for rule in default_rules:
            self.detection_rules[rule['rule_id']] = rule
            self._store_detection_rule(rule)
    
    def _load_default_threat_intelligence(self) -> None:
        """Load default threat intelligence."""
        default_ti = [
            ThreatIntelligence(
                ti_id="TI_001",
                indicator_type=ThreatIntelligenceType.IP_ADDRESS,
                indicator_value="192.168.1.100",
                threat_level=ThreatLevel.HIGH,
                confidence=0.8,
                source="Internal Honeypot",
                description="Known malicious IP address",
                first_seen=datetime.now() - timedelta(days=30),
                last_seen=datetime.now(),
                tags=["malware", "botnet", "c2"],
                metadata={"country": "Unknown", "asn": "Unknown"}
            ),
            ThreatIntelligence(
                ti_id="TI_002",
                indicator_type=ThreatIntelligenceType.DOMAIN,
                indicator_value="malicious-domain.com",
                threat_level=ThreatLevel.CRITICAL,
                confidence=0.9,
                source="External Feed",
                description="Malicious domain used for C2",
                first_seen=datetime.now() - timedelta(days=7),
                last_seen=datetime.now(),
                tags=["c2", "malware", "phishing"],
                metadata={"registrar": "Unknown", "creation_date": "2023-01-01"}
            ),
            ThreatIntelligence(
                ti_id="TI_003",
                indicator_type=ThreatIntelligenceType.HASH,
                indicator_value="a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0",
                threat_level=ThreatLevel.HIGH,
                confidence=0.85,
                source="VirusTotal",
                description="Known malware hash",
                first_seen=datetime.now() - timedelta(days=14),
                last_seen=datetime.now(),
                tags=["malware", "trojan", "backdoor"],
                metadata={"file_type": "exe", "size": "1024000"}
            )
        ]
        
        for ti in default_ti:
            self.threat_intelligence[ti.ti_id] = ti
            self._store_threat_intelligence(ti)
    
    def process_security_event(self, event_data: Dict[str, Any]) -> Optional[SecurityEvent]:
        """
        Process a security event.
        
        Args:
            event_data: Security event data
            
        Returns:
            Processed security event or None if filtered
        """
        try:
            # Create security event
            event = SecurityEvent(
                event_id=f"EVENT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(str(event_data).encode()).hexdigest()[:8]}",
                event_type=SecurityEventType(event_data.get('event_type', 'POLICY_VIOLATION')),
                threat_level=ThreatLevel(event_data.get('threat_level', 'LOW')),
                timestamp=datetime.fromisoformat(event_data.get('timestamp', datetime.now().isoformat())),
                source_ip=event_data.get('source_ip'),
                source_user=event_data.get('source_user'),
                target_system=event_data.get('target_system', 'unknown'),
                description=event_data.get('description', ''),
                raw_data=event_data,
                indicators=event_data.get('indicators', []),
                context=event_data.get('context', {})
            )
            
            # Check against threat intelligence
            self._check_threat_intelligence(event)
            
            # Run detection rules
            self._run_detection_rules(event)
            
            # Store event
            self.security_events.append(event)
            self._store_security_event(event)
            
            logger.info(f"Processed security event: {event.event_id} - {event.event_type.value}")
            return event
            
        except Exception as e:
            logger.error(f"Error processing security event: {e}")
            return None
    
    def _check_threat_intelligence(self, event: SecurityEvent) -> None:
        """Check event against threat intelligence."""
        # Check source IP against threat intelligence
        if event.source_ip:
            for ti in self.threat_intelligence.values():
                if (ti.indicator_type == ThreatIntelligenceType.IP_ADDRESS and 
                    ti.indicator_value == event.source_ip):
                    # Update event threat level if TI indicates higher threat
                    if ti.threat_level.value in ['CRITICAL', 'HIGH']:
                        event.threat_level = ti.threat_level
                        event.context['threat_intelligence'] = {
                            'ti_id': ti.ti_id,
                            'confidence': ti.confidence,
                            'description': ti.description
                        }
        
        # Check indicators against threat intelligence
        for indicator in event.indicators:
            for ti in self.threat_intelligence.values():
                if ti.indicator_value == indicator:
                    if ti.threat_level.value in ['CRITICAL', 'HIGH']:
                        event.threat_level = ti.threat_level
                        event.context['threat_intelligence'] = {
                            'ti_id': ti.ti_id,
                            'confidence': ti.confidence,
                            'description': ti.description
                        }
    
    def _run_detection_rules(self, event: SecurityEvent) -> None:
        """Run detection rules against the event."""
        for rule_id, rule in self.detection_rules.items():
            if not rule.get('enabled', True):
                continue
            
            try:
                if rule['rule_type'] == 'SINGLE_EVENT':
                    if self._evaluate_single_event_rule(rule, event):
                        self._execute_rule_actions(rule, event)
                
                elif rule['rule_type'] == 'CORRELATION':
                    if self._evaluate_correlation_rule(rule, event):
                        self._execute_rule_actions(rule, event)
                
                elif rule['rule_type'] == 'ANOMALY':
                    if self._evaluate_anomaly_rule(rule, event):
                        self._execute_rule_actions(rule, event)
            
            except Exception as e:
                logger.error(f"Error running detection rule {rule_id}: {e}")
    
    def _evaluate_single_event_rule(self, rule: Dict[str, Any], event: SecurityEvent) -> bool:
        """Evaluate single event rule."""
        conditions = rule['conditions']
        
        # Check event type
        if conditions.get('event_type') and event.event_type.value != conditions['event_type']:
            return False
        
        # Check threat level
        if conditions.get('threat_level') and event.threat_level.value != conditions['threat_level']:
            return False
        
        return True
    
    def _evaluate_correlation_rule(self, rule: Dict[str, Any], event: SecurityEvent) -> bool:
        """Evaluate correlation rule."""
        conditions = rule['conditions']
        
        # Check event type
        if conditions.get('event_type') and event.event_type.value != conditions['event_type']:
            return False
        
        # Get time window
        time_window = conditions.get('time_window', 300)
        start_time = event.timestamp - timedelta(seconds=time_window)
        
        # Get group by field
        group_by = conditions.get('group_by', 'source_ip')
        group_value = getattr(event, group_by, None)
        
        if not group_value:
            return False
        
        # Count similar events in time window
        similar_events = [
            e for e in self.security_events
            if (e.timestamp >= start_time and 
                e.timestamp <= event.timestamp and
                getattr(e, group_by, None) == group_value and
                e.event_type == event.event_type)
        ]
        
        threshold = conditions.get('threshold', 5)
        return len(similar_events) >= threshold
    
    def _evaluate_anomaly_rule(self, rule: Dict[str, Any], event: SecurityEvent) -> bool:
        """Evaluate anomaly rule."""
        # This is a simplified anomaly detection
        # In a real implementation, this would use machine learning or statistical methods
        
        conditions = rule['conditions']
        
        # Check event type
        if conditions.get('event_type') and event.event_type.value != conditions['event_type']:
            return False
        
        # Simple anomaly detection based on frequency
        time_window = conditions.get('time_window', 3600)
        start_time = event.timestamp - timedelta(seconds=time_window)
        
        group_by = conditions.get('group_by', 'source_user')
        group_value = getattr(event, group_by, None)
        
        if not group_value:
            return False
        
        # Count events in time window
        recent_events = [
            e for e in self.security_events
            if (e.timestamp >= start_time and 
                e.timestamp <= event.timestamp and
                getattr(e, group_by, None) == group_value)
        ]
        
        # Calculate baseline (average events per hour for this user)
        baseline_events = [
            e for e in self.security_events
            if (e.timestamp >= event.timestamp - timedelta(hours=24) and
                getattr(e, group_by, None) == group_value)
        ]
        
        baseline_rate = len(baseline_events) / 24  # events per hour
        current_rate = len(recent_events) / (time_window / 3600)  # events per hour
        
        # Anomaly if current rate is significantly higher than baseline
        threshold = conditions.get('threshold', 3)
        return current_rate > baseline_rate * threshold
    
    def _execute_rule_actions(self, rule: Dict[str, Any], event: SecurityEvent) -> None:
        """Execute rule actions."""
        actions = rule.get('actions', [])
        
        for action in actions:
            try:
                if action == 'CREATE_ALERT':
                    self._create_alert(rule, event)
                elif action == 'CREATE_INCIDENT':
                    self._create_incident(rule, event)
                elif action == 'BLOCK_IP':
                    self._block_ip(event.source_ip)
                elif action == 'BLOCK_USER':
                    self._block_user(event.source_user)
                elif action == 'NOTIFY_ADMIN':
                    self._notify_admin(rule, event)
                elif action == 'QUARANTINE_FILE':
                    self._quarantine_file(event)
            
            except Exception as e:
                logger.error(f"Error executing action {action}: {e}")
    
    def _create_alert(self, rule: Dict[str, Any], event: SecurityEvent) -> None:
        """Create security alert."""
        alert = {
            'alert_id': f"ALERT_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'rule_id': rule['rule_id'],
            'rule_name': rule['name'],
            'event_id': event.event_id,
            'threat_level': event.threat_level.value,
            'description': f"Alert triggered by rule: {rule['name']}",
            'timestamp': datetime.now().isoformat(),
            'status': 'NEW'
        }
        
        logger.warning(f"Security alert created: {alert['alert_id']} - {rule['name']}")
    
    def _create_incident(self, rule: Dict[str, Any], event: SecurityEvent) -> None:
        """Create security incident."""
        incident_id = f"INCIDENT_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        incident = IncidentResponse(
            incident_id=incident_id,
            title=f"Security Incident: {rule['name']}",
            description=f"Incident triggered by rule: {rule['name']} - {event.description}",
            threat_level=event.threat_level,
            status=IncidentStatus.NEW,
            assigned_to=None,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            resolved_at=None,
            affected_systems=[event.target_system],
            indicators=event.indicators,
            response_actions=[],
            lessons_learned=[]
        )
        
        self.incidents.append(incident)
        self._store_incident(incident)
        
        logger.warning(f"Security incident created: {incident_id} - {rule['name']}")
    
    def _block_ip(self, ip_address: str) -> None:
        """Block IP address."""
        if ip_address:
            logger.warning(f"Blocking IP address: {ip_address}")
            # In a real implementation, this would update firewall rules
    
    def _block_user(self, username: str) -> None:
        """Block user account."""
        if username:
            logger.warning(f"Blocking user account: {username}")
            # In a real implementation, this would disable the user account
    
    def _notify_admin(self, rule: Dict[str, Any], event: SecurityEvent) -> None:
        """Notify administrators."""
        logger.warning(f"Admin notification: {rule['name']} - {event.description}")
        # In a real implementation, this would send email/SMS notifications
    
    def _quarantine_file(self, event: SecurityEvent) -> None:
        """Quarantine malicious file."""
        logger.warning(f"Quarantining file from event: {event.event_id}")
        # In a real implementation, this would move the file to quarantine
    
    def add_threat_intelligence(self, ti_data: Dict[str, Any]) -> str:
        """
        Add threat intelligence indicator.
        
        Args:
            ti_data: Threat intelligence data
            
        Returns:
            Threat intelligence ID
        """
        ti_id = f"TI_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        ti = ThreatIntelligence(
            ti_id=ti_id,
            indicator_type=ThreatIntelligenceType(ti_data.get('indicator_type', 'IP_ADDRESS')),
            indicator_value=ti_data.get('indicator_value', ''),
            threat_level=ThreatLevel(ti_data.get('threat_level', 'LOW')),
            confidence=ti_data.get('confidence', 0.5),
            source=ti_data.get('source', 'Unknown'),
            description=ti_data.get('description', ''),
            first_seen=datetime.fromisoformat(ti_data.get('first_seen', datetime.now().isoformat())),
            last_seen=datetime.fromisoformat(ti_data.get('last_seen', datetime.now().isoformat())),
            tags=ti_data.get('tags', []),
            metadata=ti_data.get('metadata', {})
        )
        
        self.threat_intelligence[ti_id] = ti
        self._store_threat_intelligence(ti)
        
        logger.info(f"Added threat intelligence: {ti_id} - {ti.indicator_value}")
        return ti_id
    
    def update_incident_status(self, incident_id: str, status: IncidentStatus, 
                              assigned_to: str = None, response_actions: List[str] = None) -> bool:
        """
        Update incident status.
        
        Args:
            incident_id: Incident ID
            status: New status
            assigned_to: Assigned to user
            response_actions: Response actions taken
            
        Returns:
            True if updated successfully
        """
        for incident in self.incidents:
            if incident.incident_id == incident_id:
                incident.status = status
                incident.updated_at = datetime.now()
                
                if assigned_to:
                    incident.assigned_to = assigned_to
                
                if response_actions:
                    incident.response_actions.extend(response_actions)
                
                if status == IncidentStatus.RESOLVED:
                    incident.resolved_at = datetime.now()
                
                self._update_incident(incident)
                
                logger.info(f"Updated incident status: {incident_id} -> {status.value}")
                return True
        
        return False
    
    def get_security_events(self, start_date: datetime = None, end_date: datetime = None,
                           event_type: SecurityEventType = None, threat_level: ThreatLevel = None) -> List[SecurityEvent]:
        """
        Get security events with filters.
        
        Args:
            start_date: Filter by start date
            end_date: Filter by end date
            event_type: Filter by event type
            threat_level: Filter by threat level
            
        Returns:
            List of matching security events
        """
        events = self.security_events.copy()
        
        if start_date:
            events = [e for e in events if e.timestamp >= start_date]
        
        if end_date:
            events = [e for e in events if e.timestamp <= end_date]
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if threat_level:
            events = [e for e in events if e.threat_level == threat_level]
        
        return sorted(events, key=lambda x: x.timestamp, reverse=True)
    
    def get_incidents(self, status: IncidentStatus = None, threat_level: ThreatLevel = None) -> List[IncidentResponse]:
        """
        Get incidents with filters.
        
        Args:
            status: Filter by status
            threat_level: Filter by threat level
            
        Returns:
            List of matching incidents
        """
        incidents = self.incidents.copy()
        
        if status:
            incidents = [i for i in incidents if i.status == status]
        
        if threat_level:
            incidents = [i for i in incidents if i.threat_level == threat_level]
        
        return sorted(incidents, key=lambda x: x.created_at, reverse=True)
    
    def generate_security_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Generate security monitoring report.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            Security report dictionary
        """
        # Get events in date range
        events = self.get_security_events(start_date, end_date)
        
        # Get incidents in date range
        incidents = [i for i in self.incidents if start_date <= i.created_at <= end_date]
        
        # Calculate statistics
        event_type_counts = {}
        threat_level_counts = {}
        
        for event in events:
            event_type_counts[event.event_type.value] = event_type_counts.get(event.event_type.value, 0) + 1
            threat_level_counts[event.threat_level.value] = threat_level_counts.get(event.threat_level.value, 0) + 1
        
        # Calculate incident statistics
        incident_status_counts = {}
        incident_threat_level_counts = {}
        
        for incident in incidents:
            incident_status_counts[incident.status.value] = incident_status_counts.get(incident.status.value, 0) + 1
            incident_threat_level_counts[incident.threat_level.value] = incident_threat_level_counts.get(incident.threat_level.value, 0) + 1
        
        return {
            'report_period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            },
            'summary': {
                'total_events': len(events),
                'total_incidents': len(incidents),
                'threat_intelligence_indicators': len(self.threat_intelligence),
                'active_detection_rules': len([r for r in self.detection_rules.values() if r.get('enabled', True)])
            },
            'event_statistics': {
                'by_type': event_type_counts,
                'by_threat_level': threat_level_counts
            },
            'incident_statistics': {
                'by_status': incident_status_counts,
                'by_threat_level': incident_threat_level_counts
            },
            'top_events': [
                {
                    'event_id': e.event_id,
                    'event_type': e.event_type.value,
                    'threat_level': e.threat_level.value,
                    'timestamp': e.timestamp.isoformat(),
                    'description': e.description
                }
                for e in events[:10]
            ],
            'active_incidents': [
                {
                    'incident_id': i.incident_id,
                    'title': i.title,
                    'threat_level': i.threat_level.value,
                    'status': i.status.value,
                    'created_at': i.created_at.isoformat()
                }
                for i in incidents if i.status != IncidentStatus.CLOSED
            ],
            'generated_at': datetime.now().isoformat()
        }
    
    def _store_security_event(self, event: SecurityEvent) -> None:
        """Store security event in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO security_events 
            (event_id, event_type, threat_level, timestamp, source_ip, source_user,
             target_system, description, raw_data, indicators, context, correlation_id, processed, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            event.event_id, event.event_type.value, event.threat_level.value,
            event.timestamp.isoformat(), event.source_ip, event.source_user,
            event.target_system, event.description, json.dumps(event.raw_data),
            json.dumps(event.indicators), json.dumps(event.context),
            event.correlation_id, event.processed, event.created_at.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def _store_threat_intelligence(self, ti: ThreatIntelligence) -> None:
        """Store threat intelligence in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO threat_intelligence 
            (ti_id, indicator_type, indicator_value, threat_level, confidence, source,
             description, first_seen, last_seen, tags, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ti.ti_id, ti.indicator_type.value, ti.indicator_value, ti.threat_level.value,
            ti.confidence, ti.source, ti.description, ti.first_seen.isoformat(),
            ti.last_seen.isoformat(), json.dumps(ti.tags), json.dumps(ti.metadata),
            ti.created_at.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def _store_incident(self, incident: IncidentResponse) -> None:
        """Store incident in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO incidents 
            (incident_id, title, description, threat_level, status, assigned_to,
             created_at, updated_at, resolved_at, affected_systems, indicators,
             response_actions, lessons_learned)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            incident.incident_id, incident.title, incident.description,
            incident.threat_level.value, incident.status.value, incident.assigned_to,
            incident.created_at.isoformat(), incident.updated_at.isoformat(),
            incident.resolved_at.isoformat() if incident.resolved_at else None,
            json.dumps(incident.affected_systems), json.dumps(incident.indicators),
            json.dumps(incident.response_actions), json.dumps(incident.lessons_learned)
        ))
        
        conn.commit()
        conn.close()
    
    def _update_incident(self, incident: IncidentResponse) -> None:
        """Update incident in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE incidents
            SET status = ?, assigned_to = ?, updated_at = ?, resolved_at = ?,
                response_actions = ?, lessons_learned = ?
            WHERE incident_id = ?
        """, (
            incident.status.value, incident.assigned_to, incident.updated_at.isoformat(),
            incident.resolved_at.isoformat() if incident.resolved_at else None,
            json.dumps(incident.response_actions), json.dumps(incident.lessons_learned),
            incident.incident_id
        ))
        
        conn.commit()
        conn.close()
    
    def _store_detection_rule(self, rule: Dict[str, Any]) -> None:
        """Store detection rule in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO detection_rules 
            (rule_id, name, description, rule_type, conditions, actions, enabled, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            rule['rule_id'], rule['name'], rule['description'], rule['rule_type'],
            json.dumps(rule['conditions']), json.dumps(rule['actions']),
            rule.get('enabled', True), datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
