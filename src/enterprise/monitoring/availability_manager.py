"""
Availability Management System

This module implements a comprehensive availability management system with
uptime tracking, downtime monitoring, availability calculations, and
enterprise-grade availability reporting for SLA compliance.

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
import requests
import socket
import threading
import time
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AvailabilityStatus(Enum):
    """Availability status levels."""
    UP = "UP"
    DOWN = "DOWN"
    DEGRADED = "DEGRADED"
    UNKNOWN = "UNKNOWN"

class DowntimeReason(Enum):
    """Downtime reason categories."""
    PLANNED_MAINTENANCE = "PLANNED_MAINTENANCE"
    UNPLANNED_OUTAGE = "UNPLANNED_OUTAGE"
    NETWORK_ISSUE = "NETWORK_ISSUE"
    HARDWARE_FAILURE = "HARDWARE_FAILURE"
    SOFTWARE_BUG = "SOFTWARE_BUG"
    CAPACITY_ISSUE = "CAPACITY_ISSUE"
    SECURITY_INCIDENT = "SECURITY_INCIDENT"
    THIRD_PARTY_ISSUE = "THIRD_PARTY_ISSUE"
    UNKNOWN = "UNKNOWN"

class AvailabilityLevel(Enum):
    """Availability level classifications."""
    EXCELLENT = "EXCELLENT"  # > 99.9%
    GOOD = "GOOD"           # 99.0% - 99.9%
    FAIR = "FAIR"           # 95.0% - 99.0%
    POOR = "POOR"           # 90.0% - 95.0%
    CRITICAL = "CRITICAL"   # < 90.0%

@dataclass
class AvailabilityMetric:
    """Availability metric definition."""
    metric_id: str
    service: str
    endpoint: str
    status: AvailabilityStatus
    response_time: float
    timestamp: datetime
    metadata: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class DowntimeEvent:
    """Downtime event definition."""
    event_id: str
    service: str
    endpoint: str
    start_time: datetime
    end_time: Optional[datetime]
    duration_minutes: float
    reason: DowntimeReason
    severity: str
    description: str
    impact_assessment: str
    resolution_actions: List[str]
    resolved: bool = False
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class AvailabilityReport:
    """Availability report definition."""
    report_id: str
    service: str
    report_period_start: datetime
    report_period_end: datetime
    uptime_percentage: float
    downtime_minutes: float
    availability_level: AvailabilityLevel
    total_checks: int
    successful_checks: int
    failed_checks: int
    average_response_time: float
    max_response_time: float
    min_response_time: float
    downtime_events: List[DowntimeEvent]
    sla_compliance: bool
    recommendations: List[str]
    generated_at: datetime = field(default_factory=datetime.now)

class AvailabilityManager:
    """
    Comprehensive availability management system.
    
    Features:
    - Real-time availability monitoring
    - Uptime tracking and calculation
    - Downtime event detection and management
    - SLA compliance monitoring
    - Availability reporting and analytics
    - Automated health checks
    - Service dependency tracking
    """
    
    def __init__(self, db_path: str = "data/availability_management.db"):
        """
        Initialize availability manager.
        
        Args:
            db_path: Path to availability management database
        """
        self.db_path = db_path
        self.availability_metrics: List[AvailabilityMetric] = []
        self.downtime_events: List[DowntimeEvent] = []
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Service configurations
        self.service_configs = {
            'trading_system': {
                'endpoints': ['http://localhost:8000/health', 'http://localhost:8000/api/status'],
                'check_interval': 30,  # seconds
                'timeout': 10,  # seconds
                'expected_status_codes': [200, 201],
                'sla_target': 99.9
            },
            'api_gateway': {
                'endpoints': ['http://localhost:8080/health', 'http://localhost:8080/status'],
                'check_interval': 30,
                'timeout': 10,
                'expected_status_codes': [200],
                'sla_target': 99.5
            },
            'database': {
                'endpoints': ['tcp://localhost:5432'],  # Database connection check
                'check_interval': 60,
                'timeout': 5,
                'expected_status_codes': [],  # Not applicable for database
                'sla_target': 99.95
            },
            'ai_models': {
                'endpoints': ['http://localhost:9000/health', 'http://localhost:9000/models/status'],
                'check_interval': 60,
                'timeout': 15,
                'expected_status_codes': [200],
                'sla_target': 99.0
            },
            'market_data': {
                'endpoints': ['http://localhost:7000/health', 'http://localhost:7000/feed/status'],
                'check_interval': 30,
                'timeout': 10,
                'expected_status_codes': [200],
                'sla_target': 99.99
            }
        }
        
        # Availability thresholds
        self.availability_thresholds = {
            AvailabilityLevel.EXCELLENT: 99.9,
            AvailabilityLevel.GOOD: 99.0,
            AvailabilityLevel.FAIR: 95.0,
            AvailabilityLevel.POOR: 90.0,
            AvailabilityLevel.CRITICAL: 0.0
        }
        
        # Monitoring configuration
        self.monitoring_config = {
            'data_retention_days': 90,
            'downtime_detection_threshold': 3,  # consecutive failures
            'recovery_detection_threshold': 2,  # consecutive successes
            'max_response_time_warning': 5000,  # milliseconds
            'max_response_time_critical': 10000  # milliseconds
        }
        
        # Initialize database
        self._init_database()
        
        logger.info("Availability Manager initialized")
    
    def _init_database(self) -> None:
        """Initialize availability management database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create availability metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS availability_metrics (
                metric_id TEXT PRIMARY KEY,
                service TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                status TEXT NOT NULL,
                response_time REAL NOT NULL,
                timestamp TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT NOT NULL
            )
        """)
        
        # Create downtime events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS downtime_events (
                event_id TEXT PRIMARY KEY,
                service TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT,
                duration_minutes REAL NOT NULL,
                reason TEXT NOT NULL,
                severity TEXT NOT NULL,
                description TEXT,
                impact_assessment TEXT,
                resolution_actions TEXT,
                resolved INTEGER,
                created_at TEXT NOT NULL
            )
        """)
        
        # Create availability reports table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS availability_reports (
                report_id TEXT PRIMARY KEY,
                service TEXT NOT NULL,
                report_period_start TEXT NOT NULL,
                report_period_end TEXT NOT NULL,
                uptime_percentage REAL NOT NULL,
                downtime_minutes REAL NOT NULL,
                availability_level TEXT NOT NULL,
                total_checks INTEGER NOT NULL,
                successful_checks INTEGER NOT NULL,
                failed_checks INTEGER NOT NULL,
                average_response_time REAL NOT NULL,
                max_response_time REAL NOT NULL,
                min_response_time REAL NOT NULL,
                downtime_events TEXT,
                sla_compliance INTEGER NOT NULL,
                recommendations TEXT,
                generated_at TEXT NOT NULL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def start_monitoring(self) -> None:
        """Start availability monitoring."""
        if self.monitoring_active:
            logger.warning("Availability monitoring is already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Availability monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop availability monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.info("Availability monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Check all services
                for service, config in self.service_configs.items():
                    self._check_service_availability(service, config)
                
                # Sleep for the minimum check interval
                min_interval = min(config['check_interval'] for config in self.service_configs.values())
                time.sleep(min_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)  # Short sleep on error
    
    def _check_service_availability(self, service: str, config: Dict[str, Any]) -> None:
        """Check availability for a specific service."""
        try:
            for endpoint in config['endpoints']:
                # Perform health check
                status, response_time, metadata = self._perform_health_check(
                    endpoint, config['timeout'], config['expected_status_codes']
                )
                
                # Store metric
                self._store_availability_metric(service, endpoint, status, response_time, metadata)
                
                # Check for downtime events
                self._check_downtime_events(service, endpoint, status)
                
        except Exception as e:
            logger.error(f"Error checking service {service}: {e}")
    
    def _perform_health_check(self, endpoint: str, timeout: int, 
                            expected_status_codes: List[int]) -> Tuple[AvailabilityStatus, float, Dict[str, Any]]:
        """Perform health check on an endpoint."""
        try:
            start_time = time.time()
            
            if endpoint.startswith('tcp://'):
                # Database connection check
                host_port = endpoint.replace('tcp://', '')
                if ':' in host_port:
                    host, port = host_port.split(':')
                    port = int(port)
                else:
                    host = host_port
                    port = 5432  # Default PostgreSQL port
                
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(timeout)
                result = sock.connect_ex((host, port))
                sock.close()
                
                response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
                
                if result == 0:
                    return AvailabilityStatus.UP, response_time, {'connection': 'successful'}
                else:
                    return AvailabilityStatus.DOWN, response_time, {'connection': 'failed', 'error_code': result}
            
            else:
                # HTTP endpoint check
                response = requests.get(endpoint, timeout=timeout)
                response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
                
                if response.status_code in expected_status_codes:
                    return AvailabilityStatus.UP, response_time, {
                        'status_code': response.status_code,
                        'content_length': len(response.content)
                    }
                else:
                    return AvailabilityStatus.DOWN, response_time, {
                        'status_code': response.status_code,
                        'error': 'Unexpected status code'
                    }
        
        except requests.exceptions.Timeout:
            response_time = timeout * 1000
            return AvailabilityStatus.DOWN, response_time, {'error': 'timeout'}
        
        except requests.exceptions.ConnectionError:
            response_time = (time.time() - start_time) * 1000
            return AvailabilityStatus.DOWN, response_time, {'error': 'connection_error'}
        
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return AvailabilityStatus.DOWN, response_time, {'error': str(e)}
    
    def _store_availability_metric(self, service: str, endpoint: str, status: AvailabilityStatus, 
                                 response_time: float, metadata: Dict[str, Any]) -> None:
        """Store availability metric."""
        metric_id = f"{service}_{endpoint}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        metric = AvailabilityMetric(
            metric_id=metric_id,
            service=service,
            endpoint=endpoint,
            status=status,
            response_time=response_time,
            timestamp=datetime.now(),
            metadata=metadata
        )
        
        self.availability_metrics.append(metric)
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO availability_metrics 
            (metric_id, service, endpoint, status, response_time, timestamp, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metric_id, service, endpoint, status.value, response_time,
            metric.timestamp.isoformat(), json.dumps(metadata), datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def _check_downtime_events(self, service: str, endpoint: str, status: AvailabilityStatus) -> None:
        """Check for downtime events."""
        try:
            # Get recent metrics for this service/endpoint
            recent_metrics = self._get_recent_metrics(service, endpoint, minutes=10)
            
            if not recent_metrics:
                return
            
            # Check for downtime detection
            if status == AvailabilityStatus.DOWN:
                consecutive_failures = self._count_consecutive_failures(recent_metrics)
                
                if consecutive_failures >= self.monitoring_config['downtime_detection_threshold']:
                    # Check if we already have an active downtime event
                    active_event = self._get_active_downtime_event(service, endpoint)
                    
                    if not active_event:
                        # Create new downtime event
                        self._create_downtime_event(service, endpoint, recent_metrics[-1])
            
            # Check for recovery detection
            elif status == AvailabilityStatus.UP:
                consecutive_successes = self._count_consecutive_successes(recent_metrics)
                
                if consecutive_successes >= self.monitoring_config['recovery_detection_threshold']:
                    # Close any active downtime event
                    active_event = self._get_active_downtime_event(service, endpoint)
                    
                    if active_event:
                        self._close_downtime_event(active_event)
        
        except Exception as e:
            logger.error(f"Error checking downtime events for {service}/{endpoint}: {e}")
    
    def _get_recent_metrics(self, service: str, endpoint: str, minutes: int = 10) -> List[AvailabilityMetric]:
        """Get recent metrics for a service/endpoint."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [
            m for m in self.availability_metrics
            if m.service == service and m.endpoint == endpoint and m.timestamp >= cutoff_time
        ]
    
    def _count_consecutive_failures(self, metrics: List[AvailabilityMetric]) -> int:
        """Count consecutive failures from the end of the list."""
        count = 0
        for metric in reversed(metrics):
            if metric.status == AvailabilityStatus.DOWN:
                count += 1
            else:
                break
        return count
    
    def _count_consecutive_successes(self, metrics: List[AvailabilityMetric]) -> int:
        """Count consecutive successes from the end of the list."""
        count = 0
        for metric in reversed(metrics):
            if metric.status == AvailabilityStatus.UP:
                count += 1
            else:
                break
        return count
    
    def _get_active_downtime_event(self, service: str, endpoint: str) -> Optional[DowntimeEvent]:
        """Get active downtime event for service/endpoint."""
        for event in self.downtime_events:
            if (event.service == service and event.endpoint == endpoint and 
                not event.resolved and event.end_time is None):
                return event
        return None
    
    def _create_downtime_event(self, service: str, endpoint: str, metric: AvailabilityMetric) -> None:
        """Create new downtime event."""
        event_id = f"DOWN_{service}_{endpoint}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Determine reason and severity
        reason = self._determine_downtime_reason(metric)
        severity = self._determine_downtime_severity(service, metric)
        
        event = DowntimeEvent(
            event_id=event_id,
            service=service,
            endpoint=endpoint,
            start_time=metric.timestamp,
            end_time=None,
            duration_minutes=0.0,
            reason=reason,
            severity=severity,
            description=f"Service {service} endpoint {endpoint} is down",
            impact_assessment=self._assess_downtime_impact(service),
            resolution_actions=self._get_downtime_resolution_actions(service, reason)
        )
        
        self.downtime_events.append(event)
        self._store_downtime_event(event)
        
        logger.warning(f"Downtime event created: {event_id} - {service}/{endpoint}")
    
    def _close_downtime_event(self, event: DowntimeEvent) -> None:
        """Close downtime event."""
        event.end_time = datetime.now()
        event.duration_minutes = (event.end_time - event.start_time).total_seconds() / 60
        event.resolved = True
        
        self._update_downtime_event(event)
        
        logger.info(f"Downtime event resolved: {event.event_id} - Duration: {event.duration_minutes:.2f} minutes")
    
    def _determine_downtime_reason(self, metric: AvailabilityMetric) -> DowntimeReason:
        """Determine downtime reason from metric metadata."""
        metadata = metric.metadata
        
        if 'timeout' in str(metadata.get('error', '')).lower():
            return DowntimeReason.NETWORK_ISSUE
        elif 'connection_error' in str(metadata.get('error', '')).lower():
            return DowntimeReason.NETWORK_ISSUE
        elif 'status_code' in metadata:
            if metadata['status_code'] >= 500:
                return DowntimeReason.SOFTWARE_BUG
            elif metadata['status_code'] >= 400:
                return DowntimeReason.SOFTWARE_BUG
        elif 'connection' in metadata and metadata['connection'] == 'failed':
            return DowntimeReason.HARDWARE_FAILURE
        
        return DowntimeReason.UNKNOWN
    
    def _determine_downtime_severity(self, service: str, metric: AvailabilityMetric) -> str:
        """Determine downtime severity."""
        # Critical services
        if service in ['trading_system', 'market_data', 'database']:
            return 'CRITICAL'
        # Important services
        elif service in ['api_gateway']:
            return 'HIGH'
        # Standard services
        else:
            return 'MEDIUM'
    
    def _assess_downtime_impact(self, service: str) -> str:
        """Assess downtime impact."""
        impact_assessments = {
            'trading_system': 'Critical impact - Trading operations may be affected',
            'api_gateway': 'High impact - API access may be limited',
            'database': 'Critical impact - Data access may be unavailable',
            'ai_models': 'Medium impact - AI predictions may be unavailable',
            'market_data': 'Critical impact - Real-time market data may be unavailable'
        }
        
        return impact_assessments.get(service, 'Unknown impact - Service availability affected')
    
    def _get_downtime_resolution_actions(self, service: str, reason: DowntimeReason) -> List[str]:
        """Get resolution actions for downtime."""
        base_actions = [
            "Check service logs for errors",
            "Verify service configuration",
            "Check system resources (CPU, memory, disk)",
            "Contact service administrator"
        ]
        
        if reason == DowntimeReason.NETWORK_ISSUE:
            base_actions.extend([
                "Check network connectivity",
                "Verify firewall rules",
                "Test DNS resolution"
            ])
        elif reason == DowntimeReason.HARDWARE_FAILURE:
            base_actions.extend([
                "Check hardware status",
                "Verify power and cooling",
                "Contact hardware support"
            ])
        elif reason == DowntimeReason.SOFTWARE_BUG:
            base_actions.extend([
                "Review recent deployments",
                "Check for known issues",
                "Contact development team"
            ])
        
        return base_actions
    
    def _store_downtime_event(self, event: DowntimeEvent) -> None:
        """Store downtime event in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO downtime_events 
            (event_id, service, endpoint, start_time, end_time, duration_minutes,
             reason, severity, description, impact_assessment, resolution_actions, resolved, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            event.event_id, event.service, event.endpoint, event.start_time.isoformat(),
            event.end_time.isoformat() if event.end_time else None, event.duration_minutes,
            event.reason.value, event.severity, event.description, event.impact_assessment,
            json.dumps(event.resolution_actions), event.resolved, event.created_at.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def _update_downtime_event(self, event: DowntimeEvent) -> None:
        """Update downtime event in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE downtime_events
            SET end_time = ?, duration_minutes = ?, resolved = ?
            WHERE event_id = ?
        """, (
            event.end_time.isoformat() if event.end_time else None,
            event.duration_minutes, event.resolved, event.event_id
        ))
        
        conn.commit()
        conn.close()
    
    def get_availability_summary(self, service: str = None, hours: int = 24) -> Dict[str, Any]:
        """
        Get availability summary.
        
        Args:
            service: Filter by service
            hours: Time window in hours
            
        Returns:
            Availability summary dictionary
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [m for m in self.availability_metrics if m.timestamp >= cutoff_time]
        
        if service:
            recent_metrics = [m for m in recent_metrics if m.service == service]
        
        if not recent_metrics:
            return {
                'summary': {
                    'total_checks': 0,
                    'uptime_percentage': 0.0,
                    'availability_level': 'UNKNOWN',
                    'monitoring_active': self.monitoring_active
                },
                'services': {},
                'downtime_events': []
            }
        
        # Calculate uptime percentage
        total_checks = len(recent_metrics)
        successful_checks = len([m for m in recent_metrics if m.status == AvailabilityStatus.UP])
        uptime_percentage = (successful_checks / total_checks) * 100 if total_checks > 0 else 0.0
        
        # Determine availability level
        availability_level = self._get_availability_level(uptime_percentage)
        
        # Group by service
        service_summary = {}
        for metric in recent_metrics:
            if metric.service not in service_summary:
                service_summary[metric.service] = {
                    'total_checks': 0,
                    'successful_checks': 0,
                    'uptime_percentage': 0.0,
                    'average_response_time': 0.0,
                    'max_response_time': 0.0,
                    'min_response_time': 0.0
                }
            
            service_summary[metric.service]['total_checks'] += 1
            if metric.status == AvailabilityStatus.UP:
                service_summary[metric.service]['successful_checks'] += 1
        
        # Calculate service statistics
        for service_name, stats in service_summary.items():
            stats['uptime_percentage'] = (stats['successful_checks'] / stats['total_checks']) * 100
            
            service_metrics = [m for m in recent_metrics if m.service == service_name]
            response_times = [m.response_time for m in service_metrics]
            
            if response_times:
                stats['average_response_time'] = np.mean(response_times)
                stats['max_response_time'] = np.max(response_times)
                stats['min_response_time'] = np.min(response_times)
        
        # Get recent downtime events
        recent_downtime_events = [
            {
                'event_id': e.event_id,
                'service': e.service,
                'endpoint': e.endpoint,
                'start_time': e.start_time.isoformat(),
                'end_time': e.end_time.isoformat() if e.end_time else None,
                'duration_minutes': e.duration_minutes,
                'reason': e.reason.value,
                'severity': e.severity,
                'resolved': e.resolved
            }
            for e in self.downtime_events
            if e.start_time >= cutoff_time
        ]
        
        return {
            'summary': {
                'total_checks': total_checks,
                'successful_checks': successful_checks,
                'uptime_percentage': uptime_percentage,
                'availability_level': availability_level.value,
                'monitoring_active': self.monitoring_active,
                'time_window_hours': hours
            },
            'services': service_summary,
            'downtime_events': recent_downtime_events
        }
    
    def _get_availability_level(self, uptime_percentage: float) -> AvailabilityLevel:
        """Get availability level from uptime percentage."""
        if uptime_percentage >= 99.9:
            return AvailabilityLevel.EXCELLENT
        elif uptime_percentage >= 99.0:
            return AvailabilityLevel.GOOD
        elif uptime_percentage >= 95.0:
            return AvailabilityLevel.FAIR
        elif uptime_percentage >= 90.0:
            return AvailabilityLevel.POOR
        else:
            return AvailabilityLevel.CRITICAL
    
    def generate_availability_report(self, service: str, start_date: datetime, 
                                   end_date: datetime) -> AvailabilityReport:
        """
        Generate availability report.
        
        Args:
            service: Service to report on
            start_date: Report start date
            end_date: Report end date
            
        Returns:
            Availability report
        """
        # Get metrics for the period
        period_metrics = [
            m for m in self.availability_metrics
            if m.service == service and start_date <= m.timestamp <= end_date
        ]
        
        # Get downtime events for the period
        period_downtime_events = [
            e for e in self.downtime_events
            if e.service == service and start_date <= e.start_time <= end_date
        ]
        
        # Calculate statistics
        total_checks = len(period_metrics)
        successful_checks = len([m for m in period_metrics if m.status == AvailabilityStatus.UP])
        failed_checks = total_checks - successful_checks
        
        uptime_percentage = (successful_checks / total_checks) * 100 if total_checks > 0 else 0.0
        
        # Calculate downtime
        total_downtime_minutes = sum(e.duration_minutes for e in period_downtime_events)
        
        # Calculate response time statistics
        response_times = [m.response_time for m in period_metrics if m.status == AvailabilityStatus.UP]
        average_response_time = np.mean(response_times) if response_times else 0.0
        max_response_time = np.max(response_times) if response_times else 0.0
        min_response_time = np.min(response_times) if response_times else 0.0
        
        # Determine availability level
        availability_level = self._get_availability_level(uptime_percentage)
        
        # Check SLA compliance
        sla_target = self.service_configs.get(service, {}).get('sla_target', 99.0)
        sla_compliance = uptime_percentage >= sla_target
        
        # Generate recommendations
        recommendations = self._generate_availability_recommendations(
            uptime_percentage, sla_target, period_downtime_events, response_times
        )
        
        report = AvailabilityReport(
            report_id=f"AVAIL_REPORT_{service}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}",
            service=service,
            report_period_start=start_date,
            report_period_end=end_date,
            uptime_percentage=uptime_percentage,
            downtime_minutes=total_downtime_minutes,
            availability_level=availability_level,
            total_checks=total_checks,
            successful_checks=successful_checks,
            failed_checks=failed_checks,
            average_response_time=average_response_time,
            max_response_time=max_response_time,
            min_response_time=min_response_time,
            downtime_events=period_downtime_events,
            sla_compliance=sla_compliance,
            recommendations=recommendations
        )
        
        # Store report
        self._store_availability_report(report)
        
        return report
    
    def _generate_availability_recommendations(self, uptime_percentage: float, sla_target: float,
                                             downtime_events: List[DowntimeEvent], 
                                             response_times: List[float]) -> List[str]:
        """Generate availability recommendations."""
        recommendations = []
        
        # SLA compliance recommendations
        if uptime_percentage < sla_target:
            recommendations.append(f"Improve availability - currently {uptime_percentage:.2f}%, target {sla_target:.2f}%")
        
        # Downtime event recommendations
        if downtime_events:
            recommendations.append(f"Address {len(downtime_events)} downtime events")
            
            # Analyze downtime reasons
            reasons = [e.reason.value for e in downtime_events]
            reason_counts = {}
            for reason in reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            
            for reason, count in reason_counts.items():
                if count > 1:
                    recommendations.append(f"Investigate recurring {reason.lower()} issues ({count} events)")
        
        # Response time recommendations
        if response_times:
            avg_response_time = np.mean(response_times)
            if avg_response_time > 1000:  # 1 second
                recommendations.append(f"Optimize response time - currently {avg_response_time:.2f}ms average")
        
        # General recommendations
        recommendations.extend([
            "Implement proactive monitoring and alerting",
            "Regular availability testing and validation",
            "Incident response process improvements",
            "Capacity planning and redundancy strategies"
        ])
        
        return recommendations
    
    def _store_availability_report(self, report: AvailabilityReport) -> None:
        """Store availability report in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO availability_reports 
            (report_id, service, report_period_start, report_period_end, uptime_percentage,
             downtime_minutes, availability_level, total_checks, successful_checks, failed_checks,
             average_response_time, max_response_time, min_response_time, downtime_events,
             sla_compliance, recommendations, generated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            report.report_id, report.service, report.report_period_start.isoformat(),
            report.report_period_end.isoformat(), report.uptime_percentage,
            report.downtime_minutes, report.availability_level.value, report.total_checks,
            report.successful_checks, report.failed_checks, report.average_response_time,
            report.max_response_time, report.min_response_time,
            json.dumps([e.__dict__ for e in report.downtime_events]),
            report.sla_compliance, json.dumps(report.recommendations),
            report.generated_at.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def get_service_health(self, service: str) -> Dict[str, Any]:
        """
        Get current health status for a service.
        
        Args:
            service: Service name
            
        Returns:
            Service health dictionary
        """
        if service not in self.service_configs:
            return {'error': f'Service {service} not configured'}
        
        # Get recent metrics for the service
        recent_metrics = self._get_recent_metrics(service, '', minutes=5)
        
        if not recent_metrics:
            return {
                'service': service,
                'status': 'UNKNOWN',
                'last_check': None,
                'endpoints': []
            }
        
        # Group by endpoint
        endpoint_status = {}
        for metric in recent_metrics:
            endpoint = metric.endpoint
            if endpoint not in endpoint_status:
                endpoint_status[endpoint] = {
                    'status': metric.status.value,
                    'response_time': metric.response_time,
                    'last_check': metric.timestamp.isoformat(),
                    'metadata': metric.metadata
                }
        
        # Determine overall service status
        all_up = all(status['status'] == 'UP' for status in endpoint_status.values())
        any_down = any(status['status'] == 'DOWN' for status in endpoint_status.values())
        
        if all_up:
            overall_status = 'UP'
        elif any_down:
            overall_status = 'DOWN'
        else:
            overall_status = 'DEGRADED'
        
        return {
            'service': service,
            'status': overall_status,
            'last_check': max(m.timestamp for m in recent_metrics).isoformat(),
            'endpoints': endpoint_status,
            'sla_target': self.service_configs[service]['sla_target']
        }
