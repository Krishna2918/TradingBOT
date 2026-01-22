"""
Performance Tracking System

This module implements a comprehensive performance tracking system with
real-time metrics collection, performance analysis, and automated
performance alerting for enterprise-grade performance monitoring.

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
import psutil
import time
import threading
import asyncio
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Performance metric types."""
    CPU_USAGE = "CPU_USAGE"
    MEMORY_USAGE = "MEMORY_USAGE"
    DISK_USAGE = "DISK_USAGE"
    NETWORK_IO = "NETWORK_IO"
    RESPONSE_TIME = "RESPONSE_TIME"
    THROUGHPUT = "THROUGHPUT"
    ERROR_RATE = "ERROR_RATE"
    CONCURRENT_USERS = "CONCURRENT_USERS"
    QUEUE_LENGTH = "QUEUE_LENGTH"
    CACHE_HIT_RATE = "CACHE_HIT_RATE"

class PerformanceLevel(Enum):
    """Performance levels."""
    EXCELLENT = "EXCELLENT"
    GOOD = "GOOD"
    FAIR = "FAIR"
    POOR = "POOR"
    CRITICAL = "CRITICAL"

class AlertType(Enum):
    """Performance alert types."""
    THRESHOLD_EXCEEDED = "THRESHOLD_EXCEEDED"
    ANOMALY_DETECTED = "ANOMALY_DETECTED"
    TREND_DEGRADATION = "TREND_DEGRADATION"
    CAPACITY_WARNING = "CAPACITY_WARNING"
    PERFORMANCE_RECOVERY = "PERFORMANCE_RECOVERY"

@dataclass
class PerformanceMetric:
    """Performance metric definition."""
    metric_id: str
    metric_type: MetricType
    value: float
    unit: str
    timestamp: datetime
    component: str
    instance: str
    metadata: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class PerformanceAlert:
    """Performance alert definition."""
    alert_id: str
    metric_type: MetricType
    component: str
    instance: str
    alert_type: AlertType
    severity: str
    title: str
    description: str
    current_value: float
    threshold_value: float
    deviation_percentage: float
    timestamp: datetime
    acknowledged: bool = False
    resolved: bool = False
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class PerformanceReport:
    """Performance report definition."""
    report_id: str
    component: str
    report_period_start: datetime
    report_period_end: datetime
    metrics_summary: Dict[str, Dict[str, float]]
    performance_level: PerformanceLevel
    alerts_count: int
    alerts: List[PerformanceAlert]
    recommendations: List[str]
    trends: Dict[str, str]
    generated_at: datetime = field(default_factory=datetime.now)

class PerformanceTracker:
    """
    Comprehensive performance tracking system.
    
    Features:
    - Real-time performance metrics collection
    - System resource monitoring
    - Application performance tracking
    - Automated performance alerting
    - Performance trend analysis
    - Capacity planning insights
    - Performance optimization recommendations
    """
    
    def __init__(self, db_path: str = "data/performance_tracking.db"):
        """
        Initialize performance tracker.
        
        Args:
            db_path: Path to performance tracking database
        """
        self.db_path = db_path
        self.metrics: List[PerformanceMetric] = []
        self.alerts: List[PerformanceAlert] = []
        self.tracking_active = False
        self.tracking_thread = None
        
        # Performance thresholds
        self.performance_thresholds = {
            MetricType.CPU_USAGE: {'warning': 70.0, 'critical': 90.0},
            MetricType.MEMORY_USAGE: {'warning': 80.0, 'critical': 95.0},
            MetricType.DISK_USAGE: {'warning': 85.0, 'critical': 95.0},
            MetricType.RESPONSE_TIME: {'warning': 1000.0, 'critical': 5000.0},
            MetricType.THROUGHPUT: {'warning': 100.0, 'critical': 50.0},
            MetricType.ERROR_RATE: {'warning': 1.0, 'critical': 5.0},
            MetricType.CONCURRENT_USERS: {'warning': 1000.0, 'critical': 2000.0},
            MetricType.QUEUE_LENGTH: {'warning': 100.0, 'critical': 500.0},
            MetricType.CACHE_HIT_RATE: {'warning': 80.0, 'critical': 60.0}
        }
        
        # Tracking configuration
        self.tracking_config = {
            'collection_interval': 30,  # seconds
            'data_retention_days': 30,
            'alert_cooldown_minutes': 10,
            'anomaly_detection_window': 60,  # minutes
            'trend_analysis_window': 24  # hours
        }
        
        # Performance baselines
        self.performance_baselines = {}
        
        # Initialize database
        self._init_database()
        
        logger.info("Performance Tracker initialized")
    
    def _init_database(self) -> None:
        """Initialize performance tracking database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create performance metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                metric_id TEXT PRIMARY KEY,
                metric_type TEXT NOT NULL,
                value REAL NOT NULL,
                unit TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                component TEXT NOT NULL,
                instance TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT NOT NULL
            )
        """)
        
        # Create performance alerts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_alerts (
                alert_id TEXT PRIMARY KEY,
                metric_type TEXT NOT NULL,
                component TEXT NOT NULL,
                instance TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                current_value REAL NOT NULL,
                threshold_value REAL NOT NULL,
                deviation_percentage REAL NOT NULL,
                timestamp TEXT NOT NULL,
                acknowledged INTEGER,
                resolved INTEGER,
                created_at TEXT NOT NULL
            )
        """)
        
        # Create performance baselines table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_baselines (
                baseline_id TEXT PRIMARY KEY,
                metric_type TEXT NOT NULL,
                component TEXT NOT NULL,
                instance TEXT NOT NULL,
                baseline_value REAL NOT NULL,
                standard_deviation REAL NOT NULL,
                calculation_period TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def start_tracking(self) -> None:
        """Start performance tracking."""
        if self.tracking_active:
            logger.warning("Performance tracking is already active")
            return
        
        self.tracking_active = True
        self.tracking_thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self.tracking_thread.start()
        
        logger.info("Performance tracking started")
    
    def stop_tracking(self) -> None:
        """Stop performance tracking."""
        self.tracking_active = False
        if self.tracking_thread:
            self.tracking_thread.join(timeout=5)
        
        logger.info("Performance tracking stopped")
    
    def _tracking_loop(self) -> None:
        """Main tracking loop."""
        while self.tracking_active:
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Collect application metrics
                self._collect_application_metrics()
                
                # Analyze performance
                self._analyze_performance()
                
                # Sleep for the configured interval
                time.sleep(self.tracking_config['collection_interval'])
                
            except Exception as e:
                logger.error(f"Error in tracking loop: {e}")
                time.sleep(5)  # Short sleep on error
    
    def _collect_system_metrics(self) -> None:
        """Collect system performance metrics."""
        try:
            current_time = datetime.now()
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self._store_metric(
                MetricType.CPU_USAGE, cpu_percent, "%", "system", "cpu", 
                {"cores": psutil.cpu_count()}, current_time
            )
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            self._store_metric(
                MetricType.MEMORY_USAGE, memory_percent, "%", "system", "memory",
                {"total": memory.total, "available": memory.available}, current_time
            )
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self._store_metric(
                MetricType.DISK_USAGE, disk_percent, "%", "system", "disk",
                {"total": disk.total, "used": disk.used, "free": disk.free}, current_time
            )
            
            # Network I/O
            network = psutil.net_io_counters()
            network_io = network.bytes_sent + network.bytes_recv
            self._store_metric(
                MetricType.NETWORK_IO, network_io, "bytes", "system", "network",
                {"sent": network.bytes_sent, "recv": network.bytes_recv}, current_time
            )
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    def _collect_application_metrics(self) -> None:
        """Collect application performance metrics."""
        try:
            current_time = datetime.now()
            
            # Simulate application metrics
            response_time = np.random.normal(150, 30)  # Simulated response time
            throughput = np.random.normal(500, 50)     # Simulated throughput
            error_rate = np.random.exponential(0.5)   # Simulated error rate
            concurrent_users = np.random.normal(200, 50)  # Simulated concurrent users
            queue_length = np.random.poisson(10)       # Simulated queue length
            cache_hit_rate = np.random.normal(85, 5)   # Simulated cache hit rate
            
            # Store application metrics
            self._store_metric(
                MetricType.RESPONSE_TIME, response_time, "ms", "application", "api",
                {"endpoint": "trading_api"}, current_time
            )
            
            self._store_metric(
                MetricType.THROUGHPUT, throughput, "req/s", "application", "api",
                {"endpoint": "trading_api"}, current_time
            )
            
            self._store_metric(
                MetricType.ERROR_RATE, error_rate, "%", "application", "api",
                {"endpoint": "trading_api"}, current_time
            )
            
            self._store_metric(
                MetricType.CONCURRENT_USERS, concurrent_users, "users", "application", "web",
                {"service": "trading_platform"}, current_time
            )
            
            self._store_metric(
                MetricType.QUEUE_LENGTH, queue_length, "requests", "application", "queue",
                {"queue_name": "trading_queue"}, current_time
            )
            
            self._store_metric(
                MetricType.CACHE_HIT_RATE, cache_hit_rate, "%", "application", "cache",
                {"cache_name": "redis_cache"}, current_time
            )
            
        except Exception as e:
            logger.error(f"Error collecting application metrics: {e}")
    
    def _store_metric(self, metric_type: MetricType, value: float, unit: str, 
                     component: str, instance: str, metadata: Dict[str, Any], 
                     timestamp: datetime) -> None:
        """Store performance metric."""
        metric_id = f"{metric_type.value}_{component}_{instance}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        metric = PerformanceMetric(
            metric_id=metric_id,
            metric_type=metric_type,
            value=value,
            unit=unit,
            timestamp=timestamp,
            component=component,
            instance=instance,
            metadata=metadata
        )
        
        self.metrics.append(metric)
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO performance_metrics 
            (metric_id, metric_type, value, unit, timestamp, component, instance, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metric_id, metric_type.value, value, unit, timestamp.isoformat(),
            component, instance, json.dumps(metadata), datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def _analyze_performance(self) -> None:
        """Analyze performance and detect issues."""
        try:
            # Get recent metrics
            recent_metrics = self._get_recent_metrics()
            
            if not recent_metrics:
                return
            
            # Check thresholds
            self._check_performance_thresholds(recent_metrics)
            
            # Detect anomalies
            self._detect_performance_anomalies(recent_metrics)
            
            # Analyze trends
            self._analyze_performance_trends(recent_metrics)
            
        except Exception as e:
            logger.error(f"Error analyzing performance: {e}")
    
    def _get_recent_metrics(self, minutes: int = 60) -> List[PerformanceMetric]:
        """Get recent metrics."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [m for m in self.metrics if m.timestamp >= cutoff_time]
    
    def _check_performance_thresholds(self, metrics: List[PerformanceMetric]) -> None:
        """Check performance against thresholds."""
        for metric in metrics:
            if metric.metric_type not in self.performance_thresholds:
                continue
            
            thresholds = self.performance_thresholds[metric.metric_type]
            current_value = metric.value
            
            # Check critical threshold
            if current_value >= thresholds['critical']:
                self._create_performance_alert(
                    metric, AlertType.THRESHOLD_EXCEEDED, "CRITICAL",
                    f"{metric.metric_type.value} critical threshold exceeded",
                    current_value, thresholds['critical']
                )
            
            # Check warning threshold
            elif current_value >= thresholds['warning']:
                self._create_performance_alert(
                    metric, AlertType.THRESHOLD_EXCEEDED, "WARNING",
                    f"{metric.metric_type.value} warning threshold exceeded",
                    current_value, thresholds['warning']
                )
    
    def _detect_performance_anomalies(self, metrics: List[PerformanceMetric]) -> None:
        """Detect performance anomalies."""
        # Group metrics by type and component
        metric_groups = {}
        for metric in metrics:
            key = (metric.metric_type, metric.component, metric.instance)
            if key not in metric_groups:
                metric_groups[key] = []
            metric_groups[key].append(metric)
        
        # Analyze each group for anomalies
        for key, group_metrics in metric_groups.items():
            if len(group_metrics) < 10:  # Need enough data points
                continue
            
            values = [m.value for m in group_metrics]
            
            # Simple anomaly detection using z-score
            mean_value = np.mean(values)
            std_value = np.std(values)
            
            if std_value > 0:
                z_scores = [(v - mean_value) / std_value for v in values]
                
                # Check for anomalies (z-score > 3)
                for i, z_score in enumerate(z_scores):
                    if abs(z_score) > 3:
                        metric = group_metrics[i]
                        self._create_performance_alert(
                            metric, AlertType.ANOMALY_DETECTED, "WARNING",
                            f"Performance anomaly detected in {metric.metric_type.value}",
                            metric.value, mean_value
                        )
    
    def _analyze_performance_trends(self, metrics: List[PerformanceMetric]) -> None:
        """Analyze performance trends."""
        # Group metrics by type and component
        metric_groups = {}
        for metric in metrics:
            key = (metric.metric_type, metric.component, metric.instance)
            if key not in metric_groups:
                metric_groups[key] = []
            metric_groups[key].append(metric)
        
        # Analyze trends for each group
        for key, group_metrics in metric_groups.items():
            if len(group_metrics) < 20:  # Need enough data points
                continue
            
            # Sort by timestamp
            group_metrics.sort(key=lambda x: x.timestamp)
            values = [m.value for m in group_metrics]
            
            # Calculate trend using linear regression
            x = np.arange(len(values))
            y = np.array(values)
            
            # Simple linear regression
            n = len(x)
            sum_x = np.sum(x)
            sum_y = np.sum(y)
            sum_xy = np.sum(x * y)
            sum_x2 = np.sum(x * x)
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            
            # Check for significant degradation trend
            if slope > 0.1:  # Positive slope indicates degradation
                latest_metric = group_metrics[-1]
                self._create_performance_alert(
                    latest_metric, AlertType.TREND_DEGRADATION, "WARNING",
                    f"Performance degradation trend detected in {latest_metric.metric_type.value}",
                    latest_metric.value, values[0]  # Compare with first value
                )
    
    def _create_performance_alert(self, metric: PerformanceMetric, alert_type: AlertType, 
                                severity: str, description: str, current_value: float, 
                                threshold_value: float) -> None:
        """Create performance alert."""
        alert_id = f"PERF_ALERT_{metric.metric_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        deviation_percentage = abs((current_value - threshold_value) / threshold_value) * 100
        
        alert = PerformanceAlert(
            alert_id=alert_id,
            metric_type=metric.metric_type,
            component=metric.component,
            instance=metric.instance,
            alert_type=alert_type,
            severity=severity,
            title=f"Performance Alert: {metric.metric_type.value}",
            description=description,
            current_value=current_value,
            threshold_value=threshold_value,
            deviation_percentage=deviation_percentage,
            timestamp=datetime.now()
        )
        
        self.alerts.append(alert)
        self._store_performance_alert(alert)
        
        logger.warning(f"Performance alert created: {alert_id} - {description}")
    
    def _store_performance_alert(self, alert: PerformanceAlert) -> None:
        """Store performance alert in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO performance_alerts 
            (alert_id, metric_type, component, instance, alert_type, severity, title,
             description, current_value, threshold_value, deviation_percentage, timestamp,
             acknowledged, resolved, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            alert.alert_id, alert.metric_type.value, alert.component, alert.instance,
            alert.alert_type.value, alert.severity, alert.title, alert.description,
            alert.current_value, alert.threshold_value, alert.deviation_percentage,
            alert.timestamp.isoformat(), alert.acknowledged, alert.resolved,
            alert.created_at.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def get_performance_summary(self, component: str = None, hours: int = 24) -> Dict[str, Any]:
        """
        Get performance summary.
        
        Args:
            component: Filter by component
            hours: Time window in hours
            
        Returns:
            Performance summary dictionary
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [m for m in self.metrics if m.timestamp >= cutoff_time]
        
        if component:
            recent_metrics = [m for m in recent_metrics if m.component == component]
        
        # Group metrics by type
        metric_summary = {}
        for metric in recent_metrics:
            metric_type = metric.metric_type.value
            if metric_type not in metric_summary:
                metric_summary[metric_type] = []
            metric_summary[metric_type].append(metric.value)
        
        # Calculate statistics
        summary = {
            'summary': {
                'total_metrics': len(recent_metrics),
                'active_alerts': len([a for a in self.alerts if not a.resolved]),
                'tracking_active': self.tracking_active,
                'time_window_hours': hours
            },
            'metrics_summary': {},
            'performance_level': self._calculate_overall_performance_level(metric_summary),
            'recent_alerts': [
                {
                    'alert_id': a.alert_id,
                    'metric_type': a.metric_type.value,
                    'component': a.component,
                    'severity': a.severity,
                    'title': a.title,
                    'timestamp': a.timestamp.isoformat()
                }
                for a in self.alerts[-10:]  # Last 10 alerts
            ]
        }
        
        # Calculate metric statistics
        for metric_type, values in metric_summary.items():
            if values:
                summary['metrics_summary'][metric_type] = {
                    'count': len(values),
                    'average': np.mean(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'std': np.std(values),
                    'p95': np.percentile(values, 95),
                    'p99': np.percentile(values, 99)
                }
        
        return summary
    
    def _calculate_overall_performance_level(self, metric_summary: Dict[str, List[float]]) -> str:
        """Calculate overall performance level."""
        if not metric_summary:
            return PerformanceLevel.FAIR.value
        
        # Check critical metrics
        critical_issues = 0
        warning_issues = 0
        
        for metric_type, values in metric_summary.items():
            if not values:
                continue
            
            avg_value = np.mean(values)
            
            if metric_type in self.performance_thresholds:
                thresholds = self.performance_thresholds[metric_type]
                
                if avg_value >= thresholds['critical']:
                    critical_issues += 1
                elif avg_value >= thresholds['warning']:
                    warning_issues += 1
        
        if critical_issues > 0:
            return PerformanceLevel.CRITICAL.value
        elif warning_issues > 2:
            return PerformanceLevel.POOR.value
        elif warning_issues > 0:
            return PerformanceLevel.FAIR.value
        else:
            return PerformanceLevel.GOOD.value
    
    def generate_performance_report(self, component: str, start_date: datetime, 
                                  end_date: datetime) -> PerformanceReport:
        """
        Generate performance report.
        
        Args:
            component: Component to report on
            start_date: Report start date
            end_date: Report end date
            
        Returns:
            Performance report
        """
        # Get metrics for the period
        period_metrics = [
            m for m in self.metrics
            if m.component == component and start_date <= m.timestamp <= end_date
        ]
        
        # Get alerts for the period
        period_alerts = [
            a for a in self.alerts
            if a.component == component and start_date <= a.timestamp <= end_date
        ]
        
        # Calculate metrics summary
        metrics_summary = {}
        for metric in period_metrics:
            metric_type = metric.metric_type.value
            if metric_type not in metrics_summary:
                metrics_summary[metric_type] = []
            metrics_summary[metric_type].append(metric.value)
        
        # Calculate statistics for each metric type
        metrics_stats = {}
        for metric_type, values in metrics_summary.items():
            if values:
                metrics_stats[metric_type] = {
                    'average': np.mean(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'std': np.std(values),
                    'p95': np.percentile(values, 95),
                    'p99': np.percentile(values, 99)
                }
        
        # Calculate performance level
        performance_level = self._calculate_overall_performance_level(metrics_summary)
        
        # Generate recommendations
        recommendations = self._generate_performance_recommendations(
            metrics_stats, period_alerts
        )
        
        # Analyze trends
        trends = self._analyze_performance_trends_for_report(period_metrics)
        
        report = PerformanceReport(
            report_id=f"PERF_REPORT_{component}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}",
            component=component,
            report_period_start=start_date,
            report_period_end=end_date,
            metrics_summary=metrics_stats,
            performance_level=PerformanceLevel(performance_level),
            alerts_count=len(period_alerts),
            alerts=period_alerts,
            recommendations=recommendations,
            trends=trends
        )
        
        return report
    
    def _generate_performance_recommendations(self, metrics_stats: Dict[str, Dict[str, float]], 
                                            alerts: List[PerformanceAlert]) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []
        
        # Analyze metrics and generate recommendations
        for metric_type, stats in metrics_stats.items():
            avg_value = stats['average']
            p95_value = stats['p95']
            
            if metric_type in self.performance_thresholds:
                thresholds = self.performance_thresholds[metric_type]
                
                if avg_value >= thresholds['critical']:
                    recommendations.append(f"CRITICAL: {metric_type} is critically high ({avg_value:.2f}). Immediate action required.")
                elif avg_value >= thresholds['warning']:
                    recommendations.append(f"WARNING: {metric_type} is above warning threshold ({avg_value:.2f}). Monitor closely.")
                
                # Check P95 values
                if p95_value >= thresholds['critical']:
                    recommendations.append(f"P95 {metric_type} is critically high ({p95_value:.2f}). Consider capacity planning.")
        
        # Analyze alerts
        critical_alerts = [a for a in alerts if a.severity == "CRITICAL"]
        warning_alerts = [a for a in alerts if a.severity == "WARNING"]
        
        if critical_alerts:
            recommendations.append(f"Address {len(critical_alerts)} critical performance alerts")
        
        if warning_alerts:
            recommendations.append(f"Monitor {len(warning_alerts)} warning-level performance alerts")
        
        # General recommendations
        recommendations.extend([
            "Implement performance monitoring and alerting",
            "Regular performance optimization reviews",
            "Capacity planning and scaling strategies",
            "Performance testing and benchmarking"
        ])
        
        return recommendations
    
    def _analyze_performance_trends_for_report(self, metrics: List[PerformanceMetric]) -> Dict[str, str]:
        """Analyze performance trends for report."""
        trends = {}
        
        # Group metrics by type
        metric_groups = {}
        for metric in metrics:
            metric_type = metric.metric_type.value
            if metric_type not in metric_groups:
                metric_groups[metric_type] = []
            metric_groups[metric_type].append(metric)
        
        # Analyze trends for each metric type
        for metric_type, group_metrics in metric_groups.items():
            if len(group_metrics) < 10:
                continue
            
            # Sort by timestamp
            group_metrics.sort(key=lambda x: x.timestamp)
            values = [m.value for m in group_metrics]
            
            # Calculate trend
            x = np.arange(len(values))
            y = np.array(values)
            
            # Simple linear regression
            n = len(x)
            sum_x = np.sum(x)
            sum_y = np.sum(y)
            sum_xy = np.sum(x * y)
            sum_x2 = np.sum(x * x)
            
            if n * sum_x2 - sum_x * sum_x != 0:
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
                
                if slope > 0.1:
                    trends[metric_type] = "DEGRADING"
                elif slope < -0.1:
                    trends[metric_type] = "IMPROVING"
                else:
                    trends[metric_type] = "STABLE"
            else:
                trends[metric_type] = "STABLE"
        
        return trends
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """
        Acknowledge a performance alert.
        
        Args:
            alert_id: Alert ID
            
        Returns:
            True if acknowledged successfully
        """
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                self._update_performance_alert(alert)
                logger.info(f"Performance alert acknowledged: {alert_id}")
                return True
        
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """
        Resolve a performance alert.
        
        Args:
            alert_id: Alert ID
            
        Returns:
            True if resolved successfully
        """
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                self._update_performance_alert(alert)
                logger.info(f"Performance alert resolved: {alert_id}")
                return True
        
        return False
    
    def _update_performance_alert(self, alert: PerformanceAlert) -> None:
        """Update performance alert in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE performance_alerts
            SET acknowledged = ?, resolved = ?
            WHERE alert_id = ?
        """, (alert.acknowledged, alert.resolved, alert.alert_id))
        
        conn.commit()
        conn.close()
    
    def get_performance_trends(self, metric_type: MetricType, component: str, 
                             hours: int = 24) -> Dict[str, Any]:
        """
        Get performance trends for a specific metric.
        
        Args:
            metric_type: Metric type
            component: Component
            hours: Time window in hours
            
        Returns:
            Performance trends dictionary
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [
            m for m in self.metrics
            if m.metric_type == metric_type and m.component == component and m.timestamp >= cutoff_time
        ]
        
        if not recent_metrics:
            return {'trend': 'NO_DATA', 'values': [], 'timestamps': []}
        
        # Sort by timestamp
        recent_metrics.sort(key=lambda x: x.timestamp)
        
        values = [m.value for m in recent_metrics]
        timestamps = [m.timestamp.isoformat() for m in recent_metrics]
        
        # Calculate trend
        x = np.arange(len(values))
        y = np.array(values)
        
        if len(values) > 1:
            # Simple linear regression
            n = len(x)
            sum_x = np.sum(x)
            sum_y = np.sum(y)
            sum_xy = np.sum(x * y)
            sum_x2 = np.sum(x * x)
            
            if n * sum_x2 - sum_x * sum_x != 0:
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
                
                if slope > 0.1:
                    trend = "INCREASING"
                elif slope < -0.1:
                    trend = "DECREASING"
                else:
                    trend = "STABLE"
            else:
                trend = "STABLE"
        else:
            trend = "INSUFFICIENT_DATA"
        
        return {
            'trend': trend,
            'values': values,
            'timestamps': timestamps,
            'average': np.mean(values) if values else 0,
            'min': np.min(values) if values else 0,
            'max': np.max(values) if values else 0,
            'std': np.std(values) if values else 0
        }
