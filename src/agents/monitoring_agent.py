"""
Monitoring Agent

CRITICAL priority agent that monitors system health, performance, and proactively
identifies issues before they impact trading operations.

Responsibilities:
- System health monitoring
- Performance metrics tracking
- Proactive issue detection
- Alert generation
- System optimization recommendations
"""

import logging
import asyncio
import psutil
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass

from .base_agent import BaseAgent, AgentPriority, AgentStatus, ResourceRequirements

logger = logging.getLogger(__name__)


@dataclass
class SystemHealthMetrics:
    """System health metrics snapshot"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    network_io: Dict[str, int]
    active_processes: int
    load_average: float
    uptime_seconds: float


@dataclass
class Alert:
    """System alert"""
    timestamp: datetime
    severity: str  # 'info', 'warning', 'critical'
    category: str  # 'cpu', 'memory', 'disk', 'network', 'process'
    message: str
    value: float
    threshold: float
    recommendation: str


class MonitoringAgent(BaseAgent):
    """
    Monitoring Agent - CRITICAL priority
    
    Continuously monitors system health and performance to ensure
    optimal trading operations.
    """
    
    def __init__(self):
        super().__init__(
            agent_id='monitoring_agent',
            name='System Monitoring Agent',
            priority=AgentPriority.CRITICAL,  # Must always run
            resource_requirements=ResourceRequirements(
                min_cpu_percent=3.0,
                min_memory_mb=80.0,
                max_cpu_percent=10.0,
                max_memory_mb=200.0
            )
        )
        
        # Monitoring state
        self.health_history: List[SystemHealthMetrics] = []
        self.active_alerts: List[Alert] = []
        self.monitoring_active = False
        self.last_health_check = None
        
        # Thresholds
        self.thresholds = {
            'cpu_warning': 70.0,
            'cpu_critical': 85.0,
            'memory_warning': 75.0,
            'memory_critical': 90.0,
            'disk_warning': 80.0,
            'disk_critical': 95.0
        }
        
        # Performance tracking
        self.checks_performed = 0
        self.alerts_generated = 0
        self.optimizations_suggested = 0
    
    async def initialize(self) -> bool:
        """Initialize monitoring systems"""
        try:
            self.status = AgentStatus.IDLE
            self.monitoring_active = True
            
            # Start background monitoring
            asyncio.create_task(self._background_monitoring())
            
            logger.info("Monitoring Agent initialized and background monitoring started")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Monitoring Agent: {e}")
            self.status = AgentStatus.ERROR
            return False
    
    async def shutdown(self) -> bool:
        """Gracefully shutdown monitoring"""
        logger.info("Monitoring Agent shutting down")
        self.monitoring_active = False
        self.status = AgentStatus.STOPPED
        return True
    
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process monitoring tasks.
        
        Task types:
        - 'health_check': Perform immediate health check
        - 'get_alerts': Get current active alerts
        - 'get_metrics': Get system performance metrics
        - 'optimize_system': Analyze and suggest optimizations
        - 'clear_alert': Clear specific alert
        """
        task_type = task.get('type')
        
        if task_type == 'health_check':
            return await self._perform_health_check()
        elif task_type == 'get_alerts':
            return await self._get_active_alerts()
        elif task_type == 'get_metrics':
            return await self._get_system_metrics()
        elif task_type == 'optimize_system':
            return await self._analyze_optimizations()
        elif task_type == 'clear_alert':
            return await self._clear_alert(task.get('alert_id'))
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    async def _background_monitoring(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                await self._perform_health_check()
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Background monitoring error: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive system health check"""
        try:
            # Collect system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network_io = psutil.net_io_counters()
            boot_time = psutil.boot_time()
            uptime = time.time() - boot_time
            
            # Create health snapshot
            health_metrics = SystemHealthMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_usage_percent=disk.percent,
                network_io={
                    'bytes_sent': network_io.bytes_sent,
                    'bytes_recv': network_io.bytes_recv,
                    'packets_sent': network_io.packets_sent,
                    'packets_recv': network_io.packets_recv
                },
                active_processes=len(psutil.pids()),
                load_average=psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0,
                uptime_seconds=uptime
            )
            
            # Store in history
            self.health_history.append(health_metrics)
            if len(self.health_history) > 100:  # Keep last 100 snapshots
                self.health_history.pop(0)
            
            # Check for alerts
            new_alerts = self._check_thresholds(health_metrics)
            self.active_alerts.extend(new_alerts)
            
            # Update tracking
            self.checks_performed += 1
            self.last_health_check = datetime.now()
            
            return {
                'timestamp': health_metrics.timestamp.isoformat(),
                'health_status': self._determine_health_status(health_metrics),
                'metrics': {
                    'cpu_percent': health_metrics.cpu_percent,
                    'memory_percent': health_metrics.memory_percent,
                    'disk_usage_percent': health_metrics.disk_usage_percent,
                    'active_processes': health_metrics.active_processes,
                    'uptime_hours': health_metrics.uptime_seconds / 3600
                },
                'alerts_generated': len(new_alerts),
                'total_alerts': len(self.active_alerts)
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _check_thresholds(self, metrics: SystemHealthMetrics) -> List[Alert]:
        """Check metrics against thresholds and generate alerts"""
        alerts = []
        
        # CPU checks
        if metrics.cpu_percent > self.thresholds['cpu_critical']:
            alerts.append(Alert(
                timestamp=datetime.now(),
                severity='critical',
                category='cpu',
                message=f'CPU usage critically high: {metrics.cpu_percent:.1f}%',
                value=metrics.cpu_percent,
                threshold=self.thresholds['cpu_critical'],
                recommendation='Consider reducing active processes or upgrading hardware'
            ))
        elif metrics.cpu_percent > self.thresholds['cpu_warning']:
            alerts.append(Alert(
                timestamp=datetime.now(),
                severity='warning',
                category='cpu',
                message=f'CPU usage high: {metrics.cpu_percent:.1f}%',
                value=metrics.cpu_percent,
                threshold=self.thresholds['cpu_warning'],
                recommendation='Monitor closely, consider optimizing processes'
            ))
        
        # Memory checks
        if metrics.memory_percent > self.thresholds['memory_critical']:
            alerts.append(Alert(
                timestamp=datetime.now(),
                severity='critical',
                category='memory',
                message=f'Memory usage critically high: {metrics.memory_percent:.1f}%',
                value=metrics.memory_percent,
                threshold=self.thresholds['memory_critical'],
                recommendation='Free memory immediately or add more RAM'
            ))
        elif metrics.memory_percent > self.thresholds['memory_warning']:
            alerts.append(Alert(
                timestamp=datetime.now(),
                severity='warning',
                category='memory',
                message=f'Memory usage high: {metrics.memory_percent:.1f}%',
                value=metrics.memory_percent,
                threshold=self.thresholds['memory_warning'],
                recommendation='Monitor memory usage, consider cleanup'
            ))
        
        # Disk checks
        if metrics.disk_usage_percent > self.thresholds['disk_critical']:
            alerts.append(Alert(
                timestamp=datetime.now(),
                severity='critical',
                category='disk',
                message=f'Disk usage critically high: {metrics.disk_usage_percent:.1f}%',
                value=metrics.disk_usage_percent,
                threshold=self.thresholds['disk_critical'],
                recommendation='Free disk space immediately'
            ))
        elif metrics.disk_usage_percent > self.thresholds['disk_warning']:
            alerts.append(Alert(
                timestamp=datetime.now(),
                severity='warning',
                category='disk',
                message=f'Disk usage high: {metrics.disk_usage_percent:.1f}%',
                value=metrics.disk_usage_percent,
                threshold=self.thresholds['disk_warning'],
                recommendation='Monitor disk usage, consider cleanup'
            ))
        
        self.alerts_generated += len(alerts)
        return alerts
    
    def _determine_health_status(self, metrics: SystemHealthMetrics) -> str:
        """Determine overall system health status"""
        if (metrics.cpu_percent > self.thresholds['cpu_critical'] or
            metrics.memory_percent > self.thresholds['memory_critical'] or
            metrics.disk_usage_percent > self.thresholds['disk_critical']):
            return 'critical'
        elif (metrics.cpu_percent > self.thresholds['cpu_warning'] or
              metrics.memory_percent > self.thresholds['memory_warning'] or
              metrics.disk_usage_percent > self.thresholds['disk_warning']):
            return 'warning'
        else:
            return 'healthy'
    
    async def _get_active_alerts(self) -> Dict[str, Any]:
        """Get all active alerts"""
        # Clean old alerts (older than 1 hour)
        cutoff_time = datetime.now() - timedelta(hours=1)
        self.active_alerts = [alert for alert in self.active_alerts if alert.timestamp > cutoff_time]
        
        return {
            'total_alerts': len(self.active_alerts),
            'alerts': [
                {
                    'timestamp': alert.timestamp.isoformat(),
                    'severity': alert.severity,
                    'category': alert.category,
                    'message': alert.message,
                    'value': alert.value,
                    'threshold': alert.threshold,
                    'recommendation': alert.recommendation
                }
                for alert in self.active_alerts
            ],
            'severity_counts': {
                'critical': len([a for a in self.active_alerts if a.severity == 'critical']),
                'warning': len([a for a in self.active_alerts if a.severity == 'warning']),
                'info': len([a for a in self.active_alerts if a.severity == 'info'])
            }
        }
    
    async def _get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        if not self.health_history:
            return {'error': 'No health data available'}
        
        latest = self.health_history[-1]
        
        # Calculate trends (last 10 measurements)
        recent_metrics = self.health_history[-10:] if len(self.health_history) >= 10 else self.health_history
        
        cpu_trend = self._calculate_trend([m.cpu_percent for m in recent_metrics])
        memory_trend = self._calculate_trend([m.memory_percent for m in recent_metrics])
        
        return {
            'current': {
                'timestamp': latest.timestamp.isoformat(),
                'cpu_percent': latest.cpu_percent,
                'memory_percent': latest.memory_percent,
                'disk_usage_percent': latest.disk_usage_percent,
                'active_processes': latest.active_processes,
                'uptime_hours': latest.uptime_seconds / 3600
            },
            'trends': {
                'cpu_trend': cpu_trend,
                'memory_trend': memory_trend
            },
            'history_length': len(self.health_history),
            'checks_performed': self.checks_performed,
            'alerts_generated': self.alerts_generated
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from values"""
        if len(values) < 2:
            return 'stable'
        
        # Simple linear trend
        first_half = sum(values[:len(values)//2]) / (len(values)//2)
        second_half = sum(values[len(values)//2:]) / (len(values) - len(values)//2)
        
        if second_half > first_half * 1.05:  # 5% increase
            return 'increasing'
        elif second_half < first_half * 0.95:  # 5% decrease
            return 'decreasing'
        else:
            return 'stable'
    
    async def _analyze_optimizations(self) -> Dict[str, Any]:
        """Analyze system and suggest optimizations"""
        if not self.health_history:
            return {'error': 'Insufficient data for analysis'}
        
        latest = self.health_history[-1]
        optimizations = []
        
        # CPU optimization
        if latest.cpu_percent > 50:
            optimizations.append({
                'category': 'cpu',
                'priority': 'high' if latest.cpu_percent > 70 else 'medium',
                'issue': f'High CPU usage: {latest.cpu_percent:.1f}%',
                'recommendation': 'Consider reducing background processes or upgrading CPU',
                'impact': 'Performance degradation'
            })
        
        # Memory optimization
        if latest.memory_percent > 60:
            optimizations.append({
                'category': 'memory',
                'priority': 'high' if latest.memory_percent > 80 else 'medium',
                'issue': f'High memory usage: {latest.memory_percent:.1f}%',
                'recommendation': 'Consider memory cleanup or adding more RAM',
                'impact': 'Potential system slowdown'
            })
        
        # Process optimization
        if latest.active_processes > 200:
            optimizations.append({
                'category': 'processes',
                'priority': 'medium',
                'issue': f'High process count: {latest.active_processes}',
                'recommendation': 'Review and terminate unnecessary processes',
                'impact': 'Resource contention'
            })
        
        self.optimizations_suggested += len(optimizations)
        
        return {
            'total_optimizations': len(optimizations),
            'optimizations': optimizations,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    async def _clear_alert(self, alert_id: str) -> Dict[str, Any]:
        """Clear specific alert"""
        # Simple implementation - clear by message content
        original_count = len(self.active_alerts)
        self.active_alerts = [alert for alert in self.active_alerts if alert.message != alert_id]
        cleared_count = original_count - len(self.active_alerts)
        
        return {
            'cleared': cleared_count > 0,
            'alerts_remaining': len(self.active_alerts)
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status with monitoring-specific metrics"""
        base_status = super().get_status()
        
        # Add monitoring-specific metrics
        base_status['monitoring_metrics'] = {
            'monitoring_active': self.monitoring_active,
            'checks_performed': self.checks_performed,
            'alerts_generated': self.alerts_generated,
            'active_alerts': len(self.active_alerts),
            'optimizations_suggested': self.optimizations_suggested,
            'last_health_check': self.last_health_check.isoformat() if self.last_health_check else None,
            'health_history_length': len(self.health_history)
        }
        
        return base_status
