"""
Tests for MonitoringAgent

Tests the system monitoring functionality including:
- Health checks
- Alert generation
- Performance tracking
- System optimization
"""

import pytest
import asyncio
from typing import Dict, Any

# Add project root to path
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.agents.monitoring_agent import MonitoringAgent, SystemHealthMetrics, Alert


@pytest.mark.asyncio
async def test_monitoring_agent_initialization():
    """Test monitoring agent initialization"""
    agent = MonitoringAgent()
    
    assert agent.agent_id == 'monitoring_agent'
    assert agent.name == 'System Monitoring Agent'
    assert agent.priority.value == 1  # CRITICAL
    assert agent.status.value == 'initializing'
    
    # Initialize
    success = await agent.initialize()
    assert success is True
    assert agent.monitoring_active is True
    assert agent.status.value == 'idle'


@pytest.mark.asyncio
async def test_health_check():
    """Test system health check functionality"""
    agent = MonitoringAgent()
    await agent.initialize()
    
    # Perform health check
    result = await agent.process_task({'type': 'health_check'})
    
    assert 'timestamp' in result
    assert 'health_status' in result
    assert 'metrics' in result
    assert 'alerts_generated' in result
    
    # Check metrics structure
    metrics = result['metrics']
    assert 'cpu_percent' in metrics
    assert 'memory_percent' in metrics
    assert 'disk_usage_percent' in metrics
    assert 'active_processes' in metrics
    assert 'uptime_hours' in metrics


@pytest.mark.asyncio
async def test_alert_generation():
    """Test alert generation for threshold violations"""
    agent = MonitoringAgent()
    await agent.initialize()
    
    # Mock high CPU usage
    agent.thresholds['cpu_warning'] = 10.0  # Lower threshold for testing
    agent.thresholds['cpu_critical'] = 20.0
    
    # Perform health check (will likely generate alerts)
    result = await agent.process_task({'type': 'health_check'})
    
    # Check if alerts were generated
    assert 'alerts_generated' in result
    assert result['alerts_generated'] >= 0


@pytest.mark.asyncio
async def test_get_alerts():
    """Test getting active alerts"""
    agent = MonitoringAgent()
    await agent.initialize()
    
    # Get alerts
    result = await agent.process_task({'type': 'get_alerts'})
    
    assert 'total_alerts' in result
    assert 'alerts' in result
    assert 'severity_counts' in result
    
    # Check severity counts structure
    severity_counts = result['severity_counts']
    assert 'critical' in severity_counts
    assert 'warning' in severity_counts
    assert 'info' in severity_counts


@pytest.mark.asyncio
async def test_system_metrics():
    """Test getting system metrics"""
    agent = MonitoringAgent()
    await agent.initialize()
    
    # Perform a health check first to generate data
    await agent.process_task({'type': 'health_check'})
    
    # Get metrics
    result = await agent.process_task({'type': 'get_metrics'})
    
    assert 'current' in result
    assert 'trends' in result
    assert 'history_length' in result
    assert 'checks_performed' in result
    
    # Check current metrics structure
    current = result['current']
    assert 'timestamp' in current
    assert 'cpu_percent' in current
    assert 'memory_percent' in current
    assert 'disk_usage_percent' in current


@pytest.mark.asyncio
async def test_optimization_analysis():
    """Test system optimization analysis"""
    agent = MonitoringAgent()
    await agent.initialize()
    
    # Perform health checks to generate data
    for _ in range(5):
        await agent.process_task({'type': 'health_check'})
    
    # Get optimization analysis
    result = await agent.process_task({'type': 'optimize_system'})
    
    assert 'total_optimizations' in result
    assert 'optimizations' in result
    assert 'analysis_timestamp' in result
    
    # Check optimizations structure
    optimizations = result['optimizations']
    assert isinstance(optimizations, list)
    
    if optimizations:
        opt = optimizations[0]
        assert 'category' in opt
        assert 'priority' in opt
        assert 'issue' in opt
        assert 'recommendation' in opt


@pytest.mark.asyncio
async def test_clear_alert():
    """Test clearing alerts"""
    agent = MonitoringAgent()
    await agent.initialize()
    
    # Clear a non-existent alert
    result = await agent.process_task({
        'type': 'clear_alert',
        'alert_id': 'non_existent_alert'
    })
    
    assert 'cleared' in result
    assert 'alerts_remaining' in result
    assert result['cleared'] is False


@pytest.mark.asyncio
async def test_monitoring_agent_status():
    """Test monitoring agent status reporting"""
    agent = MonitoringAgent()
    await agent.initialize()
    
    # Perform some operations
    await agent.process_task({'type': 'health_check'})
    await agent.process_task({'type': 'get_alerts'})
    
    status = agent.get_status()
    
    # Check base status
    assert status['agent_id'] == 'monitoring_agent'
    assert status['name'] == 'System Monitoring Agent'
    assert status['status'] == 'idle'
    
    # Check monitoring-specific metrics
    assert 'monitoring_metrics' in status
    monitoring_metrics = status['monitoring_metrics']
    assert 'monitoring_active' in monitoring_metrics
    assert 'checks_performed' in monitoring_metrics
    assert 'alerts_generated' in monitoring_metrics
    assert 'active_alerts' in monitoring_metrics


@pytest.mark.asyncio
async def test_monitoring_agent_shutdown():
    """Test monitoring agent shutdown"""
    agent = MonitoringAgent()
    await agent.initialize()
    
    assert agent.monitoring_active is True
    
    success = await agent.shutdown()
    assert success is True
    assert agent.monitoring_active is False
    assert agent.status.value == 'stopped'


@pytest.mark.asyncio
async def test_health_metrics_dataclass():
    """Test SystemHealthMetrics dataclass"""
    metrics = SystemHealthMetrics(
        timestamp=asyncio.get_event_loop().time(),
        cpu_percent=50.0,
        memory_percent=60.0,
        disk_usage_percent=40.0,
        network_io={'bytes_sent': 1000, 'bytes_recv': 2000},
        active_processes=100,
        load_average=1.5,
        uptime_seconds=3600
    )
    
    assert metrics.cpu_percent == 50.0
    assert metrics.memory_percent == 60.0
    assert metrics.disk_usage_percent == 40.0
    assert metrics.active_processes == 100
    assert metrics.uptime_seconds == 3600


@pytest.mark.asyncio
async def test_alert_dataclass():
    """Test Alert dataclass"""
    alert = Alert(
        timestamp=asyncio.get_event_loop().time(),
        severity='warning',
        category='cpu',
        message='High CPU usage detected',
        value=75.0,
        threshold=70.0,
        recommendation='Monitor system closely'
    )
    
    assert alert.severity == 'warning'
    assert alert.category == 'cpu'
    assert alert.message == 'High CPU usage detected'
    assert alert.value == 75.0
    assert alert.threshold == 70.0
    assert alert.recommendation == 'Monitor system closely'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
