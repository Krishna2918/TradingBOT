#!/usr/bin/env python3
"""
Test script for the monitoring and health management system.
"""

import asyncio
import sys
import os
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, os.path.abspath('.'))

from continuous_data_collection.monitoring import (
    ProgressTracker, HealthMonitor, AlertSystem, MonitoringSystem
)
from continuous_data_collection.core.models import CollectionResult, DataSource


async def test_progress_tracker():
    """Test the ProgressTracker functionality."""
    print("Testing ProgressTracker...")
    
    tracker = ProgressTracker(total_target=100)
    
    # Simulate some collection results
    results = [
        CollectionResult(
            symbol="AAPL",
            success=True,
            source=DataSource.ALPHA_VANTAGE,
            data_quality_score=0.95,
            years_of_data=20.5,
            record_count=5000,
            processing_duration=2.5
        ),
        CollectionResult(
            symbol="GOOGL",
            success=True,
            source=DataSource.YFINANCE,
            data_quality_score=0.88,
            years_of_data=15.2,
            record_count=3800,
            processing_duration=3.1
        ),
        CollectionResult(
            symbol="MSFT",
            success=False,
            source=DataSource.ALPHA_VANTAGE,
            data_quality_score=0.0,
            years_of_data=0.0,
            record_count=0,
            error_message="API rate limit exceeded",
            processing_duration=1.2
        )
    ]
    
    tracker.update_progress(results)
    stats = tracker.get_current_stats()
    
    print(f"  Completed: {stats.completed}")
    print(f"  Failed: {stats.failed}")
    print(f"  Success Rate: {stats.success_rate:.2%}")
    print(f"  Avg Quality: {stats.data_quality_avg:.2f}")
    print(f"  ETA: {stats.eta}")
    
    # Generate detailed report
    report = tracker.generate_progress_report()
    print(f"  Completion %: {report['overview']['completion_percentage']:.1f}%")
    
    print("✓ ProgressTracker test passed\n")


async def test_health_monitor():
    """Test the HealthMonitor functionality."""
    print("Testing HealthMonitor...")
    
    monitor = HealthMonitor(check_interval=5)
    
    # Test resource monitoring
    resources = await monitor.monitor_resources()
    print(f"  CPU Usage: {resources['cpu_percent']:.1f}%")
    print(f"  Memory Usage: {resources['memory_percent']:.1f}%")
    print(f"  Disk Usage: {resources['disk_percent']:.1f}%")
    
    # Test API connectivity
    connectivity = await monitor.check_api_connectivity()
    print(f"  API Connectivity: {connectivity}")
    
    # Test health check
    health_status = await monitor.check_system_health()
    print(f"  System Healthy: {health_status.is_healthy}")
    print(f"  Issues: {health_status.issues}")
    
    print("✓ HealthMonitor test passed\n")


async def test_alert_system():
    """Test the AlertSystem functionality."""
    print("Testing AlertSystem...")
    
    alert_system = AlertSystem()
    
    # Create a mock health status with issues
    from continuous_data_collection.core.models import HealthStatus
    
    health_status = HealthStatus(
        is_healthy=False,
        cpu_usage=95.0,  # High CPU to trigger alert
        memory_usage=85.0,
        disk_usage=70.0,
        collection_rate=25.0,  # Low rate to trigger alert
        error_rate=0.15  # High error rate
    )
    
    # Check health status against rules
    await alert_system.check_health_status(health_status)
    
    # Get active alerts
    active_alerts = alert_system.get_active_alerts()
    print(f"  Active Alerts: {len(active_alerts)}")
    
    for alert in active_alerts:
        print(f"    - {alert.rule_name}: {alert.message} ({alert.severity.value})")
    
    # Get statistics
    stats = alert_system.get_alert_statistics()
    print(f"  Alert Statistics: {stats}")
    
    print("✓ AlertSystem test passed\n")


async def test_monitoring_integration():
    """Test the integrated MonitoringSystem."""
    print("Testing MonitoringSystem integration...")
    
    monitoring = MonitoringSystem(total_target_stocks=1000)
    
    # Start monitoring briefly
    await monitoring.start_monitoring()
    
    # Simulate some progress updates
    results = [
        CollectionResult(
            symbol="TSLA",
            success=True,
            source=DataSource.ALPHA_VANTAGE,
            data_quality_score=0.92,
            years_of_data=12.3,
            record_count=3200,
            processing_duration=2.8
        )
    ]
    
    monitoring.update_progress(results)
    
    # Get comprehensive status
    status = monitoring.get_comprehensive_status()
    print(f"  Monitoring Active: {status['monitoring_active']}")
    print(f"  Progress Completed: {status['progress']['stats']['completed']}")
    print(f"  System Healthy: {status['health']['is_healthy']}")
    print(f"  Active Alerts: {status['alerts']['active_count']}")
    
    # Get performance summary
    summary = monitoring.get_performance_summary()
    print(f"  Performance Summary: {summary}")
    
    # Stop monitoring
    await monitoring.stop_monitoring()
    
    print("✓ MonitoringSystem integration test passed\n")


async def main():
    """Run all monitoring system tests."""
    print("=== Monitoring and Health Management System Tests ===\n")
    
    try:
        await test_progress_tracker()
        await test_health_monitor()
        await test_alert_system()
        await test_monitoring_integration()
        
        print("🎉 All monitoring system tests passed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)