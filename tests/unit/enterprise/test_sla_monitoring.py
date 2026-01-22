"""
Comprehensive Unit Tests for Enterprise-Grade SLA Monitoring System

This module contains comprehensive unit tests for the SLA monitoring,
performance tracking, availability management, alert management,
SLA analytics, and SLA dashboard systems.

Author: AI Trading System
Version: 1.0.0
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import json
import sqlite3
import tempfile
import os
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from enterprise.monitoring.sla_monitor import (
    SLAMonitor, SLA, SLAViolation, SLAAlert, SLAReport,
    SLAStatus, SLAViolationType, AlertSeverity, SLAAlertType
)
from enterprise.monitoring.performance_tracker import (
    PerformanceTracker, PerformanceMetric, PerformanceAlert, PerformanceReport,
    MetricType, PerformanceLevel, AlertType
)
from enterprise.monitoring.availability_manager import (
    AvailabilityManager, AvailabilityMetric, DowntimeEvent, AvailabilityReport,
    AvailabilityStatus, DowntimeReason, AvailabilityLevel
)
from enterprise.monitoring.alert_manager import (
    AlertManager, Alert, AlertRule, AlertChannel, AlertEscalation,
    AlertSeverity, AlertStatus, AlertChannel, EscalationLevel
)
from enterprise.monitoring.sla_analytics import (
    SLAAnalytics, SLATrend, SLAForecast, SLAInsight,
    TrendDirection, ForecastAccuracy, InsightType
)
from enterprise.monitoring.sla_dashboard import (
    SLADashboard, SLAWidget, SLAChart, SLADashboardConfig,
    WidgetType, ChartType, DashboardLayout
)

class TestSLAMonitor:
    """Test cases for SLA monitoring system."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        os.unlink(db_path)
    
    @pytest.fixture
    def sla_monitor(self, temp_db):
        """Create SLA monitor instance."""
        return SLAMonitor(db_path=temp_db)
    
    def test_sla_monitor_initialization(self, sla_monitor):
        """Test SLA monitor initialization."""
        assert sla_monitor is not None
        assert sla_monitor.db_path is not None
        assert len(sla_monitor.slas) > 0  # Should have default SLAs
        assert len(sla_monitor.violations) == 0
        assert len(sla_monitor.alerts) == 0
        assert sla_monitor.monitoring_active == False
    
    def test_add_sla(self, sla_monitor):
        """Test adding SLA definition."""
        sla = SLA(
            sla_id="TEST_SLA_001",
            name="Test SLA",
            description="Test SLA for unit testing",
            service="test_service",
            metric="test_metric",
            target_value=99.5,
            measurement_period=5,
            evaluation_window=24,
            uptime_target=99.5,
            response_time_target=100.0,
            throughput_target=1000.0,
            error_rate_target=0.5,
            availability_target=99.5,
            business_impact="Test impact",
            escalation_policy="test_policy"
        )
        
        sla_monitor.add_sla(sla)
        
        assert "TEST_SLA_001" in sla_monitor.slas
        assert sla_monitor.slas["TEST_SLA_001"].name == "Test SLA"
    
    def test_start_monitoring(self, sla_monitor):
        """Test starting SLA monitoring."""
        sla_monitor.start_monitoring()
        
        assert sla_monitor.monitoring_active == True
        assert sla_monitor.monitoring_thread is not None
        
        # Stop monitoring
        sla_monitor.stop_monitoring()
        assert sla_monitor.monitoring_active == False
    
    def test_get_sla_status(self, sla_monitor):
        """Test getting SLA status."""
        # Get status for default SLA
        sla_id = list(sla_monitor.slas.keys())[0]
        status = sla_monitor.get_sla_status(sla_id)
        
        assert status is not None
        assert 'sla_id' in status
        assert 'current_status' in status
        assert 'current_values' in status
        assert 'target_values' in status
    
    def test_generate_sla_report(self, sla_monitor):
        """Test generating SLA report."""
        sla_id = list(sla_monitor.slas.keys())[0]
        start_date = datetime.now() - timedelta(days=1)
        end_date = datetime.now()
        
        report = sla_monitor.generate_sla_report(sla_id, start_date, end_date)
        
        assert report is not None
        assert report.sla_id == sla_id
        assert report.report_period_start == start_date
        assert report.report_period_end == end_date
        assert report.uptime_percentage >= 0.0
        assert report.uptime_percentage <= 100.0
        assert report.sla_status in [SLAStatus.EXCELLENT, SLAStatus.GOOD, SLAStatus.WARNING, SLAStatus.CRITICAL]
    
    def test_get_sla_summary(self, sla_monitor):
        """Test getting SLA summary."""
        start_date = datetime.now() - timedelta(days=1)
        end_date = datetime.now()
        
        summary = sla_monitor.get_sla_summary(start_date, end_date)
        
        assert summary is not None
        assert 'summary' in summary
        assert 'sla_status_breakdown' in summary
        assert 'violation_breakdown' in summary
        assert 'alert_breakdown' in summary
        assert summary['summary']['total_slas'] > 0

class TestPerformanceTracker:
    """Test cases for performance tracking system."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        os.unlink(db_path)
    
    @pytest.fixture
    def performance_tracker(self, temp_db):
        """Create performance tracker instance."""
        return PerformanceTracker(db_path=temp_db)
    
    def test_performance_tracker_initialization(self, performance_tracker):
        """Test performance tracker initialization."""
        assert performance_tracker is not None
        assert performance_tracker.db_path is not None
        assert len(performance_tracker.metrics) == 0
        assert len(performance_tracker.alerts) == 0
        assert performance_tracker.tracking_active == False
    
    def test_start_tracking(self, performance_tracker):
        """Test starting performance tracking."""
        performance_tracker.start_tracking()
        
        assert performance_tracker.tracking_active == True
        assert performance_tracker.tracking_thread is not None
        
        # Stop tracking
        performance_tracker.stop_tracking()
        assert performance_tracker.tracking_active == False
    
    def test_get_performance_summary(self, performance_tracker):
        """Test getting performance summary."""
        summary = performance_tracker.get_performance_summary()
        
        assert summary is not None
        assert 'summary' in summary
        assert 'metrics_summary' in summary
        assert 'performance_level' in summary
        assert 'recent_alerts' in summary
        assert summary['summary']['tracking_active'] == False
    
    def test_generate_performance_report(self, performance_tracker):
        """Test generating performance report."""
        component = "test_component"
        start_date = datetime.now() - timedelta(days=1)
        end_date = datetime.now()
        
        report = performance_tracker.generate_performance_report(component, start_date, end_date)
        
        assert report is not None
        assert report.component == component
        assert report.report_period_start == start_date
        assert report.report_period_end == end_date
        assert report.performance_level in [PerformanceLevel.EXCELLENT, PerformanceLevel.GOOD, 
                                          PerformanceLevel.FAIR, PerformanceLevel.POOR, PerformanceLevel.CRITICAL]
    
    def test_acknowledge_alert(self, performance_tracker):
        """Test acknowledging performance alert."""
        # Create a test alert
        alert = PerformanceAlert(
            alert_id="TEST_ALERT_001",
            metric_type=MetricType.CPU_USAGE,
            component="test_component",
            instance="test_instance",
            alert_type=AlertType.THRESHOLD_EXCEEDED,
            severity="WARNING",
            title="Test Alert",
            description="Test alert description",
            current_value=85.0,
            threshold_value=80.0,
            deviation_percentage=6.25,
            timestamp=datetime.now()
        )
        performance_tracker.alerts.append(alert)
        
        result = performance_tracker.acknowledge_alert("TEST_ALERT_001")
        assert result == True
        assert alert.acknowledged == True
    
    def test_resolve_alert(self, performance_tracker):
        """Test resolving performance alert."""
        # Create a test alert
        alert = PerformanceAlert(
            alert_id="TEST_ALERT_002",
            metric_type=MetricType.MEMORY_USAGE,
            component="test_component",
            instance="test_instance",
            alert_type=AlertType.THRESHOLD_EXCEEDED,
            severity="CRITICAL",
            title="Test Alert",
            description="Test alert description",
            current_value=95.0,
            threshold_value=90.0,
            deviation_percentage=5.56,
            timestamp=datetime.now()
        )
        performance_tracker.alerts.append(alert)
        
        result = performance_tracker.resolve_alert("TEST_ALERT_002")
        assert result == True
        assert alert.resolved == True

class TestAvailabilityManager:
    """Test cases for availability management system."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        os.unlink(db_path)
    
    @pytest.fixture
    def availability_manager(self, temp_db):
        """Create availability manager instance."""
        return AvailabilityManager(db_path=temp_db)
    
    def test_availability_manager_initialization(self, availability_manager):
        """Test availability manager initialization."""
        assert availability_manager is not None
        assert availability_manager.db_path is not None
        assert len(availability_manager.availability_metrics) == 0
        assert len(availability_manager.downtime_events) == 0
        assert availability_manager.monitoring_active == False
    
    def test_start_monitoring(self, availability_manager):
        """Test starting availability monitoring."""
        availability_manager.start_monitoring()
        
        assert availability_manager.monitoring_active == True
        assert availability_manager.monitoring_thread is not None
        
        # Stop monitoring
        availability_manager.stop_monitoring()
        assert availability_manager.monitoring_active == False
    
    def test_get_availability_summary(self, availability_manager):
        """Test getting availability summary."""
        summary = availability_manager.get_availability_summary()
        
        assert summary is not None
        assert 'summary' in summary
        assert 'services' in summary
        assert 'downtime_events' in summary
        assert summary['summary']['monitoring_active'] == False
    
    def test_generate_availability_report(self, availability_manager):
        """Test generating availability report."""
        service = "test_service"
        start_date = datetime.now() - timedelta(days=1)
        end_date = datetime.now()
        
        report = availability_manager.generate_availability_report(service, start_date, end_date)
        
        assert report is not None
        assert report.service == service
        assert report.report_period_start == start_date
        assert report.report_period_end == end_date
        assert report.uptime_percentage >= 0.0
        assert report.uptime_percentage <= 100.0
        assert report.availability_level in [AvailabilityLevel.EXCELLENT, AvailabilityLevel.GOOD,
                                           AvailabilityLevel.FAIR, AvailabilityLevel.POOR, AvailabilityLevel.CRITICAL]
    
    def test_get_service_health(self, availability_manager):
        """Test getting service health."""
        service = "trading_system"
        health = availability_manager.get_service_health(service)
        
        assert health is not None
        assert 'service' in health
        assert 'status' in health
        assert 'endpoints' in health
        assert health['service'] == service

class TestAlertManager:
    """Test cases for alert management system."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        os.unlink(db_path)
    
    @pytest.fixture
    def alert_manager(self, temp_db):
        """Create alert manager instance."""
        return AlertManager(db_path=temp_db)
    
    def test_alert_manager_initialization(self, alert_manager):
        """Test alert manager initialization."""
        assert alert_manager is not None
        assert alert_manager.db_path is not None
        assert len(alert_manager.alert_rules) > 0  # Should have default rules
        assert len(alert_manager.alert_channels) > 0  # Should have default channels
        assert len(alert_manager.alerts) == 0
        assert alert_manager.processing_active == False
    
    def test_add_alert_rule(self, alert_manager):
        """Test adding alert rule."""
        rule = AlertRule(
            rule_id="TEST_RULE_001",
            name="Test Rule",
            description="Test alert rule",
            condition="test_condition > 100",
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.EMAIL],
            escalation_policy="default",
            suppression_rules=[]
        )
        
        alert_manager.add_alert_rule(rule)
        
        assert "TEST_RULE_001" in alert_manager.alert_rules
        assert alert_manager.alert_rules["TEST_RULE_001"].name == "Test Rule"
    
    def test_add_alert_channel(self, alert_manager):
        """Test adding alert channel."""
        channel = AlertChannel(
            channel_id="TEST_CHANNEL_001",
            channel_type=AlertChannel.EMAIL,
            name="Test Email Channel",
            configuration={
                "smtp_server": "test.smtp.com",
                "smtp_port": 587,
                "username": "test@test.com",
                "password": "password"
            }
        )
        
        alert_manager.add_alert_channel(channel)
        
        assert "TEST_CHANNEL_001" in alert_manager.alert_channels
        assert alert_manager.alert_channels["TEST_CHANNEL_001"].name == "Test Email Channel"
    
    def test_start_processing(self, alert_manager):
        """Test starting alert processing."""
        alert_manager.start_processing()
        
        assert alert_manager.processing_active == True
        assert alert_manager.processing_thread is not None
        
        # Stop processing
        alert_manager.stop_processing()
        assert alert_manager.processing_active == False
    
    def test_create_alert(self, alert_manager):
        """Test creating alert."""
        rule_id = list(alert_manager.alert_rules.keys())[0]
        
        alert = alert_manager.create_alert(
            rule_id=rule_id,
            title="Test Alert",
            description="Test alert description",
            source="test_source",
            component="test_component",
            metric="test_metric",
            current_value=95.0,
            threshold_value=90.0
        )
        
        assert alert is not None
        assert alert.rule_id == rule_id
        assert alert.title == "Test Alert"
        assert alert.severity in [AlertSeverity.INFO, AlertSeverity.WARNING, AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]
        assert len(alert_manager.alerts) > 0
    
    def test_acknowledge_alert(self, alert_manager):
        """Test acknowledging alert."""
        # Create a test alert
        alert = Alert(
            alert_id="TEST_ALERT_001",
            rule_id="TEST_RULE_001",
            title="Test Alert",
            description="Test alert description",
            severity=AlertSeverity.WARNING,
            status=AlertStatus.NEW,
            source="test_source",
            component="test_component",
            metric="test_metric",
            current_value=95.0,
            threshold_value=90.0,
            deviation_percentage=5.56,
            timestamp=datetime.now()
        )
        alert_manager.alerts.append(alert)
        
        result = alert_manager.acknowledge_alert("TEST_ALERT_001", "test_user")
        assert result == True
        assert alert.status == AlertStatus.ACKNOWLEDGED
        assert alert.acknowledged_by == "test_user"
    
    def test_resolve_alert(self, alert_manager):
        """Test resolving alert."""
        # Create a test alert
        alert = Alert(
            alert_id="TEST_ALERT_002",
            rule_id="TEST_RULE_001",
            title="Test Alert",
            description="Test alert description",
            severity=AlertSeverity.CRITICAL,
            status=AlertStatus.ACKNOWLEDGED,
            source="test_source",
            component="test_component",
            metric="test_metric",
            current_value=95.0,
            threshold_value=90.0,
            deviation_percentage=5.56,
            timestamp=datetime.now()
        )
        alert_manager.alerts.append(alert)
        
        result = alert_manager.resolve_alert("TEST_ALERT_002", "test_user")
        assert result == True
        assert alert.status == AlertStatus.RESOLVED
        assert alert.resolved_by == "test_user"
    
    def test_get_alert_summary(self, alert_manager):
        """Test getting alert summary."""
        summary = alert_manager.get_alert_summary()
        
        assert summary is not None
        assert 'summary' in summary
        assert 'severity_breakdown' in summary
        assert 'source_breakdown' in summary
        assert 'component_breakdown' in summary
        assert 'recent_alerts' in summary
        assert summary['summary']['processing_active'] == False

class TestSLAAnalytics:
    """Test cases for SLA analytics system."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        os.unlink(db_path)
    
    @pytest.fixture
    def sla_analytics(self, temp_db):
        """Create SLA analytics instance."""
        return SLAAnalytics(db_path=temp_db)
    
    def test_sla_analytics_initialization(self, sla_analytics):
        """Test SLA analytics initialization."""
        assert sla_analytics is not None
        assert sla_analytics.db_path is not None
        assert len(sla_analytics.trends) == 0
        assert len(sla_analytics.forecasts) == 0
        assert len(sla_analytics.insights) == 0
    
    def test_analyze_trends(self, sla_analytics):
        """Test analyzing SLA trends."""
        sla_id = "TEST_SLA_001"
        metric_type = "uptime"
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()
        
        trend = sla_analytics.analyze_trends(sla_id, metric_type, start_date, end_date)
        
        assert trend is not None
        assert trend.sla_id == sla_id
        assert trend.metric_type == metric_type
        assert trend.trend_direction in [TrendDirection.IMPROVING, TrendDirection.STABLE, TrendDirection.DEGRADING]
        assert 0.0 <= trend.confidence <= 1.0
        assert trend.data_points > 0
    
    def test_generate_forecast(self, sla_analytics):
        """Test generating SLA forecast."""
        sla_id = "TEST_SLA_001"
        metric_type = "response_time"
        forecast_days = 7
        
        forecast = sla_analytics.generate_forecast(sla_id, metric_type, forecast_days)
        
        assert forecast is not None
        assert forecast.sla_id == sla_id
        assert forecast.metric_type == metric_type
        assert forecast.forecast_period == forecast_days
        assert len(forecast.forecast_values) == forecast_days
        assert len(forecast.confidence_intervals) == forecast_days
        assert forecast.accuracy in [ForecastAccuracy.HIGH, ForecastAccuracy.MEDIUM, ForecastAccuracy.LOW]
    
    def test_detect_anomalies(self, sla_analytics):
        """Test detecting anomalies."""
        sla_id = "TEST_SLA_001"
        metric_type = "error_rate"
        start_date = datetime.now() - timedelta(days=7)
        end_date = datetime.now()
        
        anomalies = sla_analytics.detect_anomalies(sla_id, metric_type, start_date, end_date)
        
        assert isinstance(anomalies, list)
        # Anomalies may or may not be detected depending on simulated data
        for anomaly in anomalies:
            assert 'timestamp' in anomaly
            assert 'value' in anomaly
            assert 'z_score' in anomaly
            assert 'severity' in anomaly
            assert 'description' in anomaly
    
    def test_analyze_seasonality(self, sla_analytics):
        """Test analyzing seasonality."""
        sla_id = "TEST_SLA_001"
        metric_type = "throughput"
        start_date = datetime.now() - timedelta(days=14)
        end_date = datetime.now()
        
        seasonality = sla_analytics.analyze_seasonality(sla_id, metric_type, start_date, end_date)
        
        assert isinstance(seasonality, dict)
        # Seasonality analysis may return empty dict if insufficient data
        if seasonality:
            assert 'has_daily_seasonality' in seasonality
            assert 'has_hourly_seasonality' in seasonality
            assert 'daily_patterns' in seasonality
            assert 'hourly_patterns' in seasonality
    
    def test_generate_insights(self, sla_analytics):
        """Test generating SLA insights."""
        sla_id = "TEST_SLA_001"
        start_date = datetime.now() - timedelta(days=7)
        end_date = datetime.now()
        
        insights = sla_analytics.generate_insights(sla_id, start_date, end_date)
        
        assert isinstance(insights, list)
        # Insights may or may not be generated depending on simulated data
        for insight in insights:
            assert insight.sla_id == sla_id
            assert insight.insight_type in [InsightType.PERFORMANCE_TREND, InsightType.CAPACITY_PLANNING,
                                         InsightType.ANOMALY_DETECTION, InsightType.SEASONALITY]
            assert 0.0 <= insight.confidence <= 1.0
            assert len(insight.recommendations) > 0
    
    def test_get_analytics_summary(self, sla_analytics):
        """Test getting analytics summary."""
        summary = sla_analytics.get_analytics_summary()
        
        assert summary is not None
        assert 'summary' in summary
        assert 'trend_breakdown' in summary
        assert 'insight_breakdown' in summary
        assert 'forecast_accuracy_breakdown' in summary
        assert 'recent_insights' in summary

class TestSLADashboard:
    """Test cases for SLA dashboard system."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        os.unlink(db_path)
    
    @pytest.fixture
    def sla_dashboard(self, temp_db):
        """Create SLA dashboard instance."""
        return SLADashboard(db_path=temp_db)
    
    def test_sla_dashboard_initialization(self, sla_dashboard):
        """Test SLA dashboard initialization."""
        assert sla_dashboard is not None
        assert sla_dashboard.db_path is not None
        assert len(sla_dashboard.dashboards) > 0  # Should have default dashboard
        assert len(sla_dashboard.widgets) > 0  # Should have default widgets
        assert len(sla_dashboard.charts) == 0
    
    def test_create_dashboard(self, sla_dashboard):
        """Test creating dashboard."""
        widgets = [
            SLAWidget(
                widget_id="TEST_WIDGET_001",
                widget_type=WidgetType.METRIC_CARD,
                title="Test Widget",
                description="Test widget description",
                position=(0, 0),
                size=(2, 1),
                data_source="test_data_source",
                configuration={"test": "config"},
                refresh_interval=30
            )
        ]
        
        dashboard_config = SLADashboardConfig(
            config_id="TEST_DASHBOARD_001",
            name="Test Dashboard",
            description="Test dashboard description",
            layout=DashboardLayout.GRID,
            widgets=widgets,
            refresh_interval=30
        )
        
        sla_dashboard.create_dashboard(dashboard_config)
        
        assert "TEST_DASHBOARD_001" in sla_dashboard.dashboards
        assert sla_dashboard.dashboards["TEST_DASHBOARD_001"].name == "Test Dashboard"
        assert "TEST_WIDGET_001" in sla_dashboard.widgets
    
    def test_add_widget(self, sla_dashboard):
        """Test adding widget."""
        widget = SLAWidget(
            widget_id="TEST_WIDGET_002",
            widget_type=WidgetType.LINE_CHART,
            title="Test Line Chart",
            description="Test line chart widget",
            position=(1, 0),
            size=(4, 2),
            data_source="test_line_data",
            configuration={"chart_type": "line"},
            refresh_interval=60
        )
        
        sla_dashboard.add_widget(widget)
        
        assert "TEST_WIDGET_002" in sla_dashboard.widgets
        assert sla_dashboard.widgets["TEST_WIDGET_002"].title == "Test Line Chart"
    
    def test_create_chart(self, sla_dashboard):
        """Test creating chart."""
        chart_data = [
            {"timestamp": "2023-01-01", "value": 100},
            {"timestamp": "2023-01-02", "value": 105},
            {"timestamp": "2023-01-03", "value": 98}
        ]
        
        chart = SLAChart(
            chart_id="TEST_CHART_001",
            chart_type=ChartType.LINE,
            title="Test Chart",
            data=chart_data,
            x_axis="timestamp",
            y_axis="value",
            configuration={"color": "blue"}
        )
        
        chart_id = sla_dashboard.create_chart(chart)
        
        assert chart_id == "TEST_CHART_001"
        assert "TEST_CHART_001" in sla_dashboard.charts
        assert sla_dashboard.charts["TEST_CHART_001"].title == "Test Chart"
    
    def test_generate_widget_data(self, sla_dashboard):
        """Test generating widget data."""
        widget = SLAWidget(
            widget_id="TEST_WIDGET_003",
            widget_type=WidgetType.METRIC_CARD,
            title="Test Metric Card",
            description="Test metric card widget",
            position=(0, 0),
            size=(2, 1),
            data_source="sla_summary",
            configuration={"metric": "uptime_percentage"},
            refresh_interval=30
        )
        
        data = sla_dashboard.generate_widget_data(widget)
        
        assert data is not None
        assert 'current_value' in data
        assert 'target_value' in data
        assert 'status' in data
        assert 'last_updated' in data
    
    def test_get_dashboard_data(self, sla_dashboard):
        """Test getting dashboard data."""
        dashboard_id = list(sla_dashboard.dashboards.keys())[0]
        data = sla_dashboard.get_dashboard_data(dashboard_id)
        
        assert data is not None
        assert 'dashboard' in data
        assert 'widgets' in data
        assert 'last_updated' in data
        assert data['dashboard']['config_id'] == dashboard_id
    
    def test_export_dashboard(self, sla_dashboard):
        """Test exporting dashboard."""
        dashboard_id = list(sla_dashboard.dashboards.keys())[0]
        export_data = sla_dashboard.export_dashboard(dashboard_id, "JSON")
        
        assert export_data is not None
        assert 'dashboard' in export_data
        assert 'exported_at' in export_data
        assert export_data['dashboard']['config_id'] == dashboard_id
    
    def test_get_dashboard_summary(self, sla_dashboard):
        """Test getting dashboard summary."""
        summary = sla_dashboard.get_dashboard_summary()
        
        assert summary is not None
        assert 'summary' in summary
        assert 'dashboards' in summary
        assert 'widget_types' in summary
        assert 'chart_types' in summary
        assert summary['summary']['total_dashboards'] > 0

class TestIntegration:
    """Integration tests for the SLA monitoring system."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        os.unlink(db_path)
    
    def test_end_to_end_sla_monitoring_workflow(self, temp_db):
        """Test complete SLA monitoring workflow."""
        # Initialize all components
        sla_monitor = SLAMonitor(db_path=temp_db)
        performance_tracker = PerformanceTracker(db_path=temp_db)
        availability_manager = AvailabilityManager(db_path=temp_db)
        alert_manager = AlertManager(db_path=temp_db)
        sla_analytics = SLAAnalytics(db_path=temp_db)
        sla_dashboard = SLADashboard(db_path=temp_db)
        
        # 1. Start monitoring
        sla_monitor.start_monitoring()
        performance_tracker.start_tracking()
        availability_manager.start_monitoring()
        alert_manager.start_processing()
        
        # 2. Get SLA status
        sla_id = list(sla_monitor.slas.keys())[0]
        sla_status = sla_monitor.get_sla_status(sla_id)
        
        # 3. Get performance summary
        performance_summary = performance_tracker.get_performance_summary()
        
        # 4. Get availability summary
        availability_summary = availability_manager.get_availability_summary()
        
        # 5. Create alert
        rule_id = list(alert_manager.alert_rules.keys())[0]
        alert = alert_manager.create_alert(
            rule_id=rule_id,
            title="Integration Test Alert",
            description="Test alert for integration testing",
            source="integration_test",
            component="test_component",
            metric="test_metric",
            current_value=95.0,
            threshold_value=90.0
        )
        
        # 6. Analyze trends
        start_date = datetime.now() - timedelta(days=7)
        end_date = datetime.now()
        trend = sla_analytics.analyze_trends(sla_id, "uptime", start_date, end_date)
        
        # 7. Generate insights
        insights = sla_analytics.generate_insights(sla_id, start_date, end_date)
        
        # 8. Get dashboard data
        dashboard_id = list(sla_dashboard.dashboards.keys())[0]
        dashboard_data = sla_dashboard.get_dashboard_data(dashboard_id)
        
        # 9. Generate reports
        sla_report = sla_monitor.generate_sla_report(sla_id, start_date, end_date)
        performance_report = performance_tracker.generate_performance_report("test_component", start_date, end_date)
        availability_report = availability_manager.generate_availability_report("trading_system", start_date, end_date)
        
        # 10. Stop monitoring
        sla_monitor.stop_monitoring()
        performance_tracker.stop_tracking()
        availability_manager.stop_monitoring()
        alert_manager.stop_processing()
        
        # Verify all components worked together
        assert sla_status is not None
        assert performance_summary is not None
        assert availability_summary is not None
        assert alert is not None
        assert trend is not None
        assert isinstance(insights, list)
        assert dashboard_data is not None
        assert sla_report is not None
        assert performance_report is not None
        assert availability_report is not None
    
    def test_sla_monitoring_integration(self, temp_db):
        """Test SLA monitoring integration with other components."""
        sla_monitor = SLAMonitor(db_path=temp_db)
        alert_manager = AlertManager(db_path=temp_db)
        sla_analytics = SLAAnalytics(db_path=temp_db)
        
        # Start monitoring
        sla_monitor.start_monitoring()
        alert_manager.start_processing()
        
        # Get SLA status
        sla_id = list(sla_monitor.slas.keys())[0]
        sla_status = sla_monitor.get_sla_status(sla_id)
        
        # Create alert based on SLA status
        if sla_status.get('current_status') == 'CRITICAL':
            rule_id = list(alert_manager.alert_rules.keys())[0]
            alert = alert_manager.create_alert(
                rule_id=rule_id,
                title="SLA Critical Alert",
                description="SLA status is critical",
                source="sla_monitor",
                component=sla_id,
                metric="uptime",
                current_value=sla_status.get('current_values', {}).get('uptime', 0),
                threshold_value=95.0
            )
            
            assert alert is not None
            assert alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]
        
        # Analyze trends for the SLA
        start_date = datetime.now() - timedelta(days=7)
        end_date = datetime.now()
        trend = sla_analytics.analyze_trends(sla_id, "uptime", start_date, end_date)
        
        # Generate insights
        insights = sla_analytics.generate_insights(sla_id, start_date, end_date)
        
        # Verify integration
        assert sla_status is not None
        assert trend is not None
        assert isinstance(insights, list)
        
        # Stop monitoring
        sla_monitor.stop_monitoring()
        alert_manager.stop_processing()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
