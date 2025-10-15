"""
SLA Dashboard System

This module implements a comprehensive SLA dashboard system with
real-time monitoring, visualization, alerting, and enterprise-grade
dashboard management for SLA monitoring and reporting.

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
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class WidgetType(Enum):
    """Widget types."""
    METRIC_CARD = "METRIC_CARD"
    LINE_CHART = "LINE_CHART"
    BAR_CHART = "BAR_CHART"
    PIE_CHART = "PIE_CHART"
    HEATMAP = "HEATMAP"
    GAUGE = "GAUGE"
    TABLE = "TABLE"
    ALERT_LIST = "ALERT_LIST"

class ChartType(Enum):
    """Chart types."""
    LINE = "LINE"
    BAR = "BAR"
    PIE = "PIE"
    SCATTER = "SCATTER"
    AREA = "AREA"
    HEATMAP = "HEATMAP"

class DashboardLayout(Enum):
    """Dashboard layouts."""
    GRID = "GRID"
    SINGLE_COLUMN = "SINGLE_COLUMN"
    TWO_COLUMN = "TWO_COLUMN"
    THREE_COLUMN = "THREE_COLUMN"
    CUSTOM = "CUSTOM"

@dataclass
class SLAWidget:
    """SLA widget definition."""
    widget_id: str
    widget_type: WidgetType
    title: str
    description: str
    position: Tuple[int, int]  # (row, column)
    size: Tuple[int, int]      # (width, height)
    data_source: str
    configuration: Dict[str, Any]
    refresh_interval: int      # seconds
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class SLAChart:
    """SLA chart definition."""
    chart_id: str
    chart_type: ChartType
    title: str
    data: List[Dict[str, Any]]
    x_axis: str
    y_axis: str
    configuration: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class SLADashboardConfig:
    """SLA dashboard configuration."""
    config_id: str
    name: str
    description: str
    layout: DashboardLayout
    widgets: List[SLAWidget]
    refresh_interval: int      # seconds
    auto_refresh: bool = True
    public_access: bool = False
    created_at: datetime = field(default_factory=datetime.now)

class SLADashboard:
    """
    Comprehensive SLA dashboard system.
    
    Features:
    - Real-time SLA monitoring dashboard
    - Customizable widgets and charts
    - Multiple dashboard layouts
    - Interactive visualizations
    - Alert integration
    - Performance metrics display
    - Trend analysis visualization
    - Export capabilities
    """
    
    def __init__(self, db_path: str = "data/sla_dashboard.db"):
        """
        Initialize SLA dashboard.
        
        Args:
            db_path: Path to SLA dashboard database
        """
        self.db_path = db_path
        self.dashboards: Dict[str, SLADashboardConfig] = {}
        self.widgets: Dict[str, SLAWidget] = {}
        self.charts: Dict[str, SLAChart] = {}
        
        # Dashboard configuration
        self.dashboard_config = {
            'default_refresh_interval': 30,  # seconds
            'max_widgets_per_dashboard': 20,
            'chart_cache_duration': 300,     # seconds
            'export_formats': ['PNG', 'PDF', 'SVG', 'JSON']
        }
        
        # Initialize database
        self._init_database()
        
        # Load default dashboard
        self._load_default_dashboard()
        
        logger.info("SLA Dashboard initialized")
    
    def _init_database(self) -> None:
        """Initialize SLA dashboard database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create dashboards table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dashboards (
                config_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                layout TEXT NOT NULL,
                widgets TEXT NOT NULL,
                refresh_interval INTEGER NOT NULL,
                auto_refresh INTEGER NOT NULL,
                public_access INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        
        # Create widgets table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS widgets (
                widget_id TEXT PRIMARY KEY,
                widget_type TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                position TEXT NOT NULL,
                size TEXT NOT NULL,
                data_source TEXT NOT NULL,
                configuration TEXT NOT NULL,
                refresh_interval INTEGER NOT NULL,
                enabled INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        
        # Create charts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS charts (
                chart_id TEXT PRIMARY KEY,
                chart_type TEXT NOT NULL,
                title TEXT NOT NULL,
                data TEXT NOT NULL,
                x_axis TEXT NOT NULL,
                y_axis TEXT NOT NULL,
                configuration TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _load_default_dashboard(self) -> None:
        """Load default dashboard configuration."""
        # Create default widgets
        default_widgets = [
            SLAWidget(
                widget_id="WIDGET_001",
                widget_type=WidgetType.METRIC_CARD,
                title="Overall SLA Status",
                description="Current overall SLA status",
                position=(0, 0),
                size=(2, 1),
                data_source="sla_summary",
                configuration={
                    "metric": "uptime_percentage",
                    "target": 99.9,
                    "format": "percentage",
                    "color_scheme": "traffic_light"
                },
                refresh_interval=30
            ),
            SLAWidget(
                widget_id="WIDGET_002",
                widget_type=WidgetType.LINE_CHART,
                title="SLA Trends",
                description="SLA performance trends over time",
                position=(0, 2),
                size=(4, 2),
                data_source="sla_trends",
                configuration={
                    "chart_type": "line",
                    "time_range": "24h",
                    "metrics": ["uptime", "response_time", "error_rate"]
                },
                refresh_interval=60
            ),
            SLAWidget(
                widget_id="WIDGET_003",
                widget_type=WidgetType.PIE_CHART,
                title="SLA Status Distribution",
                description="Distribution of SLA status across services",
                position=(2, 0),
                size=(2, 2),
                data_source="sla_status_distribution",
                configuration={
                    "chart_type": "pie",
                    "group_by": "service",
                    "metric": "status"
                },
                refresh_interval=60
            ),
            SLAWidget(
                widget_id="WIDGET_004",
                widget_type=WidgetType.ALERT_LIST,
                title="Active Alerts",
                description="Currently active SLA alerts",
                position=(2, 2),
                size=(4, 2),
                data_source="active_alerts",
                configuration={
                    "max_items": 10,
                    "severity_filter": ["CRITICAL", "WARNING"],
                    "sort_by": "timestamp"
                },
                refresh_interval=30
            ),
            SLAWidget(
                widget_id="WIDGET_005",
                widget_type=WidgetType.TABLE,
                title="Service Performance",
                description="Performance metrics for all services",
                position=(4, 0),
                size=(6, 2),
                data_source="service_performance",
                configuration={
                    "columns": ["service", "uptime", "response_time", "error_rate", "status"],
                    "sort_by": "uptime",
                    "sort_order": "desc"
                },
                refresh_interval=60
            )
        ]
        
        # Create default dashboard
        default_dashboard = SLADashboardConfig(
            config_id="DASHBOARD_001",
            name="SLA Monitoring Dashboard",
            description="Default SLA monitoring dashboard",
            layout=DashboardLayout.GRID,
            widgets=default_widgets,
            refresh_interval=30,
            auto_refresh=True,
            public_access=False
        )
        
        self.create_dashboard(default_dashboard)
    
    def create_dashboard(self, dashboard_config: SLADashboardConfig) -> None:
        """
        Create new dashboard.
        
        Args:
            dashboard_config: Dashboard configuration
        """
        self.dashboards[dashboard_config.config_id] = dashboard_config
        
        # Store widgets
        for widget in dashboard_config.widgets:
            self.widgets[widget.widget_id] = widget
            self._store_widget(widget)
        
        # Store dashboard
        self._store_dashboard(dashboard_config)
        
        logger.info(f"Created dashboard: {dashboard_config.config_id} - {dashboard_config.name}")
    
    def add_widget(self, widget: SLAWidget) -> None:
        """
        Add widget to dashboard.
        
        Args:
            widget: Widget definition
        """
        self.widgets[widget.widget_id] = widget
        self._store_widget(widget)
        
        logger.info(f"Added widget: {widget.widget_id} - {widget.title}")
    
    def create_chart(self, chart: SLAChart) -> str:
        """
        Create chart.
        
        Args:
            chart: Chart definition
            
        Returns:
            Chart ID
        """
        self.charts[chart.chart_id] = chart
        self._store_chart(chart)
        
        logger.info(f"Created chart: {chart.chart_id} - {chart.title}")
        return chart.chart_id
    
    def generate_widget_data(self, widget: SLAWidget) -> Dict[str, Any]:
        """
        Generate data for widget.
        
        Args:
            widget: Widget definition
            
        Returns:
            Widget data
        """
        try:
            if widget.data_source == "sla_summary":
                return self._generate_sla_summary_data(widget)
            elif widget.data_source == "sla_trends":
                return self._generate_sla_trends_data(widget)
            elif widget.data_source == "sla_status_distribution":
                return self._generate_sla_status_distribution_data(widget)
            elif widget.data_source == "active_alerts":
                return self._generate_active_alerts_data(widget)
            elif widget.data_source == "service_performance":
                return self._generate_service_performance_data(widget)
            else:
                return {"error": f"Unknown data source: {widget.data_source}"}
        
        except Exception as e:
            logger.error(f"Error generating widget data for {widget.widget_id}: {e}")
            return {"error": str(e)}
    
    def _generate_sla_summary_data(self, widget: SLAWidget) -> Dict[str, Any]:
        """Generate SLA summary data."""
        # Simulate SLA summary data
        return {
            "current_value": 99.95,
            "target_value": 99.9,
            "status": "EXCELLENT",
            "trend": "STABLE",
            "last_updated": datetime.now().isoformat(),
            "violations_today": 0,
            "alerts_active": 2
        }
    
    def _generate_sla_trends_data(self, widget: SLAWidget) -> Dict[str, Any]:
        """Generate SLA trends data."""
        # Generate time series data
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=24)
        
        timestamps = []
        uptime_values = []
        response_time_values = []
        error_rate_values = []
        
        current_time = start_time
        while current_time <= end_time:
            timestamps.append(current_time.isoformat())
            
            # Simulate realistic data with some noise
            uptime = 99.9 + np.random.normal(0, 0.05)
            response_time = 100 + np.random.normal(0, 20)
            error_rate = 0.1 + np.random.exponential(0.05)
            
            uptime_values.append(max(0, min(100, uptime)))
            response_time_values.append(max(0, response_time))
            error_rate_values.append(max(0, min(10, error_rate)))
            
            current_time += timedelta(hours=1)
        
        return {
            "timestamps": timestamps,
            "series": {
                "uptime": uptime_values,
                "response_time": response_time_values,
                "error_rate": error_rate_values
            },
            "last_updated": datetime.now().isoformat()
        }
    
    def _generate_sla_status_distribution_data(self, widget: SLAWidget) -> Dict[str, Any]:
        """Generate SLA status distribution data."""
        # Simulate service status distribution
        services = ["Trading System", "API Gateway", "Database", "AI Models", "Market Data"]
        statuses = ["EXCELLENT", "GOOD", "WARNING", "CRITICAL"]
        
        distribution = []
        for service in services:
            # Simulate status distribution
            status = np.random.choice(statuses, p=[0.4, 0.3, 0.2, 0.1])
            count = np.random.randint(1, 10)
            
            distribution.append({
                "service": service,
                "status": status,
                "count": count
            })
        
        return {
            "distribution": distribution,
            "last_updated": datetime.now().isoformat()
        }
    
    def _generate_active_alerts_data(self, widget: SLAWidget) -> Dict[str, Any]:
        """Generate active alerts data."""
        # Simulate active alerts
        alerts = []
        alert_types = ["SLA Violation", "High Response Time", "Service Down", "High Error Rate"]
        severities = ["CRITICAL", "WARNING", "INFO"]
        
        for i in range(np.random.randint(0, 5)):
            alert = {
                "alert_id": f"ALERT_{i+1:03d}",
                "title": np.random.choice(alert_types),
                "severity": np.random.choice(severities),
                "service": f"Service {i+1}",
                "timestamp": (datetime.now() - timedelta(minutes=np.random.randint(1, 60))).isoformat(),
                "description": f"Alert description for alert {i+1}"
            }
            alerts.append(alert)
        
        return {
            "alerts": alerts,
            "total_count": len(alerts),
            "last_updated": datetime.now().isoformat()
        }
    
    def _generate_service_performance_data(self, widget: SLAWidget) -> Dict[str, Any]:
        """Generate service performance data."""
        # Simulate service performance data
        services = ["Trading System", "API Gateway", "Database", "AI Models", "Market Data"]
        performance_data = []
        
        for service in services:
            uptime = 99.0 + np.random.normal(0, 0.5)
            response_time = 50 + np.random.normal(0, 20)
            error_rate = 0.1 + np.random.exponential(0.05)
            
            # Determine status based on uptime
            if uptime >= 99.9:
                status = "EXCELLENT"
            elif uptime >= 99.0:
                status = "GOOD"
            elif uptime >= 95.0:
                status = "WARNING"
            else:
                status = "CRITICAL"
            
            performance_data.append({
                "service": service,
                "uptime": round(uptime, 2),
                "response_time": round(response_time, 1),
                "error_rate": round(error_rate, 2),
                "status": status
            })
        
        return {
            "services": performance_data,
            "last_updated": datetime.now().isoformat()
        }
    
    def generate_chart_image(self, chart: SLAChart, format: str = "PNG") -> str:
        """
        Generate chart image.
        
        Args:
            chart: Chart definition
            format: Image format (PNG, SVG, PDF)
            
        Returns:
            Base64 encoded image
        """
        try:
            # Set up matplotlib
            plt.style.use('seaborn-v0_8')
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Prepare data
            df = pd.DataFrame(chart.data)
            
            if chart.chart_type == ChartType.LINE:
                ax.plot(df[chart.x_axis], df[chart.y_axis], marker='o', linewidth=2)
            elif chart.chart_type == ChartType.BAR:
                ax.bar(df[chart.x_axis], df[chart.y_axis])
            elif chart.chart_type == ChartType.PIE:
                ax.pie(df[chart.y_axis], labels=df[chart.x_axis], autopct='%1.1f%%')
            elif chart.chart_type == ChartType.AREA:
                ax.fill_between(df[chart.x_axis], df[chart.y_axis], alpha=0.7)
            elif chart.chart_type == ChartType.SCATTER:
                ax.scatter(df[chart.x_axis], df[chart.y_axis])
            
            # Customize chart
            ax.set_title(chart.title, fontsize=14, fontweight='bold')
            ax.set_xlabel(chart.x_axis)
            ax.set_ylabel(chart.y_axis)
            ax.grid(True, alpha=0.3)
            
            # Apply configuration
            if 'color' in chart.configuration:
                ax.set_color(chart.configuration['color'])
            if 'x_rotation' in chart.configuration:
                plt.xticks(rotation=chart.configuration['x_rotation'])
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format=format.lower(), dpi=300, bbox_inches='tight')
            buffer.seek(0)
            
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
        
        except Exception as e:
            logger.error(f"Error generating chart image for {chart.chart_id}: {e}")
            return ""
    
    def get_dashboard_data(self, dashboard_id: str) -> Dict[str, Any]:
        """
        Get dashboard data.
        
        Args:
            dashboard_id: Dashboard ID
            
        Returns:
            Dashboard data
        """
        if dashboard_id not in self.dashboards:
            return {"error": f"Dashboard {dashboard_id} not found"}
        
        dashboard = self.dashboards[dashboard_id]
        
        # Generate data for all widgets
        widgets_data = {}
        for widget in dashboard.widgets:
            if widget.enabled:
                widgets_data[widget.widget_id] = self.generate_widget_data(widget)
        
        return {
            "dashboard": {
                "config_id": dashboard.config_id,
                "name": dashboard.name,
                "description": dashboard.description,
                "layout": dashboard.layout.value,
                "refresh_interval": dashboard.refresh_interval,
                "auto_refresh": dashboard.auto_refresh,
                "public_access": dashboard.public_access
            },
            "widgets": widgets_data,
            "last_updated": datetime.now().isoformat()
        }
    
    def export_dashboard(self, dashboard_id: str, format: str = "JSON") -> Dict[str, Any]:
        """
        Export dashboard.
        
        Args:
            dashboard_id: Dashboard ID
            format: Export format (JSON, PNG, PDF)
            
        Returns:
            Export data
        """
        if dashboard_id not in self.dashboards:
            return {"error": f"Dashboard {dashboard_id} not found"}
        
        dashboard = self.dashboards[dashboard_id]
        
        if format == "JSON":
            return {
                "dashboard": {
                    "config_id": dashboard.config_id,
                    "name": dashboard.name,
                    "description": dashboard.description,
                    "layout": dashboard.layout.value,
                    "widgets": [
                        {
                            "widget_id": w.widget_id,
                            "widget_type": w.widget_type.value,
                            "title": w.title,
                            "description": w.description,
                            "position": w.position,
                            "size": w.size,
                            "data_source": w.data_source,
                            "configuration": w.configuration,
                            "refresh_interval": w.refresh_interval,
                            "enabled": w.enabled
                        }
                        for w in dashboard.widgets
                    ],
                    "refresh_interval": dashboard.refresh_interval,
                    "auto_refresh": dashboard.auto_refresh,
                    "public_access": dashboard.public_access
                },
                "exported_at": datetime.now().isoformat()
            }
        else:
            # For image formats, generate dashboard image
            return self._generate_dashboard_image(dashboard, format)
    
    def _generate_dashboard_image(self, dashboard: SLADashboardConfig, format: str) -> Dict[str, Any]:
        """Generate dashboard image."""
        try:
            # Create figure with subplots based on dashboard layout
            if dashboard.layout == DashboardLayout.GRID:
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            elif dashboard.layout == DashboardLayout.TWO_COLUMN:
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            else:
                fig, axes = plt.subplots(1, 1, figsize=(10, 6))
            
            # Generate charts for each widget
            for i, widget in enumerate(dashboard.widgets[:6]):  # Limit to 6 widgets
                if widget.enabled and widget.widget_type in [WidgetType.LINE_CHART, WidgetType.BAR_CHART, WidgetType.PIE_CHART]:
                    ax = axes.flat[i] if hasattr(axes, 'flat') else axes
                    
                    # Generate widget data
                    data = self.generate_widget_data(widget)
                    
                    # Create chart based on widget type
                    if widget.widget_type == WidgetType.LINE_CHART and 'series' in data:
                        for series_name, values in data['series'].items():
                            ax.plot(values, label=series_name)
                        ax.legend()
                    elif widget.widget_type == WidgetType.BAR_CHART and 'distribution' in data:
                        services = [item['service'] for item in data['distribution']]
                        counts = [item['count'] for item in data['distribution']]
                        ax.bar(services, counts)
                    elif widget.widget_type == WidgetType.PIE_CHART and 'distribution' in data:
                        services = [item['service'] for item in data['distribution']]
                        counts = [item['count'] for item in data['distribution']]
                        ax.pie(counts, labels=services, autopct='%1.1f%%')
                    
                    ax.set_title(widget.title)
                    ax.grid(True, alpha=0.3)
            
            plt.suptitle(dashboard.name, fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format=format.lower(), dpi=300, bbox_inches='tight')
            buffer.seek(0)
            
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return {
                "image": image_base64,
                "format": format,
                "exported_at": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error generating dashboard image: {e}")
            return {"error": str(e)}
    
    def _store_dashboard(self, dashboard: SLADashboardConfig) -> None:
        """Store dashboard in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO dashboards 
            (config_id, name, description, layout, widgets, refresh_interval, auto_refresh, public_access, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            dashboard.config_id, dashboard.name, dashboard.description, dashboard.layout.value,
            json.dumps([w.__dict__ for w in dashboard.widgets]), dashboard.refresh_interval,
            dashboard.auto_refresh, dashboard.public_access, dashboard.created_at.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def _store_widget(self, widget: SLAWidget) -> None:
        """Store widget in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO widgets 
            (widget_id, widget_type, title, description, position, size, data_source, configuration, refresh_interval, enabled, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            widget.widget_id, widget.widget_type.value, widget.title, widget.description,
            json.dumps(widget.position), json.dumps(widget.size), widget.data_source,
            json.dumps(widget.configuration), widget.refresh_interval, widget.enabled,
            widget.created_at.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def _store_chart(self, chart: SLAChart) -> None:
        """Store chart in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO charts 
            (chart_id, chart_type, title, data, x_axis, y_axis, configuration, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            chart.chart_id, chart.chart_type.value, chart.title, json.dumps(chart.data),
            chart.x_axis, chart.y_axis, json.dumps(chart.configuration), chart.created_at.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def get_dashboard_summary(self) -> Dict[str, Any]:
        """
        Get dashboard summary.
        
        Returns:
            Dashboard summary dictionary
        """
        return {
            'summary': {
                'total_dashboards': len(self.dashboards),
                'total_widgets': len(self.widgets),
                'total_charts': len(self.charts)
            },
            'dashboards': [
                {
                    'config_id': d.config_id,
                    'name': d.name,
                    'description': d.description,
                    'layout': d.layout.value,
                    'widget_count': len(d.widgets),
                    'refresh_interval': d.refresh_interval,
                    'auto_refresh': d.auto_refresh,
                    'public_access': d.public_access,
                    'created_at': d.created_at.isoformat()
                }
                for d in self.dashboards.values()
            ],
            'widget_types': {
                widget_type.value: len([w for w in self.widgets.values() if w.widget_type == widget_type])
                for widget_type in WidgetType
            },
            'chart_types': {
                chart_type.value: len([c for c in self.charts.values() if c.chart_type == chart_type])
                for chart_type in ChartType
            }
        }
