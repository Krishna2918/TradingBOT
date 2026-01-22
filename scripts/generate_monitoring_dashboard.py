#!/usr/bin/env python3
"""
Feature Consistency Monitoring Dashboard Generator

This script generates an HTML dashboard from feature consistency monitoring data.
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_monitoring_data(dashboard_path: str) -> Optional[Dict[str, Any]]:
    """Load monitoring data from JSON file."""
    try:
        with open(dashboard_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Dashboard data file not found: {dashboard_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing dashboard data: {e}")
        return None


def load_metrics_history(metrics_path: str) -> List[Dict[str, Any]]:
    """Load historical metrics data."""
    try:
        with open(metrics_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Metrics file not found: {metrics_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error parsing metrics data: {e}")
        return []


def load_alerts(alerts_path: str) -> List[Dict[str, Any]]:
    """Load alerts data."""
    try:
        with open(alerts_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    except json.JSONDecodeError as e:
        print(f"Error parsing alerts data: {e}")
        return []


def generate_status_card(title: str, value: str, status: str = "normal", 
                        subtitle: str = "") -> str:
    """Generate HTML for a status card."""
    status_colors = {
        "normal": "#28a745",
        "warning": "#ffc107", 
        "error": "#dc3545",
        "info": "#17a2b8"
    }
    
    color = status_colors.get(status, "#6c757d")
    
    return f"""
    <div class="status-card">
        <div class="status-card-header" style="border-left: 4px solid {color};">
            <h3>{title}</h3>
            <div class="status-value">{value}</div>
            {f'<div class="status-subtitle">{subtitle}</div>' if subtitle else ''}
        </div>
    </div>
    """


def generate_metrics_table(metrics: Dict[str, Any]) -> str:
    """Generate HTML table for metrics."""
    if not metrics:
        return "<p>No metrics data available</p>"
    
    current_run = metrics.get('current_run', {})
    
    rows = []
    
    # Processing summary
    processing = current_run.get('processing_summary', {})
    rows.extend([
        ("Symbols Processed", processing.get('symbols_processed', 'N/A')),
        ("Symbols Included", processing.get('symbols_included', 'N/A')),
        ("Symbols Excluded", processing.get('symbols_excluded', 'N/A')),
        ("Exclusion Rate", f"{processing.get('exclusion_rate_pct', 'N/A')}%"),
        ("Processing Time", f"{processing.get('processing_time_minutes', 'N/A')} min")
    ])
    
    # Feature summary
    features = current_run.get('feature_summary', {})
    rows.extend([
        ("Total Features", features.get('total_features', 'N/A')),
        ("Stable Features", features.get('stable_features', 'N/A')),
        ("Unstable Features", features.get('unstable_features', 'N/A')),
        ("Stability Rate", f"{features.get('stability_rate_pct', 'N/A')}%")
    ])
    
    # Coverage summary
    coverage = current_run.get('coverage_summary', {})
    rows.extend([
        ("Average Coverage", f"{coverage.get('avg_coverage_pct', 'N/A')}%"),
        ("Min Coverage", f"{coverage.get('min_coverage_pct', 'N/A')}%"),
        ("Max Coverage", f"{coverage.get('max_coverage_pct', 'N/A')}%")
    ])
    
    table_rows = ""
    for label, value in rows:
        table_rows += f"<tr><td>{label}</td><td>{value}</td></tr>"
    
    return f"""
    <table class="metrics-table">
        <thead>
            <tr><th>Metric</th><th>Value</th></tr>
        </thead>
        <tbody>
            {table_rows}
        </tbody>
    </table>
    """


def generate_alerts_section(alerts: List[Dict[str, Any]]) -> str:
    """Generate HTML for alerts section."""
    if not alerts:
        return "<p>No recent alerts</p>"
    
    # Sort alerts by timestamp (most recent first)
    sorted_alerts = sorted(alerts, key=lambda x: x.get('timestamp', ''), reverse=True)
    recent_alerts = sorted_alerts[:10]  # Show last 10 alerts
    
    alert_rows = ""
    for alert in recent_alerts:
        severity = alert.get('severity', 'info')
        severity_colors = {
            'info': '#17a2b8',
            'warning': '#ffc107',
            'error': '#dc3545',
            'critical': '#721c24'
        }
        color = severity_colors.get(severity, '#6c757d')
        
        timestamp = alert.get('timestamp', 'Unknown')
        try:
            # Format timestamp
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            formatted_time = dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            formatted_time = timestamp
        
        alert_rows += f"""
        <tr>
            <td><span class="severity-badge" style="background-color: {color};">{severity.upper()}</span></td>
            <td>{alert.get('description', 'No description')}</td>
            <td>{formatted_time}</td>
            <td>{alert.get('actual_value', 'N/A')}</td>
        </tr>
        """
    
    return f"""
    <table class="alerts-table">
        <thead>
            <tr><th>Severity</th><th>Description</th><th>Time</th><th>Value</th></tr>
        </thead>
        <tbody>
            {alert_rows}
        </tbody>
    </table>
    """


def generate_trends_chart_data(metrics_history: List[Dict[str, Any]]) -> str:
    """Generate JavaScript data for trends charts."""
    if not metrics_history:
        return "const trendsData = {};"
    
    # Extract time series data
    timestamps = []
    exclusion_rates = []
    stability_rates = []
    processing_times = []
    
    for metrics in metrics_history[-20:]:  # Last 20 runs
        if 'processing_start_time' in metrics:
            try:
                dt = datetime.fromisoformat(metrics['processing_start_time'])
                timestamps.append(dt.strftime('%m-%d %H:%M'))
            except:
                timestamps.append('Unknown')
        else:
            timestamps.append('Unknown')
        
        exclusion_rates.append(metrics.get('exclusion_rate', 0) * 100)
        stability_rates.append(metrics.get('feature_stability_rate', 0) * 100)
        processing_times.append(metrics.get('total_processing_time_seconds', 0) / 60)  # Convert to minutes
    
    return f"""
    const trendsData = {{
        timestamps: {json.dumps(timestamps)},
        exclusionRates: {json.dumps(exclusion_rates)},
        stabilityRates: {json.dumps(stability_rates)},
        processingTimes: {json.dumps(processing_times)}
    }};
    """


def generate_dashboard_html(dashboard_data: Dict[str, Any], 
                           metrics_history: List[Dict[str, Any]],
                           alerts: List[Dict[str, Any]]) -> str:
    """Generate complete HTML dashboard."""
    
    current_run = dashboard_data.get('current_run', {})
    processing = current_run.get('processing_summary', {})
    features = current_run.get('feature_summary', {})
    coverage = current_run.get('coverage_summary', {})
    drift = current_run.get('drift_detection', {})
    alert_summary = current_run.get('alerts', {})
    
    # Determine overall status
    exclusion_rate = processing.get('exclusion_rate_pct', 0)
    stability_rate = features.get('stability_rate_pct', 100)
    total_alerts = alert_summary.get('total_alerts', 0)
    
    overall_status = "normal"
    if exclusion_rate > 50 or stability_rate < 50 or total_alerts > 5:
        overall_status = "error"
    elif exclusion_rate > 20 or stability_rate < 80 or total_alerts > 0:
        overall_status = "warning"
    
    # Generate status cards
    status_cards = "".join([
        generate_status_card(
            "Overall Status", 
            "Healthy" if overall_status == "normal" else "Issues Detected" if overall_status == "warning" else "Critical",
            overall_status
        ),
        generate_status_card(
            "Symbols Processed", 
            str(processing.get('symbols_processed', 'N/A')),
            "warning" if exclusion_rate > 20 else "normal",
            f"{processing.get('exclusion_rate_pct', 'N/A')}% excluded"
        ),
        generate_status_card(
            "Feature Stability", 
            f"{features.get('stability_rate_pct', 'N/A')}%",
            "warning" if stability_rate < 80 else "normal",
            f"{features.get('stable_features', 'N/A')} stable features"
        ),
        generate_status_card(
            "Active Alerts", 
            str(total_alerts),
            "error" if total_alerts > 5 else "warning" if total_alerts > 0 else "normal"
        ),
        generate_status_card(
            "Feature Drift", 
            "Detected" if drift.get('drift_detected', False) else "None",
            "warning" if drift.get('drift_detected', False) else "normal",
            f"{drift.get('net_feature_change', 0):+d} features" if drift.get('drift_detected', False) else ""
        )
    ])
    
    # Generate trends chart data
    trends_js = generate_trends_chart_data(metrics_history)
    
    # Get last updated time
    last_updated = dashboard_data.get('last_updated', 'Unknown')
    try:
        dt = datetime.fromisoformat(last_updated)
        formatted_updated = dt.strftime('%Y-%m-%d %H:%M:%S UTC')
    except:
        formatted_updated = last_updated
    
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Feature Consistency Monitoring Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f8f9fa;
            color: #333;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            text-align: center;
        }}
        
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        
        .header .subtitle {{
            margin: 10px 0 0 0;
            opacity: 0.9;
            font-size: 1.1em;
        }}
        
        .status-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .status-card {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        
        .status-card-header {{
            padding: 20px;
        }}
        
        .status-card h3 {{
            margin: 0 0 10px 0;
            font-size: 1.1em;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .status-value {{
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }}
        
        .status-subtitle {{
            margin-top: 5px;
            color: #666;
            font-size: 0.9em;
        }}
        
        .dashboard-section {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            overflow: hidden;
        }}
        
        .section-header {{
            background: #f8f9fa;
            padding: 15px 20px;
            border-bottom: 1px solid #dee2e6;
        }}
        
        .section-header h2 {{
            margin: 0;
            color: #495057;
            font-size: 1.3em;
        }}
        
        .section-content {{
            padding: 20px;
        }}
        
        .metrics-table, .alerts-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        
        .metrics-table th, .metrics-table td,
        .alerts-table th, .alerts-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }}
        
        .metrics-table th, .alerts-table th {{
            background-color: #f8f9fa;
            font-weight: 600;
            color: #495057;
        }}
        
        .severity-badge {{
            padding: 4px 8px;
            border-radius: 4px;
            color: white;
            font-size: 0.8em;
            font-weight: bold;
        }}
        
        .chart-container {{
            position: relative;
            height: 300px;
            margin-top: 20px;
        }}
        
        .footer {{
            text-align: center;
            color: #666;
            margin-top: 30px;
            padding: 20px;
            font-size: 0.9em;
        }}
        
        .grid-2 {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        
        @media (max-width: 768px) {{
            .grid-2 {{
                grid-template-columns: 1fr;
            }}
            
            .status-cards {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Feature Consistency Monitoring</h1>
        <div class="subtitle">Real-time monitoring and alerting for LSTM training pipeline</div>
        <div class="subtitle">Last Updated: {formatted_updated}</div>
    </div>
    
    <div class="status-cards">
        {status_cards}
    </div>
    
    <div class="grid-2">
        <div class="dashboard-section">
            <div class="section-header">
                <h2>Current Run Metrics</h2>
            </div>
            <div class="section-content">
                {generate_metrics_table(dashboard_data)}
            </div>
        </div>
        
        <div class="dashboard-section">
            <div class="section-header">
                <h2>Recent Alerts</h2>
            </div>
            <div class="section-content">
                {generate_alerts_section(alerts)}
            </div>
        </div>
    </div>
    
    <div class="dashboard-section">
        <div class="section-header">
            <h2>Historical Trends</h2>
        </div>
        <div class="section-content">
            <div class="chart-container">
                <canvas id="trendsChart"></canvas>
            </div>
        </div>
    </div>
    
    <div class="footer">
        <p>Feature Consistency Monitoring Dashboard | Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
    </div>
    
    <script>
        {trends_js}
        
        // Create trends chart
        const ctx = document.getElementById('trendsChart').getContext('2d');
        const trendsChart = new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: trendsData.timestamps || [],
                datasets: [
                    {{
                        label: 'Exclusion Rate (%)',
                        data: trendsData.exclusionRates || [],
                        borderColor: '#dc3545',
                        backgroundColor: 'rgba(220, 53, 69, 0.1)',
                        tension: 0.4
                    }},
                    {{
                        label: 'Stability Rate (%)',
                        data: trendsData.stabilityRates || [],
                        borderColor: '#28a745',
                        backgroundColor: 'rgba(40, 167, 69, 0.1)',
                        tension: 0.4
                    }},
                    {{
                        label: 'Processing Time (min)',
                        data: trendsData.processingTimes || [],
                        borderColor: '#17a2b8',
                        backgroundColor: 'rgba(23, 162, 184, 0.1)',
                        tension: 0.4,
                        yAxisID: 'y1'
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                interaction: {{
                    intersect: false,
                    mode: 'index'
                }},
                scales: {{
                    y: {{
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {{
                            display: true,
                            text: 'Percentage (%)'
                        }}
                    }},
                    y1: {{
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {{
                            display: true,
                            text: 'Time (minutes)'
                        }},
                        grid: {{
                            drawOnChartArea: false,
                        }},
                    }}
                }},
                plugins: {{
                    legend: {{
                        position: 'top',
                    }},
                    title: {{
                        display: true,
                        text: 'Feature Consistency Metrics Over Time'
                    }}
                }}
            }}
        }});
        
        // Auto-refresh every 5 minutes
        setTimeout(() => {{
            location.reload();
        }}, 5 * 60 * 1000);
    </script>
</body>
</html>
    """


def main():
    """Main function to generate dashboard."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Feature Consistency Monitoring Dashboard")
    parser.add_argument('--dashboard-data', default='monitoring/feature_consistency_dashboard.json',
                       help='Path to dashboard data JSON file')
    parser.add_argument('--metrics-data', default='monitoring/feature_consistency_metrics.json',
                       help='Path to metrics history JSON file')
    parser.add_argument('--alerts-data', default='monitoring/alerts.json',
                       help='Path to alerts JSON file')
    parser.add_argument('--output', default='monitoring/dashboard.html',
                       help='Output HTML file path')
    
    args = parser.parse_args()
    
    # Load data
    print("Loading monitoring data...")
    dashboard_data = load_monitoring_data(args.dashboard_data)
    metrics_history = load_metrics_history(args.metrics_data)
    alerts = load_alerts(args.alerts_data)
    
    if not dashboard_data:
        print("No dashboard data available. Creating empty dashboard.")
        dashboard_data = {
            'last_updated': datetime.now().isoformat(),
            'current_run': {},
            'recent_alerts': [],
            'historical_trends': {}
        }
    
    # Generate HTML
    print("Generating dashboard HTML...")
    html_content = generate_dashboard_html(dashboard_data, metrics_history, alerts)
    
    # Save HTML file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Dashboard generated successfully: {output_path}")
    print(f"Open {output_path.absolute()} in your browser to view the dashboard")


if __name__ == "__main__":
    main()