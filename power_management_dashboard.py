#!/usr/bin/env python3
"""
Power Management Dashboard - Real-time monitoring
Displays power savings, GPU usage, cache performance, and system health
"""

import sys
import os
import io
import json
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

# Fix Unicode encoding for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np

# Import power management
from power_management import (
    get_gpu_manager,
    get_precision_manager,
    get_worker_scaler,
    get_cache_manager,
    get_network_optimizer,
    get_schedule_manager,
    get_power_monitor,
    DEFAULT_CONFIG
)

# Initialize components
print("Initializing Power Management Dashboard...")

# Initialize all power management components
gpu_manager = get_gpu_manager(DEFAULT_CONFIG.gpu)
precision_manager = get_precision_manager(DEFAULT_CONFIG.mixed_precision)
worker_scaler = get_worker_scaler(DEFAULT_CONFIG.workers, current_workers=4)
cache_manager = get_cache_manager(DEFAULT_CONFIG.caching)
schedule_manager = get_schedule_manager(DEFAULT_CONFIG.schedule)

# Initialize monitor
power_monitor = get_power_monitor(DEFAULT_CONFIG)
power_monitor.set_managers(
    gpu_manager=gpu_manager,
    precision_manager=precision_manager,
    worker_scaler=worker_scaler,
    cache_manager=cache_manager,
    schedule_manager=schedule_manager
)

print("✅ Power Management components initialized")

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Power Management Dashboard"

# Custom CSS for dark theme
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            :root {
                --bg-primary: #0a0e27;
                --bg-secondary: #1a1f3a;
                --bg-card: #252b48;
                --text-primary: #e0e6ed;
                --text-secondary: #8b93a7;
                --accent-green: #00d4aa;
                --accent-blue: #4a9eff;
                --accent-yellow: #ffd93d;
                --accent-red: #ff6b6b;
                --border-color: #2d3561;
            }

            body {
                background-color: var(--bg-primary);
                color: var(--text-primary);
                font-family: 'Inter', 'Segoe UI', sans-serif;
                margin: 0;
                padding: 0;
            }

            .dashboard-header {
                background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-primary) 100%);
                padding: 30px;
                border-bottom: 2px solid var(--accent-green);
                text-align: center;
            }

            .dashboard-header h1 {
                margin: 0;
                font-size: 2.5rem;
                font-weight: 700;
                background: linear-gradient(90deg, var(--accent-green), var(--accent-blue));
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }

            .dashboard-header p {
                margin: 10px 0 0 0;
                color: var(--text-secondary);
                font-size: 1rem;
            }

            .metric-card {
                background-color: var(--bg-card);
                border: 1px solid var(--border-color);
                border-radius: 12px;
                padding: 20px;
                margin: 15px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
                transition: transform 0.2s, box-shadow 0.2s;
            }

            .metric-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 12px rgba(0, 212, 170, 0.2);
            }

            .metric-value {
                font-size: 2.5rem;
                font-weight: 700;
                margin: 10px 0;
            }

            .metric-label {
                font-size: 0.9rem;
                color: var(--text-secondary);
                text-transform: uppercase;
                letter-spacing: 1px;
            }

            .status-badge {
                display: inline-block;
                padding: 6px 12px;
                border-radius: 20px;
                font-size: 0.85rem;
                font-weight: 600;
                margin: 5px;
            }

            .status-healthy {
                background-color: rgba(0, 212, 170, 0.2);
                color: var(--accent-green);
                border: 1px solid var(--accent-green);
            }

            .status-degraded {
                background-color: rgba(255, 217, 61, 0.2);
                color: var(--accent-yellow);
                border: 1px solid var(--accent-yellow);
            }

            .status-unhealthy {
                background-color: rgba(255, 107, 107, 0.2);
                color: var(--accent-red);
                border: 1px solid var(--accent-red);
            }

            .chart-container {
                background-color: var(--bg-card);
                border: 1px solid var(--border-color);
                border-radius: 12px;
                padding: 20px;
                margin: 15px;
            }

            .component-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 15px;
                padding: 15px;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("⚡ Power Management Dashboard"),
        html.P("Real-time monitoring of power optimizations and system efficiency"),
    ], className='dashboard-header'),

    # Auto-refresh interval
    dcc.Interval(
        id='interval-component',
        interval=2000,  # Update every 2 seconds
        n_intervals=0
    ),

    # Main content
    html.Div([
        # Top metrics row
        html.Div([
            html.Div([
                html.Div([
                    html.Div("Power Savings", className='metric-label'),
                    html.Div(id='total-savings', className='metric-value'),
                    html.Div(id='savings-target', style={'fontSize': '0.9rem', 'marginTop': '10px'}),
                ], className='metric-card'),
            ], style={'width': '25%', 'display': 'inline-block'}),

            html.Div([
                html.Div([
                    html.Div("System Health", className='metric-label'),
                    html.Div(id='system-health', className='metric-value'),
                    html.Div(id='health-status', style={'marginTop': '10px'}),
                ], className='metric-card'),
            ], style={'width': '25%', 'display': 'inline-block'}),

            html.Div([
                html.Div([
                    html.Div("GPU Power", className='metric-label'),
                    html.Div(id='gpu-power', className='metric-value'),
                    html.Div(id='gpu-mode', style={'fontSize': '0.9rem', 'marginTop': '10px'}),
                ], className='metric-card'),
            ], style={'width': '25%', 'display': 'inline-block'}),

            html.Div([
                html.Div([
                    html.Div("Cache Hit Rate", className='metric-label'),
                    html.Div(id='cache-hit-rate', className='metric-value'),
                    html.Div(id='cache-stats', style={'fontSize': '0.9rem', 'marginTop': '10px'}),
                ], className='metric-card'),
            ], style={'width': '25%', 'display': 'inline-block'}),
        ], style={'width': '100%', 'display': 'flex'}),

        # Charts row
        html.Div([
            html.Div([
                dcc.Graph(id='savings-breakdown-chart'),
            ], className='chart-container', style={'width': '48%', 'display': 'inline-block'}),

            html.Div([
                dcc.Graph(id='component-status-chart'),
            ], className='chart-container', style={'width': '48%', 'display': 'inline-block', 'marginLeft': '2%'}),
        ], style={'width': '100%'}),

        # Component details
        html.Div([
            html.H3("Component Details", style={'padding': '20px', 'color': '#e0e6ed'}),
            html.Div(id='component-details', className='component-grid'),
        ]),

        # Market schedule
        html.Div([
            html.H3("Market Schedule", style={'padding': '20px', 'color': '#e0e6ed'}),
            html.Div(id='market-schedule', style={'padding': '0 20px'}),
        ]),
    ], style={'padding': '20px'}),
])


# Callbacks
@app.callback(
    [Output('total-savings', 'children'),
     Output('savings-target', 'children'),
     Output('system-health', 'children'),
     Output('health-status', 'children'),
     Output('gpu-power', 'children'),
     Output('gpu-mode', 'children'),
     Output('cache-hit-rate', 'children'),
     Output('cache-stats', 'children'),
     Output('savings-breakdown-chart', 'figure'),
     Output('component-status-chart', 'figure'),
     Output('component-details', 'children'),
     Output('market-schedule', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_dashboard(n):
    """Update all dashboard components."""

    # Get comprehensive stats
    stats = power_monitor.get_comprehensive_stats()
    health = power_monitor.get_health_status()
    savings = stats['estimated_total_savings']

    # Total savings
    savings_value = f"{savings['total_percentage']}%"
    savings_status = f"Target: {savings['target_percentage']}% | {'✅ MET' if savings['target_met'] else '❌ NOT MET'}"

    # System health
    health_icon = {'healthy': '✅', 'degraded': '⚠️', 'unhealthy': '❌'}
    health_value = health_icon.get(health['overall'], '❓')
    health_badges = [
        html.Span(comp.upper(), className=f"status-{status}")
        for comp, status in health['components'].items()
    ]

    # GPU power
    gpu_stats = stats['components'].get('gpu', {})
    gpu_power_value = f"{gpu_stats.get('current_power_watts', 0):.1f}W" if gpu_stats.get('current_power_watts') else "N/A"
    gpu_mode_text = f"Mode: {gpu_stats.get('current_mode', 'N/A')}"

    # Cache stats
    cache_stats = stats['components'].get('caching', {})
    cache_hit_value = f"{cache_stats.get('hit_rate', 0):.1f}%"
    cache_details = f"Hits: {cache_stats.get('hits', 0)} | Misses: {cache_stats.get('misses', 0)}"

    # Savings breakdown chart
    breakdown_data = savings['breakdown']
    fig_breakdown = go.Figure(data=[
        go.Bar(
            x=list(breakdown_data.values()),
            y=list(breakdown_data.keys()),
            orientation='h',
            marker=dict(
                color=list(breakdown_data.values()),
                colorscale='Viridis',
                line=dict(color='rgba(0, 212, 170, 0.6)', width=2)
            ),
            text=[f"{v}%" for v in breakdown_data.values()],
            textposition='auto',
        )
    ])

    fig_breakdown.update_layout(
        title='Power Savings Breakdown',
        xaxis_title='Savings (%)',
        yaxis_title='Component',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e6ed'),
        height=400,
    )

    # Component status chart
    component_names = list(health['components'].keys())
    component_statuses = [
        1 if status == 'healthy' else 0.5 if status == 'degraded' else 0
        for status in health['components'].values()
    ]

    fig_status = go.Figure(data=[
        go.Bar(
            x=component_names,
            y=component_statuses,
            marker=dict(
                color=['#00d4aa' if s == 1 else '#ffd93d' if s == 0.5 else '#ff6b6b' for s in component_statuses],
                line=dict(color='rgba(255, 255, 255, 0.2)', width=1)
            ),
        )
    ])

    fig_status.update_layout(
        title='Component Health Status',
        xaxis_title='Component',
        yaxis_title='Status',
        yaxis=dict(tickvals=[0, 0.5, 1], ticktext=['Unhealthy', 'Degraded', 'Healthy']),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e6ed'),
        height=400,
    )

    # Component details
    component_cards = []

    # GPU details
    if 'gpu' in stats['components']:
        gpu = stats['components']['gpu']
        component_cards.append(html.Div([
            html.H4("🖥️ GPU Power Management"),
            html.P(f"Enabled: {'✅' if gpu['enabled'] else '❌'}"),
            html.P(f"GPU Available: {'✅' if gpu['gpu_available'] else '❌'}"),
            html.P(f"Current Mode: {gpu['current_mode']}"),
            html.P(f"Power Limits: Training={gpu['power_limit_training']}%, Inference={gpu['power_limit_inference']}%"),
        ], className='metric-card'))

    # Mixed Precision details
    if 'mixed_precision' in stats['components']:
        mp = stats['components']['mixed_precision']
        component_cards.append(html.Div([
            html.H4("🔢 Mixed Precision"),
            html.P(f"Enabled: {'✅' if mp['enabled'] else '❌'}"),
            html.P(f"Framework: {mp['framework']}"),
            html.P(f"Precision: {mp['precision']}"),
            html.P(f"Savings: {mp['estimated_power_savings']}"),
        ], className='metric-card'))

    # Worker Scaling details
    if 'worker_scaling' in stats['components']:
        ws = stats['components']['worker_scaling']
        component_cards.append(html.Div([
            html.H4("⚙️ Worker Scaling"),
            html.P(f"Enabled: {'✅' if ws['enabled'] else '❌'}"),
            html.P(f"Current Workers: {ws['current_workers']}"),
            html.P(f"Range: {ws['min_workers']}-{ws['max_workers']}"),
            html.P(f"CPU Usage: {ws['current_cpu_percent']:.1f}%"),
        ], className='metric-card'))

    # Caching details
    if 'caching' in stats['components']:
        cache = stats['components']['caching']
        component_cards.append(html.Div([
            html.H4("💾 Intelligent Caching"),
            html.P(f"Enabled: {'✅' if cache['enabled'] else '❌'}"),
            html.P(f"Backend: {cache['backend']}"),
            html.P(f"Hit Rate: {cache['hit_rate']:.1f}%"),
            html.P(f"Total Hits: {cache['hits']}"),
        ], className='metric-card'))

    # Schedule details
    if 'schedule' in stats['components']:
        sched = stats['components']['schedule']
        component_cards.append(html.Div([
            html.H4("🕒 Schedule Management"),
            html.P(f"Enabled: {'✅' if sched['enabled'] else '❌'}"),
            html.P(f"Market Status: {sched['market_status']}"),
            html.P(f"Services Running: {sched['running_services']}"),
            html.P(f"Services Paused: {sched['paused_services']}"),
        ], className='metric-card'))

    # Market schedule
    if 'schedule' in stats['components']:
        sched = stats['components']['schedule']
        market_info = html.Div([
            html.Div([
                html.H4(f"Market Status: {sched['market_status'].upper()}"),
                html.P(f"Market Hours: {sched['market_open_time']} - {sched['market_close_time']}"),
                html.P(f"Next Market Open: {sched.get('next_market_open', 'N/A')}"),
                html.P(f"Time Until Open: {sched.get('time_until_open', 'N/A')}"),
                html.P(f"Total Sleep Hours: {sched.get('total_sleep_hours', 0):.1f}"),
            ], className='metric-card')
        ])
    else:
        market_info = html.P("Schedule management not available")

    return (
        savings_value, savings_status,
        health_value, html.Div(health_badges),
        gpu_power_value, gpu_mode_text,
        cache_hit_value, cache_details,
        fig_breakdown, fig_status,
        component_cards, market_info
    )


if __name__ == '__main__':
    print("\n" + "="*70)
    print("⚡ POWER MANAGEMENT DASHBOARD STARTING")
    print("="*70)
    print(f"Dashboard URL: http://localhost:8051")
    print(f"Refresh Rate: 2 seconds")
    print(f"Power Monitoring: Active")
    print("="*70 + "\n")

    # Run the app
    app.run(
        debug=True,
        host='0.0.0.0',
        port=8051
    )
