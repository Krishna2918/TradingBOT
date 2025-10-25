#!/usr/bin/env python3
"""
Clean AI Trading Dashboard - Final Version
Fixes all previous issues: Unicode, method names, orchestrator init, data pipeline
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

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Fix Unicode encoding for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import dash
from dash import dcc, html, Input, Output, State, callback_context, ctx
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np

# Import our clean components
from src.dashboard.clean_state_manager import CleanStateManager
from src.dashboard.maximum_power_ai_engine import MaximumPowerAIEngine
from src.dashboard.advanced_ai_logger import AdvancedAILogger

# Initialize components
print("Initializing Clean AI Trading Dashboard...")

# Initialize state manager
state_manager = CleanStateManager()
print("Clean State Manager initialized")

# Initialize AI engine
ai_engine = MaximumPowerAIEngine()
print("Maximum Power AI Engine initialized")

# Initialize advanced logger
advanced_logger = AdvancedAILogger()
print("Advanced AI Logger initialized")

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Clean AI Trading Dashboard"

# Custom CSS for dark theme
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            :root {
                --bg-primary: #1a1a1a;
                --bg-secondary: #2d2d2d;
                --bg-tertiary: #3d3d3d;
                --text-primary: #ffffff;
                --text-secondary: #cccccc;
                --accent-primary: #00d4aa;
                --accent-secondary: #ff6b6b;
                --border-color: #444444;
            }
            
            body {
                background-color: var(--bg-primary);
                color: var(--text-primary);
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 0;
            }
            
            .dashboard-container {
                background-color: var(--bg-primary);
                color: var(--text-primary);
                min-height: 100vh;
                padding: 20px;
            }
            
            .metric-card {
                background-color: var(--bg-secondary);
                border: 1px solid var(--border-color);
                border-radius: 8px;
                padding: 20px;
                margin: 10px;
                text-align: center;
                box-shadow: 0 2px 4px rgba(0,0,0,0.3);
            }
            
            .metric-value {
                font-size: 2em;
                font-weight: bold;
                color: var(--accent-primary);
                margin: 10px 0;
            }
            
            .metric-label {
                color: var(--text-secondary);
                font-size: 0.9em;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            
            .btn-primary {
                background-color: var(--accent-primary);
                border: none;
                color: white;
                padding: 12px 24px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 16px;
                font-weight: bold;
                transition: all 0.3s ease;
            }
            
            .btn-primary:hover {
                background-color: #00b894;
                transform: translateY(-2px);
            }
            
            .btn-danger {
                background-color: var(--accent-secondary);
                border: none;
                color: white;
                padding: 12px 24px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 16px;
                font-weight: bold;
                transition: all 0.3s ease;
            }
            
            .btn-danger:hover {
                background-color: #ff5252;
                transform: translateY(-2px);
            }
            
            .input-field {
                background-color: var(--bg-tertiary);
                border: 1px solid var(--border-color);
                color: var(--text-primary);
                padding: 10px;
                border-radius: 4px;
                width: 100%;
                margin: 5px 0;
            }
            
            .input-field:focus {
                outline: none;
                border-color: var(--accent-primary);
            }
            
            .status-badge {
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 0.8em;
                font-weight: bold;
                text-transform: uppercase;
            }
            
            .status-active {
                background-color: #00d4aa;
                color: white;
            }
            
            .status-idle {
                background-color: #6c757d;
                color: white;
            }
            
            .status-error {
                background-color: #ff6b6b;
                color: white;
            }
            
            .table-container {
                background-color: var(--bg-secondary);
                border: 1px solid var(--border-color);
                border-radius: 8px;
                margin: 20px 0;
                overflow: hidden;
            }
            
            .table-header {
                background-color: var(--bg-tertiary);
                padding: 15px;
                border-bottom: 1px solid var(--border-color);
            }
            
            .table-title {
                color: var(--text-primary);
                font-size: 1.2em;
                font-weight: bold;
                margin: 0;
            }
            
            .modal-content {
                background-color: var(--bg-secondary);
                color: var(--text-primary);
                border: 1px solid var(--border-color);
            }
            
            .modal-header {
                background-color: var(--bg-tertiary);
                border-bottom: 1px solid var(--border-color);
                color: var(--text-primary);
            }
            
            .modal-body {
                background-color: var(--bg-secondary);
                color: var(--text-primary);
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

def create_main_layout():
    """Create the main dashboard layout"""
    return html.Div([
        # Header
        html.Div([
            html.H1("Clean AI Trading Dashboard", 
                   style={'color': 'var(--accent-primary)', 'textAlign': 'center', 'marginBottom': '30px'}),
            html.P("Real-time AI-powered trading with maximum performance", 
                   style={'color': 'var(--text-secondary)', 'textAlign': 'center', 'marginBottom': '30px'})
        ]),
        
        # Control Panel
        html.Div([
            html.Div([
                html.H3("Trading Controls", style={'color': 'var(--text-primary)'}),
                html.Div([
                    html.Label("Starting Capital (CAD):", style={'color': 'var(--text-secondary)'}),
                    dcc.Input(
                        id="capital-input",
                        type="number",
                        value=10000,
                        min=1,
                        step=1,
                        placeholder="Enter any amount",
                        className="input-field"
                    )
                ], style={'margin': '10px 0'}),
                html.Div([
                    html.Label("Trading Mode:", style={'color': 'var(--text-secondary)'}),
                    dcc.Dropdown(
                        id="mode-selector",
                        options=[
                            {'label': 'DEMO', 'value': 'DEMO'},
                            {'label': 'LIVE', 'value': 'LIVE'}
                        ],
                        value='DEMO',
                        style={'backgroundColor': 'var(--bg-tertiary)', 'color': 'var(--text-primary)'}
                    )
                ], style={'margin': '10px 0'}),
                html.Button(
                    id="start-stop-button",
                    children="Start AI Trading",
                    className="btn-primary",
                    style={'width': '100%', 'margin': '20px 0'}
                )
            ], className="metric-card", style={'width': '300px'})
        ], style={'display': 'flex', 'justifyContent': 'center', 'marginBottom': '30px'}),
        
        # Status Panel
        html.Div([
            html.Div([
                html.H3("AI System Status", style={'color': 'var(--text-primary)'}),
                html.Div([
                    html.Span("Status: ", style={'color': 'var(--text-secondary)'}),
                    html.Span(id="ai-status-text", children="Idle", className="status-badge status-idle")
                ], style={'margin': '10px 0'}),
                html.Div([
                    html.Span("Session ID: ", style={'color': 'var(--text-secondary)'}),
                    html.Span(id="session-id", children="None", style={'color': 'var(--accent-primary)'})
                ], style={'margin': '10px 0'}),
                html.Div([
                    html.Span("Mode: ", style={'color': 'var(--text-secondary)'}),
                    html.Span(id="current-mode", children="DEMO", style={'color': 'var(--accent-primary)'})
                ], style={'margin': '10px 0'})
            ], className="metric-card", style={'width': '300px'})
        ], style={'display': 'flex', 'justifyContent': 'center', 'marginBottom': '30px'}),
        
        # Metrics Row
        html.Div([
            html.Div([
                html.Div(className="metric-label", children="Account Value"),
                html.Div(id="account-value", className="metric-value", children="$0.00")
            ], className="metric-card"),
            html.Div([
                html.Div(className="metric-label", children="Total P&L"),
                html.Div(id="pnl-value", className="metric-value", children="$0.00")
            ], className="metric-card"),
            html.Div([
                html.Div(className="metric-label", children="P&L %"),
                html.Div(id="pnl-percentage", className="metric-value", children="0.00%")
            ], className="metric-card"),
            html.Div([
                html.Div(className="metric-label", children="Holdings"),
                html.Div(id="holdings-count", className="metric-value", children="0")
            ], className="metric-card"),
            html.Div([
                html.Div(className="metric-label", children="AI Decisions Today"),
                html.Div(id="ai-decisions-today", className="metric-value", children="0")
            ], className="metric-card")
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center', 'marginBottom': '30px'}),
        
        # Charts Row
        html.Div([
            html.Div([
                html.Div([
                    html.H3("Portfolio Value Over Time", style={'color': 'var(--text-primary)'})
                ], className="table-header"),
                dcc.Graph(id="portfolio-chart", style={'height': '400px'})
            ], className="table-container", style={'width': '48%', 'margin': '1%'}),
            html.Div([
                html.Div([
                    html.H3("AI Performance", style={'color': 'var(--text-primary)'})
                ], className="table-header"),
                dcc.Graph(id="ai-performance-chart", style={'height': '400px'})
            ], className="table-container", style={'width': '48%', 'margin': '1%'})
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'marginBottom': '30px'}),
        
        # Tables Row
        html.Div([
            html.Div([
                html.Div([
                    html.H3("Current Holdings", style={'color': 'var(--text-primary)'})
                ], className="table-header"),
                html.Div(id="holdings-table")
            ], className="table-container", style={'width': '48%', 'margin': '1%'}),
            html.Div([
                html.Div([
                    html.H3("Recent Trades", style={'color': 'var(--text-primary)'})
                ], className="table-header"),
                html.Div(id="trades-table")
            ], className="table-container", style={'width': '48%', 'margin': '1%'})
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'marginBottom': '30px'}),
        
        # AI Logs Section
        html.Div([
            html.Div([
                html.H3("AI System Logs & Analytics", style={'color': 'var(--text-primary)'})
            ], className="table-header"),
            html.Div([
                dcc.Tabs(id="logs-tabs", value="activity", children=[
                    dcc.Tab(label="Activity Logs", value="activity"),
                    dcc.Tab(label="Trading Decisions", value="decisions"),
                    dcc.Tab(label="Performance Metrics", value="performance"),
                    dcc.Tab(label="AI Components", value="components")
                ]),
                html.Div(id="logs-content", style={'padding': '20px', 'minHeight': '300px'})
            ])
        ], className="table-container"),
        
        # Hidden components for state management
        dcc.Store(id="ai-trading-active", data=False),
        dcc.Interval(id="interval-component", interval=5000, n_intervals=0)
    ], className="dashboard-container")

# Set the layout
app.layout = create_main_layout()

# Callbacks for updating dashboard components
@app.callback(
    Output('account-value', 'children'),
    Output('pnl-value', 'children'),
    Output('pnl-percentage', 'children'),
    Output('holdings-count', 'children'),
    Output('ai-decisions-today', 'children'),
    Output('portfolio-chart', 'figure'),
    Output('ai-performance-chart', 'figure'),
    Output('holdings-table', 'children'),
    Output('trades-table', 'children'),
    Output('session-id', 'children'),
    Output('current-mode', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_dashboard_data(n):
    """Update all dashboard data"""
    try:
        state = state_manager.get_current_state()
        
        # Account metrics
        account_value = f"${state.get('current_capital', 0):,.2f}"
        total_pnl = state.get('total_pnl', 0)
        pnl_value = f"${total_pnl:,.2f}"
        pnl_percentage = f"{state.get('pnl_percentage', 0):.2f}%"
        
        # Holdings count (only count holdings with shares > 0)
        holdings = state.get('holdings', [])
        active_holdings = [h for h in holdings if h.get('qty', 0) > 0]
        holdings_count = len(active_holdings)
        
        # AI decisions
        ai_decisions = state.get('ai_decisions_today', 0)
        
        # Portfolio chart
        portfolio_history = state.get('portfolio_history', [])
        if portfolio_history:
            portfolio_fig = go.Figure()
            portfolio_fig.add_trace(go.Scatter(
                x=list(range(len(portfolio_history))),
                y=portfolio_history,
                mode='lines',
                name='Portfolio Value',
                line=dict(color='#00d4aa', width=2)
            ))
            portfolio_fig.update_layout(
                title="Portfolio Value Over Time",
                xaxis_title="Time",
                yaxis_title="Value (CAD)",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis=dict(color='white'),
                yaxis=dict(color='white')
            )
        else:
            portfolio_fig = go.Figure()
            portfolio_fig.update_layout(
                title="Portfolio Value Over Time",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
        
        # AI Performance chart
        ai_performance_fig = go.Figure()
        ai_performance_fig.add_trace(go.Bar(
            x=['Decisions Today', 'Total P&L', 'Win Rate'],
            y=[ai_decisions, total_pnl, state.get('win_rate', 0)],
            marker_color=['#00d4aa', '#ff6b6b' if total_pnl < 0 else '#00d4aa', '#00d4aa']
        ))
        ai_performance_fig.update_layout(
            title="AI Performance Metrics",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis=dict(color='white'),
            yaxis=dict(color='white')
        )
        
        # Holdings table
        if active_holdings:
            holdings_data = []
            for holding in active_holdings:
                holdings_data.append({
                    'Symbol': holding.get('symbol', ''),
                    'Quantity': holding.get('qty', 0),
                    'Avg Price': f"${holding.get('avg_price', 0):.2f}",
                    'Current Price': f"${holding.get('current_price', 0):.2f}",
                    'P&L': f"${holding.get('pnl', 0):.2f}",
                    'P&L %': f"{holding.get('pnl_percentage', 0):.2f}%"
                })
            holdings_table = html.Table([
                html.Thead([
                    html.Tr([html.Th(col) for col in ['Symbol', 'Quantity', 'Avg Price', 'Current Price', 'P&L', 'P&L %']])
                ]),
                html.Tbody([
                    html.Tr([html.Td(holdings_data[i][col]) for col in ['Symbol', 'Quantity', 'Avg Price', 'Current Price', 'P&L', 'P&L %']])
                    for i in range(len(holdings_data))
                ])
            ], style={'width': '100%', 'color': 'var(--text-primary)'})
        else:
            holdings_table = html.P("No active holdings", style={'color': 'var(--text-secondary)', 'textAlign': 'center', 'padding': '20px'})
        
        # Trades table
        recent_trades = state.get('recent_trades', [])
        if recent_trades:
            trades_data = []
            for trade in recent_trades[-10:]:  # Show last 10 trades
                trades_data.append({
                    'Time': trade.get('timestamp', ''),
                    'Symbol': trade.get('symbol', ''),
                    'Action': trade.get('action', ''),
                    'Quantity': trade.get('quantity', 0),
                    'Price': f"${trade.get('price', 0):.2f}",
                    'P&L': f"${trade.get('pnl', 0):.2f}"
                })
            trades_table = html.Table([
                html.Thead([
                    html.Tr([html.Th(col) for col in ['Time', 'Symbol', 'Action', 'Quantity', 'Price', 'P&L']])
                ]),
                html.Tbody([
                    html.Tr([html.Td(trades_data[i][col]) for col in ['Time', 'Symbol', 'Action', 'Quantity', 'Price', 'P&L']])
                    for i in range(len(trades_data))
                ])
            ], style={'width': '100%', 'color': 'var(--text-primary)'})
        else:
            trades_table = html.P("No recent trades", style={'color': 'var(--text-secondary)', 'textAlign': 'center', 'padding': '20px'})
        
        # Session info
        session_id = state.get('session_id', 'None')
        current_mode = state.get('mode', 'DEMO')
        
        return (account_value, pnl_value, pnl_percentage, holdings_count, ai_decisions,
                portfolio_fig, ai_performance_fig, holdings_table, trades_table,
                session_id, current_mode)
        
    except Exception as e:
        print(f"Error updating dashboard: {e}")
        return ("$0.00", "$0.00", "0.00%", "0", "0", 
                go.Figure(), go.Figure(), 
                html.P("Error loading holdings", style={'color': 'red'}),
                html.P("Error loading trades", style={'color': 'red'}),
                "Error", "Error")

@app.callback(
    Output('start-stop-button', 'children'),
    Output('ai-status-text', 'children'),
    Output('ai-status-text', 'className'),
    Input('start-stop-button', 'n_clicks'),
    State('capital-input', 'value'),
    State('mode-selector', 'value'),
    prevent_initial_call=True
)
def start_ai_trading(n_clicks, capital, mode):
    """Start or stop AI trading"""
    if n_clicks is None:
        raise PreventUpdate
    
    try:
        if state_manager.current_session and state_manager.current_session.is_active:
            # Stop AI trading
            ai_engine.stop_maximum_power_trading()
            state_manager.current_session.is_active = False
            state_manager.save()
            advanced_logger.log_component_activity("AI Engine", "Stopped", {"status": "AI trading gracefully terminated"})
            return "Start AI Trading", "Idle", "status-badge status-idle"
        else:
            # Start new session
            session_id = state_manager.start_new_session(capital or 10000, mode or 'DEMO')
            advanced_logger.clear_logs()
            
            # Start AI trading in background
            def ai_trading_loop():
                try:
                    ai_engine.start_maximum_power_trading()
                    advanced_logger.log_component_activity("AI Engine", "Started", {"status": "AI trading loop initiated"})
                except Exception as e:
                    advanced_logger.log_component_activity("AI Engine", "Error", {"error": f"Failed to start: {e}"})
            
            thread = threading.Thread(target=ai_trading_loop, daemon=True)
            thread.start()
            
            return "Stop AI Trading", "Active", "status-badge status-active"
            
    except Exception as e:
        advanced_logger.log_component_activity("AI Engine", "Error", {"error": f"Failed to start AI trading: {e}"})
        return "Start AI Trading", "Error", "status-badge status-error"

@app.callback(
    Output('logs-content', 'children'),
    Input('logs-tabs', 'value'),
    Input('interval-component', 'n_intervals')
)
def update_logs_content(active_tab, n_intervals):
    """Update logs content based on active tab"""
    try:
        if active_tab == "activity":
            activities = advanced_logger.get_recent_activity(10)
            if activities:
                content = html.Div([
                    html.Div([
                        html.Strong(f"{activity['timestamp']} - {activity['component']}: {activity['action']}"),
                        html.Br(),
                        html.Span(activity.get('details', {}).get('status', ''), style={'color': 'var(--text-secondary)'})
                    ], style={'padding': '10px', 'borderBottom': '1px solid var(--border-color)'})
                    for activity in activities
                ])
            else:
                content = html.P("No activity logs available", style={'color': 'var(--text-secondary)'})
        
        elif active_tab == "decisions":
            decisions = advanced_logger.get_recent_decisions(10)
            if decisions:
                content = html.Div([
                    html.Div([
                        html.Strong(f"{decision['timestamp']} - {decision['symbol']}: {decision['action']}"),
                        html.Br(),
                        html.Span(f"Confidence: {decision.get('confidence', 0):.2f}", style={'color': 'var(--text-secondary)'})
                    ], style={'padding': '10px', 'borderBottom': '1px solid var(--border-color)'})
                    for decision in decisions
                ])
            else:
                content = html.P("No trading decisions available", style={'color': 'var(--text-secondary)'})
        
        elif active_tab == "performance":
            performance = advanced_logger.get_performance_summary()
            if performance:
                content = html.Div([
                    html.Div([
                        html.Strong(f"{metric['component']}: {metric['value']}"),
                        html.Br(),
                        html.Span(metric.get('description', ''), style={'color': 'var(--text-secondary)'})
                    ], style={'padding': '10px', 'borderBottom': '1px solid var(--border-color)'})
                    for metric in performance
                ])
            else:
                content = html.P("No performance metrics available", style={'color': 'var(--text-secondary)'})
        
        elif active_tab == "components":
            # Show AI component status
            components = [
                {"name": "Master Orchestrator", "status": "Active" if ai_engine.orchestrator else "Inactive"},
                {"name": "Maximum Power Engine", "status": "Active" if ai_engine.running else "Inactive"},
                {"name": "State Manager", "status": "Active" if state_manager.current_session else "Inactive"},
                {"name": "Advanced Logger", "status": "Active"},
                {"name": "Data Pipeline", "status": "Active"},
                {"name": "Risk Manager", "status": "Active"}
            ]
            content = html.Div([
                html.Div([
                    html.Strong(comp['name']),
                    html.Span(f" - {comp['status']}", 
                             style={'color': 'var(--accent-primary)' if comp['status'] == 'Active' else 'var(--accent-secondary)'})
                ], style={'padding': '10px', 'borderBottom': '1px solid var(--border-color)'})
                for comp in components
            ])
        
        return content
        
    except Exception as e:
        print(f"Error updating logs: {e}")
        return html.P(f"Error loading logs: {e}", style={'color': 'red'})

if __name__ == '__main__':
    print("=" * 80)
    print("Clean AI Trading Dashboard Starting...")
    print("=" * 80)
    print("Features:")
    print("   - Clean state management")
    print("   - Maximum power AI engine")
    print("   - Advanced logging system")
    print("   - Real-time dashboard updates")
    print("   - No Unicode/emoji issues")
    print("   - Fixed method names")
    print("   - Proper orchestrator initialization")
    print("Dashboard URL: http://localhost:8061")
    print("Press Ctrl+C to stop the server")
    print("=" * 80)
    
    app.run_server(debug=False, host='0.0.0.0', port=8061)
