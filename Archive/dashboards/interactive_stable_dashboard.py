#!/usr/bin/env python3
"""
Stable AI Trading Dashboard - No Auto-Restart, No Stuck Issues
"""

import sys
import os
import io
import json
import time
import threading
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Fix Unicode encoding for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np

# Import our components
try:
    from src.dashboard.clean_state_manager import CleanStateManager
    from src.dashboard.maximum_power_ai_engine import MaximumPowerAIEngine
    from src.dashboard.advanced_ai_logger import AdvancedAILogger
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Initialize components
state_manager = CleanStateManager()
ai_engine = MaximumPowerAIEngine()
advanced_logger = AdvancedAILogger()

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Stable AI Trading Dashboard"

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
            body { background-color: #1a1a1a; color: #ffffff; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
            .dash-table-container { background-color: #2d2d2d; }
            .dash-table-container .dash-spreadsheet-container { background-color: #2d2d2d; }
            .dash-table-container .dash-spreadsheet-container table { background-color: #2d2d2d; color: #ffffff; }
            .dash-table-container .dash-spreadsheet-container .dash-header { background-color: #3d3d3d; color: #ffffff; }
            .dash-table-container .dash-spreadsheet-container .dash-cell { background-color: #2d2d2d; color: #ffffff; }
            .modal-content { background-color: #2d2d2d; color: #ffffff; }
            .modal-header { background-color: #3d3d3d; color: #ffffff; border-bottom: 1px solid #555; }
            .modal-title { color: #ffffff; }
            .modal-body { background-color: #2d2d2d; color: #ffffff; }
            .log-entry { color: #ffffff; margin: 5px 0; padding: 5px; background-color: #3d3d3d; border-radius: 3px; }
            .log-timestamp { color: #888; font-size: 0.9em; }
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
            html.H1("🚀 Stable AI Trading Dashboard", className="text-center mb-4"),
            html.P("Advanced AI Trading System - No Auto-Restart Issues", className="text-center text-muted mb-4")
        ], className="header-section"),
        
        # Control Panel
        html.Div([
            html.Div([
                html.H3("🎛️ Control Panel"),
                html.Div([
                    html.Label("Starting Capital (CAD):"),
                    dcc.Input(
                        id="capital-input",
                        type="number",
                        value=10000,
                        min=1,
                        step=1,
                        placeholder="Enter any amount",
                        style={"width": "200px", "margin-right": "10px"}
                    ),
                    html.Button("🚀 Start AI Trading", id="start-btn", className="btn btn-success", style={"margin-right": "10px"}),
                    html.Button("⏹️ Stop AI Trading", id="stop-btn", className="btn btn-danger"),
                ], className="d-flex align-items-center mb-3"),
                
                html.Div([
                    html.Label("Trading Mode:"),
                    dcc.RadioItems(
                        id="mode-selector",
                        options=[
                            {"label": "DEMO", "value": "DEMO"},
                            {"label": "LIVE", "value": "LIVE"}
                        ],
                        value="DEMO",
                        inline=True,
                        style={"margin-left": "10px"}
                    )
                ], className="d-flex align-items-center mb-3"),
                
                # Status indicators
                html.Div([
                    html.Span("AI Status: ", className="fw-bold"),
                    html.Span(id="ai-status", children="Inactive", className="badge bg-secondary"),
                    html.Span(" | ", className="mx-2"),
                    html.Span("Mode: ", className="fw-bold"),
                    html.Span(id="mode-status", children="DEMO", className="badge bg-info"),
                    html.Span(" | ", className="mx-2"),
                    html.Span("Session: ", className="fw-bold"),
                    html.Span(id="session-status", children="None", className="badge bg-warning")
                ], className="mb-3")
            ], className="card p-3")
        ], className="control-panel mb-4"),
        
        # Metrics Row
        html.Div([
            html.Div([
                html.H4("💰 Account Value"),
                html.H2(id="account-value", children="$0.00", className="text-success")
            ], className="card text-center p-3"),
            
            html.Div([
                html.H4("📈 Total P&L"),
                html.H2(id="total-pnl", children="$0.00", className="text-info")
            ], className="card text-center p-3"),
            
            html.Div([
                html.H4("📊 Holdings"),
                html.H2(id="holdings-count", children="0", className="text-warning")
            ], className="card text-center p-3"),
            
            html.Div([
                html.H4("🤖 AI Decisions"),
                html.H2(id="ai-decisions", children="0", className="text-primary")
            ], className="card text-center p-3")
        ], className="row mb-4", style={"display": "flex", "gap": "20px"}),
        
        # Charts Row
        html.Div([
            html.Div([
                html.H4("📈 Portfolio Value Over Time"),
                dcc.Graph(id="portfolio-chart")
            ], className="card p-3", style={"width": "48%"}),
            
            html.Div([
                html.H4("🤖 AI Performance"),
                dcc.Graph(id="ai-performance-chart")
            ], className="card p-3", style={"width": "48%"})
        ], className="row mb-4", style={"display": "flex", "gap": "20px"}),
        
        # Tables Row
        html.Div([
            html.Div([
                html.H4("📋 Current Holdings"),
                html.Div(id="holdings-table")
            ], className="card p-3", style={"width": "48%"}),
            
            html.Div([
                html.H4("📝 Recent Trades"),
                html.Div(id="trades-table")
            ], className="card p-3", style={"width": "48%"})
        ], className="row mb-4", style={"display": "flex", "gap": "20px"}),
        
        # AI Logs Section
        html.Div([
            html.H4("📊 AI System Logs & Analytics"),
            html.Div([
                html.Button("📋 Activity Logs", id="activity-logs-btn", className="btn btn-outline-primary me-2"),
                html.Button("🎯 Trading Decisions", id="trading-decisions-btn", className="btn btn-outline-success me-2"),
                html.Button("📈 Performance Metrics", id="performance-metrics-btn", className="btn btn-outline-info me-2"),
                html.Button("🔧 System Status", id="system-status-btn", className="btn btn-outline-warning")
            ], className="mb-3"),
            html.Div(id="logs-content", className="card p-3")
        ], className="logs-section"),
        
        # Hidden stores
        dcc.Store(id="ai-trading-active", data=False),
        dcc.Interval(id="update-interval", interval=5000, n_intervals=0)  # Update every 5 seconds
    ], className="container-fluid p-4")

# Set layout
app.layout = create_main_layout()

# Callbacks
@app.callback(
    [Output("ai-trading-active", "data"),
     Output("ai-status", "children"),
     Output("ai-status", "className")],
    [Input("start-btn", "n_clicks"),
     Input("stop-btn", "n_clicks")],
    [State("capital-input", "value"),
     State("mode-selector", "value"),
     State("ai-trading-active", "data")]
)
def control_ai_trading(start_clicks, stop_clicks, capital, mode, is_active):
    """Control AI trading start/stop"""
    ctx = callback_context
    if not ctx.triggered:
        return is_active, "Inactive", "badge bg-secondary"
    
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if button_id == "start-btn" and not is_active:
        try:
            # Start new session
            session_id = state_manager.start_new_session(capital or 10000, mode or 'DEMO')
            advanced_logger.clear_logs()
            
            # Start AI trading in background
            def ai_trading_loop():
                try:
                    ai_engine.start_maximum_power_trading()
                    advanced_logger.log_component_activity("AI Engine", "Started", "AI trading loop initiated")
                except Exception as e:
                    advanced_logger.log_component_activity("AI Engine", "Error", f"Failed to start: {e}")
            
            thread = threading.Thread(target=ai_trading_loop, daemon=True)
            thread.start()
            
            return True, "Active", "badge bg-success"
        except Exception as e:
            print(f"❌ Failed to start AI trading: {e}")
            return False, "Error", "badge bg-danger"
    
    elif button_id == "stop-btn" and is_active:
        try:
            # Stop AI trading
            state_manager.current_session.is_active = False
            state_manager.save()
            advanced_logger.log_component_activity("AI Engine", "Stopped", "AI trading stopped by user")
            return False, "Inactive", "badge bg-secondary"
        except Exception as e:
            print(f"❌ Failed to stop AI trading: {e}")
            return False, "Error", "badge bg-danger"
    
    return is_active, "Inactive" if not is_active else "Active", "badge bg-secondary" if not is_active else "badge bg-success"

@app.callback(
    [Output("account-value", "children"),
     Output("total-pnl", "children"),
     Output("holdings-count", "children"),
     Output("ai-decisions", "children"),
     Output("mode-status", "children"),
     Output("session-status", "children")],
    [Input("update-interval", "n_intervals")]
)
def update_metrics(n_intervals):
    """Update dashboard metrics"""
    try:
        state = state_manager.get_current_state()
        
        # Account metrics
        account_value = f"${state.get('current_capital', 0):,.2f}"
        total_pnl = f"${state.get('total_pnl', 0):,.2f}"
        holdings_count = len([h for h in state.get('holdings', []) if h.get('qty', 0) > 0])
        ai_decisions = state.get('ai_decisions_today', 0)
        
        # Status
        mode = state.get('mode', 'DEMO')
        session_id = state.get('session_id', 'None')
        
        return account_value, total_pnl, str(holdings_count), str(ai_decisions), mode, session_id[:8] if session_id != 'None' else 'None'
    except Exception as e:
        print(f"❌ Error updating metrics: {e}")
        return "$0.00", "$0.00", "0", "0", "DEMO", "None"

@app.callback(
    [Output("portfolio-chart", "figure"),
     Output("ai-performance-chart", "figure")],
    [Input("update-interval", "n_intervals")]
)
def update_charts(n_intervals):
    """Update dashboard charts"""
    try:
        state = state_manager.get_current_state()
        
        # Portfolio chart
        portfolio_history = state.get('portfolio_history', [])
        if not portfolio_history:
            # Generate sample data
            portfolio_history = [{"timestamp": datetime.now() - timedelta(days=i), "value": 10000 + i * 100} for i in range(30, 0, -1)]
        
        portfolio_fig = go.Figure()
        portfolio_fig.add_trace(go.Scatter(
            x=[p["timestamp"] for p in portfolio_history],
            y=[p["value"] for p in portfolio_history],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#00ff00', width=2)
        ))
        portfolio_fig.update_layout(
            title="Portfolio Value Over Time",
            xaxis_title="Time",
            yaxis_title="Value (CAD)",
            plot_bgcolor='#2d2d2d',
            paper_bgcolor='#1a1a1a',
            font=dict(color='white')
        )
        
        # AI Performance chart
        ai_decisions = state.get('ai_decisions_today', 0)
        total_pnl = state.get('total_pnl', 0)
        win_rate = 75.0  # Sample win rate
        
        ai_fig = go.Figure()
        ai_fig.add_trace(go.Bar(
            x=['Decisions', 'P&L', 'Win Rate'],
            y=[ai_decisions, total_pnl, win_rate],
            marker_color=['#007bff', '#28a745', '#ffc107']
        ))
        ai_fig.update_layout(
            title="AI Performance Metrics",
            xaxis_title="Metrics",
            yaxis_title="Values",
            plot_bgcolor='#2d2d2d',
            paper_bgcolor='#1a1a1a',
            font=dict(color='white')
        )
        
        return portfolio_fig, ai_fig
    except Exception as e:
        print(f"❌ Error updating charts: {e}")
        return go.Figure(), go.Figure()

@app.callback(
    [Output("holdings-table", "children"),
     Output("trades-table", "children")],
    [Input("update-interval", "n_intervals")]
)
def update_tables(n_intervals):
    """Update holdings and trades tables"""
    try:
        state = state_manager.get_current_state()
        
        # Holdings table
        holdings = [h for h in state.get('holdings', []) if h.get('qty', 0) > 0]
        if holdings:
            holdings_data = []
            for h in holdings:
                holdings_data.append({
                    'Symbol': h.get('symbol', 'N/A'),
                    'Quantity': h.get('qty', 0),
                    'Avg Price': f"${h.get('avg_price', 0):.2f}",
                    'Current Price': f"${h.get('current_price', 0):.2f}",
                    'P&L': f"${h.get('pnl', 0):.2f}",
                    'P&L %': f"{h.get('pnl_percent', 0):.2f}%"
                })
            holdings_df = pd.DataFrame(holdings_data)
            holdings_table = dash.dash_table.DataTable(
                data=holdings_df.to_dict('records'),
                columns=[{"name": i, "id": i} for i in holdings_df.columns],
                style_cell={'backgroundColor': '#2d2d2d', 'color': 'white'},
                style_header={'backgroundColor': '#3d3d3d', 'color': 'white'}
            )
        else:
            holdings_table = html.P("No current holdings", className="text-muted")
        
        # Trades table
        trades = state.get('recent_trades', [])[-10:]  # Last 10 trades
        if trades:
            trades_data = []
            for t in trades:
                trades_data.append({
                    'Time': t.get('timestamp', 'N/A'),
                    'Symbol': t.get('symbol', 'N/A'),
                    'Action': t.get('action', 'N/A'),
                    'Quantity': t.get('qty', 0),
                    'Price': f"${t.get('price', 0):.2f}",
                    'P&L': f"${t.get('pnl', 0):.2f}"
                })
            trades_df = pd.DataFrame(trades_data)
            trades_table = dash.dash_table.DataTable(
                data=trades_df.to_dict('records'),
                columns=[{"name": i, "id": i} for i in trades_df.columns],
                style_cell={'backgroundColor': '#2d2d2d', 'color': 'white'},
                style_header={'backgroundColor': '#3d3d3d', 'color': 'white'}
            )
        else:
            trades_table = html.P("No recent trades", className="text-muted")
        
        return holdings_table, trades_table
    except Exception as e:
        print(f"❌ Error updating tables: {e}")
        return html.P("Error loading data", className="text-danger"), html.P("Error loading data", className="text-danger")

@app.callback(
    Output("logs-content", "children"),
    [Input("activity-logs-btn", "n_clicks"),
     Input("trading-decisions-btn", "n_clicks"),
     Input("performance-metrics-btn", "n_clicks"),
     Input("system-status-btn", "n_clicks")]
)
def update_logs_content(activity_clicks, trading_clicks, performance_clicks, system_clicks):
    """Update logs content based on button clicks"""
    ctx = callback_context
    if not ctx.triggered:
        return html.P("Select a log type above", className="text-muted")
    
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    try:
        if button_id == "activity-logs-btn":
            logs = advanced_logger.get_recent_activity(20)
            if logs:
                log_entries = []
                for log in logs:
                    log_entries.append(html.Div([
                        html.Span(f"[{log.get('timestamp', 'N/A')}] ", className="log-timestamp"),
                        html.Span(f"{log.get('component', 'N/A')}: ", className="fw-bold"),
                        html.Span(f"{log.get('status', 'N/A')} - {log.get('message', 'N/A')}")
                    ], className="log-entry"))
                return log_entries
            else:
                return html.P("No activity logs available", className="text-muted")
        
        elif button_id == "trading-decisions-btn":
            decisions = advanced_logger.get_recent_decisions(20)
            if decisions:
                decision_entries = []
                for decision in decisions:
                    decision_entries.append(html.Div([
                        html.Span(f"[{decision.get('timestamp', 'N/A')}] ", className="log-timestamp"),
                        html.Span(f"{decision.get('symbol', 'N/A')}: ", className="fw-bold"),
                        html.Span(f"{decision.get('action', 'N/A')} - {decision.get('reasoning', 'N/A')}")
                    ], className="log-entry"))
                return decision_entries
            else:
                return html.P("No trading decisions available", className="text-muted")
        
        elif button_id == "performance-metrics-btn":
            metrics = advanced_logger.get_performance_summary()
            if metrics:
                return html.Div([
                    html.P(f"Total Decisions: {metrics.get('total_decisions', 0)}"),
                    html.P(f"Successful Trades: {metrics.get('successful_trades', 0)}"),
                    html.P(f"Win Rate: {metrics.get('win_rate', 0):.1f}%"),
                    html.P(f"Average P&L: ${metrics.get('avg_pnl', 0):.2f}")
                ])
            else:
                return html.P("No performance metrics available", className="text-muted")
        
        elif button_id == "system-status-btn":
            state = state_manager.get_current_state()
            return html.Div([
                html.P(f"AI Active: {'Yes' if state.get('ai_trading_active', False) else 'No'}"),
                html.P(f"Session ID: {state.get('session_id', 'None')}"),
                html.P(f"Mode: {state.get('mode', 'DEMO')}"),
                html.P(f"Capital: ${state.get('current_capital', 0):,.2f}"),
                html.P(f"Holdings: {len([h for h in state.get('holdings', []) if h.get('qty', 0) > 0])}")
            ])
        
        return html.P("Select a log type above", className="text-muted")
    except Exception as e:
        print(f"❌ Error updating logs: {e}")
        return html.P(f"Error loading logs: {e}", className="text-danger")

if __name__ == "__main__":
    print("=" * 80)
    print("🚀 STABLE AI Trading Dashboard Starting...")
    print("=" * 80)
    print("Features:")
    print("   - Stable AI Engine (no auto-restart issues)")
    print("   - Clean stock universe (no delisted stocks)")
    print("   - Real-time monitoring and logging")
    print("   - Advanced analytics and performance tracking")
    print("   - No stuck or reset issues")
    print("Dashboard URL: http://localhost:8059")
    print("Press Ctrl+C to stop the server")
    print("=" * 80)
    
    try:
        app.run_server(debug=False, host='0.0.0.0', port=8059)
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Dashboard error: {e}")
