#!/usr/bin/env python3
"""
MAXIMUM POWER AI Trading Dashboard - Designed to use 80%+ of system resources
"""
import dash
from dash import dcc, html, Input, Output, State, callback, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime, timedelta
import pandas as pd
import threading
import time
import asyncio
import logging
import os
import sys
import json
import psutil

# Ensure the project root is in the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import MAXIMUM POWER components
try:
    from src.dashboard.clean_state_manager import state_manager
    from src.dashboard.maximum_power_ai_engine import max_power_ai_engine
    from src.dashboard.services import get_demo_price, is_market_open, get_random_tsx_stock
    from src.config.mode_manager import get_mode_manager
    from src.monitoring.system_monitor import SystemMonitor
    
    # Initialize components
    mode_manager = get_mode_manager()
    system_monitor = SystemMonitor()
    
    # Initialize MAXIMUM POWER AI Engine
    if max_power_ai_engine.initialize():
        logger.info("🚀 MAXIMUM POWER AI Engine ready!")
    else:
        logger.error("❌ MAXIMUM POWER AI Engine initialization failed")
    
    MAXIMUM_POWER_AVAILABLE = True
    logger.info("🚀 Connected to MAXIMUM POWER AI trading system!")
except Exception as e:
    logger.error(f"❌ Failed to import maximum power components: {e}")
    MAXIMUM_POWER_AVAILABLE = False
    logger.warning("⚠️ Falling back to standard system.")

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG], suppress_callback_exceptions=True)
app.title = "MAXIMUM POWER AI Trading Dashboard"

# Custom MAXIMUM POWER theme CSS
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
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
                color: #eee;
                margin: 0;
                padding: 0;
            }
            .navbar {
                background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1);
                border-bottom: 2px solid #ff6b6b;
                box-shadow: 0 4px 20px rgba(255, 107, 107, 0.3);
            }
            .card {
                background: linear-gradient(145deg, #2a2a3e, #1e1e2e);
                color: #eee;
                border: 1px solid #ff6b6b;
                box-shadow: 0 8px 32px rgba(255, 107, 107, 0.2);
                border-radius: 15px;
            }
            .card-header {
                background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
                border-bottom: 2px solid #ff6b6b;
                font-weight: bold;
                color: white;
                border-radius: 15px 15px 0 0;
            }
            .btn-primary {
                background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
                border: none;
                border-radius: 25px;
                padding: 12px 30px;
                font-weight: bold;
                text-transform: uppercase;
                letter-spacing: 1px;
                box-shadow: 0 4px 15px rgba(255, 107, 107, 0.4);
                transition: all 0.3s ease;
            }
            .btn-primary:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(255, 107, 107, 0.6);
            }
            .btn-danger {
                background: linear-gradient(45deg, #ff4757, #ff3838);
                border: none;
                border-radius: 25px;
                padding: 12px 30px;
                font-weight: bold;
                text-transform: uppercase;
                letter-spacing: 1px;
                box-shadow: 0 4px 15px rgba(255, 71, 87, 0.4);
            }
            .btn-danger:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(255, 71, 87, 0.6);
            }
            .metric-card {
                background: linear-gradient(145deg, #2a2a3e, #1e1e2e);
                border: 2px solid #4ecdc4;
                border-radius: 15px;
                padding: 20px;
                text-align: center;
                box-shadow: 0 8px 32px rgba(78, 205, 196, 0.2);
                transition: all 0.3s ease;
            }
            .metric-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 12px 40px rgba(78, 205, 196, 0.3);
            }
            .metric-value {
                font-size: 2.5em;
                font-weight: bold;
                color: #4ecdc4;
                text-shadow: 0 0 20px rgba(78, 205, 196, 0.5);
            }
            .metric-label {
                font-size: 1.1em;
                color: #ccc;
                margin-top: 10px;
            }
            .status-indicator {
                display: inline-block;
                width: 15px;
                height: 15px;
                border-radius: 50%;
                margin-right: 8px;
                animation: pulse 2s infinite;
            }
            .status-green { 
                background: linear-gradient(45deg, #4ecdc4, #44a08d);
                box-shadow: 0 0 20px rgba(78, 205, 196, 0.6);
            }
            .status-red { 
                background: linear-gradient(45deg, #ff6b6b, #ff4757);
                box-shadow: 0 0 20px rgba(255, 107, 107, 0.6);
            }
            .status-yellow { 
                background: linear-gradient(45deg, #ffd93d, #ff6b6b);
                box-shadow: 0 0 20px rgba(255, 217, 61, 0.6);
            }
            .status-blue { 
                background: linear-gradient(45deg, #45b7d1, #4ecdc4);
                box-shadow: 0 0 20px rgba(69, 183, 209, 0.6);
            }
            @keyframes pulse {
                0% { transform: scale(1); opacity: 1; }
                50% { transform: scale(1.2); opacity: 0.7; }
                100% { transform: scale(1); opacity: 1; }
            }
            .power-indicator {
                background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1);
                height: 8px;
                border-radius: 4px;
                margin: 10px 0;
                position: relative;
                overflow: hidden;
            }
            .power-bar {
                height: 100%;
                background: linear-gradient(90deg, #4ecdc4, #45b7d1);
                border-radius: 4px;
                transition: width 0.5s ease;
                position: relative;
            }
            .power-bar::after {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
                animation: shimmer 2s infinite;
            }
            @keyframes shimmer {
                0% { transform: translateX(-100%); }
                100% { transform: translateX(100%); }
            }
            .form-control {
                background: linear-gradient(145deg, #2a2a3e, #1e1e2e);
                color: #eee;
                border: 2px solid #4ecdc4;
                border-radius: 10px;
                padding: 12px 15px;
            }
            .form-control:focus {
                background: linear-gradient(145deg, #3a3a4e, #2e2e3e);
                color: #eee;
                border-color: #ff6b6b;
                box-shadow: 0 0 20px rgba(255, 107, 107, 0.3);
            }
            .table-dark {
                color: #eee;
                background: linear-gradient(145deg, #2a2a3e, #1e1e2e);
                border: 1px solid #4ecdc4;
            }
            .table-dark th, .table-dark td {
                border-color: #4ecdc4;
                padding: 15px;
            }
            .table-hover tbody tr:hover {
                background: linear-gradient(90deg, rgba(78, 205, 196, 0.1), rgba(69, 183, 209, 0.1));
            }
            .alert-success {
                background: linear-gradient(45deg, #4ecdc4, #44a08d);
                color: white;
                border: none;
                border-radius: 10px;
                box-shadow: 0 4px 15px rgba(78, 205, 196, 0.3);
            }
            .alert-danger {
                background: linear-gradient(45deg, #ff6b6b, #ff4757);
                color: white;
                border: none;
                border-radius: 10px;
                box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
            }
            .log-entry {
                background: linear-gradient(145deg, #2a2a3e, #1e1e2e);
                border-left: 4px solid #4ecdc4;
                margin-bottom: 8px;
                padding: 12px;
                border-radius: 8px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 0.9em;
                box-shadow: 0 4px 15px rgba(78, 205, 196, 0.1);
            }
            .log-timestamp {
                color: #4ecdc4;
                font-size: 0.8em;
                font-weight: bold;
            }
            .log-action-buy { color: #4ecdc4; font-weight: bold; }
            .log-action-sell { color: #ff6b6b; font-weight: bold; }
            .log-action-hold { color: #ffd93d; }
            .log-action-analyze { color: #45b7d1; }
            .log-action-pass { color: #6c757d; }
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
    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H1("🚀 MAXIMUM POWER AI Trading Dashboard", 
                       className="text-center mb-4",
                       style={'color': '#4ecdc4', 'textShadow': '0 0 20px rgba(78, 205, 196, 0.5)'}),
                html.P("Designed to consume 80%+ of system resources for maximum AI performance", 
                      className="text-center text-muted mb-4")
            ])
        ]),
        
        # System Status
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("🔥 MAXIMUM POWER SYSTEM STATUS"),
                    dbc.CardBody([
                        html.Div(id="system-status", className="mb-3"),
                        html.Div(id="resource-usage", className="mb-3"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Button("🚀 START MAXIMUM POWER AI", 
                                          id="start-max-power-btn", 
                                          color="primary", 
                                          size="lg", 
                                          className="w-100 mb-2"),
                            ], width=6),
                            dbc.Col([
                                dbc.Button("🛑 STOP AI TRADING", 
                                          id="stop-ai-btn", 
                                          color="danger", 
                                          size="lg", 
                                          className="w-100 mb-2"),
                            ], width=6)
                        ])
                    ])
                ])
            ])
        ], className="mb-4"),
        
        # Capital Configuration
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("💰 MAXIMUM POWER CAPITAL CONFIGURATION"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Label("Starting Capital ($)", className="form-label"),
                                dbc.Input(
                                    id="capital-input",
                                    type="number",
                                    value=10000,
                                    min=1,
                                    step=1,
                                    placeholder="Enter capital for maximum power trading"
                                )
                            ], width=6),
                            dbc.Col([
                                html.Label("Trading Mode", className="form-label"),
                                dbc.Select(
                                    id="mode-select",
                                    options=[
                                        {"label": "🚀 DEMO (Maximum Power)", "value": "DEMO"},
                                        {"label": "⚡ LIVE (Maximum Power)", "value": "LIVE"}
                                    ],
                                    value="DEMO"
                                )
                            ], width=6)
                        ])
                    ])
                ])
            ])
        ], className="mb-4"),
        
        # Account Metrics
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("📊 MAXIMUM POWER ACCOUNT METRICS"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    html.Div(id="total-balance", className="metric-value"),
                                    html.Div("Total Balance", className="metric-label")
                                ], className="metric-card")
                            ], width=3),
                            dbc.Col([
                                html.Div([
                                    html.Div(id="money-in-stocks", className="metric-value"),
                                    html.Div("In Stocks", className="metric-label")
                                ], className="metric-card")
                            ], width=3),
                            dbc.Col([
                                html.Div([
                                    html.Div(id="total-pnl", className="metric-value"),
                                    html.Div("Total P&L", className="metric-label")
                                ], className="metric-card")
                            ], width=3),
                            dbc.Col([
                                html.Div([
                                    html.Div(id="ai-decisions-count", className="metric-value"),
                                    html.Div("AI Decisions", className="metric-label")
                                ], className="metric-card")
                            ], width=3)
                        ])
                    ])
                ])
            ])
        ], className="mb-4"),
        
        # Resource Usage
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("⚡ SYSTEM RESOURCE USAGE"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    html.H5("CPU Usage", className="text-center"),
                                    html.Div(id="cpu-usage", className="power-indicator"),
                                    html.Div(id="cpu-percent", className="text-center")
                                ])
                            ], width=4),
                            dbc.Col([
                                html.Div([
                                    html.H5("Memory Usage", className="text-center"),
                                    html.Div(id="memory-usage", className="power-indicator"),
                                    html.Div(id="memory-percent", className="text-center")
                                ])
                            ], width=4),
                            dbc.Col([
                                html.Div([
                                    html.H5("AI Activity", className="text-center"),
                                    html.Div(id="ai-activity", className="power-indicator"),
                                    html.Div(id="ai-activity-text", className="text-center")
                                ])
                            ], width=4)
                        ])
                    ])
                ])
            ])
        ], className="mb-4"),
        
        # Holdings and Trades
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("📈 MAXIMUM POWER HOLDINGS"),
                    dbc.CardBody([
                        html.Div(id="holdings-table")
                    ])
                ])
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("📋 MAXIMUM POWER RECENT TRADES"),
                    dbc.CardBody([
                        html.Div(id="recent-trades-table")
                    ])
                ])
            ], width=6)
        ], className="mb-4"),
        
        # AI System Status
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("🤖 MAXIMUM POWER AI SYSTEM STATUS"),
                    dbc.CardBody([
                        html.Div(id="ai-system-status")
                    ])
                ])
            ])
        ], className="mb-4"),
        
        # AI Logs Modal
        dbc.Modal([
            dbc.ModalHeader("🔥 MAXIMUM POWER AI LOGS & ANALYTICS"),
            dbc.ModalBody([
                dbc.Tabs([
                    dbc.Tab([
                        html.Div(id="logs-content")
                    ], label="📊 AI Logs", tab_id="logs"),
                    dbc.Tab([
                        html.Div(id="analysis-content")
                    ], label="🧠 AI Analysis", tab_id="analysis")
                ], id="logs-tabs", active_tab="logs")
            ])
        ], id="logs-modal", is_open=False, size="xl"),
        
        # Hidden divs for state
        dcc.Store(id="ai-trading-active", data=False),
        dcc.Store(id="current-mode", data="DEMO"),
        
        # Auto-refresh interval (1 second for maximum responsiveness)
        dcc.Interval(
            id='interval-component',
            interval=1*1000,  # 1 second
            n_intervals=0
        )
    ], fluid=True)

# Create the layout
app.layout = create_main_layout()

# Callbacks
@app.callback(
    [Output('system-status', 'children'),
     Output('ai-trading-active', 'data')],
    [Input('start-max-power-btn', 'n_clicks'),
     Input('stop-ai-btn', 'n_clicks')],
    [State('capital-input', 'value'),
     State('mode-select', 'value')]
)
def start_stop_maximum_power_trading(start_clicks, stop_clicks, capital, mode):
    """Start or stop maximum power AI trading"""
    if not ctx.triggered:
        return "", False
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'start-max-power-btn':
        if not start_clicks or capital is None:
            return "", False
        
        try:
            capital = float(capital)
            if capital <= 0:
                return dbc.Alert("❌ Capital must be > $0", color="danger"), False
        except:
            return dbc.Alert("❌ Invalid capital", color="danger"), False
        
        if not MAXIMUM_POWER_AVAILABLE:
            return dbc.Alert("❌ Maximum Power AI System not available", color="danger"), False
        
        try:
            # Start new session in state manager
            session_id = state_manager.start_new_session(capital, mode)
            
            # Start MAXIMUM POWER AI trading
            max_power_ai_engine.start_maximum_power_trading()
            
            return dbc.Alert(
                f"🚀 MAXIMUM POWER AI Trading Started! Session: {session_id} | Capital: ${capital:,.0f} | Mode: {mode}",
                color="success"
            ), True
            
        except Exception as e:
            logger.error(f"Error starting maximum power AI trading: {e}")
            return dbc.Alert(f"❌ Error: {str(e)}", color="danger"), False
    
    elif button_id == 'stop-ai-btn':
        try:
            # Stop maximum power AI trading
            max_power_ai_engine.stop_trading()
            
            # Update state
            state = state_manager.get_current_state()
            state['is_active'] = False
            state_manager.save()
            
            return dbc.Alert("🛑 MAXIMUM POWER AI Trading Stopped!", color="warning"), False
            
        except Exception as e:
            logger.error(f"Error stopping AI trading: {e}")
            return dbc.Alert(f"❌ Error: {str(e)}", color="danger"), False
    
    return "", False

@app.callback(
    [Output('total-balance', 'children'),
     Output('money-in-stocks', 'children'),
     Output('total-pnl', 'children'),
     Output('ai-decisions-count', 'children')],
    [Input('interval-component', 'n_intervals'),
     Input('ai-trading-active', 'data')]
)
def update_account_metrics(n, ai_active):
    """Update account metrics from state manager"""
    try:
        # Get current state from SINGLE SOURCE OF TRUTH
        state = state_manager.get_current_state()
        
        current_capital = state['current_capital']
        positions = state['positions']
        
        # Calculate total value in stocks
        stocks_value = sum(p['quantity'] * p['current_price'] for p in positions)
        
        # Calculate total balance
        total_balance = current_capital + stocks_value
        
        # Calculate total P&L
        total_pnl = total_balance - state['starting_capital']
        
        return (
            f"${total_balance:,.0f}",
            f"${stocks_value:,.0f}",
            f"${total_pnl:+,.0f}",
            f"{state['ai_decisions_today']}"
        )
        
    except Exception as e:
        logger.error(f"Error updating account metrics: {e}")
        return "$0", "$0", "$0", "0"

@app.callback(
    [Output('cpu-usage', 'children'),
     Output('cpu-percent', 'children'),
     Output('memory-usage', 'children'),
     Output('memory-percent', 'children'),
     Output('ai-activity', 'children'),
     Output('ai-activity-text', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_resource_usage(n):
    """Update system resource usage"""
    try:
        # Get CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_bar = dbc.Progress(
            value=cpu_percent,
            color="success" if cpu_percent >= 80 else "warning" if cpu_percent >= 50 else "danger",
            className="mb-0"
        )
        
        # Get memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_bar = dbc.Progress(
            value=memory_percent,
            color="success" if memory_percent >= 70 else "warning" if memory_percent >= 50 else "danger",
            className="mb-0"
        )
        
        # AI Activity (based on trading state)
        state = state_manager.get_current_state()
        ai_active = state.get('is_active', False)
        
        if ai_active:
            ai_activity = 100  # Maximum activity
            ai_text = "🚀 MAXIMUM POWER"
            ai_color = "success"
        else:
            ai_activity = 0
            ai_text = "🛑 STOPPED"
            ai_color = "danger"
        
        ai_bar = dbc.Progress(
            value=ai_activity,
            color=ai_color,
            className="mb-0"
        )
        
        return (
            cpu_bar,
            f"{cpu_percent:.1f}%",
            memory_bar,
            f"{memory_percent:.1f}%",
            ai_bar,
            ai_text
        )
        
    except Exception as e:
        logger.error(f"Error updating resource usage: {e}")
        return html.Div("Error"), "Error", html.Div("Error"), "Error", html.Div("Error"), "Error"

@app.callback(
    Output('holdings-table', 'children'),
    [Input('interval-component', 'n_intervals'),
     Input('ai-trading-active', 'data')]
)
def update_holdings_table(n, ai_active):
    """Update holdings table from state manager"""
    try:
        # Get positions from SINGLE SOURCE OF TRUTH
        state = state_manager.get_current_state()
        positions = state['positions']
        
        if not positions:
            return html.Div("No positions yet. Start MAXIMUM POWER AI trading!", 
                          className="text-center p-4",
                          style={'color': '#4ecdc4'})
        
        return dbc.Table([
            html.Thead([
                html.Tr([
                    html.Th("Symbol"),
                    html.Th("Shares"),
                    html.Th("Avg Price"),
                    html.Th("Current Price"),
                    html.Th("P&L ($)"),
                    html.Th("P&L (%)")
                ])
            ]),
            html.Tbody([
                html.Tr([
                    html.Td(html.Strong(p['symbol'])),
                    html.Td(f"{p['quantity']:,}"),
                    html.Td(f"${p['avg_price']:.2f}"),
                    html.Td(f"${p['current_price']:.2f}"),
                    html.Td(f"${p['pnl']:+,.0f}", 
                           className="text-success" if p['pnl'] >= 0 else "text-danger"),
                    html.Td(f"{p['pnl_pct']:+.2f}%", 
                           className="text-success" if p['pnl_pct'] >= 0 else "text-danger")
                ]) for p in positions
            ])
        ], bordered=True, dark=True, hover=True, responsive=True, striped=True)
        
    except Exception as e:
        logger.error(f"Error updating holdings table: {e}")
        return html.Div(f"Error: {str(e)}", className="text-danger")

@app.callback(
    Output('recent-trades-table', 'children'),
    [Input('interval-component', 'n_intervals'),
     Input('ai-trading-active', 'data')]
)
def update_recent_trades(n, ai_active):
    """Update recent trades table from state manager"""
    try:
        # Get trades from SINGLE SOURCE OF TRUTH
        state = state_manager.get_current_state()
        trades = state['trades'][-10:]  # Last 10 trades
        
        if not trades:
            return html.Div("No trades yet", 
                          className="text-center p-4",
                          style={'color': '#4ecdc4'})
        
        return dbc.Table([
            html.Thead([
                html.Tr([
                    html.Th("Time"),
                    html.Th("Symbol"),
                    html.Th("Side"),
                    html.Th("Qty"),
                    html.Th("Price"),
                    html.Th("Confidence")
                ])
            ]),
            html.Tbody([
                html.Tr([
                    html.Td(t['timestamp'].split('T')[1].split('.')[0]),
                    html.Td(t['symbol']),
                    html.Td(t['action'], 
                           className="text-success" if t['action'] == 'BUY' else "text-danger"),
                    html.Td(f"{t['quantity']:,}"),
                    html.Td(f"${t['price']:.2f}"),
                    html.Td(f"{t['confidence']:.1%}")
                ]) for t in reversed(trades)
            ])
        ], bordered=True, dark=True, hover=True, responsive=True, size="sm")
        
    except Exception as e:
        logger.error(f"Error updating trades table: {e}")
        return html.Div(f"Error: {str(e)}", className="text-danger")

@app.callback(
    Output('ai-system-status', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_ai_system_status(n):
    """Update AI system status"""
    try:
        state = state_manager.get_current_state()
        ai_active = state.get('is_active', False)
        
        # Get system resource usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent
        
        components = [
            ("🚀 Maximum Power AI Engine", "success" if ai_active else "danger"),
            ("🧠 Master Orchestrator", "success" if ai_active else "danger"),
            ("📊 Model Communication Hub", "success" if ai_active else "danger"),
            ("🎯 Intelligent Model Selector", "success" if ai_active else "danger"),
            ("📈 Performance Learner", "success" if ai_active else "danger"),
            ("🔍 Market Analyzer", "success" if ai_active else "danger"),
            ("✅ Cross-Model Validator", "success" if ai_active else "danger"),
            ("⚡ System Resources", "success" if cpu_percent >= 80 else "warning" if cpu_percent >= 50 else "danger")
        ]
        
        status_items = []
        for name, status in components:
            status_items.append(
                dbc.Row([
                    dbc.Col([
                        html.Span(className=f"status-indicator status-{status}"),
                        html.Strong(name)
                    ], width=8),
                    dbc.Col([
                        html.Span(status.upper(), className=f"badge bg-{status}")
                    ], width=4)
                ], className="mb-2")
            )
        
        return status_items
        
    except Exception as e:
        logger.error(f"Error updating AI system status: {e}")
        return html.Div(f"Error: {str(e)}", className="text-danger")

if __name__ == "__main__":
    # Fix Unicode encoding for Windows
    import sys
    import io
    if sys.stdout.encoding == 'cp1252':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    
    print("=" * 80)
    print("MAXIMUM POWER AI Trading Dashboard Starting...")
    print("=" * 80)
    print("Features:")
    print("   - MAXIMUM POWER AI Engine (80%+ resource usage)")
    print("   - Parallel processing across 100+ stocks")
    print("   - Real-time resource monitoring")
    print("   - Maximum computational intensity")
    print("   - Advanced technical analysis")
    print("   - GPU acceleration ready")
    print("Dashboard URL: http://localhost:8057")
    print("Press Ctrl+C to stop the server")
    print("=" * 80)
    
    app.run_server(debug=True, host='127.0.0.1', port=8057)
