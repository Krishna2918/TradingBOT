"""
Paper Trading Dashboard - Enhanced UI
Real-time paper trading with AI signals on Canadian markets
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import sys
import os
import threading
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import demo trading engine
try:
    from src.demo.demo_trading_engine import DemoTradingEngine
    demo_engine = DemoTradingEngine()
    logger.info("Demo Trading Engine initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Demo Trading Engine: {e}")
    demo_engine = None

# Initialize Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.CYBORG,
        dbc.icons.FONT_AWESOME,
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap"
    ],
    suppress_callback_exceptions=True
)
app.title = "AI Paper Trading - Canadian Markets"

# Enhanced Custom CSS
custom_css = '''
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    body {
        background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, #0f0f1a 100%);
        min-height: 100vh;
    }

    .main-container {
        background: transparent;
        min-height: 100vh;
        padding: 20px;
    }

    /* Header Styles */
    .dashboard-header {
        background: linear-gradient(135deg, rgba(83, 103, 254, 0.15) 0%, rgba(0, 208, 156, 0.15) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        backdrop-filter: blur(10px);
    }

    .logo-text {
        background: linear-gradient(135deg, #5367FE 0%, #00D09C 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 28px;
        letter-spacing: -0.5px;
    }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(145deg, rgba(30, 30, 45, 0.9) 0%, rgba(20, 20, 35, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        height: 100%;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(83, 103, 254, 0.4);
        box-shadow: 0 12px 40px rgba(83, 103, 254, 0.15);
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #5367FE, #00D09C);
        opacity: 0;
        transition: opacity 0.3s ease;
    }

    .metric-card:hover::before {
        opacity: 1;
    }

    .metric-icon {
        width: 56px;
        height: 56px;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
    }

    .metric-icon.profit {
        background: linear-gradient(135deg, rgba(0, 208, 156, 0.2) 0%, rgba(0, 208, 156, 0.05) 100%);
        color: #00D09C;
    }

    .metric-icon.loss {
        background: linear-gradient(135deg, rgba(235, 91, 60, 0.2) 0%, rgba(235, 91, 60, 0.05) 100%);
        color: #EB5B3C;
    }

    .metric-icon.neutral {
        background: linear-gradient(135deg, rgba(83, 103, 254, 0.2) 0%, rgba(83, 103, 254, 0.05) 100%);
        color: #5367FE;
    }

    .metric-value {
        font-size: 32px;
        font-weight: 700;
        letter-spacing: -1px;
        margin: 8px 0 4px 0;
    }

    .metric-label {
        color: rgba(255, 255, 255, 0.5);
        font-size: 13px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .metric-change {
        font-size: 14px;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 8px;
        display: inline-block;
    }

    .metric-change.positive {
        background: rgba(0, 208, 156, 0.15);
        color: #00D09C;
    }

    .metric-change.negative {
        background: rgba(235, 91, 60, 0.15);
        color: #EB5B3C;
    }

    /* Panel Cards */
    .panel-card {
        background: linear-gradient(145deg, rgba(30, 30, 45, 0.8) 0%, rgba(20, 20, 35, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        overflow: hidden;
        margin-bottom: 24px;
    }

    .panel-header {
        background: linear-gradient(90deg, rgba(83, 103, 254, 0.1) 0%, transparent 100%);
        padding: 16px 24px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .panel-title {
        font-size: 16px;
        font-weight: 600;
        color: #fff;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .panel-title i {
        color: #5367FE;
    }

    .panel-body {
        padding: 24px;
    }

    /* Control Buttons */
    .control-btn {
        padding: 12px 24px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 14px;
        border: none;
        transition: all 0.3s ease;
        display: inline-flex;
        align-items: center;
        gap: 8px;
    }

    .control-btn.start {
        background: linear-gradient(135deg, #00D09C 0%, #00b386 100%);
        color: white;
    }

    .control-btn.start:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 208, 156, 0.4);
    }

    .control-btn.stop {
        background: linear-gradient(135deg, #EB5B3C 0%, #d14a2d 100%);
        color: white;
    }

    .control-btn.stop:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(235, 91, 60, 0.4);
    }

    .control-btn.cycle {
        background: linear-gradient(135deg, #5367FE 0%, #4355d4 100%);
        color: white;
    }

    .control-btn.cycle:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(83, 103, 254, 0.4);
    }

    /* Stock Ticker */
    .ticker-container {
        background: rgba(0, 0, 0, 0.3);
        border-radius: 12px;
        padding: 12px 20px;
        margin-bottom: 24px;
        overflow: hidden;
    }

    .ticker-item {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 16px;
        margin-right: 24px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
    }

    .ticker-symbol {
        font-weight: 600;
        color: #fff;
    }

    .ticker-price {
        color: rgba(255, 255, 255, 0.7);
    }

    .ticker-change.up {
        color: #00D09C;
    }

    .ticker-change.down {
        color: #EB5B3C;
    }

    /* Tables */
    .custom-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
    }

    .custom-table th {
        background: rgba(83, 103, 254, 0.1);
        padding: 14px 16px;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: rgba(255, 255, 255, 0.6);
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }

    .custom-table td {
        padding: 14px 16px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        font-size: 14px;
    }

    .custom-table tr:hover td {
        background: rgba(255, 255, 255, 0.02);
    }

    /* Strategy Cards */
    .strategy-card {
        background: linear-gradient(145deg, rgba(40, 40, 55, 0.6) 0%, rgba(30, 30, 45, 0.6) 100%);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        transition: all 0.3s ease;
    }

    .strategy-card:hover {
        border-color: rgba(83, 103, 254, 0.3);
        background: linear-gradient(145deg, rgba(50, 50, 65, 0.6) 0%, rgba(40, 40, 55, 0.6) 100%);
    }

    .strategy-name {
        font-weight: 600;
        font-size: 14px;
        color: #fff;
        margin-bottom: 4px;
    }

    .strategy-allocation {
        font-size: 12px;
        color: rgba(255, 255, 255, 0.5);
    }

    .status-badge {
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
    }

    .status-badge.active {
        background: rgba(0, 208, 156, 0.15);
        color: #00D09C;
    }

    .status-badge.paper {
        background: rgba(253, 176, 34, 0.15);
        color: #FDB022;
    }

    /* Live Indicator */
    .live-indicator {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 14px;
        background: rgba(0, 208, 156, 0.1);
        border: 1px solid rgba(0, 208, 156, 0.3);
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        color: #00D09C;
    }

    .live-dot {
        width: 8px;
        height: 8px;
        background: #00D09C;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(1.2); }
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb {
        background: rgba(83, 103, 254, 0.5);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: rgba(83, 103, 254, 0.7);
    }
</style>
'''

# Custom index string with enhanced CSS
app.index_string = f'''
<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>{{%title%}}</title>
        {{%favicon%}}
        {{%css%}}
        {custom_css}
    </head>
    <body>
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
    </body>
</html>
'''

# Trading state
is_trading = False
auto_trading_enabled = False
activity_log = []  # Store recent trading activity

def add_activity(message, activity_type="info"):
    """Add an activity to the log"""
    global activity_log
    activity_log.insert(0, {
        'time': datetime.now().strftime('%H:%M:%S'),
        'message': message,
        'type': activity_type  # 'buy', 'sell', 'info', 'signal'
    })
    # Keep only last 50 activities
    activity_log = activity_log[:50]

def create_metric_card(title, value, change=None, icon="fa-chart-line", icon_type="neutral"):
    """Create an enhanced metric card"""
    change_class = "positive" if change and change >= 0 else "negative" if change else ""
    change_text = f"+{change:.2f}%" if change and change >= 0 else f"{change:.2f}%" if change else None

    return html.Div([
        html.Div([
            html.Div([
                html.I(className=f"fas {icon}")
            ], className=f"metric-icon {icon_type}"),
            html.Div([
                html.Div(title, className="metric-label"),
                html.Div(value, className="metric-value"),
                html.Span(change_text, className=f"metric-change {change_class}") if change_text else None
            ], style={'marginLeft': '16px', 'flex': '1'})
        ], style={'display': 'flex', 'alignItems': 'center'})
    ], className="metric-card")

def create_stock_ticker():
    """Create a live stock ticker display with real data only"""
    if not demo_engine or not demo_engine.current_prices:
        return html.Div([
            html.Span("Waiting for market data...", style={'color': 'rgba(255,255,255,0.5)', 'padding': '12px'})
        ], className="ticker-container")

    stocks = []
    for symbol, price in list(demo_engine.current_prices.items())[:8]:
        # Use previous price if available for real change calculation
        prev_price = demo_engine.previous_prices.get(symbol, price) if hasattr(demo_engine, 'previous_prices') else price
        change_pct = ((price - prev_price) / prev_price * 100) if prev_price else 0
        stocks.append((symbol, f"${price:.2f}", f"{change_pct:+.1f}%", change_pct >= 0))

    ticker_items = []
    for symbol, price, change, is_up in stocks:
        ticker_items.append(
            html.Div([
                html.Span(symbol, className="ticker-symbol"),
                html.Span(price, className="ticker-price"),
                html.Span(change, className=f"ticker-change {'up' if is_up else 'down'}")
            ], className="ticker-item")
        )

    return html.Div(ticker_items, className="ticker-container", style={'whiteSpace': 'nowrap'})

# Main layout
app.layout = html.Div([
    # Auto-refresh interval (UI updates)
    dcc.Interval(id='refresh-interval', interval=5000, n_intervals=0),
    # Auto-trading interval (executes trades when enabled)
    dcc.Interval(id='auto-trade-interval', interval=15000, n_intervals=0, disabled=True),
    # Store for auto-trading state
    dcc.Store(id='auto-trade-state', data={'enabled': False, 'cycles': 0}),

    # Header
    html.Div([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.I(className="fas fa-robot", style={'fontSize': '32px', 'color': '#5367FE', 'marginRight': '16px'}),
                    html.Div([
                        html.H1("AI Paper Trading", className="logo-text mb-0"),
                        html.P("Canadian Markets (TSX) - Real-time AI Analysis",
                               style={'color': 'rgba(255,255,255,0.6)', 'margin': 0, 'fontSize': '14px'})
                    ])
                ], style={'display': 'flex', 'alignItems': 'center'})
            ], width=6),
            dbc.Col([
                html.Div([
                    html.Div([
                        html.Div(className="live-dot"),
                        html.Span("LIVE DATA")
                    ], className="live-indicator me-3"),
                    html.Span("PAPER TRADING", className="status-badge paper"),
                    html.Span(id="market-status-badge", className="status-badge active ms-2")
                ], style={'display': 'flex', 'justifyContent': 'flex-end', 'alignItems': 'center'})
            ], width=6)
        ])
    ], className="dashboard-header"),

    # Live Ticker
    html.Div(id="stock-ticker"),

    # Trading Controls
    html.Div([
        html.Div([
            html.Div([
                html.Div([
                    html.I(className="fas fa-sliders-h me-2"),
                    "Trading Controls"
                ], className="panel-title"),
                html.Div(id="trading-time", style={'color': 'rgba(255,255,255,0.5)', 'fontSize': '13px'})
            ], className="panel-header"),
            html.Div([
                dbc.Row([
                    dbc.Col([
                        html.Button([
                            html.I(className="fas fa-robot me-2"),
                            "Auto Trade"
                        ], id="auto-trade-btn", className="control-btn start me-2", style={'minWidth': '130px'}),
                        html.Button([
                            html.I(className="fas fa-sync-alt me-2"),
                            "Run Cycle"
                        ], id="cycle-btn", className="control-btn cycle me-2"),
                        html.Button([
                            html.I(className="fas fa-undo me-2"),
                            "Reset"
                        ], id="reset-btn", className="control-btn stop"),
                    ], width=4),
                    dbc.Col([
                        html.Div([
                            html.Label("Starting Capital (CAD)", style={'color': 'rgba(255,255,255,0.6)', 'fontSize': '12px', 'marginBottom': '4px'}),
                            html.Div([
                                dcc.Input(
                                    id="capital-input",
                                    type="number",
                                    value=100000,
                                    min=1000,
                                    max=10000000,
                                    step=1000,
                                    style={
                                        'background': 'rgba(255,255,255,0.1)',
                                        'border': '1px solid rgba(255,255,255,0.2)',
                                        'borderRadius': '8px',
                                        'padding': '8px 12px',
                                        'color': '#fff',
                                        'width': '130px',
                                        'marginRight': '8px'
                                    }
                                ),
                                html.Button([
                                    html.I(className="fas fa-check me-1"),
                                    "Apply"
                                ], id="apply-capital-btn", className="control-btn cycle", style={'padding': '8px 16px'})
                            ], style={'display': 'flex', 'alignItems': 'center'})
                        ])
                    ], width=4),
                    dbc.Col([
                        html.Div(id="trading-status"),
                        html.Div(id="capital-status", style={'marginTop': '4px'}),
                        html.Div(id="auto-trade-status", style={'marginTop': '4px'})
                    ], width=4)
                ])
            ], className="panel-body")
        ], className="panel-card")
    ]),

    # Key Metrics Row
    dbc.Row([
        dbc.Col([html.Div(id="portfolio-value-card")], width=3),
        dbc.Col([html.Div(id="cash-card")], width=3),
        dbc.Col([html.Div(id="positions-card")], width=3),
        dbc.Col([html.Div(id="trades-card")], width=3)
    ], className="mb-4"),

    # Charts Row
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Div([
                    html.Div([
                        html.I(className="fas fa-chart-area me-2"),
                        "Portfolio Performance"
                    ], className="panel-title"),
                ], className="panel-header"),
                html.Div([
                    dcc.Graph(id="portfolio-chart", config={'displayModeBar': False})
                ], className="panel-body", style={'padding': '16px'})
            ], className="panel-card")
        ], width=8),
        dbc.Col([
            html.Div([
                html.Div([
                    html.Div([
                        html.I(className="fas fa-pie-chart me-2"),
                        "Allocation"
                    ], className="panel-title"),
                ], className="panel-header"),
                html.Div([
                    dcc.Graph(id="allocation-chart", config={'displayModeBar': False})
                ], className="panel-body", style={'padding': '16px'})
            ], className="panel-card")
        ], width=4)
    ]),

    # Positions and Trades
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Div([
                    html.Div([
                        html.I(className="fas fa-briefcase me-2"),
                        "Open Positions"
                    ], className="panel-title"),
                    html.Span(id="positions-count", className="status-badge active")
                ], className="panel-header"),
                html.Div([
                    html.Div(id="positions-table")
                ], className="panel-body")
            ], className="panel-card")
        ], width=6),
        dbc.Col([
            html.Div([
                html.Div([
                    html.Div([
                        html.I(className="fas fa-history me-2"),
                        "Recent Trades"
                    ], className="panel-title"),
                    html.Span(id="trades-count", className="status-badge active")
                ], className="panel-header"),
                html.Div([
                    html.Div(id="trades-table")
                ], className="panel-body")
            ], className="panel-card")
        ], width=6)
    ]),

    # Activity Feed and AI Strategies Row
    dbc.Row([
        dbc.Col([
            # Live Activity Feed
            html.Div([
                html.Div([
                    html.Div([
                        html.I(className="fas fa-stream me-2"),
                        "Live Activity Feed"
                    ], className="panel-title"),
                    html.Span(id="activity-count", className="status-badge active")
                ], className="panel-header"),
                html.Div([
                    html.Div(id="activity-feed", style={'maxHeight': '300px', 'overflowY': 'auto'})
                ], className="panel-body")
            ], className="panel-card")
        ], width=6),
        dbc.Col([
            # AI Strategies
            html.Div([
                html.Div([
                    html.Div([
                        html.I(className="fas fa-brain me-2"),
                        "AI Trading Strategies"
                    ], className="panel-title"),
                ], className="panel-header"),
                html.Div([
                    html.Div(id="strategies-panel")
                ], className="panel-body")
            ], className="panel-card")
        ], width=6)
    ])

], className="main-container")

# Callbacks

@app.callback(
    [Output("trading-status", "children"),
     Output("auto-trade-interval", "disabled"),
     Output("auto-trade-btn", "className"),
     Output("auto-trade-btn", "children")],
    [Input("auto-trade-btn", "n_clicks"),
     Input("cycle-btn", "n_clicks"),
     Input("reset-btn", "n_clicks")],
    prevent_initial_call=True
)
def handle_trading_controls(auto_clicks, cycle_clicks, reset_clicks):
    """Handle trading control buttons"""
    global is_trading, auto_trading_enabled, activity_log

    ctx = callback_context
    if not ctx.triggered:
        return "", True, "control-btn start me-2", [html.I(className="fas fa-robot me-2"), "Auto Trade"]

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if not demo_engine:
        return html.Div("Engine not initialized", style={'color': '#EB5B3C'}), True, "control-btn start me-2", [html.I(className="fas fa-robot me-2"), "Auto Trade"]

    if button_id == "auto-trade-btn":
        auto_trading_enabled = not auto_trading_enabled
        if auto_trading_enabled:
            add_activity("Auto-trading STARTED", "info")
            return (
                html.Div([
                    html.I(className="fas fa-robot me-2", style={'color': '#00D09C'}),
                    "Auto-Trading Active"
                ], style={'color': '#00D09C', 'fontWeight': '600'}),
                False,  # Enable interval
                "control-btn stop me-2",  # Change button style
                [html.I(className="fas fa-stop me-2"), "Stop Auto"]
            )
        else:
            add_activity("Auto-trading STOPPED", "info")
            return (
                html.Div([
                    html.I(className="fas fa-pause-circle me-2", style={'color': '#FDB022'}),
                    "Auto-Trading Stopped"
                ], style={'color': '#FDB022', 'fontWeight': '600'}),
                True,  # Disable interval
                "control-btn start me-2",
                [html.I(className="fas fa-robot me-2"), "Auto Trade"]
            )

    elif button_id == "cycle-btn":
        try:
            # Run a trading cycle
            old_trades = len(demo_engine.account.trade_history)
            demo_engine.run_demo_cycle()
            new_trades = len(demo_engine.account.trade_history)

            # Log new trades
            for trade in demo_engine.account.trade_history[old_trades:]:
                pnl_str = f" P&L: ${trade.get('pnl', 0):+,.2f}" if trade.get('pnl') else ""
                add_activity(
                    f"{trade['side']} {trade['quantity']} {trade['symbol']} @ ${trade['price']:.2f}{pnl_str}",
                    trade['side'].lower()
                )

            add_activity(f"Cycle completed - {new_trades - old_trades} trades", "info")

            return (
                html.Div([
                    html.I(className="fas fa-sync me-2", style={'color': '#5367FE'}),
                    f"Cycle done {datetime.now().strftime('%H:%M:%S')}"
                ], style={'color': '#5367FE', 'fontWeight': '600'}),
                True if not auto_trading_enabled else False,
                "control-btn stop me-2" if auto_trading_enabled else "control-btn start me-2",
                [html.I(className="fas fa-stop me-2"), "Stop Auto"] if auto_trading_enabled else [html.I(className="fas fa-robot me-2"), "Auto Trade"]
            )
        except Exception as e:
            add_activity(f"Error: {str(e)[:30]}", "info")
            return (
                html.Div([
                    html.I(className="fas fa-exclamation-circle me-2"),
                    f"Error: {str(e)[:30]}"
                ], style={'color': '#EB5B3C'}),
                True if not auto_trading_enabled else False,
                "control-btn stop me-2" if auto_trading_enabled else "control-btn start me-2",
                [html.I(className="fas fa-stop me-2"), "Stop Auto"] if auto_trading_enabled else [html.I(className="fas fa-robot me-2"), "Auto Trade"]
            )

    elif button_id == "reset-btn":
        # Reset the account
        auto_trading_enabled = False
        demo_engine.account.cash = demo_engine.account.starting_capital
        demo_engine.account.positions = {}
        demo_engine.account.trade_history = []
        activity_log = []
        add_activity(f"Account RESET to ${demo_engine.account.starting_capital:,.0f}", "info")
        return (
            html.Div([
                html.I(className="fas fa-undo me-2", style={'color': '#FDB022'}),
                "Account Reset"
            ], style={'color': '#FDB022', 'fontWeight': '600'}),
            True,
            "control-btn start me-2",
            [html.I(className="fas fa-robot me-2"), "Auto Trade"]
        )

    return "", True, "control-btn start me-2", [html.I(className="fas fa-robot me-2"), "Auto Trade"]

@app.callback(
    Output("auto-trade-status", "children"),
    Input("auto-trade-interval", "n_intervals"),
    prevent_initial_call=True
)
def execute_auto_trade(n_intervals):
    """Execute auto-trading cycle"""
    global auto_trading_enabled

    if not auto_trading_enabled or not demo_engine:
        return ""

    try:
        old_trades = len(demo_engine.account.trade_history)
        demo_engine.run_demo_cycle()
        new_trades = len(demo_engine.account.trade_history)

        # Log new trades
        for trade in demo_engine.account.trade_history[old_trades:]:
            pnl_str = f" P&L: ${trade.get('pnl', 0):+,.2f}" if trade.get('pnl') else ""
            add_activity(
                f"{trade['side']} {trade['quantity']} {trade['symbol']} @ ${trade['price']:.2f}{pnl_str}",
                trade['side'].lower()
            )

        if new_trades > old_trades:
            add_activity(f"Auto-cycle: {new_trades - old_trades} trades executed", "info")

        prices = demo_engine.current_prices or {}
        summary = demo_engine.account.get_summary(prices)
        pnl_color = '#00D09C' if summary['total_pnl'] >= 0 else '#EB5B3C'

        return html.Div([
            html.Span(f"Cycle #{n_intervals}", style={'color': 'rgba(255,255,255,0.5)', 'fontSize': '11px'}),
            html.Span(f" | P&L: ${summary['total_pnl']:+,.2f}", style={'color': pnl_color, 'fontSize': '11px', 'fontWeight': '600'})
        ])
    except Exception as e:
        return html.Div(f"Error: {str(e)[:20]}", style={'color': '#EB5B3C', 'fontSize': '11px'})

@app.callback(
    Output("capital-status", "children"),
    Input("apply-capital-btn", "n_clicks"),
    State("capital-input", "value"),
    prevent_initial_call=True
)
def update_starting_capital(n_clicks, new_capital):
    """Update the starting capital for the demo account"""
    if not n_clicks or not new_capital:
        return ""

    if not demo_engine:
        return html.Div("Engine not initialized", style={'color': '#EB5B3C', 'fontSize': '12px'})

    try:
        new_capital = float(new_capital)
        if new_capital < 1000:
            return html.Div("Min $1,000", style={'color': '#EB5B3C', 'fontSize': '12px'})
        if new_capital > 10000000:
            return html.Div("Max $10M", style={'color': '#EB5B3C', 'fontSize': '12px'})

        # Reset the demo account with new capital
        demo_engine.account.starting_capital = new_capital
        demo_engine.account.cash = new_capital
        demo_engine.account.positions = {}
        demo_engine.account.trade_history = []
        if hasattr(demo_engine.account, 'equity_history'):
            demo_engine.account.equity_history = []

        return html.Div([
            html.I(className="fas fa-check-circle me-2", style={'color': '#00D09C'}),
            f"Capital set to ${new_capital:,.0f}"
        ], style={'color': '#00D09C', 'fontSize': '12px', 'fontWeight': '600'})
    except Exception as e:
        return html.Div(f"Error: {str(e)[:20]}", style={'color': '#EB5B3C', 'fontSize': '12px'})

@app.callback(
    [Output("stock-ticker", "children"),
     Output("trading-time", "children"),
     Output("market-status-badge", "children")],
    Input("refresh-interval", "n_intervals")
)
def update_header(n):
    """Update header elements"""
    ticker = create_stock_ticker()
    current_time = datetime.now().strftime("%H:%M:%S EST")

    now = datetime.now()
    market_open = now.replace(hour=9, minute=30, second=0)
    market_close = now.replace(hour=16, minute=0, second=0)

    if market_open <= now <= market_close and now.weekday() < 5:
        market_status = "MARKET OPEN"
    else:
        market_status = "AFTER HOURS"

    return ticker, current_time, market_status

@app.callback(
    [Output("portfolio-value-card", "children"),
     Output("cash-card", "children"),
     Output("positions-card", "children"),
     Output("trades-card", "children")],
    Input("refresh-interval", "n_intervals")
)
def update_metrics(n):
    """Update key metrics cards"""
    if not demo_engine:
        return (
            create_metric_card("Portfolio Value", "$100,000", icon="fa-wallet", icon_type="neutral"),
            create_metric_card("Cash Available", "$100,000", icon="fa-coins", icon_type="neutral"),
            create_metric_card("Open Positions", "0", icon="fa-briefcase", icon_type="neutral"),
            create_metric_card("Total Trades", "0", icon="fa-exchange-alt", icon_type="neutral")
        )

    prices = demo_engine.current_prices if demo_engine.current_prices else {}
    summary = demo_engine.account.get_summary(prices)

    pnl_type = "profit" if summary['total_pnl'] >= 0 else "loss"

    portfolio_card = create_metric_card(
        "Portfolio Value",
        f"${summary['total_value']:,.2f}",
        change=summary['total_return_pct'],
        icon="fa-wallet",
        icon_type=pnl_type
    )

    cash_card = create_metric_card(
        "Cash Available",
        f"${summary['cash']:,.2f}",
        icon="fa-coins",
        icon_type="neutral"
    )

    positions_card = create_metric_card(
        "Open Positions",
        str(summary['num_positions']),
        icon="fa-briefcase",
        icon_type="neutral"
    )

    trades_card = create_metric_card(
        "Total Trades",
        str(summary['num_trades']),
        icon="fa-exchange-alt",
        icon_type="neutral"
    )

    return portfolio_card, cash_card, positions_card, trades_card

@app.callback(
    Output("portfolio-chart", "figure"),
    Input("refresh-interval", "n_intervals")
)
def update_portfolio_chart(n):
    """Update portfolio value chart with real equity history"""
    if not demo_engine:
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", height=320, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        return fig

    prices = demo_engine.current_prices if demo_engine.current_prices else {}
    current_value = demo_engine.account.get_total_value(prices) if prices else demo_engine.account.starting_capital

    # Use real equity history if available
    if hasattr(demo_engine.account, 'equity_history') and demo_engine.account.equity_history:
        times = [entry['timestamp'] for entry in demo_engine.account.equity_history]
        values = [entry['value'] for entry in demo_engine.account.equity_history]
        # Add current value
        times.append(datetime.now())
        values.append(current_value)
    else:
        # Show only current state - no synthetic data
        times = [datetime.now()]
        values = [current_value]

    fig = go.Figure()

    is_profit = current_value >= demo_engine.account.starting_capital
    color = '#00D09C' if is_profit else '#EB5B3C'
    fill_color = 'rgba(0, 208, 156, 0.1)' if is_profit else 'rgba(235, 91, 60, 0.1)'

    fig.add_trace(go.Scatter(
        x=times,
        y=values,
        mode='lines+markers' if len(values) < 10 else 'lines',
        fill='tozeroy',
        line=dict(color=color, width=2),
        fillcolor=fill_color,
        marker=dict(size=6, color=color) if len(values) < 10 else None,
        hovertemplate='$%{y:,.2f}<extra></extra>'
    ))

    # Starting capital reference
    fig.add_hline(
        y=demo_engine.account.starting_capital,
        line_dash="dash",
        line_color="rgba(255,255,255,0.3)",
        annotation_text=f"Start: ${demo_engine.account.starting_capital:,.0f}",
        annotation_font_color="rgba(255,255,255,0.5)"
    )

    fig.update_layout(
        template="plotly_dark",
        height=320,
        margin=dict(l=0, r=0, t=20, b=0),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', zeroline=False),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        hovermode='x unified'
    )

    return fig

@app.callback(
    Output("allocation-chart", "figure"),
    Input("refresh-interval", "n_intervals")
)
def update_allocation_chart(n):
    """Update capital allocation pie chart"""
    if not demo_engine:
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", height=320, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        return fig

    prices = demo_engine.current_prices if demo_engine.current_prices else {}

    cash = demo_engine.account.cash
    positions_value = sum(
        pos['quantity'] * prices.get(symbol, pos['avg_price'])
        for symbol, pos in demo_engine.account.positions.items()
    )

    labels = ['Cash', 'Invested']
    values = [cash, positions_value]
    colors = ['#5367FE', '#00D09C']

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.7,
        marker_colors=colors,
        textinfo='percent',
        textfont_size=14,
        hovertemplate='%{label}: $%{value:,.2f}<extra></extra>'
    )])

    # Add center text
    total = cash + positions_value
    fig.add_annotation(
        text=f"${total:,.0f}",
        x=0.5, y=0.5,
        font_size=20,
        font_color='white',
        showarrow=False
    )

    fig.update_layout(
        template="plotly_dark",
        height=320,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(size=12)
        )
    )

    return fig

@app.callback(
    [Output("positions-table", "children"),
     Output("positions-count", "children")],
    Input("refresh-interval", "n_intervals")
)
def update_positions_table(n):
    """Update open positions table"""
    if not demo_engine or not demo_engine.account.positions:
        return html.P("No open positions", style={'color': 'rgba(255,255,255,0.5)', 'textAlign': 'center', 'padding': '40px'}), "0"

    prices = demo_engine.current_prices if demo_engine.current_prices else {}

    rows = []
    for symbol, pos in demo_engine.account.positions.items():
        current_price = prices.get(symbol, pos['avg_price'])
        pnl = (current_price - pos['avg_price']) * pos['quantity']
        pnl_pct = ((current_price - pos['avg_price']) / pos['avg_price']) * 100

        pnl_color = '#00D09C' if pnl >= 0 else '#EB5B3C'

        rows.append(html.Tr([
            html.Td(html.Strong(symbol, style={'color': '#fff'})),
            html.Td(str(pos['quantity'])),
            html.Td(f"${pos['avg_price']:.2f}"),
            html.Td(f"${current_price:.2f}"),
            html.Td(f"${pnl:+,.2f}", style={'color': pnl_color, 'fontWeight': '600'}),
            html.Td(f"{pnl_pct:+.2f}%", style={'color': pnl_color}),
        ]))

    table = html.Table([
        html.Thead(html.Tr([
            html.Th("Symbol"),
            html.Th("Qty"),
            html.Th("Avg Price"),
            html.Th("Current"),
            html.Th("P&L"),
            html.Th("Return"),
        ])),
        html.Tbody(rows)
    ], className="custom-table")

    return table, str(len(demo_engine.account.positions))

@app.callback(
    [Output("trades-table", "children"),
     Output("trades-count", "children")],
    Input("refresh-interval", "n_intervals")
)
def update_trades_table(n):
    """Update recent trades table"""
    if not demo_engine or not demo_engine.account.trade_history:
        return html.P("No trades yet", style={'color': 'rgba(255,255,255,0.5)', 'textAlign': 'center', 'padding': '40px'}), "0"

    recent_trades = demo_engine.account.trade_history[-8:][::-1]

    rows = []
    for trade in recent_trades:
        side_color = '#00D09C' if trade['side'] == 'BUY' else '#EB5B3C'
        pnl = trade.get('pnl', 0)
        pnl_text = f"${pnl:+,.2f}" if pnl else "-"
        pnl_color = '#00D09C' if pnl >= 0 else '#EB5B3C' if pnl else 'rgba(255,255,255,0.5)'

        rows.append(html.Tr([
            html.Td(trade['timestamp'].strftime('%H:%M:%S')),
            html.Td(trade['side'], style={'color': side_color, 'fontWeight': '700'}),
            html.Td(html.Strong(trade['symbol'])),
            html.Td(str(trade['quantity'])),
            html.Td(f"${trade['price']:.2f}"),
            html.Td(pnl_text, style={'color': pnl_color, 'fontWeight': '600'}),
        ]))

    table = html.Table([
        html.Thead(html.Tr([
            html.Th("Time"),
            html.Th("Side"),
            html.Th("Symbol"),
            html.Th("Qty"),
            html.Th("Price"),
            html.Th("P&L"),
        ])),
        html.Tbody(rows)
    ], className="custom-table")

    return table, str(len(demo_engine.account.trade_history))

@app.callback(
    Output("strategies-panel", "children"),
    Input("refresh-interval", "n_intervals")
)
def update_strategies_panel(n):
    """Update AI strategies panel"""
    strategies = [
        {"name": "Momentum Scalping", "allocation": "20%", "icon": "fa-bolt", "color": "#5367FE"},
        {"name": "News-Volatility", "allocation": "20%", "icon": "fa-newspaper", "color": "#00D09C"},
        {"name": "Gamma/OI Squeeze", "allocation": "15%", "icon": "fa-compress-arrows-alt", "color": "#FDB022"},
        {"name": "Statistical Arbitrage", "allocation": "15%", "icon": "fa-balance-scale", "color": "#EB5B3C"},
        {"name": "AI/ML Patterns", "allocation": "30%", "icon": "fa-brain", "color": "#9B59B6"},
    ]

    cards = []
    for strat in strategies:
        cards.append(
            dbc.Col([
                html.Div([
                    html.Div([
                        html.Div([
                            html.I(className=f"fas {strat['icon']}", style={'color': strat['color'], 'fontSize': '18px'}),
                        ], style={'marginRight': '12px'}),
                        html.Div([
                            html.Div(strat['name'], className="strategy-name"),
                            html.Div(f"Allocation: {strat['allocation']}", className="strategy-allocation"),
                        ]),
                        html.Span("Active", className="status-badge active", style={'marginLeft': 'auto'})
                    ], style={'display': 'flex', 'alignItems': 'center'})
                ], className="strategy-card")
            ], width=12, lg=4, className="mb-2")
        )

    return dbc.Row(cards)

@app.callback(
    [Output("activity-feed", "children"),
     Output("activity-count", "children")],
    Input("refresh-interval", "n_intervals")
)
def update_activity_feed(n):
    """Update the live activity feed with trading decisions"""
    global activity_log

    if not activity_log:
        return html.P("No activity yet - Start auto-trading or run a cycle",
                     style={'color': 'rgba(255,255,255,0.5)', 'textAlign': 'center', 'padding': '40px'}), "0"

    activity_items = []
    for activity in activity_log[:20]:  # Show last 20 activities
        # Determine icon and color based on activity type
        if activity['type'] == 'buy':
            icon = "fa-arrow-up"
            color = "#00D09C"
            bg_color = "rgba(0, 208, 156, 0.1)"
        elif activity['type'] == 'sell':
            icon = "fa-arrow-down"
            color = "#EB5B3C"
            bg_color = "rgba(235, 91, 60, 0.1)"
        elif activity['type'] == 'signal':
            icon = "fa-bolt"
            color = "#FDB022"
            bg_color = "rgba(253, 176, 34, 0.1)"
        else:  # info
            icon = "fa-info-circle"
            color = "#5367FE"
            bg_color = "rgba(83, 103, 254, 0.1)"

        activity_items.append(
            html.Div([
                html.Div([
                    html.I(className=f"fas {icon}", style={'color': color, 'fontSize': '14px'})
                ], style={
                    'width': '32px',
                    'height': '32px',
                    'borderRadius': '8px',
                    'background': bg_color,
                    'display': 'flex',
                    'alignItems': 'center',
                    'justifyContent': 'center',
                    'marginRight': '12px',
                    'flexShrink': '0'
                }),
                html.Div([
                    html.Div(activity['message'], style={
                        'color': '#fff',
                        'fontSize': '13px',
                        'fontWeight': '500'
                    }),
                    html.Div(activity['time'], style={
                        'color': 'rgba(255,255,255,0.4)',
                        'fontSize': '11px'
                    })
                ])
            ], style={
                'display': 'flex',
                'alignItems': 'center',
                'padding': '10px 12px',
                'borderBottom': '1px solid rgba(255,255,255,0.05)',
                'transition': 'background 0.2s'
            })
        )

    return html.Div(activity_items), str(len(activity_log))

if __name__ == '__main__':
    print("\n" + "="*60)
    print("     AI PAPER TRADING DASHBOARD")
    print("="*60)
    print(f"\n  Mode: DEMO (Paper Trading)")
    print(f"  Capital: $100,000 CAD")
    print(f"  Market: Canadian (TSX)")
    print(f"\n  Dashboard: http://localhost:8054")
    print(f"  Risk Monitor: http://localhost:8053")
    print("\n  Press Ctrl+C to stop")
    print("="*60 + "\n")

    app.run(debug=False, host='0.0.0.0', port=8054)
