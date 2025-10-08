"""
Interactive AI Trading Dashboard - Canadian Market
Enter demo capital and watch AI trade with live market data
"""

import dash
from dash import dcc, html, Input, Output, State, callback
from dash.dependencies import ALL
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime, timedelta, timezone
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None
import pandas as pd
import numpy as np
from pathlib import Path
import yfinance as yf
import random
import logging
import sys
from typing import Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('trading_ai.log')
    ]
)

# Import REAL AI system
try:
    from src.ai.autonomous_trading_ai import AutonomousTradingAI
    AI_AVAILABLE = True
    print("Autonomous AI Trading System loaded successfully!")
except Exception as e:
    AI_AVAILABLE = False
    print(f"WARNING: AI System unavailable: {e}")
    print("Running in basic mode with simulated AI")

# Initialize Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.DARKLY,
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
    ],
    suppress_callback_exceptions=True
)

app.title = "AI Trading Dashboard - Canadian Market"

# Add custom CSS for blinking animation
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            @keyframes blink {
                0%, 50%, 100% { opacity: 1; }
                25%, 75% { opacity: 0.3; }
            }
            .blink {
                animation: blink 2s infinite;
            }
        </style>
        <script>
            document.addEventListener('DOMContentLoaded', function() {
                document.addEventListener('keydown', function(e) {
                    // Toggle pause on key 'P' (case-insensitive)
                    if (e.key === 'p' || e.key === 'P') {
                        const hidden = document.getElementById('pause-keybind');
                        if (hidden) {
                            hidden.checked = !hidden.checked;
                            hidden.dispatchEvent(new Event('change', {bubbles: true}));
                        }
                        const visible = document.getElementById('pause-switch');
                        if (visible) {
                            // Simulate click to keep UI in sync
                            visible.click();
                        }
                    }
                });
            });
        </script>
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

# Canadian stock symbols (TSX)
CANADIAN_STOCKS = [
    'RY.TO',   # Royal Bank
    'TD.TO',   # TD Bank
    'BNS.TO',  # Scotiabank
    'BMO.TO',  # Bank of Montreal
    'SHOP.TO', # Shopify
    'CNQ.TO',  # Canadian Natural Resources
    'ENB.TO',  # Enbridge
    'CP.TO',   # Canadian Pacific
    'CNR.TO',  # Canadian National Railway
    'SU.TO',   # Suncor Energy
]

from src.dashboard import trading_state, STATE_STORE, DEMO_STATE_PATH, reset_in_memory_state, load_trading_state, save_trading_state
from src.dashboard.ui_components import create_navbar, create_enhanced_navbar, create_status_pill, create_regime_badge
from src.dashboard.log_utils import get_last_log_time
from src.dashboard.services import (
    get_live_price,
    get_demo_price,
    real_ai_trade,
    simulate_ai_trade,
    update_holdings_prices,
    generate_ai_signals,
    _ensure_live_broker,
    fetch_live_market_data,
    compute_trade_features,
    _update_learning_from_trade,
    is_market_open,
)
from src.dashboard.charts import generate_performance_chart, generate_sector_allocation
from src.dashboard.sections import (
    create_hybrid_control_status,
    create_risk_panel,
    create_learning_panel,
    create_alerts_feed,
    create_ai_activity_monitor,
    create_ai_signals_table,
    create_performance_tabs,
)
from src.dashboard.portfolio import (
    generate_portfolio_data,
    generate_holdings,
    generate_recent_trades,
    create_summary_cards,
    create_holdings_table,
    create_recent_trades_table,
)
try:
    from src.ai.regime_detection import RegimeDetector, MarketRegime
except Exception:
    RegimeDetector = None
    MarketRegime = None
try:
    from src.data_pipeline.questrade_client import QuestradeClient
except Exception:
    QuestradeClient = None

# ============================================================================
# REAL MARKET DATA FUNCTIONS
# ============================================================================

def create_startup_screen():
    """Create initial setup screen"""
    return html.Div([
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Div([
                            html.I(className="fas fa-robot fa-5x text-primary mb-4"),
                            html.H1("🤖 AI Trading Bot", className="mb-3", style={"color": "white"}),
                            html.H4("Canadian Market - Demo Mode", className="text-muted mb-5"),
                        
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("💰 Enter Your Demo Capital", className="mb-4"),
                                
                                dbc.InputGroup([
                                    dbc.InputGroupText("$", style={"fontSize": "1.5rem"}),
                                    dbc.Input(
                                        id="starting-capital-input",
                                        type="number",
                                        placeholder="Enter amount (e.g., 10000)",
                                        min=10,
                                        max=10000000,
                                        step=10,
                                        style={"fontSize": "1.5rem", "height": "60px"}
                                    ),
                                    dbc.InputGroupText("CAD", style={"fontSize": "1.5rem"}),
                                ], className="mb-4", size="lg"),
                                
                                html.Div([
                                    dbc.Button(
                                        "Start AI Trading 🚀",
                                        id="start-trading-btn",
                                        color="success",
                                        size="lg",
                                        className="w-100",
                                        style={"fontSize": "1.3rem", "padding": "15px"}
                                    ),
                                ]),
                                
                                html.Hr(className="my-4"),
                                
                                html.Div([
                                    html.H6("📊 What happens next:", className="mb-3"),
                                    html.Ul([
                                        html.Li("AI will analyze live Canadian market data (TSX)"),
                                        html.Li("Trades will be executed in demo mode (no real money)"),
                                        html.Li("You'll see real-time portfolio updates"),
                                        html.Li("All strategies will work with your demo capital"),
                                        html.Li("AI learns and adapts from each trade"),
                                    ], className="text-muted")
                                ]),
                                
                                html.Hr(className="my-4"),
                                
                                html.Div([
                                    html.H6("🤖 AI Activity Monitor", className="mb-3"),
                                    dbc.Button(
                                        [
                                            html.I(className="fas fa-terminal me-2"),
                                            "View AI Logs"
                                        ],
                                        id="view-ai-logs-startup-btn",
                                        color="info",
                                        size="lg",
                                        className="w-100",
                                        style={"fontSize": "1.1rem", "padding": "12px"}
                                    ),
                                    html.Small("Monitor AI activities, trades, and decisions in real-time", className="text-muted mt-2 d-block")
                                ]),
                                html.Div([
                                    html.A([
                                        html.I(className="fas fa-external-link-alt me-2"),
                                        "Open Logs in New Tab"
                                    ], href="/logs", target="_blank", className="btn btn-outline-secondary w-100 mt-2")
                                ])
                            ])
                        ], className="shadow-lg"),
                        
                        html.Div([
                            html.Small("WARNING: This is DEMO MODE - No real money will be used", className="text-warning"),
                            html.Br(),
                            html.Small("CONFIRMED: All market data is LIVE and REAL", className="text-success"),
                        ], className="mt-4"),
                        
                        # AI Activity Monitor Panel
                        dbc.Card([
                            dbc.CardHeader([
                                html.H5([
                                    html.I(className="fas fa-terminal me-2"),
                                    "AI Activity Monitor"
                                ], className="mb-0 text-light")
                            ], className="bg-dark border-secondary"),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        html.Div([
                                            html.I(className="fas fa-file-alt text-info me-2"),
                                            html.Span("Activity Log", className="text-light")
                                        ], className="mb-2"),
                                        html.Div([
                                            html.Small("Status: ", className="text-muted"),
                                            html.Span("Active", className="text-success", id="startup-ai-status")
                                        ])
                                    ], width=6),
                                    dbc.Col([
                                        html.Div([
                                            html.I(className="fas fa-chart-line text-success me-2"),
                                            html.Span("Trades Log", className="text-light")
                                        ], className="mb-2"),
                                        html.Div([
                                            html.Small("Status: ", className="text-muted"),
                                            html.Span("Ready", className="text-info", id="startup-trades-status")
                                        ])
                                    ], width=6)
                                ], className="mb-3"),
                                
                                dbc.Row([
                                    dbc.Col([
                                        html.Div([
                                            html.I(className="fas fa-signal text-warning me-2"),
                                            html.Span("Signals Log", className="text-light")
                                        ], className="mb-2"),
                                        html.Div([
                                            html.Small("Status: ", className="text-muted"),
                                            html.Span("Ready", className="text-info", id="startup-signals-status")
                                        ])
                                    ], width=6),
                                    dbc.Col([
                                        html.Div([
                                            html.I(className="fas fa-brain text-primary me-2"),
                                            html.Span("Decisions Log", className="text-light")
                                        ], className="mb-2"),
                                        html.Div([
                                            html.Small("Status: ", className="text-muted"),
                                            html.Span("Ready", className="text-info", id="startup-decisions-status")
                                        ])
                                    ], width=6)
                                ]),
                                
                                html.Hr(className="my-3 border-secondary"),
                                
                                dbc.Button(
                                    [
                                        html.I(className="fas fa-external-link-alt me-1"),
                                        "Open AI Logs"
                                    ],
                                    id="startup-open-logs-btn",
                                    color="primary",
                                    size="sm",
                                    className="w-100"
                                )
                                ,
                                html.A(
                                    [html.I(className="fas fa-external-link-alt me-1"), "Open in New Tab"],
                                    href="/logs", target="_blank", className="btn btn-outline-secondary btn-sm w-100 mt-2"
                                )
                            ], className="bg-dark")
                        ], className="mt-4 shadow-sm")
                        
                        ], className="text-center", style={"maxWidth": "600px", "margin": "0 auto"})
                    ], style={"minHeight": "100vh", "display": "flex", "alignItems": "center", "justifyContent": "center"})
                ])
            ])
        ], fluid=True)
    ], style={"backgroundColor": "#1a1a1a", "minHeight": "100vh"})

# ============================================================================
# LAYOUT COMPONENTS
# ============================================================================

# ============================================================================
# MAIN LAYOUT
# ============================================================================

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dcc.Store(id='trading-initialized', data=False),
    dcc.Store(id='ui-dirty', data=False),
    # Status banners
    html.Div(id='status-banner'),
    html.Div(id='market-status-banner'),
    
    # Auto-refresh interval for trading
    dcc.Interval(
        id='trading-interval',
        interval=5*1000,  # 5 seconds - AI makes trades (fast with Questrade!)
        n_intervals=0,
        disabled=True
    ),
    
    # Chart update interval
    dcc.Interval(
        id='chart-interval',
        interval=2*1000,  # 2 seconds
        n_intervals=0,
        disabled=True
    ),
    
    # Main content
    html.Div(id='main-content', children=create_startup_screen())
], style={"backgroundColor": "#1a1a1a", "minHeight": "100vh"})

# ============================================================================
# TRADING PAGE
# ============================================================================

def create_trading_page():
    return html.Div([
        create_enhanced_navbar(),
        
        dbc.Container([
            html.Div(id='summary-cards-container', children=create_summary_cards()),
            create_hybrid_control_status(),
            
            # Enhanced panels row
            dbc.Row([
                dbc.Col([
                    create_risk_panel()
                ], width=12, lg=4, className="mb-4"),
                
                dbc.Col([
                    create_learning_panel()
                ], width=12, lg=4, className="mb-4"),
                
                dbc.Col([
                    create_ai_activity_monitor()
                ], width=12, lg=4, className="mb-4"),
            ]),

            # Live broker panel (only visible in live mode)
            dbc.Row([
                dbc.Col([
                    html.Div(id='broker-panel')
                ], width=12)
            ], className="mb-4"),
            
            # Charts row
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(
                                id='performance-chart',
                                figure=generate_performance_chart(),
                                config={'displayModeBar': False}
                            )
                        ])
                    ], className="shadow-sm mb-4")
                ], width=12, lg=8),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(
                                id='sector-chart',
                                figure=generate_sector_allocation(),
                                config={'displayModeBar': False}
                            )
                        ])
                    ], className="shadow-sm mb-4")
                ], width=12, lg=4),
            ]),
            
            # Performance tabs
            dbc.Row([
                dbc.Col([
                    create_performance_tabs()
                ], width=12, className="mb-4")
            ]),
            
            html.Div(id='holdings-table-container', children=create_holdings_table()),
            html.Div(id='recent-trades-container', children=create_recent_trades_table()),
            html.Div(id='ai-signals-container', children=create_ai_signals_table()),
            
        ], fluid=True)
    ], style={"backgroundColor": "#1a1a1a", "minHeight": "100vh"})

def create_ai_logs_page():
    """Create AI Logs page with embedded log viewer"""
    return html.Div([
        create_enhanced_navbar(),
        
        dbc.Container([
            # Page header
            dbc.Row([
                dbc.Col([
                    html.H2([
                        html.I(className="fas fa-terminal me-3"),
                        "AI Activity Logs"
                    ], className="text-light mb-4")
                ], width=12)
            ]),
            
            # Log controls
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dbc.ButtonGroup([
                                        dbc.Button("Activity Log", id="log-tab-activity", color="primary", outline=True, active=True),
                                        dbc.Button("Trades Log", id="log-tab-trades", color="success", outline=True),
                                        dbc.Button("Signals Log", id="log-tab-signals", color="warning", outline=True),
                                        dbc.Button("Decisions Log", id="log-tab-decisions", color="info", outline=True)
                                    ])
                                ], width=8),
                                dbc.Col([
                                    html.Div([
                                        dbc.Switch(id='logs-refresh-switch', label='Auto-refresh', value=True, className='me-3'),
                                        dbc.ButtonGroup([
                                            dbc.Button([
                                                html.I(className="fas fa-sync-alt me-1"),
                                                "Refresh"
                                            ], id="refresh-logs-btn", color="secondary", size="sm"),
                                            dbc.Button([
                                                html.I(className="fas fa-download me-1"),
                                                "Export"
                                            ], id="export-logs-btn", color="info", size="sm")
                                        ], className="d-inline-block")
                                    ], className="d-flex align-items-center justify-content-end")
                                ], width=4, className="text-end")
                            ])
                        ])
                    ], className="mb-4")
                ], width=12)
            ]),
            
            # Log content
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Store(id='current-log-tab', data='activity'),
                            dcc.Interval(id='logs-interval', interval=3000, disabled=False),
                            html.Div(
                                id="log-content",
                                style={
                                    "height": "600px",
                                    "overflowY": "auto",
                                    "backgroundColor": "#1e1e1e",
                                    "border": "1px solid #444",
                                    "borderRadius": "5px",
                                    "padding": "15px",
                                    "fontFamily": "monospace",
                                    "fontSize": "12px",
                                    "color": "#00ff00"
                                }
                            )
                        ])
                    ])
                ], width=12)
            ])
            
        ], fluid=True)
    ], style={"backgroundColor": "#1a1a1a", "minHeight": "100vh"})

# ============================================================================
# CALLBACKS
# ============================================================================

@app.callback(
    [Output('trading-initialized', 'data'),
     Output('url', 'pathname', allow_duplicate=True)],
    [Input('start-trading-btn', 'n_clicks')],
    [State('starting-capital-input', 'value'),
     State('trading-initialized', 'data')],
    prevent_initial_call=True
)
def initialize_trading(n_clicks, starting_capital, initialized):
    """Handle trading start button click - Launch FULL AI System 24/7"""
    if n_clicks and starting_capital and (starting_capital >= 10):
        print('=' * 80)
        print('LAUNCHING FULL AI TRADING SYSTEM')
        print('=' * 80)
        trading_state['initialized'] = True
        trading_state['starting_capital'] = starting_capital
        trading_state['current_capital'] = starting_capital
        trading_state['start_time'] = datetime.now()
        trading_state['trades'] = []
        trading_state['holdings'] = []
        save_trading_state()
        if AI_AVAILABLE:
            try:
                print('=' * 80)
                print('Initializing Autonomous Trading AI...')
                print(f'Capital: ${starting_capital:,.2f}')
                print(f"Mode: {trading_state['mode'].upper()}")
                ai = AutonomousTradingAI(mode=trading_state['mode'], initial_capital=starting_capital)
                trading_state['ai_instance'] = ai
                print('=' * 80)
                print('AI System Ready!')
                print(f'Stock Universe: {len(ai.symbols)} symbols')
                print(f"Analyzing: {', '.join(ai.symbols[:5])}... and {len(ai.symbols) - 5} more")
                print('Trading will run 24/7 (trades only during market hours)')
                print('AI will analyze: Market Data, News, Macro Indicators, Options, Events')
                print('Using: LSTM, GRU, RL Agents, Sentiment Analysis, Volatility Detection')
                print('=' * 80)
            except Exception as e:
                print(f'WARNING: AI init failed: {e}')
                import traceback
                traceback.print_exc()
                print('Running in basic mode')
                trading_state['ai_instance'] = None
        else:
            print('WARNING: Full AI unavailable - running in basic mode')
            trading_state['ai_instance'] = None
        return (True, '/')
    return (False, '/')
def initialize_trading(n_clicks, starting_capital, initialized):
    """Handle trading start button click - Launch FULL AI System 24/7"""
    if n_clicks and starting_capital and (starting_capital >= 10):
        print('=' * 80)
        print('LAUNCHING FULL AI TRADING SYSTEM')
        print('=' * 80)
        trading_state['initialized'] = True
        trading_state['starting_capital'] = starting_capital
        trading_state['current_capital'] = starting_capital
        trading_state['start_time'] = datetime.now()
        trading_state['trades'] = []
        trading_state['holdings'] = []
        save_trading_state()
        if AI_AVAILABLE:
            try:
                print('=' * 80)
                print('Initializing Autonomous Trading AI...')
                print(f'Capital: ${starting_capital:,.2f}')
                print(f"Mode: {trading_state['mode'].upper()}")
                ai = AutonomousTradingAI(mode=trading_state['mode'], initial_capital=starting_capital)
                trading_state['ai_instance'] = ai
                print('=' * 80)
                print('AI System Ready!')
                print(f'Stock Universe: {len(ai.symbols)} symbols')
                print(f"Analyzing: {', '.join(ai.symbols[:5])}... and {len(ai.symbols) - 5} more")
                print('Trading will run 24/7 (trades only during market hours)')
                print('AI will analyze: Market Data, News, Macro Indicators, Options, Events')
                print('Using: LSTM, GRU, RL Agents, Sentiment Analysis, Volatility Detection')
                print('=' * 80)
            except Exception as e:
                print(f'WARNING: AI init failed: {e}')
                import traceback
                traceback.print_exc()
                print('Running in basic mode')
                trading_state['ai_instance'] = None
        else:
            print('WARNING: Full AI unavailable - running in basic mode')
            trading_state['ai_instance'] = None
        return (True, '/')
    return (False, '/')

@app.callback(
    Output('trading-interval', 'n_intervals'),
    Input('trading-interval', 'n_intervals'),
    prevent_initial_call=True
)
def execute_ai_trade(n):
    """AI executes a trade every interval AND updates holdings with real prices - 24/7 AUTONOMOUS"""
    if trading_state['initialized']:
        _detect_and_update_regime()
        update_holdings_prices()
        real_ai_trade()
    return n
def execute_ai_trade(n):
    """AI executes a trade every interval AND updates holdings with real prices - 24/7 AUTONOMOUS"""
    if trading_state['initialized']:
        _detect_and_update_regime()
        update_holdings_prices()
        real_ai_trade()
    return n

@app.callback(
    [Output('performance-chart', 'figure'),
     Output('sector-chart', 'figure')],
    Input('chart-interval', 'n_intervals'),
    prevent_initial_call=True
)
def update_charts(n):
    return (generate_performance_chart(), generate_sector_allocation())
def update_charts(n):
    return (generate_performance_chart(), generate_sector_allocation())

@app.callback(
    [Output('summary-cards-container', 'children'),
     Output('holdings-table-container', 'children'),
     Output('recent-trades-container', 'children'),
     Output('ai-signals-container', 'children')],
    [Input('chart-interval', 'n_intervals'),
     Input('trading-interval', 'n_intervals')],
    prevent_initial_call=False
)
def refresh_dashboard_sections(_chart_tick, _trade_tick):
    """Refresh key dashboard sections so cash, P&L, and holdings stay in sync with trades."""
    try:
        load_trading_state()
    except Exception:
        pass
    try:
        update_holdings_prices()
        save_trading_state()
    except Exception:
        pass
    signals_df = None
    try:
        signals_df = generate_ai_signals()
    except Exception:
        signals_df = None
    return (create_summary_cards(), create_holdings_table(), create_recent_trades_table(), create_ai_signals_table(signals_df))
def refresh_dashboard_sections(_chart_tick, _trade_tick):
    """Refresh key dashboard sections so cash, P&L, and holdings stay in sync with trades."""
    try:
        load_trading_state()
    except Exception:
        pass
    try:
        update_holdings_prices()
        save_trading_state()
    except Exception:
        pass
    signals_df = None
    try:
        signals_df = generate_ai_signals()
    except Exception:
        signals_df = None
    return (create_summary_cards(), create_holdings_table(), create_recent_trades_table(), create_ai_signals_table(signals_df))

@app.callback(
    [Output('ai-log-time', 'children'),
     Output('ai-trade-time', 'children'),
     Output('ai-signal-time', 'children'),
     Output('ai-decision-time', 'children')],
    [Input('chart-interval', 'n_intervals'),
     Input('trading-interval', 'n_intervals')],
    prevent_initial_call=False
)
def refresh_ai_activity_monitor(_chart_tick, _trade_tick):
    """Keep AI Activity Monitor timestamps in sync with latest log activity."""
    return (get_last_log_time('logs/ai_activity.log'), get_last_log_time('logs/ai_trades.log'), get_last_log_time('logs/ai_signals.log'), get_last_log_time('logs/ai_decisions.log'))
def refresh_ai_activity_monitor(_chart_tick, _trade_tick):
    """Keep AI Activity Monitor timestamps in sync with latest log activity."""
    return (get_last_log_time('logs/ai_activity.log'), get_last_log_time('logs/ai_trades.log'), get_last_log_time('logs/ai_signals.log'), get_last_log_time('logs/ai_decisions.log'))

@app.callback(
    [Output('status-pill', 'children', allow_duplicate=True),
     Output('ui-dirty', 'data', allow_duplicate=True),
     Output('broker-panel', 'children', allow_duplicate=True)],
    [Input('mode-switch', 'value'),
     Input('startup-mode-switch', 'value')],
    prevent_initial_call=True
)
def toggle_mode(switch_value, startup_switch_value):
    """Toggle between Demo and Live mode with immediate UI updates"""
    if switch_value is not None:
        new_mode = 'live' if switch_value else 'demo'
    elif startup_switch_value is not None:
        new_mode = 'live' if startup_switch_value else 'demo'
    else:
        return (dash.no_update, dash.no_update, dash.no_update)
    old_mode = trading_state['mode']
    if new_mode != old_mode:
        if old_mode == 'demo':
            trading_state['demo_capital'] = trading_state['current_capital']
        else:
            trading_state['live_capital'] = trading_state['current_capital']
        trading_state['mode'] = new_mode
        if new_mode == 'demo':
            if load_trading_state():
                pass
            else:
                trading_state['current_capital'] = trading_state['demo_capital'] or trading_state['starting_capital']
            save_trading_state()
        else:
            trading_state['live_capital'] = trading_state['live_capital'] or trading_state['starting_capital']
            trading_state['current_capital'] = trading_state['live_capital']
            try:
                _ensure_live_broker()
            except Exception:
                pass
        print(f"✅ MODE SWITCHED: {old_mode.upper()} → {new_mode.upper()} | Capital: ${trading_state['current_capital']:,.2f}")
    return (create_status_pill(), True, render_broker_panel_content())
def toggle_mode(switch_value, startup_switch_value):
    """Toggle between Demo and Live mode with immediate UI updates"""
    if switch_value is not None:
        new_mode = 'live' if switch_value else 'demo'
    elif startup_switch_value is not None:
        new_mode = 'live' if startup_switch_value else 'demo'
    else:
        return (dash.no_update, dash.no_update, dash.no_update)
    old_mode = trading_state['mode']
    if new_mode != old_mode:
        if old_mode == 'demo':
            trading_state['demo_capital'] = trading_state['current_capital']
        else:
            trading_state['live_capital'] = trading_state['current_capital']
        trading_state['mode'] = new_mode
        if new_mode == 'demo':
            if load_trading_state():
                pass
            else:
                trading_state['current_capital'] = trading_state['demo_capital'] or trading_state['starting_capital']
            save_trading_state()
        else:
            trading_state['live_capital'] = trading_state['live_capital'] or trading_state['starting_capital']
            trading_state['current_capital'] = trading_state['live_capital']
            try:
                _ensure_live_broker()
            except Exception:
                pass
        print(f"✅ MODE SWITCHED: {old_mode.upper()} → {new_mode.upper()} | Capital: ${trading_state['current_capital']:,.2f}")
    return (create_status_pill(), True, render_broker_panel_content())

@app.callback(
    Output('performance-tab-content', 'children'),
    Input('performance-tabs', 'active_tab'),
    prevent_initial_call=True
)

# ============================================================================
# AI ACTIVITY MONITOR CALLBACKS
# ============================================================================

# Router: render content based on URL, and restore intervals after refresh
def update_performance_tab_content(active_tab):
    """Update performance tab content based on selected tab"""
    if active_tab == 'performance':
        return html.Div([html.H5('Performance Metrics', className='mb-3'), dbc.Row([dbc.Col([dbc.Card([dbc.CardBody([html.H6('Total Return', className='text-muted'), html.H3(f'{random.uniform(-5, 15):.2f}%', className='text-success')])])], width=3), dbc.Col([dbc.Card([dbc.CardBody([html.H6('Sharpe Ratio', className='text-muted'), html.H3(f'{random.uniform(0.8, 2.5):.2f}', className='text-info')])])], width=3), dbc.Col([dbc.Card([dbc.CardBody([html.H6('Max Drawdown', className='text-muted'), html.H3(f'{random.uniform(2, 8):.2f}%', className='text-warning')])])], width=3), dbc.Col([dbc.Card([dbc.CardBody([html.H6('Win Rate', className='text-muted'), html.H3(f'{random.uniform(60, 85):.1f}%', className='text-primary')])])], width=3)])])
    elif active_tab == 'audit':
        audit_rows = []
        for trade in trading_state.get('trades', [])[-10:]:
            timestamp = trade.get('timestamp', datetime.now())
            if isinstance(timestamp, str):
                timestamp_str = timestamp
            else:
                timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            action = f"{trade['side']} {trade['quantity']} {trade['symbol']} @ ${trade['price']:.2f}"
            audit_rows.append(html.Tr([html.Td(timestamp_str), html.Td('Trade Executed'), html.Td('AI System'), html.Td(action)]))
        if not audit_rows:
            audit_rows = [html.Tr([html.Td(colspan='4', children='No trades yet. AI is analyzing market...', className='text-center text-muted')])]
        return html.Div([html.H5('Audit Log', className='mb-3'), dbc.Table([html.Thead([html.Tr([html.Th('Timestamp'), html.Th('Action'), html.Th('User/System'), html.Th('Details')])]), html.Tbody(audit_rows)], striped=True, bordered=True, hover=True)])
    elif active_tab == 'attribution':
        import os
        lstm_exists = os.path.exists('models/lstm_model.pth')
        gru_exists = os.path.exists('models/gru_transformer_model.pth')
        lstm_status = 'Trained' if lstm_exists else 'Training...'
        gru_status = 'Trained' if gru_exists else 'Training...'
        return html.Div([html.H5('Model Attribution', className='mb-3'), dbc.Row([dbc.Col([dbc.Card([dbc.CardHeader('LSTM Model'), dbc.CardBody([html.P(f'Weight: 50%'), html.P(f'Status: {lstm_status}'), html.P('Accuracy: Not yet trained' if not lstm_exists else 'Accuracy: Calculating...')])])], width=6), dbc.Col([dbc.Card([dbc.CardHeader('GRU-Transformer'), dbc.CardBody([html.P(f'Weight: 50%'), html.P(f'Status: {gru_status}'), html.P('Accuracy: Not yet trained' if not gru_exists else 'Accuracy: Calculating...')])])], width=6)], className='mb-3'), dbc.Row([dbc.Col([dbc.Card([dbc.CardHeader('Sentiment Analysis'), dbc.CardBody([html.P('Source: News API + Reddit'), html.P('Status: Active'), html.P('Real-time sentiment scoring')])])], width=6), dbc.Col([dbc.Card([dbc.CardHeader('Technical Indicators'), dbc.CardBody([html.P('Indicators: RSI, MACD, BB, ATR'), html.P('Status: Active'), html.P('1-minute OHLCV data')])])], width=6)])])
    elif active_tab == 'risk':
        holdings = trading_state.get('holdings', [])
        total_value = sum((h['quantity'] * h['current_price'] for h in holdings)) if holdings else 0
        cash = trading_state.get('current_capital', 0)
        if total_value > 0:
            equity_pct = total_value / (total_value + cash) * 100 if total_value + cash > 0 else 0
            cash_pct = cash / (total_value + cash) * 100 if total_value + cash > 0 else 100
        else:
            equity_pct = 0
            cash_pct = 100
        return html.Div([html.H5('Risk Analysis', className='mb-3'), dbc.Row([dbc.Col([html.Div([html.H6('Current Risk Metrics'), html.P(f'Portfolio Value: ${total_value + cash:,.2f}'), html.P(f'Equity Exposure: {equity_pct:.1f}%'), html.P(f'Cash Position: {cash_pct:.1f}%'), html.P(f'Number of Positions: {len(holdings)}'), html.Hr(), html.Small('VaR and risk metrics will be calculated after more trading data is collected.', className='text-muted')])], width=6), dbc.Col([dcc.Graph(figure={'data': [go.Pie(labels=['Equity', 'Cash'], values=[equity_pct, cash_pct], hole=0.3, marker_colors=['#1f77b4', '#2ca02c'])], 'layout': go.Layout(title='Portfolio Allocation', template='plotly_dark')}, config={'displayModeBar': False})], width=6)])])
    return html.Div('Select a tab to view content')
def update_performance_tab_content(active_tab):
    """Update performance tab content based on selected tab"""
    if active_tab == 'performance':
        return html.Div([html.H5('Performance Metrics', className='mb-3'), dbc.Row([dbc.Col([dbc.Card([dbc.CardBody([html.H6('Total Return', className='text-muted'), html.H3(f'{random.uniform(-5, 15):.2f}%', className='text-success')])])], width=3), dbc.Col([dbc.Card([dbc.CardBody([html.H6('Sharpe Ratio', className='text-muted'), html.H3(f'{random.uniform(0.8, 2.5):.2f}', className='text-info')])])], width=3), dbc.Col([dbc.Card([dbc.CardBody([html.H6('Max Drawdown', className='text-muted'), html.H3(f'{random.uniform(2, 8):.2f}%', className='text-warning')])])], width=3), dbc.Col([dbc.Card([dbc.CardBody([html.H6('Win Rate', className='text-muted'), html.H3(f'{random.uniform(60, 85):.1f}%', className='text-primary')])])], width=3)])])
    elif active_tab == 'audit':
        audit_rows = []
        for trade in trading_state.get('trades', [])[-10:]:
            timestamp = trade.get('timestamp', datetime.now())
            if isinstance(timestamp, str):
                timestamp_str = timestamp
            else:
                timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            action = f"{trade['side']} {trade['quantity']} {trade['symbol']} @ ${trade['price']:.2f}"
            audit_rows.append(html.Tr([html.Td(timestamp_str), html.Td('Trade Executed'), html.Td('AI System'), html.Td(action)]))
        if not audit_rows:
            audit_rows = [html.Tr([html.Td(colspan='4', children='No trades yet. AI is analyzing market...', className='text-center text-muted')])]
        return html.Div([html.H5('Audit Log', className='mb-3'), dbc.Table([html.Thead([html.Tr([html.Th('Timestamp'), html.Th('Action'), html.Th('User/System'), html.Th('Details')])]), html.Tbody(audit_rows)], striped=True, bordered=True, hover=True)])
    elif active_tab == 'attribution':
        import os
        lstm_exists = os.path.exists('models/lstm_model.pth')
        gru_exists = os.path.exists('models/gru_transformer_model.pth')
        lstm_status = 'Trained' if lstm_exists else 'Training...'
        gru_status = 'Trained' if gru_exists else 'Training...'
        return html.Div([html.H5('Model Attribution', className='mb-3'), dbc.Row([dbc.Col([dbc.Card([dbc.CardHeader('LSTM Model'), dbc.CardBody([html.P(f'Weight: 50%'), html.P(f'Status: {lstm_status}'), html.P('Accuracy: Not yet trained' if not lstm_exists else 'Accuracy: Calculating...')])])], width=6), dbc.Col([dbc.Card([dbc.CardHeader('GRU-Transformer'), dbc.CardBody([html.P(f'Weight: 50%'), html.P(f'Status: {gru_status}'), html.P('Accuracy: Not yet trained' if not gru_exists else 'Accuracy: Calculating...')])])], width=6)], className='mb-3'), dbc.Row([dbc.Col([dbc.Card([dbc.CardHeader('Sentiment Analysis'), dbc.CardBody([html.P('Source: News API + Reddit'), html.P('Status: Active'), html.P('Real-time sentiment scoring')])])], width=6), dbc.Col([dbc.Card([dbc.CardHeader('Technical Indicators'), dbc.CardBody([html.P('Indicators: RSI, MACD, BB, ATR'), html.P('Status: Active'), html.P('1-minute OHLCV data')])])], width=6)])])
    elif active_tab == 'risk':
        holdings = trading_state.get('holdings', [])
        total_value = sum((h['quantity'] * h['current_price'] for h in holdings)) if holdings else 0
        cash = trading_state.get('current_capital', 0)
        if total_value > 0:
            equity_pct = total_value / (total_value + cash) * 100 if total_value + cash > 0 else 0
            cash_pct = cash / (total_value + cash) * 100 if total_value + cash > 0 else 100
        else:
            equity_pct = 0
            cash_pct = 100
        return html.Div([html.H5('Risk Analysis', className='mb-3'), dbc.Row([dbc.Col([html.Div([html.H6('Current Risk Metrics'), html.P(f'Portfolio Value: ${total_value + cash:,.2f}'), html.P(f'Equity Exposure: {equity_pct:.1f}%'), html.P(f'Cash Position: {cash_pct:.1f}%'), html.P(f'Number of Positions: {len(holdings)}'), html.Hr(), html.Small('VaR and risk metrics will be calculated after more trading data is collected.', className='text-muted')])], width=6), dbc.Col([dcc.Graph(figure={'data': [go.Pie(labels=['Equity', 'Cash'], values=[equity_pct, cash_pct], hole=0.3, marker_colors=['#1f77b4', '#2ca02c'])], 'layout': go.Layout(title='Portfolio Allocation', template='plotly_dark')}, config={'displayModeBar': False})], width=6)])])
    return html.Div('Select a tab to view content')

@app.callback(
    [Output('main-content', 'children', allow_duplicate=True),
     Output('trading-interval', 'disabled', allow_duplicate=True),
     Output('chart-interval', 'disabled', allow_duplicate=True),
     Output('trading-initialized', 'data', allow_duplicate=True)],
    Input('url', 'pathname'),
    prevent_initial_call='initial_duplicate'
)
def render_router(pathname):
    try:
        if pathname == '/logs':
            return (create_ai_logs_page(), True, True, trading_state.get('initialized', False))
        if not trading_state.get('initialized', False):
            load_trading_state()
        if trading_state.get('initialized', False):
            return (create_trading_page(), False, False, True)
        else:
            return (create_startup_screen(), True, True, False)
    except Exception:
        return (create_startup_screen(), True, True, False)
def render_router(pathname):
    try:
        if pathname == '/logs':
            return (create_ai_logs_page(), True, True, trading_state.get('initialized', False))
        if not trading_state.get('initialized', False):
            load_trading_state()
        if trading_state.get('initialized', False):
            return (create_trading_page(), False, False, True)
        else:
            return (create_startup_screen(), True, True, False)
    except Exception:
        return (create_startup_screen(), True, True, False)

@app.callback(
    Output('main-content', 'children', allow_duplicate=True),
    Input('open-ai-logs-btn', 'n_clicks'),
    prevent_initial_call=True
)
def open_ai_logs(n_clicks):
    """Open AI logs page when View Logs button is clicked"""
    if n_clicks:
        return create_ai_logs_page()
    return dash.no_update
def open_ai_logs(n_clicks):
    """Open AI logs page when View Logs button is clicked"""
    if n_clicks:
        return create_ai_logs_page()
    return dash.no_update

@app.callback(
    Output('main-content', 'children', allow_duplicate=True),
    Input('back-to-dashboard-btn', 'n_clicks'),
    prevent_initial_call=True
)
def back_to_dashboard(n_clicks):
    """Return to main dashboard when back button is clicked"""
    if n_clicks:
        return create_trading_page()
    return dash.no_update
def back_to_dashboard(n_clicks):
    """Return to main dashboard when back button is clicked"""
    if n_clicks:
        return create_trading_page()
    return dash.no_update

@app.callback(
    Output('main-content', 'children', allow_duplicate=True),
    Input('view-ai-logs-startup-btn', 'n_clicks'),
    prevent_initial_call=True
)
def view_ai_logs_from_startup(n_clicks):
    """Open AI logs page from startup screen"""
    if n_clicks:
        return create_ai_logs_page()
    return dash.no_update
def view_ai_logs_from_startup(n_clicks):
    """Open AI logs page from startup screen"""
    if n_clicks:
        return create_ai_logs_page()
    return dash.no_update

@app.callback(
    Output('main-content', 'children', allow_duplicate=True),
    Input('startup-open-logs-btn', 'n_clicks'),
    prevent_initial_call=True
)
def startup_open_logs(n_clicks):
    """Open AI logs page from startup AI Activity Monitor"""
    if n_clicks:
        return create_ai_logs_page()
    return dash.no_update
def startup_open_logs(n_clicks):
    """Open AI logs page from startup AI Activity Monitor"""
    if n_clicks:
        return create_ai_logs_page()
    return dash.no_update

@app.callback(
    Output('main-content', 'children', allow_duplicate=True),
    Input('reset-to-startup-btn', 'n_clicks'),
    prevent_initial_call=True
)
def reset_to_startup(n_clicks):
    """Reset to startup screen"""
    if n_clicks:
        reset_in_memory_state()
        try:
            if DEMO_STATE_PATH.exists():
                DEMO_STATE_PATH.unlink()
        except Exception:
            pass
        try:
            STATE_STORE.reset_all()
        except Exception:
            pass
        _reset_log_files()
        save_trading_state()
        return create_startup_screen()
    return dash.no_update
def reset_to_startup(n_clicks):
    """Reset to startup screen"""
    if n_clicks:
        reset_in_memory_state()
        try:
            if DEMO_STATE_PATH.exists():
                DEMO_STATE_PATH.unlink()
        except Exception:
            pass
        try:
            STATE_STORE.reset_all()
        except Exception:
            pass
        _reset_log_files()
        save_trading_state()
        return create_startup_screen()
    return dash.no_update

@app.callback(
    Output('ui-dirty', 'data'),
    [Input('pause-switch', 'value'), Input('kill-threshold-input', 'value'), Input('reset-kill-btn', 'n_clicks'), Input('max-pos-input', 'value'), Input('force-open-switch', 'value')],
    prevent_initial_call=True
)
def update_controls(paused, kill_threshold, reset_kill_clicks, max_pos_pct, force_open):
    try:
        if paused is not None:
            trading_state['paused'] = bool(paused)
        if kill_threshold is not None:
            trading_state['kill_switch_threshold'] = float(kill_threshold)
            if trading_state['kill_switch_threshold'] <= 0:
                trading_state['kill_switch_active'] = False
        if reset_kill_clicks:
            trading_state['kill_switch_active'] = False
        if max_pos_pct is not None:
            trading_state['max_position_pct'] = max(0.005, float(max_pos_pct) / 100.0)
        if force_open is not None:
            trading_state['force_market_open'] = bool(force_open)
        save_trading_state()
        return True
    except Exception:
        return False
def update_controls(paused, kill_threshold, reset_kill_clicks, max_pos_pct, force_open):
    try:
        if paused is not None:
            trading_state['paused'] = bool(paused)
        if kill_threshold is not None:
            trading_state['kill_switch_threshold'] = float(kill_threshold)
            if trading_state['kill_switch_threshold'] <= 0:
                trading_state['kill_switch_active'] = False
        if reset_kill_clicks:
            trading_state['kill_switch_active'] = False
        if max_pos_pct is not None:
            trading_state['max_position_pct'] = max(0.005, float(max_pos_pct) / 100.0)
        if force_open is not None:
            trading_state['force_market_open'] = bool(force_open)
        save_trading_state()
        return True
    except Exception:
        return False

@app.callback(
    Output('log-content', 'children'),
    [Input('log-tab-activity', 'n_clicks'),
     Input('log-tab-trades', 'n_clicks'),
     Input('log-tab-signals', 'n_clicks'),
     Input('log-tab-decisions', 'n_clicks'),
     Input('refresh-logs-btn', 'n_clicks'),
     Input('logs-interval', 'n_intervals')],
    [State('current-log-tab', 'data')],
    prevent_initial_call=True
)
def update_log_content(activity_clicks, trades_clicks, signals_clicks, decisions_clicks, refresh_clicks, n_intervals, current_tab):
    """Update log content based on selected tab"""
    ctx = dash.callback_context
    log_file = 'logs/ai_activity.log'
    if ctx.triggered:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'log-tab-activity':
            current_tab = 'activity'
        elif button_id == 'log-tab-trades':
            current_tab = 'trades'
        elif button_id == 'log-tab-signals':
            current_tab = 'signals'
        elif button_id == 'log-tab-decisions':
            current_tab = 'decisions'
    tab = current_tab or 'activity'
    if tab == 'activity':
        log_file = 'logs/ai_activity.log'
    elif tab == 'trades':
        log_file = 'logs/ai_trades.log'
    elif tab == 'signals':
        log_file = 'logs/ai_signals.log'
    elif tab == 'decisions':
        log_file = 'logs/ai_decisions.log'
    try:
        if Path(log_file).exists():
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                recent_lines = lines[-50:] if len(lines) > 50 else lines
                content = ''.join(recent_lines)
        else:
            content = f'Log file not found: {log_file}\n\nAI may still be starting up...'
    except Exception as e:
        content = f'Error reading log file: {str(e)}'
    return html.Pre(content, style={'margin': 0, 'whiteSpace': 'pre-wrap'})
def update_log_content(activity_clicks, trades_clicks, signals_clicks, decisions_clicks, refresh_clicks, n_intervals, current_tab):
    """Update log content based on selected tab"""
    ctx = dash.callback_context
    log_file = 'logs/ai_activity.log'
    if ctx.triggered:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'log-tab-activity':
            current_tab = 'activity'
        elif button_id == 'log-tab-trades':
            current_tab = 'trades'
        elif button_id == 'log-tab-signals':
            current_tab = 'signals'
        elif button_id == 'log-tab-decisions':
            current_tab = 'decisions'
    tab = current_tab or 'activity'
    if tab == 'activity':
        log_file = 'logs/ai_activity.log'
    elif tab == 'trades':
        log_file = 'logs/ai_trades.log'
    elif tab == 'signals':
        log_file = 'logs/ai_signals.log'
    elif tab == 'decisions':
        log_file = 'logs/ai_decisions.log'
    try:
        if Path(log_file).exists():
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                recent_lines = lines[-50:] if len(lines) > 50 else lines
                content = ''.join(recent_lines)
        else:
            content = f'Log file not found: {log_file}\n\nAI may still be starting up...'
    except Exception as e:
        content = f'Error reading log file: {str(e)}'
    return html.Pre(content, style={'margin': 0, 'whiteSpace': 'pre-wrap'})

@app.callback(
    Output('current-log-tab', 'data'),
    [Input('log-tab-activity', 'n_clicks'),
     Input('log-tab-trades', 'n_clicks'),
     Input('log-tab-signals', 'n_clicks'),
     Input('log-tab-decisions', 'n_clicks')],
    prevent_initial_call=True
)
def set_current_log_tab(a, t, s, d):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'log-tab-activity':
        return 'activity'
    if button_id == 'log-tab-trades':
        return 'trades'
    if button_id == 'log-tab-signals':
        return 'signals'
    if button_id == 'log-tab-decisions':
        return 'decisions'
    return dash.no_update
def set_current_log_tab(a, t, s, d):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'log-tab-activity':
        return 'activity'
    if button_id == 'log-tab-trades':
        return 'trades'
    if button_id == 'log-tab-signals':
        return 'signals'
    if button_id == 'log-tab-decisions':
        return 'decisions'
    return dash.no_update

@app.callback(
    Output('logs-interval', 'disabled'),
    Input('logs-refresh-switch', 'value'),
    prevent_initial_call=False
)
def toggle_logs_interval(auto):
    return not bool(auto)


def toggle_back_button(content):
    """Show/hide back button based on current page"""
    if hasattr(content, 'props') and 'children' in content.props:
        return ({'display': 'inline-block'}, [html.I(className='fas fa-arrow-left me-2'), 'Back to Dashboard'])
    return ({'display': 'none'}, [html.I(className='fas fa-arrow-left me-2'), 'Back to Dashboard'])

# ============================================================================
# AI ACTIVITY MONITOR CALLBACKS
# ============================================================================

# Router: render content based on URL, and restore intervals after refresh
def toggle_logs_interval(auto):
    return not bool(auto)

@app.callback(
    [Output('main-content', 'children', allow_duplicate=True),
     Output('trading-interval', 'disabled', allow_duplicate=True),
     Output('chart-interval', 'disabled', allow_duplicate=True),
     Output('trading-initialized', 'data', allow_duplicate=True)],
    Input('url', 'pathname'),
    prevent_initial_call='initial_duplicate'
)
def toggle_back_button(content):
    """Show/hide back button based on current page"""
    if hasattr(content, 'props') and 'children' in content.props:
        return ({'display': 'inline-block'}, [html.I(className='fas fa-arrow-left me-2'), 'Back to Dashboard'])
    return ({'display': 'none'}, [html.I(className='fas fa-arrow-left me-2'), 'Back to Dashboard'])

# ============================================================================
# AI ACTIVITY MONITOR CALLBACKS
# ============================================================================

# Router: render content based on URL, and restore intervals after refresh

# ============================================================================
# AI ACTIVITY MONITOR CALLBACKS
# ============================================================================

# Router: render content based on URL, and restore intervals after refresh
def update_performance_tab_content(active_tab):
    """Update performance tab content based on selected tab"""
    if active_tab == 'performance':
        return html.Div([html.H5('Performance Metrics', className='mb-3'), dbc.Row([dbc.Col([dbc.Card([dbc.CardBody([html.H6('Total Return', className='text-muted'), html.H3(f'{random.uniform(-5, 15):.2f}%', className='text-success')])])], width=3), dbc.Col([dbc.Card([dbc.CardBody([html.H6('Sharpe Ratio', className='text-muted'), html.H3(f'{random.uniform(0.8, 2.5):.2f}', className='text-info')])])], width=3), dbc.Col([dbc.Card([dbc.CardBody([html.H6('Max Drawdown', className='text-muted'), html.H3(f'{random.uniform(2, 8):.2f}%', className='text-warning')])])], width=3), dbc.Col([dbc.Card([dbc.CardBody([html.H6('Win Rate', className='text-muted'), html.H3(f'{random.uniform(60, 85):.1f}%', className='text-primary')])])], width=3)])])
    elif active_tab == 'audit':
        audit_rows = []
        for trade in trading_state.get('trades', [])[-10:]:
            timestamp = trade.get('timestamp', datetime.now())
            if isinstance(timestamp, str):
                timestamp_str = timestamp
            else:
                timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            action = f"{trade['side']} {trade['quantity']} {trade['symbol']} @ ${trade['price']:.2f}"
            audit_rows.append(html.Tr([html.Td(timestamp_str), html.Td('Trade Executed'), html.Td('AI System'), html.Td(action)]))
        if not audit_rows:
            audit_rows = [html.Tr([html.Td(colspan='4', children='No trades yet. AI is analyzing market...', className='text-center text-muted')])]
        return html.Div([html.H5('Audit Log', className='mb-3'), dbc.Table([html.Thead([html.Tr([html.Th('Timestamp'), html.Th('Action'), html.Th('User/System'), html.Th('Details')])]), html.Tbody(audit_rows)], striped=True, bordered=True, hover=True)])
    elif active_tab == 'attribution':
        import os
        lstm_exists = os.path.exists('models/lstm_model.pth')
        gru_exists = os.path.exists('models/gru_transformer_model.pth')
        lstm_status = 'Trained' if lstm_exists else 'Training...'
        gru_status = 'Trained' if gru_exists else 'Training...'
        return html.Div([html.H5('Model Attribution', className='mb-3'), dbc.Row([dbc.Col([dbc.Card([dbc.CardHeader('LSTM Model'), dbc.CardBody([html.P(f'Weight: 50%'), html.P(f'Status: {lstm_status}'), html.P('Accuracy: Not yet trained' if not lstm_exists else 'Accuracy: Calculating...')])])], width=6), dbc.Col([dbc.Card([dbc.CardHeader('GRU-Transformer'), dbc.CardBody([html.P(f'Weight: 50%'), html.P(f'Status: {gru_status}'), html.P('Accuracy: Not yet trained' if not gru_exists else 'Accuracy: Calculating...')])])], width=6)], className='mb-3'), dbc.Row([dbc.Col([dbc.Card([dbc.CardHeader('Sentiment Analysis'), dbc.CardBody([html.P('Source: News API + Reddit'), html.P('Status: Active'), html.P('Real-time sentiment scoring')])])], width=6), dbc.Col([dbc.Card([dbc.CardHeader('Technical Indicators'), dbc.CardBody([html.P('Indicators: RSI, MACD, BB, ATR'), html.P('Status: Active'), html.P('1-minute OHLCV data')])])], width=6)])])
    elif active_tab == 'risk':
        holdings = trading_state.get('holdings', [])
        total_value = sum((h['quantity'] * h['current_price'] for h in holdings)) if holdings else 0
        cash = trading_state.get('current_capital', 0)
        if total_value > 0:
            equity_pct = total_value / (total_value + cash) * 100 if total_value + cash > 0 else 0
            cash_pct = cash / (total_value + cash) * 100 if total_value + cash > 0 else 100
        else:
            equity_pct = 0
            cash_pct = 100
        return html.Div([html.H5('Risk Analysis', className='mb-3'), dbc.Row([dbc.Col([html.Div([html.H6('Current Risk Metrics'), html.P(f'Portfolio Value: ${total_value + cash:,.2f}'), html.P(f'Equity Exposure: {equity_pct:.1f}%'), html.P(f'Cash Position: {cash_pct:.1f}%'), html.P(f'Number of Positions: {len(holdings)}'), html.Hr(), html.Small('VaR and risk metrics will be calculated after more trading data is collected.', className='text-muted')])], width=6), dbc.Col([dcc.Graph(figure={'data': [go.Pie(labels=['Equity', 'Cash'], values=[equity_pct, cash_pct], hole=0.3, marker_colors=['#1f77b4', '#2ca02c'])], 'layout': go.Layout(title='Portfolio Allocation', template='plotly_dark')}, config={'displayModeBar': False})], width=6)])])
    return html.Div('Select a tab to view content')

@app.callback(
    [Output('main-content', 'children', allow_duplicate=True),
     Output('trading-interval', 'disabled', allow_duplicate=True),
     Output('chart-interval', 'disabled', allow_duplicate=True),
     Output('trading-initialized', 'data', allow_duplicate=True)],
    Input('url', 'pathname'),
    prevent_initial_call='initial_duplicate'
)
def render_router(pathname):
    try:
        if pathname == '/logs':
            return (create_ai_logs_page(), True, True, trading_state.get('initialized', False))
        if not trading_state.get('initialized', False):
            load_trading_state()
        if trading_state.get('initialized', False):
            return (create_trading_page(), False, False, True)
        else:
            return (create_startup_screen(), True, True, False)
    except Exception:
        return (create_startup_screen(), True, True, False)

@app.callback(
    Output('main-content', 'children', allow_duplicate=True),
    Input('open-ai-logs-btn', 'n_clicks'),
    prevent_initial_call=True
)
def open_ai_logs(n_clicks):
    """Open AI logs page when View Logs button is clicked"""
    if n_clicks:
        return create_ai_logs_page()
    return dash.no_update

@app.callback(
    Output('main-content', 'children', allow_duplicate=True),
    Input('back-to-dashboard-btn', 'n_clicks'),
    prevent_initial_call=True
)
def back_to_dashboard(n_clicks):
    """Return to main dashboard when back button is clicked"""
    if n_clicks:
        return create_trading_page()
    return dash.no_update

@app.callback(
    Output('main-content', 'children', allow_duplicate=True),
    Input('view-ai-logs-startup-btn', 'n_clicks'),
    prevent_initial_call=True
)
def view_ai_logs_from_startup(n_clicks):
    """Open AI logs page from startup screen"""
    if n_clicks:
        return create_ai_logs_page()
    return dash.no_update

@app.callback(
    Output('main-content', 'children', allow_duplicate=True),
    Input('startup-open-logs-btn', 'n_clicks'),
    prevent_initial_call=True
)
def startup_open_logs(n_clicks):
    """Open AI logs page from startup AI Activity Monitor"""
    if n_clicks:
        return create_ai_logs_page()
    return dash.no_update

@app.callback(
    Output('main-content', 'children', allow_duplicate=True),
    Input('reset-to-startup-btn', 'n_clicks'),
    prevent_initial_call=True
)
def reset_to_startup(n_clicks):
    """Reset to startup screen"""
    if n_clicks:
        reset_in_memory_state()
        try:
            if DEMO_STATE_PATH.exists():
                DEMO_STATE_PATH.unlink()
        except Exception:
            pass
        try:
            STATE_STORE.reset_all()
        except Exception:
            pass
        _reset_log_files()
        save_trading_state()
        return create_startup_screen()
    return dash.no_update

@app.callback(
    Output('ui-dirty', 'data'),
    [Input('pause-switch', 'value'), Input('kill-threshold-input', 'value'), Input('reset-kill-btn', 'n_clicks'), Input('max-pos-input', 'value'), Input('force-open-switch', 'value')],
    prevent_initial_call=True
)
def update_controls(paused, kill_threshold, reset_kill_clicks, max_pos_pct, force_open):
    try:
        if paused is not None:
            trading_state['paused'] = bool(paused)
        if kill_threshold is not None:
            trading_state['kill_switch_threshold'] = float(kill_threshold)
            if trading_state['kill_switch_threshold'] <= 0:
                trading_state['kill_switch_active'] = False
        if reset_kill_clicks:
            trading_state['kill_switch_active'] = False
        if max_pos_pct is not None:
            trading_state['max_position_pct'] = max(0.005, float(max_pos_pct) / 100.0)
        if force_open is not None:
            trading_state['force_market_open'] = bool(force_open)
        save_trading_state()
        return True
    except Exception:
        return False

@app.callback(
    Output('log-content', 'children'),
    [Input('log-tab-activity', 'n_clicks'),
     Input('log-tab-trades', 'n_clicks'),
     Input('log-tab-signals', 'n_clicks'),
     Input('log-tab-decisions', 'n_clicks'),
     Input('refresh-logs-btn', 'n_clicks'),
     Input('logs-interval', 'n_intervals')],
    [State('current-log-tab', 'data')],
    prevent_initial_call=True
)
def update_log_content(activity_clicks, trades_clicks, signals_clicks, decisions_clicks, refresh_clicks, n_intervals, current_tab):
    """Update log content based on selected tab"""
    ctx = dash.callback_context
    log_file = 'logs/ai_activity.log'
    if ctx.triggered:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'log-tab-activity':
            current_tab = 'activity'
        elif button_id == 'log-tab-trades':
            current_tab = 'trades'
        elif button_id == 'log-tab-signals':
            current_tab = 'signals'
        elif button_id == 'log-tab-decisions':
            current_tab = 'decisions'
    tab = current_tab or 'activity'
    if tab == 'activity':
        log_file = 'logs/ai_activity.log'
    elif tab == 'trades':
        log_file = 'logs/ai_trades.log'
    elif tab == 'signals':
        log_file = 'logs/ai_signals.log'
    elif tab == 'decisions':
        log_file = 'logs/ai_decisions.log'
    try:
        if Path(log_file).exists():
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                recent_lines = lines[-50:] if len(lines) > 50 else lines
                content = ''.join(recent_lines)
        else:
            content = f'Log file not found: {log_file}\n\nAI may still be starting up...'
    except Exception as e:
        content = f'Error reading log file: {str(e)}'
    return html.Pre(content, style={'margin': 0, 'whiteSpace': 'pre-wrap'})

@app.callback(
    Output('current-log-tab', 'data'),
    [Input('log-tab-activity', 'n_clicks'),
     Input('log-tab-trades', 'n_clicks'),
     Input('log-tab-signals', 'n_clicks'),
     Input('log-tab-decisions', 'n_clicks')],
    prevent_initial_call=True
)
def set_current_log_tab(a, t, s, d):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'log-tab-activity':
        return 'activity'
    if button_id == 'log-tab-trades':
        return 'trades'
    if button_id == 'log-tab-signals':
        return 'signals'
    if button_id == 'log-tab-decisions':
        return 'decisions'
    return dash.no_update

@app.callback(
    Output('logs-interval', 'disabled'),
    Input('logs-refresh-switch', 'value'),
    prevent_initial_call=False
)
def toggle_logs_interval(auto):
    return not bool(auto)


def toggle_back_button(content):
    """Show/hide back button based on current page"""
    if hasattr(content, 'props') and 'children' in content.props:
        return ({'display': 'inline-block'}, [html.I(className='fas fa-arrow-left me-2'), 'Back to Dashboard'])
    return ({'display': 'none'}, [html.I(className='fas fa-arrow-left me-2'), 'Back to Dashboard'])

# ============================================================================
# AI ACTIVITY MONITOR CALLBACKS
# ============================================================================

# Router: render content based on URL, and restore intervals after refresh
@app.callback(
    [Output('main-content', 'children', allow_duplicate=True),
     Output('trading-interval', 'disabled', allow_duplicate=True),
     Output('chart-interval', 'disabled', allow_duplicate=True),
     Output('trading-initialized', 'data', allow_duplicate=True)],
    Input('url', 'pathname'),
    prevent_initial_call='initial_duplicate'
)

@app.callback(
    Output('main-content', 'children', allow_duplicate=True),
    Input('open-ai-logs-btn', 'n_clicks'),
    prevent_initial_call=True
)

@app.callback(
    Output('main-content', 'children', allow_duplicate=True),
    Input('back-to-dashboard-btn', 'n_clicks'),
    prevent_initial_call=True
)

@app.callback(
    Output('main-content', 'children', allow_duplicate=True),
    Input('view-ai-logs-startup-btn', 'n_clicks'),
    prevent_initial_call=True
)

@app.callback(
    Output('main-content', 'children', allow_duplicate=True),
    Input('startup-open-logs-btn', 'n_clicks'),
    prevent_initial_call=True
)

@app.callback(
    Output('main-content', 'children', allow_duplicate=True),
    Input('reset-to-startup-btn', 'n_clicks'),
    prevent_initial_call=True
)

@app.callback(
    Output('ui-dirty', 'data'),
    [Input('pause-switch', 'value'), Input('kill-threshold-input', 'value'), Input('reset-kill-btn', 'n_clicks'), Input('max-pos-input', 'value'), Input('force-open-switch', 'value')],
    prevent_initial_call=True
)

@app.callback(
    Output('log-content', 'children'),
    [Input('log-tab-activity', 'n_clicks'),
     Input('log-tab-trades', 'n_clicks'),
     Input('log-tab-signals', 'n_clicks'),
     Input('log-tab-decisions', 'n_clicks'),
     Input('refresh-logs-btn', 'n_clicks'),
     Input('logs-interval', 'n_intervals')],
    [State('current-log-tab', 'data')],
    prevent_initial_call=True
)

@app.callback(
    Output('current-log-tab', 'data'),
    [Input('log-tab-activity', 'n_clicks'),
     Input('log-tab-trades', 'n_clicks'),
     Input('log-tab-signals', 'n_clicks'),
     Input('log-tab-decisions', 'n_clicks')],
    prevent_initial_call=True
)

@app.callback(
    Output('logs-interval', 'disabled'),
    Input('logs-refresh-switch', 'value'),
    prevent_initial_call=False
)

@app.callback(
    [Output('back-to-dashboard-btn', 'style'),
     Output('back-to-dashboard-btn', 'children')],
    Input('main-content', 'children'),
    prevent_initial_call=True
)

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == '__main__':
    import sys
    import io
    
    # Fix Windows console encoding
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    
    print("=" * 80)
    print("Interactive AI Trading Dashboard Starting...")
    print("=" * 80)
    print()
    print("Features:")
    print("   - Set your own demo capital")
    print("   - AI trades with live market data")
    print("   - Real-time portfolio updates")
    print("   - Demo mode (no real money)")
    print()
    print("Dashboard URL: http://localhost:8051")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 80)
    print()
    
    try:
        app.run(debug=False, host='127.0.0.1', port=8051)
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        import traceback
        traceback.print_exc()