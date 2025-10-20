"""
Modern AI Trading Dashboard - Default Implementation
Based on the Dashboard/index.html design system
"""

import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
import logging
import sys
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import trading system components
try:
    from src.dashboard import trading_state, STATE_STORE, DEMO_STATE_PATH, reset_in_memory_state, load_trading_state, save_trading_state
    from src.dashboard.services import (
        get_live_price, get_demo_price, real_ai_trade, simulate_ai_trade,
        update_holdings_prices, generate_ai_signals, is_market_open
    )
    from src.dashboard.portfolio import (
        generate_portfolio_data, generate_holdings, generate_recent_trades,
        create_summary_cards, create_holdings_table, create_recent_trades_table
    )
    TRADING_SYSTEM_AVAILABLE = True
except Exception as e:
    logger.warning(f"Trading system unavailable: {e}")
    TRADING_SYSTEM_AVAILABLE = False

# Initialize Dash app with modern theme
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
    ],
    suppress_callback_exceptions=True
)

app.title = "AI Trading Bot Dashboard"

# Custom CSS for modern design system
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
                /* Color System */
                --color-primary: #1FB8CD;
                --color-primary-hover: #1A9FB3;
                --color-success: #1FB8CD;
                --color-error: #B4413C;
                --color-warning: #E68161;
                --color-info: #626C71;
                
                /* Background Colors - COMPLETELY DARK */
                --color-bg-primary: #0D1117;
                --color-bg-secondary: #161B22;
                --color-bg-dark: #0D1117;
                --color-bg-surface: #21262D;
                
                /* Text Colors - DARK THEME */
                --color-text-primary: #F0F6FC;
                --color-text-secondary: #8B949E;
                --color-text-light: #F0F6FC;
                
                /* Spacing */
                --space-xs: 4px;
                --space-sm: 8px;
                --space-md: 16px;
                --space-lg: 24px;
                --space-xl: 32px;
                
                /* Border Radius */
                --radius-sm: 6px;
                --radius-md: 8px;
                --radius-lg: 12px;
                --radius-full: 9999px;
                
                /* Shadows */
                --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.04), 0 1px 2px rgba(0, 0, 0, 0.02);
                --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.04), 0 2px 4px -1px rgba(0, 0, 0, 0.02);
                --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.04), 0 4px 6px -2px rgba(0, 0, 0, 0.02);
            }
            
            /* Dark mode support */
            @media (prefers-color-scheme: dark) {
                :root {
                    --color-bg-primary: #1F2121;
                    --color-bg-secondary: #262828;
                    --color-text-primary: #F5F5F5;
                    --color-text-secondary: #A7A9A9;
                }
            }
            
            body {
                background-color: var(--color-bg-primary) !important;
                color: var(--color-text-primary) !important;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                margin: 0;
                padding: 0;
            }
            
            /* Force dark theme on all elements */
            * {
                background-color: inherit;
                color: inherit;
            }
            
            .dash-table-container {
                background-color: var(--color-bg-secondary) !important;
                color: var(--color-text-primary) !important;
            }
            
            .dash-table-container .dash-spreadsheet-container {
                background-color: var(--color-bg-secondary) !important;
            }
            
            .dash-table-container table {
                background-color: var(--color-bg-secondary) !important;
                color: var(--color-text-primary) !important;
            }
            
            .dash-table-container th {
                background-color: var(--color-bg-surface) !important;
                color: var(--color-text-primary) !important;
                border-color: var(--color-text-secondary) !important;
            }
            
            .dash-table-container td {
                background-color: var(--color-bg-secondary) !important;
                color: var(--color-text-primary) !important;
                border-color: var(--color-text-secondary) !important;
            }
            
            .dashboard-container {
                min-height: 100vh;
                display: flex;
                flex-direction: column;
            }
            
            .navbar {
                background-color: var(--color-bg-secondary);
                border-bottom: 1px solid rgba(98, 108, 113, 0.2);
                padding: var(--space-md) var(--space-lg);
                display: flex;
                justify-content: space-between;
                align-items: center;
                box-shadow: var(--shadow-sm);
            }
            
            .navbar-brand {
                font-size: 1.25rem;
                font-weight: 600;
                color: var(--color-primary);
                display: flex;
                align-items: center;
                gap: var(--space-sm);
            }
            
            .navbar-info {
                display: flex;
                gap: var(--space-lg);
                align-items: center;
            }
            
            .status-indicator {
                display: flex;
                align-items: center;
                gap: var(--space-sm);
                font-size: 0.875rem;
            }
            
            .status-dot {
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background-color: var(--color-success);
            }
            
            .status-dot.warning {
                background-color: var(--color-warning);
            }
            
            .live-indicator {
                display: flex;
                align-items: center;
                gap: var(--space-xs);
                font-size: 0.75rem;
                color: var(--color-success);
            }
            
            .live-dot {
                width: 6px;
                height: 6px;
                background-color: var(--color-success);
                border-radius: 50%;
                animation: pulse 2s infinite;
            }
            
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.3; }
            }
            
            .main-content {
                flex: 1;
                padding: var(--space-lg);
                max-width: 1600px;
                margin: 0 auto;
                width: 100%;
            }
            
            .grid {
                display: grid;
                gap: var(--space-lg);
            }
            
            .grid-4 {
                grid-template-columns: repeat(4, 1fr);
            }
            
            .grid-2 {
                grid-template-columns: 1fr 1fr;
            }
            
            .grid-3-1 {
                grid-template-columns: 2fr 1fr;
            }
            
            @media (max-width: 1024px) {
                .grid-4 { grid-template-columns: repeat(2, 1fr); }
                .grid-3-1 { grid-template-columns: 1fr; }
            }
            
            @media (max-width: 640px) {
                .grid-4, .grid-2 { grid-template-columns: 1fr; }
            }
            
            .metric-card {
                background-color: var(--color-bg-secondary);
                padding: var(--space-lg);
                border-radius: var(--radius-lg);
                border: 1px solid rgba(98, 108, 113, 0.12);
                box-shadow: var(--shadow-sm);
                transition: box-shadow 0.2s ease;
            }
            
            .metric-card:hover {
                box-shadow: var(--shadow-md);
            }
            
            .metric-label {
                font-size: 0.875rem;
                color: var(--color-text-secondary);
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                margin-bottom: var(--space-sm);
            }
            
            .metric-value {
                font-size: 1.875rem;
                font-weight: 700;
                margin: var(--space-sm) 0;
                color: var(--color-text-primary);
            }
            
            .metric-value.positive {
                color: var(--color-success);
            }
            
            .metric-value.negative {
                color: var(--color-error);
            }
            
            .metric-change {
                font-size: 0.875rem;
                color: var(--color-text-secondary);
            }
            
            .section-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: var(--space-md);
            }
            
            .section-title {
                font-size: 1.25rem;
                font-weight: 600;
                color: var(--color-text-primary);
            }
            
            .table-container {
                background-color: var(--color-bg-secondary);
                border-radius: var(--radius-lg);
                border: 1px solid rgba(98, 108, 113, 0.12);
                overflow: hidden;
                box-shadow: var(--shadow-sm);
            }
            
            .table {
                width: 100%;
                border-collapse: collapse;
            }
            
            .table th {
                background-color: rgba(98, 108, 113, 0.12);
                padding: var(--space-md);
                text-align: left;
                font-weight: 600;
                font-size: 0.875rem;
                color: var(--color-text-primary);
                border-bottom: 1px solid rgba(98, 108, 113, 0.2);
            }
            
            .table td {
                padding: var(--space-md);
                border-bottom: 1px solid rgba(98, 108, 113, 0.12);
            }
            
            .table tbody tr:hover {
                background-color: rgba(98, 108, 113, 0.12);
            }
            
            .positive {
                color: var(--color-success);
            }
            
            .negative {
                color: var(--color-error);
            }
            
            .ai-agents {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: var(--space-md);
            }
            
            .agent-card {
                background-color: var(--color-bg-secondary);
                padding: var(--space-md);
                border-radius: var(--radius-md);
                border: 1px solid rgba(98, 108, 113, 0.12);
            }
            
            .agent-header {
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                margin-bottom: var(--space-sm);
            }
            
            .agent-name {
                font-size: 0.875rem;
                font-weight: 600;
                color: var(--color-text-primary);
            }
            
            .agent-status {
                font-size: 0.75rem;
                padding: 2px 6px;
                border-radius: var(--radius-full);
            }
            
            .agent-status.active {
                background-color: rgba(31, 184, 205, 0.15);
                color: var(--color-success);
            }
            
            .agent-status.idle {
                background-color: rgba(230, 129, 97, 0.15);
                color: var(--color-warning);
            }
            
            .agent-time {
                font-size: 0.75rem;
                color: var(--color-text-secondary);
            }
            
            .chart-container {
                background-color: var(--color-bg-secondary);
                padding: var(--space-lg);
                border-radius: var(--radius-lg);
                border: 1px solid rgba(98, 108, 113, 0.12);
                height: 300px;
                box-shadow: var(--shadow-sm);
            }
            
            .chart-title {
                font-size: 1rem;
                font-weight: 600;
                margin-bottom: var(--space-md);
                color: var(--color-text-primary);
            }
            
            .floating-logs-btn {
                position: fixed;
                bottom: 30px;
                right: 30px;
                background: var(--color-primary);
                color: white;
                border: none;
                border-radius: var(--radius-full);
                padding: var(--space-md) var(--space-lg);
                font-size: 0.875rem;
                font-weight: 500;
                cursor: pointer;
                box-shadow: var(--shadow-lg);
                transition: all 0.2s ease;
            }
            
            .floating-logs-btn:hover {
                background: var(--color-primary-hover);
                transform: translateY(-2px);
            }
            
            .footer {
                background-color: var(--color-bg-secondary);
                border-top: 1px solid rgba(98, 108, 113, 0.2);
                padding: var(--space-md) var(--space-lg);
                display: flex;
                justify-content: space-between;
                align-items: center;
                font-size: 0.875rem;
                color: var(--color-text-secondary);
            }
            
            .status-badge {
                display: inline-flex;
                align-items: center;
                padding: 4px 8px;
                border-radius: var(--radius-full);
                font-weight: 500;
                font-size: 0.75rem;
            }
            
            .status-badge.success {
                background-color: rgba(31, 184, 205, 0.15);
                color: var(--color-success);
            }
            
            .status-badge.error {
                background-color: rgba(180, 65, 60, 0.15);
                color: var(--color-error);
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

# Mock data for demonstration
MOCK_DATA = {
    'account': {
        'total_balance': 125000,
        'invested_amount': 87500,
        'cash_available': 37500,
        'total_pnl_amount': 12500,
        'total_pnl_percentage': 16.67
    },
    'holdings': [
        {
            'symbol': 'AAPL',
            'company': 'Apple Inc.',
            'shares': 150,
            'buy_price': 175.50,
            'current_price': 192.30,
            'pnl_amount': 2520,
            'pnl_percentage': 9.57
        },
        {
            'symbol': 'TSLA',
            'company': 'Tesla Inc.',
            'shares': 50,
            'buy_price': 245.80,
            'current_price': 268.90,
            'pnl_amount': 1155,
            'pnl_percentage': 9.40
        },
        {
            'symbol': 'SHOP.TO',
            'company': 'Shopify Inc.',
            'shares': 80,
            'buy_price': 85.20,
            'current_price': 94.50,
            'pnl_amount': 744,
            'pnl_percentage': 10.92
        },
        {
            'symbol': 'NXE.TO',
            'company': 'NexGen Energy Ltd.',
            'shares': 500,
            'buy_price': 4.25,
            'current_price': 5.80,
            'pnl_amount': 775,
            'pnl_percentage': 36.47
        }
    ],
    'recent_trades': [
        {
            'time': '2025-10-11 15:45:30',
            'action': 'BUY',
            'symbol': 'NXE.TO',
            'shares': 200,
            'price': 5.75,
            'total': 1150
        },
        {
            'time': '2025-10-11 14:20:15',
            'action': 'SELL',
            'symbol': 'AAPL',
            'shares': 50,
            'price': 191.80,
            'total': 9590
        },
        {
            'time': '2025-10-11 13:10:45',
            'action': 'BUY',
            'symbol': 'SHOP.TO',
            'shares': 30,
            'price': 93.20,
            'total': 2796
        }
    ],
    'ai_agents': [
        {'name': 'Market Data Collector', 'status': 'Active', 'last_update': '2025-10-11 15:47:23'},
        {'name': 'Technical Analyst', 'status': 'Active', 'last_update': '2025-10-11 15:47:20'},
        {'name': 'Risk Manager', 'status': 'Active', 'last_update': '2025-10-11 15:47:22'},
        {'name': 'Order Executor', 'status': 'Active', 'last_update': '2025-10-11 15:47:25'},
        {'name': 'Sentiment Analyzer', 'status': 'Idle', 'last_update': '2025-10-11 15:30:15'},
        {'name': 'Fundamental Analyst', 'status': 'Idle', 'last_update': '2025-10-11 15:30:15'}
    ]
}

def create_navbar():
    """Create the navigation bar"""
    return html.Nav([
        html.Div([
            html.Span("🤖", style={"fontSize": "1.5rem"}),
            html.Span("AI Trading Bot Dashboard", className="navbar-brand")
        ], className="navbar-brand"),
        html.Div([
            html.Div([
                html.Div(className="status-dot"),
                html.Span("Market Open")
            ], className="status-indicator"),
            html.Div([
                html.Div(className="live-dot"),
                html.Span("LIVE")
            ], className="live-indicator"),
            html.Div(id="current-time", className="status-indicator")
        ], className="navbar-info")
    ], className="navbar")

def create_metric_card(label, value, change=None, positive=None):
    """Create a metric card component"""
    value_class = ""
    if positive is not None:
        value_class = "positive" if positive else "negative"
    
    return html.Div([
        html.Div(label, className="metric-label"),
        html.Div(value, className=f"metric-value {value_class}"),
        html.Div(change or "", className="metric-change")
    ], className="metric-card")

# Removed hardcoded table functions - now using dynamic callbacks

def create_ai_agents_section():
    """Create the AI agents status section"""
    agents = MOCK_DATA['ai_agents']
    
    agent_cards = []
    for agent in agents:
        status_class = "active" if agent['status'] == 'Active' else "idle"
        agent_cards.append(html.Div([
            html.Div([
                html.Div(agent['name'], className="agent-name"),
                html.Div(agent['status'], className=f"agent-status {status_class}")
            ], className="agent-header"),
            html.Div(f"Last: {agent['last_update'].split(' ')[1]}", className="agent-time")
        ], className="agent-card"))
    
    return html.Div(agent_cards, className="ai-agents")

def create_performance_charts():
    """Create performance charts"""
    # Mock performance data
    dates = pd.date_range(start='2025-10-07', end='2025-10-11', freq='D')
    portfolio_values = [112500, 115200, 118900, 122300, 125000]
    daily_pnl = [2500, 2700, 3700, 3400, 2700]
    
    portfolio_chart = go.Figure()
    portfolio_chart.add_trace(go.Scatter(
        x=dates,
        y=portfolio_values,
        mode='lines+markers',
        name='Portfolio Value',
        line=dict(color='#1FB8CD', width=3),
        marker=dict(size=6)
    ))
    portfolio_chart.update_layout(
        title="Portfolio Value Over Time",
        xaxis_title="Date",
        yaxis_title="Value ($)",
        template="plotly_white",
        height=300,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    pnl_chart = go.Figure()
    colors = ['#1FB8CD' if pnl >= 0 else '#B4413C' for pnl in daily_pnl]
    pnl_chart.add_trace(go.Bar(
        x=dates,
        y=daily_pnl,
        name='Daily P&L',
        marker_color=colors
    ))
    pnl_chart.update_layout(
        title="Daily P&L",
        xaxis_title="Date",
        yaxis_title="P&L ($)",
        template="plotly_white",
        height=300,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return portfolio_chart, pnl_chart

def create_main_layout():
    """Create the main dashboard layout"""
    portfolio_chart, pnl_chart = create_performance_charts()
    
    return html.Div([
        create_navbar(),
        
        html.Main([
            # Demo Capital Configuration Section
            html.Section([
                html.Div([
                    html.H2("Demo Capital Configuration", className="section-title"),
                    html.Div([
                        dbc.InputGroup([
                            dbc.InputGroupText("Demo Capital ($)"),
                            dbc.Input(
                                id="demo-capital-input",
                                type="number",
                                value=125000,  # Will be updated by callback
                                min=1000,
                                max=1000000,
                                step=1000,
                                style={"width": "200px"}
                            ),
                            dbc.Button("Update Capital", id="update-capital-btn", color="primary", size="sm")
                        ], style={"width": "400px"}),
                        html.Div(id="capital-update-status", style={"marginLeft": "16px", "fontSize": "0.875rem"})
                    ], className="flex", style={"alignItems": "center"})
                ], className="section-header"),
            ], style={"marginBottom": "24px"}),
            
            # Account Overview Section
            html.Section([
                html.Div([
                    html.H2("Account Overview", className="section-title"),
                    html.Div([
                        html.Div(className="loading-indicator", style={"width": "12px", "height": "12px", "border": "2px solid #ccc", "borderTop": "2px solid #1FB8CD", "borderRadius": "50%", "animation": "spin 1s linear infinite"}),
                        html.Span("Auto-refreshing...", style={"marginLeft": "8px", "fontSize": "0.875rem"})
                    ], className="live-indicator")
                ], className="section-header"),
                
                html.Div([
                    html.Div([
                        html.Div("Total Account Balance", className="metric-label"),
                        html.Div(id="total-balance", className="metric-value"),
                        html.Div("CAD", className="metric-change")
                    ], className="metric-card"),
                    html.Div([
                        html.Div("Money in Stocks", className="metric-label"),
                        html.Div(id="invested-amount", className="metric-value"),
                        html.Div(id="invested-percentage", className="metric-change")
                    ], className="metric-card"),
                    html.Div([
                        html.Div("Total P&L", className="metric-label"),
                        html.Div(id="total-pnl", className="metric-value"),
                        html.Div("CAD", className="metric-change")
                    ], className="metric-card"),
                    html.Div([
                        html.Div("Total P&L %", className="metric-label"),
                        html.Div(id="total-pnl-percent", className="metric-value"),
                        html.Div("Return", className="metric-change")
                    ], className="metric-card")
                ], className="grid grid-4")
            ]),
            
            # Holdings Section
            html.Section([
                html.Div([
                    html.H2("Current Holdings", className="section-title"),
                    html.Div([
                        dbc.Input(
                            placeholder="Search holdings...",
                            style={"width": "200px"},
                            id="holdings-search"
                        )
                    ])
                ], className="section-header"),
                html.Div(id="holdings-table-container")
            ]),
            
            # Trading Activity & AI Status
            html.Div([
                # Recent Trading Activity
                html.Section([
                    html.Div([
                        html.H2("Recent Trading Activity", className="section-title"),
                        dbc.Select(
                            options=[
                                {"label": "All Trades", "value": "all"},
                                {"label": "Buy Orders", "value": "BUY"},
                                {"label": "Sell Orders", "value": "SELL"}
                            ],
                            value="all",
                            style={"width": "150px"},
                            id="trade-filter"
                        )
                    ], className="section-header"),
                    html.Div(id="trades-table-container")
                ]),
                
                # AI System Status
                html.Section([
                    html.Div([
                        html.H2("AI Agents Status", className="section-title")
                    ], className="section-header"),
                    create_ai_agents_section()
                ])
            ], className="grid grid-3-1"),
            
            # Performance Charts
            html.Div([
                html.Div([
                    html.Div([
                        html.Div("Portfolio Value Over Time", className="chart-title"),
                        dcc.Graph(figure=portfolio_chart, config={'displayModeBar': False})
                    ], className="chart-container")
                ]),
                html.Div([
                    html.Div([
                        html.Div("Daily P&L", className="chart-title"),
                        dcc.Graph(figure=pnl_chart, config={'displayModeBar': False})
                    ], className="chart-container")
                ])
            ], className="grid grid-2")
            
        ], className="main-content"),
        
        # Footer
        html.Footer([
            html.Div([
                html.Span("System Status: "),
                html.Span("All Systems Operational", className="status-badge success")
            ]),
            html.Div([
                "Last Updated: ",
                html.Span(id="last-update")
            ])
        ], className="footer"),
        
        # Floating Logs Button
        html.Button(
            "📊 AI Logs & Analytics",
            className="floating-logs-btn",
            id="logs-btn"
        )
        
    ], className="dashboard-container")

# App layout
app.layout = html.Div([
    dcc.Store(id='trading-data'),
    dcc.Interval(
        id='interval-component',
        interval=5*1000,  # Update every 5 seconds
        n_intervals=0
    ),
    create_main_layout()
])

# Callbacks
@app.callback(
    Output('current-time', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_time(n):
    now = datetime.now()
    return now.strftime('%H:%M:%S EDT')

@app.callback(
    Output('last-update', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_last_update(n):
    now = datetime.now()
    return now.strftime('%H:%M:%S')

@app.callback(
    Output('trading-data', 'data'),
    Input('interval-component', 'n_intervals')
)
def update_trading_data(n):
    """Update trading data from the system"""
    if TRADING_SYSTEM_AVAILABLE:
        try:
            load_trading_state()
            update_holdings_prices()
            save_trading_state()
            return trading_state
        except Exception as e:
            logger.error(f"Error updating trading data: {e}")
            return MOCK_DATA
    return MOCK_DATA

@app.callback(
    [Output('total-balance', 'children'),
     Output('total-balance', 'className'),
     Output('invested-amount', 'children'),
     Output('invested-percentage', 'children'),
     Output('total-pnl', 'children'),
     Output('total-pnl', 'className'),
     Output('total-pnl-percent', 'children'),
     Output('total-pnl-percent', 'className')],
    [Input('interval-component', 'n_intervals'),
     Input('demo-capital-input', 'value')]
)
def update_account_metrics(n_intervals, demo_capital):
    """Update account metrics with real data"""
    if TRADING_SYSTEM_AVAILABLE:
        try:
            # Load current trading state
            load_trading_state()
            
            # Get real data - use demo_capital if provided, otherwise use trading state
            if demo_capital and demo_capital > 0:
                total_balance = float(demo_capital)
            else:
                total_balance = trading_state.get('current_capital', 125000)
            holdings = trading_state.get('holdings', {})
            
            # Calculate invested amount and P&L
            invested_amount = 0
            total_pnl = 0
            
            # Handle both dict and list formats for holdings
            if isinstance(holdings, dict):
                holdings_iter = holdings.items()
            elif isinstance(holdings, list):
                holdings_iter = [(h.get('symbol', f'stock_{i}'), h) for i, h in enumerate(holdings)]
            else:
                holdings_iter = []
            
            for symbol, holding in holdings_iter:
                if isinstance(holding, dict) and 'shares' in holding and 'current_price' in holding:
                    invested_amount += holding['shares'] * holding['current_price']
                    if 'buy_price' in holding:
                        total_pnl += holding['shares'] * (holding['current_price'] - holding['buy_price'])
            
            # Calculate percentages
            invested_percentage = (invested_amount / total_balance * 100) if total_balance > 0 else 0
            pnl_percentage = (total_pnl / (total_balance - total_pnl) * 100) if (total_balance - total_pnl) > 0 else 0
            
            # Format values
            total_balance_str = f"${total_balance:,.0f}"
            invested_amount_str = f"${invested_amount:,.0f}"
            invested_percentage_str = f"{invested_percentage:.1f}% of total"
            total_pnl_str = f"{'+' if total_pnl >= 0 else ''}${total_pnl:,.0f}"
            pnl_percentage_str = f"{'+' if pnl_percentage >= 0 else ''}{pnl_percentage:.2f}%"
            
            # Set classes for positive/negative
            pnl_class = "metric-value positive" if total_pnl >= 0 else "metric-value negative"
            pnl_percent_class = "metric-value positive" if pnl_percentage >= 0 else "metric-value negative"
            
            return (total_balance_str, "metric-value",
                    invested_amount_str, invested_percentage_str,
                    total_pnl_str, pnl_class,
                    pnl_percentage_str, pnl_percent_class)
                    
        except Exception as e:
            logger.error(f"Error updating account metrics: {e}")
            # Fallback to demo data
            return (f"${demo_capital or 125000:,.0f}", "metric-value",
                    "$87,500", "70% of total",
                    "+$12,500", "metric-value positive",
                    "+16.67%", "metric-value positive")
    else:
        # Use mock data with demo capital
        demo_capital = demo_capital or 125000
        return (f"${demo_capital:,.0f}", "metric-value",
                "$87,500", "70% of total",
                "+$12,500", "metric-value positive",
                "+16.67%", "metric-value positive")

@app.callback(
    Output('capital-update-status', 'children'),
    [Input('update-capital-btn', 'n_clicks')],
    [State('demo-capital-input', 'value')]
)
def update_demo_capital(n_clicks, new_capital):
    """Update demo capital in the trading system"""
    if n_clicks and new_capital and new_capital > 0:
        if TRADING_SYSTEM_AVAILABLE:
            try:
                # Update the trading state with new capital
                load_trading_state()
                
                # Update capital values
                old_capital = trading_state.get('current_capital', 125000)
                trading_state['current_capital'] = float(new_capital)
                trading_state['initial_capital'] = float(new_capital)
                
                # Adjust cash available if needed
                if 'cash_available' not in trading_state:
                    trading_state['cash_available'] = float(new_capital)
                else:
                    # Keep the same cash ratio
                    cash_ratio = trading_state['cash_available'] / old_capital if old_capital > 0 else 0.3
                    trading_state['cash_available'] = float(new_capital) * cash_ratio
                
                # Save the updated state
                save_trading_state()
                
                logger.info(f"Capital updated from ${old_capital:,.0f} to ${new_capital:,.0f}")
                return html.Span(f"✅ Capital updated to ${new_capital:,.0f}", style={"color": "var(--color-success)"})
                
            except Exception as e:
                logger.error(f"Error updating capital: {e}")
                return html.Span(f"❌ Error updating capital: {e}", style={"color": "var(--color-error)"})
        else:
            return html.Span(f"✅ Capital set to ${new_capital:,.0f} (demo mode)", style={"color": "var(--color-success)"})
    elif n_clicks and (not new_capital or new_capital <= 0):
        return html.Span("❌ Please enter a valid capital amount", style={"color": "var(--color-error)"})
    return ""

@app.callback(
    Output('demo-capital-input', 'value'),
    Input('interval-component', 'n_intervals')
)
def initialize_demo_capital(n_intervals):
    """Initialize demo capital input with real value from trading system"""
    if TRADING_SYSTEM_AVAILABLE:
        try:
            load_trading_state()
            return trading_state.get('current_capital', 125000)
        except Exception as e:
            logger.error(f"Error loading capital: {e}")
            return 125000
    return 125000

@app.callback(
    Output('holdings-table-container', 'children'),
    [Input('interval-component', 'n_intervals'),
     Input('holdings-search', 'value')]
)
def update_holdings_table(n_intervals, search_term):
    """Update holdings table with real data"""
    if TRADING_SYSTEM_AVAILABLE:
        try:
            # Load current trading state
            load_trading_state()
            holdings = trading_state.get('holdings', {})
            
            # Convert holdings to list format for table
            holdings_list = []
            
            # Handle both dict and list formats for holdings
            if isinstance(holdings, dict):
                holdings_iter = holdings.items()
            elif isinstance(holdings, list):
                holdings_iter = [(h.get('symbol', f'stock_{i}'), h) for i, h in enumerate(holdings)]
            else:
                holdings_iter = []
            
            for symbol, holding in holdings_iter:
                if isinstance(holding, dict) and 'shares' in holding:
                    current_price = holding.get('current_price', 0)
                    buy_price = holding.get('buy_price', current_price)
                    shares = holding.get('shares', 0)
                    
                    pnl_amount = shares * (current_price - buy_price)
                    pnl_percentage = ((current_price - buy_price) / buy_price * 100) if buy_price > 0 else 0
                    
                    holdings_list.append({
                        'symbol': symbol,
                        'company': holding.get('company', symbol),
                        'shares': shares,
                        'buy_price': buy_price,
                        'current_price': current_price,
                        'pnl_amount': pnl_amount,
                        'pnl_percentage': pnl_percentage
                    })
        except Exception as e:
            logger.error(f"Error loading holdings: {e}")
            holdings_list = MOCK_DATA['holdings']
    else:
        holdings_list = MOCK_DATA['holdings']
    
    # Filter by search term
    if search_term:
        search_lower = search_term.lower()
        holdings_list = [h for h in holdings_list if 
                        search_lower in h['symbol'].lower() or 
                        search_lower in h['company'].lower()]
    
    # Create table rows
    table_rows = []
    for holding in holdings_list:
        pnl_class = "positive" if holding['pnl_amount'] >= 0 else "negative"
        table_rows.append(html.Tr([
            html.Td(html.Strong(holding['symbol'])),
            html.Td(holding['company']),
            html.Td(f"{holding['shares']:,}"),
            html.Td(f"${holding['buy_price']:.2f}"),
            html.Td(f"${holding['current_price']:.2f}"),
            html.Td(f"${holding['pnl_amount']:+,.0f}", className=pnl_class),
            html.Td(f"{holding['pnl_percentage']:+.2f}%", className=pnl_class)
        ]))
    
    return html.Div([
        html.Table([
            html.Thead([
                html.Tr([
                    html.Th("Symbol"),
                    html.Th("Company"),
                    html.Th("Shares"),
                    html.Th("Buy Price"),
                    html.Th("Current Price"),
                    html.Th("P&L ($)"),
                    html.Th("P&L (%)")
                ])
            ]),
            html.Tbody(table_rows)
        ], className="table")
    ], className="table-container")

@app.callback(
    Output('trades-table-container', 'children'),
    [Input('interval-component', 'n_intervals'),
     Input('trade-filter', 'value')]
)
def update_trades_table(n_intervals, filter_value):
    """Update trades table with real data"""
    if TRADING_SYSTEM_AVAILABLE:
        try:
            # Load current trading state
            load_trading_state()
            trades = trading_state.get('recent_trades', [])
            
            # Convert trades to list format for table
            trades_list = []
            for trade in trades:
                if isinstance(trade, dict):
                    trades_list.append({
                        'time': trade.get('timestamp', trade.get('time', 'N/A')),
                        'action': trade.get('action', 'N/A'),
                        'symbol': trade.get('symbol', 'N/A'),
                        'shares': trade.get('shares', 0),
                        'price': trade.get('price', 0),
                        'total': trade.get('total', trade.get('shares', 0) * trade.get('price', 0))
                    })
        except Exception as e:
            logger.error(f"Error loading trades: {e}")
            trades_list = MOCK_DATA['recent_trades']
    else:
        trades_list = MOCK_DATA['recent_trades']
    
    # Filter trades
    if filter_value and filter_value != 'all':
        trades_list = [t for t in trades_list if t['action'] == filter_value]
    
    # Create table rows
    table_rows = []
    for trade in trades_list[:20]:  # Show last 20 trades
        action = trade.get('action', 'N/A')
        action_class = "success" if action == 'BUY' else "error"
        time_str = trade.get('time', 'N/A')
        time_display = time_str.split(' ')[1] if ' ' in time_str else time_str
        table_rows.append(html.Tr([
            html.Td(time_display),
            html.Td(html.Span(action, className=f"status-badge {action_class}")),
            html.Td(html.Strong(trade.get('symbol', 'N/A'))),
            html.Td(f"{trade.get('shares', 0):,}"),
            html.Td(f"${trade.get('price', 0):.2f}"),
            html.Td(f"${trade.get('total', 0):,.0f}")
        ]))
    
    return html.Div([
        html.Table([
            html.Thead([
                html.Tr([
                    html.Th("Time"),
                    html.Th("Action"),
                    html.Th("Symbol"),
                    html.Th("Shares"),
                    html.Th("Price"),
                    html.Th("Total")
                ])
            ]),
            html.Tbody(table_rows)
        ], className="table")
    ], className="table-container")

if __name__ == '__main__':
    print("=" * 80)
    print("Modern AI Trading Dashboard Starting...")
    print("=" * 80)
    print()
    print("Features:")
    print("   - Modern design system")
    print("   - Real-time portfolio updates")
    print("   - AI agent monitoring")
    print("   - Interactive charts")
    print("   - Responsive layout")
    print()
    print("Dashboard URL: http://localhost:8052")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 80)
    print()
    
    try:
        app.run(
            debug=True,
            host='127.0.0.1',
            port=8052,
            dev_tools_hot_reload=False
        )
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        import traceback
        traceback.print_exc()
