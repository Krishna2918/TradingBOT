"""
Trading Bot Dashboard - Groww-inspired UI with Real Questrade Data
Real-time monitoring of portfolio, trades, and AI decisions
"""

import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from datetime import datetime, timedelta
import pandas as pd
import sys
import os
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_pipeline.questrade_client import get_questrade_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Questrade client
try:
    questrade_client = get_questrade_client()
    questrade_authenticated = False
    logger.info(" Questrade client initialized")
except Exception as e:
    logger.error(f" Failed to initialize Questrade client: {e}")
    questrade_client = None
    questrade_authenticated = False

# Initialize Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True
)
app.title = "Trading Bot Dashboard - Questrade"

# Groww colors
COLORS = {
    'primary': '#00D09C',
    'secondary': '#44475B',
    'background': '#FAFAFA',
    'card': '#FFFFFF',
    'text': '#44475B',
    'success': '#00D09C',
    'danger': '#EB5B3C',
    'warning': '#FDB022',
    'info': '#5367FE',
    'muted': '#8B8B8B'
}

# Layout
app.layout = html.Div([
    # Interval for updates
    dcc.Interval(
        id='interval-component',
        interval=5*1000,  # Update every 5 seconds
        n_intervals=0
    ),
    
    # Header
    dbc.Navbar(
        dbc.Container([
            html.Div([
                html.I(className="fas fa-robot", style={'fontSize': '28px', 'color': COLORS['primary'], 'marginRight': '12px'}),
                html.Span("Trading Bot Dashboard", style={'fontSize': '24px', 'fontWeight': 'bold', 'color': COLORS['text']})
            ]),
            html.Div([
                dbc.Badge("LIVE", color="success", className="me-2", style={'fontSize': '14px'}),
                html.Span(id='live-time', children="--:--:--", style={'fontSize': '16px', 'color': COLORS['text'], 'fontWeight': '500'})
            ])
        ], fluid=True, className="d-flex justify-content-between align-items-center")
        , color=COLORS['card'], dark=False, className="shadow-sm mb-4", style={'padding': '16px 0'}
    ),
    
    dbc.Container([
        # Connection Status
        dbc.Alert(
            id='connection-status',
            children="Connecting to Questrade...",
            color="info",
            className="mb-4"
        ),
        
        # Metrics Row
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([html.I(className="fas fa-wallet", style={'fontSize': '24px', 'color': COLORS['primary']})], className="mb-2"),
                        html.H4(id='metric-total-value', children="$0.00", className="mb-0", style={'color': COLORS['text'], 'fontWeight': 'bold'}),
                        html.P("Total Portfolio Value", className="text-muted mb-0", style={'fontSize': '14px'}),
                        html.Small(id='metric-total-pnl', children="$0.00 (0.00%)", style={'color': COLORS['muted'], 'fontSize': '12px'})
                    ])
                ], className="shadow-sm border-0")
            ], width=12, md=3),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([html.I(className="fas fa-money-bill-wave", style={'fontSize': '24px', 'color': COLORS['success']})], className="mb-2"),
                        html.H4(id='metric-cash', children="$0.00", className="mb-0", style={'color': COLORS['text'], 'fontWeight': 'bold'}),
                        html.P("Available Cash", className="text-muted mb-0", style={'fontSize': '14px'}),
                        html.Small("CAD", style={'color': COLORS['muted'], 'fontSize': '12px'})
                    ])
                ], className="shadow-sm border-0")
            ], width=12, md=3),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([html.I(className="fas fa-chart-line", style={'fontSize': '24px', 'color': COLORS['info']})], className="mb-2"),
                        html.H4(id='metric-positions-count', children="0", className="mb-0", style={'color': COLORS['text'], 'fontWeight': 'bold'}),
                        html.P("Open Positions", className="text-muted mb-0", style={'fontSize': '14px'}),
                        html.Small(id='metric-positions-value', children="$0.00", style={'color': COLORS['muted'], 'fontSize': '12px'})
                    ])
                ], className="shadow-sm border-0")
            ], width=12, md=3),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([html.I(className="fas fa-robot", style={'fontSize': '24px', 'color': COLORS['warning']})], className="mb-2"),
                        html.H4("READ-ONLY", className="mb-0", style={'color': COLORS['warning'], 'fontWeight': 'bold', 'fontSize': '18px'}),
                        html.P("Questrade Mode", className="text-muted mb-0", style={'fontSize': '14px'}),
                        html.Small("Manual trading only", style={'color': COLORS['muted'], 'fontSize': '12px'})
                    ])
                ], className="shadow-sm border-0")
            ], width=12, md=3),
        ], className="mb-4"),
        
        # Holdings Table
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-briefcase me-2", style={'color': COLORS['primary']}),
                        html.Strong("Portfolio Holdings", style={'color': COLORS['text']})
                    ], style={'backgroundColor': COLORS['background'], 'border': 'none'}),
                    dbc.CardBody(id='holdings-table', children=[
                        html.P("No positions", className="text-center text-muted", style={'padding': '40px'})
                    ])
                ], className="shadow-sm border-0 mb-4")
            ], width=12),
        ]),
        
        # Footer
        html.Div([
            html.Hr(),
            html.P([
                " Trading Bot Dashboard • ",
                html.Span("Connected to Questrade", style={'color': COLORS['primary']}),
                " • Read-Only Mode"
            ], className="text-center text-muted", style={'fontSize': '14px'})
        ], className="mt-4 mb-3")
        
    ], fluid=True, style={'backgroundColor': COLORS['background'], 'minHeight': '100vh', 'padding': '20px'})
], style={'backgroundColor': COLORS['background']})

# Callbacks
@app.callback(
    [
        Output('connection-status', 'children'),
        Output('connection-status', 'color'),
        Output('metric-total-value', 'children'),
        Output('metric-total-pnl', 'children'),
        Output('metric-total-pnl', 'style'),
        Output('metric-cash', 'children'),
        Output('metric-positions-count', 'children'),
        Output('metric-positions-value', 'children'),
        Output('holdings-table', 'children'),
    ],
    Input('interval-component', 'n_intervals')
)
def update_dashboard(n):
    """Update dashboard with real Questrade data"""
    global questrade_authenticated
    
    # Default values
    default_returns = (
        "Connecting to Questrade...",
        "info",
        "$0.00",
        "$0.00 (0.00%)",
        {'color': COLORS['muted'], 'fontSize': '12px'},
        "$0.00",
        "0",
        "$0.00",
        [html.P("No positions", className="text-center text-muted", style={'padding': '40px'})]
    )
    
    if not questrade_client:
        return (
            " Questrade client not available. Check your configuration.",
            "danger",
            *default_returns[2:]
        )
    
    try:
        # Authenticate if needed
        if not questrade_authenticated:
            logger.info(" Authenticating with Questrade...")
            if questrade_client.authenticate():
                questrade_authenticated = True
                logger.info(" Questrade authentication successful")
            else:
                return (
                    " Questrade authentication failed. Check your refresh token.",
                    "danger",
                    *default_returns[2:]
                )
        
        # Get account summary
        summary = questrade_client.get_portfolio_summary()
        
        # Extract metrics
        total_value = summary.get('total_value', 0.0)
        cash = summary.get('cash', 0.0)
        positions_value = summary.get('positions_value', 0.0)
        total_pnl = summary.get('pnl_total', 0.0)
        positions = summary.get('positions', [])
        
        # Calculate P&L percentage
        invested = total_value - total_pnl if total_value > 0 else 1
        pnl_percent = (total_pnl / invested * 100) if invested > 0 else 0.0
        
        # Format values
        total_value_str = f"${total_value:,.2f}"
        cash_str = f"${cash:,.2f}"
        positions_count_str = str(len(positions))
        positions_value_str = f"${positions_value:,.2f}"
        
        # P&L styling
        pnl_color = COLORS['success'] if total_pnl >= 0 else COLORS['danger']
        pnl_symbol = "+" if total_pnl >= 0 else ""
        total_pnl_str = f"{pnl_symbol}${total_pnl:,.2f} ({pnl_symbol}{pnl_percent:.2f}%)"
        pnl_style = {'color': pnl_color, 'fontSize': '12px', 'fontWeight': '600'}
        
        # Create holdings table
        if positions:
            holdings_rows = []
            for pos in positions:
                pnl = pos.get('pnl', 0.0)
                pnl_pct = pos.get('pnl_percent', 0.0)
                pnl_color = COLORS['success'] if pnl >= 0 else COLORS['danger']
                pnl_symbol = "+" if pnl >= 0 else ""
                
                holdings_rows.append(
                    html.Tr([
                        html.Td([
                            html.Strong(pos.get('symbol', ''), style={'color': COLORS['text']}),
                            html.Br(),
                            html.Small(pos.get('symbol', ''), style={'color': COLORS['muted'], 'fontSize': '11px'})
                        ]),
                        html.Td(f"{pos.get('quantity', 0)}", style={'textAlign': 'center'}),
                        html.Td(f"${pos.get('avg_price', 0.0):.2f}", style={'textAlign': 'right'}),
                        html.Td(f"${pos.get('current_price', 0.0):.2f}", style={'textAlign': 'right'}),
                        html.Td(f"${pos.get('market_value', 0.0):,.2f}", style={'textAlign': 'right', 'fontWeight': '600'}),
                        html.Td([
                            html.Strong(f"{pnl_symbol}${pnl:,.2f}", style={'color': pnl_color}),
                            html.Br(),
                            html.Small(f"({pnl_symbol}{pnl_pct:.2f}%)", style={'color': pnl_color, 'fontSize': '11px'})
                        ], style={'textAlign': 'right'}),
                    ])
                )
            
            holdings_table = html.Div([
                html.Table([
                    html.Thead([
                        html.Tr([
                            html.Th("Symbol", style={'color': COLORS['muted'], 'fontWeight': '600'}),
                            html.Th("Quantity", style={'color': COLORS['muted'], 'fontWeight': '600', 'textAlign': 'center'}),
                            html.Th("Avg Price", style={'color': COLORS['muted'], 'fontWeight': '600', 'textAlign': 'right'}),
                            html.Th("Current", style={'color': COLORS['muted'], 'fontWeight': '600', 'textAlign': 'right'}),
                            html.Th("Market Value", style={'color': COLORS['muted'], 'fontWeight': '600', 'textAlign': 'right'}),
                            html.Th("P&L", style={'color': COLORS['muted'], 'fontWeight': '600', 'textAlign': 'right'}),
                        ])
                    ]),
                    html.Tbody(holdings_rows)
                ], className="table table-hover", style={'marginBottom': '0'})
            ], style={'overflowX': 'auto'})
        else:
            holdings_table = html.P("No positions currently held", className="text-center text-muted", style={'padding': '40px'})
        
        return (
            f" Connected to Questrade • Last updated: {datetime.now().strftime('%H:%M:%S')}",
            "success",
            total_value_str,
            total_pnl_str,
            pnl_style,
            cash_str,
            positions_count_str,
            positions_value_str,
            holdings_table
        )
        
    except Exception as e:
        logger.error(f" Error updating dashboard: {e}")
        return (
            f" Error: {str(e)}",
            "danger",
            *default_returns[2:]
        )

@app.callback(
    Output('live-time', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_time(n):
    return datetime.now().strftime('%I:%M:%S %p')

if __name__ == '__main__':
    print("=" * 70)
    print(" Trading Bot Dashboard Starting...")
    print("=" * 70)
    print(" Real Questrade Data Integration")
    print(" Authenticating with Questrade API...")
    print(" Dashboard URL: http://localhost:8050")
    print("=" * 70)
    print(" Dashboard is now running...")
    print("   Press Ctrl+C to stop")
    print("=" * 70)
    
    app.run(debug=True, host='0.0.0.0', port=8050)

