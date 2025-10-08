"""
Demo Trading Dashboard - AI Trading Monitor
Real-time view of AI-controlled demo trading
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

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.demo.demo_trading_engine import DemoTradingEngine
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize demo engine
try:
    demo_engine = DemoTradingEngine()
    logger.info(" Demo engine initialized for dashboard")
except Exception as e:
    logger.error(f" Failed to initialize demo engine: {e}")
    demo_engine = None

# Initialize Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True
)
app.title = "Demo Trading Dashboard - AI Controlled"

# Colors
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
                html.Span("Demo Trading Dashboard - AI Controlled", style={'fontSize': '24px', 'fontWeight': 'bold', 'color': COLORS['text']})
            ]),
            html.Div([
                dbc.Badge("DEMO", color="warning", className="me-2", style={'fontSize': '14px'}),
                html.Span(id='live-time', children="--:--:--", style={'fontSize': '16px', 'color': COLORS['text'], 'fontWeight': '500'})
            ])
        ], fluid=True, className="d-flex justify-content-between align-items-center")
        , color=COLORS['card'], dark=False, className="shadow-sm mb-4", style={'padding': '16px 0'}
    ),
    
    dbc.Container([
        # Demo Status Banner
        dbc.Alert(
            id='demo-status',
            children="Demo trading active",
            color="success",
            className="mb-4"
        ),
        
        # Metrics Row
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([html.I(className="fas fa-wallet", style={'fontSize': '24px', 'color': COLORS['primary']})], className="mb-2"),
                        html.H4(id='metric-total-value', children="$50,000.00", className="mb-0", style={'color': COLORS['text'], 'fontWeight': 'bold'}),
                        html.P("Total Portfolio Value", className="text-muted mb-0", style={'fontSize': '14px'}),
                        html.Small(id='metric-total-pnl', children="$0.00 (0.00%)", style={'color': COLORS['muted'], 'fontSize': '12px', 'fontWeight': '600'})
                    ])
                ], className="shadow-sm border-0")
            ], width=12, md=3),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([html.I(className="fas fa-money-bill-wave", style={'fontSize': '24px', 'color': COLORS['success']})], className="mb-2"),
                        html.H4(id='metric-cash', children="$50,000.00", className="mb-0", style={'color': COLORS['text'], 'fontWeight': 'bold'}),
                        html.P("Available Cash", className="text-muted mb-0", style={'fontSize': '14px'}),
                        html.Small("CAD", style={'color': COLORS['muted'], 'fontSize': '12px'})
                    ])
                ], className="shadow-sm border-0")
            ], width=12, md=3),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([html.I(className="fas fa-chart-line", style={'fontSize': '24px', 'color': COLORS['info']})], className="mb-2"),
                        html.H4(id='metric-trades-count', children="0", className="mb-0", style={'color': COLORS['text'], 'fontWeight': 'bold'}),
                        html.P("Total Trades", className="text-muted mb-0", style={'fontSize': '14px'}),
                        html.Small(id='metric-positions-count', children="0 open positions", style={'color': COLORS['muted'], 'fontSize': '12px'})
                    ])
                ], className="shadow-sm border-0")
            ], width=12, md=3),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([html.I(className="fas fa-robot", style={'fontSize': '24px', 'color': COLORS['warning']})], className="mb-2"),
                        html.H4("AI MODE", className="mb-0", style={'color': COLORS['warning'], 'fontWeight': 'bold', 'fontSize': '18px'}),
                        html.P("Trading Status", className="text-muted mb-0", style={'fontSize': '14px'}),
                        html.Small("5 strategies active", style={'color': COLORS['muted'], 'fontSize': '12px'})
                    ])
                ], className="shadow-sm border-0")
            ], width=12, md=3),
        ], className="mb-4"),
        
        # Portfolio Performance Chart
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-chart-area me-2", style={'color': COLORS['primary']}),
                        html.Strong("Portfolio Performance", style={'color': COLORS['text']})
                    ], style={'backgroundColor': COLORS['background'], 'border': 'none'}),
                    dbc.CardBody(
                        dcc.Graph(id='performance-chart', config={'displayModeBar': False})
                    )
                ], className="shadow-sm border-0 mb-4")
            ], width=12),
        ]),
        
        # Holdings and Recent Trades
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-briefcase me-2", style={'color': COLORS['primary']}),
                        html.Strong("Current Positions", style={'color': COLORS['text']})
                    ], style={'backgroundColor': COLORS['background'], 'border': 'none'}),
                    dbc.CardBody(id='positions-table')
                ], className="shadow-sm border-0 mb-4")
            ], width=12, lg=6),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-history me-2", style={'color': COLORS['primary']}),
                        html.Strong("Recent Trades", style={'color': COLORS['text']})
                    ], style={'backgroundColor': COLORS['background'], 'border': 'none'}),
                    dbc.CardBody(id='trades-table', style={'maxHeight': '400px', 'overflowY': 'auto'})
                ], className="shadow-sm border-0 mb-4")
            ], width=12, lg=6),
        ]),
        
        # Footer
        html.Div([
            html.Hr(),
            html.P([
                " Demo Trading Dashboard • ",
                html.Span("AI-Controlled", style={'color': COLORS['primary']}),
                " • Canadian Markets • 1 Week Trial"
            ], className="text-center text-muted", style={'fontSize': '14px'})
        ], className="mt-4 mb-3")
        
    ], fluid=True, style={'backgroundColor': COLORS['background'], 'minHeight': '100vh', 'padding': '20px'})
], style={'backgroundColor': COLORS['background']})

# Callbacks
@app.callback(
    [
        Output('demo-status', 'children'),
        Output('demo-status', 'color'),
        Output('metric-total-value', 'children'),
        Output('metric-total-pnl', 'children'),
        Output('metric-total-pnl', 'style'),
        Output('metric-cash', 'children'),
        Output('metric-trades-count', 'children'),
        Output('metric-positions-count', 'children'),
        Output('performance-chart', 'figure'),
        Output('positions-table', 'children'),
        Output('trades-table', 'children'),
    ],
    Input('interval-component', 'n_intervals')
)
def update_dashboard(n):
    """Update dashboard with demo trading data"""
    
    if not demo_engine:
        return (
            " Demo engine not available",
            "danger",
            "$0.00", "$0.00 (0.00%)", {'color': COLORS['muted'], 'fontSize': '12px'},
            "$0.00", "0", "0 open positions",
            go.Figure(), html.P("No data"), html.P("No trades")
        )
    
    # Run one demo cycle
    demo_engine.run_demo_cycle()
    
    # Get account summary
    summary = demo_engine.account.get_summary(demo_engine.current_prices)
    
    # Format metrics
    total_value = summary['total_value']
    cash = summary['cash']
    total_pnl = summary['total_pnl']
    total_return = summary['total_return_pct']
    num_trades = summary['num_trades']
    num_positions = summary['num_positions']
    
    # Status
    days_left = (demo_engine.demo_end - datetime.now()).days
    status_text = f" Demo trading active • {days_left} days remaining • Real Canadian market data"
    status_color = "success"
    
    # P&L styling
    pnl_color = COLORS['success'] if total_pnl >= 0 else COLORS['danger']
    pnl_symbol = "+" if total_pnl >= 0 else ""
    pnl_text = f"{pnl_symbol}${total_pnl:,.2f} ({pnl_symbol}{total_return:.2f}%)"
    pnl_style = {'color': pnl_color, 'fontSize': '12px', 'fontWeight': '600'}
    
    # Performance chart
    # (Simplified - in real version, track historical data)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[datetime.now() - timedelta(hours=i) for i in range(24, 0, -1)],
        y=[demo_engine.account.starting_capital + (total_pnl * i/24) for i in range(24)],
        mode='lines',
        name='Portfolio Value',
        line=dict(color=COLORS['primary'], width=3),
        fill='tozeroy',
        fillcolor=f'rgba(0, 208, 156, 0.1)'
    ))
    fig.update_layout(
        xaxis=dict(title='', showgrid=False),
        yaxis=dict(title='Value (CAD)', showgrid=True, gridcolor='#E8E8E8'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=300,
        margin=dict(l=50, r=20, t=20, b=40),
        showlegend=False
    )
    
    # Positions table
    if demo_engine.account.positions:
        positions_rows = []
        for symbol, pos in demo_engine.account.positions.items():
            current_price = demo_engine.current_prices.get(symbol, pos['avg_price'])
            pnl = (current_price - pos['avg_price']) * pos['quantity']
            pnl_pct = ((current_price - pos['avg_price']) / pos['avg_price']) * 100
            pnl_color = COLORS['success'] if pnl >= 0 else COLORS['danger']
            pnl_symbol = "+" if pnl >= 0 else ""
            
            positions_rows.append(
                html.Tr([
                    html.Td(html.Strong(symbol, style={'color': COLORS['text']})),
                    html.Td(f"{pos['quantity']}", style={'textAlign': 'center'}),
                    html.Td(f"${pos['avg_price']:.2f}", style={'textAlign': 'right'}),
                    html.Td(f"${current_price:.2f}", style={'textAlign': 'right'}),
                    html.Td(
                        html.Strong(f"{pnl_symbol}${pnl:.2f}", style={'color': pnl_color}),
                        style={'textAlign': 'right'}
                    ),
                ])
            )
        
        positions_table = html.Table([
            html.Thead([
                html.Tr([
                    html.Th("Symbol", style={'color': COLORS['muted']}),
                    html.Th("Qty", style={'color': COLORS['muted'], 'textAlign': 'center'}),
                    html.Th("Avg Price", style={'color': COLORS['muted'], 'textAlign': 'right'}),
                    html.Th("Current", style={'color': COLORS['muted'], 'textAlign': 'right'}),
                    html.Th("P&L", style={'color': COLORS['muted'], 'textAlign': 'right'}),
                ])
            ]),
            html.Tbody(positions_rows)
        ], className="table table-hover", style={'marginBottom': '0'})
    else:
        positions_table = html.P("No open positions", className="text-center text-muted", style={'padding': '20px'})
    
    # Trades table
    if demo_engine.account.trade_history:
        recent_trades = demo_engine.account.trade_history[-10:]  # Last 10 trades
        trades_rows = []
        
        for trade in reversed(recent_trades):
            side_color = COLORS['success'] if trade['side'] == 'BUY' else COLORS['danger']
            pnl_value = trade.get('pnl', 0)
            pnl_color = COLORS['success'] if pnl_value >= 0 else COLORS['danger']
            
            trades_rows.append(
                html.Tr([
                    html.Td(trade['timestamp'].strftime('%H:%M:%S')),
                    html.Td(html.Strong(trade['symbol'], style={'color': COLORS['text']})),
                    html.Td(
                        dbc.Badge(trade['side'], color='success' if trade['side']=='BUY' else 'danger'),
                        style={'textAlign': 'center'}
                    ),
                    html.Td(f"{trade['quantity']}", style={'textAlign': 'center'}),
                    html.Td(f"${trade['price']:.2f}", style={'textAlign': 'right'}),
                    html.Td(
                        html.Strong(f"${pnl_value:+.2f}" if pnl_value else "-", style={'color': pnl_color if pnl_value else COLORS['muted']}),
                        style={'textAlign': 'right'}
                    ),
                ])
            )
        
        trades_table = html.Table([
            html.Thead([
                html.Tr([
                    html.Th("Time", style={'color': COLORS['muted']}),
                    html.Th("Symbol", style={'color': COLORS['muted']}),
                    html.Th("Side", style={'color': COLORS['muted'], 'textAlign': 'center'}),
                    html.Th("Qty", style={'color': COLORS['muted'], 'textAlign': 'center'}),
                    html.Th("Price", style={'color': COLORS['muted'], 'textAlign': 'right'}),
                    html.Th("P&L", style={'color': COLORS['muted'], 'textAlign': 'right'}),
                ])
            ]),
            html.Tbody(trades_rows)
        ], className="table table-hover", style={'marginBottom': '0', 'fontSize': '14px'})
    else:
        trades_table = html.P("No trades yet", className="text-center text-muted", style={'padding': '20px'})
    
    return (
        status_text, status_color,
        f"${total_value:,.2f}", pnl_text, pnl_style,
        f"${cash:,.2f}", str(num_trades), f"{num_positions} open positions",
        fig, positions_table, trades_table
    )

@app.callback(
    Output('live-time', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_time(n):
    return datetime.now().strftime('%I:%M:%S %p')

if __name__ == '__main__':
    print("=" * 70)
    print(" Demo Trading Dashboard Starting...")
    print("=" * 70)
    print(" Dashboard URL: http://localhost:8051")
    print(" AI trading with real Canadian market data")
    print("=" * 70)
    print(" Dashboard is now running...")
    print("   Press Ctrl+C to stop")
    print("=" * 70)
    
    app.run(debug=True, host='0.0.0.0', port=8051)

