"""
Trading Bot Dashboard - Groww-inspired UI
Real-time monitoring of trades, portfolio, and AI decisions
"""

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime, timedelta
import pandas as pd
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from risk_management.capital_allocation import CapitalAllocator
from strategies.strategy_manager import StrategyManager
from data_pipeline.collectors.canadian_market_collector import CanadianMarketCollector
from execution.etf_allocator import ETFAllocator
from data_pipeline.questrade_client import get_questrade_client
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Questrade client
try:
    questrade_client = get_questrade_client()
    questrade_authenticated = False
    logger.info(" Questrade client initialized, will authenticate on first data request")
except Exception as e:
    logger.error(f" Failed to initialize Questrade client: {e}")
    questrade_client = None
    questrade_authenticated = False

# Initialize Dash app with Bootstrap theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True
)

# Groww color palette
COLORS = {
    'primary': '#00D09C',      # Groww green
    'secondary': '#44475B',    # Dark blue-grey
    'background': '#FAFAFA',   # Light grey background
    'card': '#FFFFFF',         # White cards
    'text': '#44475B',         # Dark text
    'success': '#00D09C',      # Green
    'danger': '#EB5B3C',       # Red
    'warning': '#FDB022',      # Orange
    'info': '#5367FE',         # Blue
    'muted': '#8B8B8B'         # Grey
}

# Real Questrade data functions
def get_real_portfolio_data():
    """Fetch real portfolio data from Questrade"""
    global questrade_authenticated
    
    if not questrade_client:
        return pd.DataFrame(columns=['symbol', 'name', 'quantity', 'avg_price', 'current_price', 'invested', 'current', 'pnl', 'pnl_pct'])
    
    try:
        # Authenticate if needed
        if not questrade_authenticated:
            logger.info(" Authenticating with Questrade...")
            if questrade_client.authenticate():
                questrade_authenticated = True
                logger.info(" Questrade authentication successful")
            else:
                logger.error(" Questrade authentication failed")
                return pd.DataFrame(columns=['symbol', 'name', 'quantity', 'avg_price', 'current_price', 'invested', 'current', 'pnl', 'pnl_pct'])
        
        # Get positions
        positions = questrade_client.get_positions()
        
        if not positions:
            return pd.DataFrame(columns=['symbol', 'name', 'quantity', 'avg_price', 'current_price', 'invested', 'current', 'pnl', 'pnl_pct'])
        
        # Convert to DataFrame
        portfolio_data = []
        for pos in positions:
            quantity = pos.get('openQuantity', 0)
            avg_price = pos.get('averageEntryPrice', 0.0)
            current_price = pos.get('currentPrice', 0.0)
            invested = quantity * avg_price
            current = pos.get('currentMarketValue', 0.0)
            pnl = pos.get('openPnl', 0.0)
            pnl_pct = (pnl / invested * 100) if invested > 0 else 0.0
            
            portfolio_data.append({
                'symbol': pos.get('symbol', ''),
                'name': pos.get('symbolDescription', pos.get('symbol', '')),
                'quantity': quantity,
                'avg_price': avg_price,
                'current_price': current_price,
                'invested': invested,
                'current': current,
                'pnl': pnl,
                'pnl_pct': pnl_pct
            })
        
        return pd.DataFrame(portfolio_data)
        
    except Exception as e:
        logger.error(f" Error fetching portfolio data: {e}")
        return pd.DataFrame(columns=['symbol', 'name', 'quantity', 'avg_price', 'current_price', 'invested', 'current', 'pnl', 'pnl_pct'])

def get_real_account_summary():
    """Fetch real account summary from Questrade"""
    global questrade_authenticated
    
    if not questrade_client:
        return {'total_value': 0.0, 'cash': 0.0, 'positions_value': 0.0, 'pnl_today': 0.0, 'pnl_total': 0.0}
    
    try:
        # Authenticate if needed
        if not questrade_authenticated:
            if questrade_client.authenticate():
                questrade_authenticated = True
        
        # Get portfolio summary
        summary = questrade_client.get_portfolio_summary()
        return summary
        
    except Exception as e:
        logger.error(f" Error fetching account summary: {e}")
        return {'total_value': 0.0, 'cash': 0.0, 'positions_value': 0.0, 'pnl_today': 0.0, 'pnl_total': 0.0}

# Empty data for trades and signals (will be populated by trading strategies)
def generate_mock_trades():
    """Generate empty trade data - trades will come from strategy execution"""
    return pd.DataFrame(columns=['time', 'symbol', 'strategy', 'side', 'quantity', 'price', 'pnl', 'status'])

def generate_mock_signals():
    """Generate empty signals data - signals will come from AI strategies"""
    return pd.DataFrame(columns=['time', 'symbol', 'strategy', 'signal', 'confidence', 'reason', 'status'])

# Create layout components
def create_header():
    """Create dashboard header"""
    return dbc.Navbar(
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.I(className="fas fa-robot", style={'fontSize': '28px', 'color': COLORS['primary']}),
                        html.Span("Trading Bot Dashboard", className="ms-2", style={
                            'fontSize': '24px',
                            'fontWeight': 'bold',
                            'color': COLORS['text']
                        })
                    ])
                ], width="auto"),
                dbc.Col([
                    html.Div([
                        dbc.Badge("LIVE", color="success", className="me-2"),
                        html.Span(id='live-time', style={'color': COLORS['muted']})
                    ], className="d-flex align-items-center justify-content-end")
                ])
            ], className="w-100 align-items-center")
        ], fluid=True),
        color=COLORS['card'],
        dark=False,
        className="mb-3 shadow-sm",
        style={'borderBottom': f'3px solid {COLORS["primary"]}'}
    )

def create_summary_cards():
    """Create summary statistics cards"""
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-wallet", style={'fontSize': '24px', 'color': COLORS['primary']}),
                    ], className="mb-2"),
                    html.H4("$82,450", className="mb-0", style={'color': COLORS['text'], 'fontWeight': 'bold'}),
                    html.P("Total Value", className="text-muted mb-0", style={'fontSize': '14px'}),
                    html.Small("+$2,450 (3.06%)", style={'color': COLORS['success'], 'fontWeight': '600'})
                ])
            ], className="shadow-sm border-0")
        ], width=12, md=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-chart-line", style={'fontSize': '24px', 'color': COLORS['info']}),
                    ], className="mb-2"),
                    html.H4("$737.50", className="mb-0", style={'color': COLORS['text'], 'fontWeight': 'bold'}),
                    html.P("Today's P&L", className="text-muted mb-0", style={'fontSize': '14px'}),
                    html.Small("+0.92%", style={'color': COLORS['success'], 'fontWeight': '600'})
                ])
            ], className="shadow-sm border-0")
        ], width=12, md=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-exchange-alt", style={'fontSize': '24px', 'color': COLORS['warning']}),
                    ], className="mb-2"),
                    html.H4("18", className="mb-0", style={'color': COLORS['text'], 'fontWeight': 'bold'}),
                    html.P("Trades Today", className="text-muted mb-0", style={'fontSize': '14px'}),
                    html.Small("14 wins, 4 losses", style={'color': COLORS['muted'], 'fontSize': '12px'})
                ])
            ], className="shadow-sm border-0")
        ], width=12, md=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-brain", style={'fontSize': '24px', 'color': COLORS['danger']}),
                    ], className="mb-2"),
                    html.H4("4", className="mb-0", style={'color': COLORS['text'], 'fontWeight': 'bold'}),
                    html.P("Active Signals", className="text-muted mb-0", style={'fontSize': '14px'}),
                    html.Small("5 strategies active", style={'color': COLORS['muted'], 'fontSize': '12px'})
                ])
            ], className="shadow-sm border-0")
        ], width=12, md=3),
    ], className="mb-4")

def create_chart_section():
    """Create portfolio performance chart"""
    # Generate sample data
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    portfolio_value = [80000 + (i * 100) + (50 * (i % 7)) for i in range(30)]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=portfolio_value,
        mode='lines',
        name='Portfolio Value',
        line=dict(color=COLORS['primary'], width=3),
        fill='tozeroy',
        fillcolor=f'rgba(0, 208, 156, 0.1)'
    ))
    
    fig.update_layout(
        title=dict(text='Portfolio Performance (30 Days)', font=dict(size=18, color=COLORS['text'], weight='bold')),
        xaxis=dict(title='Date', showgrid=False),
        yaxis=dict(title='Value (CAD)', showgrid=True, gridcolor='#E8E8E8'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='x unified',
        height=300,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return dbc.Card([
        dbc.CardBody([
            dcc.Graph(figure=fig, config={'displayModeBar': False})
        ])
    ], className="shadow-sm border-0 mb-4")

def create_portfolio_table():
    """Create portfolio holdings table"""
    df = generate_mock_portfolio()
    
    rows = []
    for _, row in df.iterrows():
        pnl_color = COLORS['success'] if row['pnl'] >= 0 else COLORS['danger']
        rows.append(
            html.Tr([
                html.Td([
                    html.Div([
                        html.Strong(row['symbol'], style={'color': COLORS['text']}),
                        html.Br(),
                        html.Small(row['name'], style={'color': COLORS['muted']})
                    ])
                ]),
                html.Td(f"{row['quantity']}", style={'textAlign': 'center'}),
                html.Td(f"${row['avg_price']:.2f}", style={'textAlign': 'right'}),
                html.Td(f"${row['current_price']:.2f}", style={'textAlign': 'right'}),
                html.Td([
                    html.Div([
                        html.Strong(f"${abs(row['pnl']):.2f}", style={'color': pnl_color}),
                        html.Br(),
                        html.Small(f"{row['pnl_pct']:+.1f}%", style={'color': pnl_color})
                    ])
                ], style={'textAlign': 'right'}),
            ])
        )
    
    return dbc.Card([
        dbc.CardHeader([
            html.I(className="fas fa-briefcase me-2", style={'color': COLORS['primary']}),
            html.Strong("Portfolio Holdings", style={'color': COLORS['text']})
        ], style={'backgroundColor': COLORS['background'], 'border': 'none'}),
        dbc.CardBody([
            html.Div([
                html.Table([
                    html.Thead([
                        html.Tr([
                            html.Th("Stock", style={'color': COLORS['muted'], 'fontWeight': '600'}),
                            html.Th("Qty", style={'color': COLORS['muted'], 'fontWeight': '600', 'textAlign': 'center'}),
                            html.Th("Avg Price", style={'color': COLORS['muted'], 'fontWeight': '600', 'textAlign': 'right'}),
                            html.Th("LTP", style={'color': COLORS['muted'], 'fontWeight': '600', 'textAlign': 'right'}),
                            html.Th("P&L", style={'color': COLORS['muted'], 'fontWeight': '600', 'textAlign': 'right'}),
                        ])
                    ]),
                    html.Tbody(rows)
                ], className="table table-hover", style={'marginBottom': '0'})
            ], style={'overflowX': 'auto'})
        ], style={'padding': '0'})
    ], className="shadow-sm border-0 mb-4")

def create_ai_signals_panel():
    """Create AI signals panel"""
    df = generate_mock_signals()
    
    signals = []
    for _, row in df.iterrows():
        signal_color = COLORS['success'] if row['signal'] == 'BUY' else COLORS['danger']
        status_badge_color = 'success' if row['status'] == 'Executed' else 'warning'
        
        signals.append(
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                dbc.Badge(row['signal'], color='success' if row['signal']=='BUY' else 'danger', className="me-2"),
                                html.Strong(row['symbol'], style={'color': COLORS['text']}),
                            ]),
                            html.Small(row['strategy'], style={'color': COLORS['muted']})
                        ], width=8),
                        dbc.Col([
                            html.Div([
                                html.Div([
                                    html.Strong(f"{row['confidence']}%", style={'color': COLORS['primary'], 'fontSize': '18px'}),
                                    html.Br(),
                                    html.Small("Confidence", style={'color': COLORS['muted'], 'fontSize': '11px'})
                                ], style={'textAlign': 'right'})
                            ])
                        ], width=4),
                    ]),
                    html.Hr(style={'margin': '8px 0'}),
                    html.Div([
                        html.I(className="fas fa-info-circle me-2", style={'color': COLORS['info']}),
                        html.Small(row['reason'], style={'color': COLORS['text']})
                    ]),
                    html.Div([
                        dbc.Badge(row['status'], color=status_badge_color, className="mt-2"),
                        html.Small(f" • {row['time']}", style={'color': COLORS['muted'], 'marginLeft': '8px'})
                    ])
                ], style={'padding': '12px'})
            ], className="mb-2 shadow-sm border-0")
        )
    
    return dbc.Card([
        dbc.CardHeader([
            html.I(className="fas fa-brain me-2", style={'color': COLORS['primary']}),
            html.Strong("AI Trading Signals", style={'color': COLORS['text']}),
            dbc.Badge("4 Active", color="success", className="ms-2")
        ], style={'backgroundColor': COLORS['background'], 'border': 'none'}),
        dbc.CardBody(signals, style={'maxHeight': '500px', 'overflowY': 'auto'})
    ], className="shadow-sm border-0 mb-4")

def create_trades_table():
    """Create recent trades table"""
    df = generate_mock_trades()
    
    rows = []
    for _, row in df.iterrows():
        side_color = COLORS['success'] if row['side'] == 'BUY' else COLORS['danger']
        pnl_color = COLORS['success'] if row['pnl'] >= 0 else COLORS['danger']
        
        rows.append(
            html.Tr([
                html.Td(row['time']),
                html.Td([
                    html.Strong(row['symbol'], style={'color': COLORS['text']}),
                    html.Br(),
                    html.Small(row['strategy'], style={'color': COLORS['muted'], 'fontSize': '11px'})
                ]),
                html.Td(
                    dbc.Badge(row['side'], color='success' if row['side']=='BUY' else 'danger'),
                    style={'textAlign': 'center'}
                ),
                html.Td(f"{row['quantity']}", style={'textAlign': 'center'}),
                html.Td(f"${row['price']:.2f}", style={'textAlign': 'right'}),
                html.Td(
                    html.Strong(f"${row['pnl']:+,.0f}", style={'color': pnl_color}),
                    style={'textAlign': 'right'}
                ),
                html.Td(
                    dbc.Badge(row['status'], color='success' if row['status']=='Completed' else 'warning'),
                    style={'textAlign': 'center'}
                ),
            ])
        )
    
    return dbc.Card([
        dbc.CardHeader([
            html.I(className="fas fa-history me-2", style={'color': COLORS['primary']}),
            html.Strong("Recent Trades", style={'color': COLORS['text']})
        ], style={'backgroundColor': COLORS['background'], 'border': 'none'}),
        dbc.CardBody([
            html.Div([
                html.Table([
                    html.Thead([
                        html.Tr([
                            html.Th("Time", style={'color': COLORS['muted'], 'fontWeight': '600'}),
                            html.Th("Stock", style={'color': COLORS['muted'], 'fontWeight': '600'}),
                            html.Th("Side", style={'color': COLORS['muted'], 'fontWeight': '600', 'textAlign': 'center'}),
                            html.Th("Qty", style={'color': COLORS['muted'], 'fontWeight': '600', 'textAlign': 'center'}),
                            html.Th("Price", style={'color': COLORS['muted'], 'fontWeight': '600', 'textAlign': 'right'}),
                            html.Th("P&L", style={'color': COLORS['muted'], 'fontWeight': '600', 'textAlign': 'right'}),
                            html.Th("Status", style={'color': COLORS['muted'], 'fontWeight': '600', 'textAlign': 'center'}),
                        ])
                    ]),
                    html.Tbody(rows)
                ], className="table table-hover", style={'marginBottom': '0'})
            ], style={'overflowX': 'auto', 'maxHeight': '400px', 'overflowY': 'auto'})
        ], style={'padding': '0'})
    ], className="shadow-sm border-0")

def create_strategy_performance():
    """Create strategy performance chart"""
    strategies = ['Momentum\nScalping', 'News\nVolatility', 'Gamma/OI\nSqueeze', 'Arbitrage', 'AI/ML\nPatterns']
    returns = [5.8, 3.2, 2.1, 4.5, 6.2]
    trades = [45, 32, 18, 28, 38]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Returns (%)',
        x=strategies,
        y=returns,
        marker_color=COLORS['primary'],
        text=[f'{r}%' for r in returns],
        textposition='outside'
    ))
    
    fig.update_layout(
        title=dict(text='Strategy Performance', font=dict(size=16, color=COLORS['text'])),
        xaxis=dict(title='', showgrid=False),
        yaxis=dict(title='Returns (%)', showgrid=True, gridcolor='#E8E8E8'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=300,
        margin=dict(l=40, r=40, t=60, b=60),
        showlegend=False
    )
    
    return dbc.Card([
        dbc.CardBody([
            dcc.Graph(figure=fig, config={'displayModeBar': False})
        ])
    ], className="shadow-sm border-0 mb-4")

# Main app layout
app.layout = html.Div([
    dcc.Interval(id='interval-component', interval=5000, n_intervals=0),
    
    create_header(),
    
    dbc.Container([
        create_summary_cards(),
        
        dbc.Row([
            dbc.Col([
                create_chart_section(),
                create_portfolio_table(),
            ], width=12, lg=8),
            
            dbc.Col([
                create_ai_signals_panel(),
            ], width=12, lg=4),
        ]),
        
        dbc.Row([
            dbc.Col([
                create_trades_table(),
            ], width=12, lg=8),
            
            dbc.Col([
                create_strategy_performance(),
            ], width=12, lg=4),
        ]),
        
        # Footer
        html.Div([
            html.Hr(),
            html.P([
                " Trading Bot Dashboard • ",
                html.Span("Powered by AI", style={'color': COLORS['primary']}),
                " • Real-time Updates"
            ], className="text-center text-muted", style={'fontSize': '14px'})
        ], className="mt-4 mb-3")
        
    ], fluid=True, style={'backgroundColor': COLORS['background'], 'minHeight': '100vh', 'padding': '20px'})
], style={'backgroundColor': COLORS['background']})

# Callback for live time update
@app.callback(
    Output('live-time', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_time(n):
    return datetime.now().strftime('%I:%M:%S %p')

if __name__ == '__main__':
    print(" Starting Trading Bot Dashboard...")
    print(" Dashboard URL: http://localhost:8050")
    print(" Access the dashboard in your browser")
    app.run(debug=True, host='0.0.0.0', port=8050)

