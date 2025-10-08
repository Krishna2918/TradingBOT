"""
Complete Trading Dashboard - Canadian Market AI Trading Bot
Full-featured dashboard with real-time data, AI monitoring, and analysis
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
from pathlib import Path

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

# ============================================================================
# DATA GENERATION FUNCTIONS
# ============================================================================

def generate_portfolio_data():
    """Generate mock portfolio data"""
    return {
        'total_value': 100000.00,
        'cash': 25000.00,
        'invested': 75000.00,
        'daily_pnl': 1250.00,
        'daily_pnl_pct': 1.25,
        'total_pnl': 10000.00,
        'total_pnl_pct': 11.11,
        'win_rate': 68.5,
        'total_trades': 207,
        'winning_trades': 142
    }

def generate_holdings():
    """Generate mock holdings"""
    holdings = [
        {'symbol': 'SHOP.TO', 'name': 'Shopify Inc', 'qty': 50, 'avg_price': 80.50, 'current_price': 82.45, 'pnl': 97.50, 'pnl_pct': 2.42},
        {'symbol': 'TD.TO', 'name': 'TD Bank', 'qty': 100, 'avg_price': 77.30, 'current_price': 78.90, 'pnl': 160.00, 'pnl_pct': 2.07},
        {'symbol': 'ENGH.TO', 'name': 'Enbridge Inc', 'qty': 200, 'avg_price': 49.80, 'current_price': 51.20, 'pnl': 280.00, 'pnl_pct': 2.81},
        {'symbol': 'RY.TO', 'name': 'Royal Bank', 'qty': 75, 'avg_price': 138.20, 'current_price': 140.50, 'pnl': 172.50, 'pnl_pct': 1.66},
        {'symbol': 'SU.TO', 'name': 'Suncor Energy', 'qty': 300, 'avg_price': 42.10, 'current_price': 43.85, 'pnl': 525.00, 'pnl_pct': 4.16},
    ]
    return pd.DataFrame(holdings)

def generate_recent_trades():
    """Generate mock recent trades"""
    trades = []
    symbols = ['SHOP.TO', 'TD.TO', 'ENGH.TO', 'RY.TO', 'SU.TO', 'BMO.TO', 'CNQ.TO']
    
    for i in range(10):
        trade = {
            'time': (datetime.now() - timedelta(hours=i, minutes=np.random.randint(0, 60))).strftime('%H:%M:%S'),
            'symbol': np.random.choice(symbols),
            'side': np.random.choice(['BUY', 'SELL']),
            'qty': np.random.randint(10, 100),
            'price': round(np.random.uniform(40, 150), 2),
            'status': np.random.choice(['FILLED', 'FILLED', 'FILLED', 'PENDING']),
            'pnl': round(np.random.uniform(-50, 150), 2) if np.random.random() > 0.3 else None
        }
        trades.append(trade)
    
    return pd.DataFrame(trades)

def generate_performance_chart():
    """Generate portfolio performance chart"""
    dates = pd.date_range(start='2024-01-01', end=datetime.now(), freq='D')
    
    # Generate realistic portfolio growth with some volatility
    returns = np.random.randn(len(dates)) * 0.015 + 0.001
    portfolio_values = 90000 * np.exp(np.cumsum(returns))
    
    fig = go.Figure()
    
    # Portfolio value line
    fig.add_trace(go.Scatter(
        x=dates,
        y=portfolio_values,
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#00d4ff', width=3),
        fill='tozeroy',
        fillcolor='rgba(0, 212, 255, 0.1)'
    ))
    
    # Add benchmark (TSX)
    benchmark_returns = np.random.randn(len(dates)) * 0.01 + 0.0005
    benchmark_values = 90000 * np.exp(np.cumsum(benchmark_returns))
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=benchmark_values,
        mode='lines',
        name='TSX Benchmark',
        line=dict(color='#ffa500', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title='Portfolio Performance vs TSX',
        xaxis_title='Date',
        yaxis_title='Value (CAD)',
        template='plotly_dark',
        height=400,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def generate_sector_allocation():
    """Generate sector allocation pie chart"""
    sectors = {
        'Technology': 25,
        'Financials': 30,
        'Energy': 20,
        'Healthcare': 10,
        'Consumer': 8,
        'Industrials': 7
    }
    
    fig = go.Figure(data=[go.Pie(
        labels=list(sectors.keys()),
        values=list(sectors.values()),
        hole=0.4,
        marker=dict(colors=['#00d4ff', '#00ff88', '#ffa500', '#ff4444', '#aa44ff', '#ffff44'])
    )])
    
    fig.update_layout(
        title='Sector Allocation',
        template='plotly_dark',
        height=350
    )
    
    return fig

def generate_ai_signals():
    """Generate AI trading signals"""
    signals = [
        {'symbol': 'SHOP.TO', 'signal': 'BUY', 'confidence': 0.87, 'price': 82.45, 'target': 88.50, 'reason': 'Technical breakout + Positive sentiment'},
        {'symbol': 'TD.TO', 'signal': 'HOLD', 'confidence': 0.65, 'price': 78.90, 'target': 82.00, 'reason': 'Consolidation phase'},
        {'symbol': 'SU.TO', 'signal': 'BUY', 'confidence': 0.92, 'price': 43.85, 'target': 48.00, 'reason': 'Oil price surge + Strong momentum'},
        {'symbol': 'BMO.TO', 'signal': 'SELL', 'confidence': 0.78, 'price': 125.30, 'target': 120.00, 'reason': 'Overbought conditions'},
    ]
    return pd.DataFrame(signals)

def generate_strategy_performance():
    """Generate strategy performance comparison"""
    strategies = ['Momentum', 'Mean Reversion', 'News Trading', 'Options Flow', 'AI Ensemble']
    returns = [0.15, 0.08, 0.12, 0.10, 0.18]
    win_rates = [65, 58, 70, 62, 72]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Returns (%)',
        x=strategies,
        y=returns,
        marker_color='#00d4ff',
        yaxis='y',
        offsetgroup=1
    ))
    
    fig.add_trace(go.Bar(
        name='Win Rate (%)',
        x=strategies,
        y=win_rates,
        marker_color='#00ff88',
        yaxis='y2',
        offsetgroup=2
    ))
    
    fig.update_layout(
        title='Strategy Performance Comparison',
        xaxis=dict(title='Strategy'),
        yaxis=dict(title='Returns (%)', side='left'),
        yaxis2=dict(title='Win Rate (%)', side='right', overlaying='y'),
        template='plotly_dark',
        height=350,
        barmode='group',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def generate_risk_metrics():
    """Generate risk metrics"""
    return {
        'var_95': -2450.00,
        'var_99': -4200.00,
        'max_drawdown': -3.5,
        'sharpe_ratio': 1.85,
        'sortino_ratio': 2.12,
        'volatility': 12.5,
        'beta': 0.92,
        'alpha': 0.08
    }

# ============================================================================
# LAYOUT COMPONENTS
# ============================================================================

def create_navbar():
    """Create navigation bar"""
    return dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("📊 Overview", href="/", active=True)),
            dbc.NavItem(dbc.NavLink("📈 Analysis", href="/analysis")),
            dbc.NavItem(dbc.NavLink("🤖 AI Signals", href="/signals")),
            dbc.NavItem(dbc.NavLink("⚠️ Risk", href="/risk")),
            dbc.NavItem(dbc.NavLink("⚙️ Settings", href="/settings")),
            dbc.DropdownMenu(
                children=[
                    dbc.DropdownMenuItem("Demo Mode", id="mode-demo"),
                    dbc.DropdownMenuItem("Live Mode", id="mode-live", disabled=True),
                    dbc.DropdownMenuItem(divider=True),
                    dbc.DropdownMenuItem("Export Data", id="export-data"),
                ],
                nav=True,
                in_navbar=True,
                label="More",
            ),
        ],
        brand="🤖 AI Trading Bot - Canadian Market",
        brand_style={"fontSize": "24px", "fontWeight": "bold"},
        color="dark",
        dark=True,
        className="mb-4",
        sticky="top"
    )

def create_summary_cards():
    """Create summary cards for key metrics"""
    portfolio = generate_portfolio_data()
    
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-wallet fa-2x text-info mb-2"),
                        html.H6("Portfolio Value", className="text-muted"),
                        html.H3(f"${portfolio['total_value']:,.2f}", className="mb-1"),
                        html.P([
                            html.Span(f"+${portfolio['total_pnl']:,.2f} ", className="text-success"),
                            html.Small(f"(+{portfolio['total_pnl_pct']:.2f}%)", className="text-success")
                        ], className="mb-0")
                    ])
                ])
            ], className="h-100 shadow-sm")
        ], width=12, md=6, lg=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-chart-line fa-2x text-success mb-2"),
                        html.H6("Today's P&L", className="text-muted"),
                        html.H3(f"+${portfolio['daily_pnl']:,.2f}", className="mb-1 text-success"),
                        html.P([
                            html.Small(f"+{portfolio['daily_pnl_pct']:.2f}%", className="text-success")
                        ], className="mb-0")
                    ])
                ])
            ], className="h-100 shadow-sm")
        ], width=12, md=6, lg=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-trophy fa-2x text-warning mb-2"),
                        html.H6("Win Rate", className="text-muted"),
                        html.H3(f"{portfolio['win_rate']:.1f}%", className="mb-1"),
                        html.P([
                            html.Small(f"{portfolio['winning_trades']}/{portfolio['total_trades']} trades")
                        ], className="mb-0")
                    ])
                ])
            ], className="h-100 shadow-sm")
        ], width=12, md=6, lg=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-robot fa-2x text-primary mb-2"),
                        html.H6("AI Status", className="text-muted"),
                        html.H3("ACTIVE", className="mb-1 text-success"),
                        html.P([
                            html.I(className="fas fa-circle text-success me-2"),
                            html.Small("Demo Mode")
                        ], className="mb-0")
                    ])
                ])
            ], className="h-100 shadow-sm")
        ], width=12, md=6, lg=3),
    ], className="mb-4")

def create_holdings_table():
    """Create holdings table"""
    df = generate_holdings()
    
    rows = []
    for _, row in df.iterrows():
        pnl_color = "text-success" if row['pnl'] > 0 else "text-danger"
        rows.append(
            html.Tr([
                html.Td(html.Strong(row['symbol'])),
                html.Td(row['name']),
                html.Td(f"{row['qty']}", className="text-end"),
                html.Td(f"${row['avg_price']:.2f}", className="text-end"),
                html.Td(f"${row['current_price']:.2f}", className="text-end"),
                html.Td(f"${row['pnl']:.2f}", className=f"text-end {pnl_color}"),
                html.Td(f"{row['pnl_pct']:.2f}%", className=f"text-end {pnl_color}"),
                html.Td(f"${row['current_price'] * row['qty']:,.2f}", className="text-end"),
            ])
        )
    
    return dbc.Card([
        dbc.CardHeader([
            html.I(className="fas fa-briefcase me-2"),
            "Current Holdings"
        ]),
        dbc.CardBody([
            dbc.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Symbol"),
                        html.Th("Name"),
                        html.Th("Qty", className="text-end"),
                        html.Th("Avg Price", className="text-end"),
                        html.Th("Current", className="text-end"),
                        html.Th("P&L", className="text-end"),
                        html.Th("P&L %", className="text-end"),
                        html.Th("Value", className="text-end"),
                    ])
                ]),
                html.Tbody(rows)
            ], striped=True, hover=True, responsive=True, className="mb-0")
        ])
    ], className="shadow-sm mb-4")

def create_recent_trades_table():
    """Create recent trades table"""
    df = generate_recent_trades()
    
    rows = []
    for _, row in df.iterrows():
        side_badge = dbc.Badge("BUY", color="success", className="me-2") if row['side'] == 'BUY' else dbc.Badge("SELL", color="danger", className="me-2")
        status_badge = dbc.Badge(row['status'], color="success" if row['status'] == 'FILLED' else "warning")
        
        pnl_text = f"${row['pnl']:.2f}" if pd.notna(row['pnl']) else "-"
        pnl_color = "text-success" if pd.notna(row['pnl']) and row['pnl'] > 0 else "text-danger" if pd.notna(row['pnl']) else ""
        
        rows.append(
            html.Tr([
                html.Td(row['time']),
                html.Td([side_badge, row['symbol']]),
                html.Td(f"{row['qty']}", className="text-end"),
                html.Td(f"${row['price']:.2f}", className="text-end"),
                html.Td(status_badge),
                html.Td(pnl_text, className=f"text-end {pnl_color}"),
            ])
        )
    
    return dbc.Card([
        dbc.CardHeader([
            html.I(className="fas fa-history me-2"),
            "Recent Trades"
        ]),
        dbc.CardBody([
            dbc.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Time"),
                        html.Th("Symbol"),
                        html.Th("Qty", className="text-end"),
                        html.Th("Price", className="text-end"),
                        html.Th("Status"),
                        html.Th("P&L", className="text-end"),
                    ])
                ]),
                html.Tbody(rows)
            ], striped=True, hover=True, responsive=True, className="mb-0")
        ])
    ], className="shadow-sm mb-4")

def create_ai_signals_table():
    """Create AI signals table"""
    df = generate_ai_signals()
    
    rows = []
    for _, row in df.iterrows():
        if row['signal'] == 'BUY':
            signal_badge = dbc.Badge("BUY", color="success")
        elif row['signal'] == 'SELL':
            signal_badge = dbc.Badge("SELL", color="danger")
        else:
            signal_badge = dbc.Badge("HOLD", color="secondary")
        
        confidence_bar = dbc.Progress(
            value=row['confidence'] * 100,
            color="success" if row['confidence'] > 0.8 else "warning",
            className="mb-0"
        )
        
        rows.append(
            html.Tr([
                html.Td(html.Strong(row['symbol'])),
                html.Td(signal_badge),
                html.Td([confidence_bar, html.Small(f"{row['confidence']:.0%}", className="text-muted")]),
                html.Td(f"${row['price']:.2f}", className="text-end"),
                html.Td(f"${row['target']:.2f}", className="text-end"),
                html.Td(row['reason'], style={"fontSize": "0.85rem"}),
            ])
        )
    
    return dbc.Card([
        dbc.CardHeader([
            html.I(className="fas fa-brain me-2"),
            "AI Trading Signals",
            dbc.Badge("LIVE", color="success", className="ms-2")
        ]),
        dbc.CardBody([
            dbc.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Symbol"),
                        html.Th("Signal"),
                        html.Th("Confidence"),
                        html.Th("Price", className="text-end"),
                        html.Th("Target", className="text-end"),
                        html.Th("Reason"),
                    ])
                ]),
                html.Tbody(rows)
            ], hover=True, responsive=True, className="mb-0")
        ])
    ], className="shadow-sm mb-4")

def create_risk_dashboard():
    """Create risk metrics dashboard"""
    metrics = generate_risk_metrics()
    
    return dbc.Card([
        dbc.CardHeader([
            html.I(className="fas fa-shield-alt me-2"),
            "Risk Metrics"
        ]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H6("Value at Risk (95%)", className="text-muted mb-2"),
                        html.H4(f"${metrics['var_95']:,.2f}", className="text-warning mb-0")
                    ])
                ], width=6, md=3),
                dbc.Col([
                    html.Div([
                        html.H6("Max Drawdown", className="text-muted mb-2"),
                        html.H4(f"{metrics['max_drawdown']:.2f}%", className="text-danger mb-0")
                    ])
                ], width=6, md=3),
                dbc.Col([
                    html.Div([
                        html.H6("Sharpe Ratio", className="text-muted mb-2"),
                        html.H4(f"{metrics['sharpe_ratio']:.2f}", className="text-success mb-0")
                    ])
                ], width=6, md=3),
                dbc.Col([
                    html.Div([
                        html.H6("Volatility", className="text-muted mb-2"),
                        html.H4(f"{metrics['volatility']:.1f}%", className="text-info mb-0")
                    ])
                ], width=6, md=3),
            ])
        ])
    ], className="shadow-sm mb-4")

# ============================================================================
# MAIN LAYOUT
# ============================================================================

app.layout = dbc.Container([
    dcc.Location(id='url', refresh=False),
    
    create_navbar(),
    
    # Auto-refresh interval
    dcc.Interval(
        id='interval-component',
        interval=5*1000,  # 5 seconds
        n_intervals=0
    ),
    
    # Main content
    html.Div(id='page-content'),
    
], fluid=True, style={"backgroundColor": "#1a1a1a", "minHeight": "100vh"})

# ============================================================================
# OVERVIEW PAGE
# ============================================================================

def create_overview_page():
    return html.Div([
        create_summary_cards(),
        
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
        
        create_holdings_table(),
        create_recent_trades_table(),
        create_ai_signals_table(),
    ])

# ============================================================================
# ANALYSIS PAGE
# ============================================================================

def create_analysis_page():
    return html.Div([
        html.H2("📈 Strategy Analysis", className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(
                            figure=generate_strategy_performance(),
                            config={'displayModeBar': False}
                        )
                    ])
                ], className="shadow-sm")
            ], width=12, lg=8),
            
            dbc.Col([
                create_risk_dashboard()
            ], width=12, lg=4),
        ]),
    ])

# ============================================================================
# CALLBACKS
# ============================================================================

@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    if pathname == '/analysis':
        return create_analysis_page()
    elif pathname == '/signals':
        return html.Div([
            html.H2("🤖 AI Signals", className="mb-4"),
            create_ai_signals_table()
        ])
    elif pathname == '/risk':
        return html.Div([
            html.H2("⚠️ Risk Management", className="mb-4"),
            create_risk_dashboard()
        ])
    else:
        return create_overview_page()

@app.callback(
    [Output('performance-chart', 'figure'),
     Output('sector-chart', 'figure')],
    Input('interval-component', 'n_intervals'),
    prevent_initial_call=True
)
def update_charts(n):
    return generate_performance_chart(), generate_sector_allocation()

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("🚀 AI Trading Dashboard Starting...")
    print("=" * 80)
    print()
    print("📊 Features:")
    print("   ✅ Real-time portfolio tracking")
    print("   ✅ AI trading signals")
    print("   ✅ Strategy performance analysis")
    print("   ✅ Risk management dashboard")
    print("   ✅ Holdings & trades monitoring")
    print()
    print("🌐 Dashboard URL: http://localhost:8051")
    print()
    print("🔄 Auto-refresh: Every 5 seconds")
    print("⚙️  Mode: DEMO (Paper Trading)")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 80)
    print()
    
    try:
        app.run(debug=True, host='127.0.0.1', port=8051)
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        import traceback
        traceback.print_exc()

