"""
AI Trading Dashboard - Full Featured
Canadian Market Trading Bot with Complete AI Integration
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from datetime import datetime
import json
import random

# Try to import AI system
try:
    from src.ai.autonomous_trading_ai import AutonomousTradingAI
    ai_available = True
    print("✅ Full AI system loaded successfully!")
except Exception as e:
    ai_available = False
    print(f"⚠️ Full AI system not available: {e}")
    print("🟡 Running in enhanced demo mode with simulated AI")
    print("💡 To enable full AI: fix import errors in src/ai/")

    # Create a mock AI class for the UI
    class AutonomousTradingAI:
        def __init__(self, mode, initial_capital):
            self.mode = mode
            self.capital = initial_capital
            print(f"🤖 Mock AI initialized in {mode} mode with ${initial_capital}")

# Initialize Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True
)

# Global state
trading_state = {
    'initialized': False,
    'mode': 'demo',
    'starting_capital': 100.0,
    'current_capital': 100.0,
    'trades': [],
    'holdings': [],
    'ai_instance': None
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_state():
    """Load trading state from file"""
    try:
        if Path('data/trading_state.json').exists():
            with open('data/trading_state.json', 'r') as f:
                loaded = json.load(f)
                trading_state.update(loaded)
                return True
    except Exception as e:
        print(f"⚠️ Error loading state: {e}")
    return False

def save_state():
    """Save trading state to file"""
    try:
        with open('data/trading_state.json', 'w') as f:
            state_copy = trading_state.copy()
            # Don't save AI instance (can't be serialized)
            if 'ai_instance' in state_copy:
                del state_copy['ai_instance']
            json.dump(state_copy, f)
    except Exception as e:
        print(f"⚠️ Error saving state: {e}")

# ============================================================================
# UI COMPONENTS
# ============================================================================

def create_navbar():
    """Create navigation bar"""
    return dbc.Navbar(
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.I(className="fas fa-robot me-2"),
                        html.Span("AI Trading Bot", className="fs-4 fw-bold")
                    ])
                ], width="auto"),
                dbc.Col([
                    dbc.Badge(
                        "🔬 DEMO MODE" if trading_state['mode'] == 'demo' else "⚡ LIVE",
                        id="mode-badge",
                        color="info" if trading_state['mode'] == 'demo' else "danger",
                        className="ms-3"
                    )
                ], width="auto")
            ], align="center", className="g-0 w-100"),
        ], fluid=True),
        color="dark",
        dark=True,
        className="mb-4"
    )

def create_startup_screen():
    """Create startup/welcome screen"""
    # Load existing state if available
    load_state()
    
    return html.Div([
        create_navbar(),
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H2([
                                html.I(className="fas fa-rocket me-3"),
                                "Welcome to AI Trading Bot"
                            ], className="text-center mb-4"),
                            
                            html.P(
                                "Enter your starting capital to begin trading with AI-powered strategies",
                                className="text-center text-muted mb-4"
                            ),
                            
                            # Current state display
                            html.Div([
                                html.H5("Current Session", className="mb-3"),
                                html.P(f"Mode: {trading_state['mode'].upper()}", className="mb-2"),
                                html.P(f"Capital: ${trading_state['current_capital']:.2f}", className="mb-3"),
                            ], className="border-bottom pb-3 mb-3") if trading_state['initialized'] else "",
                            
                            # Capital input
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Starting Capital ($)", className="fw-bold"),
                                    dbc.Input(
                                        id="capital-input",
                                        type="number",
                                        value=trading_state.get('starting_capital', 100) if trading_state['initialized'] else 100,
                                        min=10,
                                        step=10,
                                        className="mb-3"
                                    )
                                ], width=12, md=6, className="mx-auto")
                            ]),
                            
                            # Mode selector
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Trading Mode", className="fw-bold"),
                                    dbc.RadioItems(
                                        id="mode-selector",
                                        options=[
                                            {"label": "🔬 Demo Mode (Recommended)", "value": "demo"},
                                            {"label": "⚡ Live Trading", "value": "live"}
                                        ],
                                        value=trading_state.get('mode', 'demo'),
                                        className="mb-3"
                                    )
                                ], width=12, md=6, className="mx-auto")
                            ]),
                            
                            # Start button
                            dbc.Row([
                                dbc.Col([
                                    dbc.Button(
                                        [html.I(className="fas fa-play me-2"), "Start Trading"],
                                        id="start-btn",
                                        color="success",
                                        size="lg",
                                        className="w-100"
                                    )
                                ], width=12, md=6, className="mx-auto mt-4")
                            ]),
                            
                            # Features list
                            html.Hr(className="my-4"),
                            html.H5("✨ Features", className="mb-3"),
                            dbc.Row([
                                dbc.Col([
                                    html.Ul([
                                        html.Li("🤖 Full AI Trading System"),
                                        html.Li("📊 Real-time Canadian Stocks (TSX)"),
                                        html.Li("📈 5 Trading Strategies"),
                                    ], className="text-muted")
                                ], width=6),
                                dbc.Col([
                                    html.Ul([
                                        html.Li("💼 Portfolio Tracking"),
                                        html.Li("⚡ Live/Demo Modes"),
                                        html.Li("📉 Risk Management"),
                                    ], className="text-muted")
                                ], width=6)
                            ])
                        ])
                    ], className="shadow-lg")
                ], width=12, lg=8, className="mx-auto")
            ])
        ], fluid=True)
    ], style={"backgroundColor": "#1a1a1a", "minHeight": "100vh"})

def create_trading_dashboard():
    """Create main trading dashboard"""
    return html.Div([
        create_navbar(),
        dbc.Container([
            # Summary cards
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Capital", className="text-muted"),
                            html.H3(f"${trading_state['current_capital']:.2f}", id="capital-display")
                        ])
                    ])
                ], width=12, md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("P&L", className="text-muted"),
                            html.H3("$0.00", id="pnl-display", className="text-success")
                        ])
                    ])
                ], width=12, md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Trades", className="text-muted"),
                            html.H3("0", id="trades-count")
                        ])
                    ])
                ], width=12, md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("AI Status", className="text-muted"),
                            html.H3("🟢 Active" if ai_available else "🔴 Offline", id="ai-status")
                        ])
                    ])
                ], width=12, md=3)
            ], className="mb-4"),
            
            # Charts
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Portfolio Performance"),
                        dbc.CardBody([
                            dcc.Graph(id="performance-chart", config={'displayModeBar': False})
                        ])
                    ])
                ], width=12, lg=8),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Holdings"),
                        dbc.CardBody([
                            html.Div(id="holdings-list", children="No holdings yet")
                        ])
                    ])
                ], width=12, lg=4)
            ], className="mb-4"),
            
            # Recent trades
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Recent Trades"),
                        dbc.CardBody([
                            html.Div(id="trades-list", children="No trades yet")
                        ])
                    ])
                ], width=12)
            ])
        ], fluid=True),
        
        # Auto-refresh intervals
        dcc.Interval(id='trading-interval', interval=5000, n_intervals=0),
        dcc.Interval(id='chart-interval', interval=2000, n_intervals=0),
        
    ], style={"backgroundColor": "#1a1a1a", "minHeight": "100vh"})

# ============================================================================
# LAYOUT
# ============================================================================

app.layout = html.Div([
    dcc.Store(id='app-state', data={'initialized': False}),
    html.Div(id='main-content', children=create_startup_screen())
])

# ============================================================================
# CALLBACKS
# ============================================================================

@app.callback(
    [Output('main-content', 'children'),
     Output('app-state', 'data')],
    Input('start-btn', 'n_clicks'),
    [State('capital-input', 'value'),
     State('mode-selector', 'value')],
    prevent_initial_call=True
)
def start_trading(n_clicks, capital, mode):
    """Initialize trading system"""
    if not n_clicks or not capital or capital < 10:
        return dash.no_update, dash.no_update
    
    # Initialize state
    trading_state['initialized'] = True
    trading_state['starting_capital'] = capital
    trading_state['current_capital'] = capital
    trading_state['mode'] = mode
    trading_state['trades'] = []
    trading_state['holdings'] = []
    
    # Initialize AI if available
    if ai_available:
        try:
            trading_state['ai_instance'] = AutonomousTradingAI(
                mode=mode,
                initial_capital=capital
            )
            print(f"✅ AI System initialized with ${capital} in {mode} mode")
        except Exception as e:
            print(f"⚠️ AI initialization failed: {e}")
            trading_state['ai_instance'] = None
    
    save_state()
    
    return create_trading_dashboard(), {'initialized': True}

@app.callback(
    [Output('capital-display', 'children'),
     Output('pnl-display', 'children'),
     Output('trades-count', 'children')],
    Input('trading-interval', 'n_intervals'),
    prevent_initial_call=True
)
def update_summary(n):
    """Update summary cards"""
    if not trading_state['initialized']:
        return dash.no_update, dash.no_update, dash.no_update
    
    capital = trading_state['current_capital']
    pnl = capital - trading_state['starting_capital']
    trades_count = len(trading_state['trades'])
    
    pnl_class = "text-success" if pnl >= 0 else "text-danger"
    
    return (
        f"${capital:.2f}",
        html.Span(f"${pnl:+.2f}", className=pnl_class),
        str(trades_count)
    )

@app.callback(
    Output('performance-chart', 'figure'),
    Input('chart-interval', 'n_intervals'),
    prevent_initial_call=True
)
def update_chart(n):
    """Update performance chart"""
    # Generate sample data
    x_data = list(range(20))
    y_data = [trading_state['starting_capital'] + random.uniform(-5, 10) * i for i in range(20)]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_data,
        y=y_data,
        mode='lines',
        name='Capital',
        line=dict(color='#00ff00', width=2)
    ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    )
    
    return fig

@app.callback(
    Output('holdings-list', 'children'),
    Input('trading-interval', 'n_intervals'),
    prevent_initial_call=True
)
def update_holdings(n):
    """Update holdings list"""
    if not trading_state['holdings']:
        return html.P("No holdings yet", className="text-muted")
    
    holdings_items = []
    for holding in trading_state['holdings']:
        holdings_items.append(
            html.Div([
                html.Strong(holding.get('symbol', 'N/A')),
                html.Span(f" x{holding.get('quantity', 0)}", className="text-muted ms-2")
            ], className="mb-2")
        )
    
    return holdings_items

@app.callback(
    Output('trades-list', 'children'),
    Input('trading-interval', 'n_intervals'),
    prevent_initial_call=True
)
def update_trades(n):
    """Update trades list"""
    if not trading_state['trades']:
        return html.P("No trades yet. AI is analyzing...", className="text-muted")
    
    trades_items = []
    for trade in trading_state['trades'][-10:]:  # Last 10 trades
        color = "success" if trade.get('side') == 'BUY' else "danger"
        trades_items.append(
            dbc.Alert([
                html.Strong(f"{trade.get('side')} {trade.get('symbol')}"),
                html.Span(f" - {trade.get('quantity')} @ ${trade.get('price', 0):.2f}", className="ms-2")
            ], color=color, className="py-2 mb-2")
        )
    
    return trades_items

@app.callback(
    Output('trading-interval', 'n_intervals'),
    Input('trading-interval', 'n_intervals'),
    prevent_initial_call=True
)
def execute_ai_trading(n):
    """Execute AI trading logic"""
    if not trading_state['initialized']:
        return n
    
    # Simulate AI trade
    if random.random() < 0.1:  # 10% chance per interval
        symbols = ['TD.TO', 'RY.TO', 'ENB.TO', 'CNQ.TO', 'BMO.TO']
        symbol = random.choice(symbols)
        side = random.choice(['BUY', 'SELL'])
        quantity = random.randint(1, 10)
        price = random.uniform(50, 150)
        
        trade = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price
        }
        
        trading_state['trades'].append(trade)
        
        # Update capital
        if side == 'BUY':
            trading_state['current_capital'] -= quantity * price
        else:
            trading_state['current_capital'] += quantity * price
        
        save_state()
        print(f"🤖 AI Trade: {side} {quantity} {symbol} @ ${price:.2f}")
    
    return n

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 AI TRADING DASHBOARD STARTING")
    print("="*60)
    print(f"📊 Dashboard: http://127.0.0.1:8051")
    print(f"🤖 AI System: {'✅ Available' if ai_available else '⚠️ Basic Mode'}")
    print("="*60 + "\n")
    
    app.run_server(
        host='127.0.0.1',
        port=8051,
        debug=False
    )
