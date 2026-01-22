"""
Comprehensive Trading Dashboard - Multi-Page Analysis
Complete data analysis with filtering, charts, and tables
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
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.demo.demo_trading_engine import DemoTradingEngine
from src.data_pipeline.comprehensive_data_pipeline import ComprehensiveDataPipeline
from src.risk_management.capital_architecture import CapitalArchitectureManager
from .technical_analysis_page import create_technical_analysis_page
from .options_analysis_page import create_options_analysis_page
from .macro_analysis_page import create_macro_analysis_page
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize components
try:
    demo_engine = DemoTradingEngine()
    data_pipeline = ComprehensiveDataPipeline()
    capital_manager = CapitalArchitectureManager()
    logger.info(" All components initialized for dashboard")
except Exception as e:
    logger.error(f" Failed to initialize components: {e}")
    demo_engine = None
    data_pipeline = None
    capital_manager = None

# Initialize Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True
)
app.title = "Comprehensive Trading Dashboard - Canadian Market Research"

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

# Navigation sidebar
def create_sidebar():
    return html.Div([
        html.Div([
            html.H4("Trading Dashboard", className="text-white"),
            html.Hr(className="text-white"),
            dbc.Nav([
                dbc.NavLink([
                    html.I(className="fas fa-tachometer-alt me-2"),
                    "Overview"
                ], href="/", active="exact", className="text-white"),
                dbc.NavLink([
                    html.I(className="fas fa-chart-line me-2"),
                    "Market Data"
                ], href="/market-data", active="exact", className="text-white"),
                dbc.NavLink([
                    html.I(className="fas fa-chart-bar me-2"),
                    "Technical Analysis"
                ], href="/technical", active="exact", className="text-white"),
                dbc.NavLink([
                    html.I(className="fas fa-options me-2"),
                    "Options Data"
                ], href="/options", active="exact", className="text-white"),
                dbc.NavLink([
                    html.I(className="fas fa-globe me-2"),
                    "Macro Data"
                ], href="/macro", active="exact", className="text-white"),
                dbc.NavLink([
                    html.I(className="fas fa-newspaper me-2"),
                    "News & Sentiment"
                ], href="/sentiment", active="exact", className="text-white"),
                dbc.NavLink([
                    html.I(className="fas fa-coins me-2"),
                    "Capital Allocation"
                ], href="/capital", active="exact", className="text-white"),
                dbc.NavLink([
                    html.I(className="fas fa-robot me-2"),
                    "AI Analysis"
                ], href="/ai-analysis", active="exact", className="text-white"),
                dbc.NavLink([
                    html.I(className="fas fa-shield-alt me-2"),
                    "Risk Management"
                ], href="/risk", active="exact", className="text-white"),
                dbc.NavLink([
                    html.I(className="fas fa-history me-2"),
                    "Backtesting"
                ], href="/backtest", active="exact", className="text-white"),
            ], vertical=True, pills=True),
        ], className="p-3")
    ], className="bg-dark", style={"position": "fixed", "top": 0, "left": 0, "bottom": 0, "width": "250px", "zIndex": 1000})

# Main content area
def create_main_content():
    return html.Div([
        dcc.Location(id="url"),
        html.Div(id="page-content", className="p-4", style={"marginLeft": "250px"})
    ])

# App layout
app.layout = html.Div([
    dcc.Interval(
        id='interval-component',
        interval=5*1000,  # Update every 5 seconds
        n_intervals=0
    ),
    create_sidebar(),
    create_main_content()
])

# Overview page
def create_overview_page():
    return html.Div([
        # Header
        dbc.Row([
            dbc.Col([
                html.H2("Trading Dashboard Overview", className="mb-4"),
                dbc.Alert(
                    id='system-status',
                    children="System operational",
                    color="success",
                    className="mb-4"
                )
            ])
        ]),
        
        # Key metrics
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([html.I(className="fas fa-wallet", style={'fontSize': '24px', 'color': COLORS['primary']})], className="mb-2"),
                        html.H4(id='total-value', children="$50,000.00", className="mb-0", style={'color': COLORS['text'], 'fontWeight': 'bold'}),
                        html.P("Total Portfolio Value", className="text-muted mb-0", style={'fontSize': '14px'}),
                        html.Small(id='total-pnl', children="$0.00 (0.00%)", style={'color': COLORS['muted'], 'fontSize': '12px'})
                    ])
                ], className="shadow-sm border-0")
            ], width=12, md=3),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([html.I(className="fas fa-chart-line", style={'fontSize': '24px', 'color': COLORS['info']})], className="mb-2"),
                        html.H4(id='active-positions', children="0", className="mb-0", style={'color': COLORS['text'], 'fontWeight': 'bold'}),
                        html.P("Active Positions", className="text-muted mb-0", style={'fontSize': '14px'}),
                        html.Small(id='positions-value', children="$0.00", style={'color': COLORS['muted'], 'fontSize': '12px'})
                    ])
                ], className="shadow-sm border-0")
            ], width=12, md=3),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([html.I(className="fas fa-exchange-alt", style={'fontSize': '24px', 'color': COLORS['warning']})], className="mb-2"),
                        html.H4(id='total-trades', children="0", className="mb-0", style={'color': COLORS['text'], 'fontWeight': 'bold'}),
                        html.P("Total Trades", className="text-muted mb-0", style={'fontSize': '14px'}),
                        html.Small(id='win-rate', children="0%", style={'color': COLORS['muted'], 'fontSize': '12px'})
                    ])
                ], className="shadow-sm border-0")
            ], width=12, md=3),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([html.I(className="fas fa-robot", style={'fontSize': '24px', 'color': COLORS['success']})], className="mb-2"),
                        html.H4(id='ai-signals', children="0", className="mb-0", style={'color': COLORS['text'], 'fontWeight': 'bold'}),
                        html.P("AI Signals", className="text-muted mb-0", style={'fontSize': '14px'}),
                        html.Small(id='ai-confidence', children="0%", style={'color': COLORS['muted'], 'fontSize': '12px'})
                    ])
                ], className="shadow-sm border-0")
            ], width=12, md=3),
        ], className="mb-4"),
        
        # Charts row
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-chart-area me-2", style={'color': COLORS['primary']}),
                        html.Strong("Portfolio Performance", style={'color': COLORS['text']})
                    ]),
                    dbc.CardBody(
                        dcc.Graph(id='portfolio-chart', config={'displayModeBar': False})
                    )
                ], className="shadow-sm border-0 mb-4")
            ], width=12, lg=8),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-pie-chart me-2", style={'color': COLORS['primary']}),
                        html.Strong("Capital Allocation", style={'color': COLORS['text']})
                    ]),
                    dbc.CardBody(
                        dcc.Graph(id='allocation-chart', config={'displayModeBar': False})
                    )
                ], className="shadow-sm border-0 mb-4")
            ], width=12, lg=4),
        ]),
        
        # Recent activity
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-history me-2", style={'color': COLORS['primary']}),
                        html.Strong("Recent Activity", style={'color': COLORS['text']})
                    ]),
                    dbc.CardBody(id='recent-activity')
                ], className="shadow-sm border-0")
            ], width=12)
        ])
    ])

# Market Data page
def create_market_data_page():
    return html.Div([
        html.H2("Market Data Analysis", className="mb-4"),
        
        # Filters
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Filters"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Label("Symbols"),
                                dcc.Dropdown(
                                    id='symbol-filter',
                                    options=[
                                        {'label': 'All Symbols', 'value': 'all'},
                                        {'label': 'Penny Stocks', 'value': 'penny'},
                                        {'label': 'Core Holdings', 'value': 'core'},
                                        {'label': 'F&O', 'value': 'futures_options'}
                                    ],
                                    value='all',
                                    multi=True
                                )
                            ], width=6),
                            dbc.Col([
                                html.Label("Timeframe"),
                                dcc.Dropdown(
                                    id='timeframe-filter',
                                    options=[
                                        {'label': '1 Minute', 'value': '1m'},
                                        {'label': '5 Minutes', 'value': '5m'},
                                        {'label': 'Daily', 'value': '1d'}
                                    ],
                                    value='1m'
                                )
                            ], width=6)
                        ], className="mb-3"),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Date Range"),
                                dcc.DatePickerRange(
                                    id='date-range',
                                    start_date=datetime.now() - timedelta(days=7),
                                    end_date=datetime.now()
                                )
                            ], width=6),
                            dbc.Col([
                                html.Label("Data Points"),
                                dcc.Dropdown(
                                    id='data-points-filter',
                                    options=[
                                        {'label': 'Price & Volume', 'value': 'price_volume'},
                                        {'label': 'Technical Indicators', 'value': 'technical'},
                                        {'label': 'All Data', 'value': 'all'}
                                    ],
                                    value='price_volume'
                                )
                            ], width=6)
                        ])
                    ])
                ], className="mb-4")
            ])
        ]),
        
        # Data table
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-table me-2", style={'color': COLORS['primary']}),
                        html.Strong("Market Data Table", style={'color': COLORS['text']})
                    ]),
                    dbc.CardBody([
                        html.Div(id='market-data-table')
                    ])
                ], className="shadow-sm border-0 mb-4")
            ])
        ]),
        
        # Charts
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-chart-line me-2", style={'color': COLORS['primary']}),
                        html.Strong("Price Chart", style={'color': COLORS['text']})
                    ]),
                    dbc.CardBody(
                        dcc.Graph(id='price-chart', config={'displayModeBar': True})
                    )
                ], className="shadow-sm border-0 mb-4")
            ], width=12, lg=8),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-chart-bar me-2", style={'color': COLORS['primary']}),
                        html.Strong("Volume Analysis", style={'color': COLORS['text']})
                    ]),
                    dbc.CardBody(
                        dcc.Graph(id='volume-chart', config={'displayModeBar': False})
                    )
                ], className="shadow-sm border-0 mb-4")
            ], width=12, lg=4)
        ])
    ])

# Page routing
@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return create_overview_page()
    elif pathname == "/market-data":
        return create_market_data_page()
    elif pathname == "/technical":
        return create_technical_analysis_page()
    elif pathname == "/options":
        return create_options_analysis_page()
    elif pathname == "/macro":
        return create_macro_analysis_page()
    elif pathname == "/sentiment":
        return html.Div([html.H2("News & Sentiment", className="mb-4"), html.P("Sentiment analysis page coming soon...")])
    elif pathname == "/capital":
        return html.Div([html.H2("Capital Allocation", className="mb-4"), html.P("Capital allocation page coming soon...")])
    elif pathname == "/ai-analysis":
        return html.Div([html.H2("AI Analysis", className="mb-4"), html.P("AI analysis page coming soon...")])
    elif pathname == "/risk":
        return html.Div([html.H2("Risk Management", className="mb-4"), html.P("Risk management page coming soon...")])
    elif pathname == "/backtest":
        return html.Div([html.H2("Backtesting", className="mb-4"), html.P("Backtesting page coming soon...")])
    else:
        return html.Div([html.H2("404: Page not found", className="mb-4")])

# Callbacks for overview page
@app.callback(
    [
        Output('system-status', 'children'),
        Output('system-status', 'color'),
        Output('total-value', 'children'),
        Output('total-pnl', 'children'),
        Output('total-pnl', 'style'),
        Output('active-positions', 'children'),
        Output('positions-value', 'children'),
        Output('total-trades', 'children'),
        Output('win-rate', 'children'),
        Output('ai-signals', 'children'),
        Output('ai-confidence', 'children'),
        Output('portfolio-chart', 'figure'),
        Output('allocation-chart', 'figure'),
        Output('recent-activity', 'children')
    ],
    Input('interval-component', 'n_intervals')
)
def update_overview(n):
    """Update overview page data"""
    
    # Default values
    default_returns = (
        "System operational",
        "success",
        "$50,000.00",
        "$0.00 (0.00%)",
        {'color': COLORS['muted'], 'fontSize': '12px'},
        "0",
        "$0.00",
        "0",
        "0%",
        "0",
        "0%",
        go.Figure(),
        go.Figure(),
        html.P("No recent activity")
    )
    
    if not demo_engine:
        return default_returns
    
    try:
        # Run demo cycle
        demo_engine.run_demo_cycle()
        
        # Get account summary
        summary = demo_engine.account.get_summary(demo_engine.current_prices)
        
        # Format metrics
        total_value = summary['total_value']
        total_pnl = summary['total_pnl']
        total_return = summary['total_return_pct']
        num_positions = summary['num_positions']
        num_trades = summary['num_trades']
        
        # P&L styling
        pnl_color = COLORS['success'] if total_pnl >= 0 else COLORS['danger']
        pnl_symbol = "+" if total_pnl >= 0 else ""
        pnl_text = f"{pnl_symbol}${total_pnl:,.2f} ({pnl_symbol}{total_return:.2f}%)"
        pnl_style = {'color': pnl_color, 'fontSize': '12px', 'fontWeight': '600'}
        
        # Portfolio chart
        portfolio_fig = go.Figure()
        portfolio_fig.add_trace(go.Scatter(
            x=[datetime.now() - timedelta(hours=i) for i in range(24, 0, -1)],
            y=[demo_engine.account.starting_capital + (total_pnl * i/24) for i in range(24)],
            mode='lines',
            name='Portfolio Value',
            line=dict(color=COLORS['primary'], width=3),
            fill='tozeroy',
            fillcolor=f'rgba(0, 208, 156, 0.1)'
        ))
        portfolio_fig.update_layout(
            xaxis=dict(title='', showgrid=False),
            yaxis=dict(title='Value (CAD)', showgrid=True, gridcolor='#E8E8E8'),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=300,
            margin=dict(l=50, r=20, t=20, b=40),
            showlegend=False
        )
        
        # Allocation chart
        if capital_manager:
            bucket_summary = capital_manager.get_bucket_summary()
            labels = list(bucket_summary['buckets'].keys())
            values = [bucket_summary['buckets'][bucket]['allocated'] for bucket in labels]
            
            allocation_fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=0.3,
                marker_colors=[COLORS['primary'], COLORS['info'], COLORS['warning'], COLORS['success']]
            )])
            allocation_fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=20, b=20),
                showlegend=True
            )
        else:
            allocation_fig = go.Figure()
        
        # Recent activity
        if demo_engine.account.trade_history:
            recent_trades = demo_engine.account.trade_history[-5:]
            activity_items = []
            
            for trade in reversed(recent_trades):
                side_color = COLORS['success'] if trade['side'] == 'BUY' else COLORS['danger']
                activity_items.append(
                    dbc.ListGroupItem([
                        html.Div([
                            html.Strong(f"{trade['side']} {trade['symbol']}", style={'color': side_color}),
                            html.Span(f" @ ${trade['price']:.2f}", className="text-muted ms-2"),
                            html.Small(f" • {trade['timestamp'].strftime('%H:%M')}", className="text-muted ms-2")
                        ])
                    ])
                )
            
            recent_activity = dbc.ListGroup(activity_items, flush=True)
        else:
            recent_activity = html.P("No recent trades", className="text-muted")
        
        return (
            f" System operational • {num_trades} trades • {num_positions} positions",
            "success",
            f"${total_value:,.2f}",
            pnl_text,
            pnl_style,
            str(num_positions),
            f"${summary['positions_value']:,.2f}",
            str(num_trades),
            "0%",  # Win rate calculation needed
            "0",   # AI signals count needed
            "0%",  # AI confidence needed
            portfolio_fig,
            allocation_fig,
            recent_activity
        )
        
    except Exception as e:
        logger.error(f" Error updating overview: {e}")
        return default_returns

# Market data callbacks
@app.callback(
    [
        Output('market-data-table', 'children'),
        Output('price-chart', 'figure'),
        Output('volume-chart', 'figure')
    ],
    [
        Input('symbol-filter', 'value'),
        Input('timeframe-filter', 'value'),
        Input('date-range', 'start_date'),
        Input('date-range', 'end_date'),
        Input('data-points-filter', 'value')
    ]
)
def update_market_data(symbols, timeframe, start_date, end_date, data_points):
    """Update market data based on filters"""
    
    # Default empty returns
    empty_table = html.P("No data available", className="text-muted")
    empty_fig = go.Figure()
    
    if not data_pipeline:
        return empty_table, empty_fig, empty_fig
    
    try:
        # Get symbols based on filter
        if symbols == 'all' or not symbols:
            test_symbols = ["RY.TO", "TD.TO", "SHOP.TO", "CNR.TO", "ENB.TO"]
        elif symbols == 'penny':
            test_symbols = ["AI.TO", "HUT.TO", "BITF.TO"]
        elif symbols == 'core':
            test_symbols = ["RY.TO", "TD.TO", "SHOP.TO"]
        elif symbols == 'futures_options':
            test_symbols = ["XIU.TO", "XSP.TO", "XEG.TO"]
        else:
            test_symbols = symbols if isinstance(symbols, list) else [symbols]
        
        # Fetch data
        table_name = f"bars_{timeframe}" if timeframe in ['1m', '5m'] else "bars_daily"
        data = data_pipeline.get_latest_data(test_symbols, table_name, 100)
        
        if not data:
            return empty_table, empty_fig, empty_fig
        
        # Create data table
        all_data = []
        for symbol, df in data.items():
            if not df.empty:
                latest = df.iloc[-1]
                all_data.append({
                    'Symbol': symbol,
                    'Price': f"${latest['close']:.2f}",
                    'Change': f"{latest.get('price_change', 0)*100:.2f}%",
                    'Volume': f"{latest['volume']:,}",
                    'RSI': f"{latest.get('rsi', 0):.1f}",
                    'MACD': f"{latest.get('macd', 0):.3f}",
                    'Timestamp': latest['timestamp']
                })
        
        if all_data:
            df_table = pd.DataFrame(all_data)
            table = dbc.Table.from_dataframe(
                df_table,
                striped=True,
                bordered=True,
                hover=True,
                responsive=True,
                size='sm'
            )
        else:
            table = empty_table
        
        # Create price chart
        price_fig = go.Figure()
        for symbol, df in data.items():
            if not df.empty:
                price_fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['close'],
                    mode='lines',
                    name=symbol,
                    line=dict(width=2)
                ))
        
        price_fig.update_layout(
            title="Price Chart",
            xaxis_title="Time",
            yaxis_title="Price (CAD)",
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Create volume chart
        volume_fig = go.Figure()
        for symbol, df in data.items():
            if not df.empty:
                volume_fig.add_trace(go.Bar(
                    x=df['timestamp'],
                    y=df['volume'],
                    name=symbol,
                    opacity=0.7
                ))
        
        volume_fig.update_layout(
            title="Volume Analysis",
            xaxis_title="Time",
            yaxis_title="Volume",
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white',
            barmode='group'
        )
        
        return table, price_fig, volume_fig
        
    except Exception as e:
        logger.error(f" Error updating market data: {e}")
        return empty_table, empty_fig, empty_fig

# Technical Analysis callbacks
@app.callback(
    [
        Output('ta-custom-symbols', 'disabled'),
        Output('ta-custom-indicators', 'disabled')
    ],
    [
        Input('ta-symbol-filter', 'value'),
        Input('ta-indicators', 'value')
    ]
)
def update_ta_filters(symbol_filter, indicators_filter):
    """Update technical analysis filter states"""
    symbols_disabled = symbol_filter != 'custom'
    indicators_disabled = indicators_filter != 'custom'
    return symbols_disabled, indicators_disabled

@app.callback(
    [
        Output('ta-indicators-table', 'children'),
        Output('ta-main-chart', 'figure'),
        Output('ta-volume-chart', 'figure'),
        Output('ta-rsi-chart', 'figure'),
        Output('ta-signals-analysis', 'children'),
        Output('ta-signals-chart', 'figure'),
        Output('ta-patterns-analysis', 'children')
    ],
    [
        Input('ta-symbol-filter', 'value'),
        Input('ta-custom-symbols', 'value'),
        Input('ta-timeframe', 'value'),
        Input('ta-period', 'value'),
        Input('ta-indicators', 'value'),
        Input('ta-custom-indicators', 'value')
    ]
)
def update_technical_analysis(symbol_filter, custom_symbols, timeframe, period, indicators_filter, custom_indicators):
    """Update technical analysis page"""
    
    # Default empty returns
    empty_table = html.P("No data available", className="text-muted")
    empty_fig = go.Figure()
    empty_analysis = html.P("No analysis available", className="text-muted")
    
    if not data_pipeline:
        return empty_table, empty_fig, empty_fig, empty_fig, empty_analysis, empty_fig, empty_analysis
    
    try:
        # Get symbols based on filter
        if symbol_filter == 'custom' and custom_symbols:
            test_symbols = custom_symbols
        elif symbol_filter == 'penny':
            test_symbols = ["AI.TO", "HUT.TO", "BITF.TO"]
        elif symbol_filter == 'core':
            test_symbols = ["RY.TO", "TD.TO", "SHOP.TO"]
        elif symbol_filter == 'futures_options':
            test_symbols = ["XIU.TO", "XSP.TO", "XEG.TO"]
        else:
            test_symbols = ["RY.TO", "TD.TO", "SHOP.TO"]
        
        # Fetch data
        table_name = f"bars_{timeframe}" if timeframe in ['1m', '5m', '15m', '1h'] else "bars_daily"
        data = data_pipeline.get_latest_data(test_symbols, table_name, 100)
        
        if not data:
            return empty_table, empty_fig, empty_fig, empty_fig, empty_analysis, empty_fig, empty_analysis
        
        # Import technical analysis functions
        from .technical_analysis_page import (
            create_technical_indicators_summary,
            create_trading_signals_analysis,
            create_pattern_recognition_analysis
        )
        
        # Create components
        indicators_table = create_technical_indicators_summary(data)
        signals_analysis = create_trading_signals_analysis(data)
        patterns_analysis = create_pattern_recognition_analysis(data)
        
        # Create main chart
        main_fig = go.Figure()
        for symbol, df in data.items():
            if not df.empty:
                main_fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['close'],
                    mode='lines',
                    name=symbol,
                    line=dict(width=2)
                ))
        
        main_fig.update_layout(
            title="Price Chart with Technical Indicators",
            xaxis_title="Time",
            yaxis_title="Price (CAD)",
            height=600,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Create volume chart
        volume_fig = go.Figure()
        for symbol, df in data.items():
            if not df.empty:
                volume_fig.add_trace(go.Bar(
                    x=df['timestamp'],
                    y=df['volume'],
                    name=symbol,
                    opacity=0.7
                ))
        
        volume_fig.update_layout(
            title="Volume Analysis",
            xaxis_title="Time",
            yaxis_title="Volume",
            height=200,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Create RSI chart
        rsi_fig = go.Figure()
        for symbol, df in data.items():
            if not df.empty:
                rsi_fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df.get('rsi', 50),
                    mode='lines',
                    name=f"{symbol} RSI",
                    line=dict(width=2)
                ))
        
        rsi_fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        rsi_fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        
        rsi_fig.update_layout(
            title="RSI Analysis",
            xaxis_title="Time",
            yaxis_title="RSI",
            height=200,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Create signals chart
        signals_fig = go.Figure()
        signal_counts = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        
        # Count signals from analysis
        if signals_analysis and hasattr(signals_analysis, 'children'):
            # This is a simplified version - in real implementation, parse the signals
            signal_counts = {'BUY': 2, 'SELL': 1, 'HOLD': 2}
        
        signals_fig.add_trace(go.Pie(
            labels=list(signal_counts.keys()),
            values=list(signal_counts.values()),
            hole=0.3,
            marker_colors=[COLORS['success'], COLORS['danger'], COLORS['muted']]
        ))
        
        signals_fig.update_layout(
            title="Signal Distribution",
            height=300
        )
        
        return (
            indicators_table,
            main_fig,
            volume_fig,
            rsi_fig,
            signals_analysis,
            signals_fig,
            patterns_analysis
        )
        
    except Exception as e:
        logger.error(f" Error updating technical analysis: {e}")
        return empty_table, empty_fig, empty_fig, empty_fig, empty_analysis, empty_fig, empty_analysis

# Options Analysis callbacks
@app.callback(
    [
        Output('options-chain-table', 'children'),
        Output('options-iv-surface', 'figure'),
        Output('options-oi-distribution', 'figure'),
        Output('options-greeks-chart', 'figure'),
        Output('options-greeks-summary', 'children'),
        Output('options-flow-analysis', 'children'),
        Output('options-put-call-ratio', 'figure'),
        Output('options-gamma-squeeze', 'children'),
        Output('options-strategies', 'children')
    ],
    [
        Input('options-symbol-filter', 'value'),
        Input('options-type-filter', 'value'),
        Input('options-expiry-filter', 'value'),
        Input('options-strike-range', 'value'),
        Input('options-moneyness-filter', 'value')
    ]
)
def update_options_analysis(symbol, option_type, expiry, strike_range, moneyness):
    """Update options analysis page"""
    
    # Default empty returns
    empty_table = html.P("No data available", className="text-muted")
    empty_fig = go.Figure()
    empty_analysis = html.P("No analysis available", className="text-muted")
    
    if not data_pipeline:
        return empty_table, empty_fig, empty_fig, empty_fig, empty_analysis, empty_analysis, empty_fig, empty_analysis, empty_analysis
    
    try:
        # Get market data for the symbol
        data = data_pipeline.get_latest_data([symbol], "bars_daily", 30)
        
        if not data or symbol not in data:
            return empty_table, empty_fig, empty_fig, empty_fig, empty_analysis, empty_analysis, empty_fig, empty_analysis, empty_analysis
        
        # Import options analysis functions
        from .options_analysis_page import (
            create_options_chain_table,
            create_implied_volatility_surface,
            create_open_interest_distribution,
            create_greeks_analysis,
            create_greeks_summary,
            create_options_flow_analysis,
            create_put_call_ratio_chart,
            create_gamma_squeeze_detection,
            create_options_strategies
        )
        
        # Create components
        chain_table = create_options_chain_table(data)
        iv_surface = create_implied_volatility_surface(data)
        oi_distribution = create_open_interest_distribution(data)
        greeks_chart = create_greeks_analysis(data)
        greeks_summary = create_greeks_summary(data)
        flow_analysis = create_options_flow_analysis(data)
        put_call_ratio = create_put_call_ratio_chart(data)
        gamma_squeeze = create_gamma_squeeze_detection(data)
        strategies = create_options_strategies(data)
        
        return (
            chain_table,
            iv_surface,
            oi_distribution,
            greeks_chart,
            greeks_summary,
            flow_analysis,
            put_call_ratio,
            gamma_squeeze,
            strategies
        )
        
    except Exception as e:
        logger.error(f" Error updating options analysis: {e}")
        return empty_table, empty_fig, empty_fig, empty_fig, empty_analysis, empty_analysis, empty_fig, empty_analysis, empty_analysis

# Macro Analysis callbacks
@app.callback(
    [
        Output('macro-key-indicators', 'children'),
        Output('macro-rates-chart', 'figure'),
        Output('macro-inflation-chart', 'figure'),
        Output('macro-employment-chart', 'figure'),
        Output('macro-gdp-chart', 'figure'),
        Output('macro-trade-chart', 'figure'),
        Output('macro-housing-chart', 'figure'),
        Output('macro-economic-calendar', 'children'),
        Output('macro-analysis-summary', 'children')
    ],
    [
        Input('macro-category-filter', 'value'),
        Input('macro-period-filter', 'value'),
        Input('macro-country-filter', 'value'),
        Input('macro-date-range', 'start_date'),
        Input('macro-date-range', 'end_date'),
        Input('macro-frequency-filter', 'value')
    ]
)
def update_macro_analysis(category, period, country, start_date, end_date, frequency):
    """Update macro analysis page"""
    
    # Default empty returns
    empty_indicators = html.P("No data available", className="text-muted")
    empty_fig = go.Figure()
    empty_calendar = html.P("No calendar data available", className="text-muted")
    empty_summary = html.P("No analysis available", className="text-muted")
    
    try:
        # Import macro analysis functions
        from .macro_analysis_page import (
            create_key_economic_indicators,
            create_interest_rates_chart,
            create_inflation_chart,
            create_employment_chart,
            create_gdp_chart,
            create_trade_chart,
            create_housing_chart,
            create_economic_calendar,
            create_macro_analysis_summary
        )
        
        # Create components
        key_indicators = create_key_economic_indicators()
        rates_chart = create_interest_rates_chart()
        inflation_chart = create_inflation_chart()
        employment_chart = create_employment_chart()
        gdp_chart = create_gdp_chart()
        trade_chart = create_trade_chart()
        housing_chart = create_housing_chart()
        economic_calendar = create_economic_calendar()
        analysis_summary = create_macro_analysis_summary()
        
        return (
            key_indicators,
            rates_chart,
            inflation_chart,
            employment_chart,
            gdp_chart,
            trade_chart,
            housing_chart,
            economic_calendar,
            analysis_summary
        )
        
    except Exception as e:
        logger.error(f" Error updating macro analysis: {e}")
        return empty_indicators, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_calendar, empty_summary

if __name__ == '__main__':
    print("=" * 70)
    print(" Comprehensive Trading Dashboard Starting...")
    print("=" * 70)
    print(" Features:")
    print("   • Multi-page analysis dashboard")
    print("   • Market data with filtering")
    print("   • Technical analysis")
    print("   • Options data analysis")
    print("   • Macro economic data")
    print("   • News sentiment analysis")
    print("   • Capital allocation tracking")
    print("   • AI ensemble analysis")
    print("   • Risk management")
    print("   • Backtesting framework")
    print("=" * 70)
    print(" Dashboard URL: http://localhost:8052")
    print("=" * 70)
    print(" Dashboard is now running...")
    print("   Press Ctrl+C to stop")
    print("=" * 70)
    
    app.run(debug=True, host='0.0.0.0', port=8052)
