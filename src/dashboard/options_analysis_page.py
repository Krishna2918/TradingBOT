"""
Options Data Analysis Page - Comprehensive Options Analytics
"""

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

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

def create_options_analysis_page():
    """Create comprehensive options analysis page"""
    
    return html.Div([
        html.H2("Options Data Analysis", className="mb-4"),
        
        # Options Filters
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-filter me-2", style={'color': COLORS['primary']}),
                        html.Strong("Options Filters", style={'color': COLORS['text']})
                    ]),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Label("Underlying Symbol"),
                                dcc.Dropdown(
                                    id='options-symbol-filter',
                                    options=[
                                        {'label': 'XIU.TO - TSX 60 ETF', 'value': 'XIU.TO'},
                                        {'label': 'XSP.TO - S&P 500 ETF', 'value': 'XSP.TO'},
                                        {'label': 'XEG.TO - Energy ETF', 'value': 'XEG.TO'},
                                        {'label': 'XFN.TO - Financials ETF', 'value': 'XFN.TO'},
                                        {'label': 'XIT.TO - Technology ETF', 'value': 'XIT.TO'},
                                        {'label': 'RY.TO - Royal Bank', 'value': 'RY.TO'},
                                        {'label': 'TD.TO - TD Bank', 'value': 'TD.TO'},
                                        {'label': 'SHOP.TO - Shopify', 'value': 'SHOP.TO'}
                                    ],
                                    value='XIU.TO'
                                )
                            ], width=4),
                            dbc.Col([
                                html.Label("Option Type"),
                                dcc.Dropdown(
                                    id='options-type-filter',
                                    options=[
                                        {'label': 'All Options', 'value': 'all'},
                                        {'label': 'Calls Only', 'value': 'calls'},
                                        {'label': 'Puts Only', 'value': 'puts'}
                                    ],
                                    value='all'
                                )
                            ], width=4),
                            dbc.Col([
                                html.Label("Expiration"),
                                dcc.Dropdown(
                                    id='options-expiry-filter',
                                    options=[
                                        {'label': 'All Expirations', 'value': 'all'},
                                        {'label': 'This Week', 'value': 'week'},
                                        {'label': 'This Month', 'value': 'month'},
                                        {'label': 'Next Month', 'value': 'next_month'},
                                        {'label': 'Quarterly', 'value': 'quarterly'}
                                    ],
                                    value='all'
                                )
                            ], width=4)
                        ], className="mb-3"),
                        
                        dbc.Row([
                            dbc.Col([
                                html.Label("Strike Range"),
                                dcc.RangeSlider(
                                    id='options-strike-range',
                                    min=0,
                                    max=100,
                                    step=1,
                                    value=[20, 80],
                                    marks={i: f'${i}' for i in range(0, 101, 20)},
                                    tooltip={"placement": "bottom", "always_visible": True}
                                )
                            ], width=6),
                            dbc.Col([
                                html.Label("Moneyness"),
                                dcc.Dropdown(
                                    id='options-moneyness-filter',
                                    options=[
                                        {'label': 'All Options', 'value': 'all'},
                                        {'label': 'In The Money (ITM)', 'value': 'itm'},
                                        {'label': 'At The Money (ATM)', 'value': 'atm'},
                                        {'label': 'Out Of The Money (OTM)', 'value': 'otm'}
                                    ],
                                    value='all'
                                )
                            ], width=6)
                        ])
                    ])
                ], className="mb-4")
            ])
        ]),
        
        # Options Chain Table
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-table me-2", style={'color': COLORS['primary']}),
                        html.Strong("Options Chain", style={'color': COLORS['text']})
                    ]),
                    dbc.CardBody([
                        html.Div(id='options-chain-table')
                    ])
                ], className="shadow-sm border-0 mb-4")
            ])
        ]),
        
        # Options Analytics Charts
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-chart-line me-2", style={'color': COLORS['primary']}),
                        html.Strong("Implied Volatility Surface", style={'color': COLORS['text']})
                    ]),
                    dbc.CardBody([
                        dcc.Graph(
                            id='options-iv-surface',
                            config={'displayModeBar': True},
                            style={'height': '400px'}
                        )
                    ])
                ], className="shadow-sm border-0 mb-4")
            ], width=12, lg=6),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-chart-bar me-2", style={'color': COLORS['primary']}),
                        html.Strong("Open Interest Distribution", style={'color': COLORS['text']})
                    ]),
                    dbc.CardBody([
                        dcc.Graph(
                            id='options-oi-distribution',
                            config={'displayModeBar': False},
                            style={'height': '400px'}
                        )
                    ])
                ], className="shadow-sm border-0 mb-4")
            ], width=12, lg=6)
        ]),
        
        # Greeks Analysis
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-chart-area me-2", style={'color': COLORS['primary']}),
                        html.Strong("Greeks Analysis", style={'color': COLORS['text']})
                    ]),
                    dbc.CardBody([
                        dcc.Graph(
                            id='options-greeks-chart',
                            config={'displayModeBar': True},
                            style={'height': '400px'}
                        )
                    ])
                ], className="shadow-sm border-0 mb-4")
            ], width=12, lg=8),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-calculator me-2", style={'color': COLORS['primary']}),
                        html.Strong("Greeks Summary", style={'color': COLORS['text']})
                    ]),
                    dbc.CardBody([
                        html.Div(id='options-greeks-summary')
                    ])
                ], className="shadow-sm border-0 mb-4")
            ], width=12, lg=4)
        ]),
        
        # Options Flow Analysis
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-water me-2", style={'color': COLORS['primary']}),
                        html.Strong("Options Flow Analysis", style={'color': COLORS['text']})
                    ]),
                    dbc.CardBody([
                        html.Div(id='options-flow-analysis')
                    ])
                ], className="shadow-sm border-0 mb-4")
            ], width=12, lg=6),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-chart-pie me-2", style={'color': COLORS['primary']}),
                        html.Strong("Put/Call Ratio", style={'color': COLORS['text']})
                    ]),
                    dbc.CardBody([
                        dcc.Graph(
                            id='options-put-call-ratio',
                            config={'displayModeBar': False},
                            style={'height': '300px'}
                        )
                    ])
                ], className="shadow-sm border-0 mb-4")
            ], width=12, lg=6)
        ]),
        
        # Gamma Squeeze Detection
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-exclamation-triangle me-2", style={'color': COLORS['primary']}),
                        html.Strong("Gamma Squeeze Detection", style={'color': COLORS['text']})
                    ]),
                    dbc.CardBody([
                        html.Div(id='options-gamma-squeeze')
                    ])
                ], className="shadow-sm border-0 mb-4")
            ])
        ]),
        
        # Options Strategies
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-chess me-2", style={'color': COLORS['primary']}),
                        html.Strong("Options Strategies", style={'color': COLORS['text']})
                    ]),
                    dbc.CardBody([
                        html.Div(id='options-strategies')
                    ])
                ], className="shadow-sm border-0")
            ])
        ])
    ])

def create_options_chain_table(data):
    """Create options chain table"""
    if not data:
        return html.P("No options data available", className="text-muted")
    
    # Simulate options chain data
    options_data = []
    
    for symbol, df in data.items():
        if df.empty:
            continue
        
        # Generate mock options data
        current_price = df['close'].iloc[-1]
        
        # Generate strikes around current price
        strikes = np.arange(current_price * 0.8, current_price * 1.2, current_price * 0.05)
        
        for strike in strikes:
            # Call option
            call_iv = np.random.uniform(0.15, 0.35)
            call_delta = np.random.uniform(0.1, 0.9)
            call_gamma = np.random.uniform(0.01, 0.05)
            call_theta = -np.random.uniform(0.01, 0.05)
            call_vega = np.random.uniform(0.1, 0.3)
            call_oi = np.random.randint(100, 10000)
            
            # Put option
            put_iv = np.random.uniform(0.15, 0.35)
            put_delta = np.random.uniform(-0.9, -0.1)
            put_gamma = np.random.uniform(0.01, 0.05)
            put_theta = -np.random.uniform(0.01, 0.05)
            put_vega = np.random.uniform(0.1, 0.3)
            put_oi = np.random.randint(100, 10000)
            
            options_data.append({
                'Strike': f"${strike:.2f}",
                'Call Bid': f"${np.random.uniform(0.1, 5.0):.2f}",
                'Call Ask': f"${np.random.uniform(0.1, 5.0):.2f}",
                'Call IV': f"{call_iv:.1%}",
                'Call Delta': f"{call_delta:.2f}",
                'Call OI': f"{call_oi:,}",
                'Put Bid': f"${np.random.uniform(0.1, 5.0):.2f}",
                'Put Ask': f"${np.random.uniform(0.1, 5.0):.2f}",
                'Put IV': f"{put_iv:.1%}",
                'Put Delta': f"{put_delta:.2f}",
                'Put OI': f"{put_oi:,}"
            })
    
    if not options_data:
        return html.P("No options data available", className="text-muted")
    
    df_options = pd.DataFrame(options_data)
    
    # Create styled table
    table = dbc.Table.from_dataframe(
        df_options,
        striped=True,
        bordered=True,
        hover=True,
        responsive=True,
        size='sm'
    )
    
    return table

def create_implied_volatility_surface(data):
    """Create implied volatility surface chart"""
    if not data:
        return go.Figure()
    
    # Generate mock IV surface data
    strikes = np.arange(20, 80, 2)
    expiries = [7, 14, 30, 60, 90, 180]
    
    iv_surface = []
    for expiry in expiries:
        for strike in strikes:
            # Simulate IV smile/skew
            atm_iv = 0.25
            skew = (strike - 50) * 0.001
            iv = atm_iv + skew + np.random.normal(0, 0.02)
            iv = max(0.1, min(0.5, iv))  # Clamp between 10% and 50%
            
            iv_surface.append({
                'strike': strike,
                'expiry': expiry,
                'iv': iv
            })
    
    df_iv = pd.DataFrame(iv_surface)
    
    # Create 3D surface plot
    fig = go.Figure(data=[go.Surface(
        x=df_iv['strike'].values.reshape(len(expiries), len(strikes)),
        y=df_iv['expiry'].values.reshape(len(expiries), len(strikes)),
        z=df_iv['iv'].values.reshape(len(expiries), len(strikes)),
        colorscale='Viridis'
    )])
    
    fig.update_layout(
        title="Implied Volatility Surface",
        scene=dict(
            xaxis_title="Strike Price",
            yaxis_title="Days to Expiry",
            zaxis_title="Implied Volatility"
        ),
        height=400
    )
    
    return fig

def create_open_interest_distribution(data):
    """Create open interest distribution chart"""
    if not data:
        return go.Figure()
    
    # Generate mock OI data
    strikes = np.arange(20, 80, 2)
    call_oi = np.random.randint(100, 10000, len(strikes))
    put_oi = np.random.randint(100, 10000, len(strikes))
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=strikes,
        y=call_oi,
        name='Call OI',
        marker_color=COLORS['success'],
        opacity=0.7
    ))
    
    fig.add_trace(go.Bar(
        x=strikes,
        y=put_oi,
        name='Put OI',
        marker_color=COLORS['danger'],
        opacity=0.7
    ))
    
    fig.update_layout(
        title="Open Interest Distribution",
        xaxis_title="Strike Price",
        yaxis_title="Open Interest",
        barmode='group',
        height=400
    )
    
    return fig

def create_greeks_analysis(data):
    """Create Greeks analysis chart"""
    if not data:
        return go.Figure()
    
    # Generate mock Greeks data
    strikes = np.arange(20, 80, 2)
    
    # Delta (for calls)
    delta = 1 / (1 + np.exp(-(strikes - 50) * 0.1))
    
    # Gamma
    gamma = np.exp(-((strikes - 50) ** 2) / 100) * 0.1
    
    # Theta (time decay)
    theta = -np.ones_like(strikes) * 0.02
    
    # Vega (volatility sensitivity)
    vega = np.ones_like(strikes) * 0.15
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=strikes,
        y=delta,
        mode='lines+markers',
        name='Delta',
        line=dict(color=COLORS['primary'], width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=strikes,
        y=gamma,
        mode='lines+markers',
        name='Gamma',
        line=dict(color=COLORS['info'], width=3),
        yaxis='y2'
    ))
    
    fig.add_trace(go.Scatter(
        x=strikes,
        y=theta,
        mode='lines+markers',
        name='Theta',
        line=dict(color=COLORS['warning'], width=3),
        yaxis='y3'
    ))
    
    fig.add_trace(go.Scatter(
        x=strikes,
        y=vega,
        mode='lines+markers',
        name='Vega',
        line=dict(color=COLORS['success'], width=3),
        yaxis='y4'
    ))
    
    fig.update_layout(
        title="Greeks Analysis",
        xaxis_title="Strike Price",
        yaxis=dict(title="Delta", side="left"),
        yaxis2=dict(title="Gamma", side="right", overlaying="y"),
        yaxis3=dict(title="Theta", side="right", overlaying="y", position=0.85),
        yaxis4=dict(title="Vega", side="right", overlaying="y", position=0.95),
        height=400
    )
    
    return fig

def create_greeks_summary(data):
    """Create Greeks summary"""
    if not data:
        return html.P("No data available", className="text-muted")
    
    # Mock Greeks summary
    greeks_data = [
        {'Greek': 'Delta', 'Value': '0.65', 'Impact': 'Price sensitivity', 'Color': COLORS['primary']},
        {'Greek': 'Gamma', 'Value': '0.03', 'Impact': 'Delta acceleration', 'Color': COLORS['info']},
        {'Greek': 'Theta', 'Value': '-0.02', 'Impact': 'Time decay', 'Color': COLORS['warning']},
        {'Greek': 'Vega', 'Value': '0.15', 'Impact': 'Volatility sensitivity', 'Color': COLORS['success']},
        {'Greek': 'Rho', 'Value': '0.05', 'Impact': 'Interest rate sensitivity', 'Color': COLORS['muted']}
    ]
    
    greeks_cards = []
    for greek in greeks_data:
        greeks_cards.append(
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.H6(greek['Greek'], className="mb-1"),
                        html.H4(greek['Value'], style={'color': greek['Color']}, className="mb-1"),
                        html.Small(greek['Impact'], className="text-muted")
                    ])
                ])
            ], className="mb-2")
        )
    
    return html.Div(greeks_cards)

def create_options_flow_analysis(data):
    """Create options flow analysis"""
    if not data:
        return html.P("No data available", className="text-muted")
    
    # Mock options flow data
    flow_data = [
        {'Time': '09:30', 'Symbol': 'XIU.TO', 'Strike': '$50', 'Type': 'Call', 'Volume': 1500, 'Premium': '$2.50', 'Flow': 'Bullish'},
        {'Time': '10:15', 'Symbol': 'XIU.TO', 'Strike': '$48', 'Type': 'Put', 'Volume': 800, 'Premium': '$1.20', 'Flow': 'Bearish'},
        {'Time': '11:00', 'Symbol': 'XIU.TO', 'Strike': '$52', 'Type': 'Call', 'Volume': 2200, 'Premium': '$1.80', 'Flow': 'Bullish'},
        {'Time': '13:30', 'Symbol': 'XIU.TO', 'Strike': '$49', 'Type': 'Put', 'Volume': 1200, 'Premium': '$0.95', 'Flow': 'Bearish'},
        {'Time': '14:45', 'Symbol': 'XIU.TO', 'Strike': '$51', 'Type': 'Call', 'Volume': 1800, 'Premium': '$2.10', 'Flow': 'Bullish'}
    ]
    
    flow_cards = []
    for flow in flow_data:
        flow_color = COLORS['success'] if flow['Flow'] == 'Bullish' else COLORS['danger']
        flow_cards.append(
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Div([
                            html.Strong(flow['Time'], className="me-2"),
                            html.Span(flow['Symbol'], className="me-2"),
                            html.Span(f"{flow['Strike']} {flow['Type']}", className="me-2")
                        ], className="mb-1"),
                        html.Div([
                            html.Span(f"Volume: {flow['Volume']:,}", className="me-3"),
                            html.Span(f"Premium: {flow['Premium']}", className="me-3"),
                            html.Span(flow['Flow'], style={'color': flow_color, 'fontWeight': 'bold'})
                        ])
                    ])
                ])
            ], className="mb-2")
        )
    
    return html.Div(flow_cards)

def create_put_call_ratio_chart(data):
    """Create put/call ratio chart"""
    if not data:
        return go.Figure()
    
    # Mock put/call ratio data
    dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
    put_call_ratios = np.random.uniform(0.5, 1.5, len(dates))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=put_call_ratios,
        mode='lines+markers',
        name='Put/Call Ratio',
        line=dict(color=COLORS['primary'], width=2)
    ))
    
    # Add reference lines
    fig.add_hline(y=1.0, line_dash="dash", line_color=COLORS['muted'], annotation_text="Neutral")
    fig.add_hline(y=1.2, line_dash="dash", line_color=COLORS['danger'], annotation_text="Bearish")
    fig.add_hline(y=0.8, line_dash="dash", line_color=COLORS['success'], annotation_text="Bullish")
    
    fig.update_layout(
        title="Put/Call Ratio Trend",
        xaxis_title="Date",
        yaxis_title="Put/Call Ratio",
        height=300
    )
    
    return fig

def create_gamma_squeeze_detection(data):
    """Create gamma squeeze detection"""
    if not data:
        return html.P("No data available", className="text-muted")
    
    # Mock gamma squeeze analysis
    squeeze_indicators = [
        {'Indicator': 'Gamma Exposure', 'Value': 'High', 'Status': 'Warning', 'Color': COLORS['warning']},
        {'Indicator': 'Call OI Concentration', 'Value': '85%', 'Status': 'Alert', 'Color': COLORS['danger']},
        {'Indicator': 'Strike Clustering', 'Value': 'Moderate', 'Status': 'Watch', 'Color': COLORS['info']},
        {'Indicator': 'IV Skew', 'Value': 'Steep', 'Status': 'Warning', 'Color': COLORS['warning']},
        {'Indicator': 'Delta Hedging Pressure', 'Value': 'High', 'Status': 'Alert', 'Color': COLORS['danger']}
    ]
    
    squeeze_cards = []
    for indicator in squeeze_indicators:
        squeeze_cards.append(
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Div([
                            html.Strong(indicator['Indicator'], className="me-2"),
                            html.Span(indicator['Value'], style={'color': indicator['Color']}, className="me-2")
                        ], className="mb-1"),
                        html.Small(indicator['Status'], style={'color': indicator['Color']})
                    ])
                ])
            ], className="mb-2")
        )
    
    return html.Div(squeeze_cards)

def create_options_strategies(data):
    """Create options strategies recommendations"""
    if not data:
        return html.P("No data available", className="text-muted")
    
    # Mock strategy recommendations
    strategies = [
        {
            'Strategy': 'Long Straddle',
            'Description': 'Buy call and put at same strike',
            'Market View': 'High volatility expected',
            'Risk': 'Limited to premium paid',
            'Reward': 'Unlimited',
            'Color': COLORS['info']
        },
        {
            'Strategy': 'Iron Condor',
            'Description': 'Sell call spread + sell put spread',
            'Market View': 'Low volatility, range-bound',
            'Risk': 'Limited',
            'Reward': 'Limited to premium received',
            'Color': COLORS['success']
        },
        {
            'Strategy': 'Bull Call Spread',
            'Description': 'Buy lower strike call, sell higher strike call',
            'Market View': 'Moderately bullish',
            'Risk': 'Limited to net debit',
            'Reward': 'Limited to spread width minus debit',
            'Color': COLORS['primary']
        }
    ]
    
    strategy_cards = []
    for strategy in strategies:
        strategy_cards.append(
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.H6(strategy['Strategy'], style={'color': strategy['Color']}, className="mb-2"),
                        html.P(strategy['Description'], className="mb-2"),
                        html.Div([
                            html.Small(f"Market View: {strategy['Market View']}", className="d-block"),
                            html.Small(f"Risk: {strategy['Risk']}", className="d-block"),
                            html.Small(f"Reward: {strategy['Reward']}", className="d-block")
                        ], className="text-muted")
                    ])
                ])
            ], className="mb-2")
        )
    
    return html.Div(strategy_cards)
