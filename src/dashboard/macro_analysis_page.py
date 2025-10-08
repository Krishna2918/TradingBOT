"""
Macro Economic Data Analysis Page - Comprehensive Macro Analytics
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

def create_macro_analysis_page():
    """Create comprehensive macro economic analysis page"""
    
    return html.Div([
        html.H2("Macro Economic Data Analysis", className="mb-4"),
        
        # Macro Data Filters
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-filter me-2", style={'color': COLORS['primary']}),
                        html.Strong("Macro Data Filters", style={'color': COLORS['text']})
                    ]),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Label("Data Category"),
                                dcc.Dropdown(
                                    id='macro-category-filter',
                                    options=[
                                        {'label': 'All Categories', 'value': 'all'},
                                        {'label': 'Interest Rates', 'value': 'rates'},
                                        {'label': 'Inflation', 'value': 'inflation'},
                                        {'label': 'Employment', 'value': 'employment'},
                                        {'label': 'GDP & Growth', 'value': 'gdp'},
                                        {'label': 'Trade & Current Account', 'value': 'trade'},
                                        {'label': 'Housing', 'value': 'housing'},
                                        {'label': 'Consumer Data', 'value': 'consumer'},
                                        {'label': 'Business Data', 'value': 'business'}
                                    ],
                                    value='all'
                                )
                            ], width=4),
                            dbc.Col([
                                html.Label("Time Period"),
                                dcc.Dropdown(
                                    id='macro-period-filter',
                                    options=[
                                        {'label': 'Last 3 Months', 'value': '3m'},
                                        {'label': 'Last 6 Months', 'value': '6m'},
                                        {'label': 'Last Year', 'value': '1y'},
                                        {'label': 'Last 2 Years', 'value': '2y'},
                                        {'label': 'Last 5 Years', 'value': '5y'},
                                        {'label': 'Custom Range', 'value': 'custom'}
                                    ],
                                    value='1y'
                                )
                            ], width=4),
                            dbc.Col([
                                html.Label("Country/Region"),
                                dcc.Dropdown(
                                    id='macro-country-filter',
                                    options=[
                                        {'label': 'Canada', 'value': 'canada'},
                                        {'label': 'United States', 'value': 'usa'},
                                        {'label': 'Global', 'value': 'global'},
                                        {'label': 'G7 Countries', 'value': 'g7'},
                                        {'label': 'Emerging Markets', 'value': 'emerging'}
                                    ],
                                    value='canada'
                                )
                            ], width=4)
                        ], className="mb-3"),
                        
                        dbc.Row([
                            dbc.Col([
                                html.Label("Custom Date Range"),
                                dcc.DatePickerRange(
                                    id='macro-date-range',
                                    start_date=datetime.now() - timedelta(days=365),
                                    end_date=datetime.now(),
                                    display_format='YYYY-MM-DD'
                                )
                            ], width=6),
                            dbc.Col([
                                html.Label("Data Frequency"),
                                dcc.Dropdown(
                                    id='macro-frequency-filter',
                                    options=[
                                        {'label': 'Daily', 'value': 'daily'},
                                        {'label': 'Weekly', 'value': 'weekly'},
                                        {'label': 'Monthly', 'value': 'monthly'},
                                        {'label': 'Quarterly', 'value': 'quarterly'},
                                        {'label': 'Annual', 'value': 'annual'}
                                    ],
                                    value='monthly'
                                )
                            ], width=6)
                        ])
                    ])
                ], className="mb-4")
            ])
        ]),
        
        # Key Economic Indicators
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-chart-line me-2", style={'color': COLORS['primary']}),
                        html.Strong("Key Economic Indicators", style={'color': COLORS['text']})
                    ]),
                    dbc.CardBody([
                        html.Div(id='macro-key-indicators')
                    ])
                ], className="shadow-sm border-0 mb-4")
            ])
        ]),
        
        # Economic Data Charts
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-chart-area me-2", style={'color': COLORS['primary']}),
                        html.Strong("Interest Rates & Monetary Policy", style={'color': COLORS['text']})
                    ]),
                    dbc.CardBody([
                        dcc.Graph(
                            id='macro-rates-chart',
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
                        html.Strong("Inflation & CPI", style={'color': COLORS['text']})
                    ]),
                    dbc.CardBody([
                        dcc.Graph(
                            id='macro-inflation-chart',
                            config={'displayModeBar': True},
                            style={'height': '400px'}
                        )
                    ])
                ], className="shadow-sm border-0 mb-4")
            ], width=12, lg=6)
        ]),
        
        # Employment & GDP
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-users me-2", style={'color': COLORS['primary']}),
                        html.Strong("Employment Data", style={'color': COLORS['text']})
                    ]),
                    dbc.CardBody([
                        dcc.Graph(
                            id='macro-employment-chart',
                            config={'displayModeBar': True},
                            style={'height': '400px'}
                        )
                    ])
                ], className="shadow-sm border-0 mb-4")
            ], width=12, lg=6),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-chart-pie me-2", style={'color': COLORS['primary']}),
                        html.Strong("GDP & Growth", style={'color': COLORS['text']})
                    ]),
                    dbc.CardBody([
                        dcc.Graph(
                            id='macro-gdp-chart',
                            config={'displayModeBar': True},
                            style={'height': '400px'}
                        )
                    ])
                ], className="shadow-sm border-0 mb-4")
            ], width=12, lg=6)
        ]),
        
        # Trade & Housing
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-ship me-2", style={'color': COLORS['primary']}),
                        html.Strong("Trade & Current Account", style={'color': COLORS['text']})
                    ]),
                    dbc.CardBody([
                        dcc.Graph(
                            id='macro-trade-chart',
                            config={'displayModeBar': True},
                            style={'height': '400px'}
                        )
                    ])
                ], className="shadow-sm border-0 mb-4")
            ], width=12, lg=6),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-home me-2", style={'color': COLORS['primary']}),
                        html.Strong("Housing Market", style={'color': COLORS['text']})
                    ]),
                    dbc.CardBody([
                        dcc.Graph(
                            id='macro-housing-chart',
                            config={'displayModeBar': True},
                            style={'height': '400px'}
                        )
                    ])
                ], className="shadow-sm border-0 mb-4")
            ], width=12, lg=6)
        ]),
        
        # Economic Calendar
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-calendar me-2", style={'color': COLORS['primary']}),
                        html.Strong("Economic Calendar", style={'color': COLORS['text']})
                    ]),
                    dbc.CardBody([
                        html.Div(id='macro-economic-calendar')
                    ])
                ], className="shadow-sm border-0 mb-4")
            ])
        ]),
        
        # Macro Analysis Summary
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-chart-line me-2", style={'color': COLORS['primary']}),
                        html.Strong("Macro Analysis Summary", style={'color': COLORS['text']})
                    ]),
                    dbc.CardBody([
                        html.Div(id='macro-analysis-summary')
                    ])
                ], className="shadow-sm border-0")
            ])
        ])
    ])

def create_key_economic_indicators():
    """Create key economic indicators cards"""
    
    # Mock economic indicators data
    indicators = [
        {
            'name': 'Bank of Canada Rate',
            'value': '5.00%',
            'change': '+0.25%',
            'change_color': COLORS['danger'],
            'trend': 'Rising',
            'impact': 'High',
            'icon': 'fas fa-percentage'
        },
        {
            'name': 'CPI Inflation',
            'value': '3.1%',
            'change': '-0.2%',
            'change_color': COLORS['success'],
            'trend': 'Falling',
            'impact': 'High',
            'icon': 'fas fa-chart-line'
        },
        {
            'name': 'Unemployment Rate',
            'value': '5.4%',
            'change': '+0.1%',
            'change_color': COLORS['warning'],
            'trend': 'Rising',
            'impact': 'Medium',
            'icon': 'fas fa-users'
        },
        {
            'name': 'GDP Growth',
            'value': '2.1%',
            'change': '+0.3%',
            'change_color': COLORS['success'],
            'trend': 'Rising',
            'impact': 'High',
            'icon': 'fas fa-chart-area'
        },
        {
            'name': 'CAD/USD',
            'value': '0.7350',
            'change': '-0.0020',
            'change_color': COLORS['danger'],
            'trend': 'Falling',
            'impact': 'High',
            'icon': 'fas fa-exchange-alt'
        },
        {
            'name': 'Oil Price (WTI)',
            'value': '$78.50',
            'change': '+$2.30',
            'change_color': COLORS['success'],
            'trend': 'Rising',
            'impact': 'Medium',
            'icon': 'fas fa-oil-can'
        }
    ]
    
    indicator_cards = []
    for indicator in indicators:
        indicator_cards.append(
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.Div([
                                html.I(className=indicator['icon'], style={'fontSize': '24px', 'color': COLORS['primary']}),
                                html.H5(indicator['name'], className="ms-2 mb-1", style={'color': COLORS['text']})
                            ], className="d-flex align-items-center mb-2"),
                            html.H3(indicator['value'], className="mb-1", style={'color': COLORS['text'], 'fontWeight': 'bold'}),
                            html.Div([
                                html.Span(indicator['change'], style={'color': indicator['change_color']}, className="me-2"),
                                html.Small(f"({indicator['trend']})", className="text-muted")
                            ], className="mb-1"),
                            html.Small(f"Impact: {indicator['impact']}", className="text-muted")
                        ])
                    ])
                ], className="shadow-sm border-0 h-100")
            ], width=12, md=6, lg=4, className="mb-3")
        )
    
    return dbc.Row(indicator_cards)

def create_interest_rates_chart():
    """Create interest rates chart"""
    
    # Mock interest rates data
    dates = pd.date_range(start=datetime.now() - timedelta(days=365), end=datetime.now(), freq='M')
    
    boc_rate = 4.5 + np.cumsum(np.random.normal(0, 0.1, len(dates)))
    boc_rate = np.maximum(boc_rate, 0)  # Ensure non-negative rates
    
    fed_rate = 4.0 + np.cumsum(np.random.normal(0, 0.1, len(dates)))
    fed_rate = np.maximum(fed_rate, 0)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=boc_rate,
        mode='lines+markers',
        name='Bank of Canada Rate',
        line=dict(color=COLORS['primary'], width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=fed_rate,
        mode='lines+markers',
        name='Federal Reserve Rate',
        line=dict(color=COLORS['info'], width=3)
    ))
    
    fig.update_layout(
        title="Central Bank Interest Rates",
        xaxis_title="Date",
        yaxis_title="Interest Rate (%)",
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

def create_inflation_chart():
    """Create inflation chart"""
    
    # Mock inflation data
    dates = pd.date_range(start=datetime.now() - timedelta(days=365), end=datetime.now(), freq='M')
    
    cpi_canada = 3.0 + np.cumsum(np.random.normal(0, 0.2, len(dates)))
    cpi_usa = 3.2 + np.cumsum(np.random.normal(0, 0.2, len(dates)))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=cpi_canada,
        mode='lines+markers',
        name='Canada CPI',
        line=dict(color=COLORS['primary'], width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=cpi_usa,
        mode='lines+markers',
        name='US CPI',
        line=dict(color=COLORS['info'], width=3)
    ))
    
    # Add target line
    fig.add_hline(y=2.0, line_dash="dash", line_color=COLORS['success'], annotation_text="Target 2%")
    
    fig.update_layout(
        title="Consumer Price Index (CPI)",
        xaxis_title="Date",
        yaxis_title="CPI (%)",
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

def create_employment_chart():
    """Create employment chart"""
    
    # Mock employment data
    dates = pd.date_range(start=datetime.now() - timedelta(days=365), end=datetime.now(), freq='M')
    
    unemployment_rate = 5.5 + np.cumsum(np.random.normal(0, 0.1, len(dates)))
    unemployment_rate = np.maximum(unemployment_rate, 3.0)  # Minimum 3%
    unemployment_rate = np.minimum(unemployment_rate, 10.0)  # Maximum 10%
    
    job_creation = np.random.normal(20000, 5000, len(dates))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=unemployment_rate,
        mode='lines+markers',
        name='Unemployment Rate (%)',
        line=dict(color=COLORS['primary'], width=3),
        yaxis='y'
    ))
    
    fig.add_trace(go.Bar(
        x=dates,
        y=job_creation,
        name='Job Creation',
        marker_color=COLORS['success'],
        opacity=0.7,
        yaxis='y2'
    ))
    
    fig.update_layout(
        title="Employment Data",
        xaxis_title="Date",
        yaxis=dict(title="Unemployment Rate (%)", side="left"),
        yaxis2=dict(title="Job Creation", side="right", overlaying="y"),
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

def create_gdp_chart():
    """Create GDP chart"""
    
    # Mock GDP data
    quarters = ['Q1 2023', 'Q2 2023', 'Q3 2023', 'Q4 2023', 'Q1 2024', 'Q2 2024', 'Q3 2024', 'Q4 2024']
    gdp_growth = [2.1, 1.8, 2.3, 2.0, 2.2, 1.9, 2.1, 2.0]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=quarters,
        y=gdp_growth,
        marker_color=[COLORS['success'] if x > 2.0 else COLORS['warning'] if x > 1.5 else COLORS['danger'] for x in gdp_growth],
        name='GDP Growth (%)'
    ))
    
    # Add trend line
    fig.add_trace(go.Scatter(
        x=quarters,
        y=gdp_growth,
        mode='lines+markers',
        name='Trend',
        line=dict(color=COLORS['primary'], width=2),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title="GDP Growth Rate",
        xaxis_title="Quarter",
        yaxis_title="GDP Growth (%)",
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

def create_trade_chart():
    """Create trade chart"""
    
    # Mock trade data
    dates = pd.date_range(start=datetime.now() - timedelta(days=365), end=datetime.now(), freq='M')
    
    trade_balance = np.random.normal(-2000, 1000, len(dates))
    exports = np.random.normal(50000, 5000, len(dates))
    imports = exports + trade_balance
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=exports,
        mode='lines+markers',
        name='Exports',
        line=dict(color=COLORS['success'], width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=imports,
        mode='lines+markers',
        name='Imports',
        line=dict(color=COLORS['danger'], width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=trade_balance,
        mode='lines+markers',
        name='Trade Balance',
        line=dict(color=COLORS['primary'], width=3),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title="Trade Data",
        xaxis_title="Date",
        yaxis=dict(title="Trade Volume (CAD Million)", side="left"),
        yaxis2=dict(title="Trade Balance (CAD Million)", side="right", overlaying="y"),
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

def create_housing_chart():
    """Create housing chart"""
    
    # Mock housing data
    dates = pd.date_range(start=datetime.now() - timedelta(days=365), end=datetime.now(), freq='M')
    
    home_prices = 650000 + np.cumsum(np.random.normal(1000, 5000, len(dates)))
    home_sales = np.random.normal(50000, 5000, len(dates))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=home_prices,
        mode='lines+markers',
        name='Average Home Price (CAD)',
        line=dict(color=COLORS['primary'], width=3),
        yaxis='y'
    ))
    
    fig.add_trace(go.Bar(
        x=dates,
        y=home_sales,
        name='Home Sales',
        marker_color=COLORS['success'],
        opacity=0.7,
        yaxis='y2'
    ))
    
    fig.update_layout(
        title="Housing Market",
        xaxis_title="Date",
        yaxis=dict(title="Home Price (CAD)", side="left"),
        yaxis2=dict(title="Home Sales", side="right", overlaying="y"),
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

def create_economic_calendar():
    """Create economic calendar"""
    
    # Mock economic calendar data
    events = [
        {
            'date': '2024-01-15',
            'time': '08:30',
            'event': 'Bank of Canada Rate Decision',
            'importance': 'High',
            'forecast': '5.00%',
            'previous': '4.75%',
            'impact': 'High'
        },
        {
            'date': '2024-01-16',
            'time': '08:30',
            'event': 'CPI Inflation (Dec)',
            'importance': 'High',
            'forecast': '3.1%',
            'previous': '3.3%',
            'impact': 'High'
        },
        {
            'date': '2024-01-17',
            'time': '08:30',
            'event': 'Employment Change (Dec)',
            'importance': 'Medium',
            'forecast': '15K',
            'previous': '12K',
            'impact': 'Medium'
        },
        {
            'date': '2024-01-18',
            'time': '08:30',
            'event': 'Retail Sales (Nov)',
            'importance': 'Medium',
            'forecast': '0.3%',
            'previous': '0.2%',
            'impact': 'Medium'
        },
        {
            'date': '2024-01-19',
            'time': '08:30',
            'event': 'GDP Growth (Q4)',
            'importance': 'High',
            'forecast': '2.1%',
            'previous': '1.8%',
            'impact': 'High'
        }
    ]
    
    calendar_items = []
    for event in events:
        importance_color = COLORS['danger'] if event['importance'] == 'High' else COLORS['warning'] if event['importance'] == 'Medium' else COLORS['success']
        impact_color = COLORS['danger'] if event['impact'] == 'High' else COLORS['warning'] if event['impact'] == 'Medium' else COLORS['success']
        
        calendar_items.append(
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Div([
                            html.Strong(event['date'], className="me-2"),
                            html.Span(event['time'], className="me-2"),
                            dbc.Badge(event['importance'], color="danger" if event['importance'] == 'High' else "warning", className="me-2")
                        ], className="mb-2"),
                        html.H6(event['event'], className="mb-2"),
                        html.Div([
                            html.Small(f"Forecast: {event['forecast']}", className="me-3"),
                            html.Small(f"Previous: {event['previous']}", className="me-3"),
                            html.Small(f"Impact: {event['impact']}", style={'color': impact_color})
                        ])
                    ])
                ])
            ], className="mb-2")
        )
    
    return html.Div(calendar_items)

def create_macro_analysis_summary():
    """Create macro analysis summary"""
    
    # Mock macro analysis
    analysis_points = [
        {
            'category': 'Monetary Policy',
            'outlook': 'Hawkish',
            'description': 'Bank of Canada likely to maintain higher rates to combat inflation',
            'impact': 'Negative for growth stocks, positive for financials',
            'color': COLORS['warning']
        },
        {
            'category': 'Inflation',
            'outlook': 'Moderating',
            'description': 'CPI showing signs of cooling but still above target',
            'impact': 'Mixed impact on different sectors',
            'color': COLORS['info']
        },
        {
            'category': 'Employment',
            'outlook': 'Stable',
            'description': 'Labor market remains tight with moderate job creation',
            'impact': 'Positive for consumer spending',
            'color': COLORS['success']
        },
        {
            'category': 'Trade',
            'outlook': 'Challenging',
            'description': 'Trade balance under pressure from global slowdown',
            'impact': 'Negative for export-oriented companies',
            'color': COLORS['danger']
        },
        {
            'category': 'Housing',
            'outlook': 'Cooling',
            'description': 'Higher rates impacting housing market activity',
            'impact': 'Negative for real estate and construction',
            'color': COLORS['warning']
        }
    ]
    
    analysis_cards = []
    for analysis in analysis_points:
        analysis_cards.append(
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Div([
                            html.H6(analysis['category'], className="mb-1"),
                            html.Span(analysis['outlook'], style={'color': analysis['color'], 'fontWeight': 'bold'}, className="mb-2 d-block")
                        ]),
                        html.P(analysis['description'], className="mb-2"),
                        html.Small(f"Market Impact: {analysis['impact']}", className="text-muted")
                    ])
                ])
            ], className="mb-2")
        )
    
    return html.Div(analysis_cards)
