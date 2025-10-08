"""
Mode Switcher Component for Dashboard
Live/Demo mode toggle with detailed information and safety features
"""

from dash import html, dcc
import dash_bootstrap_components as dbc
from typing import Dict, Optional

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
    'muted': '#8B8B8B',
    'demo': '#5367FE',
    'live': '#EB5B3C'
}

def create_mode_switcher(current_mode: str = 'demo', demo_info: Optional[Dict] = None, live_info: Optional[Dict] = None):
    """
    Create mode switcher component with slider
    
    Args:
        current_mode: 'demo' or 'live'
        demo_info: Demo account information
        live_info: Live account information
    """
    
    demo_info = demo_info or {'capital': 100000.0, 'num_trades': 0}
    live_info = live_info or {'capital': 0.0, 'num_trades': 0}
    
    is_demo = current_mode == 'demo'
    
    return dbc.Card([
        dbc.CardBody([
            # Header
            html.Div([
                html.H5([
                    html.I(className="fas fa-exchange-alt me-2", style={'color': COLORS['primary']}),
                    "Trading Mode"
                ], className="mb-0", style={'color': COLORS['text']})
            ], className="mb-3"),
            
            # Mode Slider
            html.Div([
                html.Div([
                    # Demo Label
                    html.Div([
                        html.Div([
                            html.I(className="fas fa-graduation-cap", style={'fontSize': '24px', 'color': COLORS['demo']}),
                            html.Div([
                                html.Strong("DEMO MODE", style={'color': COLORS['demo'], 'fontSize': '14px'}),
                                html.Div("Practice Trading", style={'fontSize': '11px', 'color': COLORS['muted']})
                            ], className="ms-2")
                        ], className="d-flex align-items-center")
                    ], style={'flex': '1', 'textAlign': 'left'}),
                    
                    # Slider
                    html.Div([
                        dcc.RadioItems(
                            id='mode-switcher',
                            options=[
                                {'label': '', 'value': 'demo'},
                                {'label': '', 'value': 'live'}
                            ],
                            value=current_mode,
                            inline=True,
                            className='mode-switch-slider',
                            style={
                                'display': 'flex',
                                'justifyContent': 'center',
                                'gap': '40px'
                            }
                        )
                    ], style={'flex': '0 0 auto', 'margin': '0 30px'}),
                    
                    # Live Label
                    html.Div([
                        html.Div([
                            html.I(className="fas fa-rocket", style={'fontSize': '24px', 'color': COLORS['live']}),
                            html.Div([
                                html.Strong("LIVE MODE", style={'color': COLORS['live'], 'fontSize': '14px'}),
                                html.Div("Real Money", style={'fontSize': '11px', 'color': COLORS['muted']})
                            ], className="ms-2")
                        ], className="d-flex align-items-center")
                    ], style={'flex': '1', 'textAlign': 'right'})
                ], className="d-flex align-items-center justify-content-between", style={'padding': '20px'})
            ], style={
                'backgroundColor': COLORS['background'],
                'borderRadius': '10px',
                'marginBottom': '20px'
            }),
            
            # Current Mode Status
            html.Div([
                dbc.Alert([
                    html.Div([
                        html.I(
                            className="fas fa-graduation-cap me-2" if is_demo else "fas fa-rocket me-2",
                            style={'fontSize': '20px'}
                        ),
                        html.Strong(
                            f"Currently in {current_mode.upper()} Mode",
                            style={'fontSize': '16px'}
                        )
                    ], className="d-flex align-items-center justify-content-center")
                ], color="info" if is_demo else "danger", className="mb-3"),
            ]),
            
            # Mode Information Cards
            dbc.Row([
                # Demo Mode Info
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="fas fa-graduation-cap", style={'fontSize': '20px', 'color': COLORS['demo']}),
                                html.Strong("Demo Account", className="ms-2", style={'color': COLORS['text']})
                            ], className="d-flex align-items-center mb-3"),
                            
                            html.Div([
                                html.Div([
                                    html.Small("Capital", className="text-muted"),
                                    html.H5(f"${demo_info['capital']:,.2f}", className="mb-0", style={'color': COLORS['text']})
                                ], className="mb-2"),
                                
                                html.Div([
                                    html.Small("Trades", className="text-muted"),
                                    html.H5(f"{demo_info['num_trades']}", className="mb-0", style={'color': COLORS['text']})
                                ], className="mb-2"),
                                
                                html.Div([
                                    html.I(className="fas fa-check-circle me-1", style={'color': COLORS['success'], 'fontSize': '12px'}),
                                    html.Small("Real-time market data", style={'color': COLORS['success']})
                                ], className="mb-1"),
                                
                                html.Div([
                                    html.I(className="fas fa-check-circle me-1", style={'color': COLORS['success'], 'fontSize': '12px'}),
                                    html.Small("Full feature access", style={'color': COLORS['success']})
                                ], className="mb-1"),
                                
                                html.Div([
                                    html.I(className="fas fa-check-circle me-1", style={'color': COLORS['success'], 'fontSize': '12px'}),
                                    html.Small("No financial risk", style={'color': COLORS['success']})
                                ])
                            ])
                        ])
                    ], className="shadow-sm border-0", style={
                        'border': f"3px solid {COLORS['demo']}" if is_demo else f"1px solid {COLORS['muted']}",
                        'backgroundColor': COLORS['card']
                    })
                ], width=6),
                
                # Live Mode Info
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="fas fa-rocket", style={'fontSize': '20px', 'color': COLORS['live']}),
                                html.Strong("Live Account", className="ms-2", style={'color': COLORS['text']})
                            ], className="d-flex align-items-center mb-3"),
                            
                            html.Div([
                                html.Div([
                                    html.Small("Capital", className="text-muted"),
                                    html.H5(
                                        f"${live_info['capital']:,.2f}" if live_info['capital'] > 0 else "Not Set",
                                        className="mb-0",
                                        style={'color': COLORS['text']}
                                    )
                                ], className="mb-2"),
                                
                                html.Div([
                                    html.Small("Trades", className="text-muted"),
                                    html.H5(f"{live_info['num_trades']}", className="mb-0", style={'color': COLORS['text']})
                                ], className="mb-2"),
                                
                                html.Div([
                                    html.I(className="fas fa-exclamation-triangle me-1", style={'color': COLORS['warning'], 'fontSize': '12px'}),
                                    html.Small("Real money at risk", style={'color': COLORS['warning']})
                                ], className="mb-1"),
                                
                                html.Div([
                                    html.I(className="fas fa-shield-alt me-1", style={'color': COLORS['info'], 'fontSize': '12px'}),
                                    html.Small("Enhanced risk controls", style={'color': COLORS['info']})
                                ], className="mb-1"),
                                
                                html.Div([
                                    html.I(className="fas fa-lock me-1", style={'color': COLORS['muted'], 'fontSize': '12px'}),
                                    html.Small("Manual execution only", style={'color': COLORS['muted']})
                                ])
                            ])
                        ])
                    ], className="shadow-sm border-0", style={
                        'border': f"3px solid {COLORS['live']}" if not is_demo else f"1px solid {COLORS['muted']}",
                        'backgroundColor': COLORS['card']
                    })
                ], width=6)
            ], className="mb-3"),
            
            # Features Comparison
            html.Div([
                html.H6("Features Comparison", className="mb-3", style={'color': COLORS['text']}),
                
                dbc.Table([
                    html.Thead([
                        html.Tr([
                            html.Th("Feature", style={'width': '50%'}),
                            html.Th("Demo", className="text-center", style={'color': COLORS['demo']}),
                            html.Th("Live", className="text-center", style={'color': COLORS['live']})
                        ])
                    ]),
                    html.Tbody([
                        html.Tr([
                            html.Td("Stocks Trading"),
                            html.Td(html.I(className="fas fa-check", style={'color': COLORS['success']}), className="text-center"),
                            html.Td(html.I(className="fas fa-check", style={'color': COLORS['success']}), className="text-center")
                        ]),
                        html.Tr([
                            html.Td("Partial Shares"),
                            html.Td(html.I(className="fas fa-check", style={'color': COLORS['success']}), className="text-center"),
                            html.Td(html.I(className="fas fa-check", style={'color': COLORS['success']}), className="text-center")
                        ]),
                        html.Tr([
                            html.Td("Options Trading"),
                            html.Td(html.I(className="fas fa-check", style={'color': COLORS['success']}), className="text-center"),
                            html.Td(html.I(className="fas fa-check", style={'color': COLORS['success']}), className="text-center")
                        ]),
                        html.Tr([
                            html.Td("SIP (Systematic Investment)"),
                            html.Td(html.I(className="fas fa-check", style={'color': COLORS['success']}), className="text-center"),
                            html.Td(html.I(className="fas fa-check", style={'color': COLORS['success']}), className="text-center")
                        ]),
                        html.Tr([
                            html.Td("Max Position Size"),
                            html.Td("20%", className="text-center"),
                            html.Td("10%", className="text-center")
                        ]),
                        html.Tr([
                            html.Td("Daily Loss Limit"),
                            html.Td("5%", className="text-center"),
                            html.Td("3%", className="text-center")
                        ]),
                        html.Tr([
                            html.Td("AI Trading"),
                            html.Td(html.I(className="fas fa-check", style={'color': COLORS['success']}), className="text-center"),
                            html.Td(html.I(className="fas fa-check", style={'color': COLORS['success']}), className="text-center")
                        ]),
                        html.Tr([
                            html.Td("Trade Sharing & Learning"),
                            html.Td(html.I(className="fas fa-check", style={'color': COLORS['success']}), className="text-center"),
                            html.Td(html.I(className="fas fa-check", style={'color': COLORS['success']}), className="text-center")
                        ])
                    ])
                ], bordered=True, hover=True, size='sm', className="mb-0")
            ], className="mb-3"),
            
            # Important Notice
            dbc.Alert([
                html.Div([
                    html.I(className="fas fa-info-circle me-2"),
                    html.Strong("Important: ", style={'fontSize': '14px'}),
                    html.Span(
                        "Both modes share AI learning insights. Demo mode helps the AI learn with zero risk, while live mode validates strategies with real market execution.",
                        style={'fontSize': '13px'}
                    )
                ])
            ], color="info", className="mb-0")
        ])
    ], className="shadow-sm border-0 mb-4")

def create_mode_comparison_chart():
    """Create side-by-side comparison chart for demo vs live performance"""
    
    return dbc.Card([
        dbc.CardHeader([
            html.I(className="fas fa-chart-bar me-2", style={'color': COLORS['primary']}),
            html.Strong("Mode Performance Comparison", style={'color': COLORS['text']})
        ], style={'backgroundColor': COLORS['background'], 'border': 'none'}),
        dbc.CardBody([
            dcc.Graph(
                id='mode-comparison-chart',
                config={'displayModeBar': False},
                style={'height': '400px'}
            )
        ])
    ], className="shadow-sm border-0 mb-4")

def create_learning_insights_panel():
    """Create panel showing shared learning insights"""
    
    return dbc.Card([
        dbc.CardHeader([
            html.I(className="fas fa-brain me-2", style={'color': COLORS['primary']}),
            html.Strong("AI Learning Insights", style={'color': COLORS['text']}),
            dbc.Badge("Shared Learning", color="success", className="ms-2")
        ], style={'backgroundColor': COLORS['background'], 'border': 'none'}),
        dbc.CardBody([
            html.Div(id='learning-insights-content')
        ])
    ], className="shadow-sm border-0 mb-4")

def create_trade_sharing_panel():
    """Create panel showing trade information sharing between modes"""
    
    return dbc.Card([
        dbc.CardHeader([
            html.I(className="fas fa-share-alt me-2", style={'color': COLORS['primary']}),
            html.Strong("Trade Information Sharing", style={'color': COLORS['text']})
        ], style={'backgroundColor': COLORS['background'], 'border': 'none'}),
        dbc.CardBody([
            html.Div(id='trade-sharing-content')
        ])
    ], className="shadow-sm border-0")

# CSS for mode switcher (to be added to dashboard)
MODE_SWITCHER_CSS = """
<style>
.mode-switch-slider input[type="radio"] {
    appearance: none;
    width: 60px;
    height: 30px;
    background: #ddd;
    border-radius: 15px;
    position: relative;
    cursor: pointer;
    transition: all 0.3s ease;
}

.mode-switch-slider input[type="radio"]:checked {
    background: #00D09C;
}

.mode-switch-slider input[type="radio"]:after {
    content: '';
    position: absolute;
    width: 26px;
    height: 26px;
    border-radius: 50%;
    background: white;
    top: 2px;
    left: 2px;
    transition: all 0.3s ease;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
}

.mode-switch-slider input[type="radio"]:checked:after {
    left: 32px;
}

.mode-switch-slider label {
    display: none;
}
</style>
"""

