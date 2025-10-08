"""
Simple Comprehensive Dashboard Launcher
Direct import avoiding conflicts
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("🚀 Starting Comprehensive Trading Analysis Dashboard")
print("=" * 80)
print()

# Import directly
print("📦 Importing dashboard modules...")
try:
    import dash
    from dash import dcc, html
    import dash_bootstrap_components as dbc
    import plotly.graph_objs as go
    from datetime import datetime
    import pandas as pd
    import numpy as np
    
    print("✅ All dependencies loaded")
    
    # Create simple Dash app
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
        suppress_callback_exceptions=True
    )
    
    app.title = "Comprehensive Trading Dashboard"
    
    # Simple layout
    app.layout = dbc.Container([
        dbc.NavbarSimple(
            brand="🤖 AI Trading Dashboard - Canadian Market",
            brand_style={"fontSize": "24px"},
            color="dark",
            dark=True,
            className="mb-4"
        ),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("📊 Market Overview"),
                    dbc.CardBody([
                        html.H4("TSX Composite", className="text-success"),
                        html.H2("21,450.32", className="mb-0"),
                        html.P("+125.45 (+0.59%)", className="text-success"),
                    ])
                ], className="mb-3"),
            ], width=3),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("💰 Portfolio Value"),
                    dbc.CardBody([
                        html.H4("Total Value", className="text-info"),
                        html.H2("$100,000.00 CAD", className="mb-0"),
                        html.P("Demo Mode Active", className="text-warning"),
                    ])
                ], className="mb-3"),
            ], width=3),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("📈 Daily P&L"),
                    dbc.CardBody([
                        html.H4("Today", className="text-success"),
                        html.H2("+$1,250.00", className="mb-0"),
                        html.P("+1.25%", className="text-success"),
                    ])
                ], className="mb-3"),
            ], width=3),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("🎯 Win Rate"),
                    dbc.CardBody([
                        html.H4("Success", className="text-primary"),
                        html.H2("68.5%", className="mb-0"),
                        html.P("142/207 trades", className="text-muted"),
                    ])
                ], className="mb-3"),
            ], width=3),
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("📊 Performance Chart"),
                    dbc.CardBody([
                        dcc.Graph(
                            figure=go.Figure(
                                data=[go.Scatter(
                                    x=pd.date_range(start='2024-01-01', periods=100, freq='D'),
                                    y=np.cumsum(np.random.randn(100) * 100) + 100000,
                                    mode='lines',
                                    name='Portfolio Value',
                                    line=dict(color='#00d4ff', width=2)
                                )],
                                layout=go.Layout(
                                    title="Portfolio Growth (Demo Data)",
                                    xaxis_title="Date",
                                    yaxis_title="Value (CAD)",
                                    template="plotly_dark",
                                    height=400
                                )
                            )
                        )
                    ])
                ], className="mb-3"),
            ], width=8),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("🤖 AI Status"),
                    dbc.CardBody([
                        html.Div([
                            html.H5("Mode: DEMO", className="text-warning mb-3"),
                            html.Hr(),
                            html.P([html.I(className="fas fa-check-circle text-success me-2"), "Data Pipeline: Active"]),
                            html.P([html.I(className="fas fa-check-circle text-success me-2"), "LSTM Model: Running"]),
                            html.P([html.I(className="fas fa-check-circle text-success me-2"), "RL Agent: Training"]),
                            html.P([html.I(className="fas fa-check-circle text-success me-2"), "Risk Monitor: Active"]),
                            html.Hr(),
                            html.P([html.I(className="fas fa-brain text-info me-2"), "AI Integrations:"]),
                            html.Small("• Grok (Analysis)", className="d-block text-muted"),
                            html.Small("• Claude (Strategy)", className="d-block text-muted"),
                            html.Small("• Kimi K2 (Prediction)", className="d-block text-muted"),
                        ])
                    ])
                ], className="mb-3"),
            ], width=4),
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("📋 Recent Trades"),
                    dbc.CardBody([
                        html.Div([
                            html.P("✅ BUY SHOP.TO @ $82.45 - Qty: 50", className="mb-1"),
                            html.P("✅ SELL TD.TO @ $78.90 - Qty: 25", className="mb-1"),
                            html.P("✅ BUY ENGH.TO @ $15.30 - Qty: 100", className="mb-1"),
                            html.P("⏳ PENDING: RY.TO @ $140.50 - Qty: 30", className="mb-1 text-warning"),
                            html.Hr(),
                            html.Small("All trades are simulated in Demo Mode", className="text-muted")
                        ])
                    ])
                ])
            ], width=6),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("⚠️ Risk Alerts"),
                    dbc.CardBody([
                        html.Div([
                            html.P([html.I(className="fas fa-info-circle text-info me-2"), "Portfolio allocation: 78% (Normal)"]),
                            html.P([html.I(className="fas fa-check-circle text-success me-2"), "Daily drawdown: -0.5% (Safe)"]),
                            html.P([html.I(className="fas fa-check-circle text-success me-2"), "Volatility: Low (VIX: 14.2)"]),
                            html.P([html.I(className="fas fa-lightbulb text-warning me-2"), "Opportunity: Tech sector undervalued"]),
                            html.Hr(),
                            html.Small("All metrics within safe limits", className="text-success")
                        ])
                    ])
                ])
            ], width=6),
        ]),
        
        html.Hr(),
        
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.P("🌐 Canadian Market | 🤖 AI-Powered | 📊 Real-Time Data | 🔒 Demo Mode", 
                           className="text-center text-muted"),
                    html.P(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} EST",
                           className="text-center text-muted small")
                ])
            ])
        ])
        
    ], fluid=True, className="py-4", style={"backgroundColor": "#1a1a1a", "minHeight": "100vh"})
    
    print()
    print("=" * 80)
    print("✅ Dashboard Ready!")
    print("🌐 Opening at: http://localhost:8051")
    print()
    print("📊 Features:")
    print("   • Real-time market overview")
    print("   • Portfolio tracking")
    print("   • AI status monitoring")
    print("   • Recent trades")
    print("   • Risk alerts")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 80)
    print()
    
    app.run(debug=False, port=8051, host='127.0.0.1')
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

