"""
Risk Management Dashboard

Real-time risk monitoring with:
- Portfolio risk metrics (VaR, CVaR, drawdown)
- Kill switch controls
- Position limits
- Exposure tracking
- Alert management
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging
from typing import Dict, List

# Import risk components
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.risk_management import CapitalAllocator, LeverageGovernor, KillSwitchManager
from src.execution import get_execution_engine
from src.trading_modes import ModeManager

logger = logging.getLogger(__name__)

# Initialize components
try:
    capital_allocator = CapitalAllocator("config/risk_config.yaml")
    leverage_governor = LeverageGovernor("config/risk_config.yaml")
    kill_switch_manager = KillSwitchManager("config/risk_config.yaml")
    execution_engine = get_execution_engine()
    mode_manager = ModeManager()
    logger.info(" Risk dashboard components initialized")
except Exception as e:
    logger.error(f" Failed to initialize components: {e}")
    capital_allocator = None
    leverage_governor = None
    kill_switch_manager = None
    execution_engine = None
    mode_manager = None

# Initialize Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True
)
app.title = "Risk Management Dashboard"

# Colors
COLORS = {
    'danger': '#DC3545',
    'warning': '#FFC107',
    'success': '#28A745',
    'info': '#17A2B8',
    'primary': '#007BFF',
    'dark': '#343A40',
    'light': '#F8F9FA'
}

def create_risk_metric_card(title: str, value: str, status: str, icon: str):
    """Create a risk metric card"""
    
    color_map = {
        'danger': COLORS['danger'],
        'warning': COLORS['warning'],
        'success': COLORS['success'],
        'info': COLORS['info']
    }
    
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.I(className=f"fas {icon} fa-2x", style={'color': color_map.get(status, COLORS['info'])}),
                html.Div([
                    html.H6(title, className="text-muted mb-1"),
                    html.H3(value, className="mb-0", style={'color': color_map.get(status, COLORS['info'])})
                ], className="ms-3")
            ], className="d-flex align-items-center")
        ])
    ], className="mb-3")

def create_kill_switch_panel():
    """Create kill switch control panel"""
    
    is_active = kill_switch_manager.is_kill_switch_active() if kill_switch_manager else False
    
    return dbc.Card([
        dbc.CardHeader([
            html.I(className="fas fa-power-off me-2"),
            "Kill Switch Control"
        ], className="bg-danger text-white"),
        dbc.CardBody([
            html.Div([
                html.H4("Status:", className="d-inline me-2"),
                dbc.Badge(
                    "ACTIVE" if is_active else "INACTIVE",
                    color="danger" if is_active else "success",
                    className="fs-5"
                )
            ], className="mb-3"),
            
            html.P(
                "The kill switch immediately halts all trading activity. "
                "Use this in emergency situations or market anomalies.",
                className="text-muted"
            ),
            
            html.Div([
                dbc.Button(
                    [html.I(className="fas fa-stop-circle me-2"), "ACTIVATE KILL SWITCH"],
                    id="activate-kill-switch-btn",
                    color="danger",
                    size="lg",
                    className="me-2",
                    disabled=is_active
                ),
                dbc.Button(
                    [html.I(className="fas fa-play-circle me-2"), "DEACTIVATE KILL SWITCH"],
                    id="deactivate-kill-switch-btn",
                    color="success",
                    size="lg",
                    disabled=not is_active
                )
            ]),
            
            html.Div(id="kill-switch-status", className="mt-3")
        ])
    ], className="mb-3")

def create_position_limits_panel():
    """Create position limits monitoring panel"""
    
    return dbc.Card([
        dbc.CardHeader([
            html.I(className="fas fa-chart-bar me-2"),
            "Position Limits"
        ]),
        dbc.CardBody([
            dcc.Graph(id="position-limits-chart"),
            
            html.Hr(),
            
            html.Div([
                html.H6("Active Positions", className="mb-3"),
                html.Div(id="active-positions-list")
            ])
        ])
    ], className="mb-3")

def create_drawdown_monitor():
    """Create drawdown monitoring panel"""
    
    return dbc.Card([
        dbc.CardHeader([
            html.I(className="fas fa-chart-line me-2"),
            "Drawdown Monitor"
        ]),
        dbc.CardBody([
            dcc.Graph(id="drawdown-chart"),
            
            html.Div([
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H6("Current Drawdown"),
                            html.H4(id="current-drawdown", className="text-danger")
                        ])
                    ], width=4),
                    dbc.Col([
                        html.Div([
                            html.H6("Max Drawdown"),
                            html.H4(id="max-drawdown", className="text-danger")
                        ])
                    ], width=4),
                    dbc.Col([
                        html.Div([
                            html.H6("Drawdown Limit"),
                            html.H4("20%", className="text-warning")
                        ])
                    ], width=4)
                ])
            ], className="mt-3")
        ])
    ])

def create_var_analysis():
    """Create VaR (Value at Risk) analysis panel"""
    
    return dbc.Card([
        dbc.CardHeader([
            html.I(className="fas fa-exclamation-triangle me-2"),
            "Value at Risk (VaR) Analysis"
        ]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H6("VaR (95%)", className="text-muted"),
                        html.H4(id="var-95", className="text-warning")
                    ])
                ], width=4),
                dbc.Col([
                    html.Div([
                        html.H6("VaR (99%)", className="text-muted"),
                        html.H4(id="var-99", className="text-danger")
                    ])
                ], width=4),
                dbc.Col([
                    html.Div([
                        html.H6("CVaR (95%)", className="text-muted"),
                        html.H4(id="cvar-95", className="text-danger")
                    ])
                ], width=4)
            ]),
            
            html.Hr(),
            
            dcc.Graph(id="var-distribution-chart")
        ])
    ], className="mb-3")

def create_alert_panel():
    """Create alert management panel"""
    
    return dbc.Card([
        dbc.CardHeader([
            html.I(className="fas fa-bell me-2"),
            "Risk Alerts"
        ]),
        dbc.CardBody([
            html.Div(id="risk-alerts-list")
        ])
    ], className="mb-3")

# Main layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1([
                html.I(className="fas fa-shield-alt me-3"),
                "Risk Management Dashboard"
            ], className="mb-4")
        ])
    ]),
    
    # Auto-refresh interval
    dcc.Interval(id='risk-refresh-interval', interval=5000, n_intervals=0),  # 5 seconds
    
    # Key Risk Metrics Row
    dbc.Row([
        dbc.Col([
            html.Div(id="risk-metrics-cards")
        ], width=12)
    ]),
    
    # Kill Switch Panel
    dbc.Row([
        dbc.Col([
            create_kill_switch_panel()
        ], width=12)
    ]),
    
    # Risk Monitoring Panels
    dbc.Row([
        dbc.Col([
            create_position_limits_panel()
        ], width=6),
        dbc.Col([
            create_drawdown_monitor()
        ], width=6)
    ]),
    
    # VaR Analysis
    dbc.Row([
        dbc.Col([
            create_var_analysis()
        ], width=12)
    ]),
    
    # Alerts
    dbc.Row([
        dbc.Col([
            create_alert_panel()
        ], width=12)
    ])
    
], fluid=True, className="p-4")

# Callbacks

@app.callback(
    Output("risk-metrics-cards", "children"),
    Input("risk-refresh-interval", "n_intervals")
)
def update_risk_metrics(n):
    """Update risk metric cards"""
    
    if not capital_allocator or not leverage_governor or not mode_manager:
        return html.Div(" Risk components not initialized", className="alert alert-warning")
    
    try:
        # Get capital state
        capital_state = capital_allocator.get_capital_state()
        leverage_state = leverage_governor.get_leverage_state()
        account_info = mode_manager.get_current_account_info()
        
        # Calculate metrics
        capital_utilization = (capital_state.active_capital / capital_state.total_capital) * 100
        
        # Determine status
        leverage_status = "danger" if leverage_state.current_leverage > 3 else "warning" if leverage_state.current_leverage > 2 else "success"
        capital_status = "danger" if capital_utilization > 90 else "warning" if capital_utilization > 70 else "success"
        
        return dbc.Row([
            dbc.Col([
                create_risk_metric_card(
                    "Total Capital",
                    f"${capital_state.total_capital:,.0f}",
                    "info",
                    "fa-wallet"
                )
            ], width=3),
            dbc.Col([
                create_risk_metric_card(
                    "Active Capital",
                    f"${capital_state.active_capital:,.0f}",
                    capital_status,
                    "fa-chart-line"
                )
            ], width=3),
            dbc.Col([
                create_risk_metric_card(
                    "Current Leverage",
                    f"{leverage_state.current_leverage:.1f}x",
                    leverage_status,
                    "fa-layer-group"
                )
            ], width=3),
            dbc.Col([
                create_risk_metric_card(
                    "Open Positions",
                    str(account_info.get('num_positions', 0)),
                    "info",
                    "fa-briefcase"
                )
            ], width=3)
        ])
        
    except Exception as e:
        logger.error(f"Error updating risk metrics: {e}")
        return html.Div(f" Error: {e}", className="alert alert-danger")

@app.callback(
    Output("kill-switch-status", "children"),
    [Input("activate-kill-switch-btn", "n_clicks"),
     Input("deactivate-kill-switch-btn", "n_clicks")],
    prevent_initial_call=True
)
def handle_kill_switch(activate_clicks, deactivate_clicks):
    """Handle kill switch activation/deactivation"""
    
    if not kill_switch_manager:
        return dbc.Alert(" Kill switch manager not available", color="warning")
    
    ctx = callback_context
    if not ctx.triggered:
        return ""
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    try:
        if button_id == "activate-kill-switch-btn":
            kill_switch_manager.activate_kill_switch("manual_activation", "dashboard")
            return dbc.Alert([
                html.I(className="fas fa-exclamation-triangle me-2"),
                "Kill switch ACTIVATED. All trading halted."
            ], color="danger", className="mt-3")
        
        elif button_id == "deactivate-kill-switch-btn":
            kill_switch_manager.deactivate_kill_switch("manual_deactivation")
            return dbc.Alert([
                html.I(className="fas fa-check-circle me-2"),
                "Kill switch DEACTIVATED. Trading resumed."
            ], color="success", className="mt-3")
    
    except Exception as e:
        return dbc.Alert(f" Error: {e}", color="danger", className="mt-3")

@app.callback(
    Output("position-limits-chart", "figure"),
    Input("risk-refresh-interval", "n_intervals")
)
def update_position_limits_chart(n):
    """Update position limits chart"""
    
    # Mock data - would use actual position data
    categories = ['Stocks', 'Options', 'Futures', 'ETFs']
    current = [45, 25, 15, 15]  # Current exposure %
    limits = [50, 30, 20, 20]  # Limits %
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Current Exposure',
        x=categories,
        y=current,
        marker_color=COLORS['info']
    ))
    
    fig.add_trace(go.Bar(
        name='Position Limits',
        x=categories,
        y=limits,
        marker_color=COLORS['warning']
    ))
    
    fig.update_layout(
        barmode='group',
        title="Position Exposure vs Limits",
        xaxis_title="Asset Class",
        yaxis_title="Exposure (%)",
        height=300,
        showlegend=True
    )
    
    return fig

@app.callback(
    Output("active-positions-list", "children"),
    Input("risk-refresh-interval", "n_intervals")
)
def update_active_positions(n):
    """Update active positions list"""
    
    if not execution_engine:
        return html.P("No execution engine available", className="text-muted")
    
    # Get open orders as proxy for positions
    open_orders = execution_engine.get_open_orders()
    
    if not open_orders:
        return html.P("No active positions", className="text-muted")
    
    position_items = []
    for order in open_orders[:10]:  # Show top 10
        position_items.append(
            dbc.ListGroupItem([
                html.Div([
                    html.Strong(order.symbol),
                    dbc.Badge(
                        order.side.value.upper(),
                        color="success" if order.side.value == "buy" else "danger",
                        className="ms-2"
                    ),
                    html.Span(f" | {order.quantity} shares", className="text-muted ms-2")
                ])
            ])
        )
    
    return dbc.ListGroup(position_items)

@app.callback(
    [Output("drawdown-chart", "figure"),
     Output("current-drawdown", "children"),
     Output("max-drawdown", "children")],
    Input("risk-refresh-interval", "n_intervals")
)
def update_drawdown_chart(n):
    """Update drawdown chart and metrics"""
    
    # Mock drawdown data - would use actual performance data
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    equity_curve = 100000 * (1 + np.cumsum(np.random.randn(30) * 0.02))
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=drawdown,
        fill='tozeroy',
        name='Drawdown',
        line=dict(color=COLORS['danger'])
    ))
    
    fig.update_layout(
        title="Portfolio Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        height=300,
        showlegend=False
    )
    
    current_dd = f"{drawdown[-1]:.2f}%"
    max_dd = f"{drawdown.min():.2f}%"
    
    return fig, current_dd, max_dd

@app.callback(
    [Output("var-95", "children"),
     Output("var-99", "children"),
     Output("cvar-95", "children"),
     Output("var-distribution-chart", "figure")],
    Input("risk-refresh-interval", "n_intervals")
)
def update_var_analysis(n):
    """Update VaR analysis"""
    
    # Mock returns data - would use actual portfolio returns
    returns = np.random.randn(1000) * 0.02  # 2% daily volatility
    
    # Calculate VaR
    var_95 = np.percentile(returns, 5) * 100
    var_99 = np.percentile(returns, 1) * 100
    
    # Calculate CVaR (Conditional VaR)
    cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * 100
    
    # Create distribution chart
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=returns * 100,
        nbinsx=50,
        name='Return Distribution',
        marker_color=COLORS['info']
    ))
    
    # Add VaR lines
    fig.add_vline(x=var_95, line_dash="dash", line_color=COLORS['warning'], 
                  annotation_text="VaR 95%")
    fig.add_vline(x=var_99, line_dash="dash", line_color=COLORS['danger'],
                  annotation_text="VaR 99%")
    
    fig.update_layout(
        title="Returns Distribution & VaR",
        xaxis_title="Daily Return (%)",
        yaxis_title="Frequency",
        height=300,
        showlegend=False
    )
    
    return f"{var_95:.2f}%", f"{var_99:.2f}%", f"{cvar_95:.2f}%", fig

@app.callback(
    Output("risk-alerts-list", "children"),
    Input("risk-refresh-interval", "n_intervals")
)
def update_risk_alerts(n):
    """Update risk alerts"""
    
    # Mock alerts - would use actual risk monitoring
    alerts = [
        {"level": "warning", "message": "Capital utilization above 75%", "time": "2 min ago"},
        {"level": "info", "message": "Volatility spike detected in SHOP.TO", "time": "5 min ago"},
        {"level": "success", "message": "All positions within limits", "time": "10 min ago"}
    ]
    
    alert_items = []
    for alert in alerts:
        color_map = {"danger": "danger", "warning": "warning", "success": "success", "info": "info"}
        alert_items.append(
            dbc.Alert([
                html.Div([
                    html.Strong(alert["message"]),
                    html.Span(f" • {alert['time']}", className="text-muted ms-2")
                ])
            ], color=color_map.get(alert["level"], "info"), className="mb-2")
        )
    
    return html.Div(alert_items) if alert_items else html.P("No alerts", className="text-muted")

if __name__ == '__main__':
    logger.info(" Starting Risk Management Dashboard...")
    app.run_server(debug=True, port=8052)

