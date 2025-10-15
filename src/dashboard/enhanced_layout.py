"""
Enhanced Dashboard Layout - Phase 8
Integrates new panels for API budgets, phase durations, confidence calibration,
ensemble weights, drawdown/regime state, and model rationale traces.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html
from typing import Dict, Any

from .state_manager import trading_state
from .connector import DashboardConnector
from .charts import (
    generate_api_budget_chart,
    generate_phase_duration_timeline,
    generate_confidence_calibration_chart,
    generate_ensemble_weights_chart,
    generate_drawdown_regime_chart,
    generate_system_health_chart,
    generate_rationale_trace_chart
)


def create_enhanced_navbar():
    """Create enhanced navigation bar with mode switcher and system status."""
    mode = trading_state.get("mode", "demo")
    mode_label = "LIVE MODE" if mode == "live" else "DEMO MODE"
    
    return dbc.NavbarSimple(
        brand="AI Trading Dashboard - Enhanced",
        brand_href="#",
        color="dark",
        dark=True,
        children=[
            dbc.Badge(mode_label, color="success" if mode == "demo" else "danger", className="ms-2"),
            dbc.Badge("Phase 8 Enhanced", color="info", className="ms-1"),
        ],
    )


def create_api_budget_panel() -> dbc.Card:
    """Create API budget and rate-limit status panel."""
    connector = DashboardConnector()
    api_data = connector.api_budget_status()
    chart = generate_api_budget_chart(api_data)
    
    return dbc.Card([
        dbc.CardHeader([
            html.H5("API Budget Status", className="mb-0"),
            dbc.Badge("Real-time", color="success", className="ms-2")
        ]),
        dbc.CardBody([
            dcc.Graph(figure=chart, id="api-budget-chart"),
            html.Small("Shows API usage, rate limits, and budget status", className="text-muted")
        ])
    ], className="mb-4")


def create_phase_duration_panel() -> dbc.Card:
    """Create phase duration timeline panel."""
    connector = DashboardConnector()
    phase_data = connector.phase_duration_timeline()
    chart = generate_phase_duration_timeline(phase_data)
    
    return dbc.Card([
        dbc.CardHeader([
            html.H5("Phase Duration Timeline", className="mb-0"),
            dbc.Badge("Performance", color="info", className="ms-2")
        ]),
        dbc.CardBody([
            dcc.Graph(figure=chart, id="phase-duration-chart"),
            html.Small("Tracks execution time for each trading phase", className="text-muted")
        ])
    ], className="mb-4")


def create_confidence_calibration_panel() -> dbc.Card:
    """Create confidence calibration panel."""
    connector = DashboardConnector()
    confidence_data = connector.confidence_calibration_data()
    chart = generate_confidence_calibration_chart(confidence_data)
    
    return dbc.Card([
        dbc.CardHeader([
            html.H5("Confidence Calibration", className="mb-0"),
            dbc.Badge("AI Quality", color="warning", className="ms-2")
        ]),
        dbc.CardBody([
            dcc.Graph(figure=chart, id="confidence-calibration-chart"),
            html.Small("Raw vs calibrated confidence for recent trades", className="text-muted")
        ])
    ], className="mb-4")


def create_ensemble_weights_panel() -> dbc.Card:
    """Create ensemble weights history panel."""
    connector = DashboardConnector()
    weights_data = connector.ensemble_weights_history()
    chart = generate_ensemble_weights_chart(weights_data)
    
    return dbc.Card([
        dbc.CardHeader([
            html.H5("Ensemble Weights History", className="mb-0"),
            dbc.Badge("Adaptive", color="primary", className="ms-2")
        ]),
        dbc.CardBody([
            dcc.Graph(figure=chart, id="ensemble-weights-chart"),
            html.Small("Model weight evolution over the last 7 days", className="text-muted")
        ])
    ], className="mb-4")


def create_drawdown_regime_panel() -> dbc.Card:
    """Create drawdown and regime state panel."""
    connector = DashboardConnector()
    regime_data = connector.drawdown_and_regime_data()
    chart = generate_drawdown_regime_chart(regime_data)
    
    return dbc.Card([
        dbc.CardHeader([
            html.H5("Portfolio Performance & Market Regime", className="mb-0"),
            dbc.Badge("Risk Management", color="danger", className="ms-2")
        ]),
        dbc.CardBody([
            dcc.Graph(figure=chart, id="drawdown-regime-chart"),
            html.Small("Portfolio performance with market regime overlay", className="text-muted")
        ])
    ], className="mb-4")


def create_system_health_panel() -> dbc.Card:
    """Create system health metrics panel."""
    connector = DashboardConnector()
    health_data = connector.system_health_metrics()
    chart = generate_system_health_chart(health_data)
    
    return dbc.Card([
        dbc.CardHeader([
            html.H5("System Health Metrics", className="mb-0"),
            dbc.Badge("Monitoring", color="secondary", className="ms-2")
        ]),
        dbc.CardBody([
            dcc.Graph(figure=chart, id="system-health-chart"),
            html.Small("API response times and phase performance", className="text-muted")
        ])
    ], className="mb-4")


def create_rationale_trace_panel() -> dbc.Card:
    """Create model rationale trace panel."""
    connector = DashboardConnector()
    rationale_data = connector.model_rationale_trace()
    chart = generate_rationale_trace_chart(rationale_data)
    
    return dbc.Card([
        dbc.CardHeader([
            html.H5("Model Rationale Trace", className="mb-0"),
            dbc.Badge("Explainable AI", color="success", className="ms-2")
        ]),
        dbc.CardBody([
            dcc.Graph(figure=chart, id="rationale-trace-chart"),
            html.Small("Detailed model explanations and decision traces", className="text-muted")
        ])
    ], className="mb-4")


def create_enhanced_dashboard_layout() -> html.Div:
    """Create the enhanced dashboard layout with all new panels."""
    
    # Create the main layout
    layout = html.Div([
        # Navigation
        create_enhanced_navbar(),
        
        # Main content
        dbc.Container([
            # Page header
            dbc.Row([
                dbc.Col([
                    html.H2("Enhanced Trading Dashboard", className="mb-3"),
                    html.P("Real-time monitoring with advanced analytics and regime awareness", 
                           className="text-muted mb-4")
                ])
            ]),
            
            # System Health Row
            dbc.Row([
                dbc.Col([
                    create_system_health_panel()
                ], width=12)
            ]),
            
            # API and Performance Row
            dbc.Row([
                dbc.Col([
                    create_api_budget_panel()
                ], width=6),
                dbc.Col([
                    create_phase_duration_panel()
                ], width=6)
            ]),
            
            # AI Quality Row
            dbc.Row([
                dbc.Col([
                    create_confidence_calibration_panel()
                ], width=6),
                dbc.Col([
                    create_ensemble_weights_panel()
                ], width=6)
            ]),
            
            # Risk Management Row
            dbc.Row([
                dbc.Col([
                    create_drawdown_regime_panel()
                ], width=12)
            ]),
            
            # Explainable AI Row
            dbc.Row([
                dbc.Col([
                    create_rationale_trace_panel()
                ], width=12)
            ]),
            
            # Auto-refresh interval
            dcc.Interval(
                id='interval-component',
                interval=30*1000,  # Update every 30 seconds
                n_intervals=0
            )
            
        ], fluid=True, className="py-4")
    ])
    
    return layout


def create_legacy_compatibility_layout() -> html.Div:
    """Create a layout that maintains compatibility with existing dashboard while adding new panels."""
    
    # Import existing layout components
    from .app_layout import create_navbar
    
    layout = html.Div([
        # Use existing navbar
        create_navbar(),
        
        # Main content with both legacy and new panels
        dbc.Container([
            # Legacy panels (if they exist)
            dbc.Row([
                dbc.Col([
                    html.H3("Legacy Dashboard", className="mb-3"),
                    html.P("Existing dashboard components", className="text-muted")
                ])
            ], className="mb-4"),
            
            # New enhanced panels
            dbc.Row([
                dbc.Col([
                    html.H3("Enhanced Analytics", className="mb-3"),
                    html.P("New Phase 8 dashboard enhancements", className="text-muted")
                ])
            ], className="mb-4"),
            
            # System Health Row
            dbc.Row([
                dbc.Col([
                    create_system_health_panel()
                ], width=12)
            ]),
            
            # API and Performance Row
            dbc.Row([
                dbc.Col([
                    create_api_budget_panel()
                ], width=6),
                dbc.Col([
                    create_phase_duration_panel()
                ], width=6)
            ]),
            
            # AI Quality Row
            dbc.Row([
                dbc.Col([
                    create_confidence_calibration_panel()
                ], width=6),
                dbc.Col([
                    create_ensemble_weights_panel()
                ], width=6)
            ]),
            
            # Risk Management Row
            dbc.Row([
                dbc.Col([
                    create_drawdown_regime_panel()
                ], width=12)
            ]),
            
            # Explainable AI Row
            dbc.Row([
                dbc.Col([
                    create_rationale_trace_panel()
                ], width=12)
            ]),
            
            # Auto-refresh interval
            dcc.Interval(
                id='interval-component',
                interval=30*1000,  # Update every 30 seconds
                n_intervals=0
            )
            
        ], fluid=True, className="py-4")
    ])
    
    return layout


def create_minimal_enhanced_layout() -> html.Div:
    """Create a minimal enhanced layout with just the new panels."""
    
    layout = html.Div([
        # Simple header
        dbc.NavbarSimple(
            brand="Enhanced Trading Dashboard",
            color="primary",
            dark=True,
        ),
        
        # Main content
        dbc.Container([
            # Page header
            dbc.Row([
                dbc.Col([
                    html.H2("Phase 8 Dashboard Enhancements", className="mb-3"),
                    html.P("Advanced analytics and monitoring capabilities", className="text-muted mb-4")
                ])
            ]),
            
            # Compact grid layout
            dbc.Row([
                dbc.Col([
                    create_system_health_panel()
                ], width=12, className="mb-3")
            ]),
            
            dbc.Row([
                dbc.Col([
                    create_api_budget_panel()
                ], width=6, className="mb-3"),
                dbc.Col([
                    create_phase_duration_panel()
                ], width=6, className="mb-3")
            ]),
            
            dbc.Row([
                dbc.Col([
                    create_confidence_calibration_panel()
                ], width=6, className="mb-3"),
                dbc.Col([
                    create_ensemble_weights_panel()
                ], width=6, className="mb-3")
            ]),
            
            dbc.Row([
                dbc.Col([
                    create_drawdown_regime_panel()
                ], width=12, className="mb-3")
            ]),
            
            dbc.Row([
                dbc.Col([
                    create_rationale_trace_panel()
                ], width=12, className="mb-3")
            ]),
            
            # Auto-refresh interval
            dcc.Interval(
                id='interval-component',
                interval=30*1000,  # Update every 30 seconds
                n_intervals=0
            )
            
        ], fluid=True, className="py-4")
    ])
    
    return layout


# Export the main layout function
def get_enhanced_layout() -> html.Div:
    """Get the enhanced dashboard layout."""
    return create_enhanced_dashboard_layout()


__all__ = [
    "create_enhanced_dashboard_layout",
    "create_legacy_compatibility_layout", 
    "create_minimal_enhanced_layout",
    "get_enhanced_layout",
    "create_api_budget_panel",
    "create_phase_duration_panel",
    "create_confidence_calibration_panel",
    "create_ensemble_weights_panel",
    "create_drawdown_regime_panel",
    "create_system_health_panel",
    "create_rationale_trace_panel"
]
