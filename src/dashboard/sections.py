"""Reusable dashboard sections and panels."""

from __future__ import annotations

from typing import Optional

import dash_bootstrap_components as dbc
from dash import dcc, html

from .state_manager import trading_state
import pandas as pd

from .portfolio import generate_portfolio_data, generate_holdings, generate_recent_trades
from .log_utils import get_last_log_time


def create_hybrid_control_status() -> dbc.Row:
    """Card summarising hybrid control components."""
    status_items = [
        {"icon": "fas fa-brain", "label": "Local Reasoner", "value": "Ready"},
        {"icon": "fas fa-project-diagram", "label": "Meta-Ensemble", "value": "Enabled"},
        {"icon": "fas fa-robot", "label": "AI Orchestrator", "value": "Online" if trading_state.get("ai_instance") else "Idle"},
        {"icon": "fas fa-shield-alt", "label": "Risk Engine", "value": "Active"},
    ]

    cards = [
        dbc.Col(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.I(className=f"{item['icon']} fa-2x text-info mb-2"),
                        html.H6(item["label"], className="text-muted"),
                        html.H5(item["value"], className="text-light"),
                    ]
                ),
                className="text-center shadow-sm",
            ),
            width=12,
            md=6,
            lg=3,
            className="mb-3",
        )
        for item in status_items
    ]

    return dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader(
                            [html.I(className="fas fa-layer-group me-2"), "Hybrid Control Plane Status"],
                            className="text-white",
                        ),
                        dbc.CardBody(dbc.Row(cards, className="g-3")),
                    ],
                    className="shadow-sm",
                ),
                width=12,
            )
        ],
        className="mb-4",
    )


def create_risk_panel() -> dbc.Card:
    """Panel showing high-level risk metrics."""
    portfolio = generate_portfolio_data()
    total_value = portfolio["total_value"]
    cash = portfolio["cash"]
    invested = portfolio["invested"]
    realized = portfolio["realized_pnl"]
    unrealized = portfolio["unrealized_pnl"]
    total_pnl = portfolio["total_pnl"]
    total_pct = portfolio["total_pnl_pct"]

    kill_threshold = trading_state.get("kill_switch_threshold", 0)
    kill_active = trading_state.get("kill_switch_active", False)

    return dbc.Card(
        [
            dbc.CardHeader([html.I(className="fas fa-shield-alt me-2"), "Risk Dashboard"], className="text-white"),
            dbc.CardBody(
                [
                    html.P(
                        [
                            html.Strong("Equity: "),
                            f"${total_value:,.2f} (Cash ${cash:,.2f} | Invested ${invested:,.2f})",
                        ],
                        className="mb-2",
                    ),
                    html.P(
                        [
                            html.Strong("P&L: "),
                            f"${total_pnl:,.2f} ({total_pct:.2f}%) • Realized ${realized:,.2f} • Unrealized ${unrealized:,.2f}",
                        ],
                        className="mb-2",
                    ),
                    html.P(
                        [
                            html.Strong("Kill Switch: "),
                            f"{kill_threshold:.1f}% threshold | ",
                            html.Span("ACTIVE" if kill_active else "Idle", className="text-danger" if kill_active else "text-success"),
                        ],
                        className="mb-2",
                    ),
                    html.Small("Drawdown logic now tracks total equity instead of cash-only.", className="text-muted"),
                ]
            ),
            dbc.CardFooter(
                html.Small("Risk panel refreshes with every market tick.", className="text-muted"),
            ),
        ],
        className="shadow-sm h-100",
    )


def create_learning_panel() -> dbc.Card:
    """Panel showing latest learning reflections."""
    learning_log = trading_state.get("learning_log", [])
    recent = list(reversed(learning_log[-3:])) if learning_log else []

    entries = (
        [
            html.Li(
                [
                    html.Strong(entry.get("timestamp", "")),
                    html.Span(f" – {entry.get('reflection', 'No reflection recorded')}", className="ms-2"),
                ]
            )
            for entry in recent
        ]
        if recent
        else [html.Li("No learning reflections yet. Losses will generate insights automatically.")]
    )

    return dbc.Card(
        [
            dbc.CardHeader([html.I(className="fas fa-lightbulb me-2"), "Learning & Improvements"], className="text-white"),
            dbc.CardBody(html.Ul(entries, className="mb-0")),
        ],
        className="shadow-sm h-100",
    )


def create_alerts_feed() -> dbc.Card:
    """Simple alerts feed highlighting recent notices."""
    portfolio = generate_portfolio_data()
    alerts = [
        {"message": "Market regime: {regime}".format(regime=trading_state.get("regime", "SIDEWAYS")), "tone": "info"},
        {
            "message": f"Cash available ${portfolio['cash']:,.2f}; consider redeploying idle capital.",
            "tone": "warning" if portfolio["cash"] > portfolio["total_value"] * 0.3 else "secondary",
        },
        {
            "message": f"Drawdown {portfolio['total_pnl_pct']:.2f}%",
            "tone": "danger" if portfolio["total_pnl_pct"] < -trading_state.get("kill_switch_threshold", 0) else "success",
        },
    ]

    badges = {
        "info": "fas fa-info-circle text-info",
        "warning": "fas fa-exclamation-triangle text-warning",
        "danger": "fas fa-exclamation-circle text-danger",
        "secondary": "fas fa-bell text-secondary",
        "success": "fas fa-check-circle text-success",
    }

    rows = [
        dbc.ListGroupItem(
            [
                html.I(className=f"{badges.get(alert['tone'], 'fas fa-info-circle')} me-2"),
                html.Span(alert["message"]),
            ],
            color=alert["tone"] if alert["tone"] in ("warning", "danger") else None,
        )
        for alert in alerts
    ]

    return dbc.Card(
        [
            dbc.CardHeader([html.I(className="fas fa-bell me-2"), "Alerts"], className="text-white"),
            dbc.ListGroup(rows, flush=True),
        ],
        className="shadow-sm h-100",
    )


def create_ai_activity_monitor() -> dbc.Card:
    """AI activity monitor card summarizing latest log activity."""
    activity_time = get_last_log_time("logs/ai_activity.log")
    trade_time = get_last_log_time("logs/ai_trades.log")
    signal_time = get_last_log_time("logs/ai_signals.log")
    decision_time = get_last_log_time("logs/ai_decisions.log")

    return dbc.Card(
        [
            dbc.CardHeader([html.I(className="fas fa-terminal me-2"), "AI Activity Monitor"], className="text-white"),
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Div(
                                        [
                                            html.I(className="fas fa-file-alt text-info me-2"),
                                            html.Span("Activity Log", className="text-light"),
                                        ],
                                        className="mb-2",
                                    ),
                                    html.Small(f"Last updated: {activity_time}", className="text-muted"),
                                ],
                                width=6,
                            ),
                            dbc.Col(
                                [
                                    html.Div(
                                        [
                                            html.I(className="fas fa-chart-line text-success me-2"),
                                            html.Span("Trades Log", className="text-light"),
                                        ],
                                        className="mb-2",
                                    ),
                                    html.Small(f"Last trade: {trade_time}", className="text-muted"),
                                ],
                                width=6,
                            ),
                        ],
                        className="mb-3",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Div(
                                        [
                                            html.I(className="fas fa-signal text-warning me-2"),
                                            html.Span("Signals Log", className="text-light"),
                                        ],
                                        className="mb-2",
                                    ),
                                    html.Small(f"Last signal: {signal_time}", className="text-muted"),
                                ],
                                width=6,
                            ),
                            dbc.Col(
                                [
                                    html.Div(
                                        [
                                            html.I(className="fas fa-brain text-primary me-2"),
                                            html.Span("Decisions Log", className="text-light"),
                                        ],
                                        className="mb-2",
                                    ),
                                    html.Small(f"Last decision: {decision_time}", className="text-muted"),
                                ],
                                width=6,
                            ),
                        ],
                        className="mb-3",
                    ),
                    html.Hr(className="border-secondary"),
                    dbc.ButtonGroup(
                        [
                            dbc.Button([html.I(className="fas fa-eye me-1"), "Monitor"], id="start-monitor-btn", color="success", size="sm"),
                            dbc.Button(
                                [html.I(className="fas fa-download me-1"), "Export"], id="export-logs-btn", color="info", size="sm"
                            ),
                        ]
                    ),
                    dcc.Interval(id="activity-monitor-interval", interval=5000, n_intervals=0),
                ]
            ),
        ],
        className="shadow-sm h-100",
    )


def create_ai_signals_table(signals_df: Optional[pd.DataFrame] = None) -> dbc.Card:
    """Render AI signals table based on the provided dataframe."""
    if signals_df is None or signals_df.empty:
        body = html.Div(
            [
                html.I(className="fas fa-robot fa-2x text-muted mb-3"),
                html.P("No signals available yet. AI is analyzing the market...", className="text-muted"),
            ],
            className="text-center py-4",
        )
    else:
        rows = []
        for _, row in signals_df.iterrows():
            rows.append(
                html.Tr(
                    [
                        html.Td(row["symbol"]),
                        html.Td(row["signal"]),
                        html.Td(f"{row['confidence']:.0%}", className="text-end"),
                        html.Td(f"${row['price']:.2f}", className="text-end"),
                        html.Td(f"${row['target']:.2f}", className="text-end"),
                        html.Td("; ".join(row["reason"]) if isinstance(row["reason"], (list, tuple)) else row["reason"]),
                    ]
                )
            )
        body = dbc.Table(
            [
                html.Thead(
                    html.Tr(
                        [
                            html.Th("Symbol"),
                            html.Th("Signal"),
                            html.Th("Confidence", className="text-end"),
                            html.Th("Price", className="text-end"),
                            html.Th("Target", className="text-end"),
                            html.Th("Reason"),
                        ]
                    )
                ),
                html.Tbody(rows),
            ],
            striped=True,
            hover=True,
            responsive=True,
            className="mb-0",
        )

    return dbc.Card(
        [
            dbc.CardHeader([html.I(className="fas fa-bolt me-2"), "AI Trading Signals"], className="text-white"),
            dbc.CardBody(body),
        ],
        className="shadow-sm mb-4",
    )


def create_performance_tabs() -> dbc.Card:
    """Tabbed performance view with key metrics, audit trail, and risk overview."""
    portfolio = generate_portfolio_data()
    holdings_df = generate_holdings()
    trades_df = generate_recent_trades()

    performance_content = html.Div(
        [
            html.H6("Equity Curve Snapshot", className="text-muted mb-3"),
            html.Ul(
                [
                    html.Li(f"Total Value: ${portfolio['total_value']:,.2f}"),
                    html.Li(f"Cash Available: ${portfolio['cash']:,.2f}"),
                    html.Li(f"Invested Capital: ${portfolio['invested']:,.2f}"),
                    html.Li(f"Total P&L: ${portfolio['total_pnl']:,.2f} ({portfolio['total_pnl_pct']:.2f}%)"),
                    html.Li(f"Realized P&L: ${portfolio['realized_pnl']:,.2f}"),
                    html.Li(f"Unrealized P&L: ${portfolio['unrealized_pnl']:,.2f}"),
                ],
                className="mb-0",
            ),
        ]
    )

    audit_rows = trades_df.tail(10).to_dict("records") if not trades_df.empty else []
    audit_content = (
        dbc.Table(
            [
                html.Thead(
                    html.Tr(
                        [
                            html.Th("Time"),
                            html.Th("Symbol"),
                            html.Th("Side"),
                            html.Th("Qty", className="text-end"),
                            html.Th("Price", className="text-end"),
                            html.Th("Status"),
                            html.Th("P&L", className="text-end"),
                        ]
                    )
                ),
                html.Tbody(
                    [
                        html.Tr(
                            [
                                html.Td(row["time"]),
                                html.Td(row["symbol"]),
                                html.Td(row["side"]),
                                html.Td(f"{row['qty']}", className="text-end"),
                                html.Td(f"${row['price']:.2f}", className="text-end"),
                                html.Td(row["status"]),
                                html.Td("-" if row.get("pnl") is None else f"${row['pnl']:.2f}", className="text-end"),
                            ]
                        )
                        for row in audit_rows
                    ]
                ),
            ],
            bordered=False,
            striped=True,
            hover=True,
            size="sm",
            className="mb-0",
        )
        if audit_rows
        else html.P("No trades yet today.", className="text-muted mb-0")
    )

    risk_content = html.Div(
        [
            html.P("Risk Snapshot", className="text-muted"),
            html.Ul(
                [
                    html.Li(f"Open Positions: {len(holdings_df)}"),
                    html.Li(f"Kill Switch Threshold: {trading_state.get('kill_switch_threshold', 0):.1f}%"),
                    html.Li(
                        [
                            "Kill Switch Status: ",
                            html.Span(
                                "ACTIVE" if trading_state.get("kill_switch_active") else "Idle",
                                className="text-danger" if trading_state.get("kill_switch_active") else "text-success",
                            ),
                        ]
                    ),
                ],
                className="mb-2",
            ),
            create_alerts_feed(),
        ],
        className="d-grid gap-2",
    )

    return dbc.Card(
        [
            dbc.CardHeader([html.I(className="fas fa-chart-pie me-2"), "Performance Overview"], className="text-white"),
            dbc.CardBody(
                dbc.Tabs(
                    [
                        dbc.Tab(performance_content, label="Performance"),
                        dbc.Tab(audit_content, label="Audit Log"),
                        dbc.Tab(risk_content, label="Risk"),
                    ]
                )
            ),
        ],
        className="shadow-sm",
    )


__all__ = [
    "create_hybrid_control_status",
    "create_risk_panel",
    "create_learning_panel",
    "create_alerts_feed",
    "create_ai_activity_monitor",
    "create_ai_signals_table",
    "create_performance_tabs",
]
