"""Dashboard UI components."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html

from .state_manager import trading_state


def create_navbar() -> dbc.Navbar:
    """Create the compact navbar used on the startup screen."""
    mode = trading_state.get("mode", "demo")
    mode_label = "LIVE MODE" if mode == "live" else "DEMO MODE"
    mode_color = "danger" if mode == "live" else "success"

    return dbc.Navbar(
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            html.A(
                                dbc.Row(
                                    [
                                        dbc.Col(html.I(className="fas fa-robot fa-2x")),
                                        dbc.Col(dbc.NavbarBrand(f"AI Trading Bot - {mode_label}", className="ms-2")),
                                    ],
                                    align="center",
                                    className="g-0",
                                ),
                                style={"textDecoration": "none"},
                            ),
                            width="auto",
                        ),
                        dbc.Col(
                            dbc.Nav(
                                [
                                    dbc.NavItem(dbc.NavLink("📊 Overview", href="/", active=True)),
                                    dbc.NavItem(dbc.NavLink("🤖 AI Signals", href="/signals")),
                                    dbc.NavItem(
                                        html.A(
                                            "📋 View Logs",
                                            href="/logs",
                                            target="_blank",
                                            className="nav-link",
                                            style={"color": "#17a2b8"},
                                        )
                                    ),
                                    dbc.NavItem(dbc.NavLink("⚙️ Settings", href="/settings")),
                                ],
                                navbar=True,
                            ),
                            width="auto",
                            className="mx-auto",
                        ),
                        dbc.Col(
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Span("Demo", style={"marginRight": "10px", "fontSize": "14px"}),
                                            dbc.Switch(id="startup-mode-switch", value=(mode == "live"), className="d-inline-block"),
                                            html.Span(
                                                "Live",
                                                style={
                                                    "marginLeft": "10px",
                                                    "fontSize": "14px",
                                                    "fontWeight": "bold",
                                                    "color": "#ff4444",
                                                },
                                            ),
                                        ],
                                        style={"display": "flex", "alignItems": "center"},
                                    ),
                                    dbc.Badge(
                                        f"${trading_state['current_capital']:,.2f}",
                                        color=mode_color,
                                        className="ms-3",
                                        style={"fontSize": "1rem"},
                                    ),
                                ],
                                style={"display": "flex", "alignItems": "center"},
                            ),
                            width="auto",
                        ),
                    ],
                    className="w-100 align-items-center justify-content-between",
                )
            ],
            fluid=True,
        ),
        color="dark",
        dark=True,
        className="mb-4",
        sticky="top",
    )


def create_regime_badge() -> dbc.Badge:
    """Create market regime badge."""
    current_regime = trading_state.get("regime", "SIDEWAYS") or "SIDEWAYS"
    regime_colors = {
        "BULL": "success",
        "BEAR": "danger",
        "SIDEWAYS": "warning",
        "LOW_VOL": "secondary",
        "HIGH_VOL": "danger",
        "TRENDING": "info",
        "VOLATILE": "info",
    }
    regime_icons = {
        "BULL": "fas fa-arrow-up",
        "BEAR": "fas fa-arrow-down",
        "SIDEWAYS": "fas fa-arrows-alt-h",
        "LOW_VOL": "fas fa-wave-square",
        "HIGH_VOL": "fas fa-bolt",
        "TRENDING": "fas fa-chart-line",
        "VOLATILE": "fas fa-chart-line",
    }
    return dbc.Badge(
        [
            html.I(className=regime_icons.get(current_regime, "fas fa-arrows-alt-h"), style={"marginRight": "5px"}),
            current_regime,
        ],
        color=regime_colors.get(current_regime, "secondary"),
        className="fs-6 px-3 py-2",
    )


def create_status_pill() -> dbc.Badge:
    """Render the live/demo status pill for navbars."""
    mode = trading_state.get("mode", "demo")
    if mode != "live":
        return dbc.Badge("DEMO", color="success", pill=True)

    status = {"authenticated": False, "practice_mode": True, "allow_trading": False}
    try:
        br = trading_state.get("broker")
        if br and hasattr(br, "get_auth_status"):
            status = br.get_auth_status()
    except Exception:
        pass

    parts = ["LIVE"]
    if status.get("practice_mode"):
        parts.append("PAPER")
    parts.append("AUTH" if status.get("authenticated") else "NOAUTH")
    label = " • ".join(parts)
    color = "danger" if status.get("authenticated") else "secondary"
    return dbc.Badge(label, color=color, pill=True)


def create_enhanced_navbar() -> dbc.Navbar:
    """Navbar used on primary trading/log pages."""
    mode = trading_state.get("mode", "demo")
    mode_label = "LIVE MODE" if mode == "live" else "DEMO MODE"

    return dbc.Navbar(
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Button(
                                    [html.I(className="fas fa-arrow-left me-2"), "Back to Dashboard"],
                                    id="back-to-dashboard-btn",
                                    color="secondary",
                                    size="sm",
                                    className="me-3",
                                    style={"display": "none"},
                                ),
                                html.A(
                                    dbc.Row(
                                        [
                                            dbc.Col(html.I(className="fas fa-robot fa-2x")),
                                            dbc.Col(dbc.NavbarBrand(f"AI Trading Bot - {mode_label}", className="ms-2")),
                                        ],
                                        align="center",
                                        className="g-0",
                                    ),
                                    style={"textDecoration": "none"},
                                ),
                            ],
                            width="auto",
                        ),
                        dbc.Col(html.Span(id="regime-badge", children=create_regime_badge()), width="auto", className="ms-auto me-3"),
                        dbc.Col(
                            [
                                html.A(
                                    dbc.Button(
                                        [html.I(className="fas fa-file-alt me-2"), "View Logs"],
                                        color="info",
                                        size="sm",
                                        className="me-2",
                                    ),
                                    href="/logs",
                                    target="_blank",
                                    style={"textDecoration": "none"},
                                ),
                                dbc.Button("Reset", id="reset-to-startup-btn", color="warning", size="sm", className="me-2"),
                                create_status_pill(),
                            ],
                            width="auto",
                            className="d-flex align-items-center",
                        ),
                    ],
                    className="w-100 align-items-center justify-content-between",
                )
            ],
            fluid=True,
        ),
        color="dark",
        dark=True,
        className="mb-4",
        sticky="top",
    )


__all__ = ["create_navbar", "create_enhanced_navbar", "create_status_pill", "create_regime_badge"]
