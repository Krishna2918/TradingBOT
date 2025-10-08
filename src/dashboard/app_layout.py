"""Dashboard layout definitions."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

from .state_manager import trading_state


def create_navbar():
    """Create navigation bar with mode switcher."""
    mode = trading_state.get("mode", "demo")
    mode_label = "LIVE MODE" if mode == "live" else "DEMO MODE"
    return dbc.NavbarSimple(
        brand="AI Trading Dashboard",
        brand_href="#",
        color="dark",
        dark=True,
        children=[
            dbc.Badge(mode_label, color="success" if mode == "demo" else "danger", className="ms-2"),
        ],
    )

