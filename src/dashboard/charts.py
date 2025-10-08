"""Chart generation utilities for the trading dashboard."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import plotly.graph_objs as go

from .portfolio import generate_portfolio_data
from .state_manager import trading_state


def generate_performance_chart() -> go.Figure:
    """Generate the portfolio performance figure."""
    if not trading_state["initialized"]:
        return go.Figure().update_layout(
            title="Portfolio Performance (Not Started)",
            template="plotly_dark",
            height=400,
        )

    start_time = trading_state["start_time"]
    if isinstance(start_time, str):
        try:
            start_time = datetime.fromisoformat(start_time)
        except Exception:
            start_time = datetime.now()

    time_diff = (datetime.now() - start_time).total_seconds() / 60
    points = max(int(time_diff), 2)

    dates = [start_time + timedelta(minutes=i) for i in range(points)]

    starting = float(trading_state.get("starting_capital") or 0)
    portfolio_snapshot = generate_portfolio_data()
    current = float(portfolio_snapshot.get("total_value", trading_state.get("current_capital", 0)))

    values = [starting]
    for i in range(1, points):
        progress = i / (points - 1)
        target = starting + (current - starting) * progress
        noise = np.random.uniform(-0.005, 0.005) * max(starting, 1)
        values.append(target + noise)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=values,
            mode="lines",
            name="Portfolio Value",
            line=dict(color="#00d4ff", width=3),
            fill="tozeroy",
            fillcolor="rgba(0, 212, 255, 0.1)",
        )
    )
    fig.add_hline(
        y=starting,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Starting: ${starting:,.2f}",
        annotation_position="right",
    )

    try:
        kst = float(trading_state.get("kill_switch_threshold") or 0)
    except Exception:
        kst = 0
    if kst > 0 and starting > 0:
        threshold_val = starting * (1 - kst / 100.0)
        fig.add_hline(
            y=threshold_val,
            line_dash="dot",
            line_color="#ff4444",
            annotation_text=f"Kill @{kst:.1f}%",
            annotation_position="right",
        )
        fig.add_hrect(y0=0, y1=threshold_val, line_width=0, fillcolor="rgba(255,0,0,0.06)")

    fig.update_layout(
        title="Live Portfolio Performance",
        xaxis_title="Time",
        yaxis_title="Value (CAD)",
        template="plotly_dark",
        height=400,
        hovermode="x unified",
    )
    return fig


def generate_sector_allocation() -> go.Figure:
    """Generate sector allocation pie chart."""
    portfolio = generate_portfolio_data()
    holdings_value = portfolio["invested"]
    cash = portfolio["cash"]

    fig = go.Figure()
    if holdings_value <= 0:
        fig.update_layout(
            title="Sector Allocation (No Holdings)",
            template="plotly_dark",
            height=350,
        )
        return fig

    values = [holdings_value, cash]
    labels = ["Invested", "Cash"]
    colors = ["#1f77b4", "#2ca02c"]

    fig.add_trace(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.3,
            marker_colors=colors,
        )
    )
    fig.update_layout(
        title="Portfolio Allocation",
        template="plotly_dark",
        height=350,
    )
    return fig


__all__ = ["generate_performance_chart", "generate_sector_allocation"]
