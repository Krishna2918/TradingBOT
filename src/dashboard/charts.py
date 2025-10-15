"""Chart generation utilities for the trading dashboard."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
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


# Phase 8: Enhanced Dashboard Charts

def generate_api_budget_chart(api_data: pd.DataFrame) -> go.Figure:
    """Generate API budget and rate-limit status chart."""
    if api_data.empty:
        fig = go.Figure()
        fig.update_layout(
            title="API Budget Status (No Data)",
            template="plotly_dark",
            height=400,
        )
        return fig
    
    fig = go.Figure()
    
    # Add budget usage bars
    fig.add_trace(go.Bar(
        x=api_data['api_name'],
        y=api_data['requests_made'],
        name='Requests Made',
        marker_color='lightblue',
        yaxis='y'
    ))
    
    # Add rate limit hits
    fig.add_trace(go.Bar(
        x=api_data['api_name'],
        y=api_data['rate_limit_hits'],
        name='Rate Limit Hits',
        marker_color='red',
        yaxis='y'
    ))
    
    fig.update_layout(
        title="API Budget Status",
        xaxis_title="API Name",
        yaxis_title="Request Count",
        template="plotly_dark",
        height=400,
        barmode='group'
    )
    
    return fig

def generate_phase_duration_timeline(phase_data: pd.DataFrame) -> go.Figure:
    """Generate phase duration timeline chart."""
    if phase_data.empty:
        fig = go.Figure()
        fig.update_layout(
            title="Phase Duration Timeline (No Data)",
            template="plotly_dark",
            height=400,
        )
        return fig
    
    # Convert timestamp to datetime
    phase_data['timestamp'] = pd.to_datetime(phase_data['timestamp'])
    
    fig = go.Figure()
    
    # Group by phase and create traces
    for phase in phase_data['phase_name'].unique():
        phase_subset = phase_data[phase_data['phase_name'] == phase]
        fig.add_trace(go.Scatter(
            x=phase_subset['timestamp'],
            y=phase_subset['duration_ms'],
            mode='markers+lines',
            name=phase,
            marker=dict(size=8),
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title="Phase Duration Timeline",
        xaxis_title="Time",
        yaxis_title="Duration (ms)",
        template="plotly_dark",
        height=400,
        hovermode='x unified'
    )
    
    return fig

def generate_confidence_calibration_chart(confidence_data: pd.DataFrame) -> go.Figure:
    """Generate calibrated vs raw confidence comparison chart."""
    if confidence_data.empty:
        fig = go.Figure()
        fig.update_layout(
            title="Confidence Calibration (No Data)",
            template="plotly_dark",
            height=400,
        )
        return fig
    
    fig = go.Figure()
    
    # Add raw confidence
    fig.add_trace(go.Scatter(
        x=confidence_data['raw_confidence'],
        y=confidence_data['calibrated_confidence'],
        mode='markers',
        name='Calibrated vs Raw',
        marker=dict(
            size=8,
            color=confidence_data['outcome'],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Outcome")
        ),
        text=confidence_data['symbol'],
        hovertemplate='<b>%{text}</b><br>' +
                      'Raw: %{x:.3f}<br>' +
                      'Calibrated: %{y:.3f}<br>' +
                      '<extra></extra>'
    ))
    
    # Add perfect calibration line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Perfect Calibration',
        line=dict(dash='dash', color='gray'),
        showlegend=True
    ))
    
    fig.update_layout(
        title="Confidence Calibration: Raw vs Calibrated",
        xaxis_title="Raw Confidence",
        yaxis_title="Calibrated Confidence",
        template="plotly_dark",
        height=400
    )
    
    return fig

def generate_ensemble_weights_chart(weights_data: pd.DataFrame) -> go.Figure:
    """Generate ensemble weights history chart."""
    if weights_data.empty:
        fig = go.Figure()
        fig.update_layout(
            title="Ensemble Weights History (No Data)",
            template="plotly_dark",
            height=400,
        )
        return fig
    
    # Convert timestamp to datetime
    weights_data['timestamp'] = pd.to_datetime(weights_data['timestamp'])
    
    fig = go.Figure()
    
    # Group by model and create traces
    for model in weights_data['model'].unique():
        model_subset = weights_data[weights_data['model'] == model]
        fig.add_trace(go.Scatter(
            x=model_subset['timestamp'],
            y=model_subset['weight'],
            mode='lines+markers',
            name=model,
            line=dict(width=3),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title="Ensemble Weights History (7 Days)",
        xaxis_title="Time",
        yaxis_title="Weight",
        template="plotly_dark",
        height=400,
        hovermode='x unified'
    )
    
    return fig

def generate_drawdown_regime_chart(regime_data: pd.DataFrame) -> go.Figure:
    """Generate drawdown and regime state chart."""
    if regime_data.empty:
        fig = go.Figure()
        fig.update_layout(
            title="Drawdown & Regime State (No Data)",
            template="plotly_dark",
            height=400,
        )
        return fig
    
    # Convert timestamp to datetime
    regime_data['timestamp'] = pd.to_datetime(regime_data['timestamp'])
    
    fig = go.Figure()
    
    # Add portfolio value
    fig.add_trace(go.Scatter(
        x=regime_data['timestamp'],
        y=regime_data['portfolio_value'],
        mode='lines',
        name='Portfolio Value',
        yaxis='y',
        line=dict(color='blue', width=2)
    ))
    
    # Add daily P&L
    fig.add_trace(go.Scatter(
        x=regime_data['timestamp'],
        y=regime_data['daily_pnl'],
        mode='lines+markers',
        name='Daily P&L',
        yaxis='y2',
        line=dict(color='green', width=2),
        marker=dict(size=4)
    ))
    
    # Add regime state as background color
    if 'regime' in regime_data.columns:
        regime_colors = {
            'TRENDING_LOW_VOL': 'rgba(0, 255, 0, 0.1)',
            'TRENDING_HIGH_VOL': 'rgba(255, 165, 0, 0.1)',
            'CHOPPY_LOW_VOL': 'rgba(0, 0, 255, 0.1)',
            'CHOPPY_HIGH_VOL': 'rgba(255, 0, 0, 0.1)',
            'TRANSITION': 'rgba(128, 128, 128, 0.1)'
        }
        
        for regime, color in regime_colors.items():
            regime_subset = regime_data[regime_data['regime'] == regime]
            if not regime_subset.empty:
                fig.add_trace(go.Scatter(
                    x=regime_subset['timestamp'],
                    y=regime_subset['portfolio_value'],
                    mode='markers',
                    name=f'Regime: {regime}',
                    marker=dict(color=color, size=10, symbol='square'),
                    showlegend=True
                ))
    
    fig.update_layout(
        title="Portfolio Performance & Market Regime",
        xaxis_title="Time",
        yaxis=dict(title="Portfolio Value", side="left"),
        yaxis2=dict(title="Daily P&L", side="right", overlaying="y"),
        template="plotly_dark",
        height=400,
        hovermode='x unified'
    )
    
    return fig

def generate_system_health_chart(health_data: Dict) -> go.Figure:
    """Generate system health metrics chart."""
    fig = go.Figure()
    
    # API Status
    if health_data.get('api_status'):
        api_df = pd.DataFrame(health_data['api_status'])
        fig.add_trace(go.Bar(
            x=api_df['api_name'],
            y=api_df['response_time_ms'],
            name='API Response Time (ms)',
            marker_color='lightblue',
            yaxis='y'
        ))
    
    # Phase Performance
    if health_data.get('phase_performance'):
        phase_df = pd.DataFrame(health_data['phase_performance'])
        fig.add_trace(go.Bar(
            x=phase_df['phase_name'],
            y=phase_df['avg_duration_ms'],
            name='Avg Phase Duration (ms)',
            marker_color='orange',
            yaxis='y2'
        ))
    
    fig.update_layout(
        title="System Health Metrics",
        xaxis_title="Component",
        yaxis=dict(title="Response Time (ms)", side="left"),
        yaxis2=dict(title="Duration (ms)", side="right", overlaying="y"),
        template="plotly_dark",
        height=400,
        barmode='group'
    )
    
    return fig

def generate_rationale_trace_chart(rationale_data: List[Dict]) -> go.Figure:
    """Generate model rationale trace chart."""
    if not rationale_data:
        fig = go.Figure()
        fig.update_layout(
            title="Model Rationale Trace (No Data)",
            template="plotly_dark",
            height=400,
        )
        return fig
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(rationale_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    fig = go.Figure()
    
    # Group by model and create traces
    for model in df['model_name'].unique():
        model_subset = df[df['model_name'] == model]
        fig.add_trace(go.Scatter(
            x=model_subset['timestamp'],
            y=model_subset['score'],
            mode='markers+lines',
            name=model,
            marker=dict(size=8),
            line=dict(width=2),
            text=model_subset['symbol'],
            hovertemplate='<b>%{text}</b><br>' +
                          'Model: ' + model + '<br>' +
                          'Score: %{y:.3f}<br>' +
                          'Time: %{x}<br>' +
                          '<extra></extra>'
        ))
    
    fig.update_layout(
        title="Model Rationale Trace (Last 7 Days)",
        xaxis_title="Time",
        yaxis_title="Score",
        template="plotly_dark",
        height=400,
        hovermode='x unified'
    )
    
    return fig


__all__ = [
    "generate_performance_chart", 
    "generate_sector_allocation",
    "generate_api_budget_chart",
    "generate_phase_duration_timeline",
    "generate_confidence_calibration_chart",
    "generate_ensemble_weights_chart",
    "generate_drawdown_regime_chart",
    "generate_system_health_chart",
    "generate_rationale_trace_chart"
]
