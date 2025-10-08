"""Portfolio data computations and layout components."""

from __future__ import annotations

import pandas as pd
import dash_bootstrap_components as dbc
from dash import html

from .state_manager import trading_state
from .ui_components import create_status_pill


def _normalize_holding_entry(holding: dict) -> dict:
    """Return a holding dict with derived P&L metrics filled in."""
    symbol = holding.get("symbol")
    qty_raw = float(holding.get("qty") or 0.0)
    qty = max(0, int(round(qty_raw)))
    avg_price = float(holding.get("avg_price") or 0.0)
    current_price = float(holding.get("current_price") or (avg_price if avg_price else 0.0))
    pnl = holding.get("pnl")
    if pnl is None:
        pnl = (current_price - avg_price) * qty if qty and avg_price else 0.0
    pnl_pct = holding.get("pnl_pct")
    if pnl_pct is None:
        pnl_pct = ((current_price - avg_price) / avg_price * 100) if avg_price else 0.0
    return {
        "symbol": symbol,
        "name": holding.get("name") or (symbol or "").replace(".TO", ""),
        "qty": qty,
        "avg_price": avg_price,
        "current_price": current_price,
        "pnl": float(pnl),
        "pnl_pct": float(pnl_pct),
    }


def generate_portfolio_data() -> dict:
    """Generate portfolio summary metrics based on current state."""
    if not trading_state["initialized"]:
        return {
            "total_value": 0,
            "cash": 0,
            "invested": 0,
            "daily_pnl": 0,
            "daily_pnl_pct": 0,
            "total_pnl": 0,
            "total_pnl_pct": 0,
            "win_rate": 0,
            "total_trades": 0,
            "winning_trades": 0,
            "realized_pnl": 0,
            "unrealized_pnl": 0,
        }

    cash = trading_state["current_capital"]
    normalized_holdings = [_normalize_holding_entry(h) for h in trading_state["holdings"]]
    holdings_value = sum(h["current_price"] * h["qty"] for h in normalized_holdings)
    total_value = cash + holdings_value

    trades = trading_state["trades"]
    realized_pnl = sum(t.get("pnl", 0) for t in trades if t.get("pnl") is not None)
    holdings_cost = sum(h["avg_price"] * h["qty"] for h in normalized_holdings)
    contributed_capital = holdings_cost + cash - realized_pnl

    if contributed_capital <= 0:
        fallback_start = trading_state.get("starting_capital") or total_value
        contributed_capital = fallback_start if fallback_start > 0 else total_value

    total_pnl = total_value - contributed_capital
    total_pnl_pct = (total_pnl / contributed_capital * 100) if contributed_capital > 0 else 0

    unrealized_pnl = sum(h["pnl"] for h in normalized_holdings)

    trades_with_pnl = [t for t in trades if t.get("pnl") is not None]
    winning = sum(1 for t in trades_with_pnl if t["pnl"] > 0)
    win_rate = (winning / len(trades_with_pnl) * 100) if trades_with_pnl else 0

    daily_pnl = total_pnl
    daily_pnl_pct = total_pnl_pct

    return {
        "total_value": total_value,
        "cash": cash,
        "invested": holdings_value,
        "daily_pnl": daily_pnl,
        "daily_pnl_pct": daily_pnl_pct,
        "total_pnl": total_pnl,
        "total_pnl_pct": total_pnl_pct,
        "win_rate": win_rate,
        "total_trades": len(trades),
        "winning_trades": winning,
        "realized_pnl": realized_pnl,
        "unrealized_pnl": unrealized_pnl,
    }


def generate_holdings() -> pd.DataFrame:
    """Generate holdings dataframe based on current positions."""
    if not trading_state["initialized"] or not trading_state["holdings"]:
        return pd.DataFrame(columns=["symbol", "name", "qty", "avg_price", "current_price", "pnl", "pnl_pct"])
    normalized = [_normalize_holding_entry(h) for h in trading_state["holdings"]]
    return pd.DataFrame(normalized)


def generate_recent_trades() -> pd.DataFrame:
    """Get recent trades dataframe."""
    if not trading_state["initialized"] or not trading_state["trades"]:
        return pd.DataFrame(columns=["time", "symbol", "side", "qty", "price", "status", "pnl"])
    return pd.DataFrame(trading_state["trades"][-20:])


def create_summary_cards():
    """Render the summary cards row."""
    portfolio = generate_portfolio_data()

    total_pnl_value = portfolio["total_pnl"]
    total_pnl_color = "success" if total_pnl_value >= 0 else "danger"
    total_direction = "▲" if total_pnl_value >= 0 else "▼"
    total_amount_str = f"${total_pnl_value:,.2f}"

    daily_pnl_value = portfolio["daily_pnl"]
    pnl_color = "success" if daily_pnl_value >= 0 else "danger"
    daily_amount_str = f"${daily_pnl_value:,.2f}"
    daily_prefix = "+" if daily_pnl_value >= 0 else ""

    cash_value = portfolio["cash"]
    invested_value = portfolio["invested"]

    win_rate = portfolio["win_rate"]
    winning_trades = portfolio["winning_trades"]
    total_trades = portfolio["total_trades"]

    cards = [
        dbc.Col(
            dbc.Card(
                dbc.CardBody(
                    html.Div(
                        [
                            html.I(className="fas fa-wallet fa-2x text-info mb-2"),
                            html.H6("Portfolio Value", className="text-muted"),
                            html.H3(f"${portfolio['total_value']:,.2f}", className="mb-1"),
                            html.P(
                                [
                                    html.Span(f"{total_amount_str} ", className=f"text-{total_pnl_color}"),
                                    html.Span(total_direction, className=f"text-{total_pnl_color}"),
                                    html.Small(f" ({portfolio['total_pnl_pct']:.2f}%)", className=f"text-{total_pnl_color}"),
                                ],
                                className="mb-0",
                            ),
                        ]
                    )
                ),
                className="h-100 shadow-sm",
            ),
            width=12,
            md=6,
            lg=3,
        ),
        dbc.Col(
            dbc.Card(
                dbc.CardBody(
                    html.Div(
                        [
                            html.I(className="fas fa-piggy-bank fa-2x text-primary mb-2"),
                            html.H6("Cash Available", className="text-muted"),
                            html.H3(f"${cash_value:,.2f}", className="mb-1 text-info"),
                            html.P(
                                [
                                    html.Small(f"Invested: ${invested_value:,.2f}", className="text-muted"),
                                ],
                                className="mb-0",
                            ),
                        ]
                    )
                ),
                className="h-100 shadow-sm",
            ),
            width=12,
            md=6,
            lg=3,
        ),
        dbc.Col(
            dbc.Card(
                dbc.CardBody(
                    html.Div(
                        [
                            html.I(className=f"fas fa-chart-line fa-2x text-{pnl_color} mb-2"),
                            html.H6("Today's P&L", className="text-muted"),
                            html.H3(daily_amount_str, className=f"mb-1 text-{pnl_color}"),
                            html.P(
                                [
                                    html.Small(f"{daily_prefix}{portfolio['daily_pnl_pct']:.2f}%", className=f"text-{pnl_color}"),
                                ],
                                className="mb-0",
                            ),
                        ]
                    )
                ),
                className="h-100 shadow-sm",
            ),
            width=12,
            md=6,
            lg=3,
        ),
        dbc.Col(
            dbc.Card(
                dbc.CardBody(
                    html.Div(
                        [
                            html.I(className="fas fa-trophy fa-2x text-warning mb-2"),
                            html.H6("Win Rate", className="text-muted"),
                            html.H3(f"{win_rate:.1f}%", className="mb-1"),
                            html.P(
                                [
                                    html.Small(f"{winning_trades}/{total_trades} trades", className="text-muted"),
                                ],
                                className="mb-0",
                            ),
                        ]
                    )
                ),
                className="h-100 shadow-sm",
            ),
            width=12,
            md=6,
            lg=3,
        ),
        dbc.Col(
            dbc.Card(
                dbc.CardBody(
                    html.Div(
                        [
                            html.I(className="fas fa-robot fa-2x text-primary mb-2"),
                            html.H6("AI Status", className="text-muted"),
                            create_status_pill(),
                        ]
                    )
                ),
                className="h-100 shadow-sm",
            ),
            width=12,
            md=6,
            lg=3,
        ),
    ]

    return dbc.Row(cards, className="mb-4 g-3")


def create_holdings_table():
    """Render holdings table component."""
    df = generate_holdings()

    if df.empty:
        return dbc.Card(
            [
                dbc.CardHeader([html.I(className="fas fa-briefcase me-2"), "Current Holdings"]),
                dbc.CardBody(
                    html.Div(
                        [
                            html.I(className="fas fa-briefcase fa-2x text-muted mb-3"),
                            html.P("No positions yet. Waiting for AI to identify trading opportunities.", className="text-muted"),
                            html.Small(
                                "AI monitors the market 24/7 and trades only during TSX hours (9:30 AM - 4:00 PM ET)",
                                className="text-muted",
                            ),
                        ],
                        className="text-center py-4",
                    )
                ),
            ],
            className="shadow-sm mb-4",
        )

    rows = []
    for _, row in df.iterrows():
        pnl_color = "text-success" if row["pnl"] > 0 else "text-danger"
        rows.append(
            html.Tr(
                [
                    html.Td(html.Strong(row["symbol"])),
                    html.Td(row["name"]),
                    html.Td(f"{row['qty']}", className="text-end"),
                    html.Td(f"${row['avg_price']:.2f}", className="text-end"),
                    html.Td(f"${row['current_price']:.2f}", className="text-end"),
                    html.Td(f"${row['pnl']:.2f}", className=f"text-end {pnl_color}"),
                    html.Td(f"{row['pnl_pct']:.2f}%", className=f"text-end {pnl_color}"),
                    html.Td(f"${row['current_price'] * row['qty']:,.2f}", className="text-end"),
                ]
            )
        )

    return dbc.Card(
        [
            dbc.CardHeader([html.I(className="fas fa-briefcase me-2"), "Current Holdings"]),
            dbc.CardBody(
                dbc.Table(
                    [
                        html.Thead(
                            html.Tr(
                                [
                                    html.Th("Symbol"),
                                    html.Th("Name"),
                                    html.Th("Qty", className="text-end"),
                                    html.Th("Buy Price", className="text-end"),
                                    html.Th("Current", className="text-end"),
                                    html.Th("P&L", className="text-end"),
                                    html.Th("P&L %", className="text-end"),
                                    html.Th("Value", className="text-end"),
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
            ),
        ],
        className="shadow-sm mb-4",
    )


def create_recent_trades_table():
    """Render recent trades table."""
    df = generate_recent_trades()
    if df.empty:
        return dbc.Card(
            [
                dbc.CardHeader([html.I(className="fas fa-history me-2"), "Recent Trades"]),
                dbc.CardBody(
                    html.Div(
                        [
                            html.I(className="fas fa-chart-line fa-2x text-muted mb-3"),
                            html.P("No trades executed yet. AI is evaluating market conditions.", className="text-muted"),
                            html.Small(
                                "Trades will appear here when AI identifies high-confidence opportunities", className="text-muted"
                            ),
                        ],
                        className="text-center py-4",
                    )
                ),
            ],
            className="shadow-sm mb-4",
        )

    rows = []
    for _, row in df.iterrows():
        side_badge = dbc.Badge("BUY", color="success", className="me-2") if row["side"] == "BUY" else dbc.Badge("SELL", color="danger", className="me-2")
        status_badge = dbc.Badge(row["status"], color="success" if row["status"] == "FILLED" else "warning")

        pnl_text = f"${row['pnl']:.2f}" if pd.notna(row["pnl"]) else "-"
        pnl_color = "text-success" if pd.notna(row["pnl"]) and row["pnl"] > 0 else "text-danger" if pd.notna(row["pnl"]) else ""

        rows.append(
            html.Tr(
                [
                    html.Td(row["time"]),
                    html.Td([side_badge, row["symbol"]]),
                    html.Td(f"{row['qty']}", className="text-end"),
                    html.Td(f"${row['price']:.2f}", className="text-end"),
                    html.Td(status_badge),
                    html.Td(pnl_text, className=f"text-end {pnl_color}"),
                ]
            )
        )

    return dbc.Card(
        [
            dbc.CardHeader([html.I(className="fas fa-history me-2"), f"Recent Trades ({len(df)} total)"]),
            dbc.CardBody(
                dbc.Table(
                    [
                        html.Thead(
                            html.Tr(
                                [
                                    html.Th("Time"),
                                    html.Th("Symbol"),
                                    html.Th("Qty", className="text-end"),
                                    html.Th("Price", className="text-end"),
                                    html.Th("Status"),
                                    html.Th("P&L", className="text-end"),
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
            ),
        ],
        className="shadow-sm mb-4",
    )


__all__ = [
    "_normalize_holding_entry",
    "generate_portfolio_data",
    "generate_holdings",
    "generate_recent_trades",
    "create_summary_cards",
    "create_holdings_table",
    "create_recent_trades_table",
]
