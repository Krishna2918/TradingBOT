"""Database utilities for bootstrapping the SQLite and DuckDB stores."""
from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Iterable

try:
    import duckdb  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - environment specific
    duckdb = None  # type: ignore
    _DUCKDB_IMPORT_ERROR = exc
else:
    _DUCKDB_IMPORT_ERROR = None

SQLITE_PATH = Path(__file__).resolve().parents[2] / "data" / "trading_state.db"
DUCKDB_PATH = Path(__file__).resolve().parents[2] / "data" / "market_data.duckdb"


SQLITE_TABLE_STATEMENTS: Iterable[str] = (
    """
    CREATE TABLE IF NOT EXISTS ai_selections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        score REAL NOT NULL,
        rationale TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS positions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        quantity REAL NOT NULL,
        average_price REAL NOT NULL,
        status TEXT NOT NULL,
        opened_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        closed_at TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        position_id INTEGER,
        symbol TEXT NOT NULL,
        side TEXT NOT NULL,
        quantity REAL NOT NULL,
        price REAL NOT NULL,
        executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(position_id) REFERENCES positions(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS risk_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_type TEXT NOT NULL,
        description TEXT,
        triggered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
)


DUCKDB_TABLE_STATEMENTS: Iterable[str] = (
    """
    CREATE TABLE IF NOT EXISTS price_bars (
        symbol TEXT,
        timestamp TIMESTAMP,
        open DOUBLE,
        high DOUBLE,
        low DOUBLE,
        close DOUBLE,
        volume BIGINT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS technical_indicators (
        symbol TEXT,
        indicator TEXT,
        value DOUBLE,
        timestamp TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS news_articles (
        source TEXT,
        symbol TEXT,
        headline TEXT,
        url TEXT,
        published_at TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS sentiment_scores (
        symbol TEXT,
        provider TEXT,
        score DOUBLE,
        assessed_at TIMESTAMP
    )
    """,
)


def bootstrap_sqlite() -> None:
    """Ensure the SQLite trading state database exists with core tables."""

    SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(SQLITE_PATH)) as connection:
        for statement in SQLITE_TABLE_STATEMENTS:
            connection.execute(statement)
        connection.commit()


def bootstrap_duckdb() -> None:
    """Ensure the DuckDB analytical store exists with required tables."""

    if duckdb is None:
        raise RuntimeError(
            "duckdb package is required to bootstrap the analytical store"
        ) from _DUCKDB_IMPORT_ERROR

    DUCKDB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with closing(duckdb.connect(str(DUCKDB_PATH))) as connection:
        for statement in DUCKDB_TABLE_STATEMENTS:
            connection.execute(statement)


def bootstrap_all() -> None:
    """Bootstrap both database backends."""

    bootstrap_sqlite()
    bootstrap_duckdb()


__all__ = [
    "bootstrap_sqlite",
    "bootstrap_duckdb",
    "bootstrap_all",
    "SQLITE_PATH",
    "DUCKDB_PATH",
]
