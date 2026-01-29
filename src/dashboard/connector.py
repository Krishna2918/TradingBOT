"""
Dashboard Connector
===================

Provides read access to AI selection history and risk events stored in SQLite.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml


class DashboardConnector:
    """
    Connect to the trading state database and retrieve AI picks and risk events.
    """

    def __init__(self, config_path: str = "config/risk.yaml") -> None:
        self.sqlite_path = self._load_sqlite_path(config_path)
        self._ensure_tables_exist()

    def _load_sqlite_path(self, config_path: str) -> str:
        config_file = Path(config_path)
        if not config_file.exists():
            return "data/trading_state.db"
        with open(config_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data.get("risk", {}).get("sqlite_path", "data/trading_state.db")

    def _ensure_tables_exist(self) -> None:
        Path(self.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.sqlite_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS ai_selections (
                    trade_date TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    score REAL NOT NULL,
                    explanation TEXT NOT NULL,
                    shares INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    PRIMARY KEY (trade_date, symbol)
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS positions (
                    trade_date TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    shares INTEGER NOT NULL,
                    stop_loss REAL NOT NULL,
                    take_profit REAL NOT NULL,
                    risk_alloc REAL NOT NULL,
                    PRIMARY KEY (trade_date, symbol)
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS risk_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    symbol TEXT,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def ai_picks(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Retrieve recent AI stock selections.

        Returns a DataFrame with columns:
            trade_date, symbol, score, explanation, shares, confidence
        """
        query = """
            SELECT trade_date, symbol, score, explanation, shares, confidence
            FROM ai_selections
            ORDER BY trade_date DESC, score DESC
        """
        if limit:
            query += f" LIMIT {int(limit)}"

        with sqlite3.connect(self.sqlite_path) as conn:
            df = pd.read_sql_query(query, conn)
        return df

    def risk_events(self, limit: int = 10) -> pd.DataFrame:
        """
        Retrieve recent risk events.

        Returns a DataFrame with columns:
            id, timestamp, event_type, symbol, severity, message
        """
        query = f"""
            SELECT id, timestamp, event_type, symbol, severity, message
            FROM risk_events
            ORDER BY id DESC
            LIMIT {int(limit)}
        """
        with sqlite3.connect(self.sqlite_path) as conn:
            df = pd.read_sql_query(query, conn)
        return df

    def positions(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Retrieve current/recent positions.

        Returns a DataFrame with columns:
            trade_date, symbol, entry_price, shares, stop_loss, take_profit, risk_alloc
        """
        query = """
            SELECT trade_date, symbol, entry_price, shares, stop_loss, take_profit, risk_alloc
            FROM positions
            ORDER BY trade_date DESC
        """
        if limit:
            query += f" LIMIT {int(limit)}"

        with sqlite3.connect(self.sqlite_path) as conn:
            df = pd.read_sql_query(query, conn)
        return df

    def log_risk_event(
        self,
        event_type: str,
        severity: str,
        message: str,
        symbol: Optional[str] = None,
    ) -> None:
        """Log a risk event to the database."""
        from datetime import datetime

        with sqlite3.connect(self.sqlite_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO risk_events (timestamp, event_type, symbol, severity, message)
                VALUES (?, ?, ?, ?, ?)
                """,
                (datetime.utcnow().isoformat(), event_type, symbol, severity, message),
            )
            conn.commit()


__all__ = ["DashboardConnector"]
