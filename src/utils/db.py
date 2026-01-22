"""
Database bootstrap utilities used by scripts and tests.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import duckdb

from src.ai.features import FeatureEngineer
from src.ai.factors import FactorBuilder
from src.ai.scoring import ScoreEngine
from src.trading.risk import RiskManager


def bootstrap_duckdb(path: str = "data/market_data.duckdb") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # Initialising these classes ensures their tables exist
    FeatureEngineer(duckdb_path=path)
    FactorBuilder(duckdb_path=path)
    ScoreEngine(duckdb_path=path)
    # Also ensure DB file is created
    with duckdb.connect(path) as conn:
        conn.execute("PRAGMA verify_parallelism").fetchall()


def bootstrap_sqlite(path: str = "data/trading_state.db") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # The RiskManager ensures core tables
    RiskManager()
    # Touch DB
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.commit()


__all__ = ["bootstrap_duckdb", "bootstrap_sqlite"]

