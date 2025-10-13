import sqlite3

import pytest

duckdb = pytest.importorskip("duckdb")

from src.utils import db


def test_bootstrap_creates_tables(tmp_path, monkeypatch):
    sqlite_path = tmp_path / "trading_state.db"
    duckdb_path = tmp_path / "market_data.duckdb"

    monkeypatch.setattr(db, "SQLITE_PATH", sqlite_path)
    monkeypatch.setattr(db, "DUCKDB_PATH", duckdb_path)

    db.bootstrap_all()

    with sqlite3.connect(sqlite_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }

    assert {"ai_selections", "positions", "trades", "risk_events"}.issubset(tables)

    with duckdb.connect(str(duckdb_path)) as connection:
        duck_tables = {
            row[0]
            for row in connection.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
            ).fetchall()
        }

    assert {
        "price_bars",
        "technical_indicators",
        "news_articles",
        "sentiment_scores",
    }.issubset(duck_tables)
