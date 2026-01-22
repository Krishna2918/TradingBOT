"""
SQLite-backed state store for dashboard persistence.
Stores trading state, trades, holdings, and learning logs.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class SQLiteStateStore:
    def __init__(self, db_path: str = "data/trading_state.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            c = conn.cursor()
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS state_meta (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    initialized INTEGER DEFAULT 0,
                    starting_capital REAL DEFAULT 0,
                    current_capital REAL DEFAULT 0,
                    mode TEXT DEFAULT 'demo',
                    start_time TEXT,
                    demo_capital REAL DEFAULT 0,
                    live_capital REAL DEFAULT 0,
                    paused INTEGER DEFAULT 0,
                    kill_switch_threshold REAL DEFAULT 0,
                    kill_switch_active INTEGER DEFAULT 0,
                    learning_state TEXT
                )
                """
            )
            c.execute(
                """
                INSERT OR IGNORE INTO state_meta (id, learning_state)
                VALUES (1, '{}')
                """
            )

            c.execute(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    symbol TEXT,
                    side TEXT,
                    qty REAL,
                    price REAL,
                    pnl REAL,
                    status TEXT,
                    extra TEXT
                )
                """
            )

            c.execute(
                """
                CREATE TABLE IF NOT EXISTS holdings (
                    symbol TEXT PRIMARY KEY,
                    qty REAL,
                    avg_price REAL,
                    current_price REAL
                )
                """
            )

            c.execute(
                """
                CREATE TABLE IF NOT EXISTS learning_log (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    pnl REAL,
                    win_streak INTEGER,
                    loss_streak INTEGER,
                    risk_multiplier REAL,
                    reflection TEXT,
                    features TEXT
                )
                """
            )
            conn.commit()

    # --- State ---
    def save_trading_state(self, state: Dict[str, Any]) -> None:
        with self._connect() as conn:
            c = conn.cursor()
            c.execute(
                """
                UPDATE state_meta
                SET initialized=?, starting_capital=?, current_capital=?, mode=?, start_time=?,
                    demo_capital=?, live_capital=?, paused=?, kill_switch_threshold=?,
                    kill_switch_active=?, learning_state=?
                WHERE id=1
                """,
                (
                    1 if state.get("initialized") else 0,
                    float(state.get("starting_capital", 0) or 0),
                    float(state.get("current_capital", 0) or 0),
                    state.get("mode", "demo"),
                    state.get("start_time"),
                    float(state.get("demo_capital", 0) or 0),
                    float(state.get("live_capital", 0) or 0),
                    1 if state.get("paused") else 0,
                    float(state.get("kill_switch_threshold", 0) or 0),
                    1 if state.get("kill_switch_active") else 0,
                    json.dumps(state.get("learning_state", {})),
                ),
            )
            conn.commit()

    def load_trading_state(self) -> Dict[str, Any]:
        with self._connect() as conn:
            c = conn.cursor()
            row = c.execute("SELECT * FROM state_meta WHERE id=1").fetchone()
            if not row:
                return {}
            out = dict(row)
            try:
                out["learning_state"] = json.loads(out.get("learning_state") or "{}")
            except Exception:
                out["learning_state"] = {}
            return out

    # --- Trades ---
    def insert_trade(self, trade: Dict[str, Any]) -> None:
        with self._connect() as conn:
            c = conn.cursor()
            extra = {k: v for k, v in trade.items() if k not in {"time", "symbol", "side", "qty", "price", "status", "pnl"}}
            c.execute(
                """
                INSERT INTO trades (timestamp, symbol, side, qty, price, pnl, status, extra)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trade.get("time"),
                    trade.get("symbol"),
                    trade.get("side"),
                    float(trade.get("qty", 0) or 0),
                    float(trade.get("price", 0) or 0),
                    None if trade.get("pnl") is None else float(trade.get("pnl")),
                    trade.get("status"),
                    json.dumps(extra) if extra else None,
                ),
            )
            conn.commit()

    def fetch_trades(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            c = conn.cursor()
            rows = c.execute(
                "SELECT * FROM trades ORDER BY trade_id DESC LIMIT ?",
                (limit,),
            ).fetchall()
            out: List[Dict[str, Any]] = []
            for r in rows:
                d = dict(r)
                if d.get("extra"):
                    try:
                        d["extra"] = json.loads(d["extra"])
                    except Exception:
                        pass
                out.append(d)
            return out

    # --- Holdings ---
    def upsert_holding(self, symbol: str, qty: float, avg_price: float, current_price: float) -> None:
        with self._connect() as conn:
            c = conn.cursor()
            c.execute(
                """
                INSERT INTO holdings(symbol, qty, avg_price, current_price)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    qty=excluded.qty,
                    avg_price=excluded.avg_price,
                    current_price=excluded.current_price
                """,
                (symbol, float(qty), float(avg_price), float(current_price)),
            )
            conn.commit()

    def delete_holding(self, symbol: str) -> None:
        with self._connect() as conn:
            c = conn.cursor()
            c.execute("DELETE FROM holdings WHERE symbol=?", (symbol,))
            conn.commit()

    def fetch_holdings(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            c = conn.cursor()
            rows = c.execute("SELECT * FROM holdings").fetchall()
            return [dict(r) for r in rows]

    # --- Learning log ---
    def insert_learning_entry(
        self,
        *,
        timestamp: str,
        pnl: Optional[float],
        win_streak: int,
        loss_streak: int,
        risk_multiplier: float,
        reflection: Optional[str] = None,
        features: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._connect() as conn:
            c = conn.cursor()
            c.execute(
                """
                INSERT INTO learning_log(timestamp, pnl, win_streak, loss_streak, risk_multiplier, reflection, features)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    None if pnl is None else float(pnl),
                    int(win_streak),
                    int(loss_streak),
                    float(risk_multiplier),
                    reflection,
                    json.dumps(features) if features else None,
                ),
            )
            conn.commit()

    def fetch_learning_log(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            c = conn.cursor()
            rows = c.execute(
                "SELECT * FROM learning_log ORDER BY log_id DESC LIMIT ?",
                (limit,),
            ).fetchall()
            out: List[Dict[str, Any]] = []
            for r in rows:
                d = dict(r)
                if d.get("features"):
                    try:
                        d["features"] = json.loads(d["features"])
                    except Exception:
                        pass
                out.append(d)
            return out

    # --- Maintenance ---
    def reset_all(self) -> None:
        """Clear all persisted state so the demo can restart from scratch."""
        with self._connect() as conn:
            c = conn.cursor()
            c.execute("DELETE FROM trades")
            c.execute("DELETE FROM holdings")
            c.execute("DELETE FROM learning_log")
            c.execute(
                """
                UPDATE state_meta
                SET initialized=0,
                    starting_capital=0,
                    current_capital=0,
                    mode='demo',
                    start_time=NULL,
                    demo_capital=0,
                    live_capital=0,
                    paused=0,
                    kill_switch_threshold=0,
                    kill_switch_active=0,
                    learning_state='{}'
                WHERE id=1
                """
            )
            conn.commit()
