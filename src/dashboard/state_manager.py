"""
Encapsulated trading state management for the dashboard.

Provides a single mutable state dictionary plus helpers to load/save/reset
persistent data so that other modules do not manipulate SQLite/JSON directly.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

from src.infrastructure.state_store import SQLiteStateStore

# Persistent storage locations
STATE_STORE = SQLiteStateStore("data/trading_state.db")
DEMO_STATE_PATH = Path("data/demo_state.json")


def _default_trading_state() -> Dict[str, Any]:
    """Factory for a fresh trading state dictionary."""
    return {
        "initialized": False,
        "starting_capital": 0.0,
        "current_capital": 0.0,
        "trades": [],
        "holdings": [],
        "start_time": None,
        "mode": "demo",
        "demo_capital": 0.0,
        "live_capital": 0.0,
        "live_prices": {},
        "ai_instance": None,
        "ai_decisions": [],
        "learning_log": [],
        "learning_state": {
            "win_streak": 0,
            "loss_streak": 0,
            "risk_multiplier": 1.0,
        },
        "broker": None,
        "paused": False,
        "kill_switch_threshold": 5.0,
        "kill_switch_active": False,
        "max_position_pct": 0.05,
        "regime": "SIDEWAYS",
        "force_market_open": False,
        "questrade_client": None,
    }


trading_state: Dict[str, Any] = _default_trading_state()


def reset_in_memory_state() -> None:
    """Clear the runtime trading state back to defaults."""
    trading_state.clear()
    trading_state.update(_default_trading_state())


def _ensure_snapshot_dir() -> None:
    try:
        DEMO_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def load_trading_state() -> bool:
    """
    Hydrate the in-memory state from persistent storage.
    Returns True if any persisted state was loaded.
    """
    try:
        data = STATE_STORE.load_trading_state() or {}
        if data:
            trading_state["initialized"] = bool(data.get("initialized", 0))
            trading_state["starting_capital"] = float(data.get("starting_capital") or 0)
            trading_state["current_capital"] = float(data.get("current_capital") or 0)
            trading_state["mode"] = data.get("mode", "demo")
            trading_state["start_time"] = data.get("start_time")
            trading_state["demo_capital"] = float(data.get("demo_capital") or 0)
            trading_state["live_capital"] = float(data.get("live_capital") or 0)
            trading_state["paused"] = bool(data.get("paused", 0))
            trading_state["kill_switch_threshold"] = float(data.get("kill_switch_threshold") or 0)
            trading_state["kill_switch_active"] = bool(data.get("kill_switch_active", 0))
            trading_state["learning_state"] = data.get("learning_state") or trading_state["learning_state"]
            try:
                trading_state["holdings"] = STATE_STORE.fetch_holdings()
            except Exception:
                pass
            try:
                recent = STATE_STORE.fetch_trades(limit=200)
                trading_state["trades"] = [
                    {
                        "time": t.get("timestamp"),
                        "symbol": t.get("symbol"),
                        "side": t.get("side"),
                        "qty": t.get("qty"),
                        "price": t.get("price"),
                        "status": t.get("status"),
                        "pnl": t.get("pnl"),
                        **(t.get("extra") or {}),
                    }
                    for t in recent
                ]
            except Exception:
                pass
            return True
        if DEMO_STATE_PATH.exists():
            with open(DEMO_STATE_PATH, "r", encoding="utf-8") as fh:
                data = json.load(fh) or {}
            reset_in_memory_state()
            trading_state.update(data)
            return True
    except Exception:
        pass
    return False


def save_trading_state() -> None:
    """Persist the current runtime state to SQLite and JSON snapshot."""
    try:
        STATE_STORE.save_trading_state(trading_state)
        try:
            _ensure_snapshot_dir()
            snap = {
                "initialized": trading_state.get("initialized", False),
                "starting_capital": trading_state.get("starting_capital", 0),
                "current_capital": trading_state.get("current_capital", 0),
                "mode": trading_state.get("mode", "demo"),
                "trades": trading_state.get("trades", []),
                "holdings": trading_state.get("holdings", []),
                "start_time": trading_state.get("start_time"),
                "learning_log": trading_state.get("learning_log", []),
                "learning_state": trading_state.get("learning_state", {}),
                "demo_capital": trading_state.get("current_capital", 0),
            }
            with open(DEMO_STATE_PATH, "w", encoding="utf-8") as fh:
                json.dump(snap, fh, indent=2, default=str)
        except Exception:
            pass
    except Exception:
        pass


__all__ = [
    "trading_state",
    "STATE_STORE",
    "DEMO_STATE_PATH",
    "reset_in_memory_state",
    "load_trading_state",
    "save_trading_state",
]
