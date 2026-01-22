import os
import sqlite3
from datetime import date

import pytest
import requests

from src.agents.ai_selector import AISelectorAgent


def _ollama_available() -> bool:
    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    try:
        resp = requests.get(f"{base}/api/tags", timeout=5)
        return resp.status_code == 200
    except requests.RequestException:
        return False


@pytest.mark.integration
def test_full_pipeline_runs_and_persists():
    if not _ollama_available():
        pytest.skip("Ollama endpoint is not reachable")

    agent = AISelectorAgent()
    trade_dt = date.today().isoformat()
    with sqlite3.connect(agent.sqlite_path) as conn:
        conn.execute("DELETE FROM ai_selections WHERE trade_date = ?", (trade_dt,))
        conn.execute("DELETE FROM positions WHERE trade_date = ?", (trade_dt,))
        order_before = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
        trade_before = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        conn.commit()

    results = agent.run(limit=5)
    assert len(results) == 5
    assert all(r.shares > 0 for r in results)

    with sqlite3.connect(agent.sqlite_path) as conn:
        sel_count = conn.execute(
            "SELECT COUNT(*) FROM ai_selections WHERE trade_date = ?", (trade_dt,)
        ).fetchone()[0]
        pos_count = conn.execute(
            "SELECT COUNT(*) FROM positions WHERE trade_date = ?", (trade_dt,)
        ).fetchone()[0]
        order_count = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
        trade_count = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]

    assert sel_count == 5
    assert pos_count == 5
    assert order_count - order_before >= 5
    assert trade_count - trade_before >= 5
