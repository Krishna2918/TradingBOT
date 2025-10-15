"""
Unified data access layer for the interactive dashboard.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import date
from typing import List, Dict

import duckdb
import pandas as pd


@dataclass
class DashboardConnector:
    sqlite_path: str = "data/trading_state.db"
    duckdb_path: str = "data/market_data.duckdb"

    # ------------------------------------------------------------------ #
    # SQLite sources
    # ------------------------------------------------------------------ #
    def ai_picks(self) -> pd.DataFrame:
        with sqlite3.connect(self.sqlite_path) as conn:
            query = """
                SELECT
                    s.trade_date,
                    s.symbol,
                    s.score,
                    s.explanation,
                    s.shares,
                    s.confidence,
                    p.entry_price,
                    p.stop_loss,
                    p.take_profit
                FROM ai_selections s
                LEFT JOIN positions p
                    ON s.trade_date = p.trade_date AND s.symbol = p.symbol
                ORDER BY s.trade_date DESC, s.score DESC
            """
            frame = pd.read_sql_query(query, conn)
        return frame

    def pnl_summary(self) -> pd.DataFrame:
        with sqlite3.connect(self.sqlite_path) as conn:
            query = """
                SELECT trade_date, symbol, realized, unrealized
                FROM pnl_tracking
                ORDER BY ts DESC
            """
            return pd.read_sql_query(query, conn)

    def risk_events(self, limit: int = 50) -> pd.DataFrame:
        with sqlite3.connect(self.sqlite_path) as conn:
            query = """
                SELECT ts, event_type, message
                FROM risk_events
                ORDER BY ts DESC
                LIMIT ?
            """
            return pd.read_sql_query(query, conn, params=(limit,))

    def open_orders(self) -> pd.DataFrame:
        with sqlite3.connect(self.sqlite_path) as conn:
            query = """
                SELECT order_id, ts, symbol, side, order_type, quantity, price
                FROM orders
                ORDER BY ts DESC
            """
            return pd.read_sql_query(query, conn)

    # ------------------------------------------------------------------ #
    # DuckDB insights
    # ------------------------------------------------------------------ #
    def model_explanations(self) -> List[Dict]:
        with duckdb.connect(self.duckdb_path, read_only=True) as conn:
            query = """
                SELECT ts, symbol, score, details_json
                FROM ai_scores
                WHERE ts >= NOW() - INTERVAL 7 DAY
                ORDER BY ts DESC
                LIMIT 200
            """
            df = conn.execute(query).fetch_df()
        explanations = []
        for _, row in df.iterrows():
            detail = json.loads(row["details_json"])
            explanations.append(
                {
                    "timestamp": row["ts"],
                    "symbol": row["symbol"],
                    "score": row["score"],
                    "technical": detail.get("technical"),
                    "sentiment": detail.get("sentiment"),
                    "fundamental": detail.get("fundamental"),
                    "momentum": detail.get("momentum"),
                    "volume": detail.get("volume"),
                    "rationale": detail.get("sentiment_json"),
                }
            )
        return explanations

    def indicator_snapshot(self, symbol: str) -> pd.DataFrame:
        with duckdb.connect(self.duckdb_path, read_only=True) as conn:
            query = """
                SELECT *
                FROM indicators
                WHERE symbol = ?
                ORDER BY ts DESC
                LIMIT 120
            """
            return conn.execute(query, [symbol]).fetch_df()

    # ------------------------------------------------------------------ #
    # Phase 8: Enhanced Dashboard Data Providers
    # ------------------------------------------------------------------ #
    
    def api_budget_status(self) -> pd.DataFrame:
        """Get API budget and rate-limit status."""
        with sqlite3.connect(self.sqlite_path) as conn:
            query = """
                SELECT 
                    api_name,
                    requests_made,
                    requests_limit,
                    daily_requests,
                    daily_limit,
                    rate_limit_hits,
                    last_request_time,
                    window_start,
                    window_end,
                    mode
                FROM api_usage_metrics
                WHERE window_end >= datetime('now', '-1 day')
                ORDER BY last_request_time DESC
            """
            return pd.read_sql_query(query, conn)
    
    def phase_duration_timeline(self, limit: int = 100) -> pd.DataFrame:
        """Get phase duration timeline data."""
        with sqlite3.connect(self.sqlite_path) as conn:
            query = """
                SELECT 
                    timestamp,
                    phase_name,
                    step_name,
                    duration_ms,
                    status,
                    detail,
                    mode
                FROM phase_execution_tracking
                ORDER BY timestamp DESC
                LIMIT ?
            """
            return pd.read_sql_query(query, conn, params=(limit,))
    
    def confidence_calibration_data(self, limit: int = 50) -> pd.DataFrame:
        """Get calibrated vs raw confidence data for recent trades."""
        with sqlite3.connect(self.sqlite_path) as conn:
            query = """
                SELECT 
                    trade_date,
                    model,
                    symbol,
                    raw_confidence,
                    calibrated_confidence,
                    outcome,
                    window_id,
                    created_at
                FROM confidence_calibration
                ORDER BY trade_date DESC
                LIMIT ?
            """
            return pd.read_sql_query(query, conn, params=(limit,))
    
    def ensemble_weights_history(self, days: int = 7) -> pd.DataFrame:
        """Get ensemble weights history for the last N days."""
        with sqlite3.connect(self.sqlite_path) as conn:
            query = """
                SELECT 
                    timestamp,
                    model,
                    weight,
                    brier_score,
                    accuracy,
                    sample_count,
                    created_at
                FROM model_performance
                WHERE timestamp >= datetime('now', '-{} days')
                ORDER BY timestamp DESC, model
            """.format(days)
            return pd.read_sql_query(query, conn)
    
    def drawdown_and_regime_data(self, limit: int = 30) -> pd.DataFrame:
        """Get drawdown scalar and regime state data."""
        with sqlite3.connect(self.sqlite_path) as conn:
            query = """
                SELECT 
                    timestamp,
                    portfolio_value,
                    daily_pnl,
                    total_pnl,
                    mode
                FROM portfolio_snapshots
                ORDER BY timestamp DESC
                LIMIT ?
            """
            portfolio_data = pd.read_sql_query(query, conn, params=(limit,))
        
        # Get regime state data
        with sqlite3.connect(self.sqlite_path) as conn:
            query = """
                SELECT 
                    timestamp,
                    regime,
                    trend_direction,
                    volatility_level,
                    trend_strength,
                    volatility_ratio,
                    regime_confidence,
                    transition_probability,
                    mode
                FROM regime_state
                ORDER BY timestamp DESC
                LIMIT ?
            """
            regime_data = pd.read_sql_query(query, conn, params=(limit,))
        
        # Merge portfolio and regime data
        if not portfolio_data.empty and not regime_data.empty:
            # Convert timestamps to datetime for merging
            portfolio_data['timestamp'] = pd.to_datetime(portfolio_data['timestamp'])
            regime_data['timestamp'] = pd.to_datetime(regime_data['timestamp'])
            
            # Merge on timestamp (approximate match)
            merged_data = pd.merge_asof(
                portfolio_data.sort_values('timestamp'),
                regime_data.sort_values('timestamp'),
                on='timestamp',
                direction='backward'
            )
            return merged_data
        else:
            return pd.concat([portfolio_data, regime_data], ignore_index=True)
    
    def model_rationale_trace(self, symbol: str = None, limit: int = 20) -> List[Dict]:
        """Get detailed rationale trace for model explanations."""
        with duckdb.connect(self.duckdb_path, read_only=True) as conn:
            if symbol:
                query = """
                    SELECT 
                        ts, symbol, score, details_json
                    FROM ai_scores
                    WHERE symbol = ? AND ts >= NOW() - INTERVAL 7 DAY
                    ORDER BY ts DESC
                    LIMIT ?
                """
                df = conn.execute(query, [symbol, limit]).fetch_df()
            else:
                query = """
                    SELECT 
                        ts, symbol, score, details_json
                    FROM ai_scores
                    WHERE ts >= NOW() - INTERVAL 7 DAY
                    ORDER BY ts DESC
                    LIMIT ?
                """
                df = conn.execute(query, [limit]).fetch_df()
        
        rationale_traces = []
        for _, row in df.iterrows():
            try:
                detail = json.loads(row["details_json"]) if row["details_json"] else {}
                rationale_traces.append({
                    "timestamp": row["ts"],
                    "symbol": row["symbol"],
                    "score": row["score"],
                    "model_name": "ensemble",  # Default to ensemble since model_name column doesn't exist
                    "technical_analysis": detail.get("technical", {}),
                    "sentiment_analysis": detail.get("sentiment", {}),
                    "fundamental_analysis": detail.get("fundamental", {}),
                    "risk_assessment": detail.get("risk", {}),
                    "market_regime": detail.get("regime", {}),
                    "confidence_breakdown": detail.get("confidence", {}),
                    "rationale_summary": detail.get("summary", ""),
                })
            except (json.JSONDecodeError, KeyError) as e:
                # Handle malformed JSON gracefully
                rationale_traces.append({
                    "timestamp": row["ts"],
                    "symbol": row["symbol"],
                    "score": row["score"],
                    "model_name": row.get("model_name", "ensemble"),
                    "error": f"Failed to parse details: {str(e)}"
                })
        
        return rationale_traces
    
    def system_health_metrics(self) -> Dict:
        """Get comprehensive system health metrics."""
        with sqlite3.connect(self.sqlite_path) as conn:
            # Get recent API validation status
            api_query = """
                SELECT 
                    api_name,
                    status,
                    response_time_ms,
                    last_validated
                FROM api_validation_log
                WHERE last_validated >= datetime('now', '-1 hour')
                ORDER BY last_validated DESC
            """
            api_status = pd.read_sql_query(api_query, conn)
            
            # Get recent phase performance
            phase_query = """
                SELECT 
                    phase_name,
                    AVG(duration_ms) as avg_duration_ms,
                    COUNT(*) as execution_count,
                    SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as success_count
                FROM phase_execution_tracking
                WHERE timestamp >= datetime('now', '-1 day')
                GROUP BY phase_name
            """
            phase_performance = pd.read_sql_query(phase_query, conn)
            
            # Get recent risk events
            risk_query = """
                SELECT 
                    event_type,
                    COUNT(*) as event_count
                FROM risk_events
                WHERE ts >= datetime('now', '-1 day')
                GROUP BY event_type
            """
            risk_events = pd.read_sql_query(risk_query, conn)
        
        return {
            "api_status": api_status.to_dict('records') if not api_status.empty else [],
            "phase_performance": phase_performance.to_dict('records') if not phase_performance.empty else [],
            "risk_events": risk_events.to_dict('records') if not risk_events.empty else [],
            "timestamp": pd.Timestamp.now().isoformat()
        }


__all__ = ["DashboardConnector"]
