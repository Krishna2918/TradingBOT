"""
Multi-factor scoring engine.

This module aggregates technical indicators, sentiment, and fundamentals into a
single weighted score that downstream agents can consume.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Sequence

import duckdb
import numpy as np
import pandas as pd

from src.ai.features import FeatureEngineer, FeatureEngineeringError, get_feature_matrix
from src.ai.factors import FactorBuilder, FactorError


logger = logging.getLogger(__name__)

SCORES_TABLE = "ai_scores"


@dataclass
class ScoreEngine:
    """
    Compute weighted scores across technical, sentiment, and fundamental signals.
    """

    duckdb_path: str = "data/market_data.duckdb"
    feature_window: int = 90
    factors: FactorBuilder = None
    features: FeatureEngineer = None

    def __post_init__(self) -> None:
        self.factors = self.factors or FactorBuilder(duckdb_path=self.duckdb_path)
        self.features = self.features or FeatureEngineer(duckdb_path=self.duckdb_path)
        self._ensure_table()

    def score_universe(self, symbols: Sequence[str]) -> pd.DataFrame:
        if not symbols:
            return pd.DataFrame(columns=["symbol", "score", "details_json"])

        # Ensure features/factors are up to date
        try:
            self.features.build_for_universe(symbols)
        except FeatureEngineeringError as exc:
            logger.error("Feature engineering failed: %s", exc)

        try:
            self.factors.build_for_universe(symbols)
        except FactorError as exc:
            logger.error("Factor build failed: %s", exc)

        feature_df = get_feature_matrix(symbols, window=self.feature_window, duckdb_path=self.duckdb_path)
        factor_df = self.factors.latest_factors(symbols)

        latest_features = (
            feature_df.sort_values(["symbol", "ts"]).groupby("symbol").tail(1).set_index("symbol")
        )
        factor_df = factor_df.set_index("symbol") if not factor_df.empty else pd.DataFrame()

        rows = []
        for symbol in symbols:
            if symbol not in latest_features.index:
                logger.warning("Missing features for %s", symbol)
                continue
            feat_row = latest_features.loc[symbol]
            factor_row = factor_df.loc[symbol] if symbol in factor_df.index else None

            technical, momentum, volume = self._technical_scores(feat_row)
            sentiment = (
                float(factor_row["sentiment_score"])
                if factor_row is not None and not pd.isna(factor_row["sentiment_score"])
                else 0.0
            )
            fundamental = (
                float(factor_row["fundamental_score"])
                if factor_row is not None and not pd.isna(factor_row["fundamental_score"])
                else 0.5
            )
            weighted_score, components = self._combine_scores(
                technical=technical,
                sentiment=sentiment,
                fundamental=fundamental,
                momentum=momentum,
                volume=volume,
            )
            details = {
                "technical": technical,
                "sentiment": components["sentiment"],
                "fundamental": fundamental,
                "momentum": components["momentum"],
                "volume": components["volume"],
                "shape": components["shape"],
                "ts": feat_row["ts"].isoformat() if isinstance(feat_row["ts"], datetime) else str(feat_row["ts"]),
                "feature_source": feat_row.get("source"),
                "sentiment_raw": sentiment,
                "momentum_raw": float(feat_row.get("momentum", 0.0)),
                "volume_raw": float(feat_row.get("volume_zscore", 0.0)),
                "fundamentals_json": factor_row["fundamental_details"] if factor_row is not None else None,
                "sentiment_json": factor_row["sentiment_details"] if factor_row is not None else None,
            }
            rows.append(
                {
                    "ts": datetime.utcnow(),
                    "symbol": symbol,
                    "score": weighted_score,
                    "details_json": json.dumps(details),
                }
            )

        scores_df = pd.DataFrame(rows)
        if not scores_df.empty:
            self._persist(scores_df)
        return scores_df[["symbol", "score", "details_json"]]

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _ensure_table(self) -> None:
        with duckdb.connect(self.duckdb_path) as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {SCORES_TABLE} (
                    ts TIMESTAMP NOT NULL,
                    symbol VARCHAR NOT NULL,
                    score DOUBLE NOT NULL,
                    details_json JSON,
                    PRIMARY KEY (symbol, ts)
                )
                """
            )

    def _persist(self, frame: pd.DataFrame) -> None:
        with duckdb.connect(self.duckdb_path) as conn:
            conn.register("incoming_scores", frame)
            conn.execute(
                f"""
                INSERT INTO {SCORES_TABLE} (ts, symbol, score, details_json)
                SELECT ts, symbol, score, details_json
                FROM incoming_scores
                """
            )
            conn.unregister("incoming_scores")

    def _technical_scores(self, row: pd.Series) -> tuple[float, float, float]:
        close = float(row.get("close", 0.0))
        boll_upper = float(row.get("bollinger_upper", np.nan))
        boll_lower = float(row.get("bollinger_lower", np.nan))
        macd_hist = float(row.get("macd_hist", np.nan))
        adx = float(row.get("adx", np.nan))
        rsi = float(row.get("rsi", np.nan))
        momentum = float(row.get("momentum", 0.0))
        volume_z = float(row.get("volume_zscore", 0.0))

        features: list[float] = []
        if not np.isnan(rsi):
            features.append(1 - min(abs(rsi - 50) / 50, 1))
        if not np.isnan(macd_hist):
            features.append(0.5 + 0.5 * np.tanh(macd_hist))
        if not np.isnan(adx):
            features.append(min(adx / 50, 1))
        if not np.isnan(boll_upper) and not np.isnan(boll_lower) and boll_upper != boll_lower:
            position = (close - boll_lower) / (boll_upper - boll_lower)
            features.append(1 - abs(position - 0.5) * 2)

        technical_score = float(np.nanmean(features)) if features else 0.5
        momentum_score = 0.5 + 0.5 * np.tanh(momentum * 5)
        volume_score = 0.5 + 0.5 * np.tanh(volume_z / 3)
        return technical_score, momentum_score, volume_score

    def _combine_scores(
        self,
        *,
        technical: float,
        sentiment: float,
        fundamental: float,
        momentum: float,
        volume: float,
    ) -> tuple[float, Dict[str, float]]:
        sentiment_scaled = 0.5 + (sentiment / 2)
        detail = {
            "technical": technical,
            "sentiment": sentiment_scaled,
            "fundamental": fundamental,
            "momentum": momentum,
            "volume": volume,
            "shape": {
                "technical": 0.30,
                "sentiment": 0.25,
                "fundamental": 0.20,
                "momentum": 0.15,
                "volume": 0.10,
            },
        }
        weighted = (
            0.30 * technical
            + 0.25 * sentiment_scaled
            + 0.20 * fundamental
            + 0.15 * momentum
            + 0.10 * volume
        )
        return float(weighted), detail


__all__ = ["ScoreEngine"]
