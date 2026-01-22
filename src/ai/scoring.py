"""
Multi-factor scoring engine.

This module aggregates technical indicators, sentiment, and fundamentals into a
single weighted score that downstream agents can consume.

Supports two modes:
1. Rule-based scoring (default): Uses weighted combination of technical/sentiment/fundamental factors
2. Model-based scoring: Uses a trained ML model from the model registry for predictions

Usage:
    # Rule-based (default)
    engine = ScoreEngine()

    # Model-based (uses production model from registry)
    engine = ScoreEngine(use_trained_model=True, model_name="lstm_attention")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

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

    Can optionally use a trained ML model for predictions instead of rule-based scoring.
    """

    duckdb_path: str = "data/market_data.duckdb"
    feature_window: int = 90
    factors: FactorBuilder = None
    features: FeatureEngineer = None

    # Model-based scoring options
    use_trained_model: bool = False
    model_name: Optional[str] = None
    model_registry_path: str = "models/registry"

    # Internal state
    _loaded_model: Any = field(default=None, init=False, repr=False)
    _model_metadata: Any = field(default=None, init=False, repr=False)
    _model_registry: Any = field(default=None, init=False, repr=False)
    _production_model_key: Optional[str] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.factors = self.factors or FactorBuilder(duckdb_path=self.duckdb_path)
        self.features = self.features or FeatureEngineer(duckdb_path=self.duckdb_path)
        self._ensure_table()

        # Load trained model if requested
        if self.use_trained_model and self.model_name:
            self._load_production_model()

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

            # Branch: model-based vs rule-based scoring
            if self.use_trained_model and self._loaded_model is not None:
                # Model-based scoring
                weighted_score, model_details = self._model_based_score(
                    symbol, feat_row, feature_df
                )
                details = {
                    "scoring_mode": "model_based",
                    **model_details,
                    "ts": feat_row["ts"].isoformat() if isinstance(feat_row["ts"], datetime) else str(feat_row["ts"]),
                    "feature_source": feat_row.get("source"),
                }
            else:
                # Rule-based scoring (default)
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
                    "scoring_mode": "rule_based",
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

    def score_from_features(
        self, feature_df: pd.DataFrame, symbols: Sequence[str]
    ) -> pd.DataFrame:
        """
        Score symbols using a pre-loaded feature matrix (for offline/snapshot mode).

        This method does NOT call live APIs - it uses the provided feature data directly.
        Useful for deterministic/reproducible scoring runs using snapshot data.

        Args:
            feature_df: Pre-loaded feature matrix with 'symbol', 'ts', and feature columns
            symbols: List of symbols to score

        Returns:
            DataFrame with columns [symbol, score, details_json]
        """
        if feature_df.empty:
            logger.warning("Empty feature matrix provided to score_from_features")
            return pd.DataFrame(columns=["symbol", "score", "details_json"])

        logger.info(
            "Scoring %d symbols from pre-loaded features (offline mode)", len(symbols)
        )

        # Get latest features per symbol (handle snapshots without 'ts' column)
        if "ts" in feature_df.columns:
            latest_features = (
                feature_df.sort_values(["symbol", "ts"])
                .groupby("symbol")
                .tail(1)
                .set_index("symbol")
            )
        else:
            # No timestamp column - take last row per symbol
            latest_features = (
                feature_df.groupby("symbol")
                .tail(1)
                .set_index("symbol")
            )

        rows = []
        for symbol in symbols:
            if symbol not in latest_features.index:
                logger.debug("Missing features for %s in snapshot", symbol)
                continue

            feat_row = latest_features.loc[symbol]

            # Branch: model-based vs rule-based scoring
            if self.use_trained_model and self._loaded_model is not None:
                # Model-based scoring with snapshot features
                weighted_score, model_details = self._model_based_score(
                    symbol, feat_row, feature_df
                )
                details = {
                    "scoring_mode": "model_based",
                    "data_source": "snapshot",
                    **model_details,
                    "ts": feat_row["ts"].isoformat()
                    if hasattr(feat_row.get("ts"), "isoformat")
                    else str(feat_row.get("ts", "")),
                }
            else:
                # Rule-based scoring with snapshot features
                technical, momentum, volume = self._technical_scores(feat_row)

                # In snapshot mode, we may not have factor data, use defaults
                sentiment = 0.5
                fundamental = 0.5

                weighted_score, components = self._combine_scores(
                    technical=technical,
                    sentiment=sentiment,
                    fundamental=fundamental,
                    momentum=momentum,
                    volume=volume,
                )

                details = {
                    "scoring_mode": "rule_based",
                    "data_source": "snapshot",
                    "technical": technical,
                    "sentiment": components["sentiment"],
                    "fundamental": fundamental,
                    "momentum": components["momentum"],
                    "volume": components["volume"],
                    "shape": components["shape"],
                    "ts": feat_row["ts"].isoformat()
                    if hasattr(feat_row.get("ts"), "isoformat")
                    else str(feat_row.get("ts", "")),
                    "note": "Scored from snapshot (offline mode)",
                }

            rows.append({
                "ts": datetime.utcnow(),
                "symbol": symbol,
                "score": weighted_score,
                "details_json": json.dumps(details),
            })

        scores_df = pd.DataFrame(rows)
        if not scores_df.empty:
            self._persist(scores_df)
            logger.info("Scored %d symbols from snapshot", len(scores_df))

        return scores_df[["symbol", "score", "details_json"]] if not scores_df.empty else pd.DataFrame(columns=["symbol", "score", "details_json"])

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

    def _load_production_model(self) -> None:
        """Load the production model from the registry."""
        try:
            from src.ai.model_registry import ModelRegistry

            registry = ModelRegistry(storage_path=self.model_registry_path)
            self._model_registry = registry  # Store for feature validation
            result = registry.get_production_model(self.model_name)

            if result:
                self._loaded_model, self._model_metadata = result
                # Store model key for feature validation
                self._production_model_key = f"{self._model_metadata.model_id}_{self._model_metadata.version}"
                logger.info(
                    "Loaded production model: %s (version=%s, metrics=%s)",
                    self.model_name,
                    self._model_metadata.version,
                    self._model_metadata.performance_metrics,
                )
            else:
                logger.warning(
                    "No production model found for %s, falling back to rule-based scoring",
                    self.model_name,
                )
                self.use_trained_model = False
        except ImportError:
            logger.warning("ModelRegistry not available, using rule-based scoring")
            self.use_trained_model = False
        except Exception as exc:
            logger.error("Failed to load model %s: %s", self.model_name, exc)
            self.use_trained_model = False

    def _model_based_score(
        self, symbol: str, features: pd.Series, feature_df: pd.DataFrame
    ) -> tuple[float, Dict[str, Any]]:
        """
        Generate score using the trained model.

        Returns score and details dict.
        """
        if self._loaded_model is None:
            logger.warning("No model loaded for %s, using rule-based fallback", symbol)
            technical, momentum, volume = self._technical_scores(features)
            weighted, components = self._combine_scores(
                technical=technical,
                sentiment=0.5,
                fundamental=0.5,
                momentum=momentum,
                volume=volume,
            )
            return weighted, {"source": "rule_fallback", **components}

        try:
            # Get recent feature history for the symbol
            symbol_features = feature_df[feature_df["symbol"] == symbol].copy()
            if len(symbol_features) < 10:
                logger.warning(
                    "Insufficient feature history for %s (%d rows), using rule-based",
                    symbol,
                    len(symbol_features),
                )
                technical, momentum, volume = self._technical_scores(features)
                weighted, components = self._combine_scores(
                    technical=technical,
                    sentiment=0.5,
                    fundamental=0.5,
                    momentum=momentum,
                    volume=volume,
                )
                return weighted, {"source": "rule_fallback_insufficient_data", **components}

            # Prepare features for model (handle snapshots without 'ts' column)
            if "ts" in symbol_features.columns:
                symbol_features = symbol_features.sort_values("ts").tail(self.feature_window)
            else:
                symbol_features = symbol_features.tail(self.feature_window)

            # Extract numeric columns for model input
            numeric_cols = symbol_features.select_dtypes(include=[np.number]).columns
            feature_matrix = symbol_features[numeric_cols].fillna(0).values

            # Validate features before inference
            if self._model_registry and self._production_model_key:
                validation = self._model_registry.validate_features(
                    self._production_model_key,
                    list(numeric_cols),
                    strict=False
                )
                if validation["coverage_pct"] < 80:
                    logger.warning(
                        "Feature coverage %.1f%% below threshold for %s, using rule-based fallback",
                        validation["coverage_pct"],
                        symbol,
                    )
                    technical, momentum, volume = self._technical_scores(features)
                    weighted, components = self._combine_scores(
                        technical=technical,
                        sentiment=0.5,
                        fundamental=0.5,
                        momentum=momentum,
                        volume=volume,
                    )
                    return weighted, {"source": "rule_fallback_low_feature_coverage", **components}

            # Model inference
            import torch

            if hasattr(self._loaded_model, "eval"):
                self._loaded_model.eval()

            with torch.no_grad():
                # Shape: (1, seq_len, features)
                x = torch.tensor(feature_matrix, dtype=torch.float32).unsqueeze(0)
                if hasattr(self._loaded_model, "forward"):
                    output = self._loaded_model(x)
                    # Assume output is probability or score in [0, 1]
                    if isinstance(output, torch.Tensor):
                        score = float(output.squeeze().cpu().numpy())
                    else:
                        score = float(output)
                else:
                    # Sklearn-style model
                    score = float(self._loaded_model.predict(feature_matrix[-1:]))

            # Clamp to [0, 1]
            score = max(0.0, min(1.0, score))

            details = {
                "source": "trained_model",
                "model_name": self.model_name,
                "model_version": self._model_metadata.version if self._model_metadata else "unknown",
                "raw_score": score,
                "feature_count": len(numeric_cols),
                "sequence_length": len(symbol_features),
            }

            return score, details

        except Exception as exc:
            logger.error("Model inference failed for %s: %s", symbol, exc)
            # Fall back to rule-based
            technical, momentum, volume = self._technical_scores(features)
            weighted, components = self._combine_scores(
                technical=technical,
                sentiment=0.5,
                fundamental=0.5,
                momentum=momentum,
                volume=volume,
            )
            return weighted, {"source": "rule_fallback_error", "error": str(exc), **components}

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
