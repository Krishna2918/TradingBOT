"""
AI Selector Agent
=================

Coordinates the full data-to-trade workflow using live market data, AI
ensemble voting, and paper execution.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd
import requests
import yaml

from src.ai.features import FeatureEngineer, get_feature_matrix
from src.ai.factors import FactorBuilder
from src.ai.scoring import ScoreEngine
from src.ai.ensemble import EnsembleCoordinator, EnsembleSelection
from src.trading import RiskManager
from src.trading.execution import OrderExecutor as PaperExecutionEngine
from src.utils.global_rate_limiter import rate_limiters


logger = logging.getLogger(__name__)

# Static fallback universe for offline operation (S&P 500 top 50 by market cap)
STATIC_UNIVERSE = [
    {"symbol": "AAPL", "market_cap": 3000000000000, "volume": 50000000, "price": 180.0},
    {"symbol": "MSFT", "market_cap": 2800000000000, "volume": 25000000, "price": 380.0},
    {"symbol": "GOOGL", "market_cap": 1700000000000, "volume": 20000000, "price": 140.0},
    {"symbol": "AMZN", "market_cap": 1600000000000, "volume": 40000000, "price": 175.0},
    {"symbol": "NVDA", "market_cap": 1200000000000, "volume": 50000000, "price": 480.0},
    {"symbol": "META", "market_cap": 900000000000, "volume": 20000000, "price": 350.0},
    {"symbol": "BRK.B", "market_cap": 780000000000, "volume": 3000000, "price": 360.0},
    {"symbol": "LLY", "market_cap": 550000000000, "volume": 3000000, "price": 580.0},
    {"symbol": "JPM", "market_cap": 500000000000, "volume": 10000000, "price": 170.0},
    {"symbol": "V", "market_cap": 490000000000, "volume": 7000000, "price": 260.0},
    {"symbol": "UNH", "market_cap": 480000000000, "volume": 3000000, "price": 520.0},
    {"symbol": "XOM", "market_cap": 450000000000, "volume": 15000000, "price": 105.0},
    {"symbol": "JNJ", "market_cap": 420000000000, "volume": 7000000, "price": 160.0},
    {"symbol": "MA", "market_cap": 400000000000, "volume": 3000000, "price": 420.0},
    {"symbol": "PG", "market_cap": 380000000000, "volume": 7000000, "price": 155.0},
    {"symbol": "HD", "market_cap": 350000000000, "volume": 4000000, "price": 340.0},
    {"symbol": "CVX", "market_cap": 340000000000, "volume": 8000000, "price": 160.0},
    {"symbol": "MRK", "market_cap": 330000000000, "volume": 10000000, "price": 115.0},
    {"symbol": "ABBV", "market_cap": 320000000000, "volume": 7000000, "price": 160.0},
    {"symbol": "COST", "market_cap": 310000000000, "volume": 2000000, "price": 580.0},
    {"symbol": "PEP", "market_cap": 270000000000, "volume": 5000000, "price": 185.0},
    {"symbol": "KO", "market_cap": 260000000000, "volume": 12000000, "price": 60.0},
    {"symbol": "AVGO", "market_cap": 500000000000, "volume": 3000000, "price": 1000.0},
    {"symbol": "ADBE", "market_cap": 260000000000, "volume": 3000000, "price": 580.0},
    {"symbol": "WMT", "market_cap": 440000000000, "volume": 7000000, "price": 165.0},
    {"symbol": "MCD", "market_cap": 210000000000, "volume": 3000000, "price": 295.0},
    {"symbol": "CSCO", "market_cap": 200000000000, "volume": 18000000, "price": 50.0},
    {"symbol": "CRM", "market_cap": 250000000000, "volume": 5000000, "price": 260.0},
    {"symbol": "TMO", "market_cap": 220000000000, "volume": 1500000, "price": 570.0},
    {"symbol": "ACN", "market_cap": 210000000000, "volume": 2000000, "price": 340.0},
    {"symbol": "ORCL", "market_cap": 310000000000, "volume": 8000000, "price": 115.0},
    {"symbol": "LIN", "market_cap": 200000000000, "volume": 1500000, "price": 420.0},
    {"symbol": "ABT", "market_cap": 195000000000, "volume": 5000000, "price": 115.0},
    {"symbol": "DHR", "market_cap": 190000000000, "volume": 2000000, "price": 250.0},
    {"symbol": "NKE", "market_cap": 160000000000, "volume": 8000000, "price": 105.0},
    {"symbol": "INTC", "market_cap": 180000000000, "volume": 35000000, "price": 45.0},
    {"symbol": "DIS", "market_cap": 170000000000, "volume": 10000000, "price": 95.0},
    {"symbol": "VZ", "market_cap": 160000000000, "volume": 15000000, "price": 40.0},
    {"symbol": "CMCSA", "market_cap": 165000000000, "volume": 18000000, "price": 42.0},
    {"symbol": "AMD", "market_cap": 220000000000, "volume": 55000000, "price": 140.0},
    {"symbol": "TXN", "market_cap": 170000000000, "volume": 5000000, "price": 175.0},
    {"symbol": "QCOM", "market_cap": 160000000000, "volume": 6000000, "price": 145.0},
    {"symbol": "PM", "market_cap": 150000000000, "volume": 5000000, "price": 95.0},
    {"symbol": "NEE", "market_cap": 145000000000, "volume": 7000000, "price": 72.0},
    {"symbol": "HON", "market_cap": 135000000000, "volume": 3000000, "price": 205.0},
    {"symbol": "IBM", "market_cap": 155000000000, "volume": 4000000, "price": 170.0},
    {"symbol": "AMGN", "market_cap": 150000000000, "volume": 2500000, "price": 285.0},
    {"symbol": "UNP", "market_cap": 145000000000, "volume": 3000000, "price": 240.0},
    {"symbol": "LOW", "market_cap": 140000000000, "volume": 4000000, "price": 235.0},
    {"symbol": "SPGI", "market_cap": 135000000000, "volume": 1500000, "price": 430.0},
]

FINNHUB_QUOTE_URL = "https://finnhub.io/api/v1/quote"
FINNHUB_METRIC_URL = "https://finnhub.io/api/v1/stock/metric"
FINNHUB_SYMBOL_URL = "https://finnhub.io/api/v1/stock/symbol"


@dataclass
class SelectionResult:
    symbol: str
    score: float
    confidence: float
    entry_price: float
    shares: int
    stop_loss: float
    take_profit: float
    risk_alloc: float
    explanation: str
    details: Dict[str, float] = field(default_factory=dict)


class AISelectorAgent:
    def __init__(
        self,
        risk_config_path: str = "config/risk.yaml",
        universe_cache_path: str = "data/universe_cache.json",
        snapshot_path: Optional[str] = None,
        use_trained_model: bool = False,
        model_name: str = "lstm_production",
        offline_mode: bool = False,
    ) -> None:
        self.risk_config = self._load_risk_config(risk_config_path)
        self.universe_cache_path = Path(universe_cache_path)
        self.cache_dir = Path("data/cache/finnhub")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Snapshot and offline mode settings
        self.snapshot_path = Path(snapshot_path) if snapshot_path else None
        self.offline_mode = offline_mode or (snapshot_path is not None)
        self.use_trained_model = use_trained_model
        self.model_name = model_name

        # In offline mode, API keys are optional
        self.alpha_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        self.finnhub_api_key = os.getenv("FINNHUB_API_KEY")
        self.news_api_key = os.getenv("NEWSAPI_KEY")

        if not self.offline_mode:
            for key, value in (
                ("ALPHA_VANTAGE_API_KEY", self.alpha_api_key),
                ("FINNHUB_API_KEY", self.finnhub_api_key),
                ("NEWSAPI_KEY", self.news_api_key),
            ):
                if not value:
                    raise RuntimeError(f"{key} is required for AI selector (or use --offline-mode).")

        self.duckdb_path = self.risk_config["duckdb_path"]
        self.sqlite_path = self.risk_config["sqlite_path"]
        self.log_path = self.risk_config["log_path"]

        self.feature_engineer = FeatureEngineer(duckdb_path=self.duckdb_path)
        self.factor_builder = FactorBuilder(duckdb_path=self.duckdb_path)
        self.score_engine = ScoreEngine(
            duckdb_path=self.duckdb_path,
            factors=self.factor_builder,
            features=self.feature_engineer,
            use_trained_model=use_trained_model,
            model_name=model_name if use_trained_model else None,
        )
        self.ensemble = EnsembleCoordinator()
        self.risk_manager = RiskManager()  # Uses default config
        self.execution_engine = PaperExecutionEngine()

        self._configure_logging()
        self._ensure_sqlite_tables()

        # Log startup configuration
        self._log_startup_info()

    # ------------------------------------------------------------------ #
    # Public workflow
    # ------------------------------------------------------------------ #
    def run(self, limit: Optional[int] = None) -> List[SelectionResult]:
        # Offline mode: load from snapshot
        if self.offline_mode and self.snapshot_path:
            return self._run_offline(limit=limit)

        # Live mode: fetch from APIs
        return self._run_live(limit=limit)

    def _run_live(self, limit: Optional[int] = None) -> List[SelectionResult]:
        """Run in live mode, fetching data from APIs."""
        universe = self._load_filtered_universe(limit=limit)
        if universe.empty:
            logger.warning("Universe filter returned no eligible symbols.")
            return []

        symbols = universe["symbol"].tolist()

        # Score universe first (this may build/refresh features internally)
        scores_df = self.score_engine.score_universe(symbols)
        if scores_df.empty:
            logger.error("Score engine returned no results.")
            return []

        # Reload feature matrix AFTER scoring to get fresh data
        feature_matrix = get_feature_matrix(symbols, window=90, duckdb_path=self.duckdb_path)

        score_payload = self._prepare_score_payload(scores_df)
        feature_payload = self._prepare_feature_payload(feature_matrix)
        ensemble_choices = self.ensemble.run(feature_payload, score_payload)

        selections = self._build_selections(
            ensemble_choices, score_payload, feature_matrix
        )
        if not selections:
            return []

        self._persist_results(selections)
        self._log_results(selections)
        self.execution_engine.submit_orders(selections)
        return selections

    def _run_offline(self, limit: Optional[int] = None) -> List[SelectionResult]:
        """Run in offline mode using snapshot data (no API calls)."""
        logger.info("Running in OFFLINE mode using snapshot: %s", self.snapshot_path)

        # Get symbols from snapshot
        symbols = self._get_snapshot_symbols()
        if not symbols:
            logger.error("No symbols found in snapshot at %s", self.snapshot_path)
            return []

        if limit:
            symbols = symbols[:limit]

        logger.info("Processing %d symbols from snapshot", len(symbols))

        # Load features from snapshot parquet files
        feature_matrix = self._load_features_from_snapshot(symbols)
        if feature_matrix.empty:
            logger.error("No features loaded from snapshot")
            return []

        # Validate feature columns match model expectations
        if self.use_trained_model:
            self._validate_feature_columns(feature_matrix)

        # Score using snapshot features (pass to score engine)
        scores_df = self.score_engine.score_from_features(feature_matrix, symbols)
        if scores_df.empty:
            logger.warning("Score engine returned no results, using feature-based fallback")
            # Create basic scores from feature data
            scores_df = self._create_fallback_scores(feature_matrix, symbols)

        if scores_df.empty:
            logger.error("No scores available")
            return []

        score_payload = self._prepare_score_payload(scores_df)
        feature_payload = self._prepare_feature_payload(feature_matrix)
        ensemble_choices = self.ensemble.run(feature_payload, score_payload)

        selections = self._build_selections(
            ensemble_choices, score_payload, feature_matrix
        )
        if not selections:
            return []

        self._persist_results(selections)
        self._log_results(selections)
        # In offline mode, we still submit orders (paper trading)
        self.execution_engine.submit_orders(selections)
        return selections

    def _create_fallback_scores(
        self, feature_matrix: pd.DataFrame, symbols: List[str]
    ) -> pd.DataFrame:
        """Create basic scores from feature data when model scoring fails."""
        logger.info("Creating fallback scores from feature data")

        scores = []
        for symbol in symbols:
            sym_data = feature_matrix[feature_matrix["symbol"] == symbol]
            if sym_data.empty:
                continue

            latest = sym_data.iloc[-1]

            # Basic rule-based score from available features
            score = 0.5  # Base score

            # Momentum signals
            if "momentum_20d" in latest:
                mom = latest["momentum_20d"]
                score += 0.1 if mom > 0 else -0.1

            # Volatility adjustment
            if "volatility_20d_annualized" in latest:
                vol = latest["volatility_20d_annualized"]
                if vol < 0.3:
                    score += 0.05  # Lower vol preferred

            # Clamp score
            score = max(0.0, min(1.0, score))

            scores.append({
                "symbol": symbol,
                "score": score,
                "details_json": json.dumps({
                    "technical": score,
                    "sentiment": 0.5,
                    "fundamental": 0.5,
                    "momentum": score,
                    "volume": 0.5,
                }),
            })

        return pd.DataFrame(scores)

    # ------------------------------------------------------------------ #
    # Selection assembly
    # ------------------------------------------------------------------ #
    def _build_selections(
        self,
        ensemble_choices: List[EnsembleSelection],
        score_payload: Dict[str, dict],
        feature_matrix: pd.DataFrame,
    ) -> List[SelectionResult]:
        selections: List[SelectionResult] = []
        if not feature_matrix.empty:
            # Handle snapshots without 'ts' column
            if "ts" in feature_matrix.columns:
                latest_features = feature_matrix.sort_values(["symbol", "ts"]).groupby("symbol").tail(1).set_index("symbol")
            else:
                latest_features = feature_matrix.groupby("symbol").tail(1).set_index("symbol")
        else:
            latest_features = pd.DataFrame()

        for pick in ensemble_choices:
            symbol = pick.symbol
            if symbol not in score_payload:
                logger.debug("Skipping %s - no score payload", symbol)
                continue
            details = score_payload[symbol]["details"]
            score_value = float(score_payload[symbol]["score"])
            quote = self._fetch_quote(symbol)
            price = quote["price"]
            if price <= 0:
                logger.debug("Skipping %s - invalid price", symbol)
                continue

            sizing = self._kelly_sizing(pick.confidence, price)
            if sizing["shares"] <= 0:
                logger.debug("Skipping %s - Kelly sizing zero", symbol)
                continue

            feature_detail = (
                latest_features.loc[symbol].to_dict() if symbol in latest_features.index else {}
            )
            explanation = self._build_explanation(symbol, details, pick.explanation, feature_detail)
            selections.append(
                SelectionResult(
                    symbol=symbol,
                    score=score_value,
                    confidence=pick.confidence,
                    entry_price=price,
                    shares=sizing["shares"],
                    stop_loss=sizing["stop_price"],
                    take_profit=sizing["take_profit"],
                    risk_alloc=sizing["risk_fraction"],
                    explanation=explanation,
                    details={
                        "votes": pick.votes,
                        "ensemble_explanation": pick.explanation,
                        "components": details,
                    },
                )
            )
        selections.sort(key=lambda s: s.score, reverse=True)
        return selections[:5]

    def _prepare_score_payload(self, scores_df: pd.DataFrame) -> Dict[str, dict]:
        payload: Dict[str, dict] = {}
        ranked = scores_df.sort_values("score", ascending=False).head(40)
        for _, row in ranked.iterrows():
            details = json.loads(row["details_json"])
            payload[row["symbol"]] = {"score": float(row["score"]), "details": details}
        return payload

    def _prepare_feature_payload(self, feature_matrix: pd.DataFrame) -> Dict[str, dict]:
        payload: Dict[str, dict] = {}
        if feature_matrix.empty:
            return payload
        # Handle snapshots without 'ts' column
        if "ts" in feature_matrix.columns:
            latest = feature_matrix.sort_values(["symbol", "ts"]).groupby("symbol").tail(1)
        else:
            latest = feature_matrix.groupby("symbol").tail(1)
        for _, row in latest.iterrows():
            payload[row["symbol"]] = {
                "ts": str(row.get("ts", "")),
                "close": float(row.get("close", 0.0)),
                "rsi": float(row.get("rsi", 0.0)),
                "macd_hist": float(row.get("macd_hist", 0.0)),
                "adx": float(row.get("adx", 0.0)),
                "momentum": float(row.get("momentum", 0.0)),
                "volume_zscore": float(row.get("volume_zscore", 0.0)),
            }
        return payload

    # ------------------------------------------------------------------ #
    # Universe construction
    # ------------------------------------------------------------------ #
    def _load_filtered_universe(self, limit: Optional[int] = None) -> pd.DataFrame:
        # Try cache first
        if self.universe_cache_path.exists():
            with open(self.universe_cache_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            cache_ts = datetime.fromisoformat(payload["timestamp"])
            if cache_ts.date() == datetime.utcnow().date():
                universe_df = pd.DataFrame(payload["universe"])
                if limit:
                    universe_df = universe_df.head(limit)
                return universe_df

        # Try to fetch from Finnhub, with fallback to static universe
        try:
            raw_symbols = self._fetch_us_symbols()
        except Exception as exc:
            logger.warning(
                "Finnhub symbol fetch failed (%s), using static universe fallback.", exc
            )
            return self._get_static_universe(limit)

        if not raw_symbols:
            logger.warning("Finnhub returned empty symbol list, using static universe fallback.")
            return self._get_static_universe(limit)

        records = []
        for entry in raw_symbols:
            if entry["type"] not in {"Common Stock", "REIT", "ETF"}:
                continue
            if entry["mic"] not in {"XNAS", "XNYS", "ARCX", "XASE"}:
                continue
            snapshot = self._fetch_symbol_snapshot(entry["symbol"])
            if not snapshot:
                continue

            if snapshot["market_cap"] < 1_000_000_000:
                continue
            if snapshot["volume"] < 1_000_000:
                continue
            if snapshot["price"] < 5:
                continue

            records.append(
                {
                    "symbol": entry["symbol"],
                    "market_cap": snapshot["market_cap"],
                    "volume": snapshot["volume"],
                    "price": snapshot["price"],
                    "quote_raw": snapshot["quote_raw"],
                }
            )
            if limit and len(records) >= limit:
                break

        # If no records from API, fall back to static universe
        if not records:
            logger.warning("No valid symbols from Finnhub, using static universe fallback.")
            return self._get_static_universe(limit)

        universe_df = pd.DataFrame(records)
        if not universe_df.empty:
            with open(self.universe_cache_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "timestamp": datetime.utcnow().isoformat(),
                        "universe": universe_df.to_dict(orient="records"),
                    },
                    handle,
                    indent=2,
                )
        return universe_df

    def _get_static_universe(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Return the static fallback universe for offline operation.

        Loads from data/static_universe.json if available, otherwise falls back
        to the hardcoded STATIC_UNIVERSE constant.
        """
        records = []

        # Try to load from JSON file first
        static_json_path = Path("data/static_universe.json")
        if static_json_path.exists():
            try:
                with open(static_json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Convert JSON format to expected format
                for item in data.get("symbols", []):
                    records.append({
                        "symbol": item["symbol"],
                        "market_cap": item.get("market_cap_b", 100) * 1_000_000_000,
                        "volume": 10_000_000,  # Default volume
                        "price": 100.0,  # Placeholder - will be fetched later
                    })
                logger.info("Loaded static universe from %s", static_json_path)
            except Exception as exc:
                logger.warning("Failed to load static_universe.json: %s, using fallback", exc)
                records = STATIC_UNIVERSE.copy()
        else:
            # Fall back to hardcoded constant
            records = STATIC_UNIVERSE.copy()

        if limit:
            records = records[:limit]

        df = pd.DataFrame(records)
        logger.info("Using static universe with %d symbols.", len(df))
        return df

    def _fetch_us_symbols(self) -> List[Dict[str, str]]:
        params = {"exchange": "US", "token": self.finnhub_api_key}
        response = requests.get(FINNHUB_SYMBOL_URL, params=params, timeout=30)
        response.raise_for_status()
        return response.json()

    def _fetch_symbol_snapshot(self, symbol: str) -> Optional[Dict[str, float]]:
        cache_file = self.cache_dir / f"{symbol}_snapshot.json"
        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            cache_ts = datetime.fromisoformat(payload["timestamp"])
            if cache_ts > datetime.utcnow() - timedelta(hours=8):
                return payload["snapshot"]

        metrics = self._finnhub_metric(symbol)
        quote = self._fetch_quote(symbol)
        if not metrics or not quote:
            return None

        snapshot = {
            "market_cap": metrics.get("marketCapitalization") or 0.0,
            "volume": quote["volume"],
            "price": quote["price"],
            "quote_raw": quote,
        }
        if snapshot["market_cap"] <= 0 or snapshot["volume"] <= 0 or snapshot["price"] <= 0:
            return None

        with open(cache_file, "w", encoding="utf-8") as handle:
            json.dump(
                {"timestamp": datetime.utcnow().isoformat(), "snapshot": snapshot},
                handle,
                indent=2,
            )
        return snapshot

    # ------------------------------------------------------------------ #
    # External data fetchers
    # ------------------------------------------------------------------ #
    def _finnhub_metric(self, symbol: str) -> Dict[str, float]:
        # Use global rate limiter (shared across all modules)
        rate_limiters.finnhub.acquire()
        params = {"symbol": symbol, "metric": "all", "token": self.finnhub_api_key}
        response = requests.get(FINNHUB_METRIC_URL, params=params, timeout=30)
        response.raise_for_status()
        return response.json().get("metric", {})

    def _fetch_quote(self, symbol: str) -> Dict[str, float]:
        # Use global rate limiter (shared across all modules)
        rate_limiters.finnhub.acquire()
        params = {"symbol": symbol, "token": self.finnhub_api_key}
        response = requests.get(FINNHUB_QUOTE_URL, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        return {
            "price": payload.get("c") or payload.get("pc") or 0.0,
            "volume": payload.get("v") or 0.0,
        }

    # ------------------------------------------------------------------ #
    # Kelly sizing & persistence
    # ------------------------------------------------------------------ #
    def _kelly_sizing(self, confidence: float, price: float) -> Dict[str, float]:
        cfg = self.risk_config
        total_capital = cfg["total_capital"]
        max_risk_pct = cfg["max_position_risk_pct"]
        stop_loss_pct = cfg["stop_loss_pct"]
        take_profit_multiple = cfg["take_profit_multiple"]
        min_conf = cfg["min_confidence"]
        conf_low, conf_high = cfg["clamp_confidence"]

        probability = max(min(confidence, conf_high), conf_low)
        if probability < min_conf or price <= 0:
            return {
                "confidence": probability,
                "risk_fraction": 0.0,
                "shares": 0,
                "stop_price": price * (1 - stop_loss_pct) if price else 0.0,
                "take_profit": price * (1 + stop_loss_pct * take_profit_multiple)
                if price
                else 0.0,
            }

        reward_risk_ratio = take_profit_multiple
        kelly_fraction = probability - (1 - probability) / reward_risk_ratio
        kelly_fraction = max(kelly_fraction, 0.0) * cfg["kelly_sensitivity"]
        risk_fraction = min(kelly_fraction, max_risk_pct)

        per_share_risk = price * stop_loss_pct
        dollar_risk = total_capital * risk_fraction
        shares = math.floor(dollar_risk / per_share_risk) if per_share_risk else 0

        stop_price = price * (1 - stop_loss_pct)
        take_profit = price * (1 + stop_loss_pct * take_profit_multiple)

        return {
            "confidence": probability,
            "risk_fraction": risk_fraction,
            "shares": shares,
            "stop_price": stop_price,
            "take_profit": take_profit,
        }

    def _persist_results(self, selections: Sequence[SelectionResult]) -> None:
        trade_dt = date.today().isoformat()
        with sqlite3.connect(self.sqlite_path) as conn:
            cursor = conn.cursor()
            for result in selections:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO ai_selections (
                        trade_date, symbol, score, explanation, shares, confidence
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        trade_dt,
                        result.symbol,
                        float(result.score),
                        result.explanation,
                        int(result.shares),
                        float(result.confidence),
                    ),
                )
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO positions (
                        trade_date, symbol, entry_price, shares,
                        stop_loss, take_profit, risk_alloc
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        trade_dt,
                        result.symbol,
                        float(result.entry_price),
                        int(result.shares),
                        float(result.stop_loss),
                        float(result.take_profit),
                        float(result.risk_alloc),
                    ),
                )
            conn.commit()

    def _log_results(self, selections: Sequence[SelectionResult]) -> None:
        for result in selections:
            logger.info(
                "AI selection %s | score=%.3f confidence=%.3f shares=%s entry=%.2f stop=%.2f tp=%.2f",
                result.symbol,
                result.score,
                result.confidence,
                result.shares,
                result.entry_price,
                result.stop_loss,
                result.take_profit,
            )

    def _build_explanation(
        self,
        symbol: str,
        detail: Dict[str, float],
        ensemble_explanation: str,
        feature_detail: Dict[str, float],
    ) -> str:
        msg = (
            f"{symbol}: ensemble=\"{ensemble_explanation}\" | "
            f"technical={detail.get('technical'):.2f} "
            f"sentiment={detail.get('sentiment'):.2f} "
            f"fundamental={detail.get('fundamental'):.2f} "
            f"momentum={detail.get('momentum'):.2f} "
            f"volume={detail.get('volume'):.2f} "
            f"(close={feature_detail.get('close')})."
        )
        return msg

    # ------------------------------------------------------------------ #
    # Config / logging setup
    # ------------------------------------------------------------------ #
    def _load_risk_config(self, path: str) -> Dict[str, float]:
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)["risk"]
        required = [
            "total_capital",
            "max_position_risk_pct",
            "stop_loss_pct",
            "take_profit_multiple",
            "kelly_sensitivity",
            "min_confidence",
            "clamp_confidence",
            "sqlite_path",
            "duckdb_path",
            "log_path",
        ]
        for field in required:
            if field not in data:
                raise ValueError(f"risk config missing {field}")
        return data

    def _configure_logging(self) -> None:
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
        if not any(
            isinstance(handler, logging.FileHandler) and handler.baseFilename == self.log_path
            for handler in logger.handlers
        ):
            file_handler = logging.FileHandler(self.log_path)
            formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)

    def _ensure_sqlite_tables(self) -> None:
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

    def _log_startup_info(self) -> None:
        """Log startup configuration for operator visibility."""
        logger.info("=" * 60)
        logger.info("AI SELECTOR AGENT STARTUP")
        logger.info("=" * 60)

        # Mode information
        mode = "OFFLINE" if self.offline_mode else "LIVE"
        scoring_mode = "MODEL-BASED" if self.use_trained_model else "RULE-BASED"
        logger.info("Mode: %s", mode)
        logger.info("Scoring: %s", scoring_mode)

        # Snapshot information
        if self.snapshot_path:
            manifest_path = self.snapshot_path / "FEATURE_MANIFEST.json"
            if manifest_path.exists():
                try:
                    with open(manifest_path, "r", encoding="utf-8") as f:
                        manifest = json.load(f)
                    logger.info("Snapshot: %s", self.snapshot_path.name)
                    logger.info("  Created: %s", manifest.get("created_at", "unknown"))
                    logger.info("  Symbols: %d", manifest.get("total_symbols_analyzed", 0))
                    logger.info("  Features: %d", len(manifest.get("features", [])))
                except Exception as e:
                    logger.warning("Failed to read snapshot manifest: %s", e)
            else:
                logger.info("Snapshot: %s (no manifest found)", self.snapshot_path)
        else:
            logger.info("Snapshot: None (using live data)")

        # Model information
        if self.use_trained_model:
            registry_path = Path("models/registry/registry.json")
            if registry_path.exists():
                try:
                    with open(registry_path, "r", encoding="utf-8") as f:
                        registry = json.load(f)
                    prod_model = registry.get("production_model", "none")
                    logger.info("Production Model: %s", prod_model)
                    if prod_model in registry.get("models", {}):
                        model_info = registry["models"][prod_model]
                        logger.info("  Type: %s", model_info.get("model_type", "unknown"))
                        logger.info("  Version: %s", model_info.get("version", "unknown"))
                        logger.info("  Git Commit: %s", model_info.get("git_commit", "unknown"))
                        logger.info("  Snapshot ID: %s", model_info.get("data_snapshot_id", "none"))
                except Exception as e:
                    logger.warning("Failed to read model registry: %s", e)
            else:
                logger.warning("Model registry not found at %s", registry_path)
        else:
            logger.info("Model: None (rule-based scoring)")

        logger.info("=" * 60)

    def _load_features_from_snapshot(self, symbols: List[str]) -> pd.DataFrame:
        """Load features from snapshot parquet files."""
        if not self.snapshot_path or not self.snapshot_path.exists():
            logger.warning("No snapshot path configured, cannot load from snapshot")
            return pd.DataFrame()

        all_features = []
        for symbol in symbols:
            parquet_path = self.snapshot_path / f"{symbol}_features.parquet"
            if parquet_path.exists():
                try:
                    import pandas as pd
                    df = pd.read_parquet(parquet_path)
                    df["symbol"] = symbol
                    all_features.append(df)
                except Exception as e:
                    logger.warning("Failed to load snapshot for %s: %s", symbol, e)

        if all_features:
            combined = pd.concat(all_features, ignore_index=True)
            logger.info("Loaded %d rows from snapshot for %d symbols", len(combined), len(all_features))
            return combined

        logger.warning("No features loaded from snapshot")
        return pd.DataFrame()

    def _get_snapshot_symbols(self) -> List[str]:
        """Get list of symbols available in the snapshot."""
        if not self.snapshot_path or not self.snapshot_path.exists():
            return []

        symbols = []
        for parquet_file in self.snapshot_path.glob("*_features.parquet"):
            symbol = parquet_file.stem.replace("_features", "")
            symbols.append(symbol)

        logger.info("Found %d symbols in snapshot", len(symbols))
        return symbols

    def _validate_feature_columns(self, feature_matrix: pd.DataFrame) -> None:
        """
        Validate that snapshot feature columns cover the model's expected features.

        Uses ModelRegistry.validate_features() for consistent validation logic.
        """
        try:
            from src.ai.model_registry import ModelRegistry

            registry = ModelRegistry(storage_path="models/registry")
            prod_key = registry._production_model_key

            if not prod_key:
                logger.warning("Cannot validate features: no production model in registry")
                return

            # Use registry's validate_features method
            validation = registry.validate_features(
                prod_key,
                list(feature_matrix.columns),
                strict=False  # Warn but don't raise
            )

            # Log additional context for low coverage
            if validation["coverage_pct"] < 80:
                logger.error(
                    "LOW FEATURE COVERAGE (%.1f%%) - model predictions may be unreliable!",
                    validation["coverage_pct"],
                )

        except Exception as e:
            logger.warning("Feature validation failed: %s", e)


__all__ = ["AISelectorAgent", "SelectionResult"]
