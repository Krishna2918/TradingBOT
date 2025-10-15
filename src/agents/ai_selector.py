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
from src.trading import PaperExecutionEngine, RiskManager


logger = logging.getLogger(__name__)

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
    ) -> None:
        self.risk_config = self._load_risk_config(risk_config_path)
        self.universe_cache_path = Path(universe_cache_path)
        self.cache_dir = Path("data/cache/finnhub")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.alpha_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        self.finnhub_api_key = os.getenv("FINNHUB_API_KEY")
        self.news_api_key = os.getenv("NEWSAPI_KEY")

        for key, value in (
            ("ALPHA_VANTAGE_API_KEY", self.alpha_api_key),
            ("FINNHUB_API_KEY", self.finnhub_api_key),
            ("NEWSAPI_KEY", self.news_api_key),
        ):
            if not value:
                raise RuntimeError(f"{key} is required for AI selector.")

        self.duckdb_path = self.risk_config["duckdb_path"]
        self.sqlite_path = self.risk_config["sqlite_path"]
        self.log_path = self.risk_config["log_path"]

        self.feature_engineer = FeatureEngineer(duckdb_path=self.duckdb_path)
        self.factor_builder = FactorBuilder(duckdb_path=self.duckdb_path)
        self.score_engine = ScoreEngine(
            duckdb_path=self.duckdb_path,
            factors=self.factor_builder,
            features=self.feature_engineer,
        )
        self.ensemble = EnsembleCoordinator()
        self.risk_manager = RiskManager(config_path=risk_config_path)
        self.execution_engine = PaperExecutionEngine(risk=self.risk_manager)

        self._last_finnhub_call: float = 0.0

        self._configure_logging()
        self._ensure_sqlite_tables()

    # ------------------------------------------------------------------ #
    # Public workflow
    # ------------------------------------------------------------------ #
    def run(self, limit: Optional[int] = None) -> List[SelectionResult]:
        universe = self._load_filtered_universe(limit=limit)
        if universe.empty:
            logger.warning("Universe filter returned no eligible symbols.")
            return []

        symbols = universe["symbol"].tolist()
        feature_matrix = get_feature_matrix(symbols, window=90, duckdb_path=self.duckdb_path)
        scores_df = self.score_engine.score_universe(symbols)
        if scores_df.empty:
            logger.error("Score engine returned no results.")
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
        self.execution_engine.submit_orders(selections)
        return selections

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
        latest_features = (
            feature_matrix.sort_values(["symbol", "ts"]).groupby("symbol").tail(1).set_index("symbol")
            if not feature_matrix.empty
            else pd.DataFrame()
        )

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
        latest = feature_matrix.sort_values(["symbol", "ts"]).groupby("symbol").tail(1)
        for _, row in latest.iterrows():
            payload[row["symbol"]] = {
                "ts": str(row["ts"]),
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
        if self.universe_cache_path.exists():
            with open(self.universe_cache_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            cache_ts = datetime.fromisoformat(payload["timestamp"])
            if cache_ts.date() == datetime.utcnow().date():
                universe_df = pd.DataFrame(payload["universe"])
                if limit:
                    universe_df = universe_df.head(limit)
                return universe_df

        raw_symbols = self._fetch_us_symbols()
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
        self._respect_rate_limit()
        params = {"symbol": symbol, "metric": "all", "token": self.finnhub_api_key}
        response = requests.get(FINNHUB_METRIC_URL, params=params, timeout=30)
        self._last_finnhub_call = time.time()
        response.raise_for_status()
        return response.json().get("metric", {})

    def _fetch_quote(self, symbol: str) -> Dict[str, float]:
        self._respect_rate_limit()
        params = {"symbol": symbol, "token": self.finnhub_api_key}
        response = requests.get(FINNHUB_QUOTE_URL, params=params, timeout=30)
        self._last_finnhub_call = time.time()
        response.raise_for_status()
        payload = response.json()
        return {
            "price": payload.get("c") or payload.get("pc") or 0.0,
            "volume": payload.get("v") or 0.0,
        }

    def _respect_rate_limit(self) -> None:
        wait = 1.0 - (time.time() - self._last_finnhub_call)
        if wait > 0:
            time.sleep(wait)

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
                    INSERT INTO ai_selections (
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
                    INSERT INTO positions (
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
            conn.commit()


__all__ = ["AISelectorAgent", "SelectionResult"]
