"""
Feature engineering utilities for the AI trading system.

This module pulls historical market data from Alpha Vantage and Finnhub,
computes a comprehensive set of technical indicators, and persists them to
DuckDB for downstream consumers. The entry-point helper
``get_feature_matrix`` exposes a clean interface for loading the most recent
feature window for a given symbol universe.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Sequence

import duckdb
import numpy as np
import pandas as pd
import requests

# Phase 3: Import data quality validation
from ..validation.data_quality import get_data_quality_validator, QualityReport
from ..config.database import log_data_provenance, log_data_quality_violation
from ..utils.global_rate_limiter import rate_limiters


logger = logging.getLogger(__name__)

ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"
FINNHUB_CANDLE_URL = "https://finnhub.io/api/v1/stock/candle"

# Alpha Vantage allows 5 calls / minute on the free tier.
ALPHA_MIN_INTERVAL = 12.5
# Finnhub free plan provides ~60 calls / minute.
FINNHUB_MIN_INTERVAL = 1.0


class FeatureEngineeringError(RuntimeError):
    """Raised when feature engineering cannot be completed."""


@dataclass
class FeatureEngineer:
    """
    Build and persist technical indicators for equities.

    Parameters
    ----------
    duckdb_path:
        Location of the DuckDB database that stores computed indicators.
    table_name:
        Table used to persist indicators. Created automatically if needed.
    lookback_days:
        Minimum number of historical days to request when rebuilding features.
    """

    duckdb_path: str = "data/market_data.duckdb"
    table_name: str = "indicators"
    lookback_days: int = 400
    alpha_api_key: Optional[str] = None
    finnhub_api_key: Optional[str] = None
    # Note: Rate limiting is now handled by the global rate_limiters singleton

    def __post_init__(self) -> None:
        self.alpha_api_key = self.alpha_api_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        self.finnhub_api_key = self.finnhub_api_key or os.getenv("FINNHUB_API_KEY")

        if not self.alpha_api_key:
            raise FeatureEngineeringError(
                "Missing Alpha Vantage API key. Set ALPHA_VANTAGE_API_KEY."
            )
        if not self.finnhub_api_key:
            raise FeatureEngineeringError(
                "Missing Finnhub API key. Set FINNHUB_API_KEY."
            )

        Path(self.duckdb_path).parent.mkdir(parents=True, exist_ok=True)
        self._ensure_table()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def calculate_features(self, symbol_or_data, days: int = 30) -> pd.DataFrame:
        """Calculate features for a symbol or from DataFrame (alias for build_for_symbol)."""
        # If passed a DataFrame, return it as-is (for testing)
        if isinstance(symbol_or_data, pd.DataFrame):
            return symbol_or_data
        # Otherwise, treat as symbol and fetch data
        return self.build_for_symbol(symbol_or_data, days)
    
    def build_for_symbol(
        self,
        symbol: str,
        as_of: Optional[datetime] = None,
        min_history: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch prices, compute indicators, and persist them for a given symbol.

        Parameters
        ----------
        symbol:
            Equity ticker (e.g., ``"AAPL"``).
        as_of:
            Optional end timestamp. Defaults to now.
        min_history:
            Minimum number of calendar days of history to pull. Defaults to
            ``self.lookback_days``.
        """

        as_of = as_of or datetime.utcnow()
        history_days = min_history or self.lookback_days
        start = as_of - timedelta(days=history_days)

        price_frame = self._load_price_history(symbol, start=start, end=as_of)
        if price_frame.empty:
            raise FeatureEngineeringError(
                f"No price data retrieved for {symbol}. Unable to build features."
            )

        # Phase 3: Data quality validation before feature calculation
        quality_validator = get_data_quality_validator()
        quality_report = quality_validator.validate_dataframe(price_frame, symbol)
        
        # Log data provenance
        log_data_provenance(
            symbol=symbol,
            data_type="price_data",
            source="alpha_vantage",
            source_metadata={
                "start_date": start.isoformat(),
                "end_date": as_of.isoformat(),
                "rows": len(price_frame),
                "columns": list(price_frame.columns)
            },
            quality_score=quality_report.overall_score,
            quality_level=quality_report.quality_level.value
        )
        
        # Log quality violations
        for violation in quality_report.violations:
            log_data_quality_violation(
                symbol=symbol,
                violation_type=violation.violation_type,
                severity=violation.severity,
                description=violation.description,
                column_name=violation.column,
                violation_value=violation.value,
                expected_range=violation.expected_range
            )
        
        # Check if quality is sufficient for feature calculation
        if quality_validator.should_skip_sizing(quality_report, threshold=0.6):
            logger.warning(f"Data quality for {symbol} is below threshold ({quality_report.overall_score:.2%}). "
                          f"Proceeding with feature calculation but sizing may be skipped.")
        
        # Log quality event
        quality_validator.log_quality_event(quality_report)

        feature_frame = self._compute_indicators(price_frame)
        
        # Phase 3: Validate computed features
        feature_quality_report = quality_validator.validate_dataframe(feature_frame, f"{symbol}_features")
        
        # Log feature data provenance
        log_data_provenance(
            symbol=symbol,
            data_type="technical_indicators",
            source="feature_engineering",
            source_metadata={
                "indicators": list(feature_frame.columns),
                "rows": len(feature_frame),
                "base_quality_score": quality_report.overall_score
            },
            quality_score=feature_quality_report.overall_score,
            quality_level=feature_quality_report.quality_level.value
        )
        
        # Log feature quality violations
        for violation in feature_quality_report.violations:
            log_data_quality_violation(
                symbol=symbol,
                violation_type=f"FEATURE_{violation.violation_type}",
                severity=violation.severity,
                description=f"Feature calculation: {violation.description}",
                column_name=violation.column,
                violation_value=violation.value,
                expected_range=violation.expected_range
            )
        
        self._persist(symbol, feature_frame)

        return feature_frame

    def build_for_universe(
        self,
        symbols: Sequence[str],
        as_of: Optional[datetime] = None,
        min_history: Optional[int] = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Build features for a batch of symbols.

        Returns a dictionary keyed by symbol with the computed feature frames.
        """

        outputs: dict[str, pd.DataFrame] = {}
        for ticker in symbols:
            try:
                outputs[ticker] = self.build_for_symbol(
                    ticker, as_of=as_of, min_history=min_history
                )
            except Exception as exc:  # pragma: no cover - hard failures logged
                logger.exception("Failed to build features for %s: %s", ticker, exc)
        return outputs

    # ------------------------------------------------------------------ #
    # Persistence helpers
    # ------------------------------------------------------------------ #
    def _ensure_table(self) -> None:
        ddl = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            ts TIMESTAMP NOT NULL,
            symbol VARCHAR NOT NULL,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE,
            adjusted_close DOUBLE,
            rsi DOUBLE,
            macd_line DOUBLE,
            macd_signal DOUBLE,
            macd_hist DOUBLE,
            bollinger_mid DOUBLE,
            bollinger_upper DOUBLE,
            bollinger_lower DOUBLE,
            atr DOUBLE,
            adx DOUBLE,
            obv DOUBLE,
            sma_5 DOUBLE,
            sma_20 DOUBLE,
            sma_50 DOUBLE,
            sma_200 DOUBLE,
            ema_5 DOUBLE,
            ema_20 DOUBLE,
            ema_50 DOUBLE,
            ema_200 DOUBLE,
            volume_zscore DOUBLE,
            momentum DOUBLE,
            source JSON,
            PRIMARY KEY (symbol, ts)
        );
        """
        with duckdb.connect(self.duckdb_path) as conn:
            conn.execute(ddl)

    def _persist(self, symbol: str, frame: pd.DataFrame) -> None:
        payload = frame.copy()
        payload.insert(0, "symbol", symbol)
        payload["ts"] = pd.to_datetime(payload.index)
        payload.reset_index(drop=True, inplace=True)
        min_ts = payload["ts"].min()
        max_ts = payload["ts"].max()

        with duckdb.connect(self.duckdb_path) as conn:
            conn.register("incoming_indicators", payload)
            conn.execute(
                f"""
                DELETE FROM {self.table_name}
                WHERE symbol = ?
                  AND ts BETWEEN ? AND ?
                """,
                [symbol, min_ts, max_ts],
            )
            conn.execute(
                f"""
                INSERT INTO {self.table_name} (
                    ts, symbol, open, high, low, close, volume, adjusted_close,
                    rsi, macd_line, macd_signal, macd_hist,
                    bollinger_mid, bollinger_upper, bollinger_lower,
                    atr, adx, obv,
                    sma_5, sma_20, sma_50, sma_200,
                    ema_5, ema_20, ema_50, ema_200,
                    volume_zscore, momentum, source
                )
                SELECT
                    ts, symbol, open, high, low, close, volume, adjusted_close,
                    rsi, macd_line, macd_signal, macd_hist,
                    bollinger_mid, bollinger_upper, bollinger_lower,
                    atr, adx, obv,
                    sma_5, sma_20, sma_50, sma_200,
                    ema_5, ema_20, ema_50, ema_200,
                    volume_zscore, momentum, source
                FROM incoming_indicators
                """,
            )
            conn.unregister("incoming_indicators")

    # ------------------------------------------------------------------ #
    # Data ingestion
    # ------------------------------------------------------------------ #
    def _load_price_history(
        self, symbol: str, start: datetime, end: datetime
    ) -> pd.DataFrame:
        alpha = self._pull_alpha_vantage(symbol)
        finnhub = self._pull_finnhub(symbol, start=start, end=end)

        if alpha.empty and finnhub.empty:
            return pd.DataFrame()

        # Combine both sources, prioritising Alpha Vantage (adjusted data).
        source_info = {"alpha": bool(alpha.shape[0]), "finnhub": bool(finnhub.shape[0])}

        if alpha.empty:
            merged = finnhub
        elif finnhub.empty:
            merged = alpha
        else:
            merged = alpha.combine_first(finnhub)

        merged.attrs["sources"] = source_info

        # Filter to requested window.
        mask = (merged.index >= start) & (merged.index <= end)
        return merged.loc[mask].sort_index()

    def _pull_alpha_vantage(self, symbol: str) -> pd.DataFrame:
        # Use global rate limiter (shared across all modules)
        rate_limiters.alpha_vantage.acquire()

        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "apikey": self.alpha_api_key,
            "outputsize": "full",
        }
        response = requests.get(ALPHA_VANTAGE_URL, params=params, timeout=30)

        if response.status_code != 200:
            logger.error(
                "Alpha Vantage request failed for %s: [%s] %s",
                symbol,
                response.status_code,
                response.text,
            )
            return pd.DataFrame()

        payload = response.json()
        if "Error Message" in payload or "Time Series (Daily)" not in payload:
            logger.warning("Alpha Vantage returned no data for %s", symbol)
            return pd.DataFrame()

        series = payload["Time Series (Daily)"]
        records = []
        for ts, row in series.items():
            records.append(
                {
                    "ts": datetime.fromisoformat(ts),
                    "open": float(row["1. open"]),
                    "high": float(row["2. high"]),
                    "low": float(row["3. low"]),
                    "close": float(row["4. close"]),
                    "adjusted_close": float(row["5. adjusted close"]),
                    "volume": float(row["6. volume"]),
                }
            )
        frame = pd.DataFrame.from_records(records).set_index("ts").sort_index()
        frame["source"] = "alpha_vantage"
        return frame

    def _pull_finnhub(
        self, symbol: str, start: datetime, end: datetime
    ) -> pd.DataFrame:
        # Use global rate limiter (shared across all modules)
        rate_limiters.finnhub.acquire()

        params = {
            "symbol": symbol,
            "resolution": "D",
            "from": int(start.timestamp()),
            "to": int(end.timestamp()),
            "token": self.finnhub_api_key,
        }
        response = requests.get(FINNHUB_CANDLE_URL, params=params, timeout=30)

        if response.status_code != 200:
            logger.error(
                "Finnhub request failed for %s: [%s] %s",
                symbol,
                response.status_code,
                response.text,
            )
            return pd.DataFrame()

        payload = response.json()
        if payload.get("s") != "ok":
            logger.warning("Finnhub returned empty candles for %s", symbol)
            return pd.DataFrame()

        frame = pd.DataFrame(
            {
                "ts": pd.to_datetime(payload["t"], unit="s"),
                "open": payload["o"],
                "high": payload["h"],
                "low": payload["l"],
                "close": payload["c"],
                "volume": payload["v"],
            }
        ).set_index("ts")
        frame["source"] = "finnhub"
        return frame.sort_index()

    # ------------------------------------------------------------------ #
    # Indicator calculations
    # ------------------------------------------------------------------ #
    def _compute_indicators(self, frame: pd.DataFrame) -> pd.DataFrame:
        df = frame.copy()

        if "adjusted_close" not in df.columns:
            df["adjusted_close"] = df["close"]

        # Moving averages
        for window in (5, 20, 50, 200):
            df[f"sma_{window}"] = df["adjusted_close"].rolling(window).mean()
            df[f"ema_{window}"] = df["adjusted_close"].ewm(
                span=window, adjust=False
            ).mean()

        df["momentum"] = df["adjusted_close"].pct_change(periods=10)

        # RSI
        delta = df["adjusted_close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        roll_up = gain.ewm(alpha=1 / 14, adjust=False).mean()
        roll_down = loss.ewm(alpha=1 / 14, adjust=False).mean()
        rs = roll_up / roll_down.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = df["adjusted_close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["adjusted_close"].ewm(span=26, adjust=False).mean()
        df["macd_line"] = ema_12 - ema_26
        df["macd_signal"] = df["macd_line"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd_line"] - df["macd_signal"]

        # Bollinger Bands
        rolling_mean = df["adjusted_close"].rolling(20).mean()
        rolling_std = df["adjusted_close"].rolling(20).std()
        df["bollinger_mid"] = rolling_mean
        df["bollinger_upper"] = rolling_mean + (2 * rolling_std)
        df["bollinger_lower"] = rolling_mean - (2 * rolling_std)

        # ATR
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["adjusted_close"].shift()).abs()
        low_close = (df["low"] - df["adjusted_close"].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = true_range.rolling(14).mean()

        # ADX
        up_move = df["high"].diff()
        down_move = df["low"].diff() * -1
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        tr = true_range.copy()
        atr = tr.rolling(14).mean()
        plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / 14, adjust=False).mean() / atr
        minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / 14, adjust=False).mean() / atr
        dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)).replace(
            [np.inf, -np.inf], np.nan
        ) * 100
        df["adx"] = dx.rolling(14).mean()

        # OBV
        obv = [0.0]
        for i in range(1, len(df)):
            if df["adjusted_close"].iloc[i] > df["adjusted_close"].iloc[i - 1]:
                obv.append(obv[-1] + df["volume"].iloc[i])
            elif df["adjusted_close"].iloc[i] < df["adjusted_close"].iloc[i - 1]:
                obv.append(obv[-1] - df["volume"].iloc[i])
            else:
                obv.append(obv[-1])
        df["obv"] = obv

        # Volume z-score
        volume_mean = df["volume"].rolling(20).mean()
        volume_std = df["volume"].rolling(20).std()
        df["volume_zscore"] = (df["volume"] - volume_mean) / volume_std.replace(0, np.nan)

        # Embed source provenance as JSON.
        source_info = frame.attrs.get(
            "sources", {"alpha": True, "finnhub": True}
        )
        df["source"] = json.dumps(source_info)

        return df[
            [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "adjusted_close",
                "rsi",
                "macd_line",
                "macd_signal",
                "macd_hist",
                "bollinger_mid",
                "bollinger_upper",
                "bollinger_lower",
                "atr",
                "adx",
                "obv",
                "sma_5",
                "sma_20",
                "sma_50",
                "sma_200",
                "ema_5",
                "ema_20",
                "ema_50",
                "ema_200",
                "volume_zscore",
                "momentum",
                "source",
            ]
        ]


def get_feature_matrix(
    symbols: Sequence[str],
    window: int = 90,
    duckdb_path: str = "data/market_data.duckdb",
    table_name: str = "indicators",
) -> pd.DataFrame:
    """
    Load the most recent feature window for the requested symbols.

    Parameters
    ----------
    symbols:
        Iterable of tickers.
    window:
        Number of daily rows to pull, per symbol.
    duckdb_path:
        DuckDB database containing the indicators table.
    table_name:
        Name of the indicators table (defaults to ``indicators``).
    """

    if not symbols:
        return pd.DataFrame()

    with duckdb.connect(duckdb_path, read_only=True) as conn:
        query = f"""
            WITH ranked AS (
                SELECT
                    ts,
                    symbol,
                    open,
                    high,
                    low,
                    close,
                    volume,
                    adjusted_close,
                    rsi,
                    macd_line,
                    macd_signal,
                    macd_hist,
                    bollinger_mid,
                    bollinger_upper,
                    bollinger_lower,
                    atr,
                    adx,
                    obv,
                    sma_5,
                    sma_20,
                    sma_50,
                    sma_200,
                    ema_5,
                    ema_20,
                    ema_50,
                    ema_200,
                    volume_zscore,
                    momentum,
                    source,
                    ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY ts DESC) AS rn
                FROM {table_name}
                WHERE symbol IN ({','.join(['?'] * len(symbols))})
            )
            SELECT *
            FROM ranked
            WHERE rn <= ?
            ORDER BY symbol, ts DESC
        """
        result = conn.execute(query, [*symbols, window]).fetch_df()

    result = result.drop(columns=["rn"]) if "rn" in result.columns else result
    return result


__all__ = ["FeatureEngineer", "FeatureEngineeringError", "get_feature_matrix"]
