"""
Factor construction for sentiment and fundamental signals.

This module pulls live data from NewsAPI and Finnhub, normalises the factors,
and persists them to DuckDB tables that power the downstream scoring engine.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import duckdb
import numpy as np
import pandas as pd
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.utils.global_rate_limiter import rate_limiters


logger = logging.getLogger(__name__)

NEWS_TABLE = "news_sentiment"
FUND_TABLE = "fundamentals"

NEWSAPI_URL = "https://newsapi.org/v2/everything"
FINNHUB_METRIC_URL = "https://finnhub.io/api/v1/stock/metric"
FINNHUB_RECO_URL = "https://finnhub.io/api/v1/stock/recommendation"


class FactorError(RuntimeError):
    """Raised when factor construction fails."""


@dataclass
class FactorBuilder:
    """
    Build sentiment and fundamental factors for equities.

    Parameters
    ----------
    duckdb_path:
        Path to the DuckDB file that stores factor tables.
    news_window_hours:
        Time window for aggregating NewsAPI articles.
    """

    duckdb_path: str = "data/market_data.duckdb"
    news_window_hours: int = 48
    news_table: str = NEWS_TABLE
    fundamentals_table: str = FUND_TABLE
    news_api_key: Optional[str] = None
    finnhub_api_key: Optional[str] = None
    _sentiment: SentimentIntensityAnalyzer = field(
        default_factory=SentimentIntensityAnalyzer, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self.news_api_key = self.news_api_key or os.getenv("NEWSAPI_KEY")
        self.finnhub_api_key = self.finnhub_api_key or os.getenv("FINNHUB_API_KEY")

        if not self.news_api_key:
            raise FactorError("Missing NEWSAPI_KEY for sentiment processing.")
        if not self.finnhub_api_key:
            raise FactorError("Missing FINNHUB_API_KEY for fundamentals.")

        Path(self.duckdb_path).parent.mkdir(parents=True, exist_ok=True)
        self._ensure_tables()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def build_for_symbol(self, symbol: str) -> Tuple[float, float]:
        """
        Compute sentiment and fundamentals for a single symbol.

        Returns
        -------
        Tuple[float, float]
            Normalised sentiment (-1, 1) and fundamental (0, 1) scores.
        """

        sentiment_score, articles = self._fetch_sentiment(symbol)
        fund_score, fund_payload = self._fetch_fundamentals(symbol)

        self._persist_sentiment(symbol, sentiment_score, articles)
        self._persist_fundamentals(symbol, fund_score, fund_payload)

        return sentiment_score, fund_score

    def build_for_universe(self, symbols: Sequence[str]) -> pd.DataFrame:
        """
        Construct factors for multiple symbols and return the latest values.
        """

        records = []
        for symbol in symbols:
            try:
                sentiment, fundamental = self.build_for_symbol(symbol)
                records.append({"symbol": symbol, "sentiment": sentiment, "fundamental": fundamental})
            except Exception as exc:  # pragma: no cover - upstream errors logged
                logger.exception("Factor build failed for %s: %s", symbol, exc)
        return pd.DataFrame(records)

    def latest_factors(self, symbols: Sequence[str]) -> pd.DataFrame:
        """
        Load most recent sentiment and fundamentals for the given symbols.
        """

        if not symbols:
            return pd.DataFrame()

        with duckdb.connect(self.duckdb_path, read_only=True) as conn:
            query = f"""
                WITH latest_news AS (
                    SELECT *, ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY ts DESC) AS rn
                    FROM {self.news_table}
                    WHERE symbol IN ({','.join(['?'] * len(symbols))})
                ),
                latest_fund AS (
                    SELECT *, ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY ts DESC) AS rn
                    FROM {self.fundamentals_table}
                    WHERE symbol IN ({','.join(['?'] * len(symbols))})
                )
                SELECT
                    COALESCE(n.symbol, f.symbol) AS symbol,
                    n.sentiment_score,
                n.article_count,
                    n.details_json AS sentiment_details,
                    f.fundamental_score,
                    f.components_json AS fundamental_details,
                    COALESCE(n.ts, f.ts) AS ts
                FROM latest_news n
                FULL OUTER JOIN latest_fund f
                    ON n.symbol = f.symbol
                WHERE (n.rn IS NULL OR n.rn = 1)
                  AND (f.rn IS NULL OR f.rn = 1)
            """
            frame = conn.execute(query, [*symbols, *symbols]).fetch_df()
        return frame

    # ------------------------------------------------------------------ #
    # Table setup / persistence
    # ------------------------------------------------------------------ #
    def _ensure_tables(self) -> None:
        news_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.news_table} (
            ts TIMESTAMP NOT NULL,
            symbol VARCHAR NOT NULL,
            sentiment_score DOUBLE NOT NULL,
            article_count INTEGER NOT NULL,
            details_json JSON,
            PRIMARY KEY (symbol, ts)
        )
        """
        fund_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.fundamentals_table} (
            ts TIMESTAMP NOT NULL,
            symbol VARCHAR NOT NULL,
            fundamental_score DOUBLE NOT NULL,
            components_json JSON,
            PRIMARY KEY (symbol, ts)
        )
        """
        with duckdb.connect(self.duckdb_path) as conn:
            conn.execute(news_sql)
            conn.execute(fund_sql)

    def _persist_sentiment(
        self, symbol: str, score: float, articles: List[dict]
    ) -> None:
        if not articles:
            return
        ts = datetime.utcnow()
        payload = {
            "timestamp": ts.isoformat(),
            "articles": articles,
        }
        with duckdb.connect(self.duckdb_path) as conn:
            conn.execute(
                f"""
                INSERT OR REPLACE INTO {self.news_table}
                (ts, symbol, sentiment_score, article_count, details_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (ts, symbol, float(score), len(articles), json.dumps(payload)),
            )

    def _persist_fundamentals(
        self, symbol: str, score: float, components: Dict[str, float]
    ) -> None:
        ts = datetime.utcnow()
        with duckdb.connect(self.duckdb_path) as conn:
            conn.execute(
                f"""
                INSERT OR REPLACE INTO {self.fundamentals_table}
                (ts, symbol, fundamental_score, components_json)
                VALUES (?, ?, ?, ?)
                """,
                (ts, symbol, float(score), json.dumps(components)),
            )

    # ------------------------------------------------------------------ #
    # Sentiment
    # ------------------------------------------------------------------ #
    def _fetch_sentiment(self, symbol: str) -> Tuple[float, List[dict]]:
        # Use global rate limiter (shared across all modules)
        rate_limiters.newsapi.acquire()

        params = {
            "q": symbol,
            "language": "en",
            "pageSize": 25,
            "sortBy": "publishedAt",
            "from": (datetime.utcnow() - timedelta(hours=self.news_window_hours)).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "apiKey": self.news_api_key,
        }
        response = requests.get(NEWSAPI_URL, params=params, timeout=30)
        if response.status_code == 429:
            raise FactorError("NewsAPI rate limit hit; wait before retrying.")
        response.raise_for_status()
        payload = response.json()
        articles = payload.get("articles", [])
        if not articles:
            return 0.0, []

        weighted_scores: List[float] = []
        weights: List[float] = []
        final_articles: List[dict] = []
        now = datetime.utcnow()
        for article in articles:
            content = " ".join(
                filter(
                    None,
                    [
                        article.get("title", ""),
                        article.get("description", ""),
                        article.get("content", ""),
                    ],
                )
            )
            if not content.strip():
                continue
            compound = self._sentiment.polarity_scores(content)["compound"]
            published_at = article.get("publishedAt")
            weight = 1.0
            if published_at:
                try:
                    published_dt = datetime.fromisoformat(
                        published_at.replace("Z", "+00:00")
                    )
                    hours_old = max((now - published_dt).total_seconds() / 3600, 1)
                    weight = min(1.0, 6 / hours_old)
                except ValueError:
                    weight = 1.0
            weighted_scores.append(compound * weight)
            weights.append(weight)
            final_articles.append(
                {
                    "title": article.get("title"),
                    "url": article.get("url"),
                    "publishedAt": published_at,
                    "compound": compound,
                    "weight": weight,
                }
            )

        if not weighted_scores:
            return 0.0, final_articles

        weight_sum = sum(weights) or 1.0
        avg_score = float(np.clip(sum(weighted_scores) / weight_sum, -1, 1))
        return avg_score, final_articles

    # ------------------------------------------------------------------ #
    # Fundamentals
    # ------------------------------------------------------------------ #
    def _fetch_fundamentals(self, symbol: str) -> Tuple[float, Dict[str, float]]:
        metrics = self._finnhub_metric(symbol)
        reco = self._finnhub_recommendation(symbol)

        components: Dict[str, float] = {}
        if metrics:
            components.update(
                {
                    "pe_normalized": metrics.get("peNormalizedAnnual"),
                    "pb": metrics.get("pbAnnual"),
                    "debt_to_equity": metrics.get("totalDebt/totalEquityAnnual"),
                    "roe": metrics.get("roeTTM"),
                    "gross_margin": metrics.get("grossMarginTTM"),
                    "eps_growth": metrics.get("epsGrowthTTMYoy"),
                }
            )
        if reco:
            components["analyst_buy"] = reco.get("buy")
            components["analyst_sell"] = reco.get("sell")
            components["analyst_hold"] = reco.get("hold")

        score = self._normalise_fundamental_components(components)
        components["score"] = score
        return score, components

    def _finnhub_metric(self, symbol: str) -> Dict[str, float]:
        # Use global rate limiter (shared across all instances)
        rate_limiters.finnhub.acquire()
        params = {
            "symbol": symbol,
            "metric": "all",
            "token": self.finnhub_api_key,
        }
        response = requests.get(FINNHUB_METRIC_URL, params=params, timeout=30)
        response.raise_for_status()
        return response.json().get("metric", {})

    def _finnhub_recommendation(self, symbol: str) -> Dict[str, float]:
        # Use global rate limiter (shared across all instances)
        rate_limiters.finnhub.acquire()
        params = {"symbol": symbol, "token": self.finnhub_api_key}
        response = requests.get(FINNHUB_RECO_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        if not data:
            return {}
        latest = data[0]
        total = max(latest.get("buy", 0) + latest.get("hold", 0) + latest.get("sell", 0), 1)
        return {
            "buy": latest.get("buy", 0) / total,
            "hold": latest.get("hold", 0) / total,
            "sell": latest.get("sell", 0) / total,
        }

    def _normalise_fundamental_components(self, components: Dict[str, float]) -> float:
        scores: List[float] = []

        pe = components.get("pe_normalized")
        if pe and pe > 0:
            scores.append(np.clip((40 - min(pe, 80)) / 40, 0, 1))

        pb = components.get("pb")
        if pb and pb > 0:
            scores.append(np.clip((6 - min(pb, 12)) / 6, 0, 1))

        debt = components.get("debt_to_equity")
        if debt and debt > 0:
            scores.append(np.clip(1 - (min(debt, 200) / 200), 0, 1))

        roe = components.get("roe")
        if roe:
            scores.append(np.clip(roe / 25, 0, 1))

        margin = components.get("gross_margin")
        if margin:
            scores.append(np.clip(margin / 60, 0, 1))

        eps_growth = components.get("eps_growth")
        if eps_growth is not None:
            scores.append(np.clip((eps_growth + 0.25) / 0.5, 0, 1))

        buy = components.get("analyst_buy")
        sell = components.get("analyst_sell")
        if buy is not None and sell is not None:
            scores.append(np.clip(buy - sell, 0, 1))

        if not scores:
            return 0.5
        return float(np.nanmean(scores))



def get_factor_matrix(
    symbols: Sequence[str],
    duckdb_path: str = "data/market_data.duckdb",
) -> pd.DataFrame:
    """
    Convenience wrapper around :meth:`FactorBuilder.latest_factors`.
    """

    builder = FactorBuilder(duckdb_path=duckdb_path)
    return builder.latest_factors(symbols)


__all__ = ["FactorBuilder", "FactorError", "get_factor_matrix"]
