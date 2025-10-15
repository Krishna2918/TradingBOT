"""AI ensemble selector implementation for the trading scaffold."""

from __future__ import annotations

import json
import logging
import random
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence, Optional

import requests

from src.config import get_settings
from src.utils.db import SQLITE_PATH, bootstrap_sqlite
from src.ai.features import get_latest_features, calculate_and_persist_features
from src.ai.factors import analyze_sentiment, analyze_fundamentals, get_latest_sentiment, get_latest_fundamentals
from src.ai.scoring import calculate_stock_score, score_stocks
from src.ai.ensemble import analyze_stock_ensemble, analyze_stocks_ensemble
from src.ai.adaptive_confidence import get_confidence_threshold, adjust_confidence_threshold, get_confidence_info

LOGGER = logging.getLogger("ai_selector")
if not LOGGER.handlers:
    LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(LOG_DIR / "system.log", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False


OLLAMA_MODELS: Sequence[str] = (
    "qwen3-coder:480b-cloud",
    "deepseek-v3.1:671b-cloud",
    "gpt-oss:120b",
)

UNIVERSE: Sequence[str] = (
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
    "NFLX",
    "CRM",
    "AMD",
    "SHOP",
    "UBER",
)

DEMO_POSITION_STATUS = "AI_DEMO"


@dataclass
class ModelPick:
    symbol: str
    score: float
    explanation: str
    action: str = "BUY"  # BUY, SELL, HOLD
    confidence: float = 0.0
    target_price: float = 0.0
    stop_loss: float = 0.0
    position_size_pct: float = 0.0


def run_ai_scoring(limit: Optional[int] = None) -> List[ModelPick]:
    """Execute the ensemble workflow and persist top selections."""
    
    settings = get_settings()
    bootstrap_sqlite()
    
    # Get universe of stocks to analyze
    universe = list(UNIVERSE)
    if limit:
        universe = universe[:limit]
    
    LOGGER.info("Starting AI scoring for %d symbols", len(universe))
    
    # Build comprehensive feature set with real data
    stock_data = _build_comprehensive_feature_set(universe)
    LOGGER.info("Comprehensive feature set built for %d symbols", len(stock_data))
    
    # Use ensemble analysis for final selections
    ensemble_results = analyze_stocks_ensemble(stock_data)
    
    # Get dynamic confidence threshold
    confidence_threshold = get_confidence_threshold()
    LOGGER.info(f"Using dynamic confidence threshold: {confidence_threshold:.3f}")
    
    # Convert ensemble results to ModelPick format
    model_picks = []
    for result in ensemble_results:
        if result.final_action == "BUY" and result.final_confidence > confidence_threshold:
            pick = ModelPick(
                symbol=result.symbol,
                score=result.final_confidence,
                explanation=" | ".join(result.reasoning[:3]),
                action=result.final_action,
                confidence=result.final_confidence,
                target_price=result.target_price,
                stop_loss=result.stop_loss,
                position_size_pct=result.position_size_pct
            )
            model_picks.append(pick)
    
    # Sort by confidence and take top 5
    model_picks.sort(key=lambda x: x.confidence, reverse=True)
    top_picks = model_picks[:5]
    
    LOGGER.info("Ensemble produced %d top picks", len(top_picks))
    
    # Persist selections
    _persist_selections(top_picks)
    LOGGER.info("Persisted %d AI selections into SQLite", len(top_picks))
    
    # Attempt to adjust confidence threshold based on recent performance
    try:
        threshold_adjusted = adjust_confidence_threshold()
        if threshold_adjusted:
            LOGGER.info("Confidence threshold automatically adjusted based on performance")
    except Exception as e:
        LOGGER.warning(f"Could not adjust confidence threshold: {e}")
    
    return top_picks


def _build_comprehensive_feature_set(symbols: List[str]) -> Dict[str, Dict]:
    """Build comprehensive feature set with real data for the trading universe."""
    
    stock_data = {}
    
    for i, symbol in enumerate(symbols):
        try:
            # Progress logging every 10 symbols
            if i % 10 == 0:
                LOGGER.info(f"[features] processed {i}/{len(symbols)}… latest={symbol}")
            
            LOGGER.info("Building features for %s", symbol)
            
            # Get latest features
            features = get_latest_features(symbol)
            
            # Get sentiment data
            sentiment_data = get_latest_sentiment(symbol)
            if not sentiment_data:
                # Analyze sentiment if not available
                sentiment_data = analyze_sentiment(symbol)
            
            # Get fundamental data
            fundamental_data = get_latest_fundamentals(symbol)
            if not fundamental_data:
                # Analyze fundamentals if not available
                fundamental_data = analyze_fundamentals(symbol)
            
            # If no features found, use pseudo-features
            if not features:
                LOGGER.info("No features found for %s, using pseudo-features", symbol)
                pseudo_data = _build_pseudo_features(symbol)
                stock_data[symbol] = pseudo_data
                continue
            
            # Prepare market data for ensemble
            market_data = {
                'current_price': 100.0,  # Placeholder - would come from real data
                'volume': 1000000,  # Placeholder
                'price_change': 0.0,  # Placeholder
                'price_change_pct': 0.0,  # Placeholder
                '52_week_high': 120.0,  # Placeholder
                '52_week_low': 80.0,  # Placeholder
                'market_cap': 1000000000,  # Placeholder
            }
            
            # Convert sentiment data to dict
            sentiment_dict = {}
            if sentiment_data:
                sentiment_dict = {
                    'sentiment_score': sentiment_data.sentiment_score,
                    'confidence': sentiment_data.confidence,
                    'news_count': sentiment_data.news_count,
                    'positive_news': sentiment_data.positive_news,
                    'negative_news': sentiment_data.negative_news,
                    'neutral_news': sentiment_data.neutral_news
                }
            
            # Convert fundamental data to dict
            fundamental_dict = {}
            if fundamental_data:
                fundamental_dict = {
                    'pe_ratio': fundamental_data.pe_ratio,
                    'pb_ratio': fundamental_data.pb_ratio,
                    'debt_to_equity': fundamental_data.debt_to_equity,
                    'roe': fundamental_data.roe,
                    'revenue_growth': fundamental_data.revenue_growth,
                    'earnings_growth': fundamental_data.earnings_growth,
                    'market_cap': fundamental_data.market_cap
                }
            
            stock_data[symbol] = {
                'features': features,
                'sentiment_data': sentiment_dict,
                'fundamental_data': fundamental_dict,
                'market_data': market_data
            }
            
        except Exception as e:
            LOGGER.error("Error building features for %s: %s", symbol, e)
            # Fallback to pseudo-features
            stock_data[symbol] = _build_pseudo_features(symbol)
    
    return stock_data

def _build_pseudo_features(symbol: str) -> Dict[str, Dict]:
    """Generate pseudo-features as fallback when real data is unavailable."""
    
    seed = int(date.today().strftime("%Y%m%d")) + hash(symbol)
    rng = random.Random(seed)
    
    base = rng.uniform(0.4, 0.95)
    technical = min(1.0, base + rng.uniform(-0.1, 0.1))
    sentiment = min(1.0, max(0.0, base + rng.uniform(-0.15, 0.15)))
    fundamental = min(1.0, max(0.0, 0.6 + rng.uniform(-0.2, 0.2)))
    momentum = min(1.0, max(0.0, 0.5 + rng.uniform(-0.1, 0.1)))
    volume_norm = min(1.0, max(0.0, 0.55 + rng.uniform(-0.15, 0.15)))
    
    features = {
        "rsi_14": 50 + rng.uniform(-20, 20),
        "macd": rng.uniform(-1, 1),
        "sma_20": 100 + rng.uniform(-10, 10),
        "sma_50": 100 + rng.uniform(-15, 15),
        "close": 100 + rng.uniform(-20, 20),
        "volume": 1000000 + rng.uniform(-500000, 500000)
    }
    
    sentiment_dict = {
        'sentiment_score': sentiment - 0.5,  # Convert to -0.5 to 0.5 range
        'confidence': rng.uniform(0.3, 0.8),
        'news_count': rng.randint(5, 20),
        'positive_news': rng.randint(2, 10),
        'negative_news': rng.randint(1, 8),
        'neutral_news': rng.randint(1, 5)
    }
    
    fundamental_dict = {
        'pe_ratio': rng.uniform(10, 30),
        'pb_ratio': rng.uniform(1, 5),
        'debt_to_equity': rng.uniform(0.1, 0.8),
        'roe': rng.uniform(5, 25),
        'revenue_growth': rng.uniform(-10, 30),
        'earnings_growth': rng.uniform(-15, 25),
        'market_cap': rng.uniform(1000000000, 10000000000)
    }
    
    market_data = {
        'current_price': 100 + rng.uniform(-20, 20),
        'volume': 1000000 + rng.uniform(-500000, 500000),
        'price_change': rng.uniform(-5, 5),
        'price_change_pct': rng.uniform(-0.05, 0.05),
        '52_week_high': 120 + rng.uniform(-10, 10),
        '52_week_low': 80 + rng.uniform(-10, 10),
        'market_cap': rng.uniform(1000000000, 10000000000)
    }
    
    return {
        'features': features,
        'sentiment_data': sentiment_dict,
        'fundamental_data': fundamental_dict,
        'market_data': market_data
    }


def _call_ollama_model(
    model: str,
    feature_space: Dict[str, Dict[str, float]],
    *,
    base_url: str,
) -> List[ModelPick]:
    """Query an Ollama model and extract structured picks."""

    prompt = _assemble_prompt(feature_space)
    url = f"{base_url.rstrip('/')}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}

    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        text = response.json().get("response") or response.text
        picks = _parse_model_response(text, feature_space)
        if not picks:
            raise ValueError("Parsed response contained no picks")
        LOGGER.info("Ollama model %s responded successfully", model)
        return picks
    except Exception as exc:  # pragma: no cover - network variability
        LOGGER.warning(
            "Falling back to heuristic scores for %s due to: %s",
            model,
            exc,
        )
        return _fallback_model_picks(model, feature_space)


def _assemble_prompt(feature_space: Dict[str, Dict[str, float]]) -> str:
    """Craft a concise prompt instructing models to score candidates."""

    payload = {
        "instructions": (
            "Score the provided stock candidates and return a JSON object with "
            "exactly five entries under a 'picks' array. Each entry should contain "
            "symbol, score (0-1 float), and explanation (short string)."
        ),
        "candidates": feature_space,
    }
    return (
        "You are an equity selection ensemble member.\n"
        "Respond with JSON only.\n"
        f"{json.dumps(payload)}"
    )


def _parse_model_response(
    text: str,
    feature_space: Dict[str, Dict[str, float]],
) -> List[ModelPick]:
    """Extract structured picks from a model response string."""

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            return []
        try:
            data = json.loads(text[start:end])
        except json.JSONDecodeError:
            return []

    picks_data = data.get("picks") or data.get("scores") or []
    picks: List[ModelPick] = []
    for entry in picks_data:
        symbol = entry.get("symbol")
        score = entry.get("score")
        explanation = entry.get("explanation") or entry.get("reason") or ""
        if (
            isinstance(symbol, str)
            and symbol in feature_space
            and isinstance(score, (int, float))
        ):
            picks.append(
                ModelPick(
                    symbol=symbol.upper(),
                    score=float(score),
                    explanation=str(explanation),
                )
            )
    return picks[:5]


def _fallback_model_picks(
    model: str,
    feature_space: Dict[str, Dict[str, float]],
) -> List[ModelPick]:
    """Generate deterministic fallback selections when models fail."""

    symbols = list(feature_space.keys())
    seed = hash((model, date.today().isoformat())) & 0xFFFF
    rng = random.Random(seed)
    rng.shuffle(symbols)

    picks: List[ModelPick] = []
    for symbol in symbols[:5]:
        features = feature_space[symbol]
        blended_score = mean(features.values())
        picks.append(
            ModelPick(
                symbol=symbol,
                score=round(blended_score, 4),
                explanation=(
                    f"Heuristic blend from {model}: "
                    f"technical={features['technical']}, "
                    f"sentiment={features['sentiment']}"
                ),
            )
        )
    return picks


def _ensemble_select(
    model_votes: Dict[str, Iterable[ModelPick]],
    feature_space: Dict[str, Dict[str, float]],
) -> List[ModelPick]:
    """Combine model votes with factor weights to determine top picks."""

    factor_weights = {
        "technical": 0.30,
        "sentiment": 0.25,
        "fundamental": 0.20,
        "momentum": 0.15,
        "volume": 0.10,
    }

    aggregate_scores: Dict[str, Dict[str, List[float] | List[str]]] = {
        symbol: {"scores": [], "explanations": []} for symbol in feature_space
    }

    for model_name, picks in model_votes.items():
        for pick in picks:
            aggregate_scores[pick.symbol]["scores"].append(pick.score)
            aggregate_scores[pick.symbol]["explanations"].append(
                f"{model_name}: {pick.explanation}"
            )

    selections: List[ModelPick] = []
    for symbol, aggregates in aggregate_scores.items():
        features = feature_space[symbol]
        factor_score = sum(
            features[name] * weight for name, weight in factor_weights.items()
        )
        if aggregates["scores"]:
            model_mean = mean(aggregates["scores"])  # type: ignore[arg-type]
            composite = (factor_score + model_mean) / 2
        else:
            composite = factor_score
        explanations = aggregates["explanations"] or [
            f"Factor composite score {composite:.3f}"
        ]
        selections.append(
            ModelPick(
                symbol=symbol,
                score=round(composite, 4),
                explanation=" | ".join(explanations[:3]),
            )
        )

    selections.sort(key=lambda entry: entry.score, reverse=True)
    return selections[:5]


def _persist_selections(picks: Sequence[ModelPick]) -> None:
    """Write ensemble selections to the SQLite store."""

    trade_date = date.today().isoformat()
    with sqlite3.connect(SQLITE_PATH) as connection:
        connection.execute(
            "DELETE FROM ai_selections WHERE trade_date = ?", (trade_date,)
        )
        connection.execute(
            "DELETE FROM positions WHERE status = ?", (DEMO_POSITION_STATUS,)
        )
        for pick in picks:
            connection.execute(
                """
                INSERT INTO ai_selections (trade_date, symbol, score, explanation, rationale)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    trade_date,
                    pick.symbol,
                    float(pick.score),
                    pick.explanation,
                    pick.explanation,
                ),
            )
        _write_demo_positions(connection, picks)
        connection.commit()


class AISelectorAgent:
    """AI Selector Agent for stock selection and analysis."""
    
    def __init__(self):
        """Initialize the AI Selector Agent."""
        self.logger = LOGGER
        
    def run(self, limit: Optional[int] = None) -> List[ModelPick]:
        """Run the AI selection process."""
        try:
            self.logger.info("Starting AI Selector Agent with limit: %s", limit)
            picks = run_ai_scoring(limit=limit)
            self.logger.info("AI Selector Agent completed with %d picks", len(picks))
            return picks
        except Exception as e:
            self.logger.error("AI Selector Agent failed: %s", e)
            return []
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status."""
        confidence_info = get_confidence_info()
        
        return {
            'name': 'AI Selector Agent',
            'status': 'active',
            'universe_size': len(UNIVERSE),
            'models': list(OLLAMA_MODELS),
            'last_run': datetime.now().isoformat(),
            'confidence_threshold': confidence_info['current_threshold'],
            'confidence_range': f"{confidence_info['min_threshold']:.2f} - {confidence_info['max_threshold']:.2f}",
            'recent_performance': confidence_info['recent_performance'],
            'should_adjust_threshold': confidence_info['should_adjust']
        }


def _write_demo_positions(
    connection: sqlite3.Connection, picks: Sequence[ModelPick]
) -> None:
    """Populate the positions table with synthetic demo entries."""

    for rank, pick in enumerate(picks, start=1):
        quantity = max(1, int(round(100 * pick.score)))
        average_price = round(95 + rank * 3 + pick.score * 10, 2)
        connection.execute(
            """
            INSERT INTO positions (symbol, quantity, average_price, status)
            VALUES (?, ?, ?, ?)
            """,
            (
                pick.symbol,
                quantity,
                average_price,
                DEMO_POSITION_STATUS,
            ),
        )
