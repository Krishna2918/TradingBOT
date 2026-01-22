"""
Signal Aggregation Engine - Combines Multiple Intelligence Sources

Fuses signals from:
- News sentiment (NewsAPI, Finnhub)
- Insider trades (SEDI filings)
- Social media (Reddit, Twitter, StockTwits)
- Weather/commodity correlations
- Whale tracking (13F, ETF flows)
- Macro economic indicators

Outputs a unified trading signal with confidence score
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TradingSignal:
    """Unified trading signal output"""
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # 0.0 to 1.0
    strength: float  # Signal strength (can exceed 1.0)
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    position_size_pct: float = 0.02  # Default 2% of capital
    reasoning: List[str] = None
    source_scores: Dict[str, float] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.reasoning is None:
            self.reasoning = []
        if self.source_scores is None:
            self.source_scores = {}
        if self.timestamp is None:
            self.timestamp = datetime.now()


class SignalAggregator:
    """
    Combines multiple intelligence sources into unified trading signals
    """
    
    # Signal source weights (total = 1.0)
    WEIGHTS = {
        'insider_trades': 0.30,  # Highest weight - insiders know their company
        'news_sentiment': 0.25,  # High weight - market-moving information
        'social_sentiment': 0.15,  # Moderate weight - retail momentum
        'macro_alignment': 0.10,  # Low weight - affects all stocks
        'weather_commodity': 0.10,  # Low weight - sector-specific
        'whale_activity': 0.10,  # Low weight - follow smart money
    }
    
    # Confidence thresholds
    MIN_CONFIDENCE_TO_TRADE = 0.50  # 50% minimum
    HIGH_CONFIDENCE_THRESHOLD = 0.75  # 75%+ = strong signal
    VERY_HIGH_CONFIDENCE_THRESHOLD = 0.85  # 85%+ = very strong
    
    def __init__(self):
        """Initialize signal aggregator"""
        self.signal_history: Dict[str, List[TradingSignal]] = {}
        logger.info("Signal Aggregator initialized")
        logger.info(f"  Weights: {self.WEIGHTS}")
        logger.info(f"  Min confidence to trade: {self.MIN_CONFIDENCE_TO_TRADE*100:.0f}%")
    
    def aggregate_signals(
        self,
        symbol: str,
        price: float,
        sources: Dict[str, Any]
    ) -> TradingSignal:
        """
        Aggregate signals from multiple sources into unified trading decision
        
        Args:
            symbol: Stock symbol
            price: Current price
            sources: Dictionary with keys:
                - insider_trades: Dict with 'sentiment' (-1 to 1), 'transactions'
                - news_sentiment: Dict with 'score' (-1 to 1), 'article_count'
                - social_sentiment: Dict with 'score' (-1 to 1), 'mention_volume'
                - macro_alignment: Dict with 'alignment' (-1 to 1)
                - weather_commodity: Dict with 'impact' (-1 to 1)
                - whale_activity: Dict with 'flow' (-1 to 1), 'net_buying'
        
        Returns:
            TradingSignal with action, confidence, reasoning
        """
        
        # Extract individual scores
        insider_score = self._extract_score(sources.get('insider_trades'), 'sentiment', 0.0)
        news_score = self._extract_score(sources.get('news_sentiment'), 'score', 0.0)
        social_score = self._extract_score(sources.get('social_sentiment'), 'score', 0.0)
        macro_score = self._extract_score(sources.get('macro_alignment'), 'alignment', 0.0)
        weather_score = self._extract_score(sources.get('weather_commodity'), 'impact', 0.0)
        whale_score = self._extract_score(sources.get('whale_activity'), 'flow', 0.0)
        
        # Calculate weighted aggregate score
        aggregate_score = (
            self.WEIGHTS['insider_trades'] * insider_score +
            self.WEIGHTS['news_sentiment'] * news_score +
            self.WEIGHTS['social_sentiment'] * social_score +
            self.WEIGHTS['macro_alignment'] * macro_score +
            self.WEIGHTS['weather_commodity'] * weather_score +
            self.WEIGHTS['whale_activity'] * whale_score
        )
        
        # Count positive sources (signal alignment boost)
        positive_sources = sum(1 for score in [insider_score, news_score, social_score, 
                                               macro_score, weather_score, whale_score] if score > 0.3)
        negative_sources = sum(1 for score in [insider_score, news_score, social_score,
                                               macro_score, weather_score, whale_score] if score < -0.3)
        
        # Determine action
        if aggregate_score >= 0.20:
            action = 'BUY'
        elif aggregate_score <= -0.20:
            action = 'SELL'
        else:
            action = 'HOLD'
        
        # Calculate confidence
        base_confidence = abs(aggregate_score)
        
        # Confidence boost: Multiple sources agreeing
        if positive_sources >= 3:
            confidence_boost = 0.15 + (positive_sources - 3) * 0.05
        elif negative_sources >= 3:
            confidence_boost = 0.15 + (negative_sources - 3) * 0.05
        else:
            confidence_boost = 0.0
        
        # Confidence penalty: Conflicting signals
        if positive_sources >= 2 and negative_sources >= 2:
            confidence_penalty = -0.20
        else:
            confidence_penalty = 0.0
        
        confidence = np.clip(base_confidence + confidence_boost + confidence_penalty, 0.0, 1.0)
        
        # Build reasoning
        reasoning = self._build_reasoning(
            insider_score, news_score, social_score, macro_score, 
            weather_score, whale_score, sources, positive_sources, negative_sources
        )
        
        # Calculate position size (higher confidence = larger position)
        if confidence >= self.VERY_HIGH_CONFIDENCE_THRESHOLD:
            position_size_pct = 0.05  # 5% of capital
        elif confidence >= self.HIGH_CONFIDENCE_THRESHOLD:
            position_size_pct = 0.03  # 3% of capital
        else:
            position_size_pct = 0.02  # 2% of capital (default)
        
        # Calculate target price and stop loss (simple 2:1 risk-reward)
        if action == 'BUY':
            target_price = price * (1 + 0.10 * confidence)  # Up to 10% profit target
            stop_loss = price * (1 - 0.05 * confidence)  # Up to 5% stop loss
        elif action == 'SELL':
            target_price = price * (1 - 0.10 * confidence)
            stop_loss = price * (1 + 0.05 * confidence)
        else:
            target_price = None
            stop_loss = None
        
        # Create signal
        signal = TradingSignal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            strength=abs(aggregate_score),
            target_price=target_price,
            stop_loss=stop_loss,
            position_size_pct=position_size_pct,
            reasoning=reasoning,
            source_scores={
                'insider': insider_score,
                'news': news_score,
                'social': social_score,
                'macro': macro_score,
                'weather': weather_score,
                'whale': whale_score,
                'aggregate': aggregate_score,
            }
        )
        
        # Store in history
        if symbol not in self.signal_history:
            self.signal_history[symbol] = []
        self.signal_history[symbol].append(signal)
        
        # Log signal
        self._log_signal(signal, positive_sources, negative_sources)
        
        return signal
    
    def _extract_score(self, source_data: Optional[Dict], key: str, default: float) -> float:
        """Extract score from source data with fallback"""
        if source_data is None:
            return default
        value = source_data.get(key, default)
        # Ensure it's a float between -1 and 1
        try:
            return np.clip(float(value), -1.0, 1.0)
        except (TypeError, ValueError):
            return default
    
    def _build_reasoning(
        self,
        insider: float, news: float, social: float,
        macro: float, weather: float, whale: float,
        sources: Dict, positive_sources: int, negative_sources: int
    ) -> List[str]:
        """Build human-readable reasoning for the signal"""
        reasons = []
        
        # Insider trades
        if abs(insider) >= 0.3:
            direction = "buying" if insider > 0 else "selling"
            transactions = sources.get('insider_trades', {}).get('transactions', 0)
            reasons.append(f"Insiders {direction} ({transactions} transactions, score={insider:.2f})")
        
        # News sentiment
        if abs(news) >= 0.3:
            direction = "bullish" if news > 0 else "bearish"
            article_count = sources.get('news_sentiment', {}).get('article_count', 0)
            reasons.append(f"News {direction} ({article_count} articles, score={news:.2f})")
        
        # Social sentiment
        if abs(social) >= 0.3:
            direction = "positive" if social > 0 else "negative"
            mentions = sources.get('social_sentiment', {}).get('mention_volume', 0)
            reasons.append(f"Social sentiment {direction} ({mentions} mentions, score={social:.2f})")
        
        # Macro alignment
        if abs(macro) >= 0.3:
            direction = "supportive" if macro > 0 else "headwind"
            reasons.append(f"Macro environment {direction} (score={macro:.2f})")
        
        # Weather/commodity
        if abs(weather) >= 0.3:
            direction = "positive" if weather > 0 else "negative"
            reasons.append(f"Weather/commodity impact {direction} (score={weather:.2f})")
        
        # Whale activity
        if abs(whale) >= 0.3:
            direction = "accumulating" if whale > 0 else "distributing"
            net_buying = sources.get('whale_activity', {}).get('net_buying', 0)
            reasons.append(f"Institutional {direction} (net={net_buying:.0f}, score={whale:.2f})")
        
        # Alignment summary
        if positive_sources >= 3:
            reasons.append(f"✅ {positive_sources} sources aligned (strong consensus)")
        elif negative_sources >= 3:
            reasons.append(f"⚠️ {negative_sources} sources negative (strong consensus)")
        elif positive_sources >= 2 and negative_sources >= 2:
            reasons.append(f"⚠️ Mixed signals ({positive_sources} positive, {negative_sources} negative)")
        
        if not reasons:
            reasons.append("Weak signals from all sources")
        
        return reasons
    
    def _log_signal(self, signal: TradingSignal, positive: int, negative: int):
        """Log signal to console"""
        if signal.action == 'HOLD':
            return  # Don't log HOLD signals
        
        conf_symbol = "🟢" if signal.confidence >= 0.75 else "🟡" if signal.confidence >= 0.50 else "🔴"
        
        logger.info(f"\n{'='*80}")
        logger.info(f"{conf_symbol} SIGNAL: {signal.action} {signal.symbol} @ confidence={signal.confidence*100:.1f}%")
        logger.info(f"   Strength: {signal.strength:.3f} | Position size: {signal.position_size_pct*100:.1f}%")
        if signal.target_price:
            logger.info(f"   Target: ${signal.target_price:.2f} | Stop: ${signal.stop_loss:.2f}")
        logger.info(f"   Sources: {positive} positive, {negative} negative")
        logger.info(f"   Reasoning:")
        for reason in signal.reasoning:
            logger.info(f"      • {reason}")
        logger.info(f"{'='*80}\n")
    
    def get_signal_history(self, symbol: str, limit: int = 10) -> List[TradingSignal]:
        """Get recent signals for a symbol"""
        return self.signal_history.get(symbol, [])[-limit:]
    
    def should_trade(self, signal: TradingSignal) -> bool:
        """Determine if signal is strong enough to trade"""
        return (
            signal.action in ['BUY', 'SELL'] and
            signal.confidence >= self.MIN_CONFIDENCE_TO_TRADE
        )


def create_signal_aggregator() -> SignalAggregator:
    """Factory function to create SignalAggregator"""
    return SignalAggregator()

