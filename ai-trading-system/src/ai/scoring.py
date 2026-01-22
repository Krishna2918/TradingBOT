"""
AI Scoring Module - Multi-Factor Scoring Engine

This module provides weighted scoring for stocks based on technical indicators,
sentiment analysis, and fundamental data. Uses ensemble scoring with configurable weights.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ScoreType(Enum):
    """Types of scores that can be calculated."""
    TECHNICAL = "technical"
    SENTIMENT = "sentiment"
    FUNDAMENTAL = "fundamental"
    MOMENTUM = "momentum"
    VOLUME = "volume"
    OVERALL = "overall"

@dataclass
class ScoreWeights:
    """Configuration for scoring weights."""
    technical: float = 0.30
    sentiment: float = 0.25
    fundamental: float = 0.20
    momentum: float = 0.15
    volume: float = 0.10
    
    def __post_init__(self):
        """Validate that weights sum to 1.0."""
        total = sum([self.technical, self.sentiment, self.fundamental, self.momentum, self.volume])
        if abs(total - 1.0) > 0.001:
            logger.warning(f"Score weights sum to {total}, not 1.0. Normalizing...")
            self.technical /= total
            self.sentiment /= total
            self.fundamental /= total
            self.momentum /= total
            self.volume /= total

@dataclass
class StockScore:
    """Stock scoring result."""
    symbol: str
    technical_score: float
    sentiment_score: float
    fundamental_score: float
    momentum_score: float
    volume_score: float
    overall_score: float
    confidence: float
    timestamp: datetime
    details: Dict

class TechnicalScorer:
    """Technical analysis scoring."""
    
    @staticmethod
    def calculate_score(features: Dict) -> Tuple[float, Dict]:
        """Calculate technical score from features."""
        if not features:
            return 0.0, {}
        
        score = 0.0
        details = {}
        
        try:
            # RSI scoring (0-100, 50 is neutral)
            rsi = features.get('rsi_14', 50)
            if rsi is not None:
                if 30 <= rsi <= 70:  # Neutral zone
                    rsi_score = 0.5
                elif rsi < 30:  # Oversold - bullish
                    rsi_score = 0.8
                elif rsi > 70:  # Overbought - bearish
                    rsi_score = 0.2
                else:
                    rsi_score = 0.5
                score += rsi_score * 0.2
                details['rsi_score'] = rsi_score
            
            # MACD scoring
            macd = features.get('macd', 0)
            macd_signal = features.get('macd_signal', 0)
            if macd is not None and macd_signal is not None:
                if macd > macd_signal:  # Bullish crossover
                    macd_score = 0.8
                elif macd < macd_signal:  # Bearish crossover
                    macd_score = 0.2
                else:
                    macd_score = 0.5
                score += macd_score * 0.2
                details['macd_score'] = macd_score
            
            # Bollinger Bands scoring
            close = features.get('close', 0)
            bb_upper = features.get('bb_upper', 0)
            bb_lower = features.get('bb_lower', 0)
            if all(x is not None for x in [close, bb_upper, bb_lower]):
                if bb_lower < close < bb_upper:  # Within bands
                    bb_score = 0.6
                elif close <= bb_lower:  # Below lower band - oversold
                    bb_score = 0.8
                elif close >= bb_upper:  # Above upper band - overbought
                    bb_score = 0.2
                else:
                    bb_score = 0.5
                score += bb_score * 0.2
                details['bb_score'] = bb_score
            
            # Moving averages scoring
            sma_20 = features.get('sma_20', 0)
            sma_50 = features.get('sma_50', 0)
            if all(x is not None for x in [close, sma_20, sma_50]):
                if close > sma_20 > sma_50:  # Bullish alignment
                    ma_score = 0.8
                elif close < sma_20 < sma_50:  # Bearish alignment
                    ma_score = 0.2
                else:
                    ma_score = 0.5
                score += ma_score * 0.2
                details['ma_score'] = ma_score
            
            # Stochastic scoring
            stoch_k = features.get('stoch_k', 50)
            stoch_d = features.get('stoch_d', 50)
            if all(x is not None for x in [stoch_k, stoch_d]):
                if stoch_k > 80 and stoch_d > 80:  # Overbought
                    stoch_score = 0.2
                elif stoch_k < 20 and stoch_d < 20:  # Oversold
                    stoch_score = 0.8
                else:
                    stoch_score = 0.5
                score += stoch_score * 0.2
                details['stoch_score'] = stoch_score
            
            # Normalize score to 0-1 range
            score = max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Error calculating technical score: {e}")
            score = 0.5
            details['error'] = str(e)
        
        return score, details

class SentimentScorer:
    """Sentiment analysis scoring."""
    
    @staticmethod
    def calculate_score(sentiment_data: Dict) -> Tuple[float, Dict]:
        """Calculate sentiment score from sentiment data."""
        if not sentiment_data:
            return 0.5, {}
        
        score = 0.5  # Neutral default
        details = {}
        
        try:
            sentiment_score = sentiment_data.get('sentiment_score', 0.0)
            confidence = sentiment_data.get('confidence', 0.0)
            news_count = sentiment_data.get('news_count', 0)
            
            # Base sentiment score (-1 to 1) -> (0 to 1)
            base_score = (sentiment_score + 1) / 2
            
            # Adjust for confidence
            if confidence > 0.5:
                score = base_score
            else:
                # Low confidence - move toward neutral
                score = 0.5 + (base_score - 0.5) * confidence
            
            # Adjust for news volume
            if news_count > 10:
                # High news volume - maintain score
                pass
            elif news_count > 5:
                # Medium news volume - slight adjustment
                score = 0.5 + (score - 0.5) * 0.8
            else:
                # Low news volume - move toward neutral
                score = 0.5 + (score - 0.5) * 0.6
            
            details = {
                'sentiment_score': sentiment_score,
                'confidence': confidence,
                'news_count': news_count,
                'base_score': base_score,
                'final_score': score
            }
            
        except Exception as e:
            logger.error(f"Error calculating sentiment score: {e}")
            score = 0.5
            details['error'] = str(e)
        
        return score, details

class FundamentalScorer:
    """Fundamental analysis scoring."""
    
    @staticmethod
    def calculate_score(fundamental_data: Dict) -> Tuple[float, Dict]:
        """Calculate fundamental score from fundamental data."""
        if not fundamental_data:
            return 0.5, {}
        
        score = 0.5  # Neutral default
        details = {}
        
        try:
            # P/E ratio scoring
            pe_ratio = fundamental_data.get('pe_ratio')
            if pe_ratio is not None and pe_ratio > 0:
                if pe_ratio < 15:  # Undervalued
                    pe_score = 0.8
                elif pe_ratio < 25:  # Fair value
                    pe_score = 0.6
                elif pe_ratio < 40:  # Overvalued
                    pe_score = 0.4
                else:  # Highly overvalued
                    pe_score = 0.2
                score += pe_score * 0.2
                details['pe_score'] = pe_score
            
            # P/B ratio scoring
            pb_ratio = fundamental_data.get('pb_ratio')
            if pb_ratio is not None and pb_ratio > 0:
                if pb_ratio < 1.5:  # Undervalued
                    pb_score = 0.8
                elif pb_ratio < 3.0:  # Fair value
                    pb_score = 0.6
                elif pb_ratio < 5.0:  # Overvalued
                    pb_score = 0.4
                else:  # Highly overvalued
                    pb_score = 0.2
                score += pb_score * 0.2
                details['pb_score'] = pb_score
            
            # ROE scoring
            roe = fundamental_data.get('roe')
            if roe is not None:
                if roe > 15:  # Excellent
                    roe_score = 0.9
                elif roe > 10:  # Good
                    roe_score = 0.7
                elif roe > 5:  # Average
                    roe_score = 0.5
                else:  # Poor
                    roe_score = 0.3
                score += roe_score * 0.2
                details['roe_score'] = roe_score
            
            # Revenue growth scoring
            revenue_growth = fundamental_data.get('revenue_growth')
            if revenue_growth is not None:
                if revenue_growth > 20:  # High growth
                    growth_score = 0.9
                elif revenue_growth > 10:  # Good growth
                    growth_score = 0.7
                elif revenue_growth > 0:  # Positive growth
                    growth_score = 0.5
                else:  # Negative growth
                    growth_score = 0.3
                score += growth_score * 0.2
                details['growth_score'] = growth_score
            
            # Debt-to-equity scoring
            debt_to_equity = fundamental_data.get('debt_to_equity')
            if debt_to_equity is not None:
                if debt_to_equity < 0.3:  # Low debt
                    debt_score = 0.8
                elif debt_to_equity < 0.6:  # Moderate debt
                    debt_score = 0.6
                elif debt_to_equity < 1.0:  # High debt
                    debt_score = 0.4
                else:  # Very high debt
                    debt_score = 0.2
                score += debt_score * 0.2
                details['debt_score'] = debt_score
            
            # Normalize score to 0-1 range
            score = max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Error calculating fundamental score: {e}")
            score = 0.5
            details['error'] = str(e)
        
        return score, details

class MomentumScorer:
    """Momentum analysis scoring."""
    
    @staticmethod
    def calculate_score(features: Dict) -> Tuple[float, Dict]:
        """Calculate momentum score from features."""
        if not features:
            return 0.5, {}
        
        score = 0.5  # Neutral default
        details = {}
        
        try:
            # Price momentum (close vs moving averages)
            close = features.get('close', 0)
            sma_20 = features.get('sma_20', 0)
            sma_50 = features.get('sma_50', 0)
            
            if all(x is not None for x in [close, sma_20, sma_50]):
                # Short-term momentum
                if close > sma_20:
                    short_momentum = 0.8
                else:
                    short_momentum = 0.2
                
                # Long-term momentum
                if close > sma_50:
                    long_momentum = 0.8
                else:
                    long_momentum = 0.2
                
                # Combined momentum
                momentum_score = (short_momentum + long_momentum) / 2
                score += momentum_score * 0.4
                details['momentum_score'] = momentum_score
            
            # MACD momentum
            macd = features.get('macd', 0)
            macd_signal = features.get('macd_signal', 0)
            if all(x is not None for x in [macd, macd_signal]):
                if macd > macd_signal:
                    macd_momentum = 0.8
                else:
                    macd_momentum = 0.2
                score += macd_momentum * 0.3
                details['macd_momentum'] = macd_momentum
            
            # RSI momentum
            rsi = features.get('rsi_14', 50)
            if rsi is not None:
                if rsi > 50:  # Bullish momentum
                    rsi_momentum = 0.7
                else:  # Bearish momentum
                    rsi_momentum = 0.3
                score += rsi_momentum * 0.3
                details['rsi_momentum'] = rsi_momentum
            
            # Normalize score to 0-1 range
            score = max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Error calculating momentum score: {e}")
            score = 0.5
            details['error'] = str(e)
        
        return score, details

class VolumeScorer:
    """Volume analysis scoring."""
    
    @staticmethod
    def calculate_score(features: Dict) -> Tuple[float, Dict]:
        """Calculate volume score from features."""
        if not features:
            return 0.5, {}
        
        score = 0.5  # Neutral default
        details = {}
        
        try:
            # Volume vs moving average
            volume = features.get('volume', 0)
            volume_sma_20 = features.get('volume_sma_20', 0)
            
            if all(x is not None for x in [volume, volume_sma_20]) and volume_sma_20 > 0:
                volume_ratio = volume / volume_sma_20
                
                if volume_ratio > 2.0:  # High volume
                    volume_score = 0.8
                elif volume_ratio > 1.5:  # Above average volume
                    volume_score = 0.7
                elif volume_ratio > 1.0:  # Average volume
                    volume_score = 0.5
                elif volume_ratio > 0.5:  # Below average volume
                    volume_score = 0.3
                else:  # Low volume
                    volume_score = 0.2
                
                score = volume_score
                details['volume_ratio'] = volume_ratio
                details['volume_score'] = volume_score
            
            # Volume profile analysis
            volume_profile = features.get('volume_profile', {})
            if volume_profile:
                poc_price = volume_profile.get('poc_price')
                close = features.get('close', 0)
                
                if all(x is not None for x in [poc_price, close]):
                    # Price relative to POC
                    if abs(close - poc_price) / poc_price < 0.02:  # Close to POC
                        poc_score = 0.8
                    else:
                        poc_score = 0.5
                    
                    score = (score + poc_score) / 2
                    details['poc_score'] = poc_score
            
            # Normalize score to 0-1 range
            score = max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Error calculating volume score: {e}")
            score = 0.5
            details['error'] = str(e)
        
        return score, details

class ScoringEngine:
    """Main scoring engine that combines all factors."""
    
    def __init__(self, weights: ScoreWeights = None):
        """Initialize scoring engine."""
        self.weights = weights or ScoreWeights()
        self.technical_scorer = TechnicalScorer()
        self.sentiment_scorer = SentimentScorer()
        self.fundamental_scorer = FundamentalScorer()
        self.momentum_scorer = MomentumScorer()
        self.volume_scorer = VolumeScorer()
    
    def calculate_overall_score(self, symbol: str, features: Dict, 
                              sentiment_data: Dict, fundamental_data: Dict) -> StockScore:
        """Calculate overall score for a stock."""
        try:
            # Calculate individual scores
            technical_score, technical_details = self.technical_scorer.calculate_score(features)
            sentiment_score, sentiment_details = self.sentiment_scorer.calculate_score(sentiment_data)
            fundamental_score, fundamental_details = self.fundamental_scorer.calculate_score(fundamental_data)
            momentum_score, momentum_details = self.momentum_scorer.calculate_score(features)
            volume_score, volume_details = self.volume_scorer.calculate_score(features)
            
            # Calculate weighted overall score
            overall_score = (
                technical_score * self.weights.technical +
                sentiment_score * self.weights.sentiment +
                fundamental_score * self.weights.fundamental +
                momentum_score * self.weights.momentum +
                volume_score * self.weights.volume
            )
            
            # Calculate confidence based on data availability
            confidence = self._calculate_confidence(
                features, sentiment_data, fundamental_data
            )
            
            # Combine all details
            details = {
                'technical': technical_details,
                'sentiment': sentiment_details,
                'fundamental': fundamental_details,
                'momentum': momentum_details,
                'volume': volume_details,
                'weights': {
                    'technical': self.weights.technical,
                    'sentiment': self.weights.sentiment,
                    'fundamental': self.weights.fundamental,
                    'momentum': self.weights.momentum,
                    'volume': self.weights.volume
                }
            }
            
            return StockScore(
                symbol=symbol,
                technical_score=technical_score,
                sentiment_score=sentiment_score,
                fundamental_score=fundamental_score,
                momentum_score=momentum_score,
                volume_score=volume_score,
                overall_score=overall_score,
                confidence=confidence,
                timestamp=datetime.now(),
                details=details
            )
            
        except Exception as e:
            logger.error(f"Error calculating overall score for {symbol}: {e}")
            return StockScore(
                symbol=symbol,
                technical_score=0.5,
                sentiment_score=0.5,
                fundamental_score=0.5,
                momentum_score=0.5,
                volume_score=0.5,
                overall_score=0.5,
                confidence=0.0,
                timestamp=datetime.now(),
                details={'error': str(e)}
            )
    
    def _calculate_confidence(self, features: Dict, sentiment_data: Dict, 
                            fundamental_data: Dict) -> float:
        """Calculate confidence based on data availability."""
        confidence = 0.0
        
        # Technical data availability
        if features:
            technical_indicators = ['rsi_14', 'macd', 'sma_20', 'sma_50']
            available_indicators = sum(1 for indicator in technical_indicators 
                                     if features.get(indicator) is not None)
            confidence += (available_indicators / len(technical_indicators)) * 0.4
        
        # Sentiment data availability
        if sentiment_data and sentiment_data.get('news_count', 0) > 0:
            confidence += 0.3
        
        # Fundamental data availability
        if fundamental_data:
            fundamental_metrics = ['pe_ratio', 'pb_ratio', 'roe', 'revenue_growth']
            available_metrics = sum(1 for metric in fundamental_metrics 
                                  if fundamental_data.get(metric) is not None)
            confidence += (available_metrics / len(fundamental_metrics)) * 0.3
        
        return min(1.0, confidence)
    
    def score_multiple_stocks(self, stock_data: Dict[str, Dict]) -> List[StockScore]:
        """Score multiple stocks."""
        scores = []
        
        for symbol, data in stock_data.items():
            features = data.get('features', {})
            sentiment_data = data.get('sentiment', {})
            fundamental_data = data.get('fundamental', {})
            
            score = self.calculate_overall_score(symbol, features, sentiment_data, fundamental_data)
            scores.append(score)
        
        # Sort by overall score descending
        scores.sort(key=lambda x: x.overall_score, reverse=True)
        
        return scores

# Global scoring engine instance
scoring_engine = ScoringEngine()

def calculate_stock_score(symbol: str, features: Dict, sentiment_data: Dict, 
                         fundamental_data: Dict) -> StockScore:
    """Calculate score for a single stock."""
    return scoring_engine.calculate_overall_score(symbol, features, sentiment_data, fundamental_data)

def score_stocks(stock_data: Dict[str, Dict]) -> List[StockScore]:
    """Score multiple stocks."""
    return scoring_engine.score_multiple_stocks(stock_data)

def get_score_weights() -> ScoreWeights:
    """Get current score weights."""
    return scoring_engine.weights

def set_score_weights(weights: ScoreWeights):
    """Set new score weights."""
    scoring_engine.weights = weights
