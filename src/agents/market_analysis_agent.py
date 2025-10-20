"""
Market Analysis Agent

IMPORTANT priority agent that analyzes market conditions, detects regime changes,
and provides market intelligence for trading decisions.

Responsibilities:
- Market regime detection
- Volatility analysis
- Trend identification
- Market sentiment analysis
- Economic indicator tracking
- Sector rotation analysis
"""

import logging
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from .base_agent import BaseAgent, AgentPriority, AgentStatus, ResourceRequirements

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications"""
    BULL = "bull_market"
    BEAR = "bear_market"
    SIDEWAYS = "sideways_market"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    CONSOLIDATION = "consolidation"


class TrendStrength(Enum):
    """Trend strength classifications"""
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    NONE = "none"


@dataclass
class MarketMetrics:
    """Market analysis metrics"""
    timestamp: datetime
    regime: MarketRegime
    trend_strength: TrendStrength
    volatility: float
    volume_trend: str
    momentum: float
    support_level: float
    resistance_level: float
    market_sentiment: float  # -1 to 1
    sector_rotation: Dict[str, float]


@dataclass
class RegimeChange:
    """Market regime change detection"""
    timestamp: datetime
    from_regime: MarketRegime
    to_regime: MarketRegime
    confidence: float
    trigger_factors: List[str]
    expected_duration: str


class MarketAnalysisAgent(BaseAgent):
    """
    Market Analysis Agent - IMPORTANT priority
    
    Analyzes market conditions and provides intelligence for trading decisions.
    """
    
    def __init__(self):
        super().__init__(
            agent_id='market_analysis_agent',
            name='Market Analysis Agent',
            priority=AgentPriority.IMPORTANT,  # Important but not critical
            resource_requirements=ResourceRequirements(
                min_cpu_percent=10.0,
                min_memory_mb=200.0,
                max_cpu_percent=30.0,
                max_memory_mb=600.0
            )
        )
        
        # Analysis state
        self.market_history: List[MarketMetrics] = []
        self.regime_changes: List[RegimeChange] = []
        self.current_regime = MarketRegime.SIDEWAYS
        self.analysis_active = False
        
        # Analysis parameters
        self.volatility_threshold_high = 0.25  # 25% annualized
        self.volatility_threshold_low = 0.10   # 10% annualized
        self.trend_threshold = 0.02  # 2% for trend detection
        self.regime_change_confidence = 0.7  # 70% confidence required
        
        # Performance tracking
        self.analyses_performed = 0
        self.regime_changes_detected = 0
        self.predictions_made = 0
        self.accuracy_score = 0.0
        
        # Market data cache
        self.price_cache: Dict[str, List[float]] = {}
        self.volume_cache: Dict[str, List[float]] = {}
        self.cache_duration = timedelta(minutes=5)
    
    async def initialize(self) -> bool:
        """Initialize market analysis systems"""
        try:
            self.status = AgentStatus.IDLE
            self.analysis_active = True
            
            # Start background analysis
            asyncio.create_task(self._background_analysis())
            
            logger.info("Market Analysis Agent initialized and background analysis started")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Market Analysis Agent: {e}")
            self.status = AgentStatus.ERROR
            return False
    
    async def shutdown(self) -> bool:
        """Gracefully shutdown market analysis"""
        logger.info("Market Analysis Agent shutting down")
        self.analysis_active = False
        self.status = AgentStatus.STOPPED
        return True
    
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process market analysis tasks.
        
        Task types:
        - 'analyze_market': Perform comprehensive market analysis
        - 'detect_regime_change': Detect market regime changes
        - 'get_trend_analysis': Get trend analysis for specific timeframe
        - 'get_volatility_analysis': Analyze market volatility
        - 'get_sector_analysis': Analyze sector rotation
        - 'predict_market_direction': Predict short-term market direction
        - 'get_support_resistance': Calculate support and resistance levels
        """
        task_type = task.get('type')
        
        if task_type == 'analyze_market':
            return await self._analyze_market(task.get('symbols', ['SPY', 'QQQ', 'IWM']))
        elif task_type == 'detect_regime_change':
            return await self._detect_regime_change()
        elif task_type == 'get_trend_analysis':
            return await self._get_trend_analysis(task.get('timeframe', '1D'))
        elif task_type == 'get_volatility_analysis':
            return await self._get_volatility_analysis()
        elif task_type == 'get_sector_analysis':
            return await self._get_sector_analysis()
        elif task_type == 'predict_market_direction':
            return await self._predict_market_direction(task.get('horizon', '1D'))
        elif task_type == 'get_support_resistance':
            return await self._get_support_resistance(task.get('symbol', 'SPY'))
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    async def _background_analysis(self):
        """Background market analysis loop"""
        while self.analysis_active:
            try:
                # Perform market analysis every 5 minutes
                await self._analyze_market(['SPY', 'QQQ', 'IWM'])
                await asyncio.sleep(300)  # 5 minutes
            except Exception as e:
                logger.error(f"Background market analysis error: {e}")
                await asyncio.sleep(600)  # Wait longer on error
    
    async def _analyze_market(self, symbols: List[str]) -> Dict[str, Any]:
        """Perform comprehensive market analysis"""
        try:
            # Get market data for symbols
            market_data = {}
            for symbol in symbols:
                data = await self._get_market_data(symbol)
                if data:
                    market_data[symbol] = data
            
            if not market_data:
                return {'error': 'No market data available'}
            
            # Analyze each symbol
            symbol_analyses = {}
            for symbol, data in market_data.items():
                analysis = self._analyze_symbol(data, symbol)
                symbol_analyses[symbol] = analysis
            
            # Determine overall market regime
            overall_regime = self._determine_overall_regime(symbol_analyses)
            
            # Calculate market metrics
            market_metrics = MarketMetrics(
                timestamp=datetime.now(),
                regime=overall_regime,
                trend_strength=self._calculate_trend_strength(symbol_analyses),
                volatility=self._calculate_market_volatility(symbol_analyses),
                volume_trend=self._analyze_volume_trend(symbol_analyses),
                momentum=self._calculate_market_momentum(symbol_analyses),
                support_level=self._calculate_support_level(symbol_analyses),
                resistance_level=self._calculate_resistance_level(symbol_analyses),
                market_sentiment=self._calculate_market_sentiment(symbol_analyses),
                sector_rotation=self._analyze_sector_rotation(symbol_analyses)
            )
            
            # Store in history
            self.market_history.append(market_metrics)
            if len(self.market_history) > 1000:  # Keep last 1000 analyses
                self.market_history.pop(0)
            
            # Check for regime change
            if len(self.market_history) > 1:
                regime_change = self._check_regime_change(market_metrics)
                if regime_change:
                    self.regime_changes.append(regime_change)
                    self.regime_changes_detected += 1
            
            self.analyses_performed += 1
            
            return {
                'timestamp': market_metrics.timestamp.isoformat(),
                'overall_regime': market_metrics.regime.value,
                'trend_strength': market_metrics.trend_strength.value,
                'volatility': market_metrics.volatility,
                'volume_trend': market_metrics.volume_trend,
                'momentum': market_metrics.momentum,
                'support_level': market_metrics.support_level,
                'resistance_level': market_metrics.resistance_level,
                'market_sentiment': market_metrics.market_sentiment,
                'sector_rotation': market_metrics.sector_rotation,
                'symbol_analyses': symbol_analyses,
                'regime_change_detected': regime_change is not None
            }
            
        except Exception as e:
            logger.error(f"Market analysis failed: {e}")
            return {'error': str(e)}
    
    async def _get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get market data for symbol (placeholder implementation)"""
        # In real implementation, this would fetch from data sources
        # For now, generate realistic mock data
        
        import random
        
        # Generate realistic price data
        base_price = 100.0 if symbol == 'SPY' else 150.0 if symbol == 'QQQ' else 80.0
        price_change = random.uniform(-0.02, 0.02)  # ±2% daily change
        current_price = base_price * (1 + price_change)
        
        # Generate volume data
        base_volume = random.randint(1000000, 5000000)
        volume_change = random.uniform(-0.3, 0.3)  # ±30% volume change
        current_volume = base_volume * (1 + volume_change)
        
        return {
            'symbol': symbol,
            'price': current_price,
            'volume': current_volume,
            'change': price_change,
            'change_percent': price_change * 100,
            'high': current_price * 1.01,
            'low': current_price * 0.99,
            'open': base_price,
            'timestamp': datetime.now()
        }
    
    def _analyze_symbol(self, data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Analyze individual symbol data"""
        price = data['price']
        volume = data['volume']
        change_percent = data['change_percent']
        
        # Calculate technical indicators
        volatility = abs(change_percent) / 100  # Simple volatility measure
        momentum = change_percent / 100
        
        # Determine trend
        if change_percent > self.trend_threshold * 100:
            trend = 'up'
        elif change_percent < -self.trend_threshold * 100:
            trend = 'down'
        else:
            trend = 'sideways'
        
        # Volume analysis
        volume_trend = 'high' if volume > 2000000 else 'normal' if volume > 1000000 else 'low'
        
        return {
            'symbol': symbol,
            'price': price,
            'volume': volume,
            'change_percent': change_percent,
            'volatility': volatility,
            'momentum': momentum,
            'trend': trend,
            'volume_trend': volume_trend,
            'technical_score': self._calculate_technical_score(volatility, momentum, volume)
        }
    
    def _calculate_technical_score(self, volatility: float, momentum: float, volume: float) -> float:
        """Calculate technical analysis score (0-100)"""
        # Simple scoring algorithm
        momentum_score = (momentum + 1) * 50  # Convert -1 to 1 range to 0-100
        volatility_score = max(0, 100 - volatility * 1000)  # Lower volatility = higher score
        volume_score = min(100, volume / 2000000 * 100)  # Higher volume = higher score
        
        return (momentum_score + volatility_score + volume_score) / 3
    
    def _determine_overall_regime(self, symbol_analyses: Dict[str, Dict[str, Any]]) -> MarketRegime:
        """Determine overall market regime from symbol analyses"""
        if not symbol_analyses:
            return MarketRegime.SIDEWAYS
        
        # Calculate average metrics
        avg_momentum = np.mean([analysis['momentum'] for analysis in symbol_analyses.values()])
        avg_volatility = np.mean([analysis['volatility'] for analysis in symbol_analyses.values()])
        
        # Determine regime based on momentum and volatility
        if avg_volatility > self.volatility_threshold_high:
            return MarketRegime.HIGH_VOLATILITY
        elif avg_volatility < self.volatility_threshold_low:
            return MarketRegime.LOW_VOLATILITY
        elif avg_momentum > self.trend_threshold:
            return MarketRegime.TRENDING_UP
        elif avg_momentum < -self.trend_threshold:
            return MarketRegime.TRENDING_DOWN
        else:
            return MarketRegime.SIDEWAYS
    
    def _calculate_trend_strength(self, symbol_analyses: Dict[str, Dict[str, Any]]) -> TrendStrength:
        """Calculate overall trend strength"""
        if not symbol_analyses:
            return TrendStrength.NONE
        
        avg_momentum = np.mean([analysis['momentum'] for analysis in symbol_analyses.values()])
        momentum_std = np.std([analysis['momentum'] for analysis in symbol_analyses.values()])
        
        if abs(avg_momentum) > 0.03:  # 3% threshold
            return TrendStrength.STRONG
        elif abs(avg_momentum) > 0.01:  # 1% threshold
            return TrendStrength.MODERATE
        elif abs(avg_momentum) > 0.005:  # 0.5% threshold
            return TrendStrength.WEAK
        else:
            return TrendStrength.NONE
    
    def _calculate_market_volatility(self, symbol_analyses: Dict[str, Dict[str, Any]]) -> float:
        """Calculate market-wide volatility"""
        if not symbol_analyses:
            return 0.0
        
        volatilities = [analysis['volatility'] for analysis in symbol_analyses.values()]
        return np.mean(volatilities)
    
    def _analyze_volume_trend(self, symbol_analyses: Dict[str, Dict[str, Any]]) -> str:
        """Analyze overall volume trend"""
        if not symbol_analyses:
            return 'unknown'
        
        volume_trends = [analysis['volume_trend'] for analysis in symbol_analyses.values()]
        
        if volume_trends.count('high') > len(volume_trends) / 2:
            return 'increasing'
        elif volume_trends.count('low') > len(volume_trends) / 2:
            return 'decreasing'
        else:
            return 'stable'
    
    def _calculate_market_momentum(self, symbol_analyses: Dict[str, Dict[str, Any]]) -> float:
        """Calculate market momentum"""
        if not symbol_analyses:
            return 0.0
        
        momentums = [analysis['momentum'] for analysis in symbol_analyses.values()]
        return np.mean(momentums)
    
    def _calculate_support_level(self, symbol_analyses: Dict[str, Dict[str, Any]]) -> float:
        """Calculate market support level"""
        if not symbol_analyses:
            return 0.0
        
        prices = [analysis['price'] for analysis in symbol_analyses.values()]
        return min(prices) * 0.98  # 2% below lowest price
    
    def _calculate_resistance_level(self, symbol_analyses: Dict[str, Dict[str, Any]]) -> float:
        """Calculate market resistance level"""
        if not symbol_analyses:
            return 0.0
        
        prices = [analysis['price'] for analysis in symbol_analyses.values()]
        return max(prices) * 1.02  # 2% above highest price
    
    def _calculate_market_sentiment(self, symbol_analyses: Dict[str, Dict[str, Any]]) -> float:
        """Calculate market sentiment (-1 to 1)"""
        if not symbol_analyses:
            return 0.0
        
        # Simple sentiment based on momentum and volume
        avg_momentum = np.mean([analysis['momentum'] for analysis in symbol_analyses.values()])
        volume_scores = [1 if analysis['volume_trend'] == 'high' else 0.5 for analysis in symbol_analyses.values()]
        avg_volume_score = np.mean(volume_scores)
        
        # Combine momentum and volume for sentiment
        sentiment = avg_momentum * avg_volume_score
        return max(-1, min(1, sentiment))  # Clamp to -1 to 1
    
    def _analyze_sector_rotation(self, symbol_analyses: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Analyze sector rotation (placeholder)"""
        # In real implementation, this would analyze actual sector ETFs
        return {
            'technology': 0.3,
            'healthcare': 0.2,
            'financials': 0.15,
            'consumer_discretionary': 0.1,
            'industrials': 0.1,
            'energy': 0.05,
            'utilities': 0.05,
            'materials': 0.05
        }
    
    def _check_regime_change(self, current_metrics: MarketMetrics) -> Optional[RegimeChange]:
        """Check if market regime has changed"""
        if len(self.market_history) < 2:
            return None
        
        previous_metrics = self.market_history[-2]
        
        if current_metrics.regime != previous_metrics.regime:
            # Calculate confidence based on multiple factors
            confidence = self._calculate_regime_change_confidence(current_metrics, previous_metrics)
            
            if confidence >= self.regime_change_confidence:
                trigger_factors = self._identify_regime_change_triggers(current_metrics, previous_metrics)
                
                return RegimeChange(
                    timestamp=current_metrics.timestamp,
                    from_regime=previous_metrics.regime,
                    to_regime=current_metrics.regime,
                    confidence=confidence,
                    trigger_factors=trigger_factors,
                    expected_duration='1-2 weeks'  # Placeholder
                )
        
        return None
    
    def _calculate_regime_change_confidence(self, current: MarketMetrics, previous: MarketMetrics) -> float:
        """Calculate confidence in regime change"""
        # Simple confidence calculation based on multiple factors
        volatility_change = abs(current.volatility - previous.volatility)
        momentum_change = abs(current.momentum - previous.momentum)
        sentiment_change = abs(current.market_sentiment - previous.market_sentiment)
        
        # Higher changes = higher confidence
        confidence = min(1.0, (volatility_change + momentum_change + sentiment_change) / 3)
        return confidence
    
    def _identify_regime_change_triggers(self, current: MarketMetrics, previous: MarketMetrics) -> List[str]:
        """Identify factors that triggered regime change"""
        triggers = []
        
        if abs(current.volatility - previous.volatility) > 0.05:
            triggers.append('volatility_change')
        
        if abs(current.momentum - previous.momentum) > 0.02:
            triggers.append('momentum_shift')
        
        if abs(current.market_sentiment - previous.market_sentiment) > 0.3:
            triggers.append('sentiment_change')
        
        if current.volume_trend != previous.volume_trend:
            triggers.append('volume_pattern_change')
        
        return triggers
    
    async def _detect_regime_change(self) -> Dict[str, Any]:
        """Detect recent regime changes"""
        recent_changes = [change for change in self.regime_changes 
                         if change.timestamp > datetime.now() - timedelta(days=7)]
        
        return {
            'recent_changes': len(recent_changes),
            'changes': [
                {
                    'timestamp': change.timestamp.isoformat(),
                    'from_regime': change.from_regime.value,
                    'to_regime': change.to_regime.value,
                    'confidence': change.confidence,
                    'trigger_factors': change.trigger_factors,
                    'expected_duration': change.expected_duration
                }
                for change in recent_changes
            ],
            'total_regime_changes': len(self.regime_changes)
        }
    
    async def _get_trend_analysis(self, timeframe: str) -> Dict[str, Any]:
        """Get trend analysis for specific timeframe"""
        if not self.market_history:
            return {'error': 'No market history available'}
        
        # Analyze recent trends
        recent_metrics = self.market_history[-10:] if len(self.market_history) >= 10 else self.market_history
        
        trends = [metric.trend_strength.value for metric in recent_metrics]
        regimes = [metric.regime.value for metric in recent_metrics]
        
        return {
            'timeframe': timeframe,
            'current_trend': recent_metrics[-1].trend_strength.value if recent_metrics else 'unknown',
            'current_regime': recent_metrics[-1].regime.value if recent_metrics else 'unknown',
            'trend_consistency': len(set(trends)) == 1,  # True if all trends are the same
            'regime_stability': len(set(regimes)) == 1,  # True if all regimes are the same
            'trend_history': trends,
            'regime_history': regimes,
            'analysis_period': len(recent_metrics)
        }
    
    async def _get_volatility_analysis(self) -> Dict[str, Any]:
        """Analyze market volatility"""
        if not self.market_history:
            return {'error': 'No market history available'}
        
        recent_metrics = self.market_history[-20:] if len(self.market_history) >= 20 else self.market_history
        volatilities = [metric.volatility for metric in recent_metrics]
        
        avg_volatility = np.mean(volatilities)
        volatility_trend = 'increasing' if volatilities[-1] > volatilities[0] else 'decreasing'
        
        return {
            'current_volatility': volatilities[-1] if volatilities else 0,
            'average_volatility': avg_volatility,
            'volatility_trend': volatility_trend,
            'volatility_level': 'high' if avg_volatility > self.volatility_threshold_high else 'low' if avg_volatility < self.volatility_threshold_low else 'normal',
            'volatility_history': volatilities,
            'analysis_period': len(recent_metrics)
        }
    
    async def _get_sector_analysis(self) -> Dict[str, Any]:
        """Analyze sector rotation"""
        if not self.market_history:
            return {'error': 'No market history available'}
        
        latest_metrics = self.market_history[-1]
        
        return {
            'sector_rotation': latest_metrics.sector_rotation,
            'top_sector': max(latest_metrics.sector_rotation.items(), key=lambda x: x[1])[0],
            'bottom_sector': min(latest_metrics.sector_rotation.items(), key=lambda x: x[1])[0],
            'rotation_strength': max(latest_metrics.sector_rotation.values()) - min(latest_metrics.sector_rotation.values()),
            'analysis_timestamp': latest_metrics.timestamp.isoformat()
        }
    
    async def _predict_market_direction(self, horizon: str) -> Dict[str, Any]:
        """Predict short-term market direction"""
        if not self.market_history:
            return {'error': 'No market history available'}
        
        recent_metrics = self.market_history[-5:] if len(self.market_history) >= 5 else self.market_history
        
        # Simple prediction based on recent momentum and trend
        avg_momentum = np.mean([metric.momentum for metric in recent_metrics])
        avg_sentiment = np.mean([metric.market_sentiment for metric in recent_metrics])
        
        # Combine momentum and sentiment for prediction
        prediction_score = (avg_momentum + avg_sentiment) / 2
        
        if prediction_score > 0.02:
            direction = 'up'
            confidence = min(0.9, abs(prediction_score) * 10)
        elif prediction_score < -0.02:
            direction = 'down'
            confidence = min(0.9, abs(prediction_score) * 10)
        else:
            direction = 'sideways'
            confidence = 0.5
        
        self.predictions_made += 1
        
        return {
            'horizon': horizon,
            'predicted_direction': direction,
            'confidence': confidence,
            'prediction_score': prediction_score,
            'factors': {
                'momentum': avg_momentum,
                'sentiment': avg_sentiment,
                'trend_strength': recent_metrics[-1].trend_strength.value if recent_metrics else 'unknown'
            },
            'prediction_timestamp': datetime.now().isoformat()
        }
    
    async def _get_support_resistance(self, symbol: str) -> Dict[str, Any]:
        """Calculate support and resistance levels"""
        if not self.market_history:
            return {'error': 'No market history available'}
        
        latest_metrics = self.market_history[-1]
        
        return {
            'symbol': symbol,
            'support_level': latest_metrics.support_level,
            'resistance_level': latest_metrics.resistance_level,
            'current_price': 100.0,  # Placeholder - would get actual price
            'support_distance': 100.0 - latest_metrics.support_level,
            'resistance_distance': latest_metrics.resistance_level - 100.0,
            'analysis_timestamp': latest_metrics.timestamp.isoformat()
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status with market analysis-specific metrics"""
        base_status = super().get_status()
        
        # Add market analysis-specific metrics
        base_status['market_analysis_metrics'] = {
            'analysis_active': self.analysis_active,
            'analyses_performed': self.analyses_performed,
            'regime_changes_detected': self.regime_changes_detected,
            'predictions_made': self.predictions_made,
            'accuracy_score': self.accuracy_score,
            'current_regime': self.current_regime.value,
            'market_history_length': len(self.market_history),
            'regime_changes_history_length': len(self.regime_changes),
            'volatility_threshold_high': self.volatility_threshold_high,
            'volatility_threshold_low': self.volatility_threshold_low,
            'trend_threshold': self.trend_threshold,
            'regime_change_confidence': self.regime_change_confidence
        }
        
        return base_status
