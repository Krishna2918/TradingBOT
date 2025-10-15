"""
Market Condition Analyzer
========================

Comprehensive market state assessment system that analyzes market conditions
and provides recommendations for model selection and trading strategies.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import statistics
import threading

logger = logging.getLogger(__name__)

@dataclass
class MarketCondition:
    """Comprehensive market condition assessment."""
    timestamp: datetime
    regime: str  # 'trending', 'ranging', 'volatile', 'calm'
    volatility_regime: str  # 'low', 'medium', 'high', 'extreme'
    liquidity_regime: str  # 'high', 'medium', 'low'
    sentiment_regime: str  # 'bullish', 'neutral', 'bearish'
    market_phase: str  # 'pre_market', 'open', 'mid_day', 'close', 'after_hours'
    
    # Quantitative metrics
    volatility_zscore: float
    trend_strength: float
    liquidity_score: float
    sentiment_score: float
    correlation_score: float
    sector_dispersion: float
    
    # Market microstructure
    bid_ask_spread: float
    order_flow_imbalance: float
    market_depth: float
    price_impact: float
    
    # Risk metrics
    var_95: float  # Value at Risk 95%
    expected_shortfall: float
    max_drawdown: float
    sharpe_ratio: float
    
    # Recommendations
    suggested_models: List[str]
    risk_level: str  # 'low', 'medium', 'high', 'extreme'
    trading_recommendation: str  # 'aggressive', 'moderate', 'conservative', 'avoid'

@dataclass
class ConditionChange:
    """Record of market condition changes."""
    timestamp: datetime
    previous_condition: MarketCondition
    current_condition: MarketCondition
    change_magnitude: float
    change_type: str  # 'regime_shift', 'volatility_spike', 'liquidity_crisis', etc.
    impact_assessment: str  # 'low', 'medium', 'high', 'critical'

class MarketConditionAnalyzer:
    """
    Comprehensive market condition analysis system.
    
    Features:
    - Multi-dimensional market state assessment
    - Regime detection and classification
    - Risk assessment and recommendations
    - Market microstructure analysis
    - Condition change detection
    - Model selection guidance
    """
    
    def __init__(self, analysis_window: int = 100, change_threshold: float = 0.3):
        """
        Initialize the market condition analyzer.
        
        Args:
            analysis_window: Window size for analysis
            change_threshold: Threshold for detecting significant changes
        """
        self.analyzer_name = "market_condition_analyzer"
        self.analysis_window = analysis_window
        self.change_threshold = change_threshold
        
        # Data storage
        self.market_data: deque = deque(maxlen=analysis_window * 2)
        self.condition_history: deque = deque(maxlen=1000)
        self.condition_changes: deque = deque(maxlen=500)
        
        # Analysis state
        self.current_condition: Optional[MarketCondition] = None
        self.last_analysis = None
        
        # Thresholds and parameters
        self.regime_thresholds = {
            'volatility': {'low': 0.5, 'medium': 1.0, 'high': 2.0},
            'trend': {'weak': 0.3, 'moderate': 0.6, 'strong': 0.8},
            'liquidity': {'low': 0.3, 'medium': 0.6, 'high': 0.8},
            'sentiment': {'bearish': -0.3, 'neutral': 0.3, 'bullish': 0.7}
        }
        
        # Model recommendations by condition
        self.model_recommendations = {
            'trending': ['trend_following_model', 'momentum_model', 'technical_analysis_model'],
            'ranging': ['mean_reversion_model', 'statistical_arbitrage_model', 'range_trading_model'],
            'volatile': ['volatility_model', 'risk_management_model', 'adaptive_model'],
            'calm': ['fundamental_model', 'long_term_model', 'value_model']
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info(f"Market Condition Analyzer initialized: {self.analyzer_name}")
    
    def add_market_data(self, data: Dict[str, Any]) -> bool:
        """
        Add market data for analysis.
        
        Args:
            data: Market data dictionary with OHLCV, indicators, etc.
            
        Returns:
            True if data added successfully
        """
        try:
            with self._lock:
                # Add timestamp if not present
                if 'timestamp' not in data:
                    data['timestamp'] = datetime.now()
                
                # Validate required fields
                required_fields = ['open', 'high', 'low', 'close', 'volume']
                if not all(field in data for field in required_fields):
                    logger.warning("Missing required market data fields")
                    return False
                
                self.market_data.append(data.copy())
                logger.debug(f"Market data added: {data.get('timestamp', 'no_timestamp')}")
                return True
                
        except Exception as e:
            logger.error(f"Error adding market data: {e}")
            return False
    
    def analyze_current_conditions(self) -> MarketCondition:
        """
        Analyze current market conditions.
        
        Returns:
            Current market condition assessment
        """
        try:
            with self._lock:
                if len(self.market_data) < 20:  # Minimum data for analysis
                    return self._get_default_condition()
                
                # Extract recent data for analysis
                recent_data = list(self.market_data)[-self.analysis_window:]
                
                # Calculate various metrics
                volatility_metrics = self._calculate_volatility_metrics(recent_data)
                trend_metrics = self._calculate_trend_metrics(recent_data)
                liquidity_metrics = self._calculate_liquidity_metrics(recent_data)
                sentiment_metrics = self._calculate_sentiment_metrics(recent_data)
                microstructure_metrics = self._calculate_microstructure_metrics(recent_data)
                risk_metrics = self._calculate_risk_metrics(recent_data)
                
                # Determine regimes
                regime = self._determine_regime(volatility_metrics, trend_metrics)
                volatility_regime = self._determine_volatility_regime(volatility_metrics)
                liquidity_regime = self._determine_liquidity_regime(liquidity_metrics)
                sentiment_regime = self._determine_sentiment_regime(sentiment_metrics)
                market_phase = self._determine_market_phase()
                
                # Calculate sector dispersion
                sector_dispersion = self._calculate_sector_dispersion(recent_data)
                
                # Generate recommendations
                suggested_models = self._get_suggested_models(regime, volatility_regime)
                risk_level = self._assess_risk_level(risk_metrics, volatility_regime)
                trading_recommendation = self._get_trading_recommendation(risk_level, regime)
                
                # Create market condition
                condition = MarketCondition(
                    timestamp=datetime.now(),
                    regime=regime,
                    volatility_regime=volatility_regime,
                    liquidity_regime=liquidity_regime,
                    sentiment_regime=sentiment_regime,
                    market_phase=market_phase,
                    volatility_zscore=volatility_metrics['zscore'],
                    trend_strength=trend_metrics['strength'],
                    liquidity_score=liquidity_metrics['score'],
                    sentiment_score=sentiment_metrics['score'],
                    correlation_score=trend_metrics['correlation'],
                    sector_dispersion=sector_dispersion,
                    bid_ask_spread=microstructure_metrics['spread'],
                    order_flow_imbalance=microstructure_metrics['imbalance'],
                    market_depth=microstructure_metrics['depth'],
                    price_impact=microstructure_metrics['impact'],
                    var_95=risk_metrics['var_95'],
                    expected_shortfall=risk_metrics['expected_shortfall'],
                    max_drawdown=risk_metrics['max_drawdown'],
                    sharpe_ratio=risk_metrics['sharpe_ratio'],
                    suggested_models=suggested_models,
                    risk_level=risk_level,
                    trading_recommendation=trading_recommendation
                )
                
                # Check for significant changes
                if self.current_condition is not None:
                    change = self._detect_condition_change(self.current_condition, condition)
                    if change:
                        self.condition_changes.append(change)
                
                # Update state
                self.current_condition = condition
                self.condition_history.append(condition)
                self.last_analysis = datetime.now()
                
                logger.debug(f"Market condition analyzed: {regime} regime, {risk_level} risk")
                return condition
                
        except Exception as e:
            logger.error(f"Error analyzing market conditions: {e}")
            return self._get_default_condition()
    
    def get_condition_summary(self, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """
        Get summary of market conditions over time window.
        
        Args:
            time_window: Time window for summary (None for all time)
            
        Returns:
            Condition summary dictionary
        """
        try:
            with self._lock:
                if not self.condition_history:
                    return {'error': 'No condition history available'}
                
                # Filter by time window
                cutoff_time = datetime.now() - time_window if time_window else None
                recent_conditions = [
                    condition for condition in self.condition_history
                    if cutoff_time is None or condition.timestamp >= cutoff_time
                ]
                
                if not recent_conditions:
                    return {'error': 'No conditions in specified time window'}
                
                # Calculate summary statistics
                regimes = [c.regime for c in recent_conditions]
                volatility_regimes = [c.volatility_regime for c in recent_conditions]
                risk_levels = [c.risk_level for c in recent_conditions]
                
                regime_counts = {regime: regimes.count(regime) for regime in set(regimes)}
                volatility_counts = {regime: volatility_regimes.count(regime) for regime in set(volatility_regimes)}
                risk_counts = {level: risk_levels.count(level) for level in set(risk_levels)}
                
                # Calculate average metrics
                avg_volatility = statistics.mean([c.volatility_zscore for c in recent_conditions])
                avg_trend_strength = statistics.mean([c.trend_strength for c in recent_conditions])
                avg_liquidity = statistics.mean([c.liquidity_score for c in recent_conditions])
                avg_sentiment = statistics.mean([c.sentiment_score for c in recent_conditions])
                
                # Count significant changes
                significant_changes = len([
                    change for change in self.condition_changes
                    if cutoff_time is None or change.timestamp >= cutoff_time
                ])
                
                return {
                    'time_window_seconds': time_window.total_seconds() if time_window else None,
                    'total_conditions': len(recent_conditions),
                    'regime_distribution': regime_counts,
                    'volatility_distribution': volatility_counts,
                    'risk_distribution': risk_counts,
                    'average_metrics': {
                        'volatility_zscore': avg_volatility,
                        'trend_strength': avg_trend_strength,
                        'liquidity_score': avg_liquidity,
                        'sentiment_score': avg_sentiment
                    },
                    'significant_changes': significant_changes,
                    'current_condition': asdict(self.current_condition) if self.current_condition else None,
                    'last_analysis': self.last_analysis.isoformat() if self.last_analysis else None
                }
                
        except Exception as e:
            logger.error(f"Error getting condition summary: {e}")
            return {'error': str(e)}
    
    def get_condition_changes(self, time_window: Optional[timedelta] = None) -> List[ConditionChange]:
        """
        Get significant condition changes.
        
        Args:
            time_window: Time window for changes (None for all time)
            
        Returns:
            List of condition changes
        """
        try:
            with self._lock:
                if not self.condition_changes:
                    return []
                
                cutoff_time = datetime.now() - time_window if time_window else None
                return [
                    change for change in self.condition_changes
                    if cutoff_time is None or change.timestamp >= cutoff_time
                ]
                
        except Exception as e:
            logger.error(f"Error getting condition changes: {e}")
            return []
    
    def _calculate_volatility_metrics(self, data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate volatility-related metrics."""
        try:
            if len(data) < 2:
                return {'zscore': 0.0, 'atr': 0.0, 'volatility': 0.0}
            
            # Calculate returns
            closes = [d['close'] for d in data]
            returns = [np.log(closes[i] / closes[i-1]) for i in range(1, len(closes))]
            
            if not returns:
                return {'zscore': 0.0, 'atr': 0.0, 'volatility': 0.0}
            
            # Calculate volatility metrics
            volatility = np.std(returns) * np.sqrt(252)  # Annualized
            mean_return = np.mean(returns)
            
            # Calculate ATR (simplified)
            high_low = [d['high'] - d['low'] for d in data]
            atr = statistics.mean(high_low) if high_low else 0.0
            
            # Calculate z-score (simplified)
            zscore = (volatility - 0.2) / 0.1 if volatility > 0 else 0.0  # Assuming 20% annual vol
            
            return {
                'zscore': zscore,
                'atr': atr,
                'volatility': volatility,
                'mean_return': mean_return
            }
            
        except Exception as e:
            logger.error(f"Error calculating volatility metrics: {e}")
            return {'zscore': 0.0, 'atr': 0.0, 'volatility': 0.0}
    
    def _calculate_trend_metrics(self, data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate trend-related metrics."""
        try:
            if len(data) < 10:
                return {'strength': 0.0, 'direction': 0.0, 'correlation': 0.0}
            
            closes = [d['close'] for d in data]
            
            # Calculate trend strength using linear regression
            x = np.arange(len(closes))
            y = np.array(closes)
            
            # Simple linear regression
            n = len(x)
            sum_x = np.sum(x)
            sum_y = np.sum(y)
            sum_xy = np.sum(x * y)
            sum_x2 = np.sum(x * x)
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            correlation = np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0.0
            
            # Normalize trend strength
            trend_strength = min(1.0, abs(slope) / (np.std(y) / np.sqrt(n)))
            
            return {
                'strength': trend_strength,
                'direction': 1.0 if slope > 0 else -1.0,
                'correlation': abs(correlation) if not np.isnan(correlation) else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error calculating trend metrics: {e}")
            return {'strength': 0.0, 'direction': 0.0, 'correlation': 0.0}
    
    def _calculate_liquidity_metrics(self, data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate liquidity-related metrics."""
        try:
            if not data:
                return {'score': 0.5, 'volume_trend': 0.0, 'spread': 0.0}
            
            volumes = [d.get('volume', 0) for d in data]
            avg_volume = statistics.mean(volumes) if volumes else 0.0
            
            # Calculate volume trend
            if len(volumes) >= 5:
                recent_avg = statistics.mean(volumes[-5:])
                older_avg = statistics.mean(volumes[-10:-5]) if len(volumes) >= 10 else recent_avg
                volume_trend = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0.0
            else:
                volume_trend = 0.0
            
            # Normalize liquidity score (0-1)
            liquidity_score = min(1.0, max(0.0, (avg_volume / 1000000) * 0.5 + 0.5))  # Simplified
            
            return {
                'score': liquidity_score,
                'volume_trend': volume_trend,
                'spread': 0.001  # Simplified spread
            }
            
        except Exception as e:
            logger.error(f"Error calculating liquidity metrics: {e}")
            return {'score': 0.5, 'volume_trend': 0.0, 'spread': 0.0}
    
    def _calculate_sentiment_metrics(self, data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate sentiment-related metrics."""
        try:
            # Simplified sentiment calculation
            # In a real system, this would use news sentiment, social media, etc.
            
            # Use price momentum as a proxy for sentiment
            if len(data) >= 5:
                recent_closes = [d['close'] for d in data[-5:]]
                older_closes = [d['close'] for d in data[-10:-5]] if len(data) >= 10 else recent_closes
                
                recent_avg = statistics.mean(recent_closes)
                older_avg = statistics.mean(older_closes)
                
                sentiment_score = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0.0
                sentiment_score = max(-1.0, min(1.0, sentiment_score * 10))  # Scale and clamp
            else:
                sentiment_score = 0.0
            
            return {
                'score': sentiment_score,
                'momentum': sentiment_score
            }
            
        except Exception as e:
            logger.error(f"Error calculating sentiment metrics: {e}")
            return {'score': 0.0, 'momentum': 0.0}
    
    def _calculate_microstructure_metrics(self, data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate market microstructure metrics."""
        try:
            # Simplified microstructure metrics
            # In a real system, this would use order book data
            
            return {
                'spread': 0.001,  # 0.1% spread
                'imbalance': 0.0,  # Balanced order flow
                'depth': 0.8,  # Good market depth
                'impact': 0.0005  # Low price impact
            }
            
        except Exception as e:
            logger.error(f"Error calculating microstructure metrics: {e}")
            return {'spread': 0.001, 'imbalance': 0.0, 'depth': 0.5, 'impact': 0.001}
    
    def _calculate_risk_metrics(self, data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate risk metrics."""
        try:
            if len(data) < 20:
                return {'var_95': 0.02, 'expected_shortfall': 0.03, 'max_drawdown': 0.05, 'sharpe_ratio': 0.5}
            
            closes = [d['close'] for d in data]
            returns = [np.log(closes[i] / closes[i-1]) for i in range(1, len(closes))]
            
            if not returns:
                return {'var_95': 0.02, 'expected_shortfall': 0.03, 'max_drawdown': 0.05, 'sharpe_ratio': 0.5}
            
            # Calculate VaR 95%
            returns_sorted = sorted(returns)
            var_95 = abs(returns_sorted[int(0.05 * len(returns_sorted))])
            
            # Calculate expected shortfall
            tail_returns = [r for r in returns if r <= -var_95]
            expected_shortfall = abs(statistics.mean(tail_returns)) if tail_returns else var_95
            
            # Calculate max drawdown
            cumulative_returns = np.cumprod([1 + r for r in returns])
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdown = abs(np.min(drawdowns))
            
            # Calculate Sharpe ratio (simplified)
            mean_return = statistics.mean(returns)
            std_return = statistics.stdev(returns) if len(returns) > 1 else 0.01
            sharpe_ratio = (mean_return * 252) / (std_return * np.sqrt(252)) if std_return > 0 else 0.0
            
            return {
                'var_95': var_95,
                'expected_shortfall': expected_shortfall,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {'var_95': 0.02, 'expected_shortfall': 0.03, 'max_drawdown': 0.05, 'sharpe_ratio': 0.5}
    
    def _determine_regime(self, volatility_metrics: Dict[str, float], trend_metrics: Dict[str, float]) -> str:
        """Determine market regime based on volatility and trend metrics."""
        try:
            volatility = volatility_metrics['zscore']
            trend_strength = trend_metrics['strength']
            
            if volatility > 1.5:
                return 'volatile'
            elif trend_strength > 0.6:
                return 'trending'
            elif trend_strength < 0.3 and volatility < 0.5:
                return 'calm'
            else:
                return 'ranging'
                
        except Exception as e:
            logger.error(f"Error determining regime: {e}")
            return 'ranging'
    
    def _determine_volatility_regime(self, volatility_metrics: Dict[str, float]) -> str:
        """Determine volatility regime."""
        try:
            zscore = volatility_metrics['zscore']
            
            if zscore > 2.0:
                return 'extreme'
            elif zscore > 1.0:
                return 'high'
            elif zscore < -1.0:
                return 'low'
            else:
                return 'medium'
                
        except Exception as e:
            logger.error(f"Error determining volatility regime: {e}")
            return 'medium'
    
    def _determine_liquidity_regime(self, liquidity_metrics: Dict[str, float]) -> str:
        """Determine liquidity regime."""
        try:
            score = liquidity_metrics['score']
            
            if score > 0.7:
                return 'high'
            elif score < 0.3:
                return 'low'
            else:
                return 'medium'
                
        except Exception as e:
            logger.error(f"Error determining liquidity regime: {e}")
            return 'medium'
    
    def _determine_sentiment_regime(self, sentiment_metrics: Dict[str, float]) -> str:
        """Determine sentiment regime."""
        try:
            score = sentiment_metrics['score']
            
            if score > 0.3:
                return 'bullish'
            elif score < -0.3:
                return 'bearish'
            else:
                return 'neutral'
                
        except Exception as e:
            logger.error(f"Error determining sentiment regime: {e}")
            return 'neutral'
    
    def _determine_market_phase(self) -> str:
        """Determine current market phase based on time."""
        try:
            now = datetime.now()
            hour = now.hour
            
            if hour < 9:
                return 'pre_market'
            elif hour < 10:
                return 'open'
            elif hour < 15:
                return 'mid_day'
            elif hour < 16:
                return 'close'
            else:
                return 'after_hours'
                
        except Exception as e:
            logger.error(f"Error determining market phase: {e}")
            return 'mid_day'
    
    def _calculate_sector_dispersion(self, data: List[Dict[str, Any]]) -> float:
        """Calculate sector dispersion (simplified)."""
        try:
            # Simplified sector dispersion calculation
            # In a real system, this would analyze multiple sectors
            return 0.5  # Default moderate dispersion
            
        except Exception as e:
            logger.error(f"Error calculating sector dispersion: {e}")
            return 0.5
    
    def _get_suggested_models(self, regime: str, volatility_regime: str) -> List[str]:
        """Get suggested models based on market conditions."""
        try:
            base_models = self.model_recommendations.get(regime, ['general_model'])
            
            # Add volatility-specific models
            if volatility_regime in ['high', 'extreme']:
                base_models.extend(['volatility_model', 'risk_management_model'])
            
            return base_models[:5]  # Limit to 5 models
            
        except Exception as e:
            logger.error(f"Error getting suggested models: {e}")
            return ['general_model']
    
    def _assess_risk_level(self, risk_metrics: Dict[str, float], volatility_regime: str) -> str:
        """Assess overall risk level."""
        try:
            var_95 = risk_metrics['var_95']
            max_drawdown = risk_metrics['max_drawdown']
            
            if volatility_regime == 'extreme' or var_95 > 0.05 or max_drawdown > 0.1:
                return 'extreme'
            elif volatility_regime == 'high' or var_95 > 0.03 or max_drawdown > 0.05:
                return 'high'
            elif volatility_regime == 'low' and var_95 < 0.01 and max_drawdown < 0.02:
                return 'low'
            else:
                return 'medium'
                
        except Exception as e:
            logger.error(f"Error assessing risk level: {e}")
            return 'medium'
    
    def _get_trading_recommendation(self, risk_level: str, regime: str) -> str:
        """Get trading recommendation based on risk and regime."""
        try:
            if risk_level == 'extreme':
                return 'avoid'
            elif risk_level == 'high':
                return 'conservative'
            elif regime in ['trending', 'volatile'] and risk_level == 'low':
                return 'aggressive'
            else:
                return 'moderate'
                
        except Exception as e:
            logger.error(f"Error getting trading recommendation: {e}")
            return 'moderate'
    
    def _detect_condition_change(self, previous: MarketCondition, current: MarketCondition) -> Optional[ConditionChange]:
        """Detect significant changes in market conditions."""
        try:
            # Calculate change magnitude
            regime_change = 1.0 if previous.regime != current.regime else 0.0
            volatility_change = abs(current.volatility_zscore - previous.volatility_zscore)
            risk_change = 1.0 if previous.risk_level != current.risk_level else 0.0
            
            change_magnitude = (regime_change + volatility_change + risk_change) / 3.0
            
            if change_magnitude < self.change_threshold:
                return None
            
            # Determine change type
            if regime_change > 0:
                change_type = 'regime_shift'
            elif volatility_change > 1.0:
                change_type = 'volatility_spike'
            elif risk_change > 0:
                change_type = 'risk_change'
            else:
                change_type = 'condition_change'
            
            # Assess impact
            if change_magnitude > 0.7:
                impact = 'critical'
            elif change_magnitude > 0.5:
                impact = 'high'
            elif change_magnitude > 0.3:
                impact = 'medium'
            else:
                impact = 'low'
            
            return ConditionChange(
                timestamp=datetime.now(),
                previous_condition=previous,
                current_condition=current,
                change_magnitude=change_magnitude,
                change_type=change_type,
                impact_assessment=impact
            )
            
        except Exception as e:
            logger.error(f"Error detecting condition change: {e}")
            return None
    
    def _get_default_condition(self) -> MarketCondition:
        """Get default market condition when insufficient data."""
        return MarketCondition(
            timestamp=datetime.now(),
            regime='ranging',
            volatility_regime='medium',
            liquidity_regime='medium',
            sentiment_regime='neutral',
            market_phase='mid_day',
            volatility_zscore=0.0,
            trend_strength=0.0,
            liquidity_score=0.5,
            sentiment_score=0.0,
            correlation_score=0.0,
            sector_dispersion=0.5,
            bid_ask_spread=0.001,
            order_flow_imbalance=0.0,
            market_depth=0.5,
            price_impact=0.001,
            var_95=0.02,
            expected_shortfall=0.03,
            max_drawdown=0.05,
            sharpe_ratio=0.5,
            suggested_models=['general_model'],
            risk_level='medium',
            trading_recommendation='moderate'
        )

# Global market condition analyzer instance
_condition_analyzer = None

def get_condition_analyzer() -> MarketConditionAnalyzer:
    """Get the global market condition analyzer instance."""
    global _condition_analyzer
    if _condition_analyzer is None:
        _condition_analyzer = MarketConditionAnalyzer()
    return _condition_analyzer