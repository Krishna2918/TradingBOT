"""
Regime Detection - Market Condition Analysis

This module detects market regimes using rolling ATR/VIX proxies and categorizes
market conditions as trend vs chop, low vs high volatility.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from src.config.mode_manager import get_current_mode
from src.config.database import execute_query, execute_update

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Market regime enumeration."""
    TRENDING_LOW_VOL = "TRENDING_LOW_VOL"
    TRENDING_HIGH_VOL = "TRENDING_HIGH_VOL"
    CHOPPY_LOW_VOL = "CHOPPY_LOW_VOL"
    CHOPPY_HIGH_VOL = "CHOPPY_HIGH_VOL"
    TRANSITION = "TRANSITION"

class TrendDirection(Enum):
    """Trend direction enumeration."""
    UPTREND = "UPTREND"
    DOWNTREND = "DOWNTREND"
    SIDEWAYS = "SIDEWAYS"

@dataclass
class RegimeState:
    """Represents current market regime state."""
    timestamp: datetime
    regime: MarketRegime
    trend_direction: TrendDirection
    volatility_level: str  # "LOW", "HIGH"
    trend_strength: float  # 0-1, higher = stronger trend
    volatility_ratio: float  # Current vs historical volatility
    atr_percentile: float  # ATR percentile (0-1)
    regime_confidence: float  # Confidence in regime classification
    transition_probability: float  # Probability of regime change
    mode: str

@dataclass
class RegimeMetrics:
    """Regime detection metrics."""
    current_atr: float
    historical_atr_mean: float
    historical_atr_std: float
    atr_percentile: float
    price_trend_slope: float
    trend_consistency: float
    volume_trend: float
    regime_duration: int  # Days in current regime

class RegimeDetector:
    """Detects and tracks market regimes."""
    
    def __init__(self):
        """Initialize Regime Detector."""
        self.lookback_days = 30  # Days to look back for regime calculation
        self.atr_window = 14  # ATR calculation window
        self.trend_window = 20  # Trend detection window
        self.volatility_threshold = 0.7  # Percentile threshold for high volatility
        self.trend_threshold = 0.3  # Minimum trend strength threshold
        self.transition_threshold = 0.4  # Threshold for regime transition
        
        # Regime classification parameters
        self.volatility_multiplier = 1.5  # High vol = 1.5x historical mean
        self.trend_consistency_threshold = 0.6  # Minimum trend consistency
        
        logger.info("Regime Detector initialized")
    
    def detect_current_regime(self, symbol: str = "SPY", mode: Optional[str] = None) -> RegimeState:
        """
        Detect current market regime for a symbol.
        
        Args:
            symbol: Trading symbol (default: SPY for market-wide regime)
            mode: Trading mode (LIVE/DEMO)
            
        Returns:
            Current regime state
        """
        if mode is None:
            mode = get_current_mode()
        
        try:
            # Get market data for regime analysis
            market_data = self._get_market_data(symbol, mode)
            
            if market_data is None or len(market_data) < self.lookback_days:
                logger.warning(f"Insufficient data for regime detection: {symbol}")
                return self._create_default_regime(mode)
            
            # Calculate regime metrics
            metrics = self._calculate_regime_metrics(market_data)
            
            # Classify regime
            regime = self._classify_regime(metrics)
            
            # Calculate regime confidence
            confidence = self._calculate_regime_confidence(metrics, regime)
            
            # Calculate transition probability
            transition_prob = self._calculate_transition_probability(metrics, regime)
            
            # Create regime state
            regime_state = RegimeState(
                timestamp=datetime.now(),
                regime=regime,
                trend_direction=self._determine_trend_direction(metrics),
                volatility_level="HIGH" if metrics.atr_percentile > self.volatility_threshold else "LOW",
                trend_strength=abs(metrics.trend_consistency),
                volatility_ratio=metrics.current_atr / metrics.historical_atr_mean if metrics.historical_atr_mean > 0 else 1.0,
                atr_percentile=metrics.atr_percentile,
                regime_confidence=confidence,
                transition_probability=transition_prob,
                mode=mode
            )
            
            # Log regime detection
            logger.info(f"Regime detected for {symbol}: {regime.value} "
                       f"(confidence: {confidence:.2f}, transition: {transition_prob:.2f})")
            
            return regime_state
            
        except Exception as e:
            logger.error(f"Error detecting regime for {symbol}: {e}")
            return self._create_default_regime(mode)
    
    def _get_market_data(self, symbol: str, mode: str) -> Optional[pd.DataFrame]:
        """Get market data for regime analysis."""
        try:
            # Query for recent market data
            query = """
                SELECT date, open, high, low, close, volume, atr
                FROM market_data 
                WHERE symbol = ? AND mode = ?
                ORDER BY date DESC
                LIMIT ?
            """
            
            result = execute_query(query, (symbol, mode, self.lookback_days), mode)
            
            if not result:
                logger.warning(f"No market data found for {symbol}")
                return None
            
            # Convert to DataFrame with proper column names
            df = pd.DataFrame(result)
            
            # Set column names explicitly (sqlite3.Row objects don't always preserve column names in DataFrame)
            expected_columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'atr']
            if len(df.columns) == len(expected_columns):
                df.columns = expected_columns
            else:
                logger.warning(f"Column count mismatch for {symbol}: expected {len(expected_columns)}, got {len(df.columns)}")
                return None
            
            # Convert date column to datetime
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return None
    
    def _calculate_regime_metrics(self, data: pd.DataFrame) -> RegimeMetrics:
        """Calculate metrics for regime classification."""
        try:
            # Calculate ATR metrics
            current_atr = data['atr'].iloc[-1] if 'atr' in data.columns else self._calculate_atr(data)
            historical_atr = data['atr'].iloc[:-1] if 'atr' in data.columns else self._calculate_rolling_atr(data)
            
            atr_mean = historical_atr.mean()
            atr_std = historical_atr.std()
            atr_percentile = (historical_atr < current_atr).mean()
            
            # Calculate trend metrics
            price_trend_slope = self._calculate_trend_slope(data['close'])
            trend_consistency = self._calculate_trend_consistency(data['close'])
            
            # Calculate volume trend
            volume_trend = self._calculate_volume_trend(data['volume'])
            
            # Calculate regime duration (simplified)
            regime_duration = self._estimate_regime_duration(data)
            
            return RegimeMetrics(
                current_atr=current_atr,
                historical_atr_mean=atr_mean,
                historical_atr_std=atr_std,
                atr_percentile=atr_percentile,
                price_trend_slope=price_trend_slope,
                trend_consistency=trend_consistency,
                volume_trend=volume_trend,
                regime_duration=regime_duration
            )
            
        except Exception as e:
            logger.error(f"Error calculating regime metrics: {e}")
            return self._create_default_metrics()
    
    def _calculate_atr(self, data: pd.DataFrame) -> float:
        """Calculate Average True Range."""
        try:
            high = data['high']
            low = data['low']
            close = data['close']
            
            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate ATR as rolling mean
            atr = true_range.rolling(window=self.atr_window).mean().iloc[-1]
            
            return atr if not pd.isna(atr) else 0.02  # Default 2% ATR
            
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return 0.02
    
    def _calculate_rolling_atr(self, data: pd.DataFrame) -> pd.Series:
        """Calculate rolling ATR for historical comparison."""
        try:
            high = data['high']
            low = data['low']
            close = data['close']
            
            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate rolling ATR
            rolling_atr = true_range.rolling(window=self.atr_window).mean()
            
            return rolling_atr.dropna()
            
        except Exception as e:
            logger.error(f"Error calculating rolling ATR: {e}")
            return pd.Series([0.02] * len(data))
    
    def _calculate_trend_slope(self, prices: pd.Series) -> float:
        """Calculate price trend slope."""
        try:
            if len(prices) < self.trend_window:
                return 0.0
            
            # Use recent prices for trend calculation
            recent_prices = prices.tail(self.trend_window)
            
            # Calculate linear regression slope
            x = np.arange(len(recent_prices))
            y = recent_prices.values
            
            # Simple linear regression
            slope = np.polyfit(x, y, 1)[0]
            
            # Normalize by price level
            normalized_slope = slope / recent_prices.iloc[-1]
            
            return normalized_slope
            
        except Exception as e:
            logger.error(f"Error calculating trend slope: {e}")
            return 0.0
    
    def _calculate_trend_consistency(self, prices: pd.Series) -> float:
        """Calculate trend consistency (how consistent the trend is)."""
        try:
            if len(prices) < self.trend_window:
                return 0.0
            
            # Calculate short-term and long-term trends
            short_window = self.trend_window // 2
            long_window = self.trend_window
            
            short_trend = self._calculate_trend_slope(prices.tail(short_window))
            long_trend = self._calculate_trend_slope(prices.tail(long_window))
            
            # Consistency is how aligned short and long trends are
            if abs(long_trend) < 0.001:  # No clear trend
                return 0.0
            
            consistency = 1.0 - abs(short_trend - long_trend) / abs(long_trend)
            consistency = max(0.0, min(1.0, consistency))  # Clamp to [0, 1]
            
            return consistency
            
        except Exception as e:
            logger.error(f"Error calculating trend consistency: {e}")
            return 0.0
    
    def _calculate_volume_trend(self, volume: pd.Series) -> float:
        """Calculate volume trend."""
        try:
            if len(volume) < 10:
                return 0.0
            
            # Calculate volume trend as slope
            recent_volume = volume.tail(10)
            x = np.arange(len(recent_volume))
            y = recent_volume.values
            
            slope = np.polyfit(x, y, 1)[0]
            
            # Normalize by average volume
            avg_volume = recent_volume.mean()
            normalized_slope = slope / avg_volume if avg_volume > 0 else 0.0
            
            return normalized_slope
            
        except Exception as e:
            logger.error(f"Error calculating volume trend: {e}")
            return 0.0
    
    def _estimate_regime_duration(self, data: pd.DataFrame) -> int:
        """Estimate how long current regime has been active."""
        try:
            # Simplified implementation - in practice, would track regime changes
            # For now, return a default duration
            return 5  # Default 5 days
            
        except Exception as e:
            logger.error(f"Error estimating regime duration: {e}")
            return 1
    
    def _classify_regime(self, metrics: RegimeMetrics) -> MarketRegime:
        """Classify market regime based on metrics."""
        try:
            # Determine volatility level
            is_high_vol = metrics.atr_percentile > self.volatility_threshold
            
            # Determine trend vs chop
            is_trending = abs(metrics.trend_consistency) > self.trend_threshold
            
            # Classify regime
            if is_trending and not is_high_vol:
                return MarketRegime.TRENDING_LOW_VOL
            elif is_trending and is_high_vol:
                return MarketRegime.TRENDING_HIGH_VOL
            elif not is_trending and not is_high_vol:
                return MarketRegime.CHOPPY_LOW_VOL
            elif not is_trending and is_high_vol:
                return MarketRegime.CHOPPY_HIGH_VOL
            else:
                return MarketRegime.TRANSITION
                
        except Exception as e:
            logger.error(f"Error classifying regime: {e}")
            return MarketRegime.TRANSITION
    
    def _determine_trend_direction(self, metrics: RegimeMetrics) -> TrendDirection:
        """Determine trend direction."""
        try:
            if metrics.price_trend_slope > 0.001:  # Uptrend
                return TrendDirection.UPTREND
            elif metrics.price_trend_slope < -0.001:  # Downtrend
                return TrendDirection.DOWNTREND
            else:  # Sideways
                return TrendDirection.SIDEWAYS
                
        except Exception as e:
            logger.error(f"Error determining trend direction: {e}")
            return TrendDirection.SIDEWAYS
    
    def _calculate_regime_confidence(self, metrics: RegimeMetrics, regime: MarketRegime) -> float:
        """Calculate confidence in regime classification."""
        try:
            # Base confidence on how clear the regime signals are
            volatility_confidence = abs(metrics.atr_percentile - 0.5) * 2  # Distance from 50th percentile
            trend_confidence = abs(metrics.trend_consistency)
            
            # Combine confidences
            overall_confidence = (volatility_confidence + trend_confidence) / 2
            
            # Adjust for regime type
            if regime == MarketRegime.TRANSITION:
                overall_confidence *= 0.5  # Lower confidence for transitions
            
            return max(0.1, min(0.95, overall_confidence))  # Clamp to [0.1, 0.95]
            
        except Exception as e:
            logger.error(f"Error calculating regime confidence: {e}")
            return 0.5
    
    def _calculate_transition_probability(self, metrics: RegimeMetrics, regime: MarketRegime) -> float:
        """Calculate probability of regime transition."""
        try:
            # Higher transition probability if metrics are near thresholds
            volatility_near_threshold = abs(metrics.atr_percentile - self.volatility_threshold) < 0.1
            trend_near_threshold = abs(abs(metrics.trend_consistency) - self.trend_threshold) < 0.1
            
            transition_prob = 0.0
            if volatility_near_threshold:
                transition_prob += 0.3
            if trend_near_threshold:
                transition_prob += 0.3
            
            # Higher probability for transition regime
            if regime == MarketRegime.TRANSITION:
                transition_prob += 0.4
            
            return min(0.8, transition_prob)  # Cap at 80%
            
        except Exception as e:
            logger.error(f"Error calculating transition probability: {e}")
            return 0.2
    
    def _create_default_regime(self, mode: str) -> RegimeState:
        """Create default regime state when detection fails."""
        return RegimeState(
            timestamp=datetime.now(),
            regime=MarketRegime.TRANSITION,
            trend_direction=TrendDirection.SIDEWAYS,
            volatility_level="LOW",
            trend_strength=0.0,
            volatility_ratio=1.0,
            atr_percentile=0.5,
            regime_confidence=0.3,
            transition_probability=0.5,
            mode=mode
        )
    
    def _create_default_metrics(self) -> RegimeMetrics:
        """Create default metrics when calculation fails."""
        return RegimeMetrics(
            current_atr=0.02,
            historical_atr_mean=0.02,
            historical_atr_std=0.005,
            atr_percentile=0.5,
            price_trend_slope=0.0,
            trend_consistency=0.0,
            volume_trend=0.0,
            regime_duration=1
        )
    
    def log_regime_state(self, regime_state: RegimeState) -> None:
        """Log regime state to database."""
        try:
            query = """
                INSERT INTO regime_state (
                    timestamp, symbol, regime, trend_direction, volatility_level,
                    trend_strength, volatility_ratio, atr_percentile, 
                    regime_confidence, transition_probability, mode,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            execute_update(query, (
                regime_state.timestamp.isoformat(),
                "SPY",  # Default to SPY for market-wide regime
                regime_state.regime.value,
                regime_state.trend_direction.value,
                regime_state.volatility_level,
                regime_state.trend_strength,
                regime_state.volatility_ratio,
                regime_state.atr_percentile,
                regime_state.regime_confidence,
                regime_state.transition_probability,
                regime_state.mode,
                datetime.now().isoformat()
            ), regime_state.mode)
            
            logger.debug(f"Regime state logged: {regime_state.regime.value}")
            
        except Exception as e:
            logger.error(f"Error logging regime state: {e}")
    
    def get_regime_history(self, symbol: str = "SPY", mode: Optional[str] = None, 
                          limit: int = 30) -> List[Dict[str, Any]]:
        """Get regime history for a symbol."""
        if mode is None:
            mode = get_current_mode()
        
        try:
            query = """
                SELECT * FROM regime_state 
                WHERE symbol = ? AND mode = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """
            
            result = execute_query(query, (symbol, mode, limit), mode)
            return result if result else []
            
        except Exception as e:
            logger.error(f"Error getting regime history for {symbol}: {e}")
            return []

# Global regime detector instance
_regime_detector: Optional[RegimeDetector] = None

def get_regime_detector() -> RegimeDetector:
    """Get the global regime detector instance."""
    global _regime_detector
    if _regime_detector is None:
        _regime_detector = RegimeDetector()
    return _regime_detector

def detect_current_regime(symbol: str = "SPY", mode: Optional[str] = None) -> RegimeState:
    """Detect current market regime."""
    return get_regime_detector().detect_current_regime(symbol, mode)

def log_regime_state(regime_state: RegimeState) -> None:
    """Log regime state to database."""
    return get_regime_detector().log_regime_state(regime_state)

def get_regime_history(symbol: str = "SPY", mode: Optional[str] = None, 
                      limit: int = 30) -> List[Dict[str, Any]]:
    """Get regime history for a symbol."""
    return get_regime_detector().get_regime_history(symbol, mode, limit)