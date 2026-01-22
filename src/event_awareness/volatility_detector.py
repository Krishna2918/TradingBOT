"""
Volatility Detection Module

Detects abnormal volatility patterns and market conditions:
- Historical volatility calculation
- Volatility spikes and regime changes
- VIX-style volatility index
- ATR (Average True Range) analysis
- Bollinger Band width analysis
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class VolatilityRegime:
    """Volatility regime classifications"""
    VERY_LOW = "very_low"
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"

class VolatilityDetector:
    """
    Volatility detection and analysis
    
    Features:
    - Historical volatility (HV) calculation
    - Parkinson, Garman-Klass, Rogers-Satchell estimators
    - Volatility regime classification
    - Spike detection
    - ATR-based volatility
    """
    
    def __init__(
        self,
        lookback_period: int = 20,
        spike_threshold: float = 2.0,  # 2 standard deviations
        regime_thresholds: Dict[str, float] = None
    ):
        self.lookback_period = lookback_period
        self.spike_threshold = spike_threshold
        
        # Default regime thresholds (annualized volatility %)
        self.regime_thresholds = regime_thresholds or {
            VolatilityRegime.VERY_LOW: 10.0,
            VolatilityRegime.LOW: 15.0,
            VolatilityRegime.NORMAL: 25.0,
            VolatilityRegime.HIGH: 40.0,
            VolatilityRegime.EXTREME: 100.0
        }
        
        self.volatility_history: Dict[str, List[Dict]] = {}
        
        logger.info(" Volatility Detector initialized")
    
    def calculate_historical_volatility(
        self,
        prices: pd.Series,
        period: int = 20,
        annualize: bool = True
    ) -> float:
        """
        Calculate historical volatility using close-to-close method
        
        Args:
            prices: Price series
            period: Lookback period
            annualize: Whether to annualize the volatility
        
        Returns:
            Historical volatility as percentage
        """
        
        if len(prices) < period:
            return 0.0
        
        # Calculate log returns
        log_returns = np.log(prices / prices.shift(1))
        
        # Calculate standard deviation
        volatility = log_returns.rolling(window=period).std()
        
        # Annualize (assumes 252 trading days)
        if annualize:
            volatility = volatility * np.sqrt(252)
        
        # Return latest value as percentage
        return volatility.iloc[-1] * 100 if not np.isnan(volatility.iloc[-1]) else 0.0
    
    def calculate_parkinson_volatility(
        self,
        data: pd.DataFrame,
        period: int = 20,
        annualize: bool = True
    ) -> float:
        """
        Calculate Parkinson volatility (uses high-low range)
        More efficient than close-to-close for non-trending markets
        
        Args:
            data: DataFrame with 'high' and 'low' columns
            period: Lookback period
            annualize: Whether to annualize
        
        Returns:
            Parkinson volatility as percentage
        """
        
        if len(data) < period or 'high' not in data.columns or 'low' not in data.columns:
            return 0.0
        
        # Parkinson formula
        hl_ratio = np.log(data['high'] / data['low'])
        parkinson = np.sqrt((hl_ratio ** 2).rolling(window=period).mean() / (4 * np.log(2)))
        
        # Annualize
        if annualize:
            parkinson = parkinson * np.sqrt(252)
        
        return parkinson.iloc[-1] * 100 if not np.isnan(parkinson.iloc[-1]) else 0.0
    
    def calculate_garman_klass_volatility(
        self,
        data: pd.DataFrame,
        period: int = 20,
        annualize: bool = True
    ) -> float:
        """
        Calculate Garman-Klass volatility
        Uses open, high, low, close for more accurate estimation
        
        Args:
            data: DataFrame with OHLC columns
            period: Lookback period
            annualize: Whether to annualize
        
        Returns:
            Garman-Klass volatility as percentage
        """
        
        required_cols = ['open', 'high', 'low', 'close']
        if len(data) < period or not all(col in data.columns for col in required_cols):
            return 0.0
        
        # Garman-Klass formula
        hl = np.log(data['high'] / data['low'])
        co = np.log(data['close'] / data['open'])
        
        gk = 0.5 * (hl ** 2) - (2 * np.log(2) - 1) * (co ** 2)
        volatility = np.sqrt(gk.rolling(window=period).mean())
        
        # Annualize
        if annualize:
            volatility = volatility * np.sqrt(252)
        
        return volatility.iloc[-1] * 100 if not np.isnan(volatility.iloc[-1]) else 0.0
    
    def calculate_atr(
        self,
        data: pd.DataFrame,
        period: int = 14
    ) -> float:
        """
        Calculate Average True Range (ATR)
        
        Args:
            data: DataFrame with OHLC columns
            period: ATR period
        
        Returns:
            ATR value
        """
        
        required_cols = ['high', 'low', 'close']
        if len(data) < period + 1 or not all(col in data.columns for col in required_cols):
            return 0.0
        
        # Calculate True Range
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift(1))
        low_close = np.abs(data['low'] - data['close'].shift(1))
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Calculate ATR (exponential moving average of TR)
        atr = true_range.ewm(span=period, adjust=False).mean()
        
        return atr.iloc[-1] if not np.isnan(atr.iloc[-1]) else 0.0
    
    def calculate_atr_percent(
        self,
        data: pd.DataFrame,
        period: int = 14
    ) -> float:
        """
        Calculate ATR as percentage of price
        
        Returns:
            ATR as percentage
        """
        
        atr = self.calculate_atr(data, period)
        current_price = data['close'].iloc[-1]
        
        if current_price > 0:
            return (atr / current_price) * 100
        return 0.0
    
    def detect_volatility_spike(
        self,
        symbol: str,
        current_volatility: float,
        historical_data: Optional[pd.Series] = None
    ) -> Tuple[bool, float]:
        """
        Detect if current volatility is a spike
        
        Args:
            symbol: Symbol to check
            current_volatility: Current volatility value
            historical_data: Historical volatility series
        
        Returns:
            (is_spike, z_score)
        """
        
        if historical_data is None or len(historical_data) < self.lookback_period:
            return False, 0.0
        
        # Calculate z-score
        mean_vol = historical_data.tail(self.lookback_period).mean()
        std_vol = historical_data.tail(self.lookback_period).std()
        
        if std_vol == 0:
            return False, 0.0
        
        z_score = (current_volatility - mean_vol) / std_vol
        
        is_spike = abs(z_score) > self.spike_threshold
        
        if is_spike:
            logger.warning(
                f" Volatility spike detected for {symbol}: "
                f"{current_volatility:.2f}% (z-score: {z_score:.2f})"
            )
        
        return is_spike, z_score
    
    def classify_volatility_regime(
        self,
        volatility: float
    ) -> str:
        """
        Classify volatility into regime
        
        Args:
            volatility: Annualized volatility percentage
        
        Returns:
            Volatility regime
        """
        
        if volatility < self.regime_thresholds[VolatilityRegime.VERY_LOW]:
            return VolatilityRegime.VERY_LOW
        elif volatility < self.regime_thresholds[VolatilityRegime.LOW]:
            return VolatilityRegime.LOW
        elif volatility < self.regime_thresholds[VolatilityRegime.NORMAL]:
            return VolatilityRegime.NORMAL
        elif volatility < self.regime_thresholds[VolatilityRegime.HIGH]:
            return VolatilityRegime.HIGH
        else:
            return VolatilityRegime.EXTREME
    
    def analyze_volatility(
        self,
        symbol: str,
        data: pd.DataFrame
    ) -> Dict:
        """
        Comprehensive volatility analysis
        
        Args:
            symbol: Symbol to analyze
            data: OHLC data
        
        Returns:
            Volatility analysis results
        """
        
        if len(data) < self.lookback_period:
            logger.warning(f" Insufficient data for {symbol} volatility analysis")
            return {}
        
        # Calculate various volatility measures
        hv = self.calculate_historical_volatility(data['close'], self.lookback_period)
        parkinson = self.calculate_parkinson_volatility(data, self.lookback_period)
        gk = self.calculate_garman_klass_volatility(data, self.lookback_period)
        atr = self.calculate_atr(data, 14)
        atr_pct = self.calculate_atr_percent(data, 14)
        
        # Detect spike
        historical_hv = data['close'].rolling(window=20).apply(
            lambda x: np.std(np.log(x / x.shift(1))) * np.sqrt(252) * 100,
            raw=False
        )
        is_spike, z_score = self.detect_volatility_spike(symbol, hv, historical_hv)
        
        # Classify regime
        regime = self.classify_volatility_regime(hv)
        
        # Calculate volatility trend (increasing/decreasing)
        if len(data) >= self.lookback_period * 2:
            recent_hv = self.calculate_historical_volatility(
                data['close'].tail(self.lookback_period),
                self.lookback_period // 2
            )
            older_hv = self.calculate_historical_volatility(
                data['close'].tail(self.lookback_period * 2).head(self.lookback_period),
                self.lookback_period // 2
            )
            trend = "increasing" if recent_hv > older_hv * 1.1 else "decreasing" if recent_hv < older_hv * 0.9 else "stable"
        else:
            trend = "unknown"
        
        analysis = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'historical_volatility': hv,
            'parkinson_volatility': parkinson,
            'garman_klass_volatility': gk,
            'atr': atr,
            'atr_percent': atr_pct,
            'volatility_regime': regime,
            'is_spike': is_spike,
            'spike_z_score': z_score,
            'trend': trend
        }
        
        # Store in history
        if symbol not in self.volatility_history:
            self.volatility_history[symbol] = []
        
        self.volatility_history[symbol].append(analysis)
        
        # Keep only recent history (last 100 entries)
        self.volatility_history[symbol] = self.volatility_history[symbol][-100:]
        
        logger.info(
            f" Volatility analysis for {symbol}: "
            f"HV={hv:.2f}%, ATR={atr_pct:.2f}%, Regime={regime}, Trend={trend}"
        )
        
        return analysis
    
    def get_volatility_summary(self) -> Dict:
        """Get summary of all tracked volatilities"""
        
        summary = {}
        
        for symbol, history in self.volatility_history.items():
            if not history:
                continue
            
            latest = history[-1]
            summary[symbol] = {
                'current_volatility': latest['historical_volatility'],
                'regime': latest['volatility_regime'],
                'is_spike': latest['is_spike'],
                'trend': latest['trend'],
                'atr_percent': latest['atr_percent']
            }
        
        return summary

# Global detector instance
_detector_instance = None

def get_volatility_detector() -> VolatilityDetector:
    """Get global volatility detector instance"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = VolatilityDetector()
    return _detector_instance

