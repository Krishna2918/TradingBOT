"""
RSI Divergence Strategy

Identifies bullish and bearish divergences between price and RSI indicator
to catch potential trend reversals.
"""

import logging
from datetime import datetime
from typing import Dict, Optional, Any, List, Tuple
import pandas as pd
import numpy as np

from .base_strategy import BaseStrategy, StrategySignal, SignalType

logger = logging.getLogger(__name__)


class RSIDivergenceStrategy(BaseStrategy):
    """
    RSI Divergence Strategy

    Bullish Divergence (Long Signal):
    - Price makes lower lows
    - RSI makes higher lows
    - RSI in oversold zone (< 40)
    - Potential reversal to upside

    Bearish Divergence (Short Signal):
    - Price makes higher highs
    - RSI makes lower highs
    - RSI in overbought zone (> 60)
    - Potential reversal to downside

    Exit Conditions:
    - RSI returns to neutral zone
    - Stop loss hit
    - Price makes new low/high against position
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("RSI Divergence", config)

        # Strategy-specific parameters
        self.rsi_period = self.config.get('rsi_period', 14)
        self.lookback_period = self.config.get('lookback_period', 20)
        self.divergence_threshold = self.config.get('divergence_threshold', 2)  # RSI points
        self.atr_period = self.config.get('atr_period', 14)

        # RSI zones
        self.oversold_zone = self.config.get('oversold_zone', 40)
        self.overbought_zone = self.config.get('overbought_zone', 60)

    def get_required_lookback(self) -> int:
        """Required historical data for strategy"""
        return max(self.rsi_period, self.lookback_period) + 20

    def generate_signal(self, market_data: pd.DataFrame, symbol: str) -> Optional[StrategySignal]:
        """Generate trading signal based on RSI divergence"""

        if not self.validate_data(market_data):
            return None

        try:
            # Calculate RSI
            rsi = self.calculate_rsi(market_data['Close'], self.rsi_period)
            current_rsi = rsi.iloc[-1]
            current_close = market_data['Close'].iloc[-1]

            # Calculate ATR for stops
            atr = self.calculate_atr(
                market_data['High'],
                market_data['Low'],
                market_data['Close'],
                self.atr_period
            ).iloc[-1]

            # Find price and RSI pivots
            price_pivots = self._find_pivots(market_data['Close'].iloc[-self.lookback_period:])
            rsi_pivots = self._find_pivots(rsi.iloc[-self.lookback_period:])

            # Check for bullish divergence
            bullish_div = self._check_bullish_divergence(
                market_data['Close'].iloc[-self.lookback_period:],
                rsi.iloc[-self.lookback_period:],
                current_rsi
            )

            # Check for bearish divergence
            bearish_div = self._check_bearish_divergence(
                market_data['Close'].iloc[-self.lookback_period:],
                rsi.iloc[-self.lookback_period:],
                current_rsi
            )

            if bullish_div['detected'] and current_rsi < self.oversold_zone:
                confidence = self._calculate_confidence(
                    bullish_div['strength'],
                    current_rsi,
                    'bullish'
                )

                if confidence >= self.min_confidence:
                    stop_loss = market_data['Low'].iloc[-self.lookback_period:].min() - atr
                    take_profit = current_close + atr * 3

                    signal = StrategySignal(
                        signal_type=SignalType.BUY,
                        symbol=symbol,
                        price=current_close,
                        timestamp=datetime.now(),
                        confidence=confidence,
                        strategy_name=self.name,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        reason=f"Bullish RSI divergence detected (RSI: {current_rsi:.1f})",
                        indicators={
                            'rsi': current_rsi,
                            'divergence_type': 'bullish',
                            'divergence_strength': bullish_div['strength'],
                            'atr': atr
                        }
                    )
                    self.record_signal(signal)
                    return signal

            elif bearish_div['detected'] and current_rsi > self.overbought_zone:
                confidence = self._calculate_confidence(
                    bearish_div['strength'],
                    current_rsi,
                    'bearish'
                )

                if confidence >= self.min_confidence:
                    stop_loss = market_data['High'].iloc[-self.lookback_period:].max() + atr
                    take_profit = current_close - atr * 3

                    signal = StrategySignal(
                        signal_type=SignalType.SELL,
                        symbol=symbol,
                        price=current_close,
                        timestamp=datetime.now(),
                        confidence=confidence,
                        strategy_name=self.name,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        reason=f"Bearish RSI divergence detected (RSI: {current_rsi:.1f})",
                        indicators={
                            'rsi': current_rsi,
                            'divergence_type': 'bearish',
                            'divergence_strength': bearish_div['strength'],
                            'atr': atr
                        }
                    )
                    self.record_signal(signal)
                    return signal

            return None

        except Exception as e:
            logger.error(f"Error generating RSI divergence signal for {symbol}: {e}")
            return None

    def _find_pivots(self, data: pd.Series, window: int = 5) -> List[Tuple[int, float]]:
        """Find pivot highs and lows in data"""
        pivots = []

        for i in range(window, len(data) - window):
            is_high = all(data.iloc[i] > data.iloc[i-j] for j in range(1, window+1)) and \
                      all(data.iloc[i] > data.iloc[i+j] for j in range(1, window+1))

            is_low = all(data.iloc[i] < data.iloc[i-j] for j in range(1, window+1)) and \
                     all(data.iloc[i] < data.iloc[i+j] for j in range(1, window+1))

            if is_high:
                pivots.append((i, data.iloc[i], 'high'))
            elif is_low:
                pivots.append((i, data.iloc[i], 'low'))

        return pivots

    def _check_bullish_divergence(self, price: pd.Series, rsi: pd.Series,
                                   current_rsi: float) -> Dict[str, Any]:
        """
        Check for bullish divergence:
        - Price makes lower low
        - RSI makes higher low
        """
        result = {'detected': False, 'strength': 0.0}

        try:
            # Look for recent lows
            half_point = len(price) // 2

            # Find lowest points in first and second half
            first_half_low_idx = price.iloc[:half_point].idxmin()
            second_half_low_idx = price.iloc[half_point:].idxmin()

            first_half_low = price.loc[first_half_low_idx]
            second_half_low = price.loc[second_half_low_idx]

            # Get corresponding RSI values
            first_rsi = rsi.loc[first_half_low_idx]
            second_rsi = rsi.loc[second_half_low_idx]

            # Check for bullish divergence
            # Price: lower low, RSI: higher low
            if second_half_low < first_half_low and second_rsi > first_rsi + self.divergence_threshold:
                result['detected'] = True
                # Calculate strength based on divergence magnitude
                price_diff = (first_half_low - second_half_low) / first_half_low
                rsi_diff = second_rsi - first_rsi
                result['strength'] = min(1.0, (price_diff + rsi_diff / 100) * 2)

        except Exception as e:
            logger.debug(f"Error checking bullish divergence: {e}")

        return result

    def _check_bearish_divergence(self, price: pd.Series, rsi: pd.Series,
                                   current_rsi: float) -> Dict[str, Any]:
        """
        Check for bearish divergence:
        - Price makes higher high
        - RSI makes lower high
        """
        result = {'detected': False, 'strength': 0.0}

        try:
            # Look for recent highs
            half_point = len(price) // 2

            # Find highest points in first and second half
            first_half_high_idx = price.iloc[:half_point].idxmax()
            second_half_high_idx = price.iloc[half_point:].idxmax()

            first_half_high = price.loc[first_half_high_idx]
            second_half_high = price.loc[second_half_high_idx]

            # Get corresponding RSI values
            first_rsi = rsi.loc[first_half_high_idx]
            second_rsi = rsi.loc[second_half_high_idx]

            # Check for bearish divergence
            # Price: higher high, RSI: lower high
            if second_half_high > first_half_high and second_rsi < first_rsi - self.divergence_threshold:
                result['detected'] = True
                # Calculate strength based on divergence magnitude
                price_diff = (second_half_high - first_half_high) / first_half_high
                rsi_diff = first_rsi - second_rsi
                result['strength'] = min(1.0, (price_diff + rsi_diff / 100) * 2)

        except Exception as e:
            logger.debug(f"Error checking bearish divergence: {e}")

        return result

    def _calculate_confidence(self, divergence_strength: float, current_rsi: float,
                             divergence_type: str) -> float:
        """Calculate signal confidence based on divergence characteristics"""

        confidence = 0.5  # Base confidence

        # Divergence strength factor
        strength_factor = min(0.25, divergence_strength * 0.3)
        confidence += strength_factor

        # RSI extremity factor (more extreme = higher confidence)
        if divergence_type == 'bullish':
            rsi_factor = min(0.2, (self.oversold_zone - current_rsi) / 100)
        else:
            rsi_factor = min(0.2, (current_rsi - self.overbought_zone) / 100)
        confidence += rsi_factor

        return min(0.95, max(0.3, confidence))
