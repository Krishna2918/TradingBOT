"""
VWAP Crossover Strategy

Uses Volume Weighted Average Price (VWAP) as a key support/resistance level
for intraday trading with volume confirmation.
"""

import logging
from datetime import datetime
from typing import Dict, Optional, Any
import pandas as pd
import numpy as np

from .base_strategy import BaseStrategy, StrategySignal, SignalType

logger = logging.getLogger(__name__)


class VWAPCrossoverStrategy(BaseStrategy):
    """
    VWAP Crossover Strategy

    Entry Conditions (Long):
    - Price crosses above VWAP from below
    - Volume is increasing (confirmation)
    - Price momentum is positive
    - Not too far from VWAP (avoid chasing)

    Entry Conditions (Short):
    - Price crosses below VWAP from above
    - Volume is increasing
    - Price momentum is negative
    - Not too far from VWAP

    Exit Conditions:
    - Price crosses back through VWAP
    - Stop loss hit (ATR-based)
    - Take profit hit (2-3x ATR)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("VWAP Crossover", config)

        # Strategy-specific parameters
        self.atr_period = self.config.get('atr_period', 14)
        self.volume_ma_period = self.config.get('volume_ma_period', 20)
        self.max_distance_from_vwap = self.config.get('max_distance_from_vwap', 0.02)  # 2%
        self.momentum_period = self.config.get('momentum_period', 5)

    def get_required_lookback(self) -> int:
        """Required historical data for strategy"""
        return max(self.atr_period, self.volume_ma_period) + 10

    def generate_signal(self, market_data: pd.DataFrame, symbol: str) -> Optional[StrategySignal]:
        """Generate trading signal based on VWAP crossover conditions"""

        if not self.validate_data(market_data):
            return None

        try:
            # Get current and previous values
            current_close = market_data['Close'].iloc[-1]
            prev_close = market_data['Close'].iloc[-2]
            current_volume = market_data['Volume'].iloc[-1]

            # Calculate VWAP
            vwap = self.calculate_vwap(
                market_data['High'],
                market_data['Low'],
                market_data['Close'],
                market_data['Volume']
            )
            current_vwap = vwap.iloc[-1]
            prev_vwap = vwap.iloc[-2]

            # Calculate ATR for stops
            atr = self.calculate_atr(
                market_data['High'],
                market_data['Low'],
                market_data['Close'],
                self.atr_period
            ).iloc[-1]

            # Volume analysis
            avg_volume = market_data['Volume'].iloc[-self.volume_ma_period:].mean()
            volume_increasing = current_volume > avg_volume

            # Momentum calculation
            momentum = current_close - market_data['Close'].iloc[-self.momentum_period]

            # Distance from VWAP
            distance_from_vwap = (current_close - current_vwap) / current_vwap

            # Crossover detection
            crossed_above = prev_close < prev_vwap and current_close > current_vwap
            crossed_below = prev_close > prev_vwap and current_close < current_vwap

            # Not too far from VWAP check
            close_to_vwap = abs(distance_from_vwap) < self.max_distance_from_vwap

            # Long signal: cross above VWAP with confirmation
            if crossed_above and volume_increasing and momentum > 0 and close_to_vwap:
                confidence = self._calculate_confidence(
                    distance_from_vwap, current_volume, avg_volume, momentum, 'long'
                )

                if confidence >= self.min_confidence:
                    stop_loss = current_vwap - atr * 1.5  # Below VWAP
                    take_profit = current_close + atr * 3  # 3x ATR target

                    signal = StrategySignal(
                        signal_type=SignalType.BUY,
                        symbol=symbol,
                        price=current_close,
                        timestamp=datetime.now(),
                        confidence=confidence,
                        strategy_name=self.name,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        reason=f"VWAP crossover (above {current_vwap:.2f})",
                        indicators={
                            'vwap': current_vwap,
                            'distance_from_vwap': distance_from_vwap,
                            'volume_ratio': current_volume / avg_volume,
                            'momentum': momentum,
                            'atr': atr
                        }
                    )
                    self.record_signal(signal)
                    return signal

            # Short signal: cross below VWAP with confirmation
            elif crossed_below and volume_increasing and momentum < 0 and close_to_vwap:
                confidence = self._calculate_confidence(
                    distance_from_vwap, current_volume, avg_volume, momentum, 'short'
                )

                if confidence >= self.min_confidence:
                    stop_loss = current_vwap + atr * 1.5  # Above VWAP
                    take_profit = current_close - atr * 3  # 3x ATR target

                    signal = StrategySignal(
                        signal_type=SignalType.SELL,
                        symbol=symbol,
                        price=current_close,
                        timestamp=datetime.now(),
                        confidence=confidence,
                        strategy_name=self.name,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        reason=f"VWAP breakdown (below {current_vwap:.2f})",
                        indicators={
                            'vwap': current_vwap,
                            'distance_from_vwap': distance_from_vwap,
                            'volume_ratio': current_volume / avg_volume,
                            'momentum': momentum,
                            'atr': atr
                        }
                    )
                    self.record_signal(signal)
                    return signal

            return None

        except Exception as e:
            logger.error(f"Error generating VWAP crossover signal for {symbol}: {e}")
            return None

    def _calculate_confidence(self, distance_from_vwap: float, current_volume: float,
                             avg_volume: float, momentum: float, direction: str) -> float:
        """Calculate signal confidence based on multiple factors"""

        confidence = 0.5  # Base confidence

        # Volume factor (higher = better confirmation)
        volume_ratio = current_volume / avg_volume
        volume_factor = min(0.2, (volume_ratio - 1) * 0.15)
        confidence += volume_factor

        # Proximity to VWAP (closer = better entry)
        proximity_factor = 0.15 * (1 - abs(distance_from_vwap) / self.max_distance_from_vwap)
        confidence += proximity_factor

        # Momentum strength
        momentum_factor = min(0.15, abs(momentum) * 0.05)
        confidence += momentum_factor

        return min(0.95, max(0.3, confidence))
