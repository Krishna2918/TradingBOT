"""
Momentum Breakout Strategy

Identifies strong momentum moves and trades breakouts from consolidation
patterns with volume confirmation.
"""

import logging
from datetime import datetime
from typing import Dict, Optional, Any
import pandas as pd
import numpy as np

from .base_strategy import BaseStrategy, StrategySignal, SignalType

logger = logging.getLogger(__name__)


class MomentumBreakoutStrategy(BaseStrategy):
    """
    Momentum Breakout Strategy

    Entry Conditions (Long):
    - Price breaks above recent high (lookback period)
    - Volume is above average (volume confirmation)
    - RSI > 50 (momentum confirmation)
    - MACD histogram positive

    Entry Conditions (Short):
    - Price breaks below recent low
    - Volume is above average
    - RSI < 50
    - MACD histogram negative

    Exit Conditions:
    - Stop loss hit (2x ATR below entry)
    - Take profit hit (4x ATR above entry)
    - Momentum reversal (RSI cross)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("Momentum Breakout", config)

        # Strategy-specific parameters
        self.lookback_period = self.config.get('lookback_period', 20)
        self.volume_multiplier = self.config.get('volume_multiplier', 1.5)
        self.rsi_period = self.config.get('rsi_period', 14)
        self.atr_period = self.config.get('atr_period', 14)
        self.breakout_threshold = self.config.get('breakout_threshold', 0.002)  # 0.2%

    def get_required_lookback(self) -> int:
        """Required historical data for strategy"""
        return max(self.lookback_period, self.rsi_period, 26) + 10  # MACD needs 26

    def generate_signal(self, market_data: pd.DataFrame, symbol: str) -> Optional[StrategySignal]:
        """Generate trading signal based on momentum breakout conditions"""

        if not self.validate_data(market_data):
            return None

        try:
            # Get current values
            current_close = market_data['Close'].iloc[-1]
            current_high = market_data['High'].iloc[-1]
            current_low = market_data['Low'].iloc[-1]
            current_volume = market_data['Volume'].iloc[-1]

            # Calculate indicators
            rsi = self.calculate_rsi(market_data['Close'], self.rsi_period).iloc[-1]
            macd_data = self.calculate_macd(market_data['Close'])
            macd_histogram = macd_data['histogram'].iloc[-1]

            atr = self.calculate_atr(
                market_data['High'],
                market_data['Low'],
                market_data['Close'],
                self.atr_period
            ).iloc[-1]

            # Calculate breakout levels
            recent_high = market_data['High'].iloc[-self.lookback_period:-1].max()
            recent_low = market_data['Low'].iloc[-self.lookback_period:-1].min()

            # Volume confirmation
            avg_volume = market_data['Volume'].iloc[-20:].mean()
            volume_confirmation = current_volume > avg_volume * self.volume_multiplier

            # Check for breakout conditions
            long_breakout = (
                current_close > recent_high * (1 + self.breakout_threshold) and
                volume_confirmation and
                rsi > 50 and
                macd_histogram > 0
            )

            short_breakout = (
                current_close < recent_low * (1 - self.breakout_threshold) and
                volume_confirmation and
                rsi < 50 and
                macd_histogram < 0
            )

            # Generate signal
            if long_breakout:
                confidence = self._calculate_confidence(rsi, macd_histogram, current_volume, avg_volume, 'long')

                if confidence >= self.min_confidence:
                    stop_loss = self.calculate_stop_loss(current_close, 'long', atr)
                    take_profit = self.calculate_take_profit(current_close, 'long', atr)

                    signal = StrategySignal(
                        signal_type=SignalType.BUY,
                        symbol=symbol,
                        price=current_close,
                        timestamp=datetime.now(),
                        confidence=confidence,
                        strategy_name=self.name,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        reason=f"Momentum breakout above {recent_high:.2f}",
                        indicators={
                            'rsi': rsi,
                            'macd_histogram': macd_histogram,
                            'atr': atr,
                            'volume_ratio': current_volume / avg_volume,
                            'breakout_level': recent_high
                        }
                    )
                    self.record_signal(signal)
                    return signal

            elif short_breakout:
                confidence = self._calculate_confidence(rsi, macd_histogram, current_volume, avg_volume, 'short')

                if confidence >= self.min_confidence:
                    stop_loss = self.calculate_stop_loss(current_close, 'short', atr)
                    take_profit = self.calculate_take_profit(current_close, 'short', atr)

                    signal = StrategySignal(
                        signal_type=SignalType.SELL,
                        symbol=symbol,
                        price=current_close,
                        timestamp=datetime.now(),
                        confidence=confidence,
                        strategy_name=self.name,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        reason=f"Momentum breakdown below {recent_low:.2f}",
                        indicators={
                            'rsi': rsi,
                            'macd_histogram': macd_histogram,
                            'atr': atr,
                            'volume_ratio': current_volume / avg_volume,
                            'breakout_level': recent_low
                        }
                    )
                    self.record_signal(signal)
                    return signal

            return None

        except Exception as e:
            logger.error(f"Error generating momentum breakout signal for {symbol}: {e}")
            return None

    def _calculate_confidence(self, rsi: float, macd_histogram: float,
                             current_volume: float, avg_volume: float,
                             direction: str) -> float:
        """Calculate signal confidence based on multiple factors"""

        confidence = 0.5  # Base confidence

        # RSI factor (stronger momentum = higher confidence)
        if direction == 'long':
            rsi_factor = min(0.2, (rsi - 50) / 100)
        else:
            rsi_factor = min(0.2, (50 - rsi) / 100)
        confidence += rsi_factor

        # MACD factor
        macd_factor = min(0.15, abs(macd_histogram) * 10)
        confidence += macd_factor

        # Volume factor (higher volume = higher confidence)
        volume_ratio = current_volume / avg_volume
        volume_factor = min(0.15, (volume_ratio - 1) * 0.1)
        confidence += volume_factor

        return min(0.95, max(0.3, confidence))
