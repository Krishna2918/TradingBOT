"""
Mean Reversion Strategy

Identifies oversold/overbought conditions and trades the reversion to mean
using Bollinger Bands and RSI confirmation.
"""

import logging
from datetime import datetime
from typing import Dict, Optional, Any
import pandas as pd
import numpy as np

from .base_strategy import BaseStrategy, StrategySignal, SignalType

logger = logging.getLogger(__name__)


class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion Strategy

    Entry Conditions (Long):
    - Price touches or breaks below lower Bollinger Band
    - RSI < 30 (oversold)
    - Price showing signs of reversal (current close > current open)

    Entry Conditions (Short):
    - Price touches or breaks above upper Bollinger Band
    - RSI > 70 (overbought)
    - Price showing signs of reversal (current close < current open)

    Exit Conditions:
    - Price reaches middle Bollinger Band (mean)
    - Stop loss hit
    - RSI returns to neutral (40-60)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("Mean Reversion", config)

        # Strategy-specific parameters
        self.bb_period = self.config.get('bb_period', 20)
        self.bb_std_dev = self.config.get('bb_std_dev', 2.0)
        self.rsi_period = self.config.get('rsi_period', 14)
        self.rsi_oversold = self.config.get('rsi_oversold', 30)
        self.rsi_overbought = self.config.get('rsi_overbought', 70)
        self.atr_period = self.config.get('atr_period', 14)

    def get_required_lookback(self) -> int:
        """Required historical data for strategy"""
        return max(self.bb_period, self.rsi_period) + 10

    def generate_signal(self, market_data: pd.DataFrame, symbol: str) -> Optional[StrategySignal]:
        """Generate trading signal based on mean reversion conditions"""

        if not self.validate_data(market_data):
            return None

        try:
            # Get current values
            current_close = market_data['Close'].iloc[-1]
            current_open = market_data['Open'].iloc[-1]
            current_high = market_data['High'].iloc[-1]
            current_low = market_data['Low'].iloc[-1]

            # Calculate indicators
            bb = self.calculate_bollinger_bands(
                market_data['Close'],
                self.bb_period,
                self.bb_std_dev
            )
            bb_upper = bb['upper'].iloc[-1]
            bb_middle = bb['middle'].iloc[-1]
            bb_lower = bb['lower'].iloc[-1]

            rsi = self.calculate_rsi(market_data['Close'], self.rsi_period).iloc[-1]

            atr = self.calculate_atr(
                market_data['High'],
                market_data['Low'],
                market_data['Close'],
                self.atr_period
            ).iloc[-1]

            # Calculate distance from mean
            distance_from_mean = (current_close - bb_middle) / bb_middle

            # Check for reversal candle patterns
            bullish_reversal = current_close > current_open  # Green candle
            bearish_reversal = current_close < current_open  # Red candle

            # Check for long entry (oversold bounce)
            long_signal = (
                (current_low <= bb_lower or current_close <= bb_lower * 1.005) and
                rsi < self.rsi_oversold and
                bullish_reversal
            )

            # Check for short entry (overbought reversal)
            short_signal = (
                (current_high >= bb_upper or current_close >= bb_upper * 0.995) and
                rsi > self.rsi_overbought and
                bearish_reversal
            )

            if long_signal:
                confidence = self._calculate_confidence(rsi, distance_from_mean, 'long')

                if confidence >= self.min_confidence:
                    # Stop loss below recent low or 1.5x ATR
                    stop_loss = min(
                        current_low - atr * 1.5,
                        market_data['Low'].iloc[-3:].min() - atr * 0.5
                    )

                    # Target is the mean (middle band)
                    take_profit = bb_middle

                    signal = StrategySignal(
                        signal_type=SignalType.BUY,
                        symbol=symbol,
                        price=current_close,
                        timestamp=datetime.now(),
                        confidence=confidence,
                        strategy_name=self.name,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        reason=f"Oversold bounce at BB lower ({rsi:.1f} RSI)",
                        indicators={
                            'rsi': rsi,
                            'bb_lower': bb_lower,
                            'bb_middle': bb_middle,
                            'bb_upper': bb_upper,
                            'distance_from_mean': distance_from_mean,
                            'atr': atr
                        }
                    )
                    self.record_signal(signal)
                    return signal

            elif short_signal:
                confidence = self._calculate_confidence(rsi, distance_from_mean, 'short')

                if confidence >= self.min_confidence:
                    # Stop loss above recent high or 1.5x ATR
                    stop_loss = max(
                        current_high + atr * 1.5,
                        market_data['High'].iloc[-3:].max() + atr * 0.5
                    )

                    # Target is the mean (middle band)
                    take_profit = bb_middle

                    signal = StrategySignal(
                        signal_type=SignalType.SELL,
                        symbol=symbol,
                        price=current_close,
                        timestamp=datetime.now(),
                        confidence=confidence,
                        strategy_name=self.name,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        reason=f"Overbought reversal at BB upper ({rsi:.1f} RSI)",
                        indicators={
                            'rsi': rsi,
                            'bb_lower': bb_lower,
                            'bb_middle': bb_middle,
                            'bb_upper': bb_upper,
                            'distance_from_mean': distance_from_mean,
                            'atr': atr
                        }
                    )
                    self.record_signal(signal)
                    return signal

            return None

        except Exception as e:
            logger.error(f"Error generating mean reversion signal for {symbol}: {e}")
            return None

    def _calculate_confidence(self, rsi: float, distance_from_mean: float,
                             direction: str) -> float:
        """Calculate signal confidence based on multiple factors"""

        confidence = 0.5  # Base confidence

        # RSI factor (more extreme = higher confidence)
        if direction == 'long':
            rsi_factor = min(0.25, (self.rsi_oversold - rsi) / 100)
        else:
            rsi_factor = min(0.25, (rsi - self.rsi_overbought) / 100)
        confidence += rsi_factor

        # Distance from mean factor (further = higher confidence for reversion)
        distance_factor = min(0.2, abs(distance_from_mean) * 2)
        confidence += distance_factor

        return min(0.95, max(0.3, confidence))
