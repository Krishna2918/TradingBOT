"""
Opening Range Breakout (ORB) Strategy

Classic intraday strategy that trades breakouts from the first 15-30 minutes
trading range, a popular strategy for Canadian markets.
"""

import logging
from datetime import datetime, time
from typing import Dict, Optional, Any
import pandas as pd
import numpy as np
import pytz

from .base_strategy import BaseStrategy, StrategySignal, SignalType

logger = logging.getLogger(__name__)


class OpeningRangeBreakoutStrategy(BaseStrategy):
    """
    Opening Range Breakout (ORB) Strategy

    Setup:
    - Define opening range as first N minutes (default 30 min)
    - Record high and low of opening range
    - Wait for breakout above/below range

    Entry Conditions (Long):
    - Price breaks above opening range high
    - Volume confirmation
    - Within valid trading window

    Entry Conditions (Short):
    - Price breaks below opening range low
    - Volume confirmation
    - Within valid trading window

    Exit Conditions:
    - Stop loss at opposite side of opening range
    - Take profit at 2x range size
    - End of day (15:45 ET)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("Opening Range Breakout", config)

        # Strategy-specific parameters
        self.opening_range_minutes = self.config.get('opening_range_minutes', 30)
        self.breakout_threshold = self.config.get('breakout_threshold', 0.001)  # 0.1%
        self.volume_multiplier = self.config.get('volume_multiplier', 1.2)
        self.atr_period = self.config.get('atr_period', 14)

        # Trading hours (Eastern Time for TSX)
        self.market_open = time(9, 30)
        self.range_end = time(10, 0)  # 9:30 + 30 min
        self.trading_cutoff = time(15, 30)  # Stop trading 30 min before close
        self.timezone = pytz.timezone('America/Toronto')

        # Opening range state
        self.opening_range_high: Optional[float] = None
        self.opening_range_low: Optional[float] = None
        self.opening_range_set: bool = False
        self.range_date: Optional[datetime] = None

    def get_required_lookback(self) -> int:
        """Required historical data for strategy"""
        return max(self.atr_period, 50)

    def reset_opening_range(self):
        """Reset opening range for new trading day"""
        self.opening_range_high = None
        self.opening_range_low = None
        self.opening_range_set = False
        self.range_date = None

    def set_opening_range(self, high: float, low: float, date: datetime):
        """Set the opening range for the day"""
        self.opening_range_high = high
        self.opening_range_low = low
        self.opening_range_set = True
        self.range_date = date
        logger.info(f"Opening range set: High={high:.2f}, Low={low:.2f}")

    def generate_signal(self, market_data: pd.DataFrame, symbol: str) -> Optional[StrategySignal]:
        """Generate trading signal based on opening range breakout"""

        if not self.validate_data(market_data):
            return None

        try:
            current_close = market_data['Close'].iloc[-1]
            current_high = market_data['High'].iloc[-1]
            current_low = market_data['Low'].iloc[-1]
            current_volume = market_data['Volume'].iloc[-1]

            # Get current time
            now = datetime.now(self.timezone)
            current_time = now.time()

            # Check if new trading day - reset opening range
            if self.range_date is None or self.range_date.date() != now.date():
                self.reset_opening_range()

            # Calculate opening range if not set yet
            if not self.opening_range_set:
                if current_time >= self.range_end:
                    # Calculate from first 30 min of data
                    # In real implementation, would filter by time
                    recent_high = market_data['High'].iloc[-10:].max()
                    recent_low = market_data['Low'].iloc[-10:].min()
                    self.set_opening_range(recent_high, recent_low, now)
                else:
                    # Still within opening range period
                    return None

            # Check if within valid trading window
            if current_time < self.range_end or current_time > self.trading_cutoff:
                return None

            # Calculate range size
            range_size = self.opening_range_high - self.opening_range_low
            breakout_buffer = range_size * self.breakout_threshold

            # Calculate ATR
            atr = self.calculate_atr(
                market_data['High'],
                market_data['Low'],
                market_data['Close'],
                self.atr_period
            ).iloc[-1]

            # Volume confirmation
            avg_volume = market_data['Volume'].iloc[-20:].mean()
            volume_confirmation = current_volume > avg_volume * self.volume_multiplier

            # Breakout detection
            long_breakout = (
                current_close > self.opening_range_high + breakout_buffer and
                volume_confirmation
            )

            short_breakout = (
                current_close < self.opening_range_low - breakout_buffer and
                volume_confirmation
            )

            if long_breakout:
                confidence = self._calculate_confidence(
                    current_close, self.opening_range_high, range_size,
                    current_volume, avg_volume, 'long'
                )

                if confidence >= self.min_confidence:
                    # Stop loss at bottom of opening range
                    stop_loss = self.opening_range_low - atr * 0.5

                    # Take profit at 2x range size from breakout
                    take_profit = self.opening_range_high + range_size * 2

                    signal = StrategySignal(
                        signal_type=SignalType.BUY,
                        symbol=symbol,
                        price=current_close,
                        timestamp=datetime.now(),
                        confidence=confidence,
                        strategy_name=self.name,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        reason=f"ORB long breakout above {self.opening_range_high:.2f}",
                        indicators={
                            'opening_range_high': self.opening_range_high,
                            'opening_range_low': self.opening_range_low,
                            'range_size': range_size,
                            'volume_ratio': current_volume / avg_volume,
                            'atr': atr
                        }
                    )
                    self.record_signal(signal)
                    return signal

            elif short_breakout:
                confidence = self._calculate_confidence(
                    current_close, self.opening_range_low, range_size,
                    current_volume, avg_volume, 'short'
                )

                if confidence >= self.min_confidence:
                    # Stop loss at top of opening range
                    stop_loss = self.opening_range_high + atr * 0.5

                    # Take profit at 2x range size from breakdown
                    take_profit = self.opening_range_low - range_size * 2

                    signal = StrategySignal(
                        signal_type=SignalType.SELL,
                        symbol=symbol,
                        price=current_close,
                        timestamp=datetime.now(),
                        confidence=confidence,
                        strategy_name=self.name,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        reason=f"ORB short breakdown below {self.opening_range_low:.2f}",
                        indicators={
                            'opening_range_high': self.opening_range_high,
                            'opening_range_low': self.opening_range_low,
                            'range_size': range_size,
                            'volume_ratio': current_volume / avg_volume,
                            'atr': atr
                        }
                    )
                    self.record_signal(signal)
                    return signal

            return None

        except Exception as e:
            logger.error(f"Error generating ORB signal for {symbol}: {e}")
            return None

    def _calculate_confidence(self, current_close: float, breakout_level: float,
                             range_size: float, current_volume: float,
                             avg_volume: float, direction: str) -> float:
        """Calculate signal confidence based on multiple factors"""

        confidence = 0.5  # Base confidence

        # Breakout strength (how far past the level)
        breakout_distance = abs(current_close - breakout_level)
        breakout_factor = min(0.2, (breakout_distance / range_size) * 0.5)
        confidence += breakout_factor

        # Volume factor
        volume_ratio = current_volume / avg_volume
        volume_factor = min(0.2, (volume_ratio - 1) * 0.15)
        confidence += volume_factor

        # Range size factor (larger ranges = more reliable breakouts)
        range_factor = min(0.1, range_size * 0.05)
        confidence += range_factor

        return min(0.95, max(0.3, confidence))
