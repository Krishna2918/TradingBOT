"""
Market Condition Handler
========================

Handles extreme market conditions including:
- Gap risk detection (overnight/weekend gaps)
- Flash crash protection (rapid price drops)
- Volatility regime detection
- Liquidity monitoring
- Pre/post-market volatility handling

Usage:
    from src.risk_management.market_condition_handler import (
        MarketConditionHandler,
        get_market_condition_handler,
    )

    handler = get_market_condition_handler()

    # Check for gap risk before market open
    gap_risk = handler.check_gap_risk("AAPL", prev_close=150.0, current_price=145.0)

    # Check for flash crash during trading
    crash_detected = handler.check_flash_crash("AAPL", current_price=140.0)

    # Get overall market conditions
    conditions = handler.get_market_conditions("AAPL")
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time as dt_time
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger('trading.market_conditions')


# =============================================================================
# Enums
# =============================================================================

class MarketRegime(Enum):
    """Market volatility/trend regime."""
    LOW_VOLATILITY = auto()
    NORMAL = auto()
    HIGH_VOLATILITY = auto()
    EXTREME_VOLATILITY = auto()
    TRENDING_UP = auto()
    TRENDING_DOWN = auto()
    RANGING = auto()


class GapType(Enum):
    """Type of price gap."""
    NO_GAP = auto()
    GAP_UP = auto()
    GAP_DOWN = auto()
    SIGNIFICANT_GAP_UP = auto()
    SIGNIFICANT_GAP_DOWN = auto()


class TradingAction(Enum):
    """Recommended trading action based on conditions."""
    NORMAL = auto()          # Trade normally
    REDUCE_SIZE = auto()     # Reduce position sizes
    WIDEN_STOPS = auto()     # Use wider stop losses
    HALT_ENTRIES = auto()    # No new positions
    CLOSE_ALL = auto()       # Close all positions
    EMERGENCY_HALT = auto()  # Stop all trading


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class GapRiskAssessment:
    """Assessment of gap risk for a symbol."""
    symbol: str
    gap_type: GapType
    gap_percent: float
    gap_dollars: float
    previous_close: float
    current_price: float
    risk_level: str  # low, medium, high, extreme
    recommended_action: TradingAction
    timestamp: datetime = field(default_factory=datetime.now)

    def is_significant(self) -> bool:
        """Check if gap is significant (>2%)."""
        return abs(self.gap_percent) >= 2.0


@dataclass
class FlashCrashAssessment:
    """Assessment of flash crash conditions."""
    symbol: str
    is_flash_crash: bool
    price_change_percent: float
    time_window_seconds: float
    current_price: float
    window_high: float
    window_low: float
    recommended_action: TradingAction
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MarketConditions:
    """Overall market conditions for a symbol."""
    symbol: str
    regime: MarketRegime
    volatility_percentile: float  # 0-100
    liquidity_score: float  # 0-1
    bid_ask_spread_bps: float
    is_trading_halted: bool
    gap_risk: Optional[GapRiskAssessment]
    flash_crash_risk: Optional[FlashCrashAssessment]
    recommended_action: TradingAction
    reasons: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PriceSnapshot:
    """Price snapshot for tracking."""
    price: float
    timestamp: datetime
    volume: Optional[int] = None


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class MarketConditionConfig:
    """Configuration for market condition handler."""
    # Gap risk thresholds
    gap_threshold_percent: float = 2.0  # Gap >2% is significant
    extreme_gap_threshold_percent: float = 5.0  # Gap >5% is extreme

    # Flash crash thresholds
    flash_crash_threshold_percent: float = 5.0  # 5% drop in time window
    flash_crash_time_window_seconds: float = 300.0  # 5 minutes
    flash_crash_recovery_seconds: float = 600.0  # 10 min recovery time

    # Volatility thresholds
    high_volatility_percentile: float = 80.0
    extreme_volatility_percentile: float = 95.0

    # Liquidity thresholds
    min_liquidity_score: float = 0.3
    max_bid_ask_spread_bps: float = 50.0  # 0.5%

    # Trading halt conditions
    halt_on_extreme_gap: bool = True
    halt_on_flash_crash: bool = True
    halt_on_extreme_volatility: bool = True

    # Market hours (US Eastern)
    market_open_time: dt_time = dt_time(9, 30)
    market_close_time: dt_time = dt_time(16, 0)
    pre_market_start: dt_time = dt_time(4, 0)
    after_hours_end: dt_time = dt_time(20, 0)


# =============================================================================
# Flash Crash Detector
# =============================================================================

class FlashCrashDetector:
    """Detects flash crash conditions using sliding window."""

    def __init__(
        self,
        threshold_percent: float = 5.0,
        time_window_seconds: float = 300.0,
        recovery_seconds: float = 600.0,
    ):
        self.threshold_percent = threshold_percent
        self.time_window_seconds = time_window_seconds
        self.recovery_seconds = recovery_seconds

        self._price_history: Dict[str, deque] = {}
        self._crash_detected: Dict[str, datetime] = {}
        self._lock = threading.Lock()

    def record_price(self, symbol: str, price: float, timestamp: Optional[datetime] = None) -> None:
        """Record a price observation."""
        ts = timestamp or datetime.now()

        with self._lock:
            if symbol not in self._price_history:
                self._price_history[symbol] = deque(maxlen=1000)

            self._price_history[symbol].append(PriceSnapshot(price=price, timestamp=ts))

            # Clean old data
            self._cleanup_old_data(symbol)

    def _cleanup_old_data(self, symbol: str) -> None:
        """Remove data older than time window."""
        if symbol not in self._price_history:
            return

        cutoff = datetime.now() - timedelta(seconds=self.time_window_seconds * 2)
        history = self._price_history[symbol]

        while history and history[0].timestamp < cutoff:
            history.popleft()

    def check_flash_crash(self, symbol: str, current_price: float) -> FlashCrashAssessment:
        """Check if flash crash conditions exist."""
        now = datetime.now()

        with self._lock:
            # Record current price
            if symbol not in self._price_history:
                self._price_history[symbol] = deque(maxlen=1000)
            self._price_history[symbol].append(PriceSnapshot(price=current_price, timestamp=now))

            # Check if in recovery period
            if symbol in self._crash_detected:
                time_since_crash = (now - self._crash_detected[symbol]).total_seconds()
                if time_since_crash < self.recovery_seconds:
                    return FlashCrashAssessment(
                        symbol=symbol,
                        is_flash_crash=True,
                        price_change_percent=0,
                        time_window_seconds=self.time_window_seconds,
                        current_price=current_price,
                        window_high=current_price,
                        window_low=current_price,
                        recommended_action=TradingAction.HALT_ENTRIES,
                    )

            # Get prices in time window
            cutoff = now - timedelta(seconds=self.time_window_seconds)
            window_prices = [
                s.price for s in self._price_history.get(symbol, [])
                if s.timestamp >= cutoff
            ]

            if len(window_prices) < 2:
                return FlashCrashAssessment(
                    symbol=symbol,
                    is_flash_crash=False,
                    price_change_percent=0,
                    time_window_seconds=self.time_window_seconds,
                    current_price=current_price,
                    window_high=current_price,
                    window_low=current_price,
                    recommended_action=TradingAction.NORMAL,
                )

            window_high = max(window_prices)
            window_low = min(window_prices)

            # Calculate max drop from high
            if window_high > 0:
                drop_percent = ((window_high - current_price) / window_high) * 100
            else:
                drop_percent = 0

            # Check for flash crash
            is_crash = drop_percent >= self.threshold_percent

            if is_crash:
                self._crash_detected[symbol] = now
                logger.warning(
                    f"FLASH CRASH detected: {symbol} dropped {drop_percent:.1f}% "
                    f"in {self.time_window_seconds}s (high: ${window_high:.2f}, "
                    f"current: ${current_price:.2f})"
                )

            return FlashCrashAssessment(
                symbol=symbol,
                is_flash_crash=is_crash,
                price_change_percent=drop_percent,
                time_window_seconds=self.time_window_seconds,
                current_price=current_price,
                window_high=window_high,
                window_low=window_low,
                recommended_action=TradingAction.EMERGENCY_HALT if is_crash else TradingAction.NORMAL,
            )

    def is_in_recovery(self, symbol: str) -> bool:
        """Check if symbol is in flash crash recovery period."""
        with self._lock:
            if symbol not in self._crash_detected:
                return False

            time_since = (datetime.now() - self._crash_detected[symbol]).total_seconds()
            return time_since < self.recovery_seconds

    def get_status(self) -> Dict[str, Any]:
        """Get detector status."""
        with self._lock:
            return {
                "symbols_tracked": len(self._price_history),
                "active_crashes": [
                    s for s, t in self._crash_detected.items()
                    if (datetime.now() - t).total_seconds() < self.recovery_seconds
                ],
                "threshold_percent": self.threshold_percent,
                "time_window_seconds": self.time_window_seconds,
            }


# =============================================================================
# Main Handler
# =============================================================================

class MarketConditionHandler:
    """Handles detection and response to extreme market conditions."""

    def __init__(self, config: Optional[MarketConditionConfig] = None):
        self.config = config or MarketConditionConfig()

        # Flash crash detector
        self._flash_detector = FlashCrashDetector(
            threshold_percent=self.config.flash_crash_threshold_percent,
            time_window_seconds=self.config.flash_crash_time_window_seconds,
            recovery_seconds=self.config.flash_crash_recovery_seconds,
        )

        # Previous close prices for gap detection
        self._previous_closes: Dict[str, float] = {}
        self._lock = threading.Lock()

        # Volatility tracking
        self._volatility_history: Dict[str, deque] = {}

        # Callbacks for condition changes
        self._on_gap_detected: Optional[Callable[[GapRiskAssessment], None]] = None
        self._on_flash_crash: Optional[Callable[[FlashCrashAssessment], None]] = None
        self._on_halt_trading: Optional[Callable[[str, str], None]] = None

        logger.info("Market condition handler initialized")

    def set_previous_close(self, symbol: str, price: float) -> None:
        """Set previous close price for gap detection."""
        with self._lock:
            self._previous_closes[symbol] = price

    def check_gap_risk(
        self,
        symbol: str,
        current_price: float,
        prev_close: Optional[float] = None,
    ) -> GapRiskAssessment:
        """Check for gap risk at market open.

        Args:
            symbol: Stock symbol
            current_price: Current/opening price
            prev_close: Previous close (uses stored value if not provided)

        Returns:
            GapRiskAssessment with gap analysis
        """
        with self._lock:
            if prev_close is None:
                prev_close = self._previous_closes.get(symbol)

        if prev_close is None or prev_close <= 0:
            return GapRiskAssessment(
                symbol=symbol,
                gap_type=GapType.NO_GAP,
                gap_percent=0,
                gap_dollars=0,
                previous_close=0,
                current_price=current_price,
                risk_level="unknown",
                recommended_action=TradingAction.NORMAL,
            )

        # Calculate gap
        gap_dollars = current_price - prev_close
        gap_percent = (gap_dollars / prev_close) * 100

        # Determine gap type
        if abs(gap_percent) < self.config.gap_threshold_percent:
            gap_type = GapType.NO_GAP
            risk_level = "low"
            action = TradingAction.NORMAL
        elif abs(gap_percent) >= self.config.extreme_gap_threshold_percent:
            gap_type = GapType.SIGNIFICANT_GAP_UP if gap_percent > 0 else GapType.SIGNIFICANT_GAP_DOWN
            risk_level = "extreme"
            action = TradingAction.HALT_ENTRIES if self.config.halt_on_extreme_gap else TradingAction.REDUCE_SIZE
        elif gap_percent > 0:
            gap_type = GapType.GAP_UP
            risk_level = "medium" if gap_percent < self.config.extreme_gap_threshold_percent else "high"
            action = TradingAction.WIDEN_STOPS
        else:
            gap_type = GapType.GAP_DOWN
            risk_level = "medium" if abs(gap_percent) < self.config.extreme_gap_threshold_percent else "high"
            action = TradingAction.WIDEN_STOPS

        assessment = GapRiskAssessment(
            symbol=symbol,
            gap_type=gap_type,
            gap_percent=gap_percent,
            gap_dollars=gap_dollars,
            previous_close=prev_close,
            current_price=current_price,
            risk_level=risk_level,
            recommended_action=action,
        )

        if assessment.is_significant():
            logger.warning(
                f"Gap detected: {symbol} {gap_percent:+.1f}% "
                f"(${prev_close:.2f} -> ${current_price:.2f})"
            )
            if self._on_gap_detected:
                self._on_gap_detected(assessment)

        return assessment

    def check_flash_crash(self, symbol: str, current_price: float) -> FlashCrashAssessment:
        """Check for flash crash conditions.

        Args:
            symbol: Stock symbol
            current_price: Current price

        Returns:
            FlashCrashAssessment with analysis
        """
        assessment = self._flash_detector.check_flash_crash(symbol, current_price)

        if assessment.is_flash_crash and self._on_flash_crash:
            self._on_flash_crash(assessment)

        return assessment

    def record_price(self, symbol: str, price: float) -> None:
        """Record a price observation for tracking."""
        self._flash_detector.record_price(symbol, price)

    def get_market_conditions(
        self,
        symbol: str,
        current_price: float,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        volume: Optional[int] = None,
    ) -> MarketConditions:
        """Get overall market conditions for a symbol.

        Args:
            symbol: Stock symbol
            current_price: Current price
            bid: Current bid price
            ask: Current ask price
            volume: Current volume

        Returns:
            MarketConditions with comprehensive analysis
        """
        reasons = []
        recommended_action = TradingAction.NORMAL

        # Check gap risk
        gap_risk = self.check_gap_risk(symbol, current_price)
        if gap_risk.is_significant():
            reasons.append(f"Significant gap: {gap_risk.gap_percent:+.1f}%")
            if gap_risk.recommended_action.value > recommended_action.value:
                recommended_action = gap_risk.recommended_action

        # Check flash crash
        flash_crash = self.check_flash_crash(symbol, current_price)
        if flash_crash.is_flash_crash:
            reasons.append(f"Flash crash: {flash_crash.price_change_percent:.1f}% drop")
            recommended_action = TradingAction.EMERGENCY_HALT

        # Calculate bid-ask spread
        bid_ask_spread_bps = 0
        if bid and ask and bid > 0:
            bid_ask_spread_bps = ((ask - bid) / bid) * 10000
            if bid_ask_spread_bps > self.config.max_bid_ask_spread_bps:
                reasons.append(f"Wide spread: {bid_ask_spread_bps:.0f} bps")
                if recommended_action == TradingAction.NORMAL:
                    recommended_action = TradingAction.REDUCE_SIZE

        # Calculate liquidity score (simplified)
        liquidity_score = 1.0
        if bid_ask_spread_bps > 0:
            liquidity_score = max(0, 1 - (bid_ask_spread_bps / 100))
        if liquidity_score < self.config.min_liquidity_score:
            reasons.append(f"Low liquidity: {liquidity_score:.2f}")
            if recommended_action == TradingAction.NORMAL:
                recommended_action = TradingAction.REDUCE_SIZE

        # Determine regime (simplified - would use actual volatility data)
        regime = MarketRegime.NORMAL
        if flash_crash.is_flash_crash:
            regime = MarketRegime.EXTREME_VOLATILITY
        elif abs(gap_risk.gap_percent) > self.config.extreme_gap_threshold_percent:
            regime = MarketRegime.HIGH_VOLATILITY

        return MarketConditions(
            symbol=symbol,
            regime=regime,
            volatility_percentile=50.0,  # Would calculate from actual data
            liquidity_score=liquidity_score,
            bid_ask_spread_bps=bid_ask_spread_bps,
            is_trading_halted=recommended_action == TradingAction.EMERGENCY_HALT,
            gap_risk=gap_risk,
            flash_crash_risk=flash_crash,
            recommended_action=recommended_action,
            reasons=reasons,
        )

    def should_halt_trading(self, symbol: str, current_price: float) -> Tuple[bool, str]:
        """Check if trading should be halted for a symbol.

        Returns:
            Tuple of (should_halt, reason)
        """
        conditions = self.get_market_conditions(symbol, current_price)

        if conditions.recommended_action == TradingAction.EMERGENCY_HALT:
            reason = "; ".join(conditions.reasons) if conditions.reasons else "Emergency halt"
            if self._on_halt_trading:
                self._on_halt_trading(symbol, reason)
            return True, reason

        if conditions.recommended_action == TradingAction.CLOSE_ALL:
            reason = "; ".join(conditions.reasons) if conditions.reasons else "Close all positions"
            return True, reason

        return False, ""

    def is_market_hours(self, check_time: Optional[datetime] = None) -> bool:
        """Check if within regular market hours (US Eastern)."""
        if check_time is None:
            check_time = datetime.now()

        current_time = check_time.time()
        return self.config.market_open_time <= current_time <= self.config.market_close_time

    def is_extended_hours(self, check_time: Optional[datetime] = None) -> bool:
        """Check if within extended trading hours."""
        if check_time is None:
            check_time = datetime.now()

        current_time = check_time.time()
        return (
            self.config.pre_market_start <= current_time < self.config.market_open_time or
            self.config.market_close_time < current_time <= self.config.after_hours_end
        )

    def set_callbacks(
        self,
        on_gap_detected: Optional[Callable[[GapRiskAssessment], None]] = None,
        on_flash_crash: Optional[Callable[[FlashCrashAssessment], None]] = None,
        on_halt_trading: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        """Set callback functions for market events."""
        self._on_gap_detected = on_gap_detected
        self._on_flash_crash = on_flash_crash
        self._on_halt_trading = on_halt_trading

    def get_status(self) -> Dict[str, Any]:
        """Get handler status."""
        return {
            "flash_crash_detector": self._flash_detector.get_status(),
            "symbols_with_prev_close": len(self._previous_closes),
            "config": {
                "gap_threshold": self.config.gap_threshold_percent,
                "flash_crash_threshold": self.config.flash_crash_threshold_percent,
                "flash_crash_window": self.config.flash_crash_time_window_seconds,
            },
        }


# =============================================================================
# Global Instance
# =============================================================================

_handler_instance: Optional[MarketConditionHandler] = None


def get_market_condition_handler() -> MarketConditionHandler:
    """Get global market condition handler instance."""
    global _handler_instance
    if _handler_instance is None:
        _handler_instance = MarketConditionHandler()
    return _handler_instance


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    'MarketRegime',
    'GapType',
    'TradingAction',
    # Data classes
    'GapRiskAssessment',
    'FlashCrashAssessment',
    'MarketConditions',
    'MarketConditionConfig',
    # Classes
    'FlashCrashDetector',
    'MarketConditionHandler',
    # Functions
    'get_market_condition_handler',
]
