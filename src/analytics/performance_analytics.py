"""
Performance Analytics Module
============================

Provides comprehensive performance tracking and analytics:
- Real-time performance streaming
- Per-strategy attribution
- Execution quality metrics
- Slippage tracking
- Risk-adjusted returns
- Drawdown analysis

Usage:
    from src.analytics.performance_analytics import (
        PerformanceTracker,
        get_performance_tracker,
    )

    tracker = get_performance_tracker()

    # Record a trade
    tracker.record_trade(
        symbol="AAPL",
        side="buy",
        quantity=100,
        entry_price=150.0,
        exit_price=155.0,
        strategy="momentum",
    )

    # Get performance summary
    summary = tracker.get_summary()

    # Get strategy attribution
    attribution = tracker.get_strategy_attribution()
"""

from __future__ import annotations

import logging
import math
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger('trading.analytics')


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TradeRecord:
    """Record of a completed trade."""
    trade_id: str
    symbol: str
    side: str  # buy/sell or long/short
    quantity: float
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    strategy: str
    pnl: float
    pnl_percent: float
    commission: float = 0.0
    slippage: float = 0.0
    expected_price: Optional[float] = None
    holding_period_seconds: float = 0.0
    tags: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.holding_period_seconds == 0 and self.entry_time and self.exit_time:
            self.holding_period_seconds = (self.exit_time - self.entry_time).total_seconds()


@dataclass
class ExecutionMetrics:
    """Execution quality metrics for a trade."""
    trade_id: str
    symbol: str
    expected_fill_price: float
    actual_fill_price: float
    slippage_bps: float
    market_impact_bps: float
    time_to_fill_ms: float
    fill_rate: float  # Percentage filled
    venue: Optional[str] = None


@dataclass
class StrategyPerformance:
    """Performance metrics for a strategy."""
    strategy: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    average_pnl: float
    max_win: float
    max_loss: float
    profit_factor: float  # gross profit / gross loss
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    average_holding_period_hours: float
    average_slippage_bps: float


@dataclass
class DailyPerformance:
    """Daily performance summary."""
    date: date
    starting_equity: float
    ending_equity: float
    daily_pnl: float
    daily_return_percent: float
    num_trades: int
    winning_trades: int
    losing_trades: int
    max_drawdown_intraday: float
    strategies_active: List[str] = field(default_factory=list)


@dataclass
class PerformanceSummary:
    """Overall performance summary."""
    start_date: date
    end_date: date
    initial_capital: float
    current_capital: float
    total_return_percent: float
    annualized_return_percent: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_percent: float
    max_drawdown_duration_days: int
    average_trade_pnl: float
    average_winner: float
    average_loser: float
    largest_winner: float
    largest_loser: float
    average_holding_period_hours: float
    total_commission: float
    total_slippage: float
    strategies: Dict[str, StrategyPerformance] = field(default_factory=dict)


# =============================================================================
# Performance Tracker
# =============================================================================

class PerformanceTracker:
    """Tracks and analyzes trading performance."""

    def __init__(self, initial_capital: float = 100000.0, risk_free_rate: float = 0.02):
        """Initialize performance tracker.

        Args:
            initial_capital: Starting capital
            risk_free_rate: Annual risk-free rate for Sharpe calculation
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_free_rate = risk_free_rate

        # Trade storage
        self._trades: List[TradeRecord] = []
        self._execution_metrics: List[ExecutionMetrics] = []
        self._daily_returns: List[float] = []
        self._equity_curve: List[Tuple[datetime, float]] = []

        # Strategy tracking
        self._strategy_trades: Dict[str, List[TradeRecord]] = defaultdict(list)

        # Real-time tracking
        self._daily_pnl = 0.0
        self._daily_high_water = initial_capital
        self._high_water_mark = initial_capital
        self._max_drawdown = 0.0
        self._drawdown_start: Optional[datetime] = None

        # Threading
        self._lock = threading.Lock()

        # Callbacks for real-time streaming
        self._on_trade_recorded: Optional[Callable[[TradeRecord], None]] = None
        self._on_daily_close: Optional[Callable[[DailyPerformance], None]] = None

        # Initialize equity curve
        self._equity_curve.append((datetime.now(), initial_capital))

        logger.info(f"Performance tracker initialized with ${initial_capital:,.2f}")

    def record_trade(
        self,
        trade_id: str,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        exit_price: float,
        strategy: str,
        entry_time: Optional[datetime] = None,
        exit_time: Optional[datetime] = None,
        commission: float = 0.0,
        expected_price: Optional[float] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> TradeRecord:
        """Record a completed trade.

        Args:
            trade_id: Unique trade identifier
            symbol: Stock symbol
            side: Trade side (buy/sell)
            quantity: Number of shares
            entry_price: Entry price
            exit_price: Exit price
            strategy: Strategy name
            entry_time: Entry timestamp
            exit_time: Exit timestamp
            commission: Total commission paid
            expected_price: Expected fill price (for slippage calc)
            tags: Additional tags/metadata

        Returns:
            TradeRecord object
        """
        entry_time = entry_time or datetime.now() - timedelta(hours=1)
        exit_time = exit_time or datetime.now()

        # Calculate P&L
        if side.lower() in ('buy', 'long'):
            pnl = (exit_price - entry_price) * quantity - commission
        else:
            pnl = (entry_price - exit_price) * quantity - commission

        cost_basis = entry_price * quantity
        pnl_percent = (pnl / cost_basis) * 100 if cost_basis > 0 else 0

        # Calculate slippage
        slippage = 0.0
        if expected_price:
            slippage = abs(entry_price - expected_price) * quantity

        trade = TradeRecord(
            trade_id=trade_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            exit_price=exit_price,
            entry_time=entry_time,
            exit_time=exit_time,
            strategy=strategy,
            pnl=pnl,
            pnl_percent=pnl_percent,
            commission=commission,
            slippage=slippage,
            expected_price=expected_price,
            tags=tags or {},
        )

        with self._lock:
            self._trades.append(trade)
            self._strategy_trades[strategy].append(trade)

            # Update capital
            self.current_capital += pnl
            self._daily_pnl += pnl

            # Update equity curve
            self._equity_curve.append((exit_time, self.current_capital))

            # Update high water mark and drawdown
            if self.current_capital > self._high_water_mark:
                self._high_water_mark = self.current_capital
                self._drawdown_start = None
            else:
                drawdown = (self._high_water_mark - self.current_capital) / self._high_water_mark
                if drawdown > self._max_drawdown:
                    self._max_drawdown = drawdown
                    if self._drawdown_start is None:
                        self._drawdown_start = exit_time

        logger.info(
            f"Trade recorded: {trade_id} {symbol} {side} "
            f"P&L: ${pnl:.2f} ({pnl_percent:.2f}%) Strategy: {strategy}"
        )

        if self._on_trade_recorded:
            self._on_trade_recorded(trade)

        return trade

    def record_execution_metrics(
        self,
        trade_id: str,
        symbol: str,
        expected_fill_price: float,
        actual_fill_price: float,
        time_to_fill_ms: float,
        fill_rate: float = 1.0,
        venue: Optional[str] = None,
    ) -> ExecutionMetrics:
        """Record execution quality metrics.

        Args:
            trade_id: Trade identifier
            symbol: Stock symbol
            expected_fill_price: Expected fill price
            actual_fill_price: Actual fill price
            time_to_fill_ms: Time to fill in milliseconds
            fill_rate: Percentage of order filled (0-1)
            venue: Execution venue

        Returns:
            ExecutionMetrics object
        """
        slippage_bps = ((actual_fill_price - expected_fill_price) / expected_fill_price) * 10000 if expected_fill_price > 0 else 0

        metrics = ExecutionMetrics(
            trade_id=trade_id,
            symbol=symbol,
            expected_fill_price=expected_fill_price,
            actual_fill_price=actual_fill_price,
            slippage_bps=slippage_bps,
            market_impact_bps=abs(slippage_bps),  # Simplified
            time_to_fill_ms=time_to_fill_ms,
            fill_rate=fill_rate,
            venue=venue,
        )

        with self._lock:
            self._execution_metrics.append(metrics)

        return metrics

    def close_day(self, closing_equity: Optional[float] = None) -> DailyPerformance:
        """Close out the trading day and record daily performance.

        Args:
            closing_equity: Ending equity (uses current if not provided)

        Returns:
            DailyPerformance for the day
        """
        with self._lock:
            today = date.today()
            starting = self._daily_high_water
            ending = closing_equity or self.current_capital

            daily_return = ((ending - starting) / starting) * 100 if starting > 0 else 0
            self._daily_returns.append(daily_return)

            # Get today's trades
            today_start = datetime.combine(today, datetime.min.time())
            today_trades = [t for t in self._trades if t.exit_time >= today_start]

            winning = len([t for t in today_trades if t.pnl > 0])
            losing = len([t for t in today_trades if t.pnl < 0])

            strategies = list(set(t.strategy for t in today_trades))

            daily_perf = DailyPerformance(
                date=today,
                starting_equity=starting,
                ending_equity=ending,
                daily_pnl=self._daily_pnl,
                daily_return_percent=daily_return,
                num_trades=len(today_trades),
                winning_trades=winning,
                losing_trades=losing,
                max_drawdown_intraday=self._max_drawdown * 100,
                strategies_active=strategies,
            )

            # Reset daily tracking
            self._daily_pnl = 0.0
            self._daily_high_water = ending

        logger.info(f"Day closed: P&L ${daily_perf.daily_pnl:.2f} ({daily_perf.daily_return_percent:.2f}%)")

        if self._on_daily_close:
            self._on_daily_close(daily_perf)

        return daily_perf

    def get_strategy_performance(self, strategy: str) -> Optional[StrategyPerformance]:
        """Get performance metrics for a specific strategy."""
        with self._lock:
            trades = self._strategy_trades.get(strategy, [])

        if not trades:
            return None

        return self._calculate_strategy_metrics(strategy, trades)

    def _calculate_strategy_metrics(self, strategy: str, trades: List[TradeRecord]) -> StrategyPerformance:
        """Calculate performance metrics for a set of trades."""
        if not trades:
            return StrategyPerformance(
                strategy=strategy,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0,
                total_pnl=0,
                average_pnl=0,
                max_win=0,
                max_loss=0,
                profit_factor=0,
                sharpe_ratio=0,
                sortino_ratio=0,
                max_drawdown=0,
                average_holding_period_hours=0,
                average_slippage_bps=0,
            )

        pnls = [t.pnl for t in trades]
        winners = [t for t in trades if t.pnl > 0]
        losers = [t for t in trades if t.pnl < 0]

        total_pnl = sum(pnls)
        gross_profit = sum(t.pnl for t in winners) if winners else 0
        gross_loss = abs(sum(t.pnl for t in losers)) if losers else 1

        # Calculate Sharpe and Sortino
        returns = [t.pnl_percent for t in trades]
        avg_return = np.mean(returns) if returns else 0
        std_return = np.std(returns) if len(returns) > 1 else 1

        sharpe = (avg_return - self.risk_free_rate / 252) / std_return if std_return > 0 else 0

        downside_returns = [r for r in returns if r < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 1 else 1
        sortino = (avg_return - self.risk_free_rate / 252) / downside_std if downside_std > 0 else 0

        # Calculate max drawdown for strategy
        equity = 0
        peak = 0
        max_dd = 0
        for t in sorted(trades, key=lambda x: x.exit_time):
            equity += t.pnl
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        # Average slippage
        slippages = [t.slippage for t in trades if t.slippage > 0]
        avg_slippage = np.mean(slippages) if slippages else 0

        return StrategyPerformance(
            strategy=strategy,
            total_trades=len(trades),
            winning_trades=len(winners),
            losing_trades=len(losers),
            win_rate=len(winners) / len(trades) * 100 if trades else 0,
            total_pnl=total_pnl,
            average_pnl=np.mean(pnls) if pnls else 0,
            max_win=max(pnls) if pnls else 0,
            max_loss=min(pnls) if pnls else 0,
            profit_factor=gross_profit / gross_loss if gross_loss > 0 else 0,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd * 100,
            average_holding_period_hours=np.mean([t.holding_period_seconds / 3600 for t in trades]),
            average_slippage_bps=avg_slippage,
        )

    def get_strategy_attribution(self) -> Dict[str, StrategyPerformance]:
        """Get performance attribution by strategy."""
        with self._lock:
            strategies = list(self._strategy_trades.keys())

        attribution = {}
        for strategy in strategies:
            perf = self.get_strategy_performance(strategy)
            if perf:
                attribution[strategy] = perf

        return attribution

    def get_execution_quality_summary(self) -> Dict[str, Any]:
        """Get summary of execution quality metrics."""
        with self._lock:
            metrics = list(self._execution_metrics)

        if not metrics:
            return {
                "total_executions": 0,
                "average_slippage_bps": 0,
                "average_time_to_fill_ms": 0,
                "average_fill_rate": 1.0,
            }

        return {
            "total_executions": len(metrics),
            "average_slippage_bps": np.mean([m.slippage_bps for m in metrics]),
            "max_slippage_bps": max(m.slippage_bps for m in metrics),
            "average_time_to_fill_ms": np.mean([m.time_to_fill_ms for m in metrics]),
            "average_fill_rate": np.mean([m.fill_rate for m in metrics]),
            "by_venue": self._group_metrics_by_venue(metrics),
        }

    def _group_metrics_by_venue(self, metrics: List[ExecutionMetrics]) -> Dict[str, Dict]:
        """Group execution metrics by venue."""
        by_venue: Dict[str, List[ExecutionMetrics]] = defaultdict(list)
        for m in metrics:
            venue = m.venue or "unknown"
            by_venue[venue].append(m)

        result = {}
        for venue, venue_metrics in by_venue.items():
            result[venue] = {
                "count": len(venue_metrics),
                "average_slippage_bps": np.mean([m.slippage_bps for m in venue_metrics]),
                "average_time_ms": np.mean([m.time_to_fill_ms for m in venue_metrics]),
            }
        return result

    def get_summary(self) -> PerformanceSummary:
        """Get overall performance summary."""
        with self._lock:
            trades = list(self._trades)
            returns = list(self._daily_returns)

        if not trades:
            return PerformanceSummary(
                start_date=date.today(),
                end_date=date.today(),
                initial_capital=self.initial_capital,
                current_capital=self.current_capital,
                total_return_percent=0,
                annualized_return_percent=0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0,
                profit_factor=0,
                sharpe_ratio=0,
                sortino_ratio=0,
                max_drawdown_percent=0,
                max_drawdown_duration_days=0,
                average_trade_pnl=0,
                average_winner=0,
                average_loser=0,
                largest_winner=0,
                largest_loser=0,
                average_holding_period_hours=0,
                total_commission=0,
                total_slippage=0,
            )

        start_date = min(t.entry_time.date() for t in trades)
        end_date = max(t.exit_time.date() for t in trades)
        days_trading = (end_date - start_date).days + 1

        total_return = ((self.current_capital - self.initial_capital) / self.initial_capital) * 100
        annualized = total_return * (365 / days_trading) if days_trading > 0 else 0

        winners = [t for t in trades if t.pnl > 0]
        losers = [t for t in trades if t.pnl < 0]

        gross_profit = sum(t.pnl for t in winners) if winners else 0
        gross_loss = abs(sum(t.pnl for t in losers)) if losers else 1

        # Sharpe and Sortino
        if len(returns) > 1:
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe = (avg_return * 252 - self.risk_free_rate) / (std_return * math.sqrt(252)) if std_return > 0 else 0

            downside = [r for r in returns if r < 0]
            downside_std = np.std(downside) if len(downside) > 1 else 1
            sortino = (avg_return * 252 - self.risk_free_rate) / (downside_std * math.sqrt(252)) if downside_std > 0 else 0
        else:
            sharpe = 0
            sortino = 0

        # Strategy attribution
        strategies = {}
        for strategy in self._strategy_trades.keys():
            perf = self.get_strategy_performance(strategy)
            if perf:
                strategies[strategy] = perf

        return PerformanceSummary(
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            current_capital=self.current_capital,
            total_return_percent=total_return,
            annualized_return_percent=annualized,
            total_trades=len(trades),
            winning_trades=len(winners),
            losing_trades=len(losers),
            win_rate=len(winners) / len(trades) * 100 if trades else 0,
            profit_factor=gross_profit / gross_loss if gross_loss > 0 else 0,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown_percent=self._max_drawdown * 100,
            max_drawdown_duration_days=0,  # Would need to track
            average_trade_pnl=np.mean([t.pnl for t in trades]),
            average_winner=np.mean([t.pnl for t in winners]) if winners else 0,
            average_loser=np.mean([t.pnl for t in losers]) if losers else 0,
            largest_winner=max(t.pnl for t in trades) if trades else 0,
            largest_loser=min(t.pnl for t in trades) if trades else 0,
            average_holding_period_hours=np.mean([t.holding_period_seconds / 3600 for t in trades]),
            total_commission=sum(t.commission for t in trades),
            total_slippage=sum(t.slippage for t in trades),
            strategies=strategies,
        )

    def get_equity_curve(self) -> List[Tuple[datetime, float]]:
        """Get equity curve data."""
        with self._lock:
            return list(self._equity_curve)

    def set_callbacks(
        self,
        on_trade_recorded: Optional[Callable[[TradeRecord], None]] = None,
        on_daily_close: Optional[Callable[[DailyPerformance], None]] = None,
    ) -> None:
        """Set callback functions for events."""
        self._on_trade_recorded = on_trade_recorded
        self._on_daily_close = on_daily_close

    def reset(self) -> None:
        """Reset all tracking data."""
        with self._lock:
            self._trades.clear()
            self._execution_metrics.clear()
            self._daily_returns.clear()
            self._equity_curve.clear()
            self._strategy_trades.clear()

            self.current_capital = self.initial_capital
            self._daily_pnl = 0.0
            self._daily_high_water = self.initial_capital
            self._high_water_mark = self.initial_capital
            self._max_drawdown = 0.0
            self._drawdown_start = None

            self._equity_curve.append((datetime.now(), self.initial_capital))

        logger.info("Performance tracker reset")


# =============================================================================
# Global Instance
# =============================================================================

_tracker_instance: Optional[PerformanceTracker] = None


def get_performance_tracker(initial_capital: float = 100000.0) -> PerformanceTracker:
    """Get global performance tracker instance."""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = PerformanceTracker(initial_capital=initial_capital)
    return _tracker_instance


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Data classes
    'TradeRecord',
    'ExecutionMetrics',
    'StrategyPerformance',
    'DailyPerformance',
    'PerformanceSummary',
    # Classes
    'PerformanceTracker',
    # Functions
    'get_performance_tracker',
]
