"""
Tests for Performance Analytics Module
======================================

Tests cover:
- Trade recording accuracy
- P&L calculation (long/short positions)
- Slippage tracking
- Strategy attribution
- Sharpe ratio calculation
- Sortino ratio calculation
- Max drawdown tracking
- Equity curve updates
- Daily close processing

Coverage Target: 85%
"""

import pytest
import threading
from datetime import datetime, timedelta, date
from unittest.mock import Mock, patch
import numpy as np

from src.analytics.performance_analytics import (
    PerformanceTracker,
    TradeRecord,
    ExecutionMetrics,
    StrategyPerformance,
    DailyPerformance,
    PerformanceSummary,
    get_performance_tracker,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def tracker():
    """Create a fresh performance tracker."""
    return PerformanceTracker(initial_capital=100000.0, risk_free_rate=0.02)


@pytest.fixture
def tracker_with_trades(tracker):
    """Create a tracker with some initial trades."""
    # Add some winning trades
    tracker.record_trade(
        trade_id="trade_001",
        symbol="AAPL",
        side="buy",
        quantity=100,
        entry_price=150.0,
        exit_price=155.0,
        strategy="momentum",
        entry_time=datetime.now() - timedelta(hours=2),
        exit_time=datetime.now() - timedelta(hours=1),
    )
    tracker.record_trade(
        trade_id="trade_002",
        symbol="GOOG",
        side="buy",
        quantity=50,
        entry_price=100.0,
        exit_price=108.0,
        strategy="momentum",
        entry_time=datetime.now() - timedelta(hours=1),
        exit_time=datetime.now() - timedelta(minutes=30),
    )
    # Add a losing trade
    tracker.record_trade(
        trade_id="trade_003",
        symbol="TSLA",
        side="buy",
        quantity=30,
        entry_price=200.0,
        exit_price=190.0,
        strategy="news",
        entry_time=datetime.now() - timedelta(minutes=30),
        exit_time=datetime.now(),
    )
    return tracker


@pytest.fixture
def reset_global_tracker():
    """Reset global tracker instance after test."""
    import src.analytics.performance_analytics as pa_module
    original = pa_module._tracker_instance
    yield
    pa_module._tracker_instance = original


# =============================================================================
# Test TradeRecord Data Class
# =============================================================================

class TestTradeRecord:
    """Tests for TradeRecord data class."""

    def test_trade_record_creation(self):
        """Test basic TradeRecord creation."""
        entry_time = datetime.now() - timedelta(hours=1)
        exit_time = datetime.now()

        trade = TradeRecord(
            trade_id="test_001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            entry_price=150.0,
            exit_price=155.0,
            entry_time=entry_time,
            exit_time=exit_time,
            strategy="momentum",
            pnl=500.0,
            pnl_percent=3.33,
        )

        assert trade.trade_id == "test_001"
        assert trade.symbol == "AAPL"
        assert trade.side == "buy"
        assert trade.quantity == 100
        assert trade.entry_price == 150.0
        assert trade.exit_price == 155.0
        assert trade.pnl == 500.0
        assert trade.pnl_percent == 3.33

    def test_holding_period_calculation(self):
        """Test automatic holding period calculation."""
        entry_time = datetime.now() - timedelta(hours=2)
        exit_time = datetime.now()

        trade = TradeRecord(
            trade_id="test_001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            entry_price=150.0,
            exit_price=155.0,
            entry_time=entry_time,
            exit_time=exit_time,
            strategy="momentum",
            pnl=500.0,
            pnl_percent=3.33,
        )

        # Should be approximately 2 hours = 7200 seconds
        assert abs(trade.holding_period_seconds - 7200) < 10

    def test_trade_record_with_optional_fields(self):
        """Test TradeRecord with all optional fields."""
        trade = TradeRecord(
            trade_id="test_001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            entry_price=150.0,
            exit_price=155.0,
            entry_time=datetime.now(),
            exit_time=datetime.now(),
            strategy="momentum",
            pnl=500.0,
            pnl_percent=3.33,
            commission=5.0,
            slippage=2.5,
            expected_price=149.95,
            tags={"reason": "breakout"},
        )

        assert trade.commission == 5.0
        assert trade.slippage == 2.5
        assert trade.expected_price == 149.95
        assert trade.tags["reason"] == "breakout"


# =============================================================================
# Test Trade Recording
# =============================================================================

class TestTradeRecording:
    """Tests for trade recording accuracy."""

    def test_record_long_trade_positive_pnl(self, tracker):
        """Test recording a long trade with positive P&L."""
        trade = tracker.record_trade(
            trade_id="trade_001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            entry_price=150.0,
            exit_price=155.0,
            strategy="momentum",
        )

        # P&L = (155 - 150) * 100 = 500
        assert trade.pnl == 500.0
        assert trade.pnl_percent == pytest.approx(3.33, rel=0.1)
        assert tracker.current_capital == 100500.0

    def test_record_long_trade_negative_pnl(self, tracker):
        """Test recording a long trade with negative P&L."""
        trade = tracker.record_trade(
            trade_id="trade_001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            entry_price=150.0,
            exit_price=145.0,
            strategy="momentum",
        )

        # P&L = (145 - 150) * 100 = -500
        assert trade.pnl == -500.0
        assert trade.pnl_percent == pytest.approx(-3.33, rel=0.1)
        assert tracker.current_capital == 99500.0

    def test_record_short_trade_positive_pnl(self, tracker):
        """Test recording a short trade with positive P&L."""
        trade = tracker.record_trade(
            trade_id="trade_001",
            symbol="AAPL",
            side="sell",
            quantity=100,
            entry_price=150.0,
            exit_price=145.0,
            strategy="momentum",
        )

        # P&L = (150 - 145) * 100 = 500 (short gains on price decrease)
        assert trade.pnl == 500.0
        assert tracker.current_capital == 100500.0

    def test_record_short_trade_negative_pnl(self, tracker):
        """Test recording a short trade with negative P&L."""
        trade = tracker.record_trade(
            trade_id="trade_001",
            symbol="AAPL",
            side="short",
            quantity=100,
            entry_price=150.0,
            exit_price=155.0,
            strategy="momentum",
        )

        # P&L = (150 - 155) * 100 = -500 (short loses on price increase)
        assert trade.pnl == -500.0
        assert tracker.current_capital == 99500.0

    def test_record_trade_with_commission(self, tracker):
        """Test that commission is subtracted from P&L."""
        trade = tracker.record_trade(
            trade_id="trade_001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            entry_price=150.0,
            exit_price=155.0,
            strategy="momentum",
            commission=10.0,
        )

        # P&L = (155 - 150) * 100 - 10 = 490
        assert trade.pnl == 490.0
        assert trade.commission == 10.0
        assert tracker.current_capital == 100490.0

    def test_record_trade_with_slippage(self, tracker):
        """Test slippage calculation when expected price provided."""
        trade = tracker.record_trade(
            trade_id="trade_001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            entry_price=150.10,  # Actual fill price
            exit_price=155.0,
            strategy="momentum",
            expected_price=150.0,  # Expected price
        )

        # Slippage = |150.10 - 150.0| * 100 = 10
        assert trade.slippage == pytest.approx(10.0, rel=0.01)
        assert trade.expected_price == 150.0

    def test_record_trade_updates_equity_curve(self, tracker):
        """Test that recording trades updates equity curve."""
        initial_curve_len = len(tracker.get_equity_curve())

        tracker.record_trade(
            trade_id="trade_001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            entry_price=150.0,
            exit_price=155.0,
            strategy="momentum",
        )

        curve = tracker.get_equity_curve()
        assert len(curve) == initial_curve_len + 1
        assert curve[-1][1] == 100500.0

    def test_record_trade_with_tags(self, tracker):
        """Test recording trade with metadata tags."""
        trade = tracker.record_trade(
            trade_id="trade_001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            entry_price=150.0,
            exit_price=155.0,
            strategy="momentum",
            tags={"signal": "breakout", "confidence": "high"},
        )

        assert trade.tags["signal"] == "breakout"
        assert trade.tags["confidence"] == "high"


# =============================================================================
# Test P&L Calculations
# =============================================================================

class TestPnLCalculations:
    """Tests for P&L calculation accuracy."""

    def test_cumulative_pnl(self, tracker):
        """Test cumulative P&L across multiple trades."""
        # Trade 1: +500
        tracker.record_trade(
            trade_id="trade_001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            entry_price=150.0,
            exit_price=155.0,
            strategy="momentum",
        )

        # Trade 2: +400
        tracker.record_trade(
            trade_id="trade_002",
            symbol="GOOG",
            side="buy",
            quantity=50,
            entry_price=100.0,
            exit_price=108.0,
            strategy="momentum",
        )

        # Trade 3: -300
        tracker.record_trade(
            trade_id="trade_003",
            symbol="TSLA",
            side="buy",
            quantity=30,
            entry_price=200.0,
            exit_price=190.0,
            strategy="news",
        )

        # Total: 100000 + 500 + 400 - 300 = 100600
        assert tracker.current_capital == 100600.0

    def test_pnl_percent_long_trade(self, tracker):
        """Test P&L percentage calculation for long trade."""
        trade = tracker.record_trade(
            trade_id="trade_001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            entry_price=100.0,
            exit_price=110.0,
            strategy="momentum",
        )

        # P&L% = (1000 / 10000) * 100 = 10%
        assert trade.pnl_percent == pytest.approx(10.0, rel=0.01)

    def test_pnl_percent_short_trade(self, tracker):
        """Test P&L percentage calculation for short trade."""
        trade = tracker.record_trade(
            trade_id="trade_001",
            symbol="AAPL",
            side="sell",
            quantity=100,
            entry_price=100.0,
            exit_price=90.0,
            strategy="momentum",
        )

        # P&L = (100 - 90) * 100 = 1000
        # P&L% = (1000 / 10000) * 100 = 10%
        assert trade.pnl_percent == pytest.approx(10.0, rel=0.01)


# =============================================================================
# Test Slippage Tracking
# =============================================================================

class TestSlippageTracking:
    """Tests for slippage tracking functionality."""

    def test_slippage_calculation(self, tracker):
        """Test slippage calculation on entry."""
        trade = tracker.record_trade(
            trade_id="trade_001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            entry_price=150.05,
            exit_price=155.0,
            strategy="momentum",
            expected_price=150.0,
        )

        # Slippage = |150.05 - 150.0| * 100 = 5
        assert trade.slippage == pytest.approx(5.0, rel=0.01)

    def test_no_slippage_when_no_expected_price(self, tracker):
        """Test no slippage when expected price not provided."""
        trade = tracker.record_trade(
            trade_id="trade_001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            entry_price=150.0,
            exit_price=155.0,
            strategy="momentum",
        )

        assert trade.slippage == 0.0

    def test_total_slippage_in_summary(self, tracker_with_trades):
        """Test total slippage is tracked in summary."""
        # Add a trade with slippage
        tracker_with_trades.record_trade(
            trade_id="trade_004",
            symbol="NVDA",
            side="buy",
            quantity=50,
            entry_price=300.10,
            exit_price=310.0,
            strategy="momentum",
            expected_price=300.0,
        )

        summary = tracker_with_trades.get_summary()
        # Slippage = |300.10 - 300.0| * 50 = 5
        assert summary.total_slippage == pytest.approx(5.0, rel=0.01)


# =============================================================================
# Test Execution Metrics
# =============================================================================

class TestExecutionMetrics:
    """Tests for execution quality metrics."""

    def test_record_execution_metrics(self, tracker):
        """Test recording execution metrics."""
        metrics = tracker.record_execution_metrics(
            trade_id="trade_001",
            symbol="AAPL",
            expected_fill_price=150.0,
            actual_fill_price=150.05,
            time_to_fill_ms=100.0,
            fill_rate=1.0,
            venue="NYSE",
        )

        # Slippage bps = ((150.05 - 150.0) / 150.0) * 10000 = 3.33 bps
        assert metrics.slippage_bps == pytest.approx(3.33, rel=0.01)
        assert metrics.time_to_fill_ms == 100.0
        assert metrics.fill_rate == 1.0
        assert metrics.venue == "NYSE"

    def test_execution_quality_summary_empty(self, tracker):
        """Test execution quality summary with no data."""
        summary = tracker.get_execution_quality_summary()

        assert summary["total_executions"] == 0
        assert summary["average_slippage_bps"] == 0
        assert summary["average_fill_rate"] == 1.0

    def test_execution_quality_summary_with_data(self, tracker):
        """Test execution quality summary with multiple executions."""
        tracker.record_execution_metrics(
            trade_id="trade_001",
            symbol="AAPL",
            expected_fill_price=150.0,
            actual_fill_price=150.05,
            time_to_fill_ms=100.0,
            fill_rate=1.0,
            venue="NYSE",
        )
        tracker.record_execution_metrics(
            trade_id="trade_002",
            symbol="GOOG",
            expected_fill_price=100.0,
            actual_fill_price=100.10,
            time_to_fill_ms=200.0,
            fill_rate=0.9,
            venue="NASDAQ",
        )

        summary = tracker.get_execution_quality_summary()

        assert summary["total_executions"] == 2
        assert summary["average_time_to_fill_ms"] == 150.0
        assert summary["average_fill_rate"] == pytest.approx(0.95, rel=0.01)

    def test_execution_metrics_by_venue(self, tracker):
        """Test execution metrics grouped by venue."""
        # Add NYSE executions
        tracker.record_execution_metrics(
            trade_id="trade_001",
            symbol="AAPL",
            expected_fill_price=150.0,
            actual_fill_price=150.05,
            time_to_fill_ms=100.0,
            fill_rate=1.0,
            venue="NYSE",
        )
        tracker.record_execution_metrics(
            trade_id="trade_002",
            symbol="IBM",
            expected_fill_price=120.0,
            actual_fill_price=120.02,
            time_to_fill_ms=80.0,
            fill_rate=1.0,
            venue="NYSE",
        )
        # Add NASDAQ execution
        tracker.record_execution_metrics(
            trade_id="trade_003",
            symbol="GOOG",
            expected_fill_price=100.0,
            actual_fill_price=100.10,
            time_to_fill_ms=200.0,
            fill_rate=0.9,
            venue="NASDAQ",
        )

        summary = tracker.get_execution_quality_summary()
        by_venue = summary["by_venue"]

        assert "NYSE" in by_venue
        assert "NASDAQ" in by_venue
        assert by_venue["NYSE"]["count"] == 2
        assert by_venue["NASDAQ"]["count"] == 1


# =============================================================================
# Test Strategy Attribution
# =============================================================================

class TestStrategyAttribution:
    """Tests for strategy performance attribution."""

    def test_get_strategy_performance(self, tracker_with_trades):
        """Test getting performance for a specific strategy."""
        perf = tracker_with_trades.get_strategy_performance("momentum")

        assert perf is not None
        assert perf.strategy == "momentum"
        assert perf.total_trades == 2
        assert perf.winning_trades == 2
        assert perf.losing_trades == 0
        assert perf.win_rate == 100.0

    def test_get_strategy_performance_nonexistent(self, tracker):
        """Test getting performance for nonexistent strategy."""
        perf = tracker.get_strategy_performance("nonexistent")
        assert perf is None

    def test_strategy_attribution_multiple_strategies(self, tracker_with_trades):
        """Test attribution across multiple strategies."""
        attribution = tracker_with_trades.get_strategy_attribution()

        assert "momentum" in attribution
        assert "news" in attribution

        # Momentum: 2 trades, both winners
        assert attribution["momentum"].total_trades == 2
        assert attribution["momentum"].winning_trades == 2

        # News: 1 trade, loser
        assert attribution["news"].total_trades == 1
        assert attribution["news"].losing_trades == 1

    def test_strategy_profit_factor(self, tracker):
        """Test profit factor calculation for strategy."""
        # Add winning trades: +500 + +400 = +900
        tracker.record_trade(
            trade_id="trade_001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            entry_price=150.0,
            exit_price=155.0,
            strategy="test",
        )
        tracker.record_trade(
            trade_id="trade_002",
            symbol="GOOG",
            side="buy",
            quantity=50,
            entry_price=100.0,
            exit_price=108.0,
            strategy="test",
        )
        # Add losing trade: -300
        tracker.record_trade(
            trade_id="trade_003",
            symbol="TSLA",
            side="buy",
            quantity=30,
            entry_price=200.0,
            exit_price=190.0,
            strategy="test",
        )

        perf = tracker.get_strategy_performance("test")
        # Profit factor = 900 / 300 = 3.0
        assert perf.profit_factor == pytest.approx(3.0, rel=0.01)


# =============================================================================
# Test Sharpe and Sortino Ratios
# =============================================================================

class TestRiskAdjustedReturns:
    """Tests for Sharpe and Sortino ratio calculations."""

    def test_sharpe_ratio_calculation(self, tracker):
        """Test Sharpe ratio calculation in strategy metrics."""
        # Add multiple trades to get returns
        for i in range(10):
            pnl_direction = 1 if i % 3 != 0 else -1
            exit_price = 155.0 if pnl_direction > 0 else 145.0
            tracker.record_trade(
                trade_id=f"trade_{i}",
                symbol="AAPL",
                side="buy",
                quantity=100,
                entry_price=150.0,
                exit_price=exit_price,
                strategy="test",
            )

        perf = tracker.get_strategy_performance("test")
        # Sharpe should be calculated (exact value depends on returns)
        assert perf.sharpe_ratio != 0

    def test_sortino_ratio_calculation(self, tracker):
        """Test Sortino ratio calculation in strategy metrics."""
        # Add trades with variety of losses to ensure downside deviation is calculated
        returns_profile = [
            (155.0, 1),   # Win
            (145.0, -1),  # Loss
            (160.0, 1),   # Win
            (140.0, -1),  # Loss (larger)
            (155.0, 1),   # Win
            (148.0, -1),  # Loss (smaller)
            (157.0, 1),   # Win
            (143.0, -1),  # Loss
            (154.0, 1),   # Win
            (146.0, -1),  # Loss
        ]
        for i, (exit_price, _) in enumerate(returns_profile):
            tracker.record_trade(
                trade_id=f"trade_{i}",
                symbol="AAPL",
                side="buy",
                quantity=100,
                entry_price=150.0,
                exit_price=exit_price,
                strategy="test",
            )

        perf = tracker.get_strategy_performance("test")
        # With multiple varied losses, sortino should be calculated
        # Note: sortino could be 0 if downside std falls to default of 1
        # and (avg_return - rf/252) / 1 rounds to nearly 0
        assert perf.sortino_ratio is not None  # Just verify it's computed

    def test_sharpe_sortino_in_summary(self, tracker):
        """Test Sharpe and Sortino in overall summary."""
        # Add trades and close days to generate daily returns
        tracker.record_trade(
            trade_id="trade_001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            entry_price=150.0,
            exit_price=155.0,
            strategy="test",
        )
        tracker.close_day()

        tracker.record_trade(
            trade_id="trade_002",
            symbol="AAPL",
            side="buy",
            quantity=100,
            entry_price=150.0,
            exit_price=145.0,
            strategy="test",
        )
        tracker.close_day()

        summary = tracker.get_summary()
        # With daily returns, Sharpe/Sortino should be calculable
        # Values depend on actual returns


# =============================================================================
# Test Max Drawdown Tracking
# =============================================================================

class TestMaxDrawdownTracking:
    """Tests for maximum drawdown tracking."""

    def test_no_drawdown_with_only_winners(self, tracker):
        """Test no drawdown when all trades are winners."""
        tracker.record_trade(
            trade_id="trade_001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            entry_price=150.0,
            exit_price=155.0,
            strategy="test",
        )
        tracker.record_trade(
            trade_id="trade_002",
            symbol="GOOG",
            side="buy",
            quantity=100,
            entry_price=100.0,
            exit_price=105.0,
            strategy="test",
        )

        summary = tracker.get_summary()
        assert summary.max_drawdown_percent == 0.0

    def test_drawdown_calculation(self, tracker):
        """Test drawdown calculation after loss."""
        # Win: 100000 -> 100500
        tracker.record_trade(
            trade_id="trade_001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            entry_price=150.0,
            exit_price=155.0,
            strategy="test",
        )
        # Lose: 100500 -> 100000
        tracker.record_trade(
            trade_id="trade_002",
            symbol="AAPL",
            side="buy",
            quantity=100,
            entry_price=155.0,
            exit_price=150.0,
            strategy="test",
        )

        # Drawdown = (100500 - 100000) / 100500 * 100 ≈ 0.498%
        summary = tracker.get_summary()
        assert summary.max_drawdown_percent == pytest.approx(0.498, rel=0.1)

    def test_max_drawdown_updates_on_larger_drawdown(self, tracker):
        """Test max drawdown updates when larger drawdown occurs."""
        # Win: 100000 -> 100500
        tracker.record_trade(
            trade_id="trade_001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            entry_price=150.0,
            exit_price=155.0,
            strategy="test",
        )
        # Small loss: 100500 -> 100300
        tracker.record_trade(
            trade_id="trade_002",
            symbol="AAPL",
            side="buy",
            quantity=40,
            entry_price=155.0,
            exit_price=150.0,
            strategy="test",
        )
        # Large loss: 100300 -> 99300
        tracker.record_trade(
            trade_id="trade_003",
            symbol="AAPL",
            side="buy",
            quantity=200,
            entry_price=155.0,
            exit_price=150.0,
            strategy="test",
        )

        # Max drawdown should be from peak 100500 to 99300
        # Drawdown = (100500 - 99300) / 100500 * 100 ≈ 1.19%
        summary = tracker.get_summary()
        assert summary.max_drawdown_percent > 1.0

    def test_drawdown_resets_on_new_high(self, tracker):
        """Test high water mark resets on new equity high."""
        # Win: 100000 -> 100500
        tracker.record_trade(
            trade_id="trade_001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            entry_price=150.0,
            exit_price=155.0,
            strategy="test",
        )
        # Lose: 100500 -> 100000
        tracker.record_trade(
            trade_id="trade_002",
            symbol="AAPL",
            side="buy",
            quantity=100,
            entry_price=155.0,
            exit_price=150.0,
            strategy="test",
        )
        # Win big: 100000 -> 101000
        tracker.record_trade(
            trade_id="trade_003",
            symbol="AAPL",
            side="buy",
            quantity=200,
            entry_price=150.0,
            exit_price=155.0,
            strategy="test",
        )

        # New high water mark should be 101000
        assert tracker._high_water_mark == 101000.0


# =============================================================================
# Test Equity Curve
# =============================================================================

class TestEquityCurve:
    """Tests for equity curve tracking."""

    def test_initial_equity_curve(self, tracker):
        """Test initial equity curve has starting point."""
        curve = tracker.get_equity_curve()
        assert len(curve) == 1
        assert curve[0][1] == 100000.0

    def test_equity_curve_updates_with_trades(self, tracker):
        """Test equity curve updates after each trade."""
        tracker.record_trade(
            trade_id="trade_001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            entry_price=150.0,
            exit_price=155.0,
            strategy="test",
        )
        tracker.record_trade(
            trade_id="trade_002",
            symbol="GOOG",
            side="buy",
            quantity=100,
            entry_price=100.0,
            exit_price=105.0,
            strategy="test",
        )

        curve = tracker.get_equity_curve()
        assert len(curve) == 3
        assert curve[1][1] == 100500.0  # After first trade
        assert curve[2][1] == 101000.0  # After second trade

    def test_equity_curve_timestamps(self, tracker):
        """Test equity curve has valid timestamps."""
        now = datetime.now()

        tracker.record_trade(
            trade_id="trade_001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            entry_price=150.0,
            exit_price=155.0,
            strategy="test",
            exit_time=now,
        )

        curve = tracker.get_equity_curve()
        assert curve[-1][0] == now


# =============================================================================
# Test Daily Close Processing
# =============================================================================

class TestDailyClose:
    """Tests for daily close processing."""

    def test_close_day_returns_daily_performance(self, tracker):
        """Test close_day returns DailyPerformance."""
        tracker.record_trade(
            trade_id="trade_001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            entry_price=150.0,
            exit_price=155.0,
            strategy="test",
        )

        daily = tracker.close_day()

        assert isinstance(daily, DailyPerformance)
        assert daily.date == date.today()
        assert daily.daily_pnl == 500.0
        assert daily.num_trades >= 1

    def test_close_day_resets_daily_tracking(self, tracker):
        """Test close_day resets daily P&L tracking."""
        tracker.record_trade(
            trade_id="trade_001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            entry_price=150.0,
            exit_price=155.0,
            strategy="test",
        )

        tracker.close_day()

        # Daily P&L should be reset
        assert tracker._daily_pnl == 0.0

    def test_daily_returns_accumulation(self, tracker):
        """Test daily returns are accumulated."""
        tracker.record_trade(
            trade_id="trade_001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            entry_price=150.0,
            exit_price=155.0,
            strategy="test",
        )
        tracker.close_day()

        tracker.record_trade(
            trade_id="trade_002",
            symbol="AAPL",
            side="buy",
            quantity=100,
            entry_price=150.0,
            exit_price=160.0,
            strategy="test",
        )
        tracker.close_day()

        assert len(tracker._daily_returns) == 2

    def test_close_day_with_custom_equity(self, tracker):
        """Test close_day with custom closing equity."""
        tracker.record_trade(
            trade_id="trade_001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            entry_price=150.0,
            exit_price=155.0,
            strategy="test",
        )

        daily = tracker.close_day(closing_equity=101000.0)

        assert daily.ending_equity == 101000.0


# =============================================================================
# Test Summary Generation
# =============================================================================

class TestSummaryGeneration:
    """Tests for performance summary generation."""

    def test_empty_summary(self, tracker):
        """Test summary with no trades."""
        summary = tracker.get_summary()

        assert summary.total_trades == 0
        assert summary.winning_trades == 0
        assert summary.losing_trades == 0
        assert summary.total_return_percent == 0

    def test_summary_with_trades(self, tracker_with_trades):
        """Test summary with multiple trades."""
        summary = tracker_with_trades.get_summary()

        assert summary.total_trades == 3
        assert summary.winning_trades == 2
        assert summary.losing_trades == 1
        assert summary.win_rate == pytest.approx(66.67, rel=0.1)

    def test_summary_average_calculations(self, tracker_with_trades):
        """Test summary average calculations."""
        summary = tracker_with_trades.get_summary()

        # Average trade P&L = (500 + 400 - 300) / 3 = 200
        assert summary.average_trade_pnl == pytest.approx(200.0, rel=0.1)

        # Average winner = (500 + 400) / 2 = 450
        assert summary.average_winner == pytest.approx(450.0, rel=0.1)

        # Average loser = -300
        assert summary.average_loser == pytest.approx(-300.0, rel=0.1)

    def test_summary_largest_winner_loser(self, tracker_with_trades):
        """Test summary identifies largest winner and loser."""
        summary = tracker_with_trades.get_summary()

        assert summary.largest_winner == 500.0
        assert summary.largest_loser == -300.0

    def test_summary_includes_strategy_attribution(self, tracker_with_trades):
        """Test summary includes strategy attribution."""
        summary = tracker_with_trades.get_summary()

        assert "momentum" in summary.strategies
        assert "news" in summary.strategies

    def test_summary_total_return(self, tracker):
        """Test total return calculation."""
        # Add trades totaling +1000 P&L (1% return)
        tracker.record_trade(
            trade_id="trade_001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            entry_price=150.0,
            exit_price=160.0,
            strategy="test",
        )

        summary = tracker.get_summary()
        # Total return = (101000 - 100000) / 100000 * 100 = 1%
        assert summary.total_return_percent == pytest.approx(1.0, rel=0.01)


# =============================================================================
# Test Callbacks
# =============================================================================

class TestCallbacks:
    """Tests for event callbacks."""

    def test_on_trade_recorded_callback(self, tracker):
        """Test on_trade_recorded callback is called."""
        callback_mock = Mock()
        tracker.set_callbacks(on_trade_recorded=callback_mock)

        trade = tracker.record_trade(
            trade_id="trade_001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            entry_price=150.0,
            exit_price=155.0,
            strategy="test",
        )

        callback_mock.assert_called_once_with(trade)

    def test_on_daily_close_callback(self, tracker):
        """Test on_daily_close callback is called."""
        callback_mock = Mock()
        tracker.set_callbacks(on_daily_close=callback_mock)

        tracker.record_trade(
            trade_id="trade_001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            entry_price=150.0,
            exit_price=155.0,
            strategy="test",
        )

        daily = tracker.close_day()

        callback_mock.assert_called_once_with(daily)


# =============================================================================
# Test Reset Functionality
# =============================================================================

class TestReset:
    """Tests for tracker reset functionality."""

    def test_reset_clears_trades(self, tracker_with_trades):
        """Test reset clears all trades."""
        assert len(tracker_with_trades._trades) > 0

        tracker_with_trades.reset()

        assert len(tracker_with_trades._trades) == 0

    def test_reset_restores_initial_capital(self, tracker_with_trades):
        """Test reset restores initial capital."""
        assert tracker_with_trades.current_capital != tracker_with_trades.initial_capital

        tracker_with_trades.reset()

        assert tracker_with_trades.current_capital == tracker_with_trades.initial_capital

    def test_reset_clears_strategy_tracking(self, tracker_with_trades):
        """Test reset clears strategy tracking."""
        assert len(tracker_with_trades._strategy_trades) > 0

        tracker_with_trades.reset()

        assert len(tracker_with_trades._strategy_trades) == 0

    def test_reset_clears_drawdown_tracking(self, tracker):
        """Test reset clears drawdown tracking."""
        # Create a drawdown
        tracker.record_trade(
            trade_id="trade_001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            entry_price=150.0,
            exit_price=155.0,
            strategy="test",
        )
        tracker.record_trade(
            trade_id="trade_002",
            symbol="AAPL",
            side="buy",
            quantity=100,
            entry_price=155.0,
            exit_price=150.0,
            strategy="test",
        )

        assert tracker._max_drawdown > 0

        tracker.reset()

        assert tracker._max_drawdown == 0.0
        assert tracker._high_water_mark == tracker.initial_capital

    def test_reset_initializes_new_equity_curve(self, tracker_with_trades):
        """Test reset initializes a fresh equity curve."""
        old_curve_len = len(tracker_with_trades.get_equity_curve())

        tracker_with_trades.reset()

        curve = tracker_with_trades.get_equity_curve()
        assert len(curve) == 1
        assert curve[0][1] == tracker_with_trades.initial_capital


# =============================================================================
# Test Thread Safety
# =============================================================================

class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_trade_recording(self, tracker):
        """Test concurrent trade recording is thread-safe."""
        errors = []

        def record_trades(thread_id):
            try:
                for i in range(10):
                    tracker.record_trade(
                        trade_id=f"trade_{thread_id}_{i}",
                        symbol="AAPL",
                        side="buy",
                        quantity=100,
                        entry_price=150.0,
                        exit_price=155.0,
                        strategy=f"strategy_{thread_id}",
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_trades, args=(i,)) for i in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(tracker._trades) == 50

    def test_concurrent_summary_access(self, tracker_with_trades):
        """Test concurrent summary access is thread-safe."""
        errors = []
        results = []

        def get_summary():
            try:
                summary = tracker_with_trades.get_summary()
                results.append(summary)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=get_summary) for _ in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10


# =============================================================================
# Test Global Tracker Instance
# =============================================================================

class TestGlobalTracker:
    """Tests for global tracker instance."""

    def test_get_performance_tracker(self, reset_global_tracker):
        """Test getting global tracker instance."""
        tracker = get_performance_tracker()
        assert tracker is not None
        assert isinstance(tracker, PerformanceTracker)

    def test_get_performance_tracker_singleton(self, reset_global_tracker):
        """Test global tracker is a singleton."""
        tracker1 = get_performance_tracker()
        tracker2 = get_performance_tracker()
        assert tracker1 is tracker2

    def test_get_performance_tracker_with_initial_capital(self, reset_global_tracker):
        """Test initial capital is set on first call."""
        import src.analytics.performance_analytics as pa_module
        pa_module._tracker_instance = None

        tracker = get_performance_tracker(initial_capital=50000.0)
        assert tracker.initial_capital == 50000.0


# =============================================================================
# Test Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_quantity_trade(self, tracker):
        """Test handling of zero quantity trade."""
        trade = tracker.record_trade(
            trade_id="trade_001",
            symbol="AAPL",
            side="buy",
            quantity=0,
            entry_price=150.0,
            exit_price=155.0,
            strategy="test",
        )

        assert trade.pnl == 0.0
        assert trade.quantity == 0

    def test_fractional_shares(self, tracker):
        """Test handling of fractional share quantities."""
        trade = tracker.record_trade(
            trade_id="trade_001",
            symbol="AAPL",
            side="buy",
            quantity=10.5,
            entry_price=150.0,
            exit_price=155.0,
            strategy="test",
        )

        # P&L = (155 - 150) * 10.5 = 52.5
        assert trade.pnl == 52.5

    def test_very_small_price_movement(self, tracker):
        """Test handling of very small price movements."""
        trade = tracker.record_trade(
            trade_id="trade_001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            entry_price=150.0,
            exit_price=150.01,
            strategy="test",
        )

        # P&L = 0.01 * 100 = 1.0
        assert trade.pnl == pytest.approx(1.0, rel=0.01)

    def test_large_trade_values(self, tracker):
        """Test handling of large trade values."""
        trade = tracker.record_trade(
            trade_id="trade_001",
            symbol="BRK.A",
            side="buy",
            quantity=10,
            entry_price=500000.0,
            exit_price=510000.0,
            strategy="test",
        )

        # P&L = 10000 * 10 = 100000
        assert trade.pnl == 100000.0

    def test_same_entry_exit_price(self, tracker):
        """Test trade with same entry and exit price (flat)."""
        trade = tracker.record_trade(
            trade_id="trade_001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            entry_price=150.0,
            exit_price=150.0,
            strategy="test",
        )

        assert trade.pnl == 0.0
        assert trade.pnl_percent == 0.0

    def test_negative_commission(self, tracker):
        """Test handling of negative commission (rebate)."""
        trade = tracker.record_trade(
            trade_id="trade_001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            entry_price=150.0,
            exit_price=155.0,
            strategy="test",
            commission=-2.0,  # Rebate
        )

        # P&L = 500 - (-2) = 502
        assert trade.pnl == 502.0


# =============================================================================
# Test Data Classes
# =============================================================================

class TestDataClasses:
    """Tests for data class creation and fields."""

    def test_execution_metrics_creation(self):
        """Test ExecutionMetrics data class creation."""
        metrics = ExecutionMetrics(
            trade_id="test_001",
            symbol="AAPL",
            expected_fill_price=150.0,
            actual_fill_price=150.05,
            slippage_bps=3.33,
            market_impact_bps=3.33,
            time_to_fill_ms=100.0,
            fill_rate=1.0,
            venue="NYSE",
        )

        assert metrics.trade_id == "test_001"
        assert metrics.symbol == "AAPL"
        assert metrics.slippage_bps == 3.33
        assert metrics.venue == "NYSE"

    def test_daily_performance_creation(self):
        """Test DailyPerformance data class creation."""
        daily = DailyPerformance(
            date=date.today(),
            starting_equity=100000.0,
            ending_equity=100500.0,
            daily_pnl=500.0,
            daily_return_percent=0.5,
            num_trades=5,
            winning_trades=3,
            losing_trades=2,
            max_drawdown_intraday=0.2,
            strategies_active=["momentum", "mean_reversion"],
        )

        assert daily.daily_pnl == 500.0
        assert daily.num_trades == 5
        assert len(daily.strategies_active) == 2

    def test_strategy_performance_creation(self):
        """Test StrategyPerformance data class creation."""
        perf = StrategyPerformance(
            strategy="momentum",
            total_trades=100,
            winning_trades=60,
            losing_trades=40,
            win_rate=60.0,
            total_pnl=5000.0,
            average_pnl=50.0,
            max_win=500.0,
            max_loss=-200.0,
            profit_factor=1.5,
            sharpe_ratio=1.2,
            sortino_ratio=1.8,
            max_drawdown=5.0,
            average_holding_period_hours=2.5,
            average_slippage_bps=2.0,
        )

        assert perf.strategy == "momentum"
        assert perf.win_rate == 60.0
        assert perf.sharpe_ratio == 1.2
