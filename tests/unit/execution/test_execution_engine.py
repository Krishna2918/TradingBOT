"""
Unit tests for Execution Engine
===============================

Tests for order execution, retry logic, slippage, and commission calculation.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.execution.execution_engine import (
    ExecutionEngine,
    Order,
    OrderType,
    OrderSide,
    OrderStatus,
    RetryConfig,
    with_retry,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def execution_engine():
    """Create execution engine with default settings."""
    return ExecutionEngine(
        commission_rate=0.001,
        min_commission=1.0,
        slippage_bps=5.0,
        allow_fractional=True,
        max_order_size_pct=0.10,
    )


@pytest.fixture
def mock_broker():
    """Create a mock broker for testing."""
    broker = Mock()
    broker.place_order = Mock(return_value={"order_id": "TEST123", "status": "filled"})
    broker.cancel_order = Mock(return_value=True)
    broker.get_order_status = Mock(return_value={"status": "filled"})
    return broker


@pytest.fixture
def market_data():
    """Create sample market data for VWAP testing."""
    dates = pd.date_range(end=datetime.now(), periods=30, freq="1min")
    return pd.DataFrame({
        "open": np.random.uniform(99, 101, 30),
        "high": np.random.uniform(100, 102, 30),
        "low": np.random.uniform(98, 100, 30),
        "close": np.random.uniform(99, 101, 30),
        "volume": np.random.randint(1000, 10000, 30),
    }, index=dates)


# =============================================================================
# Order Tests
# =============================================================================


class TestOrder:
    """Tests for Order class."""

    def test_create_market_order(self):
        """Test creating a market order."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
        )

        assert order.symbol == "AAPL"
        assert order.side == OrderSide.BUY
        assert order.quantity == 100
        assert order.order_type == OrderType.MARKET
        assert order.status == OrderStatus.PENDING
        assert order.filled_quantity == 0.0
        assert order.remaining_quantity == 100

    def test_create_limit_order(self):
        """Test creating a limit order."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=50,
            order_type=OrderType.LIMIT,
            limit_price=150.00,
        )

        assert order.order_type == OrderType.LIMIT
        assert order.limit_price == 150.00

    def test_add_fill(self):
        """Test adding a fill to an order."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )

        order.add_fill(50, 150.00, 1.50)

        assert order.filled_quantity == 50
        assert order.remaining_quantity == 50
        assert order.average_fill_price == 150.00
        assert order.status == OrderStatus.PARTIALLY_FILLED
        assert len(order.fills) == 1

    def test_complete_fill(self):
        """Test completing an order with fills."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )

        order.add_fill(60, 150.00, 1.50)
        order.add_fill(40, 151.00, 1.00)

        assert order.filled_quantity == 100
        assert order.remaining_quantity == 0
        assert order.status == OrderStatus.FILLED
        # Average fill price = (60*150 + 40*151) / 100 = 150.40
        assert abs(order.average_fill_price - 150.40) < 0.01

    def test_cancel_order(self):
        """Test cancelling an order."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )

        result = order.cancel()

        assert result is True
        assert order.status == OrderStatus.CANCELLED

    def test_cannot_cancel_filled_order(self):
        """Test that filled orders cannot be cancelled."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )
        order.add_fill(100, 150.00, 1.50)

        result = order.cancel()

        assert result is False
        assert order.status == OrderStatus.FILLED

    def test_order_to_dict(self):
        """Test order serialization to dictionary."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            limit_price=150.00,
        )

        data = order.to_dict()

        assert data["symbol"] == "AAPL"
        assert data["side"] == "buy"
        assert data["quantity"] == 100
        assert data["limit_price"] == 150.00


# =============================================================================
# ExecutionEngine Tests
# =============================================================================


class TestExecutionEngine:
    """Tests for ExecutionEngine class."""

    def test_create_order(self, execution_engine):
        """Test creating an order through the engine."""
        order = execution_engine.create_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )

        assert order.symbol == "AAPL"
        assert order.order_id in execution_engine.orders

    def test_create_order_rounds_to_whole_shares_when_fractional_disabled(self):
        """Test that fractional shares are rounded when disabled."""
        engine = ExecutionEngine(allow_fractional=False)

        order = engine.create_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10.5,
        )

        assert order.quantity == 10

    def test_create_order_rejects_zero_quantity(self, execution_engine):
        """Test that zero quantity orders are rejected."""
        with pytest.raises(ValueError):
            execution_engine.create_order(
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=0,
            )

    def test_execute_market_order(self, execution_engine):
        """Test executing a market order."""
        order = execution_engine.create_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )

        result = execution_engine.execute_market_order(
            order=order,
            current_price=150.00,
            volume=100000,
        )

        assert result is True
        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == 100
        assert order.average_fill_price > 0

    def test_execute_market_order_with_slippage(self, execution_engine):
        """Test that market orders include slippage."""
        order = execution_engine.create_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )

        execution_engine.execute_market_order(
            order=order,
            current_price=150.00,
            volume=100000,
        )

        # Buy orders should have slippage added (higher price)
        assert order.average_fill_price >= 150.00

    def test_execute_limit_order_condition_met(self, execution_engine):
        """Test executing limit order when price condition is met."""
        order = execution_engine.create_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            limit_price=150.00,
        )

        result = execution_engine.execute_limit_order(
            order=order,
            current_price=149.00,  # Below limit price
            volume=100000,
        )

        assert result is True
        assert order.status == OrderStatus.FILLED

    def test_execute_limit_order_condition_not_met(self, execution_engine):
        """Test limit order waits when price condition not met."""
        order = execution_engine.create_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            limit_price=150.00,
        )

        result = execution_engine.execute_limit_order(
            order=order,
            current_price=151.00,  # Above limit price
            volume=100000,
        )

        assert result is False
        assert order.status == OrderStatus.OPEN

    def test_partial_fill_large_order(self, execution_engine):
        """Test partial fill for orders exceeding volume capacity."""
        order = execution_engine.create_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10000,  # Large order
        )

        execution_engine.execute_market_order(
            order=order,
            current_price=150.00,
            volume=1000,  # Low volume
        )

        # Should be partially filled due to volume constraint
        assert order.filled_quantity < order.quantity
        assert order.status == OrderStatus.PARTIALLY_FILLED

    def test_get_order(self, execution_engine):
        """Test retrieving an order by ID."""
        order = execution_engine.create_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )

        retrieved = execution_engine.get_order(order.order_id)

        assert retrieved is order

    def test_get_open_orders(self, execution_engine):
        """Test retrieving open orders."""
        order1 = execution_engine.create_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )
        order1.status = OrderStatus.OPEN

        order2 = execution_engine.create_order(
            symbol="MSFT",
            side=OrderSide.BUY,
            quantity=50,
        )
        execution_engine.execute_market_order(order2, 300.00, 100000)

        open_orders = execution_engine.get_open_orders()

        assert len(open_orders) == 1
        assert open_orders[0] == order1

    def test_get_open_orders_by_symbol(self, execution_engine):
        """Test filtering open orders by symbol."""
        order1 = execution_engine.create_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )
        order1.status = OrderStatus.OPEN

        order2 = execution_engine.create_order(
            symbol="MSFT",
            side=OrderSide.BUY,
            quantity=50,
        )
        order2.status = OrderStatus.OPEN

        aapl_orders = execution_engine.get_open_orders(symbol="AAPL")

        assert len(aapl_orders) == 1
        assert aapl_orders[0].symbol == "AAPL"

    def test_cancel_order(self, execution_engine):
        """Test cancelling an order through the engine."""
        order = execution_engine.create_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )

        result = execution_engine.cancel_order(order.order_id)

        assert result is True
        assert order.status == OrderStatus.CANCELLED

    def test_cancel_nonexistent_order(self, execution_engine):
        """Test cancelling a non-existent order."""
        result = execution_engine.cancel_order("nonexistent_id")
        assert result is False


# =============================================================================
# Commission Calculation Tests
# =============================================================================


class TestCommissionCalculation:
    """Tests for commission calculation."""

    def test_commission_calculation(self, execution_engine):
        """Test basic commission calculation."""
        commission = execution_engine._calculate_commission(100, 150.00)

        # 100 * 150 * 0.001 = 15.00
        assert commission == 15.00

    def test_minimum_commission(self, execution_engine):
        """Test minimum commission is applied."""
        commission = execution_engine._calculate_commission(1, 5.00)

        # 1 * 5 * 0.001 = 0.005, but min is 1.00
        assert commission == 1.00

    def test_custom_commission_rate(self):
        """Test custom commission rate."""
        engine = ExecutionEngine(commission_rate=0.002, min_commission=2.00)
        commission = engine._calculate_commission(100, 150.00)

        # 100 * 150 * 0.002 = 30.00
        assert commission == 30.00


# =============================================================================
# Slippage Calculation Tests
# =============================================================================


class TestSlippageCalculation:
    """Tests for slippage calculation."""

    def test_fixed_slippage(self):
        """Test fixed slippage model."""
        engine = ExecutionEngine(slippage_model="fixed", slippage_bps=5.0)

        slippage = engine._calculate_slippage(
            quantity=100,
            price=150.00,
            volume=100000,
            side=OrderSide.BUY,
        )

        # 5 bps = 0.0005
        assert abs(slippage - 0.0005) < 0.0001

    def test_proportional_slippage(self):
        """Test proportional slippage model."""
        engine = ExecutionEngine(slippage_model="proportional", slippage_bps=5.0)

        slippage = engine._calculate_slippage(
            quantity=100,
            price=150.00,
            volume=100000,
            side=OrderSide.BUY,
        )

        # Proportional slippage should be > base slippage
        assert slippage >= 0.0005

    def test_slippage_capped(self):
        """Test that slippage is capped at maximum."""
        engine = ExecutionEngine(slippage_model="proportional", slippage_bps=5.0)

        slippage = engine._calculate_slippage(
            quantity=100000,  # Very large order
            price=150.00,
            volume=100,  # Very low volume
            side=OrderSide.BUY,
        )

        # Should be capped at 1%
        assert slippage <= 0.01


# =============================================================================
# VWAP Execution Tests
# =============================================================================


class TestVWAPExecution:
    """Tests for VWAP execution algorithm."""

    def test_vwap_order_execution(self, execution_engine, market_data):
        """Test VWAP order execution."""
        order = execution_engine.create_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )

        result = execution_engine.execute_vwap_order(
            order=order,
            market_data=market_data,
            time_window_minutes=30,
        )

        assert result is True
        # VWAP may result in partial fill due to chunking/rounding
        assert order.status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]
        assert order.filled_quantity > 0
        assert len(order.fills) > 1  # Multiple fills for VWAP

    def test_vwap_already_filled_order(self, execution_engine, market_data):
        """Test VWAP rejects already filled orders."""
        order = execution_engine.create_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )
        order.status = OrderStatus.FILLED

        result = execution_engine.execute_vwap_order(
            order=order,
            market_data=market_data,
        )

        assert result is False


# =============================================================================
# Execution Statistics Tests
# =============================================================================


class TestExecutionStatistics:
    """Tests for execution statistics."""

    def test_execution_statistics(self, execution_engine):
        """Test execution statistics calculation."""
        order = execution_engine.create_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )

        execution_engine.execute_market_order(order, 150.00, 100000)

        stats = execution_engine.get_execution_statistics()

        assert stats["total_executions"] >= 1
        assert stats["total_volume"] >= 100
        assert stats["total_value"] > 0
        assert stats["total_commission"] > 0
        assert "average_slippage" in stats

    def test_empty_execution_statistics(self, execution_engine):
        """Test execution statistics when no executions."""
        stats = execution_engine.get_execution_statistics()
        assert stats == {}


# =============================================================================
# Retry Logic Tests
# =============================================================================


class TestRetryLogic:
    """Tests for retry configuration and decorator."""

    def test_retry_config_delay_calculation(self):
        """Test exponential backoff delay calculation."""
        config = RetryConfig(
            max_retries=3,
            base_delay=1.0,
            max_delay=30.0,
            exponential_base=2.0,
            jitter=False,
        )

        assert config.calculate_delay(0) == 1.0
        assert config.calculate_delay(1) == 2.0
        assert config.calculate_delay(2) == 4.0
        assert config.calculate_delay(3) == 8.0

    def test_retry_config_max_delay_cap(self):
        """Test that delay is capped at max."""
        config = RetryConfig(
            base_delay=1.0,
            max_delay=5.0,
            exponential_base=2.0,
            jitter=False,
        )

        assert config.calculate_delay(10) == 5.0

    def test_retry_config_with_jitter(self):
        """Test that jitter adds randomness."""
        config = RetryConfig(
            base_delay=1.0,
            jitter=True,
        )

        delays = [config.calculate_delay(0) for _ in range(10)]
        # With jitter, delays should vary
        assert len(set(delays)) > 1

    def test_with_retry_decorator_success(self):
        """Test retry decorator with successful function."""
        call_count = [0]

        @with_retry(RetryConfig(max_retries=3))
        def successful_func():
            call_count[0] += 1
            return "success"

        result = successful_func()

        assert result == "success"
        assert call_count[0] == 1

    def test_with_retry_decorator_retry_then_success(self):
        """Test retry decorator with eventual success."""
        call_count = [0]

        @with_retry(RetryConfig(max_retries=3, base_delay=0.01))
        def flaky_func():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ValueError("Temporary error")
            return "success"

        result = flaky_func()

        assert result == "success"
        assert call_count[0] == 3


# =============================================================================
# Broker Execution Tests
# =============================================================================


class TestBrokerExecution:
    """Tests for broker order execution."""

    def test_execute_broker_order_success(self, mock_broker):
        """Test successful broker order execution."""
        engine = ExecutionEngine(broker=mock_broker)

        result = engine.execute_broker_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )

        assert result["submitted"] is True
        assert result["symbol"] == "AAPL"
        assert result["quantity"] == 100
        mock_broker.place_order.assert_called_once()

    def test_execute_broker_order_no_broker(self):
        """Test broker order without configured broker."""
        engine = ExecutionEngine()

        with pytest.raises(RuntimeError, match="No broker configured"):
            engine.execute_broker_order(
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=100,
            )

    def test_execute_broker_order_with_retry(self, mock_broker):
        """Test broker order retries on failure."""
        call_count = [0]

        def flaky_order(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] < 2:
                raise Exception("Temporary error")
            return {"order_id": "TEST123", "status": "filled"}

        mock_broker.place_order = Mock(side_effect=flaky_order)
        engine = ExecutionEngine(
            broker=mock_broker,
            max_retries=3,
            retry_base_delay=0.01,
        )

        result = engine.execute_broker_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )

        assert result["submitted"] is True
        assert call_count[0] == 2

    def test_circuit_breaker_status(self, mock_broker):
        """Test circuit breaker status retrieval."""
        engine = ExecutionEngine(broker=mock_broker)

        status = engine.get_circuit_breaker_status()

        assert "available" in status


# =============================================================================
# Aged Order Cancellation Tests
# =============================================================================


class TestAgedOrderCancellation:
    """Tests for aged order cancellation."""

    def test_cancel_aged_orders(self, mock_broker):
        """Test cancelling aged orders."""
        engine = ExecutionEngine(broker=mock_broker)

        # Simulate an aged order
        engine._pending_orders["old_order"] = datetime.now() - timedelta(minutes=10)

        cancelled = engine.cancel_aged_orders(max_age_seconds=60)

        assert "old_order" in cancelled
        assert "old_order" not in engine._pending_orders

    def test_dont_cancel_recent_orders(self, mock_broker):
        """Test that recent orders are not cancelled."""
        engine = ExecutionEngine(broker=mock_broker)

        engine._pending_orders["recent_order"] = datetime.now()

        cancelled = engine.cancel_aged_orders(max_age_seconds=300)

        assert len(cancelled) == 0
        assert "recent_order" in engine._pending_orders
