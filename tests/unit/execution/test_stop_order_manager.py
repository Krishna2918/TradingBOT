"""Unit tests for Stop Order Manager.

Tests cover:
- Stop-loss placement and triggers
- Take-profit placement and triggers
- Trailing stop placement, updates, and triggers
- Multiple stops for same symbol
- Stop order cancellation
- Broker stop verification
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.execution.execution_engine import (
    StopOrder,
    StopOrderType,
    StopOrderManager,
    get_stop_order_manager,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_broker():
    """Create a mock broker for testing."""
    broker = Mock()
    broker.place_order = Mock(return_value={"order_id": "broker_123", "status": "accepted"})
    broker.place_trailing_stop = Mock(return_value={"order_id": "broker_trail_123", "status": "accepted"})
    broker.cancel_order = Mock(return_value={"status": "cancelled"})
    broker.get_order_status = Mock(return_value={"status": "active"})
    return broker


@pytest.fixture
def stop_manager():
    """Create a stop order manager without broker (demo mode)."""
    return StopOrderManager(broker=None)


@pytest.fixture
def stop_manager_with_broker(mock_broker):
    """Create a stop order manager with a mock broker."""
    return StopOrderManager(broker=mock_broker)


@pytest.fixture
def reset_global_stop_manager():
    """Reset global stop order manager singleton."""
    import src.execution.execution_engine as ee
    original = ee._stop_order_manager_instance
    ee._stop_order_manager_instance = None
    yield
    ee._stop_order_manager_instance = original


# =============================================================================
# Test: StopOrder Dataclass
# =============================================================================

class TestStopOrder:
    """Tests for StopOrder dataclass."""

    def test_create_stop_loss_order(self):
        """Test creating a stop-loss order."""
        stop = StopOrder(
            order_id="stop_001",
            parent_order_id="order_001",
            symbol="AAPL",
            side="sell",
            quantity=100,
            stop_type=StopOrderType.STOP_LOSS,
            trigger_price=145.0,
        )

        assert stop.order_id == "stop_001"
        assert stop.parent_order_id == "order_001"
        assert stop.symbol == "AAPL"
        assert stop.side == "sell"
        assert stop.quantity == 100
        assert stop.stop_type == StopOrderType.STOP_LOSS
        assert stop.trigger_price == 145.0
        assert stop.status == "pending"
        assert stop.broker_order_id is None

    def test_create_take_profit_order(self):
        """Test creating a take-profit order."""
        stop = StopOrder(
            order_id="tp_001",
            parent_order_id="order_001",
            symbol="AAPL",
            side="sell",
            quantity=100,
            stop_type=StopOrderType.TAKE_PROFIT,
            trigger_price=160.0,
        )

        assert stop.stop_type == StopOrderType.TAKE_PROFIT
        assert stop.trigger_price == 160.0

    def test_create_trailing_stop_order(self):
        """Test creating a trailing stop order."""
        stop = StopOrder(
            order_id="trail_001",
            parent_order_id="order_001",
            symbol="AAPL",
            side="sell",
            quantity=100,
            stop_type=StopOrderType.TRAILING_STOP,
            trigger_price=148.0,
            trailing_amount=2.0,
            trailing_percent=False,
        )

        assert stop.stop_type == StopOrderType.TRAILING_STOP
        assert stop.trailing_amount == 2.0
        assert stop.trailing_percent is False

    def test_create_stop_limit_order(self):
        """Test creating a stop-limit order."""
        stop = StopOrder(
            order_id="stop_002",
            parent_order_id="order_002",
            symbol="TSLA",
            side="sell",
            quantity=50,
            stop_type=StopOrderType.STOP_LOSS,
            trigger_price=200.0,
            limit_price=199.0,
        )

        assert stop.trigger_price == 200.0
        assert stop.limit_price == 199.0


# =============================================================================
# Test: Stop-Loss Placement
# =============================================================================

class TestStopLossPlacement:
    """Tests for stop-loss order placement."""

    def test_place_stop_loss_long_position(self, stop_manager):
        """Test placing stop-loss for a long position."""
        stop = stop_manager.place_stop_loss(
            parent_order_id="order_001",
            symbol="AAPL",
            quantity=100,
            stop_price=145.0,
            position_side="long",
        )

        assert stop is not None
        assert stop.symbol == "AAPL"
        assert stop.side == "sell"  # Sell to close long
        assert stop.quantity == 100
        assert stop.trigger_price == 145.0
        assert stop.stop_type == StopOrderType.STOP_LOSS
        assert stop.status == "active"

    def test_place_stop_loss_short_position(self, stop_manager):
        """Test placing stop-loss for a short position."""
        stop = stop_manager.place_stop_loss(
            parent_order_id="order_002",
            symbol="TSLA",
            quantity=50,
            stop_price=210.0,
            position_side="short",
        )

        assert stop is not None
        assert stop.side == "buy"  # Buy to cover short
        assert stop.trigger_price == 210.0

    def test_place_stop_loss_with_limit(self, stop_manager):
        """Test placing stop-limit order."""
        stop = stop_manager.place_stop_loss(
            parent_order_id="order_003",
            symbol="NVDA",
            quantity=30,
            stop_price=400.0,
            position_side="long",
            limit_price=398.0,
        )

        assert stop is not None
        assert stop.trigger_price == 400.0
        assert stop.limit_price == 398.0

    def test_place_stop_loss_with_broker(self, stop_manager_with_broker, mock_broker):
        """Test placing stop-loss with broker."""
        stop = stop_manager_with_broker.place_stop_loss(
            parent_order_id="order_004",
            symbol="MSFT",
            quantity=75,
            stop_price=350.0,
            position_side="long",
        )

        assert stop is not None
        assert stop.broker_order_id == "broker_123"
        assert stop.status == "active"
        mock_broker.place_order.assert_called_once()

    def test_place_stop_loss_broker_rejection(self, stop_manager_with_broker, mock_broker):
        """Test stop-loss rejection by broker."""
        mock_broker.place_order.return_value = {"error": "Insufficient margin"}

        stop = stop_manager_with_broker.place_stop_loss(
            parent_order_id="order_005",
            symbol="META",
            quantity=100,
            stop_price=300.0,
            position_side="long",
        )

        assert stop is None

    def test_place_stop_loss_broker_exception(self, stop_manager_with_broker, mock_broker):
        """Test stop-loss when broker throws exception."""
        mock_broker.place_order.side_effect = Exception("Connection error")

        stop = stop_manager_with_broker.place_stop_loss(
            parent_order_id="order_006",
            symbol="AMZN",
            quantity=50,
            stop_price=140.0,
            position_side="long",
        )

        assert stop is None


# =============================================================================
# Test: Take-Profit Placement
# =============================================================================

class TestTakeProfitPlacement:
    """Tests for take-profit order placement."""

    def test_place_take_profit_long_position(self, stop_manager):
        """Test placing take-profit for a long position."""
        tp = stop_manager.place_take_profit(
            parent_order_id="order_001",
            symbol="AAPL",
            quantity=100,
            target_price=160.0,
            position_side="long",
        )

        assert tp is not None
        assert tp.symbol == "AAPL"
        assert tp.side == "sell"  # Sell to take profit on long
        assert tp.quantity == 100
        assert tp.trigger_price == 160.0
        assert tp.stop_type == StopOrderType.TAKE_PROFIT
        assert tp.status == "active"

    def test_place_take_profit_short_position(self, stop_manager):
        """Test placing take-profit for a short position."""
        tp = stop_manager.place_take_profit(
            parent_order_id="order_002",
            symbol="TSLA",
            quantity=50,
            target_price=180.0,
            position_side="short",
        )

        assert tp is not None
        assert tp.side == "buy"  # Buy to take profit on short
        assert tp.trigger_price == 180.0

    def test_place_take_profit_with_broker(self, stop_manager_with_broker, mock_broker):
        """Test placing take-profit with broker."""
        tp = stop_manager_with_broker.place_take_profit(
            parent_order_id="order_003",
            symbol="GOOGL",
            quantity=20,
            target_price=150.0,
            position_side="long",
        )

        assert tp is not None
        assert tp.broker_order_id == "broker_123"
        mock_broker.place_order.assert_called_once()


# =============================================================================
# Test: Trailing Stop Placement
# =============================================================================

class TestTrailingStopPlacement:
    """Tests for trailing stop order placement."""

    def test_place_trailing_stop_long_dollar(self, stop_manager):
        """Test placing trailing stop with dollar amount for long position."""
        current_price = 150.0
        trailing_amount = 5.0

        trail = stop_manager.place_trailing_stop(
            parent_order_id="order_001",
            symbol="AAPL",
            quantity=100,
            trailing_amount=trailing_amount,
            current_price=current_price,
            position_side="long",
            trailing_percent=False,
        )

        assert trail is not None
        assert trail.symbol == "AAPL"
        assert trail.side == "sell"
        assert trail.stop_type == StopOrderType.TRAILING_STOP
        assert trail.trigger_price == current_price - trailing_amount  # 145.0
        assert trail.trailing_amount == trailing_amount
        assert trail.trailing_percent is False

    def test_place_trailing_stop_long_percent(self, stop_manager):
        """Test placing trailing stop with percentage for long position."""
        current_price = 200.0
        trailing_percent = 3.0  # 3%

        trail = stop_manager.place_trailing_stop(
            parent_order_id="order_002",
            symbol="TSLA",
            quantity=50,
            trailing_amount=trailing_percent,
            current_price=current_price,
            position_side="long",
            trailing_percent=True,
        )

        assert trail is not None
        expected_trigger = current_price - (current_price * 0.03)  # 194.0
        assert trail.trigger_price == expected_trigger
        assert trail.trailing_percent is True

    def test_place_trailing_stop_short(self, stop_manager):
        """Test placing trailing stop for short position."""
        current_price = 100.0
        trailing_amount = 2.0

        trail = stop_manager.place_trailing_stop(
            parent_order_id="order_003",
            symbol="AMD",
            quantity=100,
            trailing_amount=trailing_amount,
            current_price=current_price,
            position_side="short",
            trailing_percent=False,
        )

        assert trail is not None
        assert trail.side == "buy"  # Buy to cover short
        assert trail.trigger_price == current_price + trailing_amount  # 102.0

    def test_place_trailing_stop_with_broker(self, stop_manager_with_broker, mock_broker):
        """Test placing trailing stop with broker."""
        trail = stop_manager_with_broker.place_trailing_stop(
            parent_order_id="order_004",
            symbol="NVDA",
            quantity=30,
            trailing_amount=5.0,
            current_price=500.0,
            position_side="long",
            trailing_percent=False,
        )

        assert trail is not None
        assert trail.broker_order_id == "broker_trail_123"
        mock_broker.place_trailing_stop.assert_called_once()


# =============================================================================
# Test: Trailing Stop Updates
# =============================================================================

class TestTrailingStopUpdates:
    """Tests for trailing stop update logic."""

    def test_update_trailing_stop_long_price_up(self, stop_manager):
        """Test trailing stop moves up when price increases for long position."""
        # Place initial trailing stop
        trail = stop_manager.place_trailing_stop(
            parent_order_id="order_001",
            symbol="AAPL",
            quantity=100,
            trailing_amount=5.0,
            current_price=150.0,  # Initial trigger at 145.0
            position_side="long",
        )

        initial_trigger = trail.trigger_price
        assert initial_trigger == 145.0

        # Price goes up to 160
        updated = stop_manager.update_trailing_stop(
            order_id=trail.order_id,
            current_price=160.0,
            position_side="long",
        )

        assert updated is True
        assert trail.trigger_price == 155.0  # 160 - 5
        assert trail.trigger_price > initial_trigger

    def test_update_trailing_stop_long_price_down(self, stop_manager):
        """Test trailing stop does NOT move down when price decreases for long."""
        trail = stop_manager.place_trailing_stop(
            parent_order_id="order_002",
            symbol="TSLA",
            quantity=50,
            trailing_amount=10.0,
            current_price=200.0,  # Initial trigger at 190.0
            position_side="long",
        )

        initial_trigger = trail.trigger_price

        # Price goes down to 195
        updated = stop_manager.update_trailing_stop(
            order_id=trail.order_id,
            current_price=195.0,
            position_side="long",
        )

        assert updated is False
        assert trail.trigger_price == initial_trigger  # Unchanged at 190.0

    def test_update_trailing_stop_short_price_down(self, stop_manager):
        """Test trailing stop moves down when price decreases for short position."""
        trail = stop_manager.place_trailing_stop(
            parent_order_id="order_003",
            symbol="AMD",
            quantity=100,
            trailing_amount=3.0,
            current_price=100.0,  # Initial trigger at 103.0
            position_side="short",
        )

        initial_trigger = trail.trigger_price
        assert initial_trigger == 103.0

        # Price goes down to 95
        updated = stop_manager.update_trailing_stop(
            order_id=trail.order_id,
            current_price=95.0,
            position_side="short",
        )

        assert updated is True
        assert trail.trigger_price == 98.0  # 95 + 3
        assert trail.trigger_price < initial_trigger

    def test_update_trailing_stop_short_price_up(self, stop_manager):
        """Test trailing stop does NOT move up when price increases for short."""
        trail = stop_manager.place_trailing_stop(
            parent_order_id="order_004",
            symbol="INTC",
            quantity=200,
            trailing_amount=2.0,
            current_price=50.0,  # Initial trigger at 52.0
            position_side="short",
        )

        initial_trigger = trail.trigger_price

        # Price goes up to 51
        updated = stop_manager.update_trailing_stop(
            order_id=trail.order_id,
            current_price=51.0,
            position_side="short",
        )

        assert updated is False
        assert trail.trigger_price == initial_trigger  # Unchanged at 52.0

    def test_update_trailing_stop_percent_mode(self, stop_manager):
        """Test trailing stop updates correctly in percentage mode."""
        trail = stop_manager.place_trailing_stop(
            parent_order_id="order_005",
            symbol="MSFT",
            quantity=100,
            trailing_amount=5.0,  # 5%
            current_price=400.0,  # Trigger at 380.0 (400 - 20)
            position_side="long",
            trailing_percent=True,
        )

        assert trail.trigger_price == 380.0

        # Price goes up to 450
        updated = stop_manager.update_trailing_stop(
            order_id=trail.order_id,
            current_price=450.0,
            position_side="long",
        )

        assert updated is True
        # New trigger: 450 - (450 * 0.05) = 450 - 22.5 = 427.5
        assert trail.trigger_price == 427.5

    def test_update_nonexistent_order(self, stop_manager):
        """Test updating a non-existent order returns False."""
        updated = stop_manager.update_trailing_stop(
            order_id="fake_order",
            current_price=100.0,
            position_side="long",
        )

        assert updated is False

    def test_update_non_trailing_stop(self, stop_manager):
        """Test updating a non-trailing stop returns False."""
        # Place regular stop-loss
        stop = stop_manager.place_stop_loss(
            parent_order_id="order_006",
            symbol="GOOG",
            quantity=50,
            stop_price=120.0,
            position_side="long",
        )

        # Try to update as trailing
        updated = stop_manager.update_trailing_stop(
            order_id=stop.order_id,
            current_price=130.0,
            position_side="long",
        )

        assert updated is False

    def test_update_cancelled_trailing_stop(self, stop_manager):
        """Test updating a cancelled trailing stop returns False."""
        trail = stop_manager.place_trailing_stop(
            parent_order_id="order_007",
            symbol="META",
            quantity=100,
            trailing_amount=5.0,
            current_price=300.0,
            position_side="long",
        )

        # Cancel the stop
        stop_manager.cancel_stop(trail.order_id)

        # Try to update
        updated = stop_manager.update_trailing_stop(
            order_id=trail.order_id,
            current_price=310.0,
            position_side="long",
        )

        assert updated is False


# =============================================================================
# Test: Stop Trigger Detection
# =============================================================================

class TestStopTriggers:
    """Tests for stop trigger detection."""

    def test_stop_loss_trigger_at_exact_price_long(self, stop_manager):
        """Test stop-loss triggers at exact price for long position."""
        stop = stop_manager.place_stop_loss(
            parent_order_id="order_001",
            symbol="AAPL",
            quantity=100,
            stop_price=145.0,
            position_side="long",
        )

        triggered = stop_manager.check_stop_triggers({"AAPL": 145.0})

        assert len(triggered) == 1
        assert triggered[0].order_id == stop.order_id
        assert triggered[0].status == "triggered"
        assert triggered[0].triggered_at is not None

    def test_stop_loss_trigger_below_price_long(self, stop_manager):
        """Test stop-loss triggers below stop price for long position."""
        stop = stop_manager.place_stop_loss(
            parent_order_id="order_002",
            symbol="TSLA",
            quantity=50,
            stop_price=200.0,
            position_side="long",
        )

        triggered = stop_manager.check_stop_triggers({"TSLA": 195.0})

        assert len(triggered) == 1
        assert triggered[0].order_id == stop.order_id

    def test_stop_loss_not_triggered_above_price_long(self, stop_manager):
        """Test stop-loss does NOT trigger above stop price for long position."""
        stop_manager.place_stop_loss(
            parent_order_id="order_003",
            symbol="NVDA",
            quantity=30,
            stop_price=400.0,
            position_side="long",
        )

        triggered = stop_manager.check_stop_triggers({"NVDA": 410.0})

        assert len(triggered) == 0

    def test_stop_loss_trigger_short(self, stop_manager):
        """Test stop-loss triggers for short position."""
        stop = stop_manager.place_stop_loss(
            parent_order_id="order_004",
            symbol="AMD",
            quantity=100,
            stop_price=110.0,
            position_side="short",
        )

        # For short, stop triggers when price rises above stop
        triggered = stop_manager.check_stop_triggers({"AMD": 115.0})

        assert len(triggered) == 1
        assert triggered[0].order_id == stop.order_id

    def test_take_profit_trigger_long(self, stop_manager):
        """Test take-profit triggers for long position."""
        tp = stop_manager.place_take_profit(
            parent_order_id="order_005",
            symbol="MSFT",
            quantity=75,
            target_price=400.0,
            position_side="long",
        )

        triggered = stop_manager.check_stop_triggers({"MSFT": 405.0})

        assert len(triggered) == 1
        assert triggered[0].order_id == tp.order_id
        assert triggered[0].stop_type == StopOrderType.TAKE_PROFIT

    def test_take_profit_trigger_short(self, stop_manager):
        """Test take-profit triggers for short position."""
        tp = stop_manager.place_take_profit(
            parent_order_id="order_006",
            symbol="GOOGL",
            quantity=20,
            target_price=120.0,
            position_side="short",
        )

        # For short, take profit triggers when price falls below target
        triggered = stop_manager.check_stop_triggers({"GOOGL": 115.0})

        assert len(triggered) == 1
        assert triggered[0].order_id == tp.order_id

    def test_trailing_stop_trigger(self, stop_manager):
        """Test trailing stop triggers."""
        trail = stop_manager.place_trailing_stop(
            parent_order_id="order_007",
            symbol="AAPL",
            quantity=100,
            trailing_amount=5.0,
            current_price=150.0,  # Trigger at 145.0
            position_side="long",
        )

        triggered = stop_manager.check_stop_triggers({"AAPL": 144.0})

        assert len(triggered) == 1
        assert triggered[0].order_id == trail.order_id
        assert triggered[0].stop_type == StopOrderType.TRAILING_STOP

    def test_multiple_stops_same_symbol(self, stop_manager):
        """Test multiple stops on the same symbol."""
        # Place stop-loss
        stop = stop_manager.place_stop_loss(
            parent_order_id="order_008",
            symbol="AAPL",
            quantity=100,
            stop_price=145.0,
            position_side="long",
        )

        # Place take-profit
        tp = stop_manager.place_take_profit(
            parent_order_id="order_008",
            symbol="AAPL",
            quantity=100,
            target_price=160.0,
            position_side="long",
        )

        # Price rises to 165 - only take-profit triggers
        triggered = stop_manager.check_stop_triggers({"AAPL": 165.0})

        assert len(triggered) == 1
        assert triggered[0].order_id == tp.order_id

        # Check stop-loss is still active
        assert stop.status == "active"

    def test_multiple_stops_different_symbols(self, stop_manager):
        """Test multiple stops on different symbols."""
        stop_manager.place_stop_loss(
            parent_order_id="order_009",
            symbol="AAPL",
            quantity=100,
            stop_price=145.0,
            position_side="long",
        )

        stop_manager.place_stop_loss(
            parent_order_id="order_010",
            symbol="TSLA",
            quantity=50,
            stop_price=200.0,
            position_side="long",
        )

        # Both prices drop
        triggered = stop_manager.check_stop_triggers({
            "AAPL": 140.0,
            "TSLA": 195.0,
        })

        assert len(triggered) == 2

    def test_no_trigger_for_missing_symbol(self, stop_manager):
        """Test no trigger when symbol not in market prices."""
        stop_manager.place_stop_loss(
            parent_order_id="order_011",
            symbol="AAPL",
            quantity=100,
            stop_price=145.0,
            position_side="long",
        )

        # Only TSLA price provided
        triggered = stop_manager.check_stop_triggers({"TSLA": 100.0})

        assert len(triggered) == 0

    def test_already_triggered_stop_not_retriggered(self, stop_manager):
        """Test that already triggered stops are not re-triggered."""
        stop_manager.place_stop_loss(
            parent_order_id="order_012",
            symbol="AAPL",
            quantity=100,
            stop_price=145.0,
            position_side="long",
        )

        # First trigger
        triggered1 = stop_manager.check_stop_triggers({"AAPL": 140.0})
        assert len(triggered1) == 1

        # Second check - should not trigger again
        triggered2 = stop_manager.check_stop_triggers({"AAPL": 135.0})
        assert len(triggered2) == 0


# =============================================================================
# Test: Stop Order Cancellation
# =============================================================================

class TestStopCancellation:
    """Tests for stop order cancellation."""

    def test_cancel_stop_order(self, stop_manager):
        """Test cancelling a stop order."""
        stop = stop_manager.place_stop_loss(
            parent_order_id="order_001",
            symbol="AAPL",
            quantity=100,
            stop_price=145.0,
            position_side="long",
        )

        cancelled = stop_manager.cancel_stop(stop.order_id)

        assert cancelled is True
        assert stop.status == "cancelled"

    def test_cancel_nonexistent_stop(self, stop_manager):
        """Test cancelling a non-existent stop."""
        cancelled = stop_manager.cancel_stop("fake_order")

        assert cancelled is False

    def test_cancel_already_triggered_stop(self, stop_manager):
        """Test cannot cancel already triggered stop."""
        stop = stop_manager.place_stop_loss(
            parent_order_id="order_002",
            symbol="TSLA",
            quantity=50,
            stop_price=200.0,
            position_side="long",
        )

        # Trigger the stop
        stop_manager.check_stop_triggers({"TSLA": 195.0})

        # Try to cancel
        cancelled = stop_manager.cancel_stop(stop.order_id)

        assert cancelled is False
        assert stop.status == "triggered"

    def test_cancel_stop_with_broker(self, stop_manager_with_broker, mock_broker):
        """Test cancelling a stop calls broker."""
        stop = stop_manager_with_broker.place_stop_loss(
            parent_order_id="order_003",
            symbol="NVDA",
            quantity=30,
            stop_price=400.0,
            position_side="long",
        )

        cancelled = stop_manager_with_broker.cancel_stop(stop.order_id)

        assert cancelled is True
        mock_broker.cancel_order.assert_called_once_with("broker_123")

    def test_cancel_all_for_position(self, stop_manager):
        """Test cancelling all stops for a position."""
        parent_id = "order_004"

        # Place multiple stops for same position
        stop_manager.place_stop_loss(
            parent_order_id=parent_id,
            symbol="AAPL",
            quantity=100,
            stop_price=145.0,
            position_side="long",
        )

        stop_manager.place_take_profit(
            parent_order_id=parent_id,
            symbol="AAPL",
            quantity=100,
            target_price=160.0,
            position_side="long",
        )

        stop_manager.place_trailing_stop(
            parent_order_id=parent_id,
            symbol="AAPL",
            quantity=100,
            trailing_amount=5.0,
            current_price=150.0,
            position_side="long",
        )

        # Cancel all
        cancelled_count = stop_manager.cancel_all_for_position(parent_id)

        assert cancelled_count == 3

    def test_cancel_all_for_position_mixed_states(self, stop_manager):
        """Test cancel_all_for_position only cancels active stops."""
        parent_id = "order_005"

        # Active stop
        stop1 = stop_manager.place_stop_loss(
            parent_order_id=parent_id,
            symbol="AAPL",
            quantity=100,
            stop_price=145.0,
            position_side="long",
        )

        # Another stop that gets triggered
        stop2 = stop_manager.place_take_profit(
            parent_order_id=parent_id,
            symbol="AAPL",
            quantity=100,
            target_price=160.0,
            position_side="long",
        )

        # Trigger the take-profit
        stop_manager.check_stop_triggers({"AAPL": 165.0})

        # Cancel all - should only cancel the active stop-loss
        cancelled_count = stop_manager.cancel_all_for_position(parent_id)

        assert cancelled_count == 1
        assert stop1.status == "cancelled"
        assert stop2.status == "triggered"


# =============================================================================
# Test: Get Stops for Position
# =============================================================================

class TestGetStopsForPosition:
    """Tests for getting stops for a position."""

    def test_get_stops_for_position(self, stop_manager):
        """Test getting all active stops for a position."""
        parent_id = "order_001"

        stop_manager.place_stop_loss(
            parent_order_id=parent_id,
            symbol="AAPL",
            quantity=100,
            stop_price=145.0,
            position_side="long",
        )

        stop_manager.place_take_profit(
            parent_order_id=parent_id,
            symbol="AAPL",
            quantity=100,
            target_price=160.0,
            position_side="long",
        )

        stops = stop_manager.get_stops_for_position(parent_id)

        assert len(stops) == 2

    def test_get_stops_excludes_cancelled(self, stop_manager):
        """Test get_stops_for_position excludes cancelled stops."""
        parent_id = "order_002"

        stop = stop_manager.place_stop_loss(
            parent_order_id=parent_id,
            symbol="TSLA",
            quantity=50,
            stop_price=200.0,
            position_side="long",
        )

        stop_manager.cancel_stop(stop.order_id)

        stops = stop_manager.get_stops_for_position(parent_id)

        assert len(stops) == 0

    def test_get_stops_for_nonexistent_position(self, stop_manager):
        """Test getting stops for non-existent position returns empty list."""
        stops = stop_manager.get_stops_for_position("fake_order")

        assert len(stops) == 0


# =============================================================================
# Test: Broker Stop Verification
# =============================================================================

class TestBrokerVerification:
    """Tests for broker stop verification."""

    def test_verify_broker_stops_no_broker(self, stop_manager):
        """Test verification returns empty report when no broker."""
        report = stop_manager.verify_broker_stops()

        assert report["total_stops"] == 0
        assert report["verified"] == 0
        assert len(report["missing"]) == 0

    def test_verify_broker_stops_all_active(self, stop_manager_with_broker, mock_broker):
        """Test verification with all stops active at broker."""
        stop_manager_with_broker.place_stop_loss(
            parent_order_id="order_001",
            symbol="AAPL",
            quantity=100,
            stop_price=145.0,
            position_side="long",
        )

        stop_manager_with_broker.place_stop_loss(
            parent_order_id="order_002",
            symbol="TSLA",
            quantity=50,
            stop_price=200.0,
            position_side="long",
        )

        report = stop_manager_with_broker.verify_broker_stops()

        assert report["total_stops"] == 2
        assert report["verified"] == 2
        assert len(report["missing"]) == 0

    def test_verify_broker_stops_cancelled_at_broker(self, stop_manager_with_broker, mock_broker):
        """Test verification detects stops cancelled at broker."""
        stop_manager_with_broker.place_stop_loss(
            parent_order_id="order_003",
            symbol="NVDA",
            quantity=30,
            stop_price=400.0,
            position_side="long",
        )

        # Broker reports cancelled
        mock_broker.get_order_status.return_value = {"status": "cancelled"}

        report = stop_manager_with_broker.verify_broker_stops()

        assert report["total_stops"] == 1
        assert report["verified"] == 0
        assert len(report["missing"]) == 1
        assert report["missing"][0]["reason"] == "cancelled at broker"

    def test_verify_broker_stops_not_found(self, stop_manager_with_broker, mock_broker):
        """Test verification detects stops not found at broker."""
        stop_manager_with_broker.place_stop_loss(
            parent_order_id="order_004",
            symbol="MSFT",
            quantity=75,
            stop_price=350.0,
            position_side="long",
        )

        # Broker returns None (not found)
        mock_broker.get_order_status.return_value = None

        report = stop_manager_with_broker.verify_broker_stops()

        assert report["total_stops"] == 1
        assert report["verified"] == 0
        assert len(report["missing"]) == 1
        assert report["missing"][0]["reason"] == "not found at broker"


# =============================================================================
# Test: Global Stop Order Manager
# =============================================================================

class TestGlobalStopOrderManager:
    """Tests for global stop order manager singleton."""

    def test_get_stop_order_manager_creates_instance(self, reset_global_stop_manager):
        """Test get_stop_order_manager creates a new instance."""
        manager = get_stop_order_manager()

        assert manager is not None
        assert isinstance(manager, StopOrderManager)

    def test_get_stop_order_manager_returns_same_instance(self, reset_global_stop_manager):
        """Test get_stop_order_manager returns the same instance."""
        manager1 = get_stop_order_manager()
        manager2 = get_stop_order_manager()

        assert manager1 is manager2

    def test_get_stop_order_manager_with_broker(self, reset_global_stop_manager, mock_broker):
        """Test get_stop_order_manager with broker argument."""
        manager = get_stop_order_manager(broker=mock_broker)

        assert manager.broker is mock_broker


# =============================================================================
# Test: Thread Safety
# =============================================================================

class TestThreadSafety:
    """Tests for thread safety of stop order manager."""

    def test_concurrent_stop_placement(self, stop_manager):
        """Test concurrent stop order placement."""
        import threading
        import time

        results = []

        def place_stop(index):
            stop = stop_manager.place_stop_loss(
                parent_order_id=f"order_{index}",
                symbol=f"SYM{index}",
                quantity=100,
                stop_price=100.0 + index,
                position_side="long",
            )
            results.append(stop)

        threads = [threading.Thread(target=place_stop, args=(i,)) for i in range(10)]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        assert len(results) == 10
        assert all(r is not None for r in results)

    def test_concurrent_trigger_checks(self, stop_manager):
        """Test concurrent trigger checks."""
        import threading

        # Place stops
        for i in range(5):
            stop_manager.place_stop_loss(
                parent_order_id=f"order_{i}",
                symbol=f"SYM{i}",
                quantity=100,
                stop_price=100.0,
                position_side="long",
            )

        results = []

        def check_triggers():
            triggered = stop_manager.check_stop_triggers({
                f"SYM{i}": 95.0 for i in range(5)
            })
            results.append(len(triggered))

        threads = [threading.Thread(target=check_triggers) for _ in range(3)]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # Only first check should find triggered stops
        total_triggered = sum(results)
        assert total_triggered == 5  # Each stop triggers exactly once
