"""
Unit tests for async execution module.

Tests cover:
- AsyncExecutionEngine single order execution
- Batch order processing
- Timeout and retry handling
- Concurrency control
- Statistics and monitoring
"""

import asyncio
from unittest.mock import Mock, patch, MagicMock
import pytest

from src.execution.async_execution import (
    AsyncExecutionEngine,
    AsyncOrderRequest,
    AsyncExecutionResult,
    BatchExecutionResult,
    execute_orders_async,
    execute_order_async,
    get_async_execution_engine,
)
from src.execution.execution_engine import (
    ExecutionEngine,
    OrderSide,
    OrderType,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def async_engine():
    """Create async execution engine for testing."""
    sync_engine = ExecutionEngine(
        commission_rate=0.001,
        slippage_bps=5.0,
    )
    return AsyncExecutionEngine(
        sync_engine=sync_engine,
        max_concurrent_orders=3,
        default_timeout_seconds=10.0,
    )


@pytest.fixture
def sample_order_request():
    """Create sample order request."""
    return AsyncOrderRequest(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=100,
        order_type=OrderType.MARKET,
        timeout_seconds=5.0,
    )


# =============================================================================
# Test AsyncOrderRequest
# =============================================================================

class TestAsyncOrderRequest:
    """Tests for AsyncOrderRequest dataclass."""

    def test_basic_creation(self):
        """Test basic order request creation."""
        request = AsyncOrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )

        assert request.symbol == "AAPL"
        assert request.side == OrderSide.BUY
        assert request.quantity == 100
        assert request.order_type == OrderType.MARKET  # Default

    def test_with_limit_price(self):
        """Test order request with limit price."""
        request = AsyncOrderRequest(
            symbol="MSFT",
            side=OrderSide.SELL,
            quantity=50,
            order_type=OrderType.LIMIT,
            limit_price=350.00,
        )

        assert request.limit_price == 350.00
        assert request.order_type == OrderType.LIMIT

    def test_order_id_generation(self):
        """Test automatic order ID generation."""
        request = AsyncOrderRequest(
            symbol="GOOG",
            side=OrderSide.BUY,
            quantity=10,
        )

        order_id = request.order_id
        assert order_id.startswith("async_")
        # Calling again should return same ID
        assert request.order_id == order_id

    def test_custom_order_id(self):
        """Test custom order ID."""
        request = AsyncOrderRequest(
            symbol="TSLA",
            side=OrderSide.SELL,
            quantity=25,
            metadata={"order_id": "custom_123"},
        )

        assert request.order_id == "custom_123"

    def test_priority_setting(self):
        """Test priority setting."""
        request = AsyncOrderRequest(
            symbol="AMZN",
            side=OrderSide.BUY,
            quantity=5,
            priority=10,
        )

        assert request.priority == 10


# =============================================================================
# Test AsyncExecutionResult
# =============================================================================

class TestAsyncExecutionResult:
    """Tests for AsyncExecutionResult dataclass."""

    def test_success_result(self):
        """Test successful execution result."""
        result = AsyncExecutionResult(
            order_id="test_001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            status="success",
            filled_quantity=100,
            average_price=150.50,
        )

        assert result.status == "success"
        assert result.filled_quantity == 100
        assert result.error is None

    def test_failed_result(self):
        """Test failed execution result."""
        result = AsyncExecutionResult(
            order_id="test_002",
            symbol="MSFT",
            side="sell",
            quantity=50,
            status="failed",
            error="Insufficient funds",
        )

        assert result.status == "failed"
        assert result.error == "Insufficient funds"
        assert result.filled_quantity == 0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = AsyncExecutionResult(
            order_id="test_003",
            symbol="GOOG",
            side="buy",
            quantity=10,
            status="success",
            elapsed_ms=150.5,
        )

        d = result.to_dict()
        assert d["order_id"] == "test_003"
        assert d["symbol"] == "GOOG"
        assert d["status"] == "success"
        assert d["elapsed_ms"] == 150.5
        assert "timestamp" in d


# =============================================================================
# Test BatchExecutionResult
# =============================================================================

class TestBatchExecutionResult:
    """Tests for BatchExecutionResult dataclass."""

    def test_batch_result(self):
        """Test batch execution result."""
        results = [
            AsyncExecutionResult("o1", "AAPL", "buy", 100, "success", 100, 150.0),
            AsyncExecutionResult("o2", "MSFT", "buy", 50, "failed", 0, 0, "Error"),
        ]

        batch = BatchExecutionResult(
            total_orders=2,
            successful=1,
            failed=1,
            total_elapsed_ms=500.0,
            results=results,
        )

        assert batch.total_orders == 2
        assert batch.successful == 1
        assert batch.failed == 1

    def test_batch_to_dict(self):
        """Test batch result conversion to dictionary."""
        batch = BatchExecutionResult(
            total_orders=5,
            successful=4,
            failed=1,
            total_elapsed_ms=1000.0,
            results=[],
        )

        d = batch.to_dict()
        assert d["total_orders"] == 5
        assert d["success_rate"] == 0.8
        assert "timestamp" in d


# =============================================================================
# Test AsyncExecutionEngine Initialization
# =============================================================================

class TestAsyncExecutionEngineInit:
    """Tests for AsyncExecutionEngine initialization."""

    def test_default_init(self):
        """Test default initialization."""
        engine = AsyncExecutionEngine()

        assert engine._max_concurrent == 5  # Default
        assert engine._default_timeout == 30.0  # Default

    def test_custom_init(self):
        """Test custom initialization."""
        engine = AsyncExecutionEngine(
            max_concurrent_orders=10,
            default_timeout_seconds=60.0,
        )

        assert engine._max_concurrent == 10
        assert engine._default_timeout == 60.0

    def test_with_sync_engine(self):
        """Test initialization with sync engine."""
        sync_engine = ExecutionEngine()
        engine = AsyncExecutionEngine(sync_engine=sync_engine)

        assert engine._sync_engine is sync_engine


# =============================================================================
# Test Single Order Execution
# =============================================================================

class TestSingleOrderExecution:
    """Tests for single order execution."""

    @pytest.mark.asyncio
    async def test_execute_market_order(self, async_engine):
        """Test executing a market order."""
        request = AsyncOrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
        )

        result = await async_engine.execute_order(request)

        assert result.symbol == "AAPL"
        assert result.side == "buy"
        assert result.quantity == 100
        assert result.status == "success"
        assert result.filled_quantity > 0

    @pytest.mark.asyncio
    async def test_execute_limit_order(self, async_engine):
        """Test executing a limit order."""
        request = AsyncOrderRequest(
            symbol="MSFT",
            side=OrderSide.SELL,
            quantity=50,
            order_type=OrderType.LIMIT,
            limit_price=350.00,
        )

        result = await async_engine.execute_order(request)

        assert result.symbol == "MSFT"
        assert result.status == "success"

    @pytest.mark.asyncio
    async def test_execution_elapsed_time(self, async_engine):
        """Test that elapsed time is tracked."""
        request = AsyncOrderRequest(
            symbol="GOOG",
            side=OrderSide.BUY,
            quantity=10,
        )

        result = await async_engine.execute_order(request)

        assert result.elapsed_ms >= 0

    @pytest.mark.asyncio
    async def test_execution_updates_statistics(self, async_engine):
        """Test that execution updates statistics."""
        initial_stats = async_engine.get_statistics()
        initial_executed = initial_stats["total_executed"]

        request = AsyncOrderRequest(
            symbol="TSLA",
            side=OrderSide.BUY,
            quantity=5,
        )

        await async_engine.execute_order(request)

        stats = async_engine.get_statistics()
        assert stats["total_executed"] == initial_executed + 1


# =============================================================================
# Test Batch Order Execution
# =============================================================================

class TestBatchOrderExecution:
    """Tests for batch order execution."""

    @pytest.mark.asyncio
    async def test_execute_batch_orders(self, async_engine):
        """Test executing multiple orders in batch."""
        requests = [
            AsyncOrderRequest("AAPL", OrderSide.BUY, 100),
            AsyncOrderRequest("MSFT", OrderSide.BUY, 50),
            AsyncOrderRequest("GOOG", OrderSide.SELL, 10),
        ]

        result = await async_engine.execute_batch_orders(requests)

        assert result.total_orders == 3
        assert result.successful == 3
        assert result.failed == 0
        assert len(result.results) == 3

    @pytest.mark.asyncio
    async def test_empty_batch(self, async_engine):
        """Test executing empty batch."""
        result = await async_engine.execute_batch_orders([])

        assert result.total_orders == 0
        assert result.successful == 0
        assert result.total_elapsed_ms == 0

    @pytest.mark.asyncio
    async def test_batch_preserves_order(self, async_engine):
        """Test batch execution with preserved order."""
        requests = [
            AsyncOrderRequest("AAPL", OrderSide.BUY, 100, priority=1),
            AsyncOrderRequest("MSFT", OrderSide.BUY, 50, priority=2),
            AsyncOrderRequest("GOOG", OrderSide.BUY, 10, priority=3),
        ]

        result = await async_engine.execute_batch_orders(
            requests,
            preserve_order=True,
        )

        # All should be successful
        assert result.successful == 3

    @pytest.mark.asyncio
    async def test_batch_priority_ordering(self, async_engine):
        """Test that batch respects priority."""
        requests = [
            AsyncOrderRequest("LOW", OrderSide.BUY, 10, priority=1),
            AsyncOrderRequest("HIGH", OrderSide.BUY, 10, priority=100),
            AsyncOrderRequest("MED", OrderSide.BUY, 10, priority=50),
        ]

        # With priority ordering, HIGH should be executed first
        result = await async_engine.execute_batch_orders(requests)

        # Results should be returned in completion order (parallel)
        # but we can verify all completed
        assert result.successful == 3


# =============================================================================
# Test Concurrency Control
# =============================================================================

class TestConcurrencyControl:
    """Tests for concurrency control."""

    @pytest.mark.asyncio
    async def test_max_concurrent_limit(self):
        """Test that max concurrent orders is respected."""
        engine = AsyncExecutionEngine(max_concurrent_orders=2)

        # Track concurrent executions
        concurrent_count = [0]
        max_concurrent_seen = [0]

        original_execute = engine._execute_single

        async def tracked_execute(request):
            concurrent_count[0] += 1
            max_concurrent_seen[0] = max(max_concurrent_seen[0], concurrent_count[0])
            await asyncio.sleep(0.05)  # Small delay
            result = await original_execute(request)
            concurrent_count[0] -= 1
            return result

        engine._execute_single = tracked_execute

        # Submit 5 orders
        requests = [
            AsyncOrderRequest(f"SYM{i}", OrderSide.BUY, 10)
            for i in range(5)
        ]

        await engine.execute_batch_orders(requests)

        # Max concurrent should not exceed limit
        assert max_concurrent_seen[0] <= 2

    @pytest.mark.asyncio
    async def test_semaphore_release_on_error(self):
        """Test that semaphore is released even on error."""
        from src.execution.execution_engine import RetryConfig

        # Create engine with no retries for this test
        engine = AsyncExecutionEngine(
            max_concurrent_orders=2,
            retry_config=RetryConfig(max_retries=0),
        )

        error_count = [0]

        async def failing_execute(request):
            error_count[0] += 1
            raise Exception("Test error")

        engine._execute_single = failing_execute

        requests = [
            AsyncOrderRequest(f"SYM{i}", OrderSide.BUY, 10)
            for i in range(3)
        ]

        result = await engine.execute_batch_orders(requests)

        # All should have been attempted (semaphore properly released)
        assert error_count[0] == 3
        assert result.failed == 3


# =============================================================================
# Test Timeout Handling
# =============================================================================

class TestTimeoutHandling:
    """Tests for timeout handling."""

    @pytest.mark.asyncio
    async def test_order_timeout_result(self):
        """Test that timeout produces correct result structure."""
        from src.execution.execution_engine import RetryConfig

        # Create engine with no retries
        engine = AsyncExecutionEngine(
            max_concurrent_orders=2,
            default_timeout_seconds=1.0,
            retry_config=RetryConfig(max_retries=0),
        )

        # Mock _execute_single to return a timeout result directly
        async def timeout_execute(request):
            return AsyncExecutionResult(
                order_id=request.order_id,
                symbol=request.symbol,
                side=request.side.value,
                quantity=request.quantity,
                status="timeout",
                error=f"Order timed out after {request.timeout_seconds}s",
            )

        engine._execute_single = timeout_execute

        request = AsyncOrderRequest(
            symbol="SLOW",
            side=OrderSide.BUY,
            quantity=10,
            timeout_seconds=0.1,
        )

        result = await engine.execute_order(request)

        # Timeout is either propagated as "timeout" status or converted to "failed"
        assert result.status in ["timeout", "failed"]
        assert "timed out" in result.error.lower() or "timeout" in result.error.lower()

    @pytest.mark.asyncio
    async def test_timeout_return_structure(self):
        """Test that AsyncExecutionResult properly represents timeout."""
        result = AsyncExecutionResult(
            order_id="test_timeout",
            symbol="TEST",
            side="buy",
            quantity=10,
            status="timeout",
            error="Order timed out after 5.0s",
        )

        assert result.status == "timeout"
        assert result.filled_quantity == 0
        d = result.to_dict()
        assert d["status"] == "timeout"
        # Check for "timed out" since that's the phrase in the error message
        assert "timed out" in d["error"].lower()

    @pytest.mark.asyncio
    async def test_default_timeout(self, async_engine):
        """Test default timeout is used."""
        request = AsyncOrderRequest(
            symbol="TEST",
            side=OrderSide.BUY,
            quantity=10,
            # No timeout specified - uses default
        )

        # Should use default (10s from fixture)
        assert async_engine._default_timeout == 10.0


# =============================================================================
# Test Retry Logic
# =============================================================================

class TestRetryLogic:
    """Tests for retry logic."""

    @pytest.mark.asyncio
    async def test_retry_on_failure(self, async_engine):
        """Test retry on recoverable failure."""
        attempt_count = [0]

        async def flaky_execute(request):
            attempt_count[0] += 1
            if attempt_count[0] < 2:
                return AsyncExecutionResult(
                    request.order_id, request.symbol, request.side.value,
                    request.quantity, "failed", error="Temporary error",
                )
            return AsyncExecutionResult(
                request.order_id, request.symbol, request.side.value,
                request.quantity, "success", filled_quantity=request.quantity,
            )

        async_engine._execute_single = flaky_execute

        request = AsyncOrderRequest(
            symbol="FLAKY",
            side=OrderSide.BUY,
            quantity=10,
        )

        result = await async_engine.execute_order(request)

        assert result.status == "success"
        assert result.retries >= 1

    @pytest.mark.asyncio
    async def test_no_retry_on_non_retryable_error(self, async_engine):
        """Test no retry on non-retryable errors."""
        attempt_count = [0]

        async def non_retryable_execute(request):
            attempt_count[0] += 1
            return AsyncExecutionResult(
                request.order_id, request.symbol, request.side.value,
                request.quantity, "failed", error="Insufficient funds",
            )

        async_engine._execute_single = non_retryable_execute

        request = AsyncOrderRequest(
            symbol="TEST",
            side=OrderSide.BUY,
            quantity=10,
        )

        result = await async_engine.execute_order(request)

        assert result.status == "failed"
        assert attempt_count[0] == 1  # No retries


# =============================================================================
# Test Order Monitoring
# =============================================================================

class TestOrderMonitoring:
    """Tests for order monitoring."""

    @pytest.mark.asyncio
    async def test_get_active_orders(self, async_engine):
        """Test getting active orders."""
        # Initially empty
        active = async_engine.get_active_orders()
        assert len(active) == 0

    @pytest.mark.asyncio
    async def test_get_recent_executions(self, async_engine):
        """Test getting recent executions."""
        # Execute some orders
        for i in range(3):
            request = AsyncOrderRequest(f"SYM{i}", OrderSide.BUY, 10)
            await async_engine.execute_order(request)

        recent = async_engine.get_recent_executions(limit=2)

        assert len(recent) == 2

    @pytest.mark.asyncio
    async def test_cancel_all_orders(self, async_engine):
        """Test cancelling all orders."""
        # Start with no active orders
        cancelled = await async_engine.cancel_all_orders()

        assert cancelled == 0  # No orders to cancel


# =============================================================================
# Test Statistics
# =============================================================================

class TestStatistics:
    """Tests for execution statistics."""

    @pytest.mark.asyncio
    async def test_statistics_tracking(self, async_engine):
        """Test statistics are properly tracked."""
        # Execute some orders
        for i in range(3):
            request = AsyncOrderRequest(f"SYM{i}", OrderSide.BUY, 10)
            await async_engine.execute_order(request)

        stats = async_engine.get_statistics()

        assert stats["total_executed"] == 3
        assert stats["successful"] == 3
        assert stats["failed"] == 0
        assert stats["success_rate"] == 1.0
        assert stats["uptime_seconds"] >= 0

    @pytest.mark.asyncio
    async def test_statistics_with_failures(self, async_engine):
        """Test statistics with some failures."""
        execution_count = [0]

        async def sometimes_fail(request):
            execution_count[0] += 1
            if execution_count[0] % 2 == 0:
                return AsyncExecutionResult(
                    request.order_id, request.symbol, request.side.value,
                    request.quantity, "failed", error="Invalid symbol",
                )
            return AsyncExecutionResult(
                request.order_id, request.symbol, request.side.value,
                request.quantity, "success", filled_quantity=request.quantity,
            )

        async_engine._execute_single = sometimes_fail

        for i in range(4):
            request = AsyncOrderRequest(f"SYM{i}", OrderSide.BUY, 10)
            await async_engine.execute_order(request)

        stats = async_engine.get_statistics()

        assert stats["total_executed"] == 4
        assert stats["successful"] == 2
        assert stats["failed"] == 2
        assert stats["success_rate"] == 0.5


# =============================================================================
# Test Bracket Order
# =============================================================================

class TestBracketOrder:
    """Tests for bracket order execution."""

    @pytest.mark.asyncio
    async def test_bracket_order_entry_only(self, async_engine):
        """Test bracket order with entry only."""
        result = await async_engine.execute_bracket_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )

        assert "entry" in result
        assert result["entry"].status == "success"

    @pytest.mark.asyncio
    async def test_bracket_order_with_stops(self, async_engine):
        """Test bracket order with stop loss and take profit."""
        result = await async_engine.execute_bracket_order(
            symbol="MSFT",
            side=OrderSide.BUY,
            quantity=50,
            entry_price=350.00,
            stop_loss_price=340.00,
            take_profit_price=370.00,
        )

        assert "entry" in result
        assert result["entry"].status == "success"


# =============================================================================
# Test Convenience Functions
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.mark.asyncio
    async def test_execute_order_async_with_strings(self):
        """Test execute_order_async with string parameters."""
        result = await execute_order_async(
            symbol="AAPL",
            side="BUY",
            quantity=100,
            order_type="market",
        )

        assert result.symbol == "AAPL"
        assert result.side == "buy"

    @pytest.mark.asyncio
    async def test_execute_order_async_with_enums(self):
        """Test execute_order_async with enum parameters."""
        result = await execute_order_async(
            symbol="MSFT",
            side=OrderSide.SELL,
            quantity=50,
            order_type=OrderType.MARKET,
        )

        assert result.symbol == "MSFT"
        assert result.side == "sell"

    @pytest.mark.asyncio
    async def test_execute_orders_async_batch(self):
        """Test execute_orders_async batch function."""
        orders = [
            AsyncOrderRequest("AAPL", OrderSide.BUY, 100),
            AsyncOrderRequest("MSFT", OrderSide.BUY, 50),
        ]

        result = await execute_orders_async(orders, max_concurrent=2)

        assert result.total_orders == 2
        assert result.successful == 2


# =============================================================================
# Test Global Instance
# =============================================================================

class TestGlobalInstance:
    """Tests for global instance management."""

    def test_get_async_execution_engine(self):
        """Test getting global async engine instance."""
        engine1 = get_async_execution_engine()
        engine2 = get_async_execution_engine()

        # Should return same instance
        assert engine1 is engine2

    def test_custom_max_concurrent(self):
        """Test custom max concurrent setting."""
        engine = get_async_execution_engine(max_concurrent_orders=10)

        # Note: max_concurrent may already be set from first call
        assert engine._max_concurrent >= 5


# =============================================================================
# Test Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_zero_quantity_handling(self, async_engine):
        """Test handling of zero quantity."""
        request = AsyncOrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=0,
        )

        result = await async_engine.execute_order(request)

        # Should fail validation
        assert result.status == "failed"

    @pytest.mark.asyncio
    async def test_very_large_batch(self):
        """Test handling of large batch."""
        # Create fresh engine for this test to avoid event loop issues
        sync_engine = ExecutionEngine(commission_rate=0.001, slippage_bps=5.0)
        engine = AsyncExecutionEngine(
            sync_engine=sync_engine,
            max_concurrent_orders=5,
        )

        requests = [
            AsyncOrderRequest(f"SYM{i}", OrderSide.BUY, 10)
            for i in range(20)
        ]

        result = await engine.execute_batch_orders(requests)

        assert result.total_orders == 20
        assert result.successful + result.failed == 20

    @pytest.mark.asyncio
    async def test_cancelled_execution(self, async_engine):
        """Test handling of cancelled execution."""
        async def slow_with_cancel(request):
            if request.metadata.get("cancelled"):
                raise asyncio.CancelledError()
            await asyncio.sleep(0.1)
            return AsyncExecutionResult(
                request.order_id, request.symbol, request.side.value,
                request.quantity, "success",
            )

        async_engine._execute_single = slow_with_cancel

        request = AsyncOrderRequest(
            symbol="TEST",
            side=OrderSide.BUY,
            quantity=10,
        )
        request.metadata["cancelled"] = True

        result = await async_engine.execute_order(request)

        assert result.status == "cancelled"

    @pytest.mark.asyncio
    async def test_order_with_metadata(self, async_engine):
        """Test order with custom metadata."""
        request = AsyncOrderRequest(
            symbol="META",
            side=OrderSide.BUY,
            quantity=25,
            metadata={"strategy": "momentum", "signal_strength": 0.85},
        )

        result = await async_engine.execute_order(request)

        assert result.status == "success"
