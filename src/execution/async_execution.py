"""
Async Execution Module
======================

Provides asynchronous execution capabilities for the trading system:
- Async order submission
- Batch order processing
- Concurrent broker operations
- Non-blocking execution monitoring

Usage:
    from src.execution.async_execution import (
        AsyncExecutionEngine,
        execute_orders_async,
    )

    engine = AsyncExecutionEngine()
    results = await engine.execute_batch_orders(orders)
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .execution_engine import (
    ExecutionEngine,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    RetryConfig,
)

logger = logging.getLogger('trading.async_execution')


# =============================================================================
# Async Execution Result
# =============================================================================

@dataclass
class AsyncExecutionResult:
    """Result of an async order execution."""
    order_id: str
    symbol: str
    side: str
    quantity: float
    status: str  # 'success', 'failed', 'timeout', 'cancelled'
    filled_quantity: float = 0.0
    average_price: float = 0.0
    error: Optional[str] = None
    elapsed_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    broker_order_id: Optional[str] = None
    retries: int = 0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "status": self.status,
            "filled_quantity": self.filled_quantity,
            "average_price": self.average_price,
            "error": self.error,
            "elapsed_ms": self.elapsed_ms,
            "timestamp": self.timestamp.isoformat(),
            "broker_order_id": self.broker_order_id,
            "retries": self.retries,
            "details": self.details,
        }


@dataclass
class BatchExecutionResult:
    """Result of a batch order execution."""
    total_orders: int
    successful: int
    failed: int
    total_elapsed_ms: float
    results: List[AsyncExecutionResult]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_orders": self.total_orders,
            "successful": self.successful,
            "failed": self.failed,
            "total_elapsed_ms": self.total_elapsed_ms,
            "timestamp": self.timestamp.isoformat(),
            "success_rate": self.successful / self.total_orders if self.total_orders > 0 else 0,
            "results": [r.to_dict() for r in self.results],
        }


# =============================================================================
# Order Request
# =============================================================================

@dataclass
class AsyncOrderRequest:
    """Request for async order execution."""
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "Day"
    timeout_seconds: float = 30.0
    priority: int = 0  # Higher priority executes first
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def order_id(self) -> str:
        """Generate order ID if not in metadata."""
        if "order_id" not in self.metadata:
            self.metadata["order_id"] = f"async_{uuid.uuid4().hex[:12]}"
        return self.metadata["order_id"]


# =============================================================================
# Async Execution Engine
# =============================================================================

class AsyncExecutionEngine:
    """
    Provides async execution capabilities for the trading system.

    Features:
    - Async order submission with timeout
    - Batch order processing with concurrency control
    - Parallel execution with priority ordering
    - Non-blocking execution monitoring
    - Automatic retry with backoff
    """

    def __init__(
        self,
        sync_engine: Optional[ExecutionEngine] = None,
        max_concurrent_orders: int = 5,
        default_timeout_seconds: float = 30.0,
        retry_config: Optional[RetryConfig] = None,
    ):
        """
        Initialize async execution engine.

        Args:
            sync_engine: Underlying sync execution engine
            max_concurrent_orders: Max orders to execute in parallel
            default_timeout_seconds: Default timeout for orders
            retry_config: Retry configuration
        """
        self._sync_engine = sync_engine or ExecutionEngine()
        self._max_concurrent = max_concurrent_orders
        self._default_timeout = default_timeout_seconds
        self._retry_config = retry_config or RetryConfig()

        # Execution state - lazily initialized to work with different event loops
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._active_orders: Dict[str, AsyncOrderRequest] = {}
        self._execution_history: List[AsyncExecutionResult] = []
        self._lock: Optional[asyncio.Lock] = None

        # Statistics
        self._total_executed = 0
        self._total_successful = 0
        self._total_failed = 0
        self._start_time = datetime.now()

        logger.info(
            f"AsyncExecutionEngine initialized (max_concurrent={max_concurrent_orders})"
        )

    # -------------------------------------------------------------------------
    # Single Order Execution
    # -------------------------------------------------------------------------

    async def execute_order(
        self,
        request: AsyncOrderRequest,
    ) -> AsyncExecutionResult:
        """
        Execute a single order asynchronously.

        Args:
            request: Order request

        Returns:
            AsyncExecutionResult with execution details
        """
        start_time = time.time()
        order_id = request.order_id
        result = None

        # Lazily initialize lock and semaphore for current event loop
        if self._lock is None:
            self._lock = asyncio.Lock()
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._max_concurrent)

        async with self._lock:
            self._active_orders[order_id] = request

        try:
            # Acquire semaphore to limit concurrency
            async with self._semaphore:
                result = await self._execute_with_retry(request)

        except Exception as e:
            # Handle any unexpected errors
            result = AsyncExecutionResult(
                order_id=order_id,
                symbol=request.symbol,
                side=request.side.value,
                quantity=request.quantity,
                status="failed",
                error=f"Unexpected error: {e}",
            )
            logger.error(f"Unexpected error executing order {order_id}: {e}")

        finally:
            elapsed_ms = (time.time() - start_time) * 1000
            if result is not None:
                result.elapsed_ms = elapsed_ms

                async with self._lock:
                    self._active_orders.pop(order_id, None)
                    self._execution_history.append(result)
                    self._total_executed += 1
                    if result.status == "success":
                        self._total_successful += 1
                    else:
                        self._total_failed += 1

        return result

    async def _execute_with_retry(
        self,
        request: AsyncOrderRequest,
    ) -> AsyncExecutionResult:
        """Execute order with retry logic."""
        last_error = None
        retries = 0

        for attempt in range(self._retry_config.max_retries + 1):
            try:
                result = await self._execute_single(request)

                if result.status == "success":
                    result.retries = retries
                    return result

                # Check if error is retryable
                if not self._is_retryable_error(result.error):
                    result.retries = retries
                    return result

                last_error = result.error

            except asyncio.TimeoutError:
                last_error = "Order execution timed out"
            except asyncio.CancelledError:
                return AsyncExecutionResult(
                    order_id=request.order_id,
                    symbol=request.symbol,
                    side=request.side.value,
                    quantity=request.quantity,
                    status="cancelled",
                    error="Order was cancelled",
                    retries=retries,
                )
            except Exception as e:
                last_error = str(e)
                logger.error(f"Order {request.order_id} attempt {attempt + 1} failed: {e}")

            retries += 1

            if attempt < self._retry_config.max_retries:
                delay = self._retry_config.calculate_delay(attempt)
                logger.warning(
                    f"Retrying order {request.order_id} in {delay:.1f}s "
                    f"(attempt {attempt + 2}/{self._retry_config.max_retries + 1})"
                )
                await asyncio.sleep(delay)

        # All retries exhausted
        return AsyncExecutionResult(
            order_id=request.order_id,
            symbol=request.symbol,
            side=request.side.value,
            quantity=request.quantity,
            status="failed",
            error=last_error or "All retry attempts exhausted",
            retries=retries,
        )

    async def _execute_single(
        self,
        request: AsyncOrderRequest,
    ) -> AsyncExecutionResult:
        """Execute a single order (no retry)."""
        timeout = request.timeout_seconds or self._default_timeout

        try:
            # Run the sync broker call in a thread pool
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    self._sync_broker_call,
                    request,
                ),
                timeout=timeout,
            )
            return result

        except asyncio.TimeoutError:
            return AsyncExecutionResult(
                order_id=request.order_id,
                symbol=request.symbol,
                side=request.side.value,
                quantity=request.quantity,
                status="timeout",
                error=f"Order timed out after {timeout}s",
            )

    def _sync_broker_call(
        self,
        request: AsyncOrderRequest,
    ) -> AsyncExecutionResult:
        """Make synchronous broker call (runs in thread pool)."""
        try:
            if self._sync_engine.broker:
                # Use real broker
                broker_result = self._sync_engine.execute_broker_order(
                    symbol=request.symbol,
                    side=request.side,
                    quantity=request.quantity,
                    order_type=request.order_type,
                    limit_price=request.limit_price,
                    time_in_force=request.time_in_force,
                    timeout_seconds=request.timeout_seconds,
                )

                if broker_result.get("submitted"):
                    return AsyncExecutionResult(
                        order_id=request.order_id,
                        symbol=request.symbol,
                        side=request.side.value,
                        quantity=request.quantity,
                        status="success",
                        filled_quantity=request.quantity,
                        broker_order_id=broker_result.get("order_id"),
                        details=broker_result,
                    )
                else:
                    return AsyncExecutionResult(
                        order_id=request.order_id,
                        symbol=request.symbol,
                        side=request.side.value,
                        quantity=request.quantity,
                        status="failed",
                        error=broker_result.get("error", "Order rejected"),
                        details=broker_result,
                    )
            else:
                # Demo mode - simulate execution
                order = self._sync_engine.create_order(
                    symbol=request.symbol,
                    side=request.side,
                    quantity=request.quantity,
                    order_type=request.order_type,
                    limit_price=request.limit_price,
                    stop_price=request.stop_price,
                )

                # Simulate market execution
                demo_price = request.limit_price or 100.0
                self._sync_engine.execute_market_order(
                    order=order,
                    current_price=demo_price,
                    volume=10000,
                )

                return AsyncExecutionResult(
                    order_id=request.order_id,
                    symbol=request.symbol,
                    side=request.side.value,
                    quantity=request.quantity,
                    status="success",
                    filled_quantity=order.filled_quantity,
                    average_price=order.average_fill_price,
                    broker_order_id=order.order_id,
                )

        except Exception as e:
            logger.error(f"Broker call failed for {request.symbol}: {e}")
            return AsyncExecutionResult(
                order_id=request.order_id,
                symbol=request.symbol,
                side=request.side.value,
                quantity=request.quantity,
                status="failed",
                error=str(e),
            )

    def _is_retryable_error(self, error: Optional[str]) -> bool:
        """Check if an error is retryable."""
        if not error:
            return True

        error_lower = error.lower()

        # Non-retryable errors
        non_retryable = [
            "insufficient funds",
            "invalid symbol",
            "market closed",
            "account restricted",
            "permission denied",
            "invalid quantity",
            "cancelled",
        ]

        return not any(term in error_lower for term in non_retryable)

    # -------------------------------------------------------------------------
    # Batch Order Execution
    # -------------------------------------------------------------------------

    async def execute_batch_orders(
        self,
        requests: List[AsyncOrderRequest],
        preserve_order: bool = False,
    ) -> BatchExecutionResult:
        """
        Execute multiple orders concurrently.

        Args:
            requests: List of order requests
            preserve_order: If True, execute in order; otherwise parallel

        Returns:
            BatchExecutionResult with all order results
        """
        if not requests:
            return BatchExecutionResult(
                total_orders=0,
                successful=0,
                failed=0,
                total_elapsed_ms=0,
                results=[],
            )

        start_time = time.time()

        # Sort by priority (higher priority first)
        sorted_requests = sorted(requests, key=lambda r: -r.priority)

        if preserve_order:
            # Execute sequentially
            results = []
            for request in sorted_requests:
                result = await self.execute_order(request)
                results.append(result)
        else:
            # Execute concurrently
            tasks = [
                self.execute_order(request)
                for request in sorted_requests
            ]
            results = await asyncio.gather(*tasks)

        elapsed_ms = (time.time() - start_time) * 1000
        successful = sum(1 for r in results if r.status == "success")
        failed = len(results) - successful

        return BatchExecutionResult(
            total_orders=len(requests),
            successful=successful,
            failed=failed,
            total_elapsed_ms=elapsed_ms,
            results=list(results),
        )

    async def execute_bracket_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        entry_price: Optional[float] = None,
        stop_loss_price: Optional[float] = None,
        take_profit_price: Optional[float] = None,
    ) -> Dict[str, AsyncExecutionResult]:
        """
        Execute bracket order (entry + stop loss + take profit) asynchronously.

        Args:
            symbol: Trading symbol
            side: Order side
            quantity: Number of shares
            entry_price: Entry limit price (None for market)
            stop_loss_price: Stop loss price
            take_profit_price: Take profit price

        Returns:
            Dict with entry, stop_loss, and take_profit results
        """
        results = {}

        # Execute entry order first
        entry_request = AsyncOrderRequest(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.LIMIT if entry_price else OrderType.MARKET,
            limit_price=entry_price,
            priority=10,  # High priority
        )

        entry_result = await self.execute_order(entry_request)
        results["entry"] = entry_result

        if entry_result.status != "success":
            logger.error(f"Entry order failed for {symbol}: {entry_result.error}")
            return results

        # Place protective orders concurrently
        protective_requests = []
        exit_side = OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY

        if stop_loss_price:
            protective_requests.append(AsyncOrderRequest(
                symbol=symbol,
                side=exit_side,
                quantity=quantity,
                order_type=OrderType.STOP,
                stop_price=stop_loss_price,
                metadata={"type": "stop_loss"},
            ))

        if take_profit_price:
            protective_requests.append(AsyncOrderRequest(
                symbol=symbol,
                side=exit_side,
                quantity=quantity,
                order_type=OrderType.LIMIT,
                limit_price=take_profit_price,
                metadata={"type": "take_profit"},
            ))

        if protective_requests:
            protective_results = await self.execute_batch_orders(protective_requests)

            for result in protective_results.results:
                order_type = result.details.get("type", "protective")
                if result.details.get("type") == "stop_loss":
                    results["stop_loss"] = result
                elif result.details.get("type") == "take_profit":
                    results["take_profit"] = result
                else:
                    results[order_type] = result

        return results

    # -------------------------------------------------------------------------
    # Order Monitoring
    # -------------------------------------------------------------------------

    async def wait_for_fill(
        self,
        order_id: str,
        timeout_seconds: float = 60.0,
        poll_interval: float = 1.0,
    ) -> Optional[AsyncExecutionResult]:
        """
        Wait for an order to be filled.

        Args:
            order_id: Order ID to monitor
            timeout_seconds: Maximum wait time
            poll_interval: Polling interval

        Returns:
            Execution result if filled, None if timeout
        """
        start_time = time.time()

        while (time.time() - start_time) < timeout_seconds:
            async with self._lock:
                # Check if order completed
                for result in reversed(self._execution_history):
                    if result.order_id == order_id:
                        return result

                # Check if still active
                if order_id not in self._active_orders:
                    # Order finished but not found - may have been cleared
                    return None

            await asyncio.sleep(poll_interval)

        return None

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an active order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancelled, False otherwise
        """
        async with self._lock:
            if order_id in self._active_orders:
                # Order is still active - flag for cancellation
                self._active_orders[order_id].metadata["cancelled"] = True
                logger.info(f"Order {order_id} flagged for cancellation")
                return True

        logger.warning(f"Order {order_id} not found in active orders")
        return False

    async def cancel_all_orders(self) -> int:
        """
        Cancel all active orders.

        Returns:
            Number of orders cancelled
        """
        cancelled = 0

        # Handle case where no orders have been executed yet
        if not self._active_orders:
            return 0

        # Lazily initialize lock if needed
        if self._lock is None:
            self._lock = asyncio.Lock()

        async with self._lock:
            for order_id in list(self._active_orders.keys()):
                self._active_orders[order_id].metadata["cancelled"] = True
                cancelled += 1

        logger.info(f"Cancelled {cancelled} active orders")
        return cancelled

    # -------------------------------------------------------------------------
    # Statistics and Status
    # -------------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics."""
        uptime = (datetime.now() - self._start_time).total_seconds()

        return {
            "total_executed": self._total_executed,
            "successful": self._total_successful,
            "failed": self._total_failed,
            "success_rate": (
                self._total_successful / self._total_executed
                if self._total_executed > 0 else 0
            ),
            "active_orders": len(self._active_orders),
            "max_concurrent": self._max_concurrent,
            "uptime_seconds": uptime,
            "orders_per_minute": (
                self._total_executed / (uptime / 60)
                if uptime > 0 else 0
            ),
        }

    def get_active_orders(self) -> List[Dict[str, Any]]:
        """Get list of active orders."""
        return [
            {
                "order_id": order.order_id,
                "symbol": order.symbol,
                "side": order.side.value,
                "quantity": order.quantity,
            }
            for order in self._active_orders.values()
        ]

    def get_recent_executions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent execution results."""
        return [
            result.to_dict()
            for result in self._execution_history[-limit:]
        ]


# =============================================================================
# Convenience Functions
# =============================================================================

async def execute_orders_async(
    orders: List[AsyncOrderRequest],
    max_concurrent: int = 5,
    engine: Optional[AsyncExecutionEngine] = None,
) -> BatchExecutionResult:
    """
    Execute multiple orders asynchronously.

    Args:
        orders: List of order requests
        max_concurrent: Maximum concurrent executions
        engine: Optional existing engine to use

    Returns:
        BatchExecutionResult with all results
    """
    if engine is None:
        engine = AsyncExecutionEngine(max_concurrent_orders=max_concurrent)

    return await engine.execute_batch_orders(orders)


async def execute_order_async(
    symbol: str,
    side: Union[str, OrderSide],
    quantity: float,
    order_type: Union[str, OrderType] = OrderType.MARKET,
    limit_price: Optional[float] = None,
    timeout_seconds: float = 30.0,
    engine: Optional[AsyncExecutionEngine] = None,
) -> AsyncExecutionResult:
    """
    Execute a single order asynchronously.

    Args:
        symbol: Trading symbol
        side: 'BUY' or 'SELL' or OrderSide enum
        quantity: Number of shares
        order_type: Order type
        limit_price: Limit price for limit orders
        timeout_seconds: Timeout
        engine: Optional existing engine to use

    Returns:
        AsyncExecutionResult
    """
    if isinstance(side, str):
        side = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL

    if isinstance(order_type, str):
        order_type = OrderType(order_type.lower())

    request = AsyncOrderRequest(
        symbol=symbol,
        side=side,
        quantity=quantity,
        order_type=order_type,
        limit_price=limit_price,
        timeout_seconds=timeout_seconds,
    )

    if engine is None:
        engine = AsyncExecutionEngine()

    return await engine.execute_order(request)


# =============================================================================
# Global Instance
# =============================================================================

_async_engine_instance: Optional[AsyncExecutionEngine] = None


def get_async_execution_engine(
    max_concurrent_orders: int = 5,
) -> AsyncExecutionEngine:
    """Get global async execution engine instance."""
    global _async_engine_instance
    if _async_engine_instance is None:
        _async_engine_instance = AsyncExecutionEngine(
            max_concurrent_orders=max_concurrent_orders,
        )
    return _async_engine_instance


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Classes
    "AsyncExecutionEngine",
    "AsyncOrderRequest",
    "AsyncExecutionResult",
    "BatchExecutionResult",
    # Functions
    "execute_orders_async",
    "execute_order_async",
    "get_async_execution_engine",
]
