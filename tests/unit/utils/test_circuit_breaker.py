"""Unit tests for Circuit Breaker.

Tests cover:
- State transitions (CLOSED -> OPEN -> HALF_OPEN -> CLOSED)
- Failure threshold triggering
- Recovery timeout behavior
- Half-open state behavior
- CircuitBreakerOpenError raising
- Metrics tracking
- Concurrent request handling
- Registry functionality
- Decorator behavior
- Retry with circuit breaker
"""

import pytest
from unittest.mock import Mock, patch
import threading
import time
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.utils.circuit_breaker import (
    CircuitState,
    CircuitStats,
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitBreakerRegistry,
    get_circuit_breaker_registry,
    circuit_breaker,
    retry_with_circuit_breaker,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def breaker():
    """Create a basic circuit breaker for testing."""
    return CircuitBreaker(
        name="test_breaker",
        failure_threshold=3,
        recovery_timeout=1.0,  # Short timeout for testing
        half_open_max_calls=2,
        success_threshold=1,
    )


@pytest.fixture
def fast_breaker():
    """Create a circuit breaker with very short timeouts for testing."""
    return CircuitBreaker(
        name="fast_breaker",
        failure_threshold=2,
        recovery_timeout=0.1,  # 100ms timeout
        half_open_max_calls=1,
        success_threshold=1,
    )


@pytest.fixture
def registry():
    """Create a fresh circuit breaker registry."""
    return CircuitBreakerRegistry()


@pytest.fixture
def reset_global_registry():
    """Reset global registry after test."""
    registry = get_circuit_breaker_registry()
    original_breakers = registry._breakers.copy()
    yield
    registry._breakers.clear()
    registry._breakers.update(original_breakers)


# =============================================================================
# Test: CircuitState Enum
# =============================================================================

class TestCircuitState:
    """Tests for CircuitState enum."""

    def test_states_exist(self):
        """Test that all expected states exist."""
        assert CircuitState.CLOSED is not None
        assert CircuitState.OPEN is not None
        assert CircuitState.HALF_OPEN is not None

    def test_state_names(self):
        """Test state names."""
        assert CircuitState.CLOSED.name == "CLOSED"
        assert CircuitState.OPEN.name == "OPEN"
        assert CircuitState.HALF_OPEN.name == "HALF_OPEN"


# =============================================================================
# Test: CircuitStats Dataclass
# =============================================================================

class TestCircuitStats:
    """Tests for CircuitStats dataclass."""

    def test_default_values(self):
        """Test default values."""
        stats = CircuitStats()

        assert stats.total_requests == 0
        assert stats.successful_requests == 0
        assert stats.failed_requests == 0
        assert stats.rejected_requests == 0
        assert stats.last_failure_time is None
        assert stats.last_success_time is None
        assert stats.consecutive_failures == 0
        assert stats.consecutive_successes == 0
        assert stats.state_changes == 0


# =============================================================================
# Test: CircuitBreakerOpenError
# =============================================================================

class TestCircuitBreakerOpenError:
    """Tests for CircuitBreakerOpenError exception."""

    def test_error_creation(self):
        """Test error creation with name and time."""
        error = CircuitBreakerOpenError("test_breaker", 30.5)

        assert error.name == "test_breaker"
        assert error.time_until_retry == 30.5
        assert "test_breaker" in str(error)
        assert "OPEN" in str(error)
        assert "30.5" in str(error)

    def test_error_can_be_raised(self):
        """Test error can be raised and caught."""
        with pytest.raises(CircuitBreakerOpenError) as exc_info:
            raise CircuitBreakerOpenError("my_breaker", 10.0)

        assert exc_info.value.name == "my_breaker"


# =============================================================================
# Test: State Transitions
# =============================================================================

class TestStateTransitions:
    """Tests for circuit breaker state transitions."""

    def test_initial_state_is_closed(self, breaker):
        """Test circuit starts in CLOSED state."""
        assert breaker.state == CircuitState.CLOSED

    def test_closed_to_open_after_failures(self, breaker):
        """Test CLOSED -> OPEN after N consecutive failures."""
        # Record failures up to threshold
        for _ in range(breaker.failure_threshold):
            breaker.record_failure(Exception("test error"))

        assert breaker.state == CircuitState.OPEN

    def test_open_to_half_open_after_timeout(self, fast_breaker):
        """Test OPEN -> HALF_OPEN after recovery timeout."""
        # Trip the circuit
        for _ in range(fast_breaker.failure_threshold):
            fast_breaker.record_failure(Exception("test error"))

        assert fast_breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        time.sleep(0.15)  # Slightly more than 100ms

        # State check should trigger transition
        assert fast_breaker.state == CircuitState.HALF_OPEN

    def test_half_open_to_closed_on_success(self, fast_breaker):
        """Test HALF_OPEN -> CLOSED on successful request."""
        # Trip and wait
        for _ in range(fast_breaker.failure_threshold):
            fast_breaker.record_failure(Exception("test error"))
        time.sleep(0.15)

        assert fast_breaker.state == CircuitState.HALF_OPEN

        # Record success
        fast_breaker.allow_request()  # First need to allow the request
        fast_breaker.record_success()

        assert fast_breaker.state == CircuitState.CLOSED

    def test_half_open_to_open_on_failure(self, fast_breaker):
        """Test HALF_OPEN -> OPEN on failed request."""
        # Trip and wait
        for _ in range(fast_breaker.failure_threshold):
            fast_breaker.record_failure(Exception("test error"))
        time.sleep(0.15)

        assert fast_breaker.state == CircuitState.HALF_OPEN

        # Record failure
        fast_breaker.allow_request()
        fast_breaker.record_failure(Exception("still failing"))

        assert fast_breaker.state == CircuitState.OPEN

    def test_state_change_increments_counter(self, breaker):
        """Test state changes increment counter."""
        initial_changes = breaker.stats.state_changes

        # Trip the circuit
        for _ in range(breaker.failure_threshold):
            breaker.record_failure(Exception("test"))

        assert breaker.stats.state_changes > initial_changes


# =============================================================================
# Test: Request Handling
# =============================================================================

class TestRequestHandling:
    """Tests for request handling."""

    def test_allow_request_when_closed(self, breaker):
        """Test requests are allowed when CLOSED."""
        assert breaker.allow_request() is True

    def test_deny_request_when_open(self, breaker):
        """Test requests are denied when OPEN."""
        # Trip the circuit
        for _ in range(breaker.failure_threshold):
            breaker.record_failure(Exception("test"))

        assert breaker.state == CircuitState.OPEN
        assert breaker.allow_request() is False

    def test_limited_requests_when_half_open(self, fast_breaker):
        """Test limited requests allowed when HALF_OPEN."""
        # Trip and wait
        for _ in range(fast_breaker.failure_threshold):
            fast_breaker.record_failure(Exception("test"))
        time.sleep(0.15)

        assert fast_breaker.state == CircuitState.HALF_OPEN

        # Only half_open_max_calls should be allowed
        assert fast_breaker.allow_request() is True  # First call
        assert fast_breaker.allow_request() is False  # Exceeds limit

    def test_rejected_requests_tracked(self, breaker):
        """Test rejected requests are tracked in stats."""
        # Trip the circuit
        for _ in range(breaker.failure_threshold):
            breaker.record_failure(Exception("test"))

        # Try to make requests
        breaker.allow_request()
        breaker.allow_request()

        assert breaker.stats.rejected_requests >= 2


# =============================================================================
# Test: Success/Failure Recording
# =============================================================================

class TestRecording:
    """Tests for success/failure recording."""

    def test_record_success_updates_stats(self, breaker):
        """Test record_success updates statistics."""
        breaker.allow_request()
        breaker.record_success()

        stats = breaker.stats
        assert stats.total_requests == 1
        assert stats.successful_requests == 1
        assert stats.consecutive_successes == 1
        assert stats.last_success_time is not None

    def test_record_failure_updates_stats(self, breaker):
        """Test record_failure updates statistics."""
        breaker.allow_request()
        breaker.record_failure(Exception("test error"))

        stats = breaker.stats
        assert stats.total_requests == 1
        assert stats.failed_requests == 1
        assert stats.consecutive_failures == 1
        assert stats.last_failure_time is not None

    def test_success_resets_consecutive_failures(self, breaker):
        """Test success resets consecutive failure count."""
        # Record some failures
        breaker.record_failure(Exception("test"))
        breaker.record_failure(Exception("test"))

        assert breaker.stats.consecutive_failures == 2

        # Record success
        breaker.record_success()

        assert breaker.stats.consecutive_failures == 0
        assert breaker.stats.consecutive_successes == 1

    def test_failure_resets_consecutive_successes(self, breaker):
        """Test failure resets consecutive success count."""
        # Record some successes
        breaker.record_success()
        breaker.record_success()

        assert breaker.stats.consecutive_successes == 2

        # Record failure
        breaker.record_failure(Exception("test"))

        assert breaker.stats.consecutive_successes == 0
        assert breaker.stats.consecutive_failures == 1


# =============================================================================
# Test: Recovery Timeout
# =============================================================================

class TestRecoveryTimeout:
    """Tests for recovery timeout functionality."""

    def test_time_until_retry_when_closed(self, breaker):
        """Test time_until_retry is 0 when CLOSED."""
        assert breaker.time_until_retry() == 0.0

    def test_time_until_retry_when_open(self, breaker):
        """Test time_until_retry returns positive value when OPEN."""
        # Trip the circuit
        for _ in range(breaker.failure_threshold):
            breaker.record_failure(Exception("test"))

        assert breaker.state == CircuitState.OPEN
        retry_time = breaker.time_until_retry()
        assert 0 < retry_time <= breaker.recovery_timeout

    def test_time_until_retry_decreases(self, fast_breaker):
        """Test time_until_retry decreases over time."""
        # Trip the circuit
        for _ in range(fast_breaker.failure_threshold):
            fast_breaker.record_failure(Exception("test"))

        initial_time = fast_breaker.time_until_retry()
        time.sleep(0.05)
        later_time = fast_breaker.time_until_retry()

        assert later_time < initial_time


# =============================================================================
# Test: Failure Rate Threshold
# =============================================================================

class TestFailureRateThreshold:
    """Tests for failure rate threshold functionality."""

    def test_opens_on_failure_rate(self):
        """Test circuit opens when failure rate exceeds threshold."""
        breaker = CircuitBreaker(
            name="rate_breaker",
            failure_threshold=100,  # High threshold
            failure_rate_threshold=0.5,  # 50% failure rate
            window_size=60.0,
        )

        # Record mixed results - 60% failure rate
        for _ in range(4):
            breaker.record_failure(Exception("test"))
        for _ in range(2):
            breaker.record_success()

        # Should still be closed (60% > 50%)
        # Wait, we need more requests in the window
        for _ in range(4):
            breaker.record_failure(Exception("test"))

        assert breaker.state == CircuitState.OPEN


# =============================================================================
# Test: Reset and Force Open
# =============================================================================

class TestResetAndForce:
    """Tests for reset and force_open functionality."""

    def test_reset_returns_to_closed(self, breaker):
        """Test reset returns circuit to CLOSED."""
        # Trip the circuit
        for _ in range(breaker.failure_threshold):
            breaker.record_failure(Exception("test"))

        assert breaker.state == CircuitState.OPEN

        breaker.reset()

        assert breaker.state == CircuitState.CLOSED

    def test_reset_clears_stats(self, breaker):
        """Test reset clears statistics."""
        breaker.record_success()
        breaker.record_failure(Exception("test"))

        breaker.reset()

        stats = breaker.stats
        assert stats.total_requests == 0
        assert stats.successful_requests == 0
        assert stats.failed_requests == 0

    def test_force_open(self, breaker):
        """Test force_open opens the circuit."""
        assert breaker.state == CircuitState.CLOSED

        breaker.force_open()

        assert breaker.state == CircuitState.OPEN


# =============================================================================
# Test: Context Manager
# =============================================================================

class TestContextManager:
    """Tests for context manager usage."""

    def test_context_manager_success(self, breaker):
        """Test context manager records success on normal exit."""
        with breaker:
            pass  # Success

        assert breaker.stats.successful_requests == 1

    def test_context_manager_failure(self, breaker):
        """Test context manager records failure on exception."""
        with pytest.raises(ValueError):
            with breaker:
                raise ValueError("test error")

        assert breaker.stats.failed_requests == 1

    def test_context_manager_raises_when_open(self, breaker):
        """Test context manager raises error when circuit is open."""
        # Trip the circuit
        for _ in range(breaker.failure_threshold):
            breaker.record_failure(Exception("test"))

        with pytest.raises(CircuitBreakerOpenError) as exc_info:
            with breaker:
                pass

        assert exc_info.value.name == breaker.name


# =============================================================================
# Test: Callbacks
# =============================================================================

class TestCallbacks:
    """Tests for state change callbacks."""

    def test_on_open_callback(self):
        """Test on_open callback is called."""
        callback = Mock()
        breaker = CircuitBreaker(
            name="callback_test",
            failure_threshold=2,
            on_open=callback,
        )

        # Trip the circuit
        for _ in range(2):
            breaker.record_failure(Exception("test"))

        callback.assert_called_once_with("callback_test")

    def test_on_close_callback(self):
        """Test on_close callback is called."""
        callback = Mock()
        breaker = CircuitBreaker(
            name="callback_test",
            failure_threshold=2,
            recovery_timeout=0.1,
            success_threshold=1,
            on_close=callback,
        )

        # Trip and recover
        for _ in range(2):
            breaker.record_failure(Exception("test"))
        time.sleep(0.15)
        breaker.allow_request()
        breaker.record_success()

        callback.assert_called_once_with("callback_test")

    def test_on_half_open_callback(self):
        """Test on_half_open callback is called."""
        callback = Mock()
        breaker = CircuitBreaker(
            name="callback_test",
            failure_threshold=2,
            recovery_timeout=0.1,
            on_half_open=callback,
        )

        # Trip and wait
        for _ in range(2):
            breaker.record_failure(Exception("test"))
        time.sleep(0.15)
        _ = breaker.state  # Trigger check

        callback.assert_called_once_with("callback_test")


# =============================================================================
# Test: CircuitBreakerRegistry
# =============================================================================

class TestCircuitBreakerRegistry:
    """Tests for CircuitBreakerRegistry."""

    def test_get_or_create_new(self, registry):
        """Test get_or_create creates new breaker."""
        breaker = registry.get_or_create("test_new", failure_threshold=5)

        assert breaker is not None
        assert breaker.name == "test_new"
        assert breaker.failure_threshold == 5

    def test_get_or_create_existing(self, registry):
        """Test get_or_create returns existing breaker."""
        breaker1 = registry.get_or_create("test_existing")
        breaker2 = registry.get_or_create("test_existing")

        assert breaker1 is breaker2

    def test_get_returns_none_if_not_exists(self, registry):
        """Test get returns None for non-existent breaker."""
        assert registry.get("nonexistent") is None

    def test_get_returns_existing(self, registry):
        """Test get returns existing breaker."""
        registry.get_or_create("test_get")
        breaker = registry.get("test_get")

        assert breaker is not None
        assert breaker.name == "test_get"

    def test_all_returns_copy(self, registry):
        """Test all returns a copy of all breakers."""
        registry.get_or_create("breaker1")
        registry.get_or_create("breaker2")

        all_breakers = registry.all()

        assert len(all_breakers) == 2
        assert "breaker1" in all_breakers
        assert "breaker2" in all_breakers

    def test_reset_all(self, registry):
        """Test reset_all resets all breakers."""
        breaker1 = registry.get_or_create("reset1", failure_threshold=2)
        breaker2 = registry.get_or_create("reset2", failure_threshold=2)

        # Trip both
        for _ in range(2):
            breaker1.record_failure(Exception("test"))
            breaker2.record_failure(Exception("test"))

        assert breaker1.state == CircuitState.OPEN
        assert breaker2.state == CircuitState.OPEN

        registry.reset_all()

        assert breaker1.state == CircuitState.CLOSED
        assert breaker2.state == CircuitState.CLOSED

    def test_get_status(self, registry):
        """Test get_status returns status of all breakers."""
        breaker = registry.get_or_create("status_test", failure_threshold=3)
        breaker.record_success()
        breaker.record_failure(Exception("test"))

        status = registry.get_status()

        assert "status_test" in status
        assert status["status_test"]["state"] == "CLOSED"
        assert status["status_test"]["total_requests"] == 2
        assert status["status_test"]["failed_requests"] == 1


class TestGlobalRegistry:
    """Tests for global registry."""

    def test_get_circuit_breaker_registry(self, reset_global_registry):
        """Test get_circuit_breaker_registry returns global registry."""
        registry1 = get_circuit_breaker_registry()
        registry2 = get_circuit_breaker_registry()

        assert registry1 is registry2


# =============================================================================
# Test: Decorator
# =============================================================================

class TestDecorator:
    """Tests for circuit_breaker decorator."""

    def test_decorator_success(self, reset_global_registry):
        """Test decorator records success on normal execution."""
        @circuit_breaker(name="decorator_success", failure_threshold=3)
        def my_function():
            return "success"

        result = my_function()

        assert result == "success"
        assert my_function.circuit_breaker.stats.successful_requests == 1

    def test_decorator_failure(self, reset_global_registry):
        """Test decorator records failure on exception."""
        @circuit_breaker(name="decorator_failure", failure_threshold=3)
        def my_function():
            raise ValueError("test error")

        with pytest.raises(ValueError):
            my_function()

        assert my_function.circuit_breaker.stats.failed_requests == 1

    def test_decorator_opens_circuit(self, reset_global_registry):
        """Test decorator opens circuit after threshold."""
        @circuit_breaker(name="decorator_open", failure_threshold=2)
        def my_function():
            raise ValueError("test error")

        # Call twice to trip
        for _ in range(2):
            with pytest.raises(ValueError):
                my_function()

        # Third call should raise CircuitBreakerOpenError
        with pytest.raises(CircuitBreakerOpenError):
            my_function()

    def test_decorator_with_fallback(self, reset_global_registry):
        """Test decorator uses fallback when open."""
        @circuit_breaker(
            name="decorator_fallback",
            failure_threshold=2,
            fallback=lambda: "fallback_value"
        )
        def my_function():
            raise ValueError("test error")

        # Trip the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                my_function()

        # Should use fallback
        result = my_function()
        assert result == "fallback_value"

    def test_decorator_with_excluded_exceptions(self, reset_global_registry):
        """Test decorator doesn't count excluded exceptions as failures."""
        @circuit_breaker(
            name="decorator_exclude",
            failure_threshold=2,
            exclude_exceptions=(ValueError,)
        )
        def my_function(should_fail):
            if should_fail:
                raise ValueError("excluded error")
            return "success"

        # Raise excluded exceptions - should not trip circuit
        for _ in range(5):
            with pytest.raises(ValueError):
                my_function(True)

        # Circuit should still be closed
        assert my_function.circuit_breaker.state == CircuitState.CLOSED


# =============================================================================
# Test: Retry with Circuit Breaker
# =============================================================================

class TestRetryWithCircuitBreaker:
    """Tests for retry_with_circuit_breaker decorator."""

    def test_retry_success_first_attempt(self, breaker):
        """Test retry succeeds on first attempt."""
        call_count = 0

        @retry_with_circuit_breaker(breaker, max_retries=3)
        def my_function():
            nonlocal call_count
            call_count += 1
            return "success"

        result = my_function()

        assert result == "success"
        assert call_count == 1

    def test_retry_success_after_failures(self, breaker):
        """Test retry succeeds after some failures."""
        call_count = 0

        @retry_with_circuit_breaker(
            breaker,
            max_retries=3,
            base_delay=0.01,  # Fast delay for testing
        )
        def my_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("temporary error")
            return "success"

        result = my_function()

        assert result == "success"
        assert call_count == 3

    def test_retry_exhausted(self, breaker):
        """Test retry raises after all attempts exhausted."""
        @retry_with_circuit_breaker(
            breaker,
            max_retries=2,
            base_delay=0.01,
        )
        def my_function():
            raise ValueError("persistent error")

        with pytest.raises(ValueError):
            my_function()

        assert breaker.stats.failed_requests == 3  # Initial + 2 retries

    def test_retry_respects_circuit_open(self, breaker):
        """Test retry stops when circuit opens."""
        @retry_with_circuit_breaker(
            breaker,
            max_retries=10,
            base_delay=0.01,
        )
        def my_function():
            raise ValueError("error")

        # Should raise CircuitBreakerOpenError after circuit trips
        with pytest.raises(CircuitBreakerOpenError):
            my_function()

        # Should have tripped after failure_threshold failures
        assert breaker.stats.failed_requests == breaker.failure_threshold


# =============================================================================
# Test: Thread Safety
# =============================================================================

class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_record_success(self, breaker):
        """Test concurrent record_success calls are thread-safe."""
        def record_many():
            for _ in range(100):
                breaker.record_success()

        threads = [threading.Thread(target=record_many) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert breaker.stats.successful_requests == 500

    def test_concurrent_record_failure(self, breaker):
        """Test concurrent record_failure calls are thread-safe."""
        breaker = CircuitBreaker(
            name="concurrent_fail",
            failure_threshold=1000,  # High threshold to avoid tripping
        )

        def record_many():
            for _ in range(100):
                breaker.record_failure(Exception("test"))

        threads = [threading.Thread(target=record_many) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert breaker.stats.failed_requests == 500

    def test_concurrent_allow_request(self, fast_breaker):
        """Test concurrent allow_request calls are thread-safe."""
        # Trip the circuit
        for _ in range(fast_breaker.failure_threshold):
            fast_breaker.record_failure(Exception("test"))

        time.sleep(0.15)  # Wait for half-open

        results = []

        def check_allow():
            allowed = fast_breaker.allow_request()
            results.append(allowed)

        threads = [threading.Thread(target=check_allow) for _ in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Only half_open_max_calls should be allowed
        allowed_count = sum(1 for r in results if r)
        assert allowed_count == fast_breaker.half_open_max_calls


# =============================================================================
# Test: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_zero_failure_threshold(self):
        """Test breaker with zero failure threshold opens immediately."""
        breaker = CircuitBreaker(
            name="zero_threshold",
            failure_threshold=0,
        )

        breaker.record_failure(Exception("test"))

        assert breaker.state == CircuitState.OPEN

    def test_very_short_recovery_timeout(self):
        """Test breaker with very short recovery timeout."""
        breaker = CircuitBreaker(
            name="short_timeout",
            failure_threshold=1,
            recovery_timeout=0.001,  # 1ms
            success_threshold=1,
        )

        breaker.record_failure(Exception("test"))
        time.sleep(0.01)

        # Should transition to half-open
        assert breaker.state == CircuitState.HALF_OPEN

    def test_high_success_threshold(self):
        """Test breaker with high success threshold needs multiple successes."""
        breaker = CircuitBreaker(
            name="high_success",
            failure_threshold=1,
            recovery_timeout=0.01,
            success_threshold=3,
            half_open_max_calls=5,
        )

        breaker.record_failure(Exception("test"))
        time.sleep(0.02)

        # Record successes
        breaker.allow_request()
        breaker.record_success()
        assert breaker.state == CircuitState.HALF_OPEN  # Still half-open

        breaker.allow_request()
        breaker.record_success()
        assert breaker.state == CircuitState.HALF_OPEN  # Still half-open

        breaker.allow_request()
        breaker.record_success()
        assert breaker.state == CircuitState.CLOSED  # Now closed
