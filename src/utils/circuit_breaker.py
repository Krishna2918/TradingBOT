"""
Circuit Breaker Pattern Implementation
======================================

Prevents cascading failures by failing fast when a service is degraded.
Implements the standard circuit breaker pattern with three states:
- CLOSED: Normal operation, requests pass through
- OPEN: Service degraded, requests fail immediately
- HALF_OPEN: Testing if service recovered

Usage:
    from src.utils.circuit_breaker import CircuitBreaker, circuit_breaker

    # As decorator
    @circuit_breaker(failure_threshold=5, recovery_timeout=60)
    def call_external_api():
        ...

    # As context manager
    breaker = CircuitBreaker("broker_api")
    with breaker:
        call_broker()

    # Manual usage
    if breaker.allow_request():
        try:
            result = call_broker()
            breaker.record_success()
        except Exception as e:
            breaker.record_failure()
            raise
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar

logger = logging.getLogger('trading.circuit_breaker')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = auto()     # Normal operation
    OPEN = auto()       # Failing fast
    HALF_OPEN = auto()  # Testing recovery


@dataclass
class CircuitStats:
    """Statistics for a circuit breaker."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rejected_requests: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    state_changes: int = 0


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""

    def __init__(self, name: str, time_until_retry: float):
        self.name = name
        self.time_until_retry = time_until_retry
        super().__init__(
            f"Circuit breaker '{name}' is OPEN. "
            f"Retry in {time_until_retry:.1f}s"
        )


class CircuitBreaker:
    """Circuit breaker implementation.

    Attributes:
        name: Identifier for this circuit breaker
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Seconds to wait before trying half-open
        half_open_max_calls: Max calls allowed in half-open state
        success_threshold: Successes needed in half-open to close circuit
        failure_rate_threshold: Alternative: open if failure rate exceeds this
        window_size: Time window for calculating failure rate (seconds)
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 3,
        success_threshold: int = 2,
        failure_rate_threshold: Optional[float] = None,
        window_size: float = 60.0,
        on_open: Optional[Callable[[str], None]] = None,
        on_close: Optional[Callable[[str], None]] = None,
        on_half_open: Optional[Callable[[str], None]] = None,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.success_threshold = success_threshold
        self.failure_rate_threshold = failure_rate_threshold
        self.window_size = window_size

        # Callbacks
        self._on_open = on_open
        self._on_close = on_close
        self._on_half_open = on_half_open

        # State
        self._state = CircuitState.CLOSED
        self._last_state_change = datetime.utcnow()
        self._half_open_calls = 0
        self._half_open_successes = 0

        # Statistics
        self._stats = CircuitStats()

        # Thread safety
        self._lock = threading.RLock()

        # Sliding window for failure rate calculation
        self._failure_window: list[tuple[datetime, bool]] = []

    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        with self._lock:
            self._check_state_transition()
            return self._state

    @property
    def stats(self) -> CircuitStats:
        """Current statistics."""
        with self._lock:
            return CircuitStats(
                total_requests=self._stats.total_requests,
                successful_requests=self._stats.successful_requests,
                failed_requests=self._stats.failed_requests,
                rejected_requests=self._stats.rejected_requests,
                last_failure_time=self._stats.last_failure_time,
                last_success_time=self._stats.last_success_time,
                consecutive_failures=self._stats.consecutive_failures,
                consecutive_successes=self._stats.consecutive_successes,
                state_changes=self._stats.state_changes,
            )

    def _check_state_transition(self) -> None:
        """Check if state should transition based on timeout."""
        if self._state == CircuitState.OPEN:
            time_in_open = (datetime.utcnow() - self._last_state_change).total_seconds()
            if time_in_open >= self.recovery_timeout:
                self._transition_to(CircuitState.HALF_OPEN)

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        if self._state == new_state:
            return

        old_state = self._state
        self._state = new_state
        self._last_state_change = datetime.utcnow()
        self._stats.state_changes += 1

        logger.info(
            f"Circuit breaker '{self.name}': {old_state.name} -> {new_state.name}"
        )

        # Reset half-open counters
        if new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
            self._half_open_successes = 0
            if self._on_half_open:
                self._on_half_open(self.name)
        elif new_state == CircuitState.OPEN:
            if self._on_open:
                self._on_open(self.name)
        elif new_state == CircuitState.CLOSED:
            self._stats.consecutive_failures = 0
            if self._on_close:
                self._on_close(self.name)

    def _calculate_failure_rate(self) -> float:
        """Calculate failure rate in current window."""
        now = datetime.utcnow()
        cutoff = now - timedelta(seconds=self.window_size)

        # Clean old entries
        self._failure_window = [
            (ts, success) for ts, success in self._failure_window
            if ts > cutoff
        ]

        if not self._failure_window:
            return 0.0

        failures = sum(1 for _, success in self._failure_window if not success)
        return failures / len(self._failure_window)

    def allow_request(self) -> bool:
        """Check if a request should be allowed.

        Returns:
            True if request is allowed, False if circuit is open
        """
        with self._lock:
            self._check_state_transition()

            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                self._stats.rejected_requests += 1
                return False

            # HALF_OPEN: allow limited requests
            if self._half_open_calls < self.half_open_max_calls:
                self._half_open_calls += 1
                return True

            self._stats.rejected_requests += 1
            return False

    def record_success(self) -> None:
        """Record a successful request."""
        with self._lock:
            self._stats.total_requests += 1
            self._stats.successful_requests += 1
            self._stats.last_success_time = datetime.utcnow()
            self._stats.consecutive_successes += 1
            self._stats.consecutive_failures = 0

            # Add to sliding window
            self._failure_window.append((datetime.utcnow(), True))

            if self._state == CircuitState.HALF_OPEN:
                self._half_open_successes += 1
                if self._half_open_successes >= self.success_threshold:
                    self._transition_to(CircuitState.CLOSED)

    def record_failure(self, error: Optional[Exception] = None) -> None:
        """Record a failed request."""
        with self._lock:
            self._stats.total_requests += 1
            self._stats.failed_requests += 1
            self._stats.last_failure_time = datetime.utcnow()
            self._stats.consecutive_failures += 1
            self._stats.consecutive_successes = 0

            # Add to sliding window
            self._failure_window.append((datetime.utcnow(), False))

            if error:
                logger.warning(
                    f"Circuit breaker '{self.name}' recorded failure: {error}"
                )

            # Check if we should open the circuit
            should_open = False

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open reopens the circuit
                should_open = True
            elif self._state == CircuitState.CLOSED:
                # Check threshold
                if self._stats.consecutive_failures >= self.failure_threshold:
                    should_open = True
                # Check failure rate
                elif self.failure_rate_threshold:
                    if self._calculate_failure_rate() >= self.failure_rate_threshold:
                        should_open = True

            if should_open:
                self._transition_to(CircuitState.OPEN)

    def time_until_retry(self) -> float:
        """Seconds until circuit might allow requests again."""
        with self._lock:
            if self._state != CircuitState.OPEN:
                return 0.0

            elapsed = (datetime.utcnow() - self._last_state_change).total_seconds()
            return max(0.0, self.recovery_timeout - elapsed)

    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        with self._lock:
            self._transition_to(CircuitState.CLOSED)
            self._stats = CircuitStats()
            self._failure_window.clear()
            logger.info(f"Circuit breaker '{self.name}' reset")

    def force_open(self) -> None:
        """Force circuit to open state."""
        with self._lock:
            self._transition_to(CircuitState.OPEN)
            logger.warning(f"Circuit breaker '{self.name}' forced OPEN")

    def __enter__(self) -> 'CircuitBreaker':
        """Context manager entry."""
        if not self.allow_request():
            raise CircuitBreakerOpenError(self.name, self.time_until_retry())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Context manager exit."""
        if exc_type is None:
            self.record_success()
        else:
            self.record_failure(exc_val)
        return False  # Don't suppress exceptions


# =============================================================================
# Circuit Breaker Registry
# =============================================================================

class CircuitBreakerRegistry:
    """Global registry of circuit breakers."""

    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()

    def get_or_create(
        self,
        name: str,
        **kwargs,
    ) -> CircuitBreaker:
        """Get existing or create new circuit breaker."""
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name, **kwargs)
            return self._breakers[name]

    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        return self._breakers.get(name)

    def all(self) -> Dict[str, CircuitBreaker]:
        """Get all circuit breakers."""
        return self._breakers.copy()

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()

    def get_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers."""
        status = {}
        for name, breaker in self._breakers.items():
            status[name] = {
                'state': breaker.state.name,
                'consecutive_failures': breaker.stats.consecutive_failures,
                'total_requests': breaker.stats.total_requests,
                'failed_requests': breaker.stats.failed_requests,
                'rejected_requests': breaker.stats.rejected_requests,
                'time_until_retry': breaker.time_until_retry(),
            }
        return status


# Global registry instance
_registry = CircuitBreakerRegistry()


def get_circuit_breaker_registry() -> CircuitBreakerRegistry:
    """Get the global circuit breaker registry."""
    return _registry


# =============================================================================
# Decorator
# =============================================================================

T = TypeVar('T')


def circuit_breaker(
    name: Optional[str] = None,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    half_open_max_calls: int = 3,
    success_threshold: int = 2,
    fallback: Optional[Callable[..., T]] = None,
    exclude_exceptions: tuple[type, ...] = (),
) -> Callable:
    """Decorator to wrap a function with circuit breaker protection.

    Args:
        name: Circuit breaker name (defaults to function name)
        failure_threshold: Failures before opening
        recovery_timeout: Seconds before trying half-open
        half_open_max_calls: Max calls in half-open state
        success_threshold: Successes to close from half-open
        fallback: Function to call when circuit is open
        exclude_exceptions: Exceptions that don't count as failures

    Example:
        @circuit_breaker(failure_threshold=3, recovery_timeout=30)
        def call_api():
            return requests.get("https://api.example.com")

        @circuit_breaker(fallback=lambda: [])
        def get_data():
            return fetch_from_external_service()
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        breaker_name = name or f"{func.__module__}.{func.__name__}"

        breaker = _registry.get_or_create(
            breaker_name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            half_open_max_calls=half_open_max_calls,
            success_threshold=success_threshold,
        )

        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            if not breaker.allow_request():
                if fallback:
                    logger.warning(
                        f"Circuit '{breaker_name}' open, using fallback"
                    )
                    return fallback(*args, **kwargs)
                raise CircuitBreakerOpenError(
                    breaker_name,
                    breaker.time_until_retry()
                )

            try:
                result = func(*args, **kwargs)
                breaker.record_success()
                return result
            except exclude_exceptions:
                # Don't count these as failures
                breaker.record_success()
                raise
            except Exception as e:
                breaker.record_failure(e)
                raise

        # Attach breaker for testing/inspection
        wrapper.circuit_breaker = breaker
        return wrapper

    return decorator


# =============================================================================
# Retry with Circuit Breaker
# =============================================================================

def retry_with_circuit_breaker(
    breaker: CircuitBreaker,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
) -> Callable:
    """Decorator combining retry logic with circuit breaker.

    Args:
        breaker: Circuit breaker to use
        max_retries: Maximum retry attempts
        base_delay: Initial delay between retries
        max_delay: Maximum delay cap
        exponential_base: Base for exponential backoff
        jitter: Add random jitter to delays
    """
    import random

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(max_retries + 1):
                # Check circuit breaker
                if not breaker.allow_request():
                    raise CircuitBreakerOpenError(
                        breaker.name,
                        breaker.time_until_retry()
                    )

                try:
                    result = func(*args, **kwargs)
                    breaker.record_success()
                    return result
                except Exception as e:
                    last_exception = e
                    breaker.record_failure(e)

                    if attempt < max_retries:
                        # Calculate delay
                        delay = min(
                            base_delay * (exponential_base ** attempt),
                            max_delay
                        )
                        if jitter:
                            delay *= (0.5 + random.random())

                        logger.warning(
                            f"Retry {attempt + 1}/{max_retries} for {func.__name__} "
                            f"after {delay:.1f}s: {e}"
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"All {max_retries} retries exhausted for {func.__name__}"
                        )

            raise last_exception

        return wrapper
    return decorator


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    'CircuitState',
    'CircuitStats',
    'CircuitBreaker',
    'CircuitBreakerOpenError',
    'CircuitBreakerRegistry',
    'get_circuit_breaker_registry',
    'circuit_breaker',
    'retry_with_circuit_breaker',
]
