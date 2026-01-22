"""
Centralized Error Handling Framework
=====================================

Provides a hierarchical exception system and error handling utilities
for the TradingBOT system.

Usage:
    from src.utils.error_handler import (
        TradingError, BrokerError, APIError,
        handle_error, ErrorSeverity
    )

    try:
        execute_order(...)
    except BrokerError as e:
        handle_error(e, context={'order_id': order.id})
"""

from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from functools import wraps
from typing import Any, Callable, Optional, Type, TypeVar

logger = logging.getLogger('trading.errors')


class ErrorSeverity(Enum):
    """Severity levels for error classification."""
    DEBUG = auto()      # Non-critical, informational
    WARNING = auto()    # Potential issue, continue operation
    ERROR = auto()      # Operation failed, but system continues
    CRITICAL = auto()   # System-level failure, may require shutdown
    FATAL = auto()      # Unrecoverable, immediate shutdown required


class ErrorCategory(Enum):
    """Categories for error classification and routing."""
    BROKER = "broker"           # Broker/exchange related
    API = "api"                 # External API failures
    DATA = "data"               # Data quality/availability issues
    EXECUTION = "execution"     # Order execution failures
    POSITION = "position"       # Position management issues
    RISK = "risk"               # Risk limit breaches
    NETWORK = "network"         # Network/connectivity issues
    VALIDATION = "validation"   # Input validation failures
    CONFIGURATION = "config"    # Configuration errors
    SYSTEM = "system"           # System-level errors


# =============================================================================
# Base Exception Hierarchy
# =============================================================================

class TradingError(Exception):
    """Base exception for all trading system errors.

    Attributes:
        message: Human-readable error description
        severity: Error severity level
        category: Error category for routing
        recoverable: Whether the error is recoverable
        context: Additional context about the error
        original_error: Original exception if wrapping another error
    """

    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        recoverable: bool = True,
        context: Optional[dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.category = category
        self.recoverable = recoverable
        self.context = context or {}
        self.original_error = original_error
        self.timestamp = datetime.utcnow()

    def __str__(self) -> str:
        parts = [f"[{self.severity.name}] {self.category.value}: {self.message}"]
        if self.context:
            parts.append(f" | Context: {self.context}")
        return "".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Convert error to dictionary for logging/serialization."""
        return {
            'type': self.__class__.__name__,
            'message': self.message,
            'severity': self.severity.name,
            'category': self.category.value,
            'recoverable': self.recoverable,
            'context': self.context,
            'timestamp': self.timestamp.isoformat(),
            'original_error': str(self.original_error) if self.original_error else None,
        }


# =============================================================================
# Broker Errors
# =============================================================================

class BrokerError(TradingError):
    """Base exception for broker-related errors."""

    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        recoverable: bool = True,
        context: Optional[dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            severity=severity,
            category=ErrorCategory.BROKER,
            recoverable=recoverable,
            context=context,
            original_error=original_error,
        )


class BrokerConnectionError(BrokerError):
    """Failed to connect to broker."""

    def __init__(self, message: str = "Failed to connect to broker", **kwargs):
        super().__init__(message, severity=ErrorSeverity.CRITICAL, **kwargs)


class BrokerAuthenticationError(BrokerError):
    """Broker authentication failed."""

    def __init__(self, message: str = "Broker authentication failed", **kwargs):
        super().__init__(message, severity=ErrorSeverity.CRITICAL, recoverable=False, **kwargs)


class BrokerRateLimitError(BrokerError):
    """Broker rate limit exceeded."""

    def __init__(self, message: str = "Broker rate limit exceeded", **kwargs):
        super().__init__(message, severity=ErrorSeverity.WARNING, **kwargs)


class BrokerTimeoutError(BrokerError):
    """Broker request timed out."""

    def __init__(self, message: str = "Broker request timed out", **kwargs):
        super().__init__(message, severity=ErrorSeverity.ERROR, **kwargs)


# =============================================================================
# API Errors
# =============================================================================

class APIError(TradingError):
    """Base exception for external API errors."""

    def __init__(
        self,
        message: str,
        api_name: str = "unknown",
        status_code: Optional[int] = None,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        recoverable: bool = True,
        context: Optional[dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ):
        ctx = context or {}
        ctx['api_name'] = api_name
        if status_code:
            ctx['status_code'] = status_code
        super().__init__(
            message=message,
            severity=severity,
            category=ErrorCategory.API,
            recoverable=recoverable,
            context=ctx,
            original_error=original_error,
        )
        self.api_name = api_name
        self.status_code = status_code


class APIRateLimitError(APIError):
    """API rate limit exceeded."""

    def __init__(self, api_name: str, retry_after: Optional[int] = None, **kwargs):
        ctx = kwargs.pop('context', {}) or {}
        if retry_after:
            ctx['retry_after_seconds'] = retry_after
        super().__init__(
            message=f"{api_name} rate limit exceeded",
            api_name=api_name,
            severity=ErrorSeverity.WARNING,
            context=ctx,
            **kwargs,
        )
        self.retry_after = retry_after


class APITimeoutError(APIError):
    """API request timed out."""

    def __init__(self, api_name: str, timeout_seconds: Optional[float] = None, **kwargs):
        ctx = kwargs.pop('context', {}) or {}
        if timeout_seconds:
            ctx['timeout_seconds'] = timeout_seconds
        super().__init__(
            message=f"{api_name} request timed out",
            api_name=api_name,
            context=ctx,
            **kwargs,
        )


class APIResponseError(APIError):
    """Invalid or unexpected API response."""

    def __init__(self, api_name: str, message: str = "Invalid API response", **kwargs):
        super().__init__(message=message, api_name=api_name, **kwargs)


# =============================================================================
# Execution Errors
# =============================================================================

class ExecutionError(TradingError):
    """Base exception for order execution errors."""

    def __init__(
        self,
        message: str,
        order_id: Optional[str] = None,
        symbol: Optional[str] = None,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        recoverable: bool = True,
        context: Optional[dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ):
        ctx = context or {}
        if order_id:
            ctx['order_id'] = order_id
        if symbol:
            ctx['symbol'] = symbol
        super().__init__(
            message=message,
            severity=severity,
            category=ErrorCategory.EXECUTION,
            recoverable=recoverable,
            context=ctx,
            original_error=original_error,
        )
        self.order_id = order_id
        self.symbol = symbol


class OrderRejectedError(ExecutionError):
    """Order was rejected by broker."""

    def __init__(self, reason: str, **kwargs):
        super().__init__(
            message=f"Order rejected: {reason}",
            recoverable=False,
            **kwargs,
        )
        self.reason = reason


class OrderTimeoutError(ExecutionError):
    """Order execution timed out."""

    def __init__(self, **kwargs):
        super().__init__(message="Order execution timed out", **kwargs)


class InsufficientFundsError(ExecutionError):
    """Insufficient funds to execute order."""

    def __init__(self, required: float, available: float, **kwargs):
        ctx = kwargs.pop('context', {}) or {}
        ctx['required_funds'] = required
        ctx['available_funds'] = available
        super().__init__(
            message=f"Insufficient funds: need {required:.2f}, have {available:.2f}",
            recoverable=False,
            context=ctx,
            **kwargs,
        )


class PositionLimitError(ExecutionError):
    """Position limit would be exceeded."""

    def __init__(self, message: str = "Position limit exceeded", **kwargs):
        super().__init__(message=message, recoverable=False, **kwargs)


# =============================================================================
# Position Errors
# =============================================================================

class PositionError(TradingError):
    """Base exception for position management errors."""

    def __init__(
        self,
        message: str,
        symbol: Optional[str] = None,
        position_id: Optional[str] = None,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        recoverable: bool = True,
        context: Optional[dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ):
        ctx = context or {}
        if symbol:
            ctx['symbol'] = symbol
        if position_id:
            ctx['position_id'] = position_id
        super().__init__(
            message=message,
            severity=severity,
            category=ErrorCategory.POSITION,
            recoverable=recoverable,
            context=ctx,
            original_error=original_error,
        )


class PositionNotFoundError(PositionError):
    """Position not found."""

    def __init__(self, symbol: str, **kwargs):
        super().__init__(
            message=f"Position not found for {symbol}",
            symbol=symbol,
            **kwargs,
        )


class PositionSyncError(PositionError):
    """Position out of sync with broker."""

    def __init__(self, symbol: str, local_qty: float, broker_qty: float, **kwargs):
        ctx = kwargs.pop('context', {}) or {}
        ctx['local_quantity'] = local_qty
        ctx['broker_quantity'] = broker_qty
        super().__init__(
            message=f"Position sync mismatch for {symbol}: local={local_qty}, broker={broker_qty}",
            symbol=symbol,
            severity=ErrorSeverity.CRITICAL,
            context=ctx,
            **kwargs,
        )


# =============================================================================
# Risk Errors
# =============================================================================

class RiskError(TradingError):
    """Base exception for risk-related errors."""

    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.WARNING,
        recoverable: bool = True,
        context: Optional[dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            severity=severity,
            category=ErrorCategory.RISK,
            recoverable=recoverable,
            context=context,
            original_error=original_error,
        )


class RiskLimitBreachedError(RiskError):
    """Risk limit has been breached."""

    def __init__(self, limit_name: str, limit_value: float, current_value: float, **kwargs):
        ctx = kwargs.pop('context', {}) or {}
        ctx['limit_name'] = limit_name
        ctx['limit_value'] = limit_value
        ctx['current_value'] = current_value
        super().__init__(
            message=f"Risk limit breached: {limit_name} ({current_value:.2%} > {limit_value:.2%})",
            severity=ErrorSeverity.CRITICAL,
            recoverable=False,
            context=ctx,
            **kwargs,
        )


class MaxDrawdownError(RiskError):
    """Maximum drawdown limit breached."""

    def __init__(self, current_drawdown: float, max_allowed: float, **kwargs):
        ctx = kwargs.pop('context', {}) or {}
        ctx['current_drawdown'] = current_drawdown
        ctx['max_allowed'] = max_allowed
        super().__init__(
            message=f"Max drawdown breached: {current_drawdown:.2%} > {max_allowed:.2%}",
            severity=ErrorSeverity.FATAL,
            recoverable=False,
            context=ctx,
            **kwargs,
        )


class DailyLossLimitError(RiskError):
    """Daily loss limit breached."""

    def __init__(self, current_loss: float, max_allowed: float, **kwargs):
        ctx = kwargs.pop('context', {}) or {}
        ctx['current_loss'] = current_loss
        ctx['max_allowed'] = max_allowed
        super().__init__(
            message=f"Daily loss limit breached: {current_loss:.2%} > {max_allowed:.2%}",
            severity=ErrorSeverity.CRITICAL,
            recoverable=False,
            context=ctx,
            **kwargs,
        )


# =============================================================================
# Data Errors
# =============================================================================

class DataError(TradingError):
    """Base exception for data-related errors."""

    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.WARNING,
        recoverable: bool = True,
        context: Optional[dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            severity=severity,
            category=ErrorCategory.DATA,
            recoverable=recoverable,
            context=context,
            original_error=original_error,
        )


class DataStaleError(DataError):
    """Data is stale or outdated."""

    def __init__(self, data_type: str, age_seconds: float, max_age_seconds: float, **kwargs):
        ctx = kwargs.pop('context', {}) or {}
        ctx['data_type'] = data_type
        ctx['age_seconds'] = age_seconds
        ctx['max_age_seconds'] = max_age_seconds
        super().__init__(
            message=f"Stale {data_type} data: {age_seconds:.0f}s old (max: {max_age_seconds:.0f}s)",
            context=ctx,
            **kwargs,
        )


class DataValidationError(DataError):
    """Data failed validation."""

    def __init__(self, field: str, reason: str, **kwargs):
        ctx = kwargs.pop('context', {}) or {}
        ctx['field'] = field
        ctx['reason'] = reason
        super().__init__(
            message=f"Data validation failed for '{field}': {reason}",
            context=ctx,
            **kwargs,
        )


# =============================================================================
# Validation Errors
# =============================================================================

class ValidationError(TradingError):
    """Input validation error."""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Any = None,
        severity: ErrorSeverity = ErrorSeverity.WARNING,
        context: Optional[dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ):
        ctx = context or {}
        if field:
            ctx['field'] = field
        if value is not None:
            ctx['value'] = str(value)[:100]  # Truncate long values
        super().__init__(
            message=message,
            severity=severity,
            category=ErrorCategory.VALIDATION,
            recoverable=True,
            context=ctx,
            original_error=original_error,
        )


class SymbolValidationError(ValidationError):
    """Invalid symbol format."""

    def __init__(self, symbol: str, **kwargs):
        super().__init__(
            message=f"Invalid symbol format: '{symbol}'",
            field='symbol',
            value=symbol,
            **kwargs,
        )


# =============================================================================
# Configuration Errors
# =============================================================================

class ConfigurationError(TradingError):
    """Configuration error."""

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        severity: ErrorSeverity = ErrorSeverity.CRITICAL,
        context: Optional[dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ):
        ctx = context or {}
        if config_key:
            ctx['config_key'] = config_key
        super().__init__(
            message=message,
            severity=severity,
            category=ErrorCategory.CONFIGURATION,
            recoverable=False,
            context=ctx,
            original_error=original_error,
        )


# =============================================================================
# Error Context and Tracking
# =============================================================================

@dataclass
class ErrorRecord:
    """Record of an error occurrence for tracking."""
    error_type: str
    message: str
    severity: ErrorSeverity
    category: ErrorCategory
    timestamp: datetime
    context: dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None


class ErrorTracker:
    """Tracks error occurrences for monitoring and alerting."""

    def __init__(self, max_records: int = 1000):
        self._records: list[ErrorRecord] = []
        self._max_records = max_records
        self._error_counts: dict[str, int] = {}

    def record(self, error: TradingError, include_trace: bool = True) -> None:
        """Record an error occurrence."""
        record = ErrorRecord(
            error_type=error.__class__.__name__,
            message=error.message,
            severity=error.severity,
            category=error.category,
            timestamp=error.timestamp,
            context=error.context.copy(),
            stack_trace=traceback.format_exc() if include_trace else None,
        )
        self._records.append(record)

        # Update counts
        self._error_counts[record.error_type] = \
            self._error_counts.get(record.error_type, 0) + 1

        # Trim old records
        if len(self._records) > self._max_records:
            self._records = self._records[-self._max_records:]

    def get_recent(self, count: int = 10) -> list[ErrorRecord]:
        """Get most recent error records."""
        return self._records[-count:]

    def get_counts(self) -> dict[str, int]:
        """Get error counts by type."""
        return self._error_counts.copy()

    def get_by_severity(self, severity: ErrorSeverity) -> list[ErrorRecord]:
        """Get errors by severity level."""
        return [r for r in self._records if r.severity == severity]


# Global error tracker instance
_error_tracker = ErrorTracker()


def get_error_tracker() -> ErrorTracker:
    """Get the global error tracker instance."""
    return _error_tracker


# =============================================================================
# Error Handling Utilities
# =============================================================================

def handle_error(
    error: Exception,
    context: Optional[dict[str, Any]] = None,
    reraise: bool = False,
    log_level: Optional[int] = None,
) -> None:
    """Handle an error with appropriate logging and tracking.

    Args:
        error: The exception to handle
        context: Additional context to add to the error
        reraise: Whether to re-raise the error after handling
        log_level: Override log level (defaults to error severity)
    """
    # Wrap non-TradingError exceptions
    if not isinstance(error, TradingError):
        error = TradingError(
            message=str(error),
            severity=ErrorSeverity.ERROR,
            original_error=error,
            context=context,
        )
    elif context:
        error.context.update(context)

    # Determine log level
    if log_level is None:
        level_map = {
            ErrorSeverity.DEBUG: logging.DEBUG,
            ErrorSeverity.WARNING: logging.WARNING,
            ErrorSeverity.ERROR: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL,
            ErrorSeverity.FATAL: logging.CRITICAL,
        }
        log_level = level_map.get(error.severity, logging.ERROR)

    # Log the error
    logger.log(log_level, str(error), exc_info=error.original_error or error)

    # Track the error
    _error_tracker.record(error)

    if reraise:
        raise error


T = TypeVar('T')


def with_error_handling(
    default_return: T = None,
    error_types: tuple[Type[Exception], ...] = (Exception,),
    reraise_types: tuple[Type[Exception], ...] = (),
    context_func: Optional[Callable[..., dict[str, Any]]] = None,
) -> Callable:
    """Decorator for standardized error handling.

    Args:
        default_return: Value to return on error (if not reraising)
        error_types: Exception types to catch
        reraise_types: Exception types to re-raise after logging
        context_func: Function to generate context from args/kwargs

    Example:
        @with_error_handling(default_return=[], error_types=(APIError,))
        def fetch_data(symbol: str):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except reraise_types as e:
                context = context_func(*args, **kwargs) if context_func else {}
                handle_error(e, context=context, reraise=True)
            except error_types as e:
                context = context_func(*args, **kwargs) if context_func else {}
                context['function'] = func.__name__
                handle_error(e, context=context, reraise=False)
                return default_return
        return wrapper
    return decorator


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    # Severity and Category
    'ErrorSeverity',
    'ErrorCategory',

    # Base exceptions
    'TradingError',

    # Broker errors
    'BrokerError',
    'BrokerConnectionError',
    'BrokerAuthenticationError',
    'BrokerRateLimitError',
    'BrokerTimeoutError',

    # API errors
    'APIError',
    'APIRateLimitError',
    'APITimeoutError',
    'APIResponseError',

    # Execution errors
    'ExecutionError',
    'OrderRejectedError',
    'OrderTimeoutError',
    'InsufficientFundsError',
    'PositionLimitError',

    # Position errors
    'PositionError',
    'PositionNotFoundError',
    'PositionSyncError',

    # Risk errors
    'RiskError',
    'RiskLimitBreachedError',
    'MaxDrawdownError',
    'DailyLossLimitError',

    # Data errors
    'DataError',
    'DataStaleError',
    'DataValidationError',

    # Validation errors
    'ValidationError',
    'SymbolValidationError',

    # Configuration errors
    'ConfigurationError',

    # Tracking and handling
    'ErrorRecord',
    'ErrorTracker',
    'get_error_tracker',
    'handle_error',
    'with_error_handling',
]
