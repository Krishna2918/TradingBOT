"""
Robust retry manager with exponential backoff and intelligent error classification.
"""

import time
import random
import logging
from typing import Optional, Callable, Any, Dict, List
from enum import Enum
import requests
from .interfaces import RetryManager
from .config import CollectionConfig


class ErrorType(Enum):
    """Classification of error types for retry decisions."""
    TEMPORARY = "temporary"      # Network issues, timeouts, temporary API issues
    PERMANENT = "permanent"      # Invalid API key, invalid symbol, quota exceeded
    RATE_LIMIT = "rate_limit"   # Rate limit exceeded
    SYSTEM = "system"           # Disk space, memory issues


class RetryAttempt:
    """Information about a retry attempt."""
    
    def __init__(self, attempt: int, error: Exception, error_type: ErrorType, 
                 delay: float, timestamp: float):
        self.attempt = attempt
        self.error = error
        self.error_type = error_type
        self.delay = delay
        self.timestamp = timestamp
        self.error_message = str(error)


class IntelligentRetryManager(RetryManager):
    """Retry manager with intelligent error classification and exponential backoff."""
    
    def __init__(self, config: CollectionConfig):
        self.config = config
        self.max_retries = config.max_retries
        self.backoff_base = config.retry_backoff_base
        self.max_backoff = 300.0  # 5 minutes maximum
        self.jitter_range = 0.1   # ±10% jitter
        
        self.logger = logging.getLogger(__name__)
        self.retry_history: Dict[str, List[RetryAttempt]] = {}
        
        self.logger.info(f"Initialized retry manager: max_retries={self.max_retries}, "
                        f"backoff_base={self.backoff_base}")
    
    def should_retry(self, error: Exception, attempt: int) -> bool:
        """Determine if an error should be retried."""
        if attempt >= self.max_retries:
            self.logger.warning(f"Max retries ({self.max_retries}) reached for attempt {attempt}")
            return False
        
        error_type = self.classify_error(error)
        
        # Never retry permanent errors
        if error_type == ErrorType.PERMANENT:
            self.logger.info(f"Not retrying permanent error: {error}")
            return False
        
        # Always retry temporary and rate limit errors (within max attempts)
        if error_type in [ErrorType.TEMPORARY, ErrorType.RATE_LIMIT]:
            self.logger.debug(f"Will retry {error_type.value} error (attempt {attempt}): {error}")
            return True
        
        # System errors - retry with caution
        if error_type == ErrorType.SYSTEM:
            self.logger.warning(f"System error detected, will retry (attempt {attempt}): {error}")
            return True
        
        # Default: don't retry unknown errors
        self.logger.warning(f"Unknown error type, not retrying: {error}")
        return False
    
    def get_backoff_delay(self, attempt: int) -> float:
        """Get delay before next retry attempt with exponential backoff and jitter."""
        if attempt <= 0:
            return 0.0
        
        # Exponential backoff: base^attempt
        base_delay = min(self.backoff_base ** attempt, self.max_backoff)
        
        # Add jitter: ±10% random variation
        jitter = random.uniform(-self.jitter_range, self.jitter_range)
        delay = base_delay * (1 + jitter)
        
        # Ensure minimum delay of 1 second
        delay = max(delay, 1.0)
        
        self.logger.debug(f"Calculated backoff delay for attempt {attempt}: {delay:.2f}s "
                         f"(base: {base_delay:.2f}s, jitter: {jitter:.2%})")
        
        return delay
    
    def classify_error(self, error: Exception) -> ErrorType:
        """Classify error as temporary, permanent, rate limit, or system."""
        error_message = str(error).lower()
        
        # Rate limit errors
        if any(phrase in error_message for phrase in [
            "rate limit", "too many requests", "api call frequency",
            "requests per minute", "quota exceeded"
        ]):
            return ErrorType.RATE_LIMIT
        
        # Permanent errors
        if any(phrase in error_message for phrase in [
            "invalid api", "invalid symbol", "unauthorized", "forbidden",
            "not found", "invalid request", "malformed", "bad request"
        ]):
            return ErrorType.PERMANENT
        
        # System errors
        if any(phrase in error_message for phrase in [
            "disk space", "memory", "no space left", "permission denied",
            "file system", "storage"
        ]):
            return ErrorType.SYSTEM
        
        # Network and temporary errors
        if isinstance(error, (requests.exceptions.ConnectionError,
                             requests.exceptions.Timeout,
                             requests.exceptions.RequestException)):
            return ErrorType.TEMPORARY
        
        if any(phrase in error_message for phrase in [
            "timeout", "connection", "network", "temporary", "service unavailable",
            "internal server error", "bad gateway", "gateway timeout"
        ]):
            return ErrorType.TEMPORARY
        
        # Default to temporary for unknown errors (safer to retry)
        self.logger.warning(f"Unknown error type, classifying as temporary: {error}")
        return ErrorType.TEMPORARY
    
    def execute_with_retry(self, operation: Callable[[], Any], 
                          operation_name: str = "operation") -> Any:
        """Execute an operation with retry logic."""
        attempt = 0
        last_error = None
        
        while attempt <= self.max_retries:
            try:
                if attempt > 0:
                    self.logger.info(f"Retrying {operation_name} (attempt {attempt}/{self.max_retries})")
                
                result = operation()
                
                # Success - log and return
                if attempt > 0:
                    self.logger.info(f"{operation_name} succeeded after {attempt} retries")
                
                return result
                
            except Exception as error:
                last_error = error
                attempt += 1
                
                # Check if we should retry
                if not self.should_retry(error, attempt):
                    self.logger.error(f"{operation_name} failed permanently after {attempt-1} retries: {error}")
                    break
                
                # Calculate delay and wait
                if attempt <= self.max_retries:
                    delay = self.get_backoff_delay(attempt)
                    error_type = self.classify_error(error)
                    
                    # Record retry attempt
                    retry_attempt = RetryAttempt(
                        attempt=attempt,
                        error=error,
                        error_type=error_type,
                        delay=delay,
                        timestamp=time.time()
                    )
                    
                    if operation_name not in self.retry_history:
                        self.retry_history[operation_name] = []
                    self.retry_history[operation_name].append(retry_attempt)
                    
                    self.logger.warning(f"{operation_name} failed (attempt {attempt}), "
                                      f"retrying in {delay:.2f}s. Error: {error}")
                    
                    time.sleep(delay)
        
        # All retries exhausted
        self.logger.error(f"{operation_name} failed after {self.max_retries} retries. "
                         f"Final error: {last_error}")
        raise last_error
    
    def get_retry_statistics(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """Get retry statistics for analysis."""
        if operation_name:
            attempts = self.retry_history.get(operation_name, [])
            operations = {operation_name: attempts}
        else:
            operations = self.retry_history
        
        stats = {
            "total_operations": len(operations),
            "total_retry_attempts": sum(len(attempts) for attempts in operations.values()),
            "operations_with_retries": sum(1 for attempts in operations.values() if attempts),
            "error_type_breakdown": {},
            "average_retries_per_operation": 0.0,
            "max_retries_for_operation": 0,
            "most_common_errors": {}
        }
        
        if not operations:
            return stats
        
        # Calculate detailed statistics
        all_attempts = []
        for attempts in operations.values():
            all_attempts.extend(attempts)
        
        if all_attempts:
            # Error type breakdown
            error_types = {}
            error_messages = {}
            
            for attempt in all_attempts:
                error_type = attempt.error_type.value
                error_types[error_type] = error_types.get(error_type, 0) + 1
                
                error_msg = attempt.error_message[:100]  # Truncate long messages
                error_messages[error_msg] = error_messages.get(error_msg, 0) + 1
            
            stats["error_type_breakdown"] = error_types
            stats["most_common_errors"] = dict(sorted(error_messages.items(), 
                                                     key=lambda x: x[1], reverse=True)[:5])
            
            # Calculate averages
            retry_counts = [len(attempts) for attempts in operations.values()]
            stats["average_retries_per_operation"] = sum(retry_counts) / len(retry_counts)
            stats["max_retries_for_operation"] = max(retry_counts) if retry_counts else 0
        
        return stats
    
    def clear_history(self, operation_name: Optional[str] = None) -> None:
        """Clear retry history for analysis."""
        if operation_name:
            self.retry_history.pop(operation_name, None)
            self.logger.debug(f"Cleared retry history for {operation_name}")
        else:
            self.retry_history.clear()
            self.logger.debug("Cleared all retry history")
    
    def get_recent_failures(self, hours: int = 24) -> List[RetryAttempt]:
        """Get recent failures within the specified time window."""
        cutoff_time = time.time() - (hours * 3600)
        recent_failures = []
        
        for attempts in self.retry_history.values():
            for attempt in attempts:
                if attempt.timestamp >= cutoff_time:
                    recent_failures.append(attempt)
        
        # Sort by timestamp (most recent first)
        recent_failures.sort(key=lambda x: x.timestamp, reverse=True)
        return recent_failures