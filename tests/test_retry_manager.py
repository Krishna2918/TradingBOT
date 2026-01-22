"""
Tests for the intelligent retry manager.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
import requests

from src.adaptive_data_collection.retry_manager import (
    IntelligentRetryManager, ErrorType, RetryAttempt
)
from src.adaptive_data_collection.config import CollectionConfig


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    config = Mock(spec=CollectionConfig)
    config.max_retries = 3
    config.retry_backoff_base = 2.0
    return config


@pytest.fixture
def retry_manager(mock_config):
    """Create retry manager for testing."""
    return IntelligentRetryManager(mock_config)


def test_retry_manager_initialization(mock_config):
    """Test retry manager initialization."""
    manager = IntelligentRetryManager(mock_config)
    
    assert manager.config == mock_config
    assert manager.max_retries == 3
    assert manager.backoff_base == 2.0
    assert manager.max_backoff == 300.0
    assert manager.jitter_range == 0.1
    assert manager.retry_history == {}


def test_error_classification_rate_limit(retry_manager):
    """Test classification of rate limit errors."""
    errors = [
        ValueError("Rate limit exceeded"),
        ValueError("Too many requests"),
        ValueError("API call frequency exceeded"),
        ValueError("Requests per minute limit reached"),
        ValueError("Quota exceeded")
    ]
    
    for error in errors:
        assert retry_manager.classify_error(error) == ErrorType.RATE_LIMIT


def test_error_classification_permanent(retry_manager):
    """Test classification of permanent errors."""
    errors = [
        ValueError("Invalid API key"),
        ValueError("Invalid symbol"),
        ValueError("Unauthorized access"),
        ValueError("Forbidden request"),
        ValueError("Not found"),
        ValueError("Bad request format")
    ]
    
    for error in errors:
        assert retry_manager.classify_error(error) == ErrorType.PERMANENT


def test_error_classification_system(retry_manager):
    """Test classification of system errors."""
    errors = [
        ValueError("No space left on device"),
        ValueError("Memory allocation failed"),
        ValueError("Permission denied"),
        ValueError("File system error")
    ]
    
    for error in errors:
        assert retry_manager.classify_error(error) == ErrorType.SYSTEM


def test_error_classification_temporary(retry_manager):
    """Test classification of temporary errors."""
    errors = [
        requests.exceptions.ConnectionError("Connection failed"),
        requests.exceptions.Timeout("Request timeout"),
        ValueError("Network timeout"),
        ValueError("Service unavailable"),
        ValueError("Internal server error"),
        ValueError("Bad gateway")
    ]
    
    for error in errors:
        assert retry_manager.classify_error(error) == ErrorType.TEMPORARY


def test_error_classification_unknown(retry_manager):
    """Test classification of unknown errors defaults to temporary."""
    error = ValueError("Some unknown error")
    assert retry_manager.classify_error(error) == ErrorType.TEMPORARY


def test_should_retry_permanent_error(retry_manager):
    """Test that permanent errors are not retried."""
    error = ValueError("Invalid API key")
    assert retry_manager.should_retry(error, 1) is False


def test_should_retry_max_attempts(retry_manager):
    """Test that max attempts are respected."""
    error = ValueError("Temporary error")
    assert retry_manager.should_retry(error, 3) is False  # max_retries = 3
    assert retry_manager.should_retry(error, 4) is False


def test_should_retry_temporary_error(retry_manager):
    """Test that temporary errors are retried."""
    error = ValueError("Network timeout")
    assert retry_manager.should_retry(error, 1) is True
    assert retry_manager.should_retry(error, 2) is True


def test_should_retry_rate_limit_error(retry_manager):
    """Test that rate limit errors are retried."""
    error = ValueError("Rate limit exceeded")
    assert retry_manager.should_retry(error, 1) is True
    assert retry_manager.should_retry(error, 2) is True


def test_backoff_delay_calculation(retry_manager):
    """Test exponential backoff delay calculation."""
    # Test exponential growth
    delay1 = retry_manager.get_backoff_delay(1)
    delay2 = retry_manager.get_backoff_delay(2)
    delay3 = retry_manager.get_backoff_delay(3)
    
    # Should be roughly exponential (allowing for jitter)
    assert 1.0 <= delay1 <= 3.0  # 2^1 ± jitter
    assert 3.0 <= delay2 <= 5.0  # 2^2 ± jitter
    assert 7.0 <= delay3 <= 9.0  # 2^3 ± jitter
    
    # Test minimum delay
    assert retry_manager.get_backoff_delay(0) == 0.0
    
    # Test maximum delay cap
    with patch.object(retry_manager, 'max_backoff', 10.0):
        large_delay = retry_manager.get_backoff_delay(10)
        assert large_delay <= 11.0  # max_backoff + jitter


def test_execute_with_retry_success_first_attempt(retry_manager):
    """Test successful operation on first attempt."""
    mock_operation = Mock(return_value="success")
    
    result = retry_manager.execute_with_retry(mock_operation, "test_op")
    
    assert result == "success"
    mock_operation.assert_called_once()
    assert "test_op" not in retry_manager.retry_history


def test_execute_with_retry_success_after_retries(retry_manager):
    """Test successful operation after retries."""
    mock_operation = Mock(side_effect=[
        ValueError("Temporary error"),
        ValueError("Another temporary error"),
        "success"
    ])
    
    with patch('time.sleep'):  # Mock sleep to speed up test
        result = retry_manager.execute_with_retry(mock_operation, "test_op")
    
    assert result == "success"
    assert mock_operation.call_count == 3
    assert "test_op" in retry_manager.retry_history
    assert len(retry_manager.retry_history["test_op"]) == 2  # 2 failed attempts


def test_execute_with_retry_permanent_failure(retry_manager):
    """Test permanent failure stops retries immediately."""
    mock_operation = Mock(side_effect=ValueError("Invalid API key"))
    
    with pytest.raises(ValueError, match="Invalid API key"):
        retry_manager.execute_with_retry(mock_operation, "test_op")
    
    mock_operation.assert_called_once()  # No retries for permanent errors


def test_execute_with_retry_max_retries_exceeded(retry_manager):
    """Test that max retries are respected."""
    mock_operation = Mock(side_effect=ValueError("Temporary error"))
    
    with patch('time.sleep'):  # Mock sleep to speed up test
        with pytest.raises(ValueError, match="Temporary error"):
            retry_manager.execute_with_retry(mock_operation, "test_op")
    
    # Should be called max_retries times (initial + 2 retries, since max_retries=3 means 3 total attempts)
    assert mock_operation.call_count == 3
    assert len(retry_manager.retry_history["test_op"]) == 2  # 2 retry attempts after initial failure


def test_retry_history_recording(retry_manager):
    """Test that retry attempts are properly recorded."""
    mock_operation = Mock(side_effect=[
        ValueError("Network timeout"),
        "success"
    ])
    
    with patch('time.sleep'), patch('time.time', return_value=1000.0):
        retry_manager.execute_with_retry(mock_operation, "test_op")
    
    history = retry_manager.retry_history["test_op"]
    assert len(history) == 1
    
    attempt = history[0]
    assert attempt.attempt == 1
    assert "Network timeout" in attempt.error_message
    assert attempt.error_type == ErrorType.TEMPORARY
    assert attempt.timestamp == 1000.0
    assert attempt.delay > 0


def test_get_retry_statistics_empty(retry_manager):
    """Test retry statistics with no history."""
    stats = retry_manager.get_retry_statistics()
    
    assert stats["total_operations"] == 0
    assert stats["total_retry_attempts"] == 0
    assert stats["operations_with_retries"] == 0
    assert stats["average_retries_per_operation"] == 0.0
    assert stats["max_retries_for_operation"] == 0


def test_get_retry_statistics_with_data(retry_manager):
    """Test retry statistics with actual data."""
    # Simulate some retry history
    retry_manager.retry_history = {
        "op1": [
            RetryAttempt(1, ValueError("Error 1"), ErrorType.TEMPORARY, 2.0, 1000.0),
            RetryAttempt(2, ValueError("Error 2"), ErrorType.RATE_LIMIT, 4.0, 1002.0)
        ],
        "op2": [
            RetryAttempt(1, ValueError("Error 3"), ErrorType.TEMPORARY, 2.0, 1010.0)
        ],
        "op3": []  # No retries
    }
    
    stats = retry_manager.get_retry_statistics()
    
    assert stats["total_operations"] == 3
    assert stats["total_retry_attempts"] == 3
    assert stats["operations_with_retries"] == 2
    assert stats["average_retries_per_operation"] == 1.0  # (2+1+0)/3
    assert stats["max_retries_for_operation"] == 2
    
    # Check error type breakdown
    assert stats["error_type_breakdown"]["temporary"] == 2
    assert stats["error_type_breakdown"]["rate_limit"] == 1


def test_get_retry_statistics_specific_operation(retry_manager):
    """Test retry statistics for a specific operation."""
    retry_manager.retry_history = {
        "op1": [RetryAttempt(1, ValueError("Error"), ErrorType.TEMPORARY, 2.0, 1000.0)],
        "op2": [RetryAttempt(1, ValueError("Error"), ErrorType.RATE_LIMIT, 2.0, 1000.0)]
    }
    
    stats = retry_manager.get_retry_statistics("op1")
    
    assert stats["total_operations"] == 1
    assert stats["total_retry_attempts"] == 1
    assert stats["error_type_breakdown"]["temporary"] == 1
    assert "rate_limit" not in stats["error_type_breakdown"]


def test_clear_history(retry_manager):
    """Test clearing retry history."""
    retry_manager.retry_history = {
        "op1": [RetryAttempt(1, ValueError("Error"), ErrorType.TEMPORARY, 2.0, 1000.0)],
        "op2": [RetryAttempt(1, ValueError("Error"), ErrorType.RATE_LIMIT, 2.0, 1000.0)]
    }
    
    # Clear specific operation
    retry_manager.clear_history("op1")
    assert "op1" not in retry_manager.retry_history
    assert "op2" in retry_manager.retry_history
    
    # Clear all history
    retry_manager.clear_history()
    assert retry_manager.retry_history == {}


def test_get_recent_failures(retry_manager):
    """Test getting recent failures within time window."""
    current_time = time.time()
    
    retry_manager.retry_history = {
        "op1": [
            RetryAttempt(1, ValueError("Recent error"), ErrorType.TEMPORARY, 2.0, current_time - 3600),  # 1 hour ago
            RetryAttempt(2, ValueError("Old error"), ErrorType.TEMPORARY, 4.0, current_time - 48*3600)   # 48 hours ago
        ]
    }
    
    # Get failures from last 24 hours
    recent = retry_manager.get_recent_failures(24)
    
    assert len(recent) == 1
    assert "Recent error" in recent[0].error_message
    
    # Get failures from last 72 hours
    all_failures = retry_manager.get_recent_failures(72)
    assert len(all_failures) == 2


if __name__ == "__main__":
    pytest.main([__file__])