"""
Tests for rate limiter implementations.
"""

import pytest
import time
from unittest.mock import patch, Mock
import threading

from src.adaptive_data_collection.rate_limiter import TokenBucketRateLimiter, ExponentialBackoffRateLimiter


def test_token_bucket_initialization():
    """Test token bucket rate limiter initialization."""
    limiter = TokenBucketRateLimiter(rpm=60)
    
    assert limiter.rpm == 60
    assert limiter.tokens_per_second == 1.0
    assert limiter.burst_capacity == 15  # 60/4
    assert limiter.tokens == 15.0


def test_token_bucket_custom_burst():
    """Test token bucket with custom burst capacity."""
    limiter = TokenBucketRateLimiter(rpm=60, burst_capacity=10)
    
    assert limiter.burst_capacity == 10
    assert limiter.tokens == 10.0


def test_token_bucket_acquire_with_tokens():
    """Test acquiring tokens when available."""
    limiter = TokenBucketRateLimiter(rpm=60)
    initial_tokens = limiter.tokens
    
    limiter.acquire()
    
    assert limiter.tokens == initial_tokens - 1.0


def test_token_bucket_acquire_without_tokens():
    """Test acquiring tokens when none available."""
    limiter = TokenBucketRateLimiter(rpm=60)
    limiter.tokens = 0.0  # No tokens available
    
    with patch('src.adaptive_data_collection.rate_limiter.time.sleep') as mock_sleep:
        # Mock refill to add tokens after first sleep
        call_count = [0]
        def mock_refill():
            call_count[0] += 1
            if call_count[0] > 1:  # After first wait, add tokens
                limiter.tokens = 1.0
        
        with patch.object(limiter, '_refill_tokens', side_effect=mock_refill):
            limiter.acquire()
            mock_sleep.assert_called_once()


def test_token_bucket_refill():
    """Test token refill mechanism."""
    limiter = TokenBucketRateLimiter(rpm=60)  # 1 token per second
    limiter.tokens = 0.0
    
    with patch('src.adaptive_data_collection.rate_limiter.time.time') as mock_time:
        # Simulate 2 seconds passing (called twice: once for current_time, once for last_refill)
        mock_time.side_effect = [2, 2]
        limiter.last_refill = 0
        
        limiter._refill_tokens()
        
        assert limiter.tokens == 2.0  # Should have refilled 2 tokens


def test_token_bucket_refill_cap():
    """Test token refill respects burst capacity."""
    limiter = TokenBucketRateLimiter(rpm=60, burst_capacity=5)
    limiter.tokens = 0.0
    
    with patch('src.adaptive_data_collection.rate_limiter.time.time') as mock_time:
        # Simulate 10 seconds passing (would add 10 tokens, called twice)
        mock_time.side_effect = [10, 10]
        limiter.last_refill = 0
        
        limiter._refill_tokens()
        
        assert limiter.tokens == 5.0  # Should be capped at burst capacity


def test_token_bucket_get_wait_time():
    """Test wait time calculation."""
    limiter = TokenBucketRateLimiter(rpm=60)  # 1 token per second
    
    # With tokens available
    limiter.tokens = 2.0
    assert limiter.get_wait_time() == 0.0
    
    # Without tokens
    limiter.tokens = 0.5
    wait_time = limiter.get_wait_time()
    assert wait_time == 0.5  # (1.0 - 0.5) / 1.0


def test_token_bucket_update_rate_limit():
    """Test dynamic rate limit updates."""
    limiter = TokenBucketRateLimiter(rpm=60)
    initial_tokens = limiter.tokens
    
    limiter.update_rate_limit(120)  # Double the rate
    
    assert limiter.rpm == 120
    assert limiter.tokens_per_second == 2.0
    assert limiter.burst_capacity == 30
    assert limiter.tokens == initial_tokens * 2  # Tokens scaled proportionally


def test_token_bucket_thread_safety():
    """Test thread safety of token bucket."""
    limiter = TokenBucketRateLimiter(rpm=60)
    results = []
    
    def acquire_token():
        limiter.acquire()
        results.append(threading.current_thread().ident)
    
    # Create multiple threads
    threads = [threading.Thread(target=acquire_token) for _ in range(5)]
    
    # Start all threads
    for thread in threads:
        thread.start()
    
    # Wait for all threads
    for thread in threads:
        thread.join()
    
    # All threads should have completed
    assert len(results) == 5


def test_exponential_backoff_initialization():
    """Test exponential backoff rate limiter initialization."""
    limiter = ExponentialBackoffRateLimiter(base_rpm=60)
    
    assert limiter.base_limiter.rpm == 60
    assert limiter.backoff_base == 2.0
    assert limiter.max_backoff == 300.0
    assert limiter.consecutive_failures == 0


def test_exponential_backoff_no_failures():
    """Test backoff limiter with no failures."""
    limiter = ExponentialBackoffRateLimiter(base_rpm=60)
    
    with patch.object(limiter.base_limiter, 'acquire') as mock_acquire:
        limiter.acquire()
        mock_acquire.assert_called_once()


def test_exponential_backoff_with_failures():
    """Test backoff limiter with failures."""
    limiter = ExponentialBackoffRateLimiter(base_rpm=60, backoff_base=2.0)
    
    # Record a failure
    limiter.record_failure()
    assert limiter.consecutive_failures == 1
    
    with patch('time.time', return_value=0), patch('time.sleep') as mock_sleep:
        limiter.last_failure_time = 0
        
        with patch.object(limiter.base_limiter, 'acquire'):
            limiter.acquire()
            mock_sleep.assert_called_once()
            # Should wait for 2^1 = 2 seconds


def test_exponential_backoff_record_success():
    """Test recording success resets backoff."""
    limiter = ExponentialBackoffRateLimiter(base_rpm=60)
    
    # Record failures
    limiter.record_failure()
    limiter.record_failure()
    assert limiter.consecutive_failures == 2
    
    # Record success
    limiter.record_success()
    assert limiter.consecutive_failures == 0
    assert limiter.last_failure_time == 0.0


def test_exponential_backoff_max_backoff():
    """Test maximum backoff limit."""
    limiter = ExponentialBackoffRateLimiter(base_rpm=60, backoff_base=2.0, max_backoff=10.0)
    
    # Record many failures to exceed max backoff
    for _ in range(10):
        limiter.record_failure()
    
    # Calculate expected backoff (should be capped at max_backoff)
    expected_backoff = min(2.0 ** 10, 10.0)
    assert expected_backoff == 10.0
    
    with patch('time.time', return_value=0):
        limiter.last_failure_time = 0
        wait_time = limiter.get_wait_time()
        assert wait_time <= 10.0


def test_exponential_backoff_get_status():
    """Test status reporting for exponential backoff limiter."""
    limiter = ExponentialBackoffRateLimiter(base_rpm=60)
    
    # Mock the base limiter's get_status to avoid any potential waits
    with patch.object(limiter.base_limiter, 'get_status', return_value={'rpm': 60, 'tokens': 15.0}):
        with patch.object(limiter, 'get_wait_time', return_value=0.0):
            status = limiter.get_status()
    
    assert 'rpm' in status
    assert 'consecutive_failures' in status
    assert 'backoff_base' in status
    assert 'max_backoff' in status
    assert status['consecutive_failures'] == 0
    assert status['backoff_base'] == 2.0


def test_exponential_backoff_update_rate_limit():
    """Test updating rate limit in exponential backoff limiter."""
    limiter = ExponentialBackoffRateLimiter(base_rpm=60)
    
    limiter.update_rate_limit(120)
    
    assert limiter.base_limiter.rpm == 120


if __name__ == "__main__":
    pytest.main([__file__])