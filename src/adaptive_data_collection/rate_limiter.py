"""
Adaptive rate limiter with token bucket algorithm.
"""

import time
import threading
import logging
from typing import Optional
from .interfaces import RateLimiter


class TokenBucketRateLimiter(RateLimiter):
    """Token bucket rate limiter for API requests."""
    
    def __init__(self, rpm: int, burst_capacity: Optional[int] = None):
        """
        Initialize token bucket rate limiter.
        
        Args:
            rpm: Requests per minute
            burst_capacity: Maximum burst capacity (defaults to rpm/4)
        """
        self.rpm = rpm
        self.tokens_per_second = rpm / 60.0
        self.burst_capacity = burst_capacity or max(1, rpm // 4)
        
        # Token bucket state
        self.tokens = float(self.burst_capacity)
        self.last_refill = time.time()
        self.lock = threading.Lock()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized rate limiter: {rpm} RPM, burst capacity: {self.burst_capacity}")
    
    def acquire(self) -> None:
        """Acquire permission to make a request (blocks if necessary)."""
        while True:
            with self.lock:
                self._refill_tokens()
                
                if self.tokens >= 1.0:
                    # Token available, consume it
                    self.tokens -= 1.0
                    self.logger.debug(f"Token acquired, {self.tokens:.2f} tokens remaining")
                    return
                
                # No tokens available, calculate wait time
                wait_time = (1.0 - self.tokens) / self.tokens_per_second
                self.logger.debug(f"Rate limit reached, waiting {wait_time:.2f} seconds")
            
            # Wait outside the lock to allow other threads
            time.sleep(wait_time)
    
    def get_wait_time(self) -> float:
        """Get current wait time until next request is allowed."""
        with self.lock:
            self._refill_tokens()
            
            if self.tokens >= 1.0:
                return 0.0
            
            return (1.0 - self.tokens) / self.tokens_per_second
    
    def update_rate_limit(self, new_rpm: int) -> None:
        """Update the rate limit dynamically."""
        with self.lock:
            old_rpm = self.rpm
            self.rpm = new_rpm
            self.tokens_per_second = new_rpm / 60.0
            self.burst_capacity = max(1, new_rpm // 4)
            
            # Adjust current tokens proportionally
            if old_rpm > 0:
                token_ratio = new_rpm / old_rpm
                self.tokens = min(self.tokens * token_ratio, self.burst_capacity)
            
            self.logger.info(f"Rate limit updated: {old_rpm} -> {new_rpm} RPM")
    
    def _refill_tokens(self) -> None:
        """Refill tokens based on elapsed time."""
        current_time = time.time()
        elapsed = current_time - self.last_refill
        
        if elapsed > 0:
            # Add tokens based on elapsed time
            tokens_to_add = elapsed * self.tokens_per_second
            self.tokens = min(self.tokens + tokens_to_add, self.burst_capacity)
            self.last_refill = current_time
    
    def get_status(self) -> dict:
        """Get current rate limiter status."""
        with self.lock:
            self._refill_tokens()
            return {
                'rpm': self.rpm,
                'tokens_per_second': self.tokens_per_second,
                'burst_capacity': self.burst_capacity,
                'current_tokens': self.tokens,
                'wait_time': self.get_wait_time()
            }


class ExponentialBackoffRateLimiter(RateLimiter):
    """Rate limiter with exponential backoff for failed requests."""
    
    def __init__(self, base_rpm: int, backoff_base: float = 2.0, max_backoff: float = 300.0):
        """
        Initialize exponential backoff rate limiter.
        
        Args:
            base_rpm: Base requests per minute
            backoff_base: Exponential backoff base multiplier
            max_backoff: Maximum backoff time in seconds
        """
        self.base_limiter = TokenBucketRateLimiter(base_rpm)
        self.backoff_base = backoff_base
        self.max_backoff = max_backoff
        
        # Backoff state
        self.consecutive_failures = 0
        self.last_failure_time = 0.0
        self.lock = threading.Lock()
        
        self.logger = logging.getLogger(__name__)
    
    def acquire(self) -> None:
        """Acquire permission with exponential backoff consideration."""
        # Check if we need to wait due to backoff
        wait_time = 0
        with self.lock:
            if self.consecutive_failures > 0:
                backoff_time = min(
                    self.backoff_base ** self.consecutive_failures,
                    self.max_backoff
                )
                
                time_since_failure = time.time() - self.last_failure_time
                if time_since_failure < backoff_time:
                    wait_time = backoff_time - time_since_failure
                    self.logger.debug(f"Exponential backoff: waiting {wait_time:.2f}s (failure #{self.consecutive_failures})")
        
        # Wait outside the lock
        if wait_time > 0:
            time.sleep(wait_time)
        
        # Use base rate limiter
        self.base_limiter.acquire()
    
    def get_wait_time(self) -> float:
        """Get current wait time including backoff."""
        base_wait = self.base_limiter.get_wait_time()
        
        with self.lock:
            if self.consecutive_failures > 0:
                backoff_time = min(
                    self.backoff_base ** self.consecutive_failures,
                    self.max_backoff
                )
                
                time_since_failure = time.time() - self.last_failure_time
                backoff_wait = max(0, backoff_time - time_since_failure)
                
                return max(base_wait, backoff_wait)
        
        return base_wait
    
    def update_rate_limit(self, new_rpm: int) -> None:
        """Update the base rate limit."""
        self.base_limiter.update_rate_limit(new_rpm)
    
    def record_success(self) -> None:
        """Record a successful request (resets backoff)."""
        with self.lock:
            if self.consecutive_failures > 0:
                self.logger.info(f"Request succeeded, resetting backoff (was {self.consecutive_failures} failures)")
                self.consecutive_failures = 0
                self.last_failure_time = 0.0
    
    def record_failure(self) -> None:
        """Record a failed request (increases backoff)."""
        with self.lock:
            self.consecutive_failures += 1
            self.last_failure_time = time.time()
            
            backoff_time = min(
                self.backoff_base ** self.consecutive_failures,
                self.max_backoff
            )
            
            self.logger.warning(f"Request failed (failure #{self.consecutive_failures}), next backoff: {backoff_time:.2f}s")
    
    def get_status(self) -> dict:
        """Get current rate limiter status."""
        base_status = self.base_limiter.get_status()
        
        # Get wait time before acquiring lock to avoid deadlock
        current_wait = self.get_wait_time()
        
        with self.lock:
            base_status.update({
                'consecutive_failures': self.consecutive_failures,
                'backoff_base': self.backoff_base,
                'max_backoff': self.max_backoff,
                'last_failure_time': self.last_failure_time,
                'current_backoff_wait': current_wait
            })
        
        return base_status