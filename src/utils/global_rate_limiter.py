"""
Global singleton rate limiters for API services.

This module provides thread-safe, global rate limiters that are shared across
all instances and modules to prevent exceeding API rate limits.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class APIRateLimiter:
    """
    Thread-safe rate limiter using sliding window algorithm.

    Parameters
    ----------
    name : str
        Name of the API being rate-limited.
    requests_per_minute : int
        Maximum requests allowed per minute.
    min_interval : float
        Minimum seconds between requests (defaults to 60/RPM).
    """

    name: str
    requests_per_minute: int
    min_interval: Optional[float] = None
    _request_times: deque = field(default_factory=deque, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _last_request: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.min_interval is None:
            self.min_interval = 60.0 / self.requests_per_minute

    def acquire(self, timeout: float = 60.0) -> bool:
        """
        Acquire permission to make a request. Blocks until allowed or timeout.

        Parameters
        ----------
        timeout : float
            Maximum seconds to wait for permission.

        Returns
        -------
        bool
            True if permission granted, False if timeout.
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            with self._lock:
                now = time.time()

                # Clean old requests (outside 60-second window)
                while self._request_times and now - self._request_times[0] >= 60:
                    self._request_times.popleft()

                # Check if we're under the per-minute limit
                if len(self._request_times) >= self.requests_per_minute:
                    wait_time = 60 - (now - self._request_times[0])
                    logger.debug(
                        "%s rate limit: %d/%d RPM, waiting %.2fs",
                        self.name,
                        len(self._request_times),
                        self.requests_per_minute,
                        wait_time,
                    )
                else:
                    # Check minimum interval
                    elapsed = now - self._last_request
                    if elapsed >= self.min_interval:
                        # Permission granted
                        self._request_times.append(now)
                        self._last_request = now
                        return True
                    wait_time = self.min_interval - elapsed

            # Wait outside the lock
            time.sleep(min(wait_time, 0.1))

        logger.warning("%s rate limiter timeout after %.1fs", self.name, timeout)
        return False

    def get_status(self) -> Dict[str, float]:
        """Get current rate limiter status."""
        with self._lock:
            now = time.time()
            # Clean old requests
            while self._request_times and now - self._request_times[0] >= 60:
                self._request_times.popleft()

            return {
                "name": self.name,
                "requests_per_minute": self.requests_per_minute,
                "current_minute_usage": len(self._request_times),
                "remaining_capacity": self.requests_per_minute - len(self._request_times),
                "seconds_until_next_slot": max(
                    0, self.min_interval - (now - self._last_request)
                ),
            }

    def reset(self) -> None:
        """Reset the rate limiter (for testing)."""
        with self._lock:
            self._request_times.clear()
            self._last_request = 0.0


class GlobalRateLimiterRegistry:
    """
    Singleton registry for global rate limiters.

    Usage:
        from src.utils.global_rate_limiter import rate_limiters
        rate_limiters.finnhub.acquire()  # blocks until allowed
    """

    _instance: Optional["GlobalRateLimiterRegistry"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "GlobalRateLimiterRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """Initialize default rate limiters."""
        self._limiters: Dict[str, APIRateLimiter] = {}

        # Finnhub: 60 requests per minute for free tier
        self._limiters["finnhub"] = APIRateLimiter(
            name="Finnhub",
            requests_per_minute=60,
            min_interval=1.0,  # At least 1 second between requests
        )

        # NewsAPI: ~100 requests per day (free tier), we'll limit to 10/min
        self._limiters["newsapi"] = APIRateLimiter(
            name="NewsAPI",
            requests_per_minute=10,
            min_interval=6.0,
        )

        # Alpha Vantage: 5 requests per minute (free tier)
        self._limiters["alpha_vantage"] = APIRateLimiter(
            name="AlphaVantage",
            requests_per_minute=5,
            min_interval=12.5,
        )

        # Yahoo Finance: generous limits, but still rate limit
        self._limiters["yahoo_finance"] = APIRateLimiter(
            name="YahooFinance",
            requests_per_minute=60,
            min_interval=0.5,
        )

        logger.info(
            "Global rate limiters initialized: %s",
            list(self._limiters.keys()),
        )

    @property
    def finnhub(self) -> APIRateLimiter:
        """Get the Finnhub rate limiter."""
        return self._limiters["finnhub"]

    @property
    def newsapi(self) -> APIRateLimiter:
        """Get the NewsAPI rate limiter."""
        return self._limiters["newsapi"]

    @property
    def alpha_vantage(self) -> APIRateLimiter:
        """Get the Alpha Vantage rate limiter."""
        return self._limiters["alpha_vantage"]

    @property
    def yahoo_finance(self) -> APIRateLimiter:
        """Get the Yahoo Finance rate limiter."""
        return self._limiters["yahoo_finance"]

    def get(self, name: str) -> Optional[APIRateLimiter]:
        """Get a rate limiter by name."""
        return self._limiters.get(name)

    def register(
        self,
        name: str,
        requests_per_minute: int,
        min_interval: Optional[float] = None,
    ) -> APIRateLimiter:
        """Register a new rate limiter or update existing."""
        self._limiters[name] = APIRateLimiter(
            name=name,
            requests_per_minute=requests_per_minute,
            min_interval=min_interval,
        )
        return self._limiters[name]

    def get_all_status(self) -> Dict[str, Dict]:
        """Get status of all rate limiters."""
        return {name: limiter.get_status() for name, limiter in self._limiters.items()}


# Global singleton instance
rate_limiters = GlobalRateLimiterRegistry()


__all__ = ["APIRateLimiter", "GlobalRateLimiterRegistry", "rate_limiters"]
