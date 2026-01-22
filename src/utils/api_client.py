"""
Base API Client
===============

Provides a base class for all external API clients with:
- Automatic retry with exponential backoff
- Rate limiting integration
- Response validation
- Circuit breaker integration
- Consistent error handling
- Request/response logging

Usage:
    from src.utils.api_client import BaseAPIClient

    class FinnhubClient(BaseAPIClient):
        def __init__(self, api_key: str):
            super().__init__(
                name="Finnhub",
                base_url="https://finnhub.io/api/v1",
                rate_limiter_name="finnhub",
            )
            self.api_key = api_key

        def get_quote(self, symbol: str) -> dict:
            return self._get(f"/quote", params={"symbol": symbol, "token": self.api_key})
"""

from __future__ import annotations

import logging
import random
import time
from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Type, Union

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Import rate limiting
from src.utils.global_rate_limiter import rate_limiters, APIRateLimiter

# Import error handling and circuit breaker if available
try:
    from src.utils.error_handler import (
        APIError,
        DataError,
        handle_error,
        ErrorSeverity,
    )
    from src.utils.circuit_breaker import (
        CircuitBreaker,
        CircuitBreakerOpenError,
        get_circuit_breaker_registry,
    )
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    APIError = Exception
    DataError = Exception

logger = logging.getLogger('trading.api_client')


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0
    jitter: bool = True
    retry_on_status: tuple = (429, 500, 502, 503, 504)
    retry_on_exceptions: tuple = (
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        requests.exceptions.ChunkedEncodingError,
    )

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for a given attempt number."""
        delay = min(
            self.base_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        if self.jitter:
            delay *= (0.5 + random.random())
        return delay


@dataclass
class APIClientConfig:
    """Configuration for API client."""
    timeout: float = 30.0
    connect_timeout: float = 10.0
    read_timeout: float = 30.0
    max_retries: int = 3
    rate_limit_timeout: float = 60.0
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    validate_responses: bool = True
    log_requests: bool = True
    log_responses: bool = False  # Be careful with sensitive data

    # Connection pooling settings
    pool_connections: int = 10  # Number of connection pools to cache
    pool_maxsize: int = 20  # Max connections per pool
    pool_block: bool = False  # Block when pool exhausted (False = create new connection)
    keep_alive: bool = True  # Enable HTTP keep-alive
    keep_alive_timeout: int = 30  # Keep-alive timeout in seconds


# =============================================================================
# Response Models
# =============================================================================

@dataclass
class APIResponse:
    """Standardized API response wrapper."""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    status_code: Optional[int] = None
    headers: Dict[str, str] = field(default_factory=dict)
    elapsed_ms: float = 0.0
    retries: int = 0
    from_cache: bool = False
    quality_score: float = 1.0

    def __bool__(self) -> bool:
        return self.success

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from data dict."""
        if isinstance(self.data, dict):
            return self.data.get(key, default)
        return default


# =============================================================================
# Base API Client
# =============================================================================

class BaseAPIClient(ABC):
    """Base class for all API clients.

    Provides:
    - Automatic retry with exponential backoff
    - Rate limiting integration
    - Circuit breaker integration
    - Response validation
    - Consistent error handling
    - Request/response logging

    Subclasses should implement specific API methods using _get, _post, etc.
    """

    def __init__(
        self,
        name: str,
        base_url: str,
        rate_limiter_name: Optional[str] = None,
        config: Optional[APIClientConfig] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        """Initialize the API client.

        Args:
            name: Client name for logging and circuit breaker
            base_url: Base URL for all requests
            rate_limiter_name: Name of rate limiter to use (from registry)
            config: Client configuration
            headers: Default headers for all requests
        """
        self.name = name
        self.base_url = base_url.rstrip('/')
        self.config = config or APIClientConfig()
        self._default_headers = headers or {}

        # Rate limiter
        self._rate_limiter: Optional[APIRateLimiter] = None
        if rate_limiter_name:
            self._rate_limiter = rate_limiters.get(rate_limiter_name)
            if not self._rate_limiter:
                logger.warning(
                    f"Rate limiter '{rate_limiter_name}' not found for {name}"
                )

        # Circuit breaker
        self._circuit_breaker: Optional[CircuitBreaker] = None
        if UTILS_AVAILABLE:
            registry = get_circuit_breaker_registry()
            self._circuit_breaker = registry.get_or_create(
                f"api_{name.lower()}",
                failure_threshold=self.config.circuit_breaker_threshold,
                recovery_timeout=self.config.circuit_breaker_timeout,
            )

        # Retry config
        self._retry_config = RetryConfig(max_retries=self.config.max_retries)

        # Session with connection pooling
        self._session = self._create_session()

        # Request tracking
        self._request_count = 0
        self._error_count = 0
        self._last_request_time: Optional[datetime] = None

        logger.info(f"API client initialized: {name} -> {base_url}")

    def _create_session(self) -> requests.Session:
        """Create requests session with retry adapter and connection pooling.

        Uses config settings for:
        - pool_connections: Number of connection pools to cache
        - pool_maxsize: Max connections per pool
        - pool_block: Whether to block when pool is exhausted
        - keep_alive: Enable HTTP keep-alive
        """
        session = requests.Session()

        # Configure retries at transport level
        retry_strategy = Retry(
            total=self._retry_config.max_retries,
            backoff_factor=self._retry_config.base_delay,
            status_forcelist=list(self._retry_config.retry_on_status),
            allowed_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
        )

        # Create adapter with configurable connection pooling
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=self.config.pool_connections,
            pool_maxsize=self.config.pool_maxsize,
            pool_block=self.config.pool_block,
        )

        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # Configure keep-alive headers if enabled
        if self.config.keep_alive:
            session.headers.update({
                "Connection": "keep-alive",
                "Keep-Alive": f"timeout={self.config.keep_alive_timeout}",
            })
        else:
            session.headers.update({
                "Connection": "close",
            })

        return session

    def _build_url(self, endpoint: str) -> str:
        """Build full URL from endpoint."""
        if endpoint.startswith('http'):
            return endpoint
        return f"{self.base_url}/{endpoint.lstrip('/')}"

    def _prepare_headers(self, headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Merge default headers with request-specific headers."""
        result = {
            "User-Agent": f"TradingBot/{self.name}",
            "Accept": "application/json",
        }
        result.update(self._default_headers)
        if headers:
            result.update(headers)
        return result

    def _acquire_rate_limit(self) -> bool:
        """Acquire rate limit slot."""
        if self._rate_limiter:
            return self._rate_limiter.acquire(timeout=self.config.rate_limit_timeout)
        return True

    def _check_circuit_breaker(self) -> None:
        """Check if circuit breaker allows request."""
        if self._circuit_breaker:
            if not self._circuit_breaker.allow_request():
                if UTILS_AVAILABLE:
                    raise CircuitBreakerOpenError(
                        f"api_{self.name.lower()}",
                        self._circuit_breaker.time_until_retry()
                    )
                else:
                    raise RuntimeError(f"Circuit breaker open for {self.name}")

    def _record_success(self) -> None:
        """Record successful request."""
        if self._circuit_breaker:
            self._circuit_breaker.record_success()

    def _record_failure(self, error: Exception) -> None:
        """Record failed request."""
        self._error_count += 1
        if self._circuit_breaker:
            self._circuit_breaker.record_failure(error)

    def _validate_response(self, response: requests.Response) -> None:
        """Validate response status and content."""
        if not self.config.validate_responses:
            return

        # Check for error status codes
        if response.status_code >= 400:
            error_msg = f"{self.name} API error: {response.status_code}"
            try:
                error_data = response.json()
                if isinstance(error_data, dict):
                    error_msg = error_data.get('error', error_data.get('message', error_msg))
            except Exception:
                error_msg = response.text[:200] if response.text else error_msg

            if UTILS_AVAILABLE:
                raise APIError(
                    error_msg,
                    context={
                        'status_code': response.status_code,
                        'url': response.url,
                    }
                )
            else:
                raise RuntimeError(error_msg)

    def _parse_response(self, response: requests.Response) -> Any:
        """Parse response content."""
        content_type = response.headers.get('content-type', '')

        if 'application/json' in content_type:
            try:
                return response.json()
            except ValueError as e:
                logger.warning(f"{self.name}: Failed to parse JSON response: {e}")
                return response.text
        else:
            return response.text

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> APIResponse:
        """Make HTTP request with retry, rate limiting, and circuit breaker.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            params: Query parameters
            data: Form data
            json: JSON body
            headers: Additional headers
            timeout: Request timeout (uses config default if not specified)

        Returns:
            APIResponse with result or error
        """
        url = self._build_url(endpoint)
        request_headers = self._prepare_headers(headers)
        request_timeout = timeout or (self.config.connect_timeout, self.config.read_timeout)

        # Log request
        if self.config.log_requests:
            logger.debug(f"{self.name} {method} {url} params={params}")

        # Check circuit breaker
        try:
            self._check_circuit_breaker()
        except Exception as e:
            return APIResponse(
                success=False,
                error=str(e),
                status_code=None,
            )

        # Acquire rate limit
        if not self._acquire_rate_limit():
            return APIResponse(
                success=False,
                error=f"Rate limit timeout for {self.name}",
                status_code=429,
            )

        # Execute request with retry
        start_time = time.time()
        last_error = None
        retries = 0

        for attempt in range(self._retry_config.max_retries + 1):
            try:
                response = self._session.request(
                    method=method,
                    url=url,
                    params=params,
                    data=data,
                    json=json,
                    headers=request_headers,
                    timeout=request_timeout,
                )

                self._request_count += 1
                self._last_request_time = datetime.now()

                # Validate response
                self._validate_response(response)

                # Parse response
                response_data = self._parse_response(response)

                # Record success
                self._record_success()

                elapsed_ms = (time.time() - start_time) * 1000

                if self.config.log_responses:
                    logger.debug(f"{self.name} response: {response.status_code} in {elapsed_ms:.0f}ms")

                return APIResponse(
                    success=True,
                    data=response_data,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    elapsed_ms=elapsed_ms,
                    retries=retries,
                )

            except self._retry_config.retry_on_exceptions as e:
                last_error = e
                retries += 1

                if attempt < self._retry_config.max_retries:
                    delay = self._retry_config.calculate_delay(attempt)
                    logger.warning(
                        f"{self.name} request failed (attempt {attempt + 1}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    self._record_failure(e)
                    logger.error(f"{self.name} request failed after {retries} retries: {e}")

            except Exception as e:
                self._record_failure(e)
                last_error = e
                logger.error(f"{self.name} request error: {e}")
                break

        return APIResponse(
            success=False,
            error=str(last_error) if last_error else "Unknown error",
            retries=retries,
            elapsed_ms=(time.time() - start_time) * 1000,
        )

    # ==========================================================================
    # Convenience Methods
    # ==========================================================================

    def _get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> APIResponse:
        """Make GET request."""
        return self._request("GET", endpoint, params=params, headers=headers, timeout=timeout)

    def _post(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> APIResponse:
        """Make POST request."""
        return self._request("POST", endpoint, data=data, json=json, headers=headers, timeout=timeout)

    def _put(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> APIResponse:
        """Make PUT request."""
        return self._request("PUT", endpoint, data=data, json=json, headers=headers, timeout=timeout)

    def _delete(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> APIResponse:
        """Make DELETE request."""
        return self._request("DELETE", endpoint, params=params, headers=headers, timeout=timeout)

    # ==========================================================================
    # Async Methods
    # ==========================================================================

    async def _request_async(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> APIResponse:
        """Make async HTTP request with retry, rate limiting, and circuit breaker.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            params: Query parameters
            data: Form data
            json_data: JSON body
            headers: Additional headers
            timeout: Request timeout (uses config default if not specified)

        Returns:
            APIResponse with result or error
        """
        try:
            import aiohttp
        except ImportError:
            logger.error("aiohttp is required for async requests. Install with: pip install aiohttp")
            return APIResponse(
                success=False,
                error="aiohttp not installed",
            )

        import asyncio

        url = self._build_url(endpoint)
        request_headers = self._prepare_headers(headers)
        request_timeout = aiohttp.ClientTimeout(
            total=timeout or self.config.timeout,
            connect=self.config.connect_timeout,
        )

        # Log request
        if self.config.log_requests:
            logger.debug(f"{self.name} async {method} {url} params={params}")

        # Check circuit breaker
        try:
            self._check_circuit_breaker()
        except Exception as e:
            return APIResponse(
                success=False,
                error=str(e),
                status_code=None,
            )

        # Acquire rate limit (sync operation)
        if not self._acquire_rate_limit():
            return APIResponse(
                success=False,
                error=f"Rate limit timeout for {self.name}",
                status_code=429,
            )

        # Execute request with retry
        start_time = time.time()
        last_error = None
        retries = 0

        for attempt in range(self._retry_config.max_retries + 1):
            try:
                async with aiohttp.ClientSession(
                    timeout=request_timeout,
                    headers=request_headers,
                ) as session:
                    async with session.request(
                        method=method,
                        url=url,
                        params=params,
                        data=data,
                        json=json_data,
                    ) as response:
                        self._request_count += 1
                        self._last_request_time = datetime.now()

                        # Check for error status
                        if response.status >= 400:
                            error_text = await response.text()
                            error_msg = f"{self.name} API error: {response.status}"
                            try:
                                error_data = await response.json()
                                if isinstance(error_data, dict):
                                    error_msg = error_data.get('error', error_data.get('message', error_msg))
                            except Exception:
                                error_msg = error_text[:200] if error_text else error_msg

                            # Retry on certain status codes
                            if response.status in self._retry_config.retry_on_status:
                                raise aiohttp.ClientResponseError(
                                    response.request_info,
                                    response.history,
                                    status=response.status,
                                    message=error_msg,
                                )

                            self._record_failure(RuntimeError(error_msg))
                            return APIResponse(
                                success=False,
                                error=error_msg,
                                status_code=response.status,
                                elapsed_ms=(time.time() - start_time) * 1000,
                            )

                        # Parse response
                        content_type = response.headers.get('content-type', '')
                        if 'application/json' in content_type:
                            try:
                                response_data = await response.json()
                            except Exception:
                                response_data = await response.text()
                        else:
                            response_data = await response.text()

                        # Record success
                        self._record_success()

                        elapsed_ms = (time.time() - start_time) * 1000

                        if self.config.log_responses:
                            logger.debug(f"{self.name} async response: {response.status} in {elapsed_ms:.0f}ms")

                        return APIResponse(
                            success=True,
                            data=response_data,
                            status_code=response.status,
                            headers=dict(response.headers),
                            elapsed_ms=elapsed_ms,
                            retries=retries,
                        )

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_error = e
                retries += 1

                if attempt < self._retry_config.max_retries:
                    delay = self._retry_config.calculate_delay(attempt)
                    logger.warning(
                        f"{self.name} async request failed (attempt {attempt + 1}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    self._record_failure(e)
                    logger.error(f"{self.name} async request failed after {retries} retries: {e}")

            except Exception as e:
                self._record_failure(e)
                last_error = e
                logger.error(f"{self.name} async request error: {e}")
                break

        return APIResponse(
            success=False,
            error=str(last_error) if last_error else "Unknown error",
            retries=retries,
            elapsed_ms=(time.time() - start_time) * 1000,
        )

    async def _get_async(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> APIResponse:
        """Make async GET request."""
        return await self._request_async("GET", endpoint, params=params, headers=headers, timeout=timeout)

    async def _post_async(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> APIResponse:
        """Make async POST request."""
        return await self._request_async("POST", endpoint, data=data, json_data=json_data, headers=headers, timeout=timeout)

    async def _put_async(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> APIResponse:
        """Make async PUT request."""
        return await self._request_async("PUT", endpoint, data=data, json_data=json_data, headers=headers, timeout=timeout)

    async def _delete_async(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> APIResponse:
        """Make async DELETE request."""
        return await self._request_async("DELETE", endpoint, params=params, headers=headers, timeout=timeout)

    # ==========================================================================
    # Status and Diagnostics
    # ==========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get client status and statistics."""
        status = {
            "name": self.name,
            "base_url": self.base_url,
            "request_count": self._request_count,
            "error_count": self._error_count,
            "error_rate": self._error_count / max(1, self._request_count),
            "last_request": self._last_request_time.isoformat() if self._last_request_time else None,
        }

        if self._rate_limiter:
            status["rate_limiter"] = self._rate_limiter.get_status()

        if self._circuit_breaker:
            status["circuit_breaker"] = {
                "state": self._circuit_breaker.state.name,
                "consecutive_failures": self._circuit_breaker.stats.consecutive_failures,
            }

        # Add connection pool status
        status["connection_pool"] = self.get_pool_status()

        return status

    def get_pool_status(self) -> Dict[str, Any]:
        """Get connection pool statistics.

        Returns:
            Dictionary with pool configuration and health info
        """
        pool_status = {
            "config": {
                "pool_connections": self.config.pool_connections,
                "pool_maxsize": self.config.pool_maxsize,
                "pool_block": self.config.pool_block,
                "keep_alive": self.config.keep_alive,
                "keep_alive_timeout": self.config.keep_alive_timeout,
            },
            "healthy": True,
            "pools": [],
        }

        # Get pool manager stats from adapters
        try:
            for prefix in ["https://", "http://"]:
                adapter = self._session.get_adapter(prefix)
                if hasattr(adapter, 'poolmanager') and adapter.poolmanager:
                    pm = adapter.poolmanager
                    # Get pool statistics
                    pool_info = {
                        "prefix": prefix,
                        "num_pools": pm.num_pools if hasattr(pm, 'num_pools') else 0,
                    }

                    # Try to get connection counts from pools
                    if hasattr(pm, 'pools'):
                        total_connections = 0
                        for key, pool in pm.pools.items():
                            if hasattr(pool, 'num_connections'):
                                total_connections += pool.num_connections
                        pool_info["total_connections"] = total_connections

                    pool_status["pools"].append(pool_info)
        except Exception as e:
            logger.debug(f"Could not get pool stats: {e}")
            pool_status["pools_error"] = str(e)

        return pool_status

    def reset_session(self) -> None:
        """Reset the HTTP session and connection pool.

        Use this to recover from connection issues or to force
        fresh connections.
        """
        if self._session:
            try:
                self._session.close()
            except Exception as e:
                logger.warning(f"Error closing session for {self.name}: {e}")

        self._session = self._create_session()
        logger.info(f"Session reset for {self.name}")

    def check_pool_health(self) -> bool:
        """Check if connection pool is healthy.

        Returns:
            True if pool is healthy, False otherwise
        """
        try:
            # Check if session exists
            if not self._session:
                return False

            # Check if adapters are mounted
            http_adapter = self._session.get_adapter("http://")
            https_adapter = self._session.get_adapter("https://")

            if not http_adapter or not https_adapter:
                return False

            # Check if pool manager is initialized
            for adapter in [http_adapter, https_adapter]:
                if hasattr(adapter, 'poolmanager'):
                    # Pool manager exists, check if it's functional
                    if adapter.poolmanager is None:
                        return False

            return True

        except Exception as e:
            logger.warning(f"Pool health check failed for {self.name}: {e}")
            return False

    def health_check(self) -> bool:
        """Check if API is reachable. Override in subclasses."""
        return True

    def close(self) -> None:
        """Close the session and cleanup."""
        if self._session:
            self._session.close()
        logger.info(f"API client closed: {self.name}")

    def __enter__(self) -> 'BaseAPIClient':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


# =============================================================================
# Concrete Client Examples
# =============================================================================

class JSONPlaceholderClient(BaseAPIClient):
    """Example client for JSONPlaceholder API (for testing)."""

    def __init__(self):
        super().__init__(
            name="JSONPlaceholder",
            base_url="https://jsonplaceholder.typicode.com",
        )

    def get_posts(self) -> APIResponse:
        """Get all posts."""
        return self._get("/posts")

    def get_post(self, post_id: int) -> APIResponse:
        """Get a specific post."""
        return self._get(f"/posts/{post_id}")


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    'RetryConfig',
    'APIClientConfig',
    'APIResponse',
    'BaseAPIClient',
    'JSONPlaceholderClient',
]
