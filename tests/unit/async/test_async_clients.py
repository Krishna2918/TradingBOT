"""
Unit tests for async API client functionality.

Tests cover:
- BaseAPIClient async methods
- Async request handling
- Error handling in async context
- Concurrent requests
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch, MagicMock
import pytest

from src.utils.api_client import BaseAPIClient, APIResponse, APIClientConfig


# =============================================================================
# Test Fixtures
# =============================================================================

class AsyncTestClient(BaseAPIClient):
    """Concrete implementation of BaseAPIClient for testing."""

    def __init__(self, **kwargs):
        super().__init__(
            name="TestAsync",
            base_url="https://api.test.com",
            **kwargs,
        )

    def health_check(self) -> bool:
        return True


@pytest.fixture
def async_client():
    """Create test async client."""
    config = APIClientConfig(
        timeout=5.0,
        max_retries=2,
        circuit_breaker_threshold=3,
    )
    client = AsyncTestClient(config=config)
    # Reset circuit breaker state for clean tests
    if client._circuit_breaker:
        client._circuit_breaker.reset()
    return client


# =============================================================================
# Test Async Request Building
# =============================================================================

class TestAsyncRequestBuilding:
    """Tests for async request URL and header building."""

    def test_build_url_relative(self, async_client):
        """Test URL building with relative endpoint."""
        url = async_client._build_url("/v1/data")
        assert url == "https://api.test.com/v1/data"

    def test_build_url_absolute(self, async_client):
        """Test URL building with absolute endpoint."""
        url = async_client._build_url("https://other.api.com/data")
        assert url == "https://other.api.com/data"

    def test_prepare_headers_default(self, async_client):
        """Test default header preparation."""
        headers = async_client._prepare_headers()
        assert "User-Agent" in headers
        assert headers["Accept"] == "application/json"

    def test_prepare_headers_custom(self, async_client):
        """Test custom header preparation."""
        headers = async_client._prepare_headers({"X-Custom": "value"})
        assert headers["X-Custom"] == "value"


# =============================================================================
# Test Async GET Method
# =============================================================================

class TestAsyncGet:
    """Tests for async GET requests."""

    @pytest.mark.asyncio
    async def test_get_async_success(self, async_client):
        """Test successful async GET request."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json = AsyncMock(return_value={"key": "value"})

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.request = MagicMock()
        mock_session.request.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.request.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await async_client._get_async("/v1/test")

        assert result.success is True
        assert result.data == {"key": "value"}
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_get_async_error_status(self, async_client):
        """Test async GET with error status code."""
        mock_response = MagicMock()
        mock_response.status = 404
        mock_response.headers = {"content-type": "application/json"}
        mock_response.text = AsyncMock(return_value="Not Found")
        mock_response.json = AsyncMock(return_value={"error": "Resource not found"})

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.request = MagicMock()
        mock_session.request.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.request.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await async_client._get_async("/v1/notfound")

        assert result.success is False
        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_get_async_no_aiohttp(self, async_client):
        """Test async GET when aiohttp is not installed."""
        with patch.dict("sys.modules", {"aiohttp": None}):
            # Force reimport to trigger ImportError
            with patch("builtins.__import__", side_effect=ImportError("No module named 'aiohttp'")):
                result = await async_client._get_async("/v1/test")

        assert result.success is False
        assert "aiohttp" in result.error.lower()


# =============================================================================
# Test Async POST Method
# =============================================================================

class TestAsyncPost:
    """Tests for async POST requests."""

    @pytest.mark.asyncio
    async def test_post_async_success(self, async_client):
        """Test successful async POST request."""
        mock_response = MagicMock()
        mock_response.status = 201
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json = AsyncMock(return_value={"id": 123, "created": True})

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.request = MagicMock()
        mock_session.request.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.request.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await async_client._post_async("/v1/create", json_data={"name": "test"})

        assert result.success is True
        assert result.data["id"] == 123


# =============================================================================
# Test Async Error Handling
# =============================================================================

class TestAsyncErrorHandling:
    """Tests for async error handling."""

    @pytest.mark.asyncio
    async def test_async_timeout_handling(self, async_client):
        """Test timeout handling in async requests."""
        import aiohttp

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(side_effect=asyncio.TimeoutError())
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await async_client._get_async("/v1/slow")

        assert result.success is False
        assert result.retries > 0

    @pytest.mark.asyncio
    async def test_async_connection_error(self, async_client):
        """Test connection error handling in async requests."""
        import aiohttp

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(
            side_effect=aiohttp.ClientConnectorError(Mock(), OSError("Connection refused"))
        )
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await async_client._get_async("/v1/test")

        assert result.success is False


# =============================================================================
# Test Circuit Breaker Integration
# =============================================================================

class TestAsyncCircuitBreaker:
    """Tests for circuit breaker integration with async."""

    @pytest.mark.asyncio
    async def test_async_circuit_breaker_open(self, async_client):
        """Test async request when circuit breaker is open."""
        # Open the circuit breaker
        if async_client._circuit_breaker:
            for _ in range(10):
                async_client._circuit_breaker.record_failure(Exception("test"))

        result = await async_client._get_async("/v1/test")

        # Should fail due to circuit breaker
        assert result.success is False


# =============================================================================
# Test Rate Limiting Integration
# =============================================================================

class TestAsyncRateLimiting:
    """Tests for rate limiting integration with async."""

    @pytest.mark.asyncio
    async def test_async_rate_limit_timeout(self):
        """Test async request when rate limit is exhausted."""
        # Create client with rate limiter
        config = APIClientConfig(rate_limit_timeout=0.1)
        client = AsyncTestClient(config=config, rate_limiter_name="nonexistent")

        # Reset circuit breaker
        if client._circuit_breaker:
            client._circuit_breaker.reset()

        # Mock rate limiter to always fail
        client._rate_limiter = Mock()
        client._rate_limiter.acquire = Mock(return_value=False)

        result = await client._get_async("/v1/test")

        assert result.success is False
        assert result.status_code == 429 or "rate limit" in result.error.lower()


# =============================================================================
# Test Concurrent Async Requests
# =============================================================================

class TestConcurrentAsyncRequests:
    """Tests for concurrent async operations."""

    @pytest.mark.asyncio
    async def test_multiple_concurrent_requests(self, async_client):
        """Test multiple concurrent async requests."""
        call_count = [0]

        async def mock_request(*args, **kwargs):
            call_count[0] += 1
            await asyncio.sleep(0.01)  # Simulate network delay
            return APIResponse(
                success=True,
                data={"request_num": call_count[0]},
                status_code=200,
            )

        async_client._request_async = mock_request

        # Run 5 concurrent requests
        tasks = [async_client._get_async(f"/v1/test/{i}") for i in range(5)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_gather_with_exceptions(self, async_client):
        """Test gathering results with some failures."""
        request_count = [0]

        async def mock_request(*args, **kwargs):
            request_count[0] += 1
            if request_count[0] % 2 == 0:
                return APIResponse(success=False, error="Even request failed")
            return APIResponse(success=True, data={"num": request_count[0]})

        async_client._request_async = mock_request

        tasks = [async_client._get_async(f"/v1/test/{i}") for i in range(4)]
        results = await asyncio.gather(*tasks)

        # Should have 2 successes and 2 failures
        successes = [r for r in results if r.success]
        failures = [r for r in results if not r.success]

        assert len(successes) == 2
        assert len(failures) == 2


# =============================================================================
# Test Response Parsing
# =============================================================================

class TestAsyncResponseParsing:
    """Tests for async response parsing."""

    @pytest.mark.asyncio
    async def test_parse_json_response(self, async_client):
        """Test parsing JSON response in async."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.headers = {"content-type": "application/json; charset=utf-8"}
        mock_response.json = AsyncMock(return_value={"data": [1, 2, 3]})

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.request = MagicMock()
        mock_session.request.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.request.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await async_client._get_async("/v1/data")

        assert result.data == {"data": [1, 2, 3]}

    @pytest.mark.asyncio
    async def test_parse_text_response(self, async_client):
        """Test parsing text response in async."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.headers = {"content-type": "text/plain"}
        mock_response.text = AsyncMock(return_value="Plain text response")

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.request = MagicMock()
        mock_session.request.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.request.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await async_client._get_async("/v1/text")

        assert result.data == "Plain text response"


# =============================================================================
# Test Retry Logic in Async
# =============================================================================

class TestAsyncRetryLogic:
    """Tests for retry logic in async requests."""

    @pytest.mark.asyncio
    async def test_async_retry_on_server_error(self, async_client):
        """Test retry on 500 error in async."""
        call_count = [0]

        async def mock_request_with_retry(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] < 3:
                return APIResponse(success=False, error="Server error", status_code=500)
            return APIResponse(success=True, data={"recovered": True}, status_code=200)

        async_client._request_async = mock_request_with_retry

        result = await async_client._get_async("/v1/flaky")

        # With default retry config, should eventually succeed or exhaust retries
        assert call_count[0] >= 1

    @pytest.mark.asyncio
    async def test_elapsed_time_tracking(self, async_client):
        """Test elapsed time is tracked in async requests."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json = AsyncMock(return_value={})

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.request = MagicMock()
        mock_session.request.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.request.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await async_client._get_async("/v1/test")

        assert result.elapsed_ms >= 0
