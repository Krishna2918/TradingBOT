"""
Unit tests for AlphaVantageClient
==================================

Tests for the refactored AlphaVantageClient that inherits from BaseAPIClient.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import tempfile
from pathlib import Path

import pandas as pd

from src.adaptive_data_collection.alpha_vantage_client import AlphaVantageClient
from src.adaptive_data_collection.config import CollectionConfig
from src.utils.api_client import APIResponse


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_config():
    """Create mock collection config."""
    return CollectionConfig(
        alpha_vantage_api_key="test_api_key_12345",
        alpha_vantage_rpm=5,
        years_to_collect=5,
        us_symbols_file="test_symbols.txt",
    )


@pytest.fixture
def client(mock_config):
    """Create AlphaVantageClient for testing."""
    return AlphaVantageClient(mock_config)


@pytest.fixture
def sample_daily_data():
    """Create sample daily time series data."""
    return {
        "Meta Data": {
            "1. Information": "Daily Time Series with Splits and Dividend Events",
            "2. Symbol": "AAPL",
        },
        "Time Series (Daily)": {
            "2024-01-15": {
                "1. open": "150.00",
                "2. high": "152.00",
                "3. low": "149.00",
                "4. close": "151.50",
                "5. adjusted close": "151.50",
                "6. volume": "50000000",
            },
            "2024-01-14": {
                "1. open": "148.00",
                "2. high": "151.00",
                "3. low": "147.00",
                "4. close": "150.00",
                "5. adjusted close": "150.00",
                "6. volume": "45000000",
            },
        },
    }


@pytest.fixture
def sample_quote_data():
    """Create sample global quote data."""
    return {
        "Global Quote": {
            "01. symbol": "AAPL",
            "02. open": "150.00",
            "03. high": "152.00",
            "04. low": "149.00",
            "05. price": "151.50",
            "06. volume": "50000000",
            "07. latest trading day": "2024-01-15",
            "08. previous close": "150.00",
            "09. change": "1.50",
            "10. change percent": "1.00%",
        }
    }


# =============================================================================
# Initialization Tests
# =============================================================================

class TestAlphaVantageClientInit:
    """Tests for AlphaVantageClient initialization."""

    def test_init_with_config(self, mock_config):
        """Test client initializes with config."""
        client = AlphaVantageClient(mock_config)

        assert client.api_key == "test_api_key_12345"
        assert client.collection_config == mock_config
        assert client.base_url == "https://www.alphavantage.co"

    def test_init_tracking_stats(self, client):
        """Test client initializes tracking statistics."""
        assert client._request_count == 0
        assert client._error_count == 0
        assert client._last_request_time == 0.0


# =============================================================================
# API Request Tests
# =============================================================================

class TestAPIRequests:
    """Tests for API request methods."""

    def test_make_api_request_success(self, client, sample_quote_data):
        """Test successful API request."""
        mock_response = APIResponse(
            success=True,
            data=sample_quote_data,
            status_code=200,
        )

        with patch.object(client, "_get", return_value=mock_response):
            response = client._make_api_request({"function": "GLOBAL_QUOTE", "symbol": "AAPL"})

            assert response.success is True
            assert "Global Quote" in response.data

    def test_make_api_request_error_message(self, client):
        """Test API request with error message in response."""
        mock_response = APIResponse(
            success=True,
            data={"Error Message": "Invalid API call"},
            status_code=200,
        )

        with patch.object(client, "_get", return_value=mock_response):
            response = client._make_api_request({"function": "INVALID"})

            assert response.success is False
            assert "Invalid API call" in response.error

    def test_make_api_request_rate_limit(self, client):
        """Test API request with rate limit note."""
        mock_response = APIResponse(
            success=True,
            data={"Note": "Thank you for using Alpha Vantage! Our standard API call frequency is 5 calls per minute."},
            status_code=200,
        )

        with patch.object(client, "_get", return_value=mock_response):
            response = client._make_api_request({"function": "GLOBAL_QUOTE"})

            assert response.success is False
            assert response.status_code == 429

    def test_make_api_request_tracks_stats(self, client, sample_quote_data):
        """Test that API requests track statistics."""
        mock_response = APIResponse(
            success=True,
            data=sample_quote_data,
            status_code=200,
        )

        with patch.object(client, "_get", return_value=mock_response):
            client._make_api_request({"function": "GLOBAL_QUOTE"})

            assert client._request_count == 1
            assert client._last_request_time > 0


# =============================================================================
# Data Collection Tests
# =============================================================================

class TestDataCollection:
    """Tests for data collection methods."""

    def test_collect_ticker_data_success(self, client, sample_daily_data):
        """Test successful ticker data collection."""
        mock_response = APIResponse(
            success=True,
            data=sample_daily_data,
            status_code=200,
        )

        with patch.object(client, "_get", return_value=mock_response):
            df = client.collect_ticker_data("AAPL")

            assert isinstance(df, pd.DataFrame)
            assert len(df) == 2
            assert "symbol" in df.columns
            assert "close" in df.columns
            assert df["symbol"].iloc[0] == "AAPL"

    def test_collect_ticker_data_failure(self, client):
        """Test ticker data collection failure."""
        mock_response = APIResponse(
            success=False,
            error="API error",
            status_code=500,
        )

        with patch.object(client, "_get", return_value=mock_response):
            with pytest.raises(ValueError) as exc_info:
                client.collect_ticker_data("AAPL")

            assert "Failed to collect data" in str(exc_info.value)

    def test_collect_ticker_data_unexpected_format(self, client):
        """Test ticker data collection with unexpected format."""
        mock_response = APIResponse(
            success=True,
            data={"unexpected": "format"},
            status_code=200,
        )

        with patch.object(client, "_get", return_value=mock_response):
            with pytest.raises(ValueError) as exc_info:
                client.collect_ticker_data("AAPL")

            assert "Unexpected response format" in str(exc_info.value)


# =============================================================================
# Global Quote Tests
# =============================================================================

class TestGlobalQuote:
    """Tests for global quote retrieval."""

    def test_get_global_quote_success(self, client, sample_quote_data):
        """Test successful quote retrieval."""
        mock_response = APIResponse(
            success=True,
            data=sample_quote_data,
            status_code=200,
        )

        with patch.object(client, "_get", return_value=mock_response):
            quote = client.get_global_quote("AAPL")

            assert quote is not None
            assert quote["symbol"] == "AAPL"
            assert quote["price"] == 151.50
            assert quote["volume"] == 50000000

    def test_get_global_quote_failure(self, client):
        """Test quote retrieval failure."""
        mock_response = APIResponse(
            success=False,
            error="API error",
            status_code=500,
        )

        with patch.object(client, "_get", return_value=mock_response):
            quote = client.get_global_quote("AAPL")

            assert quote is None


# =============================================================================
# Connection Test
# =============================================================================

class TestConnection:
    """Tests for connection testing."""

    def test_test_connection_success(self, client, sample_quote_data):
        """Test successful connection test."""
        mock_response = APIResponse(
            success=True,
            data=sample_quote_data,
            status_code=200,
        )

        with patch.object(client, "_get", return_value=mock_response):
            result = client.test_connection()

            assert result is True

    def test_test_connection_failure(self, client):
        """Test connection test failure."""
        mock_response = APIResponse(
            success=False,
            error="Network error",
            status_code=None,
        )

        with patch.object(client, "_get", return_value=mock_response):
            result = client.test_connection()

            assert result is False

    def test_health_check_alias(self, client, sample_quote_data):
        """Test health_check is alias for test_connection."""
        mock_response = APIResponse(
            success=True,
            data=sample_quote_data,
            status_code=200,
        )

        with patch.object(client, "_get", return_value=mock_response):
            result = client.health_check()

            assert result is True


# =============================================================================
# Symbol Management Tests
# =============================================================================

class TestSymbolManagement:
    """Tests for symbol management."""

    def test_get_available_symbols(self, client):
        """Test loading available symbols."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("AAPL\nMSFT\nGOOG\n")
            temp_path = f.name

        try:
            client.collection_config.us_symbols_file = temp_path
            symbols = client.get_available_symbols()

            assert len(symbols) == 3
            assert "AAPL" in symbols
            assert "MSFT" in symbols
        finally:
            Path(temp_path).unlink()

    def test_get_available_symbols_file_not_found(self, client):
        """Test loading symbols from non-existent file."""
        client.collection_config.us_symbols_file = "nonexistent.txt"

        with pytest.raises(FileNotFoundError):
            client.get_available_symbols()

    def test_validate_symbol(self, client):
        """Test symbol validation."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("AAPL\nMSFT\n")
            temp_path = f.name

        try:
            client.collection_config.us_symbols_file = temp_path

            assert client.validate_symbol("AAPL") is True
            assert client.validate_symbol("INVALID") is False
        finally:
            Path(temp_path).unlink()


# =============================================================================
# API Status Tests
# =============================================================================

class TestAPIStatus:
    """Tests for API status retrieval."""

    def test_get_api_status(self, client):
        """Test getting API status."""
        status = client.get_api_status()

        assert "api_key_set" in status
        assert status["api_key_set"] is True
        assert "rate_limit_rpm" in status
        assert status["rate_limit_rpm"] == 5
        assert "request_count" in status
        assert "error_count" in status

    def test_get_api_status_after_requests(self, client, sample_quote_data):
        """Test API status after making requests."""
        mock_response = APIResponse(
            success=True,
            data=sample_quote_data,
            status_code=200,
        )

        with patch.object(client, "_get", return_value=mock_response):
            client._make_api_request({"function": "GLOBAL_QUOTE"})
            client._make_api_request({"function": "GLOBAL_QUOTE"})

            status = client.get_api_status()

            assert status["request_count"] == 2


# =============================================================================
# Intraday Data Tests
# =============================================================================

class TestIntradayData:
    """Tests for intraday data retrieval."""

    def test_get_intraday_data_success(self, client):
        """Test successful intraday data retrieval."""
        intraday_data = {
            "Time Series (5min)": {
                "2024-01-15 16:00:00": {
                    "1. open": "150.00",
                    "2. high": "150.50",
                    "3. low": "149.50",
                    "4. close": "150.25",
                    "5. volume": "1000000",
                },
            }
        }
        mock_response = APIResponse(
            success=True,
            data=intraday_data,
            status_code=200,
        )

        with patch.object(client, "_get", return_value=mock_response):
            df = client.get_intraday_data("AAPL", "5min")

            assert df is not None
            assert len(df) == 1
            assert "close" in df.columns

    def test_get_intraday_data_failure(self, client):
        """Test intraday data retrieval failure."""
        mock_response = APIResponse(
            success=False,
            error="API error",
            status_code=500,
        )

        with patch.object(client, "_get", return_value=mock_response):
            df = client.get_intraday_data("AAPL")

            assert df is None
