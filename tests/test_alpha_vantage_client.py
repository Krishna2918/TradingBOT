"""
Tests for Alpha Vantage API client.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import pandas as pd
import requests

from src.adaptive_data_collection.alpha_vantage_client import AlphaVantageClient
from src.adaptive_data_collection.config import CollectionConfig


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    config = Mock(spec=CollectionConfig)
    config.alpha_vantage_api_key = "test_api_key"
    config.alpha_vantage_rpm = 60
    config.years_to_collect = 25
    config.us_symbols_file = "lists/us_100.txt"
    config.max_retries = 3
    config.retry_backoff_base = 2.0
    return config


@pytest.fixture
def mock_alpha_vantage_response():
    """Mock Alpha Vantage API response."""
    return {
        "Meta Data": {
            "1. Information": "Daily Prices (open, high, low, close) and Volumes",
            "2. Symbol": "AAPL",
            "3. Last Refreshed": "2024-01-15",
            "4. Output Size": "Full size",
            "5. Time Zone": "US/Eastern"
        },
        "Time Series (Daily)": {
            "2024-01-15": {
                "1. open": "185.92",
                "2. high": "186.40",
                "3. low": "182.34",
                "4. close": "185.59",
                "5. adjusted close": "185.59",
                "6. volume": "47317442"
            },
            "2024-01-12": {
                "1. open": "182.08",
                "2. high": "186.74",
                "3. low": "180.41",
                "4. close": "185.92",
                "5. adjusted close": "185.92",
                "6. volume": "54844788"
            }
        }
    }


@pytest.fixture
def alpha_vantage_client(mock_config):
    """Create Alpha Vantage client for testing."""
    return AlphaVantageClient(mock_config)


def test_client_initialization(mock_config):
    """Test client initialization."""
    client = AlphaVantageClient(mock_config)
    
    assert client.config == mock_config
    assert client.api_key == "test_api_key"
    assert client.base_url == "https://www.alphavantage.co/query"
    assert client.request_interval == 1.0  # 60 RPM = 1 request per second


def test_rate_limiting(alpha_vantage_client):
    """Test rate limiting functionality."""
    # Mock time to control timing
    with patch('time.time') as mock_time, patch('time.sleep') as mock_sleep:
        # First call - sufficient time has passed
        mock_time.return_value = 10  # Well after any previous request
        alpha_vantage_client.last_request_time = 0  # Long ago
        alpha_vantage_client._wait_for_rate_limit()
        mock_sleep.assert_not_called()
        
        # Second call immediately after - should wait
        mock_time.return_value = 10.5  # Only 0.5 seconds later (need 1.0 for 60 RPM)
        alpha_vantage_client._wait_for_rate_limit()
        mock_sleep.assert_called_once()


@patch('requests.Session.get')
def test_make_request_success(mock_get, alpha_vantage_client, mock_alpha_vantage_response):
    """Test successful API request."""
    # Mock successful response
    mock_response = Mock()
    mock_response.json.return_value = mock_alpha_vantage_response
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response
    
    params = {'function': 'TIME_SERIES_DAILY_ADJUSTED', 'symbol': 'AAPL'}
    result = alpha_vantage_client._make_request(params)
    
    assert result == mock_alpha_vantage_response
    mock_get.assert_called_once()
    
    # Check that API key was added
    call_args = mock_get.call_args
    assert call_args[1]['params']['apikey'] == 'test_api_key'


@patch('requests.Session.get')
def test_make_request_api_error(mock_get, alpha_vantage_client):
    """Test API error handling."""
    # Mock API error response
    mock_response = Mock()
    mock_response.json.return_value = {"Error Message": "Invalid API call"}
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response
    
    params = {'function': 'TIME_SERIES_DAILY_ADJUSTED', 'symbol': 'INVALID'}
    
    with pytest.raises(ValueError, match="Alpha Vantage API error"):
        alpha_vantage_client._make_request(params)


@patch('requests.Session.get')
def test_make_request_rate_limit_error(mock_get, alpha_vantage_client):
    """Test rate limit error handling."""
    # Mock rate limit response
    mock_response = Mock()
    mock_response.json.return_value = {
        "Note": "Thank you for using Alpha Vantage! Our standard API call frequency is 5 calls per minute"
    }
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response
    
    params = {'function': 'TIME_SERIES_DAILY_ADJUSTED', 'symbol': 'AAPL'}
    
    with pytest.raises(ValueError, match="Rate limit exceeded"):
        alpha_vantage_client._make_request(params)


@patch('requests.Session.get')
def test_collect_ticker_data_success(mock_get, alpha_vantage_client, mock_alpha_vantage_response):
    """Test successful ticker data collection."""
    # Mock successful response
    mock_response = Mock()
    mock_response.json.return_value = mock_alpha_vantage_response
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response
    
    df = alpha_vantage_client.collect_ticker_data('AAPL')
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2  # Two data points in mock response
    assert list(df.columns) == ['symbol', 'date', 'open', 'high', 'low', 'close', 'adj_close', 'volume', 'source']
    assert df['symbol'].iloc[0] == 'AAPL'
    assert df['source'].iloc[0] == 'alpha_vantage'
    assert df['open'].iloc[0] == 182.08  # Should be sorted by date ascending


@patch('requests.Session.get')
def test_collect_ticker_data_invalid_response(mock_get, alpha_vantage_client):
    """Test handling of invalid response format."""
    # Mock response without expected time series key
    mock_response = Mock()
    mock_response.json.return_value = {"Meta Data": {}, "Invalid Key": {}}
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response
    
    with pytest.raises(ValueError, match="Unexpected response format"):
        alpha_vantage_client.collect_ticker_data('AAPL')


@patch('builtins.open')
def test_get_available_symbols(mock_open, alpha_vantage_client):
    """Test loading symbols from file."""
    # Mock file content using mock_open helper
    from unittest.mock import mock_open as mock_open_helper
    mock_open.return_value = mock_open_helper(read_data='AAPL\nMSFT\nGOOGL\n\n')()
    
    symbols = alpha_vantage_client.get_available_symbols()
    
    assert symbols == ['AAPL', 'MSFT', 'GOOGL']
    mock_open.assert_called_once_with('lists/us_100.txt', 'r')


def test_validate_symbol(alpha_vantage_client):
    """Test symbol validation."""
    with patch.object(alpha_vantage_client, 'get_available_symbols', return_value=['AAPL', 'MSFT', 'GOOGL']):
        assert alpha_vantage_client.validate_symbol('AAPL') is True
        assert alpha_vantage_client.validate_symbol('INVALID') is False


@patch('requests.Session.get')
def test_test_connection_success(mock_get, alpha_vantage_client):
    """Test successful connection test."""
    # Mock successful response
    mock_response = Mock()
    mock_response.json.return_value = {
        "Global Quote": {
            "01. symbol": "AAPL",
            "05. price": "185.59"
        }
    }
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response
    
    result = alpha_vantage_client.test_connection()
    assert result is True


@patch('requests.Session.get')
def test_test_connection_failure(mock_get, alpha_vantage_client):
    """Test connection test failure."""
    # Mock failed response
    mock_get.side_effect = requests.exceptions.RequestException("Connection failed")
    
    result = alpha_vantage_client.test_connection()
    assert result is False


def test_get_api_status(alpha_vantage_client):
    """Test API status reporting."""
    status = alpha_vantage_client.get_api_status()
    
    assert 'api_key_set' in status
    assert 'rate_limit_rpm' in status
    assert 'request_interval' in status
    assert 'retry_statistics' in status
    assert status['api_key_set'] is True
    assert status['rate_limit_rpm'] == 60
    assert status['request_interval'] == 1.0


@patch('requests.Session.get')
def test_collect_ticker_data_with_retry(mock_get, alpha_vantage_client, mock_alpha_vantage_response):
    """Test ticker data collection with retry on temporary failure."""
    # Mock first call fails, second succeeds
    mock_response_fail = Mock()
    mock_response_fail.side_effect = requests.exceptions.ConnectionError("Network error")
    
    mock_response_success = Mock()
    mock_response_success.json.return_value = mock_alpha_vantage_response
    mock_response_success.raise_for_status.return_value = None
    
    mock_get.side_effect = [mock_response_fail, mock_response_success]
    
    with patch('time.sleep'):  # Mock sleep to speed up test
        df = alpha_vantage_client.collect_ticker_data('AAPL')
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert mock_get.call_count == 2  # First call failed, second succeeded


@patch('requests.Session.get')
def test_collect_ticker_data_permanent_failure(mock_get, alpha_vantage_client):
    """Test ticker data collection with permanent failure (no retry)."""
    # Mock permanent error response
    mock_response = Mock()
    mock_response.json.return_value = {"Error Message": "Invalid API key"}
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response
    
    with pytest.raises(ValueError, match="Alpha Vantage API error"):
        alpha_vantage_client.collect_ticker_data('AAPL')
    
    # Should only be called once (no retries for permanent errors)
    mock_get.assert_called_once()


def test_get_retry_statistics(alpha_vantage_client):
    """Test getting retry statistics."""
    stats = alpha_vantage_client.get_retry_statistics()
    
    assert 'total_operations' in stats
    assert 'total_retry_attempts' in stats
    assert 'operations_with_retries' in stats


def test_get_recent_failures(alpha_vantage_client):
    """Test getting recent failures."""
    failures = alpha_vantage_client.get_recent_failures(24)
    
    assert isinstance(failures, list)


def test_clear_retry_history(alpha_vantage_client):
    """Test clearing retry history."""
    # Should not raise any exceptions
    alpha_vantage_client.clear_retry_history()
    alpha_vantage_client.clear_retry_history('AAPL')


if __name__ == "__main__":
    pytest.main([__file__])