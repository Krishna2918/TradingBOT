"""
Tests for the US market data collector.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

from src.adaptive_data_collection.us_market_collector import USMarketDataCollector
from src.adaptive_data_collection.config import CollectionConfig


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    config = Mock(spec=CollectionConfig)
    config.alpha_vantage_api_key = "test_api_key"
    config.alpha_vantage_rpm = 60
    config.years_to_collect = 25
    config.us_symbols_file = "test_symbols.txt"
    config.max_retries = 3
    config.retry_backoff_base = 2.0
    config.output_base_dir = "test_data"
    return config


@pytest.fixture
def temp_symbols_file():
    """Create a temporary symbols file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("AAPL\nMSFT\nGOOGL\nTSLA\n")
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def us_collector(mock_config, temp_symbols_file):
    """Create US market collector for testing."""
    mock_config.us_symbols_file = temp_symbols_file
    
    with patch('src.adaptive_data_collection.us_market_collector.AlphaVantageClient'), \
         patch('src.adaptive_data_collection.us_market_collector.JSONProgressTracker'):
        collector = USMarketDataCollector(mock_config)
        
        # Mock the components
        collector.alpha_vantage_client = Mock()
        collector.progress_tracker = Mock()
        
        return collector


def test_collector_initialization(mock_config, temp_symbols_file):
    """Test collector initialization."""
    mock_config.us_symbols_file = temp_symbols_file
    
    with patch('src.adaptive_data_collection.us_market_collector.AlphaVantageClient'), \
         patch('src.adaptive_data_collection.us_market_collector.JSONProgressTracker'):
        collector = USMarketDataCollector(mock_config)
        
        assert collector.config == mock_config
        assert not collector.is_running
        assert not collector.should_stop


def test_load_symbols(us_collector, temp_symbols_file):
    """Test loading symbols from file."""
    us_collector.config.us_symbols_file = temp_symbols_file
    
    symbols = us_collector._load_symbols()
    
    assert symbols == ["AAPL", "MSFT", "GOOGL", "TSLA"]


def test_load_symbols_file_not_found(us_collector):
    """Test loading symbols when file doesn't exist."""
    us_collector.config.us_symbols_file = "nonexistent_file.txt"
    
    with pytest.raises(FileNotFoundError):
        us_collector._load_symbols()


def test_load_symbols_empty_file(us_collector):
    """Test loading symbols from empty file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        empty_file = f.name
    
    try:
        us_collector.config.us_symbols_file = empty_file
        
        with pytest.raises(ValueError, match="No symbols found"):
            us_collector._load_symbols()
    finally:
        Path(empty_file).unlink(missing_ok=True)


def test_collect_symbol_data_success(us_collector):
    """Test successful symbol data collection."""
    symbol = "AAPL"
    mock_df = pd.DataFrame({
        'symbol': ['AAPL'] * 100,
        'date': pd.date_range('2020-01-01', periods=100),
        'close': range(100)
    })
    
    # Mock successful data collection
    us_collector.alpha_vantage_client.collect_ticker_data.return_value = mock_df
    
    result = us_collector._collect_symbol_data(symbol)
    
    assert result is True
    us_collector.progress_tracker.mark_symbol_started.assert_called_once_with(symbol)
    us_collector.progress_tracker.mark_symbol_complete.assert_called_once_with(symbol, success=True)
    us_collector.progress_tracker.update_symbol_data.assert_called_once()


def test_collect_symbol_data_failure(us_collector):
    """Test failed symbol data collection."""
    symbol = "AAPL"
    error_message = "API error"
    
    # Mock failed data collection
    us_collector.alpha_vantage_client.collect_ticker_data.side_effect = Exception(error_message)
    
    result = us_collector._collect_symbol_data(symbol)
    
    assert result is False
    us_collector.progress_tracker.mark_symbol_started.assert_called_once_with(symbol)
    us_collector.progress_tracker.mark_symbol_complete.assert_called_once_with(
        symbol, success=False, error=error_message
    )


def test_collect_symbol_data_empty_result(us_collector):
    """Test symbol data collection with empty result."""
    symbol = "AAPL"
    
    # Mock empty data collection
    us_collector.alpha_vantage_client.collect_ticker_data.return_value = pd.DataFrame()
    
    result = us_collector._collect_symbol_data(symbol)
    
    assert result is False
    us_collector.progress_tracker.mark_symbol_complete.assert_called_once()
    
    # Check the call arguments
    call_args = us_collector.progress_tracker.mark_symbol_complete.call_args
    assert call_args[0][0] == symbol  # First positional arg is symbol
    assert call_args[1]["success"] is False  # Keyword arg success=False
    assert "No data returned" in call_args[1]["error"]  # Keyword arg error contains message


def test_collect_all_symbols_fresh_start(us_collector, temp_symbols_file):
    """Test collecting all symbols from fresh start."""
    us_collector.config.us_symbols_file = temp_symbols_file
    
    # Mock successful data collection for all symbols
    mock_df = pd.DataFrame({'symbol': ['TEST'], 'close': [100]})
    us_collector.alpha_vantage_client.collect_ticker_data.return_value = mock_df
    
    # Mock progress tracker
    us_collector.progress_tracker.get_pending_symbols.return_value = ["AAPL", "MSFT", "GOOGL", "TSLA"]
    us_collector.progress_tracker.get_progress_summary.return_value = {
        "total_symbols": 4,
        "completed_symbols": 4,
        "failed_symbols": 0,
        "progress_percentage": 100.0,
        "success_rate": 100.0
    }
    
    result = us_collector.collect_all_symbols(resume=False)
    
    assert result["symbols_succeeded"] == 4
    assert result["symbols_failed"] == 0
    assert not us_collector.is_running
    
    # Verify progress tracker calls
    us_collector.progress_tracker.reset_progress.assert_called_once()
    us_collector.progress_tracker.initialize_symbols.assert_called_once()


def test_collect_all_symbols_resume(us_collector, temp_symbols_file):
    """Test resuming collection from previous progress."""
    us_collector.config.us_symbols_file = temp_symbols_file
    
    # Mock partial progress (2 symbols remaining)
    us_collector.progress_tracker.get_pending_symbols.return_value = ["GOOGL", "TSLA"]
    us_collector.progress_tracker.get_progress_summary.return_value = {
        "total_symbols": 4,
        "completed_symbols": 4,
        "failed_symbols": 0,
        "progress_percentage": 100.0,
        "success_rate": 100.0
    }
    
    # Mock successful data collection
    mock_df = pd.DataFrame({'symbol': ['TEST'], 'close': [100]})
    us_collector.alpha_vantage_client.collect_ticker_data.return_value = mock_df
    
    result = us_collector.collect_all_symbols(resume=True)
    
    assert result["symbols_succeeded"] == 2
    
    # Should not reset progress when resuming
    us_collector.progress_tracker.reset_progress.assert_not_called()


def test_stop_collection(us_collector):
    """Test stopping collection gracefully."""
    us_collector.stop_collection()
    
    assert us_collector.should_stop is True


def test_get_collection_status(us_collector):
    """Test getting collection status."""
    # Mock return values
    us_collector.progress_tracker.get_progress_summary.return_value = {"progress": "test"}
    us_collector.alpha_vantage_client.get_api_status.return_value = {"api": "test"}
    us_collector.alpha_vantage_client.get_retry_statistics.return_value = {"retry": "test"}
    
    status = us_collector.get_collection_status()
    
    assert "is_running" in status
    assert "should_stop" in status
    assert "progress" in status
    assert "alpha_vantage_status" in status
    assert "retry_statistics" in status


def test_get_failed_symbols(us_collector):
    """Test getting failed symbols information."""
    mock_progress_data = {
        "symbols": {
            "AAPL": {"status": "completed"},
            "MSFT": {"status": "failed", "error": "API error", "attempts": 3, "end_time": "2024-01-01"},
            "GOOGL": {"status": "failed", "error": "Network error", "attempts": 2, "end_time": "2024-01-02"}
        }
    }
    
    us_collector.progress_tracker.load_progress.return_value = mock_progress_data
    
    failed = us_collector.get_failed_symbols()
    
    assert len(failed) == 2
    assert failed[0]["symbol"] == "MSFT"
    assert failed[0]["error"] == "API error"
    assert failed[1]["symbol"] == "GOOGL"
    assert failed[1]["error"] == "Network error"


def test_retry_failed_symbols(us_collector):
    """Test retrying failed symbols."""
    # Mock failed symbols
    us_collector.progress_tracker.get_failed_symbols.return_value = ["MSFT", "GOOGL"]
    
    mock_progress_data = {
        "symbols": {
            "MSFT": {"status": "failed", "error": "API error"},
            "GOOGL": {"status": "failed", "error": "Network error"}
        },
        "failed_symbols": 2
    }
    
    us_collector.progress_tracker.load_progress.return_value = mock_progress_data
    
    # Mock collect_all_symbols to return immediately
    with patch.object(us_collector, 'collect_all_symbols', return_value={"retried": True}):
        result = us_collector.retry_failed_symbols()
    
    assert result["retried"] is True
    us_collector.progress_tracker.save_progress.assert_called_once()


def test_retry_failed_symbols_none_failed(us_collector):
    """Test retrying when no symbols have failed."""
    us_collector.progress_tracker.get_failed_symbols.return_value = []
    
    result = us_collector.retry_failed_symbols()
    
    assert result["retried_count"] == 0
    assert "No failed symbols" in result["message"]


def test_validate_setup_success(us_collector, temp_symbols_file):
    """Test successful setup validation."""
    us_collector.config.us_symbols_file = temp_symbols_file
    us_collector.alpha_vantage_client.test_connection.return_value = True
    
    with patch('pathlib.Path.exists', return_value=True):
        result = us_collector.validate_setup()
    
    assert result["valid"] is True
    assert len(result["issues"]) == 0


def test_validate_setup_api_failure(us_collector, temp_symbols_file):
    """Test setup validation with API connection failure."""
    us_collector.config.us_symbols_file = temp_symbols_file
    us_collector.alpha_vantage_client.test_connection.return_value = False
    
    result = us_collector.validate_setup()
    
    assert result["valid"] is False
    assert any("API connection failed" in issue for issue in result["issues"])


def test_validate_setup_symbols_file_missing(us_collector):
    """Test setup validation with missing symbols file."""
    us_collector.config.us_symbols_file = "nonexistent_file.txt"
    us_collector.alpha_vantage_client.test_connection.return_value = True
    
    result = us_collector.validate_setup()
    
    assert result["valid"] is False
    assert any("Symbols file error" in issue for issue in result["issues"])


def test_export_collection_report(us_collector):
    """Test exporting collection report."""
    output_file = "test_report.json"
    
    us_collector.export_collection_report(output_file)
    
    us_collector.progress_tracker.export_progress_report.assert_called_once_with(output_file)


if __name__ == "__main__":
    pytest.main([__file__])