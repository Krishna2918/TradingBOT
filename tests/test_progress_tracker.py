"""
Tests for the progress tracker.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock
from datetime import datetime

from src.adaptive_data_collection.progress_tracker import JSONProgressTracker


@pytest.fixture
def temp_progress_file():
    """Create a temporary progress file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def progress_tracker(temp_progress_file):
    """Create progress tracker with temporary file."""
    return JSONProgressTracker(temp_progress_file)


def test_progress_tracker_initialization(temp_progress_file):
    """Test progress tracker initialization."""
    tracker = JSONProgressTracker(temp_progress_file)
    
    assert tracker.progress_file == Path(temp_progress_file)
    assert "session_id" in tracker.progress_data
    assert "start_time" in tracker.progress_data
    assert tracker.progress_data["total_symbols"] == 0
    assert tracker.progress_data["completed_symbols"] == 0
    assert tracker.progress_data["failed_symbols"] == 0
    assert tracker.progress_data["symbols"] == {}


def test_initialize_symbols(progress_tracker):
    """Test initializing symbols for tracking."""
    symbols = ["AAPL", "MSFT", "GOOGL"]
    
    progress_tracker.initialize_symbols(symbols)
    
    assert progress_tracker.progress_data["total_symbols"] == 3
    for symbol in symbols:
        assert symbol in progress_tracker.progress_data["symbols"]
        assert progress_tracker.progress_data["symbols"][symbol]["status"] == "not_started"


def test_mark_symbol_started(progress_tracker):
    """Test marking a symbol as started."""
    symbol = "AAPL"
    progress_tracker.initialize_symbols([symbol])
    
    progress_tracker.mark_symbol_started(symbol)
    
    symbol_data = progress_tracker.progress_data["symbols"][symbol]
    assert symbol_data["status"] == "in_progress"
    assert symbol_data["start_time"] is not None
    assert symbol_data["attempts"] == 1


def test_mark_symbol_complete_success(progress_tracker):
    """Test marking a symbol as successfully completed."""
    symbol = "AAPL"
    progress_tracker.initialize_symbols([symbol])
    progress_tracker.mark_symbol_started(symbol)
    
    progress_tracker.mark_symbol_complete(symbol, success=True)
    
    symbol_data = progress_tracker.progress_data["symbols"][symbol]
    assert symbol_data["status"] == "completed"
    assert symbol_data["end_time"] is not None
    assert symbol_data["error"] is None
    assert progress_tracker.progress_data["completed_symbols"] == 1


def test_mark_symbol_complete_failure(progress_tracker):
    """Test marking a symbol as failed."""
    symbol = "AAPL"
    error_message = "API error"
    progress_tracker.initialize_symbols([symbol])
    progress_tracker.mark_symbol_started(symbol)
    
    progress_tracker.mark_symbol_complete(symbol, success=False, error=error_message)
    
    symbol_data = progress_tracker.progress_data["symbols"][symbol]
    assert symbol_data["status"] == "failed"
    assert symbol_data["end_time"] is not None
    assert symbol_data["error"] == error_message
    assert progress_tracker.progress_data["failed_symbols"] == 1


def test_update_symbol_data(progress_tracker):
    """Test updating symbol data statistics."""
    symbol = "AAPL"
    progress_tracker.initialize_symbols([symbol])
    
    progress_tracker.update_symbol_data(symbol, data_points=1000, file_size_bytes=50000)
    
    symbol_data = progress_tracker.progress_data["symbols"][symbol]
    assert symbol_data["data_points"] == 1000
    assert symbol_data["file_size_bytes"] == 50000


def test_get_pending_symbols(progress_tracker):
    """Test getting pending symbols."""
    symbols = ["AAPL", "MSFT", "GOOGL"]
    progress_tracker.initialize_symbols(symbols)
    
    # Mark one as completed, one as failed
    progress_tracker.mark_symbol_complete("AAPL", success=True)
    progress_tracker.mark_symbol_complete("MSFT", success=False, error="Test error")
    
    pending = progress_tracker.get_pending_symbols()
    
    # Should include failed symbol and not started symbol
    assert "MSFT" in pending  # Failed symbols are pending for retry
    assert "GOOGL" in pending  # Not started
    assert "AAPL" not in pending  # Completed


def test_get_completed_symbols(progress_tracker):
    """Test getting completed symbols."""
    symbols = ["AAPL", "MSFT", "GOOGL"]
    progress_tracker.initialize_symbols(symbols)
    
    progress_tracker.mark_symbol_complete("AAPL", success=True)
    progress_tracker.mark_symbol_complete("MSFT", success=False, error="Test error")
    
    completed = progress_tracker.get_completed_symbols()
    
    assert completed == ["AAPL"]


def test_get_failed_symbols(progress_tracker):
    """Test getting failed symbols."""
    symbols = ["AAPL", "MSFT", "GOOGL"]
    progress_tracker.initialize_symbols(symbols)
    
    progress_tracker.mark_symbol_complete("AAPL", success=True)
    progress_tracker.mark_symbol_complete("MSFT", success=False, error="Test error")
    
    failed = progress_tracker.get_failed_symbols()
    
    assert failed == ["MSFT"]


def test_get_progress_summary(progress_tracker):
    """Test getting progress summary."""
    symbols = ["AAPL", "MSFT", "GOOGL", "TSLA"]
    progress_tracker.initialize_symbols(symbols)
    
    # Complete some symbols
    progress_tracker.mark_symbol_complete("AAPL", success=True)
    progress_tracker.mark_symbol_complete("MSFT", success=False, error="Test error")
    progress_tracker.mark_symbol_started("GOOGL")  # In progress
    
    summary = progress_tracker.get_progress_summary()
    
    assert summary["total_symbols"] == 4
    assert summary["completed_symbols"] == 1
    assert summary["failed_symbols"] == 1
    assert summary["in_progress_symbols"] == 1
    assert summary["pending_symbols"] == 1
    assert summary["progress_percentage"] == 25.0  # 1 completed out of 4


def test_save_and_load_progress(temp_progress_file):
    """Test saving and loading progress from file."""
    # Create tracker and add some data
    tracker1 = JSONProgressTracker(temp_progress_file)
    tracker1.initialize_symbols(["AAPL", "MSFT"])
    tracker1.mark_symbol_complete("AAPL", success=True)
    
    # Create new tracker with same file
    tracker2 = JSONProgressTracker(temp_progress_file)
    
    # Should load existing data
    assert tracker2.progress_data["total_symbols"] == 2
    assert tracker2.progress_data["completed_symbols"] == 1
    assert "AAPL" in tracker2.progress_data["symbols"]
    assert tracker2.progress_data["symbols"]["AAPL"]["status"] == "completed"


def test_reset_progress(progress_tracker, temp_progress_file):
    """Test resetting progress."""
    # Add some data
    progress_tracker.initialize_symbols(["AAPL", "MSFT"])
    progress_tracker.mark_symbol_complete("AAPL", success=True)
    
    # Reset
    progress_tracker.reset_progress()
    
    # Should be back to initial state
    assert progress_tracker.progress_data["total_symbols"] == 0
    assert progress_tracker.progress_data["completed_symbols"] == 0
    assert progress_tracker.progress_data["symbols"] == {}
    assert not Path(temp_progress_file).exists()


def test_export_progress_report(progress_tracker):
    """Test exporting progress report."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        report_file = f.name
    
    try:
        # Add some data
        progress_tracker.initialize_symbols(["AAPL", "MSFT"])
        progress_tracker.mark_symbol_complete("AAPL", success=True)
        
        # Export report
        progress_tracker.export_progress_report(report_file)
        
        # Verify report file
        assert Path(report_file).exists()
        
        with open(report_file, 'r') as f:
            report = json.load(f)
        
        assert "summary" in report
        assert "detailed_symbols" in report
        assert "export_time" in report
        assert report["summary"]["total_symbols"] == 2
        assert report["summary"]["completed_symbols"] == 1
        
    finally:
        Path(report_file).unlink(missing_ok=True)


def test_statistics_update(progress_tracker):
    """Test that statistics are updated correctly."""
    symbols = ["AAPL", "MSFT"]
    progress_tracker.initialize_symbols(symbols)
    
    # Complete one symbol with data
    progress_tracker.update_symbol_data("AAPL", data_points=1000, file_size_bytes=50000)
    progress_tracker.mark_symbol_complete("AAPL", success=True)
    
    # Fail one symbol
    progress_tracker.mark_symbol_complete("MSFT", success=False, error="Test error")
    
    stats = progress_tracker.progress_data["statistics"]
    
    assert stats["total_data_points"] == 1000
    assert stats["total_file_size_bytes"] == 50000
    assert stats["success_rate"] == 50.0  # 1 success out of 2 attempts


if __name__ == "__main__":
    pytest.main([__file__])