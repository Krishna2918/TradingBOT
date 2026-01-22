"""
Data Collection Module

Comprehensive historical and intraday data collection system for Canadian AI Trading Bot.
Collects 20 years of TSX/TSXV data using yfinance with intelligent progress tracking.
"""

from .symbol_manager import SymbolManager, get_tsx_symbols
from .progress_tracker import ProgressTracker
from .storage_manager import StorageManager
from .historical_appender import HistoricalAppender
from .intraday_appender import IntradayAppender
from .data_validator import DataValidator

__all__ = [
    'SymbolManager',
    'get_tsx_symbols', 
    'ProgressTracker',
    'StorageManager',
    'HistoricalAppender',
    'IntradayAppender',
    'DataValidator'
]

__version__ = "1.0.0"