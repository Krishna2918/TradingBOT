"""
Core interfaces for the Adaptive Data Collection System.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd
from dataclasses import dataclass


@dataclass
class MarketDataPoint:
    """Raw market data point."""
    symbol: str
    date: datetime
    open: float
    high: float
    low: float
    close: float
    adj_close: float
    volume: int
    source: str


@dataclass
class EnhancedMarketData(MarketDataPoint):
    """Enhanced market data with technical indicators."""
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    rsi_14: Optional[float] = None
    atr_14: Optional[float] = None


@dataclass
class CatalogEntry:
    """Data catalog entry."""
    file_path: str
    symbol: str
    market: str
    row_count: int
    start_date: datetime
    end_date: datetime
    collection_timestamp: datetime
    data_source: str
    file_size_bytes: int
    indicators_included: List[str]


class DataCollector(ABC):
    """Abstract base class for data collectors."""
    
    @abstractmethod
    def collect_ticker_data(self, symbol: str) -> pd.DataFrame:
        """Collect data for a single ticker."""
        pass
    
    @abstractmethod
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols to collect."""
        pass
    
    @abstractmethod
    def validate_symbol(self, symbol: str) -> bool:
        """Validate if symbol is available for collection."""
        pass


class DataProcessor(ABC):
    """Abstract base class for data processors."""
    
    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean raw data."""
        pass
    
    @abstractmethod
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators."""
        pass
    
    @abstractmethod
    def sort_and_clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """Sort data by date and perform final cleaning."""
        pass


class StorageManager(ABC):
    """Abstract base class for storage managers."""
    
    @abstractmethod
    def save_ticker_data(self, symbol: str, data: pd.DataFrame) -> str:
        """Save ticker data to storage and return file path."""
        pass
    
    @abstractmethod
    def update_catalog(self, entry: CatalogEntry) -> None:
        """Update the data catalog with new entry."""
        pass
    
    @abstractmethod
    def get_catalog(self) -> pd.DataFrame:
        """Get the current data catalog."""
        pass
    
    @abstractmethod
    def file_exists(self, symbol: str) -> bool:
        """Check if data file exists for symbol."""
        pass


class ProgressTracker(ABC):
    """Abstract base class for progress tracking."""
    
    @abstractmethod
    def save_progress(self, progress_data: Dict[str, Any]) -> None:
        """Save current progress state."""
        pass
    
    @abstractmethod
    def load_progress(self) -> Dict[str, Any]:
        """Load previous progress state."""
        pass
    
    @abstractmethod
    def mark_symbol_complete(self, symbol: str, success: bool, error: Optional[str] = None) -> None:
        """Mark a symbol as complete (success or failure)."""
        pass
    
    @abstractmethod
    def get_pending_symbols(self) -> List[str]:
        """Get list of symbols that still need to be processed."""
        pass


class RateLimiter(ABC):
    """Abstract base class for rate limiting."""
    
    @abstractmethod
    def acquire(self) -> None:
        """Acquire permission to make a request (blocks if necessary)."""
        pass
    
    @abstractmethod
    def get_wait_time(self) -> float:
        """Get current wait time until next request is allowed."""
        pass
    
    @abstractmethod
    def update_rate_limit(self, new_rpm: int) -> None:
        """Update the rate limit dynamically."""
        pass


class RetryManager(ABC):
    """Abstract base class for retry management."""
    
    @abstractmethod
    def should_retry(self, error: Exception, attempt: int) -> bool:
        """Determine if an error should be retried."""
        pass
    
    @abstractmethod
    def get_backoff_delay(self, attempt: int) -> float:
        """Get delay before next retry attempt."""
        pass
    
    @abstractmethod
    def classify_error(self, error: Exception) -> str:
        """Classify error as temporary, permanent, or system."""
        pass