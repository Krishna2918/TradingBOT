"""
Interface for market data providers.

This interface ensures consistent data access across all components
and enables integration with multiple data sources while respecting
API rate limits.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import pandas as pd
from datetime import datetime, timedelta


class IDataProvider(ABC):
    """
    Interface for market data providers.
    
    All data providers must implement this interface to ensure consistent
    data access and enable seamless switching between data sources.
    """
    
    @abstractmethod
    def get_market_data(
        self, 
        symbols: List[str], 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        frequency: str = 'daily'
    ) -> pd.DataFrame:
        """
        Retrieve market data for specified symbols.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date for data retrieval
            end_date: End date for data retrieval  
            frequency: Data frequency ('daily', 'hourly', 'minute')
            
        Returns:
            DataFrame with OHLCV data indexed by date
            
        Raises:
            DataError: If data retrieval fails
        """
        pass
    
    @abstractmethod
    def get_factor_data(
        self, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Retrieve factor data (momentum, value, quality, etc.).
        
        Args:
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            
        Returns:
            DataFrame with factor returns indexed by date
            
        Raises:
            DataError: If factor data retrieval fails
        """
        pass
    
    @abstractmethod
    def get_risk_free_rate(
        self, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.Series:
        """
        Retrieve risk-free rate data.
        
        Args:
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            
        Returns:
            Series with risk-free rates indexed by date
        """
        pass
    
    @abstractmethod
    def get_sector_data(self, symbols: List[str]) -> Dict[str, str]:
        """
        Get sector classification for symbols.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary mapping symbol to sector
        """
        pass
    
    @abstractmethod
    def get_market_cap_data(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get market capitalization data for symbols.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary mapping symbol to market cap
        """
        pass
    
    @abstractmethod
    def get_trading_volume_data(
        self, 
        symbols: List[str],
        lookback_days: int = 30
    ) -> Dict[str, float]:
        """
        Get average daily trading volume for symbols.
        
        Args:
            symbols: List of stock symbols
            lookback_days: Number of days to average over
            
        Returns:
            Dictionary mapping symbol to average daily volume
        """
        pass
    
    @abstractmethod
    def is_data_stale(self, symbol: str, max_age_minutes: int = 60) -> bool:
        """
        Check if data for a symbol is stale.
        
        Args:
            symbol: Stock symbol to check
            max_age_minutes: Maximum acceptable data age in minutes
            
        Returns:
            True if data is stale, False otherwise
        """
        pass
    
    @abstractmethod
    def get_data_quality_score(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get data quality scores for symbols.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary mapping symbol to quality score (0-1)
        """
        pass
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get the name of the data provider"""
        pass
    
    @property
    @abstractmethod
    def rate_limit_per_minute(self) -> int:
        """Get the rate limit for this provider (calls per minute)"""
        pass
    
    @property
    @abstractmethod
    def supported_frequencies(self) -> List[str]:
        """Get list of supported data frequencies"""
        pass
    
    def calculate_returns(
        self, 
        price_data: pd.DataFrame, 
        method: str = 'simple'
    ) -> pd.DataFrame:
        """
        Calculate returns from price data.
        
        Args:
            price_data: DataFrame with price data
            method: Return calculation method ('simple' or 'log')
            
        Returns:
            DataFrame with calculated returns
        """
        if method == 'log':
            return np.log(price_data / price_data.shift(1)).dropna()
        else:
            return (price_data / price_data.shift(1) - 1).dropna()
    
    def validate_symbols(self, symbols: List[str]) -> Tuple[List[str], List[str]]:
        """
        Validate symbols and return valid/invalid lists.
        
        Args:
            symbols: List of symbols to validate
            
        Returns:
            Tuple of (valid_symbols, invalid_symbols)
        """
        valid_symbols = []
        invalid_symbols = []
        
        for symbol in symbols:
            try:
                # Try to get recent data for validation
                data = self.get_market_data([symbol], 
                                          start_date=datetime.now() - timedelta(days=5))
                if not data.empty:
                    valid_symbols.append(symbol)
                else:
                    invalid_symbols.append(symbol)
            except Exception:
                invalid_symbols.append(symbol)
        
        return valid_symbols, invalid_symbols