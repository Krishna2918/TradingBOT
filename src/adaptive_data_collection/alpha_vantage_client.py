"""
Alpha Vantage API client with authentication, rate limiting, and retry logic.
"""

import time
import requests
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import json
from .interfaces import DataCollector
from .config import CollectionConfig
from .retry_manager import IntelligentRetryManager
import pandas as pd


class AlphaVantageClient(DataCollector):
    """Alpha Vantage API client with rate limiting and authentication."""
    
    def __init__(self, config: CollectionConfig):
        self.config = config
        self.api_key = config.alpha_vantage_api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.session = requests.Session()
        self.logger = logging.getLogger(__name__)
        
        # Initialize retry manager
        self.retry_manager = IntelligentRetryManager(config)
        
        # Set up session headers
        self.session.headers.update({
            'User-Agent': 'AdaptiveDataCollection/1.0.0',
            'Accept': 'application/json'
        })
        
        # Rate limiting state
        self.last_request_time = 0.0
        self.request_interval = 60.0 / config.alpha_vantage_rpm  # seconds between requests
        
        self.logger.info(f"Initialized Alpha Vantage client with {config.alpha_vantage_rpm} RPM limit")
    
    def _wait_for_rate_limit(self) -> None:
        """Wait if necessary to respect rate limits."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.request_interval:
            wait_time = self.request_interval - time_since_last
            self.logger.debug(f"Rate limiting: waiting {wait_time:.2f} seconds")
            time.sleep(wait_time)
        
        self.last_request_time = time.time()
    
    def _make_request(self, params: Dict[str, str]) -> Dict[str, Any]:
        """Make authenticated request to Alpha Vantage API."""
        self._wait_for_rate_limit()
        
        # Add API key to parameters
        params['apikey'] = self.api_key
        
        start_time = time.time()
        
        try:
            response = self.session.get(
                self.base_url,
                params=params,
                timeout=30
            )
            
            response_time = time.time() - start_time
            self.logger.debug(f"API request completed in {response_time:.2f}s")
            
            response.raise_for_status()
            data = response.json()
            
            # Check for API errors
            if "Error Message" in data:
                raise ValueError(f"Alpha Vantage API error: {data['Error Message']}")
            
            if "Note" in data:
                if "API call frequency" in data["Note"]:
                    raise ValueError(f"Rate limit exceeded: {data['Note']}")
                else:
                    self.logger.warning(f"Alpha Vantage note: {data['Note']}")
            
            # Log successful request
            self.logger.info(f"Successful API request: {params.get('function', 'unknown')} for {params.get('symbol', 'unknown')}")
            
            return data
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed: {e}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {e}")
            raise ValueError(f"Invalid JSON response from Alpha Vantage: {e}")
    
    def collect_ticker_data(self, symbol: str) -> pd.DataFrame:
        """Collect daily historical data for a ticker with retry logic."""
        self.logger.info(f"Collecting data for {symbol}")
        
        def _collect_data():
            params = {
                'function': 'TIME_SERIES_DAILY_ADJUSTED',
                'symbol': symbol,
                'outputsize': 'full'  # Get full historical data
            }
            
            data = self._make_request(params)
            
            # Extract time series data
            time_series_key = "Time Series (Daily)"
            if time_series_key not in data:
                available_keys = list(data.keys())
                self.logger.error(f"Expected key '{time_series_key}' not found. Available keys: {available_keys}")
                raise ValueError(f"Unexpected response format for {symbol}")
            
            time_series = data[time_series_key]
            
            # Convert to DataFrame
            df_data = []
            for date_str, values in time_series.items():
                try:
                    df_data.append({
                        'symbol': symbol,
                        'date': datetime.strptime(date_str, '%Y-%m-%d'),
                        'open': float(values['1. open']),
                        'high': float(values['2. high']),
                        'low': float(values['3. low']),
                        'close': float(values['4. close']),
                        'adj_close': float(values['5. adjusted close']),
                        'volume': int(values['6. volume']),
                        'source': 'alpha_vantage'
                    })
                except (KeyError, ValueError) as e:
                    self.logger.warning(f"Skipping invalid data point for {symbol} on {date_str}: {e}")
                    continue
            
            if not df_data:
                raise ValueError(f"No valid data points found for {symbol}")
            
            df = pd.DataFrame(df_data)
            
            # Filter to requested years
            cutoff_date = datetime.now() - timedelta(days=self.config.years_to_collect * 365)
            df = df[df['date'] >= cutoff_date]
            
            # Sort by date ascending
            df = df.sort_values('date').reset_index(drop=True)
            
            self.logger.info(f"Collected {len(df)} data points for {symbol} from {df['date'].min()} to {df['date'].max()}")
            
            return df
        
        # Execute with retry logic
        return self.retry_manager.execute_with_retry(
            operation=_collect_data,
            operation_name=f"collect_data_{symbol}"
        )
    
    def get_available_symbols(self) -> list[str]:
        """Get list of available symbols from the symbols file."""
        try:
            with open(self.config.us_symbols_file, 'r') as f:
                symbols = [line.strip() for line in f if line.strip()]
            
            self.logger.info(f"Loaded {len(symbols)} symbols from {self.config.us_symbols_file}")
            return symbols
            
        except FileNotFoundError:
            self.logger.error(f"Symbols file not found: {self.config.us_symbols_file}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to load symbols: {e}")
            raise
    
    def validate_symbol(self, symbol: str) -> bool:
        """Validate if symbol is available for collection."""
        # Basic validation - check if symbol is in our list
        available_symbols = self.get_available_symbols()
        return symbol in available_symbols
    
    def test_connection(self) -> bool:
        """Test API connection with a simple request and retry logic."""
        def _test_connection():
            # Use a simple quote request to test connection
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': 'AAPL'  # Use AAPL as test symbol
            }
            
            data = self._make_request(params)
            
            # Check if we got valid data
            if "Global Quote" in data:
                self.logger.info("Alpha Vantage API connection test successful")
                return True
            else:
                self.logger.error(f"Unexpected response format in connection test: {data}")
                raise ValueError("Unexpected response format in connection test")
        
        try:
            return self.retry_manager.execute_with_retry(
                operation=_test_connection,
                operation_name="connection_test"
            )
        except Exception as e:
            self.logger.error(f"Alpha Vantage API connection test failed after retries: {e}")
            return False
    
    def get_api_status(self) -> Dict[str, Any]:
        """Get current API status and usage information."""
        return {
            'api_key_set': bool(self.api_key),
            'rate_limit_rpm': self.config.alpha_vantage_rpm,
            'request_interval': self.request_interval,
            'last_request_time': self.last_request_time,
            'time_since_last_request': time.time() - self.last_request_time if self.last_request_time > 0 else None,
            'retry_statistics': self.retry_manager.get_retry_statistics()
        }
    
    def get_retry_statistics(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get retry statistics for a specific symbol or all operations."""
        if symbol:
            operation_name = f"collect_data_{symbol}"
            return self.retry_manager.get_retry_statistics(operation_name)
        return self.retry_manager.get_retry_statistics()
    
    def get_recent_failures(self, hours: int = 24) -> list:
        """Get recent failures within the specified time window."""
        failures = self.retry_manager.get_recent_failures(hours)
        return [
            {
                'timestamp': failure.timestamp,
                'attempt': failure.attempt,
                'error_type': failure.error_type.value,
                'error_message': failure.error_message,
                'delay': failure.delay
            }
            for failure in failures
        ]
    
    def clear_retry_history(self, symbol: Optional[str] = None) -> None:
        """Clear retry history for a specific symbol or all operations."""
        if symbol:
            operation_name = f"collect_data_{symbol}"
            self.retry_manager.clear_history(operation_name)
        else:
            self.retry_manager.clear_history()