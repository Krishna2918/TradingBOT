"""
Alpha Vantage Data Collector

Alternative data source using Alpha Vantage API for reliable historical data collection.
Complements yfinance with professional-grade API service.
"""

import logging
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class AlphaVantageCollector:
    """Professional data collection using Alpha Vantage API"""
    
    def __init__(self, api_key: Optional[str] = None):
        # Get API key from environment or parameter
        self.api_key = api_key or os.getenv('ALPHA_VANTAGE_API_KEY')
        
        if not self.api_key:
            raise ValueError("Alpha Vantage API key not found. Set ALPHA_VANTAGE_API_KEY environment variable.")
        
        self.base_url = "https://www.alphavantage.co/query"
        self.rate_limit_delay = 0.8  # 75 requests per minute = 0.8 seconds between requests (premium)
        
        logger.info(f"🔑 Alpha Vantage Collector initialized with API key: {self.api_key[:8]}...")
    
    def _make_request(self, params: Dict) -> Optional[Dict]:
        """Make API request with rate limiting"""
        try:
            # Add API key to parameters
            params['apikey'] = self.api_key
            
            # Make request
            logger.debug(f"📡 Alpha Vantage request: {params.get('function')} for {params.get('symbol')}")
            response = requests.get(self.base_url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check for API errors
                if "Error Message" in data:
                    logger.error(f"❌ Alpha Vantage error: {data['Error Message']}")
                    return None
                
                if "Note" in data:
                    logger.warning(f"⚠️ Alpha Vantage note: {data['Note']}")
                    return None
                
                logger.debug(f"✅ Alpha Vantage response received")
                return data
            else:
                logger.error(f"❌ Alpha Vantage HTTP error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Alpha Vantage request failed: {e}")
            return None
        finally:
            # Rate limiting
            time.sleep(self.rate_limit_delay)
    
    def fetch_daily_data(self, symbol: str, outputsize: str = "full") -> Optional[pd.DataFrame]:
        """Fetch daily historical data (20+ years available)"""
        
        # Remove .TO suffix for Alpha Vantage (they use different format)
        av_symbol = symbol.replace('.TO', '.TRT')  # Toronto exchange format
        
        params = {
            'function': 'TIME_SERIES_DAILY_ADJUSTED',
            'symbol': av_symbol,
            'outputsize': outputsize  # 'compact' = 100 days, 'full' = 20+ years
        }
        
        data = self._make_request(params)
        
        if not data or 'Time Series (Daily)' not in data:
            logger.error(f"❌ No daily data returned for {symbol}")
            return None
        
        # Convert to DataFrame
        time_series = data['Time Series (Daily)']
        
        df_data = []
        for date_str, values in time_series.items():
            df_data.append({
                'Date': pd.to_datetime(date_str),
                'Open': float(values['1. open']),
                'High': float(values['2. high']),
                'Low': float(values['3. low']),
                'Close': float(values['4. close']),
                'Adj_Close': float(values['5. adjusted close']),
                'Volume': int(values['6. volume'])
            })
        
        df = pd.DataFrame(df_data)
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        
        # Use adjusted close as close price
        df['Close'] = df['Adj_Close']
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        logger.info(f"✅ Alpha Vantage daily data for {symbol}: {len(df)} rows from {df.index.min()} to {df.index.max()}")
        return df
    
    def fetch_intraday_data(self, symbol: str, interval: str = "5min") -> Optional[pd.DataFrame]:
        """Fetch intraday data (last 30 days)"""
        
        # Remove .TO suffix for Alpha Vantage
        av_symbol = symbol.replace('.TO', '.TRT')
        
        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': av_symbol,
            'interval': interval,
            'outputsize': 'full'  # Last 30 days
        }
        
        data = self._make_request(params)
        
        time_series_key = f'Time Series ({interval})'
        if not data or time_series_key not in data:
            logger.error(f"❌ No intraday data returned for {symbol} {interval}")
            return None
        
        # Convert to DataFrame
        time_series = data[time_series_key]
        
        df_data = []
        for datetime_str, values in time_series.items():
            df_data.append({
                'Datetime': pd.to_datetime(datetime_str),
                'Open': float(values['1. open']),
                'High': float(values['2. high']),
                'Low': float(values['3. low']),
                'Close': float(values['4. close']),
                'Volume': int(values['5. volume'])
            })
        
        df = pd.DataFrame(df_data)
        df.set_index('Datetime', inplace=True)
        df.sort_index(inplace=True)
        
        logger.info(f"✅ Alpha Vantage {interval} data for {symbol}: {len(df)} rows")
        return df
    
    def test_connectivity(self) -> bool:
        """Test Alpha Vantage API connectivity"""
        try:
            # Test with a simple request
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': 'RY.TRT',  # Royal Bank in Alpha Vantage format
                'outputsize': 'compact'
            }
            
            data = self._make_request(params)
            
            if data and 'Time Series (Daily)' in data:
                logger.info("✅ Alpha Vantage connectivity test: SUCCESS")
                return True
            else:
                logger.error("❌ Alpha Vantage connectivity test: FAILED")
                return False
                
        except Exception as e:
            logger.error(f"❌ Alpha Vantage connectivity test error: {e}")
            return False

# For testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    try:
        collector = AlphaVantageCollector()
        
        # Test connectivity
        if collector.test_connectivity():
            print("✅ Alpha Vantage API is working!")
            
            # Test data collection
            print("🧪 Testing data collection...")
            data = collector.fetch_daily_data("RY.TO", outputsize="compact")
            
            if data is not None and not data.empty:
                print(f"✅ Successfully fetched {len(data)} rows of RY.TO daily data")
                print(f"   Date range: {data.index.min()} to {data.index.max()}")
                print(f"   Latest close: ${data['Close'].iloc[-1]:.2f}")
            else:
                print("❌ No data returned")
        else:
            print("❌ Alpha Vantage API connectivity failed")
            
    except Exception as e:
        print(f"❌ Alpha Vantage test failed: {e}")
        print("Check your ALPHA_VANTAGE_API_KEY environment variable")