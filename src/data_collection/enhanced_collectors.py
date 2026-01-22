"""
Enhanced Data Collectors with Rate Limiting and Multi-Source Fallback

Implements robust data collection with automatic fallback between Yahoo Finance,
Alpha Vantage, and other sources when rate limits or errors occur.
"""

import time
import random
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import os
from pathlib import Path

from .alpha_vantage_collector import AlphaVantageCollector

logger = logging.getLogger(__name__)

class EnhancedYahooFinanceCollector:
    """Enhanced Yahoo Finance collector with rate limiting and error handling"""
    
    def __init__(self):
        self.session = self._setup_session()
        self.last_request_time = 0
        self.min_delay = 2.0  # Minimum 2 seconds between requests
        self.rate_limit_delay = 60  # Wait 60 seconds if rate limited
        self.max_retries = 3
        
        logger.info("🚀 Enhanced Yahoo Finance Collector initialized")
    
    def _setup_session(self) -> requests.Session:
        """Setup requests session with proper retry strategy and headers"""
        
        session = requests.Session()
        
        # Rotate user agents to avoid detection
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/121.0'
        ]
        
        session.headers.update({
            'User-Agent': random.choice(user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0'
        })
        
        # Retry strategy for network issues
        retry_strategy = Retry(
            total=3,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=1, pool_maxsize=1)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def _wait_for_rate_limit(self):
        """Implement intelligent rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_delay:
            wait_time = self.min_delay - time_since_last
            logger.debug(f"⏳ Rate limiting: waiting {wait_time:.1f}s")
            time.sleep(wait_time)
        
        self.last_request_time = time.time()
    
    def fetch_data(self, symbol: str, period: str = "max", interval: str = "1d") -> Optional[pd.DataFrame]:
        """Fetch data with enhanced error handling and rate limiting"""
        
        for attempt in range(self.max_retries):
            try:
                # Wait for rate limit
                self._wait_for_rate_limit()
                
                logger.debug(f"📥 Fetching {symbol} {interval} data (attempt {attempt + 1}/{self.max_retries})")
                
                # Create ticker with custom session
                ticker = yf.Ticker(symbol, session=self.session)
                
                # Fetch data with timeout
                data = ticker.history(period=period, interval=interval, timeout=30)
                
                if not data.empty:
                    logger.info(f"✅ Yahoo Finance: {symbol} {interval} - {len(data)} rows")
                    return data
                else:
                    logger.warning(f"⚠️ Yahoo Finance: Empty data for {symbol} {interval}")
                    
            except Exception as e:
                error_str = str(e).lower()
                
                if "429" in error_str or "rate" in error_str or "too many requests" in error_str:
                    logger.warning(f"🚫 Rate limited for {symbol}, waiting {self.rate_limit_delay}s...")
                    time.sleep(self.rate_limit_delay)
                    self.rate_limit_delay = min(self.rate_limit_delay * 1.5, 300)  # Exponential backoff, max 5 min
                elif "timeout" in error_str or "connection" in error_str:
                    logger.warning(f"🌐 Network issue for {symbol}: {e}")
                    time.sleep(5 * (attempt + 1))  # Progressive delay
                else:
                    logger.error(f"❌ Yahoo Finance error for {symbol}: {e}")
                
                if attempt == self.max_retries - 1:
                    logger.error(f"❌ All attempts failed for {symbol}")
        
        return None
    
    def test_connectivity(self) -> bool:
        """Test if Yahoo Finance is accessible"""
        try:
            # Test with a simple, reliable symbol
            test_data = self.fetch_data("SPY", period="5d", interval="1d")
            return test_data is not None and not test_data.empty
        except Exception as e:
            logger.error(f"❌ Yahoo Finance connectivity test failed: {e}")
            return False

class MultiSourceDataCollector:
    """Multi-source data collector with intelligent fallback"""
    
    def __init__(self):
        self.yahoo_collector = EnhancedYahooFinanceCollector()
        self.alpha_vantage_collector = None
        
        # Try to initialize Alpha Vantage if API key is available
        try:
            self.alpha_vantage_collector = AlphaVantageCollector()
            logger.info("✅ Alpha Vantage collector initialized")
        except Exception as e:
            logger.warning(f"⚠️ Alpha Vantage not available: {e}")
        
        # Source priority (1 = highest priority) - Alpha Vantage is PAID with 75 calls/min
        self.source_priority = {
            'alpha_vantage': 1,  # PRIMARY - Paid service, 75 calls/min
            'yahoo': 2           # BACKUP - Free but rate limited
        }
        
        # Track source health
        self.source_health = {
            'yahoo': True,
            'alpha_vantage': True
        }
        
        logger.info("🚀 Multi-Source Data Collector initialized")
    
    def _test_source_health(self, source: str) -> bool:
        """Test if a data source is healthy"""
        try:
            if source == 'yahoo':
                return self.yahoo_collector.test_connectivity()
            elif source == 'alpha_vantage' and self.alpha_vantage_collector:
                return self.alpha_vantage_collector.test_connectivity()
            return False
        except Exception as e:
            logger.error(f"❌ Health check failed for {source}: {e}")
            return False
    
    def _get_available_sources(self) -> List[str]:
        """Get list of available sources sorted by priority"""
        sources = []
        
        # Check Yahoo Finance
        if self.source_health['yahoo']:
            sources.append('yahoo')
        
        # Check Alpha Vantage (PRIORITY - paid service)
        if self.alpha_vantage_collector and self.source_health['alpha_vantage']:
            sources.append('alpha_vantage')
        
        # Sort by priority (Alpha Vantage first - it's paid and reliable)
        sources.sort(key=lambda x: self.source_priority.get(x, 999))
        
        return sources
    
    def fetch_data(self, symbol: str, period: str = "max", interval: str = "1d") -> Tuple[Optional[pd.DataFrame], str]:
        """Fetch data with automatic fallback between sources"""
        
        available_sources = self._get_available_sources()
        
        if not available_sources:
            logger.error("❌ No data sources available")
            return None, "no_sources"
        
        logger.info(f"📊 Fetching {symbol} {interval} data from {len(available_sources)} sources")
        
        for source in available_sources:
            try:
                logger.debug(f"🔄 Trying {source} for {symbol}")
                
                data = None
                
                if source == 'yahoo':
                    data = self.yahoo_collector.fetch_data(symbol, period, interval)
                elif source == 'alpha_vantage':
                    # Convert interval for Alpha Vantage
                    if interval == "1d":
                        data = self.alpha_vantage_collector.fetch_daily_data(symbol, "full")
                    elif interval in ["1min", "5min", "15min", "30min"]:
                        data = self.alpha_vantage_collector.fetch_intraday_data(symbol, interval)
                
                if data is not None and not data.empty:
                    logger.info(f"✅ Success: {symbol} from {source} - {len(data)} rows")
                    return data, source
                else:
                    logger.warning(f"⚠️ No data from {source} for {symbol}")
                    
            except Exception as e:
                logger.error(f"❌ Error from {source} for {symbol}: {e}")
                
                # Mark source as unhealthy if multiple failures
                self.source_health[source] = False
                
                # Re-test source health after a delay
                time.sleep(1)
                self.source_health[source] = self._test_source_health(source)
        
        logger.error(f"❌ All sources failed for {symbol}")
        return None, "all_failed"
    
    def get_source_status(self) -> Dict[str, Dict]:
        """Get status of all data sources"""
        status = {}
        
        for source in ['yahoo', 'alpha_vantage']:
            is_available = False
            is_healthy = self.source_health.get(source, False)
            
            if source == 'yahoo':
                is_available = True
            elif source == 'alpha_vantage':
                is_available = self.alpha_vantage_collector is not None
            
            status[source] = {
                'available': is_available,
                'healthy': is_healthy,
                'priority': self.source_priority.get(source, 999)
            }
        
        return status
    
    def refresh_source_health(self):
        """Refresh health status of all sources"""
        logger.info("🔄 Refreshing source health status...")
        
        for source in ['yahoo', 'alpha_vantage']:
            old_health = self.source_health[source]
            new_health = self._test_source_health(source)
            self.source_health[source] = new_health
            
            if old_health != new_health:
                status = "✅ HEALTHY" if new_health else "❌ UNHEALTHY"
                logger.info(f"🔄 {source.title()} status changed: {status}")

def test_enhanced_collectors():
    """Test the enhanced collectors system"""
    
    print("🧪 TESTING ENHANCED DATA COLLECTORS")
    print("=" * 50)
    
    # Initialize collector
    collector = MultiSourceDataCollector()
    
    # Show source status
    print("\n📊 Data Source Status:")
    status = collector.get_source_status()
    for source, info in status.items():
        available = "✅" if info['available'] else "❌"
        healthy = "✅" if info['healthy'] else "❌"
        print(f"   {source.title()}: Available {available} | Healthy {healthy} | Priority {info['priority']}")
    
    # Test symbols
    test_symbols = ["SPY", "RY.TO", "TD.TO", "SHOP.TO"]
    
    print(f"\n🧪 Testing {len(test_symbols)} symbols...")
    
    results = {}
    
    for symbol in test_symbols:
        print(f"\n📊 Testing {symbol}...")
        
        data, source = collector.fetch_data(symbol, period="5d", interval="1d")
        
        if data is not None and not data.empty:
            print(f"   ✅ SUCCESS from {source}: {len(data)} rows")
            print(f"   📅 Date range: {data.index.min()} to {data.index.max()}")
            print(f"   💰 Latest close: ${data['Close'].iloc[-1]:.2f}")
            results[symbol] = True
        else:
            print(f"   ❌ FAILED: No data from any source")
            results[symbol] = False
        
        # Small delay between symbols
        time.sleep(2)
    
    # Summary
    success_count = sum(results.values())
    total_count = len(results)
    success_rate = success_count / total_count * 100
    
    print(f"\n🎯 TEST RESULTS:")
    print(f"   Success: {success_count}/{total_count} ({success_rate:.1f}%)")
    
    if success_rate >= 75:
        print("   ✅ SYSTEM WORKING - Ready for data collection")
        return True
    else:
        print("   ⚠️ SYSTEM ISSUES - Some sources may need attention")
        
        # Refresh source health
        collector.refresh_source_health()
        return False

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run test
    test_enhanced_collectors()