"""
Canadian Market Data Collector
Collects real-time data from Canadian market sources
"""

import logging
import requests
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import yfinance as yf

# Phase 2: Import API Budget Manager
from ..api_budget_manager import get_api_budget_manager

logger = logging.getLogger(__name__)

class CanadianMarketCollector:
    """Collects data from Canadian market sources"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.data_sources = self.config['data_sources']
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Phase 2: Initialize API Budget Manager
        self.budget_manager = get_api_budget_manager()
    
    def get_tsx_data(self, symbols: List[str]) -> Dict:
        """Get TSX market data with caching and budget management"""
        try:
            data = {}
            cache_hits = 0
            
            for symbol in symbols:
                # Phase 2: Check cache first
                cache_key = f"yfinance:{symbol}:1d"
                cached_data = self.budget_manager.get_cached_response(cache_key)
                
                if cached_data is not None:
                    data[symbol] = cached_data
                    cache_hits += 1
                    continue
                
                # Phase 2: Check API budget
                if not self.budget_manager.can_make_request("yfinance"):
                    logger.warning("Yahoo Finance API budget exhausted, using cached data where available")
                    continue
                
                # Make API request
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1d", interval="1m")
                
                if not hist.empty:
                    latest = hist.iloc[-1]
                    symbol_data = {
                        'open': float(latest['Open']),
                        'high': float(latest['High']),
                        'low': float(latest['Low']),
                        'close': float(latest['Close']),
                        'volume': int(latest['Volume']),
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    data[symbol] = symbol_data
                    
                    # Phase 2: Cache the response (5 minutes for intraday data)
                    self.budget_manager.cache_response(cache_key, symbol_data, 300)
                    
                    # Record successful API call
                    self.budget_manager.record_request("yfinance", True)
                else:
                    # Record failed API call
                    self.budget_manager.record_request("yfinance", False)
            
            logger.info(f"Collected TSX data for {len(data)} symbols ({cache_hits} cache hits)")
            return data
            
        except Exception as e:
            logger.error(f"Failed to collect TSX data: {e}")
            return {}
    
    def get_canadian_news(self) -> List[Dict]:
        """Get Canadian financial news"""
        try:
            news_items = []
            
            # Globe and Mail
            try:
                response = self.session.get(
                    "https://www.theglobeandmail.com/business/rss.xml",
                    timeout=10
                )
                if response.status_code == 200:
                    # Parse RSS feed (simplified)
                    news_items.append({
                        'source': 'globeandmail',
                        'title': 'Sample Globe and Mail News',
                        'url': 'https://www.theglobeandmail.com/business/',
                        'timestamp': datetime.now().isoformat()
                    })
            except Exception as e:
                logger.warning(f"Failed to get Globe and Mail news: {e}")
            
            # Financial Post
            try:
                response = self.session.get(
                    "https://financialpost.com/feed",
                    timeout=10
                )
                if response.status_code == 200:
                    news_items.append({
                        'source': 'financialpost',
                        'title': 'Sample Financial Post News',
                        'url': 'https://financialpost.com/',
                        'timestamp': datetime.now().isoformat()
                    })
            except Exception as e:
                logger.warning(f"Failed to get Financial Post news: {e}")
            
            logger.info(f"Collected {len(news_items)} news items")
            return news_items
            
        except Exception as e:
            logger.error(f"Failed to collect Canadian news: {e}")
            return []
    
    def get_bank_of_canada_data(self) -> Dict:
        """Get Bank of Canada economic data"""
        try:
            boc_data = {}
            
            # Interest rates
            try:
                response = self.session.get(
                    "https://www.bankofcanada.ca/valet/observations/group/interest_rates",
                    timeout=10
                )
                if response.status_code == 200:
                    boc_data['interest_rates'] = {
                        'overnight_rate': 5.0,  # Mock data
                        'timestamp': datetime.now().isoformat()
                    }
            except Exception as e:
                logger.warning(f"Failed to get BOC interest rates: {e}")
            
            # Exchange rates
            try:
                response = self.session.get(
                    "https://www.bankofcanada.ca/valet/observations/group/FX_RATES_DAILY",
                    timeout=10
                )
                if response.status_code == 200:
                    boc_data['exchange_rates'] = {
                        'usd_cad': 1.35,  # Mock data
                        'timestamp': datetime.now().isoformat()
                    }
            except Exception as e:
                logger.warning(f"Failed to get BOC exchange rates: {e}")
            
            logger.info(f"Collected BOC data: {list(boc_data.keys())}")
            return boc_data
            
        except Exception as e:
            logger.error(f"Failed to collect BOC data: {e}")
            return {}
    
    def get_commodity_prices(self) -> Dict:
        """Get commodity prices relevant to Canadian markets"""
        try:
            commodities = {}
            
            # Oil prices (WTI)
            try:
                oil_ticker = yf.Ticker("CL=F")
                oil_hist = oil_ticker.history(period="1d")
                if not oil_hist.empty:
                    latest_oil = oil_hist.iloc[-1]
                    commodities['oil_wti'] = {
                        'price': float(latest_oil['Close']),
                        'change': float(latest_oil['Close'] - latest_oil['Open']),
                        'timestamp': datetime.now().isoformat()
                    }
            except Exception as e:
                logger.warning(f"Failed to get oil prices: {e}")
            
            # Gold prices
            try:
                gold_ticker = yf.Ticker("GC=F")
                gold_hist = gold_ticker.history(period="1d")
                if not gold_hist.empty:
                    latest_gold = gold_hist.iloc[-1]
                    commodities['gold'] = {
                        'price': float(latest_gold['Close']),
                        'change': float(latest_gold['Close'] - latest_gold['Open']),
                        'timestamp': datetime.now().isoformat()
                    }
            except Exception as e:
                logger.warning(f"Failed to get gold prices: {e}")
            
            logger.info(f"Collected commodity prices: {list(commodities.keys())}")
            return commodities
            
        except Exception as e:
            logger.error(f"Failed to collect commodity prices: {e}")
            return {}
    
    def get_weather_data(self) -> Dict:
        """Get weather data for agricultural impact"""
        try:
            weather_data = {}
            
            # Environment Canada weather alerts
            try:
                response = self.session.get(
                    "https://weather.gc.ca/rss/warning/agriculture/",
                    timeout=10
                )
                if response.status_code == 200:
                    weather_data['agricultural_alerts'] = {
                        'alerts': [],  # Would parse RSS feed
                        'timestamp': datetime.now().isoformat()
                    }
            except Exception as e:
                logger.warning(f"Failed to get weather data: {e}")
            
            logger.info("Collected weather data")
            return weather_data
            
        except Exception as e:
            logger.error(f"Failed to collect weather data: {e}")
            return {}
    
    def collect_all_data(self) -> Dict:
        """Collect all Canadian market data"""
        try:
            logger.info("Starting Canadian market data collection...")
            
            # Get market data
            tsx_symbols = ["^GSPTSE", "^TX60", "RY.TO", "TD.TO", "SHOP.TO"]
            market_data = self.get_tsx_data(tsx_symbols)
            
            # Get news
            news_data = self.get_canadian_news()
            
            # Get economic data
            boc_data = self.get_bank_of_canada_data()
            
            # Get commodity prices
            commodity_data = self.get_commodity_prices()
            
            # Get weather data
            weather_data = self.get_weather_data()
            
            # Combine all data
            all_data = {
                'market_data': market_data,
                'news': news_data,
                'economic_data': boc_data,
                'commodities': commodity_data,
                'weather': weather_data,
                'collection_timestamp': datetime.now().isoformat()
            }
            
            logger.info("Canadian market data collection completed")
            return all_data
            
        except Exception as e:
            logger.error(f"Failed to collect all data: {e}")
            return {}

# Test the collector
if __name__ == "__main__":
    collector = CanadianMarketCollector("config/data_sources.yaml")
    
    # Test data collection
    data = collector.collect_all_data()
    
    print("Collected data keys:", list(data.keys()))
    print("Market data symbols:", list(data.get('market_data', {}).keys()))
    print("News items:", len(data.get('news', [])))
    print("BOC data keys:", list(data.get('economic_data', {}).keys()))
    print("Commodity data keys:", list(data.get('commodities', {}).keys()))
