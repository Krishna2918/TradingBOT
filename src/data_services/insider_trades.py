"""
Insider Trading Tracker - SEDI Scraper for Canadian Stocks

Tracks CEO/CFO buy/sell transactions from SEDI (System for Electronic Disclosure by Insiders)
Calculates insider sentiment score for trading signals

Data source: https://www.sedi.ca/sedi/
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import random

logger = logging.getLogger(__name__)


class InsiderTradesTracker:
    """
    Tracks insider trading activity for TSX/TSXV stocks
    """
    
    def __init__(self, demo_mode: bool = True):
        """
        Initialize insider trades tracker
        
        Args:
            demo_mode: If True, generates simulated data
        """
        self.demo_mode = demo_mode
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_expiry = timedelta(hours=6)  # Cache for 6 hours
        
        logger.info(f"Insider Trades Tracker initialized (demo_mode={demo_mode})")
    
    def get_insider_sentiment(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get insider trading sentiment for given symbols
        
        Args:
            symbols: List of stock symbols
        
        Returns:
            Dict mapping symbol to:
                - sentiment: -1 to 1 (negative = selling, positive = buying)
                - transactions: Number of recent transactions
                - net_value: Net $ value (positive = buying, negative = selling)
                - last_update: Timestamp of last transaction
        """
        results = {}
        
        for symbol in symbols:
            # Check cache
            cached = self.cache.get(symbol)
            if cached and (datetime.now() - cached['timestamp']) < self.cache_expiry:
                results[symbol] = cached['data']
                continue
            
            # Fetch fresh data
            if self.demo_mode:
                data = self._get_demo_data(symbol)
            else:
                data = self._scrape_sedi(symbol)
            
            # Cache it
            self.cache[symbol] = {
                'data': data,
                'timestamp': datetime.now()
            }
            
            results[symbol] = data
        
        return results
    
    def _get_demo_data(self, symbol: str) -> Dict[str, Any]:
        """Generate simulated insider trading data"""
        # Random but realistic patterns
        
        # 70% chance of no recent activity
        if random.random() < 0.7:
            return {
                'sentiment': 0.0,
                'transactions': 0,
                'net_value': 0.0,
                'last_update': None,
            }
        
        # 30% chance of activity
        transaction_count = random.randint(1, 5)
        
        # Buying vs selling (slight bias towards buying = 60/40)
        is_buying = random.random() < 0.6
        
        if is_buying:
            sentiment = random.uniform(0.5, 1.0)
            net_value = random.uniform(50_000, 2_000_000)
        else:
            sentiment = random.uniform(-1.0, -0.5)
            net_value = random.uniform(-2_000_000, -50_000)
        
        return {
            'sentiment': sentiment,
            'transactions': transaction_count,
            'net_value': net_value,
            'last_update': datetime.now() - timedelta(days=random.randint(1, 30)),
        }
    
    def _scrape_sedi(self, symbol: str) -> Dict[str, Any]:
        """
        Scrape SEDI for real insider trading data
        
        NOTE: This is a placeholder for future implementation.
        Real implementation would:
        1. Query SEDI website (https://www.sedi.ca/sedi/)
        2. Parse HTML/XML responses
        3. Filter for recent transactions (last 90 days)
        4. Calculate net buying/selling
        5. Weight by transaction size and insider role (CEO/CFO > Director)
        """
        logger.warning(f"SEDI scraping not yet implemented for {symbol}, using demo data")
        return self._get_demo_data(symbol)


def create_insider_tracker(demo_mode: bool = True) -> InsiderTradesTracker:
    """Factory function to create InsiderTradesTracker"""
    return InsiderTradesTracker(demo_mode=demo_mode)

