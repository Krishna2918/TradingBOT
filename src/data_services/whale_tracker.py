"""
Whale Tracker - Institutional Investor & Famous Investor Tracking

Tracks:
- 13F filings (Warren Buffett, Cathie Wood, Bill Ackman, etc.)
- Canadian pension funds (CPP, OMERS)
- ETF flows (large institutional moves)
- Hedge fund 13D filings (activist positions)

Provides "follow the smart money" signals
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import random

logger = logging.getLogger(__name__)


class WhaleTracker:
    """
    Tracks institutional investor and famous investor activity
    """
    
    def __init__(self, demo_mode: bool = True):
        """
        Initialize whale tracker
        
        Args:
            demo_mode: If True, generates simulated data
        """
        self.demo_mode = demo_mode
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_expiry = timedelta(days=1)  # Cache for 1 day (13F updates quarterly)
        
        # Famous investors to track
        self.famous_investors = [
            'Warren Buffett (Berkshire Hathaway)',
            'Cathie Wood (ARK Invest)',
            'Bill Ackman (Pershing Square)',
            'CPP Investments (Canada Pension Plan)',
            'OMERS (Ontario Municipal)',
            'Caisse de dépôt (Quebec)',
        ]
        
        logger.info(f"Whale Tracker initialized (demo_mode={demo_mode})")
        logger.info(f"  Tracking: {', '.join(self.famous_investors)}")
    
    def get_whale_activity(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get whale/institutional activity for given symbols
        
        Args:
            symbols: List of stock symbols
        
        Returns:
            Dict mapping symbol to:
                - flow: -1 to 1 (negative = selling, positive = buying)
                - net_buying: Net shares/$ bought (positive) or sold (negative)
                - notable_investors: List of famous investors with positions
                - etf_flows: Net ETF flows
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
                data = self._fetch_whale_data(symbol)
            
            # Cache it
            self.cache[symbol] = {
                'data': data,
                'timestamp': datetime.now()
            }
            
            results[symbol] = data
        
        return results
    
    def _get_demo_data(self, symbol: str) -> Dict[str, Any]:
        """Generate simulated whale activity data"""
        
        # 85% of stocks have no notable whale activity
        if random.random() < 0.85:
            return {
                'flow': 0.0,
                'net_buying': 0,
                'notable_investors': [],
                'etf_flows': 0,
            }
        
        # 15% have whale activity
        is_buying = random.random() < 0.6  # 60% chance buying vs 40% selling
        
        if is_buying:
            flow_score = random.uniform(0.4, 0.9)
            net_buying = random.randint(1_000_000, 50_000_000)
            etf_flows = random.randint(500_000, 10_000_000)
        else:
            flow_score = random.uniform(-0.9, -0.4)
            net_buying = random.randint(-50_000_000, -1_000_000)
            etf_flows = random.randint(-10_000_000, -500_000)
        
        # Pick random notable investors (1-3)
        num_investors = random.randint(1, 3)
        notable_investors = random.sample(self.famous_investors, num_investors)
        
        return {
            'flow': flow_score,
            'net_buying': net_buying,
            'notable_investors': notable_investors,
            'etf_flows': etf_flows,
        }
    
    def _fetch_whale_data(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch real whale/institutional data
        
        NOTE: This is a placeholder for future implementation.
        Real implementation would:
        1. Query SEC EDGAR for 13F filings (quarterly)
        2. Parse XML/text filings for position changes
        3. Track CPP/OMERS public disclosures
        4. Query ETF.com or similar for ETF flow data
        5. Calculate net institutional buying/selling
        6. Weight by investor reputation (Buffett > random hedge fund)
        """
        logger.warning(f"Whale tracking APIs not yet implemented for {symbol}, using demo data")
        return self._get_demo_data(symbol)


def create_whale_tracker(demo_mode: bool = True) -> WhaleTracker:
    """Factory function to create WhaleTracker"""
    return WhaleTracker(demo_mode=demo_mode)

