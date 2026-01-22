"""
Social Media Sentiment Tracker

Aggregates sentiment from:
- Reddit (r/CanadianInvestor, r/Baystreetbets, r/stocks)
- Twitter/X mentions
- StockTwits sentiment

Provides retail investor sentiment and mention volume for trading signals
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import random

logger = logging.getLogger(__name__)


class SocialSentimentTracker:
    """
    Tracks social media sentiment for stocks
    """
    
    def __init__(self, demo_mode: bool = True):
        """
        Initialize social sentiment tracker
        
        Args:
            demo_mode: If True, generates simulated data
        """
        self.demo_mode = demo_mode
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_expiry = timedelta(minutes=15)  # Cache for 15 minutes (social moves fast)
        
        logger.info(f"Social Sentiment Tracker initialized (demo_mode={demo_mode})")
    
    def get_social_sentiment(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get social media sentiment for given symbols
        
        Args:
            symbols: List of stock symbols
        
        Returns:
            Dict mapping symbol to:
                - score: -1 to 1 (negative to positive sentiment)
                - mention_volume: Number of mentions (last 24h)
                - trending: Boolean if stock is trending
                - sources: Dict with breakdown by source (reddit, twitter, stocktwits)
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
                data = self._fetch_social_data(symbol)
            
            # Cache it
            self.cache[symbol] = {
                'data': data,
                'timestamp': datetime.now()
            }
            
            results[symbol] = data
        
        return results
    
    def _get_demo_data(self, symbol: str) -> Dict[str, Any]:
        """Generate simulated social sentiment data"""
        
        # Most stocks have low mention volume (80%)
        if random.random() < 0.8:
            mention_volume = random.randint(0, 20)
            trending = False
        else:
            # 20% are actively discussed
            mention_volume = random.randint(50, 500)
            trending = mention_volume > 200
        
        # Sentiment skewed positive (retail is optimistic)
        if mention_volume > 0:
            sentiment_score = random.uniform(-0.3, 0.7)
        else:
            sentiment_score = 0.0
        
        # Generate source breakdown
        reddit_mentions = int(mention_volume * 0.5)
        twitter_mentions = int(mention_volume * 0.3)
        stocktwits_mentions = int(mention_volume * 0.2)
        
        return {
            'score': sentiment_score,
            'mention_volume': mention_volume,
            'trending': trending,
            'sources': {
                'reddit': {
                    'mentions': reddit_mentions,
                    'sentiment': sentiment_score * random.uniform(0.8, 1.2),
                },
                'twitter': {
                    'mentions': twitter_mentions,
                    'sentiment': sentiment_score * random.uniform(0.8, 1.2),
                },
                'stocktwits': {
                    'mentions': stocktwits_mentions,
                    'sentiment': sentiment_score * random.uniform(0.8, 1.2),
                },
            },
        }
    
    def _fetch_social_data(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch real social media data
        
        NOTE: This is a placeholder for future implementation.
        Real implementation would:
        1. Query Reddit API (r/CanadianInvestor, r/Baystreetbets, r/stocks)
        2. Search Twitter API for mentions
        3. Query StockTwits API
        4. Parse sentiment using NLP (positive/negative keywords or pre-trained model)
        5. Aggregate weighted by source reliability
        """
        logger.warning(f"Social media APIs not yet implemented for {symbol}, using demo data")
        return self._get_demo_data(symbol)


def create_social_tracker(demo_mode: bool = True) -> SocialSentimentTracker:
    """Factory function to create SocialSentimentTracker"""
    return SocialSentimentTracker(demo_mode=demo_mode)

