"""
Free API Services Integration
Integrates News API, Alpha Vantage, and Reddit API for enhanced data
"""

import requests
import os
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FreeAPIsIntegration:
    """Integration for free API services: News API, Alpha Vantage, Reddit API"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_keys = config.get('api_keys', {})
        self.rate_limits = config.get('rate_limits', {})
        
        # Rate limiting tracking
        self.last_request = {
            'newsapi': 0,
            'alpha_vantage': 0,
            'reddit': 0,
            'finnhub': 0
        }
        
        # Request counters
        self.request_counts = {
            'newsapi': {'minute': 0, 'hour': 0},
            'alpha_vantage': {'minute': 0, 'hour': 0},
            'reddit': {'minute': 0, 'hour': 0},
            'finnhub': {'minute': 0, 'hour': 0}
        }
        
        # Initialize services
        # Prefer environment variables if set (safer than hard-coding in YAML)
        env_news = os.getenv('NEWSAPI_KEY')
        env_alpha = os.getenv('ALPHAVANTAGE_KEY')
        env_finnhub = os.getenv('FINNHUB_KEY')

        self.newsapi_key = (env_news or self.api_keys.get('newsapi', 'demo')).strip() if (env_news or self.api_keys.get('newsapi')) else 'demo'
        self.alpha_vantage_key = (env_alpha or self.api_keys.get('alpha_vantage', 'demo')).strip() if (env_alpha or self.api_keys.get('alpha_vantage')) else 'demo'
        # Keep reddit config as-is; not environment-driven here
        self.reddit_config = self.api_keys.get('reddit', {})

        # Optional: store finnhub for future extension
        self.finnhub_key = (env_finnhub or self.api_keys.get('finnhub', 'demo')).strip() if (env_finnhub or self.api_keys.get('finnhub')) else 'demo'
        
        logger.info("Free APIs Integration initialized")
        logger.info(f"News API: {' Configured' if self.newsapi_key != 'demo' else ' Demo mode'}")
        logger.info(f"Alpha Vantage: {' Configured' if self.alpha_vantage_key != 'demo' else ' Demo mode'}")
        logger.info(f"Finnhub: {' Configured' if self.finnhub_key != 'demo' else ' Demo mode'}")
        logger.info(f"Reddit API: {' Configured' if self.reddit_config.get('client_id') != 'demo' else ' Demo mode'}")
    
    def _check_rate_limit(self, service: str) -> bool:
        """Check if we can make a request based on rate limits"""
        now = time.time()
        limits = self.rate_limits.get(service, {})
        
        # Reset counters if needed
        if now - self.last_request[service] > 60:  # Reset minute counter
            self.request_counts[service]['minute'] = 0
        if now - self.last_request[service] > 3600:  # Reset hour counter
            self.request_counts[service]['hour'] = 0
        
        # Check limits
        min_limit = limits.get('requests_per_minute', 1000)
        hour_limit = limits.get('requests_per_hour', 10000)
        
        if (self.request_counts[service]['minute'] >= min_limit or 
            self.request_counts[service]['hour'] >= hour_limit):
            return False
        
        return True
    
    def _update_rate_limit(self, service: str):
        """Update rate limit counters"""
        now = time.time()
        self.last_request[service] = now
        self.request_counts[service]['minute'] += 1
        self.request_counts[service]['hour'] += 1
    
    def get_news_sentiment(self, symbols: List[str], days_back: int = 7) -> Dict[str, Any]:
        """Get news sentiment for given symbols using News API"""
        if self.newsapi_key == 'demo':
            return self._get_demo_news_sentiment(symbols)
        
        if not self._check_rate_limit('newsapi'):
            logger.warning("News API rate limit reached")
            return self._get_demo_news_sentiment(symbols)
        
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            sentiment_data = {
                'symbols': {},
                'overall_sentiment': 0.0,
                'total_articles': 0,
                'last_updated': datetime.now().isoformat()
            }
            
            for symbol in symbols:
                # Search for news about the symbol
                query = f"{symbol} stock OR {symbol} earnings OR {symbol} financial"
                
                url = "https://newsapi.org/v2/everything"
                params = {
                    'q': query,
                    'from': start_date.strftime('%Y-%m-%d'),
                    'to': end_date.strftime('%Y-%m-%d'),
                    'sortBy': 'publishedAt',
                    'language': 'en',
                    'pageSize': 50,
                    'apiKey': self.newsapi_key
                }
                
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                articles = data.get('articles', [])
                
                # Simple sentiment analysis (positive/negative keywords)
                positive_keywords = [
                    'profit', 'growth', 'increase', 'rise', 'gain', 'positive', 
                    'strong', 'beat', 'exceed', 'outperform', 'bullish', 'upgrade'
                ]
                negative_keywords = [
                    'loss', 'decline', 'decrease', 'fall', 'drop', 'negative',
                    'weak', 'miss', 'disappoint', 'underperform', 'bearish', 'downgrade'
                ]
                
                sentiment_score = 0.0
                for article in articles:
                    title = article.get('title', '').lower()
                    description = article.get('description', '').lower()
                    text = f"{title} {description}"
                    
                    positive_count = sum(1 for word in positive_keywords if word in text)
                    negative_count = sum(1 for word in negative_keywords if word in text)
                    
                    if positive_count > negative_count:
                        sentiment_score += 1.0
                    elif negative_count > positive_count:
                        sentiment_score -= 1.0
                
                # Normalize sentiment (-1 to 1)
                if len(articles) > 0:
                    sentiment_score = sentiment_score / len(articles)
                
                sentiment_data['symbols'][symbol] = {
                    'sentiment': sentiment_score,
                    'article_count': len(articles),
                    'articles': articles[:5]  # Keep top 5 articles
                }
                
                sentiment_data['total_articles'] += len(articles)
            
            # Calculate overall sentiment
            if sentiment_data['symbols']:
                sentiment_data['overall_sentiment'] = sum(
                    data['sentiment'] for data in sentiment_data['symbols'].values()
                ) / len(sentiment_data['symbols'])
            
            self._update_rate_limit('newsapi')
            logger.info(f"News sentiment fetched for {len(symbols)} symbols")
            return sentiment_data
            
        except Exception as e:
            logger.error(f"News API error: {e}")
            return self._get_demo_news_sentiment(symbols)
    
    def _get_demo_news_sentiment(self, symbols: List[str]) -> Dict[str, Any]:
        """Demo news sentiment data"""
        import random
        
        sentiment_data = {
            'symbols': {},
            'overall_sentiment': random.uniform(-0.3, 0.3),
            'total_articles': random.randint(10, 50),
            'last_updated': datetime.now().isoformat(),
            'demo_mode': True
        }
        
        for symbol in symbols:
            sentiment_data['symbols'][symbol] = {
                'sentiment': random.uniform(-0.5, 0.5),
                'article_count': random.randint(2, 8),
                'articles': []
            }
        
        return sentiment_data
    
    def get_technical_indicators(self, symbol: str, interval: str = 'daily') -> Dict[str, Any]:
        """Get technical indicators from Alpha Vantage"""
        if self.alpha_vantage_key == 'demo':
            return self._get_demo_technical_indicators(symbol)
        
        if not self._check_rate_limit('alpha_vantage'):
            logger.warning("Alpha Vantage rate limit reached")
            return self._get_demo_technical_indicators(symbol)
        
        try:
            indicators = {}
            
            # Get SMA (Simple Moving Average)
            sma_url = "https://www.alphavantage.co/query"
            sma_params = {
                'function': 'SMA',
                'symbol': symbol,
                'interval': interval,
                'time_period': 20,
                'series_type': 'close',
                'apikey': self.alpha_vantage_key
            }
            
            response = requests.get(sma_url, params=sma_params, timeout=10)
            response.raise_for_status()
            sma_data = response.json()
            
            if 'Technical Analysis: SMA' in sma_data:
                latest_sma = list(sma_data['Technical Analysis: SMA'].items())[0]
                indicators['sma_20'] = float(latest_sma[1]['SMA'])
            
            # Get RSI (Relative Strength Index)
            rsi_params = {
                'function': 'RSI',
                'symbol': symbol,
                'interval': interval,
                'time_period': 14,
                'series_type': 'close',
                'apikey': self.alpha_vantage_key
            }
            
            response = requests.get(sma_url, params=rsi_params, timeout=10)
            response.raise_for_status()
            rsi_data = response.json()
            
            if 'Technical Analysis: RSI' in rsi_data:
                latest_rsi = list(rsi_data['Technical Analysis: RSI'].items())[0]
                indicators['rsi_14'] = float(latest_rsi[1]['RSI'])
            
            # Get MACD
            macd_params = {
                'function': 'MACD',
                'symbol': symbol,
                'interval': interval,
                'series_type': 'close',
                'apikey': self.alpha_vantage_key
            }
            
            response = requests.get(sma_url, params=macd_params, timeout=10)
            response.raise_for_status()
            macd_data = response.json()
            
            if 'Technical Analysis: MACD' in macd_data:
                latest_macd = list(macd_data['Technical Analysis: MACD'].items())[0]
                indicators['macd'] = float(latest_macd[1]['MACD'])
                indicators['macd_signal'] = float(latest_macd[1]['MACD_Signal'])
                indicators['macd_hist'] = float(latest_macd[1]['MACD_Hist'])
            
            indicators['last_updated'] = datetime.now().isoformat()
            self._update_rate_limit('alpha_vantage')
            
            logger.info(f"Technical indicators fetched for {symbol}")
            return indicators
            
        except Exception as e:
            logger.error(f"Alpha Vantage error: {e}")
            return self._get_demo_technical_indicators(symbol)
    
    def _get_demo_technical_indicators(self, symbol: str) -> Dict[str, Any]:
        """Demo technical indicators"""
        import random
        
        return {
            'sma_20': random.uniform(50, 150),
            'rsi_14': random.uniform(20, 80),
            'macd': random.uniform(-2, 2),
            'macd_signal': random.uniform(-2, 2),
            'macd_hist': random.uniform(-1, 1),
            'last_updated': datetime.now().isoformat(),
            'demo_mode': True
        }
    
    def get_reddit_sentiment(self, symbols: List[str]) -> Dict[str, Any]:
        """Get Reddit sentiment for given symbols"""
        if self.reddit_config.get('client_id') == 'demo':
            return self._get_demo_reddit_sentiment(symbols)
        
        if not self._check_rate_limit('reddit'):
            logger.warning("Reddit API rate limit reached")
            return self._get_demo_reddit_sentiment(symbols)
        
        try:
            # Reddit API requires OAuth2 authentication
            # For now, we'll use a simplified approach
            sentiment_data = {
                'symbols': {},
                'overall_sentiment': 0.0,
                'total_mentions': 0,
                'last_updated': datetime.now().isoformat()
            }
            
            # This would require proper Reddit API setup
            # For demo purposes, we'll return mock data
            logger.info("Reddit API integration requires OAuth2 setup")
            return self._get_demo_reddit_sentiment(symbols)
            
        except Exception as e:
            logger.error(f"Reddit API error: {e}")
            return self._get_demo_reddit_sentiment(symbols)
    
    def _get_demo_reddit_sentiment(self, symbols: List[str]) -> Dict[str, Any]:
        """Demo Reddit sentiment data"""
        import random
        
        sentiment_data = {
            'symbols': {},
            'overall_sentiment': random.uniform(-0.2, 0.2),
            'total_mentions': random.randint(5, 25),
            'last_updated': datetime.now().isoformat(),
            'demo_mode': True
        }
        
        for symbol in symbols:
            sentiment_data['symbols'][symbol] = {
                'sentiment': random.uniform(-0.4, 0.4),
                'mentions': random.randint(1, 5),
                'subreddits': ['wallstreetbets', 'stocks', 'investing']
            }
        
        return sentiment_data
    
    def get_finnhub_news(self, symbol: str, days_back: int = 7) -> Dict[str, Any]:
        """
        Get company news from Finnhub API
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'TD.TO')
            days_back: Number of days to look back
        
        Returns:
            Dict with news headlines and sentiment
        """
        if self.finnhub_key == 'demo':
            return self._get_demo_finnhub_news(symbol)
        
        if not self._check_rate_limit('finnhub'):
            logger.warning("Finnhub API rate limit reached")
            time.sleep(1)  # Backoff 1 second
            if not self._check_rate_limit('finnhub'):
                logger.error("Finnhub API rate limit still exceeded after backoff")
                return self._get_demo_finnhub_news(symbol)
        
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Format symbol (remove .TO suffix for Canadian stocks)
            finnhub_symbol = symbol.replace('.TO', '')
            
            # Finnhub API endpoint
            url = 'https://finnhub.io/api/v1/company-news'
            params = {
                'symbol': finnhub_symbol,
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'token': self.finnhub_key
            }
            
            self._update_rate_limit('finnhub')
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            news_items = response.json()
            
            # Process news with sentiment
            processed_news = []
            total_sentiment = 0
            
            for item in news_items[:20]:  # Limit to 20 most recent
                # Simple sentiment based on headline keywords
                headline = item.get('headline', '').lower()
                summary = item.get('summary', '').lower()
                
                # Basic keyword-based sentiment
                positive_keywords = ['gain', 'profit', 'growth', 'success', 'beat', 'surge', 'rise']
                negative_keywords = ['loss', 'decline', 'fall', 'miss', 'drop', 'concern', 'risk']
                
                sentiment_score = 0
                for keyword in positive_keywords:
                    if keyword in headline or keyword in summary:
                        sentiment_score += 0.1
                for keyword in negative_keywords:
                    if keyword in headline or keyword in summary:
                        sentiment_score -= 0.1
                
                sentiment_score = max(-1, min(1, sentiment_score))  # Clamp to [-1, 1]
                total_sentiment += sentiment_score
                
                processed_news.append({
                    'headline': item.get('headline', ''),
                    'summary': item.get('summary', ''),
                    'url': item.get('url', ''),
                    'source': item.get('source', ''),
                    'datetime': item.get('datetime', 0),
                    'sentiment_score': sentiment_score
                })
            
            avg_sentiment = total_sentiment / len(processed_news) if processed_news else 0
            
            logger.info(f"Finnhub: Retrieved {len(processed_news)} news items for {symbol}")
            
            return {
                'symbol': symbol,
                'news_count': len(processed_news),
                'avg_sentiment': avg_sentiment,
                'news_items': processed_news,
                'date_range': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                'last_updated': datetime.now().isoformat(),
                'demo_mode': False
            }
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                logger.error("Finnhub API rate limit exceeded (429)")
                time.sleep(60)  # Backoff 60 seconds
            else:
                logger.error(f"Finnhub API HTTP error: {e}")
            return self._get_demo_finnhub_news(symbol)
        except Exception as e:
            logger.error(f"Finnhub API error: {e}")
            return self._get_demo_finnhub_news(symbol)
    
    def _get_demo_finnhub_news(self, symbol: str) -> Dict[str, Any]:
        """Demo Finnhub news data"""
        import random
        
        demo_headlines = [
            {'headline': f'{symbol} reports strong quarterly earnings', 'sentiment': 0.6},
            {'headline': f'Analysts upgrade {symbol} price target', 'sentiment': 0.4},
            {'headline': f'{symbol} faces headwinds in current market', 'sentiment': -0.3},
            {'headline': f'CEO of {symbol} discusses growth strategy', 'sentiment': 0.2},
            {'headline': f'{symbol} announces new product launch', 'sentiment': 0.5}
        ]
        
        news_items = []
        total_sentiment = 0
        
        for headline_data in demo_headlines[:random.randint(3, 5)]:
            sentiment = headline_data['sentiment'] + random.uniform(-0.1, 0.1)
            total_sentiment += sentiment
            
            news_items.append({
                'headline': headline_data['headline'],
                'summary': f"Demo summary for {headline_data['headline']}",
                'url': 'https://demo.url',
                'source': 'Demo Source',
                'datetime': int(time.time()) - random.randint(0, 604800),
                'sentiment_score': sentiment
            })
        
        avg_sentiment = total_sentiment / len(news_items) if news_items else 0
        
        return {
            'symbol': symbol,
            'news_count': len(news_items),
            'avg_sentiment': avg_sentiment,
            'news_items': news_items,
            'date_range': 'demo',
            'last_updated': datetime.now().isoformat(),
            'demo_mode': True
        }
    
    def get_comprehensive_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Get comprehensive data from all free APIs"""
        logger.info(f"Fetching comprehensive data for {len(symbols)} symbols")
        
        # Get news sentiment
        news_data = self.get_news_sentiment(symbols)
        
        # Get technical indicators for each symbol
        technical_data = {}
        for symbol in symbols:
            technical_data[symbol] = self.get_technical_indicators(symbol)
        
        # Get Reddit sentiment
        reddit_data = self.get_reddit_sentiment(symbols)
        
        # Get Finnhub news for each symbol
        finnhub_data = {}
        for symbol in symbols:
            finnhub_data[symbol] = self.get_finnhub_news(symbol)
        
        return {
            'news_sentiment': news_data,
            'technical_indicators': technical_data,
            'reddit_sentiment': reddit_data,
            'finnhub_news': finnhub_data,
            'timestamp': datetime.now().isoformat(),
            'symbols': symbols
        }
    
    def get_api_status(self) -> Dict[str, Any]:
        """Get status of all API services"""
        return {
            'news_api': {
                'configured': self.newsapi_key != 'demo',
                'rate_limit_status': self._check_rate_limit('newsapi'),
                'requests_this_minute': self.request_counts['newsapi']['minute'],
                'requests_this_hour': self.request_counts['newsapi']['hour']
            },
            'alpha_vantage': {
                'configured': self.alpha_vantage_key != 'demo',
                'rate_limit_status': self._check_rate_limit('alpha_vantage'),
                'requests_this_minute': self.request_counts['alpha_vantage']['minute'],
                'requests_this_hour': self.request_counts['alpha_vantage']['hour']
            },
            'reddit_api': {
                'configured': self.reddit_config.get('client_id') != 'demo',
                'rate_limit_status': self._check_rate_limit('reddit'),
                'requests_this_minute': self.request_counts['reddit']['minute'],
                'requests_this_hour': self.request_counts['reddit']['hour']
            },
            'finnhub': {
                'configured': self.finnhub_key != 'demo',
                'rate_limit_status': self._check_rate_limit('finnhub'),
                'requests_this_minute': self.request_counts['finnhub']['minute'],
                'requests_this_hour': self.request_counts['finnhub']['hour']
            }
        }


def create_free_apis_config() -> Dict[str, Any]:
    """Create configuration for free APIs integration"""
    return {
        'api_keys': {
            'newsapi': 'demo',  # Replace with your News API key
            'alpha_vantage': 'demo',  # Replace with your Alpha Vantage key
            'reddit': {
                'client_id': 'demo',  # Replace with your Reddit client ID
                'client_secret': 'demo',  # Replace with your Reddit client secret
                'user_agent': 'TradingBot/1.0'
            }
        },
        'rate_limits': {
            'newsapi': {
                'requests_per_minute': 30,
                'requests_per_hour': 1000
            },
            'alpha_vantage': {
                'requests_per_minute': 5,
                'requests_per_hour': 500
            },
            'reddit': {
                'requests_per_minute': 60,
                'requests_per_hour': 3600
            },
            'finnhub': {
                'requests_per_minute': 60,
                'requests_per_hour': 3600
            }
        }
    }


# Example usage
if __name__ == "__main__":
    # Create config
    config = create_free_apis_config()
    
    # Initialize integration
    api_integration = FreeAPIsIntegration(config)
    
    # Test with Canadian stocks
    symbols = ['TD.TO', 'RY.TO', 'SHOP.TO']
    
    # Get comprehensive data
    data = api_integration.get_comprehensive_data(symbols)
    
    print("API Status:")
    print(json.dumps(api_integration.get_api_status(), indent=2))
    
    print("\nComprehensive Data:")
    print(json.dumps(data, indent=2))
