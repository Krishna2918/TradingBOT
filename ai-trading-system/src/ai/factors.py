"""
AI Factors Module - Sentiment and Fundamental Analysis

This module provides sentiment analysis using NewsAPI and fundamental analysis
using Finnhub API. All data is cached and persisted for performance.
"""

import logging
import requests
import json
import time
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import duckdb
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

@dataclass
class SentimentData:
    """Sentiment analysis data structure."""
    symbol: str
    sentiment_score: float  # -1 to 1
    confidence: float  # 0 to 1
    news_count: int
    positive_news: int
    negative_news: int
    neutral_news: int
    timestamp: datetime

@dataclass
class FundamentalData:
    """Fundamental analysis data structure."""
    symbol: str
    pe_ratio: Optional[float]
    pb_ratio: Optional[float]
    debt_to_equity: Optional[float]
    roe: Optional[float]
    revenue_growth: Optional[float]
    earnings_growth: Optional[float]
    market_cap: Optional[float]
    timestamp: datetime

class NewsAPIClient:
    """NewsAPI client for sentiment analysis."""
    
    def __init__(self, api_key: str = None):
        """Initialize NewsAPI client."""
        self.api_key = api_key or os.getenv('NEWSAPI_KEY')
        if not self.api_key:
            raise ValueError("NEWSAPI_KEY not found in environment variables")
        
        self.base_url = "https://newsapi.org/v2"
        self.session = requests.Session()
        self.session.headers.update({
            'X-API-Key': self.api_key,
            'User-Agent': 'AI-Trading-System/1.0'
        })
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # 1 second between requests
    
    def _rate_limit(self):
        """Apply rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()
    
    def get_news(self, symbol: str, days_back: int = 7) -> List[Dict]:
        """Get news articles for a symbol."""
        # Check if NewsAPI is disabled
        if os.getenv('NEWSAPI_DISABLE') == '1':
            logger.info(f"NewsAPI disabled, skipping news for {symbol}")
            return []
        
        self._rate_limit()
        
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Search for news
            params = {
                'q': f'"{symbol}" OR "{symbol}.TO"',
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'sortBy': 'publishedAt',
                'pageSize': 100,
                'language': 'en'
            }
            
            response = self.session.get(f"{self.base_url}/everything", params=params)
            response.raise_for_status()
            
            data = response.json()
            articles = data.get('articles', [])
            
            logger.info(f"Retrieved {len(articles)} news articles for {symbol}")
            return articles
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching news for {symbol}: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching news for {symbol}: {e}")
            return []
    
    def analyze_sentiment(self, articles: List[Dict]) -> Tuple[float, float, Dict]:
        """Analyze sentiment of news articles."""
        if not articles:
            return 0.0, 0.0, {'positive': 0, 'negative': 0, 'neutral': 0}
        
        # Simple sentiment analysis based on keywords
        positive_keywords = [
            'bullish', 'positive', 'growth', 'profit', 'gain', 'rise', 'increase',
            'strong', 'outperform', 'beat', 'exceed', 'surge', 'rally', 'upgrade'
        ]
        
        negative_keywords = [
            'bearish', 'negative', 'decline', 'loss', 'fall', 'drop', 'decrease',
            'weak', 'underperform', 'miss', 'disappoint', 'crash', 'plunge', 'downgrade'
        ]
        
        sentiment_scores = []
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        
        for article in articles:
            title = article.get('title', '') or ''
            description = article.get('description', '') or ''
            content = f"{title} {description}".lower()
            
            # Count keyword matches
            positive_count = sum(1 for keyword in positive_keywords if keyword in content)
            negative_count = sum(1 for keyword in negative_keywords if keyword in content)
            
            # Calculate sentiment score for this article
            if positive_count > negative_count:
                sentiment_scores.append(1.0)
                sentiment_counts['positive'] += 1
            elif negative_count > positive_count:
                sentiment_scores.append(-1.0)
                sentiment_counts['negative'] += 1
            else:
                sentiment_scores.append(0.0)
                sentiment_counts['neutral'] += 1
        
        # Calculate overall sentiment
        if sentiment_scores:
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            confidence = abs(avg_sentiment)  # Higher absolute value = higher confidence
        else:
            avg_sentiment = 0.0
            confidence = 0.0
        
        return avg_sentiment, confidence, sentiment_counts

class FinnhubClient:
    """Finnhub client for fundamental analysis."""
    
    def __init__(self, api_key: str = None):
        """Initialize Finnhub client."""
        self.api_key = api_key or os.getenv('FINNHUB_API_KEY')
        if not self.api_key:
            raise ValueError("FINNHUB_API_KEY not found in environment variables")
        
        self.base_url = "https://finnhub.io/api/v1"
        self.session = requests.Session()
        self.session.params = {'token': self.api_key}
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # 1 second between requests
    
    def _rate_limit(self):
        """Apply rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()
    
    def get_company_profile(self, symbol: str) -> Dict:
        """Get company profile data."""
        # Check if Finnhub is disabled
        if os.getenv('FINNHUB_DISABLE') == '1':
            logger.info(f"Finnhub disabled, skipping profile for {symbol}")
            return {}
        
        self._rate_limit()
        
        try:
            response = self.session.get(f"{self.base_url}/stock/profile2", 
                                      params={'symbol': symbol})
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching company profile for {symbol}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error fetching company profile for {symbol}: {e}")
            return {}
    
    def get_financial_metrics(self, symbol: str) -> Dict:
        """Get financial metrics."""
        # Check if Finnhub is disabled
        if os.getenv('FINNHUB_DISABLE') == '1':
            logger.info(f"Finnhub disabled, skipping metrics for {symbol}")
            return {}
        
        self._rate_limit()
        
        try:
            response = self.session.get(f"{self.base_url}/stock/metric", 
                                      params={'symbol': symbol, 'metric': 'all'})
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching financial metrics for {symbol}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error fetching financial metrics for {symbol}: {e}")
            return {}
    
    def get_earnings(self, symbol: str) -> List[Dict]:
        """Get earnings data."""
        # Check if Finnhub is disabled
        if os.getenv('FINNHUB_DISABLE') == '1':
            logger.info(f"Finnhub disabled, skipping earnings for {symbol}")
            return []
        
        self._rate_limit()
        
        try:
            response = self.session.get(f"{self.base_url}/stock/earnings", 
                                      params={'symbol': symbol})
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching earnings for {symbol}: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching earnings for {symbol}: {e}")
            return []

class FactorEngine:
    """Factor analysis engine with caching and persistence."""
    
    def __init__(self, db_path: str = "data/market_data.duckdb"):
        """Initialize factor engine."""
        self.db_path = db_path
        self.conn = None
        self.news_client = NewsAPIClient()
        self.finnhub_client = FinnhubClient()
        self._ensure_db_exists()
        self._create_schema()
    
    def _ensure_db_exists(self):
        """Ensure database directory exists."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
    
    def _create_schema(self):
        """Create database schema for factors."""
        self.conn = duckdb.connect(self.db_path)
        
        # Sentiment data table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS sentiment_data (
                symbol VARCHAR,
                sentiment_score DOUBLE,
                confidence DOUBLE,
                news_count INTEGER,
                positive_news INTEGER,
                negative_news INTEGER,
                neutral_news INTEGER,
                timestamp TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, timestamp)
            )
        """)
        
        # Fundamental data table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS fundamental_data (
                symbol VARCHAR,
                pe_ratio DOUBLE,
                pb_ratio DOUBLE,
                debt_to_equity DOUBLE,
                roe DOUBLE,
                revenue_growth DOUBLE,
                earnings_growth DOUBLE,
                market_cap DOUBLE,
                timestamp TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, timestamp)
            )
        """)
        
        # News articles table - with primary key for INSERT OR REPLACE
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS news_articles (
                id INTEGER PRIMARY KEY,
                symbol VARCHAR,
                title VARCHAR,
                description VARCHAR,
                url VARCHAR,
                published_at TIMESTAMP,
                source VARCHAR,
                sentiment_score DOUBLE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, url, published_at)
            )
        """)
        
        logger.info("Factor database schema created successfully")
    
    def analyze_sentiment(self, symbol: str, days_back: int = 7) -> SentimentData:
        """Analyze sentiment for a symbol."""
        try:
            # Get news articles
            articles = self.news_client.get_news(symbol, days_back)
            
            # Analyze sentiment
            sentiment_score, confidence, sentiment_counts = self.news_client.analyze_sentiment(articles)
            
            # Create sentiment data
            sentiment_data = SentimentData(
                symbol=symbol,
                sentiment_score=sentiment_score,
                confidence=confidence,
                news_count=len(articles),
                positive_news=sentiment_counts['positive'],
                negative_news=sentiment_counts['negative'],
                neutral_news=sentiment_counts['neutral'],
                timestamp=datetime.now()
            )
            
            # Persist to database
            self._persist_sentiment_data(sentiment_data)
            
            # Persist articles
            self._persist_news_articles(symbol, articles)
            
            logger.info(f"Analyzed sentiment for {symbol}: {sentiment_score:.3f} (confidence: {confidence:.3f})")
            return sentiment_data
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment for {symbol}: {e}")
            return SentimentData(
                symbol=symbol,
                sentiment_score=0.0,
                confidence=0.0,
                news_count=0,
                positive_news=0,
                negative_news=0,
                neutral_news=0,
                timestamp=datetime.now()
            )
    
    def analyze_fundamentals(self, symbol: str) -> FundamentalData:
        """Analyze fundamentals for a symbol."""
        try:
            # Get company profile
            profile = self.finnhub_client.get_company_profile(symbol)
            
            # Get financial metrics
            metrics = self.finnhub_client.get_financial_metrics(symbol)
            
            # Get earnings data
            earnings = self.finnhub_client.get_earnings(symbol)
            
            # Extract fundamental data
            fundamental_data = FundamentalData(
                symbol=symbol,
                pe_ratio=self._extract_metric(metrics, 'pe'),
                pb_ratio=self._extract_metric(metrics, 'pb'),
                debt_to_equity=self._extract_metric(metrics, 'debtToEquity'),
                roe=self._extract_metric(metrics, 'roe'),
                revenue_growth=self._extract_metric(metrics, 'revenueGrowth'),
                earnings_growth=self._extract_metric(metrics, 'earningsGrowth'),
                market_cap=profile.get('marketCapitalization'),
                timestamp=datetime.now()
            )
            
            # Persist to database
            self._persist_fundamental_data(fundamental_data)
            
            logger.info(f"Analyzed fundamentals for {symbol}")
            return fundamental_data
            
        except Exception as e:
            logger.error(f"Error analyzing fundamentals for {symbol}: {e}")
            return FundamentalData(
                symbol=symbol,
                pe_ratio=None,
                pb_ratio=None,
                debt_to_equity=None,
                roe=None,
                revenue_growth=None,
                earnings_growth=None,
                market_cap=None,
                timestamp=datetime.now()
            )
    
    def _extract_metric(self, metrics: Dict, key: str) -> Optional[float]:
        """Extract metric value from Finnhub response."""
        try:
            if 'metric' in metrics and key in metrics['metric']:
                value = metrics['metric'][key]
                return float(value) if value is not None else None
            return None
        except (ValueError, TypeError):
            return None
    
    def _persist_sentiment_data(self, sentiment_data: SentimentData):
        """Persist sentiment data to database."""
        try:
            self.conn.execute("""
                INSERT OR REPLACE INTO sentiment_data 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, [
                sentiment_data.symbol,
                sentiment_data.sentiment_score,
                sentiment_data.confidence,
                sentiment_data.news_count,
                sentiment_data.positive_news,
                sentiment_data.negative_news,
                sentiment_data.neutral_news,
                sentiment_data.timestamp
            ])
        except Exception as e:
            logger.error(f"Error persisting sentiment data for {sentiment_data.symbol}: {e}")
    
    def _persist_fundamental_data(self, fundamental_data: FundamentalData):
        """Persist fundamental data to database."""
        try:
            self.conn.execute("""
                INSERT OR REPLACE INTO fundamental_data 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, [
                fundamental_data.symbol,
                fundamental_data.pe_ratio,
                fundamental_data.pb_ratio,
                fundamental_data.debt_to_equity,
                fundamental_data.roe,
                fundamental_data.revenue_growth,
                fundamental_data.earnings_growth,
                fundamental_data.market_cap,
                fundamental_data.timestamp
            ])
        except Exception as e:
            logger.error(f"Error persisting fundamental data for {fundamental_data.symbol}: {e}")
    
    def _persist_news_articles(self, symbol: str, articles: List[Dict]):
        """Persist news articles to database."""
        try:
            for article in articles:
                # Calculate sentiment for individual article
                title = article.get('title', '').lower()
                description = article.get('description', '').lower()
                content = f"{title} {description}"
                
                # Simple sentiment scoring
                positive_keywords = ['bullish', 'positive', 'growth', 'profit', 'gain', 'rise']
                negative_keywords = ['bearish', 'negative', 'decline', 'loss', 'fall', 'drop']
                
                positive_count = sum(1 for keyword in positive_keywords if keyword in content)
                negative_count = sum(1 for keyword in negative_keywords if keyword in content)
                
                if positive_count > negative_count:
                    sentiment_score = 1.0
                elif negative_count > positive_count:
                    sentiment_score = -1.0
                else:
                    sentiment_score = 0.0
                
                # Parse published date
                published_at = None
                if article.get('publishedAt'):
                    try:
                        published_at = datetime.fromisoformat(
                            article['publishedAt'].replace('Z', '+00:00')
                        )
                    except:
                        published_at = datetime.now()
                
                self.conn.execute("""
                    INSERT OR REPLACE INTO news_articles 
                    (symbol, title, description, url, published_at, source, sentiment_score, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, [
                    symbol,
                    article.get('title', ''),
                    article.get('description', ''),
                    article.get('url', ''),
                    published_at,
                    article.get('source', {}).get('name', ''),
                    sentiment_score
                ])
        except Exception as e:
            logger.error(f"Error persisting news articles for {symbol}: {e}")
    
    def get_latest_sentiment(self, symbol: str) -> Optional[SentimentData]:
        """Get latest sentiment data for a symbol."""
        try:
            query = """
                SELECT * FROM sentiment_data 
                WHERE symbol = ? 
                ORDER BY timestamp DESC 
                LIMIT 1
            """
            
            result = self.conn.execute(query, [symbol]).fetchdf()
            
            if result.empty:
                return None
            
            row = result.iloc[0]
            return SentimentData(
                symbol=row['symbol'],
                sentiment_score=row['sentiment_score'],
                confidence=row['confidence'],
                news_count=row['news_count'],
                positive_news=row['positive_news'],
                negative_news=row['negative_news'],
                neutral_news=row['neutral_news'],
                timestamp=row['timestamp']
            )
            
        except Exception as e:
            logger.error(f"Error getting latest sentiment for {symbol}: {e}")
            return None
    
    def get_latest_fundamentals(self, symbol: str) -> Optional[FundamentalData]:
        """Get latest fundamental data for a symbol."""
        try:
            query = """
                SELECT * FROM fundamental_data 
                WHERE symbol = ? 
                ORDER BY timestamp DESC 
                LIMIT 1
            """
            
            result = self.conn.execute(query, [symbol]).fetchdf()
            
            if result.empty:
                return None
            
            row = result.iloc[0]
            return FundamentalData(
                symbol=row['symbol'],
                pe_ratio=row['pe_ratio'],
                pb_ratio=row['pb_ratio'],
                debt_to_equity=row['debt_to_equity'],
                roe=row['roe'],
                revenue_growth=row['revenue_growth'],
                earnings_growth=row['earnings_growth'],
                market_cap=row['market_cap'],
                timestamp=row['timestamp']
            )
            
        except Exception as e:
            logger.error(f"Error getting latest fundamentals for {symbol}: {e}")
            return None
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

# Global factor engine instance
factor_engine = FactorEngine()

def analyze_sentiment(symbol: str, days_back: int = 7) -> SentimentData:
    """Analyze sentiment for a symbol."""
    return factor_engine.analyze_sentiment(symbol, days_back)

def analyze_fundamentals(symbol: str) -> FundamentalData:
    """Analyze fundamentals for a symbol."""
    return factor_engine.analyze_fundamentals(symbol)

def get_latest_sentiment(symbol: str) -> Optional[SentimentData]:
    """Get latest sentiment data for a symbol."""
    return factor_engine.get_latest_sentiment(symbol)

def get_latest_fundamentals(symbol: str) -> Optional[FundamentalData]:
    """Get latest fundamental data for a symbol."""
    return factor_engine.get_latest_fundamentals(symbol)
