"""
News Processing and Aggregation for Financial Market Intelligence

This module implements news processing, aggregation, and analysis
for financial market intelligence and trading signal generation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime, timedelta
import re
import hashlib
from collections import defaultdict, Counter
import warnings

# Web scraping and news APIs
try:
    import requests
    from bs4 import BeautifulSoup
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    warnings.warn("Requests/BeautifulSoup not available. Web scraping features will be limited.")

try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False
    warnings.warn("Feedparser not available. RSS feed features will be limited.")

logger = logging.getLogger(__name__)

class NewsProcessor:
    """
    News processor for financial news analysis and processing.
    """
    
    def __init__(
        self,
        model_name: str = "news_processor",
        clean_text: bool = True,
        extract_entities: bool = True
    ):
        """
        Initialize news processor.
        
        Args:
            model_name: Name for the processor
            clean_text: Whether to clean and preprocess text
            extract_entities: Whether to extract financial entities
        """
        self.model_name = model_name
        self.clean_text = clean_text
        self.extract_entities = extract_entities
        
        # Financial entity patterns
        self.ticker_pattern = re.compile(r'\b[A-Z]{1,5}\b')
        self.price_pattern = re.compile(r'\$[\d,]+\.?\d*')
        self.percentage_pattern = re.compile(r'[\d,]+\.?\d*%')
        self.currency_pattern = re.compile(r'[\d,]+\.?\d*\s*(USD|EUR|GBP|JPY|CAD|AUD)')
        
        # Common financial terms
        self.financial_terms = {
            'earnings', 'revenue', 'profit', 'loss', 'dividend', 'stock', 'share',
            'market', 'trading', 'investment', 'portfolio', 'fund', 'bond',
            'equity', 'debt', 'credit', 'loan', 'interest', 'rate', 'yield',
            'volatility', 'risk', 'return', 'performance', 'growth', 'decline'
        }
        
        # News categories
        self.news_categories = {
            'earnings': ['earnings', 'revenue', 'profit', 'loss', 'quarterly', 'annual'],
            'mergers': ['merger', 'acquisition', 'takeover', 'buyout', 'deal'],
            'ipo': ['ipo', 'initial public offering', 'public offering', 'listing'],
            'dividend': ['dividend', 'payout', 'distribution', 'yield'],
            'analyst': ['analyst', 'rating', 'upgrade', 'downgrade', 'target', 'price'],
            'regulatory': ['sec', 'regulation', 'compliance', 'investigation', 'fine'],
            'market': ['market', 'trading', 'volume', 'price', 'volatility']
        }
        
        # Processed news history
        self.news_history = []
        
        logger.info(f"Initialized News Processor: {model_name}")
    
    def clean_news_text(self, text: str) -> str:
        """
        Clean and preprocess news text.
        
        Args:
            text: Raw news text
            
        Returns:
            Cleaned text
        """
        if not self.clean_text:
            return text
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters but keep financial symbols
        text = re.sub(r'[^\w\s\$%\.\,\-]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_tickers(self, text: str) -> List[str]:
        """
        Extract stock tickers from text.
        
        Args:
            text: Input text
            
        Returns:
            List of potential tickers
        """
        tickers = self.ticker_pattern.findall(text)
        
        # Filter out common words that match ticker pattern
        common_words = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'HAD', 'BUT', 'WILL', 'NEW', 'NOW', 'OLD', 'SEE', 'HIM', 'TWO', 'HOW', 'ITS', 'WHO', 'OIL', 'GET', 'HAS', 'HAD', 'MAY', 'SAY', 'USE', 'MAN', 'DAY', 'TOO', 'ANY', 'SAME', 'TELL', 'WELL', 'WERE', 'BEEN', 'GOOD', 'MUCH', 'SOME', 'TIME', 'VERY', 'WHEN', 'COME', 'HERE', 'JUST', 'LIKE', 'LONG', 'MAKE', 'MANY', 'OVER', 'SUCH', 'TAKE', 'THAN', 'THEM', 'WELL', 'WERE', 'WHAT', 'YEAR', 'YOUR', 'ABLE', 'ABOUT', 'ABOVE', 'ACRE', 'ACROSS', 'ACTED', 'ACTOR', 'ADDED', 'ADULT', 'AFTER', 'AGAIN', 'AGENT', 'AGREE', 'AHEAD', 'ALARM', 'ALBUM', 'ALERT', 'ALIEN', 'ALIGN', 'ALIKE', 'ALIVE', 'ALLOW', 'ALONE', 'ALONG', 'ALTER', 'AMONG', 'ANGER', 'ANGLE', 'ANGRY', 'APART', 'APPLE', 'APPLY', 'ARENA', 'ARGUE', 'ARISE', 'ARRAY', 'ASIDE', 'ASSET', 'AVOID', 'AWAKE', 'AWARD', 'AWARE', 'BADLY', 'BASIC', 'BEACH', 'BEGAN', 'BEGIN', 'BEING', 'BELOW', 'BENCH', 'BILLY', 'BIRTH', 'BLACK', 'BLAME', 'BLANK', 'BLIND', 'BLOCK', 'BLOOD', 'BOARD', 'BOOST', 'BOOTH', 'BOUND', 'BRAIN', 'BRAND', 'BRASS', 'BRAVE', 'BREAD', 'BREAK', 'BREED', 'BRIEF', 'BRING', 'BROAD', 'BROKE', 'BROWN', 'BUILD', 'BUILT', 'BUYER', 'CABLE', 'CALIF', 'CARRY', 'CATCH', 'CAUSE', 'CHAIN', 'CHAIR', 'CHAOS', 'CHARM', 'CHART', 'CHASE', 'CHEAP', 'CHECK', 'CHEST', 'CHIEF', 'CHILD', 'CHINA', 'CHOSE', 'CIVIL', 'CLAIM', 'CLASS', 'CLEAN', 'CLEAR', 'CLICK', 'CLIMB', 'CLOCK', 'CLOSE', 'CLOUD', 'COACH', 'COAST', 'COULD', 'COUNT', 'COURT', 'COVER', 'CRAFT', 'CRASH', 'CRAZY', 'CREAM', 'CRIME', 'CROSS', 'CROWD', 'CROWN', 'CRUDE', 'CURVE', 'CYCLE', 'DAILY', 'DANCE', 'DATED', 'DEALT', 'DEATH', 'DEBUT', 'DELAY', 'DEPTH', 'DOING', 'DOUBT', 'DOZEN', 'DRAFT', 'DRAMA', 'DRANK', 'DRAWN', 'DREAM', 'DRESS', 'DRILL', 'DRINK', 'DRIVE', 'DROVE', 'DYING', 'EAGER', 'EARLY', 'EARTH', 'EIGHT', 'ELITE', 'EMPTY', 'ENEMY', 'ENJOY', 'ENTER', 'ENTRY', 'EQUAL', 'ERROR', 'EVENT', 'EVERY', 'EXACT', 'EXIST', 'EXTRA', 'FAITH', 'FALSE', 'FAULT', 'FIBER', 'FIELD', 'FIFTH', 'FIFTY', 'FIGHT', 'FINAL', 'FIRST', 'FIXED', 'FLASH', 'FLEET', 'FLOOR', 'FLUID', 'FOCUS', 'FORCE', 'FORTH', 'FORTY', 'FORUM', 'FOUND', 'FRAME', 'FRANK', 'FRAUD', 'FRESH', 'FRONT', 'FROST', 'FRUIT', 'FULLY', 'FUNNY', 'GIANT', 'GIVEN', 'GLASS', 'GLOBE', 'GOING', 'GRACE', 'GRADE', 'GRAND', 'GRANT', 'GRASS', 'GRAVE', 'GREAT', 'GREEN', 'GROSS', 'GROUP', 'GROWN', 'GUARD', 'GUESS', 'GUEST', 'GUIDE', 'HAPPY', 'HARRY', 'HEART', 'HEAVY', 'HORSE', 'HOTEL', 'HOUSE', 'HUMAN', 'IDEAL', 'IMAGE', 'INDEX', 'INNER', 'INPUT', 'ISSUE', 'JAPAN', 'JIMMY', 'JOINT', 'JONES', 'JUDGE', 'KNOWN', 'LABEL', 'LARGE', 'LASER', 'LATER', 'LAUGH', 'LAYER', 'LEARN', 'LEASE', 'LEAST', 'LEAVE', 'LEGAL', 'LEVEL', 'LEWIS', 'LIGHT', 'LIMIT', 'LINKS', 'LIVES', 'LOCAL', 'LOOSE', 'LOWER', 'LUCKY', 'LUNCH', 'LYING', 'MAGIC', 'MAJOR', 'MAKER', 'MARCH', 'MARIA', 'MATCH', 'MAYBE', 'MAYOR', 'MEANT', 'MEDIA', 'METAL', 'MIGHT', 'MINOR', 'MINUS', 'MIXED', 'MODEL', 'MONEY', 'MONTH', 'MORAL', 'MOTOR', 'MOUNT', 'MOUSE', 'MOUTH', 'MOVED', 'MOVIE', 'MUSIC', 'NEEDS', 'NEVER', 'NEWLY', 'NIGHT', 'NOISE', 'NORTH', 'NOTED', 'NOVEL', 'NURSE', 'OCCUR', 'OCEAN', 'OFFER', 'OFTEN', 'ORDER', 'OTHER', 'OUGHT', 'PAINT', 'PANEL', 'PAPER', 'PARTY', 'PEACE', 'PETER', 'PHASE', 'PHONE', 'PHOTO', 'PIANO', 'PIECE', 'PILOT', 'PITCH', 'PLACE', 'PLAIN', 'PLANE', 'PLANT', 'PLATE', 'PLAZA', 'PLOT', 'PLUG', 'PLUS', 'POINT', 'POUND', 'POWER', 'PRESS', 'PRICE', 'PRIDE', 'PRIME', 'PRINT', 'PRIOR', 'PRIZE', 'PROOF', 'PROUD', 'PROVE', 'QUEEN', 'QUICK', 'QUIET', 'QUITE', 'RADIO', 'RAISE', 'RANGE', 'RAPID', 'RATIO', 'REACH', 'READY', 'REALM', 'REBEL', 'REFER', 'RELAX', 'REPAY', 'REPLY', 'RIGHT', 'RIGID', 'RIVER', 'ROBIN', 'ROGER', 'ROMAN', 'ROUGH', 'ROUND', 'ROUTE', 'ROYAL', 'RURAL', 'SCALE', 'SCENE', 'SCOPE', 'SCORE', 'SENSE', 'SERVE', 'SETUP', 'SEVEN', 'SHALL', 'SHAPE', 'SHARE', 'SHARP', 'SHEET', 'SHELF', 'SHELL', 'SHIFT', 'SHINE', 'SHIRT', 'SHOCK', 'SHOOT', 'SHORT', 'SHOWN', 'SIDED', 'SIGHT', 'SILLY', 'SINCE', 'SIXTH', 'SIXTY', 'SIZED', 'SKILL', 'SLEEP', 'SLIDE', 'SMALL', 'SMART', 'SMILE', 'SMITH', 'SMOKE', 'SNAKE', 'SNOW', 'SOLAR', 'SOLID', 'SOLVE', 'SORRY', 'SOUND', 'SOUTH', 'SPACE', 'SPARE', 'SPEAK', 'SPEED', 'SPEND', 'SPENT', 'SPLIT', 'SPOKE', 'SPORT', 'STAFF', 'STAGE', 'STAKE', 'STAND', 'START', 'STATE', 'STEAM', 'STEEL', 'STEEP', 'STEER', 'STEPS', 'STICK', 'STILL', 'STOCK', 'STONE', 'STOOD', 'STORE', 'STORM', 'STORY', 'STRIP', 'STUCK', 'STUDY', 'STUFF', 'STYLE', 'SUGAR', 'SUITE', 'SUPER', 'SWEET', 'TABLE', 'TAKEN', 'TASTE', 'TAXES', 'TEACH', 'TEETH', 'TERRY', 'TEXAS', 'THANK', 'THEFT', 'THEIR', 'THEME', 'THERE', 'THESE', 'THICK', 'THING', 'THINK', 'THIRD', 'THOSE', 'THREE', 'THREW', 'THROW', 'THUMB', 'TIGHT', 'TIMER', 'TIMES', 'TITLE', 'TODAY', 'TOPIC', 'TOTAL', 'TOUCH', 'TOUGH', 'TOWER', 'TRACK', 'TRADE', 'TRAIN', 'TREAT', 'TREND', 'TRIAL', 'TRIBE', 'TRICK', 'TRIED', 'TRIES', 'TRIPS', 'TRULY', 'TRUNK', 'TRUST', 'TRUTH', 'TWICE', 'TWIST', 'TYLER', 'TYPES', 'UNCLE', 'UNDER', 'UNDUE', 'UNION', 'UNITY', 'UNTIL', 'UPPER', 'UPSET', 'URBAN', 'URGED', 'USAGE', 'USUAL', 'VALID', 'VALUE', 'VIDEO', 'VIRUS', 'VISIT', 'VITAL', 'VOCAL', 'WASTE', 'WATCH', 'WATER', 'WAVES', 'WAYS', 'WEAK', 'WEALTH', 'WEAPON', 'WEARY', 'WEAVE', 'WEDGE', 'WELSH', 'WHEEL', 'WHERE', 'WHICH', 'WHILE', 'WHITE', 'WHOLE', 'WHOSE', 'WOMAN', 'WOMEN', 'WORLD', 'WORRY', 'WORSE', 'WORST', 'WORTH', 'WOULD', 'WRITE', 'WRONG', 'WROTE', 'YARDS', 'YOUNG', 'YOUTH'}
        
        # Filter out common words
        filtered_tickers = [ticker for ticker in tickers if ticker not in common_words]
        
        return list(set(filtered_tickers))  # Remove duplicates
    
    def extract_prices(self, text: str) -> List[str]:
        """
        Extract price information from text.
        
        Args:
            text: Input text
            
        Returns:
            List of price strings
        """
        prices = self.price_pattern.findall(text)
        return prices
    
    def extract_percentages(self, text: str) -> List[str]:
        """
        Extract percentage information from text.
        
        Args:
            text: Input text
            
        Returns:
            List of percentage strings
        """
        percentages = self.percentage_pattern.findall(text)
        return percentages
    
    def extract_currencies(self, text: str) -> List[str]:
        """
        Extract currency information from text.
        
        Args:
            text: Input text
            
        Returns:
            List of currency strings
        """
        currencies = self.currency_pattern.findall(text)
        return currencies
    
    def extract_financial_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract financial entities from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of extracted entities
        """
        entities = {
            'tickers': self.extract_tickers(text),
            'prices': self.extract_prices(text),
            'percentages': self.extract_percentages(text),
            'currencies': self.extract_currencies(text)
        }
        
        return entities
    
    def categorize_news(self, text: str) -> Dict[str, float]:
        """
        Categorize news based on content.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of category scores
        """
        text_lower = text.lower()
        category_scores = {}
        
        for category, keywords in self.news_categories.items():
            score = 0
            for keyword in keywords:
                score += text_lower.count(keyword)
            
            # Normalize by text length
            text_length = len(text.split())
            if text_length > 0:
                score = score / text_length
            
            category_scores[category] = score
        
        return category_scores
    
    def calculate_news_importance(self, text: str, entities: Dict[str, List[str]]) -> float:
        """
        Calculate news importance score.
        
        Args:
            text: Input text
            entities: Extracted entities
            
        Returns:
            Importance score
        """
        importance = 0.0
        
        # Base importance from text length
        text_length = len(text.split())
        importance += min(text_length / 100, 1.0) * 0.2
        
        # Importance from financial terms
        financial_term_count = sum(1 for term in self.financial_terms if term in text.lower())
        importance += min(financial_term_count / 10, 1.0) * 0.3
        
        # Importance from entities
        entity_count = sum(len(entity_list) for entity_list in entities.values())
        importance += min(entity_count / 5, 1.0) * 0.3
        
        # Importance from categories
        categories = self.categorize_news(text)
        max_category_score = max(categories.values()) if categories else 0
        importance += min(max_category_score * 10, 1.0) * 0.2
        
        return min(importance, 1.0)
    
    def process_news_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single news article.
        
        Args:
            article: News article dictionary
            
        Returns:
            Processed article with analysis
        """
        # Extract basic information
        title = article.get('title', '')
        content = article.get('content', '')
        url = article.get('url', '')
        published_date = article.get('published_date', datetime.now().isoformat())
        source = article.get('source', '')
        
        # Combine title and content
        full_text = f"{title} {content}"
        
        # Clean text
        cleaned_text = self.clean_news_text(full_text)
        
        # Extract entities
        entities = self.extract_financial_entities(cleaned_text)
        
        # Categorize news
        categories = self.categorize_news(cleaned_text)
        
        # Calculate importance
        importance = self.calculate_news_importance(cleaned_text, entities)
        
        # Generate article hash for deduplication
        article_hash = hashlib.md5(cleaned_text.encode()).hexdigest()
        
        # Create processed article
        processed_article = {
            'title': title,
            'content': content,
            'cleaned_text': cleaned_text,
            'url': url,
            'published_date': published_date,
            'source': source,
            'entities': entities,
            'categories': categories,
            'importance': importance,
            'article_hash': article_hash,
            'processed_at': datetime.now().isoformat()
        }
        
        # Store in history
        self.news_history.append(processed_article)
        
        return processed_article
    
    def process_news_batch(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of news articles.
        
        Args:
            articles: List of news articles
            
        Returns:
            List of processed articles
        """
        processed_articles = []
        
        for article in articles:
            try:
                processed_article = self.process_news_article(article)
                processed_articles.append(processed_article)
            except Exception as e:
                logger.warning(f"Failed to process article: {e}")
                continue
        
        return processed_articles
    
    def deduplicate_news(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate news articles.
        
        Args:
            articles: List of articles
            
        Returns:
            Deduplicated list of articles
        """
        seen_hashes = set()
        deduplicated = []
        
        for article in articles:
            article_hash = article.get('article_hash', '')
            if article_hash and article_hash not in seen_hashes:
                seen_hashes.add(article_hash)
                deduplicated.append(article)
        
        return deduplicated
    
    def get_news_statistics(self) -> Dict[str, Any]:
        """
        Get news processing statistics.
        
        Returns:
            Statistics dictionary
        """
        if not self.news_history:
            return {}
        
        # Extract statistics
        importance_scores = [article['importance'] for article in self.news_history]
        
        # Count categories
        all_categories = []
        for article in self.news_history:
            categories = article['categories']
            if categories:
                max_category = max(categories, key=categories.get)
                all_categories.append(max_category)
        
        category_counts = Counter(all_categories)
        
        # Count entities
        total_entities = 0
        entity_types = defaultdict(int)
        for article in self.news_history:
            entities = article['entities']
            for entity_type, entity_list in entities.items():
                entity_types[entity_type] += len(entity_list)
                total_entities += len(entity_list)
        
        # Count sources
        sources = [article['source'] for article in self.news_history if article['source']]
        source_counts = Counter(sources)
        
        return {
            'total_articles': len(self.news_history),
            'importance_stats': {
                'mean': np.mean(importance_scores),
                'std': np.std(importance_scores),
                'min': np.min(importance_scores),
                'max': np.max(importance_scores)
            },
            'category_distribution': dict(category_counts),
            'entity_counts': dict(entity_types),
            'total_entities': total_entities,
            'source_distribution': dict(source_counts),
            'unique_sources': len(source_counts)
        }


class NewsAggregator:
    """
    News aggregator for collecting and managing financial news from multiple sources.
    """
    
    def __init__(
        self,
        model_name: str = "news_aggregator",
        update_interval_minutes: int = 15
    ):
        """
        Initialize news aggregator.
        
        Args:
            model_name: Name for the aggregator
            update_interval_minutes: Update interval in minutes
        """
        self.model_name = model_name
        self.update_interval_minutes = update_interval_minutes
        
        # News sources configuration
        self.news_sources = {
            'rss_feeds': [],
            'api_sources': [],
            'web_scraping': []
        }
        
        # News processor
        self.news_processor = NewsProcessor()
        
        # Aggregated news
        self.aggregated_news = []
        self.last_update = None
        
        logger.info(f"Initialized News Aggregator: {model_name}")
    
    def add_rss_feed(self, feed_url: str, source_name: str) -> None:
        """
        Add RSS feed source.
        
        Args:
            feed_url: RSS feed URL
            source_name: Name of the source
        """
        if not FEEDPARSER_AVAILABLE:
            logger.warning("Feedparser not available. Cannot add RSS feed.")
            return
        
        self.news_sources['rss_feeds'].append({
            'url': feed_url,
            'name': source_name
        })
        
        logger.info(f"Added RSS feed: {source_name}")
    
    def add_api_source(self, api_config: Dict[str, Any]) -> None:
        """
        Add API news source.
        
        Args:
            api_config: API configuration dictionary
        """
        self.news_sources['api_sources'].append(api_config)
        
        logger.info(f"Added API source: {api_config.get('name', 'Unknown')}")
    
    def fetch_rss_news(self) -> List[Dict[str, Any]]:
        """
        Fetch news from RSS feeds.
        
        Returns:
            List of news articles
        """
        if not FEEDPARSER_AVAILABLE:
            logger.warning("Feedparser not available. Cannot fetch RSS news.")
            return []
        
        articles = []
        
        for feed_config in self.news_sources['rss_feeds']:
            try:
                feed_url = feed_config['url']
                source_name = feed_config['name']
                
                # Parse RSS feed
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries:
                    article = {
                        'title': entry.get('title', ''),
                        'content': entry.get('summary', ''),
                        'url': entry.get('link', ''),
                        'published_date': entry.get('published', datetime.now().isoformat()),
                        'source': source_name
                    }
                    articles.append(article)
                
                logger.info(f"Fetched {len(feed.entries)} articles from {source_name}")
                
            except Exception as e:
                logger.warning(f"Failed to fetch RSS feed {feed_config['name']}: {e}")
                continue
        
        return articles
    
    def fetch_api_news(self) -> List[Dict[str, Any]]:
        """
        Fetch news from API sources.
        
        Returns:
            List of news articles
        """
        if not REQUESTS_AVAILABLE:
            logger.warning("Requests not available. Cannot fetch API news.")
            return []
        
        articles = []
        
        for api_config in self.news_sources['api_sources']:
            try:
                api_url = api_config['url']
                api_key = api_config.get('api_key', '')
                headers = api_config.get('headers', {})
                params = api_config.get('params', {})
                
                # Add API key to params if provided
                if api_key:
                    params['api_key'] = api_key
                
                # Make API request
                response = requests.get(api_url, headers=headers, params=params, timeout=30)
                response.raise_for_status()
                
                # Parse response based on API format
                data = response.json()
                articles_data = data.get('articles', data.get('results', []))
                
                for article_data in articles_data:
                    article = {
                        'title': article_data.get('title', ''),
                        'content': article_data.get('description', article_data.get('content', '')),
                        'url': article_data.get('url', ''),
                        'published_date': article_data.get('publishedAt', article_data.get('published_date', datetime.now().isoformat())),
                        'source': api_config.get('name', 'API')
                    }
                    articles.append(article)
                
                logger.info(f"Fetched {len(articles_data)} articles from {api_config.get('name', 'API')}")
                
            except Exception as e:
                logger.warning(f"Failed to fetch API news from {api_config.get('name', 'API')}: {e}")
                continue
        
        return articles
    
    def aggregate_news(self) -> List[Dict[str, Any]]:
        """
        Aggregate news from all sources.
        
        Returns:
            List of aggregated news articles
        """
        logger.info("Starting news aggregation")
        
        all_articles = []
        
        # Fetch from RSS feeds
        rss_articles = self.fetch_rss_news()
        all_articles.extend(rss_articles)
        
        # Fetch from API sources
        api_articles = self.fetch_api_news()
        all_articles.extend(api_articles)
        
        # Process articles
        processed_articles = self.news_processor.process_news_batch(all_articles)
        
        # Deduplicate
        deduplicated_articles = self.news_processor.deduplicate_news(processed_articles)
        
        # Sort by importance
        sorted_articles = sorted(deduplicated_articles, key=lambda x: x['importance'], reverse=True)
        
        # Update aggregated news
        self.aggregated_news = sorted_articles
        self.last_update = datetime.now()
        
        logger.info(f"Aggregated {len(sorted_articles)} unique articles")
        
        return sorted_articles
    
    def get_latest_news(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get latest news articles.
        
        Args:
            limit: Maximum number of articles to return
            
        Returns:
            List of latest news articles
        """
        # Check if we need to update
        if (self.last_update is None or 
            datetime.now() - self.last_update > timedelta(minutes=self.update_interval_minutes)):
            self.aggregate_news()
        
        return self.aggregated_news[:limit]
    
    def get_news_by_category(self, category: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get news articles by category.
        
        Args:
            category: News category
            limit: Maximum number of articles to return
            
        Returns:
            List of news articles in the category
        """
        if not self.aggregated_news:
            self.aggregate_news()
        
        category_articles = []
        
        for article in self.aggregated_news:
            categories = article['categories']
            if categories and categories.get(category, 0) > 0:
                category_articles.append(article)
        
        # Sort by category score
        category_articles.sort(key=lambda x: x['categories'].get(category, 0), reverse=True)
        
        return category_articles[:limit]
    
    def get_news_by_ticker(self, ticker: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get news articles mentioning a specific ticker.
        
        Args:
            ticker: Stock ticker symbol
            limit: Maximum number of articles to return
            
        Returns:
            List of news articles mentioning the ticker
        """
        if not self.aggregated_news:
            self.aggregate_news()
        
        ticker_articles = []
        
        for article in self.aggregated_news:
            entities = article['entities']
            if ticker.upper() in entities.get('tickers', []):
                ticker_articles.append(article)
        
        # Sort by importance
        ticker_articles.sort(key=lambda x: x['importance'], reverse=True)
        
        return ticker_articles[:limit]
    
    def get_news_statistics(self) -> Dict[str, Any]:
        """
        Get news aggregation statistics.
        
        Returns:
            Statistics dictionary
        """
        processor_stats = self.news_processor.get_news_statistics()
        
        return {
            'aggregator_name': self.model_name,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'update_interval_minutes': self.update_interval_minutes,
            'total_sources': len(self.news_sources['rss_feeds']) + len(self.news_sources['api_sources']),
            'rss_feeds': len(self.news_sources['rss_feeds']),
            'api_sources': len(self.news_sources['api_sources']),
            **processor_stats
        }

