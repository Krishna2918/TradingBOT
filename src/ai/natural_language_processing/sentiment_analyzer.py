"""
Sentiment Analysis for Financial News and Market Intelligence

This module implements sentiment analysis models specifically designed
for financial text data and market sentiment extraction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime, timedelta
import re
import string
from collections import Counter
import warnings

# Text processing libraries
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import WordNetLemmatizer
    from nltk.sentiment import SentimentIntensityAnalyzer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    warnings.warn("NLTK not available. Some features will be limited.")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    warnings.warn("TextBlob not available. Some features will be limited.")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VaderSentiment
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    warnings.warn("VADER not available. Some features will be limited.")

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Base sentiment analyzer for financial text data.
    """
    
    def __init__(
        self,
        model_name: str = "sentiment_analyzer",
        preprocess: bool = True
    ):
        """
        Initialize sentiment analyzer.
        
        Args:
            model_name: Name for the analyzer
            preprocess: Whether to preprocess text
        """
        self.model_name = model_name
        self.preprocess = preprocess
        
        # Initialize text processing components
        if NLTK_AVAILABLE:
            try:
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('wordnet', quiet=True)
                nltk.download('vader_lexicon', quiet=True)
                
                self.stop_words = set(stopwords.words('english'))
                self.lemmatizer = WordNetLemmatizer()
                self.nltk_sia = SentimentIntensityAnalyzer()
            except Exception as e:
                logger.warning(f"Failed to initialize NLTK components: {e}")
                self.stop_words = set()
                self.lemmatizer = None
                self.nltk_sia = None
        else:
            self.stop_words = set()
            self.lemmatizer = None
            self.nltk_sia = None
        
        if VADER_AVAILABLE:
            self.vader_sia = VaderSentiment()
        else:
            self.vader_sia = None
        
        # Financial sentiment history
        self.sentiment_history = []
        
        logger.info(f"Initialized Sentiment Analyzer: {model_name}")
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for sentiment analysis.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        if not self.preprocess:
            return text
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        if NLTK_AVAILABLE:
            try:
                tokens = word_tokenize(text)
            except Exception:
                tokens = text.split()
        else:
            tokens = text.split()
        
        return tokens
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Filtered tokens
        """
        if not self.stop_words:
            return tokens
        
        return [token for token in tokens if token not in self.stop_words]
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Lemmatized tokens
        """
        if not self.lemmatizer:
            return tokens
        
        try:
            return [self.lemmatizer.lemmatize(token) for token in tokens]
        except Exception:
            return tokens
    
    def analyze_sentiment_nltk(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using NLTK VADER.
        
        Args:
            text: Input text
            
        Returns:
            Sentiment scores
        """
        if not self.nltk_sia:
            return {'compound': 0.0, 'pos': 0.0, 'neu': 0.0, 'neg': 0.0}
        
        try:
            scores = self.nltk_sia.polarity_scores(text)
            return scores
        except Exception as e:
            logger.warning(f"NLTK sentiment analysis failed: {e}")
            return {'compound': 0.0, 'pos': 0.0, 'neu': 0.0, 'neg': 0.0}
    
    def analyze_sentiment_vader(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using VADER.
        
        Args:
            text: Input text
            
        Returns:
            Sentiment scores
        """
        if not self.vader_sia:
            return {'compound': 0.0, 'pos': 0.0, 'neu': 0.0, 'neg': 0.0}
        
        try:
            scores = self.vader_sia.polarity_scores(text)
            return scores
        except Exception as e:
            logger.warning(f"VADER sentiment analysis failed: {e}")
            return {'compound': 0.0, 'pos': 0.0, 'neu': 0.0, 'neg': 0.0}
    
    def analyze_sentiment_textblob(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using TextBlob.
        
        Args:
            text: Input text
            
        Returns:
            Sentiment scores
        """
        if not TEXTBLOB_AVAILABLE:
            return {'polarity': 0.0, 'subjectivity': 0.0}
        
        try:
            blob = TextBlob(text)
            return {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            }
        except Exception as e:
            logger.warning(f"TextBlob sentiment analysis failed: {e}")
            return {'polarity': 0.0, 'subjectivity': 0.0}
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment using multiple methods.
        
        Args:
            text: Input text
            
        Returns:
            Comprehensive sentiment analysis
        """
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Get sentiment scores from different methods
        nltk_scores = self.analyze_sentiment_nltk(processed_text)
        vader_scores = self.analyze_sentiment_vader(processed_text)
        textblob_scores = self.analyze_sentiment_textblob(processed_text)
        
        # Calculate overall sentiment
        compound_score = (nltk_scores['compound'] + vader_scores['compound']) / 2
        polarity_score = textblob_scores['polarity']
        subjectivity_score = textblob_scores['subjectivity']
        
        # Determine sentiment label
        if compound_score >= 0.05:
            sentiment_label = 'positive'
        elif compound_score <= -0.05:
            sentiment_label = 'negative'
        else:
            sentiment_label = 'neutral'
        
        # Calculate confidence
        confidence = abs(compound_score)
        
        result = {
            'text': text,
            'processed_text': processed_text,
            'sentiment_label': sentiment_label,
            'compound_score': compound_score,
            'polarity_score': polarity_score,
            'subjectivity_score': subjectivity_score,
            'confidence': confidence,
            'nltk_scores': nltk_scores,
            'vader_scores': vader_scores,
            'textblob_scores': textblob_scores,
            'timestamp': datetime.now().isoformat()
        }
        
        # Store in history
        self.sentiment_history.append(result)
        
        return result
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze sentiment for a batch of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of sentiment analysis results
        """
        results = []
        
        for text in texts:
            result = self.analyze_sentiment(text)
            results.append(result)
        
        return results
    
    def get_sentiment_statistics(self) -> Dict[str, Any]:
        """
        Get sentiment analysis statistics.
        
        Returns:
            Statistics dictionary
        """
        if not self.sentiment_history:
            return {}
        
        # Extract scores
        compound_scores = [r['compound_score'] for r in self.sentiment_history]
        polarity_scores = [r['polarity_score'] for r in self.sentiment_history]
        subjectivity_scores = [r['subjectivity_score'] for r in self.sentiment_history]
        confidence_scores = [r['confidence'] for r in self.sentiment_history]
        
        # Count sentiment labels
        sentiment_labels = [r['sentiment_label'] for r in self.sentiment_history]
        sentiment_counts = Counter(sentiment_labels)
        
        return {
            'total_analyses': len(self.sentiment_history),
            'compound_stats': {
                'mean': np.mean(compound_scores),
                'std': np.std(compound_scores),
                'min': np.min(compound_scores),
                'max': np.max(compound_scores)
            },
            'polarity_stats': {
                'mean': np.mean(polarity_scores),
                'std': np.std(polarity_scores),
                'min': np.min(polarity_scores),
                'max': np.max(polarity_scores)
            },
            'subjectivity_stats': {
                'mean': np.mean(subjectivity_scores),
                'std': np.std(subjectivity_scores),
                'min': np.min(subjectivity_scores),
                'max': np.max(subjectivity_scores)
            },
            'confidence_stats': {
                'mean': np.mean(confidence_scores),
                'std': np.std(confidence_scores),
                'min': np.min(confidence_scores),
                'max': np.max(confidence_scores)
            },
            'sentiment_distribution': dict(sentiment_counts)
        }


class FinancialSentimentAnalyzer(SentimentAnalyzer):
    """
    Specialized sentiment analyzer for financial news and market data.
    """
    
    def __init__(
        self,
        model_name: str = "financial_sentiment_analyzer",
        preprocess: bool = True
    ):
        """
        Initialize financial sentiment analyzer.
        
        Args:
            model_name: Name for the analyzer
            preprocess: Whether to preprocess text
        """
        super().__init__(model_name, preprocess)
        
        # Financial-specific terms and weights
        self.bullish_terms = {
            'bullish', 'bull', 'rally', 'surge', 'gain', 'rise', 'up', 'positive',
            'optimistic', 'strong', 'growth', 'profit', 'earnings', 'beat', 'exceed',
            'outperform', 'upgrade', 'buy', 'outperform', 'strong', 'robust'
        }
        
        self.bearish_terms = {
            'bearish', 'bear', 'decline', 'fall', 'drop', 'down', 'negative',
            'pessimistic', 'weak', 'loss', 'miss', 'underperform', 'downgrade',
            'sell', 'crash', 'plunge', 'tumble', 'slump', 'recession'
        }
        
        self.volatility_terms = {
            'volatile', 'volatility', 'uncertain', 'unstable', 'fluctuate',
            'swing', 'wild', 'turbulent', 'choppy', 'erratic'
        }
        
        self.uncertainty_terms = {
            'uncertain', 'unclear', 'unknown', 'unpredictable', 'risky',
            'caution', 'warning', 'concern', 'worry', 'fear'
        }
        
        # Financial sentiment history
        self.financial_sentiment_history = []
        
        logger.info(f"Initialized Financial Sentiment Analyzer: {model_name}")
    
    def extract_financial_terms(self, text: str) -> Dict[str, List[str]]:
        """
        Extract financial-specific terms from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of extracted terms by category
        """
        processed_text = self.preprocess_text(text)
        tokens = self.tokenize_text(processed_text)
        
        extracted_terms = {
            'bullish': [],
            'bearish': [],
            'volatility': [],
            'uncertainty': []
        }
        
        for token in tokens:
            if token in self.bullish_terms:
                extracted_terms['bullish'].append(token)
            elif token in self.bearish_terms:
                extracted_terms['bearish'].append(token)
            elif token in self.volatility_terms:
                extracted_terms['volatility'].append(token)
            elif token in self.uncertainty_terms:
                extracted_terms['uncertainty'].append(token)
        
        return extracted_terms
    
    def calculate_financial_sentiment_score(self, text: str) -> Dict[str, float]:
        """
        Calculate financial-specific sentiment score.
        
        Args:
            text: Input text
            
        Returns:
            Financial sentiment scores
        """
        extracted_terms = self.extract_financial_terms(text)
        
        # Calculate term-based scores
        bullish_score = len(extracted_terms['bullish'])
        bearish_score = len(extracted_terms['bearish'])
        volatility_score = len(extracted_terms['volatility'])
        uncertainty_score = len(extracted_terms['uncertainty'])
        
        # Normalize scores by text length
        text_length = len(self.tokenize_text(text))
        if text_length > 0:
            bullish_score = bullish_score / text_length
            bearish_score = bearish_score / text_length
            volatility_score = volatility_score / text_length
            uncertainty_score = uncertainty_score / text_length
        
        # Calculate overall financial sentiment
        financial_sentiment = bullish_score - bearish_score
        
        return {
            'financial_sentiment': financial_sentiment,
            'bullish_score': bullish_score,
            'bearish_score': bearish_score,
            'volatility_score': volatility_score,
            'uncertainty_score': uncertainty_score,
            'extracted_terms': extracted_terms
        }
    
    def analyze_financial_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze financial sentiment using both general and financial-specific methods.
        
        Args:
            text: Input text
            
        Returns:
            Comprehensive financial sentiment analysis
        """
        # Get general sentiment analysis
        general_sentiment = self.analyze_sentiment(text)
        
        # Get financial-specific sentiment
        financial_sentiment = self.calculate_financial_sentiment_score(text)
        
        # Combine scores
        combined_sentiment_score = (
            general_sentiment['compound_score'] * 0.6 +
            financial_sentiment['financial_sentiment'] * 0.4
        )
        
        # Determine financial sentiment label
        if combined_sentiment_score >= 0.1:
            financial_sentiment_label = 'bullish'
        elif combined_sentiment_score <= -0.1:
            financial_sentiment_label = 'bearish'
        else:
            financial_sentiment_label = 'neutral'
        
        # Calculate market impact score
        market_impact = (
            abs(combined_sentiment_score) +
            financial_sentiment['volatility_score'] * 0.3 +
            financial_sentiment['uncertainty_score'] * 0.2
        )
        
        result = {
            **general_sentiment,
            'financial_sentiment_score': combined_sentiment_score,
            'financial_sentiment_label': financial_sentiment_label,
            'market_impact_score': market_impact,
            'bullish_score': financial_sentiment['bullish_score'],
            'bearish_score': financial_sentiment['bearish_score'],
            'volatility_score': financial_sentiment['volatility_score'],
            'uncertainty_score': financial_sentiment['uncertainty_score'],
            'extracted_terms': financial_sentiment['extracted_terms']
        }
        
        # Store in financial sentiment history
        self.financial_sentiment_history.append(result)
        
        return result
    
    def analyze_financial_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze financial sentiment for a batch of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of financial sentiment analysis results
        """
        results = []
        
        for text in texts:
            result = self.analyze_financial_sentiment(text)
            results.append(result)
        
        return results
    
    def get_financial_sentiment_statistics(self) -> Dict[str, Any]:
        """
        Get financial sentiment analysis statistics.
        
        Returns:
            Financial sentiment statistics
        """
        if not self.financial_sentiment_history:
            return {}
        
        # Extract financial scores
        financial_scores = [r['financial_sentiment_score'] for r in self.financial_sentiment_history]
        market_impact_scores = [r['market_impact_score'] for r in self.financial_sentiment_history]
        bullish_scores = [r['bullish_score'] for r in self.financial_sentiment_history]
        bearish_scores = [r['bearish_score'] for r in self.financial_sentiment_history]
        volatility_scores = [r['volatility_score'] for r in self.financial_sentiment_history]
        uncertainty_scores = [r['uncertainty_score'] for r in self.financial_sentiment_history]
        
        # Count financial sentiment labels
        financial_labels = [r['financial_sentiment_label'] for r in self.financial_sentiment_history]
        financial_label_counts = Counter(financial_labels)
        
        # Get general statistics
        general_stats = self.get_sentiment_statistics()
        
        return {
            **general_stats,
            'financial_sentiment_stats': {
                'mean': np.mean(financial_scores),
                'std': np.std(financial_scores),
                'min': np.min(financial_scores),
                'max': np.max(financial_scores)
            },
            'market_impact_stats': {
                'mean': np.mean(market_impact_scores),
                'std': np.std(market_impact_scores),
                'min': np.min(market_impact_scores),
                'max': np.max(market_impact_scores)
            },
            'bullish_stats': {
                'mean': np.mean(bullish_scores),
                'std': np.std(bullish_scores),
                'min': np.min(bullish_scores),
                'max': np.max(bullish_scores)
            },
            'bearish_stats': {
                'mean': np.mean(bearish_scores),
                'std': np.std(bearish_scores),
                'min': np.min(bearish_scores),
                'max': np.max(bearish_scores)
            },
            'volatility_stats': {
                'mean': np.mean(volatility_scores),
                'std': np.std(volatility_scores),
                'min': np.min(volatility_scores),
                'max': np.max(volatility_scores)
            },
            'uncertainty_stats': {
                'mean': np.mean(uncertainty_scores),
                'std': np.std(uncertainty_scores),
                'min': np.min(uncertainty_scores),
                'max': np.max(uncertainty_scores)
            },
            'financial_sentiment_distribution': dict(financial_label_counts)
        }
    
    def get_market_sentiment_summary(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """
        Get market sentiment summary for a time window.
        
        Args:
            time_window_hours: Time window in hours
            
        Returns:
            Market sentiment summary
        """
        if not self.financial_sentiment_history:
            return {}
        
        # Filter by time window
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        recent_sentiments = [
            s for s in self.financial_sentiment_history
            if datetime.fromisoformat(s['timestamp']) >= cutoff_time
        ]
        
        if not recent_sentiments:
            return {}
        
        # Calculate aggregate scores
        avg_financial_sentiment = np.mean([s['financial_sentiment_score'] for s in recent_sentiments])
        avg_market_impact = np.mean([s['market_impact_score'] for s in recent_sentiments])
        avg_volatility = np.mean([s['volatility_score'] for s in recent_sentiments])
        avg_uncertainty = np.mean([s['uncertainty_score'] for s in recent_sentiments])
        
        # Determine overall market sentiment
        if avg_financial_sentiment >= 0.1:
            overall_sentiment = 'bullish'
        elif avg_financial_sentiment <= -0.1:
            overall_sentiment = 'bearish'
        else:
            overall_sentiment = 'neutral'
        
        return {
            'time_window_hours': time_window_hours,
            'total_articles': len(recent_sentiments),
            'overall_sentiment': overall_sentiment,
            'avg_financial_sentiment': avg_financial_sentiment,
            'avg_market_impact': avg_market_impact,
            'avg_volatility': avg_volatility,
            'avg_uncertainty': avg_uncertainty,
            'sentiment_trend': 'increasing' if avg_financial_sentiment > 0 else 'decreasing',
            'market_risk_level': 'high' if avg_volatility > 0.1 or avg_uncertainty > 0.1 else 'low'
        }

