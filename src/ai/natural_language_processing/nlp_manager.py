"""
Natural Language Processing Manager for Financial Market Intelligence

This module provides a unified interface for managing NLP models,
including sentiment analysis, news classification, and text processing.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime, timedelta
import json
import os
from collections import defaultdict, Counter
import warnings

# Import NLP components
from .sentiment_analyzer import SentimentAnalyzer, FinancialSentimentAnalyzer
from .news_processor import NewsProcessor, NewsAggregator
from .bert_models import BERTSentimentModel, BERTNewsClassifier

logger = logging.getLogger(__name__)

class NaturalLanguageProcessingManager:
    """
    Unified manager for all NLP models and text processing capabilities.
    """
    
    def __init__(
        self,
        model_name: str = "nlp_manager",
        enable_bert: bool = True,
        enable_financial_analysis: bool = True,
        cache_results: bool = True
    ):
        """
        Initialize NLP manager.
        
        Args:
            model_name: Name for the manager
            enable_bert: Whether to enable BERT models
            enable_financial_analysis: Whether to enable financial-specific analysis
            cache_results: Whether to cache analysis results
        """
        self.model_name = model_name
        self.enable_bert = enable_bert
        self.enable_financial_analysis = enable_financial_analysis
        self.cache_results = cache_results
        
        # Initialize components
        self.sentiment_analyzer = SentimentAnalyzer()
        self.news_processor = NewsProcessor()
        self.news_aggregator = NewsAggregator()
        
        if self.enable_financial_analysis:
            self.financial_sentiment_analyzer = FinancialSentimentAnalyzer()
        
        if self.enable_bert:
            try:
                self.bert_sentiment_model = BERTSentimentModel()
                self.bert_news_classifier = BERTNewsClassifier()
                logger.info("BERT models initialized successfully")
            except ImportError as e:
                logger.warning(f"BERT models not available: {e}")
                self.bert_sentiment_model = None
                self.bert_news_classifier = None
                self.enable_bert = False
        else:
            self.bert_sentiment_model = None
            self.bert_news_classifier = None
        
        # Results cache
        self.results_cache = {}
        
        # Analysis history
        self.analysis_history = []
        
        # Performance metrics
        self.performance_metrics = {
            'total_analyses': 0,
            'bert_analyses': 0,
            'financial_analyses': 0,
            'news_analyses': 0,
            'average_processing_time': 0.0
        }
        
        logger.info(f"Initialized NLP Manager: {model_name}")
    
    def analyze_sentiment(
        self,
        text: str,
        use_bert: bool = False,
        use_financial: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze sentiment of text using specified methods.
        
        Args:
            text: Input text
            use_bert: Whether to use BERT model
            use_financial: Whether to use financial-specific analysis
            
        Returns:
            Sentiment analysis results
        """
        start_time = datetime.now()
        
        # Check cache first
        if self.cache_results:
            cache_key = f"sentiment_{hash(text)}_{use_bert}_{use_financial}"
            if cache_key in self.results_cache:
                return self.results_cache[cache_key]
        
        results = {
            'text': text,
            'timestamp': datetime.now().isoformat(),
            'methods_used': []
        }
        
        # Basic sentiment analysis
        basic_sentiment = self.sentiment_analyzer.analyze_sentiment(text)
        results['basic_sentiment'] = basic_sentiment
        results['methods_used'].append('basic')
        
        # Financial sentiment analysis
        if use_financial and self.enable_financial_analysis:
            financial_sentiment = self.financial_sentiment_analyzer.analyze_financial_sentiment(text)
            results['financial_sentiment'] = financial_sentiment
            results['methods_used'].append('financial')
        
        # BERT sentiment analysis
        if use_bert and self.enable_bert and self.bert_sentiment_model:
            try:
                bert_sentiment = self.bert_sentiment_model.predict_sentiment(text)
                results['bert_sentiment'] = bert_sentiment
                results['methods_used'].append('bert')
            except Exception as e:
                logger.warning(f"BERT sentiment analysis failed: {e}")
                results['bert_error'] = str(e)
        
        # Combine results
        results['combined_sentiment'] = self._combine_sentiment_results(results)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        results['processing_time'] = processing_time
        
        # Update performance metrics
        self._update_performance_metrics('sentiment', processing_time)
        
        # Cache results
        if self.cache_results:
            self.results_cache[cache_key] = results
        
        # Store in history
        self.analysis_history.append(results)
        
        return results
    
    def classify_news(
        self,
        text: str,
        use_bert: bool = False
    ) -> Dict[str, Any]:
        """
        Classify news article into categories.
        
        Args:
            text: Input text
            use_bert: Whether to use BERT model
            
        Returns:
            News classification results
        """
        start_time = datetime.now()
        
        # Check cache first
        if self.cache_results:
            cache_key = f"classification_{hash(text)}_{use_bert}"
            if cache_key in self.results_cache:
                return self.results_cache[cache_key]
        
        results = {
            'text': text,
            'timestamp': datetime.now().isoformat(),
            'methods_used': []
        }
        
        # Basic news processing
        processed_article = self.news_processor.process_news_article({
            'title': '',
            'content': text,
            'url': '',
            'published_date': datetime.now().isoformat(),
            'source': 'manual_input'
        })
        
        results['basic_classification'] = {
            'categories': processed_article['categories'],
            'importance': processed_article['importance'],
            'entities': processed_article['entities']
        }
        results['methods_used'].append('basic')
        
        # BERT news classification
        if use_bert and self.enable_bert and self.bert_news_classifier:
            try:
                bert_classification = self.bert_news_classifier.classify_news(text)
                results['bert_classification'] = bert_classification
                results['methods_used'].append('bert')
            except Exception as e:
                logger.warning(f"BERT news classification failed: {e}")
                results['bert_error'] = str(e)
        
        # Combine results
        results['combined_classification'] = self._combine_classification_results(results)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        results['processing_time'] = processing_time
        
        # Update performance metrics
        self._update_performance_metrics('classification', processing_time)
        
        # Cache results
        if self.cache_results:
            self.results_cache[cache_key] = results
        
        # Store in history
        self.analysis_history.append(results)
        
        return results
    
    def analyze_news_batch(
        self,
        articles: List[Dict[str, Any]],
        use_bert: bool = False,
        use_financial: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Analyze a batch of news articles.
        
        Args:
            articles: List of news articles
            use_bert: Whether to use BERT models
            use_financial: Whether to use financial-specific analysis
            
        Returns:
            List of analysis results
        """
        results = []
        
        for article in articles:
            # Extract text from article
            text = f"{article.get('title', '')} {article.get('content', '')}"
            
            # Analyze sentiment
            sentiment_result = self.analyze_sentiment(text, use_bert, use_financial)
            
            # Classify news
            classification_result = self.classify_news(text, use_bert)
            
            # Combine results
            combined_result = {
                'article': article,
                'sentiment_analysis': sentiment_result,
                'news_classification': classification_result,
                'timestamp': datetime.now().isoformat()
            }
            
            results.append(combined_result)
        
        return results
    
    def get_market_sentiment_summary(
        self,
        time_window_hours: int = 24,
        use_financial: bool = True
    ) -> Dict[str, Any]:
        """
        Get market sentiment summary for a time window.
        
        Args:
            time_window_hours: Time window in hours
            use_financial: Whether to use financial-specific analysis
            
        Returns:
            Market sentiment summary
        """
        if use_financial and self.enable_financial_analysis:
            return self.financial_sentiment_analyzer.get_market_sentiment_summary(time_window_hours)
        else:
            # Use basic sentiment analyzer
            cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
            recent_analyses = [
                a for a in self.analysis_history
                if datetime.fromisoformat(a['timestamp']) >= cutoff_time
            ]
            
            if not recent_analyses:
                return {}
            
            # Calculate aggregate sentiment
            sentiment_scores = []
            for analysis in recent_analyses:
                if 'basic_sentiment' in analysis:
                    sentiment_scores.append(analysis['basic_sentiment']['compound_score'])
            
            if not sentiment_scores:
                return {}
            
            avg_sentiment = np.mean(sentiment_scores)
            
            # Determine overall sentiment
            if avg_sentiment >= 0.05:
                overall_sentiment = 'positive'
            elif avg_sentiment <= -0.05:
                overall_sentiment = 'negative'
            else:
                overall_sentiment = 'neutral'
            
            return {
                'time_window_hours': time_window_hours,
                'total_analyses': len(recent_analyses),
                'overall_sentiment': overall_sentiment,
                'avg_sentiment_score': avg_sentiment,
                'sentiment_trend': 'increasing' if avg_sentiment > 0 else 'decreasing'
            }
    
    def get_news_by_category(
        self,
        category: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get news articles by category.
        
        Args:
            category: News category
            limit: Maximum number of articles
            
        Returns:
            List of news articles in the category
        """
        return self.news_aggregator.get_news_by_category(category, limit)
    
    def get_news_by_ticker(
        self,
        ticker: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get news articles mentioning a specific ticker.
        
        Args:
            ticker: Stock ticker symbol
            limit: Maximum number of articles
            
        Returns:
            List of news articles mentioning the ticker
        """
        return self.news_aggregator.get_news_by_ticker(ticker, limit)
    
    def aggregate_news(self) -> List[Dict[str, Any]]:
        """
        Aggregate news from all sources.
        
        Returns:
            List of aggregated news articles
        """
        return self.news_aggregator.aggregate_news()
    
    def _combine_sentiment_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine sentiment results from different methods.
        
        Args:
            results: Results dictionary
            
        Returns:
            Combined sentiment results
        """
        combined = {
            'sentiment_label': 'neutral',
            'sentiment_score': 0.0,
            'confidence': 0.0,
            'method_weights': {}
        }
        
        # Get basic sentiment
        if 'basic_sentiment' in results:
            basic = results['basic_sentiment']
            combined['sentiment_score'] += basic['compound_score'] * 0.4
            combined['confidence'] += basic['confidence'] * 0.4
            combined['method_weights']['basic'] = 0.4
        
        # Get financial sentiment
        if 'financial_sentiment' in results:
            financial = results['financial_sentiment']
            combined['sentiment_score'] += financial['financial_sentiment_score'] * 0.3
            combined['confidence'] += financial['confidence'] * 0.3
            combined['method_weights']['financial'] = 0.3
        
        # Get BERT sentiment
        if 'bert_sentiment' in results:
            bert = results['bert_sentiment']
            combined['sentiment_score'] += bert['sentiment_score'] * 0.3
            combined['confidence'] += bert['confidence'] * 0.3
            combined['method_weights']['bert'] = 0.3
        
        # Determine sentiment label
        if combined['sentiment_score'] >= 0.1:
            combined['sentiment_label'] = 'positive'
        elif combined['sentiment_score'] <= -0.1:
            combined['sentiment_label'] = 'negative'
        else:
            combined['sentiment_label'] = 'neutral'
        
        return combined
    
    def _combine_classification_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine classification results from different methods.
        
        Args:
            results: Results dictionary
            
        Returns:
            Combined classification results
        """
        combined = {
            'primary_category': 'market',
            'category_scores': {},
            'confidence': 0.0,
            'method_weights': {}
        }
        
        # Get basic classification
        if 'basic_classification' in results:
            basic = results['basic_classification']
            for category, score in basic['categories'].items():
                if category not in combined['category_scores']:
                    combined['category_scores'][category] = 0.0
                combined['category_scores'][category] += score * 0.5
            combined['confidence'] += basic['importance'] * 0.5
            combined['method_weights']['basic'] = 0.5
        
        # Get BERT classification
        if 'bert_classification' in results:
            bert = results['bert_classification']
            for category_info in bert['top_categories']:
                category = category_info['category']
                score = category_info['probability']
                if category not in combined['category_scores']:
                    combined['category_scores'][category] = 0.0
                combined['category_scores'][category] += score * 0.5
            combined['confidence'] += bert['confidence'] * 0.5
            combined['method_weights']['bert'] = 0.5
        
        # Determine primary category
        if combined['category_scores']:
            combined['primary_category'] = max(combined['category_scores'], key=combined['category_scores'].get)
        
        return combined
    
    def _update_performance_metrics(self, analysis_type: str, processing_time: float) -> None:
        """
        Update performance metrics.
        
        Args:
            analysis_type: Type of analysis performed
            processing_time: Processing time in seconds
        """
        self.performance_metrics['total_analyses'] += 1
        
        if analysis_type == 'sentiment':
            if 'bert' in analysis_type:
                self.performance_metrics['bert_analyses'] += 1
            if 'financial' in analysis_type:
                self.performance_metrics['financial_analyses'] += 1
        elif analysis_type == 'classification':
            self.performance_metrics['news_analyses'] += 1
        
        # Update average processing time
        total_analyses = self.performance_metrics['total_analyses']
        current_avg = self.performance_metrics['average_processing_time']
        self.performance_metrics['average_processing_time'] = (
            (current_avg * (total_analyses - 1) + processing_time) / total_analyses
        )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics.
        
        Returns:
            Performance metrics dictionary
        """
        return {
            **self.performance_metrics,
            'cache_size': len(self.results_cache),
            'history_size': len(self.analysis_history),
            'bert_enabled': self.enable_bert,
            'financial_analysis_enabled': self.enable_financial_analysis,
            'cache_enabled': self.cache_results
        }
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """
        Get analysis statistics.
        
        Returns:
            Analysis statistics dictionary
        """
        if not self.analysis_history:
            return {}
        
        # Extract statistics
        sentiment_labels = []
        categories = []
        processing_times = []
        
        for analysis in self.analysis_history:
            if 'combined_sentiment' in analysis:
                sentiment_labels.append(analysis['combined_sentiment']['sentiment_label'])
            if 'combined_classification' in analysis:
                categories.append(analysis['combined_classification']['primary_category'])
            if 'processing_time' in analysis:
                processing_times.append(analysis['processing_time'])
        
        # Count sentiment labels
        sentiment_counts = Counter(sentiment_labels)
        
        # Count categories
        category_counts = Counter(categories)
        
        return {
            'total_analyses': len(self.analysis_history),
            'sentiment_distribution': dict(sentiment_counts),
            'category_distribution': dict(category_counts),
            'processing_time_stats': {
                'mean': np.mean(processing_times) if processing_times else 0.0,
                'std': np.std(processing_times) if processing_times else 0.0,
                'min': np.min(processing_times) if processing_times else 0.0,
                'max': np.max(processing_times) if processing_times else 0.0
            }
        }
    
    def clear_cache(self) -> None:
        """
        Clear the results cache.
        """
        self.results_cache.clear()
        logger.info("Results cache cleared")
    
    def clear_history(self) -> None:
        """
        Clear the analysis history.
        """
        self.analysis_history.clear()
        logger.info("Analysis history cleared")
    
    def save_results(self, filepath: str) -> None:
        """
        Save analysis results to file.
        
        Args:
            filepath: Path to save results
        """
        results_data = {
            'analysis_history': self.analysis_history,
            'performance_metrics': self.performance_metrics,
            'analysis_statistics': self.get_analysis_statistics(),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
    
    def load_results(self, filepath: str) -> None:
        """
        Load analysis results from file.
        
        Args:
            filepath: Path to load results from
        """
        if not os.path.exists(filepath):
            logger.warning(f"Results file not found: {filepath}")
            return
        
        with open(filepath, 'r') as f:
            results_data = json.load(f)
        
        self.analysis_history = results_data.get('analysis_history', [])
        self.performance_metrics = results_data.get('performance_metrics', self.performance_metrics)
        
        logger.info(f"Results loaded from {filepath}")
    
    def get_model_status(self) -> Dict[str, Any]:
        """
        Get status of all models.
        
        Returns:
            Model status dictionary
        """
        return {
            'sentiment_analyzer': 'available',
            'news_processor': 'available',
            'news_aggregator': 'available',
            'financial_sentiment_analyzer': 'available' if self.enable_financial_analysis else 'disabled',
            'bert_sentiment_model': 'available' if self.bert_sentiment_model else 'unavailable',
            'bert_news_classifier': 'available' if self.bert_news_classifier else 'unavailable',
            'bert_enabled': self.enable_bert,
            'financial_analysis_enabled': self.enable_financial_analysis,
            'cache_enabled': self.cache_results
        }

