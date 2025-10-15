"""
Unit tests for Natural Language Processing models.

This module contains comprehensive unit tests for all NLP components
including sentiment analysis, news processing, BERT models, and text classification.
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import warnings

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore")

# Import NLP components
from src.ai.natural_language_processing.sentiment_analyzer import SentimentAnalyzer, FinancialSentimentAnalyzer
from src.ai.natural_language_processing.news_processor import NewsProcessor, NewsAggregator
from src.ai.natural_language_processing.text_classifier import TextClassifier, NewsClassifier
from src.ai.natural_language_processing.nlp_manager import NaturalLanguageProcessingManager


class TestSentimentAnalyzer(unittest.TestCase):
    """Test cases for SentimentAnalyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = SentimentAnalyzer()
        self.sample_texts = [
            "This is a great product!",
            "I hate this terrible service.",
            "The weather is okay today.",
            "Amazing results and excellent performance!",
            "Very disappointing and poor quality."
        ]
    
    def test_initialization(self):
        """Test sentiment analyzer initialization."""
        self.assertEqual(self.analyzer.model_name, "sentiment_analyzer")
        self.assertTrue(self.analyzer.preprocess)
        self.assertEqual(len(self.analyzer.sentiment_history), 0)
    
    def test_preprocess_text(self):
        """Test text preprocessing."""
        text = "This is a GREAT product! Visit https://example.com for more info."
        processed = self.analyzer.preprocess_text(text)
        
        self.assertIn("great", processed)
        self.assertNotIn("https://example.com", processed)
        self.assertNotIn("!", processed)
    
    def test_analyze_sentiment(self):
        """Test sentiment analysis."""
        result = self.analyzer.analyze_sentiment("This is a great product!")
        
        self.assertIn('sentiment_label', result)
        self.assertIn('compound_score', result)
        self.assertIn('confidence', result)
        self.assertIn('timestamp', result)
        self.assertEqual(result['text'], "This is a great product!")
    
    def test_analyze_batch(self):
        """Test batch sentiment analysis."""
        results = self.analyzer.analyze_batch(self.sample_texts)
        
        self.assertEqual(len(results), len(self.sample_texts))
        for result in results:
            self.assertIn('sentiment_label', result)
            self.assertIn('compound_score', result)
    
    def test_sentiment_statistics(self):
        """Test sentiment statistics."""
        # Analyze some texts first
        self.analyzer.analyze_batch(self.sample_texts)
        
        stats = self.analyzer.get_sentiment_statistics()
        
        self.assertIn('total_analyses', stats)
        self.assertIn('compound_stats', stats)
        self.assertIn('sentiment_distribution', stats)
        self.assertEqual(stats['total_analyses'], len(self.sample_texts))


class TestFinancialSentimentAnalyzer(unittest.TestCase):
    """Test cases for FinancialSentimentAnalyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = FinancialSentimentAnalyzer()
        self.financial_texts = [
            "The stock market is bullish and showing strong growth.",
            "Bearish sentiment is causing a market decline.",
            "High volatility and uncertainty in the market.",
            "Earnings beat expectations with robust performance.",
            "Market crash and economic recession fears."
        ]
    
    def test_initialization(self):
        """Test financial sentiment analyzer initialization."""
        self.assertEqual(self.analyzer.model_name, "financial_sentiment_analyzer")
        self.assertIn('bullish', self.analyzer.bullish_terms)
        self.assertIn('bearish', self.analyzer.bearish_terms)
        self.assertIn('volatile', self.analyzer.volatility_terms)
    
    def test_extract_financial_terms(self):
        """Test financial term extraction."""
        text = "The market is bullish with high volatility and uncertainty."
        terms = self.analyzer.extract_financial_terms(text)
        
        self.assertIn('bullish', terms['bullish'])
        self.assertIn('volatility', terms['volatility'])
        self.assertIn('uncertainty', terms['uncertainty'])
    
    def test_calculate_financial_sentiment_score(self):
        """Test financial sentiment score calculation."""
        text = "Bullish market with strong growth and positive earnings."
        scores = self.analyzer.calculate_financial_sentiment_score(text)
        
        self.assertIn('financial_sentiment', scores)
        self.assertIn('bullish_score', scores)
        self.assertIn('bearish_score', scores)
        self.assertGreater(scores['bullish_score'], 0)
    
    def test_analyze_financial_sentiment(self):
        """Test financial sentiment analysis."""
        result = self.analyzer.analyze_financial_sentiment(
            "The market is bullish with strong earnings growth."
        )
        
        self.assertIn('financial_sentiment_score', result)
        self.assertIn('financial_sentiment_label', result)
        self.assertIn('market_impact_score', result)
        self.assertIn('extracted_terms', result)
    
    def test_analyze_financial_batch(self):
        """Test batch financial sentiment analysis."""
        results = self.analyzer.analyze_financial_batch(self.financial_texts)
        
        self.assertEqual(len(results), len(self.financial_texts))
        for result in results:
            self.assertIn('financial_sentiment_score', result)
            self.assertIn('financial_sentiment_label', result)
    
    def test_financial_sentiment_statistics(self):
        """Test financial sentiment statistics."""
        # Analyze some texts first
        self.analyzer.analyze_financial_batch(self.financial_texts)
        
        stats = self.analyzer.get_financial_sentiment_statistics()
        
        self.assertIn('financial_sentiment_stats', stats)
        self.assertIn('market_impact_stats', stats)
        self.assertIn('financial_sentiment_distribution', stats)
    
    def test_market_sentiment_summary(self):
        """Test market sentiment summary."""
        # Analyze some texts first
        self.analyzer.analyze_financial_batch(self.financial_texts)
        
        summary = self.analyzer.get_market_sentiment_summary(time_window_hours=24)
        
        self.assertIn('overall_sentiment', summary)
        self.assertIn('avg_financial_sentiment', summary)
        self.assertIn('market_risk_level', summary)


class TestNewsProcessor(unittest.TestCase):
    """Test cases for NewsProcessor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = NewsProcessor()
        self.sample_articles = [
            {
                'title': 'AAPL Reports Strong Q3 Earnings',
                'content': 'Apple Inc. (AAPL) reported earnings of $2.50 per share, beating estimates.',
                'url': 'https://example.com/aapl-earnings',
                'published_date': datetime.now().isoformat(),
                'source': 'Financial News'
            },
            {
                'title': 'Market Volatility Increases',
                'content': 'The S&P 500 index showed high volatility with a 2% swing.',
                'url': 'https://example.com/market-volatility',
                'published_date': datetime.now().isoformat(),
                'source': 'Market Watch'
            }
        ]
    
    def test_initialization(self):
        """Test news processor initialization."""
        self.assertEqual(self.processor.model_name, "news_processor")
        self.assertTrue(self.processor.clean_text)
        self.assertTrue(self.processor.extract_entities)
    
    def test_clean_news_text(self):
        """Test news text cleaning."""
        text = "<p>This is a <b>great</b> article!</p>"
        cleaned = self.processor.clean_news_text(text)
        
        self.assertNotIn('<p>', cleaned)
        self.assertNotIn('<b>', cleaned)
        self.assertNotIn('</b>', cleaned)
        self.assertNotIn('</p>', cleaned)
    
    def test_extract_tickers(self):
        """Test ticker extraction."""
        text = "AAPL and MSFT stocks are performing well."
        tickers = self.processor.extract_tickers(text)
        
        self.assertIn('AAPL', tickers)
        self.assertIn('MSFT', tickers)
    
    def test_extract_prices(self):
        """Test price extraction."""
        text = "The stock price is $150.50 and increased by 5%."
        prices = self.processor.extract_prices(text)
        
        self.assertIn('$150.50', prices)
    
    def test_extract_percentages(self):
        """Test percentage extraction."""
        text = "The stock increased by 5% and decreased by 2.5%."
        percentages = self.processor.extract_percentages(text)
        
        self.assertIn('5%', percentages)
        self.assertIn('2.5%', percentages)
    
    def test_extract_financial_entities(self):
        """Test financial entity extraction."""
        text = "AAPL stock is trading at $150.50, up 5% today."
        entities = self.processor.extract_financial_entities(text)
        
        self.assertIn('tickers', entities)
        self.assertIn('prices', entities)
        self.assertIn('percentages', entities)
        self.assertIn('AAPL', entities['tickers'])
        self.assertIn('$150.50', entities['prices'])
        self.assertIn('5%', entities['percentages'])
    
    def test_categorize_news(self):
        """Test news categorization."""
        text = "The company reported strong earnings and revenue growth."
        categories = self.processor.categorize_news(text)
        
        self.assertIn('earnings', categories)
        self.assertGreater(categories['earnings'], 0)
    
    def test_calculate_news_importance(self):
        """Test news importance calculation."""
        text = "AAPL reported strong earnings with $2.50 EPS, beating estimates."
        entities = self.processor.extract_financial_entities(text)
        importance = self.processor.calculate_news_importance(text, entities)
        
        self.assertGreaterEqual(importance, 0.0)
        self.assertLessEqual(importance, 1.0)
    
    def test_process_news_article(self):
        """Test news article processing."""
        article = self.sample_articles[0]
        processed = self.processor.process_news_article(article)
        
        self.assertIn('title', processed)
        self.assertIn('content', processed)
        self.assertIn('entities', processed)
        self.assertIn('categories', processed)
        self.assertIn('importance', processed)
        self.assertIn('article_hash', processed)
    
    def test_process_news_batch(self):
        """Test batch news processing."""
        processed_articles = self.processor.process_news_batch(self.sample_articles)
        
        self.assertEqual(len(processed_articles), len(self.sample_articles))
        for article in processed_articles:
            self.assertIn('entities', article)
            self.assertIn('categories', article)
            self.assertIn('importance', article)
    
    def test_deduplicate_news(self):
        """Test news deduplication."""
        # Create duplicate articles
        duplicate_articles = self.sample_articles + self.sample_articles
        processed = self.processor.process_news_batch(duplicate_articles)
        deduplicated = self.processor.deduplicate_news(processed)
        
        self.assertEqual(len(deduplicated), len(self.sample_articles))
    
    def test_news_statistics(self):
        """Test news processing statistics."""
        # Process some articles first
        self.processor.process_news_batch(self.sample_articles)
        
        stats = self.processor.get_news_statistics()
        
        self.assertIn('total_articles', stats)
        self.assertIn('importance_stats', stats)
        self.assertIn('category_distribution', stats)


class TestNewsAggregator(unittest.TestCase):
    """Test cases for NewsAggregator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.aggregator = NewsAggregator()
    
    def test_initialization(self):
        """Test news aggregator initialization."""
        self.assertEqual(self.aggregator.model_name, "news_aggregator")
        self.assertEqual(self.aggregator.update_interval_minutes, 15)
        self.assertIsInstance(self.aggregator.news_processor, NewsProcessor)
    
    def test_add_rss_feed(self):
        """Test adding RSS feed."""
        self.aggregator.add_rss_feed("https://example.com/feed.xml", "Test Feed")
        
        self.assertEqual(len(self.aggregator.news_sources['rss_feeds']), 1)
        self.assertEqual(self.aggregator.news_sources['rss_feeds'][0]['name'], "Test Feed")
    
    def test_add_api_source(self):
        """Test adding API source."""
        api_config = {
            'name': 'Test API',
            'url': 'https://api.example.com/news',
            'api_key': 'test_key'
        }
        self.aggregator.add_api_source(api_config)
        
        self.assertEqual(len(self.aggregator.news_sources['api_sources']), 1)
        self.assertEqual(self.aggregator.news_sources['api_sources'][0]['name'], "Test API")
    
    @patch('src.ai.natural_language_processing.news_processor.feedparser')
    def test_fetch_rss_news(self, mock_feedparser):
        """Test RSS news fetching."""
        # Mock feedparser
        mock_feed = Mock()
        mock_feed.entries = [
            Mock(title="Test News", summary="Test content", link="https://example.com", published="2023-01-01")
        ]
        mock_feedparser.parse.return_value = mock_feed
        
        # Add RSS feed
        self.aggregator.add_rss_feed("https://example.com/feed.xml", "Test Feed")
        
        # Fetch news
        articles = self.aggregator.fetch_rss_news()
        
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0]['title'], "Test News")
    
    @patch('src.ai.natural_language_processing.news_processor.requests')
    def test_fetch_api_news(self, mock_requests):
        """Test API news fetching."""
        # Mock requests
        mock_response = Mock()
        mock_response.json.return_value = {
            'articles': [
                {
                    'title': 'Test News',
                    'description': 'Test content',
                    'url': 'https://example.com',
                    'publishedAt': '2023-01-01'
                }
            ]
        }
        mock_requests.get.return_value = mock_response
        
        # Add API source
        api_config = {
            'name': 'Test API',
            'url': 'https://api.example.com/news',
            'api_key': 'test_key'
        }
        self.aggregator.add_api_source(api_config)
        
        # Fetch news
        articles = self.aggregator.fetch_api_news()
        
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0]['title'], "Test News")
    
    def test_get_news_statistics(self):
        """Test news aggregation statistics."""
        stats = self.aggregator.get_news_statistics()
        
        self.assertIn('aggregator_name', stats)
        self.assertIn('total_sources', stats)
        self.assertIn('rss_feeds', stats)
        self.assertIn('api_sources', stats)


class TestTextClassifier(unittest.TestCase):
    """Test cases for TextClassifier."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.classifier = TextClassifier()
        self.sample_texts = [
            "This is a positive review.",
            "This is a negative review.",
            "This is a neutral review."
        ]
        self.sample_labels = ["positive", "negative", "neutral"]
    
    def test_initialization(self):
        """Test text classifier initialization."""
        self.assertEqual(self.classifier.model_name, "text_classifier")
        self.assertEqual(self.classifier.classifier_type, "logistic_regression")
        self.assertEqual(self.classifier.vectorizer_type, "tfidf")
    
    def test_preprocess_text(self):
        """Test text preprocessing."""
        text = "This is a GREAT product! Visit https://example.com"
        processed = self.classifier.preprocess_text(text)
        
        self.assertIn("great", processed)
        self.assertNotIn("https://example.com", processed)
        self.assertNotIn("!", processed)
    
    def test_predict(self):
        """Test text prediction."""
        # Train the classifier first
        self.classifier.train(self.sample_texts, self.sample_labels)
        
        result = self.classifier.predict("This is a positive review.")
        
        self.assertIn('predicted_class', result)
        self.assertIn('confidence', result)
        self.assertIn('class_probabilities', result)
    
    def test_predict_batch(self):
        """Test batch prediction."""
        # Train the classifier first
        self.classifier.train(self.sample_texts, self.sample_labels)
        
        results = self.classifier.predict_batch(self.sample_texts)
        
        self.assertEqual(len(results), len(self.sample_texts))
        for result in results:
            self.assertIn('predicted_class', result)
            self.assertIn('confidence', result)
    
    def test_evaluate(self):
        """Test model evaluation."""
        # Train the classifier first
        self.classifier.train(self.sample_texts, self.sample_labels)
        
        # Evaluate on the same data
        results = self.classifier.evaluate(self.sample_texts, self.sample_labels)
        
        self.assertIn('accuracy', results)
        self.assertIn('classification_report', results)
        self.assertIn('confusion_matrix', results)
    
    def test_get_feature_importance(self):
        """Test feature importance extraction."""
        # Train the classifier first
        self.classifier.train(self.sample_texts, self.sample_labels)
        
        importance = self.classifier.get_feature_importance()
        
        # Should return empty dict for logistic regression
        self.assertIsInstance(importance, dict)
    
    def test_classification_statistics(self):
        """Test classification statistics."""
        # Train and predict first
        self.classifier.train(self.sample_texts, self.sample_labels)
        self.classifier.predict_batch(self.sample_texts)
        
        stats = self.classifier.get_classification_statistics()
        
        self.assertIn('total_classifications', stats)
        self.assertIn('class_distribution', stats)
        self.assertIn('confidence_stats', stats)


class TestNewsClassifier(unittest.TestCase):
    """Test cases for NewsClassifier."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.classifier = NewsClassifier()
        self.sample_news = [
            "Apple reported strong quarterly earnings.",
            "Microsoft announced a major acquisition.",
            "Tesla stock shows high volatility."
        ]
        self.sample_labels = ["earnings", "mergers", "market"]
    
    def test_initialization(self):
        """Test news classifier initialization."""
        self.assertEqual(self.classifier.model_name, "news_classifier")
        self.assertIn('earnings', self.classifier.financial_categories)
        self.assertIn('mergers', self.classifier.financial_categories)
    
    def test_preprocess_news_text(self):
        """Test news text preprocessing."""
        text = "The company said it reported strong earnings."
        processed = self.classifier.preprocess_news_text(text)
        
        self.assertNotIn("said", processed)
        self.assertNotIn("company", processed)
    
    def test_classify_news(self):
        """Test news classification."""
        # Train the classifier first
        self.classifier.train(self.sample_news, self.sample_labels)
        
        result = self.classifier.classify_news("Apple reported strong quarterly earnings.")
        
        self.assertIn('news_category', result)
        self.assertIn('news_confidence', result)
        self.assertIn('category_scores', result)
        self.assertIn('prediction_consistency', result)
    
    def test_classify_news_batch(self):
        """Test batch news classification."""
        # Train the classifier first
        self.classifier.train(self.sample_news, self.sample_labels)
        
        results = self.classifier.classify_news_batch(self.sample_news)
        
        self.assertEqual(len(results), len(self.sample_news))
        for result in results:
            self.assertIn('news_category', result)
            self.assertIn('category_scores', result)
    
    def test_get_category_keywords(self):
        """Test getting category keywords."""
        keywords = self.classifier.get_category_keywords('earnings')
        
        self.assertIn('earnings', keywords)
        self.assertIn('revenue', keywords)
        self.assertIn('profit', keywords)
    
    def test_add_category_keywords(self):
        """Test adding category keywords."""
        initial_count = len(self.classifier.get_category_keywords('earnings'))
        
        self.classifier.add_category_keywords('earnings', ['new_keyword'])
        
        new_count = len(self.classifier.get_category_keywords('earnings'))
        self.assertEqual(new_count, initial_count + 1)
    
    def test_get_all_categories(self):
        """Test getting all categories."""
        categories = self.classifier.get_all_categories()
        
        self.assertIn('earnings', categories)
        self.assertIn('mergers', categories)
        self.assertIn('market', categories)
    
    def test_get_category_statistics(self):
        """Test getting category statistics."""
        stats = self.classifier.get_category_statistics()
        
        self.assertIn('total_categories', stats)
        self.assertIn('categories', stats)
        self.assertIn('category_keyword_counts', stats)


class TestNaturalLanguageProcessingManager(unittest.TestCase):
    """Test cases for NaturalLanguageProcessingManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.manager = NaturalLanguageProcessingManager(enable_bert=False, enable_financial_analysis=True)
        self.sample_text = "The market is bullish with strong earnings growth."
        self.sample_articles = [
            {
                'title': 'AAPL Reports Strong Earnings',
                'content': 'Apple Inc. reported strong quarterly earnings.',
                'url': 'https://example.com',
                'published_date': datetime.now().isoformat(),
                'source': 'Financial News'
            }
        ]
    
    def test_initialization(self):
        """Test NLP manager initialization."""
        self.assertEqual(self.manager.model_name, "nlp_manager")
        self.assertFalse(self.manager.enable_bert)
        self.assertTrue(self.manager.enable_financial_analysis)
        self.assertTrue(self.manager.cache_results)
    
    def test_analyze_sentiment(self):
        """Test sentiment analysis."""
        result = self.manager.analyze_sentiment(self.sample_text, use_financial=True)
        
        self.assertIn('basic_sentiment', result)
        self.assertIn('financial_sentiment', result)
        self.assertIn('combined_sentiment', result)
        self.assertIn('methods_used', result)
        self.assertIn('processing_time', result)
    
    def test_classify_news(self):
        """Test news classification."""
        result = self.manager.classify_news(self.sample_text)
        
        self.assertIn('basic_classification', result)
        self.assertIn('combined_classification', result)
        self.assertIn('methods_used', result)
        self.assertIn('processing_time', result)
    
    def test_analyze_news_batch(self):
        """Test batch news analysis."""
        results = self.manager.analyze_news_batch(self.sample_articles, use_financial=True)
        
        self.assertEqual(len(results), len(self.sample_articles))
        for result in results:
            self.assertIn('article', result)
            self.assertIn('sentiment_analysis', result)
            self.assertIn('news_classification', result)
    
    def test_get_market_sentiment_summary(self):
        """Test market sentiment summary."""
        # Analyze some text first
        self.manager.analyze_sentiment(self.sample_text, use_financial=True)
        
        summary = self.manager.get_market_sentiment_summary(use_financial=True)
        
        self.assertIn('overall_sentiment', summary)
        self.assertIn('avg_financial_sentiment', summary)
        self.assertIn('market_risk_level', summary)
    
    def test_get_news_by_category(self):
        """Test getting news by category."""
        # This will return empty list since no news is aggregated
        news = self.manager.get_news_by_category('earnings')
        
        self.assertIsInstance(news, list)
    
    def test_get_news_by_ticker(self):
        """Test getting news by ticker."""
        # This will return empty list since no news is aggregated
        news = self.manager.get_news_by_ticker('AAPL')
        
        self.assertIsInstance(news, list)
    
    def test_aggregate_news(self):
        """Test news aggregation."""
        # This will return empty list since no sources are configured
        news = self.manager.aggregate_news()
        
        self.assertIsInstance(news, list)
    
    def test_get_performance_metrics(self):
        """Test performance metrics."""
        # Analyze some text first
        self.manager.analyze_sentiment(self.sample_text)
        
        metrics = self.manager.get_performance_metrics()
        
        self.assertIn('total_analyses', metrics)
        self.assertIn('average_processing_time', metrics)
        self.assertIn('bert_enabled', metrics)
        self.assertIn('financial_analysis_enabled', metrics)
    
    def test_get_analysis_statistics(self):
        """Test analysis statistics."""
        # Analyze some text first
        self.manager.analyze_sentiment(self.sample_text)
        
        stats = self.manager.get_analysis_statistics()
        
        self.assertIn('total_analyses', stats)
        self.assertIn('sentiment_distribution', stats)
        self.assertIn('processing_time_stats', stats)
    
    def test_clear_cache(self):
        """Test cache clearing."""
        # Analyze some text to populate cache
        self.manager.analyze_sentiment(self.sample_text)
        
        # Clear cache
        self.manager.clear_cache()
        
        self.assertEqual(len(self.manager.results_cache), 0)
    
    def test_clear_history(self):
        """Test history clearing."""
        # Analyze some text to populate history
        self.manager.analyze_sentiment(self.sample_text)
        
        # Clear history
        self.manager.clear_history()
        
        self.assertEqual(len(self.manager.analysis_history), 0)
    
    def test_get_model_status(self):
        """Test model status."""
        status = self.manager.get_model_status()
        
        self.assertIn('sentiment_analyzer', status)
        self.assertIn('news_processor', status)
        self.assertIn('financial_sentiment_analyzer', status)
        self.assertIn('bert_enabled', status)
        self.assertIn('financial_analysis_enabled', status)


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestSentimentAnalyzer,
        TestFinancialSentimentAnalyzer,
        TestNewsProcessor,
        TestNewsAggregator,
        TestTextClassifier,
        TestNewsClassifier,
        TestNaturalLanguageProcessingManager
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*50}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")

