"""
Unit tests for Free APIs Integration
====================================

Tests for the refactored FreeAPIsIntegration that uses individual
API clients inheriting from BaseAPIClient.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from src.data_services.free_apis_integration import (
    NewsAPIClient,
    FinnhubAPIClient,
    RedditAPIClient,
    AlphaVantageTechnicalClient,
    FreeAPIsIntegration,
    create_free_apis_config,
)
from src.utils.api_client import APIResponse


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def clear_env_vars(monkeypatch):
    """Clear API key environment variables for demo mode testing."""
    monkeypatch.delenv("NEWSAPI_KEY", raising=False)
    monkeypatch.delenv("FINNHUB_KEY", raising=False)
    monkeypatch.delenv("ALPHAVANTAGE_KEY", raising=False)


@pytest.fixture
def default_config():
    """Create default configuration."""
    return create_free_apis_config()


@pytest.fixture
def newsapi_client(clear_env_vars):
    """Create NewsAPI client in demo mode."""
    return NewsAPIClient("demo")


@pytest.fixture
def finnhub_client(clear_env_vars):
    """Create Finnhub client in demo mode."""
    return FinnhubAPIClient("demo")


@pytest.fixture
def reddit_client(clear_env_vars):
    """Create Reddit client in demo mode."""
    return RedditAPIClient()


@pytest.fixture
def alpha_vantage_technical_client(clear_env_vars):
    """Create Alpha Vantage Technical client in demo mode."""
    return AlphaVantageTechnicalClient("demo")


@pytest.fixture
def integration(default_config, clear_env_vars):
    """Create FreeAPIsIntegration."""
    return FreeAPIsIntegration(default_config)


# =============================================================================
# NewsAPIClient Tests
# =============================================================================

class TestNewsAPIClient:
    """Tests for NewsAPIClient."""

    def test_init_demo_mode(self, newsapi_client):
        """Test client initializes in demo mode."""
        assert newsapi_client.demo_mode is True
        assert newsapi_client.api_key == "demo"
        assert newsapi_client.base_url == "https://newsapi.org/v2"

    def test_init_with_api_key(self, clear_env_vars):
        """Test client initializes with API key."""
        client = NewsAPIClient("real_api_key")
        assert client.demo_mode is False
        assert client.api_key == "real_api_key"

    def test_get_news_sentiment_demo_mode(self, newsapi_client):
        """Test news sentiment in demo mode."""
        symbols = ["AAPL", "MSFT"]
        result = newsapi_client.get_news_sentiment(symbols)

        assert "symbols" in result
        assert "overall_sentiment" in result
        assert "total_articles" in result
        assert "demo_mode" in result
        assert result["demo_mode"] is True
        assert "AAPL" in result["symbols"]
        assert "MSFT" in result["symbols"]

    def test_get_news_sentiment_with_api(self, clear_env_vars):
        """Test news sentiment with API response."""
        client = NewsAPIClient("real_key")

        mock_response = APIResponse(
            success=True,
            data={
                "articles": [
                    {"title": "Stock gains momentum", "description": "Profit increase"},
                    {"title": "Decline in market", "description": "Loss reported"},
                ]
            },
            status_code=200,
        )

        with patch.object(client, "_get", return_value=mock_response):
            result = client.get_news_sentiment(["AAPL"])

            assert "symbols" in result
            assert "AAPL" in result["symbols"]
            assert result["symbols"]["AAPL"]["article_count"] == 2

    def test_analyze_sentiment(self, newsapi_client):
        """Test sentiment analysis."""
        articles = [
            {"title": "Company reports profit growth", "description": "Strong performance"},
            {"title": "Stock decline expected", "description": "Weak outlook"},
            {"title": "Neutral news", "description": "Regular update"},
        ]

        sentiment = newsapi_client._analyze_sentiment(articles)
        assert isinstance(sentiment, float)
        assert -1 <= sentiment <= 1

    def test_health_check_demo_mode(self, newsapi_client):
        """Test health check in demo mode."""
        assert newsapi_client.health_check() is True


# =============================================================================
# FinnhubAPIClient Tests
# =============================================================================

class TestFinnhubAPIClient:
    """Tests for FinnhubAPIClient."""

    def test_init_demo_mode(self, finnhub_client):
        """Test client initializes in demo mode."""
        assert finnhub_client.demo_mode is True
        assert finnhub_client.api_key == "demo"
        assert finnhub_client.base_url == "https://finnhub.io/api/v1"

    def test_init_with_api_key(self, clear_env_vars):
        """Test client initializes with API key."""
        client = FinnhubAPIClient("real_api_key")
        assert client.demo_mode is False
        assert client.api_key == "real_api_key"

    def test_get_company_news_demo_mode(self, finnhub_client):
        """Test company news in demo mode."""
        result = finnhub_client.get_company_news("AAPL")

        assert "symbol" in result
        assert result["symbol"] == "AAPL"
        assert "news_count" in result
        assert "avg_sentiment" in result
        assert "news_items" in result
        assert "demo_mode" in result
        assert result["demo_mode"] is True

    def test_get_company_news_with_api(self, clear_env_vars):
        """Test company news with API response."""
        client = FinnhubAPIClient("real_key")

        mock_response = APIResponse(
            success=True,
            data=[
                {"headline": "Profit surge", "summary": "Growth expected"},
                {"headline": "Market decline", "summary": "Concerns raised"},
            ],
            status_code=200,
        )

        with patch.object(client, "_get", return_value=mock_response):
            result = client.get_company_news("AAPL")

            assert result["symbol"] == "AAPL"
            assert result["news_count"] == 2
            assert result["demo_mode"] is False

    def test_calculate_sentiment(self, finnhub_client):
        """Test sentiment calculation."""
        headline = "company reports strong profit growth"
        summary = "beat expectations with surge"

        sentiment = finnhub_client._calculate_sentiment(headline, summary)

        assert isinstance(sentiment, float)
        assert sentiment > 0  # Positive keywords

    def test_calculate_sentiment_negative(self, finnhub_client):
        """Test negative sentiment calculation."""
        headline = "company faces loss and decline"
        summary = "concerns about risk"

        sentiment = finnhub_client._calculate_sentiment(headline, summary)

        assert sentiment < 0  # Negative keywords

    def test_get_quote_demo_mode(self, finnhub_client):
        """Test quote in demo mode."""
        result = finnhub_client.get_quote("AAPL")
        assert result is None

    def test_health_check_demo_mode(self, finnhub_client):
        """Test health check in demo mode."""
        assert finnhub_client.health_check() is True


# =============================================================================
# RedditAPIClient Tests
# =============================================================================

class TestRedditAPIClient:
    """Tests for RedditAPIClient."""

    def test_init_demo_mode(self, reddit_client):
        """Test client initializes in demo mode."""
        assert reddit_client.demo_mode is True
        assert reddit_client.client_id == "demo"
        assert reddit_client.base_url == "https://oauth.reddit.com"

    def test_init_with_credentials(self, clear_env_vars):
        """Test client initializes with credentials."""
        client = RedditAPIClient(
            client_id="real_id",
            client_secret="real_secret",
            user_agent="Test/1.0",
        )
        assert client.demo_mode is False
        assert client.client_id == "real_id"

    def test_get_reddit_sentiment_demo_mode(self, reddit_client):
        """Test Reddit sentiment in demo mode."""
        symbols = ["AAPL", "MSFT"]
        result = reddit_client.get_reddit_sentiment(symbols)

        assert "symbols" in result
        assert "overall_sentiment" in result
        assert "total_mentions" in result
        assert "demo_mode" in result
        assert result["demo_mode"] is True
        assert "AAPL" in result["symbols"]
        assert "MSFT" in result["symbols"]

    def test_health_check(self, reddit_client):
        """Test health check."""
        assert reddit_client.health_check() is True

    def test_prepare_headers(self, reddit_client):
        """Test header preparation."""
        headers = reddit_client._prepare_headers()

        assert "User-Agent" in headers
        assert headers["User-Agent"] == "TradingBot/1.0"


# =============================================================================
# AlphaVantageTechnicalClient Tests
# =============================================================================

class TestAlphaVantageTechnicalClient:
    """Tests for AlphaVantageTechnicalClient."""

    def test_init_demo_mode(self, alpha_vantage_technical_client):
        """Test client initializes in demo mode."""
        assert alpha_vantage_technical_client.demo_mode is True
        assert alpha_vantage_technical_client.api_key == "demo"
        assert alpha_vantage_technical_client.base_url == "https://www.alphavantage.co"

    def test_init_with_api_key(self, clear_env_vars):
        """Test client initializes with API key."""
        client = AlphaVantageTechnicalClient("real_api_key")
        assert client.demo_mode is False
        assert client.api_key == "real_api_key"

    def test_get_technical_indicators_demo_mode(self, alpha_vantage_technical_client):
        """Test technical indicators in demo mode."""
        result = alpha_vantage_technical_client.get_technical_indicators("AAPL")

        assert "sma_20" in result
        assert "rsi_14" in result
        assert "macd" in result
        assert "macd_signal" in result
        assert "macd_hist" in result
        assert "demo_mode" in result
        assert result["demo_mode"] is True

    def test_get_technical_indicators_with_api(self, clear_env_vars):
        """Test technical indicators with API responses."""
        client = AlphaVantageTechnicalClient("real_key")

        sma_response = APIResponse(
            success=True,
            data={"Technical Analysis: SMA": {"2024-01-15": {"SMA": "150.5"}}},
            status_code=200,
        )
        rsi_response = APIResponse(
            success=True,
            data={"Technical Analysis: RSI": {"2024-01-15": {"RSI": "55.2"}}},
            status_code=200,
        )
        macd_response = APIResponse(
            success=True,
            data={
                "Technical Analysis: MACD": {
                    "2024-01-15": {"MACD": "1.5", "MACD_Signal": "1.2", "MACD_Hist": "0.3"}
                }
            },
            status_code=200,
        )

        with patch.object(client, "_get") as mock_get:
            mock_get.side_effect = [sma_response, rsi_response, macd_response]
            result = client.get_technical_indicators("AAPL")

            assert result["sma_20"] == 150.5
            assert result["rsi_14"] == 55.2
            assert result["macd"] == 1.5

    def test_health_check_demo_mode(self, alpha_vantage_technical_client):
        """Test health check in demo mode."""
        assert alpha_vantage_technical_client.health_check() is True


# =============================================================================
# FreeAPIsIntegration Tests
# =============================================================================

class TestFreeAPIsIntegration:
    """Tests for FreeAPIsIntegration facade."""

    def test_init(self, integration):
        """Test integration initializes all clients."""
        assert integration.newsapi_client is not None
        assert integration.finnhub_client is not None
        assert integration.reddit_client is not None
        assert integration.alpha_vantage_client is not None

    def test_init_demo_mode(self, integration):
        """Test all clients in demo mode."""
        assert integration.newsapi_client.demo_mode is True
        assert integration.finnhub_client.demo_mode is True
        assert integration.reddit_client.demo_mode is True
        assert integration.alpha_vantage_client.demo_mode is True

    def test_get_news_sentiment(self, integration):
        """Test news sentiment delegation."""
        result = integration.get_news_sentiment(["AAPL"])

        assert "symbols" in result
        assert "demo_mode" in result

    def test_get_technical_indicators(self, integration):
        """Test technical indicators delegation."""
        result = integration.get_technical_indicators("AAPL")

        assert "sma_20" in result
        assert "demo_mode" in result

    def test_get_reddit_sentiment(self, integration):
        """Test Reddit sentiment delegation."""
        result = integration.get_reddit_sentiment(["AAPL"])

        assert "symbols" in result
        assert "demo_mode" in result

    def test_get_finnhub_news(self, integration):
        """Test Finnhub news delegation."""
        result = integration.get_finnhub_news("AAPL")

        assert "symbol" in result
        assert "demo_mode" in result

    def test_get_comprehensive_data(self, integration):
        """Test comprehensive data collection."""
        symbols = ["AAPL", "MSFT"]
        result = integration.get_comprehensive_data(symbols)

        assert "news_sentiment" in result
        assert "technical_indicators" in result
        assert "reddit_sentiment" in result
        assert "finnhub_news" in result
        assert "timestamp" in result
        assert "symbols" in result
        assert result["symbols"] == symbols

    def test_get_api_status(self, integration):
        """Test API status retrieval."""
        status = integration.get_api_status()

        assert "news_api" in status
        assert "alpha_vantage" in status
        assert "reddit_api" in status
        assert "finnhub" in status

        for api_status in status.values():
            assert "configured" in api_status
            assert "health_check" in api_status


# =============================================================================
# Configuration Tests
# =============================================================================

class TestConfiguration:
    """Tests for configuration functions."""

    def test_create_free_apis_config(self):
        """Test default config creation."""
        config = create_free_apis_config()

        assert "api_keys" in config
        assert "rate_limits" in config

        api_keys = config["api_keys"]
        assert api_keys["newsapi"] == "demo"
        assert api_keys["alpha_vantage"] == "demo"
        assert api_keys["finnhub"] == "demo"
        assert "reddit" in api_keys

        rate_limits = config["rate_limits"]
        assert "newsapi" in rate_limits
        assert "alpha_vantage" in rate_limits
        assert "reddit" in rate_limits
        assert "finnhub" in rate_limits


# =============================================================================
# Environment Variable Tests
# =============================================================================

class TestEnvironmentVariables:
    """Tests for environment variable handling."""

    def test_newsapi_env_var(self, monkeypatch):
        """Test NewsAPI uses environment variable."""
        monkeypatch.setenv("NEWSAPI_KEY", "env_key")
        client = NewsAPIClient("demo")

        assert client.api_key == "env_key"
        assert client.demo_mode is False

    def test_finnhub_env_var(self, monkeypatch):
        """Test Finnhub uses environment variable."""
        monkeypatch.setenv("FINNHUB_KEY", "env_finnhub_key")
        client = FinnhubAPIClient("demo")

        assert client.api_key == "env_finnhub_key"
        assert client.demo_mode is False

    def test_alpha_vantage_env_var(self, monkeypatch):
        """Test Alpha Vantage uses environment variable."""
        monkeypatch.setenv("ALPHAVANTAGE_KEY", "env_av_key")
        client = AlphaVantageTechnicalClient("demo")

        assert client.api_key == "env_av_key"
        assert client.demo_mode is False


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error handling."""

    def test_newsapi_api_error_fallback(self, clear_env_vars):
        """Test NewsAPI falls back to demo on error."""
        client = NewsAPIClient("real_key")

        mock_response = APIResponse(
            success=False,
            error="API Error",
            status_code=500,
        )

        with patch.object(client, "_get", return_value=mock_response):
            result = client.get_news_sentiment(["AAPL"])

            # Should still return data (default sentiment)
            assert "symbols" in result
            assert "AAPL" in result["symbols"]

    def test_finnhub_api_error_fallback(self, clear_env_vars):
        """Test Finnhub falls back to demo on error."""
        client = FinnhubAPIClient("real_key")

        mock_response = APIResponse(
            success=False,
            error="API Error",
            status_code=500,
        )

        with patch.object(client, "_get", return_value=mock_response):
            result = client.get_company_news("AAPL")

            assert "demo_mode" in result
            assert result["demo_mode"] is True

    def test_alpha_vantage_api_error_fallback(self, clear_env_vars):
        """Test Alpha Vantage falls back to demo on error."""
        client = AlphaVantageTechnicalClient("real_key")

        mock_response = APIResponse(
            success=False,
            error="API Error",
            status_code=500,
        )

        with patch.object(client, "_get", return_value=mock_response):
            result = client.get_technical_indicators("AAPL")

            # Should still return demo data with expected fields
            assert "sma_20" in result
            assert "rsi_14" in result
