"""
Unit tests for QuestradeOptionsAPI
===================================

Tests for the refactored QuestradeOptionsAPI that inherits from QuestradeBaseClient.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
from pathlib import Path
import tempfile

from src.options.questrade_options_api import (
    QuestradeOptionsAPI,
    OptionContract,
    OptionsChain,
)
from src.utils.api_client import APIResponse


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def temp_config_file(temp_dir):
    """Create a temporary config file."""
    import yaml
    config = {
        "questrade": {
            "api": {
                "timeout": 30,
                "token_cache_path": str(Path(temp_dir) / "token_cache.json"),
            },
        }
    }
    config_path = Path(temp_dir) / "questrade_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return str(config_path)


@pytest.fixture
def authenticated_client(temp_config_file, temp_dir):
    """Create authenticated client for testing."""
    with patch("src.data_pipeline.questrade_base_client.yaml.safe_load") as mock_yaml:
        mock_yaml.return_value = {"questrade": {"api": {}}}
        client = QuestradeOptionsAPI(
            config_path=temp_config_file,
            cache_enabled=True,
        )
        # Set up valid token
        client.access_token = "valid_token"
        client.api_server = "https://api.questrade.com"
        client.base_url = "https://api.questrade.com"
        client.token_expiry = datetime.now() + timedelta(hours=1)
        return client


@pytest.fixture
def sample_options_data():
    """Create sample options chain data."""
    return {
        "underlyingPrice": 150.0,
        "optionChain": [
            {
                "symbol": "AAPL240119C00150000",
                "underlying": "AAPL",
                "strikePrice": 150.0,
                "expiryDate": "2024-01-19",
                "bidPrice": 5.50,
                "askPrice": 5.80,
                "lastTradePriceTrHrs": 5.60,
                "volume": 1000,
                "openInterest": 5000,
                "volatility": 0.25,
                "delta": 0.50,
                "gamma": 0.03,
                "theta": -0.05,
                "vega": 0.15,
            },
            {
                "symbol": "AAPL240119P00150000",
                "underlying": "AAPL",
                "strikePrice": 150.0,
                "expiryDate": "2024-01-19",
                "bidPrice": 5.40,
                "askPrice": 5.70,
                "lastTradePriceTrHrs": 5.55,
                "volume": 800,
                "openInterest": 4000,
                "volatility": 0.25,
                "delta": -0.50,
                "gamma": 0.03,
                "theta": -0.05,
                "vega": 0.15,
            },
            {
                "symbol": "AAPL240119C00160000",
                "underlying": "AAPL",
                "strikePrice": 160.0,
                "expiryDate": "2024-01-19",
                "bidPrice": 2.50,
                "askPrice": 2.80,
                "lastTradePriceTrHrs": 2.60,
                "volume": 500,
                "openInterest": 3000,
                "volatility": 0.28,
                "delta": 0.30,
                "gamma": 0.02,
                "theta": -0.03,
                "vega": 0.10,
            },
        ],
    }


# =============================================================================
# Initialization Tests
# =============================================================================

class TestQuestradeOptionsAPIInit:
    """Tests for QuestradeOptionsAPI initialization."""

    def test_init_defaults(self, temp_config_file):
        """Test client initializes with defaults."""
        with patch("src.data_pipeline.questrade_base_client.yaml.safe_load") as mock_yaml:
            mock_yaml.return_value = {"questrade": {"api": {}}}
            client = QuestradeOptionsAPI(config_path=temp_config_file)

            assert client.cache_enabled is True
            assert client.cache_ttl == 60

    def test_init_custom_cache_settings(self, temp_config_file):
        """Test client with custom cache settings."""
        with patch("src.data_pipeline.questrade_base_client.yaml.safe_load") as mock_yaml:
            mock_yaml.return_value = {"questrade": {"api": {}}}
            client = QuestradeOptionsAPI(
                config_path=temp_config_file,
                cache_enabled=False,
                cache_ttl_seconds=120,
            )

            assert client.cache_enabled is False
            assert client.cache_ttl == 120

    def test_init_legacy_config(self, temp_config_file):
        """Test client with legacy config dict."""
        legacy_config = {"token_cache_path": "/tmp/token.json"}

        with patch("src.data_pipeline.questrade_base_client.yaml.safe_load") as mock_yaml:
            mock_yaml.return_value = {"questrade": {"api": {}}}
            client = QuestradeOptionsAPI(
                config=legacy_config,
                config_path=temp_config_file,
            )

            # Should accept legacy config without error
            assert client is not None


# =============================================================================
# Options Chain Tests
# =============================================================================

class TestOptionsChain:
    """Tests for options chain retrieval."""

    def test_get_options_chain_success(self, authenticated_client, sample_options_data):
        """Test successful options chain retrieval."""
        symbol_response = APIResponse(
            success=True,
            data={"symbols": [{"symbol": "AAPL", "symbolId": 8049}]},
            status_code=200,
        )
        chain_response = APIResponse(
            success=True,
            data=sample_options_data,
            status_code=200,
        )

        with patch.object(authenticated_client, "_get") as mock_get:
            mock_get.side_effect = [symbol_response, chain_response]
            chain = authenticated_client.get_options_chain("AAPL")

            assert chain is not None
            assert chain.symbol == "AAPL"
            assert chain.underlying_price == 150.0
            assert len(chain.calls) == 2
            assert len(chain.puts) == 1

    def test_get_options_chain_uses_cache(self, authenticated_client, sample_options_data):
        """Test that options chain is cached."""
        symbol_response = APIResponse(
            success=True,
            data={"symbols": [{"symbol": "AAPL", "symbolId": 8049}]},
            status_code=200,
        )
        chain_response = APIResponse(
            success=True,
            data=sample_options_data,
            status_code=200,
        )

        with patch.object(authenticated_client, "_get") as mock_get:
            mock_get.side_effect = [symbol_response, chain_response]

            # First call should hit API
            chain1 = authenticated_client.get_options_chain("AAPL")
            # Second call should use cache
            chain2 = authenticated_client.get_options_chain("AAPL")

            assert chain1 is not None
            assert chain2 is not None
            # Only 2 calls to _get (symbol search + chain) not 4
            assert mock_get.call_count == 2

    def test_get_options_chain_auth_failure(self, temp_config_file):
        """Test options chain when authentication fails."""
        with patch("src.data_pipeline.questrade_base_client.yaml.safe_load") as mock_yaml:
            mock_yaml.return_value = {"questrade": {"api": {}}}
            client = QuestradeOptionsAPI(config_path=temp_config_file)
            # No token set

            with patch.object(client, "authenticate", return_value=False):
                chain = client.get_options_chain("AAPL")

                assert chain is None


# =============================================================================
# Option Quote Tests
# =============================================================================

class TestOptionQuote:
    """Tests for option quote retrieval."""

    def test_get_option_quote_success(self, authenticated_client):
        """Test successful option quote retrieval."""
        symbol_response = APIResponse(
            success=True,
            data={"symbols": [{"symbol": "AAPL240119C00150000", "symbolId": 12345}]},
            status_code=200,
        )
        quote_response = APIResponse(
            success=True,
            data={
                "quotes": [
                    {
                        "symbol": "AAPL240119C00150000",
                        "bidPrice": 5.50,
                        "askPrice": 5.80,
                        "strikePrice": 150.0,
                        "expiryDate": "2024-01-19",
                    }
                ]
            },
            status_code=200,
        )

        with patch.object(authenticated_client, "_get") as mock_get:
            mock_get.side_effect = [symbol_response, quote_response]
            quote = authenticated_client.get_option_quote("AAPL240119C00150000")

            assert quote is not None
            assert quote.bid == 5.50
            assert quote.ask == 5.80

    def test_get_option_quote_not_found(self, authenticated_client):
        """Test option quote when symbol not found."""
        symbol_response = APIResponse(
            success=True,
            data={"symbols": []},
            status_code=200,
        )

        with patch.object(authenticated_client, "_get", return_value=symbol_response):
            quote = authenticated_client.get_option_quote("INVALID")

            assert quote is None


# =============================================================================
# Options Search Tests
# =============================================================================

class TestOptionsSearch:
    """Tests for options search functionality."""

    def test_search_options_by_type(self, authenticated_client, sample_options_data):
        """Test searching options by type."""
        symbol_response = APIResponse(
            success=True,
            data={"symbols": [{"symbol": "AAPL", "symbolId": 8049}]},
            status_code=200,
        )
        chain_response = APIResponse(
            success=True,
            data=sample_options_data,
            status_code=200,
        )

        with patch.object(authenticated_client, "_get") as mock_get:
            mock_get.side_effect = [symbol_response, chain_response]
            calls = authenticated_client.search_options("AAPL", "CALL")

            assert len(calls) == 2
            for option in calls:
                assert option.option_type == "CALL"

    def test_search_options_by_strike(self, authenticated_client, sample_options_data):
        """Test searching options by strike range."""
        symbol_response = APIResponse(
            success=True,
            data={"symbols": [{"symbol": "AAPL", "symbolId": 8049}]},
            status_code=200,
        )
        chain_response = APIResponse(
            success=True,
            data=sample_options_data,
            status_code=200,
        )

        with patch.object(authenticated_client, "_get") as mock_get:
            mock_get.side_effect = [symbol_response, chain_response]
            options = authenticated_client.search_options(
                "AAPL",
                "CALL",
                min_strike=155.0,
                max_strike=165.0,
            )

            assert len(options) == 1
            assert options[0].strike == 160.0


# =============================================================================
# ATM Options Tests
# =============================================================================

class TestATMOptions:
    """Tests for ATM options retrieval."""

    def test_get_atm_options(self, authenticated_client, sample_options_data):
        """Test getting ATM options."""
        symbol_response = APIResponse(
            success=True,
            data={"symbols": [{"symbol": "AAPL", "symbolId": 8049}]},
            status_code=200,
        )
        chain_response = APIResponse(
            success=True,
            data=sample_options_data,
            status_code=200,
        )

        with patch.object(authenticated_client, "_get") as mock_get:
            mock_get.side_effect = [symbol_response, chain_response]
            atm = authenticated_client.get_atm_options("AAPL", 150.0)

            assert "call" in atm
            assert "put" in atm
            assert atm["call"].strike == 150.0
            assert atm["put"].strike == 150.0


# =============================================================================
# Statistics and Cache Tests
# =============================================================================

class TestStatisticsAndCache:
    """Tests for statistics and cache methods."""

    def test_get_statistics(self, authenticated_client):
        """Test getting API statistics."""
        stats = authenticated_client.get_statistics()

        assert "authenticated" in stats
        assert "cache_size" in stats
        assert "cache_enabled" in stats
        assert stats["cache_enabled"] is True

    def test_clear_cache(self, authenticated_client, sample_options_data):
        """Test clearing the cache."""
        symbol_response = APIResponse(
            success=True,
            data={"symbols": [{"symbol": "AAPL", "symbolId": 8049}]},
            status_code=200,
        )
        chain_response = APIResponse(
            success=True,
            data=sample_options_data,
            status_code=200,
        )

        with patch.object(authenticated_client, "_get") as mock_get:
            mock_get.side_effect = [symbol_response, chain_response]

            # Populate cache
            authenticated_client.get_options_chain("AAPL")
            assert len(authenticated_client.options_cache) == 1

            # Clear cache
            authenticated_client.clear_cache()
            assert len(authenticated_client.options_cache) == 0


# =============================================================================
# Option Contract Parsing Tests
# =============================================================================

class TestOptionContractParsing:
    """Tests for option contract parsing."""

    def test_parse_option_contract_call(self, authenticated_client):
        """Test parsing a call option contract."""
        data = {
            "symbol": "AAPL240119C00150000",
            "underlying": "AAPL",
            "strikePrice": 150.0,
            "expiryDate": "2024-01-19",
            "bidPrice": 5.50,
            "askPrice": 5.80,
            "lastTradePriceTrHrs": 5.60,
            "volume": 1000,
            "openInterest": 5000,
            "volatility": 0.25,
            "delta": 0.50,
            "gamma": 0.03,
            "theta": -0.05,
            "vega": 0.15,
        }

        option = authenticated_client._parse_option_contract(data)

        assert option.symbol == "AAPL240119C00150000"
        assert option.option_type == "CALL"
        assert option.strike == 150.0
        assert option.bid == 5.50
        assert option.ask == 5.80
        assert option.delta == 0.50

    def test_parse_option_contract_put(self, authenticated_client):
        """Test parsing a put option contract."""
        data = {
            "symbol": "AAPL240119P00150000",
            "underlying": "AAPL",
            "strikePrice": 150.0,
            "expiryDate": "2024-01-19",
            "bidPrice": 5.40,
            "askPrice": 5.70,
        }

        option = authenticated_client._parse_option_contract(data)

        assert option.option_type == "PUT"
