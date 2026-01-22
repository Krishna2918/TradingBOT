"""
Unit tests for QuestradeClient
===============================

Tests for the refactored QuestradeClient that inherits from QuestradeBaseClient.
"""

import json
import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from pathlib import Path
import tempfile

from src.data_pipeline.questrade_client import (
    QuestradeClient,
    get_questrade_client,
    env_token_matches,
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
            "oauth": {
                "token_endpoint": "https://login.questrade.com/oauth2/token",
            },
        }
    }
    config_path = Path(temp_dir) / "questrade_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return str(config_path)


@pytest.fixture
def temp_token_cache(temp_dir):
    """Create path for temporary token cache."""
    return str(Path(temp_dir) / "token_cache.json")


@pytest.fixture
def mock_budget_manager():
    """Create mock budget manager."""
    manager = Mock()
    manager.can_make_request.return_value = True
    manager.record_request.return_value = None
    manager.record_rate_limit.return_value = None
    manager.get_usage_stats.return_value = {"requests": 10, "errors": 0}
    return manager


@pytest.fixture
def authenticated_client(temp_config_file, temp_token_cache, mock_budget_manager):
    """Create authenticated client for testing."""
    with patch("src.data_pipeline.questrade_client.get_api_budget_manager") as mock_get_mgr:
        mock_get_mgr.return_value = mock_budget_manager
        with patch("src.data_pipeline.questrade_base_client.yaml.safe_load") as mock_yaml:
            mock_yaml.return_value = {"questrade": {"api": {}}}
            client = QuestradeClient(
                config_path=temp_config_file,
                token_cache_path=temp_token_cache,
                allow_trading=True,
            )
            # Set up valid token
            client.access_token = "valid_token"
            client.api_server = "https://api.questrade.com"
            client.base_url = "https://api.questrade.com"
            client.token_expiry = datetime.now() + timedelta(hours=1)
            client.account_id = "12345678"
            client.budget_manager = mock_budget_manager
            return client


# =============================================================================
# Initialization Tests
# =============================================================================

class TestQuestradeClientInit:
    """Tests for QuestradeClient initialization."""

    def test_init_defaults(self, temp_config_file, temp_token_cache):
        """Test client initializes with defaults."""
        with patch("src.data_pipeline.questrade_client.get_api_budget_manager") as mock_mgr:
            mock_mgr.return_value = Mock()
            with patch("src.data_pipeline.questrade_base_client.yaml.safe_load") as mock_yaml:
                mock_yaml.return_value = {"questrade": {"api": {}}}
                client = QuestradeClient(
                    config_path=temp_config_file,
                    token_cache_path=temp_token_cache,
                )

                assert client.allow_trading is False
                assert client.practice_mode is True

    def test_init_allow_trading(self, temp_config_file, temp_token_cache):
        """Test client with trading enabled."""
        with patch("src.data_pipeline.questrade_client.get_api_budget_manager") as mock_mgr:
            mock_mgr.return_value = Mock()
            with patch("src.data_pipeline.questrade_base_client.yaml.safe_load") as mock_yaml:
                mock_yaml.return_value = {"questrade": {"api": {}}}
                client = QuestradeClient(
                    config_path=temp_config_file,
                    token_cache_path=temp_token_cache,
                    allow_trading=True,
                    practice_mode=False,
                )

                assert client.allow_trading is True
                assert client.practice_mode is False

    def test_init_env_override(self, temp_config_file, temp_token_cache):
        """Test environment variables override init params."""
        with patch.dict(os.environ, {"QUESTRADE_ALLOW_TRADING": "true"}):
            with patch("src.data_pipeline.questrade_client.get_api_budget_manager") as mock_mgr:
                mock_mgr.return_value = Mock()
                with patch("src.data_pipeline.questrade_base_client.yaml.safe_load") as mock_yaml:
                    mock_yaml.return_value = {"questrade": {"api": {}}}
                    client = QuestradeClient(
                        config_path=temp_config_file,
                        token_cache_path=temp_token_cache,
                        allow_trading=False,  # Should be overridden
                    )

                    assert client.allow_trading is True


# =============================================================================
# Account Management Tests
# =============================================================================

class TestAccountManagement:
    """Tests for account management methods."""

    def test_get_accounts_success(self, authenticated_client):
        """Test successful account retrieval."""
        mock_response = APIResponse(
            success=True,
            data={"accounts": [{"number": "12345678", "type": "Margin"}]},
            status_code=200,
        )
        with patch.object(authenticated_client, "_get", return_value=mock_response):
            accounts = authenticated_client.get_accounts()

            assert accounts is not None
            assert len(accounts) == 1
            assert accounts[0]["number"] == "12345678"

    def test_get_accounts_sets_account_id(self, temp_config_file, temp_token_cache, mock_budget_manager):
        """Test that get_accounts sets account_id if not set."""
        with patch("src.data_pipeline.questrade_client.get_api_budget_manager") as mock_get_mgr:
            mock_get_mgr.return_value = mock_budget_manager
            with patch("src.data_pipeline.questrade_base_client.yaml.safe_load") as mock_yaml:
                mock_yaml.return_value = {"questrade": {"api": {}}}
                client = QuestradeClient(
                    config_path=temp_config_file,
                    token_cache_path=temp_token_cache,
                )
                client.access_token = "valid"
                client.api_server = "https://api.questrade.com"
                client.base_url = "https://api.questrade.com"
                client.token_expiry = datetime.now() + timedelta(hours=1)
                client.budget_manager = mock_budget_manager

                mock_response = APIResponse(
                    success=True,
                    data={"accounts": [{"number": "99999999"}]},
                    status_code=200,
                )
                with patch.object(client, "_get", return_value=mock_response):
                    accounts = client.get_accounts()

                    assert client.account_id == "99999999"

    def test_get_account_balances_success(self, authenticated_client):
        """Test successful balance retrieval."""
        mock_response = APIResponse(
            success=True,
            data={
                "perCurrencyBalances": [
                    {"currency": "CAD", "cash": 10000.0, "totalEquity": 50000.0}
                ]
            },
            status_code=200,
        )
        with patch.object(authenticated_client, "_get", return_value=mock_response):
            balances = authenticated_client.get_account_balances()

            assert balances is not None
            assert "perCurrencyBalances" in balances

    def test_get_positions_success(self, authenticated_client):
        """Test successful positions retrieval."""
        mock_response = APIResponse(
            success=True,
            data={
                "positions": [
                    {"symbol": "AAPL", "openQuantity": 100, "currentPrice": 150.0}
                ]
            },
            status_code=200,
        )
        with patch.object(authenticated_client, "_get", return_value=mock_response):
            positions = authenticated_client.get_positions()

            assert positions is not None
            assert len(positions) == 1
            assert positions[0]["symbol"] == "AAPL"


# =============================================================================
# Quote Tests
# =============================================================================

class TestQuoteRetrieval:
    """Tests for quote retrieval methods."""

    def test_get_quote_success(self, authenticated_client):
        """Test successful single quote retrieval."""
        # Mock symbol search
        symbol_response = APIResponse(
            success=True,
            data={"symbols": [{"symbol": "AAPL", "symbolId": 8049}]},
            status_code=200,
        )
        quote_response = APIResponse(
            success=True,
            data={"quotes": [{"symbol": "AAPL", "lastTradePrice": 150.0}]},
            status_code=200,
        )

        with patch.object(authenticated_client, "_get") as mock_get:
            mock_get.side_effect = [symbol_response, quote_response]
            quote = authenticated_client.get_quote("AAPL")

            assert quote is not None
            assert quote["symbol"] == "AAPL"

    def test_get_quotes_multiple(self, authenticated_client):
        """Test multiple quotes retrieval."""
        symbol_response1 = APIResponse(
            success=True,
            data={"symbols": [{"symbol": "AAPL", "symbolId": 8049}]},
            status_code=200,
        )
        symbol_response2 = APIResponse(
            success=True,
            data={"symbols": [{"symbol": "MSFT", "symbolId": 9051}]},
            status_code=200,
        )
        quote_response = APIResponse(
            success=True,
            data={
                "quotes": [
                    {"symbol": "AAPL", "lastTradePrice": 150.0},
                    {"symbol": "MSFT", "lastTradePrice": 300.0},
                ]
            },
            status_code=200,
        )

        with patch.object(authenticated_client, "_get") as mock_get:
            mock_get.side_effect = [symbol_response1, symbol_response2, quote_response]
            quotes = authenticated_client.get_quotes(["AAPL", "MSFT"])

            assert quotes is not None
            assert len(quotes) == 2


# =============================================================================
# Order Tests
# =============================================================================

class TestOrderManagement:
    """Tests for order management methods."""

    def test_place_order_trading_disabled(self, temp_config_file, temp_token_cache, mock_budget_manager):
        """Test order rejected when trading disabled."""
        with patch("src.data_pipeline.questrade_client.get_api_budget_manager") as mock_get_mgr:
            mock_get_mgr.return_value = mock_budget_manager
            with patch("src.data_pipeline.questrade_base_client.yaml.safe_load") as mock_yaml:
                mock_yaml.return_value = {"questrade": {"api": {}}}
                client = QuestradeClient(
                    config_path=temp_config_file,
                    token_cache_path=temp_token_cache,
                    allow_trading=False,
                )

                result = client.place_order(
                    symbol="AAPL",
                    quantity=100,
                    action="Buy",
                    limit_price=150.0,
                )

                assert result is not None
                assert result["error"] == "TRADING_DISABLED"

    def test_place_order_success(self, authenticated_client):
        """Test successful order placement."""
        symbol_response = APIResponse(
            success=True,
            data={"symbols": [{"symbol": "AAPL", "symbolId": 8049}]},
            status_code=200,
        )
        order_response = APIResponse(
            success=True,
            data={"id": 12345, "status": "Submitted"},
            status_code=200,
        )

        with patch.object(authenticated_client, "_get", return_value=symbol_response):
            with patch.object(authenticated_client, "_post", return_value=order_response):
                result = authenticated_client.place_order(
                    symbol="AAPL",
                    quantity=100,
                    action="Buy",
                    limit_price=150.0,
                )

                assert result is not None
                assert result["id"] == 12345

    def test_place_order_limit_price_required(self, authenticated_client):
        """Test limit order requires limit_price."""
        symbol_response = APIResponse(
            success=True,
            data={"symbols": [{"symbol": "AAPL", "symbolId": 8049}]},
            status_code=200,
        )

        with patch.object(authenticated_client, "_get", return_value=symbol_response):
            with pytest.raises(ValueError) as exc_info:
                authenticated_client.place_order(
                    symbol="AAPL",
                    quantity=100,
                    action="Buy",
                    order_type="Limit",
                    # No limit_price
                )

            assert "limit_price is required" in str(exc_info.value)

    def test_cancel_order_success(self, authenticated_client):
        """Test successful order cancellation."""
        mock_response = APIResponse(
            success=True,
            data={},
            status_code=200,
        )

        with patch.object(authenticated_client, "_post", return_value=mock_response):
            result = authenticated_client.cancel_order(12345)

            assert result is True

    def test_get_open_orders(self, authenticated_client):
        """Test getting open orders."""
        mock_response = APIResponse(
            success=True,
            data={"orders": [{"id": 12345, "status": "Open"}]},
            status_code=200,
        )

        with patch.object(authenticated_client, "_get", return_value=mock_response):
            orders = authenticated_client.get_open_orders()

            assert orders is not None
            assert len(orders) == 1


# =============================================================================
# Portfolio Summary Tests
# =============================================================================

class TestPortfolioSummary:
    """Tests for portfolio summary."""

    def test_get_portfolio_summary(self, authenticated_client):
        """Test portfolio summary calculation."""
        balance_response = APIResponse(
            success=True,
            data={
                "perCurrencyBalances": [
                    {"currency": "CAD", "cash": 10000.0, "totalEquity": 50000.0}
                ]
            },
            status_code=200,
        )
        positions_response = APIResponse(
            success=True,
            data={
                "positions": [
                    {
                        "symbol": "AAPL",
                        "openQuantity": 100,
                        "averageEntryPrice": 140.0,
                        "currentPrice": 150.0,
                        "currentMarketValue": 15000.0,
                        "openPnl": 1000.0,
                    }
                ]
            },
            status_code=200,
        )

        with patch.object(authenticated_client, "_get") as mock_get:
            mock_get.side_effect = [balance_response, positions_response]
            summary = authenticated_client.get_portfolio_summary()

            assert summary["cash"] == 10000.0
            assert summary["total_value"] == 50000.0
            assert len(summary["positions"]) == 1
            assert summary["pnl_total"] == 1000.0


# =============================================================================
# Budget Manager Tests
# =============================================================================

class TestBudgetManager:
    """Tests for budget manager integration."""

    def test_budget_exhausted_blocks_request(self, authenticated_client):
        """Test that exhausted budget blocks requests."""
        authenticated_client.budget_manager.can_make_request.return_value = False

        result = authenticated_client._tracked_get("/v1/accounts")

        assert result is None

    def test_get_api_usage_stats(self, authenticated_client):
        """Test API usage stats retrieval."""
        stats = authenticated_client.get_api_usage_stats()

        assert stats["requests"] == 10
        authenticated_client.budget_manager.get_usage_stats.assert_called_once_with("questrade")


# =============================================================================
# Utility Tests
# =============================================================================

class TestUtilities:
    """Tests for utility functions."""

    def test_env_token_matches_true(self):
        """Test env_token_matches returns True when matching."""
        with patch.dict(os.environ, {"TEST_TOKEN": "my_token"}):
            assert env_token_matches("TEST_TOKEN", "my_token") is True

    def test_env_token_matches_false(self):
        """Test env_token_matches returns False when not matching."""
        with patch.dict(os.environ, {"TEST_TOKEN": "my_token"}):
            assert env_token_matches("TEST_TOKEN", "different_token") is False

    def test_env_token_matches_not_set(self):
        """Test env_token_matches returns False when env not set."""
        os.environ.pop("NONEXISTENT_TOKEN", None)
        assert env_token_matches("NONEXISTENT_TOKEN", "any_token") is False

    def test_resolve_symbol_id(self, authenticated_client):
        """Test symbol ID resolution."""
        mock_response = APIResponse(
            success=True,
            data={"symbols": [{"symbol": "AAPL", "symbolId": 8049}]},
            status_code=200,
        )

        with patch.object(authenticated_client, "_get", return_value=mock_response):
            symbol_id = authenticated_client._resolve_symbol_id("AAPL")

            assert symbol_id == 8049


# =============================================================================
# Singleton Tests
# =============================================================================

class TestSingleton:
    """Tests for singleton pattern."""

    def test_get_questrade_client_singleton(self):
        """Test get_questrade_client returns singleton."""
        # Reset singleton
        import src.data_pipeline.questrade_client as module
        module._questrade_client = None

        with patch("src.data_pipeline.questrade_client.QuestradeClient") as MockClient:
            mock_instance = Mock()
            MockClient.return_value = mock_instance

            client1 = get_questrade_client()
            client2 = get_questrade_client()

            assert client1 is client2
            MockClient.assert_called_once()
