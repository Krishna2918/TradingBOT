"""
Unit tests for QuestradeBaseClient Authentication
==================================================

Tests for OAuth2 token management, caching, and authentication flow.
"""

import json
import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from pathlib import Path
import tempfile

from src.data_pipeline.questrade_base_client import QuestradeBaseClient
from src.utils.api_client import APIResponse


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_config():
    """Create mock config for tests."""
    return {
        "questrade": {
            "api": {
                "timeout": 30,
                "token_cache_path": "test_token_cache.json",
            },
            "oauth": {
                "token_endpoint": "https://login.questrade.com/oauth2/token",
            },
            "rate_limits": {
                "requests_per_second": 1,
            },
        }
    }


@pytest.fixture
def mock_auth_response():
    """Create mock OAuth2 token response."""
    return {
        "access_token": "test_access_token_12345",
        "api_server": "https://api01.iq.questrade.com/",
        "expires_in": 1800,
        "refresh_token": "new_refresh_token_67890",
        "token_type": "Bearer",
    }


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def temp_config_file(temp_dir, mock_config):
    """Create a temporary config file."""
    config_path = Path(temp_dir) / "questrade_config.yaml"
    import yaml
    with open(config_path, "w") as f:
        yaml.dump(mock_config, f)
    return str(config_path)


@pytest.fixture
def temp_token_cache(temp_dir):
    """Create path for temporary token cache."""
    return str(Path(temp_dir) / "token_cache.json")


# =============================================================================
# Token Cache Tests
# =============================================================================

class TestTokenCacheManagement:
    """Tests for token cache loading and saving."""

    def test_load_token_cache_not_exists(self, temp_config_file, temp_token_cache):
        """Test loading when cache file doesn't exist."""
        with patch("src.data_pipeline.questrade_base_client.yaml.safe_load") as mock_yaml:
            mock_yaml.return_value = {"questrade": {"api": {}}}
            client = QuestradeBaseClient(
                config_path=temp_config_file,
                token_cache_path=temp_token_cache,
            )
            assert client.access_token is None
            assert client.api_server is None or client.api_server == "https://api01.iq.questrade.com"

    def test_load_token_cache_exists(self, temp_config_file, temp_token_cache):
        """Test loading when cache file exists with valid data."""
        # Create cache file
        cache_data = {
            "access_token": "cached_access_token",
            "api_server": "https://cached.api.server.com",
            "expires_at": (datetime.now() + timedelta(hours=1)).isoformat(),
            "refresh_token": "cached_refresh_token",
        }
        with open(temp_token_cache, "w") as f:
            json.dump(cache_data, f)

        with patch("src.data_pipeline.questrade_base_client.yaml.safe_load") as mock_yaml:
            mock_yaml.return_value = {"questrade": {"api": {}}}
            client = QuestradeBaseClient(
                config_path=temp_config_file,
                token_cache_path=temp_token_cache,
            )
            assert client.access_token == "cached_access_token"
            assert client.api_server == "https://cached.api.server.com"
            assert client._cached_refresh_token == "cached_refresh_token"

    def test_save_token_cache(self, temp_config_file, temp_token_cache):
        """Test saving token cache to file."""
        with patch("src.data_pipeline.questrade_base_client.yaml.safe_load") as mock_yaml:
            mock_yaml.return_value = {"questrade": {"api": {}}}
            client = QuestradeBaseClient(
                config_path=temp_config_file,
                token_cache_path=temp_token_cache,
            )

            # Set token state
            client.access_token = "new_access_token"
            client.api_server = "https://new.api.server.com"
            client.token_expiry = datetime.now() + timedelta(hours=1)

            # Save cache
            client._save_token_cache("new_refresh_token")

            # Verify file contents
            with open(temp_token_cache, "r") as f:
                saved = json.load(f)

            assert saved["access_token"] == "new_access_token"
            assert saved["api_server"] == "https://new.api.server.com"
            assert saved["refresh_token"] == "new_refresh_token"

    def test_clear_token_cache(self, temp_config_file, temp_token_cache):
        """Test clearing token cache."""
        # Create cache file
        with open(temp_token_cache, "w") as f:
            json.dump({"access_token": "test"}, f)

        with patch("src.data_pipeline.questrade_base_client.yaml.safe_load") as mock_yaml:
            mock_yaml.return_value = {"questrade": {"api": {}}}
            client = QuestradeBaseClient(
                config_path=temp_config_file,
                token_cache_path=temp_token_cache,
            )
            client.access_token = "test"
            client.api_server = "https://test.com"
            client.token_expiry = datetime.now()

            client._clear_token_cache()

            assert client.access_token is None
            assert client.api_server is None
            assert client.token_expiry is None
            assert not Path(temp_token_cache).exists()


# =============================================================================
# Authentication Tests
# =============================================================================

class TestAuthentication:
    """Tests for OAuth2 authentication flow."""

    def test_authenticate_success_from_env(
        self, temp_config_file, temp_token_cache, mock_auth_response
    ):
        """Test successful authentication using env variable."""
        with patch.dict(os.environ, {"QUESTRADE_REFRESH_TOKEN": "env_refresh_token"}):
            with patch("src.data_pipeline.questrade_base_client.yaml.safe_load") as mock_yaml:
                mock_yaml.return_value = {"questrade": {"api": {}}}
                with patch("requests.get") as mock_get:
                    mock_response = Mock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = mock_auth_response
                    mock_get.return_value = mock_response

                    client = QuestradeBaseClient(
                        config_path=temp_config_file,
                        token_cache_path=temp_token_cache,
                    )

                    result = client.authenticate()

                    assert result is True
                    assert client.access_token == "test_access_token_12345"
                    assert "api01.iq.questrade.com" in client.api_server

    def test_authenticate_success_from_cache(
        self, temp_config_file, temp_token_cache, mock_auth_response
    ):
        """Test authentication using cached refresh token."""
        # Create cache with refresh token
        cache_data = {
            "access_token": "old_token",
            "api_server": "https://old.server.com",
            "expires_at": (datetime.now() - timedelta(hours=1)).isoformat(),  # Expired
            "refresh_token": "cached_refresh_token",
        }
        with open(temp_token_cache, "w") as f:
            json.dump(cache_data, f)

        with patch("src.data_pipeline.questrade_base_client.yaml.safe_load") as mock_yaml:
            mock_yaml.return_value = {"questrade": {"api": {}}}
            with patch("requests.get") as mock_get:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = mock_auth_response
                mock_get.return_value = mock_response

                client = QuestradeBaseClient(
                    config_path=temp_config_file,
                    token_cache_path=temp_token_cache,
                )

                result = client.authenticate()

                assert result is True
                # Should have used cached_refresh_token first
                mock_get.assert_called()
                call_args = mock_get.call_args
                assert "refresh_token" in call_args.kwargs.get("params", {})

    def test_authenticate_failure_invalid_token(
        self, temp_config_file, temp_token_cache
    ):
        """Test authentication failure with invalid token."""
        with patch.dict(os.environ, {"QUESTRADE_REFRESH_TOKEN": "invalid_token"}):
            with patch("src.data_pipeline.questrade_base_client.yaml.safe_load") as mock_yaml:
                mock_yaml.return_value = {"questrade": {"api": {}}}
                with patch("requests.get") as mock_get:
                    mock_response = Mock()
                    mock_response.status_code = 401
                    mock_response.text = "Invalid refresh token"
                    mock_get.return_value = mock_response

                    client = QuestradeBaseClient(
                        config_path=temp_config_file,
                        token_cache_path=temp_token_cache,
                    )

                    result = client.authenticate()

                    assert result is False

    def test_authenticate_failure_no_token(
        self, temp_config_file, temp_token_cache
    ):
        """Test authentication failure when no token available."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove env var if exists
            os.environ.pop("QUESTRADE_REFRESH_TOKEN", None)

            with patch("src.data_pipeline.questrade_base_client.yaml.safe_load") as mock_yaml:
                mock_yaml.return_value = {"questrade": {"api": {}}}

                client = QuestradeBaseClient(
                    config_path=temp_config_file,
                    token_cache_path=temp_token_cache,
                )

                result = client.authenticate()

                assert result is False

    def test_authenticate_network_error(
        self, temp_config_file, temp_token_cache
    ):
        """Test authentication with network error."""
        with patch.dict(os.environ, {"QUESTRADE_REFRESH_TOKEN": "test_token"}):
            with patch("src.data_pipeline.questrade_base_client.yaml.safe_load") as mock_yaml:
                mock_yaml.return_value = {"questrade": {"api": {}}}
                with patch("requests.get") as mock_get:
                    import requests as req
                    mock_get.side_effect = req.RequestException("Network error")

                    client = QuestradeBaseClient(
                        config_path=temp_config_file,
                        token_cache_path=temp_token_cache,
                    )

                    result = client.authenticate()

                    assert result is False


# =============================================================================
# Token Validation Tests
# =============================================================================

class TestTokenValidation:
    """Tests for token validation logic."""

    def test_is_token_valid_true(self, temp_config_file, temp_token_cache):
        """Test token is valid when not expired."""
        with patch("src.data_pipeline.questrade_base_client.yaml.safe_load") as mock_yaml:
            mock_yaml.return_value = {"questrade": {"api": {}}}
            client = QuestradeBaseClient(
                config_path=temp_config_file,
                token_cache_path=temp_token_cache,
            )

            client.access_token = "valid_token"
            client.api_server = "https://api.server.com"
            client.token_expiry = datetime.now() + timedelta(hours=1)

            assert client._is_token_valid() is True

    def test_is_token_valid_false_expired(self, temp_config_file, temp_token_cache):
        """Test token is invalid when expired."""
        with patch("src.data_pipeline.questrade_base_client.yaml.safe_load") as mock_yaml:
            mock_yaml.return_value = {"questrade": {"api": {}}}
            client = QuestradeBaseClient(
                config_path=temp_config_file,
                token_cache_path=temp_token_cache,
            )

            client.access_token = "expired_token"
            client.api_server = "https://api.server.com"
            client.token_expiry = datetime.now() - timedelta(hours=1)

            assert client._is_token_valid() is False

    def test_is_token_valid_false_within_buffer(self, temp_config_file, temp_token_cache):
        """Test token is invalid within 5 minute buffer."""
        with patch("src.data_pipeline.questrade_base_client.yaml.safe_load") as mock_yaml:
            mock_yaml.return_value = {"questrade": {"api": {}}}
            client = QuestradeBaseClient(
                config_path=temp_config_file,
                token_cache_path=temp_token_cache,
            )

            client.access_token = "expiring_token"
            client.api_server = "https://api.server.com"
            client.token_expiry = datetime.now() + timedelta(minutes=3)  # Within 5 min buffer

            assert client._is_token_valid() is False

    def test_is_token_valid_false_no_token(self, temp_config_file, temp_token_cache):
        """Test token is invalid when no token."""
        with patch("src.data_pipeline.questrade_base_client.yaml.safe_load") as mock_yaml:
            mock_yaml.return_value = {"questrade": {"api": {}}}
            client = QuestradeBaseClient(
                config_path=temp_config_file,
                token_cache_path=temp_token_cache,
            )

            assert client._is_token_valid() is False


# =============================================================================
# Header Preparation Tests
# =============================================================================

class TestHeaderPreparation:
    """Tests for header preparation with auth."""

    def test_prepare_headers_with_token(self, temp_config_file, temp_token_cache):
        """Test headers include bearer token when authenticated."""
        with patch("src.data_pipeline.questrade_base_client.yaml.safe_load") as mock_yaml:
            mock_yaml.return_value = {"questrade": {"api": {}}}
            client = QuestradeBaseClient(
                config_path=temp_config_file,
                token_cache_path=temp_token_cache,
            )

            client.access_token = "test_token"

            headers = client._prepare_headers()

            assert "Authorization" in headers
            assert headers["Authorization"] == "Bearer test_token"

    def test_prepare_headers_without_token(self, temp_config_file, temp_token_cache):
        """Test headers without token when not authenticated."""
        with patch("src.data_pipeline.questrade_base_client.yaml.safe_load") as mock_yaml:
            mock_yaml.return_value = {"questrade": {"api": {}}}
            client = QuestradeBaseClient(
                config_path=temp_config_file,
                token_cache_path=temp_token_cache,
            )

            headers = client._prepare_headers()

            assert "Authorization" not in headers or headers.get("Authorization") is None

    def test_prepare_headers_with_custom_headers(self, temp_config_file, temp_token_cache):
        """Test custom headers are merged with auth."""
        with patch("src.data_pipeline.questrade_base_client.yaml.safe_load") as mock_yaml:
            mock_yaml.return_value = {"questrade": {"api": {}}}
            client = QuestradeBaseClient(
                config_path=temp_config_file,
                token_cache_path=temp_token_cache,
            )

            client.access_token = "test_token"
            custom = {"X-Custom-Header": "custom_value"}

            headers = client._prepare_headers(custom)

            assert headers["Authorization"] == "Bearer test_token"
            assert headers["X-Custom-Header"] == "custom_value"


# =============================================================================
# Health Check Tests
# =============================================================================

class TestHealthCheck:
    """Tests for health check functionality."""

    def test_health_check_success(self, temp_config_file, temp_token_cache):
        """Test health check returns True when authenticated and API responds."""
        with patch("src.data_pipeline.questrade_base_client.yaml.safe_load") as mock_yaml:
            mock_yaml.return_value = {"questrade": {"api": {}}}
            client = QuestradeBaseClient(
                config_path=temp_config_file,
                token_cache_path=temp_token_cache,
            )

            # Set up valid token
            client.access_token = "valid_token"
            client.api_server = "https://api.server.com"
            client.token_expiry = datetime.now() + timedelta(hours=1)

            # Mock the API call
            mock_response = APIResponse(
                success=True,
                data={"accounts": []},
                status_code=200,
            )

            with patch.object(client, "_request", return_value=mock_response):
                assert client.health_check() is True

    def test_health_check_not_authenticated(self, temp_config_file, temp_token_cache):
        """Test health check returns False when not authenticated."""
        with patch("src.data_pipeline.questrade_base_client.yaml.safe_load") as mock_yaml:
            mock_yaml.return_value = {"questrade": {"api": {}}}
            client = QuestradeBaseClient(
                config_path=temp_config_file,
                token_cache_path=temp_token_cache,
            )

            # No token set
            with patch.object(client, "authenticate", return_value=False):
                assert client.health_check() is False


# =============================================================================
# Token Info Tests
# =============================================================================

class TestTokenInfo:
    """Tests for token info retrieval."""

    def test_get_token_info_authenticated(self, temp_config_file, temp_token_cache):
        """Test get_token_info with valid token."""
        with patch("src.data_pipeline.questrade_base_client.yaml.safe_load") as mock_yaml:
            mock_yaml.return_value = {"questrade": {"api": {}}}
            client = QuestradeBaseClient(
                config_path=temp_config_file,
                token_cache_path=temp_token_cache,
            )

            client.access_token = "test_token"
            client.api_server = "https://api.server.com"
            client.token_expiry = datetime.now() + timedelta(hours=1)

            info = client.get_token_info()

            assert info["authenticated"] is True
            assert info["api_server"] == "https://api.server.com"
            assert info["token_expiry"] is not None
            assert info["seconds_until_expiry"] > 0

    def test_get_token_info_not_authenticated(self, temp_config_file, temp_token_cache):
        """Test get_token_info without token."""
        with patch("src.data_pipeline.questrade_base_client.yaml.safe_load") as mock_yaml:
            mock_yaml.return_value = {"questrade": {"api": {}}}
            client = QuestradeBaseClient(
                config_path=temp_config_file,
                token_cache_path=temp_token_cache,
            )

            info = client.get_token_info()

            assert info["authenticated"] is False
            assert info["token_expiry"] is None
            assert info["seconds_until_expiry"] is None
