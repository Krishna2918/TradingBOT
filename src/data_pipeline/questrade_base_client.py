"""
Questrade Base Client
=====================

Shared base client for Questrade API integrations. Handles:
- OAuth2 token refresh
- Token caching with file persistence
- Authorization header injection
- Rate limiting via global rate limiter

Usage:
    from src.data_pipeline.questrade_base_client import QuestradeBaseClient

    class MyQuestradeClient(QuestradeBaseClient):
        def my_method(self):
            response = self._get("/v1/accounts")
            return response.data
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import yaml

from src.utils.api_client import BaseAPIClient, APIResponse, APIClientConfig

logger = logging.getLogger(__name__)


class QuestradeBaseClient(BaseAPIClient):
    """
    Base client for Questrade API with OAuth2 token management.

    Inherits from BaseAPIClient for:
    - Automatic retry with exponential backoff
    - Circuit breaker for failure handling
    - Global rate limiting (questrade limiter)
    - Consistent error handling

    Adds:
    - OAuth2 token refresh logic
    - Token caching with file persistence
    - Automatic authentication on demand
    """

    # OAuth2 endpoints
    TOKEN_ENDPOINT = "https://login.questrade.com/oauth2/token"

    def __init__(
        self,
        config_path: str = "config/questrade_config.yaml",
        token_cache_path: Optional[str] = None,
        refresh_env_var: str = "QUESTRADE_REFRESH_TOKEN",
        name: str = "Questrade",
    ) -> None:
        """Initialize Questrade base client.

        Args:
            config_path: Path to Questrade configuration YAML file.
            token_cache_path: Path to token cache file. Defaults to config value.
            refresh_env_var: Environment variable name for refresh token.
            name: Client name for logging.
        """
        # Load config first
        self.questrade_config = self._load_questrade_config(config_path)

        # Determine token cache path
        default_cache = self.questrade_config.get("api", {}).get(
            "token_cache_path", "config/questrade_token_cache.json"
        )
        self.token_cache_path = Path(token_cache_path or default_cache)
        self.refresh_env_var = refresh_env_var

        # Token state
        self.access_token: Optional[str] = None
        self.api_server: Optional[str] = None
        self.token_expiry: Optional[datetime] = None
        self._cached_refresh_token: Optional[str] = None

        # Configure for Questrade - longer timeouts, uses questrade rate limiter
        api_config = APIClientConfig(
            timeout=self.questrade_config.get("api", {}).get("timeout", 30.0),
            connect_timeout=5.0,
            read_timeout=30.0,
            max_retries=2,
            circuit_breaker_threshold=5,
            circuit_breaker_timeout=60.0,
            validate_responses=True,
            log_requests=True,
            log_responses=False,
        )

        # Load token cache before initializing base (needed for api_server)
        self._load_token_cache()

        # Initialize base client with api_server as base_url (or placeholder)
        super().__init__(
            name=name,
            base_url=self.api_server or "https://api01.iq.questrade.com",
            rate_limiter_name="questrade",
            config=api_config,
        )

        logger.info("%s base client initialized", name)

    def _load_questrade_config(self, path: str) -> Dict[str, Any]:
        """Load Questrade configuration from YAML file."""
        try:
            with open(path, "r", encoding="utf-8") as handle:
                raw = yaml.safe_load(handle) or {}
                return raw.get("questrade", {})
        except FileNotFoundError:
            logger.warning("Questrade config %s not found; using defaults", path)
        except Exception as exc:
            logger.error("Failed to load Questrade config: %s", exc)
        return {
            "api": {"timeout": 30},
            "oauth": {"token_endpoint": "https://login.questrade.com/oauth2/token"},
            "rate_limits": {"requests_per_second": 1},
        }

    # ------------------------------------------------------------------
    # Token Cache Management
    # ------------------------------------------------------------------

    def _load_token_cache(self) -> None:
        """Load cached token from file."""
        if not self.token_cache_path.exists():
            return
        try:
            with open(self.token_cache_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception as exc:
            logger.warning("Could not read token cache %s: %s", self.token_cache_path, exc)
            return

        self.access_token = data.get("access_token")
        self.api_server = data.get("api_server")
        self._cached_refresh_token = data.get("refresh_token")

        expires_at = data.get("expires_at")
        if expires_at:
            try:
                self.token_expiry = datetime.fromisoformat(expires_at)
            except ValueError:
                logger.debug("Invalid expiry in token cache; ignoring value")
                self.token_expiry = None

    def _save_token_cache(self, refresh_token: str) -> None:
        """Persist token to cache file."""
        payload = {
            "access_token": self.access_token,
            "api_server": self.api_server,
            "expires_at": self.token_expiry.isoformat() if self.token_expiry else None,
            "refresh_token": refresh_token,
            "updated_at": datetime.now().isoformat(),
        }
        try:
            self.token_cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.token_cache_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        except Exception as exc:
            logger.warning("Unable to persist token cache: %s", exc)

    def _clear_token_cache(self) -> None:
        """Clear all cached token data."""
        self.access_token = None
        self.api_server = None
        self.token_expiry = None
        if self.token_cache_path.exists():
            try:
                self.token_cache_path.unlink()
            except OSError:
                logger.debug("Failed to remove token cache at %s", self.token_cache_path)

    # ------------------------------------------------------------------
    # OAuth2 Authentication
    # ------------------------------------------------------------------

    def authenticate(self) -> bool:
        """
        Obtain a fresh access token.

        Tries tokens in order:
        1. Cached refresh token
        2. Environment variable
        3. Config file

        Returns:
            True if authentication successful, False otherwise.
        """
        candidates: List[str] = []

        if self._cached_refresh_token:
            candidates.append(self._cached_refresh_token)

        env_token = os.getenv(self.refresh_env_var)
        if env_token and env_token not in candidates:
            candidates.append(env_token)

        config_token = self.questrade_config.get("api", {}).get("refresh_token")
        if config_token and config_token not in candidates:
            logger.warning(
                "Refresh token loaded from config; move it to env var %s",
                self.refresh_env_var,
            )
            candidates.append(config_token)

        if not candidates:
            logger.error(
                "No Questrade refresh token available. Set %s in your environment.",
                self.refresh_env_var,
            )
            return False

        for token in candidates:
            if self._perform_auth(token):
                return True

        logger.error("All refresh token attempts failed")
        return False

    def _perform_auth(self, refresh_token: str) -> bool:
        """Perform OAuth2 token refresh."""
        token_endpoint = self.questrade_config.get("oauth", {}).get(
            "token_endpoint", self.TOKEN_ENDPOINT
        )
        timeout = self.questrade_config.get("api", {}).get("timeout", 30)

        try:
            response = requests.get(
                token_endpoint,
                params={"grant_type": "refresh_token", "refresh_token": refresh_token},
                timeout=timeout,
            )
        except requests.RequestException as exc:
            logger.error("Authentication request failed: %s", exc)
            return False

        if response.status_code != 200:
            logger.error(
                "Authentication rejected (%s): %s",
                response.status_code,
                response.text,
            )
            # Clear cache if cached token was rejected
            if response.status_code in (400, 401) and refresh_token == self._cached_refresh_token:
                self._clear_token_cache()
                self._cached_refresh_token = None
            return False

        data = response.json()

        self.access_token = data["access_token"]
        self.api_server = data["api_server"].rstrip("/")
        expires_in = int(data.get("expires_in", 1800))
        self.token_expiry = datetime.now() + timedelta(seconds=expires_in)

        # Update base_url to use new api_server
        self.base_url = self.api_server

        # Save new refresh token
        new_refresh = data.get("refresh_token", refresh_token)
        self._cached_refresh_token = new_refresh
        self._save_token_cache(new_refresh)

        logger.info(
            "Authenticated against %s; token valid until %s",
            self.api_server,
            self.token_expiry.strftime("%Y-%m-%d %H:%M:%S"),
        )

        return True

    def _is_token_valid(self) -> bool:
        """Check if current token is valid (with 5 minute buffer)."""
        if not self.access_token or not self.api_server or not self.token_expiry:
            return False
        return datetime.now() < (self.token_expiry - timedelta(minutes=5))

    def _ensure_authenticated(self) -> bool:
        """Ensure we have a valid token, refreshing if needed."""
        if self._is_token_valid():
            return True
        logger.info("Access token missing or expired; refreshing")
        return self.authenticate()

    # ------------------------------------------------------------------
    # BaseAPIClient Overrides
    # ------------------------------------------------------------------

    def _prepare_headers(self, headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Prepare request headers with OAuth2 bearer token."""
        result = super()._prepare_headers(headers)
        if self.access_token:
            result["Authorization"] = f"Bearer {self.access_token}"
        return result

    def _get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> APIResponse:
        """Make authenticated GET request.

        Automatically ensures authentication before request.
        """
        if not self._ensure_authenticated():
            return APIResponse(
                success=False,
                error="Authentication failed",
                status_code=None,
            )
        return super()._get(endpoint, params=params, **kwargs)

    def _post(
        self,
        endpoint: str,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        **kwargs: Any,
    ) -> APIResponse:
        """Make authenticated POST request.

        Automatically ensures authentication before request.
        """
        if not self._ensure_authenticated():
            return APIResponse(
                success=False,
                error="Authentication failed",
                status_code=None,
            )
        return super()._post(endpoint, json=json, data=data, **kwargs)

    def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs: Any,
    ) -> APIResponse:
        """Make authenticated request with any HTTP method.

        Handles 401 responses by re-authenticating once.
        """
        if not self._ensure_authenticated():
            return APIResponse(
                success=False,
                error="Authentication failed",
                status_code=None,
            )

        response = super()._request(method, endpoint, **kwargs)

        # Handle 401 by re-authenticating once
        if response.status_code == 401:
            logger.info("Access token rejected; attempting re-authenticate")
            if self.authenticate():
                return super()._request(method, endpoint, **kwargs)

        return response

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def health_check(self) -> bool:
        """Check if Questrade API is reachable and authenticated.

        Returns:
            True if API is healthy and authenticated.
        """
        if not self._ensure_authenticated():
            return False

        # Try to list accounts as a health check
        response = super()._get("/v1/accounts")
        return response.success

    def is_authenticated(self) -> bool:
        """Check if currently authenticated with valid token."""
        return self._is_token_valid()

    def get_token_info(self) -> Dict[str, Any]:
        """Get information about current token state."""
        return {
            "authenticated": self._is_token_valid(),
            "api_server": self.api_server,
            "token_expiry": self.token_expiry.isoformat() if self.token_expiry else None,
            "seconds_until_expiry": (
                (self.token_expiry - datetime.now()).total_seconds()
                if self.token_expiry
                else None
            ),
        }
