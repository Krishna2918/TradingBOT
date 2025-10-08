"""
Questrade API client with automatic token refresh, local caching, and
convenience helpers for quotes, positions, and order placement.
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


logger = logging.getLogger(__name__)


class QuestradeClient:
    """Thin wrapper around the Questrade REST API."""

    def __init__(
        self,
        config_path: str = "config/questrade_config.yaml",
        token_cache_path: Optional[str] = None,
        refresh_env_var: str = "QUESTRADE_REFRESH_TOKEN",
        allow_trading: bool = False,
        practice_mode: bool = True,
    ) -> None:
        self.config = self._load_config(config_path)
        default_cache = self.config.get("api", {}).get(
            "token_cache_path", "config/questrade_token_cache.json"
        )
        self.token_cache_path = Path(token_cache_path or default_cache)
        self.refresh_env_var = refresh_env_var
        
        # Trading controls - can be overridden by environment variables
        self.allow_trading = os.getenv("QUESTRADE_ALLOW_TRADING", str(allow_trading)).lower() in ("true", "1", "yes")
        self.practice_mode = os.getenv("QUESTRADE_PRACTICE_MODE", str(practice_mode)).lower() in ("true", "1", "yes")

        self.access_token: Optional[str] = None
        self.api_server: Optional[str] = None
        self.token_expiry: Optional[datetime] = None
        self.account_id: Optional[str] = None
        self._cached_refresh_token: Optional[str] = None

        rate_cfg = self.config.get("rate_limits", {})
        rps = rate_cfg.get("requests_per_second", 1)
        self.min_request_interval = 1.0 / max(rps, 1)
        self.last_request_time = 0.0

        self._load_token_cache()
        
        # Log trading mode
        if not self.allow_trading:
            logger.warning("TRADING DISABLED - Orders will be rejected")
        elif self.practice_mode:
            logger.warning("PRACTICE MODE ENABLED - Using paper/practice account")
        else:
            logger.warning("LIVE TRADING ENABLED - Using real money account")
        
        logger.info("Questrade client ready; will authenticate on demand")

    def _load_config(self, path: str) -> Dict[str, Any]:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                raw = yaml.safe_load(handle) or {}
                return raw.get("questrade", {})
        except FileNotFoundError:
            logger.warning("Questrade config %s not found; using defaults", path)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to load Questrade config: %s", exc)
        return {
            "api": {"timeout": 30},
            "oauth": {"token_endpoint": "https://login.questrade.com/oauth2/token"},
            "rate_limits": {"requests_per_second": 1},
        }

    def _load_token_cache(self) -> None:
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
        self.account_id = data.get("account_id") or self.config.get("account", {}).get(
            "account_id"
        )
        self._cached_refresh_token = data.get("refresh_token")

        expires_at = data.get("expires_at")
        if expires_at:
            try:
                self.token_expiry = datetime.fromisoformat(expires_at)
            except ValueError:
                logger.debug("Invalid expiry in token cache; ignoring value")
                self.token_expiry = None

    def _save_token_cache(self, refresh_token: str) -> None:
        payload = {
            "access_token": self.access_token,
            "api_server": self.api_server,
            "expires_at": self.token_expiry.isoformat() if self.token_expiry else None,
            "refresh_token": refresh_token,
            "account_id": self.account_id,
            "updated_at": datetime.now().isoformat(),
        }
        try:
            self.token_cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.token_cache_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        except Exception as exc:  # pragma: no cover - disk issues
            logger.warning("Unable to persist token cache: %s", exc)

    def _clear_token_cache(self) -> None:
        self.access_token = None
        self.api_server = None
        self.token_expiry = None
        if self.token_cache_path.exists():
            try:
                self.token_cache_path.unlink()
            except OSError:
                logger.debug("Failed to remove token cache at %s", self.token_cache_path)

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------
    def authenticate(self) -> bool:
        """Obtain a fresh access token; tries cached, env, then config refresh tokens."""

        candidates: List[str] = []

        if self._cached_refresh_token:
            candidates.append(self._cached_refresh_token)

        env_token = os.getenv(self.refresh_env_var)
        if env_token and env_token not in candidates:
            candidates.append(env_token)

        config_token = self.config.get("api", {}).get("refresh_token")
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
        token_endpoint = self.config.get("oauth", {}).get(
            "token_endpoint", "https://login.questrade.com/oauth2/token"
        )
        timeout = self.config.get("api", {}).get("timeout", 30)

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
            if response.status_code in (400, 401) and refresh_token == self._cached_refresh_token:
                self._clear_token_cache()
                self._cached_refresh_token = None
            return False

        data = response.json()

        self.access_token = data["access_token"]
        self.api_server = data["api_server"].rstrip("/")
        expires_in = int(data.get("expires_in", 1800))
        self.token_expiry = datetime.now() + timedelta(seconds=expires_in)

        new_refresh = data.get("refresh_token", refresh_token)
        self._cached_refresh_token = new_refresh
        self._save_token_cache(new_refresh)

        logger.info(
            "Authenticated against %s; token valid until %s",
            self.api_server,
            self.token_expiry.strftime("%Y-%m-%d %H:%M:%S"),
        )

        if new_refresh != refresh_token and not env_token_matches(self.refresh_env_var, new_refresh):
            logger.info(
                "New refresh token issued. Update %s to keep parity with cache.",
                self.refresh_env_var,
            )

        return True

    def _is_token_valid(self) -> bool:
        if not self.access_token or not self.api_server or not self.token_expiry:
            return False
        return datetime.now() < (self.token_expiry - timedelta(minutes=5))

    def _ensure_authenticated(self) -> bool:
        if self._is_token_valid():
            return True
        logger.info("Access token missing or expired; refreshing")
        return self.authenticate()

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------
    def _rate_limit(self) -> None:
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def _request(
        self,
        method: str,
        endpoint: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        payload: Optional[Dict[str, Any]] = None,
        retry: bool = True,
    ) -> Optional[Dict[str, Any]]:
        if not self._ensure_authenticated():
            return None

        self._rate_limit()
        url = f"{self.api_server}{endpoint}"
        headers = {"Authorization": f"Bearer {self.access_token}"}
        if payload is not None:
            headers["Content-Type"] = "application/json"

        timeout = self.config.get("api", {}).get("timeout", 30)

        try:
            response = requests.request(
                method,
                url,
                params=params,
                json=payload,
                headers=headers,
                timeout=timeout,
            )
        except requests.RequestException as exc:
            logger.error("Request to %s failed: %s", endpoint, exc)
            return None

        if response.status_code == 401 and retry:
            logger.info("Access token rejected; attempting one re-authenticate cycle")
            self.authenticate()
            return self._request(
                method,
                endpoint,
                params=params,
                payload=payload,
                retry=False,
            )

        if response.status_code >= 400:
            logger.error(
                "Questrade API error %s on %s: %s",
                response.status_code,
                endpoint,
                response.text,
            )
            return None

        if not response.content:
            return None

        try:
            return response.json()
        except ValueError:
            logger.error("Non-JSON response from %s", endpoint)
            return None

    def _get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        return self._request("GET", endpoint, params=params)

    def _post(self, endpoint: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return self._request("POST", endpoint, payload=payload)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_accounts(self) -> Optional[List[Dict[str, Any]]]:
        result = self._get("/v1/accounts")
        if not result:
            return None

        accounts = result.get("accounts", [])
        if accounts and not self.account_id:
            self.account_id = str(accounts[0]["number"])
            if self._cached_refresh_token:
                self._save_token_cache(self._cached_refresh_token)
        return accounts

    def _ensure_account(self) -> Optional[str]:
        if self.account_id:
            return self.account_id
        accounts = self.get_accounts()
        if accounts:
            return self.account_id
        logger.error("No Questrade account available; run get_accounts() first")
        return None

    def get_account_balances(self, account_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        acc = account_id or self._ensure_account()
        if not acc:
            return None
        return self._get(f"/v1/accounts/{acc}/balances")

    def get_balances(self, account_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get account balances including cash, market value, and buying power"""
        acc = account_id or self._ensure_account()
        if not acc:
            return None
        return self._get(f"/v1/accounts/{acc}/balances")
    
    def get_positions(self, account_id: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        acc = account_id or self._ensure_account()
        if not acc:
            return None
        result = self._get(f"/v1/accounts/{acc}/positions")
        if not result:
            return None
        return result.get("positions", [])

    def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        quote_list = self.get_quotes([symbol]) or []
        return quote_list[0] if quote_list else None

    def get_quotes(self, symbols: List[str]) -> Optional[List[Dict[str, Any]]]:
        symbol_ids: List[str] = []
        for symbol in symbols:
            symbol_id = self._resolve_symbol_id(symbol)
            if symbol_id:
                symbol_ids.append(str(symbol_id))

        if not symbol_ids:
            logger.error("No symbol IDs resolved for request")
            return None

        result = self._get("/v1/markets/quotes", {"ids": ",".join(symbol_ids)})
        if not result:
            return None
        return result.get("quotes", [])

    def place_order(
        self,
        *,
        symbol: str,
        quantity: float,
        action: str,
        order_type: str = "Limit",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "Day",
        primary_route: str = "AUTO",
        secondary_route: str = "AUTO",
        is_all_or_none: bool = False,
        is_anonymous: bool = False,
        account_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        # Trading safety check
        if not self.allow_trading:
            logger.error("TRADING DISABLED - Order rejected: %s %s x%s", action, symbol, quantity)
            return {
                "error": "TRADING_DISABLED",
                "message": "Trading is disabled. Set allow_trading=True to enable.",
                "rejected_order": {
                    "symbol": symbol,
                    "quantity": quantity,
                    "action": action,
                    "order_type": order_type
                }
            }
        
        # Practice mode warning
        if self.practice_mode:
            logger.warning("PRACTICE MODE - Order will execute in paper account: %s %s x%s", 
                         action, symbol, quantity)
        
        acc = account_id or self._ensure_account()
        if not acc:
            return None

        symbol_id = self._resolve_symbol_id(symbol)
        if not symbol_id:
            logger.error("Unable to resolve symbol ID for %s", symbol)
            return None

        normalized_action = action.strip().replace(" ", "").capitalize()

        payload: Dict[str, Any] = {
            "accountNumber": acc,
            "symbolId": symbol_id,
            "quantity": quantity,
            "icebergQuantity": 0,
            "isAllOrNone": is_all_or_none,
            "isAnonymous": is_anonymous,
            "orderType": order_type,
            "timeInForce": time_in_force,
            "action": normalized_action,
            "primaryRoute": primary_route,
            "secondaryRoute": secondary_route,
        }

        if order_type.lower() in {"limit", "stoplimit"} and limit_price is None:
            raise ValueError("limit_price is required for limit or stop-limit orders")

        if limit_price is not None:
            payload["limitPrice"] = limit_price
        if stop_price is not None:
            payload["stopPrice"] = stop_price

        result = self._post(f"/v1/accounts/{acc}/orders", payload)
        if result:
            logger.info(
                "Submitted %s order for %s x%s (orderId=%s)",
                normalized_action,
                symbol,
                quantity,
                result.get("id"),
            )
        return result

    def cancel_order(self, order_id: int, account_id: Optional[str] = None) -> bool:
        acc = account_id or self._ensure_account()
        if not acc:
            return False
        result = self._request("POST", f"/v1/accounts/{acc}/orders/{order_id}/cancel")
        return bool(result)

    def get_portfolio_summary(self) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "total_value": 0.0,
            "cash": 0.0,
            "positions_value": 0.0,
            "positions": [],
            "pnl_total": 0.0,
            "timestamp": datetime.now().isoformat(),
        }

        balances = self.get_account_balances()
        if balances:
            for currency_balance in balances.get("perCurrencyBalances", []):
                if currency_balance.get("currency") == "CAD":
                    summary["cash"] = currency_balance.get("cash", 0.0)
                    summary["total_value"] = currency_balance.get("totalEquity", 0.0)

        positions = self.get_positions() or []
        total_market_value = 0.0
        total_pnl = 0.0

        for pos in positions:
            qty = pos.get("openQuantity", 0.0)
            avg_price = pos.get("averageEntryPrice", 0.0)
            current_price = pos.get("currentPrice", 0.0)
            market_value = pos.get("currentMarketValue", qty * current_price)
            pnl = pos.get("openPnl", 0.0)
            cost = qty * avg_price

            summary["positions"].append(
                {
                    "symbol": pos.get("symbol"),
                    "quantity": qty,
                    "avg_price": avg_price,
                    "current_price": current_price,
                    "market_value": market_value,
                    "pnl": pnl,
                    "pnl_percent": (pnl / cost) * 100 if cost else 0.0,
                }
            )

            total_market_value += market_value
            total_pnl += pnl

        summary["positions_value"] = total_market_value
        summary["pnl_total"] = total_pnl
        return summary

    def get_open_orders(self, account_id: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        """Fetch open orders for the account."""
        acc = account_id or self._ensure_account()
        if not acc:
            return None
        result = self._get(f"/v1/accounts/{acc}/orders", {"state": "open"})
        if not result:
            return []
        return result.get("orders", [])

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _resolve_symbol_id(self, symbol: str) -> Optional[int]:
        search = self._get("/v1/symbols/search", {"prefix": symbol.upper()})
        if not search:
            return None
        for candidate in search.get("symbols", []):
            if candidate.get("symbol") == symbol.upper():
                return candidate.get("symbolId")
        symbols = search.get("symbols", [])
        if symbols:
            return symbols[0].get("symbolId")
        return None


def env_token_matches(env_var: str, refresh_token: str) -> bool:
    current = os.getenv(env_var)
    return bool(current and current.strip() == refresh_token.strip())


_questrade_client: Optional[QuestradeClient] = None


def get_questrade_client() -> QuestradeClient:
    global _questrade_client
    if _questrade_client is None:
        _questrade_client = QuestradeClient()
    return _questrade_client


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = QuestradeClient()
    if client.authenticate():
        accounts = client.get_accounts() or []
        logger.info("Discovered %s account(s)", len(accounts))
        if accounts:
            logger.info("Primary account %s", client.account_id)
        summary = client.get_portfolio_summary()
        logger.info(
            "Portfolio value %.2f CAD with %s positions",
            summary["total_value"],
            len(summary["positions"]),
        )
