"""
Questrade Broker Adapter

Bridges ExecutionEngine to Questrade via QuestradeClient.
Supports paper (practice) and live modes, controlled by flags/env.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .base_broker import BrokerAPI
from src.data_pipeline.questrade_client import QuestradeClient


logger = logging.getLogger(__name__)


class QuestradeBroker(BrokerAPI):
    """Adapter over QuestradeClient implementing BrokerAPI."""

    def __init__(
        self,
        *,
        config_path: str = "config/questrade_config.yaml",
        allow_trading: bool = False,
        practice_mode: bool = True,
        token_cache_path: Optional[str] = None,
    ) -> None:
        self.client = QuestradeClient(
            config_path=config_path,
            token_cache_path=token_cache_path,
            allow_trading=allow_trading,
            practice_mode=practice_mode,
        )
        # Authenticate lazily on first request; pre-warm to surface issues early
        self._ready = self.client.authenticate()
        if not self._ready:
            logger.warning("QuestradeBroker failed to authenticate; API calls may fail")

    # --- Portfolio ---
    def get_balances(self) -> Optional[Dict[str, Any]]:
        return self.client.get_account_balances()

    def get_positions(self) -> Optional[List[Dict[str, Any]]]:
        return self.client.get_positions()

    # --- Market Data ---
    def get_quotes(self, symbols: List[str]) -> Optional[List[Dict[str, Any]]]:
        return self.client.get_quotes(symbols)

    # --- Orders ---
    def place_order(
        self,
        *,
        symbol: str,
        quantity: float,
        action: str,
        order_type: str = "Market",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "Day",
    ) -> Optional[Dict[str, Any]]:
        # Questrade stock orders do not support fractional; round down to int
        qty = int(quantity) if quantity >= 1 else 0
        if qty <= 0:
            logger.warning("Refusing to place order with non-positive quantity: %s", quantity)
            return {
                "error": "INVALID_QUANTITY",
                "message": f"Quantity {quantity} not supported for Questrade stocks",
            }

        return self.client.place_order(
            symbol=symbol,
            quantity=qty,
            action=action,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            time_in_force=time_in_force,
        )

    def cancel_order(self, order_id: Any) -> bool:
        try:
            return self.client.cancel_order(int(order_id))
        except Exception:
            logger.exception("Failed to cancel order %s", order_id)
            return False

    def get_open_orders(self) -> Optional[List[Dict[str, Any]]]:
        try:
            return self.client.get_open_orders()
        except Exception:
            logger.exception("Failed to fetch open orders")
            return None

    def get_auth_status(self) -> Dict[str, Any]:
        return {
            "authenticated": bool(self.client.access_token and self.client.api_server),
            "practice_mode": bool(self.client.practice_mode),
            "allow_trading": bool(self.client.allow_trading),
            "account_id": self.client.account_id,
        }
