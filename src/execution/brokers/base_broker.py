"""
Broker API base interfaces.

Defines a minimal interface for execution backends (paper/live).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BrokerAPI(ABC):
    """Abstract execution backend for placing and managing orders."""

    @abstractmethod
    def get_balances(self) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def get_positions(self) -> Optional[List[Dict[str, Any]]]:
        raise NotImplementedError

    @abstractmethod
    def get_quotes(self, symbols: List[str]) -> Optional[List[Dict[str, Any]]]:
        raise NotImplementedError

    @abstractmethod
    def place_order(
        self,
        *,
        symbol: str,
        quantity: float,
        action: str,  # "Buy" or "Sell"
        order_type: str = "Market",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "Day",
    ) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def cancel_order(self, order_id: Any) -> bool:
        raise NotImplementedError

    # Optional live helpers
    def get_open_orders(self) -> Optional[List[Dict[str, Any]]]:
        return None

    def get_auth_status(self) -> Dict[str, Any]:
        return {"authenticated": False, "practice_mode": True, "allow_trading": False}
