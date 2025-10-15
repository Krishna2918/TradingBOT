"""Runtime configuration loading for the AI Trading System scaffold."""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

# Ingest values from .env (if present) before inspecting the environment.
load_dotenv()


class Settings(BaseModel):
    """Typed view over environment-driven configuration."""

    alpha_vantage_api_key: str = Field(..., alias="ALPHA_VANTAGE_API_KEY")
    newsapi_key: str = Field(..., alias="NEWSAPI_KEY")
    finnhub_api_key: str = Field(..., alias="FINNHUB_API_KEY")
    questrade_refresh_token: str = Field(..., alias="QUESTRADE_REFRESH_TOKEN")
    ollama_base_url: str = Field(..., alias="OLLAMA_BASE_URL")
    demo_mode: bool = Field(False, alias="DEMO_MODE")
    initial_capital: float = Field(100_000.0, alias="INITIAL_CAPITAL")
    max_daily_risk: float = Field(0.03, alias="MAX_DAILY_RISK")
    target_daily_return: float = Field(0.05, alias="TARGET_DAILY_RETURN")
    max_position_risk: float = Field(0.02, alias="MAX_POSITION_RISK")

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


def _env_values() -> dict[str, Any]:
    """Collect environment variables using field aliases."""

    values: dict[str, Any] = {}
    for field in Settings.model_fields.values():
        alias = field.alias or field.serialization_alias
        if not alias:
            continue
        value = os.getenv(alias)
        if value is not None:
            values[alias] = value
    return values


@lru_cache(maxsize=1)
def get_settings(override: Optional[dict[str, Any]] = None) -> Settings:
    """Return cached settings, optionally overriding fields for tests."""

    data = _env_values()
    if override:
        data.update(override)
    return Settings.model_validate(data)


__all__ = ["Settings", "get_settings"]
