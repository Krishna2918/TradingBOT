"""Runtime configuration loading for the AI Trading System scaffold."""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

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

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True


@lru_cache(maxsize=1)
def get_settings(override: Optional[dict[str, object]] = None) -> Settings:
    """Return cached settings, optionally overriding fields for tests."""

    if override:
        return Settings(**override)
    return Settings()


__all__ = ["Settings", "get_settings"]
