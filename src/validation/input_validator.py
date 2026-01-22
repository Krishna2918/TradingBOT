"""
Input Validation Layer
======================

Pydantic models for validating all external inputs to the trading system.
Prevents invalid data from causing incorrect results or silent failures.

Usage:
    from src.validation.input_validator import (
        OrderRequest, PositionRequest, SymbolValidator, validate_order
    )

    # Validate order input
    try:
        order = OrderRequest(
            symbol="AAPL",
            side="buy",
            quantity=100,
            price=150.00,
        )
    except ValidationError as e:
        print(f"Invalid order: {e}")

    # Validate with decorator
    @validate_inputs(OrderRequest)
    def place_order(symbol, side, quantity, price=None):
        ...
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
    ValidationError,
)

logger = logging.getLogger('trading.validation')


# =============================================================================
# Enums
# =============================================================================

class OrderSideEnum(str, Enum):
    """Valid order sides."""
    BUY = "buy"
    SELL = "sell"
    LONG = "long"
    SHORT = "short"


class OrderTypeEnum(str, Enum):
    """Valid order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class TimeInForceEnum(str, Enum):
    """Valid time in force values."""
    DAY = "day"
    GTC = "gtc"  # Good Till Cancelled
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill


class DataQualityLevel(str, Enum):
    """Data quality levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


# =============================================================================
# Symbol Validation
# =============================================================================

# Common US exchanges
VALID_EXCHANGES = {"NYSE", "NASDAQ", "AMEX", "ARCA", "BATS", "IEX", "TSX", "TSXV"}

# Pattern for valid stock symbols
SYMBOL_PATTERN = re.compile(r'^[A-Z]{1,5}(\.[A-Z]{1,2})?$')

# Known invalid/problematic symbols
BLACKLISTED_SYMBOLS = {
    "TEST", "DEMO", "NULL", "NONE", "N/A", "NA", "TBD",
}


class SymbolValidator(BaseModel):
    """Validates stock symbols."""

    model_config = ConfigDict(str_strip_whitespace=True)

    symbol: str = Field(..., min_length=1, max_length=10)

    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate symbol format and content."""
        v = v.upper().strip()

        if not v:
            raise ValueError("Symbol cannot be empty")

        if v in BLACKLISTED_SYMBOLS:
            raise ValueError(f"Invalid symbol: {v} is blacklisted")

        if not SYMBOL_PATTERN.match(v):
            raise ValueError(
                f"Invalid symbol format: {v}. "
                "Expected 1-5 uppercase letters, optionally followed by .XX suffix"
            )

        return v


# =============================================================================
# Price Validation
# =============================================================================

class PriceValidator(BaseModel):
    """Validates price values."""

    model_config = ConfigDict(str_strip_whitespace=True)

    price: float = Field(..., gt=0)
    min_price: float = Field(default=0.0001)
    max_price: float = Field(default=1_000_000.0)

    @field_validator('price')
    @classmethod
    def validate_price_range(cls, v: float) -> float:
        """Validate price is within reasonable range."""
        if v <= 0:
            raise ValueError("Price must be positive")
        if v > 1_000_000:
            raise ValueError(f"Price {v} exceeds maximum allowed (1,000,000)")
        if v < 0.0001:
            raise ValueError(f"Price {v} below minimum allowed (0.0001)")
        return round(v, 4)


# =============================================================================
# Order Validation
# =============================================================================

class OrderRequest(BaseModel):
    """Validates order request inputs."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        use_enum_values=True,
    )

    symbol: str = Field(..., min_length=1, max_length=10)
    side: OrderSideEnum
    quantity: float = Field(..., gt=0)
    order_type: OrderTypeEnum = Field(default=OrderTypeEnum.MARKET)
    price: Optional[float] = Field(default=None, gt=0)
    stop_price: Optional[float] = Field(default=None, gt=0)
    time_in_force: TimeInForceEnum = Field(default=TimeInForceEnum.DAY)
    stop_loss: Optional[float] = Field(default=None, gt=0)
    take_profit: Optional[float] = Field(default=None, gt=0)

    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate and normalize symbol."""
        v = v.upper().strip()
        if not SYMBOL_PATTERN.match(v):
            raise ValueError(f"Invalid symbol format: {v}")
        if v in BLACKLISTED_SYMBOLS:
            raise ValueError(f"Symbol {v} is blacklisted")
        return v

    @field_validator('quantity')
    @classmethod
    def validate_quantity(cls, v: float) -> float:
        """Validate quantity."""
        if v <= 0:
            raise ValueError("Quantity must be positive")
        if v > 1_000_000:
            raise ValueError(f"Quantity {v} exceeds maximum allowed")
        return v

    @model_validator(mode='after')
    def validate_order_type_requirements(self) -> 'OrderRequest':
        """Validate order type specific requirements."""
        if self.order_type == OrderTypeEnum.LIMIT and self.price is None:
            raise ValueError("Limit orders require a price")
        if self.order_type == OrderTypeEnum.STOP and self.stop_price is None:
            raise ValueError("Stop orders require a stop_price")
        if self.order_type == OrderTypeEnum.STOP_LIMIT:
            if self.stop_price is None or self.price is None:
                raise ValueError("Stop-limit orders require both price and stop_price")
        return self

    @model_validator(mode='after')
    def validate_stop_take_profit(self) -> 'OrderRequest':
        """Validate stop loss and take profit relative to entry."""
        if self.price and self.stop_loss and self.take_profit:
            is_buy = self.side in (OrderSideEnum.BUY, OrderSideEnum.LONG)

            if is_buy:
                # For buy orders: stop_loss < price < take_profit
                if self.stop_loss >= self.price:
                    raise ValueError(
                        f"Stop loss ({self.stop_loss}) must be below entry price ({self.price}) for buy orders"
                    )
                if self.take_profit <= self.price:
                    raise ValueError(
                        f"Take profit ({self.take_profit}) must be above entry price ({self.price}) for buy orders"
                    )
            else:
                # For sell/short orders: take_profit < price < stop_loss
                if self.stop_loss <= self.price:
                    raise ValueError(
                        f"Stop loss ({self.stop_loss}) must be above entry price ({self.price}) for sell orders"
                    )
                if self.take_profit >= self.price:
                    raise ValueError(
                        f"Take profit ({self.take_profit}) must be below entry price ({self.price}) for sell orders"
                    )

        return self


# =============================================================================
# Position Validation
# =============================================================================

class PositionRequest(BaseModel):
    """Validates position tracking inputs."""

    model_config = ConfigDict(str_strip_whitespace=True)

    symbol: str = Field(..., min_length=1, max_length=10)
    entry_price: float = Field(..., gt=0)
    quantity: int = Field(..., gt=0)
    stop_loss: Optional[float] = Field(default=None, gt=0)
    take_profit: Optional[float] = Field(default=None, gt=0)
    mode: str = Field(default="DEMO", pattern="^(LIVE|DEMO)$")

    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate and normalize symbol."""
        return v.upper().strip()

    @field_validator('entry_price')
    @classmethod
    def validate_entry_price(cls, v: float) -> float:
        """Validate entry price."""
        if v <= 0:
            raise ValueError("Entry price must be positive")
        if v > 1_000_000:
            raise ValueError(f"Entry price {v} exceeds maximum allowed")
        return round(v, 4)


# =============================================================================
# Market Data Validation
# =============================================================================

class MarketDataInput(BaseModel):
    """Validates market data inputs."""

    model_config = ConfigDict(str_strip_whitespace=True)

    symbol: str = Field(..., min_length=1, max_length=10)
    price: Optional[float] = Field(default=None, gt=0)
    open: Optional[float] = Field(default=None, gt=0)
    high: Optional[float] = Field(default=None, gt=0)
    low: Optional[float] = Field(default=None, gt=0)
    close: Optional[float] = Field(default=None, gt=0)
    volume: Optional[int] = Field(default=None, ge=0)
    timestamp: Optional[datetime] = None

    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate and normalize symbol."""
        return v.upper().strip()

    @model_validator(mode='after')
    def validate_ohlc_consistency(self) -> 'MarketDataInput':
        """Validate OHLC data consistency."""
        if all([self.open, self.high, self.low, self.close]):
            if self.high < self.low:
                raise ValueError(f"High ({self.high}) cannot be less than Low ({self.low})")
            if self.high < max(self.open, self.close):
                raise ValueError(f"High ({self.high}) must be >= Open and Close")
            if self.low > min(self.open, self.close):
                raise ValueError(f"Low ({self.low}) must be <= Open and Close")
        return self


# =============================================================================
# Risk Parameter Validation
# =============================================================================

class RiskParameters(BaseModel):
    """Validates risk management parameters."""

    model_config = ConfigDict()

    max_position_size: float = Field(default=0.05, gt=0, le=1.0)
    max_portfolio_risk: float = Field(default=0.02, gt=0, le=1.0)
    max_single_loss: float = Field(default=0.01, gt=0, le=1.0)
    max_daily_loss: float = Field(default=0.05, gt=0, le=1.0)
    stop_loss_percent: float = Field(default=0.02, gt=0, le=0.5)
    take_profit_percent: float = Field(default=0.05, gt=0, le=1.0)
    max_correlation: float = Field(default=0.7, gt=0, le=1.0)
    max_leverage: float = Field(default=1.0, ge=1.0, le=10.0)

    @model_validator(mode='after')
    def validate_risk_ratios(self) -> 'RiskParameters':
        """Validate risk parameter relationships."""
        if self.take_profit_percent <= self.stop_loss_percent:
            raise ValueError(
                f"Take profit ({self.take_profit_percent}) should be greater than "
                f"stop loss ({self.stop_loss_percent}) for positive risk/reward"
            )
        if self.max_single_loss > self.max_daily_loss:
            raise ValueError(
                f"Max single loss ({self.max_single_loss}) should not exceed "
                f"max daily loss ({self.max_daily_loss})"
            )
        return self


# =============================================================================
# API Response Validation
# =============================================================================

class BrokerOrderResponse(BaseModel):
    """Validates broker order response."""

    model_config = ConfigDict(extra='allow')

    order_id: Optional[str] = None
    status: Optional[str] = None
    filled_quantity: Optional[float] = Field(default=None, ge=0)
    average_price: Optional[float] = Field(default=None, ge=0)
    error: Optional[str] = None
    error_code: Optional[str] = None

    @model_validator(mode='after')
    def validate_response(self) -> 'BrokerOrderResponse':
        """Validate response has either order_id or error."""
        if not self.order_id and not self.error:
            logger.warning("Broker response missing both order_id and error")
        return self


class QuoteResponse(BaseModel):
    """Validates quote response from data provider."""

    model_config = ConfigDict(extra='allow')

    symbol: str
    bid: Optional[float] = Field(default=None, ge=0)
    ask: Optional[float] = Field(default=None, ge=0)
    last: Optional[float] = Field(default=None, ge=0)
    volume: Optional[int] = Field(default=None, ge=0)
    timestamp: Optional[datetime] = None

    @model_validator(mode='after')
    def validate_spread(self) -> 'QuoteResponse':
        """Validate bid/ask spread."""
        if self.bid and self.ask:
            if self.ask < self.bid:
                raise ValueError(f"Ask ({self.ask}) cannot be less than bid ({self.bid})")
            spread_pct = (self.ask - self.bid) / self.bid * 100
            if spread_pct > 10:
                logger.warning(f"Wide spread detected for {self.symbol}: {spread_pct:.2f}%")
        return self


# =============================================================================
# Data Quality Tracking
# =============================================================================

class DataQualityScore(BaseModel):
    """Tracks data quality for inputs."""

    model_config = ConfigDict()

    symbol: str
    quality_level: DataQualityLevel = DataQualityLevel.UNKNOWN
    completeness: float = Field(default=1.0, ge=0, le=1)
    freshness_seconds: Optional[float] = None
    source: Optional[str] = None
    is_fallback: bool = False
    is_placeholder: bool = False
    warnings: List[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def calculate_quality(self) -> 'DataQualityScore':
        """Calculate overall quality level."""
        if self.is_placeholder:
            self.quality_level = DataQualityLevel.LOW
            self.warnings.append("Using placeholder data")
        elif self.is_fallback:
            self.quality_level = DataQualityLevel.MEDIUM
            self.warnings.append("Using fallback data source")
        elif self.completeness < 0.8:
            self.quality_level = DataQualityLevel.MEDIUM
            self.warnings.append(f"Low data completeness: {self.completeness:.1%}")
        elif self.freshness_seconds and self.freshness_seconds > 300:
            self.quality_level = DataQualityLevel.MEDIUM
            self.warnings.append(f"Stale data: {self.freshness_seconds:.0f}s old")
        else:
            self.quality_level = DataQualityLevel.HIGH

        return self


# =============================================================================
# Validation Decorators
# =============================================================================

def validate_inputs(model_class: Type[BaseModel]):
    """Decorator to validate function inputs against a Pydantic model.

    Args:
        model_class: Pydantic model class to validate against

    Usage:
        @validate_inputs(OrderRequest)
        def place_order(symbol, side, quantity, price=None):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                # Try to construct model from kwargs
                validated = model_class(**kwargs)
                # Pass validated data as kwargs
                return func(*args, **validated.model_dump())
            except ValidationError as e:
                logger.error(f"Input validation failed for {func.__name__}: {e}")
                raise ValueError(f"Invalid input: {e}") from e

        return wrapper
    return decorator


def validate_symbol(symbol: str) -> str:
    """Validate and normalize a symbol.

    Args:
        symbol: Stock symbol to validate

    Returns:
        Normalized symbol (uppercase)

    Raises:
        ValueError: If symbol is invalid
    """
    try:
        validated = SymbolValidator(symbol=symbol)
        return validated.symbol
    except ValidationError as e:
        raise ValueError(f"Invalid symbol: {e}") from e


def validate_order(
    symbol: str,
    side: str,
    quantity: float,
    order_type: str = "market",
    price: Optional[float] = None,
    stop_price: Optional[float] = None,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
) -> OrderRequest:
    """Validate order parameters.

    Args:
        symbol: Stock symbol
        side: Order side (buy/sell)
        quantity: Order quantity
        order_type: Order type (market/limit/stop)
        price: Limit price
        stop_price: Stop trigger price
        stop_loss: Stop loss price
        take_profit: Take profit price

    Returns:
        Validated OrderRequest object

    Raises:
        ValueError: If any parameter is invalid
    """
    try:
        return OrderRequest(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            price=price,
            stop_price=stop_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
    except ValidationError as e:
        raise ValueError(f"Invalid order: {e}") from e


def validate_market_data(data: Dict[str, Any]) -> MarketDataInput:
    """Validate market data dictionary.

    Args:
        data: Market data dictionary

    Returns:
        Validated MarketDataInput object

    Raises:
        ValueError: If data is invalid
    """
    try:
        return MarketDataInput(**data)
    except ValidationError as e:
        raise ValueError(f"Invalid market data: {e}") from e


def get_data_quality_score(
    symbol: str,
    data: Optional[Dict[str, Any]] = None,
    source: Optional[str] = None,
    is_fallback: bool = False,
    is_placeholder: bool = False,
    data_timestamp: Optional[datetime] = None,
) -> DataQualityScore:
    """Calculate data quality score for input data.

    Args:
        symbol: Stock symbol
        data: Optional data dictionary to score
        source: Data source name
        is_fallback: Whether this is fallback data
        is_placeholder: Whether this is placeholder data
        data_timestamp: When the data was generated

    Returns:
        DataQualityScore object
    """
    completeness = 1.0
    if data:
        total_fields = len(data)
        non_null_fields = sum(1 for v in data.values() if v is not None)
        completeness = non_null_fields / total_fields if total_fields > 0 else 0

    freshness = None
    if data_timestamp:
        freshness = (datetime.now() - data_timestamp).total_seconds()

    return DataQualityScore(
        symbol=symbol,
        completeness=completeness,
        freshness_seconds=freshness,
        source=source,
        is_fallback=is_fallback,
        is_placeholder=is_placeholder,
    )


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Enums
    'OrderSideEnum',
    'OrderTypeEnum',
    'TimeInForceEnum',
    'DataQualityLevel',
    # Models
    'SymbolValidator',
    'PriceValidator',
    'OrderRequest',
    'PositionRequest',
    'MarketDataInput',
    'RiskParameters',
    'BrokerOrderResponse',
    'QuoteResponse',
    'DataQualityScore',
    # Functions
    'validate_inputs',
    'validate_symbol',
    'validate_order',
    'validate_market_data',
    'get_data_quality_score',
    # Re-export ValidationError for convenience
    'ValidationError',
]
