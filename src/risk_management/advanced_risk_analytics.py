"""
Advanced Risk Analytics Module
==============================

Provides advanced risk analytics capabilities:
- Correlation matrix computation and monitoring
- Sector concentration tracking
- Options Greeks calculation
- Portfolio risk decomposition
- Stress testing scenarios

Usage:
    from src.risk_management.advanced_risk_analytics import (
        AdvancedRiskAnalytics,
        get_risk_analytics,
    )

    analytics = get_risk_analytics()

    # Update returns data
    analytics.update_returns("AAPL", returns_series)

    # Get correlation matrix
    corr_matrix = analytics.get_correlation_matrix()

    # Calculate Greeks for options
    greeks = analytics.calculate_greeks(
        option_type="call",
        spot_price=150.0,
        strike_price=155.0,
        time_to_expiry=30/365,
        risk_free_rate=0.05,
        volatility=0.25
    )
"""

from __future__ import annotations

import logging
import math
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger('trading.risk_analytics')


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class GreeksResult:
    """Options Greeks calculation result."""
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    option_type: str
    spot_price: float
    strike_price: float
    time_to_expiry: float
    volatility: float
    risk_free_rate: float
    option_price: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SectorExposure:
    """Sector exposure information."""
    sector: str
    exposure_percent: float
    symbols: List[str]
    total_value: float
    is_over_limit: bool
    limit_percent: float


@dataclass
class CorrelationAlert:
    """Alert for significant correlation changes."""
    symbol1: str
    symbol2: str
    old_correlation: float
    new_correlation: float
    change: float
    timestamp: datetime


@dataclass
class StressTestResult:
    """Result of a stress test scenario."""
    scenario_name: str
    portfolio_impact_percent: float
    portfolio_impact_value: float
    worst_affected_positions: List[Tuple[str, float]]
    var_under_stress: float
    timestamp: datetime


class Sector(Enum):
    """Market sectors."""
    TECHNOLOGY = "Technology"
    HEALTHCARE = "Healthcare"
    FINANCIALS = "Financials"
    CONSUMER_DISCRETIONARY = "Consumer Discretionary"
    CONSUMER_STAPLES = "Consumer Staples"
    INDUSTRIALS = "Industrials"
    ENERGY = "Energy"
    UTILITIES = "Utilities"
    MATERIALS = "Materials"
    REAL_ESTATE = "Real Estate"
    COMMUNICATION_SERVICES = "Communication Services"
    UNKNOWN = "Unknown"


# =============================================================================
# Sector Mappings
# =============================================================================

# Common stock to sector mappings (can be extended or loaded from config)
DEFAULT_SECTOR_MAPPINGS: Dict[str, Sector] = {
    # Technology
    "AAPL": Sector.TECHNOLOGY,
    "MSFT": Sector.TECHNOLOGY,
    "GOOGL": Sector.TECHNOLOGY,
    "GOOG": Sector.TECHNOLOGY,
    "META": Sector.TECHNOLOGY,
    "NVDA": Sector.TECHNOLOGY,
    "AMD": Sector.TECHNOLOGY,
    "INTC": Sector.TECHNOLOGY,
    "CRM": Sector.TECHNOLOGY,
    "ORCL": Sector.TECHNOLOGY,
    "ADBE": Sector.TECHNOLOGY,
    "CSCO": Sector.TECHNOLOGY,
    "IBM": Sector.TECHNOLOGY,
    "QCOM": Sector.TECHNOLOGY,
    "TXN": Sector.TECHNOLOGY,
    "AVGO": Sector.TECHNOLOGY,
    "NOW": Sector.TECHNOLOGY,
    "SHOP": Sector.TECHNOLOGY,
    "SQ": Sector.TECHNOLOGY,
    "PYPL": Sector.TECHNOLOGY,
    # Healthcare
    "JNJ": Sector.HEALTHCARE,
    "UNH": Sector.HEALTHCARE,
    "PFE": Sector.HEALTHCARE,
    "ABBV": Sector.HEALTHCARE,
    "MRK": Sector.HEALTHCARE,
    "LLY": Sector.HEALTHCARE,
    "TMO": Sector.HEALTHCARE,
    "ABT": Sector.HEALTHCARE,
    "DHR": Sector.HEALTHCARE,
    "BMY": Sector.HEALTHCARE,
    "AMGN": Sector.HEALTHCARE,
    "GILD": Sector.HEALTHCARE,
    "MRNA": Sector.HEALTHCARE,
    "BIIB": Sector.HEALTHCARE,
    # Financials
    "JPM": Sector.FINANCIALS,
    "BAC": Sector.FINANCIALS,
    "WFC": Sector.FINANCIALS,
    "GS": Sector.FINANCIALS,
    "MS": Sector.FINANCIALS,
    "C": Sector.FINANCIALS,
    "BLK": Sector.FINANCIALS,
    "AXP": Sector.FINANCIALS,
    "SCHW": Sector.FINANCIALS,
    "USB": Sector.FINANCIALS,
    "V": Sector.FINANCIALS,
    "MA": Sector.FINANCIALS,
    # Consumer Discretionary
    "AMZN": Sector.CONSUMER_DISCRETIONARY,
    "TSLA": Sector.CONSUMER_DISCRETIONARY,
    "HD": Sector.CONSUMER_DISCRETIONARY,
    "MCD": Sector.CONSUMER_DISCRETIONARY,
    "NKE": Sector.CONSUMER_DISCRETIONARY,
    "SBUX": Sector.CONSUMER_DISCRETIONARY,
    "LOW": Sector.CONSUMER_DISCRETIONARY,
    "TGT": Sector.CONSUMER_DISCRETIONARY,
    "BKNG": Sector.CONSUMER_DISCRETIONARY,
    "F": Sector.CONSUMER_DISCRETIONARY,
    "GM": Sector.CONSUMER_DISCRETIONARY,
    # Consumer Staples
    "PG": Sector.CONSUMER_STAPLES,
    "KO": Sector.CONSUMER_STAPLES,
    "PEP": Sector.CONSUMER_STAPLES,
    "COST": Sector.CONSUMER_STAPLES,
    "WMT": Sector.CONSUMER_STAPLES,
    "PM": Sector.CONSUMER_STAPLES,
    "MO": Sector.CONSUMER_STAPLES,
    "CL": Sector.CONSUMER_STAPLES,
    "KMB": Sector.CONSUMER_STAPLES,
    # Industrials
    "BA": Sector.INDUSTRIALS,
    "CAT": Sector.INDUSTRIALS,
    "UPS": Sector.INDUSTRIALS,
    "HON": Sector.INDUSTRIALS,
    "UNP": Sector.INDUSTRIALS,
    "RTX": Sector.INDUSTRIALS,
    "LMT": Sector.INDUSTRIALS,
    "DE": Sector.INDUSTRIALS,
    "GE": Sector.INDUSTRIALS,
    "MMM": Sector.INDUSTRIALS,
    # Energy
    "XOM": Sector.ENERGY,
    "CVX": Sector.ENERGY,
    "COP": Sector.ENERGY,
    "SLB": Sector.ENERGY,
    "EOG": Sector.ENERGY,
    "PXD": Sector.ENERGY,
    "MPC": Sector.ENERGY,
    "VLO": Sector.ENERGY,
    "OXY": Sector.ENERGY,
    # Utilities
    "NEE": Sector.UTILITIES,
    "DUK": Sector.UTILITIES,
    "SO": Sector.UTILITIES,
    "D": Sector.UTILITIES,
    "AEP": Sector.UTILITIES,
    "EXC": Sector.UTILITIES,
    # Communication Services
    "NFLX": Sector.COMMUNICATION_SERVICES,
    "DIS": Sector.COMMUNICATION_SERVICES,
    "CMCSA": Sector.COMMUNICATION_SERVICES,
    "VZ": Sector.COMMUNICATION_SERVICES,
    "T": Sector.COMMUNICATION_SERVICES,
    "TMUS": Sector.COMMUNICATION_SERVICES,
    # Real Estate
    "AMT": Sector.REAL_ESTATE,
    "PLD": Sector.REAL_ESTATE,
    "CCI": Sector.REAL_ESTATE,
    "EQIX": Sector.REAL_ESTATE,
    "PSA": Sector.REAL_ESTATE,
    "SPG": Sector.REAL_ESTATE,
    # Materials
    "LIN": Sector.MATERIALS,
    "APD": Sector.MATERIALS,
    "SHW": Sector.MATERIALS,
    "ECL": Sector.MATERIALS,
    "FCX": Sector.MATERIALS,
    "NEM": Sector.MATERIALS,
}


# =============================================================================
# Options Greeks Calculator
# =============================================================================

class GreeksCalculator:
    """Black-Scholes Greeks calculator for options."""

    @staticmethod
    def calculate_d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d1 in Black-Scholes formula."""
        if T <= 0 or sigma <= 0:
            return 0.0
        return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))

    @staticmethod
    def calculate_d2(d1: float, sigma: float, T: float) -> float:
        """Calculate d2 in Black-Scholes formula."""
        if T <= 0:
            return 0.0
        return d1 - sigma * math.sqrt(T)

    @staticmethod
    def calculate_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate call option price using Black-Scholes."""
        if T <= 0:
            return max(0, S - K)
        d1 = GreeksCalculator.calculate_d1(S, K, T, r, sigma)
        d2 = GreeksCalculator.calculate_d2(d1, sigma, T)
        return S * stats.norm.cdf(d1) - K * math.exp(-r * T) * stats.norm.cdf(d2)

    @staticmethod
    def calculate_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate put option price using Black-Scholes."""
        if T <= 0:
            return max(0, K - S)
        d1 = GreeksCalculator.calculate_d1(S, K, T, r, sigma)
        d2 = GreeksCalculator.calculate_d2(d1, sigma, T)
        return K * math.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)

    @staticmethod
    def calculate_greeks(
        option_type: str,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float
    ) -> GreeksResult:
        """
        Calculate all Greeks for an option.

        Args:
            option_type: "call" or "put"
            S: Spot price
            K: Strike price
            T: Time to expiry (in years)
            r: Risk-free rate (annual)
            sigma: Volatility (annual)

        Returns:
            GreeksResult with all Greeks
        """
        if T <= 0 or sigma <= 0:
            # At or past expiry, or invalid volatility
            option_price = max(0, S - K) if option_type.lower() == "call" else max(0, K - S)
            return GreeksResult(
                delta=1.0 if option_type.lower() == "call" and S > K else 0.0,
                gamma=0.0,
                theta=0.0,
                vega=0.0,
                rho=0.0,
                option_type=option_type,
                spot_price=S,
                strike_price=K,
                time_to_expiry=T,
                volatility=sigma,
                risk_free_rate=r,
                option_price=option_price,
            )

        d1 = GreeksCalculator.calculate_d1(S, K, T, r, sigma)
        d2 = GreeksCalculator.calculate_d2(d1, sigma, T)

        # Standard normal PDF at d1
        pdf_d1 = stats.norm.pdf(d1)
        sqrt_T = math.sqrt(T)

        if option_type.lower() == "call":
            # Call Greeks
            delta = stats.norm.cdf(d1)
            theta = (
                -(S * pdf_d1 * sigma) / (2 * sqrt_T)
                - r * K * math.exp(-r * T) * stats.norm.cdf(d2)
            ) / 365  # Daily theta
            rho = K * T * math.exp(-r * T) * stats.norm.cdf(d2) / 100  # Per 1% change
            option_price = GreeksCalculator.calculate_call_price(S, K, T, r, sigma)
        else:
            # Put Greeks
            delta = stats.norm.cdf(d1) - 1
            theta = (
                -(S * pdf_d1 * sigma) / (2 * sqrt_T)
                + r * K * math.exp(-r * T) * stats.norm.cdf(-d2)
            ) / 365  # Daily theta
            rho = -K * T * math.exp(-r * T) * stats.norm.cdf(-d2) / 100  # Per 1% change
            option_price = GreeksCalculator.calculate_put_price(S, K, T, r, sigma)

        # Gamma and Vega are same for calls and puts
        gamma = pdf_d1 / (S * sigma * sqrt_T)
        vega = S * pdf_d1 * sqrt_T / 100  # Per 1% change in volatility

        return GreeksResult(
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            rho=rho,
            option_type=option_type,
            spot_price=S,
            strike_price=K,
            time_to_expiry=T,
            volatility=sigma,
            risk_free_rate=r,
            option_price=option_price,
        )


# =============================================================================
# Advanced Risk Analytics
# =============================================================================

class AdvancedRiskAnalytics:
    """
    Advanced risk analytics for portfolio management.

    Provides:
    - Correlation matrix computation and monitoring
    - Sector concentration tracking
    - Options Greeks calculation
    - Stress testing
    """

    def __init__(
        self,
        sector_limit_percent: float = 30.0,
        correlation_change_threshold: float = 0.2,
        correlation_lookback_days: int = 60,
        sector_mappings: Optional[Dict[str, Sector]] = None,
    ):
        """
        Initialize advanced risk analytics.

        Args:
            sector_limit_percent: Maximum sector concentration (default 30%)
            correlation_change_threshold: Threshold for correlation change alerts
            correlation_lookback_days: Days of data for correlation calculation
            sector_mappings: Custom sector mappings (uses default if None)
        """
        self.sector_limit_percent = sector_limit_percent
        self.correlation_change_threshold = correlation_change_threshold
        self.correlation_lookback_days = correlation_lookback_days
        self.sector_mappings = sector_mappings or DEFAULT_SECTOR_MAPPINGS.copy()

        # Data storage
        self._returns_data: Dict[str, pd.Series] = {}
        self._correlation_matrix: Optional[pd.DataFrame] = None
        self._previous_correlation_matrix: Optional[pd.DataFrame] = None
        self._correlation_alerts: List[CorrelationAlert] = []

        # Position data
        self._positions: Dict[str, Dict[str, Any]] = {}

        # Greeks calculator
        self._greeks_calculator = GreeksCalculator()

        # Threading - use RLock for reentrant locking (needed for get_status)
        self._lock = threading.RLock()

        # Callbacks
        self._on_sector_limit_breach: Optional[Callable[[SectorExposure], None]] = None
        self._on_correlation_alert: Optional[Callable[[CorrelationAlert], None]] = None

        logger.info(
            f"Advanced Risk Analytics initialized "
            f"(sector_limit={sector_limit_percent}%, "
            f"corr_threshold={correlation_change_threshold})"
        )

    # -------------------------------------------------------------------------
    # Returns Data Management
    # -------------------------------------------------------------------------

    def update_returns(self, symbol: str, returns: pd.Series) -> None:
        """
        Update returns data for a symbol.

        Args:
            symbol: Stock symbol
            returns: Returns series (indexed by date)
        """
        with self._lock:
            self._returns_data[symbol.upper()] = returns.copy()
            # Invalidate correlation matrix
            self._previous_correlation_matrix = self._correlation_matrix
            self._correlation_matrix = None

        logger.debug(f"Updated returns for {symbol.upper()}")

    def update_returns_bulk(self, returns_dict: Dict[str, pd.Series]) -> None:
        """
        Update returns data for multiple symbols.

        Args:
            returns_dict: Dictionary of symbol -> returns series
        """
        with self._lock:
            for symbol, returns in returns_dict.items():
                self._returns_data[symbol.upper()] = returns.copy()
            self._previous_correlation_matrix = self._correlation_matrix
            self._correlation_matrix = None

        logger.info(f"Bulk updated returns for {len(returns_dict)} symbols")

    def get_symbols(self) -> List[str]:
        """Get list of symbols with returns data."""
        with self._lock:
            return list(self._returns_data.keys())

    # -------------------------------------------------------------------------
    # Correlation Matrix
    # -------------------------------------------------------------------------

    def compute_correlation_matrix(self) -> pd.DataFrame:
        """
        Compute correlation matrix from returns data.

        Returns:
            Correlation matrix as DataFrame
        """
        with self._lock:
            if self._correlation_matrix is not None:
                return self._correlation_matrix.copy()

            if len(self._returns_data) < 2:
                return pd.DataFrame()

            # Create DataFrame from returns
            returns_df = pd.DataFrame(self._returns_data)

            # Use last N days
            if len(returns_df) > self.correlation_lookback_days:
                returns_df = returns_df.tail(self.correlation_lookback_days)

            # Compute correlation
            self._correlation_matrix = returns_df.corr()

            # Check for significant changes
            if self._previous_correlation_matrix is not None:
                self._check_correlation_changes()

            return self._correlation_matrix.copy()

    def get_correlation_matrix(self) -> pd.DataFrame:
        """Get current correlation matrix (computes if needed)."""
        return self.compute_correlation_matrix()

    def get_correlation(self, symbol1: str, symbol2: str) -> Optional[float]:
        """
        Get correlation between two symbols.

        Args:
            symbol1: First symbol
            symbol2: Second symbol

        Returns:
            Correlation coefficient or None if not available
        """
        corr_matrix = self.compute_correlation_matrix()
        if corr_matrix.empty:
            return None

        s1, s2 = symbol1.upper(), symbol2.upper()
        if s1 in corr_matrix.columns and s2 in corr_matrix.columns:
            return corr_matrix.loc[s1, s2]
        return None

    def get_highest_correlations(self, symbol: str, n: int = 5) -> List[Tuple[str, float]]:
        """
        Get symbols with highest correlation to given symbol.

        Args:
            symbol: Target symbol
            n: Number of results

        Returns:
            List of (symbol, correlation) tuples
        """
        corr_matrix = self.compute_correlation_matrix()
        if corr_matrix.empty or symbol.upper() not in corr_matrix.columns:
            return []

        correlations = corr_matrix[symbol.upper()].drop(symbol.upper())
        top_n = correlations.abs().nlargest(n)

        return [(sym, correlations[sym]) for sym in top_n.index]

    def _check_correlation_changes(self) -> None:
        """Check for significant correlation changes and generate alerts."""
        if self._previous_correlation_matrix is None or self._correlation_matrix is None:
            return

        common_symbols = set(self._previous_correlation_matrix.columns) & set(
            self._correlation_matrix.columns
        )

        for i, s1 in enumerate(common_symbols):
            for s2 in list(common_symbols)[i + 1:]:
                old_corr = self._previous_correlation_matrix.loc[s1, s2]
                new_corr = self._correlation_matrix.loc[s1, s2]
                change = abs(new_corr - old_corr)

                if change >= self.correlation_change_threshold:
                    alert = CorrelationAlert(
                        symbol1=s1,
                        symbol2=s2,
                        old_correlation=old_corr,
                        new_correlation=new_corr,
                        change=change,
                        timestamp=datetime.now(),
                    )
                    self._correlation_alerts.append(alert)
                    logger.warning(
                        f"Correlation change alert: {s1}-{s2} "
                        f"changed by {change:.3f} ({old_corr:.3f} -> {new_corr:.3f})"
                    )
                    if self._on_correlation_alert:
                        self._on_correlation_alert(alert)

    def get_correlation_alerts(
        self, since: Optional[datetime] = None
    ) -> List[CorrelationAlert]:
        """Get correlation alerts, optionally filtered by time."""
        with self._lock:
            if since is None:
                return list(self._correlation_alerts)
            return [a for a in self._correlation_alerts if a.timestamp >= since]

    # -------------------------------------------------------------------------
    # Sector Concentration
    # -------------------------------------------------------------------------

    def update_position(
        self,
        symbol: str,
        quantity: float,
        market_value: float,
        sector: Optional[Sector] = None,
    ) -> None:
        """
        Update position data for sector tracking.

        Args:
            symbol: Stock symbol
            quantity: Number of shares
            market_value: Current market value
            sector: Sector override (uses mapping if None)
        """
        symbol = symbol.upper()
        resolved_sector = sector or self.sector_mappings.get(symbol, Sector.UNKNOWN)

        with self._lock:
            self._positions[symbol] = {
                "quantity": quantity,
                "market_value": market_value,
                "sector": resolved_sector,
                "updated_at": datetime.now(),
            }

    def update_positions_bulk(
        self, positions: Dict[str, Dict[str, Any]]
    ) -> None:
        """
        Update multiple positions at once.

        Args:
            positions: Dict of symbol -> {quantity, market_value, sector (optional)}
        """
        with self._lock:
            for symbol, data in positions.items():
                symbol = symbol.upper()
                sector = data.get("sector") or self.sector_mappings.get(
                    symbol, Sector.UNKNOWN
                )
                self._positions[symbol] = {
                    "quantity": data["quantity"],
                    "market_value": data["market_value"],
                    "sector": sector,
                    "updated_at": datetime.now(),
                }

    def get_sector_exposure(self) -> Dict[str, SectorExposure]:
        """
        Calculate sector exposure percentages.

        Returns:
            Dictionary of sector -> SectorExposure
        """
        with self._lock:
            if not self._positions:
                return {}

            total_value = sum(p["market_value"] for p in self._positions.values())
            if total_value <= 0:
                return {}

            # Group by sector
            sector_data: Dict[Sector, Dict] = defaultdict(
                lambda: {"value": 0.0, "symbols": []}
            )

            for symbol, pos in self._positions.items():
                sector = pos["sector"]
                sector_data[sector]["value"] += pos["market_value"]
                sector_data[sector]["symbols"].append(symbol)

            # Create exposure objects
            exposures = {}
            for sector, data in sector_data.items():
                exposure_pct = (data["value"] / total_value) * 100
                is_over = exposure_pct > self.sector_limit_percent

                exposure = SectorExposure(
                    sector=sector.value,
                    exposure_percent=exposure_pct,
                    symbols=data["symbols"],
                    total_value=data["value"],
                    is_over_limit=is_over,
                    limit_percent=self.sector_limit_percent,
                )
                exposures[sector.value] = exposure

                if is_over and self._on_sector_limit_breach:
                    self._on_sector_limit_breach(exposure)

            return exposures

    def check_sector_limits(self) -> List[SectorExposure]:
        """
        Check which sectors are over concentration limits.

        Returns:
            List of SectorExposure objects that are over limit
        """
        exposures = self.get_sector_exposure()
        return [e for e in exposures.values() if e.is_over_limit]

    def add_sector_mapping(self, symbol: str, sector: Sector) -> None:
        """Add or update sector mapping for a symbol."""
        self.sector_mappings[symbol.upper()] = sector

    def get_sector_for_symbol(self, symbol: str) -> Sector:
        """Get sector for a symbol."""
        return self.sector_mappings.get(symbol.upper(), Sector.UNKNOWN)

    # -------------------------------------------------------------------------
    # Options Greeks
    # -------------------------------------------------------------------------

    def calculate_greeks(
        self,
        option_type: str,
        spot_price: float,
        strike_price: float,
        time_to_expiry: float,
        risk_free_rate: float,
        volatility: float,
    ) -> GreeksResult:
        """
        Calculate Greeks for an option.

        Args:
            option_type: "call" or "put"
            spot_price: Current price of underlying
            strike_price: Option strike price
            time_to_expiry: Time to expiry in years (e.g., 30/365 for 30 days)
            risk_free_rate: Risk-free interest rate (annual)
            volatility: Implied volatility (annual)

        Returns:
            GreeksResult with all Greeks
        """
        return GreeksCalculator.calculate_greeks(
            option_type=option_type,
            S=spot_price,
            K=strike_price,
            T=time_to_expiry,
            r=risk_free_rate,
            sigma=volatility,
        )

    def calculate_portfolio_greeks(
        self,
        options_positions: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Calculate aggregate Greeks for options portfolio.

        Args:
            options_positions: List of dicts with keys:
                - option_type: "call" or "put"
                - spot_price: Current price
                - strike_price: Strike
                - time_to_expiry: Years to expiry
                - risk_free_rate: Rate
                - volatility: IV
                - quantity: Number of contracts (positive=long, negative=short)

        Returns:
            Dictionary with aggregate Greeks
        """
        total_delta = 0.0
        total_gamma = 0.0
        total_theta = 0.0
        total_vega = 0.0
        total_rho = 0.0

        for pos in options_positions:
            greeks = self.calculate_greeks(
                option_type=pos["option_type"],
                spot_price=pos["spot_price"],
                strike_price=pos["strike_price"],
                time_to_expiry=pos["time_to_expiry"],
                risk_free_rate=pos["risk_free_rate"],
                volatility=pos["volatility"],
            )
            qty = pos.get("quantity", 1) * 100  # Standard contract = 100 shares

            total_delta += greeks.delta * qty
            total_gamma += greeks.gamma * qty
            total_theta += greeks.theta * qty
            total_vega += greeks.vega * qty
            total_rho += greeks.rho * qty

        return {
            "delta": total_delta,
            "gamma": total_gamma,
            "theta": total_theta,
            "vega": total_vega,
            "rho": total_rho,
        }

    # -------------------------------------------------------------------------
    # Stress Testing
    # -------------------------------------------------------------------------

    def run_stress_test(
        self,
        scenario_name: str,
        price_shocks: Dict[str, float],
        portfolio_value: float,
    ) -> StressTestResult:
        """
        Run a stress test scenario on the portfolio.

        Args:
            scenario_name: Name of the scenario
            price_shocks: Dict of symbol -> percent change (e.g., -0.20 for -20%)
            portfolio_value: Current portfolio value

        Returns:
            StressTestResult with impact analysis
        """
        with self._lock:
            if not self._positions:
                return StressTestResult(
                    scenario_name=scenario_name,
                    portfolio_impact_percent=0.0,
                    portfolio_impact_value=0.0,
                    worst_affected_positions=[],
                    var_under_stress=0.0,
                    timestamp=datetime.now(),
                )

            total_impact = 0.0
            position_impacts: List[Tuple[str, float]] = []

            for symbol, pos in self._positions.items():
                shock = price_shocks.get(symbol, price_shocks.get("DEFAULT", 0.0))
                impact = pos["market_value"] * shock
                total_impact += impact
                position_impacts.append((symbol, impact))

            # Sort by absolute impact
            position_impacts.sort(key=lambda x: abs(x[1]), reverse=True)

            impact_percent = (total_impact / portfolio_value) * 100 if portfolio_value > 0 else 0

            return StressTestResult(
                scenario_name=scenario_name,
                portfolio_impact_percent=impact_percent,
                portfolio_impact_value=total_impact,
                worst_affected_positions=position_impacts[:5],
                var_under_stress=abs(impact_percent),
                timestamp=datetime.now(),
            )

    def run_standard_scenarios(
        self, portfolio_value: float
    ) -> Dict[str, StressTestResult]:
        """
        Run standard stress test scenarios.

        Args:
            portfolio_value: Current portfolio value

        Returns:
            Dictionary of scenario name -> StressTestResult
        """
        scenarios = {
            "market_crash_10": {"DEFAULT": -0.10},
            "market_crash_20": {"DEFAULT": -0.20},
            "market_rally_10": {"DEFAULT": 0.10},
            "tech_selloff": {
                symbol: -0.25
                for symbol, sector in self.sector_mappings.items()
                if sector == Sector.TECHNOLOGY
            },
            "financials_crisis": {
                symbol: -0.30
                for symbol, sector in self.sector_mappings.items()
                if sector == Sector.FINANCIALS
            },
            "energy_shock": {
                symbol: -0.35
                for symbol, sector in self.sector_mappings.items()
                if sector == Sector.ENERGY
            },
        }

        # Add DEFAULT for sector scenarios
        for scenario in ["tech_selloff", "financials_crisis", "energy_shock"]:
            scenarios[scenario]["DEFAULT"] = 0.0

        results = {}
        for name, shocks in scenarios.items():
            results[name] = self.run_stress_test(name, shocks, portfolio_value)

        return results

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def set_callbacks(
        self,
        on_sector_limit_breach: Optional[Callable[[SectorExposure], None]] = None,
        on_correlation_alert: Optional[Callable[[CorrelationAlert], None]] = None,
    ) -> None:
        """Set callback functions for alerts."""
        self._on_sector_limit_breach = on_sector_limit_breach
        self._on_correlation_alert = on_correlation_alert

    # -------------------------------------------------------------------------
    # Status and Reset
    # -------------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Get current status of risk analytics."""
        with self._lock:
            return {
                "symbols_tracked": len(self._returns_data),
                "positions_tracked": len(self._positions),
                "correlation_matrix_computed": self._correlation_matrix is not None,
                "correlation_alerts_count": len(self._correlation_alerts),
                "sector_limit_percent": self.sector_limit_percent,
                "sectors_over_limit": len(self.check_sector_limits()),
            }

    def reset(self) -> None:
        """Reset all tracked data."""
        with self._lock:
            self._returns_data.clear()
            self._positions.clear()
            self._correlation_matrix = None
            self._previous_correlation_matrix = None
            self._correlation_alerts.clear()

        logger.info("Advanced Risk Analytics reset")


# =============================================================================
# Global Instance
# =============================================================================

_analytics_instance: Optional[AdvancedRiskAnalytics] = None


def get_risk_analytics(
    sector_limit_percent: float = 30.0,
    correlation_change_threshold: float = 0.2,
) -> AdvancedRiskAnalytics:
    """Get global risk analytics instance."""
    global _analytics_instance
    if _analytics_instance is None:
        _analytics_instance = AdvancedRiskAnalytics(
            sector_limit_percent=sector_limit_percent,
            correlation_change_threshold=correlation_change_threshold,
        )
    return _analytics_instance


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Data classes
    "GreeksResult",
    "SectorExposure",
    "CorrelationAlert",
    "StressTestResult",
    # Enums
    "Sector",
    # Classes
    "GreeksCalculator",
    "AdvancedRiskAnalytics",
    # Functions
    "get_risk_analytics",
    # Constants
    "DEFAULT_SECTOR_MAPPINGS",
]
