"""
Portfolio Risk Management Module
================================

Provides portfolio-level risk management capabilities:
- Sector concentration limits enforcement
- Position concentration limits
- Overall portfolio exposure tracking
- Risk limit alerts and notifications
- Portfolio diversification scoring

Usage:
    from src.risk_management.portfolio_risk import (
        PortfolioRiskManager,
        get_portfolio_risk_manager,
    )

    manager = get_portfolio_risk_manager()

    # Update positions
    manager.update_portfolio({
        "AAPL": {"quantity": 100, "market_value": 15000},
        "MSFT": {"quantity": 50, "market_value": 17500},
    })

    # Check sector limits
    violations = manager.check_sector_limits()

    # Get portfolio risk report
    report = manager.get_risk_report()
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.risk_management.advanced_risk_analytics import (
    AdvancedRiskAnalytics,
    Sector,
    SectorExposure,
    DEFAULT_SECTOR_MAPPINGS,
    get_risk_analytics,
)

logger = logging.getLogger('trading.portfolio_risk')


# =============================================================================
# Data Classes
# =============================================================================

class RiskLevel(Enum):
    """Risk level classifications."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PositionRisk:
    """Risk information for a single position."""
    symbol: str
    sector: Sector
    market_value: float
    portfolio_percent: float
    sector_contribution: float
    is_over_position_limit: bool
    is_contributing_to_sector_breach: bool
    risk_level: RiskLevel


@dataclass
class ConcentrationAlert:
    """Alert for concentration limit violations."""
    alert_type: str  # "sector" or "position"
    identifier: str  # sector name or symbol
    current_percent: float
    limit_percent: float
    excess_percent: float
    risk_level: RiskLevel
    recommended_action: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PortfolioRiskReport:
    """Comprehensive portfolio risk report."""
    total_value: float
    position_count: int
    sector_count: int
    sectors_over_limit: List[SectorExposure]
    positions_over_limit: List[PositionRisk]
    concentration_alerts: List[ConcentrationAlert]
    diversification_score: float
    overall_risk_level: RiskLevel
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SectorLimitConfig:
    """Configuration for sector concentration limits."""
    default_limit_percent: float = 30.0
    warning_threshold_percent: float = 25.0
    custom_limits: Dict[str, float] = field(default_factory=dict)


@dataclass
class PositionLimitConfig:
    """Configuration for position concentration limits."""
    default_limit_percent: float = 10.0
    warning_threshold_percent: float = 8.0
    custom_limits: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# Portfolio Risk Manager
# =============================================================================

class PortfolioRiskManager:
    """Manages portfolio-level risk including sector concentration.

    Features:
    - Sector concentration limits with configurable thresholds
    - Position concentration limits
    - Real-time risk monitoring
    - Alert generation for limit violations
    - Diversification scoring
    """

    def __init__(
        self,
        sector_limit_config: Optional[SectorLimitConfig] = None,
        position_limit_config: Optional[PositionLimitConfig] = None,
        risk_analytics: Optional[AdvancedRiskAnalytics] = None,
    ):
        """Initialize the portfolio risk manager.

        Args:
            sector_limit_config: Sector concentration limits configuration
            position_limit_config: Position concentration limits configuration
            risk_analytics: Risk analytics instance (uses global if None)
        """
        self.sector_config = sector_limit_config or SectorLimitConfig()
        self.position_config = position_limit_config or PositionLimitConfig()

        self._risk_analytics = risk_analytics or get_risk_analytics()
        self._lock = threading.RLock()  # RLock allows recursive locking

        # Position tracking
        self._positions: Dict[str, Dict[str, Any]] = {}
        self._total_portfolio_value: float = 0.0

        # Alert callbacks
        self._on_sector_alert: Optional[Callable[[ConcentrationAlert], None]] = None
        self._on_position_alert: Optional[Callable[[ConcentrationAlert], None]] = None
        self._on_critical_alert: Optional[Callable[[ConcentrationAlert], None]] = None

        # Alert history
        self._alert_history: List[ConcentrationAlert] = []
        self._max_alert_history = 1000

        logger.info(
            f"PortfolioRiskManager initialized "
            f"(sector_limit={self.sector_config.default_limit_percent}%, "
            f"position_limit={self.position_config.default_limit_percent}%)"
        )

    # =========================================================================
    # Portfolio Updates
    # =========================================================================

    def update_portfolio(self, positions: Dict[str, Dict[str, Any]]) -> None:
        """Update portfolio positions.

        Args:
            positions: Dict of symbol -> {quantity, market_value, sector (optional)}
        """
        with self._lock:
            self._positions = {}
            self._total_portfolio_value = 0.0

            for symbol, data in positions.items():
                market_value = data.get("market_value", 0)
                quantity = data.get("quantity", 0)
                sector = data.get("sector") or self._get_sector_for_symbol(symbol)

                self._positions[symbol.upper()] = {
                    "quantity": quantity,
                    "market_value": market_value,
                    "sector": sector,
                }
                self._total_portfolio_value += market_value

            # Also update the risk analytics
            self._risk_analytics.update_positions_bulk(positions)

            logger.debug(
                f"Portfolio updated: {len(self._positions)} positions, "
                f"${self._total_portfolio_value:,.2f} total value"
            )

    def update_position(
        self,
        symbol: str,
        quantity: float,
        market_value: float,
        sector: Optional[Sector] = None,
    ) -> None:
        """Update a single position.

        Args:
            symbol: Stock symbol
            quantity: Number of shares
            market_value: Total market value
            sector: Sector override
        """
        with self._lock:
            symbol = symbol.upper()
            resolved_sector = sector or self._get_sector_for_symbol(symbol)

            # Remove old value from total
            if symbol in self._positions:
                self._total_portfolio_value -= self._positions[symbol]["market_value"]

            # Update position
            self._positions[symbol] = {
                "quantity": quantity,
                "market_value": market_value,
                "sector": resolved_sector,
            }
            self._total_portfolio_value += market_value

            # Update risk analytics
            self._risk_analytics.update_position(symbol, quantity, market_value, resolved_sector)

    def remove_position(self, symbol: str) -> None:
        """Remove a position from the portfolio.

        Args:
            symbol: Stock symbol to remove
        """
        with self._lock:
            symbol = symbol.upper()
            if symbol in self._positions:
                self._total_portfolio_value -= self._positions[symbol]["market_value"]
                del self._positions[symbol]

    # =========================================================================
    # Sector Concentration
    # =========================================================================

    def get_sector_exposure(self) -> Dict[str, SectorExposure]:
        """Get current sector exposure.

        Returns:
            Dictionary of sector -> SectorExposure
        """
        with self._lock:
            if self._total_portfolio_value <= 0:
                return {}

            sector_data: Dict[Sector, Dict] = defaultdict(
                lambda: {"value": 0.0, "symbols": []}
            )

            for symbol, pos in self._positions.items():
                sector = pos["sector"]
                sector_data[sector]["value"] += pos["market_value"]
                sector_data[sector]["symbols"].append(symbol)

            exposures = {}
            for sector, data in sector_data.items():
                exposure_pct = (data["value"] / self._total_portfolio_value) * 100
                limit = self._get_sector_limit(sector.value)
                is_over = exposure_pct > limit

                exposure = SectorExposure(
                    sector=sector.value,
                    exposure_percent=exposure_pct,
                    symbols=data["symbols"],
                    total_value=data["value"],
                    is_over_limit=is_over,
                    limit_percent=limit,
                )
                exposures[sector.value] = exposure

            return exposures

    def check_sector_limits(self) -> List[SectorExposure]:
        """Check which sectors are over concentration limits.

        Returns:
            List of SectorExposure objects that are over limit
        """
        exposures = self.get_sector_exposure()
        violations = [exp for exp in exposures.values() if exp.is_over_limit]

        for violation in violations:
            self._generate_sector_alert(violation)

        return violations

    def get_sector_limit(self, sector: str) -> float:
        """Get the concentration limit for a sector.

        Args:
            sector: Sector name

        Returns:
            Limit percentage
        """
        return self._get_sector_limit(sector)

    def set_sector_limit(self, sector: str, limit_percent: float) -> None:
        """Set a custom sector concentration limit.

        Args:
            sector: Sector name
            limit_percent: Limit percentage (0-100)
        """
        if not 0 <= limit_percent <= 100:
            raise ValueError("Limit must be between 0 and 100")

        self.sector_config.custom_limits[sector] = limit_percent
        logger.info(f"Set sector limit for {sector}: {limit_percent}%")

    def _get_sector_limit(self, sector: str) -> float:
        """Get sector limit with custom override."""
        return self.sector_config.custom_limits.get(
            sector, self.sector_config.default_limit_percent
        )

    def _get_sector_for_symbol(self, symbol: str) -> Sector:
        """Get sector for a symbol."""
        return DEFAULT_SECTOR_MAPPINGS.get(symbol.upper(), Sector.UNKNOWN)

    # =========================================================================
    # Position Concentration
    # =========================================================================

    def get_position_exposures(self) -> List[PositionRisk]:
        """Get exposure information for all positions.

        Returns:
            List of PositionRisk objects
        """
        with self._lock:
            if self._total_portfolio_value <= 0:
                return []

            exposures = []
            sector_exposures = self.get_sector_exposure()

            for symbol, pos in self._positions.items():
                portfolio_pct = (pos["market_value"] / self._total_portfolio_value) * 100
                position_limit = self._get_position_limit(symbol)
                sector = pos["sector"]

                # Check if sector is over limit
                sector_exp = sector_exposures.get(sector.value)
                sector_contribution = 0.0
                is_sector_breach = False

                if sector_exp:
                    sector_contribution = (pos["market_value"] / sector_exp.total_value) * 100
                    is_sector_breach = sector_exp.is_over_limit

                risk_level = self._calculate_position_risk_level(
                    portfolio_pct, position_limit, is_sector_breach
                )

                exposures.append(PositionRisk(
                    symbol=symbol,
                    sector=sector,
                    market_value=pos["market_value"],
                    portfolio_percent=portfolio_pct,
                    sector_contribution=sector_contribution,
                    is_over_position_limit=portfolio_pct > position_limit,
                    is_contributing_to_sector_breach=is_sector_breach,
                    risk_level=risk_level,
                ))

            return sorted(exposures, key=lambda x: x.portfolio_percent, reverse=True)

    def check_position_limits(self) -> List[PositionRisk]:
        """Check which positions are over concentration limits.

        Returns:
            List of PositionRisk objects that are over limit
        """
        exposures = self.get_position_exposures()
        violations = [exp for exp in exposures if exp.is_over_position_limit]

        for violation in violations:
            self._generate_position_alert(violation)

        return violations

    def get_position_limit(self, symbol: str) -> float:
        """Get the concentration limit for a position.

        Args:
            symbol: Stock symbol

        Returns:
            Limit percentage
        """
        return self._get_position_limit(symbol)

    def set_position_limit(self, symbol: str, limit_percent: float) -> None:
        """Set a custom position concentration limit.

        Args:
            symbol: Stock symbol
            limit_percent: Limit percentage (0-100)
        """
        if not 0 <= limit_percent <= 100:
            raise ValueError("Limit must be between 0 and 100")

        self.position_config.custom_limits[symbol.upper()] = limit_percent
        logger.info(f"Set position limit for {symbol}: {limit_percent}%")

    def _get_position_limit(self, symbol: str) -> float:
        """Get position limit with custom override."""
        return self.position_config.custom_limits.get(
            symbol.upper(), self.position_config.default_limit_percent
        )

    def _calculate_position_risk_level(
        self,
        portfolio_pct: float,
        limit_pct: float,
        sector_breach: bool,
    ) -> RiskLevel:
        """Calculate risk level for a position."""
        if portfolio_pct > limit_pct:
            return RiskLevel.CRITICAL if sector_breach else RiskLevel.HIGH

        warning = self.position_config.warning_threshold_percent
        if portfolio_pct > warning:
            return RiskLevel.MEDIUM

        return RiskLevel.LOW

    # =========================================================================
    # Diversification
    # =========================================================================

    def calculate_diversification_score(self) -> float:
        """Calculate portfolio diversification score.

        Returns a score from 0-100:
        - Higher is more diversified
        - Penalizes sector concentration
        - Penalizes position concentration
        - Rewards having positions across multiple sectors

        Returns:
            Diversification score (0-100)
        """
        with self._lock:
            if not self._positions or self._total_portfolio_value <= 0:
                return 0.0

            sector_exposures = self.get_sector_exposure()
            position_exposures = self.get_position_exposures()

            # Start with perfect score
            score = 100.0

            # Penalize sector concentration (up to 30 points)
            sector_penalty = 0.0
            for sector_exp in sector_exposures.values():
                if sector_exp.exposure_percent > sector_exp.limit_percent:
                    excess = sector_exp.exposure_percent - sector_exp.limit_percent
                    sector_penalty += min(excess * 0.5, 15)  # Max 15 per sector

            score -= min(sector_penalty, 30)

            # Penalize position concentration (up to 30 points)
            position_penalty = 0.0
            for pos in position_exposures:
                limit = self._get_position_limit(pos.symbol)
                if pos.portfolio_percent > limit:
                    excess = pos.portfolio_percent - limit
                    position_penalty += min(excess * 0.3, 10)  # Max 10 per position

            score -= min(position_penalty, 30)

            # Reward sector diversity (up to 20 points)
            num_sectors = len([s for s in sector_exposures.values() if s.exposure_percent > 5])
            sector_diversity_bonus = min(num_sectors * 2.5, 20)
            score = min(score + sector_diversity_bonus, 100)

            # Reward position count (up to 10 points)
            position_bonus = min(len(self._positions) * 0.5, 10)
            score = min(score + position_bonus, 100)

            # Penalize if very few positions
            if len(self._positions) < 5:
                score -= (5 - len(self._positions)) * 5

            return max(0.0, min(100.0, score))

    # =========================================================================
    # Alerts
    # =========================================================================

    def _generate_sector_alert(self, exposure: SectorExposure) -> None:
        """Generate alert for sector concentration violation."""
        excess = exposure.exposure_percent - exposure.limit_percent
        risk_level = RiskLevel.CRITICAL if excess > 10 else RiskLevel.HIGH

        # Calculate reduction needed
        if self._total_portfolio_value > 0:
            target_value = (exposure.limit_percent / 100) * self._total_portfolio_value
            reduction_needed = exposure.total_value - target_value
            reduction_pct = (reduction_needed / exposure.total_value) * 100

            action = (
                f"Reduce {exposure.sector} sector by ${reduction_needed:,.2f} "
                f"({reduction_pct:.1f}%) to meet {exposure.limit_percent}% limit"
            )
        else:
            action = f"Reduce {exposure.sector} sector exposure"

        alert = ConcentrationAlert(
            alert_type="sector",
            identifier=exposure.sector,
            current_percent=exposure.exposure_percent,
            limit_percent=exposure.limit_percent,
            excess_percent=excess,
            risk_level=risk_level,
            recommended_action=action,
        )

        self._add_alert(alert)

        if self._on_sector_alert:
            self._on_sector_alert(alert)

        if risk_level == RiskLevel.CRITICAL and self._on_critical_alert:
            self._on_critical_alert(alert)

    def _generate_position_alert(self, position: PositionRisk) -> None:
        """Generate alert for position concentration violation."""
        limit = self._get_position_limit(position.symbol)
        excess = position.portfolio_percent - limit

        action = (
            f"Reduce {position.symbol} position by "
            f"{excess:.1f}% of portfolio value"
        )

        alert = ConcentrationAlert(
            alert_type="position",
            identifier=position.symbol,
            current_percent=position.portfolio_percent,
            limit_percent=limit,
            excess_percent=excess,
            risk_level=position.risk_level,
            recommended_action=action,
        )

        self._add_alert(alert)

        if self._on_position_alert:
            self._on_position_alert(alert)

        if position.risk_level == RiskLevel.CRITICAL and self._on_critical_alert:
            self._on_critical_alert(alert)

    def _add_alert(self, alert: ConcentrationAlert) -> None:
        """Add alert to history."""
        self._alert_history.append(alert)
        if len(self._alert_history) > self._max_alert_history:
            self._alert_history = self._alert_history[-self._max_alert_history:]

        logger.warning(
            f"Concentration alert: {alert.alert_type} {alert.identifier} "
            f"at {alert.current_percent:.1f}% (limit: {alert.limit_percent}%)"
        )

    def set_alert_callbacks(
        self,
        on_sector_alert: Optional[Callable[[ConcentrationAlert], None]] = None,
        on_position_alert: Optional[Callable[[ConcentrationAlert], None]] = None,
        on_critical_alert: Optional[Callable[[ConcentrationAlert], None]] = None,
    ) -> None:
        """Set callbacks for concentration alerts.

        Args:
            on_sector_alert: Called when sector limit is breached
            on_position_alert: Called when position limit is breached
            on_critical_alert: Called for any critical-level alert
        """
        self._on_sector_alert = on_sector_alert
        self._on_position_alert = on_position_alert
        self._on_critical_alert = on_critical_alert

    def get_alert_history(
        self,
        limit: int = 100,
        alert_type: Optional[str] = None,
    ) -> List[ConcentrationAlert]:
        """Get recent alerts.

        Args:
            limit: Maximum number of alerts to return
            alert_type: Filter by type ("sector" or "position")

        Returns:
            List of recent alerts
        """
        alerts = self._alert_history[-limit:]

        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]

        return alerts

    # =========================================================================
    # Risk Report
    # =========================================================================

    def get_risk_report(self) -> PortfolioRiskReport:
        """Generate comprehensive portfolio risk report.

        Returns:
            PortfolioRiskReport with current risk state
        """
        with self._lock:
            sector_violations = self.check_sector_limits()
            position_violations = self.check_position_limits()

            # Calculate overall risk level
            overall_risk = RiskLevel.LOW

            if sector_violations or position_violations:
                overall_risk = RiskLevel.MEDIUM

                # Check for critical
                critical_sectors = [s for s in sector_violations if s.exposure_percent > s.limit_percent + 10]
                critical_positions = [p for p in position_violations if p.risk_level == RiskLevel.CRITICAL]

                if critical_sectors or critical_positions:
                    overall_risk = RiskLevel.CRITICAL
                elif sector_violations or position_violations:
                    overall_risk = RiskLevel.HIGH

            return PortfolioRiskReport(
                total_value=self._total_portfolio_value,
                position_count=len(self._positions),
                sector_count=len(set(p["sector"] for p in self._positions.values())),
                sectors_over_limit=sector_violations,
                positions_over_limit=position_violations,
                concentration_alerts=self.get_alert_history(limit=50),
                diversification_score=self.calculate_diversification_score(),
                overall_risk_level=overall_risk,
            )

    def get_status(self) -> Dict[str, Any]:
        """Get portfolio risk status summary.

        Returns:
            Dictionary with status information
        """
        report = self.get_risk_report()

        return {
            "total_value": report.total_value,
            "position_count": report.position_count,
            "sector_count": report.sector_count,
            "sectors_over_limit": len(report.sectors_over_limit),
            "positions_over_limit": len(report.positions_over_limit),
            "diversification_score": report.diversification_score,
            "overall_risk_level": report.overall_risk_level.value,
            "sector_limit_default": self.sector_config.default_limit_percent,
            "position_limit_default": self.position_config.default_limit_percent,
        }

    # =========================================================================
    # Validation
    # =========================================================================

    def validate_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
    ) -> Tuple[bool, Optional[str]]:
        """Validate if a trade would violate concentration limits.

        Args:
            symbol: Stock symbol
            side: "buy" or "sell"
            quantity: Number of shares
            price: Price per share

        Returns:
            Tuple of (is_valid, reason if invalid)
        """
        trade_value = quantity * price
        symbol = symbol.upper()

        with self._lock:
            if side.lower() == "sell":
                return True, None  # Selling always OK for concentration

            # Calculate new portfolio value
            new_total = self._total_portfolio_value + trade_value

            # Check position concentration
            current_value = self._positions.get(symbol, {}).get("market_value", 0)
            new_position_value = current_value + trade_value
            new_position_pct = (new_position_value / new_total) * 100

            position_limit = self._get_position_limit(symbol)
            if new_position_pct > position_limit:
                return False, (
                    f"Trade would put {symbol} at {new_position_pct:.1f}% "
                    f"of portfolio (limit: {position_limit}%)"
                )

            # Check sector concentration
            sector = self._get_sector_for_symbol(symbol)
            sector_exposures = self.get_sector_exposure()
            sector_exp = sector_exposures.get(sector.value)

            if sector_exp:
                new_sector_value = sector_exp.total_value + trade_value
            else:
                new_sector_value = trade_value

            new_sector_pct = (new_sector_value / new_total) * 100
            sector_limit = self._get_sector_limit(sector.value)

            if new_sector_pct > sector_limit:
                return False, (
                    f"Trade would put {sector.value} sector at {new_sector_pct:.1f}% "
                    f"of portfolio (limit: {sector_limit}%)"
                )

            return True, None


# =============================================================================
# Global Instance
# =============================================================================

_portfolio_risk_manager: Optional[PortfolioRiskManager] = None
_portfolio_risk_lock = threading.Lock()


def get_portfolio_risk_manager(
    sector_limit_config: Optional[SectorLimitConfig] = None,
    position_limit_config: Optional[PositionLimitConfig] = None,
) -> PortfolioRiskManager:
    """Get or create the global portfolio risk manager.

    Args:
        sector_limit_config: Sector limit configuration
        position_limit_config: Position limit configuration

    Returns:
        PortfolioRiskManager instance
    """
    global _portfolio_risk_manager

    with _portfolio_risk_lock:
        if _portfolio_risk_manager is None:
            _portfolio_risk_manager = PortfolioRiskManager(
                sector_limit_config=sector_limit_config,
                position_limit_config=position_limit_config,
            )

        return _portfolio_risk_manager


def reset_portfolio_risk_manager() -> None:
    """Reset the global portfolio risk manager."""
    global _portfolio_risk_manager

    with _portfolio_risk_lock:
        _portfolio_risk_manager = None


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Data classes
    "RiskLevel",
    "PositionRisk",
    "ConcentrationAlert",
    "PortfolioRiskReport",
    "SectorLimitConfig",
    "PositionLimitConfig",
    # Manager
    "PortfolioRiskManager",
    # Factory
    "get_portfolio_risk_manager",
    "reset_portfolio_risk_manager",
]
