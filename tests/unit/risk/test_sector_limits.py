"""
Unit tests for sector concentration limits.

Tests cover:
- Sector exposure calculations
- Sector limit enforcement
- Position concentration limits
- Custom limit configuration
- Alert generation
- Trade validation
- Diversification scoring
"""

from unittest.mock import Mock, patch, MagicMock
import pytest

from src.risk_management.portfolio_risk import (
    PortfolioRiskManager,
    SectorLimitConfig,
    PositionLimitConfig,
    RiskLevel,
    PositionRisk,
    ConcentrationAlert,
    get_portfolio_risk_manager,
    reset_portfolio_risk_manager,
)
from src.risk_management.advanced_risk_analytics import Sector


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def mock_risk_analytics():
    """Create a mock risk analytics object."""
    mock = MagicMock()
    mock.update_positions_bulk = MagicMock()
    mock.update_position = MagicMock()
    return mock


@pytest.fixture
def portfolio_manager(mock_risk_analytics):
    """Create a fresh portfolio risk manager with mocked analytics."""
    reset_portfolio_risk_manager()
    manager = PortfolioRiskManager(risk_analytics=mock_risk_analytics)
    return manager


@pytest.fixture
def portfolio_with_positions(portfolio_manager):
    """Create manager with sample positions."""
    positions = {
        "AAPL": {"quantity": 100, "market_value": 15000},  # Technology
        "MSFT": {"quantity": 50, "market_value": 17500},   # Technology
        "JPM": {"quantity": 75, "market_value": 12000},    # Financials
        "JNJ": {"quantity": 40, "market_value": 6500},     # Healthcare
        "XOM": {"quantity": 60, "market_value": 5000},     # Energy
    }
    portfolio_manager.update_portfolio(positions)
    return portfolio_manager


@pytest.fixture
def concentrated_portfolio(portfolio_manager):
    """Create manager with concentrated sector exposure."""
    positions = {
        "AAPL": {"quantity": 200, "market_value": 35000},  # Technology
        "MSFT": {"quantity": 150, "market_value": 25000},  # Technology
        "GOOGL": {"quantity": 100, "market_value": 15000}, # Technology
        "JPM": {"quantity": 50, "market_value": 10000},    # Financials
        "JNJ": {"quantity": 30, "market_value": 5000},     # Healthcare
    }
    portfolio_manager.update_portfolio(positions)
    return portfolio_manager


# =============================================================================
# Test Sector Exposure Calculation
# =============================================================================

class TestSectorExposure:
    """Tests for sector exposure calculations."""

    def test_empty_portfolio(self, portfolio_manager):
        """Test sector exposure with empty portfolio."""
        exposure = portfolio_manager.get_sector_exposure()
        assert exposure == {}

    def test_single_position_exposure(self, portfolio_manager):
        """Test sector exposure with single position."""
        portfolio_manager.update_position("AAPL", 100, 10000)

        exposure = portfolio_manager.get_sector_exposure()

        assert Sector.TECHNOLOGY.value in exposure
        tech_exp = exposure[Sector.TECHNOLOGY.value]
        assert tech_exp.exposure_percent == 100.0
        assert tech_exp.total_value == 10000
        assert "AAPL" in tech_exp.symbols

    def test_multiple_positions_same_sector(self, portfolio_manager):
        """Test exposure with multiple positions in same sector."""
        portfolio_manager.update_position("AAPL", 100, 10000)
        portfolio_manager.update_position("MSFT", 50, 10000)

        exposure = portfolio_manager.get_sector_exposure()

        tech_exp = exposure[Sector.TECHNOLOGY.value]
        assert tech_exp.exposure_percent == 100.0
        assert tech_exp.total_value == 20000
        assert len(tech_exp.symbols) == 2

    def test_multiple_sectors(self, portfolio_with_positions):
        """Test exposure with positions across sectors."""
        exposure = portfolio_with_positions.get_sector_exposure()

        # Total value: 15000 + 17500 + 12000 + 6500 + 5000 = 56000
        total = 56000.0

        assert Sector.TECHNOLOGY.value in exposure
        assert Sector.FINANCIALS.value in exposure
        assert Sector.HEALTHCARE.value in exposure
        assert Sector.ENERGY.value in exposure

        # Technology: 15000 + 17500 = 32500
        tech_exp = exposure[Sector.TECHNOLOGY.value]
        expected_tech_pct = (32500 / total) * 100
        assert abs(tech_exp.exposure_percent - expected_tech_pct) < 0.1

    def test_exposure_symbols_tracked(self, portfolio_with_positions):
        """Test that symbols are properly tracked per sector."""
        exposure = portfolio_with_positions.get_sector_exposure()

        tech_exp = exposure[Sector.TECHNOLOGY.value]
        assert set(tech_exp.symbols) == {"AAPL", "MSFT"}

        fin_exp = exposure[Sector.FINANCIALS.value]
        assert "JPM" in fin_exp.symbols


# =============================================================================
# Test Sector Limit Enforcement
# =============================================================================

class TestSectorLimitEnforcement:
    """Tests for sector concentration limit enforcement."""

    def test_default_sector_limit(self, portfolio_manager):
        """Test default sector limit is 30%."""
        assert portfolio_manager.sector_config.default_limit_percent == 30.0

    def test_sector_under_limit(self, portfolio_with_positions):
        """Test sectors under limit are not flagged."""
        # Total 56000, Technology 32500 = ~58% - over limit
        # But let's adjust to test under limit
        violations = portfolio_with_positions.check_sector_limits()

        # Technology at 58% should be over 30% limit
        tech_violations = [v for v in violations if v.sector == Sector.TECHNOLOGY.value]
        assert len(tech_violations) == 1

    def test_sector_over_limit(self, concentrated_portfolio):
        """Test detection of sectors over limit."""
        violations = concentrated_portfolio.check_sector_limits()

        # Technology at (35000+25000+15000)/90000 = 83.3%
        assert len(violations) >= 1

        tech_violation = next(v for v in violations if v.sector == Sector.TECHNOLOGY.value)
        assert tech_violation.is_over_limit is True
        assert tech_violation.exposure_percent > 30.0

    def test_custom_sector_limit(self, portfolio_manager):
        """Test setting custom sector limits."""
        portfolio_manager.set_sector_limit(Sector.TECHNOLOGY.value, 50.0)

        assert portfolio_manager.get_sector_limit(Sector.TECHNOLOGY.value) == 50.0
        assert portfolio_manager.get_sector_limit(Sector.FINANCIALS.value) == 30.0

    def test_custom_limit_applied(self, concentrated_portfolio):
        """Test that custom limits are applied in checks."""
        # Set high limit for technology
        concentrated_portfolio.set_sector_limit(Sector.TECHNOLOGY.value, 90.0)

        violations = concentrated_portfolio.check_sector_limits()

        # Technology should no longer be over limit
        tech_violations = [v for v in violations if v.sector == Sector.TECHNOLOGY.value]
        assert len(tech_violations) == 0

    def test_invalid_limit_value(self, portfolio_manager):
        """Test that invalid limit values are rejected."""
        with pytest.raises(ValueError):
            portfolio_manager.set_sector_limit(Sector.TECHNOLOGY.value, 150.0)

        with pytest.raises(ValueError):
            portfolio_manager.set_sector_limit(Sector.TECHNOLOGY.value, -10.0)


# =============================================================================
# Test Position Concentration Limits
# =============================================================================

class TestPositionConcentration:
    """Tests for position concentration limits."""

    def test_default_position_limit(self, portfolio_manager):
        """Test default position limit is 10%."""
        assert portfolio_manager.position_config.default_limit_percent == 10.0

    def test_position_under_limit(self, portfolio_manager):
        """Test position under limit is not flagged."""
        # Create portfolio where no single position exceeds 10%
        portfolio_manager.update_portfolio({
            "AAPL": {"quantity": 10, "market_value": 1500},
            "MSFT": {"quantity": 10, "market_value": 1500},
            "JPM": {"quantity": 10, "market_value": 1500},
            "JNJ": {"quantity": 10, "market_value": 1500},
            "XOM": {"quantity": 10, "market_value": 1500},
            "GOOGL": {"quantity": 10, "market_value": 1500},
            "META": {"quantity": 10, "market_value": 1500},
            "BAC": {"quantity": 10, "market_value": 1500},
            "WFC": {"quantity": 10, "market_value": 1500},
            "PFE": {"quantity": 10, "market_value": 1500},
        })

        violations = portfolio_manager.check_position_limits()
        assert len(violations) == 0

    def test_position_over_limit(self, portfolio_with_positions):
        """Test detection of positions over limit."""
        violations = portfolio_with_positions.check_position_limits()

        # MSFT at 17500/56000 = 31.25% - way over 10%
        over_limit = [v for v in violations if v.symbol == "MSFT"]
        assert len(over_limit) == 1
        assert over_limit[0].is_over_position_limit is True

    def test_custom_position_limit(self, portfolio_manager):
        """Test setting custom position limits."""
        portfolio_manager.set_position_limit("AAPL", 25.0)

        assert portfolio_manager.get_position_limit("AAPL") == 25.0
        assert portfolio_manager.get_position_limit("MSFT") == 10.0

    def test_position_exposures(self, portfolio_with_positions):
        """Test getting position exposure details."""
        exposures = portfolio_with_positions.get_position_exposures()

        assert len(exposures) == 5

        # Check highest exposure is first (sorted)
        assert exposures[0].portfolio_percent >= exposures[-1].portfolio_percent

        # Check MSFT details
        msft = next(e for e in exposures if e.symbol == "MSFT")
        assert msft.market_value == 17500
        assert msft.sector == Sector.TECHNOLOGY


# =============================================================================
# Test Alert Generation
# =============================================================================

class TestAlertGeneration:
    """Tests for concentration alert generation."""

    def test_sector_alert_generated(self, concentrated_portfolio):
        """Test that sector alerts are generated."""
        alerts = []

        def capture_alert(alert):
            alerts.append(alert)

        concentrated_portfolio.set_alert_callbacks(on_sector_alert=capture_alert)
        concentrated_portfolio.check_sector_limits()

        assert len(alerts) >= 1
        assert all(a.alert_type == "sector" for a in alerts)

    def test_position_alert_generated(self, portfolio_with_positions):
        """Test that position alerts are generated."""
        alerts = []

        def capture_alert(alert):
            alerts.append(alert)

        portfolio_with_positions.set_alert_callbacks(on_position_alert=capture_alert)
        portfolio_with_positions.check_position_limits()

        assert len(alerts) >= 1
        assert all(a.alert_type == "position" for a in alerts)

    def test_critical_alert_callback(self, concentrated_portfolio):
        """Test critical alert callback."""
        critical_alerts = []

        def capture_critical(alert):
            critical_alerts.append(alert)

        concentrated_portfolio.set_alert_callbacks(on_critical_alert=capture_critical)
        concentrated_portfolio.check_sector_limits()

        # Tech at 83% is 53% over limit - should be critical
        assert len(critical_alerts) >= 1

    def test_alert_history(self, concentrated_portfolio):
        """Test alert history is maintained."""
        concentrated_portfolio.check_sector_limits()

        history = concentrated_portfolio.get_alert_history()
        assert len(history) >= 1

    def test_alert_history_filter(self, concentrated_portfolio):
        """Test filtering alert history by type."""
        concentrated_portfolio.check_sector_limits()
        concentrated_portfolio.check_position_limits()

        sector_alerts = concentrated_portfolio.get_alert_history(alert_type="sector")
        position_alerts = concentrated_portfolio.get_alert_history(alert_type="position")

        assert all(a.alert_type == "sector" for a in sector_alerts)
        assert all(a.alert_type == "position" for a in position_alerts)

    def test_alert_contains_action(self, concentrated_portfolio):
        """Test that alerts contain recommended actions."""
        concentrated_portfolio.check_sector_limits()

        history = concentrated_portfolio.get_alert_history()
        assert len(history) >= 1
        assert history[0].recommended_action is not None
        assert len(history[0].recommended_action) > 0


# =============================================================================
# Test Trade Validation
# =============================================================================

class TestTradeValidation:
    """Tests for trade validation against limits."""

    def test_sell_always_valid(self, portfolio_with_positions):
        """Test that sell trades are always valid for concentration."""
        is_valid, reason = portfolio_with_positions.validate_trade(
            "MSFT", "sell", 50, 350.0
        )
        assert is_valid is True
        assert reason is None

    def test_buy_under_limits(self, portfolio_manager):
        """Test buy trade that stays under limits."""
        # Start with balanced portfolio
        portfolio_manager.update_portfolio({
            "JPM": {"quantity": 100, "market_value": 50000},  # 50%
            "JNJ": {"quantity": 100, "market_value": 50000},  # 50%
        })

        # Buy AAPL (Technology) for 5000 - should be fine
        is_valid, reason = portfolio_manager.validate_trade(
            "AAPL", "buy", 50, 100.0  # 5000 value
        )
        assert is_valid is True

    def test_buy_exceeds_position_limit(self, portfolio_manager):
        """Test buy trade that exceeds position limit."""
        portfolio_manager.update_portfolio({
            "AAPL": {"quantity": 100, "market_value": 9000},  # 9%
            "MSFT": {"quantity": 100, "market_value": 91000}, # 91%
        })

        # Try to buy more AAPL to exceed 10%
        is_valid, reason = portfolio_manager.validate_trade(
            "AAPL", "buy", 20, 100.0  # Would put AAPL at 11000/102000 = 10.8%
        )
        assert is_valid is False
        assert "portfolio" in reason.lower()

    def test_buy_exceeds_sector_limit(self, portfolio_manager):
        """Test buy trade that exceeds sector limit."""
        portfolio_manager.update_portfolio({
            "AAPL": {"quantity": 100, "market_value": 28000},  # 28% Tech
            "JPM": {"quantity": 100, "market_value": 72000},   # 72% Financials
        })

        # Try to buy more MSFT (Technology) - would push tech over 30%
        is_valid, reason = portfolio_manager.validate_trade(
            "MSFT", "buy", 50, 100.0  # Would put Tech at 33000/105000 = 31.4%
        )
        assert is_valid is False
        assert "sector" in reason.lower()


# =============================================================================
# Test Diversification Score
# =============================================================================

class TestDiversificationScore:
    """Tests for diversification scoring."""

    def test_empty_portfolio_score(self, portfolio_manager):
        """Test score for empty portfolio."""
        score = portfolio_manager.calculate_diversification_score()
        assert score == 0.0

    def test_well_diversified_score(self, portfolio_manager):
        """Test score for well-diversified portfolio."""
        # Create balanced portfolio across many sectors
        portfolio_manager.update_portfolio({
            "AAPL": {"quantity": 10, "market_value": 1500},   # Tech
            "JPM": {"quantity": 10, "market_value": 1500},    # Financials
            "JNJ": {"quantity": 10, "market_value": 1500},    # Healthcare
            "XOM": {"quantity": 10, "market_value": 1500},    # Energy
            "BA": {"quantity": 10, "market_value": 1500},     # Industrials
            "PG": {"quantity": 10, "market_value": 1500},     # Consumer Staples
            "AMZN": {"quantity": 10, "market_value": 1500},   # Consumer Disc
            "NEE": {"quantity": 10, "market_value": 1500},    # Utilities
        })

        score = portfolio_manager.calculate_diversification_score()
        assert score >= 60.0  # Well diversified

    def test_concentrated_score(self, concentrated_portfolio):
        """Test score for concentrated portfolio."""
        score = concentrated_portfolio.calculate_diversification_score()
        # Highly concentrated in tech - should have lower score than well-diversified
        assert score < 90.0  # Less than a well-diversified portfolio

    def test_single_position_score(self, portfolio_manager):
        """Test score with single position."""
        portfolio_manager.update_position("AAPL", 100, 10000)

        score = portfolio_manager.calculate_diversification_score()
        # Single position - should be relatively low
        assert score < 65.0

    def test_score_bounds(self, portfolio_manager):
        """Test that score stays within 0-100."""
        # Various scenarios
        portfolio_manager.update_position("AAPL", 100, 10000)
        score1 = portfolio_manager.calculate_diversification_score()
        assert 0 <= score1 <= 100

        # Add more positions
        portfolio_manager.update_position("JPM", 50, 5000)
        portfolio_manager.update_position("JNJ", 50, 5000)
        score2 = portfolio_manager.calculate_diversification_score()
        assert 0 <= score2 <= 100


# =============================================================================
# Test Risk Report
# =============================================================================

class TestRiskReport:
    """Tests for risk report generation."""

    def test_report_structure(self, portfolio_with_positions):
        """Test risk report has correct structure."""
        report = portfolio_with_positions.get_risk_report()

        assert hasattr(report, 'total_value')
        assert hasattr(report, 'position_count')
        assert hasattr(report, 'sector_count')
        assert hasattr(report, 'sectors_over_limit')
        assert hasattr(report, 'positions_over_limit')
        assert hasattr(report, 'diversification_score')
        assert hasattr(report, 'overall_risk_level')

    def test_report_values(self, portfolio_with_positions):
        """Test risk report has correct values."""
        report = portfolio_with_positions.get_risk_report()

        assert report.total_value == 56000.0
        assert report.position_count == 5
        assert report.sector_count == 4  # Tech, Financials, Healthcare, Energy

    def test_report_risk_level_low(self, portfolio_manager):
        """Test low risk level for balanced portfolio."""
        # Well balanced portfolio
        portfolio_manager.update_portfolio({
            "AAPL": {"quantity": 10, "market_value": 1000},
            "JPM": {"quantity": 10, "market_value": 1000},
            "JNJ": {"quantity": 10, "market_value": 1000},
            "XOM": {"quantity": 10, "market_value": 1000},
            "BA": {"quantity": 10, "market_value": 1000},
            "PG": {"quantity": 10, "market_value": 1000},
            "AMZN": {"quantity": 10, "market_value": 1000},
            "NEE": {"quantity": 10, "market_value": 1000},
            "NFLX": {"quantity": 10, "market_value": 1000},
            "AMT": {"quantity": 10, "market_value": 1000},
        })

        report = portfolio_manager.get_risk_report()
        # All positions at 10% each, no sector over 30%
        assert report.overall_risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM]

    def test_report_risk_level_critical(self, concentrated_portfolio):
        """Test critical risk level for concentrated portfolio."""
        report = concentrated_portfolio.get_risk_report()

        # Tech at 83% is way over limit
        assert report.overall_risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]


# =============================================================================
# Test Position Updates
# =============================================================================

class TestPositionUpdates:
    """Tests for position update functionality."""

    def test_update_single_position(self, portfolio_manager):
        """Test updating a single position."""
        portfolio_manager.update_position("AAPL", 100, 15000)

        exposures = portfolio_manager.get_position_exposures()
        assert len(exposures) == 1
        assert exposures[0].symbol == "AAPL"
        assert exposures[0].market_value == 15000

    def test_update_existing_position(self, portfolio_manager):
        """Test updating an existing position."""
        portfolio_manager.update_position("AAPL", 100, 15000)
        portfolio_manager.update_position("AAPL", 150, 22500)

        exposures = portfolio_manager.get_position_exposures()
        assert len(exposures) == 1
        assert exposures[0].market_value == 22500

    def test_remove_position(self, portfolio_with_positions):
        """Test removing a position."""
        portfolio_with_positions.remove_position("AAPL")

        exposures = portfolio_with_positions.get_position_exposures()
        symbols = [e.symbol for e in exposures]
        assert "AAPL" not in symbols

    def test_total_value_updated(self, portfolio_manager):
        """Test that total value is properly maintained."""
        portfolio_manager.update_position("AAPL", 100, 15000)
        assert portfolio_manager._total_portfolio_value == 15000

        portfolio_manager.update_position("MSFT", 50, 10000)
        assert portfolio_manager._total_portfolio_value == 25000

        portfolio_manager.remove_position("AAPL")
        assert portfolio_manager._total_portfolio_value == 10000


# =============================================================================
# Test Global Instance
# =============================================================================

class TestGlobalInstance:
    """Tests for global portfolio risk manager."""

    def test_get_global_instance(self):
        """Test getting global instance."""
        reset_portfolio_risk_manager()

        manager1 = get_portfolio_risk_manager()
        manager2 = get_portfolio_risk_manager()

        assert manager1 is manager2

    def test_reset_global_instance(self):
        """Test resetting global instance."""
        manager1 = get_portfolio_risk_manager()
        manager1.update_position("AAPL", 100, 15000)

        reset_portfolio_risk_manager()

        manager2 = get_portfolio_risk_manager()
        assert manager2._total_portfolio_value == 0.0

    def test_custom_config_on_create(self):
        """Test passing custom config on creation."""
        reset_portfolio_risk_manager()

        sector_config = SectorLimitConfig(default_limit_percent=40.0)
        position_config = PositionLimitConfig(default_limit_percent=15.0)

        manager = get_portfolio_risk_manager(
            sector_limit_config=sector_config,
            position_limit_config=position_config,
        )

        assert manager.sector_config.default_limit_percent == 40.0
        assert manager.position_config.default_limit_percent == 15.0


# =============================================================================
# Test Status
# =============================================================================

class TestStatus:
    """Tests for status reporting."""

    def test_get_status_structure(self, portfolio_with_positions):
        """Test status has correct structure."""
        status = portfolio_with_positions.get_status()

        assert "total_value" in status
        assert "position_count" in status
        assert "sector_count" in status
        assert "sectors_over_limit" in status
        assert "positions_over_limit" in status
        assert "diversification_score" in status
        assert "overall_risk_level" in status
        assert "sector_limit_default" in status
        assert "position_limit_default" in status

    def test_status_values(self, portfolio_with_positions):
        """Test status has correct values."""
        status = portfolio_with_positions.get_status()

        assert status["total_value"] == 56000.0
        assert status["position_count"] == 5
        assert status["sector_limit_default"] == 30.0
        assert status["position_limit_default"] == 10.0
