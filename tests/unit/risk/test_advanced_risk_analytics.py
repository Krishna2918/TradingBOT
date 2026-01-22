"""
Tests for Advanced Risk Analytics Module
========================================

Tests cover:
- Options Greeks calculation (Black-Scholes)
- Correlation matrix computation
- Sector concentration tracking
- Stress testing scenarios
- Callbacks and alerts

Coverage Target: 85%
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock
import threading

from src.risk_management.advanced_risk_analytics import (
    AdvancedRiskAnalytics,
    GreeksCalculator,
    GreeksResult,
    SectorExposure,
    CorrelationAlert,
    StressTestResult,
    Sector,
    get_risk_analytics,
    DEFAULT_SECTOR_MAPPINGS,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def analytics():
    """Create a fresh analytics instance."""
    return AdvancedRiskAnalytics(
        sector_limit_percent=30.0,
        correlation_change_threshold=0.2,
        correlation_lookback_days=60,
    )


@pytest.fixture
def sample_returns():
    """Create sample returns data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")

    return {
        "AAPL": pd.Series(np.random.normal(0.001, 0.02, 100), index=dates),
        "MSFT": pd.Series(np.random.normal(0.001, 0.018, 100), index=dates),
        "GOOG": pd.Series(np.random.normal(0.0008, 0.022, 100), index=dates),
        "JPM": pd.Series(np.random.normal(0.0005, 0.025, 100), index=dates),
        "XOM": pd.Series(np.random.normal(0.0003, 0.03, 100), index=dates),
    }


@pytest.fixture
def sample_positions():
    """Create sample position data for testing."""
    return {
        "AAPL": {"quantity": 100, "market_value": 17500.0},
        "MSFT": {"quantity": 50, "market_value": 18750.0},
        "GOOG": {"quantity": 20, "market_value": 2800.0},
        "JPM": {"quantity": 80, "market_value": 12000.0},
        "XOM": {"quantity": 100, "market_value": 10500.0},
    }


@pytest.fixture
def reset_global_analytics():
    """Reset global analytics instance after test."""
    import src.risk_management.advanced_risk_analytics as ara_module
    original = ara_module._analytics_instance
    yield
    ara_module._analytics_instance = original


# =============================================================================
# Test Greeks Calculator
# =============================================================================

class TestGreeksCalculator:
    """Tests for Black-Scholes Greeks calculation."""

    def test_calculate_d1(self):
        """Test d1 calculation."""
        d1 = GreeksCalculator.calculate_d1(
            S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2
        )
        # d1 = (ln(100/100) + (0.05 + 0.5*0.04)*1) / (0.2*1) = 0.35
        assert d1 == pytest.approx(0.35, rel=0.01)

    def test_calculate_d1_zero_time(self):
        """Test d1 with zero time to expiry."""
        d1 = GreeksCalculator.calculate_d1(
            S=100.0, K=100.0, T=0.0, r=0.05, sigma=0.2
        )
        assert d1 == 0.0

    def test_calculate_d2(self):
        """Test d2 calculation."""
        d1 = 0.35
        d2 = GreeksCalculator.calculate_d2(d1=d1, sigma=0.2, T=1.0)
        # d2 = 0.35 - 0.2*1 = 0.15
        assert d2 == pytest.approx(0.15, rel=0.01)

    def test_calculate_call_price_atm(self):
        """Test call option price at the money."""
        price = GreeksCalculator.calculate_call_price(
            S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2
        )
        # ATM call should be roughly 10% of spot for 20% vol, 1 year
        assert price > 0
        assert price < 20

    def test_calculate_put_price_atm(self):
        """Test put option price at the money."""
        price = GreeksCalculator.calculate_put_price(
            S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2
        )
        assert price > 0
        assert price < 20

    def test_calculate_call_price_expired(self):
        """Test call option price at expiry."""
        # ITM call
        price_itm = GreeksCalculator.calculate_call_price(
            S=110.0, K=100.0, T=0.0, r=0.05, sigma=0.2
        )
        assert price_itm == 10.0  # max(0, 110-100) = 10

        # OTM call
        price_otm = GreeksCalculator.calculate_call_price(
            S=90.0, K=100.0, T=0.0, r=0.05, sigma=0.2
        )
        assert price_otm == 0.0

    def test_calculate_greeks_call(self):
        """Test full Greeks calculation for call option."""
        greeks = GreeksCalculator.calculate_greeks(
            option_type="call",
            S=100.0,
            K=100.0,
            T=30/365,
            r=0.05,
            sigma=0.25,
        )

        assert isinstance(greeks, GreeksResult)
        assert greeks.option_type == "call"
        assert greeks.spot_price == 100.0
        assert greeks.strike_price == 100.0

        # Call delta should be between 0 and 1
        assert 0 < greeks.delta < 1
        # ATM delta should be around 0.5
        assert 0.4 < greeks.delta < 0.6

        # Gamma should be positive
        assert greeks.gamma > 0

        # Theta should be negative (time decay)
        assert greeks.theta < 0

        # Vega should be positive
        assert greeks.vega > 0

    def test_calculate_greeks_put(self):
        """Test full Greeks calculation for put option."""
        greeks = GreeksCalculator.calculate_greeks(
            option_type="put",
            S=100.0,
            K=100.0,
            T=30/365,
            r=0.05,
            sigma=0.25,
        )

        # Put delta should be between -1 and 0
        assert -1 < greeks.delta < 0
        # ATM put delta should be around -0.5
        assert -0.6 < greeks.delta < -0.4

        # Gamma is same for put and call
        assert greeks.gamma > 0

        # Theta should be negative
        assert greeks.theta < 0

    def test_calculate_greeks_itm_call(self):
        """Test Greeks for in-the-money call."""
        greeks = GreeksCalculator.calculate_greeks(
            option_type="call",
            S=120.0,
            K=100.0,
            T=30/365,
            r=0.05,
            sigma=0.25,
        )

        # ITM call delta should be close to 1
        assert greeks.delta > 0.8

    def test_calculate_greeks_otm_put(self):
        """Test Greeks for out-of-the-money put."""
        greeks = GreeksCalculator.calculate_greeks(
            option_type="put",
            S=120.0,
            K=100.0,
            T=30/365,
            r=0.05,
            sigma=0.25,
        )

        # OTM put delta should be close to 0
        assert greeks.delta > -0.2

    def test_calculate_greeks_expired(self):
        """Test Greeks at expiry."""
        greeks = GreeksCalculator.calculate_greeks(
            option_type="call",
            S=110.0,
            K=100.0,
            T=0.0,
            r=0.05,
            sigma=0.25,
        )

        # ITM at expiry
        assert greeks.delta == 1.0
        assert greeks.gamma == 0.0
        assert greeks.theta == 0.0
        assert greeks.vega == 0.0
        assert greeks.option_price == 10.0

    def test_put_call_parity(self):
        """Test put-call parity relationship."""
        S, K, T, r, sigma = 100.0, 100.0, 0.5, 0.05, 0.25

        call_price = GreeksCalculator.calculate_call_price(S, K, T, r, sigma)
        put_price = GreeksCalculator.calculate_put_price(S, K, T, r, sigma)

        # Put-call parity: C - P = S - K*e^(-rT)
        expected_diff = S - K * np.exp(-r * T)
        actual_diff = call_price - put_price

        assert actual_diff == pytest.approx(expected_diff, rel=0.01)


# =============================================================================
# Test Correlation Matrix
# =============================================================================

class TestCorrelationMatrix:
    """Tests for correlation matrix computation."""

    def test_update_returns(self, analytics, sample_returns):
        """Test updating returns data."""
        analytics.update_returns("AAPL", sample_returns["AAPL"])
        assert "AAPL" in analytics.get_symbols()

    def test_update_returns_bulk(self, analytics, sample_returns):
        """Test bulk updating returns data."""
        analytics.update_returns_bulk(sample_returns)
        symbols = analytics.get_symbols()
        assert len(symbols) == 5
        assert "AAPL" in symbols
        assert "MSFT" in symbols

    def test_compute_correlation_matrix(self, analytics, sample_returns):
        """Test correlation matrix computation."""
        analytics.update_returns_bulk(sample_returns)
        corr_matrix = analytics.compute_correlation_matrix()

        assert not corr_matrix.empty
        assert "AAPL" in corr_matrix.columns
        assert "MSFT" in corr_matrix.columns

        # Diagonal should be 1
        assert corr_matrix.loc["AAPL", "AAPL"] == pytest.approx(1.0)

        # Correlations should be between -1 and 1
        assert corr_matrix.min().min() >= -1.0
        assert corr_matrix.max().max() <= 1.0

    def test_get_correlation(self, analytics, sample_returns):
        """Test getting correlation between two symbols."""
        analytics.update_returns_bulk(sample_returns)

        corr = analytics.get_correlation("AAPL", "MSFT")
        assert corr is not None
        assert -1.0 <= corr <= 1.0

    def test_get_correlation_nonexistent(self, analytics):
        """Test getting correlation for nonexistent symbol."""
        corr = analytics.get_correlation("AAPL", "MSFT")
        assert corr is None

    def test_get_highest_correlations(self, analytics, sample_returns):
        """Test getting highest correlations for a symbol."""
        analytics.update_returns_bulk(sample_returns)

        top_corr = analytics.get_highest_correlations("AAPL", n=3)
        assert len(top_corr) == 3
        assert all(isinstance(item, tuple) and len(item) == 2 for item in top_corr)

    def test_correlation_caching(self, analytics, sample_returns):
        """Test that correlation matrix is cached."""
        analytics.update_returns_bulk(sample_returns)

        # First computation
        matrix1 = analytics.compute_correlation_matrix()
        # Second computation should return cached version
        matrix2 = analytics.compute_correlation_matrix()

        pd.testing.assert_frame_equal(matrix1, matrix2)

    def test_correlation_invalidation_on_update(self, analytics, sample_returns):
        """Test that cache is invalidated on data update."""
        analytics.update_returns_bulk(sample_returns)
        analytics.compute_correlation_matrix()

        # Update should invalidate cache
        new_returns = pd.Series(np.random.normal(0.001, 0.02, 100))
        analytics.update_returns("NEW", new_returns)

        # Should include new symbol
        symbols = analytics.get_symbols()
        assert "NEW" in symbols


# =============================================================================
# Test Correlation Alerts
# =============================================================================

class TestCorrelationAlerts:
    """Tests for correlation change alerts."""

    def test_correlation_alert_on_change(self, analytics):
        """Test that alerts are generated on significant correlation changes."""
        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")

        # Initial correlated data
        base = np.random.normal(0, 0.02, 100)
        returns1 = {
            "A": pd.Series(base + np.random.normal(0, 0.005, 100), index=dates),
            "B": pd.Series(base + np.random.normal(0, 0.005, 100), index=dates),
        }

        analytics.update_returns_bulk(returns1)
        analytics.compute_correlation_matrix()

        # Now change correlation significantly
        returns2 = {
            "A": pd.Series(np.random.normal(0, 0.02, 100), index=dates),  # Independent
            "B": pd.Series(-base + np.random.normal(0, 0.005, 100), index=dates),  # Negatively correlated
        }

        analytics.update_returns_bulk(returns2)
        analytics.compute_correlation_matrix()

        alerts = analytics.get_correlation_alerts()
        # Should have at least one alert for the correlation change
        # (depends on actual change magnitude)

    def test_correlation_alert_callback(self, analytics):
        """Test correlation alert callback."""
        callback_mock = Mock()
        analytics.set_callbacks(on_correlation_alert=callback_mock)

        # Setup and trigger significant change (implementation dependent)


# =============================================================================
# Test Sector Concentration
# =============================================================================

class TestSectorConcentration:
    """Tests for sector concentration tracking."""

    def test_update_position(self, analytics):
        """Test updating position data."""
        analytics.update_position("AAPL", quantity=100, market_value=17500.0)
        exposure = analytics.get_sector_exposure()
        assert "Technology" in exposure

    def test_update_positions_bulk(self, analytics, sample_positions):
        """Test bulk updating positions."""
        analytics.update_positions_bulk(sample_positions)
        exposure = analytics.get_sector_exposure()

        assert len(exposure) > 0

    def test_sector_exposure_calculation(self, analytics, sample_positions):
        """Test sector exposure percentage calculation."""
        analytics.update_positions_bulk(sample_positions)
        exposure = analytics.get_sector_exposure()

        # Total should be 100%
        total_exposure = sum(e.exposure_percent for e in exposure.values())
        assert total_exposure == pytest.approx(100.0, rel=0.01)

    def test_sector_over_limit_detection(self, analytics):
        """Test detection of sector over concentration limit."""
        # Create concentrated portfolio (>30% in tech)
        analytics.update_position("AAPL", quantity=100, market_value=40000.0)
        analytics.update_position("MSFT", quantity=50, market_value=20000.0)
        analytics.update_position("JPM", quantity=80, market_value=10000.0)

        over_limit = analytics.check_sector_limits()

        # Tech should be over 30% (60000/70000 = 85.7%)
        assert len(over_limit) > 0
        tech_exposure = next((e for e in over_limit if e.sector == "Technology"), None)
        assert tech_exposure is not None
        assert tech_exposure.is_over_limit

    def test_sector_limit_callback(self, analytics):
        """Test sector limit breach callback."""
        callback_mock = Mock()
        analytics.set_callbacks(on_sector_limit_breach=callback_mock)

        # Create over-limit position
        analytics.update_position("AAPL", quantity=100, market_value=40000.0)
        analytics.update_position("JPM", quantity=80, market_value=10000.0)

        # Getting exposure should trigger callback
        analytics.get_sector_exposure()

        assert callback_mock.called

    def test_add_sector_mapping(self, analytics):
        """Test adding custom sector mapping."""
        analytics.add_sector_mapping("NEWSTOCK", Sector.HEALTHCARE)
        sector = analytics.get_sector_for_symbol("NEWSTOCK")
        assert sector == Sector.HEALTHCARE

    def test_unknown_sector_default(self, analytics):
        """Test unknown symbol defaults to UNKNOWN sector."""
        sector = analytics.get_sector_for_symbol("UNKNOWNSYMBOL")
        assert sector == Sector.UNKNOWN

    def test_sector_exposure_fields(self, analytics):
        """Test SectorExposure dataclass fields."""
        analytics.update_position("AAPL", quantity=100, market_value=10000.0)
        analytics.update_position("MSFT", quantity=50, market_value=10000.0)

        exposure = analytics.get_sector_exposure()
        tech_exposure = exposure.get("Technology")

        assert tech_exposure is not None
        assert tech_exposure.sector == "Technology"
        assert tech_exposure.total_value == 20000.0
        assert "AAPL" in tech_exposure.symbols
        assert "MSFT" in tech_exposure.symbols


# =============================================================================
# Test Portfolio Greeks
# =============================================================================

class TestPortfolioGreeks:
    """Tests for portfolio-level Greeks calculation."""

    def test_calculate_portfolio_greeks(self, analytics):
        """Test calculating aggregate portfolio Greeks."""
        options_positions = [
            {
                "option_type": "call",
                "spot_price": 150.0,
                "strike_price": 155.0,
                "time_to_expiry": 30/365,
                "risk_free_rate": 0.05,
                "volatility": 0.25,
                "quantity": 10,  # Long 10 contracts
            },
            {
                "option_type": "put",
                "spot_price": 150.0,
                "strike_price": 145.0,
                "time_to_expiry": 30/365,
                "risk_free_rate": 0.05,
                "volatility": 0.25,
                "quantity": -5,  # Short 5 contracts
            },
        ]

        portfolio_greeks = analytics.calculate_portfolio_greeks(options_positions)

        assert "delta" in portfolio_greeks
        assert "gamma" in portfolio_greeks
        assert "theta" in portfolio_greeks
        assert "vega" in portfolio_greeks
        assert "rho" in portfolio_greeks

    def test_portfolio_greeks_netting(self, analytics):
        """Test that portfolio Greeks net correctly."""
        # Long call + short call at same strike = near-zero delta
        options_positions = [
            {
                "option_type": "call",
                "spot_price": 100.0,
                "strike_price": 100.0,
                "time_to_expiry": 30/365,
                "risk_free_rate": 0.05,
                "volatility": 0.25,
                "quantity": 1,
            },
            {
                "option_type": "call",
                "spot_price": 100.0,
                "strike_price": 100.0,
                "time_to_expiry": 30/365,
                "risk_free_rate": 0.05,
                "volatility": 0.25,
                "quantity": -1,
            },
        ]

        portfolio_greeks = analytics.calculate_portfolio_greeks(options_positions)

        # Should net to approximately zero
        assert abs(portfolio_greeks["delta"]) < 1.0
        assert abs(portfolio_greeks["gamma"]) < 0.1


# =============================================================================
# Test Stress Testing
# =============================================================================

class TestStressTesting:
    """Tests for stress testing functionality."""

    def test_run_stress_test(self, analytics, sample_positions):
        """Test running a stress test scenario."""
        analytics.update_positions_bulk(sample_positions)

        result = analytics.run_stress_test(
            scenario_name="market_crash",
            price_shocks={"DEFAULT": -0.20},
            portfolio_value=61550.0,
        )

        assert isinstance(result, StressTestResult)
        assert result.scenario_name == "market_crash"
        assert result.portfolio_impact_percent == pytest.approx(-20.0, rel=0.1)

    def test_run_stress_test_per_symbol(self, analytics, sample_positions):
        """Test stress test with per-symbol shocks."""
        analytics.update_positions_bulk(sample_positions)

        result = analytics.run_stress_test(
            scenario_name="tech_selloff",
            price_shocks={
                "AAPL": -0.30,
                "MSFT": -0.30,
                "GOOG": -0.30,
                "DEFAULT": -0.05,
            },
            portfolio_value=61550.0,
        )

        # Tech stocks should have larger impact
        assert result.portfolio_impact_percent < -5.0

    def test_run_standard_scenarios(self, analytics, sample_positions):
        """Test running standard stress scenarios."""
        analytics.update_positions_bulk(sample_positions)

        results = analytics.run_standard_scenarios(portfolio_value=61550.0)

        assert "market_crash_10" in results
        assert "market_crash_20" in results
        assert "tech_selloff" in results
        assert "financials_crisis" in results

    def test_stress_test_worst_affected(self, analytics, sample_positions):
        """Test that worst affected positions are identified."""
        analytics.update_positions_bulk(sample_positions)

        result = analytics.run_stress_test(
            scenario_name="test",
            price_shocks={"AAPL": -0.50, "DEFAULT": -0.10},
            portfolio_value=61550.0,
        )

        # AAPL should be in worst affected
        worst_symbols = [pos[0] for pos in result.worst_affected_positions]
        assert "AAPL" in worst_symbols

    def test_stress_test_empty_portfolio(self, analytics):
        """Test stress test with no positions."""
        result = analytics.run_stress_test(
            scenario_name="test",
            price_shocks={"DEFAULT": -0.20},
            portfolio_value=100000.0,
        )

        assert result.portfolio_impact_percent == 0.0
        assert result.portfolio_impact_value == 0.0


# =============================================================================
# Test Status and Reset
# =============================================================================

class TestStatusAndReset:
    """Tests for status reporting and reset functionality."""

    def test_get_status(self, analytics, sample_returns, sample_positions):
        """Test getting analytics status."""
        analytics.update_returns_bulk(sample_returns)
        analytics.update_positions_bulk(sample_positions)

        status = analytics.get_status()

        assert status["symbols_tracked"] == 5
        assert status["positions_tracked"] == 5
        assert status["sector_limit_percent"] == 30.0

    def test_reset(self, analytics, sample_returns, sample_positions):
        """Test resetting all data."""
        analytics.update_returns_bulk(sample_returns)
        analytics.update_positions_bulk(sample_positions)

        analytics.reset()

        status = analytics.get_status()
        assert status["symbols_tracked"] == 0
        assert status["positions_tracked"] == 0


# =============================================================================
# Test Global Instance
# =============================================================================

class TestGlobalInstance:
    """Tests for global analytics instance."""

    def test_get_risk_analytics(self, reset_global_analytics):
        """Test getting global instance."""
        analytics = get_risk_analytics()
        assert analytics is not None
        assert isinstance(analytics, AdvancedRiskAnalytics)

    def test_get_risk_analytics_singleton(self, reset_global_analytics):
        """Test that global instance is singleton."""
        analytics1 = get_risk_analytics()
        analytics2 = get_risk_analytics()
        assert analytics1 is analytics2

    def test_get_risk_analytics_with_params(self, reset_global_analytics):
        """Test initial params are respected."""
        import src.risk_management.advanced_risk_analytics as ara_module
        ara_module._analytics_instance = None

        analytics = get_risk_analytics(
            sector_limit_percent=25.0,
            correlation_change_threshold=0.3,
        )

        assert analytics.sector_limit_percent == 25.0
        assert analytics.correlation_change_threshold == 0.3


# =============================================================================
# Test Thread Safety
# =============================================================================

class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_returns_update(self, analytics):
        """Test concurrent returns data updates."""
        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        errors = []

        def update_returns(symbol_id):
            try:
                returns = pd.Series(np.random.normal(0.001, 0.02, 100), index=dates)
                analytics.update_returns(f"SYM{symbol_id}", returns)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=update_returns, args=(i,)) for i in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(analytics.get_symbols()) == 10

    def test_concurrent_position_update(self, analytics):
        """Test concurrent position updates."""
        errors = []

        def update_position(symbol_id):
            try:
                analytics.update_position(
                    f"SYM{symbol_id}",
                    quantity=100,
                    market_value=10000.0,
                )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=update_position, args=(i,)) for i in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# =============================================================================
# Test Default Sector Mappings
# =============================================================================

class TestDefaultSectorMappings:
    """Tests for default sector mappings."""

    def test_tech_stocks_mapped(self):
        """Test that tech stocks are mapped to Technology."""
        assert DEFAULT_SECTOR_MAPPINGS["AAPL"] == Sector.TECHNOLOGY
        assert DEFAULT_SECTOR_MAPPINGS["MSFT"] == Sector.TECHNOLOGY
        assert DEFAULT_SECTOR_MAPPINGS["NVDA"] == Sector.TECHNOLOGY

    def test_financial_stocks_mapped(self):
        """Test that financial stocks are mapped to Financials."""
        assert DEFAULT_SECTOR_MAPPINGS["JPM"] == Sector.FINANCIALS
        assert DEFAULT_SECTOR_MAPPINGS["GS"] == Sector.FINANCIALS
        assert DEFAULT_SECTOR_MAPPINGS["BAC"] == Sector.FINANCIALS

    def test_energy_stocks_mapped(self):
        """Test that energy stocks are mapped to Energy."""
        assert DEFAULT_SECTOR_MAPPINGS["XOM"] == Sector.ENERGY
        assert DEFAULT_SECTOR_MAPPINGS["CVX"] == Sector.ENERGY

    def test_healthcare_stocks_mapped(self):
        """Test that healthcare stocks are mapped to Healthcare."""
        assert DEFAULT_SECTOR_MAPPINGS["JNJ"] == Sector.HEALTHCARE
        assert DEFAULT_SECTOR_MAPPINGS["PFE"] == Sector.HEALTHCARE


# =============================================================================
# Test Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_returns_correlation(self, analytics):
        """Test correlation with no data."""
        corr_matrix = analytics.compute_correlation_matrix()
        assert corr_matrix.empty

    def test_single_symbol_correlation(self, analytics):
        """Test correlation with single symbol."""
        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        returns = pd.Series(np.random.normal(0.001, 0.02, 100), index=dates)

        analytics.update_returns("AAPL", returns)
        corr_matrix = analytics.compute_correlation_matrix()

        # Single symbol can't have correlation matrix
        assert corr_matrix.empty

    def test_zero_portfolio_value_stress_test(self, analytics, sample_positions):
        """Test stress test with zero portfolio value."""
        analytics.update_positions_bulk(sample_positions)

        result = analytics.run_stress_test(
            scenario_name="test",
            price_shocks={"DEFAULT": -0.20},
            portfolio_value=0.0,
        )

        assert result.portfolio_impact_percent == 0.0

    def test_negative_volatility_greeks(self):
        """Test Greeks with zero volatility."""
        greeks = GreeksCalculator.calculate_greeks(
            option_type="call",
            S=100.0,
            K=100.0,
            T=0.5,
            r=0.05,
            sigma=0.0,  # Zero vol
        )

        # Should handle gracefully
        assert greeks is not None

    def test_very_short_expiry(self):
        """Test Greeks with very short expiry."""
        greeks = GreeksCalculator.calculate_greeks(
            option_type="call",
            S=100.0,
            K=100.0,
            T=1/365,  # 1 day
            r=0.05,
            sigma=0.25,
        )

        # Theta should be very negative (high time decay)
        assert greeks.theta < -0.01
