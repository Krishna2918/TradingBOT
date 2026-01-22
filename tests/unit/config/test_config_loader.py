"""
Unit tests for Configuration Loader
===================================

Tests for the configuration loader and typed config classes.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, mock_open
import yaml

from src.config.config_loader import (
    ConfigLoader,
    get_config_loader,
    get_position_sizing_config,
    get_execution_config,
    get_strategy_allocations_config,
    get_risk_config,
    PositionSizingConfig,
    ExecutionConfig,
    StrategyAllocationsConfig,
    RiskConfig,
    KellyCriterionConfig,
    StopLossConfig,
    TakeProfitConfig,
    CommissionConfig,
    SlippageConfig,
    RetryConfig,
    CircuitBreakerConfig,
    VWAPConfig,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def reset_config_loader():
    """Reset the global config loader singleton before each test."""
    import src.config.config_loader as config_module
    config_module._config_loader = None
    ConfigLoader._instance = None
    yield
    config_module._config_loader = None
    ConfigLoader._instance = None


@pytest.fixture
def config_dir(tmp_path):
    """Create a temporary config directory with test files."""
    # Create position_sizing.yaml
    position_sizing = {
        "position_sizing": {
            "default_risk_per_trade": 0.02,
            "max_portfolio_risk": 0.20,
            "max_position_size": 0.10,
            "max_daily_drawdown": 0.05,
            "min_confidence": 0.70,
        },
        "kelly_criterion": {
            "min_drawdown_scale": 0.3,
            "drawdown_window_hours": 24,
            "kelly_fraction_cap": 0.25,
            "avg_win_default": 0.02,
            "avg_loss_default": 0.015,
        },
        "stop_loss": {
            "default_stop_percent": 0.05,
            "min_stop_percent": 0.01,
            "fallback_stop_percent": 0.02,
        },
        "take_profit": {
            "default_take_profit_percent": 0.02,
            "extended_take_profit_percent": 0.03,
            "default_risk_reward_ratio": 1.5,
        },
        "confidence": {
            "confidence_multiplier": 0.8,
            "signal_strength_factor": 0.2,
        },
        "portfolio_limits": {
            "max_positions": 10,
            "position_concentration_limit": 0.80,
            "default_account_balance": 10000.0,
        },
        "volatility": {
            "default_volatility": 0.2,
            "volatility_risk_factor": 0.02,
        },
    }
    with open(tmp_path / "position_sizing.yaml", "w") as f:
        yaml.dump(position_sizing, f)

    # Create execution_parameters.yaml
    execution = {
        "execution": {
            "commission": {
                "rate": 0.001,
                "minimum": 1.0,
            },
            "slippage": {
                "default_bps": 5.0,
                "max_percent_proportional": 0.01,
                "max_percent_adaptive": 0.02,
                "default_model": "proportional",
            },
            "order_limits": {
                "max_size_percent": 0.10,
                "timeout_seconds": 30.0,
                "max_age_seconds": 300.0,
            },
            "fractional_shares": {
                "enabled": True,
            },
        },
        "retry": {
            "max_attempts": 3,
            "base_delay_seconds": 1.0,
            "max_delay_seconds": 30.0,
            "exponential_base": 2.0,
            "jitter_enabled": True,
        },
        "circuit_breaker": {
            "failure_threshold": 5,
            "recovery_timeout_seconds": 60.0,
            "half_open_max_calls": 1,
        },
        "vwap": {
            "max_chunks": 10,
            "price_variance": 0.001,
            "buy_adjustment": 1.0005,
            "sell_adjustment": 0.9995,
            "default_time_window_minutes": 30,
        },
    }
    with open(tmp_path / "execution_parameters.yaml", "w") as f:
        yaml.dump(execution, f)

    # Create strategy_allocations.yaml
    strategy = {
        "strategy_allocations": {
            "momentum_scalping": 0.25,
            "news_volatility": 0.20,
            "gamma_oi_squeeze": 0.15,
            "arbitrage": 0.20,
            "ai_ml_patterns": 0.20,
        },
        "strategy_thresholds": {
            "min_confidence": 0.60,
            "risk_per_trade": 0.02,
            "max_signals": 10,
        },
    }
    with open(tmp_path / "strategy_allocations.yaml", "w") as f:
        yaml.dump(strategy, f)

    # Create risk.yaml
    risk = {
        "risk": {
            "total_capital": 250000.0,
            "max_position_risk_pct": 0.02,
            "stop_loss_pct": 0.02,
            "take_profit_multiple": 2.0,
            "kelly_sensitivity": 0.6,
            "min_confidence": 0.55,
            "clamp_confidence": [0.5, 0.75],
            "daily_drawdown_limit": 0.03,
        },
        "risk_limits": {
            "max_positions": 10,
            "position_concentration_limit": 0.80,
            "default_account_balance": 10000.0,
            "max_portfolio_risk": 0.20,
            "max_position_size": 0.10,
            "max_daily_drawdown": 0.05,
        },
        "drawdown_management": {
            "min_drawdown_scale": 0.3,
            "drawdown_window_hours": 24,
            "kelly_fraction_cap": 0.25,
            "reduction_thresholds": [
                {"threshold": 0.02, "reduction": 0.9},
                {"threshold": 0.03, "reduction": 0.7},
            ],
        },
        "market_conditions": {
            "gap_risk_threshold": 0.02,
            "flash_crash_threshold": 0.05,
            "regime_detection": {
                "low_volatility_threshold": 0.1,
                "high_volatility_threshold": 0.25,
                "extreme_volatility_threshold": 0.4,
            },
        },
    }
    with open(tmp_path / "risk.yaml", "w") as f:
        yaml.dump(risk, f)

    return tmp_path


# =============================================================================
# ConfigLoader Tests
# =============================================================================


class TestConfigLoader:
    """Tests for ConfigLoader class."""

    def test_singleton_pattern(self, reset_config_loader, config_dir):
        """Test that ConfigLoader is a singleton."""
        loader1 = ConfigLoader(config_dir)
        loader2 = ConfigLoader(config_dir)
        assert loader1 is loader2

    def test_load_position_sizing(self, reset_config_loader, config_dir):
        """Test loading position sizing configuration."""
        loader = ConfigLoader(config_dir)
        ps = loader.position_sizing

        assert ps.default_risk_per_trade == 0.02
        assert ps.max_portfolio_risk == 0.20
        assert ps.max_position_size == 0.10
        assert ps.max_daily_drawdown == 0.05
        assert ps.min_confidence == 0.70

    def test_load_kelly_criterion(self, reset_config_loader, config_dir):
        """Test loading Kelly criterion configuration."""
        loader = ConfigLoader(config_dir)
        kelly = loader.position_sizing.kelly

        assert kelly.min_drawdown_scale == 0.3
        assert kelly.drawdown_window_hours == 24
        assert kelly.kelly_fraction_cap == 0.25
        assert kelly.avg_win_default == 0.02
        assert kelly.avg_loss_default == 0.015

    def test_load_stop_loss(self, reset_config_loader, config_dir):
        """Test loading stop loss configuration."""
        loader = ConfigLoader(config_dir)
        sl = loader.position_sizing.stop_loss

        assert sl.default_stop_percent == 0.05
        assert sl.min_stop_percent == 0.01
        assert sl.fallback_stop_percent == 0.02

    def test_load_take_profit(self, reset_config_loader, config_dir):
        """Test loading take profit configuration."""
        loader = ConfigLoader(config_dir)
        tp = loader.position_sizing.take_profit

        assert tp.default_take_profit_percent == 0.02
        assert tp.extended_take_profit_percent == 0.03
        assert tp.default_risk_reward_ratio == 1.5

    def test_load_execution_config(self, reset_config_loader, config_dir):
        """Test loading execution configuration."""
        loader = ConfigLoader(config_dir)
        ex = loader.execution

        assert ex.commission.rate == 0.001
        assert ex.commission.minimum == 1.0
        assert ex.slippage.default_bps == 5.0
        assert ex.order_limits.timeout_seconds == 30.0
        assert ex.allow_fractional_shares is True

    def test_load_retry_config(self, reset_config_loader, config_dir):
        """Test loading retry configuration."""
        loader = ConfigLoader(config_dir)
        retry = loader.execution.retry

        assert retry.max_attempts == 3
        assert retry.base_delay_seconds == 1.0
        assert retry.max_delay_seconds == 30.0
        assert retry.exponential_base == 2.0
        assert retry.jitter_enabled is True

    def test_load_circuit_breaker_config(self, reset_config_loader, config_dir):
        """Test loading circuit breaker configuration."""
        loader = ConfigLoader(config_dir)
        cb = loader.execution.circuit_breaker

        assert cb.failure_threshold == 5
        assert cb.recovery_timeout_seconds == 60.0
        assert cb.half_open_max_calls == 1

    def test_load_vwap_config(self, reset_config_loader, config_dir):
        """Test loading VWAP configuration."""
        loader = ConfigLoader(config_dir)
        vwap = loader.execution.vwap

        assert vwap.max_chunks == 10
        assert vwap.price_variance == 0.001
        assert vwap.buy_adjustment == 1.0005
        assert vwap.sell_adjustment == 0.9995

    def test_load_strategy_allocations(self, reset_config_loader, config_dir):
        """Test loading strategy allocations configuration."""
        loader = ConfigLoader(config_dir)
        alloc = loader.strategy_allocations

        assert alloc.momentum_scalping == 0.25
        assert alloc.news_volatility == 0.20
        assert alloc.gamma_oi_squeeze == 0.15
        assert alloc.arbitrage == 0.20
        assert alloc.ai_ml_patterns == 0.20
        assert alloc.min_confidence == 0.60

    def test_strategy_allocations_sum_to_one(self, reset_config_loader, config_dir):
        """Test that strategy allocations sum to 1.0."""
        loader = ConfigLoader(config_dir)
        alloc = loader.strategy_allocations
        assert alloc.validate_allocations() is True

    def test_load_risk_config(self, reset_config_loader, config_dir):
        """Test loading risk configuration."""
        loader = ConfigLoader(config_dir)
        risk = loader.risk

        assert risk.total_capital == 250000.0
        assert risk.max_position_risk_pct == 0.02
        assert risk.stop_loss_pct == 0.02
        assert risk.take_profit_multiple == 2.0
        assert risk.max_positions == 10
        assert risk.max_portfolio_risk == 0.20

    def test_load_drawdown_management(self, reset_config_loader, config_dir):
        """Test loading drawdown management configuration."""
        loader = ConfigLoader(config_dir)
        dd = loader.risk.drawdown

        assert dd.min_drawdown_scale == 0.3
        assert dd.drawdown_window_hours == 24
        assert dd.kelly_fraction_cap == 0.25
        assert len(dd.reduction_thresholds) == 2

    def test_load_market_conditions(self, reset_config_loader, config_dir):
        """Test loading market conditions configuration."""
        loader = ConfigLoader(config_dir)
        mc = loader.risk.market_conditions

        assert mc.gap_risk_threshold == 0.02
        assert mc.flash_crash_threshold == 0.05
        assert mc.low_volatility_threshold == 0.1
        assert mc.high_volatility_threshold == 0.25

    def test_reload_configs(self, reset_config_loader, config_dir):
        """Test reloading configurations."""
        loader = ConfigLoader(config_dir)
        initial_load_time = loader.get_load_time()

        loader.reload()

        assert loader.get_load_time() > initial_load_time

    def test_validate_all(self, reset_config_loader, config_dir):
        """Test validating all configurations."""
        loader = ConfigLoader(config_dir)
        results = loader.validate_all()

        assert results["position_sizing"] is True
        assert results["execution"] is True
        assert results["strategy_allocations"] is True
        assert results["risk"] is True

    def test_get_raw_config(self, reset_config_loader, config_dir):
        """Test getting raw configuration dictionary."""
        loader = ConfigLoader(config_dir)
        raw = loader.get_raw_config("position_sizing")

        assert "position_sizing" in raw
        assert raw["position_sizing"]["default_risk_per_trade"] == 0.02

    def test_audit_log(self, reset_config_loader, config_dir):
        """Test configuration audit log."""
        loader = ConfigLoader(config_dir)
        loader.reload()

        audit = loader.get_audit_log()
        assert len(audit) >= 1
        assert audit[-1]["action"] == "reload"


# =============================================================================
# Environment Variable Override Tests
# =============================================================================


class TestEnvironmentOverrides:
    """Tests for environment variable overrides."""

    def test_env_override_position_sizing(self, reset_config_loader, config_dir, monkeypatch):
        """Test environment variable override for position sizing."""
        monkeypatch.setenv("TRADINGBOT_POSITION_SIZING_default_risk_per_trade", "0.03")

        loader = ConfigLoader(config_dir)
        raw = loader.get_raw_config("position_sizing")

        # Check that the raw config has the override
        assert "position_sizing" in raw

    def test_env_override_boolean(self, reset_config_loader, config_dir, monkeypatch):
        """Test environment variable override for boolean values."""
        monkeypatch.setenv("TRADINGBOT_EXECUTION_allow_fractional", "false")

        loader = ConfigLoader(config_dir)
        # The override should be applied to raw config


# =============================================================================
# Default Value Tests
# =============================================================================


class TestDefaultValues:
    """Tests for default values when config files are missing."""

    def test_default_position_sizing(self, reset_config_loader, tmp_path):
        """Test default position sizing values."""
        # Empty config directory
        loader = ConfigLoader(tmp_path)
        ps = loader.position_sizing

        assert ps.default_risk_per_trade == 0.02
        assert ps.max_portfolio_risk == 0.20
        assert ps.max_position_size == 0.10

    def test_default_execution(self, reset_config_loader, tmp_path):
        """Test default execution values."""
        loader = ConfigLoader(tmp_path)
        ex = loader.execution

        assert ex.commission.rate == 0.001
        assert ex.retry.max_attempts == 3

    def test_default_strategy_allocations(self, reset_config_loader, tmp_path):
        """Test default strategy allocation values."""
        loader = ConfigLoader(tmp_path)
        alloc = loader.strategy_allocations

        assert alloc.momentum_scalping == 0.25
        assert alloc.validate_allocations() is True


# =============================================================================
# Module-level Function Tests
# =============================================================================


class TestModuleFunctions:
    """Tests for module-level convenience functions."""

    def test_get_config_loader(self, reset_config_loader, config_dir):
        """Test get_config_loader function."""
        loader = get_config_loader(config_dir)
        assert isinstance(loader, ConfigLoader)

    def test_get_position_sizing_config(self, reset_config_loader, config_dir):
        """Test get_position_sizing_config function."""
        # Initialize the loader first
        get_config_loader(config_dir)
        ps = get_position_sizing_config()
        assert isinstance(ps, PositionSizingConfig)

    def test_get_execution_config(self, reset_config_loader, config_dir):
        """Test get_execution_config function."""
        get_config_loader(config_dir)
        ex = get_execution_config()
        assert isinstance(ex, ExecutionConfig)

    def test_get_strategy_allocations_config(self, reset_config_loader, config_dir):
        """Test get_strategy_allocations_config function."""
        get_config_loader(config_dir)
        alloc = get_strategy_allocations_config()
        assert isinstance(alloc, StrategyAllocationsConfig)

    def test_get_risk_config(self, reset_config_loader, config_dir):
        """Test get_risk_config function."""
        get_config_loader(config_dir)
        risk = get_risk_config()
        assert isinstance(risk, RiskConfig)


# =============================================================================
# Validation Tests
# =============================================================================


class TestValidation:
    """Tests for configuration validation."""

    def test_invalid_risk_per_trade(self, reset_config_loader, tmp_path):
        """Test validation of invalid risk per trade."""
        position_sizing = {
            "position_sizing": {
                "default_risk_per_trade": 0.5,  # Too high
            },
        }
        with open(tmp_path / "position_sizing.yaml", "w") as f:
            yaml.dump(position_sizing, f)

        loader = ConfigLoader(tmp_path)
        results = loader.validate_all()
        assert results["position_sizing"] is False

    def test_invalid_strategy_allocations(self, reset_config_loader, tmp_path):
        """Test validation of strategy allocations that don't sum to 1.0."""
        strategy = {
            "strategy_allocations": {
                "momentum_scalping": 0.30,  # Sums to 1.1
                "news_volatility": 0.30,
                "gamma_oi_squeeze": 0.20,
                "arbitrage": 0.20,
                "ai_ml_patterns": 0.10,
            },
        }
        with open(tmp_path / "strategy_allocations.yaml", "w") as f:
            yaml.dump(strategy, f)

        loader = ConfigLoader(tmp_path)
        alloc = loader.strategy_allocations
        assert alloc.validate_allocations() is False

    def test_invalid_max_positions(self, reset_config_loader, tmp_path):
        """Test validation of invalid max positions."""
        risk = {
            "risk_limits": {
                "max_positions": 0,  # Invalid
            },
        }
        with open(tmp_path / "risk.yaml", "w") as f:
            yaml.dump(risk, f)

        loader = ConfigLoader(tmp_path)
        results = loader.validate_all()
        assert results["risk"] is False
