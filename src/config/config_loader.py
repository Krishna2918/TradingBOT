"""
Configuration Loader Module
===========================

Loads and validates all YAML configuration files using Pydantic models.
Provides environment variable overrides and centralized config access.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

logger = logging.getLogger(__name__)

# Default config directory
CONFIG_DIR = Path(__file__).parent.parent.parent / "config"


# =============================================================================
# Configuration Data Classes
# =============================================================================


@dataclass
class KellyCriterionConfig:
    """Kelly criterion configuration."""
    min_drawdown_scale: float = 0.3
    drawdown_window_hours: int = 24
    kelly_fraction_cap: float = 0.25
    avg_win_default: float = 0.02
    avg_loss_default: float = 0.015


@dataclass
class StopLossConfig:
    """Stop loss configuration."""
    default_stop_percent: float = 0.05
    min_stop_percent: float = 0.01
    fallback_stop_percent: float = 0.02


@dataclass
class TakeProfitConfig:
    """Take profit configuration."""
    default_take_profit_percent: float = 0.02
    extended_take_profit_percent: float = 0.03
    default_risk_reward_ratio: float = 1.5


@dataclass
class ConfidenceConfig:
    """Confidence configuration."""
    confidence_multiplier: float = 0.8
    signal_strength_factor: float = 0.2


@dataclass
class PortfolioLimitsConfig:
    """Portfolio limits configuration."""
    max_positions: int = 10
    position_concentration_limit: float = 0.80
    default_account_balance: float = 10000.0


@dataclass
class VolatilityConfig:
    """Volatility configuration."""
    default_volatility: float = 0.2
    volatility_risk_factor: float = 0.02


@dataclass
class PositionSizingConfig:
    """Position sizing configuration."""
    default_risk_per_trade: float = 0.02
    max_portfolio_risk: float = 0.20
    max_position_size: float = 0.10
    max_daily_drawdown: float = 0.05
    min_confidence: float = 0.70
    kelly: KellyCriterionConfig = field(default_factory=KellyCriterionConfig)
    stop_loss: StopLossConfig = field(default_factory=StopLossConfig)
    take_profit: TakeProfitConfig = field(default_factory=TakeProfitConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    portfolio_limits: PortfolioLimitsConfig = field(default_factory=PortfolioLimitsConfig)
    volatility: VolatilityConfig = field(default_factory=VolatilityConfig)


@dataclass
class CommissionConfig:
    """Commission configuration."""
    rate: float = 0.001
    minimum: float = 1.0


@dataclass
class SlippageConfig:
    """Slippage configuration."""
    default_bps: float = 5.0
    max_percent_proportional: float = 0.01
    max_percent_adaptive: float = 0.02
    default_model: str = "proportional"


@dataclass
class OrderLimitsConfig:
    """Order limits configuration."""
    max_size_percent: float = 0.10
    timeout_seconds: float = 30.0
    max_age_seconds: float = 300.0


@dataclass
class RetryConfig:
    """Retry configuration."""
    max_attempts: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0
    exponential_base: float = 2.0
    jitter_enabled: bool = True


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5
    recovery_timeout_seconds: float = 60.0
    half_open_max_calls: int = 1


@dataclass
class VWAPConfig:
    """VWAP execution configuration."""
    max_chunks: int = 10
    price_variance: float = 0.001
    buy_adjustment: float = 1.0005
    sell_adjustment: float = 0.9995
    default_time_window_minutes: int = 30


@dataclass
class ExecutionConfig:
    """Execution parameters configuration."""
    commission: CommissionConfig = field(default_factory=CommissionConfig)
    slippage: SlippageConfig = field(default_factory=SlippageConfig)
    order_limits: OrderLimitsConfig = field(default_factory=OrderLimitsConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    vwap: VWAPConfig = field(default_factory=VWAPConfig)
    allow_fractional_shares: bool = True


@dataclass
class StrategyParametersConfig:
    """Individual strategy parameters."""
    enabled: bool = True
    max_leverage: float = 1.0
    max_positions: int = 3
    max_position_pct: float = 0.10


@dataclass
class StrategyAllocationsConfig:
    """Strategy allocations configuration."""
    momentum_scalping: float = 0.25
    news_volatility: float = 0.20
    gamma_oi_squeeze: float = 0.15
    arbitrage: float = 0.20
    ai_ml_patterns: float = 0.20
    min_confidence: float = 0.60
    risk_per_trade: float = 0.02
    max_signals: int = 10

    def validate_allocations(self) -> bool:
        """Validate that allocations sum to 1.0."""
        total = (
            self.momentum_scalping +
            self.news_volatility +
            self.gamma_oi_squeeze +
            self.arbitrage +
            self.ai_ml_patterns
        )
        return abs(total - 1.0) < 0.001


@dataclass
class DrawdownThreshold:
    """Drawdown reduction threshold."""
    threshold: float
    reduction: float


@dataclass
class DrawdownManagementConfig:
    """Drawdown management configuration."""
    min_drawdown_scale: float = 0.3
    drawdown_window_hours: int = 24
    kelly_fraction_cap: float = 0.25
    reduction_thresholds: List[DrawdownThreshold] = field(default_factory=list)


@dataclass
class MarketConditionsConfig:
    """Market conditions configuration."""
    gap_risk_threshold: float = 0.02
    flash_crash_threshold: float = 0.05
    low_volatility_threshold: float = 0.1
    high_volatility_threshold: float = 0.25
    extreme_volatility_threshold: float = 0.4


@dataclass
class RiskConfig:
    """Comprehensive risk configuration."""
    total_capital: float = 250000.0
    max_position_risk_pct: float = 0.02
    stop_loss_pct: float = 0.02
    take_profit_multiple: float = 2.0
    kelly_sensitivity: float = 0.6
    min_confidence: float = 0.55
    clamp_confidence: Tuple[float, float] = (0.5, 0.75)
    daily_drawdown_limit: float = 0.03
    max_positions: int = 10
    max_portfolio_risk: float = 0.20
    max_position_size: float = 0.10
    max_daily_drawdown: float = 0.05
    drawdown: DrawdownManagementConfig = field(default_factory=DrawdownManagementConfig)
    market_conditions: MarketConditionsConfig = field(default_factory=MarketConditionsConfig)


# =============================================================================
# Configuration Loader
# =============================================================================


class ConfigLoader:
    """
    Loads and manages all configuration files.

    Features:
    - YAML config loading
    - Environment variable overrides
    - Validation
    - Caching
    - Audit trail
    """

    _instance: Optional["ConfigLoader"] = None
    _initialized: bool = False

    def __new__(cls, config_dir: Optional[Path] = None) -> "ConfigLoader":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_dir: Optional[Path] = None):
        if self._initialized:
            return

        self.config_dir = config_dir or CONFIG_DIR
        self._raw_configs: Dict[str, Dict] = {}
        self._position_sizing: Optional[PositionSizingConfig] = None
        self._execution: Optional[ExecutionConfig] = None
        self._strategy_allocations: Optional[StrategyAllocationsConfig] = None
        self._risk: Optional[RiskConfig] = None
        self._load_time: Optional[datetime] = None
        self._audit_log: List[Dict[str, Any]] = []

        self._load_all_configs()
        self._initialized = True

    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load a YAML configuration file."""
        filepath = self.config_dir / filename
        try:
            if filepath.exists():
                with open(filepath, "r") as f:
                    data = yaml.safe_load(f) or {}
                logger.debug(f"Loaded config: {filename}")
                return data
            else:
                logger.warning(f"Config file not found: {filepath}")
                return {}
        except Exception as e:
            logger.error(f"Failed to load {filename}: {e}")
            return {}

    def _apply_env_overrides(self, config: Dict, prefix: str) -> Dict:
        """Apply environment variable overrides to config.

        Environment variables should be named: TRADINGBOT_{PREFIX}_{KEY}
        For nested keys, use double underscore: TRADINGBOT_RISK__MAX_POSITIONS
        """
        result = config.copy()
        env_prefix = f"TRADINGBOT_{prefix.upper()}_"

        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                config_key = key[len(env_prefix):].lower()

                # Handle nested keys with double underscore
                if "__" in config_key:
                    parts = config_key.split("__")
                    current = result
                    for part in parts[:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                    final_key = parts[-1]
                else:
                    current = result
                    final_key = config_key

                # Type conversion
                try:
                    if value.lower() in ("true", "false"):
                        current[final_key] = value.lower() == "true"
                    elif "." in value:
                        current[final_key] = float(value)
                    elif value.isdigit():
                        current[final_key] = int(value)
                    else:
                        current[final_key] = value

                    self._log_audit("env_override", f"{key}={value}")
                except (ValueError, KeyError) as e:
                    logger.warning(f"Failed to apply env override {key}: {e}")

        return result

    def _load_all_configs(self) -> None:
        """Load all configuration files."""
        logger.info("Loading all configuration files...")

        # Load raw YAML files
        self._raw_configs["position_sizing"] = self._apply_env_overrides(
            self._load_yaml("position_sizing.yaml"), "POSITION_SIZING"
        )
        self._raw_configs["execution"] = self._apply_env_overrides(
            self._load_yaml("execution_parameters.yaml"), "EXECUTION"
        )
        self._raw_configs["strategy"] = self._apply_env_overrides(
            self._load_yaml("strategy_allocations.yaml"), "STRATEGY"
        )
        self._raw_configs["risk"] = self._apply_env_overrides(
            self._load_yaml("risk.yaml"), "RISK"
        )

        # Parse into typed configs
        self._parse_position_sizing()
        self._parse_execution()
        self._parse_strategy_allocations()
        self._parse_risk()

        self._load_time = datetime.now()
        logger.info("All configurations loaded successfully")

    def _parse_position_sizing(self) -> None:
        """Parse position sizing configuration."""
        raw = self._raw_configs.get("position_sizing", {})
        ps = raw.get("position_sizing", {})
        kelly = raw.get("kelly_criterion", {})
        sl = raw.get("stop_loss", {})
        tp = raw.get("take_profit", {})
        conf = raw.get("confidence", {})
        limits = raw.get("portfolio_limits", {})
        vol = raw.get("volatility", {})

        self._position_sizing = PositionSizingConfig(
            default_risk_per_trade=ps.get("default_risk_per_trade", 0.02),
            max_portfolio_risk=ps.get("max_portfolio_risk", 0.20),
            max_position_size=ps.get("max_position_size", 0.10),
            max_daily_drawdown=ps.get("max_daily_drawdown", 0.05),
            min_confidence=ps.get("min_confidence", 0.70),
            kelly=KellyCriterionConfig(
                min_drawdown_scale=kelly.get("min_drawdown_scale", 0.3),
                drawdown_window_hours=kelly.get("drawdown_window_hours", 24),
                kelly_fraction_cap=kelly.get("kelly_fraction_cap", 0.25),
                avg_win_default=kelly.get("avg_win_default", 0.02),
                avg_loss_default=kelly.get("avg_loss_default", 0.015),
            ),
            stop_loss=StopLossConfig(
                default_stop_percent=sl.get("default_stop_percent", 0.05),
                min_stop_percent=sl.get("min_stop_percent", 0.01),
                fallback_stop_percent=sl.get("fallback_stop_percent", 0.02),
            ),
            take_profit=TakeProfitConfig(
                default_take_profit_percent=tp.get("default_take_profit_percent", 0.02),
                extended_take_profit_percent=tp.get("extended_take_profit_percent", 0.03),
                default_risk_reward_ratio=tp.get("default_risk_reward_ratio", 1.5),
            ),
            confidence=ConfidenceConfig(
                confidence_multiplier=conf.get("confidence_multiplier", 0.8),
                signal_strength_factor=conf.get("signal_strength_factor", 0.2),
            ),
            portfolio_limits=PortfolioLimitsConfig(
                max_positions=limits.get("max_positions", 10),
                position_concentration_limit=limits.get("position_concentration_limit", 0.80),
                default_account_balance=limits.get("default_account_balance", 10000.0),
            ),
            volatility=VolatilityConfig(
                default_volatility=vol.get("default_volatility", 0.2),
                volatility_risk_factor=vol.get("volatility_risk_factor", 0.02),
            ),
        )

    def _parse_execution(self) -> None:
        """Parse execution configuration."""
        raw = self._raw_configs.get("execution", {})
        exec_cfg = raw.get("execution", {})
        comm = exec_cfg.get("commission", {})
        slip = exec_cfg.get("slippage", {})
        limits = exec_cfg.get("order_limits", {})
        frac = exec_cfg.get("fractional_shares", {})
        retry = raw.get("retry", {})
        cb = raw.get("circuit_breaker", {})
        vwap = raw.get("vwap", {})

        self._execution = ExecutionConfig(
            commission=CommissionConfig(
                rate=comm.get("rate", 0.001),
                minimum=comm.get("minimum", 1.0),
            ),
            slippage=SlippageConfig(
                default_bps=slip.get("default_bps", 5.0),
                max_percent_proportional=slip.get("max_percent_proportional", 0.01),
                max_percent_adaptive=slip.get("max_percent_adaptive", 0.02),
                default_model=slip.get("default_model", "proportional"),
            ),
            order_limits=OrderLimitsConfig(
                max_size_percent=limits.get("max_size_percent", 0.10),
                timeout_seconds=limits.get("timeout_seconds", 30.0),
                max_age_seconds=limits.get("max_age_seconds", 300.0),
            ),
            retry=RetryConfig(
                max_attempts=retry.get("max_attempts", 3),
                base_delay_seconds=retry.get("base_delay_seconds", 1.0),
                max_delay_seconds=retry.get("max_delay_seconds", 30.0),
                exponential_base=retry.get("exponential_base", 2.0),
                jitter_enabled=retry.get("jitter_enabled", True),
            ),
            circuit_breaker=CircuitBreakerConfig(
                failure_threshold=cb.get("failure_threshold", 5),
                recovery_timeout_seconds=cb.get("recovery_timeout_seconds", 60.0),
                half_open_max_calls=cb.get("half_open_max_calls", 1),
            ),
            vwap=VWAPConfig(
                max_chunks=vwap.get("max_chunks", 10),
                price_variance=vwap.get("price_variance", 0.001),
                buy_adjustment=vwap.get("buy_adjustment", 1.0005),
                sell_adjustment=vwap.get("sell_adjustment", 0.9995),
                default_time_window_minutes=vwap.get("default_time_window_minutes", 30),
            ),
            allow_fractional_shares=frac.get("enabled", True),
        )

    def _parse_strategy_allocations(self) -> None:
        """Parse strategy allocations configuration."""
        raw = self._raw_configs.get("strategy", {})
        alloc = raw.get("strategy_allocations", {})
        thresh = raw.get("strategy_thresholds", {})

        self._strategy_allocations = StrategyAllocationsConfig(
            momentum_scalping=alloc.get("momentum_scalping", 0.25),
            news_volatility=alloc.get("news_volatility", 0.20),
            gamma_oi_squeeze=alloc.get("gamma_oi_squeeze", 0.15),
            arbitrage=alloc.get("arbitrage", 0.20),
            ai_ml_patterns=alloc.get("ai_ml_patterns", 0.20),
            min_confidence=thresh.get("min_confidence", 0.60),
            risk_per_trade=thresh.get("risk_per_trade", 0.02),
            max_signals=thresh.get("max_signals", 10),
        )

        # Validate allocations sum to 1.0
        if not self._strategy_allocations.validate_allocations():
            logger.warning("Strategy allocations do not sum to 1.0!")

    def _parse_risk(self) -> None:
        """Parse risk configuration."""
        raw = self._raw_configs.get("risk", {})
        risk = raw.get("risk", {})
        limits = raw.get("risk_limits", {})
        dd = raw.get("drawdown_management", {})
        mc = raw.get("market_conditions", {})
        regime = mc.get("regime_detection", {})

        # Parse drawdown thresholds
        thresholds = []
        for t in dd.get("reduction_thresholds", []):
            thresholds.append(DrawdownThreshold(
                threshold=t.get("threshold", 0.0),
                reduction=t.get("reduction", 1.0),
            ))

        clamp = risk.get("clamp_confidence", [0.5, 0.75])
        if isinstance(clamp, list) and len(clamp) == 2:
            clamp_tuple = (clamp[0], clamp[1])
        else:
            clamp_tuple = (0.5, 0.75)

        self._risk = RiskConfig(
            total_capital=risk.get("total_capital", 250000.0),
            max_position_risk_pct=risk.get("max_position_risk_pct", 0.02),
            stop_loss_pct=risk.get("stop_loss_pct", 0.02),
            take_profit_multiple=risk.get("take_profit_multiple", 2.0),
            kelly_sensitivity=risk.get("kelly_sensitivity", 0.6),
            min_confidence=risk.get("min_confidence", 0.55),
            clamp_confidence=clamp_tuple,
            daily_drawdown_limit=risk.get("daily_drawdown_limit", 0.03),
            max_positions=limits.get("max_positions", 10),
            max_portfolio_risk=limits.get("max_portfolio_risk", 0.20),
            max_position_size=limits.get("max_position_size", 0.10),
            max_daily_drawdown=limits.get("max_daily_drawdown", 0.05),
            drawdown=DrawdownManagementConfig(
                min_drawdown_scale=dd.get("min_drawdown_scale", 0.3),
                drawdown_window_hours=dd.get("drawdown_window_hours", 24),
                kelly_fraction_cap=dd.get("kelly_fraction_cap", 0.25),
                reduction_thresholds=thresholds,
            ),
            market_conditions=MarketConditionsConfig(
                gap_risk_threshold=mc.get("gap_risk_threshold", 0.02),
                flash_crash_threshold=mc.get("flash_crash_threshold", 0.05),
                low_volatility_threshold=regime.get("low_volatility_threshold", 0.1),
                high_volatility_threshold=regime.get("high_volatility_threshold", 0.25),
                extreme_volatility_threshold=regime.get("extreme_volatility_threshold", 0.4),
            ),
        )

    def _log_audit(self, action: str, details: str) -> None:
        """Log configuration change to audit trail."""
        self._audit_log.append({
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "details": details,
        })

    def reload(self) -> None:
        """Reload all configurations from disk."""
        logger.info("Reloading all configurations...")
        self._log_audit("reload", "Full configuration reload")
        self._load_all_configs()

    @property
    def position_sizing(self) -> PositionSizingConfig:
        """Get position sizing configuration."""
        if self._position_sizing is None:
            self._parse_position_sizing()
        return self._position_sizing

    @property
    def execution(self) -> ExecutionConfig:
        """Get execution configuration."""
        if self._execution is None:
            self._parse_execution()
        return self._execution

    @property
    def strategy_allocations(self) -> StrategyAllocationsConfig:
        """Get strategy allocations configuration."""
        if self._strategy_allocations is None:
            self._parse_strategy_allocations()
        return self._strategy_allocations

    @property
    def risk(self) -> RiskConfig:
        """Get risk configuration."""
        if self._risk is None:
            self._parse_risk()
        return self._risk

    def get_raw_config(self, name: str) -> Dict[str, Any]:
        """Get raw configuration dictionary by name."""
        return self._raw_configs.get(name, {})

    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Get configuration audit log."""
        return self._audit_log.copy()

    def get_load_time(self) -> Optional[datetime]:
        """Get last configuration load time."""
        return self._load_time

    def validate_all(self) -> Dict[str, bool]:
        """Validate all configurations."""
        results = {
            "position_sizing": True,
            "execution": True,
            "strategy_allocations": True,
            "risk": True,
        }

        # Validate position sizing bounds
        ps = self.position_sizing
        if not (0 < ps.default_risk_per_trade <= 0.1):
            logger.error("default_risk_per_trade must be between 0 and 0.1")
            results["position_sizing"] = False
        if not (0 < ps.max_position_size <= 1.0):
            logger.error("max_position_size must be between 0 and 1.0")
            results["position_sizing"] = False

        # Validate strategy allocations sum
        if not self.strategy_allocations.validate_allocations():
            logger.error("Strategy allocations must sum to 1.0")
            results["strategy_allocations"] = False

        # Validate risk limits
        risk = self.risk
        if risk.max_positions < 1:
            logger.error("max_positions must be at least 1")
            results["risk"] = False

        # Validate execution parameters
        ex = self.execution
        if ex.retry.max_attempts < 1:
            logger.error("retry.max_attempts must be at least 1")
            results["execution"] = False

        return results


# =============================================================================
# Module-level singleton
# =============================================================================


_config_loader: Optional[ConfigLoader] = None


def get_config_loader(config_dir: Optional[Path] = None) -> ConfigLoader:
    """Get the global configuration loader instance."""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader(config_dir)
    return _config_loader


def get_position_sizing_config() -> PositionSizingConfig:
    """Get position sizing configuration."""
    return get_config_loader().position_sizing


def get_execution_config() -> ExecutionConfig:
    """Get execution configuration."""
    return get_config_loader().execution


def get_strategy_allocations_config() -> StrategyAllocationsConfig:
    """Get strategy allocations configuration."""
    return get_config_loader().strategy_allocations


def get_risk_config() -> RiskConfig:
    """Get risk configuration."""
    return get_config_loader().risk


def reload_all_configs() -> None:
    """Reload all configurations from disk."""
    get_config_loader().reload()


__all__ = [
    "ConfigLoader",
    "get_config_loader",
    "get_position_sizing_config",
    "get_execution_config",
    "get_strategy_allocations_config",
    "get_risk_config",
    "reload_all_configs",
    # Config types
    "PositionSizingConfig",
    "ExecutionConfig",
    "StrategyAllocationsConfig",
    "RiskConfig",
    "KellyCriterionConfig",
    "StopLossConfig",
    "TakeProfitConfig",
    "CommissionConfig",
    "SlippageConfig",
    "RetryConfig",
    "CircuitBreakerConfig",
    "VWAPConfig",
]
