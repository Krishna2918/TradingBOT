    max_position = settings.capital.max_position_pct_demo
    rsi_threshold = settings.signals.rsi_oversold
"""

import os
from typing import List, Optional
from pathlib import Path
from pydantic import BaseModel, Field
import yaml


class MarketConfig(BaseModel):
    """Market and symbol universe configuration"""
    name: str = "Canadian Market (TSX/TSXV)"
    primary_symbols: List[str] = []
    use_full_universe: bool = True
    max_universe_size: int = 200


class CapitalConfig(BaseModel):
    """Capital and position sizing configuration"""
    initial_capital_demo: float = 100.0
    initial_capital_live: float = 10000.0
    min_capital: float = 10.0
    
    max_position_pct_demo: float = 0.05
    max_position_pct_live: float = 0.03
    default_position_size_demo: float = 0.02
    default_position_size_live: float = 0.015
    
    min_cash_reserve_pct: float = 0.10
    cash_rebalance_threshold: float = 0.30


class RiskConfig(BaseModel):
    """Risk management and kill switch configuration"""
    max_drawdown_pct: float = 0.10
    max_daily_loss_pct: float = 0.05
    min_sharpe_ratio: float = -0.5
    
    profit_target_pct: float = 0.02
    stop_loss_pct: float = 0.02
    trailing_stop_pct: float = 0.015
    
    high_volatility_threshold: float = 0.03
    reduce_size_on_high_vol: float = 0.5


class SignalsConfig(BaseModel):
    """AI signal thresholds and confidence levels"""
    min_ai_model_confidence: float = 0.70
    min_signal_confidence_demo: float = 0.45
    min_signal_confidence_live: float = 0.55
    max_confidence_cap: float = 0.95
    
    sentiment_strong_positive: float = 0.5
    sentiment_moderate_positive: float = 0.1
    sentiment_neutral_lower: float = -0.1
    sentiment_neutral_upper: float = 0.1
    sentiment_moderate_negative: float = -0.5
    sentiment_strong_negative: float = -0.5
    
    rsi_oversold: int = 45
    rsi_neutral_lower: int = 45
    rsi_neutral_upper: int = 55
    rsi_overbought: int = 60
    
    options_call_put_ratio: float = 2.0
    options_min_confidence: float = 0.5
    
    chatgpt_weight: float = 0.3
    original_signals_weight: float = 0.7
    chatgpt_min_threshold: float = 0.5
    combined_decision_threshold: float = 1.0
    
    demo_confidence_multiplier: float = 1.15
    demo_enabled: bool = True


class DataConfig(BaseModel):
    """Data fetching intervals and batch sizes"""
    trading_interval_seconds: int = 5
    chart_update_interval_seconds: int = 2
    log_monitor_interval_seconds: int = 5
    holdings_update_interval_seconds: int = 3
    
    questrade_batch_size: int = 10
    yahoo_batch_size: int = 5
    yahoo_delay_seconds: float = 1.0
    
    cache_ttl_seconds: int = 60
    redis_enabled: bool = False


class QuestradeConfig(BaseModel):
    """Questrade broker configuration"""
    enabled: bool = True
    token_cache_path: str = "config/questrade_token_cache.json"
    rate_limit_per_second: int = 100


class YahooConfig(BaseModel):
    """Yahoo Finance configuration"""
    enabled: bool = True
    rate_limit_per_second: int = 10
    retry_attempts: int = 3


class BrokerConfig(BaseModel):
    """Broker configuration"""
    primary: str = "questrade"
    fallback: str = "yahoo"
    questrade: QuestradeConfig = QuestradeConfig()
    yahoo: YahooConfig = YahooConfig()


class UIConfig(BaseModel):
    """UI and display configuration"""
    theme: str = "darkly"
    
    app_title: str = "AI Trading Dashboard"
    app_subtitle: str = "Canadian Market (TSX/TSXV)"
    demo_mode_badge: str = "🔬 DEMO MODE"
    live_mode_badge: str = "⚡ LIVE TRADING"
    
    chart_height: int = 400
    chart_noise_amplitude: float = 0.005
    enable_chart_noise: bool = True
    
    alerts_max_display: int = 10
    enable_audio_alerts: bool = False
    
    trades_table_page_size: int = 10
    holdings_table_page_size: int = 10


class LearningConfig(BaseModel):
    """Learning and adaptation configuration"""
    enable_learning: bool = True
    mistake_log_max_size: int = 100
    success_pattern_max_size: int = 100
    
    min_pattern_confidence: float = 0.6
    similar_pattern_threshold: float = 0.8


class LoggingConfig(BaseModel):
    """Logging configuration"""
    level: str = "INFO"
    ai_activity_log_path: str = "logs/ai_activity.log"
    trade_log_path: str = "logs/trades.log"
    error_log_path: str = "logs/errors.log"
    
    max_log_size_mb: int = 100
    backup_count: int = 5


class ModeConfig(BaseModel):
    """Mode-specific configuration"""
    aggressive_trading: bool = True
    enable_all_signals: bool = True
    ignore_risk_limits: bool = False
    strict_risk_management: bool = False
    require_multi_signal_confirmation: bool = False


class ModesConfig(BaseModel):
    """Demo and live mode configurations"""
    demo: ModeConfig = ModeConfig(
        aggressive_trading=True,
        enable_all_signals=True,
        ignore_risk_limits=False
    )
    live: ModeConfig = ModeConfig(
        aggressive_trading=False,
        enable_all_signals=False,
        strict_risk_management=True,
        require_multi_signal_confirmation=True
    )


class FeaturesConfig(BaseModel):
    """Feature flags"""
    enable_regime_detection: bool = True
    enable_sentiment_analysis: bool = True
    enable_options_flow: bool = True
    enable_chatgpt_integration: bool = False
    enable_penny_stocks: bool = True
    enable_crypto_related: bool = True
    enable_background_worker: bool = False


class Settings(BaseModel):
    """Main configuration class"""
    market: MarketConfig = MarketConfig()
    capital: CapitalConfig = CapitalConfig()
    risk: RiskConfig = RiskConfig()
    signals: SignalsConfig = SignalsConfig()
    data: DataConfig = DataConfig()
    broker: BrokerConfig = BrokerConfig()
    ui: UIConfig = UIConfig()
    learning: LearningConfig = LearningConfig()
    logging: LoggingConfig = LoggingConfig()
    modes: ModesConfig = ModesConfig()
    features: FeaturesConfig = FeaturesConfig()
    
    class Config:
        # Pydantic v2 configuration
        pass
    
    @classmethod
    def load_from_yaml(cls, config_path: str = "config/trading_config.yaml") -> "Settings":
        """Load configuration from YAML file"""
        config_file = Path(config_path)
        
        if not config_file.exists():
            print(f"⚠️ Warning: Config file not found at {config_path}, using defaults")
            return cls()
        
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        
        if not config_data:
            print(f"⚠️ Warning: Empty config file at {config_path}, using defaults")
            return cls()
        
        # Create nested configuration objects
        return cls(
            market=MarketConfig(**config_data.get('market', {})),
            capital=CapitalConfig(**config_data.get('capital', {})),
            risk=RiskConfig(**config_data.get('risk', {})),
            signals=SignalsConfig(**config_data.get('signals', {})),
            data=DataConfig(**config_data.get('data', {})),
            broker=BrokerConfig(
                primary=config_data.get('broker', {}).get('primary', 'questrade'),
                fallback=config_data.get('broker', {}).get('fallback', 'yahoo'),
                questrade=QuestradeConfig(**config_data.get('broker', {}).get('questrade', {})),
                yahoo=YahooConfig(**config_data.get('broker', {}).get('yahoo', {}))
            ),
            ui=UIConfig(**config_data.get('ui', {})),
            learning=LearningConfig(**config_data.get('learning', {})),
            logging=LoggingConfig(**config_data.get('logging', {})),
            modes=ModesConfig(
                demo=ModeConfig(**config_data.get('modes', {}).get('demo', {})),
                live=ModeConfig(**config_data.get('modes', {}).get('live', {}))
            ),
            features=FeaturesConfig(**config_data.get('features', {}))
        )
    
    def get_mode_config(self, mode: str) -> ModeConfig:
        """Get configuration for specific mode (demo/live)"""
        if mode.lower() == 'demo':
            return self.modes.demo
        elif mode.lower() == 'live':
            return self.modes.live
        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'demo' or 'live'")
    
    def get_position_size(self, mode: str) -> float:
        """Get default position size for mode"""
        if mode.lower() == 'demo':
            return self.capital.default_position_size_demo
        return self.capital.default_position_size_live
    
    def get_max_position_pct(self, mode: str) -> float:
        """Get max position percentage for mode"""
        if mode.lower() == 'demo':
            return self.capital.max_position_pct_demo
        return self.capital.max_position_pct_live
    
    def get_min_signal_confidence(self, mode: str) -> float:
        """Get minimum signal confidence for mode"""
        if mode.lower() == 'demo':
            return self.signals.min_signal_confidence_demo
        return self.signals.min_signal_confidence_live


# Load configuration from YAML (singleton pattern)
settings = Settings.load_from_yaml()

# Print configuration summary on import
if __name__ != "__main__":
    print("\n" + "="*80)
    print("🔧 CONFIGURATION LOADED")
    print("="*80)
    print(f"Market: {settings.market.name}")
    print(f"Trading Interval: {settings.data.trading_interval_seconds}s")
    print(f"Primary Broker: {settings.broker.primary}")
    print(f"Demo Position Size: {settings.capital.default_position_size_demo*100:.1f}%")
    print(f"Live Position Size: {settings.capital.default_position_size_live*100:.1f}%")
    print(f"Kill Switch: {settings.risk.max_drawdown_pct*100:.0f}% max drawdown")
    print("="*80 + "\n")

