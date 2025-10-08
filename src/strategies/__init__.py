"""
Trading Strategies Package
Contains all 5 trading strategies for the Canadian market trading bot
"""

from .strategy_manager import StrategyManager
from .momentum_scalping import MomentumScalpingStrategy, MomentumSignal
from .news_volatility import NewsVolatilityStrategy, NewsSignal
from .gamma_oi_squeeze import GammaOISqueezeStrategy, GammaOISignal
from .arbitrage import ArbitrageStrategy, ArbitrageSignal
from .ai_ml_patterns import AIMLPatternStrategy, MLSignal

__all__ = [
    'StrategyManager',
    'MomentumScalpingStrategy',
    'MomentumSignal',
    'NewsVolatilityStrategy',
    'NewsSignal',
    'GammaOISqueezeStrategy',
    'GammaOISignal',
    'ArbitrageStrategy',
    'ArbitrageSignal',
    'AIMLPatternStrategy',
    'MLSignal'
]
