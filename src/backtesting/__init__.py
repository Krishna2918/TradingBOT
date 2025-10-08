"""
Backtesting Package

Strategy validation and performance analysis
"""

from .backtest_engine import (
    BacktestEngine,
    BacktestResult,
    get_backtest_engine
)

__all__ = [
    'BacktestEngine',
    'BacktestResult',
    'get_backtest_engine'
]

