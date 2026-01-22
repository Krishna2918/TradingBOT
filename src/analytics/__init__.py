"""Analytics module for trading performance tracking and analysis."""

from src.analytics.performance_analytics import (
    TradeRecord,
    ExecutionMetrics,
    StrategyPerformance,
    DailyPerformance,
    PerformanceSummary,
    PerformanceTracker,
    get_performance_tracker,
)

__all__ = [
    # Data classes
    'TradeRecord',
    'ExecutionMetrics',
    'StrategyPerformance',
    'DailyPerformance',
    'PerformanceSummary',
    # Classes
    'PerformanceTracker',
    # Functions
    'get_performance_tracker',
]
