"""
Advanced Portfolio Optimization Engine

A sophisticated portfolio optimization system that integrates with the existing
Canadian AI Trading Bot to provide modern portfolio theory, machine learning
integration, and real-time optimization capabilities.
"""

__version__ = "1.0.0"
__author__ = "AI Trading Bot Team"

# Core components (will be implemented in later tasks)
# from .core.portfolio_optimizer import PortfolioOptimizer
# from .core.data_processor import MarketDataProcessor
# from .core.correlation_analyzer import CorrelationAnalyzer
# from .core.factor_analyzer import FactorAnalyzer
# from .core.risk_calculator import RiskCalculator
# from .core.transaction_cost_model import TransactionCostModel
# from .core.portfolio_analytics import PortfolioAnalytics

# Data models
from .models.portfolio_state import PortfolioState, Position
from .models.optimization_config import OptimizationConfig, ConstraintSet, OptimizationMethod
from .models.risk_metrics import RiskMetrics
from .models.factor_exposure import FactorExposure
from .models.transaction_cost import TransactionCost

# Interfaces
from .interfaces.optimization_algorithm import IOptimizationAlgorithm
from .interfaces.data_provider import IDataProvider
from .interfaces.risk_model import IRiskModel

# Exceptions
from .exceptions.optimization_errors import (
    OptimizationError,
    DataError,
    ConstraintViolationError,
    ConvergenceError
)

__all__ = [
    # Data models
    'PortfolioState',
    'Position',
    'OptimizationConfig',
    'ConstraintSet',
    'OptimizationMethod',
    'RiskMetrics',
    'FactorExposure',
    'TransactionCost',
    
    # Interfaces
    'IOptimizationAlgorithm',
    'IDataProvider',
    'IRiskModel',
    
    # Exceptions
    'OptimizationError',
    'DataError',
    'ConstraintViolationError',
    'ConvergenceError'
]