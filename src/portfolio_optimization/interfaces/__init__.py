"""
Interface definitions for the Portfolio Optimization Engine.

This module defines the core interfaces that ensure consistent behavior
across all optimization components and enable easy testing and extensibility.
"""

from .optimization_algorithm import IOptimizationAlgorithm
from .data_provider import IDataProvider
from .risk_model import IRiskModel

__all__ = [
    'IOptimizationAlgorithm',
    'IDataProvider', 
    'IRiskModel'
]