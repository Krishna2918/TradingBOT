"""
Exception classes for the Portfolio Optimization Engine.

This module defines custom exceptions to provide clear error handling
and debugging information across all optimization components.
"""

from .optimization_errors import (
    OptimizationError,
    DataError,
    ConstraintViolationError,
    ConvergenceError
)

__all__ = [
    'OptimizationError',
    'DataError',
    'ConstraintViolationError',
    'ConvergenceError'
]