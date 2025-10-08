"""
Execution Package

Advanced order execution with VWAP, partial fills, and fractional shares
"""

from .execution_engine import (
    ExecutionEngine,
    Order,
    OrderType,
    OrderSide,
    OrderStatus,
    get_execution_engine
)

__all__ = [
    'ExecutionEngine',
    'Order',
    'OrderType',
    'OrderSide',
    'OrderStatus',
    'get_execution_engine'
]
