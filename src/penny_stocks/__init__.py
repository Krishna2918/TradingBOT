"""
Penny Stocks Package

Detection and analysis of penny stocks
"""

from .penny_stock_detector import (
    PennyStockDetector,
    PennyStockProfile,
    get_penny_stock_detector
)

__all__ = [
    'PennyStockDetector',
    'PennyStockProfile',
    'get_penny_stock_detector'
]

