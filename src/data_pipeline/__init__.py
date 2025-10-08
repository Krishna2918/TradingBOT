"""
Data Pipeline Package
Handles data collection, processing, and storage for the trading bot
"""

from .collectors.canadian_market_collector import CanadianMarketCollector

__all__ = [
    'CanadianMarketCollector'
]
