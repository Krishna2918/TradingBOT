"""
Adaptive Data Collection System

A robust, Alpha Vantage-powered data collection system for US market stocks
with 25 years of historical data, technical indicators, and optimized Parquet storage.
"""

__version__ = "1.0.0"
__author__ = "AI Trading System"

from .config import CollectionConfig
from .interfaces import DataCollector, DataProcessor, StorageManager

__all__ = [
    "CollectionConfig",
    "DataCollector", 
    "DataProcessor",
    "StorageManager"
]