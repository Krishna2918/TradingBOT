"""
Advanced ML Predictive Models

This module contains advanced machine learning models for predicting market crashes,
bubbles, regime shifts, volatility, and correlation breakdowns.

Components:
- CrashDetector: Market crash prediction models
- BubbleDetector: Asset bubble detection
- RegimePredictor: Market regime shift prediction
- VolatilityForecaster: Advanced volatility forecasting
- CorrelationAnalyzer: Correlation breakdown detection

Author: AI Trading System
Version: 1.0.0
"""

from .crash_detection import CrashDetector
from .bubble_detection import BubbleDetector
from .regime_prediction import RegimePredictor
from .volatility_forecasting import VolatilityForecaster
from .correlation_analysis import CorrelationAnalyzer

__all__ = [
    "CrashDetector",
    "BubbleDetector", 
    "RegimePredictor",
    "VolatilityForecaster",
    "CorrelationAnalyzer"
]
