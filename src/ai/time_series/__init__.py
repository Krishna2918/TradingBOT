"""
Time Series Models for Financial Market Prediction

This module contains specialized time series models for financial data,
including ARIMA-GARCH, Prophet, VAR models, and State Space models.
"""

from .arima_garch import ARIMAGARCHPredictor, GARCHVolatilityPredictor
from .prophet_models import ProphetPredictor, ProphetAnomalyDetector
from .var_models import VARPredictor, VECMPredictor
from .state_space import KalmanFilterPredictor, DynamicLinearModel
from .time_series_manager import TimeSeriesModelManager
from .seasonality import SeasonalityDetector, SeasonalDecomposer

__all__ = [
    'ARIMAGARCHPredictor',
    'GARCHVolatilityPredictor',
    'ProphetPredictor',
    'ProphetAnomalyDetector',
    'VARPredictor',
    'VECMPredictor',
    'KalmanFilterPredictor',
    'DynamicLinearModel',
    'TimeSeriesModelManager',
    'SeasonalityDetector',
    'SeasonalDecomposer'
]

