"""
Unit tests for time series models.

This module contains comprehensive tests for all time series models
including ARIMA-GARCH, Prophet, VAR, VECM, and State Space models.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from ai.time_series.arima_garch import ARIMAGARCHPredictor, GARCHVolatilityPredictor
from ai.time_series.prophet_models import ProphetPredictor, ProphetAnomalyDetector
from ai.time_series.var_models import VARPredictor, VECMPredictor
from ai.time_series.state_space import KalmanFilterPredictor, DynamicLinearModel
from ai.time_series.seasonality import SeasonalityDetector, SeasonalDecomposer
from ai.time_series.time_series_manager import TimeSeriesModelManager


# Module-level fixtures available to all test classes
@pytest.fixture
def sample_data():
    """Create sample market data for testing."""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'date': dates,
        'close': 100 + np.random.randn(100).cumsum() * 0.1
    })
    return data


@pytest.fixture
def multi_asset_data():
    """Create multi-asset data for testing."""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'date': dates,
        'asset1': 100 + np.random.randn(100).cumsum() * 0.1,
        'asset2': 200 + np.random.randn(100).cumsum() * 0.1,
        'asset3': 300 + np.random.randn(100).cumsum() * 0.1
    })
    return data


class TestARIMAGARCHPredictor:
    """Test ARIMA-GARCH prediction model."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample market data for testing."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'date': dates,
            'close': 100 + np.random.randn(100).cumsum() * 0.1
        })
        return data
    
    @pytest.fixture
    def arima_garch_model(self):
        """Create ARIMA-GARCH model instance."""
        return ARIMAGARCHPredictor(
            arima_order=(1, 1, 1),
            garch_order=(1, 1),
            garch_model='GARCH',
            distribution='normal'
        )
    
    def test_arima_garch_initialization(self, arima_garch_model):
        """Test ARIMA-GARCH model initialization."""
        assert arima_garch_model.arima_order == (1, 1, 1)
        assert arima_garch_model.garch_order == (1, 1)
        assert arima_garch_model.garch_model == 'GARCH'
        assert arima_garch_model.distribution == 'normal'
        assert not arima_garch_model.is_fitted
    
    def test_prepare_data(self, arima_garch_model, sample_data):
        """Test data preparation."""
        returns = arima_garch_model.prepare_data(sample_data, 'close')
        
        assert len(returns) == len(sample_data) - 1  # One less due to log returns
        assert not returns.isnull().any()
    
    @patch('statsmodels.tsa.arima.model.ARIMA')
    @patch('arch.arch_model')
    def test_fit_arima_garch(self, mock_arch, mock_arima, arima_garch_model, sample_data):
        """Test ARIMA-GARCH model fitting."""
        # Mock ARIMA model
        mock_arima_instance = Mock()
        mock_arima.return_value = mock_arima_instance
        mock_arima_fitted = Mock()
        mock_arima_instance.fit.return_value = mock_arima_fitted
        mock_arima_fitted.resid = pd.Series(np.random.randn(99))
        mock_arima_fitted.aic = 100.0
        mock_arima_fitted.bic = 105.0
        mock_arima_fitted.llf = -50.0
        mock_arima_fitted.params = pd.Series([0.1, 0.2, 0.3])
        mock_arima_fitted.pvalues = pd.Series([0.01, 0.02, 0.03])
        mock_arima_fitted.fittedvalues = pd.Series(np.random.randn(99))
        
        # Mock GARCH model
        mock_garch_instance = Mock()
        mock_arch.return_value = mock_garch_instance
        mock_garch_fitted = Mock()
        mock_garch_instance.fit.return_value = mock_garch_fitted
        mock_garch_fitted.conditional_volatility = pd.Series(np.random.randn(99))
        mock_garch_fitted.aic = 95.0
        mock_garch_fitted.bic = 100.0
        mock_garch_fitted.loglikelihood = -45.0
        mock_garch_fitted.params = pd.Series([0.1, 0.2, 0.3])
        mock_garch_fitted.pvalues = pd.Series([0.01, 0.02, 0.03])
        
        # Fit model
        results = arima_garch_model.fit(sample_data, 'close')
        
        assert 'arima_results' in results
        assert 'garch_results' in results
        assert 'stationarity_results' in results
        assert arima_garch_model.is_fitted
    
    def test_predict_next_return(self, arima_garch_model, sample_data):
        """Test next return prediction."""
        # Mock fitted model
        arima_garch_model.is_fitted = True
        arima_garch_model.arima_model = Mock()
        arima_garch_model.garch_model_fitted = Mock()
        
        # Mock forecast
        mock_forecast = Mock()
        mock_forecast.values = np.array([0.01])
        arima_garch_model.arima_model.fit.return_value.forecast.return_value = mock_forecast
        
        mock_ci = Mock()
        mock_ci.conf_int.return_value = pd.DataFrame({
            'lower': [0.005],
            'upper': [0.015]
        })
        arima_garch_model.arima_model.fit.return_value.get_forecast.return_value = mock_ci
        
        # Mock GARCH forecast
        mock_garch_forecast = Mock()
        mock_garch_forecast.variance.values = np.array([[0.0001]])
        arima_garch_model.garch_model_fitted.forecast.return_value = mock_garch_forecast
        
        # Make prediction
        result = arima_garch_model.predict_next_return(sample_data, 'close')
        
        assert 'predicted_return' in result
        assert 'volatility' in result
        assert 'confidence_interval' in result
        assert 'model_type' in result
        assert result['model_type'] == 'ARIMA-GARCH'


class TestGARCHVolatilityPredictor:
    """Test GARCH volatility prediction model."""
    
    @pytest.fixture
    def garch_model(self):
        """Create GARCH model instance."""
        return GARCHVolatilityPredictor(
            garch_order=(1, 1),
            garch_model='GARCH',
            distribution='normal'
        )
    
    def test_garch_initialization(self, garch_model):
        """Test GARCH model initialization."""
        assert garch_model.garch_order == (1, 1)
        assert garch_model.garch_model == 'GARCH'
        assert garch_model.distribution == 'normal'
        assert not garch_model.is_fitted
    
    @patch('arch.arch_model')
    def test_fit_garch(self, mock_arch, garch_model, sample_data):
        """Test GARCH model fitting."""
        # Mock GARCH model
        mock_garch_instance = Mock()
        mock_arch.return_value = mock_garch_instance
        mock_garch_fitted = Mock()
        mock_garch_instance.fit.return_value = mock_garch_fitted
        mock_garch_fitted.conditional_volatility = pd.Series(np.random.randn(99))
        mock_garch_fitted.aic = 95.0
        mock_garch_fitted.bic = 100.0
        mock_garch_fitted.loglikelihood = -45.0
        mock_garch_fitted.params = pd.Series([0.1, 0.2, 0.3])
        mock_garch_fitted.pvalues = pd.Series([0.01, 0.02, 0.03])
        
        # Fit model
        results = garch_model.fit(sample_data, 'close')
        
        assert 'model_type' in results
        assert 'order' in results
        assert 'distribution' in results
        assert garch_model.is_fitted
    
    def test_predict_next_volatility(self, garch_model, sample_data):
        """Test next volatility prediction."""
        # Mock fitted model
        garch_model.is_fitted = True
        garch_model.model = Mock()
        
        # Mock forecast
        mock_forecast = Mock()
        mock_forecast.variance.values = np.array([[0.0001]])
        garch_model.model.forecast.return_value = mock_forecast
        
        # Make prediction
        result = garch_model.predict_next_volatility(sample_data, 'close')
        
        assert 'predicted_volatility' in result
        assert 'confidence_interval' in result
        assert 'model_type' in result
        assert result['model_type'] == 'GARCH-GARCH'


class TestProphetPredictor:
    """Test Prophet prediction model."""
    
    @pytest.fixture
    def prophet_model(self):
        """Create Prophet model instance."""
        return ProphetPredictor(
            growth='linear',
            seasonality_mode='additive',
            yearly_seasonality=True,
            weekly_seasonality=True
        )
    
    def test_prophet_initialization(self, prophet_model):
        """Test Prophet model initialization."""
        assert prophet_model.growth == 'linear'
        assert prophet_model.seasonality_mode == 'additive'
        assert prophet_model.yearly_seasonality is True
        assert prophet_model.weekly_seasonality is True
        assert not prophet_model.is_fitted
    
    def test_prepare_data(self, prophet_model, sample_data):
        """Test Prophet data preparation."""
        prophet_data = prophet_model.prepare_data(sample_data, 'date', 'close')
        
        assert 'ds' in prophet_data.columns
        assert 'y' in prophet_data.columns
        assert len(prophet_data) == len(sample_data)
    
    @patch('prophet.Prophet')
    def test_fit_prophet(self, mock_prophet, prophet_model, sample_data):
        """Test Prophet model fitting."""
        # Mock Prophet model
        mock_prophet_instance = Mock()
        mock_prophet.return_value = mock_prophet_instance
        mock_prophet_instance.fit.return_value = None
        
        # Mock make_future_dataframe and predict
        mock_future = pd.DataFrame({'ds': sample_data['date']})
        mock_prophet_instance.make_future_dataframe.return_value = mock_future
        mock_prophet_instance.predict.return_value = pd.DataFrame({
            'ds': sample_data['date'],
            'yhat': np.random.randn(100)
        })
        
        # Fit model
        results = prophet_model.fit(sample_data, 'date', 'close')
        
        assert 'model_params' in results
        assert 'data_info' in results
        assert 'seasonality_detected' in results
        assert prophet_model.is_fitted
    
    def test_predict_next_value(self, prophet_model, sample_data):
        """Test next value prediction."""
        # Mock fitted model
        prophet_model.is_fitted = True
        prophet_model.model = Mock()
        
        # Mock forecast
        mock_future = pd.DataFrame({'ds': [sample_data['date'].iloc[-1] + timedelta(days=1)]})
        prophet_model.model.make_future_dataframe.return_value = mock_future
        prophet_model.model.predict.return_value = pd.DataFrame({
            'ds': mock_future['ds'],
            'yhat': [101.5],
            'yhat_lower': [100.5],
            'yhat_upper': [102.5]
        })
        
        # Make prediction
        result = prophet_model.predict_next_value(sample_data, 'date', 'close')
        
        assert 'predicted_value' in result
        assert 'confidence_interval' in result
        assert 'model_type' in result
        assert result['model_type'] == 'Prophet'


class TestProphetAnomalyDetector:
    """Test Prophet anomaly detection model."""
    
    @pytest.fixture
    def prophet_anomaly_detector(self):
        """Create Prophet anomaly detector instance."""
        return ProphetAnomalyDetector(
            anomaly_threshold=2.0,
            confidence_level=0.95
        )
    
    def test_prophet_anomaly_initialization(self, prophet_anomaly_detector):
        """Test Prophet anomaly detector initialization."""
        assert prophet_anomaly_detector.anomaly_threshold == 2.0
        assert prophet_anomaly_detector.confidence_level == 0.95
        assert not prophet_anomaly_detector.is_fitted
    
    def test_detect_anomalies(self, prophet_anomaly_detector, sample_data):
        """Test anomaly detection."""
        # Mock fitted model
        prophet_anomaly_detector.is_fitted = True
        prophet_anomaly_detector.prophet_model = Mock()
        prophet_anomaly_detector.prophet_model.model = Mock()
        
        # Mock Prophet data preparation
        prophet_data = pd.DataFrame({
            'ds': sample_data['date'],
            'y': sample_data['close']
        })
        prophet_anomaly_detector.prophet_model.prepare_data.return_value = prophet_data
        
        # Mock forecast
        mock_future = pd.DataFrame({'ds': sample_data['date']})
        prophet_anomaly_detector.prophet_model.model.make_future_dataframe.return_value = mock_future
        prophet_anomaly_detector.prophet_model.model.predict.return_value = pd.DataFrame({
            'ds': sample_data['date'],
            'yhat': sample_data['close'] + np.random.randn(100) * 0.1
        })
        
        # Detect anomalies
        result = prophet_anomaly_detector.detect_anomalies(sample_data, 'date', 'close')
        
        assert 'anomalies_detected' in result
        assert 'total_samples' in result
        assert 'anomaly_rate' in result
        assert 'anomaly_threshold' in result


class TestVARPredictor:
    """Test VAR prediction model."""
    
    @pytest.fixture
    def var_model(self):
        """Create VAR model instance."""
        return VARPredictor(
            maxlags=15,
            ic='aic'
        )
    
    @pytest.fixture
    def multi_asset_data(self):
        """Create multi-asset data for testing."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'date': dates,
            'asset1': 100 + np.random.randn(100).cumsum() * 0.1,
            'asset2': 200 + np.random.randn(100).cumsum() * 0.1,
            'asset3': 300 + np.random.randn(100).cumsum() * 0.1
        })
        return data
    
    def test_var_initialization(self, var_model):
        """Test VAR model initialization."""
        assert var_model.maxlags == 15
        assert var_model.ic == 'aic'
        assert not var_model.is_fitted
    
    def test_prepare_data(self, var_model, multi_asset_data):
        """Test VAR data preparation."""
        variables = ['asset1', 'asset2', 'asset3']
        var_data = var_model.prepare_data(multi_asset_data, variables, returns=True)
        
        assert len(var_data.columns) == 3
        assert len(var_data) == len(multi_asset_data) - 1  # One less due to log returns
    
    @patch('statsmodels.tsa.vector_ar.var_model.VAR')
    def test_fit_var(self, mock_var, var_model, multi_asset_data):
        """Test VAR model fitting."""
        # Mock VAR model
        mock_var_instance = Mock()
        mock_var.return_value = mock_var_instance
        mock_var_fitted = Mock()
        mock_var_instance.fit.return_value = mock_var_fitted
        mock_var_fitted.aic = 100.0
        mock_var_fitted.bic = 105.0
        mock_var_fitted.hqic = 102.0
        mock_var_fitted.fpe = 1.05
        mock_var_fitted.llf = -50.0
        mock_var_fitted.summary.return_value = "VAR Model Summary"
        
        # Mock lag selection
        mock_lag_results = Mock()
        mock_lag_results.aic = 2
        mock_var_instance.select_order.return_value = mock_lag_results
        
        # Fit model
        variables = ['asset1', 'asset2', 'asset3']
        results = var_model.fit(multi_asset_data, variables)
        
        assert 'model_info' in results
        assert 'stationarity_results' in results
        assert 'model_summary' in results
        assert var_model.is_fitted
    
    def test_predict_next_values(self, var_model, multi_asset_data):
        """Test next values prediction."""
        # Mock fitted model
        var_model.is_fitted = True
        var_model.fitted_model = Mock()
        var_model.variable_names = ['asset1', 'asset2', 'asset3']
        
        # Mock forecast
        mock_forecast = np.array([[0.01, 0.02, 0.03]])
        var_model.fitted_model.forecast.return_value = mock_forecast
        
        # Mock forecast interval
        mock_ci = (
            np.array([[0.005, 0.015, 0.025]]),  # Lower bound
            np.array([[0.015, 0.025, 0.035]])   # Upper bound
        )
        var_model.fitted_model.forecast_interval.return_value = mock_ci
        
        # Make prediction
        variables = ['asset1', 'asset2', 'asset3']
        result = var_model.predict_next_values(multi_asset_data, variables)
        
        assert 'predictions' in result
        assert 'model_type' in result
        assert result['model_type'] == 'VAR'
        assert len(result['predictions']) == 3


class TestVECMPredictor:
    """Test VECM prediction model."""
    
    @pytest.fixture
    def vecm_model(self):
        """Create VECM model instance."""
        return VECMPredictor(
            k_ar_diff=1,
            coint_rank=None,
            deterministic='ci'
        )
    
    def test_vecm_initialization(self, vecm_model):
        """Test VECM model initialization."""
        assert vecm_model.k_ar_diff == 1
        assert vecm_model.coint_rank is None
        assert vecm_model.deterministic == 'ci'
        assert not vecm_model.is_fitted
    
    @patch('statsmodels.tsa.vector_ar.vecm.VECM')
    def test_fit_vecm(self, mock_vecm, vecm_model, multi_asset_data):
        """Test VECM model fitting."""
        # Mock VECM model
        mock_vecm_instance = Mock()
        mock_vecm.return_value = mock_vecm_instance
        mock_vecm_fitted = Mock()
        mock_vecm_instance.fit.return_value = mock_vecm_fitted
        mock_vecm_fitted.aic = 95.0
        mock_vecm_fitted.bic = 100.0
        mock_vecm_fitted.hqic = 97.0
        mock_vecm_fitted.llf = -45.0
        
        # Fit model
        variables = ['asset1', 'asset2', 'asset3']
        results = vecm_model.fit(multi_asset_data, variables)

        assert 'model_info' in results
        assert 'cointegration_results' in results
        assert vecm_model.is_fitted
    
    def test_predict_next_values(self, vecm_model, multi_asset_data):
        """Test next values prediction."""
        # Mock fitted model
        vecm_model.is_fitted = True
        vecm_model.fitted_model = Mock()
        vecm_model.variable_names = ['asset1', 'asset2', 'asset3']

        # Mock forecast
        mock_forecast = np.array([[0.01, 0.02, 0.03]])
        vecm_model.fitted_model.forecast.return_value = mock_forecast

        # Make prediction
        variables = ['asset1', 'asset2', 'asset3']
        result = vecm_model.predict_next_values(multi_asset_data, variables)

        assert 'forecast' in result
        assert 'forecast_periods' in result
        assert 'variables' in result


class TestKalmanFilterPredictor:
    """Test Kalman Filter prediction model."""
    
    @pytest.fixture
    def kalman_model(self):
        """Create Kalman Filter model instance."""
        return KalmanFilterPredictor(
            state_dim=2,
            observation_dim=1
        )
    
    def test_kalman_initialization(self, kalman_model):
        """Test Kalman Filter initialization."""
        assert kalman_model.state_dim == 2
        assert kalman_model.observation_dim == 1
        assert not kalman_model.is_fitted
    
    @patch('statsmodels.tsa.statespace.structural.UnobservedComponents')
    def test_fit_kalman(self, mock_uc, kalman_model, sample_data):
        """Test Kalman Filter model fitting."""
        # Mock UnobservedComponents model
        mock_uc_instance = Mock()
        mock_uc.return_value = mock_uc_instance
        mock_uc_fitted = Mock()
        mock_uc_instance.fit.return_value = mock_uc_fitted
        mock_uc_fitted.aic = 90.0
        mock_uc_fitted.bic = 95.0
        mock_uc_fitted.hqic = 92.0
        mock_uc_fitted.llf = -40.0
        mock_uc_fitted.smoothed_state = np.random.randn(100, 2)
        
        # Fit model
        results = kalman_model.fit(sample_data, 'close')
        
        assert 'model_info' in results
        assert 'aic' in results
        assert 'bic' in results
        assert kalman_model.is_fitted
    
    def test_predict_next_value(self, kalman_model, sample_data):
        """Test next value prediction."""
        # Mock fitted model
        kalman_model.is_fitted = True
        kalman_model.fitted_model = Mock()
        
        # Mock forecast
        mock_forecast = pd.Series([101.5])
        kalman_model.fitted_model.forecast.return_value = mock_forecast
        
        # Mock confidence interval
        mock_ci = pd.DataFrame({
            'lower': [100.5],
            'upper': [102.5]
        })
        mock_forecast_obj = Mock()
        mock_forecast_obj.conf_int.return_value = mock_ci
        kalman_model.fitted_model.get_forecast.return_value = mock_forecast_obj
        
        # Make prediction
        result = kalman_model.predict_next_value(sample_data, 'close')
        
        assert 'predicted_value' in result
        assert 'confidence_interval' in result
        assert 'model_type' in result
        assert result['model_type'] == 'Kalman Filter'


class TestDynamicLinearModel:
    """Test Dynamic Linear Model."""
    
    @pytest.fixture
    def dlm_model(self):
        """Create Dynamic Linear Model instance."""
        return DynamicLinearModel(
            k_factors=1,
            factor_order=1
        )
    
    def test_dlm_initialization(self, dlm_model):
        """Test Dynamic Linear Model initialization."""
        assert dlm_model.k_factors == 1
        assert dlm_model.factor_order == 1
        assert not dlm_model.is_fitted
    
    @patch('statsmodels.tsa.statespace.dynamic_factor.DynamicFactor')
    def test_fit_dlm(self, mock_df, dlm_model, multi_asset_data):
        """Test Dynamic Linear Model fitting."""
        # Mock DynamicFactor model
        mock_df_instance = Mock()
        mock_df.return_value = mock_df_instance
        mock_df_fitted = Mock()
        mock_df_instance.fit.return_value = mock_df_fitted
        mock_df_fitted.aic = 85.0
        mock_df_fitted.bic = 90.0
        mock_df_fitted.hqic = 87.0
        mock_df_fitted.llf = -35.0
        mock_df_fitted.factors.filtered = np.random.randn(100, 1)
        mock_df_fitted.params = np.array([0.1, 0.2, 0.3])
        mock_df_fitted.model.endog_names = ['asset1', 'asset2', 'asset3']
        
        # Fit model
        variables = ['asset1', 'asset2', 'asset3']
        results = dlm_model.fit(multi_asset_data, variables)
        
        assert 'model_info' in results
        assert 'aic' in results
        assert 'bic' in results
        assert dlm_model.is_fitted
    
    def test_predict_next_values(self, dlm_model, multi_asset_data):
        """Test next values prediction."""
        # Mock fitted model
        dlm_model.is_fitted = True
        dlm_model.fitted_model = Mock()
        dlm_model.fitted_model.model.endog_names = ['asset1', 'asset2', 'asset3']
        
        # Mock forecast
        mock_forecast = np.array([[0.01, 0.02, 0.03]])
        dlm_model.fitted_model.forecast.return_value = mock_forecast
        
        # Mock confidence interval
        mock_ci = pd.DataFrame({
            'lower': [0.005, 0.015, 0.025],
            'upper': [0.015, 0.025, 0.035]
        })
        mock_forecast_obj = Mock()
        mock_forecast_obj.conf_int.return_value = mock_ci
        dlm_model.fitted_model.get_forecast.return_value = mock_forecast_obj
        
        # Make prediction
        variables = ['asset1', 'asset2', 'asset3']
        result = dlm_model.predict_next_values(multi_asset_data, variables)
        
        assert 'predictions' in result
        assert 'model_type' in result
        assert result['model_type'] == 'Dynamic Linear Model'
        assert len(result['predictions']) == 3


class TestSeasonalityDetector:
    """Test Seasonality Detector."""
    
    @pytest.fixture
    def seasonality_detector(self):
        """Create Seasonality Detector instance."""
        return SeasonalityDetector()
    
    def test_seasonality_detector_initialization(self, seasonality_detector):
        """Test Seasonality Detector initialization."""
        assert seasonality_detector.model_name == "seasonality_detector"
        assert not seasonality_detector.is_analyzed
    
    def test_prepare_data(self, seasonality_detector, sample_data):
        """Test data preparation for seasonality analysis."""
        series = seasonality_detector.prepare_data(sample_data, 'date', 'close')
        
        assert len(series) == len(sample_data)
        assert not series.isnull().any()
    
    @patch('scipy.signal.detrend')
    @patch('scipy.signal.periodogram')
    def test_detect_seasonal_periods(self, mock_periodogram, mock_detrend, seasonality_detector, sample_data):
        """Test seasonal period detection."""
        # Mock detrend
        mock_detrend.return_value = np.random.randn(100)
        
        # Mock periodogram
        mock_periodogram.return_value = (
            np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        )
        
        series = seasonality_detector.prepare_data(sample_data, 'date', 'close')
        results = seasonality_detector.detect_seasonal_periods(series)
        
        assert 'significant_periods' in results
        assert 'total_periods_found' in results
        assert 'periodogram' in results
    
    def test_analyze_seasonality(self, seasonality_detector, sample_data):
        """Test comprehensive seasonality analysis."""
        results = seasonality_detector.analyze_seasonality(sample_data, 'date', 'close')
        
        assert 'data_info' in results
        assert 'period_detection' in results
        assert 'seasonality_tests' in results
        assert 'overall_assessment' in results
        assert seasonality_detector.is_analyzed


class TestSeasonalDecomposer:
    """Test Seasonal Decomposer."""
    
    @pytest.fixture
    def seasonal_decomposer(self):
        """Create Seasonal Decomposer instance."""
        return SeasonalDecomposer()
    
    def test_seasonal_decomposer_initialization(self, seasonal_decomposer):
        """Test Seasonal Decomposer initialization."""
        assert seasonal_decomposer.model_name == "seasonal_decomposer"
        assert not seasonal_decomposer.is_decomposed
    
    @patch('statsmodels.tsa.seasonal.seasonal_decompose')
    def test_decompose_additive(self, mock_decompose, seasonal_decomposer, sample_data):
        """Test additive seasonal decomposition."""
        # Mock decomposition
        mock_decomposition = Mock()
        mock_decomposition.observed = pd.Series(np.random.randn(100))
        mock_decomposition.trend = pd.Series(np.random.randn(100))
        mock_decomposition.seasonal = pd.Series(np.random.randn(100))
        mock_decomposition.resid = pd.Series(np.random.randn(100))
        mock_decompose.return_value = mock_decomposition
        
        series = seasonal_decomposer.prepare_data(sample_data, 'date', 'close')
        results = seasonal_decomposer.decompose_additive(series, period=12)
        
        assert 'decomposition_type' in results
        assert 'period' in results
        assert 'observed' in results
        assert 'trend' in results
        assert 'seasonal' in results
        assert 'residual' in results
    
    def test_decompose(self, seasonal_decomposer, sample_data):
        """Test seasonal decomposition."""
        results = seasonal_decomposer.decompose(sample_data, 'date', 'close', method='additive')
        
        assert 'decomposition_type' in results
        assert 'data_info' in results
        assert 'decomposition_date' in results
        assert seasonal_decomposer.is_decomposed


class TestTimeSeriesModelManager:
    """Test Time Series Model Manager."""
    
    @pytest.fixture
    def model_manager(self):
        """Create model manager instance."""
        return TimeSeriesModelManager()
    
    def test_model_manager_initialization(self, model_manager):
        """Test model manager initialization."""
        assert model_manager.model_dir is not None
        assert len(model_manager.model_configs) > 0
        assert len(model_manager.models) == 0
    
    def test_create_model(self, model_manager):
        """Test model creation."""
        model = model_manager.create_model('arima_garch')
        assert model is not None
        assert 'arima_garch' in model_manager.models
        assert 'arima_garch' in model_manager.model_metadata
    
    def test_list_models(self, model_manager):
        """Test model listing."""
        models_info = model_manager.list_models()
        assert len(models_info) > 0
        assert all('name' in info for info in models_info)
        assert all('class' in info for info in models_info)
        assert all('parameters' in info for info in models_info)
    
    def test_get_model_status(self, model_manager):
        """Test model status retrieval."""
        status = model_manager.get_model_status()
        assert 'total_models' in status
        assert 'created_models' in status
        assert 'trained_models' in status
        assert 'models' in status
    
    def test_ensemble_prediction(self, model_manager, sample_data):
        """Test ensemble prediction."""
        # Create and mock models
        model1 = model_manager.create_model('arima_garch')
        model2 = model_manager.create_model('prophet')
        
        # Mock trained models
        model1.is_fitted = True
        model1.predict_next_return = Mock(return_value={'predicted_return': 0.01, 'confidence': 0.8})
        
        model2.is_fitted = True
        model2.predict_next_value = Mock(return_value={'predicted_value': 101.5, 'confidence': 0.7})
        
        # Update metadata
        model_manager.model_metadata['arima_garch']['is_trained'] = True
        model_manager.model_metadata['prophet']['is_trained'] = True
        
        # Make ensemble prediction
        result = model_manager.create_ensemble_prediction(sample_data, ['arima_garch', 'prophet'])
        
        assert 'ensemble_prediction' in result
        assert 'ensemble_confidence' in result
        assert 'individual_predictions' in result
        assert 'individual_confidences' in result
        assert 'weights' in result


class TestIntegration:
    """Integration tests for time series models."""
    
    @pytest.fixture
    def integration_data(self):
        """Create integration test data."""
        dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
        data = pd.DataFrame({
            'date': dates,
            'close': 100 + np.random.randn(200).cumsum() * 0.1,
            'volume': 1000000 + np.random.randint(-100000, 100000, 200)
        })
        return data
    
    def test_end_to_end_arima_garch_pipeline(self, integration_data):
        """Test end-to-end ARIMA-GARCH pipeline."""
        # Create model
        model = ARIMAGARCHPredictor()
        
        # Mock fitting
        model.is_fitted = True
        model.arima_model = Mock()
        model.garch_model_fitted = Mock()
        
        # Mock forecast
        mock_forecast = Mock()
        mock_forecast.values = np.array([0.01])
        model.arima_model.fit.return_value.forecast.return_value = mock_forecast
        
        mock_ci = Mock()
        mock_ci.conf_int.return_value = pd.DataFrame({
            'lower': [0.005],
            'upper': [0.015]
        })
        model.arima_model.fit.return_value.get_forecast.return_value = mock_ci
        
        # Mock GARCH forecast
        mock_garch_forecast = Mock()
        mock_garch_forecast.variance.values = np.array([[0.0001]])
        model.garch_model_fitted.forecast.return_value = mock_garch_forecast
        
        # Make prediction
        result = model.predict_next_return(integration_data, 'close')
        
        assert 'predicted_return' in result
        assert 'volatility' in result
        assert 'confidence_interval' in result
        assert result['predicted_return'] == 0.01
    
    def test_model_manager_integration(self, integration_data):
        """Test model manager integration."""
        # Create model manager
        manager = TimeSeriesModelManager()
        
        # Create models
        arima_model = manager.create_model('arima_garch')
        prophet_model = manager.create_model('prophet')
        
        # Mock training
        arima_model.is_fitted = True
        arima_model.predict_next_return = Mock(return_value={'predicted_return': 0.01, 'confidence': 0.8})
        
        prophet_model.is_fitted = True
        prophet_model.predict_next_value = Mock(return_value={'predicted_value': 101.5, 'confidence': 0.7})
        
        # Update metadata
        manager.model_metadata['arima_garch']['is_trained'] = True
        manager.model_metadata['prophet']['is_trained'] = True
        
        # Test ensemble prediction
        result = manager.create_ensemble_prediction(integration_data, ['arima_garch', 'prophet'])
        
        assert 'ensemble_prediction' in result
        assert 'ensemble_confidence' in result
        assert 'model_count' in result
        assert result['model_count'] == 2

