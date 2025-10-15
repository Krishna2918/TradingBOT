"""
State Space models for financial time series prediction.

This module implements Kalman filters and Dynamic Linear Models
for state space modeling of financial time series.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime, timedelta
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.statespace.tools import cfa_simulation_smoother
import joblib
import os

logger = logging.getLogger(__name__)

class KalmanFilterPredictor:
    """
    Kalman Filter for financial time series prediction.
    
    This model uses state space representation with Kalman filtering
    for optimal estimation and prediction of financial time series.
    """
    
    def __init__(
        self,
        state_dim: int = 2,
        observation_dim: int = 1,
        model_name: str = "kalman_filter_predictor"
    ):
        """
        Initialize Kalman Filter predictor.
        
        Args:
            state_dim: Dimension of state vector
            observation_dim: Dimension of observation vector
            model_name: Name for model saving/loading
        """
        self.state_dim = state_dim
        self.observation_dim = observation_dim
        self.model_name = model_name
        
        self.model = None
        self.fitted_model = None
        self.is_fitted = False
        self.state_estimates = None
        self.observation_estimates = None
        
        logger.info(f"Initialized Kalman Filter Predictor: {model_name}")
    
    def prepare_data(
        self,
        data: pd.DataFrame,
        price_column: str = 'close',
        returns: bool = True
    ) -> pd.Series:
        """
        Prepare data for Kalman filtering.
        
        Args:
            data: Input DataFrame with price data
            price_column: Column containing price data
            returns: Whether to convert prices to returns
            
        Returns:
            Time series data for filtering
        """
        if returns:
            # Convert to log returns
            prices = data[price_column].dropna()
            series = np.log(prices / prices.shift(1)).dropna()
        else:
            series = data[price_column].dropna()
        
        logger.info(f"Prepared data for Kalman filter: {len(series)} observations")
        
        return series
    
    def fit(
        self,
        data: pd.DataFrame,
        price_column: str = 'close',
        returns: bool = True,
        trend: bool = True,
        seasonal: bool = False,
        cycle: bool = False
    ) -> Dict[str, Any]:
        """
        Fit Kalman Filter model to the data.
        
        Args:
            data: Input DataFrame with price data
            price_column: Column containing price data
            returns: Whether to convert prices to returns
            trend: Whether to include trend component
            seasonal: Whether to include seasonal component
            cycle: Whether to include cycle component
            
        Returns:
            Fitting results
        """
        logger.info("Starting Kalman Filter model fitting")
        
        # Prepare data
        series = self.prepare_data(data, price_column, returns)
        
        try:
            # Create UnobservedComponents model (state space representation)
            self.model = UnobservedComponents(
                series,
                level='local level' if trend else None,
                trend='local linear trend' if trend else None,
                seasonal=12 if seasonal else None,
                cycle=cycle,
                stochastic_level=True,
                stochastic_trend=True,
                stochastic_seasonal=True if seasonal else False,
                stochastic_cycle=True if cycle else False
            )
            
            # Fit model
            self.fitted_model = self.model.fit(disp=False)
            self.is_fitted = True
            
            # Get state estimates
            self.state_estimates = self.fitted_model.smoothed_state
            self.observation_estimates = self.fitted_model.smoothed_state[0]  # Level component
            
            results = {
                'model_info': {
                    'state_dim': self.state_dim,
                    'observation_dim': self.observation_dim,
                    'trend': trend,
                    'seasonal': seasonal,
                    'cycle': cycle,
                    'observations': len(series)
                },
                'aic': self.fitted_model.aic,
                'bic': self.fitted_model.bic,
                'hqic': self.fitted_model.hqic,
                'loglikelihood': self.fitted_model.llf,
                'state_estimates': self.state_estimates.tolist() if hasattr(self.state_estimates, 'tolist') else None
            }
            
            logger.info("Kalman Filter model fitting completed")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to fit Kalman Filter model: {e}")
            raise
    
    def forecast(
        self,
        steps: int = 1,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Generate forecasts using the fitted Kalman Filter model.
        
        Args:
            steps: Number of steps ahead to forecast
            confidence_level: Confidence level for prediction intervals
            
        Returns:
            Forecast results
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        try:
            # Generate forecast
            forecast = self.fitted_model.forecast(steps=steps)
            
            # Get forecast confidence intervals
            forecast_ci = self.fitted_model.get_forecast(steps=steps).conf_int(alpha=1-confidence_level)
            
            results = {
                'forecast': forecast.tolist(),
                'confidence_intervals': {
                    'lower': forecast_ci.iloc[:, 0].tolist(),
                    'upper': forecast_ci.iloc[:, 1].tolist(),
                    'level': confidence_level
                },
                'forecast_periods': steps,
                'forecast_date': datetime.now().isoformat()
            }
            
            logger.info(f"Generated {steps}-step ahead Kalman Filter forecast")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to generate Kalman Filter forecast: {e}")
            raise
    
    def predict_next_value(
        self,
        data: pd.DataFrame,
        price_column: str = 'close'
    ) -> Dict[str, Any]:
        """
        Predict next value using Kalman Filter model.
        
        Args:
            data: Recent market data
            price_column: Column containing price data
            
        Returns:
            Prediction results
        """
        if not self.is_fitted:
            # Fit model if not already fitted
            self.fit(data, price_column)
        
        # Generate 1-step ahead forecast
        forecast_results = self.forecast(steps=1, confidence_level=0.95)
        
        return {
            'predicted_value': forecast_results['forecast'][0],
            'confidence_interval': {
                'lower': forecast_results['confidence_intervals']['lower'][0],
                'upper': forecast_results['confidence_intervals']['upper'][0]
            },
            'model_type': 'Kalman Filter',
            'state_dimension': self.state_dim,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_state_estimates(self) -> Dict[str, Any]:
        """
        Get state estimates from the Kalman Filter.
        
        Returns:
            State estimation results
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting state estimates")
        
        try:
            # Get smoothed states
            smoothed_states = self.fitted_model.smoothed_state
            
            # Get state names
            state_names = self.fitted_model.model.state_names
            
            results = {
                'state_estimates': smoothed_states.tolist() if hasattr(smoothed_states, 'tolist') else smoothed_states,
                'state_names': state_names,
                'state_dimension': self.state_dim,
                'analysis_date': datetime.now().isoformat()
            }
            
            logger.info("Generated state estimates")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get state estimates: {e}")
            raise
    
    def save_model(self, model_dir: str) -> None:
        """Save the fitted model to disk."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Save Kalman Filter model
        model_path = os.path.join(model_dir, f"{self.model_name}.pkl")
        joblib.dump(self.fitted_model, model_path)
        
        # Save metadata
        metadata = {
            'state_dim': self.state_dim,
            'observation_dim': self.observation_dim,
            'is_fitted': self.is_fitted
        }
        
        metadata_path = os.path.join(model_dir, f"{self.model_name}_metadata.pkl")
        joblib.dump(metadata, metadata_path)
        
        logger.info(f"Kalman Filter model saved to {model_dir}")
    
    def load_model(self, model_dir: str) -> None:
        """Load a fitted model from disk."""
        # Load Kalman Filter model
        model_path = os.path.join(model_dir, f"{self.model_name}.pkl")
        self.fitted_model = joblib.load(model_path)
        
        # Load metadata
        metadata_path = os.path.join(model_dir, f"{self.model_name}_metadata.pkl")
        metadata = joblib.load(metadata_path)
        
        self.state_dim = metadata['state_dim']
        self.observation_dim = metadata['observation_dim']
        self.is_fitted = metadata['is_fitted']
        
        logger.info(f"Kalman Filter model loaded from {model_dir}")


class DynamicLinearModel:
    """
    Dynamic Linear Model for financial time series prediction.
    
    This model uses state space representation with time-varying
    parameters for modeling financial time series dynamics.
    """
    
    def __init__(
        self,
        k_factors: int = 1,
        factor_order: int = 1,
        model_name: str = "dynamic_linear_model"
    ):
        """
        Initialize Dynamic Linear Model.
        
        Args:
            k_factors: Number of factors
            factor_order: Order of factor dynamics
            model_name: Name for model saving/loading
        """
        self.k_factors = k_factors
        self.factor_order = factor_order
        self.model_name = model_name
        
        self.model = None
        self.fitted_model = None
        self.is_fitted = False
        self.factor_loadings = None
        self.factor_estimates = None
        
        logger.info(f"Initialized Dynamic Linear Model: {model_name}")
    
    def prepare_data(
        self,
        data: pd.DataFrame,
        variables: List[str],
        returns: bool = True
    ) -> pd.DataFrame:
        """
        Prepare data for Dynamic Linear Model.
        
        Args:
            data: Input DataFrame with multiple time series
            variables: List of variable names to include
            returns: Whether to convert prices to returns
            
        Returns:
            DataFrame with prepared variables
        """
        # Select variables
        dlm_data = data[variables].copy()
        
        if returns:
            # Convert to log returns
            dlm_data = np.log(dlm_data / dlm_data.shift(1)).dropna()
            logger.info(f"Converted to log returns: {len(dlm_data)} observations")
        else:
            dlm_data = dlm_data.dropna()
            logger.info(f"Using levels: {len(dlm_data)} observations")
        
        return dlm_data
    
    def fit(
        self,
        data: pd.DataFrame,
        variables: List[str],
        returns: bool = True
    ) -> Dict[str, Any]:
        """
        Fit Dynamic Linear Model to the data.
        
        Args:
            data: Input DataFrame with multiple time series
            variables: List of variable names to include
            returns: Whether to convert prices to returns
            
        Returns:
            Fitting results
        """
        logger.info("Starting Dynamic Linear Model fitting")
        
        # Prepare data
        dlm_data = self.prepare_data(data, variables, returns)
        
        try:
            # Create Dynamic Factor model
            self.model = DynamicFactor(
                dlm_data,
                k_factors=self.k_factors,
                factor_order=self.factor_order,
                error_cov_type='diagonal'
            )
            
            # Fit model
            self.fitted_model = self.model.fit(disp=False)
            self.is_fitted = True
            
            # Get factor estimates
            self.factor_estimates = self.fitted_model.factors.filtered[0]
            self.factor_loadings = self.fitted_model.params
            
            results = {
                'model_info': {
                    'k_factors': self.k_factors,
                    'factor_order': self.factor_order,
                    'variables': variables,
                    'observations': len(dlm_data)
                },
                'aic': self.fitted_model.aic,
                'bic': self.fitted_model.bic,
                'hqic': self.fitted_model.hqic,
                'loglikelihood': self.fitted_model.llf,
                'factor_loadings': self.factor_loadings.tolist() if hasattr(self.factor_loadings, 'tolist') else self.factor_loadings
            }
            
            logger.info("Dynamic Linear Model fitting completed")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to fit Dynamic Linear Model: {e}")
            raise
    
    def forecast(
        self,
        steps: int = 1,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Generate forecasts using the fitted Dynamic Linear Model.
        
        Args:
            steps: Number of steps ahead to forecast
            confidence_level: Confidence level for prediction intervals
            
        Returns:
            Forecast results
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        try:
            # Generate forecast
            forecast = self.fitted_model.forecast(steps=steps)
            
            # Get forecast confidence intervals
            forecast_ci = self.fitted_model.get_forecast(steps=steps).conf_int(alpha=1-confidence_level)
            
            # Create forecast DataFrame
            forecast_df = pd.DataFrame(
                forecast,
                columns=self.fitted_model.model.endog_names,
                index=range(1, steps + 1)
            )
            
            results = {
                'forecast': forecast_df.to_dict('index'),
                'confidence_intervals': {
                    'lower': forecast_ci.iloc[:, 0].tolist(),
                    'upper': forecast_ci.iloc[:, 1].tolist(),
                    'level': confidence_level
                },
                'forecast_periods': steps,
                'variables': self.fitted_model.model.endog_names,
                'forecast_date': datetime.now().isoformat()
            }
            
            logger.info(f"Generated {steps}-step ahead Dynamic Linear Model forecast")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to generate Dynamic Linear Model forecast: {e}")
            raise
    
    def predict_next_values(
        self,
        data: pd.DataFrame,
        variables: List[str]
    ) -> Dict[str, Any]:
        """
        Predict next values for all variables.
        
        Args:
            data: Recent market data
            variables: List of variable names
            
        Returns:
            Prediction results
        """
        if not self.is_fitted:
            # Fit model if not already fitted
            self.fit(data, variables)
        
        # Generate 1-step ahead forecast
        forecast_results = self.forecast(steps=1, confidence_level=0.95)
        
        # Extract next period predictions
        next_forecast = forecast_results['forecast'][1]
        next_lower = forecast_results['confidence_intervals']['lower'][0]
        next_upper = forecast_results['confidence_intervals']['upper'][0]
        
        # Create prediction dictionary
        predictions = {}
        for variable in self.fitted_model.model.endog_names:
            predictions[variable] = {
                'predicted_value': next_forecast[variable],
                'confidence_interval': {
                    'lower': next_lower,
                    'upper': next_upper
                }
            }
        
        return {
            'predictions': predictions,
            'model_type': 'Dynamic Linear Model',
            'k_factors': self.k_factors,
            'factor_order': self.factor_order,
            'variables': self.fitted_model.model.endog_names,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_factor_analysis(self) -> Dict[str, Any]:
        """
        Get factor analysis results.
        
        Returns:
            Factor analysis results
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting factor analysis")
        
        try:
            # Get factor estimates
            factors = self.fitted_model.factors.filtered[0]
            
            # Get factor loadings
            loadings = self.fitted_model.params
            
            # Calculate factor contributions
            factor_contributions = {}
            for i, variable in enumerate(self.fitted_model.model.endog_names):
                factor_contributions[variable] = {
                    'factor_loading': float(loadings[i]) if hasattr(loadings, '__getitem__') else float(loadings),
                    'factor_contribution': float(factors[i]) if hasattr(factors, '__getitem__') else float(factors)
                }
            
            results = {
                'factor_estimates': factors.tolist() if hasattr(factors, 'tolist') else [factors],
                'factor_loadings': loadings.tolist() if hasattr(loadings, 'tolist') else [loadings],
                'factor_contributions': factor_contributions,
                'k_factors': self.k_factors,
                'variables': self.fitted_model.model.endog_names,
                'analysis_date': datetime.now().isoformat()
            }
            
            logger.info("Generated factor analysis")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get factor analysis: {e}")
            raise
    
    def save_model(self, model_dir: str) -> None:
        """Save the fitted model to disk."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Save Dynamic Linear Model
        model_path = os.path.join(model_dir, f"{self.model_name}.pkl")
        joblib.dump(self.fitted_model, model_path)
        
        # Save metadata
        metadata = {
            'k_factors': self.k_factors,
            'factor_order': self.factor_order,
            'is_fitted': self.is_fitted
        }
        
        metadata_path = os.path.join(model_dir, f"{self.model_name}_metadata.pkl")
        joblib.dump(metadata, metadata_path)
        
        logger.info(f"Dynamic Linear Model saved to {model_dir}")
    
    def load_model(self, model_dir: str) -> None:
        """Load a fitted model from disk."""
        # Load Dynamic Linear Model
        model_path = os.path.join(model_dir, f"{self.model_name}.pkl")
        self.fitted_model = joblib.load(model_path)
        
        # Load metadata
        metadata_path = os.path.join(model_dir, f"{self.model_name}_metadata.pkl")
        metadata = joblib.load(metadata_path)
        
        self.k_factors = metadata['k_factors']
        self.factor_order = metadata['factor_order']
        self.is_fitted = metadata['is_fitted']
        
        logger.info(f"Dynamic Linear Model loaded from {model_dir}")

