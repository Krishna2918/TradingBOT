"""
ARIMA-GARCH models for financial time series prediction.

This module implements ARIMA models for mean prediction and GARCH models
for volatility forecasting in financial time series.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime, timedelta
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from arch import arch_model
from arch.univariate import GARCH, EGARCH, GJR_GARCH
import joblib
import os

logger = logging.getLogger(__name__)

class ARIMAGARCHPredictor:
    """
    ARIMA-GARCH model for financial time series prediction.
    
    This model combines ARIMA for mean prediction and GARCH for volatility
    modeling to provide comprehensive forecasts with confidence intervals.
    """
    
    def __init__(
        self,
        arima_order: Tuple[int, int, int] = (1, 1, 1),
        garch_order: Tuple[int, int] = (1, 1),
        garch_model: str = 'GARCH',
        distribution: str = 'normal',
        model_name: str = "arima_garch_predictor"
    ):
        """
        Initialize ARIMA-GARCH predictor.
        
        Args:
            arima_order: ARIMA order (p, d, q)
            garch_order: GARCH order (p, q)
            garch_model: GARCH model type ('GARCH', 'EGARCH', 'GJR-GARCH')
            distribution: Error distribution ('normal', 't', 'skewt')
            model_name: Name for model saving/loading
        """
        self.arima_order = arima_order
        self.garch_order = garch_order
        self.garch_model = garch_model
        self.distribution = distribution
        self.model_name = model_name
        
        self.arima_model = None
        self.garch_model_fitted = None
        self.is_fitted = False
        self.residuals = None
        self.volatility = None
        
        logger.info(f"Initialized ARIMA-GARCH Predictor: {model_name}")
    
    def prepare_data(
        self,
        data: pd.DataFrame,
        price_column: str = 'close',
        returns_column: Optional[str] = None
    ) -> pd.Series:
        """
        Prepare data for ARIMA-GARCH modeling.
        
        Args:
            data: Input DataFrame with price data
            price_column: Column containing price data
            returns_column: Column containing returns (if None, will calculate)
            
        Returns:
            Returns series for modeling
        """
        if returns_column and returns_column in data.columns:
            returns = data[returns_column].dropna()
        else:
            # Calculate log returns
            prices = data[price_column].dropna()
            returns = np.log(prices / prices.shift(1)).dropna()
        
        logger.info(f"Prepared returns data: {len(returns)} observations")
        
        return returns
    
    def test_stationarity(self, series: pd.Series) -> Dict[str, Any]:
        """
        Test stationarity of the time series.
        
        Args:
            series: Time series to test
            
        Returns:
            Dictionary with stationarity test results
        """
        # ADF Test
        adf_result = adfuller(series.dropna())
        
        # KPSS Test
        kpss_result = kpss(series.dropna(), regression='c')
        
        results = {
            'adf_statistic': adf_result[0],
            'adf_pvalue': adf_result[1],
            'adf_critical_values': adf_result[4],
            'adf_stationary': adf_result[1] < 0.05,
            'kpss_statistic': kpss_result[0],
            'kpss_pvalue': kpss_result[1],
            'kpss_critical_values': kpss_result[3],
            'kpss_stationary': kpss_result[1] > 0.05,
            'is_stationary': adf_result[1] < 0.05 and kpss_result[1] > 0.05
        }
        
        logger.info(f"Stationarity test - ADF p-value: {results['adf_pvalue']:.4f}, "
                   f"KPSS p-value: {results['kpss_pvalue']:.4f}, "
                   f"Stationary: {results['is_stationary']}")
        
        return results
    
    def find_optimal_arima_order(
        self,
        series: pd.Series,
        max_p: int = 5,
        max_d: int = 2,
        max_q: int = 5,
        ic: str = 'aic'
    ) -> Tuple[int, int, int]:
        """
        Find optimal ARIMA order using information criteria.
        
        Args:
            series: Time series data
            max_p: Maximum AR order
            max_d: Maximum differencing order
            max_q: Maximum MA order
            ic: Information criterion ('aic', 'bic', 'hqic')
            
        Returns:
            Optimal ARIMA order (p, d, q)
        """
        best_ic = float('inf')
        best_order = (0, 0, 0)
        
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(series, order=(p, d, q))
                        fitted_model = model.fit()
                        
                        if ic == 'aic':
                            current_ic = fitted_model.aic
                        elif ic == 'bic':
                            current_ic = fitted_model.bic
                        else:  # hqic
                            current_ic = fitted_model.hqic
                        
                        if current_ic < best_ic:
                            best_ic = current_ic
                            best_order = (p, d, q)
                            
                    except Exception as e:
                        logger.debug(f"Failed to fit ARIMA({p},{d},{q}): {e}")
                        continue
        
        logger.info(f"Optimal ARIMA order: {best_order} with {ic.upper()}: {best_ic:.4f}")
        
        return best_order
    
    def fit_arima(
        self,
        series: pd.Series,
        order: Optional[Tuple[int, int, int]] = None
    ) -> Dict[str, Any]:
        """
        Fit ARIMA model to the data.
        
        Args:
            series: Time series data
            order: ARIMA order (if None, will find optimal)
            
        Returns:
            Fitting results dictionary
        """
        if order is None:
            order = self.find_optimal_arima_order(series)
        
        try:
            # Fit ARIMA model
            self.arima_model = ARIMA(series, order=order)
            fitted_model = self.arima_model.fit()
            
            # Get residuals
            self.residuals = fitted_model.resid
            
            results = {
                'order': order,
                'aic': fitted_model.aic,
                'bic': fitted_model.bic,
                'hqic': fitted_model.hqic,
                'loglikelihood': fitted_model.llf,
                'params': fitted_model.params.to_dict(),
                'pvalues': fitted_model.pvalues.to_dict(),
                'residuals': self.residuals,
                'fitted_values': fitted_model.fittedvalues
            }
            
            logger.info(f"ARIMA model fitted successfully: {order}")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to fit ARIMA model: {e}")
            raise
    
    def fit_garch(
        self,
        residuals: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Fit GARCH model to ARIMA residuals.
        
        Args:
            residuals: Residuals from ARIMA model (if None, use stored residuals)
            
        Returns:
            GARCH fitting results
        """
        if residuals is None:
            residuals = self.residuals
        
        if residuals is None:
            raise ValueError("No residuals available. Fit ARIMA model first.")
        
        try:
            # Create GARCH model
            if self.garch_model == 'GARCH':
                garch = arch_model(
                    residuals,
                    vol=self.garch_model,
                    p=self.garch_order[0],
                    q=self.garch_order[1],
                    dist=self.distribution
                )
            elif self.garch_model == 'EGARCH':
                garch = arch_model(
                    residuals,
                    vol='EGARCH',
                    p=self.garch_order[0],
                    o=1,  # Asymmetric term
                    q=self.garch_order[1],
                    dist=self.distribution
                )
            elif self.garch_model == 'GJR-GARCH':
                garch = arch_model(
                    residuals,
                    vol='GARCH',
                    p=self.garch_order[0],
                    o=1,  # Asymmetric term
                    q=self.garch_order[1],
                    dist=self.distribution
                )
            else:
                raise ValueError(f"Unknown GARCH model: {self.garch_model}")
            
            # Fit GARCH model
            self.garch_model_fitted = garch.fit(disp='off')
            
            # Get volatility
            self.volatility = self.garch_model_fitted.conditional_volatility
            
            results = {
                'model_type': self.garch_model,
                'order': self.garch_order,
                'distribution': self.distribution,
                'aic': self.garch_model_fitted.aic,
                'bic': self.garch_model_fitted.bic,
                'loglikelihood': self.garch_model_fitted.loglikelihood,
                'params': self.garch_model_fitted.params.to_dict(),
                'pvalues': self.garch_model_fitted.pvalues.to_dict(),
                'volatility': self.volatility
            }
            
            logger.info(f"GARCH model fitted successfully: {self.garch_model}")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to fit GARCH model: {e}")
            raise
    
    def fit(
        self,
        data: pd.DataFrame,
        price_column: str = 'close',
        auto_order: bool = True
    ) -> Dict[str, Any]:
        """
        Fit ARIMA-GARCH model to the data.
        
        Args:
            data: Input DataFrame with price data
            price_column: Column containing price data
            auto_order: Whether to automatically find optimal ARIMA order
            
        Returns:
            Complete fitting results
        """
        logger.info("Starting ARIMA-GARCH model fitting")
        
        # Prepare data
        returns = self.prepare_data(data, price_column)
        
        # Test stationarity
        stationarity_results = self.test_stationarity(returns)
        
        # Fit ARIMA model
        if auto_order:
            arima_order = self.find_optimal_arima_order(returns)
        else:
            arima_order = self.arima_order
        
        arima_results = self.fit_arima(returns, arima_order)
        
        # Fit GARCH model
        garch_results = self.fit_garch()
        
        self.is_fitted = True
        
        # Combine results
        results = {
            'arima_results': arima_results,
            'garch_results': garch_results,
            'stationarity_results': stationarity_results,
            'data_info': {
                'n_observations': len(returns),
                'start_date': returns.index[0] if hasattr(returns.index[0], 'date') else None,
                'end_date': returns.index[-1] if hasattr(returns.index[-1], 'date') else None
            }
        }
        
        logger.info("ARIMA-GARCH model fitting completed")
        
        return results
    
    def forecast(
        self,
        steps: int = 1,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Generate forecasts using the fitted ARIMA-GARCH model.
        
        Args:
            steps: Number of steps ahead to forecast
            confidence_level: Confidence level for prediction intervals
            
        Returns:
            Forecast results
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        try:
            # ARIMA forecast
            arima_forecast = self.arima_model.fit().forecast(steps=steps)
            arima_ci = self.arima_model.fit().get_forecast(steps=steps).conf_int(alpha=1-confidence_level)
            
            # GARCH forecast
            garch_forecast = self.garch_model_fitted.forecast(horizon=steps)
            
            # Combine forecasts
            mean_forecast = arima_forecast.values
            volatility_forecast = np.sqrt(garch_forecast.variance.values[-1])
            
            # Calculate confidence intervals
            z_score = 1.96 if confidence_level == 0.95 else 2.576  # 95% or 99%
            lower_bound = mean_forecast - z_score * volatility_forecast
            upper_bound = mean_forecast + z_score * volatility_forecast
            
            results = {
                'mean_forecast': mean_forecast.tolist(),
                'volatility_forecast': volatility_forecast.tolist(),
                'confidence_intervals': {
                    'lower': lower_bound.tolist(),
                    'upper': upper_bound.tolist(),
                    'level': confidence_level
                },
                'steps_ahead': steps,
                'forecast_date': datetime.now().isoformat()
            }
            
            logger.info(f"Generated {steps}-step ahead forecast")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to generate forecast: {e}")
            raise
    
    def predict_next_return(
        self,
        data: pd.DataFrame,
        price_column: str = 'close'
    ) -> Dict[str, Any]:
        """
        Predict next return using ARIMA-GARCH model.
        
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
            'predicted_return': forecast_results['mean_forecast'][0],
            'volatility': forecast_results['volatility_forecast'][0],
            'confidence_interval': {
                'lower': forecast_results['confidence_intervals']['lower'][0],
                'upper': forecast_results['confidence_intervals']['upper'][0]
            },
            'model_type': 'ARIMA-GARCH',
            'model_order': {
                'arima': self.arima_order,
                'garch': self.garch_order
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def get_model_diagnostics(self) -> Dict[str, Any]:
        """
        Get model diagnostics and statistics.
        
        Returns:
            Model diagnostics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting diagnostics")
        
        diagnostics = {
            'arima_diagnostics': {
                'aic': self.arima_model.fit().aic,
                'bic': self.arima_model.fit().bic,
                'loglikelihood': self.arima_model.fit().llf,
                'residuals_stats': {
                    'mean': float(self.residuals.mean()),
                    'std': float(self.residuals.std()),
                    'skewness': float(self.residuals.skew()),
                    'kurtosis': float(self.residuals.kurtosis())
                }
            },
            'garch_diagnostics': {
                'aic': self.garch_model_fitted.aic,
                'bic': self.garch_model_fitted.bic,
                'loglikelihood': self.garch_model_fitted.loglikelihood,
                'volatility_stats': {
                    'mean': float(self.volatility.mean()),
                    'std': float(self.volatility.std()),
                    'min': float(self.volatility.min()),
                    'max': float(self.volatility.max())
                }
            }
        }
        
        return diagnostics
    
    def save_model(self, model_dir: str) -> None:
        """Save the fitted model to disk."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Save ARIMA model
        arima_path = os.path.join(model_dir, f"{self.model_name}_arima.pkl")
        joblib.dump(self.arima_model, arima_path)
        
        # Save GARCH model
        garch_path = os.path.join(model_dir, f"{self.model_name}_garch.pkl")
        joblib.dump(self.garch_model_fitted, garch_path)
        
        # Save metadata
        metadata = {
            'arima_order': self.arima_order,
            'garch_order': self.garch_order,
            'garch_model': self.garch_model,
            'distribution': self.distribution,
            'is_fitted': self.is_fitted
        }
        
        metadata_path = os.path.join(model_dir, f"{self.model_name}_metadata.pkl")
        joblib.dump(metadata, metadata_path)
        
        logger.info(f"ARIMA-GARCH model saved to {model_dir}")
    
    def load_model(self, model_dir: str) -> None:
        """Load a fitted model from disk."""
        # Load ARIMA model
        arima_path = os.path.join(model_dir, f"{self.model_name}_arima.pkl")
        self.arima_model = joblib.load(arima_path)
        
        # Load GARCH model
        garch_path = os.path.join(model_dir, f"{self.model_name}_garch.pkl")
        self.garch_model_fitted = joblib.load(garch_path)
        
        # Load metadata
        metadata_path = os.path.join(model_dir, f"{self.model_name}_metadata.pkl")
        metadata = joblib.load(metadata_path)
        
        self.arima_order = metadata['arima_order']
        self.garch_order = metadata['garch_order']
        self.garch_model = metadata['garch_model']
        self.distribution = metadata['distribution']
        self.is_fitted = metadata['is_fitted']
        
        logger.info(f"ARIMA-GARCH model loaded from {model_dir}")


class GARCHVolatilityPredictor:
    """
    GARCH model specifically for volatility prediction.
    
    This model focuses on volatility forecasting using various GARCH
    specifications including GARCH, EGARCH, and GJR-GARCH.
    """
    
    def __init__(
        self,
        garch_order: Tuple[int, int] = (1, 1),
        garch_model: str = 'GARCH',
        distribution: str = 'normal',
        model_name: str = "garch_volatility_predictor"
    ):
        """
        Initialize GARCH volatility predictor.
        
        Args:
            garch_order: GARCH order (p, q)
            garch_model: GARCH model type
            distribution: Error distribution
            model_name: Name for model saving/loading
        """
        self.garch_order = garch_order
        self.garch_model = garch_model
        self.distribution = distribution
        self.model_name = model_name
        
        self.model = None
        self.is_fitted = False
        self.volatility = None
        
        logger.info(f"Initialized GARCH Volatility Predictor: {model_name}")
    
    def prepare_data(
        self,
        data: pd.DataFrame,
        price_column: str = 'close'
    ) -> pd.Series:
        """Prepare returns data for GARCH modeling."""
        prices = data[price_column].dropna()
        returns = np.log(prices / prices.shift(1)).dropna()
        
        logger.info(f"Prepared returns data for GARCH: {len(returns)} observations")
        
        return returns
    
    def fit(
        self,
        data: pd.DataFrame,
        price_column: str = 'close'
    ) -> Dict[str, Any]:
        """
        Fit GARCH model to the data.
        
        Args:
            data: Input DataFrame with price data
            price_column: Column containing price data
            
        Returns:
            Fitting results
        """
        logger.info("Starting GARCH volatility model fitting")
        
        # Prepare data
        returns = self.prepare_data(data, price_column)
        
        # Create GARCH model
        if self.garch_model == 'GARCH':
            garch = arch_model(
                returns,
                vol='GARCH',
                p=self.garch_order[0],
                q=self.garch_order[1],
                dist=self.distribution
            )
        elif self.garch_model == 'EGARCH':
            garch = arch_model(
                returns,
                vol='EGARCH',
                p=self.garch_order[0],
                o=1,
                q=self.garch_order[1],
                dist=self.distribution
            )
        elif self.garch_model == 'GJR-GARCH':
            garch = arch_model(
                returns,
                vol='GARCH',
                p=self.garch_order[0],
                o=1,
                q=self.garch_order[1],
                dist=self.distribution
            )
        else:
            raise ValueError(f"Unknown GARCH model: {self.garch_model}")
        
        # Fit model
        self.model = garch.fit(disp='off')
        self.volatility = self.model.conditional_volatility
        self.is_fitted = True
        
        results = {
            'model_type': self.garch_model,
            'order': self.garch_order,
            'distribution': self.distribution,
            'aic': self.model.aic,
            'bic': self.model.bic,
            'loglikelihood': self.model.loglikelihood,
            'params': self.model.params.to_dict(),
            'volatility': self.volatility.tolist()
        }
        
        logger.info(f"GARCH volatility model fitted successfully")
        
        return results
    
    def forecast_volatility(
        self,
        steps: int = 1,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Forecast volatility using the fitted GARCH model.
        
        Args:
            steps: Number of steps ahead to forecast
            confidence_level: Confidence level for prediction intervals
            
        Returns:
            Volatility forecast results
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        try:
            # Generate volatility forecast
            forecast = self.model.forecast(horizon=steps)
            
            # Extract volatility forecast
            volatility_forecast = np.sqrt(forecast.variance.values[-1])
            
            # Calculate confidence intervals (simplified)
            z_score = 1.96 if confidence_level == 0.95 else 2.576
            volatility_std = volatility_forecast * 0.1  # Approximate standard error
            
            lower_bound = volatility_forecast - z_score * volatility_std
            upper_bound = volatility_forecast + z_score * volatility_std
            
            results = {
                'volatility_forecast': volatility_forecast.tolist(),
                'confidence_intervals': {
                    'lower': lower_bound.tolist(),
                    'upper': upper_bound.tolist(),
                    'level': confidence_level
                },
                'steps_ahead': steps,
                'forecast_date': datetime.now().isoformat()
            }
            
            logger.info(f"Generated {steps}-step ahead volatility forecast")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to generate volatility forecast: {e}")
            raise
    
    def predict_next_volatility(
        self,
        data: pd.DataFrame,
        price_column: str = 'close'
    ) -> Dict[str, Any]:
        """
        Predict next period volatility.
        
        Args:
            data: Recent market data
            price_column: Column containing price data
            
        Returns:
            Volatility prediction results
        """
        if not self.is_fitted:
            # Fit model if not already fitted
            self.fit(data, price_column)
        
        # Generate 1-step ahead volatility forecast
        forecast_results = self.forecast_volatility(steps=1, confidence_level=0.95)
        
        return {
            'predicted_volatility': forecast_results['volatility_forecast'][0],
            'confidence_interval': {
                'lower': forecast_results['confidence_intervals']['lower'][0],
                'upper': forecast_results['confidence_intervals']['upper'][0]
            },
            'model_type': f'GARCH-{self.garch_model}',
            'model_order': self.garch_order,
            'timestamp': datetime.now().isoformat()
        }
    
    def save_model(self, model_dir: str) -> None:
        """Save the fitted model to disk."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, f"{self.model_name}.pkl")
        joblib.dump(self.model, model_path)
        
        # Save metadata
        metadata = {
            'garch_order': self.garch_order,
            'garch_model': self.garch_model,
            'distribution': self.distribution,
            'is_fitted': self.is_fitted
        }
        
        metadata_path = os.path.join(model_dir, f"{self.model_name}_metadata.pkl")
        joblib.dump(metadata, metadata_path)
        
        logger.info(f"GARCH volatility model saved to {model_dir}")
    
    def load_model(self, model_dir: str) -> None:
        """Load a fitted model from disk."""
        # Load model
        model_path = os.path.join(model_dir, f"{self.model_name}.pkl")
        self.model = joblib.load(model_path)
        
        # Load metadata
        metadata_path = os.path.join(model_dir, f"{self.model_name}_metadata.pkl")
        metadata = joblib.load(metadata_path)
        
        self.garch_order = metadata['garch_order']
        self.garch_model = metadata['garch_model']
        self.distribution = metadata['distribution']
        self.is_fitted = metadata['is_fitted']
        
        logger.info(f"GARCH volatility model loaded from {model_dir}")

