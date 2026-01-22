"""
VAR (Vector Autoregression) models for multi-asset financial prediction.

This module implements VAR and VECM models for analyzing relationships
between multiple financial time series.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime, timedelta
import warnings
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.vecm import VECM
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
import joblib
import os

logger = logging.getLogger(__name__)

class VARPredictor:
    """
    Vector Autoregression (VAR) model for multi-asset prediction.
    
    This model captures dynamic relationships between multiple
    financial time series variables.
    """
    
    def __init__(
        self,
        maxlags: int = 15,
        ic: str = 'aic',
        model_name: str = "var_predictor"
    ):
        """
        Initialize VAR predictor.
        
        Args:
            maxlags: Maximum number of lags to consider
            ic: Information criterion for lag selection ('aic', 'bic', 'hqic', 'fpe')
            model_name: Name for model saving/loading
        """
        self.maxlags = maxlags
        self.ic = ic
        self.model_name = model_name
        
        self.model = None
        self.fitted_model = None
        self.is_fitted = False
        self.variable_names = None
        self.selected_lags = None
        
        logger.info(f"Initialized VAR Predictor: {model_name}")
    
    def prepare_data(
        self,
        data: pd.DataFrame,
        variables: List[str],
        returns: bool = True
    ) -> pd.DataFrame:
        """
        Prepare data for VAR modeling.
        
        Args:
            data: Input DataFrame with multiple time series
            variables: List of variable names to include
            returns: Whether to convert prices to returns
            
        Returns:
            DataFrame with prepared variables
        """
        # Select variables
        var_data = data[variables].copy()
        
        if returns:
            # Convert to log returns
            var_data = np.log(var_data / var_data.shift(1)).dropna()
            logger.info(f"Converted to log returns: {len(var_data)} observations")
        else:
            var_data = var_data.dropna()
            logger.info(f"Using levels: {len(var_data)} observations")
        
        self.variable_names = var_data.columns.tolist()
        
        return var_data
    
    def test_stationarity(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Test stationarity of all variables.
        
        Args:
            data: DataFrame with time series variables
            
        Returns:
            Stationarity test results
        """
        results = {}
        
        for column in data.columns:
            # ADF Test
            adf_result = adfuller(data[column].dropna())
            
            results[column] = {
                'adf_statistic': adf_result[0],
                'adf_pvalue': adf_result[1],
                'adf_critical_values': adf_result[4],
                'is_stationary': adf_result[1] < 0.05
            }
        
        # Overall stationarity
        all_stationary = all(result['is_stationary'] for result in results.values())
        results['all_stationary'] = all_stationary
        
        logger.info(f"Stationarity test - All stationary: {all_stationary}")
        
        return results
    
    def select_lags(
        self,
        data: pd.DataFrame,
        maxlags: Optional[int] = None
    ) -> int:
        """
        Select optimal number of lags using information criteria.
        
        Args:
            data: DataFrame with time series variables
            maxlags: Maximum number of lags to consider
            
        Returns:
            Optimal number of lags
        """
        if maxlags is None:
            maxlags = self.maxlags
        
        try:
            # Create VAR model
            var_model = VAR(data)
            
            # Select lags
            lag_results = var_model.select_order(maxlags=maxlags)
            
            # Get selected lags based on information criterion
            if self.ic == 'aic':
                selected_lags = lag_results.aic
            elif self.ic == 'bic':
                selected_lags = lag_results.bic
            elif self.ic == 'hqic':
                selected_lags = lag_results.hqic
            elif self.ic == 'fpe':
                selected_lags = lag_results.fpe
            else:
                selected_lags = lag_results.aic  # Default to AIC
            
            self.selected_lags = selected_lags
            
            logger.info(f"Selected {selected_lags} lags using {self.ic.upper()}")
            
            return selected_lags
            
        except Exception as e:
            logger.error(f"Failed to select lags: {e}")
            # Default to 1 lag if selection fails
            self.selected_lags = 1
            return 1
    
    def fit(
        self,
        data: pd.DataFrame,
        variables: List[str],
        lags: Optional[int] = None,
        returns: bool = True
    ) -> Dict[str, Any]:
        """
        Fit VAR model to the data.
        
        Args:
            data: Input DataFrame with multiple time series
            variables: List of variable names to include
            lags: Number of lags (if None, will select optimal)
            returns: Whether to convert prices to returns
            
        Returns:
            Fitting results
        """
        logger.info("Starting VAR model fitting")
        
        # Prepare data
        var_data = self.prepare_data(data, variables, returns)
        
        # Test stationarity
        stationarity_results = self.test_stationarity(var_data)
        
        # Select lags if not provided
        if lags is None:
            lags = self.select_lags(var_data)
        else:
            self.selected_lags = lags
        
        try:
            # Create and fit VAR model
            self.model = VAR(var_data)
            self.fitted_model = self.model.fit(lags)
            self.is_fitted = True
            
            # Get model summary
            summary = self.fitted_model.summary()
            
            results = {
                'model_info': {
                    'variables': self.variable_names,
                    'lags': lags,
                    'observations': len(var_data),
                    'information_criterion': self.ic
                },
                'stationarity_results': stationarity_results,
                'model_summary': str(summary),
                'aic': self.fitted_model.aic,
                'bic': self.fitted_model.bic,
                'hqic': self.fitted_model.hqic,
                'fpe': self.fitted_model.fpe,
                'loglikelihood': self.fitted_model.llf
            }
            
            logger.info("VAR model fitting completed")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to fit VAR model: {e}")
            raise
    
    def forecast(
        self,
        steps: int = 1,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Generate forecasts using the fitted VAR model.
        
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
            forecast = self.fitted_model.forecast(
                self.fitted_model.y, 
                steps=steps
            )
            
            # Get forecast standard errors
            forecast_errors = self.fitted_model.forecast_interval(
                self.fitted_model.y,
                steps=steps,
                alpha=1-confidence_level
            )
            
            # Create forecast DataFrame
            forecast_df = pd.DataFrame(
                forecast,
                columns=self.variable_names,
                index=range(1, steps + 1)
            )
            
            # Create confidence intervals DataFrame
            lower_bound = forecast_errors[0]
            upper_bound = forecast_errors[1]

            results = {
                'forecast': forecast_df.to_dict('index'),
                'confidence_intervals': {
                    'lower': pd.DataFrame(lower_bound, columns=self.variable_names, index=range(1, steps + 1)).to_dict('index'),
                    'upper': pd.DataFrame(upper_bound, columns=self.variable_names, index=range(1, steps + 1)).to_dict('index'),
                    'level': confidence_level
                },
                'forecast_periods': steps,
                'variables': self.variable_names,
                'forecast_date': datetime.now().isoformat()
            }
            
            logger.info(f"Generated {steps}-step ahead VAR forecast")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to generate VAR forecast: {e}")
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
        next_lower = forecast_results['confidence_intervals']['lower'][1]
        next_upper = forecast_results['confidence_intervals']['upper'][1]
        
        # Create prediction dictionary
        predictions = {}
        for variable in self.variable_names:
            predictions[variable] = {
                'predicted_value': next_forecast[variable],
                'confidence_interval': {
                    'lower': next_lower[variable],
                    'upper': next_upper[variable]
                }
            }
        
        return {
            'predictions': predictions,
            'model_type': 'VAR',
            'model_lags': self.selected_lags,
            'variables': self.variable_names,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_impulse_response(
        self,
        periods: int = 10,
        orthogonalized: bool = True
    ) -> Dict[str, Any]:
        """
        Get impulse response functions.
        
        Args:
            periods: Number of periods for impulse response
            orthogonalized: Whether to use orthogonalized impulse responses
            
        Returns:
            Impulse response results
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting impulse responses")
        
        try:
            # Get impulse response functions
            if orthogonalized:
                irf = self.fitted_model.irf(periods)
            else:
                irf = self.fitted_model.irf_resim(periods)
            
            # Convert to dictionary format
            irf_results = {}
            for i, variable in enumerate(self.variable_names):
                irf_results[variable] = {}
                for j, shock_var in enumerate(self.variable_names):
                    irf_results[variable][shock_var] = irf.irfs[:, i, j].tolist()
            
            results = {
                'impulse_responses': irf_results,
                'periods': periods,
                'orthogonalized': orthogonalized,
                'variables': self.variable_names,
                'analysis_date': datetime.now().isoformat()
            }
            
            logger.info("Generated impulse response functions")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get impulse responses: {e}")
            raise
    
    def get_granger_causality(self) -> Dict[str, Any]:
        """
        Test Granger causality between variables.
        
        Returns:
            Granger causality test results
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before testing Granger causality")
        
        try:
            # Perform Granger causality tests
            gc_results = {}
            
            for variable in self.variable_names:
                gc_results[variable] = {}
                for other_var in self.variable_names:
                    if variable != other_var:
                        # Test if other_var Granger causes variable
                        test_result = self.fitted_model.test_causality(
                            variable, other_var, kind='f'
                        )
                        gc_results[variable][other_var] = {
                            'f_statistic': test_result.test_statistic,
                            'p_value': test_result.pvalue,
                            'significant': test_result.pvalue < 0.05
                        }
            
            results = {
                'granger_causality': gc_results,
                'variables': self.variable_names,
                'analysis_date': datetime.now().isoformat()
            }
            
            logger.info("Generated Granger causality tests")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to test Granger causality: {e}")
            raise
    
    def save_model(self, model_dir: str) -> None:
        """Save the fitted model to disk."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Save VAR model
        model_path = os.path.join(model_dir, f"{self.model_name}.pkl")
        joblib.dump(self.fitted_model, model_path)
        
        # Save metadata
        metadata = {
            'maxlags': self.maxlags,
            'ic': self.ic,
            'selected_lags': self.selected_lags,
            'variable_names': self.variable_names,
            'is_fitted': self.is_fitted
        }
        
        metadata_path = os.path.join(model_dir, f"{self.model_name}_metadata.pkl")
        joblib.dump(metadata, metadata_path)
        
        logger.info(f"VAR model saved to {model_dir}")
    
    def load_model(self, model_dir: str) -> None:
        """Load a fitted model from disk."""
        # Load VAR model
        model_path = os.path.join(model_dir, f"{self.model_name}.pkl")
        self.fitted_model = joblib.load(model_path)
        
        # Load metadata
        metadata_path = os.path.join(model_dir, f"{self.model_name}_metadata.pkl")
        metadata = joblib.load(metadata_path)
        
        self.maxlags = metadata['maxlags']
        self.ic = metadata['ic']
        self.selected_lags = metadata['selected_lags']
        self.variable_names = metadata['variable_names']
        self.is_fitted = metadata['is_fitted']
        
        logger.info(f"VAR model loaded from {model_dir}")


class VECMPredictor:
    """
    Vector Error Correction Model (VECM) for cointegrated time series.
    
    This model is used when variables are cointegrated and captures
    both short-run dynamics and long-run equilibrium relationships.
    """
    
    def __init__(
        self,
        k_ar_diff: int = 1,
        coint_rank: Optional[int] = None,
        deterministic: str = 'ci',
        model_name: str = "vecm_predictor"
    ):
        """
        Initialize VECM predictor.
        
        Args:
            k_ar_diff: Number of lags of differences
            coint_rank: Cointegration rank (if None, will be determined)
            deterministic: Deterministic terms ('ci', 'co', 'lo', 'li')
            model_name: Name for model saving/loading
        """
        self.k_ar_diff = k_ar_diff
        self.coint_rank = coint_rank
        self.deterministic = deterministic
        self.model_name = model_name
        
        self.model = None
        self.fitted_model = None
        self.is_fitted = False
        self.variable_names = None
        self.cointegration_results = None
        
        logger.info(f"Initialized VECM Predictor: {model_name}")
    
    def test_cointegration(
        self,
        data: pd.DataFrame,
        variables: List[str]
    ) -> Dict[str, Any]:
        """
        Test for cointegration between variables.
        
        Args:
            data: DataFrame with time series variables
            variables: List of variable names to test
            
        Returns:
            Cointegration test results
        """
        results = {}
        
        # Test all pairs
        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables[i+1:], i+1):
                try:
                    # Engle-Granger cointegration test
                    coint_result = coint(data[var1], data[var2])
                    
                    pair_name = f"{var1}_{var2}"
                    results[pair_name] = {
                        'statistic': coint_result[0],
                        'p_value': coint_result[1],
                        'critical_values': coint_result[2],
                        'is_cointegrated': coint_result[1] < 0.05
                    }
                    
                except Exception as e:
                    logger.warning(f"Failed to test cointegration for {var1}-{var2}: {e}")
                    results[f"{var1}_{var2}"] = {
                        'error': str(e),
                        'is_cointegrated': False
                    }
        
        # Overall cointegration assessment
        cointegrated_pairs = sum(1 for result in results.values() 
                               if result.get('is_cointegrated', False))
        results['cointegrated_pairs'] = cointegrated_pairs
        results['total_pairs'] = len(results)
        results['has_cointegration'] = cointegrated_pairs > 0
        
        logger.info(f"Cointegration test - {cointegrated_pairs} cointegrated pairs found")
        
        return results
    
    def fit(
        self,
        data: pd.DataFrame,
        variables: List[str],
        coint_rank: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Fit VECM model to the data.
        
        Args:
            data: Input DataFrame with multiple time series
            variables: List of variable names to include
            coint_rank: Cointegration rank (if None, will be determined)
            
        Returns:
            Fitting results
        """
        logger.info("Starting VECM model fitting")
        
        # Select variables
        vecm_data = data[variables].copy().dropna()
        self.variable_names = vecm_data.columns.tolist()
        
        # Test for cointegration
        cointegration_results = self.test_cointegration(vecm_data, variables)
        
        if not cointegration_results['has_cointegration']:
            logger.warning("No cointegration found. VECM may not be appropriate.")
        
        # Determine cointegration rank if not provided
        if coint_rank is None:
            # Use number of cointegrated pairs as proxy
            coint_rank = min(cointegration_results['cointegrated_pairs'], len(variables) - 1)
            if coint_rank == 0:
                coint_rank = 1  # Default to rank 1
        
        self.coint_rank = coint_rank
        
        try:
            # Create and fit VECM model
            self.model = VECM(
                vecm_data,
                k_ar_diff=self.k_ar_diff,
                coint_rank=coint_rank,
                deterministic=self.deterministic
            )
            self.fitted_model = self.model.fit()
            self.is_fitted = True
            self.cointegration_results = cointegration_results
            
            results = {
                'model_info': {
                    'variables': self.variable_names,
                    'k_ar_diff': self.k_ar_diff,
                    'coint_rank': coint_rank,
                    'deterministic': self.deterministic,
                    'observations': len(vecm_data)
                },
                'cointegration_results': cointegration_results,
                'aic': getattr(self.fitted_model, 'aic', None),
                'bic': getattr(self.fitted_model, 'bic', None),
                'hqic': getattr(self.fitted_model, 'hqic', None),
                'loglikelihood': getattr(self.fitted_model, 'llf', None)
            }
            
            logger.info("VECM model fitting completed")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to fit VECM model: {e}")
            raise
    
    def forecast(
        self,
        steps: int = 1
    ) -> Dict[str, Any]:
        """
        Generate forecasts using the fitted VECM model.
        
        Args:
            steps: Number of steps ahead to forecast
            
        Returns:
            Forecast results
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        try:
            # Generate forecast
            forecast = self.fitted_model.forecast(steps=steps)
            
            # Create forecast DataFrame
            forecast_df = pd.DataFrame(
                forecast,
                columns=self.variable_names,
                index=range(1, steps + 1)
            )
            
            results = {
                'forecast': forecast_df.to_dict('index'),
                'forecast_periods': steps,
                'variables': self.variable_names,
                'forecast_date': datetime.now().isoformat()
            }
            
            logger.info(f"Generated {steps}-step ahead VECM forecast")

            return results

        except Exception as e:
            logger.error(f"Failed to generate VECM forecast: {e}")
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
        forecast_results = self.forecast(steps=1)

        # Extract next period predictions
        next_forecast = forecast_results['forecast'][1]

        # Create prediction dictionary
        predictions = {}
        for variable in self.variable_names:
            predictions[variable] = {
                'predicted_value': next_forecast[variable]
            }

        return {
            'forecast': next_forecast,
            'forecast_periods': 1,
            'variables': self.variable_names,
            'predictions': predictions,
            'model_type': 'VECM',
            'timestamp': datetime.now().isoformat()
        }

    def get_cointegration_vectors(self) -> Dict[str, Any]:
        """
        Get cointegration vectors from the VECM model.
        
        Returns:
            Cointegration vector results
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting cointegration vectors")
        
        try:
            # Get cointegration vectors
            coint_vectors = self.fitted_model.beta
            
            # Create results
            results = {
                'cointegration_vectors': coint_vectors.tolist(),
                'coint_rank': self.coint_rank,
                'variables': self.variable_names,
                'analysis_date': datetime.now().isoformat()
            }
            
            logger.info("Generated cointegration vectors")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get cointegration vectors: {e}")
            raise
    
    def save_model(self, model_dir: str) -> None:
        """Save the fitted model to disk."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Save VECM model
        model_path = os.path.join(model_dir, f"{self.model_name}.pkl")
        joblib.dump(self.fitted_model, model_path)
        
        # Save metadata
        metadata = {
            'k_ar_diff': self.k_ar_diff,
            'coint_rank': self.coint_rank,
            'deterministic': self.deterministic,
            'variable_names': self.variable_names,
            'cointegration_results': self.cointegration_results,
            'is_fitted': self.is_fitted
        }
        
        metadata_path = os.path.join(model_dir, f"{self.model_name}_metadata.pkl")
        joblib.dump(metadata, metadata_path)
        
        logger.info(f"VECM model saved to {model_dir}")
    
    def load_model(self, model_dir: str) -> None:
        """Load a fitted model from disk."""
        # Load VECM model
        model_path = os.path.join(model_dir, f"{self.model_name}.pkl")
        self.fitted_model = joblib.load(model_path)
        
        # Load metadata
        metadata_path = os.path.join(model_dir, f"{self.model_name}_metadata.pkl")
        metadata = joblib.load(metadata_path)
        
        self.k_ar_diff = metadata['k_ar_diff']
        self.coint_rank = metadata['coint_rank']
        self.deterministic = metadata['deterministic']
        self.variable_names = metadata['variable_names']
        self.cointegration_results = metadata['cointegration_results']
        self.is_fitted = metadata['is_fitted']
        
        logger.info(f"VECM model loaded from {model_dir}")

