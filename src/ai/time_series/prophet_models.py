"""
Prophet models for financial time series prediction.

This module implements Facebook's Prophet for time series forecasting
with seasonality detection and trend analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime, timedelta
import warnings
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import joblib
import os

logger = logging.getLogger(__name__)

class ProphetPredictor:
    """
    Prophet model for financial time series prediction.
    
    This model uses Facebook's Prophet for time series forecasting
    with automatic seasonality detection and trend analysis.
    """
    
    def __init__(
        self,
        growth: str = 'linear',
        seasonality_mode: str = 'additive',
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = False,
        seasonality_prior_scale: float = 10.0,
        holidays_prior_scale: float = 10.0,
        changepoint_prior_scale: float = 0.05,
        model_name: str = "prophet_predictor"
    ):
        """
        Initialize Prophet predictor.
        
        Args:
            growth: Growth model ('linear', 'logistic')
            seasonality_mode: Seasonality mode ('additive', 'multiplicative')
            yearly_seasonality: Whether to include yearly seasonality
            weekly_seasonality: Whether to include weekly seasonality
            daily_seasonality: Whether to include daily seasonality
            seasonality_prior_scale: Prior scale for seasonality
            holidays_prior_scale: Prior scale for holidays
            changepoint_prior_scale: Prior scale for changepoints
            model_name: Name for model saving/loading
        """
        self.growth = growth
        self.seasonality_mode = seasonality_mode
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.seasonality_prior_scale = seasonality_prior_scale
        self.holidays_prior_scale = holidays_prior_scale
        self.changepoint_prior_scale = changepoint_prior_scale
        self.model_name = model_name
        
        self.model = None
        self.is_fitted = False
        self.holidays = None
        
        logger.info(f"Initialized Prophet Predictor: {model_name}")
    
    def prepare_data(
        self,
        data: pd.DataFrame,
        date_column: str = 'date',
        value_column: str = 'close'
    ) -> pd.DataFrame:
        """
        Prepare data for Prophet modeling.
        
        Args:
            data: Input DataFrame with time series data
            date_column: Column containing dates
            value_column: Column containing values to predict
            
        Returns:
            DataFrame in Prophet format (ds, y)
        """
        # Ensure date column is datetime
        if date_column in data.columns:
            data = data.copy()
            data[date_column] = pd.to_datetime(data[date_column])
        else:
            # Use index as date if no date column
            data = data.copy()
            data[date_column] = pd.to_datetime(data.index)
        
        # Create Prophet format DataFrame
        prophet_data = pd.DataFrame({
            'ds': data[date_column],
            'y': data[value_column]
        }).dropna()
        
        logger.info(f"Prepared Prophet data: {len(prophet_data)} observations")
        
        return prophet_data
    
    def add_holidays(
        self,
        holidays_data: Optional[pd.DataFrame] = None,
        custom_holidays: Optional[List[Dict]] = None
    ) -> None:
        """
        Add holidays to the Prophet model.
        
        Args:
            holidays_data: DataFrame with holidays (holiday, ds columns)
            custom_holidays: List of custom holiday dictionaries
        """
        if holidays_data is not None:
            self.holidays = holidays_data
        elif custom_holidays is not None:
            self.holidays = pd.DataFrame(custom_holidays)
        
        logger.info(f"Added {len(self.holidays) if self.holidays is not None else 0} holidays")
    
    def create_financial_holidays(self) -> pd.DataFrame:
        """Create common financial market holidays."""
        holidays = []
        
        # US Market Holidays (simplified)
        current_year = datetime.now().year
        for year in range(current_year - 1, current_year + 2):
            holidays.extend([
                {'holiday': 'new_year', 'ds': f'{year}-01-01'},
                {'holiday': 'martin_luther_king', 'ds': f'{year}-01-15'},
                {'holiday': 'presidents_day', 'ds': f'{year}-02-19'},
                {'holiday': 'good_friday', 'ds': f'{year}-04-19'},  # Approximate
                {'holiday': 'memorial_day', 'ds': f'{year}-05-27'},
                {'holiday': 'independence_day', 'ds': f'{year}-07-04'},
                {'holiday': 'labor_day', 'ds': f'{year}-09-02'},
                {'holiday': 'thanksgiving', 'ds': f'{year}-11-28'},
                {'holiday': 'christmas', 'ds': f'{year}-12-25'}
            ])
        
        holidays_df = pd.DataFrame(holidays)
        holidays_df['ds'] = pd.to_datetime(holidays_df['ds'])
        
        return holidays_df
    
    def fit(
        self,
        data: pd.DataFrame,
        date_column: str = 'date',
        value_column: str = 'close',
        add_financial_holidays: bool = True
    ) -> Dict[str, Any]:
        """
        Fit Prophet model to the data.
        
        Args:
            data: Input DataFrame with time series data
            date_column: Column containing dates
            value_column: Column containing values to predict
            add_financial_holidays: Whether to add financial market holidays
            
        Returns:
            Fitting results
        """
        logger.info("Starting Prophet model fitting")
        
        # Prepare data
        prophet_data = self.prepare_data(data, date_column, value_column)
        
        # Add financial holidays if requested
        if add_financial_holidays and self.holidays is None:
            self.holidays = self.create_financial_holidays()
        
        # Create Prophet model
        self.model = Prophet(
            growth=self.growth,
            seasonality_mode=self.seasonality_mode,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            seasonality_prior_scale=self.seasonality_prior_scale,
            holidays_prior_scale=self.holidays_prior_scale,
            changepoint_prior_scale=self.changepoint_prior_scale
        )
        
        # Add holidays if available
        if self.holidays is not None:
            self.model.add_country_holidays(country_name='US')
            self.model.holidays = self.holidays
        
        # Fit model
        self.model.fit(prophet_data)
        self.is_fitted = True
        
        # Get model components
        future = self.model.make_future_dataframe(periods=0)
        forecast = self.model.predict(future)
        
        results = {
            'model_params': {
                'growth': self.growth,
                'seasonality_mode': self.seasonality_mode,
                'yearly_seasonality': self.yearly_seasonality,
                'weekly_seasonality': self.weekly_seasonality,
                'daily_seasonality': self.daily_seasonality
            },
            'data_info': {
                'n_observations': len(prophet_data),
                'start_date': prophet_data['ds'].min(),
                'end_date': prophet_data['ds'].max(),
                'date_range_days': (prophet_data['ds'].max() - prophet_data['ds'].min()).days
            },
            'seasonality_detected': {
                'yearly': self.yearly_seasonality,
                'weekly': self.weekly_seasonality,
                'daily': self.daily_seasonality
            },
            'holidays_count': len(self.holidays) if self.holidays is not None else 0
        }
        
        logger.info("Prophet model fitting completed")
        
        return results
    
    def forecast(
        self,
        periods: int = 30,
        freq: str = 'D',
        include_history: bool = True
    ) -> Dict[str, Any]:
        """
        Generate forecasts using the fitted Prophet model.
        
        Args:
            periods: Number of periods to forecast
            freq: Frequency of forecasts ('D', 'W', 'M')
            include_history: Whether to include historical data
            
        Returns:
            Forecast results
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        try:
            # Create future dataframe
            future = self.model.make_future_dataframe(periods=periods, freq=freq)
            
            # Generate forecast
            forecast = self.model.predict(future)
            
            # Extract forecast components
            if include_history:
                forecast_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
            else:
                # Only future forecasts
                forecast_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods).copy()
            
            # Calculate confidence intervals
            forecast_data['confidence_interval_width'] = (
                forecast_data['yhat_upper'] - forecast_data['yhat_lower']
            )
            
            results = {
                'forecast': forecast_data.to_dict('records'),
                'forecast_periods': periods,
                'frequency': freq,
                'include_history': include_history,
                'forecast_date': datetime.now().isoformat(),
                'forecast_summary': {
                    'mean_forecast': float(forecast_data['yhat'].mean()),
                    'std_forecast': float(forecast_data['yhat'].std()),
                    'min_forecast': float(forecast_data['yhat'].min()),
                    'max_forecast': float(forecast_data['yhat'].max())
                }
            }
            
            logger.info(f"Generated {periods}-period ahead forecast")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to generate forecast: {e}")
            raise
    
    def predict_next_value(
        self,
        data: pd.DataFrame,
        date_column: str = 'date',
        value_column: str = 'close'
    ) -> Dict[str, Any]:
        """
        Predict next value using Prophet model.
        
        Args:
            data: Recent market data
            date_column: Column containing dates
            value_column: Column containing values
            
        Returns:
            Prediction results
        """
        if not self.is_fitted:
            # Fit model if not already fitted
            self.fit(data, date_column, value_column)
        
        # Generate 1-period ahead forecast
        forecast_results = self.forecast(periods=1, include_history=False)
        
        next_forecast = forecast_results['forecast'][0]
        
        return {
            'predicted_value': next_forecast['yhat'],
            'confidence_interval': {
                'lower': next_forecast['yhat_lower'],
                'upper': next_forecast['yhat_upper']
            },
            'confidence_width': next_forecast['confidence_interval_width'],
            'model_type': 'Prophet',
            'model_params': {
                'growth': self.growth,
                'seasonality_mode': self.seasonality_mode
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def get_components(
        self,
        periods: int = 30
    ) -> Dict[str, Any]:
        """
        Get forecast components (trend, seasonality, etc.).
        
        Args:
            periods: Number of periods for component analysis
            
        Returns:
            Component analysis results
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting components")
        
        try:
            # Create future dataframe
            future = self.model.make_future_dataframe(periods=periods)
            
            # Generate forecast
            forecast = self.model.predict(future)
            
            # Extract components
            components = {
                'trend': forecast['trend'].tolist(),
                'yearly': forecast.get('yearly', []).tolist() if 'yearly' in forecast.columns else [],
                'weekly': forecast.get('weekly', []).tolist() if 'weekly' in forecast.columns else [],
                'daily': forecast.get('daily', []).tolist() if 'daily' in forecast.columns else [],
                'holidays': forecast.get('holidays', []).tolist() if 'holidays' in forecast.columns else [],
                'dates': forecast['ds'].dt.strftime('%Y-%m-%d').tolist()
            }
            
            # Calculate component statistics
            component_stats = {}
            for component, values in components.items():
                if component != 'dates' and values:
                    component_stats[component] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values))
                    }
            
            results = {
                'components': components,
                'component_stats': component_stats,
                'analysis_periods': periods,
                'analysis_date': datetime.now().isoformat()
            }
            
            logger.info("Generated component analysis")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get components: {e}")
            raise
    
    def detect_changepoints(self) -> Dict[str, Any]:
        """
        Detect changepoints in the time series.
        
        Returns:
            Changepoint detection results
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before detecting changepoints")
        
        try:
            # Get changepoints from the model
            changepoints = self.model.changepoints
            
            # Get changepoint dates
            changepoint_dates = changepoints.dt.strftime('%Y-%m-%d').tolist()
            
            # Calculate changepoint significance (simplified)
            changepoint_significance = []
            for i, date in enumerate(changepoint_dates):
                # This is a simplified significance measure
                significance = 0.5 + np.random.random() * 0.5  # Placeholder
                changepoint_significance.append(significance)
            
            results = {
                'changepoint_dates': changepoint_dates,
                'changepoint_count': len(changepoint_dates),
                'changepoint_significance': changepoint_significance,
                'detection_date': datetime.now().isoformat()
            }
            
            logger.info(f"Detected {len(changepoint_dates)} changepoints")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to detect changepoints: {e}")
            raise
    
    def save_model(self, model_dir: str) -> None:
        """Save the fitted model to disk."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Save Prophet model
        model_path = os.path.join(model_dir, f"{self.model_name}.pkl")
        joblib.dump(self.model, model_path)
        
        # Save holidays if available
        if self.holidays is not None:
            holidays_path = os.path.join(model_dir, f"{self.model_name}_holidays.pkl")
            joblib.dump(self.holidays, holidays_path)
        
        # Save metadata
        metadata = {
            'growth': self.growth,
            'seasonality_mode': self.seasonality_mode,
            'yearly_seasonality': self.yearly_seasonality,
            'weekly_seasonality': self.weekly_seasonality,
            'daily_seasonality': self.daily_seasonality,
            'seasonality_prior_scale': self.seasonality_prior_scale,
            'holidays_prior_scale': self.holidays_prior_scale,
            'changepoint_prior_scale': self.changepoint_prior_scale,
            'is_fitted': self.is_fitted
        }
        
        metadata_path = os.path.join(model_dir, f"{self.model_name}_metadata.pkl")
        joblib.dump(metadata, metadata_path)
        
        logger.info(f"Prophet model saved to {model_dir}")
    
    def load_model(self, model_dir: str) -> None:
        """Load a fitted model from disk."""
        # Load Prophet model
        model_path = os.path.join(model_dir, f"{self.model_name}.pkl")
        self.model = joblib.load(model_path)
        
        # Load holidays if available
        holidays_path = os.path.join(model_dir, f"{self.model_name}_holidays.pkl")
        if os.path.exists(holidays_path):
            self.holidays = joblib.load(holidays_path)
        
        # Load metadata
        metadata_path = os.path.join(model_dir, f"{self.model_name}_metadata.pkl")
        metadata = joblib.load(metadata_path)
        
        self.growth = metadata['growth']
        self.seasonality_mode = metadata['seasonality_mode']
        self.yearly_seasonality = metadata['yearly_seasonality']
        self.weekly_seasonality = metadata['weekly_seasonality']
        self.daily_seasonality = metadata['daily_seasonality']
        self.seasonality_prior_scale = metadata['seasonality_prior_scale']
        self.holidays_prior_scale = metadata['holidays_prior_scale']
        self.changepoint_prior_scale = metadata['changepoint_prior_scale']
        self.is_fitted = metadata['is_fitted']
        
        logger.info(f"Prophet model loaded from {model_dir}")


class ProphetAnomalyDetector:
    """
    Prophet-based anomaly detector for time series data.
    
    This model uses Prophet to detect anomalies by comparing
    actual values with forecasted values and confidence intervals.
    """
    
    def __init__(
        self,
        anomaly_threshold: float = 2.0,
        confidence_level: float = 0.95,
        model_name: str = "prophet_anomaly_detector"
    ):
        """
        Initialize Prophet anomaly detector.
        
        Args:
            anomaly_threshold: Threshold for anomaly detection (in standard deviations)
            confidence_level: Confidence level for prediction intervals
            model_name: Name for model saving/loading
        """
        self.anomaly_threshold = anomaly_threshold
        self.confidence_level = confidence_level
        self.model_name = model_name
        
        self.prophet_model = ProphetPredictor()
        self.is_fitted = False
        
        logger.info(f"Initialized Prophet Anomaly Detector: {model_name}")
    
    def fit(
        self,
        data: pd.DataFrame,
        date_column: str = 'date',
        value_column: str = 'close'
    ) -> Dict[str, Any]:
        """
        Fit Prophet model for anomaly detection.
        
        Args:
            data: Input DataFrame with time series data
            date_column: Column containing dates
            value_column: Column containing values
            
        Returns:
            Fitting results
        """
        logger.info("Starting Prophet anomaly detector fitting")
        
        # Fit Prophet model
        fit_results = self.prophet_model.fit(data, date_column, value_column)
        self.is_fitted = True
        
        logger.info("Prophet anomaly detector fitting completed")
        
        return fit_results
    
    def detect_anomalies(
        self,
        data: pd.DataFrame,
        date_column: str = 'date',
        value_column: str = 'close'
    ) -> Dict[str, Any]:
        """
        Detect anomalies in the time series data.
        
        Args:
            data: Input DataFrame with time series data
            date_column: Column containing dates
            value_column: Column containing values
            
        Returns:
            Anomaly detection results
        """
        if not self.is_fitted:
            # Fit model if not already fitted
            self.fit(data, date_column, value_column)
        
        try:
            # Prepare data
            prophet_data = self.prophet_model.prepare_data(data, date_column, value_column)
            
            # Generate forecast for historical data
            future = self.prophet_model.model.make_future_dataframe(periods=0)
            forecast = self.prophet_model.model.predict(future)
            
            # Calculate residuals
            residuals = prophet_data['y'] - forecast['yhat']
            
            # Calculate anomaly scores
            residual_std = residuals.std()
            anomaly_scores = np.abs(residuals) / residual_std
            
            # Detect anomalies
            anomalies = anomaly_scores > self.anomaly_threshold
            
            # Get anomaly details
            anomaly_indices = np.where(anomalies)[0]
            anomaly_dates = prophet_data.loc[anomalies, 'ds'].tolist()
            anomaly_values = prophet_data.loc[anomalies, 'y'].tolist()
            anomaly_scores_list = anomaly_scores[anomalies].tolist()
            
            results = {
                'anomalies_detected': int(np.sum(anomalies)),
                'total_samples': len(prophet_data),
                'anomaly_rate': float(np.mean(anomalies)),
                'anomaly_threshold': self.anomaly_threshold,
                'anomaly_indices': anomaly_indices.tolist(),
                'anomaly_dates': [date.strftime('%Y-%m-%d') for date in anomaly_dates],
                'anomaly_values': anomaly_values,
                'anomaly_scores': anomaly_scores_list,
                'residual_stats': {
                    'mean': float(residuals.mean()),
                    'std': float(residuals.std()),
                    'min': float(residuals.min()),
                    'max': float(residuals.max())
                },
                'detection_date': datetime.now().isoformat()
            }
            
            logger.info(f"Detected {results['anomalies_detected']} anomalies")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to detect anomalies: {e}")
            raise
    
    def save_model(self, model_dir: str) -> None:
        """Save the fitted model to disk."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Save Prophet model
        self.prophet_model.save_model(model_dir)
        
        # Save anomaly detector metadata
        metadata = {
            'anomaly_threshold': self.anomaly_threshold,
            'confidence_level': self.confidence_level,
            'is_fitted': self.is_fitted
        }
        
        metadata_path = os.path.join(model_dir, f"{self.model_name}_metadata.pkl")
        joblib.dump(metadata, metadata_path)
        
        logger.info(f"Prophet anomaly detector saved to {model_dir}")
    
    def load_model(self, model_dir: str) -> None:
        """Load a fitted model from disk."""
        # Load Prophet model
        self.prophet_model.load_model(model_dir)
        
        # Load anomaly detector metadata
        metadata_path = os.path.join(model_dir, f"{self.model_name}_metadata.pkl")
        metadata = joblib.load(metadata_path)
        
        self.anomaly_threshold = metadata['anomaly_threshold']
        self.confidence_level = metadata['confidence_level']
        self.is_fitted = metadata['is_fitted']
        
        logger.info(f"Prophet anomaly detector loaded from {model_dir}")

