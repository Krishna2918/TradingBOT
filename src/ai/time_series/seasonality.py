"""
Seasonality detection and decomposition for financial time series.

This module provides tools for detecting and analyzing seasonal
patterns in financial time series data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime, timedelta
import warnings
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.stattools import periodogram
from scipy import signal
from scipy.stats import kstest
import joblib
import os

logger = logging.getLogger(__name__)

class SeasonalityDetector:
    """
    Seasonality detector for financial time series.
    
    This class provides comprehensive seasonality detection
    using multiple statistical methods.
    """
    
    def __init__(
        self,
        model_name: str = "seasonality_detector"
    ):
        """
        Initialize Seasonality Detector.
        
        Args:
            model_name: Name for model saving/loading
        """
        self.model_name = model_name
        self.seasonality_results = None
        self.is_analyzed = False
        
        logger.info(f"Initialized Seasonality Detector: {model_name}")
    
    def prepare_data(
        self,
        data: pd.DataFrame,
        date_column: str = 'date',
        value_column: str = 'close',
        freq: str = 'D'
    ) -> pd.Series:
        """
        Prepare data for seasonality analysis.
        
        Args:
            data: Input DataFrame with time series data
            date_column: Column containing dates
            value_column: Column containing values
            freq: Frequency of the time series
            
        Returns:
            Time series data for analysis
        """
        # Ensure date column is datetime
        if date_column in data.columns:
            data = data.copy()
            data[date_column] = pd.to_datetime(data[date_column])
            data = data.set_index(date_column)
        else:
            # Use index as date if no date column
            data = data.copy()
            data.index = pd.to_datetime(data.index)
        
        # Select value column and ensure proper frequency
        series = data[value_column].dropna()
        
        # Resample to ensure consistent frequency
        if freq == 'D':
            series = series.resample('D').mean()
        elif freq == 'W':
            series = series.resample('W').mean()
        elif freq == 'M':
            series = series.resample('M').mean()
        
        logger.info(f"Prepared data for seasonality analysis: {len(series)} observations")
        
        return series
    
    def detect_seasonal_periods(
        self,
        series: pd.Series,
        max_period: int = 50
    ) -> Dict[str, Any]:
        """
        Detect seasonal periods using periodogram analysis.
        
        Args:
            series: Time series data
            max_period: Maximum period to consider
            
        Returns:
            Seasonal period detection results
        """
        try:
            # Remove trend and mean
            detrended = signal.detrend(series.values)
            
            # Compute periodogram
            freqs, psd = periodogram(detrended)
            
            # Convert frequencies to periods
            periods = 1 / freqs[1:]  # Exclude zero frequency
            psd_periods = psd[1:]
            
            # Find significant periods
            significant_periods = []
            for i, period in enumerate(periods):
                if period <= max_period and psd_periods[i] > np.mean(psd_periods) + 2 * np.std(psd_periods):
                    significant_periods.append({
                        'period': float(period),
                        'power': float(psd_periods[i]),
                        'significance': float(psd_periods[i] / np.mean(psd_periods))
                    })
            
            # Sort by significance
            significant_periods.sort(key=lambda x: x['significance'], reverse=True)
            
            results = {
                'significant_periods': significant_periods,
                'total_periods_found': len(significant_periods),
                'max_period_analyzed': max_period,
                'periodogram': {
                    'frequencies': freqs.tolist(),
                    'power_spectral_density': psd.tolist()
                }
            }
            
            logger.info(f"Detected {len(significant_periods)} significant seasonal periods")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to detect seasonal periods: {e}")
            return {'error': str(e), 'significant_periods': []}
    
    def test_seasonality(
        self,
        series: pd.Series,
        period: int = 12
    ) -> Dict[str, Any]:
        """
        Test for seasonality at a specific period.
        
        Args:
            series: Time series data
            period: Period to test for seasonality
            
        Returns:
            Seasonality test results
        """
        try:
            # Ensure series length is sufficient
            if len(series) < 2 * period:
                return {
                    'period': period,
                    'sufficient_data': False,
                    'error': f"Insufficient data for period {period}"
                }
            
            # Create seasonal dummy variables
            seasonal_dummies = pd.get_dummies(series.index.month if hasattr(series.index, 'month') else range(len(series)) % period)
            
            # Perform seasonal decomposition
            decomposition = seasonal_decompose(series, model='additive', period=period)
            
            # Extract seasonal component
            seasonal_component = decomposition.seasonal
            
            # Calculate seasonal strength
            seasonal_strength = np.var(seasonal_component) / np.var(series)
            
            # Test for significant seasonality using K-S test
            ks_statistic, ks_pvalue = kstest(seasonal_component, 'norm')
            
            # Calculate seasonal amplitude
            seasonal_amplitude = np.max(seasonal_component) - np.min(seasonal_component)
            
            results = {
                'period': period,
                'sufficient_data': True,
                'seasonal_strength': float(seasonal_strength),
                'seasonal_amplitude': float(seasonal_amplitude),
                'ks_statistic': float(ks_statistic),
                'ks_pvalue': float(ks_pvalue),
                'is_seasonal': seasonal_strength > 0.1 and ks_pvalue < 0.05,
                'seasonal_component': seasonal_component.tolist()
            }
            
            logger.info(f"Seasonality test for period {period}: {'Significant' if results['is_seasonal'] else 'Not significant'}")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to test seasonality for period {period}: {e}")
            return {
                'period': period,
                'error': str(e),
                'is_seasonal': False
            }
    
    def analyze_seasonality(
        self,
        data: pd.DataFrame,
        date_column: str = 'date',
        value_column: str = 'close',
        periods_to_test: List[int] = [7, 12, 30, 365]
    ) -> Dict[str, Any]:
        """
        Comprehensive seasonality analysis.
        
        Args:
            data: Input DataFrame with time series data
            date_column: Column containing dates
            value_column: Column containing values
            periods_to_test: List of periods to test for seasonality
            
        Returns:
            Comprehensive seasonality analysis results
        """
        logger.info("Starting comprehensive seasonality analysis")
        
        # Prepare data
        series = self.prepare_data(data, date_column, value_column)
        
        # Detect seasonal periods
        period_detection = self.detect_seasonal_periods(series)
        
        # Test seasonality at specific periods
        seasonality_tests = {}
        for period in periods_to_test:
            seasonality_tests[f'period_{period}'] = self.test_seasonality(series, period)
        
        # Overall seasonality assessment
        significant_periods = [test for test in seasonality_tests.values() if test.get('is_seasonal', False)]
        has_seasonality = len(significant_periods) > 0
        
        results = {
            'data_info': {
                'n_observations': len(series),
                'start_date': series.index[0].strftime('%Y-%m-%d') if hasattr(series.index[0], 'strftime') else str(series.index[0]),
                'end_date': series.index[-1].strftime('%Y-%m-%d') if hasattr(series.index[-1], 'strftime') else str(series.index[-1]),
                'frequency': pd.infer_freq(series.index) or 'Unknown'
            },
            'period_detection': period_detection,
            'seasonality_tests': seasonality_tests,
            'overall_assessment': {
                'has_seasonality': has_seasonality,
                'significant_periods_count': len(significant_periods),
                'strongest_seasonal_period': max(significant_periods, key=lambda x: x.get('seasonal_strength', 0)) if significant_periods else None
            },
            'analysis_date': datetime.now().isoformat()
        }
        
        self.seasonality_results = results
        self.is_analyzed = True
        
        logger.info(f"Seasonality analysis completed - Seasonality detected: {has_seasonality}")
        
        return results
    
    def get_seasonal_patterns(self) -> Dict[str, Any]:
        """
        Get seasonal patterns from the analysis.
        
        Returns:
            Seasonal patterns results
        """
        if not self.is_analyzed:
            raise ValueError("Seasonality analysis must be performed first")
        
        patterns = {}
        
        for test_name, test_result in self.seasonality_results['seasonality_tests'].items():
            if test_result.get('is_seasonal', False):
                period = test_result['period']
                seasonal_component = test_result.get('seasonal_component', [])
                
                if seasonal_component:
                    # Calculate seasonal pattern statistics
                    pattern_stats = {
                        'mean': float(np.mean(seasonal_component)),
                        'std': float(np.std(seasonal_component)),
                        'min': float(np.min(seasonal_component)),
                        'max': float(np.max(seasonal_component)),
                        'amplitude': float(np.max(seasonal_component) - np.min(seasonal_component))
                    }
                    
                    patterns[f'period_{period}'] = {
                        'period': period,
                        'seasonal_strength': test_result['seasonal_strength'],
                        'pattern_stats': pattern_stats,
                        'seasonal_component': seasonal_component
                    }
        
        return {
            'seasonal_patterns': patterns,
            'total_patterns': len(patterns),
            'analysis_date': datetime.now().isoformat()
        }
    
    def save_results(self, filepath: str) -> None:
        """Save seasonality analysis results to disk."""
        if not self.is_analyzed:
            raise ValueError("Seasonality analysis must be performed first")
        
        results_data = {
            'seasonality_results': self.seasonality_results,
            'is_analyzed': self.is_analyzed,
            'model_name': self.model_name
        }
        
        joblib.dump(results_data, filepath)
        logger.info(f"Seasonality analysis results saved to {filepath}")
    
    def load_results(self, filepath: str) -> None:
        """Load seasonality analysis results from disk."""
        results_data = joblib.load(filepath)
        
        self.seasonality_results = results_data['seasonality_results']
        self.is_analyzed = results_data['is_analyzed']
        self.model_name = results_data['model_name']
        
        logger.info(f"Seasonality analysis results loaded from {filepath}")


class SeasonalDecomposer:
    """
    Seasonal decomposition for financial time series.
    
    This class provides advanced seasonal decomposition
    using multiple methods including STL decomposition.
    """
    
    def __init__(
        self,
        model_name: str = "seasonal_decomposer"
    ):
        """
        Initialize Seasonal Decomposer.
        
        Args:
            model_name: Name for model saving/loading
        """
        self.model_name = model_name
        self.decomposition_results = None
        self.is_decomposed = False
        
        logger.info(f"Initialized Seasonal Decomposer: {model_name}")
    
    def prepare_data(
        self,
        data: pd.DataFrame,
        date_column: str = 'date',
        value_column: str = 'close'
    ) -> pd.Series:
        """
        Prepare data for seasonal decomposition.
        
        Args:
            data: Input DataFrame with time series data
            date_column: Column containing dates
            value_column: Column containing values
            
        Returns:
            Time series data for decomposition
        """
        # Ensure date column is datetime
        if date_column in data.columns:
            data = data.copy()
            data[date_column] = pd.to_datetime(data[date_column])
            data = data.set_index(date_column)
        else:
            # Use index as date if no date column
            data = data.copy()
            data.index = pd.to_datetime(data.index)
        
        # Select value column
        series = data[value_column].dropna()
        
        logger.info(f"Prepared data for seasonal decomposition: {len(series)} observations")
        
        return series
    
    def decompose_additive(
        self,
        series: pd.Series,
        period: int = 12
    ) -> Dict[str, Any]:
        """
        Perform additive seasonal decomposition.
        
        Args:
            series: Time series data
            period: Seasonal period
            
        Returns:
            Additive decomposition results
        """
        try:
            # Perform additive decomposition
            decomposition = seasonal_decompose(series, model='additive', period=period)
            
            results = {
                'decomposition_type': 'additive',
                'period': period,
                'observed': decomposition.observed.tolist(),
                'trend': decomposition.trend.tolist(),
                'seasonal': decomposition.seasonal.tolist(),
                'residual': decomposition.resid.tolist(),
                'dates': series.index.strftime('%Y-%m-%d').tolist() if hasattr(series.index[0], 'strftime') else list(range(len(series)))
            }
            
            logger.info("Additive seasonal decomposition completed")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to perform additive decomposition: {e}")
            return {'error': str(e), 'decomposition_type': 'additive'}
    
    def decompose_multiplicative(
        self,
        series: pd.Series,
        period: int = 12
    ) -> Dict[str, Any]:
        """
        Perform multiplicative seasonal decomposition.
        
        Args:
            series: Time series data
            period: Seasonal period
            
        Returns:
            Multiplicative decomposition results
        """
        try:
            # Perform multiplicative decomposition
            decomposition = seasonal_decompose(series, model='multiplicative', period=period)
            
            results = {
                'decomposition_type': 'multiplicative',
                'period': period,
                'observed': decomposition.observed.tolist(),
                'trend': decomposition.trend.tolist(),
                'seasonal': decomposition.seasonal.tolist(),
                'residual': decomposition.resid.tolist(),
                'dates': series.index.strftime('%Y-%m-%d').tolist() if hasattr(series.index[0], 'strftime') else list(range(len(series)))
            }
            
            logger.info("Multiplicative seasonal decomposition completed")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to perform multiplicative decomposition: {e}")
            return {'error': str(e), 'decomposition_type': 'multiplicative'}
    
    def decompose_stl(
        self,
        series: pd.Series,
        period: int = 12,
        seasonal: int = 7
    ) -> Dict[str, Any]:
        """
        Perform STL (Seasonal and Trend decomposition using Loess) decomposition.
        
        Args:
            series: Time series data
            period: Seasonal period
            seasonal: Seasonal smoothing parameter
            
        Returns:
            STL decomposition results
        """
        try:
            # Perform STL decomposition
            stl = STL(series, seasonal=seasonal, period=period)
            decomposition = stl.fit()
            
            results = {
                'decomposition_type': 'stl',
                'period': period,
                'seasonal': seasonal,
                'observed': decomposition.observed.tolist(),
                'trend': decomposition.trend.tolist(),
                'seasonal': decomposition.seasonal.tolist(),
                'residual': decomposition.resid.tolist(),
                'dates': series.index.strftime('%Y-%m-%d').tolist() if hasattr(series.index[0], 'strftime') else list(range(len(series)))
            }
            
            logger.info("STL seasonal decomposition completed")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to perform STL decomposition: {e}")
            return {'error': str(e), 'decomposition_type': 'stl'}
    
    def decompose(
        self,
        data: pd.DataFrame,
        date_column: str = 'date',
        value_column: str = 'close',
        method: str = 'additive',
        period: int = 12,
        seasonal: int = 7
    ) -> Dict[str, Any]:
        """
        Perform seasonal decomposition using specified method.
        
        Args:
            data: Input DataFrame with time series data
            date_column: Column containing dates
            value_column: Column containing values
            method: Decomposition method ('additive', 'multiplicative', 'stl')
            period: Seasonal period
            seasonal: Seasonal smoothing parameter (for STL)
            
        Returns:
            Decomposition results
        """
        logger.info(f"Starting {method} seasonal decomposition")
        
        # Prepare data
        series = self.prepare_data(data, date_column, value_column)
        
        # Perform decomposition based on method
        if method == 'additive':
            decomposition_results = self.decompose_additive(series, period)
        elif method == 'multiplicative':
            decomposition_results = self.decompose_multiplicative(series, period)
        elif method == 'stl':
            decomposition_results = self.decompose_stl(series, period, seasonal)
        else:
            raise ValueError(f"Unknown decomposition method: {method}")
        
        # Add metadata
        decomposition_results.update({
            'data_info': {
                'n_observations': len(series),
                'start_date': series.index[0].strftime('%Y-%m-%d') if hasattr(series.index[0], 'strftime') else str(series.index[0]),
                'end_date': series.index[-1].strftime('%Y-%m-%d') if hasattr(series.index[-1], 'strftime') else str(series.index[-1])
            },
            'decomposition_date': datetime.now().isoformat()
        })
        
        self.decomposition_results = decomposition_results
        self.is_decomposed = True
        
        logger.info(f"{method.capitalize()} seasonal decomposition completed")
        
        return decomposition_results
    
    def get_component_statistics(self) -> Dict[str, Any]:
        """
        Get statistics for each decomposition component.
        
        Returns:
            Component statistics
        """
        if not self.is_decomposed:
            raise ValueError("Seasonal decomposition must be performed first")
        
        components = ['trend', 'seasonal', 'residual']
        statistics = {}
        
        for component in components:
            if component in self.decomposition_results:
                values = self.decomposition_results[component]
                if values and not isinstance(values, str):  # Check if values exist and are not error strings
                    statistics[component] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'skewness': float(pd.Series(values).skew()),
                        'kurtosis': float(pd.Series(values).kurtosis())
                    }
        
        return {
            'component_statistics': statistics,
            'decomposition_type': self.decomposition_results.get('decomposition_type', 'unknown'),
            'analysis_date': datetime.now().isoformat()
        }
    
    def save_results(self, filepath: str) -> None:
        """Save decomposition results to disk."""
        if not self.is_decomposed:
            raise ValueError("Seasonal decomposition must be performed first")
        
        results_data = {
            'decomposition_results': self.decomposition_results,
            'is_decomposed': self.is_decomposed,
            'model_name': self.model_name
        }
        
        joblib.dump(results_data, filepath)
        logger.info(f"Seasonal decomposition results saved to {filepath}")
    
    def load_results(self, filepath: str) -> None:
        """Load decomposition results from disk."""
        results_data = joblib.load(filepath)
        
        self.decomposition_results = results_data['decomposition_results']
        self.is_decomposed = results_data['is_decomposed']
        self.model_name = results_data['model_name']
        
        logger.info(f"Seasonal decomposition results loaded from {filepath}")

