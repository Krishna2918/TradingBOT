"""
Advanced Volatility Forecasting System

This module implements advanced machine learning models for forecasting market volatility
using GARCH models, machine learning ensembles, and regime-aware forecasting.

Author: AI Trading System
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class VolatilityForecast:
    """Volatility forecast with confidence intervals and metadata."""
    timestamp: datetime
    forecast_horizon: int  # Days ahead
    point_forecast: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    confidence_level: float
    volatility_regime: str
    indicators: Dict[str, float]
    explanation: str
    model_used: str

@dataclass
class VolatilityMetrics:
    """Volatility forecasting performance metrics."""
    mse: float
    mae: float
    rmse: float
    mape: float
    r2_score: float
    directional_accuracy: float

class VolatilityForecaster:
    """
    Advanced volatility forecasting system using multiple ML models and statistical methods.
    
    Features:
    - Multi-horizon volatility forecasting (1, 5, 10, 20 days)
    - Regime-aware volatility forecasting
    - Confidence interval estimation
    - Volatility clustering detection
    - Real-time volatility prediction
    - Historical volatility pattern recognition
    """
    
    def __init__(self, lookback_days: int = 252, forecast_horizons: List[int] = [1, 5, 10, 20]):
        """
        Initialize volatility forecaster.
        
        Args:
            lookback_days: Number of days to look back for training data
            forecast_horizons: List of forecast horizons in days
        """
        self.lookback_days = lookback_days
        self.forecast_horizons = forecast_horizons
        
        # ML Models for different horizons
        self.models = {}
        self.scalers = {}
        
        for horizon in forecast_horizons:
            self.models[horizon] = {
                'rf': RandomForestRegressor(n_estimators=100, random_state=42),
                'gb': GradientBoostingRegressor(n_estimators=100, random_state=42)
            }
            self.scalers[horizon] = StandardScaler()
        
        # Volatility regime model
        self.regime_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.regime_scaler = StandardScaler()
        
        # Model state
        self.is_trained = False
        self.volatility_regimes = ['LOW', 'NORMAL', 'HIGH', 'EXTREME']
        
        # Performance tracking
        self.performance_metrics = {}
        self.forecast_history = []
        
        logger.info(f"VolatilityForecaster initialized with {lookback_days} day lookback, horizons: {forecast_horizons}")
    
    def _calculate_volatility_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive volatility indicators from market data.
        
        Args:
            data: Market data with OHLCV columns
            
        Returns:
            DataFrame with volatility indicators
        """
        df = data.copy()
        
        # Basic returns and volatility
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['abs_returns'] = abs(df['returns'])
        
        # Realized volatility (multiple horizons)
        df['rv_1d'] = df['abs_returns']
        df['rv_5d'] = df['abs_returns'].rolling(5).sum()
        df['rv_10d'] = df['abs_returns'].rolling(10).sum()
        df['rv_20d'] = df['abs_returns'].rolling(20).sum()
        
        # Volatility of volatility
        df['vol_of_vol'] = df['rv_1d'].rolling(20).std()
        
        # Volatility ratios
        df['vol_ratio_5_20'] = df['rv_5d'] / df['rv_20d']
        df['vol_ratio_10_20'] = df['rv_10d'] / df['rv_20d']
        
        # Volatility percentiles
        df['vol_percentile_20'] = df['rv_1d'].rolling(252).rank(pct=True)
        df['vol_percentile_50'] = df['rv_5d'].rolling(252).rank(pct=True)
        
        # GARCH-like indicators
        df['garch_alpha'] = self._calculate_garch_alpha(df)
        df['garch_beta'] = self._calculate_garch_beta(df)
        
        # Volatility clustering
        df['vol_clustering'] = self._calculate_volatility_clustering(df)
        
        # Volume-volatility relationship
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['vol_volume_corr'] = df['rv_1d'].rolling(20).corr(df['volume_ratio'])
        
        # Price-volatility relationship
        df['price_vol_corr'] = df['returns'].rolling(20).corr(df['rv_1d'])
        
        # Volatility momentum
        df['vol_momentum'] = df['rv_1d'] / df['rv_1d'].shift(5)
        df['vol_acceleration'] = df['vol_momentum'] - df['vol_momentum'].shift(5)
        
        # Volatility mean reversion
        df['vol_mean_reversion'] = self._calculate_volatility_mean_reversion(df)
        
        # Volatility regime indicators
        df['vol_regime_score'] = self._calculate_volatility_regime_score(df)
        
        return df
    
    def _calculate_garch_alpha(self, df: pd.DataFrame) -> pd.Series:
        """Calculate GARCH alpha parameter (simplified)."""
        # Simplified GARCH alpha calculation
        returns = df['returns']
        squared_returns = returns ** 2
        lagged_squared_returns = squared_returns.shift(1)
        
        # Rolling correlation between squared returns and lagged squared returns
        garch_alpha = squared_returns.rolling(20).corr(lagged_squared_returns)
        
        return garch_alpha.fillna(0)
    
    def _calculate_garch_beta(self, df: pd.DataFrame) -> pd.Series:
        """Calculate GARCH beta parameter (simplified)."""
        # Simplified GARCH beta calculation
        returns = df['returns']
        squared_returns = returns ** 2
        lagged_volatility = df['rv_1d'].shift(1)
        
        # Rolling correlation between squared returns and lagged volatility
        garch_beta = squared_returns.rolling(20).corr(lagged_volatility)
        
        return garch_beta.fillna(0)
    
    def _calculate_volatility_clustering(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volatility clustering indicator."""
        # Volatility clustering is measured by autocorrelation of squared returns
        returns = df['returns']
        squared_returns = returns ** 2
        
        # Rolling autocorrelation of squared returns
        vol_clustering = squared_returns.rolling(20).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 1 else 0
        )
        
        return vol_clustering.fillna(0)
    
    def _calculate_volatility_mean_reversion(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volatility mean reversion indicator."""
        # Mean reversion is measured by how far current volatility is from long-term mean
        rv_1d = df['rv_1d']
        long_term_mean = rv_1d.rolling(252).mean()
        long_term_std = rv_1d.rolling(252).std()
        
        # Z-score of current volatility
        mean_reversion = (rv_1d - long_term_mean) / long_term_std
        
        return mean_reversion.fillna(0)
    
    def _calculate_volatility_regime_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volatility regime score."""
        # Combine multiple volatility indicators
        vol_percentile = df['vol_percentile_20']
        vol_ratio = df['vol_ratio_5_20']
        vol_clustering = df['vol_clustering']
        mean_reversion = abs(df['vol_mean_reversion'])
        
        # Weighted combination
        regime_score = (
            0.3 * vol_percentile +
            0.25 * (vol_ratio - 1) +
            0.25 * vol_clustering +
            0.2 * mean_reversion
        )
        
        return regime_score
    
    def _create_volatility_targets(self, data: pd.DataFrame) -> Dict[int, pd.Series]:
        """
        Create volatility targets for different forecast horizons.
        
        Args:
            data: Market data
            
        Returns:
            Dictionary with volatility targets for each horizon
        """
        df = data.copy()
        df['returns'] = df['close'].pct_change()
        df['abs_returns'] = abs(df['returns'])
        
        targets = {}
        for horizon in self.forecast_horizons:
            # Future realized volatility
            future_vol = df['abs_returns'].shift(-horizon).rolling(horizon).sum()
            targets[horizon] = future_vol
        
        return targets
    
    def _create_volatility_regime_labels(self, data: pd.DataFrame) -> pd.Series:
        """
        Create volatility regime labels.
        
        Args:
            data: Market data
            
        Returns:
            Series with volatility regime labels
        """
        df = data.copy()
        df['returns'] = df['close'].pct_change()
        df['rv_1d'] = abs(df['returns'])
        
        # Calculate volatility percentiles
        vol_percentile = df['rv_1d'].rolling(252).rank(pct=True)
        
        # Create regime labels
        regime_labels = pd.Series('NORMAL', index=df.index)
        regime_labels[vol_percentile < 0.25] = 'LOW'
        regime_labels[vol_percentile > 0.75] = 'HIGH'
        regime_labels[vol_percentile > 0.9] = 'EXTREME'
        
        return regime_labels
    
    def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the volatility forecasting models.
        
        Args:
            data: Historical market data for training
            
        Returns:
            Training results and performance metrics
        """
        logger.info("Training volatility forecasting models...")
        
        # Calculate indicators
        df_with_indicators = self._calculate_volatility_indicators(data)
        
        # Create targets
        volatility_targets = self._create_volatility_targets(data)
        regime_labels = self._create_volatility_regime_labels(data)
        
        # Prepare features
        feature_columns = [
            'rv_1d', 'rv_5d', 'rv_10d', 'rv_20d', 'vol_of_vol',
            'vol_ratio_5_20', 'vol_ratio_10_20', 'vol_percentile_20', 'vol_percentile_50',
            'garch_alpha', 'garch_beta', 'vol_clustering', 'volume_ratio',
            'vol_volume_corr', 'price_vol_corr', 'vol_momentum', 'vol_acceleration',
            'vol_mean_reversion', 'vol_regime_score'
        ]
        
        # Remove rows with NaN values
        valid_mask = df_with_indicators[feature_columns].notna().all(axis=1)
        for horizon in self.forecast_horizons:
            valid_mask = valid_mask & volatility_targets[horizon].notna()
        valid_mask = valid_mask & regime_labels.notna()
        
        X = df_with_indicators.loc[valid_mask, feature_columns]
        
        # Train models for each horizon
        training_results = {}
        for horizon in self.forecast_horizons:
            y = volatility_targets[horizon].loc[valid_mask]
            
            if len(X) < 100:
                logger.warning(f"Insufficient training data for horizon {horizon}")
                continue
            
            # Scale features
            X_scaled = self.scalers[horizon].fit_transform(X)
            
            # Train models
            self.models[horizon]['rf'].fit(X_scaled, y)
            self.models[horizon]['gb'].fit(X_scaled, y)
            
            # Evaluate performance
            rf_pred = self.models[horizon]['rf'].predict(X_scaled)
            gb_pred = self.models[horizon]['gb'].predict(X_scaled)
            ensemble_pred = (rf_pred + gb_pred) / 2
            
            # Calculate metrics
            mse = mean_squared_error(y, ensemble_pred)
            mae = mean_absolute_error(y, ensemble_pred)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((y - ensemble_pred) / y)) * 100
            r2 = r2_score(y, ensemble_pred)
            
            # Directional accuracy
            y_direction = np.sign(y.diff())
            pred_direction = np.sign(pd.Series(ensemble_pred).diff())
            directional_accuracy = (y_direction == pred_direction).mean()
            
            self.performance_metrics[horizon] = VolatilityMetrics(
                mse=mse,
                mae=mae,
                rmse=rmse,
                mape=mape,
                r2_score=r2,
                directional_accuracy=directional_accuracy
            )
            
            training_results[horizon] = {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'r2_score': r2,
                'directional_accuracy': directional_accuracy,
                'training_samples': len(X)
            }
        
        # Train regime model
        y_regime = regime_labels.loc[valid_mask]
        X_scaled_regime = self.regime_scaler.fit_transform(X)
        self.regime_model.fit(X_scaled_regime, y_regime)
        
        self.is_trained = True
        
        logger.info(f"Volatility forecasting models trained successfully for horizons: {list(training_results.keys())}")
        
        return {
            'horizon_results': training_results,
            'regime_training_samples': len(X),
            'regime_distribution': y_regime.value_counts().to_dict()
        }
    
    def predict_volatility(self, data: pd.DataFrame, horizon: int) -> VolatilityForecast:
        """
        Predict volatility for a specific horizon.
        
        Args:
            data: Recent market data
            horizon: Forecast horizon in days
            
        Returns:
            VolatilityForecast with point forecast and confidence intervals
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        if horizon not in self.forecast_horizons:
            raise ValueError(f"Horizon {horizon} not supported. Available horizons: {self.forecast_horizons}")
        
        # Calculate indicators for latest data
        df_with_indicators = self._calculate_volatility_indicators(data)
        latest_data = df_with_indicators.iloc[-1:]
        
        # Prepare features
        feature_columns = [
            'rv_1d', 'rv_5d', 'rv_10d', 'rv_20d', 'vol_of_vol',
            'vol_ratio_5_20', 'vol_ratio_10_20', 'vol_percentile_20', 'vol_percentile_50',
            'garch_alpha', 'garch_beta', 'vol_clustering', 'volume_ratio',
            'vol_volume_corr', 'price_vol_corr', 'vol_momentum', 'vol_acceleration',
            'vol_mean_reversion', 'vol_regime_score'
        ]
        
        X = latest_data[feature_columns].fillna(0)
        X_scaled = self.scalers[horizon].transform(X)
        
        # Get predictions from both models
        rf_pred = self.models[horizon]['rf'].predict(X_scaled)[0]
        gb_pred = self.models[horizon]['gb'].predict(X_scaled)[0]
        
        # Ensemble prediction
        point_forecast = (rf_pred + gb_pred) / 2
        
        # Calculate confidence intervals (simplified)
        prediction_std = abs(rf_pred - gb_pred) / 2
        confidence_interval_lower = point_forecast - 1.96 * prediction_std
        confidence_interval_upper = point_forecast + 1.96 * prediction_std
        
        # Predict volatility regime
        X_scaled_regime = self.regime_scaler.transform(X)
        regime_pred = self.regime_model.predict(X_scaled_regime)[0]
        
        # Create explanation
        explanation = self._generate_explanation(latest_data.iloc[0], point_forecast, horizon)
        
        forecast = VolatilityForecast(
            timestamp=datetime.now(),
            forecast_horizon=horizon,
            point_forecast=point_forecast,
            confidence_interval_lower=confidence_interval_lower,
            confidence_interval_upper=confidence_interval_upper,
            confidence_level=0.95,
            volatility_regime=regime_pred,
            indicators=latest_data.iloc[0].to_dict(),
            explanation=explanation,
            model_used='Ensemble RF+GB'
        )
        
        # Store forecast for performance tracking
        self.forecast_history.append(forecast)
        
        return forecast
    
    def predict_volatility_ensemble(self, data: pd.DataFrame) -> Dict[int, VolatilityForecast]:
        """
        Predict volatility for all supported horizons.
        
        Args:
            data: Recent market data
            
        Returns:
            Dictionary with volatility forecasts for each horizon
        """
        forecasts = {}
        for horizon in self.forecast_horizons:
            forecasts[horizon] = self.predict_volatility(data, horizon)
        
        return forecasts
    
    def _generate_explanation(self, indicators: pd.Series, forecast: float, horizon: int) -> str:
        """Generate human-readable explanation for volatility forecast."""
        explanations = []
        
        if indicators['vol_percentile_20'] > 0.8:
            explanations.append("High current volatility percentile")
        
        if indicators['vol_clustering'] > 0.3:
            explanations.append("Strong volatility clustering")
        
        if indicators['vol_momentum'] > 1.2:
            explanations.append("Accelerating volatility")
        
        if indicators['vol_mean_reversion'] > 2:
            explanations.append("Volatility mean reversion expected")
        
        if indicators['vol_regime_score'] > 0.7:
            explanations.append("High volatility regime")
        
        if not explanations:
            explanations.append("Normal volatility conditions")
        
        return f"{horizon}-day volatility forecast {forecast:.4f}: " + ", ".join(explanations)
    
    def get_performance_metrics(self) -> Dict[int, VolatilityMetrics]:
        """Get model performance metrics for all horizons."""
        return self.performance_metrics
    
    def get_forecast_history(self) -> List[VolatilityForecast]:
        """Get history of volatility forecasts."""
        return self.forecast_history
    
    def validate_model(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate model performance on test data.
        
        Args:
            test_data: Test data for validation
            
        Returns:
            Validation results
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before validation")
        
        logger.info("Validating volatility forecasting models...")
        
        # Calculate indicators
        df_with_indicators = self._calculate_volatility_indicators(test_data)
        volatility_targets = self._create_volatility_targets(test_data)
        regime_labels = self._create_volatility_regime_labels(test_data)
        
        # Prepare features
        feature_columns = [
            'rv_1d', 'rv_5d', 'rv_10d', 'rv_20d', 'vol_of_vol',
            'vol_ratio_5_20', 'vol_ratio_10_20', 'vol_percentile_20', 'vol_percentile_50',
            'garch_alpha', 'garch_beta', 'vol_clustering', 'volume_ratio',
            'vol_volume_corr', 'price_vol_corr', 'vol_momentum', 'vol_acceleration',
            'vol_mean_reversion', 'vol_regime_score'
        ]
        
        valid_mask = df_with_indicators[feature_columns].notna().all(axis=1)
        for horizon in self.forecast_horizons:
            valid_mask = valid_mask & volatility_targets[horizon].notna()
        valid_mask = valid_mask & regime_labels.notna()
        
        X = df_with_indicators.loc[valid_mask, feature_columns]
        
        validation_results = {}
        for horizon in self.forecast_horizons:
            y = volatility_targets[horizon].loc[valid_mask]
            
            if len(X) == 0:
                validation_results[horizon] = {'error': 'No valid test data'}
                continue
            
            X_scaled = self.scalers[horizon].transform(X)
            
            # Get predictions
            rf_pred = self.models[horizon]['rf'].predict(X_scaled)
            gb_pred = self.models[horizon]['gb'].predict(X_scaled)
            ensemble_pred = (rf_pred + gb_pred) / 2
            
            # Calculate metrics
            mse = mean_squared_error(y, ensemble_pred)
            mae = mean_absolute_error(y, ensemble_pred)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((y - ensemble_pred) / y)) * 100
            r2 = r2_score(y, ensemble_pred)
            
            validation_results[horizon] = {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'r2_score': r2,
                'test_samples': len(X)
            }
        
        logger.info(f"Model validation completed for horizons: {list(validation_results.keys())}")
        
        return validation_results
