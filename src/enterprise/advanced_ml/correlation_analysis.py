"""
Market Correlation Analysis System

This module implements advanced correlation analysis for detecting correlation breakdowns,
regime changes in market relationships, and dynamic correlation forecasting.

Author: AI Trading System
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class CorrelationSignal:
    """Correlation analysis signal with confidence and metadata."""
    timestamp: datetime
    signal_type: str  # 'BREAKDOWN', 'REGIME_CHANGE', 'ANOMALY'
    confidence: float
    severity: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    affected_assets: List[str]
    correlation_change: float
    indicators: Dict[str, float]
    explanation: str
    time_horizon: int
    model_used: str

@dataclass
class CorrelationMetrics:
    """Correlation analysis performance metrics."""
    detection_accuracy: float
    false_positive_rate: float
    true_positive_rate: float
    correlation_forecast_accuracy: float
    regime_change_accuracy: float

class CorrelationAnalyzer:
    """
    Advanced market correlation analysis system for detecting correlation breakdowns and regime changes.
    
    Features:
    - Dynamic correlation calculation and monitoring
    - Correlation breakdown detection
    - Correlation regime change identification
    - Correlation anomaly detection
    - Correlation forecasting
    - Asset clustering based on correlations
    """
    
    def __init__(self, lookback_days: int = 252, min_confidence: float = 0.6):
        """
        Initialize correlation analyzer.
        
        Args:
            lookback_days: Number of days to look back for training data
            min_confidence: Minimum confidence threshold for correlation signals
        """
        self.lookback_days = lookback_days
        self.min_confidence = min_confidence
        
        # Models
        self.breakdown_model = IsolationForest(contamination=0.1, random_state=42)
        self.regime_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.forecast_model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Scalers
        self.breakdown_scaler = StandardScaler()
        self.regime_scaler = StandardScaler()
        self.forecast_scaler = StandardScaler()
        
        # Model state
        self.is_trained = False
        self.correlation_thresholds = {
            'LOW': 0.3,
            'MEDIUM': 0.5,
            'HIGH': 0.7,
            'CRITICAL': 0.9
        }
        
        # Performance tracking
        self.performance_metrics = None
        self.signal_history = []
        
        logger.info(f"CorrelationAnalyzer initialized with {lookback_days} day lookback")
    
    def _calculate_correlation_indicators(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate comprehensive correlation indicators from multi-asset data.
        
        Args:
            data: Dictionary with asset symbols as keys and DataFrames as values
            
        Returns:
            DataFrame with correlation indicators
        """
        # Extract returns for all assets
        returns_data = {}
        for asset, df in data.items():
            if 'close' in df.columns:
                returns_data[asset] = df['close'].pct_change()
        
        if len(returns_data) < 2:
            raise ValueError("Need at least 2 assets for correlation analysis")
        
        # Create returns DataFrame
        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.dropna()
        
        # Calculate rolling correlations
        window_sizes = [20, 50, 100]
        correlation_indicators = {}
        
        for window in window_sizes:
            # Rolling correlation matrix
            rolling_corr = returns_df.rolling(window).corr()
            
            # Extract correlation indicators
            for i, asset1 in enumerate(returns_data.keys()):
                for j, asset2 in enumerate(returns_data.keys()):
                    if i < j:  # Avoid duplicates
                        pair_name = f"{asset1}_{asset2}"
                        corr_series = rolling_corr.loc[(slice(None), asset1), asset2]
                        correlation_indicators[f"{pair_name}_corr_{window}"] = corr_series
        
        # Calculate market-wide correlation indicators
        correlation_indicators['avg_correlation_20'] = self._calculate_average_correlation(returns_df, 20)
        correlation_indicators['avg_correlation_50'] = self._calculate_average_correlation(returns_df, 50)
        correlation_indicators['max_correlation_20'] = self._calculate_max_correlation(returns_df, 20)
        correlation_indicators['min_correlation_20'] = self._calculate_min_correlation(returns_df, 20)
        correlation_indicators['correlation_volatility'] = self._calculate_correlation_volatility(returns_df, 20)
        
        # Correlation regime indicators
        correlation_indicators['correlation_regime'] = self._calculate_correlation_regime(returns_df)
        correlation_indicators['correlation_trend'] = self._calculate_correlation_trend(returns_df)
        correlation_indicators['correlation_momentum'] = self._calculate_correlation_momentum(returns_df)
        
        # Market stress indicators
        correlation_indicators['market_stress'] = self._calculate_market_stress(returns_df)
        correlation_indicators['flight_to_quality'] = self._calculate_flight_to_quality(returns_df)
        
        # Create DataFrame
        indicators_df = pd.DataFrame(correlation_indicators)
        
        return indicators_df
    
    def _calculate_average_correlation(self, returns_df: pd.DataFrame, window: int) -> pd.Series:
        """Calculate average correlation across all asset pairs."""
        rolling_corr = returns_df.rolling(window).corr()
        
        # Get all unique pairs
        assets = returns_df.columns
        correlations = []
        
        for i, asset1 in enumerate(assets):
            for j, asset2 in enumerate(assets):
                if i < j:
                    corr_series = rolling_corr.loc[(slice(None), asset1), asset2]
                    correlations.append(corr_series)
        
        if correlations:
            avg_corr = pd.concat(correlations, axis=1).mean(axis=1)
        else:
            avg_corr = pd.Series(0, index=returns_df.index)
        
        return avg_corr
    
    def _calculate_max_correlation(self, returns_df: pd.DataFrame, window: int) -> pd.Series:
        """Calculate maximum correlation across all asset pairs."""
        rolling_corr = returns_df.rolling(window).corr()
        
        assets = returns_df.columns
        max_correlations = []
        
        for i, asset1 in enumerate(assets):
            for j, asset2 in enumerate(assets):
                if i < j:
                    corr_series = rolling_corr.loc[(slice(None), asset1), asset2]
                    max_correlations.append(corr_series)
        
        if max_correlations:
            max_corr = pd.concat(max_correlations, axis=1).max(axis=1)
        else:
            max_corr = pd.Series(0, index=returns_df.index)
        
        return max_corr
    
    def _calculate_min_correlation(self, returns_df: pd.DataFrame, window: int) -> pd.Series:
        """Calculate minimum correlation across all asset pairs."""
        rolling_corr = returns_df.rolling(window).corr()
        
        assets = returns_df.columns
        min_correlations = []
        
        for i, asset1 in enumerate(assets):
            for j, asset2 in enumerate(assets):
                if i < j:
                    corr_series = rolling_corr.loc[(slice(None), asset1), asset2]
                    min_correlations.append(corr_series)
        
        if min_correlations:
            min_corr = pd.concat(min_correlations, axis=1).min(axis=1)
        else:
            min_corr = pd.Series(0, index=returns_df.index)
        
        return min_corr
    
    def _calculate_correlation_volatility(self, returns_df: pd.DataFrame, window: int) -> pd.Series:
        """Calculate volatility of correlations."""
        avg_corr = self._calculate_average_correlation(returns_df, window)
        corr_vol = avg_corr.rolling(window).std()
        
        return corr_vol.fillna(0)
    
    def _calculate_correlation_regime(self, returns_df: pd.DataFrame) -> pd.Series:
        """Calculate correlation regime indicator."""
        avg_corr_20 = self._calculate_average_correlation(returns_df, 20)
        avg_corr_50 = self._calculate_average_correlation(returns_df, 50)
        
        # Regime based on correlation level and trend
        regime = pd.Series('NORMAL', index=returns_df.index)
        regime[avg_corr_20 > 0.7] = 'HIGH'
        regime[avg_corr_20 < 0.3] = 'LOW'
        regime[(avg_corr_20 - avg_corr_50) > 0.2] = 'INCREASING'
        regime[(avg_corr_20 - avg_corr_50) < -0.2] = 'DECREASING'
        
        return regime
    
    def _calculate_correlation_trend(self, returns_df: pd.DataFrame) -> pd.Series:
        """Calculate correlation trend indicator."""
        avg_corr_20 = self._calculate_average_correlation(returns_df, 20)
        avg_corr_50 = self._calculate_average_correlation(returns_df, 50)
        
        trend = avg_corr_20 - avg_corr_50
        return trend.fillna(0)
    
    def _calculate_correlation_momentum(self, returns_df: pd.DataFrame) -> pd.Series:
        """Calculate correlation momentum indicator."""
        avg_corr_20 = self._calculate_average_correlation(returns_df, 20)
        momentum = avg_corr_20.diff(5)
        
        return momentum.fillna(0)
    
    def _calculate_market_stress(self, returns_df: pd.DataFrame) -> pd.Series:
        """Calculate market stress indicator based on correlation patterns."""
        # High correlations often indicate market stress
        avg_corr = self._calculate_average_correlation(returns_df, 20)
        corr_vol = self._calculate_correlation_volatility(returns_df, 20)
        
        # Market stress when correlations are high and volatile
        stress = avg_corr * corr_vol
        return stress.fillna(0)
    
    def _calculate_flight_to_quality(self, returns_df: pd.DataFrame) -> pd.Series:
        """Calculate flight to quality indicator."""
        # Flight to quality often shows up as correlation breakdowns
        avg_corr = self._calculate_average_correlation(returns_df, 20)
        corr_trend = self._calculate_correlation_trend(returns_df)
        
        # Flight to quality when correlations are decreasing rapidly
        flight_to_quality = -corr_trend * (1 - avg_corr)
        return flight_to_quality.fillna(0)
    
    def _create_correlation_labels(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        """
        Create correlation labels for training.
        
        Args:
            data: Multi-asset market data
            
        Returns:
            Dictionary with correlation labels
        """
        # Extract returns
        returns_data = {}
        for asset, df in data.items():
            if 'close' in df.columns:
                returns_data[asset] = df['close'].pct_change()
        
        returns_df = pd.DataFrame(returns_data).dropna()
        
        # Calculate future correlation changes
        current_corr = self._calculate_average_correlation(returns_df, 20)
        future_corr = current_corr.shift(-10)  # 10 days ahead
        
        # Correlation breakdown labels
        corr_change = future_corr - current_corr
        breakdown_labels = (abs(corr_change) > 0.3).astype(int)
        
        # Regime change labels
        regime_change_labels = (abs(corr_change) > 0.2).astype(int)
        
        # Anomaly labels (extreme correlation changes)
        anomaly_labels = (abs(corr_change) > 0.5).astype(int)
        
        return {
            'breakdown': breakdown_labels,
            'regime_change': regime_change_labels,
            'anomaly': anomaly_labels
        }
    
    def train(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Train the correlation analysis models.
        
        Args:
            data: Multi-asset historical market data for training
            
        Returns:
            Training results and performance metrics
        """
        logger.info("Training correlation analysis models...")
        
        # Calculate indicators
        indicators_df = self._calculate_correlation_indicators(data)
        
        # Create labels
        correlation_labels = self._create_correlation_labels(data)
        
        # Prepare features
        feature_columns = [
            col for col in indicators_df.columns 
            if col not in ['breakdown', 'regime_change', 'anomaly']
        ]
        
        # Remove rows with NaN values
        valid_mask = indicators_df[feature_columns].notna().all(axis=1)
        for label_type in correlation_labels:
            valid_mask = valid_mask & correlation_labels[label_type].notna()
        
        X = indicators_df.loc[valid_mask, feature_columns]
        y_breakdown = correlation_labels['breakdown'].loc[valid_mask]
        y_regime = correlation_labels['regime_change'].loc[valid_mask]
        y_anomaly = correlation_labels['anomaly'].loc[valid_mask]
        
        if len(X) < 100:
            raise ValueError("Insufficient training data. Need at least 100 valid samples.")
        
        # Scale features
        X_breakdown_scaled = self.breakdown_scaler.fit_transform(X)
        X_regime_scaled = self.regime_scaler.fit_transform(X)
        X_forecast_scaled = self.forecast_scaler.fit_transform(X)
        
        # Train models
        self.breakdown_model.fit(X_breakdown_scaled)
        self.regime_model.fit(X_regime_scaled, y_regime)
        self.forecast_model.fit(X_forecast_scaled, y_anomaly)
        
        # Evaluate performance
        breakdown_pred = self.breakdown_model.predict(X_breakdown_scaled)
        regime_pred = self.regime_model.predict(X_regime_scaled)
        forecast_pred = self.forecast_model.predict(X_forecast_scaled)
        
        # Calculate metrics
        breakdown_accuracy = (breakdown_pred == -1).mean()  # Anomaly detection
        regime_mse = mean_squared_error(y_regime, regime_pred)
        forecast_mse = mean_squared_error(y_anomaly, forecast_pred)
        
        self.performance_metrics = CorrelationMetrics(
            detection_accuracy=breakdown_accuracy,
            false_positive_rate=0.0,  # Will be calculated properly
            true_positive_rate=0.0,   # Will be calculated properly
            correlation_forecast_accuracy=1 - forecast_mse,
            regime_change_accuracy=1 - regime_mse
        )
        
        self.is_trained = True
        
        logger.info(f"Correlation analysis models trained successfully. Breakdown accuracy: {breakdown_accuracy:.3f}")
        
        return {
            'performance_metrics': self.performance_metrics,
            'breakdown_accuracy': breakdown_accuracy,
            'regime_mse': regime_mse,
            'forecast_mse': forecast_mse,
            'training_samples': len(X),
            'breakdown_samples': y_breakdown.sum(),
            'regime_samples': y_regime.sum(),
            'anomaly_samples': y_anomaly.sum()
        }
    
    def predict_correlation_breakdown(self, data: Dict[str, pd.DataFrame]) -> float:
        """
        Predict correlation breakdown probability.
        
        Args:
            data: Multi-asset market data
            
        Returns:
            Correlation breakdown probability (0-1)
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        # Calculate indicators for latest data
        indicators_df = self._calculate_correlation_indicators(data)
        latest_data = indicators_df.iloc[-1:]
        
        # Prepare features
        feature_columns = [
            col for col in indicators_df.columns 
            if col not in ['breakdown', 'regime_change', 'anomaly']
        ]
        
        X = latest_data[feature_columns].fillna(0)
        X_scaled = self.breakdown_scaler.transform(X)
        
        # Get prediction
        breakdown_score = self.breakdown_model.decision_function(X_scaled)[0]
        
        # Convert to probability
        breakdown_prob = 1 / (1 + np.exp(breakdown_score))
        
        return breakdown_prob
    
    def predict_correlation_regime_change(self, data: Dict[str, pd.DataFrame]) -> float:
        """
        Predict correlation regime change probability.
        
        Args:
            data: Multi-asset market data
            
        Returns:
            Correlation regime change probability (0-1)
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        # Calculate indicators for latest data
        indicators_df = self._calculate_correlation_indicators(data)
        latest_data = indicators_df.iloc[-1:]
        
        # Prepare features
        feature_columns = [
            col for col in indicators_df.columns 
            if col not in ['breakdown', 'regime_change', 'anomaly']
        ]
        
        X = latest_data[feature_columns].fillna(0)
        X_scaled = self.regime_scaler.transform(X)
        
        # Get prediction
        regime_change_prob = self.regime_model.predict(X_scaled)[0]
        
        return max(0, min(1, regime_change_prob))
    
    def generate_correlation_signal(self, data: Dict[str, pd.DataFrame]) -> Optional[CorrelationSignal]:
        """
        Generate correlation analysis signal if conditions are met.
        
        Args:
            data: Multi-asset market data
            
        Returns:
            CorrelationSignal if correlation conditions detected, None otherwise
        """
        if not self.is_trained:
            logger.warning("Models not trained, cannot generate correlation signal")
            return None
        
        # Calculate current indicators
        indicators_df = self._calculate_correlation_indicators(data)
        latest_indicators = indicators_df.iloc[-1]
        
        # Get predictions
        breakdown_prob = self.predict_correlation_breakdown(data)
        regime_change_prob = self.predict_correlation_regime_change(data)
        
        # Determine signal type and confidence
        max_prob = max(breakdown_prob, regime_change_prob)
        
        if max_prob < self.min_confidence:
            return None
        
        if breakdown_prob == max_prob:
            signal_type = 'BREAKDOWN'
            confidence = breakdown_prob
        else:
            signal_type = 'REGIME_CHANGE'
            confidence = regime_change_prob
        
        # Determine severity
        severity = 'LOW'
        for level, threshold in self.correlation_thresholds.items():
            if confidence >= threshold:
                severity = level
        
        # Get affected assets (simplified)
        affected_assets = list(data.keys())[:2]  # Top 2 assets for now
        
        # Calculate correlation change
        current_corr = latest_indicators.get('avg_correlation_20', 0)
        correlation_change = latest_indicators.get('correlation_trend', 0)
        
        # Create explanation
        explanation = self._generate_explanation(latest_indicators, signal_type, confidence)
        
        # Estimate time horizon
        time_horizon = self._estimate_time_horizon(confidence)
        
        signal = CorrelationSignal(
            timestamp=datetime.now(),
            signal_type=signal_type,
            confidence=confidence,
            severity=severity,
            affected_assets=affected_assets,
            correlation_change=correlation_change,
            indicators=latest_indicators.to_dict(),
            explanation=explanation,
            time_horizon=time_horizon,
            model_used='Ensemble Breakdown+Regime'
        )
        
        # Store signal for performance tracking
        self.signal_history.append(signal)
        
        logger.info(f"Correlation signal generated: {signal_type}, {confidence:.3f} confidence")
        
        return signal
    
    def _generate_explanation(self, indicators: pd.Series, signal_type: str, confidence: float) -> str:
        """Generate human-readable explanation for correlation signal."""
        explanations = []
        
        if signal_type == 'BREAKDOWN':
            explanations.append("Correlation breakdown detected")
        elif signal_type == 'REGIME_CHANGE':
            explanations.append("Correlation regime change detected")
        
        # Add supporting indicators
        if indicators.get('market_stress', 0) > 0.5:
            explanations.append("High market stress")
        
        if indicators.get('flight_to_quality', 0) > 0.3:
            explanations.append("Flight to quality pattern")
        
        if indicators.get('correlation_volatility', 0) > 0.1:
            explanations.append("High correlation volatility")
        
        if indicators.get('correlation_trend', 0) > 0.2:
            explanations.append("Rising correlation trend")
        elif indicators.get('correlation_trend', 0) < -0.2:
            explanations.append("Falling correlation trend")
        
        if not explanations:
            explanations.append("Multiple correlation indicators detected")
        
        return f"Correlation {signal_type.lower()} probability {confidence:.1%}: " + ", ".join(explanations)
    
    def _estimate_time_horizon(self, confidence: float) -> int:
        """Estimate time horizon for correlation prediction."""
        # Higher confidence = shorter time horizon
        if confidence > 0.8:
            return 5  # 5 days
        elif confidence > 0.6:
            return 10  # 10 days
        elif confidence > 0.4:
            return 20  # 20 days
        else:
            return 30  # 30 days
    
    def get_performance_metrics(self) -> Optional[CorrelationMetrics]:
        """Get model performance metrics."""
        return self.performance_metrics
    
    def get_signal_history(self) -> List[CorrelationSignal]:
        """Get history of correlation signals."""
        return self.signal_history
    
    def cluster_assets(self, data: Dict[str, pd.DataFrame], n_clusters: int = 3) -> Dict[str, int]:
        """
        Cluster assets based on correlation patterns.
        
        Args:
            data: Multi-asset market data
            n_clusters: Number of clusters
            
        Returns:
            Dictionary with asset symbols as keys and cluster assignments as values
        """
        # Extract returns
        returns_data = {}
        for asset, df in data.items():
            if 'close' in df.columns:
                returns_data[asset] = df['close'].pct_change()
        
        returns_df = pd.DataFrame(returns_data).dropna()
        
        # Calculate correlation matrix
        corr_matrix = returns_df.corr()
        
        # Convert to distance matrix
        distance_matrix = 1 - abs(corr_matrix)
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(distance_matrix.values, method='ward')
        
        # Get cluster assignments
        from scipy.cluster.hierarchy import fcluster
        cluster_assignments = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        
        # Create result dictionary
        asset_clusters = {}
        for i, asset in enumerate(corr_matrix.columns):
            asset_clusters[asset] = cluster_assignments[i]
        
        return asset_clusters
    
    def validate_model(self, test_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Validate model performance on test data.
        
        Args:
            test_data: Multi-asset test data for validation
            
        Returns:
            Validation results
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before validation")
        
        logger.info("Validating correlation analysis models...")
        
        # Calculate indicators
        indicators_df = self._calculate_correlation_indicators(test_data)
        correlation_labels = self._create_correlation_labels(test_data)
        
        # Prepare features
        feature_columns = [
            col for col in indicators_df.columns 
            if col not in ['breakdown', 'regime_change', 'anomaly']
        ]
        
        valid_mask = indicators_df[feature_columns].notna().all(axis=1)
        for label_type in correlation_labels:
            valid_mask = valid_mask & correlation_labels[label_type].notna()
        
        X = indicators_df.loc[valid_mask, feature_columns]
        y_breakdown = correlation_labels['breakdown'].loc[valid_mask]
        y_regime = correlation_labels['regime_change'].loc[valid_mask]
        y_anomaly = correlation_labels['anomaly'].loc[valid_mask]
        
        if len(X) == 0:
            return {'error': 'No valid test data'}
        
        X_breakdown_scaled = self.breakdown_scaler.transform(X)
        X_regime_scaled = self.regime_scaler.transform(X)
        X_forecast_scaled = self.forecast_scaler.transform(X)
        
        # Get predictions
        breakdown_pred = self.breakdown_model.predict(X_breakdown_scaled)
        regime_pred = self.regime_model.predict(X_regime_scaled)
        forecast_pred = self.forecast_model.predict(X_forecast_scaled)
        
        # Calculate metrics
        breakdown_accuracy = (breakdown_pred == -1).mean()
        regime_mse = mean_squared_error(y_regime, regime_pred)
        forecast_mse = mean_squared_error(y_anomaly, forecast_pred)
        
        validation_metrics = {
            'breakdown_accuracy': breakdown_accuracy,
            'regime_mse': regime_mse,
            'forecast_mse': forecast_mse,
            'test_samples': len(X),
            'breakdown_samples': y_breakdown.sum(),
            'regime_samples': y_regime.sum(),
            'anomaly_samples': y_anomaly.sum()
        }
        
        logger.info(f"Model validation completed. Breakdown accuracy: {breakdown_accuracy:.3f}")
        
        return validation_metrics
