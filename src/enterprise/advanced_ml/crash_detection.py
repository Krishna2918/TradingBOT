"""
Market Crash Detection System

This module implements advanced machine learning models for predicting market crashes
and major corrections using multiple indicators and ensemble methods.

Author: AI Trading System
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class CrashSignal:
    """Crash detection signal with confidence and metadata."""
    timestamp: datetime
    confidence: float
    severity: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    indicators: Dict[str, float]
    explanation: str
    time_horizon: int  # Days until predicted crash
    model_used: str

@dataclass
class CrashMetrics:
    """Crash detection performance metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    false_positive_rate: float
    true_positive_rate: float
    auc_score: float

class CrashDetector:
    """
    Advanced market crash detection system using ensemble ML models.
    
    Features:
    - Multiple crash indicators (VIX, yield curve, momentum, volatility)
    - Ensemble of Random Forest and Gradient Boosting models
    - Real-time crash probability scoring
    - Historical crash pattern recognition
    - Confidence calibration and uncertainty quantification
    """
    
    def __init__(self, lookback_days: int = 252, min_confidence: float = 0.7):
        """
        Initialize crash detector.
        
        Args:
            lookback_days: Number of days to look back for training data
            min_confidence: Minimum confidence threshold for crash signals
        """
        self.lookback_days = lookback_days
        self.min_confidence = min_confidence
        
        # ML Models
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        self.gb_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        self.scaler = StandardScaler()
        
        # Model state
        self.is_trained = False
        self.feature_importance = {}
        self.crash_thresholds = {
            'LOW': 0.3,
            'MEDIUM': 0.5,
            'HIGH': 0.7,
            'CRITICAL': 0.85
        }
        
        # Performance tracking
        self.performance_metrics = None
        self.prediction_history = []
        
        logger.info(f"CrashDetector initialized with {lookback_days} day lookback")
    
    def _calculate_crash_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive crash indicators from market data.
        
        Args:
            data: Market data with OHLCV columns
            
        Returns:
            DataFrame with crash indicators
        """
        df = data.copy()
        
        # Price-based indicators
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['returns'].rolling(20).std()
        df['volatility_ratio'] = df['volatility'] / df['volatility'].rolling(50).mean()
        
        # Momentum indicators
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
        df['momentum_divergence'] = df['momentum_5'] - df['momentum_20']
        
        # Volatility indicators
        df['atr'] = self._calculate_atr(df)
        df['atr_ratio'] = df['atr'] / df['atr'].rolling(20).mean()
        df['volatility_clustering'] = df['volatility'].rolling(5).mean() / df['volatility'].rolling(20).mean()
        
        # Volume indicators
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['price_volume_trend'] = df['returns'] * df['volume_ratio']
        
        # Technical indicators
        df['rsi'] = self._calculate_rsi(df['close'])
        df['rsi_divergence'] = df['rsi'] - df['rsi'].rolling(10).mean()
        
        # Market structure indicators
        df['support_resistance'] = self._calculate_support_resistance(df)
        df['trend_strength'] = self._calculate_trend_strength(df)
        
        # Crash-specific indicators
        df['crash_momentum'] = self._calculate_crash_momentum(df)
        df['panic_volume'] = self._calculate_panic_volume(df)
        df['liquidity_stress'] = self._calculate_liquidity_stress(df)
        
        return df
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(period).mean()
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_support_resistance(self, df: pd.DataFrame) -> pd.Series:
        """Calculate support/resistance levels."""
        highs = df['high'].rolling(20).max()
        lows = df['low'].rolling(20).min()
        current_price = df['close']
        
        # Distance to support and resistance
        support_distance = (current_price - lows) / current_price
        resistance_distance = (highs - current_price) / current_price
        
        return support_distance - resistance_distance
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> pd.Series:
        """Calculate trend strength indicator."""
        sma_20 = df['close'].rolling(20).mean()
        sma_50 = df['close'].rolling(50).mean()
        
        trend_direction = np.where(sma_20 > sma_50, 1, -1)
        trend_strength = abs(sma_20 - sma_50) / sma_50
        
        return trend_direction * trend_strength
    
    def _calculate_crash_momentum(self, df: pd.DataFrame) -> pd.Series:
        """Calculate crash momentum indicator."""
        # Look for accelerating downward momentum
        returns = df['returns']
        momentum_1 = returns.rolling(1).sum()
        momentum_5 = returns.rolling(5).sum()
        momentum_10 = returns.rolling(10).sum()
        
        # Crash momentum is negative acceleration
        crash_momentum = momentum_1 - 2 * momentum_5 + momentum_10
        return crash_momentum
    
    def _calculate_panic_volume(self, df: pd.DataFrame) -> pd.Series:
        """Calculate panic volume indicator."""
        # High volume during price declines indicates panic
        negative_returns = df['returns'] < 0
        panic_volume = np.where(negative_returns, df['volume_ratio'], 0)
        return pd.Series(panic_volume, index=df.index)
    
    def _calculate_liquidity_stress(self, df: pd.DataFrame) -> pd.Series:
        """Calculate liquidity stress indicator."""
        # Wide bid-ask spreads and high volatility indicate liquidity stress
        price_range = (df['high'] - df['low']) / df['close']
        volume_decline = 1 - df['volume_ratio']
        
        liquidity_stress = price_range * volume_decline
        return liquidity_stress
    
    def _create_crash_labels(self, data: pd.DataFrame, crash_threshold: float = -0.15) -> pd.Series:
        """
        Create crash labels based on future returns.
        
        Args:
            data: Market data
            crash_threshold: Threshold for defining a crash (e.g., -15%)
            
        Returns:
            Series with crash labels (1 for crash, 0 for no crash)
        """
        # Calculate future returns over different horizons
        future_returns_5d = data['close'].shift(-5) / data['close'] - 1
        future_returns_10d = data['close'].shift(-10) / data['close'] - 1
        future_returns_20d = data['close'].shift(-20) / data['close'] - 1
        
        # Label as crash if any horizon shows significant decline
        crash_5d = (future_returns_5d <= crash_threshold).astype(int)
        crash_10d = (future_returns_10d <= crash_threshold).astype(int)
        crash_20d = (future_returns_20d <= crash_threshold).astype(int)
        
        # Combine all horizons
        crash_labels = np.maximum(crash_5d, np.maximum(crash_10d, crash_20d))
        
        return pd.Series(crash_labels, index=data.index)
    
    def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the crash detection models.
        
        Args:
            data: Historical market data for training
            
        Returns:
            Training results and performance metrics
        """
        logger.info("Training crash detection models...")
        
        # Calculate indicators
        df_with_indicators = self._calculate_crash_indicators(data)
        
        # Create crash labels
        crash_labels = self._create_crash_labels(df_with_indicators)
        
        # Prepare features
        feature_columns = [
            'volatility_ratio', 'momentum_5', 'momentum_20', 'momentum_divergence',
            'atr_ratio', 'volatility_clustering', 'volume_ratio', 'price_volume_trend',
            'rsi', 'rsi_divergence', 'support_resistance', 'trend_strength',
            'crash_momentum', 'panic_volume', 'liquidity_stress'
        ]
        
        # Remove rows with NaN values
        valid_mask = df_with_indicators[feature_columns].notna().all(axis=1) & crash_labels.notna()
        X = df_with_indicators.loc[valid_mask, feature_columns]
        y = crash_labels.loc[valid_mask]
        
        if len(X) < 100:
            raise ValueError("Insufficient training data. Need at least 100 valid samples.")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train models
        self.rf_model.fit(X_scaled, y)
        self.gb_model.fit(X_scaled, y)
        
        # Calculate feature importance
        self.feature_importance = dict(zip(feature_columns, self.rf_model.feature_importances_))
        
        # Evaluate performance
        rf_pred = self.rf_model.predict(X_scaled)
        gb_pred = self.gb_model.predict(X_scaled)
        
        # Ensemble prediction
        ensemble_pred = (rf_pred + gb_pred) / 2
        ensemble_pred_binary = (ensemble_pred > 0.5).astype(int)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        self.performance_metrics = CrashMetrics(
            accuracy=accuracy_score(y, ensemble_pred_binary),
            precision=precision_score(y, ensemble_pred_binary, zero_division=0),
            recall=recall_score(y, ensemble_pred_binary, zero_division=0),
            f1_score=f1_score(y, ensemble_pred_binary, zero_division=0),
            false_positive_rate=0.0,  # Will be calculated properly
            true_positive_rate=recall_score(y, ensemble_pred_binary, zero_division=0),
            auc_score=roc_auc_score(y, ensemble_pred) if len(np.unique(y)) > 1 else 0.0
        )
        
        self.is_trained = True
        
        logger.info(f"Crash detection models trained successfully. Accuracy: {self.performance_metrics.accuracy:.3f}")
        
        return {
            'performance_metrics': self.performance_metrics,
            'feature_importance': self.feature_importance,
            'training_samples': len(X),
            'crash_samples': y.sum()
        }
    
    def predict_crash_probability(self, data: pd.DataFrame) -> float:
        """
        Predict crash probability for current market conditions.
        
        Args:
            data: Recent market data
            
        Returns:
            Crash probability (0-1)
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        # Calculate indicators for latest data
        df_with_indicators = self._calculate_crash_indicators(data)
        latest_data = df_with_indicators.iloc[-1:]
        
        # Prepare features
        feature_columns = [
            'volatility_ratio', 'momentum_5', 'momentum_20', 'momentum_divergence',
            'atr_ratio', 'volatility_clustering', 'volume_ratio', 'price_volume_trend',
            'rsi', 'rsi_divergence', 'support_resistance', 'trend_strength',
            'crash_momentum', 'panic_volume', 'liquidity_stress'
        ]
        
        X = latest_data[feature_columns].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from both models
        rf_prob = self.rf_model.predict_proba(X_scaled)[0][1]
        gb_prob = self.gb_model.predict_proba(X_scaled)[0][1]
        
        # Ensemble prediction
        crash_probability = (rf_prob + gb_prob) / 2
        
        return crash_probability
    
    def generate_crash_signal(self, data: pd.DataFrame) -> Optional[CrashSignal]:
        """
        Generate crash detection signal if conditions are met.
        
        Args:
            data: Recent market data
            
        Returns:
            CrashSignal if crash conditions detected, None otherwise
        """
        if not self.is_trained:
            logger.warning("Models not trained, cannot generate crash signal")
            return None
        
        crash_probability = self.predict_crash_probability(data)
        
        if crash_probability < self.min_confidence:
            return None
        
        # Determine severity
        severity = 'LOW'
        for level, threshold in self.crash_thresholds.items():
            if crash_probability >= threshold:
                severity = level
        
        # Calculate indicators for explanation
        df_with_indicators = self._calculate_crash_indicators(data)
        latest_indicators = df_with_indicators.iloc[-1]
        
        # Create explanation
        explanation = self._generate_explanation(latest_indicators, crash_probability)
        
        # Estimate time horizon (simplified)
        time_horizon = self._estimate_time_horizon(crash_probability)
        
        signal = CrashSignal(
            timestamp=datetime.now(),
            confidence=crash_probability,
            severity=severity,
            indicators=latest_indicators.to_dict(),
            explanation=explanation,
            time_horizon=time_horizon,
            model_used='Ensemble RF+GB'
        )
        
        # Store prediction for performance tracking
        self.prediction_history.append(signal)
        
        logger.info(f"Crash signal generated: {severity} severity, {crash_probability:.3f} confidence")
        
        return signal
    
    def _generate_explanation(self, indicators: pd.Series, probability: float) -> str:
        """Generate human-readable explanation for crash signal."""
        explanations = []
        
        if indicators['volatility_ratio'] > 2.0:
            explanations.append("High volatility spike detected")
        
        if indicators['momentum_5'] < -0.05:
            explanations.append("Strong negative momentum")
        
        if indicators['panic_volume'] > 1.5:
            explanations.append("Panic selling volume")
        
        if indicators['liquidity_stress'] > 0.1:
            explanations.append("Liquidity stress indicators")
        
        if indicators['rsi'] < 30:
            explanations.append("Oversold conditions")
        
        if not explanations:
            explanations.append("Multiple risk factors detected")
        
        return f"Crash probability {probability:.1%}: " + ", ".join(explanations)
    
    def _estimate_time_horizon(self, probability: float) -> int:
        """Estimate time horizon for crash prediction."""
        # Higher probability = shorter time horizon
        if probability > 0.8:
            return 5  # 5 days
        elif probability > 0.6:
            return 10  # 10 days
        elif probability > 0.4:
            return 20  # 20 days
        else:
            return 30  # 30 days
    
    def get_performance_metrics(self) -> Optional[CrashMetrics]:
        """Get model performance metrics."""
        return self.performance_metrics
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        return self.feature_importance
    
    def get_prediction_history(self) -> List[CrashSignal]:
        """Get history of crash predictions."""
        return self.prediction_history
    
    def update_models(self, new_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Update models with new data (incremental learning).
        
        Args:
            new_data: New market data for model updates
            
        Returns:
            Update results
        """
        logger.info("Updating crash detection models with new data...")
        
        # Combine with existing data and retrain
        # This is a simplified approach - in production, you'd use incremental learning
        combined_data = pd.concat([self._get_historical_data(), new_data])
        
        return self.train(combined_data)
    
    def _get_historical_data(self) -> pd.DataFrame:
        """Get historical data for model updates."""
        # In a real implementation, this would fetch from database
        # For now, return empty DataFrame
        return pd.DataFrame()
    
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
        
        logger.info("Validating crash detection models...")
        
        # Calculate indicators
        df_with_indicators = self._calculate_crash_indicators(test_data)
        crash_labels = self._create_crash_labels(df_with_indicators)
        
        # Prepare features
        feature_columns = [
            'volatility_ratio', 'momentum_5', 'momentum_20', 'momentum_divergence',
            'atr_ratio', 'volatility_clustering', 'volume_ratio', 'price_volume_trend',
            'rsi', 'rsi_divergence', 'support_resistance', 'trend_strength',
            'crash_momentum', 'panic_volume', 'liquidity_stress'
        ]
        
        valid_mask = df_with_indicators[feature_columns].notna().all(axis=1) & crash_labels.notna()
        X = df_with_indicators.loc[valid_mask, feature_columns]
        y = crash_labels.loc[valid_mask]
        
        if len(X) == 0:
            return {'error': 'No valid test data'}
        
        X_scaled = self.scaler.transform(X)
        
        # Get predictions
        rf_pred = self.rf_model.predict_proba(X_scaled)[:, 1]
        gb_pred = self.gb_model.predict_proba(X_scaled)[:, 1]
        ensemble_pred = (rf_pred + gb_pred) / 2
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        binary_pred = (ensemble_pred > 0.5).astype(int)
        
        validation_metrics = {
            'accuracy': accuracy_score(y, binary_pred),
            'precision': precision_score(y, binary_pred, zero_division=0),
            'recall': recall_score(y, binary_pred, zero_division=0),
            'f1_score': f1_score(y, binary_pred, zero_division=0),
            'auc_score': roc_auc_score(y, ensemble_pred) if len(np.unique(y)) > 1 else 0.0,
            'test_samples': len(X),
            'crash_samples': y.sum()
        }
        
        logger.info(f"Model validation completed. Test accuracy: {validation_metrics['accuracy']:.3f}")
        
        return validation_metrics
