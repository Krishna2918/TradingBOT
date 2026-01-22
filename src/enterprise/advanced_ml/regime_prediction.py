"""
Market Regime Prediction System

This module implements advanced machine learning models for predicting market regime shifts
and transitions between bull/bear markets, high/low volatility periods, and trend/range-bound markets.

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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class RegimeSignal:
    """Regime prediction signal with confidence and metadata."""
    timestamp: datetime
    current_regime: str
    predicted_regime: str
    confidence: float
    transition_probability: float
    time_horizon: int  # Days until predicted transition
    indicators: Dict[str, float]
    explanation: str
    model_used: str

@dataclass
class RegimeMetrics:
    """Regime prediction performance metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    regime_transition_accuracy: float
    false_positive_rate: float

class RegimePredictor:
    """
    Advanced market regime prediction system using ensemble ML models.
    
    Features:
    - Bull/Bear market regime prediction
    - High/Low volatility regime prediction
    - Trend/Range-bound regime prediction
    - Regime transition probability scoring
    - Real-time regime classification
    - Historical regime pattern recognition
    """
    
    def __init__(self, lookback_days: int = 252, min_confidence: float = 0.6):
        """
        Initialize regime predictor.
        
        Args:
            lookback_days: Number of days to look back for training data
            min_confidence: Minimum confidence threshold for regime signals
        """
        self.lookback_days = lookback_days
        self.min_confidence = min_confidence
        
        # ML Models for different regime types
        self.bull_bear_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.volatility_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.trend_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.composite_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        
        # Scalers
        self.scaler = StandardScaler()
        self.label_encoders = {
            'bull_bear': LabelEncoder(),
            'volatility': LabelEncoder(),
            'trend': LabelEncoder()
        }
        
        # Model state
        self.is_trained = False
        self.regime_definitions = {
            'bull_bear': ['BULL', 'BEAR', 'NEUTRAL'],
            'volatility': ['HIGH_VOL', 'LOW_VOL', 'NORMAL_VOL'],
            'trend': ['TRENDING', 'RANGING', 'CHOPPY']
        }
        
        # Performance tracking
        self.performance_metrics = None
        self.prediction_history = []
        
        logger.info(f"RegimePredictor initialized with {lookback_days} day lookback")
    
    def _calculate_regime_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive regime indicators from market data.
        
        Args:
            data: Market data with OHLCV columns
            
        Returns:
            DataFrame with regime indicators
        """
        df = data.copy()
        
        # Price-based indicators
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Volatility indicators
        df['volatility_20'] = df['returns'].rolling(20).std()
        df['volatility_50'] = df['returns'].rolling(50).std()
        df['volatility_ratio'] = df['volatility_20'] / df['volatility_50']
        df['volatility_percentile'] = df['volatility_20'].rolling(252).rank(pct=True)
        
        # Trend indicators
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['sma_200'] = df['close'].rolling(200).mean()
        df['trend_strength'] = (df['sma_20'] - df['sma_50']) / df['sma_50']
        df['trend_direction'] = np.where(df['sma_20'] > df['sma_50'], 1, -1)
        
        # Momentum indicators
        df['momentum_1m'] = df['close'] / df['close'].shift(21) - 1
        df['momentum_3m'] = df['close'] / df['close'].shift(63) - 1
        df['momentum_6m'] = df['close'] / df['close'].shift(126) - 1
        df['momentum_divergence'] = df['momentum_1m'] - df['momentum_3m']
        
        # Volume indicators
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['volume_trend'] = df['volume'].rolling(20).mean() / df['volume'].rolling(50).mean()
        
        # Technical indicators
        df['rsi'] = self._calculate_rsi(df['close'])
        df['rsi_divergence'] = df['rsi'] - df['rsi'].rolling(10).mean()
        
        # Market structure indicators
        df['support_resistance'] = self._calculate_support_resistance(df)
        df['market_breadth'] = self._calculate_market_breadth(df)
        df['correlation_stability'] = self._calculate_correlation_stability(df)
        
        # Regime-specific indicators
        df['bull_bear_score'] = self._calculate_bull_bear_score(df)
        df['volatility_regime_score'] = self._calculate_volatility_regime_score(df)
        df['trend_regime_score'] = self._calculate_trend_regime_score(df)
        
        return df
    
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
    
    def _calculate_market_breadth(self, df: pd.DataFrame) -> pd.Series:
        """Calculate market breadth indicator (simplified)."""
        # In a real implementation, this would use multiple stocks
        # For now, use price momentum as a proxy
        return df['momentum_1m'].rolling(10).mean()
    
    def _calculate_correlation_stability(self, df: pd.DataFrame) -> pd.Series:
        """Calculate correlation stability indicator."""
        # Rolling correlation between returns and volume
        returns_volume_corr = df['returns'].rolling(20).corr(df['volume_ratio'])
        return returns_volume_corr.fillna(0)
    
    def _calculate_bull_bear_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate bull/bear market score."""
        # Combine multiple indicators
        trend_score = df['trend_strength']
        momentum_score = df['momentum_3m']
        volume_score = df['volume_ratio'] - 1
        
        # Weighted combination
        bull_bear_score = (
            0.4 * trend_score +
            0.3 * momentum_score +
            0.3 * volume_score
        )
        
        return bull_bear_score
    
    def _calculate_volatility_regime_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volatility regime score."""
        # Volatility percentile and ratio
        vol_percentile = df['volatility_percentile']
        vol_ratio = df['volatility_ratio']
        
        # Volatility regime score
        volatility_score = (
            0.6 * vol_percentile +
            0.4 * (vol_ratio - 1)
        )
        
        return volatility_score
    
    def _calculate_trend_regime_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate trend regime score."""
        # Trend strength and consistency
        trend_strength = abs(df['trend_strength'])
        trend_consistency = df['trend_direction'].rolling(10).mean()
        
        # Trend regime score
        trend_score = (
            0.5 * trend_strength +
            0.5 * abs(trend_consistency)
        )
        
        return trend_score
    
    def _create_regime_labels(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Create regime labels based on market conditions.
        
        Args:
            data: Market data
            
        Returns:
            Dictionary with regime labels for each regime type
        """
        df = data.copy()
        
        # Bull/Bear labels
        returns_3m = df['close'] / df['close'].shift(63) - 1
        bull_bear_labels = pd.Series('NEUTRAL', index=df.index)
        bull_bear_labels[returns_3m > 0.1] = 'BULL'
        bull_bear_labels[returns_3m < -0.1] = 'BEAR'
        
        # Volatility labels
        volatility_20 = df['returns'].rolling(20).std()
        vol_percentile = volatility_20.rolling(252).rank(pct=True)
        volatility_labels = pd.Series('NORMAL_VOL', index=df.index)
        volatility_labels[vol_percentile > 0.8] = 'HIGH_VOL'
        volatility_labels[vol_percentile < 0.2] = 'LOW_VOL'
        
        # Trend labels
        trend_strength = abs((df['sma_20'] - df['sma_50']) / df['sma_50'])
        trend_labels = pd.Series('CHOPPY', index=df.index)
        trend_labels[trend_strength > 0.05] = 'TRENDING'
        trend_labels[(trend_strength > 0.02) & (trend_strength <= 0.05)] = 'RANGING'
        
        return {
            'bull_bear': bull_bear_labels,
            'volatility': volatility_labels,
            'trend': trend_labels
        }
    
    def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the regime prediction models.
        
        Args:
            data: Historical market data for training
            
        Returns:
            Training results and performance metrics
        """
        logger.info("Training regime prediction models...")
        
        # Calculate indicators
        df_with_indicators = self._calculate_regime_indicators(data)
        
        # Create regime labels
        regime_labels = self._create_regime_labels(df_with_indicators)
        
        # Prepare features
        feature_columns = [
            'volatility_20', 'volatility_50', 'volatility_ratio', 'volatility_percentile',
            'trend_strength', 'trend_direction', 'momentum_1m', 'momentum_3m', 'momentum_6m',
            'momentum_divergence', 'volume_ratio', 'volume_trend', 'rsi', 'rsi_divergence',
            'support_resistance', 'market_breadth', 'correlation_stability',
            'bull_bear_score', 'volatility_regime_score', 'trend_regime_score'
        ]
        
        # Remove rows with NaN values
        valid_mask = df_with_indicators[feature_columns].notna().all(axis=1)
        for regime_type in regime_labels:
            valid_mask = valid_mask & regime_labels[regime_type].notna()
        
        X = df_with_indicators.loc[valid_mask, feature_columns]
        y_bull_bear = regime_labels['bull_bear'].loc[valid_mask]
        y_volatility = regime_labels['volatility'].loc[valid_mask]
        y_trend = regime_labels['trend'].loc[valid_mask]
        
        if len(X) < 100:
            raise ValueError("Insufficient training data. Need at least 100 valid samples.")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Encode labels
        y_bull_bear_encoded = self.label_encoders['bull_bear'].fit_transform(y_bull_bear)
        y_volatility_encoded = self.label_encoders['volatility'].fit_transform(y_volatility)
        y_trend_encoded = self.label_encoders['trend'].fit_transform(y_trend)
        
        # Train models
        self.bull_bear_model.fit(X_scaled, y_bull_bear_encoded)
        self.volatility_model.fit(X_scaled, y_volatility_encoded)
        self.trend_model.fit(X_scaled, y_trend_encoded)
        
        # Create composite labels for composite model
        composite_labels = []
        for i in range(len(y_bull_bear_encoded)):
            composite_label = f"{y_bull_bear[i]}_{y_volatility[i]}_{y_trend[i]}"
            composite_labels.append(composite_label)
        
        composite_encoder = LabelEncoder()
        y_composite_encoded = composite_encoder.fit_transform(composite_labels)
        self.composite_model.fit(X_scaled, y_composite_encoded)
        
        # Evaluate performance
        bull_bear_pred = self.bull_bear_model.predict(X_scaled)
        volatility_pred = self.volatility_model.predict(X_scaled)
        trend_pred = self.trend_model.predict(X_scaled)
        composite_pred = self.composite_model.predict(X_scaled)
        
        # Calculate metrics
        bull_bear_accuracy = accuracy_score(y_bull_bear_encoded, bull_bear_pred)
        volatility_accuracy = accuracy_score(y_volatility_encoded, volatility_pred)
        trend_accuracy = accuracy_score(y_trend_encoded, trend_pred)
        composite_accuracy = accuracy_score(y_composite_encoded, composite_pred)
        
        self.performance_metrics = RegimeMetrics(
            accuracy=composite_accuracy,
            precision=0.0,  # Will be calculated properly
            recall=0.0,     # Will be calculated properly
            f1_score=0.0,   # Will be calculated properly
            regime_transition_accuracy=0.0,  # Will be calculated properly
            false_positive_rate=0.0
        )
        
        self.is_trained = True
        
        logger.info(f"Regime prediction models trained successfully. Composite accuracy: {composite_accuracy:.3f}")
        
        return {
            'performance_metrics': self.performance_metrics,
            'bull_bear_accuracy': bull_bear_accuracy,
            'volatility_accuracy': volatility_accuracy,
            'trend_accuracy': trend_accuracy,
            'composite_accuracy': composite_accuracy,
            'training_samples': len(X),
            'regime_distribution': {
                'bull_bear': y_bull_bear.value_counts().to_dict(),
                'volatility': y_volatility.value_counts().to_dict(),
                'trend': y_trend.value_counts().to_dict()
            }
        }
    
    def predict_regime_probabilities(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Predict regime probabilities for current market conditions.
        
        Args:
            data: Recent market data
            
        Returns:
            Dictionary with regime probabilities for each regime type
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        # Calculate indicators for latest data
        df_with_indicators = self._calculate_regime_indicators(data)
        latest_data = df_with_indicators.iloc[-1:]
        
        # Prepare features
        feature_columns = [
            'volatility_20', 'volatility_50', 'volatility_ratio', 'volatility_percentile',
            'trend_strength', 'trend_direction', 'momentum_1m', 'momentum_3m', 'momentum_6m',
            'momentum_divergence', 'volume_ratio', 'volume_trend', 'rsi', 'rsi_divergence',
            'support_resistance', 'market_breadth', 'correlation_stability',
            'bull_bear_score', 'volatility_regime_score', 'trend_regime_score'
        ]
        
        X = latest_data[feature_columns].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from all models
        bull_bear_probs = self.bull_bear_model.predict_proba(X_scaled)[0]
        volatility_probs = self.volatility_model.predict_proba(X_scaled)[0]
        trend_probs = self.trend_model.predict_proba(X_scaled)[0]
        composite_probs = self.composite_model.predict_proba(X_scaled)[0]
        
        # Map probabilities to regime names
        bull_bear_regimes = self.label_encoders['bull_bear'].classes_
        volatility_regimes = self.label_encoders['volatility'].classes_
        trend_regimes = self.label_encoders['trend'].classes_
        
        bull_bear_prob_dict = dict(zip(bull_bear_regimes, bull_bear_probs))
        volatility_prob_dict = dict(zip(volatility_regimes, volatility_probs))
        trend_prob_dict = dict(zip(trend_regimes, trend_probs))
        
        return {
            'bull_bear': bull_bear_prob_dict,
            'volatility': volatility_prob_dict,
            'trend': trend_prob_dict,
            'composite': composite_probs
        }
    
    def get_current_regime(self, data: pd.DataFrame) -> Dict[str, str]:
        """
        Get current market regime classification.
        
        Args:
            data: Recent market data
            
        Returns:
            Dictionary with current regime for each regime type
        """
        regime_probs = self.predict_regime_probabilities(data)
        
        current_regime = {}
        for regime_type in ['bull_bear', 'volatility', 'trend']:
            max_prob = max(regime_probs[regime_type].values())
            for regime, prob in regime_probs[regime_type].items():
                if prob == max_prob:
                    current_regime[regime_type] = regime
                    break
        
        return current_regime
    
    def generate_regime_signal(self, data: pd.DataFrame) -> Optional[RegimeSignal]:
        """
        Generate regime prediction signal if conditions are met.
        
        Args:
            data: Recent market data
            
        Returns:
            RegimeSignal if regime conditions detected, None otherwise
        """
        if not self.is_trained:
            logger.warning("Models not trained, cannot generate regime signal")
            return None
        
        # Get current and predicted regimes
        current_regime = self.get_current_regime(data)
        regime_probs = self.predict_regime_probabilities(data)
        
        # Calculate transition probabilities
        transition_probs = {}
        for regime_type in ['bull_bear', 'volatility', 'trend']:
            current = current_regime[regime_type]
            probs = regime_probs[regime_type]
            max_prob = max(probs.values())
            max_regime = max(probs, key=probs.get)
            
            if max_regime != current:
                transition_probs[regime_type] = max_prob
            else:
                transition_probs[regime_type] = 0.0
        
        # Overall transition probability
        overall_transition_prob = max(transition_probs.values())
        
        if overall_transition_prob < self.min_confidence:
            return None
        
        # Determine which regime is most likely to change
        max_transition_type = max(transition_probs, key=transition_probs.get)
        predicted_regime = max(regime_probs[max_transition_type], key=regime_probs[max_transition_type].get)
        
        # Calculate indicators for explanation
        df_with_indicators = self._calculate_regime_indicators(data)
        latest_indicators = df_with_indicators.iloc[-1]
        
        # Create explanation
        explanation = self._generate_explanation(current_regime, predicted_regime, max_transition_type, latest_indicators)
        
        # Estimate time horizon
        time_horizon = self._estimate_time_horizon(overall_transition_prob)
        
        signal = RegimeSignal(
            timestamp=datetime.now(),
            current_regime=f"{current_regime['bull_bear']}_{current_regime['volatility']}_{current_regime['trend']}",
            predicted_regime=f"{predicted_regime if max_transition_type == 'bull_bear' else current_regime['bull_bear']}_{predicted_regime if max_transition_type == 'volatility' else current_regime['volatility']}_{predicted_regime if max_transition_type == 'trend' else current_regime['trend']}",
            confidence=overall_transition_prob,
            transition_probability=overall_transition_prob,
            time_horizon=time_horizon,
            indicators=latest_indicators.to_dict(),
            explanation=explanation,
            model_used='Ensemble BullBear+Volatility+Trend'
        )
        
        # Store prediction for performance tracking
        self.prediction_history.append(signal)
        
        logger.info(f"Regime signal generated: {max_transition_type} transition, {overall_transition_prob:.3f} confidence")
        
        return signal
    
    def _generate_explanation(self, current_regime: Dict[str, str], predicted_regime: str, 
                            transition_type: str, indicators: pd.Series) -> str:
        """Generate human-readable explanation for regime signal."""
        explanations = []
        
        if transition_type == 'bull_bear':
            explanations.append(f"Bull/Bear regime transition from {current_regime['bull_bear']} to {predicted_regime}")
        elif transition_type == 'volatility':
            explanations.append(f"Volatility regime transition from {current_regime['volatility']} to {predicted_regime}")
        elif transition_type == 'trend':
            explanations.append(f"Trend regime transition from {current_regime['trend']} to {predicted_regime}")
        
        # Add supporting indicators
        if indicators['volatility_ratio'] > 1.5:
            explanations.append("High volatility ratio")
        
        if abs(indicators['trend_strength']) > 0.05:
            explanations.append("Strong trend strength")
        
        if indicators['momentum_divergence'] > 0.02:
            explanations.append("Momentum divergence")
        
        return "Regime transition predicted: " + ", ".join(explanations)
    
    def _estimate_time_horizon(self, probability: float) -> int:
        """Estimate time horizon for regime transition prediction."""
        # Higher probability = shorter time horizon
        if probability > 0.8:
            return 5  # 5 days
        elif probability > 0.6:
            return 10  # 10 days
        elif probability > 0.4:
            return 20  # 20 days
        else:
            return 30  # 30 days
    
    def get_performance_metrics(self) -> Optional[RegimeMetrics]:
        """Get model performance metrics."""
        return self.performance_metrics
    
    def get_prediction_history(self) -> List[RegimeSignal]:
        """Get history of regime predictions."""
        return self.prediction_history
    
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
        
        logger.info("Validating regime prediction models...")
        
        # Calculate indicators
        df_with_indicators = self._calculate_regime_indicators(test_data)
        regime_labels = self._create_regime_labels(df_with_indicators)
        
        # Prepare features
        feature_columns = [
            'volatility_20', 'volatility_50', 'volatility_ratio', 'volatility_percentile',
            'trend_strength', 'trend_direction', 'momentum_1m', 'momentum_3m', 'momentum_6m',
            'momentum_divergence', 'volume_ratio', 'volume_trend', 'rsi', 'rsi_divergence',
            'support_resistance', 'market_breadth', 'correlation_stability',
            'bull_bear_score', 'volatility_regime_score', 'trend_regime_score'
        ]
        
        valid_mask = df_with_indicators[feature_columns].notna().all(axis=1)
        for regime_type in regime_labels:
            valid_mask = valid_mask & regime_labels[regime_type].notna()
        
        X = df_with_indicators.loc[valid_mask, feature_columns]
        y_bull_bear = regime_labels['bull_bear'].loc[valid_mask]
        y_volatility = regime_labels['volatility'].loc[valid_mask]
        y_trend = regime_labels['trend'].loc[valid_mask]
        
        if len(X) == 0:
            return {'error': 'No valid test data'}
        
        X_scaled = self.scaler.transform(X)
        
        # Encode labels
        y_bull_bear_encoded = self.label_encoders['bull_bear'].transform(y_bull_bear)
        y_volatility_encoded = self.label_encoders['volatility'].transform(y_volatility)
        y_trend_encoded = self.label_encoders['trend'].transform(y_trend)
        
        # Get predictions
        bull_bear_pred = self.bull_bear_model.predict(X_scaled)
        volatility_pred = self.volatility_model.predict(X_scaled)
        trend_pred = self.trend_model.predict(X_scaled)
        
        # Calculate metrics
        bull_bear_accuracy = accuracy_score(y_bull_bear_encoded, bull_bear_pred)
        volatility_accuracy = accuracy_score(y_volatility_encoded, volatility_pred)
        trend_accuracy = accuracy_score(y_trend_encoded, trend_pred)
        
        validation_metrics = {
            'bull_bear_accuracy': bull_bear_accuracy,
            'volatility_accuracy': volatility_accuracy,
            'trend_accuracy': trend_accuracy,
            'test_samples': len(X),
            'regime_distribution': {
                'bull_bear': y_bull_bear.value_counts().to_dict(),
                'volatility': y_volatility.value_counts().to_dict(),
                'trend': y_trend.value_counts().to_dict()
            }
        }
        
        logger.info(f"Model validation completed. Bull/Bear: {bull_bear_accuracy:.3f}, Volatility: {volatility_accuracy:.3f}, Trend: {trend_accuracy:.3f}")
        
        return validation_metrics
