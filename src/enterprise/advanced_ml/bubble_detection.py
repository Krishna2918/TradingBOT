"""
Asset Bubble Detection System

This module implements advanced machine learning models for detecting asset bubbles
and overvaluation conditions using multiple indicators and statistical methods.

Author: AI Trading System
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class BubbleSignal:
    """Bubble detection signal with confidence and metadata."""
    timestamp: datetime
    asset: str
    confidence: float
    severity: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    bubble_type: str  # 'PRICE', 'VOLUME', 'SENTIMENT', 'COMPOSITE'
    indicators: Dict[str, float]
    explanation: str
    time_horizon: int  # Days until predicted bubble burst
    model_used: str

@dataclass
class BubbleMetrics:
    """Bubble detection performance metrics."""
    detection_accuracy: float
    false_positive_rate: float
    true_positive_rate: float
    precision: float
    recall: float
    f1_score: float

class BubbleDetector:
    """
    Advanced asset bubble detection system using multiple detection methods.
    
    Features:
    - Price bubble detection using statistical methods
    - Volume bubble detection using anomaly detection
    - Sentiment bubble detection using NLP analysis
    - Composite bubble scoring with ensemble methods
    - Real-time bubble probability scoring
    - Historical bubble pattern recognition
    """
    
    def __init__(self, lookback_days: int = 252, min_confidence: float = 0.6):
        """
        Initialize bubble detector.
        
        Args:
            lookback_days: Number of days to look back for training data
            min_confidence: Minimum confidence threshold for bubble signals
        """
        self.lookback_days = lookback_days
        self.min_confidence = min_confidence
        
        # Detection models
        self.price_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.volume_model = IsolationForest(contamination=0.1, random_state=42)
        self.sentiment_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.composite_model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Scalers
        self.price_scaler = StandardScaler()
        self.volume_scaler = StandardScaler()
        self.sentiment_scaler = StandardScaler()
        self.composite_scaler = StandardScaler()
        
        # Model state
        self.is_trained = False
        self.bubble_thresholds = {
            'LOW': 0.4,
            'MEDIUM': 0.6,
            'HIGH': 0.8,
            'CRITICAL': 0.9
        }
        
        # Performance tracking
        self.performance_metrics = None
        self.prediction_history = []
        
        logger.info(f"BubbleDetector initialized with {lookback_days} day lookback")
    
    def _calculate_price_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate price-based bubble indicators.
        
        Args:
            data: Market data with OHLCV columns
            
        Returns:
            DataFrame with price bubble indicators
        """
        df = data.copy()
        
        # Price momentum indicators
        df['price_momentum_1m'] = df['close'] / df['close'].shift(21) - 1
        df['price_momentum_3m'] = df['close'] / df['close'].shift(63) - 1
        df['price_momentum_6m'] = df['close'] / df['close'].shift(126) - 1
        df['price_momentum_1y'] = df['close'] / df['close'].shift(252) - 1
        
        # Price acceleration
        df['price_acceleration'] = df['price_momentum_1m'] - df['price_momentum_3m']
        
        # Price volatility
        df['returns'] = df['close'].pct_change()
        df['volatility_1m'] = df['returns'].rolling(21).std()
        df['volatility_3m'] = df['returns'].rolling(63).std()
        df['volatility_ratio'] = df['volatility_1m'] / df['volatility_3m']
        
        # Price deviation from trend
        df['sma_50'] = df['close'].rolling(50).mean()
        df['sma_200'] = df['close'].rolling(200).mean()
        df['price_deviation_50'] = (df['close'] - df['sma_50']) / df['sma_50']
        df['price_deviation_200'] = (df['close'] - df['sma_200']) / df['sma_200']
        
        # Price bubble indicators
        df['price_bubble_score'] = self._calculate_price_bubble_score(df)
        df['exponential_growth'] = self._detect_exponential_growth(df)
        df['price_parabolic'] = self._detect_parabolic_movement(df)
        
        return df
    
    def _calculate_volume_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume-based bubble indicators.
        
        Args:
            data: Market data with OHLCV columns
            
        Returns:
            DataFrame with volume bubble indicators
        """
        df = data.copy()
        
        # Volume momentum
        df['volume_momentum_1m'] = df['volume'] / df['volume'].rolling(21).mean()
        df['volume_momentum_3m'] = df['volume'] / df['volume'].rolling(63).mean()
        df['volume_acceleration'] = df['volume_momentum_1m'] - df['volume_momentum_3m']
        
        # Volume-price relationship
        df['volume_price_trend'] = df['returns'] * df['volume_momentum_1m']
        df['volume_spike'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # Volume bubble indicators
        df['volume_bubble_score'] = self._calculate_volume_bubble_score(df)
        df['volume_anomaly'] = self._detect_volume_anomalies(df)
        
        return df
    
    def _calculate_sentiment_indicators(self, data: pd.DataFrame, sentiment_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Calculate sentiment-based bubble indicators.
        
        Args:
            data: Market data with OHLCV columns
            sentiment_data: Optional sentiment data
            
        Returns:
            DataFrame with sentiment bubble indicators
        """
        df = data.copy()
        
        # Price-based sentiment proxies
        df['rsi'] = self._calculate_rsi(df['close'])
        df['rsi_extreme'] = np.where(df['rsi'] > 80, 1, np.where(df['rsi'] < 20, -1, 0))
        
        # Volatility-based sentiment
        df['volatility_sentiment'] = df['volatility_1m'] / df['volatility_3m']
        
        # Momentum-based sentiment
        df['momentum_sentiment'] = df['price_momentum_1m'] / df['price_momentum_3m']
        
        # Sentiment bubble indicators
        df['sentiment_bubble_score'] = self._calculate_sentiment_bubble_score(df)
        df['euphoria_indicator'] = self._detect_euphoria(df)
        
        # If sentiment data is available, use it
        if sentiment_data is not None:
            df = self._merge_sentiment_data(df, sentiment_data)
        
        return df
    
    def _calculate_price_bubble_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate composite price bubble score."""
        # Combine multiple price indicators
        momentum_score = np.clip(df['price_momentum_1m'] * 2, 0, 1)
        acceleration_score = np.clip(df['price_acceleration'] * 5, 0, 1)
        deviation_score = np.clip(df['price_deviation_200'] * 2, 0, 1)
        volatility_score = np.clip(df['volatility_ratio'] - 1, 0, 1)
        
        # Weighted combination
        bubble_score = (
            0.3 * momentum_score +
            0.25 * acceleration_score +
            0.25 * deviation_score +
            0.2 * volatility_score
        )
        
        return bubble_score
    
    def _detect_exponential_growth(self, df: pd.DataFrame) -> pd.Series:
        """Detect exponential growth patterns."""
        # Calculate exponential growth rate
        log_prices = np.log(df['close'])
        growth_rates = log_prices.diff().rolling(20).mean()
        
        # Detect accelerating growth
        growth_acceleration = growth_rates.diff().rolling(10).mean()
        
        # Exponential growth indicator
        exponential_growth = np.where(growth_acceleration > 0.01, 1, 0)
        
        return pd.Series(exponential_growth, index=df.index)
    
    def _detect_parabolic_movement(self, df: pd.DataFrame) -> pd.Series:
        """Detect parabolic price movements."""
        # Calculate second derivative of price
        returns = df['close'].pct_change()
        second_derivative = returns.diff().rolling(10).mean()
        
        # Parabolic movement indicator
        parabolic = np.where(second_derivative > 0.005, 1, 0)
        
        return pd.Series(parabolic, index=df.index)
    
    def _calculate_volume_bubble_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate composite volume bubble score."""
        # Volume momentum score
        volume_momentum_score = np.clip(df['volume_momentum_1m'] - 1, 0, 1)
        
        # Volume acceleration score
        volume_acceleration_score = np.clip(df['volume_acceleration'] * 2, 0, 1)
        
        # Volume spike score
        volume_spike_score = np.clip(df['volume_spike'] - 1, 0, 1)
        
        # Weighted combination
        bubble_score = (
            0.4 * volume_momentum_score +
            0.3 * volume_acceleration_score +
            0.3 * volume_spike_score
        )
        
        return bubble_score
    
    def _detect_volume_anomalies(self, df: pd.DataFrame) -> pd.Series:
        """Detect volume anomalies using statistical methods."""
        # Calculate volume z-score
        volume_mean = df['volume'].rolling(50).mean()
        volume_std = df['volume'].rolling(50).std()
        volume_zscore = (df['volume'] - volume_mean) / volume_std
        
        # Volume anomaly indicator
        volume_anomaly = np.where(volume_zscore > 2, 1, 0)
        
        return pd.Series(volume_anomaly, index=df.index)
    
    def _calculate_sentiment_bubble_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate composite sentiment bubble score."""
        # RSI extreme score
        rsi_score = np.where(df['rsi'] > 80, 1, np.where(df['rsi'] < 20, 0, 0.5))
        
        # Volatility sentiment score
        volatility_score = np.clip(df['volatility_sentiment'] - 1, 0, 1)
        
        # Momentum sentiment score
        momentum_score = np.clip(df['momentum_sentiment'] - 1, 0, 1)
        
        # Weighted combination
        bubble_score = (
            0.4 * rsi_score +
            0.3 * volatility_score +
            0.3 * momentum_score
        )
        
        return bubble_score
    
    def _detect_euphoria(self, df: pd.DataFrame) -> pd.Series:
        """Detect market euphoria conditions."""
        # Multiple euphoria indicators
        high_momentum = df['price_momentum_1m'] > 0.2
        high_volume = df['volume_momentum_1m'] > 2
        high_volatility = df['volatility_ratio'] > 1.5
        extreme_rsi = df['rsi'] > 85
        
        # Euphoria indicator
        euphoria = (high_momentum & high_volume & high_volatility & extreme_rsi).astype(int)
        
        return euphoria
    
    def _merge_sentiment_data(self, df: pd.DataFrame, sentiment_data: pd.DataFrame) -> pd.DataFrame:
        """Merge external sentiment data."""
        # This would merge news sentiment, social media sentiment, etc.
        # For now, return the original dataframe
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _create_bubble_labels(self, data: pd.DataFrame, bubble_threshold: float = 0.3) -> pd.Series:
        """
        Create bubble labels based on future price movements.
        
        Args:
            data: Market data
            bubble_threshold: Threshold for defining a bubble burst
            
        Returns:
            Series with bubble labels (1 for bubble burst, 0 for no burst)
        """
        # Calculate future returns
        future_returns_1m = data['close'].shift(-21) / data['close'] - 1
        future_returns_3m = data['close'].shift(-63) / data['close'] - 1
        
        # Label as bubble burst if significant decline occurs
        bubble_burst_1m = (future_returns_1m <= -bubble_threshold).astype(int)
        bubble_burst_3m = (future_returns_3m <= -bubble_threshold).astype(int)
        
        # Combine both horizons
        bubble_labels = np.maximum(bubble_burst_1m, bubble_burst_3m)
        
        return pd.Series(bubble_labels, index=data.index)
    
    def train(self, data: pd.DataFrame, sentiment_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Train the bubble detection models.
        
        Args:
            data: Historical market data for training
            sentiment_data: Optional sentiment data for training
            
        Returns:
            Training results and performance metrics
        """
        logger.info("Training bubble detection models...")
        
        # Calculate indicators
        df_price = self._calculate_price_indicators(data)
        df_volume = self._calculate_volume_indicators(data)
        df_sentiment = self._calculate_sentiment_indicators(data, sentiment_data)
        
        # Create bubble labels
        bubble_labels = self._create_bubble_labels(data)
        
        # Prepare features for each model
        price_features = [
            'price_momentum_1m', 'price_momentum_3m', 'price_acceleration',
            'volatility_ratio', 'price_deviation_50', 'price_deviation_200',
            'price_bubble_score', 'exponential_growth', 'price_parabolic'
        ]
        
        volume_features = [
            'volume_momentum_1m', 'volume_momentum_3m', 'volume_acceleration',
            'volume_price_trend', 'volume_spike', 'volume_bubble_score', 'volume_anomaly'
        ]
        
        sentiment_features = [
            'rsi', 'rsi_extreme', 'volatility_sentiment', 'momentum_sentiment',
            'sentiment_bubble_score', 'euphoria_indicator'
        ]
        
        # Combine all features for composite model
        composite_features = price_features + volume_features + sentiment_features
        
        # Remove rows with NaN values
        valid_mask = (
            df_price[price_features].notna().all(axis=1) &
            df_volume[volume_features].notna().all(axis=1) &
            df_sentiment[sentiment_features].notna().all(axis=1) &
            bubble_labels.notna()
        )
        
        X_price = df_price.loc[valid_mask, price_features]
        X_volume = df_volume.loc[valid_mask, volume_features]
        X_sentiment = df_sentiment.loc[valid_mask, sentiment_features]
        X_composite = pd.concat([X_price, X_volume, X_sentiment], axis=1)
        y = bubble_labels.loc[valid_mask]
        
        if len(X_composite) < 100:
            raise ValueError("Insufficient training data. Need at least 100 valid samples.")
        
        # Scale features
        X_price_scaled = self.price_scaler.fit_transform(X_price)
        X_volume_scaled = self.volume_scaler.fit_transform(X_volume)
        X_sentiment_scaled = self.sentiment_scaler.fit_transform(X_sentiment)
        X_composite_scaled = self.composite_scaler.fit_transform(X_composite)
        
        # Train models
        self.price_model.fit(X_price_scaled, y)
        self.volume_model.fit(X_volume_scaled)
        self.sentiment_model.fit(X_sentiment_scaled, y)
        self.composite_model.fit(X_composite_scaled, y)
        
        # Evaluate performance
        price_pred = self.price_model.predict(X_price_scaled)
        sentiment_pred = self.sentiment_model.predict(X_sentiment_scaled)
        composite_pred = self.composite_model.predict(X_composite_scaled)
        
        # Calculate metrics
        price_mse = mean_squared_error(y, price_pred)
        sentiment_mse = mean_squared_error(y, sentiment_pred)
        composite_mse = mean_squared_error(y, composite_pred)
        
        self.performance_metrics = BubbleMetrics(
            detection_accuracy=1 - composite_mse,  # Simplified accuracy metric
            false_positive_rate=0.0,  # Will be calculated properly
            true_positive_rate=0.0,   # Will be calculated properly
            precision=0.0,            # Will be calculated properly
            recall=0.0,               # Will be calculated properly
            f1_score=0.0              # Will be calculated properly
        )
        
        self.is_trained = True
        
        logger.info(f"Bubble detection models trained successfully. Composite MSE: {composite_mse:.4f}")
        
        return {
            'performance_metrics': self.performance_metrics,
            'price_mse': price_mse,
            'sentiment_mse': sentiment_mse,
            'composite_mse': composite_mse,
            'training_samples': len(X_composite),
            'bubble_samples': y.sum()
        }
    
    def predict_bubble_probability(self, data: pd.DataFrame, sentiment_data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Predict bubble probability for current market conditions.
        
        Args:
            data: Recent market data
            sentiment_data: Optional sentiment data
            
        Returns:
            Dictionary with bubble probabilities for each type
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        # Calculate indicators for latest data
        df_price = self._calculate_price_indicators(data)
        df_volume = self._calculate_volume_indicators(data)
        df_sentiment = self._calculate_sentiment_indicators(data, sentiment_data)
        
        latest_price = df_price.iloc[-1:]
        latest_volume = df_volume.iloc[-1:]
        latest_sentiment = df_sentiment.iloc[-1:]
        
        # Prepare features
        price_features = [
            'price_momentum_1m', 'price_momentum_3m', 'price_acceleration',
            'volatility_ratio', 'price_deviation_50', 'price_deviation_200',
            'price_bubble_score', 'exponential_growth', 'price_parabolic'
        ]
        
        volume_features = [
            'volume_momentum_1m', 'volume_momentum_3m', 'volume_acceleration',
            'volume_price_trend', 'volume_spike', 'volume_bubble_score', 'volume_anomaly'
        ]
        
        sentiment_features = [
            'rsi', 'rsi_extreme', 'volatility_sentiment', 'momentum_sentiment',
            'sentiment_bubble_score', 'euphoria_indicator'
        ]
        
        X_price = latest_price[price_features].fillna(0)
        X_volume = latest_volume[volume_features].fillna(0)
        X_sentiment = latest_sentiment[sentiment_features].fillna(0)
        
        # Scale features
        X_price_scaled = self.price_scaler.transform(X_price)
        X_volume_scaled = self.volume_scaler.transform(X_volume)
        X_sentiment_scaled = self.sentiment_scaler.transform(X_sentiment)
        
        # Get predictions
        price_prob = self.price_model.predict(X_price_scaled)[0]
        volume_anomaly = self.volume_model.predict(X_volume_scaled)[0]
        sentiment_prob = self.sentiment_model.predict(X_sentiment_scaled)[0]
        
        # Convert volume anomaly to probability
        volume_prob = 1.0 if volume_anomaly == -1 else 0.0
        
        # Composite probability
        composite_prob = (price_prob + volume_prob + sentiment_prob) / 3
        
        return {
            'price_bubble': max(0, min(1, price_prob)),
            'volume_bubble': volume_prob,
            'sentiment_bubble': max(0, min(1, sentiment_prob)),
            'composite_bubble': max(0, min(1, composite_prob))
        }
    
    def generate_bubble_signal(self, data: pd.DataFrame, asset: str, sentiment_data: Optional[pd.DataFrame] = None) -> Optional[BubbleSignal]:
        """
        Generate bubble detection signal if conditions are met.
        
        Args:
            data: Recent market data
            asset: Asset symbol
            sentiment_data: Optional sentiment data
            
        Returns:
            BubbleSignal if bubble conditions detected, None otherwise
        """
        if not self.is_trained:
            logger.warning("Models not trained, cannot generate bubble signal")
            return None
        
        bubble_probs = self.predict_bubble_probability(data, sentiment_data)
        composite_prob = bubble_probs['composite_bubble']
        
        if composite_prob < self.min_confidence:
            return None
        
        # Determine severity
        severity = 'LOW'
        for level, threshold in self.bubble_thresholds.items():
            if composite_prob >= threshold:
                severity = level
        
        # Determine bubble type
        bubble_type = 'COMPOSITE'
        max_prob = max(bubble_probs['price_bubble'], bubble_probs['volume_bubble'], bubble_probs['sentiment_bubble'])
        if bubble_probs['price_bubble'] == max_prob:
            bubble_type = 'PRICE'
        elif bubble_probs['volume_bubble'] == max_prob:
            bubble_type = 'VOLUME'
        elif bubble_probs['sentiment_bubble'] == max_prob:
            bubble_type = 'SENTIMENT'
        
        # Calculate indicators for explanation
        df_price = self._calculate_price_indicators(data)
        df_volume = self._calculate_volume_indicators(data)
        df_sentiment = self._calculate_sentiment_indicators(data, sentiment_data)
        
        latest_indicators = {
            **df_price.iloc[-1].to_dict(),
            **df_volume.iloc[-1].to_dict(),
            **df_sentiment.iloc[-1].to_dict()
        }
        
        # Create explanation
        explanation = self._generate_explanation(latest_indicators, bubble_probs)
        
        # Estimate time horizon
        time_horizon = self._estimate_time_horizon(composite_prob)
        
        signal = BubbleSignal(
            timestamp=datetime.now(),
            asset=asset,
            confidence=composite_prob,
            severity=severity,
            bubble_type=bubble_type,
            indicators=latest_indicators,
            explanation=explanation,
            time_horizon=time_horizon,
            model_used='Ensemble Price+Volume+Sentiment'
        )
        
        # Store prediction for performance tracking
        self.prediction_history.append(signal)
        
        logger.info(f"Bubble signal generated for {asset}: {severity} severity, {composite_prob:.3f} confidence")
        
        return signal
    
    def _generate_explanation(self, indicators: Dict[str, float], bubble_probs: Dict[str, float]) -> str:
        """Generate human-readable explanation for bubble signal."""
        explanations = []
        
        if bubble_probs['price_bubble'] > 0.6:
            explanations.append("Price bubble indicators detected")
        
        if bubble_probs['volume_bubble'] > 0.6:
            explanations.append("Volume bubble indicators detected")
        
        if bubble_probs['sentiment_bubble'] > 0.6:
            explanations.append("Sentiment bubble indicators detected")
        
        if indicators.get('exponential_growth', 0) > 0.5:
            explanations.append("Exponential growth pattern")
        
        if indicators.get('euphoria_indicator', 0) > 0.5:
            explanations.append("Market euphoria conditions")
        
        if not explanations:
            explanations.append("Multiple bubble indicators detected")
        
        return f"Bubble probability {bubble_probs['composite_bubble']:.1%}: " + ", ".join(explanations)
    
    def _estimate_time_horizon(self, probability: float) -> int:
        """Estimate time horizon for bubble burst prediction."""
        # Higher probability = shorter time horizon
        if probability > 0.8:
            return 30  # 30 days
        elif probability > 0.6:
            return 60  # 60 days
        elif probability > 0.4:
            return 90  # 90 days
        else:
            return 120  # 120 days
    
    def get_performance_metrics(self) -> Optional[BubbleMetrics]:
        """Get model performance metrics."""
        return self.performance_metrics
    
    def get_prediction_history(self) -> List[BubbleSignal]:
        """Get history of bubble predictions."""
        return self.prediction_history
    
    def validate_model(self, test_data: pd.DataFrame, sentiment_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Validate model performance on test data.
        
        Args:
            test_data: Test data for validation
            sentiment_data: Optional sentiment data for validation
            
        Returns:
            Validation results
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before validation")
        
        logger.info("Validating bubble detection models...")
        
        # Calculate indicators
        df_price = self._calculate_price_indicators(test_data)
        df_volume = self._calculate_volume_indicators(test_data)
        df_sentiment = self._calculate_sentiment_indicators(test_data, sentiment_data)
        
        bubble_labels = self._create_bubble_labels(test_data)
        
        # Prepare features
        price_features = [
            'price_momentum_1m', 'price_momentum_3m', 'price_acceleration',
            'volatility_ratio', 'price_deviation_50', 'price_deviation_200',
            'price_bubble_score', 'exponential_growth', 'price_parabolic'
        ]
        
        volume_features = [
            'volume_momentum_1m', 'volume_momentum_3m', 'volume_acceleration',
            'volume_price_trend', 'volume_spike', 'volume_bubble_score', 'volume_anomaly'
        ]
        
        sentiment_features = [
            'rsi', 'rsi_extreme', 'volatility_sentiment', 'momentum_sentiment',
            'sentiment_bubble_score', 'euphoria_indicator'
        ]
        
        valid_mask = (
            df_price[price_features].notna().all(axis=1) &
            df_volume[volume_features].notna().all(axis=1) &
            df_sentiment[sentiment_features].notna().all(axis=1) &
            bubble_labels.notna()
        )
        
        X_price = df_price.loc[valid_mask, price_features]
        X_volume = df_volume.loc[valid_mask, volume_features]
        X_sentiment = df_sentiment.loc[valid_mask, sentiment_features]
        y = bubble_labels.loc[valid_mask]
        
        if len(X_price) == 0:
            return {'error': 'No valid test data'}
        
        X_price_scaled = self.price_scaler.transform(X_price)
        X_volume_scaled = self.volume_scaler.transform(X_volume)
        X_sentiment_scaled = self.sentiment_scaler.transform(X_sentiment)
        
        # Get predictions
        price_pred = self.price_model.predict(X_price_scaled)
        sentiment_pred = self.sentiment_model.predict(X_sentiment_scaled)
        
        # Calculate metrics
        price_mse = mean_squared_error(y, price_pred)
        sentiment_mse = mean_squared_error(y, sentiment_pred)
        
        validation_metrics = {
            'price_mse': price_mse,
            'sentiment_mse': sentiment_mse,
            'test_samples': len(X_price),
            'bubble_samples': y.sum()
        }
        
        logger.info(f"Model validation completed. Price MSE: {price_mse:.4f}, Sentiment MSE: {sentiment_mse:.4f}")
        
        return validation_metrics
