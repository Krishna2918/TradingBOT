"""
Anomaly Detection Module

Detects unusual market behavior using:
- Isolation Forest algorithm
- Statistical anomaly detection
- Volume anomalies
- Price movement anomalies
- Pattern-based anomaly detection
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class AnomalyType:
    """Anomaly types"""
    VOLUME = "volume"
    PRICE = "price"
    SPREAD = "spread"
    VOLATILITY = "volatility"
    PATTERN = "pattern"
    COMPOSITE = "composite"

class AnomalyDetector:
    """
    Anomaly detection using Isolation Forest and statistical methods
    
    Features:
    - Multi-dimensional anomaly detection
    - Volume spike detection
    - Unusual price movements
    - Spread anomalies
    - Pattern-based detection
    """
    
    def __init__(
        self,
        contamination: float = 0.05,  # Expected proportion of outliers
        n_estimators: int = 100,
        random_state: int = 42
    ):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        
        # Isolation Forest model
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        
        # Scaler for feature normalization
        self.scaler = StandardScaler()
        
        # Anomaly history
        self.anomaly_history: Dict[str, List[Dict]] = {}
        
        # Model trained flag
        self.is_trained = False
        
        logger.info(" Anomaly Detector initialized")
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for anomaly detection
        
        Args:
            data: OHLCV data
        
        Returns:
            Feature DataFrame
        """
        
        features = pd.DataFrame(index=data.index)
        
        # Price-based features
        features['returns'] = data['close'].pct_change()
        features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        features['price_range'] = (data['high'] - data['low']) / data['close']
        features['body_size'] = abs(data['close'] - data['open']) / data['close']
        
        # Volume features
        features['volume'] = data['volume']
        features['volume_change'] = data['volume'].pct_change()
        features['volume_ma_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        
        # Volatility features
        features['high_low_ratio'] = data['high'] / data['low']
        features['close_open_ratio'] = data['close'] / data['open']
        
        # Momentum features
        features['rsi'] = self._calculate_rsi(data['close'], 14)
        features['momentum_5'] = data['close'].pct_change(5)
        features['momentum_10'] = data['close'].pct_change(10)
        
        # Spread features (if bid/ask available)
        if 'bid' in data.columns and 'ask' in data.columns:
            features['spread'] = (data['ask'] - data['bid']) / data['close']
            features['spread_ma_ratio'] = features['spread'] / features['spread'].rolling(20).mean()
        
        # Fill NaN values
        features = features.fillna(method='ffill').fillna(0)
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def train(self, data: pd.DataFrame):
        """
        Train anomaly detection model
        
        Args:
            data: Historical OHLCV data
        """
        
        logger.info(" Training anomaly detection model...")
        
        # Prepare features
        features = self.prepare_features(data)
        
        # Remove any infinite or NaN values
        features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Train Isolation Forest
        self.model.fit(features_scaled)
        
        self.is_trained = True
        
        logger.info(" Anomaly detection model trained")
    
    def detect_anomalies(
        self,
        symbol: str,
        data: pd.DataFrame,
        return_scores: bool = False
    ) -> pd.DataFrame:
        """
        Detect anomalies in data
        
        Args:
            symbol: Symbol being analyzed
            data: OHLCV data
            return_scores: Whether to return anomaly scores
        
        Returns:
            DataFrame with anomaly flags and scores
        """
        
        if not self.is_trained:
            logger.warning(" Model not trained, training now...")
            self.train(data)
        
        # Prepare features
        features = self.prepare_features(data)
        
        # Remove any infinite or NaN values
        features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict anomalies (-1 for anomaly, 1 for normal)
        predictions = self.model.predict(features_scaled)
        
        # Get anomaly scores (lower is more anomalous)
        scores = self.model.score_samples(features_scaled)
        
        # Create results DataFrame
        results = pd.DataFrame(index=data.index)
        results['is_anomaly'] = predictions == -1
        
        if return_scores:
            results['anomaly_score'] = scores
        
        # Count anomalies
        n_anomalies = results['is_anomaly'].sum()
        
        if n_anomalies > 0:
            logger.info(f" Detected {n_anomalies} anomalies in {symbol} ({len(data)} samples)")
        
        return results
    
    def detect_volume_anomaly(
        self,
        symbol: str,
        current_volume: float,
        historical_volume: pd.Series,
        threshold: float = 3.0
    ) -> Tuple[bool, float]:
        """
        Detect volume anomalies using z-score
        
        Args:
            symbol: Symbol being analyzed
            current_volume: Current volume
            historical_volume: Historical volume series
            threshold: Z-score threshold
        
        Returns:
            (is_anomaly, z_score)
        """
        
        if len(historical_volume) < 20:
            return False, 0.0
        
        # Calculate z-score
        mean_vol = historical_volume.mean()
        std_vol = historical_volume.std()
        
        if std_vol == 0:
            return False, 0.0
        
        z_score = (current_volume - mean_vol) / std_vol
        
        is_anomaly = abs(z_score) > threshold
        
        if is_anomaly:
            logger.warning(
                f" Volume anomaly detected for {symbol}: "
                f"{current_volume:,.0f} (z-score: {z_score:.2f})"
            )
            
            # Record anomaly
            self._record_anomaly(
                symbol=symbol,
                anomaly_type=AnomalyType.VOLUME,
                severity=abs(z_score) / threshold,
                details={
                    'current_volume': current_volume,
                    'mean_volume': mean_vol,
                    'z_score': z_score
                }
            )
        
        return is_anomaly, z_score
    
    def detect_price_anomaly(
        self,
        symbol: str,
        current_price: float,
        historical_prices: pd.Series,
        threshold: float = 3.0
    ) -> Tuple[bool, float]:
        """
        Detect price movement anomalies
        
        Args:
            symbol: Symbol being analyzed
            current_price: Current price
            historical_prices: Historical price series
            threshold: Z-score threshold
        
        Returns:
            (is_anomaly, z_score)
        """
        
        if len(historical_prices) < 20:
            return False, 0.0
        
        # Calculate returns
        returns = historical_prices.pct_change().dropna()
        current_return = (current_price - historical_prices.iloc[-1]) / historical_prices.iloc[-1]
        
        # Calculate z-score
        mean_return = returns.mean()
        std_return = returns.std()
        
        if std_return == 0:
            return False, 0.0
        
        z_score = (current_return - mean_return) / std_return
        
        is_anomaly = abs(z_score) > threshold
        
        if is_anomaly:
            logger.warning(
                f" Price anomaly detected for {symbol}: "
                f"{current_return:.2%} return (z-score: {z_score:.2f})"
            )
            
            # Record anomaly
            self._record_anomaly(
                symbol=symbol,
                anomaly_type=AnomalyType.PRICE,
                severity=abs(z_score) / threshold,
                details={
                    'current_price': current_price,
                    'return': current_return,
                    'z_score': z_score
                }
            )
        
        return is_anomaly, z_score
    
    def analyze_anomalies(
        self,
        symbol: str,
        data: pd.DataFrame
    ) -> Dict:
        """
        Comprehensive anomaly analysis
        
        Args:
            symbol: Symbol to analyze
            data: OHLCV data
        
        Returns:
            Anomaly analysis results
        """
        
        if len(data) < 30:
            logger.warning(f" Insufficient data for {symbol} anomaly analysis")
            return {}
        
        # Detect anomalies using Isolation Forest
        anomaly_results = self.detect_anomalies(symbol, data, return_scores=True)
        
        # Get latest anomaly status
        is_anomaly = anomaly_results['is_anomaly'].iloc[-1]
        anomaly_score = anomaly_results['anomaly_score'].iloc[-1]
        
        # Detect volume anomaly
        current_volume = data['volume'].iloc[-1]
        historical_volume = data['volume'].iloc[:-1]
        is_volume_anomaly, volume_z_score = self.detect_volume_anomaly(
            symbol, current_volume, historical_volume
        )
        
        # Detect price anomaly
        current_price = data['close'].iloc[-1]
        historical_prices = data['close'].iloc[:-1]
        is_price_anomaly, price_z_score = self.detect_price_anomaly(
            symbol, current_price, historical_prices
        )
        
        # Count recent anomalies
        recent_anomalies = anomaly_results['is_anomaly'].tail(20).sum()
        
        analysis = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'is_anomaly': bool(is_anomaly),
            'anomaly_score': float(anomaly_score),
            'is_volume_anomaly': is_volume_anomaly,
            'volume_z_score': volume_z_score,
            'is_price_anomaly': is_price_anomaly,
            'price_z_score': price_z_score,
            'recent_anomaly_count': int(recent_anomalies),
            'anomaly_types': []
        }
        
        # Determine anomaly types
        if is_anomaly:
            analysis['anomaly_types'].append(AnomalyType.COMPOSITE)
        if is_volume_anomaly:
            analysis['anomaly_types'].append(AnomalyType.VOLUME)
        if is_price_anomaly:
            analysis['anomaly_types'].append(AnomalyType.PRICE)
        
        logger.info(
            f" Anomaly analysis for {symbol}: "
            f"Anomaly={is_anomaly}, Score={anomaly_score:.3f}, "
            f"Volume Z={volume_z_score:.2f}, Price Z={price_z_score:.2f}"
        )
        
        return analysis
    
    def _record_anomaly(
        self,
        symbol: str,
        anomaly_type: str,
        severity: float,
        details: Dict
    ):
        """Record anomaly in history"""
        
        if symbol not in self.anomaly_history:
            self.anomaly_history[symbol] = []
        
        anomaly_record = {
            'timestamp': datetime.now().isoformat(),
            'type': anomaly_type,
            'severity': severity,
            'details': details
        }
        
        self.anomaly_history[symbol].append(anomaly_record)
        
        # Keep only recent history (last 100 entries)
        self.anomaly_history[symbol] = self.anomaly_history[symbol][-100:]
    
    def get_anomaly_summary(self) -> Dict:
        """Get summary of all detected anomalies"""
        
        summary = {}
        
        for symbol, history in self.anomaly_history.items():
            if not history:
                continue
            
            recent_anomalies = [a for a in history if a['timestamp'] >= (datetime.now() - pd.Timedelta(hours=24)).isoformat()]
            
            summary[symbol] = {
                'total_anomalies': len(history),
                'recent_anomalies_24h': len(recent_anomalies),
                'latest_anomaly': history[-1] if history else None
            }
        
        return summary

# Global detector instance
_anomaly_detector_instance = None

def get_anomaly_detector() -> AnomalyDetector:
    """Get global anomaly detector instance"""
    global _anomaly_detector_instance
    if _anomaly_detector_instance is None:
        _anomaly_detector_instance = AnomalyDetector()
    return _anomaly_detector_instance

