"""
Regime Detection Module
Detects market regimes (bull/bear/sideways) using volatility, correlation, and dispersion metrics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Market regime types"""
    LOW_VOL = "low_volatility"
    MID_VOL = "mid_volatility"
    HIGH_VOL = "high_volatility"
    BULL = "bull_market"
    BEAR = "bear_market"
    SIDEWAYS = "sideways"
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"

@dataclass
class RegimeMetrics:
    """Regime detection metrics"""
    volatility_10d: float
    volatility_30d: float
    correlation_breakdown: float  # FIXED: was correlation_30_90d
    dispersion: float
    momentum: float
    volume_profile: float
    regime: MarketRegime
    confidence: float
    timestamp: datetime

class RegimeDetector:
    """Detects market regimes using multiple indicators"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.lookback_days = config.get('lookback_days', 90)
        self.volatility_window = config.get('volatility_window', 10)
        self.correlation_window_short = config.get('correlation_window_short', 30)
        self.correlation_window_long = config.get('correlation_window_long', 90)
        
        # Regime thresholds
        self.vol_thresholds = {
            'low': 0.15,    # 15% annualized
            'high': 0.35    # 35% annualized
        }
        
        self.correlation_threshold = 0.7
        self.dispersion_threshold = 0.3
        
        # Clustering model
        self.kmeans_model = None
        self.scaler = StandardScaler()
        self.regime_history = []
        
        logger.info("Regime Detector initialized")
    
    def calculate_volatility_metrics(self, prices: pd.Series) -> Dict[str, float]:
        """Calculate volatility metrics"""
        try:
            if len(prices) < self.volatility_window:
                return {'vol_10d': 0.0, 'vol_30d': 0.0}
            
            # Calculate returns
            returns = prices.pct_change().dropna()
            
            # Short-term volatility (10 days)
            vol_10d = returns.tail(self.volatility_window).std() * np.sqrt(252)
            
            # Long-term volatility (30 days)
            vol_30d = returns.tail(30).std() * np.sqrt(252) if len(returns) >= 30 else vol_10d
            
            return {
                'vol_10d': vol_10d,
                'vol_30d': vol_30d
            }
            
        except Exception as e:
            logger.error(f"Error calculating volatility metrics: {e}")
            return {'vol_10d': 0.0, 'vol_30d': 0.0}
    
    def calculate_correlation_breakdown(self, prices_df: pd.DataFrame) -> float:
        """Calculate correlation breakdown between short and long periods"""
        try:
            if len(prices_df) < self.correlation_window_long:
                return 0.0
            
            # Calculate returns
            returns = prices_df.pct_change().dropna()
            
            # Short-term correlation (30 days)
            if len(returns) >= self.correlation_window_short:
                corr_short = returns.tail(self.correlation_window_short).corr().values
                corr_short = corr_short[np.triu_indices_from(corr_short, k=1)].mean()
            else:
                corr_short = 0.0
            
            # Long-term correlation (90 days)
            if len(returns) >= self.correlation_window_long:
                corr_long = returns.tail(self.correlation_window_long).corr().values
                corr_long = corr_long[np.triu_indices_from(corr_long, k=1)].mean()
            else:
                corr_long = corr_short
            
            # Correlation breakdown
            correlation_breakdown = abs(corr_short - corr_long)
            
            return correlation_breakdown
            
        except Exception as e:
            logger.error(f"Error calculating correlation breakdown: {e}")
            return 0.0
    
    def calculate_dispersion(self, prices_df: pd.DataFrame) -> float:
        """Calculate cross-sectional dispersion"""
        try:
            if len(prices_df) < 30:
                return 0.0
            
            # Calculate returns
            returns = prices_df.pct_change().dropna()
            
            # Recent returns (last 30 days)
            recent_returns = returns.tail(30)
            
            # Calculate cross-sectional standard deviation
            dispersion = recent_returns.std(axis=1).mean()
            
            return dispersion
            
        except Exception as e:
            logger.error(f"Error calculating dispersion: {e}")
            return 0.0
    
    def calculate_momentum(self, prices: pd.Series) -> float:
        """Calculate price momentum"""
        try:
            if len(prices) < 20:
                return 0.0
            
            # 20-day momentum
            momentum = (prices.iloc[-1] / prices.iloc[-20] - 1) * 100
            
            return momentum
            
        except Exception as e:
            logger.error(f"Error calculating momentum: {e}")
            return 0.0
    
    def calculate_volume_profile(self, volume: pd.Series) -> float:
        """Calculate volume profile indicator"""
        try:
            if len(volume) < 20:
                return 0.0
            
            # Recent volume vs historical average
            recent_volume = volume.tail(5).mean()
            historical_volume = volume.tail(20).mean()
            
            if historical_volume == 0:
                return 0.0
            
            volume_profile = (recent_volume / historical_volume - 1) * 100
            
            return volume_profile
            
        except Exception as e:
            logger.error(f"Error calculating volume profile: {e}")
            return 0.0
    
    def detect_regime_clustering(self, metrics: Dict[str, float]) -> Tuple[MarketRegime, float]:
        """Detect regime using K-means clustering"""
        try:
            # Prepare features for clustering
            features = np.array([
                metrics['vol_10d'],
                metrics['correlation_breakdown'],
                metrics['dispersion'],
                metrics['momentum'],
                metrics['volume_profile']
            ]).reshape(1, -1)
            
            # If we don't have enough history, use rule-based detection
            if len(self.regime_history) < 50:
                return self._detect_regime_rules(metrics)
            
            # Train K-means if not already trained
            if self.kmeans_model is None:
                self._train_clustering_model()
            
            # Predict regime
            if self.kmeans_model is not None:
                cluster = self.kmeans_model.predict(features)[0]
                confidence = self._calculate_cluster_confidence(features, cluster)
                
                # Map cluster to regime
                regime = self._map_cluster_to_regime(cluster, metrics)
                
                return regime, confidence
            else:
                return self._detect_regime_rules(metrics)
                
        except Exception as e:
            logger.error(f"Error in regime clustering: {e}")
            return self._detect_regime_rules(metrics)
    
    def _train_clustering_model(self):
        """Train K-means clustering model on historical data"""
        try:
            if len(self.regime_history) < 50:
                return
            
            # Prepare training data
            features = []
            for metrics in self.regime_history[-200:]:  # Use last 200 data points
                features.append([
                    metrics.volatility_10d,
                    metrics.correlation_breakdown,
                    metrics.dispersion,
                    metrics.momentum,
                    metrics.volume_profile
                ])
            
            features = np.array(features)
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # Train K-means
            self.kmeans_model = KMeans(n_clusters=6, random_state=42, n_init=10)
            self.kmeans_model.fit(features_scaled)
            
            logger.info("K-means clustering model trained")
            
        except Exception as e:
            logger.error(f"Error training clustering model: {e}")
            self.kmeans_model = None
    
    def _calculate_cluster_confidence(self, features: np.ndarray, cluster: int) -> float:
        """Calculate confidence in cluster assignment"""
        try:
            if self.kmeans_model is None:
                return 0.5
            
            # Calculate distance to cluster center
            center = self.kmeans_model.cluster_centers_[cluster]
            distance = np.linalg.norm(features - center)
            
            # Convert distance to confidence (closer = higher confidence)
            max_distance = np.max([np.linalg.norm(center - other_center) 
                                 for other_center in self.kmeans_model.cluster_centers_])
            
            confidence = max(0.1, 1.0 - (distance / max_distance))
            return confidence
            
        except Exception as e:
            logger.error(f"Error calculating cluster confidence: {e}")
            return 0.5
    
    def _map_cluster_to_regime(self, cluster: int, metrics: Dict[str, float]) -> MarketRegime:
        """Map cluster to market regime"""
        try:
            # Simple mapping based on cluster characteristics
            # This would be more sophisticated in practice
            
            vol_10d = metrics['vol_10d']
            momentum = metrics['momentum']
            dispersion = metrics['dispersion']
            
            if vol_10d < self.vol_thresholds['low']:
                return MarketRegime.LOW_VOL
            elif vol_10d > self.vol_thresholds['high']:
                return MarketRegime.HIGH_VOL
            elif momentum > 5:
                return MarketRegime.BULL
            elif momentum < -5:
                return MarketRegime.BEAR
            elif dispersion > self.dispersion_threshold:
                return MarketRegime.TRENDING
            else:
                return MarketRegime.SIDEWAYS
                
        except Exception as e:
            logger.error(f"Error mapping cluster to regime: {e}")
            return MarketRegime.SIDEWAYS
    
    def _detect_regime_rules(self, metrics: Dict[str, float]) -> Tuple[MarketRegime, float]:
        """Detect regime using rule-based approach"""
        try:
            vol_10d = metrics['vol_10d']
            momentum = metrics['momentum']
            dispersion = metrics['dispersion']
            correlation_breakdown = metrics['correlation_breakdown']
            
            # Rule-based regime detection
            if vol_10d < self.vol_thresholds['low']:
                regime = MarketRegime.LOW_VOL
                confidence = 0.8
            elif vol_10d > self.vol_thresholds['high']:
                regime = MarketRegime.HIGH_VOL
                confidence = 0.8
            elif momentum > 5 and dispersion < self.dispersion_threshold:
                regime = MarketRegime.BULL
                confidence = 0.7
            elif momentum < -5 and dispersion < self.dispersion_threshold:
                regime = MarketRegime.BEAR
                confidence = 0.7
            elif correlation_breakdown > 0.3:
                regime = MarketRegime.TRENDING
                confidence = 0.6
            else:
                regime = MarketRegime.SIDEWAYS
                confidence = 0.5
            
            return regime, confidence
            
        except Exception as e:
            logger.error(f"Error in rule-based regime detection: {e}")
            return MarketRegime.SIDEWAYS, 0.5
    
    def detect_regime(self, prices_df: pd.DataFrame, volume_df: pd.DataFrame = None) -> RegimeMetrics:
        """Detect current market regime"""
        try:
            # Calculate all metrics
            vol_metrics = self.calculate_volatility_metrics(prices_df.iloc[:, 0])
            correlation_breakdown = self.calculate_correlation_breakdown(prices_df)
            dispersion = self.calculate_dispersion(prices_df)
            momentum = self.calculate_momentum(prices_df.iloc[:, 0])
            
            volume_profile = 0.0
            if volume_df is not None and not volume_df.empty:
                volume_profile = self.calculate_volume_profile(volume_df.iloc[:, 0])
            
            # Combine metrics
            metrics = {
                'vol_10d': vol_metrics['vol_10d'],
                'vol_30d': vol_metrics['vol_30d'],
                'correlation_breakdown': correlation_breakdown,
                'dispersion': dispersion,
                'momentum': momentum,
                'volume_profile': volume_profile
            }
            
            # Detect regime
            regime, confidence = self.detect_regime_clustering(metrics)
            
            # Create regime metrics
            regime_metrics = RegimeMetrics(
                volatility_10d=vol_metrics['vol_10d'],
                volatility_30d=vol_metrics['vol_30d'],
                correlation_breakdown=correlation_breakdown,
                dispersion=dispersion,
                momentum=momentum,
                volume_profile=volume_profile,
                regime=regime,
                confidence=confidence,
                timestamp=datetime.now()
            )
            
            # Store in history
            self.regime_history.append(regime_metrics)
            
            # Keep only recent history
            if len(self.regime_history) > 500:
                self.regime_history = self.regime_history[-500:]
            
            return regime_metrics
            
        except Exception as e:
            logger.error(f"Error detecting regime: {e}")
            return RegimeMetrics(
                volatility_10d=0.0,
                volatility_30d=0.0,
                correlation_breakdown=0.0,
                dispersion=0.0,
                momentum=0.0,
                volume_profile=0.0,
                regime=MarketRegime.SIDEWAYS,
                confidence=0.0,
                timestamp=datetime.now()
            )
    
    def get_regime_transition_probability(self) -> Dict[str, float]:
        """Calculate transition probabilities between regimes"""
        try:
            if len(self.regime_history) < 10:
                return {}
            
            # Count transitions
            transitions = {}
            current_regime = None
            
            for metrics in self.regime_history:
                if current_regime is not None:
                    transition_key = f"{current_regime.value} -> {metrics.regime.value}"
                    transitions[transition_key] = transitions.get(transition_key, 0) + 1
                
                current_regime = metrics.regime
            
            # Calculate probabilities
            total_transitions = sum(transitions.values())
            probabilities = {}
            
            for transition, count in transitions.items():
                probabilities[transition] = count / total_transitions
            
            return probabilities
            
        except Exception as e:
            logger.error(f"Error calculating transition probabilities: {e}")
            return {}
    
    def get_regime_stability(self) -> float:
        """Calculate regime stability (how often regime changes)"""
        try:
            if len(self.regime_history) < 10:
                return 0.0
            
            # Count regime changes
            changes = 0
            current_regime = None
            
            for metrics in self.regime_history:
                if current_regime is not None and current_regime != metrics.regime:
                    changes += 1
                current_regime = metrics.regime
            
            # Calculate stability (lower changes = higher stability)
            stability = 1.0 - (changes / len(self.regime_history))
            return max(0.0, min(1.0, stability))
            
        except Exception as e:
            logger.error(f"Error calculating regime stability: {e}")
            return 0.0
    
    def get_regime_summary(self) -> Dict:
        """Get comprehensive regime summary"""
        try:
            if not self.regime_history:
                return {}
            
            current_regime = self.regime_history[-1]
            
            return {
                'current_regime': current_regime.regime.value,
                'confidence': current_regime.confidence,
                'volatility_10d': current_regime.volatility_10d,
                'volatility_30d': current_regime.volatility_30d,
                'correlation_breakdown': current_regime.correlation_breakdown,
                'dispersion': current_regime.dispersion,
                'momentum': current_regime.momentum,
                'volume_profile': current_regime.volume_profile,
                'stability': self.get_regime_stability(),
                'transition_probabilities': self.get_regime_transition_probability(),
                'timestamp': current_regime.timestamp
            }
            
        except Exception as e:
            logger.error(f"Error getting regime summary: {e}")
            return {}

class RegimeManager:
    """Manages regime detection and provides regime-aware features"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.detector = RegimeDetector(config)
        self.current_regime = MarketRegime.SIDEWAYS
        self.regime_history = []
        
        logger.info("Regime Manager initialized")
    
    def update_regime(self, prices_df: pd.DataFrame, volume_df: pd.DataFrame = None) -> RegimeMetrics:
        """Update current regime"""
        regime_metrics = self.detector.detect_regime(prices_df, volume_df)
        self.current_regime = regime_metrics.regime
        self.regime_history.append(regime_metrics)
        
        # Keep only recent history
        if len(self.regime_history) > 100:
            self.regime_history = self.regime_history[-100:]
        
        return regime_metrics
    
    def get_regime_features(self) -> Dict[str, float]:
        """Get regime features for ML models"""
        try:
            if not self.regime_history:
                return {}
            
            current = self.regime_history[-1]
            
            # One-hot encode regime
            regime_features = {}
            for regime in MarketRegime:
                regime_features[f'regime_{regime.value}'] = 1.0 if current.regime == regime else 0.0
            
            # Add continuous features
            regime_features.update({
                'regime_confidence': current.confidence,
                'volatility_10d': current.volatility_10d,
                'volatility_30d': current.volatility_30d,
                'correlation_breakdown': current.correlation_breakdown,
                'dispersion': current.dispersion,
                'momentum': current.momentum,
                'volume_profile': current.volume_profile
            })
            
            return regime_features
            
        except Exception as e:
            logger.error(f"Error getting regime features: {e}")
            return {}
    
    def should_trigger_escalation(self) -> bool:
        """Check if regime change should trigger GPT-5 escalation"""
        try:
            if len(self.regime_history) < 2:
                return False
            
            current = self.regime_history[-1]
            previous = self.regime_history[-2]
            
            # Trigger escalation if:
            # 1. Regime changed
            # 2. High confidence in new regime
            # 3. Significant volatility change
            
            regime_changed = current.regime != previous.regime
            high_confidence = current.confidence > 0.7
            vol_spike = abs(current.volatility_10d - previous.volatility_10d) > 0.1
            
            return regime_changed and (high_confidence or vol_spike)
            
        except Exception as e:
            logger.error(f"Error checking escalation trigger: {e}")
            return False
    
    def get_regime_recommendations(self) -> Dict[str, str]:
        """Get trading recommendations based on current regime"""
        try:
            if not self.regime_history:
                return {}
            
            current = self.regime_history[-1]
            regime = current.regime
            
            recommendations = {}
            
            if regime == MarketRegime.LOW_VOL:
                recommendations.update({
                    'strategy': 'mean_reversion',
                    'position_sizing': 'increase',
                    'risk_management': 'standard',
                    'timeframe': 'short_term'
                })
            elif regime == MarketRegime.HIGH_VOL:
                recommendations.update({
                    'strategy': 'momentum',
                    'position_sizing': 'reduce',
                    'risk_management': 'aggressive',
                    'timeframe': 'medium_term'
                })
            elif regime == MarketRegime.BULL:
                recommendations.update({
                    'strategy': 'trend_following',
                    'position_sizing': 'increase',
                    'risk_management': 'standard',
                    'timeframe': 'long_term'
                })
            elif regime == MarketRegime.BEAR:
                recommendations.update({
                    'strategy': 'defensive',
                    'position_sizing': 'reduce',
                    'risk_management': 'aggressive',
                    'timeframe': 'short_term'
                })
            elif regime == MarketRegime.TRENDING:
                recommendations.update({
                    'strategy': 'momentum',
                    'position_sizing': 'normal',
                    'risk_management': 'standard',
                    'timeframe': 'medium_term'
                })
            else:  # SIDEWAYS
                recommendations.update({
                    'strategy': 'mean_reversion',
                    'position_sizing': 'normal',
                    'risk_management': 'standard',
                    'timeframe': 'short_term'
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting regime recommendations: {e}")
            return {}
