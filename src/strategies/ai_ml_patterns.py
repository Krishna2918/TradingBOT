"""
AI/ML Pattern Discovery Strategy
Uses machine learning to identify trading patterns
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

logger = logging.getLogger(__name__)

@dataclass
class MLSignal:
    """Machine learning trading signal"""
    symbol: str
    side: str  # 'BUY' or 'SELL'
    strength: float
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    pattern_type: str
    feature_importance: Dict[str, float]
    model_version: str
    timestamp: datetime

class AIMLPatternStrategy:
    """AI/ML Pattern Discovery Strategy for Canadian Markets"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.enabled = config.get('enabled', True)
        self.allocation = config.get('allocation', 0.20)
        self.max_leverage = config.get('max_leverage', 1.8)
        
        # Strategy parameters
        self.ml_confidence = config.get('signals', {}).get('ml_confidence', 0.8)
        self.pattern_score = config.get('signals', {}).get('pattern_score', 0.75)
        self.sentiment_alignment = config.get('signals', {}).get('sentiment_alignment', 0.7)
        self.stop_loss = config.get('risk', {}).get('stop_loss', 0.4)
        self.take_profit = config.get('risk', {}).get('take_profit', 0.8)
        self.max_holding_time = config.get('risk', {}).get('max_holding_time', 120)
        
        # Canadian market instruments
        self.instruments = config.get('instruments', [
            "RY.TO", "TD.TO", "SHOP.TO", "CNR.TO", "CP.TO", "ENB.TO", "TRP.TO"
        ])
        
        # ML models
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.model_version = "1.0"
        
        # Pattern recognition
        self.patterns = {
            'momentum_breakout': {'weight': 0.3, 'threshold': 0.7},
            'mean_reversion': {'weight': 0.25, 'threshold': 0.6},
            'volume_spike': {'weight': 0.2, 'threshold': 0.8},
            'support_resistance': {'weight': 0.15, 'threshold': 0.65},
            'news_sentiment': {'weight': 0.1, 'threshold': 0.7}
        }
        
        self.active_positions = {}
        self.signal_history = []
        self.price_history = {}
        self.feature_history = {}
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models for pattern recognition"""
        try:
            # Create feature names
            self.feature_names = [
                'price_change_1m', 'price_change_5m', 'price_change_15m',
                'volume_ratio', 'rsi_14', 'bollinger_position',
                'macd_signal', 'vwap_deviation', 'volatility_20m',
                'momentum_5m', 'mean_reversion_score', 'support_resistance_score'
            ]
            
            # Initialize models for each instrument
            for symbol in self.instruments:
                # Random Forest for pattern classification
                self.models[symbol] = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    class_weight='balanced'
                )
                
                # Feature scaler
                self.scalers[symbol] = StandardScaler()
            
            logger.info(f"Initialized ML models for {len(self.instruments)} instruments")
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
    
    def analyze_ml_patterns(self, market_data: Dict, news_data: List[Dict] = None) -> List[MLSignal]:
        """Analyze market data using ML pattern recognition"""
        try:
            signals = []
            
            # Update feature history
            self._update_feature_history(market_data)
            
            for symbol in self.instruments:
                if symbol in market_data:
                    signal = self._analyze_symbol_patterns(symbol, market_data[symbol], news_data)
                    if signal:
                        signals.append(signal)
            
            logger.info(f"Generated {len(signals)} ML pattern signals")
            return signals
            
        except Exception as e:
            logger.error(f"Failed to analyze ML patterns: {e}")
            return []
    
    def _update_feature_history(self, market_data: Dict):
        """Update feature history for ML analysis"""
        try:
            current_time = datetime.now()
            
            for symbol, data in market_data.items():
                if symbol in self.instruments:
                    # Calculate features
                    features = self._calculate_features(symbol, data)
                    
                    if symbol not in self.feature_history:
                        self.feature_history[symbol] = []
                    
                    self.feature_history[symbol].append({
                        'features': features,
                        'timestamp': current_time,
                        'price': data.get('close', 0)
                    })
                    
                    # Keep only last 100 data points
                    if len(self.feature_history[symbol]) > 100:
                        self.feature_history[symbol] = self.feature_history[symbol][-100:]
            
        except Exception as e:
            logger.error(f"Failed to update feature history: {e}")
    
    def _calculate_features(self, symbol: str, data: Dict) -> List[float]:
        """Calculate technical features for ML model"""
        try:
            features = []
            current_price = data.get('close', 0)
            volume = data.get('volume', 0)
            high = data.get('high', 0)
            low = data.get('low', 0)
            
            if current_price <= 0:
                return [0.0] * len(self.feature_names)
            
            # Price changes (mock calculations)
            features.append(0.02)  # price_change_1m
            features.append(0.05)  # price_change_5m
            features.append(0.08)  # price_change_15m
            
            # Volume ratio
            avg_volume = 1000000  # Mock average volume
            volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
            features.append(volume_ratio)
            
            # RSI (mock calculation)
            rsi = 50 + np.random.normal(0, 10)  # Mock RSI
            features.append(max(0, min(100, rsi)))
            
            # Bollinger position (mock)
            bb_position = 0.5 + np.random.normal(0, 0.2)
            features.append(max(0, min(1, bb_position)))
            
            # MACD signal (mock)
            macd_signal = np.random.normal(0, 0.1)
            features.append(macd_signal)
            
            # VWAP deviation (mock)
            vwap_deviation = np.random.normal(0, 0.02)
            features.append(vwap_deviation)
            
            # Volatility (mock)
            volatility = abs(np.random.normal(0, 0.01))
            features.append(volatility)
            
            # Momentum (mock)
            momentum = np.random.normal(0, 0.05)
            features.append(momentum)
            
            # Mean reversion score (mock)
            mean_reversion = np.random.normal(0, 0.3)
            features.append(mean_reversion)
            
            # Support/resistance score (mock)
            support_resistance = np.random.normal(0, 0.2)
            features.append(support_resistance)
            
            return features
            
        except Exception as e:
            logger.error(f"Failed to calculate features for {symbol}: {e}")
            return [0.0] * len(self.feature_names)
    
    def _analyze_symbol_patterns(self, symbol: str, data: Dict, news_data: List[Dict] = None) -> Optional[MLSignal]:
        """Analyze patterns for individual symbol using ML"""
        try:
            current_price = data.get('close', 0)
            if current_price <= 0:
                return None
            
            # Get recent features
            if symbol not in self.feature_history or len(self.feature_history[symbol]) < 10:
                return None
            
            recent_features = self.feature_history[symbol][-1]['features']
            
            # Predict using ML model (mock prediction)
            prediction = self._predict_pattern(symbol, recent_features)
            
            if prediction is None:
                return None
            
            pattern_type, confidence, side = prediction
            
            # Check confidence thresholds
            if confidence < self.ml_confidence:
                return None
            
            # Calculate pattern score
            pattern_score = self._calculate_pattern_score(symbol, recent_features, pattern_type)
            if pattern_score < self.pattern_score:
                return None
            
            # Check sentiment alignment if news data available
            if news_data:
                sentiment_score = self._calculate_news_sentiment(symbol, news_data)
                if abs(sentiment_score) < self.sentiment_alignment:
                    return None
            
            # Calculate feature importance (mock)
            feature_importance = self._calculate_feature_importance(symbol, recent_features)
            
            return MLSignal(
                symbol=symbol,
                side=side,
                strength=min(confidence * pattern_score, 1.0),
                entry_price=current_price,
                stop_loss=self._calculate_stop_loss(current_price, side),
                take_profit=self._calculate_take_profit(current_price, side),
                confidence=confidence,
                pattern_type=pattern_type,
                feature_importance=feature_importance,
                model_version=self.model_version,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze patterns for {symbol}: {e}")
            return None
    
    def _predict_pattern(self, symbol: str, features: List[float]) -> Optional[Tuple[str, float, str]]:
        """Predict pattern using ML model"""
        try:
            # Mock ML prediction
            # In real implementation, this would use trained models
            
            # Simulate pattern prediction
            pattern_scores = {}
            for pattern, config in self.patterns.items():
                score = np.random.random()
                if score > config['threshold']:
                    pattern_scores[pattern] = score
            
            if not pattern_scores:
                return None
            
            # Get best pattern
            best_pattern = max(pattern_scores, key=pattern_scores.get)
            confidence = pattern_scores[best_pattern]
            
            # Determine side based on pattern
            if best_pattern in ['momentum_breakout', 'volume_spike']:
                side = 'BUY' if np.random.random() > 0.5 else 'SELL'
            elif best_pattern == 'mean_reversion':
                side = 'SELL' if np.random.random() > 0.5 else 'BUY'
            else:
                side = 'BUY' if np.random.random() > 0.5 else 'SELL'
            
            return best_pattern, confidence, side
            
        except Exception as e:
            logger.error(f"Failed to predict pattern for {symbol}: {e}")
            return None
    
    def _calculate_pattern_score(self, symbol: str, features: List[float], pattern_type: str) -> float:
        """Calculate pattern score based on features"""
        try:
            # Mock pattern scoring
            # In real implementation, this would use specific pattern recognition algorithms
            
            base_score = np.random.random()
            
            # Adjust based on pattern type
            if pattern_type == 'momentum_breakout':
                # Use momentum-related features
                momentum_score = abs(features[9])  # momentum_5m
                base_score = (base_score + momentum_score) / 2
            elif pattern_type == 'mean_reversion':
                # Use mean reversion features
                mr_score = abs(features[10])  # mean_reversion_score
                base_score = (base_score + mr_score) / 2
            elif pattern_type == 'volume_spike':
                # Use volume features
                volume_score = features[3]  # volume_ratio
                base_score = (base_score + min(volume_score, 1.0)) / 2
            
            return min(base_score, 1.0)
            
        except Exception as e:
            logger.error(f"Failed to calculate pattern score: {e}")
            return 0.5
    
    def _calculate_news_sentiment(self, symbol: str, news_data: List[Dict]) -> float:
        """Calculate news sentiment for symbol"""
        try:
            if not news_data:
                return 0.0
            
            # Mock sentiment calculation
            # In real implementation, this would use NLP models
            
            sentiment_scores = []
            for news in news_data:
                # Simple keyword-based sentiment
                title = news.get('title', '').lower()
                if any(word in title for word in ['positive', 'strong', 'growth', 'beat']):
                    sentiment_scores.append(0.5)
                elif any(word in title for word in ['negative', 'weak', 'decline', 'miss']):
                    sentiment_scores.append(-0.5)
                else:
                    sentiment_scores.append(0.0)
            
            return np.mean(sentiment_scores) if sentiment_scores else 0.0
            
        except Exception as e:
            logger.error(f"Failed to calculate news sentiment: {e}")
            return 0.0
    
    def _calculate_feature_importance(self, symbol: str, features: List[float]) -> Dict[str, float]:
        """Calculate feature importance for the prediction"""
        try:
            # Mock feature importance
            # In real implementation, this would come from the ML model
            
            importance = {}
            for i, feature_name in enumerate(self.feature_names):
                importance[feature_name] = abs(features[i]) if i < len(features) else 0.0
            
            # Normalize importance scores
            total_importance = sum(importance.values())
            if total_importance > 0:
                for key in importance:
                    importance[key] /= total_importance
            
            return importance
            
        except Exception as e:
            logger.error(f"Failed to calculate feature importance: {e}")
            return {}
    
    def _calculate_stop_loss(self, entry_price: float, side: str) -> float:
        """Calculate stop loss price"""
        if side == 'BUY':
            return entry_price * (1 - self.stop_loss / 100)
        else:
            return entry_price * (1 + self.stop_loss / 100)
    
    def _calculate_take_profit(self, entry_price: float, side: str) -> float:
        """Calculate take profit price"""
        if side == 'BUY':
            return entry_price * (1 + self.take_profit / 100)
        else:
            return entry_price * (1 - self.take_profit / 100)
    
    def should_exit_position(self, symbol: str, current_price: float, entry_time: datetime) -> bool:
        """Check if position should be exited"""
        try:
            # Check holding time
            holding_time = datetime.now() - entry_time
            if holding_time.total_seconds() > (self.max_holding_time * 60):
                logger.info(f"Exiting {symbol} due to max holding time")
                return True
            
            # Check stop loss and take profit
            if symbol in self.active_positions:
                position = self.active_positions[symbol]
                side = position['side']
                
                if side == 'BUY':
                    if current_price <= position['stop_loss'] or current_price >= position['take_profit']:
                        return True
                else:
                    if current_price >= position['stop_loss'] or current_price <= position['take_profit']:
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check exit conditions for {symbol}: {e}")
            return False
    
    def retrain_models(self, historical_data: Dict):
        """Retrain ML models with new data"""
        try:
            logger.info("Retraining ML models...")
            
            # Mock retraining
            # In real implementation, this would use actual historical data
            
            for symbol in self.instruments:
                if symbol in historical_data:
                    # Mock model retraining
                    logger.info(f"Retrained model for {symbol}")
            
            self.model_version = f"1.{int(datetime.now().timestamp())}"
            logger.info(f"Models retrained successfully. New version: {self.model_version}")
            
        except Exception as e:
            logger.error(f"Failed to retrain models: {e}")
    
    def get_strategy_status(self) -> Dict:
        """Get current strategy status"""
        return {
            'enabled': self.enabled,
            'allocation': self.allocation,
            'max_leverage': self.max_leverage,
            'active_positions': len(self.active_positions),
            'signals_generated': len(self.signal_history),
            'instruments': self.instruments,
            'model_version': self.model_version,
            'patterns': list(self.patterns.keys()),
            'parameters': {
                'ml_confidence': self.ml_confidence,
                'pattern_score': self.pattern_score,
                'sentiment_alignment': self.sentiment_alignment,
                'stop_loss': self.stop_loss,
                'take_profit': self.take_profit,
                'max_holding_time': self.max_holding_time
            }
        }

# Test the strategy
if __name__ == "__main__":
    config = {
        'enabled': True,
        'allocation': 0.20,
        'max_leverage': 1.8,
        'instruments': ['RY.TO', 'TD.TO', 'SHOP.TO'],
        'signals': {
            'ml_confidence': 0.8,
            'pattern_score': 0.75,
            'sentiment_alignment': 0.7
        },
        'risk': {
            'stop_loss': 0.4,
            'take_profit': 0.8,
            'max_holding_time': 120
        }
    }
    
    strategy = AIMLPatternStrategy(config)
    
    # Test with mock market data
    mock_market_data = {
        'RY.TO': {
            'open': 100.0,
            'high': 102.0,
            'low': 99.0,
            'close': 101.5,
            'volume': 2000000
        }
    }
    
    signals = strategy.analyze_ml_patterns(mock_market_data)
    print(f"Generated {len(signals)} ML pattern signals")
    
    status = strategy.get_strategy_status()
    print(f"Strategy status: {status}")
