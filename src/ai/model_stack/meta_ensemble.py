"""
Meta-Ensemble Model
Combines LSTM (short-term) and GRU-Transformer (mid-term) predictions
with regime detection and confidence weighting
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path

from .lstm_model import LSTMPredictor
from .gru_transformer_model import GRUTransformerPredictor

logger = logging.getLogger(__name__)

class RegimeDetector:
    """
    Market Regime Detector
    
    Detects market conditions:
    - Trending (bullish/bearish)
    - Range-bound
    - High volatility
    - Low volatility
    """
    
    def __init__(self):
        self.regimes = {
            'bullish_trend': 0,
            'bearish_trend': 1,
            'range_bound': 2,
            'high_volatility': 3,
            'low_volatility': 4
        }
    
    def detect_regime(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Detect current market regime
        
        Returns probabilities for each regime
        """
        
        # Calculate indicators for regime detection
        returns = df['close'].pct_change()
        volatility = returns.rolling(window=20).std()
        
        # Trend detection
        sma_50 = df['close'].rolling(window=50).mean()
        sma_200 = df['close'].rolling(window=200).mean()
        
        current_price = df['close'].iloc[-1]
        current_vol = volatility.iloc[-1]
        
        # Calculate regime probabilities
        regimes = {}
        
        # Bullish trend
        if pd.notna(sma_50.iloc[-1]) and pd.notna(sma_200.iloc[-1]):
            if current_price > sma_50.iloc[-1] > sma_200.iloc[-1]:
                regimes['bullish_trend'] = 0.8
            else:
                regimes['bullish_trend'] = 0.2
        else:
            regimes['bullish_trend'] = 0.5
        
        # Bearish trend
        if pd.notna(sma_50.iloc[-1]) and pd.notna(sma_200.iloc[-1]):
            if current_price < sma_50.iloc[-1] < sma_200.iloc[-1]:
                regimes['bearish_trend'] = 0.8
            else:
                regimes['bearish_trend'] = 0.2
        else:
            regimes['bearish_trend'] = 0.5
        
        # Range-bound
        price_range = (df['high'].rolling(window=20).max() - df['low'].rolling(window=20).min()).iloc[-1]
        price_std = df['close'].rolling(window=20).std().iloc[-1]
        if pd.notna(price_range) and pd.notna(price_std) and price_std > 0:
            range_ratio = price_range / (current_price * 0.1)  # 10% reference
            if range_ratio < 1.5:
                regimes['range_bound'] = 0.7
            else:
                regimes['range_bound'] = 0.3
        else:
            regimes['range_bound'] = 0.5
        
        # High volatility
        if pd.notna(current_vol):
            avg_vol = volatility.mean()
            if avg_vol > 0 and current_vol > avg_vol * 1.5:
                regimes['high_volatility'] = 0.8
            else:
                regimes['high_volatility'] = 0.2
        else:
            regimes['high_volatility'] = 0.5
        
        # Low volatility
        if pd.notna(current_vol):
            avg_vol = volatility.mean()
            if avg_vol > 0 and current_vol < avg_vol * 0.7:
                regimes['low_volatility'] = 0.8
            else:
                regimes['low_volatility'] = 0.2
        else:
            regimes['low_volatility'] = 0.5
        
        # Normalize probabilities
        total = sum(regimes.values())
        if total > 0:
            regimes = {k: v / total for k, v in regimes.items()}
        
        return regimes

class MetaEnsemble:
    """
    Meta-Ensemble Model
    
    Combines predictions from:
    - LSTM (short-term, 1-minute)
    - GRU-Transformer (mid-term, 5-15 minute)
    
    With:
    - Regime-based weighting
    - Confidence-based aggregation
    - Consensus analysis
    """
    
    def __init__(
        self,
        lstm_model_path: str = "models/lstm_model.pth",
        gru_model_path: str = "models/gru_transformer_model.pth",
        meta_model_path: str = "models/meta_ensemble.pkl"
    ):
        self.lstm_predictor = LSTMPredictor(model_path=lstm_model_path)
        self.gru_predictor = GRUTransformerPredictor(model_path=gru_model_path)
        self.regime_detector = RegimeDetector()
        
        self.meta_model_path = Path(meta_model_path)
        self.meta_model_path.parent.mkdir(parents=True, exist_ok=True)
        self.meta_model = None  # RandomForest meta-learner
        
        logger.info(" Meta-Ensemble initialized")
    
    def calculate_model_weights(self, regimes: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate model weights based on market regime
        
        LSTM performs better in:
        - High volatility
        - Short-term trends
        
        GRU-Transformer performs better in:
        - Trending markets
        - Low volatility
        - Regime-based patterns
        """
        
        lstm_weight = 0.5  # Default equal weight
        gru_weight = 0.5
        
        # Adjust based on regimes
        if regimes.get('high_volatility', 0) > 0.6:
            lstm_weight += 0.2
            gru_weight -= 0.2
        
        if regimes.get('low_volatility', 0) > 0.6:
            lstm_weight -= 0.1
            gru_weight += 0.1
        
        if regimes.get('bullish_trend', 0) > 0.6 or regimes.get('bearish_trend', 0) > 0.6:
            lstm_weight -= 0.1
            gru_weight += 0.1
        
        if regimes.get('range_bound', 0) > 0.6:
            lstm_weight += 0.1
            gru_weight -= 0.1
        
        # Normalize weights
        total_weight = lstm_weight + gru_weight
        lstm_weight /= total_weight
        gru_weight /= total_weight
        
        return {
            'lstm': lstm_weight,
            'gru_transformer': gru_weight
        }
    
    def aggregate_predictions(
        self,
        lstm_pred: Dict[str, float],
        gru_pred: Dict[str, float],
        weights: Dict[str, float]
    ) -> Dict[str, any]:
        """
        Aggregate predictions from multiple models
        
        Uses weighted averaging of probabilities
        """
        
        # Extract probabilities
        lstm_probs = np.array(lstm_pred['probabilities'])
        gru_probs = np.array(gru_pred['probabilities'])
        
        # Weighted average
        ensemble_probs = (
            lstm_probs * weights['lstm'] +
            gru_probs * weights['gru_transformer']
        )
        
        # Get direction and confidence
        predicted_class = np.argmax(ensemble_probs)
        confidence = float(ensemble_probs[predicted_class])
        
        direction_map = {0: 'down', 1: 'neutral', 2: 'up'}
        direction = direction_map[predicted_class]
        
        # Calculate agreement
        lstm_direction = lstm_pred['direction']
        gru_direction = gru_pred['direction']
        
        agreement = 1.0 if lstm_direction == gru_direction else 0.5
        
        # Calculate consensus strength
        consensus_strength = (
            lstm_pred['confidence'] * weights['lstm'] +
            gru_pred['confidence'] * weights['gru_transformer']
        ) * agreement
        
        return {
            'direction': direction,
            'confidence': confidence,
            'probabilities': ensemble_probs.tolist(),
            'consensus_strength': consensus_strength,
            'agreement': agreement,
            'individual_predictions': {
                'lstm': lstm_pred,
                'gru_transformer': gru_pred
            },
            'weights': weights
        }
    
    def predict(
        self,
        df: pd.DataFrame,
        macro_data: Optional[Dict] = None,
        options_data: Optional[Dict] = None
    ) -> Dict[str, any]:
        """
        Generate ensemble prediction
        
        Returns comprehensive prediction with:
        - Direction (up/down/neutral)
        - Confidence
        - Probabilities
        - Regime analysis
        - Individual model predictions
        - Consensus analysis
        """
        
        logger.info(" Generating meta-ensemble prediction...")
        
        # Detect market regime
        regimes = self.regime_detector.detect_regime(df)
        dominant_regime = max(regimes.items(), key=lambda x: x[1])[0]
        
        logger.info(f" Market regime: {dominant_regime} ({regimes[dominant_regime]:.2f})")
        
        # Calculate model weights based on regime
        weights = self.calculate_model_weights(regimes)
        
        logger.info(f" Model weights: LSTM={weights['lstm']:.2f}, GRU-Transformer={weights['gru_transformer']:.2f}")
        
        # Get predictions from individual models
        try:
            lstm_pred = self.lstm_predictor.predict(df)
            logger.info(f" LSTM: {lstm_pred['direction']} ({lstm_pred['confidence']:.2f})")
        except Exception as e:
            logger.error(f" LSTM prediction failed: {e}")
            lstm_pred = {
                'direction': 'neutral',
                'confidence': 0.33,
                'probabilities': [0.33, 0.34, 0.33]
            }
        
        try:
            gru_pred = self.gru_predictor.predict(df, macro_data, options_data)
            logger.info(f" GRU-Transformer: {gru_pred['direction']} ({gru_pred['confidence']:.2f})")
        except Exception as e:
            logger.error(f" GRU-Transformer prediction failed: {e}")
            gru_pred = {
                'direction': 'neutral',
                'confidence': 0.33,
                'probabilities': [0.33, 0.34, 0.33]
            }
        
        # Aggregate predictions
        ensemble_pred = self.aggregate_predictions(lstm_pred, gru_pred, weights)
        
        # Add regime information
        ensemble_pred['regime'] = {
            'dominant': dominant_regime,
            'probabilities': regimes
        }
        
        logger.info(
            f" Ensemble: {ensemble_pred['direction']} "
            f"(confidence={ensemble_pred['confidence']:.2f}, "
            f"consensus={ensemble_pred['consensus_strength']:.2f})"
        )
        
        return ensemble_pred
    
    def train_meta_model(
        self,
        historical_predictions: List[Dict],
        actual_outcomes: List[int]
    ):
        """
        Train meta-learner on historical predictions and outcomes
        
        The meta-model learns to weight individual model predictions
        based on their historical performance in different regimes
        """
        
        logger.info(" Training meta-learner...")
        
        # Prepare training data
        X = []
        y = actual_outcomes
        
        for pred in historical_predictions:
            features = [
                # Individual model predictions
                pred['individual_predictions']['lstm']['probabilities'][0],
                pred['individual_predictions']['lstm']['probabilities'][1],
                pred['individual_predictions']['lstm']['probabilities'][2],
                pred['individual_predictions']['gru_transformer']['probabilities'][0],
                pred['individual_predictions']['gru_transformer']['probabilities'][1],
                pred['individual_predictions']['gru_transformer']['probabilities'][2],
                
                # Model confidences
                pred['individual_predictions']['lstm']['confidence'],
                pred['individual_predictions']['gru_transformer']['confidence'],
                
                # Regime information
                pred['regime']['probabilities'].get('bullish_trend', 0),
                pred['regime']['probabilities'].get('bearish_trend', 0),
                pred['regime']['probabilities'].get('range_bound', 0),
                pred['regime']['probabilities'].get('high_volatility', 0),
                pred['regime']['probabilities'].get('low_volatility', 0),
                
                # Agreement metrics
                pred['agreement'],
                pred['consensus_strength']
            ]
            X.append(features)
        
        X = np.array(X)
        y = np.array(y)
        
        # Train Random Forest meta-learner
        self.meta_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        self.meta_model.fit(X, y)
        
        # Save meta-model
        joblib.dump(self.meta_model, self.meta_model_path)
        
        logger.info(" Meta-learner training completed")
    
    def predict_with_meta_model(
        self,
        df: pd.DataFrame,
        macro_data: Optional[Dict] = None,
        options_data: Optional[Dict] = None
    ) -> Dict[str, any]:
        """
        Generate prediction using trained meta-model
        
        Uses the meta-learner to combine base model predictions
        """
        
        # Get base ensemble prediction
        base_pred = self.predict(df, macro_data, options_data)
        
        # If meta-model is trained, use it for final prediction
        if self.meta_model is not None:
            # Prepare features for meta-model
            features = [
                base_pred['individual_predictions']['lstm']['probabilities'][0],
                base_pred['individual_predictions']['lstm']['probabilities'][1],
                base_pred['individual_predictions']['lstm']['probabilities'][2],
                base_pred['individual_predictions']['gru_transformer']['probabilities'][0],
                base_pred['individual_predictions']['gru_transformer']['probabilities'][1],
                base_pred['individual_predictions']['gru_transformer']['probabilities'][2],
                base_pred['individual_predictions']['lstm']['confidence'],
                base_pred['individual_predictions']['gru_transformer']['confidence'],
                base_pred['regime']['probabilities'].get('bullish_trend', 0),
                base_pred['regime']['probabilities'].get('bearish_trend', 0),
                base_pred['regime']['probabilities'].get('range_bound', 0),
                base_pred['regime']['probabilities'].get('high_volatility', 0),
                base_pred['regime']['probabilities'].get('low_volatility', 0),
                base_pred['agreement'],
                base_pred['consensus_strength']
            ]
            
            X = np.array([features])
            
            # Get meta-model prediction
            meta_pred = self.meta_model.predict(X)[0]
            meta_proba = self.meta_model.predict_proba(X)[0]
            
            # Update prediction with meta-model results
            direction_map = {0: 'down', 1: 'neutral', 2: 'up'}
            base_pred['meta_direction'] = direction_map[meta_pred]
            base_pred['meta_confidence'] = float(meta_proba[meta_pred])
            base_pred['meta_probabilities'] = meta_proba.tolist()
            
            logger.info(f" Meta-model: {base_pred['meta_direction']} ({base_pred['meta_confidence']:.2f})")
        
        return base_pred
    
    def get_model_performance(self) -> Dict[str, any]:
        """Get performance metrics for individual models"""
        
        return {
            'lstm': {
                'description': 'Short-term (1-minute) price prediction',
                'features': len(self.lstm_predictor.feature_names),
                'sequence_length': self.lstm_predictor.sequence_length
            },
            'gru_transformer': {
                'description': 'Mid-term (5-15 minute) price prediction',
                'features': len(self.gru_predictor.feature_names),
                'sequence_length': self.gru_predictor.sequence_length
            },
            'meta_model': {
                'trained': self.meta_model is not None,
                'type': 'Random Forest Classifier'
            }
        }

def create_meta_ensemble() -> MetaEnsemble:
    """Factory function to create meta-ensemble instance"""
    return MetaEnsemble()

