"""
Deep Learning Model Ensemble for Trading Predictions
Combines LSTM (short-term) and GRU-Transformer (mid-term) predictions
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json

from src.ai.production_lstm_loader import ProductionLSTMPredictor
from src.ai.model_stack.gru_transformer_model import GRUTransformerPredictor

logger = logging.getLogger(__name__)


class ModelEnsemble:
    """
    Ensemble of deep learning models for trading predictions

    Models:
    1. LSTM - Short-term (1-minute) predictions
    2. GRU-Transformer - Mid-term (5-15 minute) predictions

    Voting methods:
    - Weighted Average: Combine probabilities with custom weights
    - Majority Vote: Take the class with most votes
    - Confidence Weighted: Weight by each model's confidence
    """

    def __init__(
        self,
        lstm_model_path: str = "models/lstm_best.pth",
        gru_model_path: str = "models/gru_transformer_10h.pth",
        lstm_weight: float = 0.4,
        gru_weight: float = 0.6,
        voting_method: str = "confidence_weighted"  # weighted_average, majority_vote, confidence_weighted
    ):
        self.lstm_model_path = Path(lstm_model_path)
        self.gru_model_path = Path(gru_model_path)

        # Model weights (should sum to 1.0)
        self.lstm_weight = lstm_weight
        self.gru_weight = gru_weight

        # Normalize weights
        total_weight = self.lstm_weight + self.gru_weight
        self.lstm_weight /= total_weight
        self.gru_weight /= total_weight

        self.voting_method = voting_method

        # Initialize models
        logger.info("Initializing model ensemble...")

        # Production LSTM with trained model configuration
        self.lstm = ProductionLSTMPredictor(
            model_path=str(self.lstm_model_path),
            input_size=58,
            hidden_size=256,
            num_layers=3,
            dropout=0.3,
            sequence_length=30
        )

        # GRU-Transformer with default configuration
        self.gru = GRUTransformerPredictor(model_path=str(self.gru_model_path))

        # Load models
        self.lstm_loaded = False
        self.gru_loaded = False

        self.load_models()

        # Performance tracking
        self.prediction_history = []
        self.performance_stats = {
            'lstm_accuracy': [],
            'gru_accuracy': [],
            'ensemble_accuracy': []
        }

    def load_models(self):
        """Load all models"""
        logger.info("Loading deep learning models...")

        try:
            self.lstm_loaded = self.lstm.load_model()
            if self.lstm_loaded:
                logger.info(f"✓ LSTM model loaded from {self.lstm_model_path}")
            else:
                logger.warning(f"✗ LSTM model failed to load from {self.lstm_model_path}")
        except Exception as e:
            logger.error(f"✗ LSTM model error: {e}")
            self.lstm_loaded = False

        try:
            self.gru_loaded = self.gru.load_model()
            if self.gru_loaded:
                logger.info(f"✓ GRU-Transformer model loaded from {self.gru_model_path}")
            else:
                logger.warning(f"✗ GRU-Transformer model failed to load from {self.gru_model_path}")
        except Exception as e:
            logger.error(f"✗ GRU-Transformer model error: {e}")
            self.gru_loaded = False

        if not self.lstm_loaded and not self.gru_loaded:
            logger.error("No models loaded! Ensemble will return neutral predictions.")
        elif self.lstm_loaded and self.gru_loaded:
            logger.info(f"✓ All models loaded successfully. Voting method: {self.voting_method}")
        else:
            logger.warning(f"⚠ Only {'LSTM' if self.lstm_loaded else 'GRU'} loaded. Using single model predictions.")

    def predict(
        self,
        df: pd.DataFrame,
        macro_data: Optional[Dict] = None,
        options_data: Optional[Dict] = None,
        symbol: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Generate ensemble prediction

        Args:
            df: DataFrame with OHLCV data
            macro_data: Optional macro indicators
            options_data: Optional options data
            symbol: Optional stock symbol for logging

        Returns:
            {
                'direction': 'up'/'down'/'neutral',
                'confidence': 0.0-1.0,
                'probabilities': [down_prob, neutral_prob, up_prob],
                'lstm_prediction': {...},
                'gru_prediction': {...},
                'voting_method': str,
                'model_agreement': bool
            }
        """

        predictions = {}

        # Get LSTM prediction (short-term)
        if self.lstm_loaded:
            try:
                lstm_pred = self.lstm.predict(df)
                predictions['lstm'] = lstm_pred
            except Exception as e:
                logger.error(f"LSTM prediction failed: {e}")
                predictions['lstm'] = None
        else:
            predictions['lstm'] = None

        # Get GRU-Transformer prediction (mid-term)
        if self.gru_loaded:
            try:
                gru_pred = self.gru.predict(df, macro_data, options_data)
                predictions['gru'] = gru_pred
            except Exception as e:
                logger.error(f"GRU prediction failed: {e}")
                predictions['gru'] = None
        else:
            predictions['gru'] = None

        # Check if we have any valid predictions
        valid_predictions = [p for p in predictions.values() if p is not None]

        if not valid_predictions:
            logger.warning("No valid predictions available, returning neutral")
            return self._neutral_prediction(predictions)

        # If only one model available, use it directly
        if len(valid_predictions) == 1:
            model_name = 'lstm' if predictions['lstm'] is not None else 'gru'
            logger.info(f"Using single model prediction from {model_name.upper()}")
            return self._format_single_prediction(predictions, model_name)

        # Ensemble voting
        if self.voting_method == "weighted_average":
            result = self._weighted_average_voting(predictions)
        elif self.voting_method == "majority_vote":
            result = self._majority_voting(predictions)
        elif self.voting_method == "confidence_weighted":
            result = self._confidence_weighted_voting(predictions)
        else:
            logger.warning(f"Unknown voting method: {self.voting_method}, using weighted_average")
            result = self._weighted_average_voting(predictions)

        # Add predictions from individual models
        result['lstm_prediction'] = predictions['lstm']
        result['gru_prediction'] = predictions['gru']
        result['voting_method'] = self.voting_method

        # Check model agreement
        if predictions['lstm'] and predictions['gru']:
            result['model_agreement'] = predictions['lstm']['direction'] == predictions['gru']['direction']
        else:
            result['model_agreement'] = None

        # Log prediction
        if symbol:
            lstm_dir = predictions['lstm']['direction'] if predictions['lstm'] else 'N/A'
            lstm_conf = f"{predictions['lstm']['confidence']:.2%}" if predictions['lstm'] else 'N/A'
            gru_dir = predictions['gru']['direction'] if predictions['gru'] else 'N/A'
            gru_conf = f"{predictions['gru']['confidence']:.2%}" if predictions['gru'] else 'N/A'

            logger.info(
                f"{symbol} Ensemble: {result['direction'].upper()} "
                f"(confidence: {result['confidence']:.2%}) | "
                f"LSTM: {lstm_dir} ({lstm_conf}) | "
                f"GRU: {gru_dir} ({gru_conf}) | "
                f"Agreement: {result['model_agreement']}"
            )

        # Track prediction
        self.prediction_history.append({
            'symbol': symbol,
            'timestamp': pd.Timestamp.now(),
            'prediction': result
        })

        return result

    def _weighted_average_voting(self, predictions: Dict) -> Dict:
        """Combine predictions using weighted average of probabilities"""

        lstm_pred = predictions['lstm']
        gru_pred = predictions['gru']

        # Get probability arrays
        lstm_probs = np.array(lstm_pred['probabilities'])
        gru_probs = np.array(gru_pred['probabilities'])

        # Weighted average
        ensemble_probs = (lstm_probs * self.lstm_weight + gru_probs * self.gru_weight)

        # Normalize
        ensemble_probs /= ensemble_probs.sum()

        # Get direction and confidence
        predicted_class = np.argmax(ensemble_probs)
        confidence = float(ensemble_probs[predicted_class])

        direction_map = {0: 'down', 1: 'neutral', 2: 'up'}
        direction = direction_map[predicted_class]

        return {
            'direction': direction,
            'confidence': confidence,
            'probabilities': ensemble_probs.tolist()
        }

    def _majority_voting(self, predictions: Dict) -> Dict:
        """Take majority vote from model predictions"""

        lstm_pred = predictions['lstm']
        gru_pred = predictions['gru']

        # Count votes
        votes = {
            'lstm': lstm_pred['direction'],
            'gru': gru_pred['direction']
        }

        # If models agree, use their consensus
        if lstm_pred['direction'] == gru_pred['direction']:
            # Average the probabilities
            lstm_probs = np.array(lstm_pred['probabilities'])
            gru_probs = np.array(gru_pred['probabilities'])
            ensemble_probs = (lstm_probs + gru_probs) / 2

            return {
                'direction': lstm_pred['direction'],
                'confidence': float(np.max(ensemble_probs)),
                'probabilities': ensemble_probs.tolist()
            }

        # Models disagree, use confidence to break tie
        if lstm_pred['confidence'] > gru_pred['confidence']:
            winning_pred = lstm_pred
        else:
            winning_pred = gru_pred

        return {
            'direction': winning_pred['direction'],
            'confidence': winning_pred['confidence'],
            'probabilities': winning_pred['probabilities']
        }

    def _confidence_weighted_voting(self, predictions: Dict) -> Dict:
        """Weight predictions by their confidence scores"""

        lstm_pred = predictions['lstm']
        gru_pred = predictions['gru']

        # Get confidences
        lstm_conf = lstm_pred['confidence']
        gru_conf = gru_pred['confidence']

        # Normalize confidences
        total_conf = lstm_conf + gru_conf
        lstm_weight = lstm_conf / total_conf
        gru_weight = gru_conf / total_conf

        # Get probability arrays
        lstm_probs = np.array(lstm_pred['probabilities'])
        gru_probs = np.array(gru_pred['probabilities'])

        # Confidence-weighted average
        ensemble_probs = (lstm_probs * lstm_weight + gru_probs * gru_weight)

        # Normalize
        ensemble_probs /= ensemble_probs.sum()

        # Get direction and confidence
        predicted_class = np.argmax(ensemble_probs)
        confidence = float(ensemble_probs[predicted_class])

        direction_map = {0: 'down', 1: 'neutral', 2: 'up'}
        direction = direction_map[predicted_class]

        return {
            'direction': direction,
            'confidence': confidence,
            'probabilities': ensemble_probs.tolist()
        }

    def _neutral_prediction(self, predictions: Dict) -> Dict:
        """Return neutral prediction when no models available"""
        return {
            'direction': 'neutral',
            'confidence': 0.33,
            'probabilities': [0.33, 0.34, 0.33],
            'lstm_prediction': predictions.get('lstm'),
            'gru_prediction': predictions.get('gru'),
            'voting_method': self.voting_method,
            'model_agreement': None
        }

    def _format_single_prediction(self, predictions: Dict, model_name: str) -> Dict:
        """Format prediction when only one model is available"""
        pred = predictions[model_name]
        return {
            'direction': pred['direction'],
            'confidence': pred['confidence'],
            'probabilities': pred['probabilities'],
            'lstm_prediction': predictions.get('lstm'),
            'gru_prediction': predictions.get('gru'),
            'voting_method': f'single_model_{model_name}',
            'model_agreement': None
        }

    def batch_predict(
        self,
        symbols: List[str],
        data_dict: Dict[str, pd.DataFrame],
        macro_data: Optional[Dict] = None,
        options_data_dict: Optional[Dict[str, Dict]] = None
    ) -> Dict[str, Dict]:
        """
        Generate predictions for multiple symbols

        Args:
            symbols: List of stock symbols
            data_dict: Dictionary mapping symbols to DataFrames
            macro_data: Optional macro indicators (shared across symbols)
            options_data_dict: Optional dict mapping symbols to options data

        Returns:
            Dictionary mapping symbols to predictions
        """

        predictions = {}

        for symbol in symbols:
            if symbol not in data_dict:
                logger.warning(f"No data available for {symbol}, skipping")
                continue

            df = data_dict[symbol]
            options_data = options_data_dict.get(symbol) if options_data_dict else None

            pred = self.predict(df, macro_data, options_data, symbol)
            predictions[symbol] = pred

        return predictions

    def get_top_predictions(
        self,
        predictions: Dict[str, Dict],
        top_n: int = 10,
        min_confidence: float = 0.5,
        direction_filter: Optional[str] = None
    ) -> List[Tuple[str, Dict]]:
        """
        Get top N predictions by confidence

        Args:
            predictions: Dictionary mapping symbols to predictions
            top_n: Number of top predictions to return
            min_confidence: Minimum confidence threshold
            direction_filter: Optional filter ('up', 'down', or None for all)

        Returns:
            List of (symbol, prediction) tuples sorted by confidence
        """

        # Filter by confidence and direction
        filtered = []
        for symbol, pred in predictions.items():
            if pred['confidence'] < min_confidence:
                continue

            if direction_filter and pred['direction'] != direction_filter:
                continue

            filtered.append((symbol, pred))

        # Sort by confidence
        filtered.sort(key=lambda x: x[1]['confidence'], reverse=True)

        return filtered[:top_n]

    def update_performance(self, symbol: str, actual_direction: str):
        """
        Update performance statistics with actual outcome

        Args:
            symbol: Stock symbol
            actual_direction: Actual price movement ('up', 'down', 'neutral')
        """

        # Find most recent prediction for this symbol
        for i in range(len(self.prediction_history) - 1, -1, -1):
            if self.prediction_history[i]['symbol'] == symbol:
                pred = self.prediction_history[i]['prediction']

                # Update ensemble accuracy
                ensemble_correct = pred['direction'] == actual_direction
                self.performance_stats['ensemble_accuracy'].append(int(ensemble_correct))

                # Update individual model accuracies
                if pred.get('lstm_prediction'):
                    lstm_correct = pred['lstm_prediction']['direction'] == actual_direction
                    self.performance_stats['lstm_accuracy'].append(int(lstm_correct))

                if pred.get('gru_prediction'):
                    gru_correct = pred['gru_prediction']['direction'] == actual_direction
                    self.performance_stats['gru_accuracy'].append(int(gru_correct))

                break

    def get_performance_summary(self) -> Dict:
        """Get summary of model performance"""

        summary = {}

        for model_name, accuracies in self.performance_stats.items():
            if accuracies:
                summary[model_name] = {
                    'accuracy': np.mean(accuracies),
                    'total_predictions': len(accuracies),
                    'correct': sum(accuracies),
                    'incorrect': len(accuracies) - sum(accuracies)
                }
            else:
                summary[model_name] = {
                    'accuracy': 0.0,
                    'total_predictions': 0,
                    'correct': 0,
                    'incorrect': 0
                }

        return summary

    def save_performance_stats(self, filepath: str = "models/ensemble_performance.json"):
        """Save performance statistics to file"""
        stats = self.get_performance_summary()

        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Performance stats saved to {filepath}")

    def __repr__(self):
        return (
            f"ModelEnsemble(\n"
            f"  LSTM: {'✓ Loaded' if self.lstm_loaded else '✗ Not loaded'}\n"
            f"  GRU-Transformer: {'✓ Loaded' if self.gru_loaded else '✗ Not loaded'}\n"
            f"  Weights: LSTM={self.lstm_weight:.2f}, GRU={self.gru_weight:.2f}\n"
            f"  Voting: {self.voting_method}\n"
            f"  Predictions: {len(self.prediction_history)}\n"
            f")"
        )


__all__ = ['ModelEnsemble']
