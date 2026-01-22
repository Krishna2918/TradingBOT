"""
Production LSTM Model Loader
Loads the trained LSTM model from lstm_best.pth with correct architecture
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class ProductionLSTMModel(nn.Module):
    """Production LSTM model architecture (matches train_lstm_production.py)"""

    def __init__(self, input_dim: int, hidden_size: int = 256, num_layers: int = 3, dropout: float = 0.3):
        super(ProductionLSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        # Attention layer
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )

        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)  # Binary classification: up/down
        )

    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden)

        # Attention mechanism
        attention_weights = self.attention(lstm_out)  # (batch, seq, 1)
        context = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, hidden)

        # Classification
        output = self.fc(context)
        return output


class ProductionLSTMPredictor:
    """
    Production LSTM Predictor for loading and using the trained lstm_best.pth model

    Features:
    - 58 input features (from production training data)
    - Binary classification (up/down)
    - Attention mechanism
    - Sequence length: 30
    """

    def __init__(
        self,
        model_path: str = "models/lstm_best.pth",
        input_size: int = 58,
        hidden_size: int = 256,
        num_layers: int = 3,
        dropout: float = 0.3,
        sequence_length: int = 30
    ):
        self.model_path = Path(model_path)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.sequence_length = sequence_length

        self.model = None
        self.scaler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        logger.info(f"Production LSTM Predictor initialized")
        logger.info(f"  Input size: {self.input_size}")
        logger.info(f"  Hidden size: {self.hidden_size}")
        logger.info(f"  Num layers: {self.num_layers}")
        logger.info(f"  Sequence length: {self.sequence_length}")

    def load_model(self) -> bool:
        """Load the trained model and scaler"""
        try:
            if not self.model_path.exists():
                logger.warning(f"Model file not found: {self.model_path}")
                return False

            # Initialize model with correct architecture
            self.model = ProductionLSTMModel(
                input_dim=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout
            ).to(self.device)

            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)

            # Extract model state dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                epoch = checkpoint.get('epoch', 'unknown')
                val_acc = checkpoint.get('val_acc', 'unknown')
                logger.info(f"Loaded checkpoint from epoch {epoch}, val_acc: {val_acc}")
            else:
                self.model.load_state_dict(checkpoint)

            self.model.eval()

            # Load scaler
            scaler_path = str(self.model_path).replace('.pth', '_scaler.pkl')
            if Path(scaler_path).exists():
                self.scaler = joblib.load(scaler_path)
                logger.info(f"Scaler loaded from {scaler_path}")
            else:
                logger.warning(f"Scaler not found at {scaler_path}")
                from sklearn.preprocessing import StandardScaler
                self.scaler = StandardScaler()

            logger.info(f"✓ Production LSTM model loaded successfully from {self.model_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def predict(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Generate prediction from market data

        Returns:
            {
                'direction': 'up'/'down'/'neutral',
                'confidence': 0.0-1.0,
                'probabilities': [down_prob, neutral_prob, up_prob],
                'raw_output': [down_logit, up_logit]
            }
        """

        if self.model is None:
            logger.warning("Model not loaded, returning neutral prediction")
            return self._neutral_prediction()

        try:
            # Prepare features (you'll need to match the 58 features from training)
            # For now, use a placeholder - you'll need to implement proper feature engineering
            features = self._prepare_features(df)

            if features is None or len(features) < self.sequence_length:
                logger.warning(f"Insufficient data for prediction (need {self.sequence_length} rows)")
                return self._neutral_prediction()

            # Scale features
            if hasattr(self.scaler, 'mean_'):  # Check if scaler is fitted
                features_scaled = self.scaler.transform(features)
            else:
                logger.warning("Scaler not fitted, using unscaled features")
                features_scaled = features

            # Get last sequence
            sequence = features_scaled[-self.sequence_length:]
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)

            # Predict
            self.model.eval()
            with torch.no_grad():
                logits = self.model(sequence_tensor)
                probabilities_binary = torch.softmax(logits, dim=1).cpu().numpy()[0]

            # Convert binary (down/up) to 3-class (down/neutral/up)
            # Map: 0=down, 1=up from binary model
            # To: 0=down, 1=neutral, 2=up for ensemble

            down_prob = probabilities_binary[0]
            up_prob = probabilities_binary[1]

            # Create neutral zone for low confidence predictions
            neutral_threshold = 0.1
            if abs(up_prob - down_prob) < neutral_threshold:
                # Low confidence - treat as neutral
                probabilities_3class = np.array([
                    down_prob * 0.4,
                    0.2,  # Neutral
                    up_prob * 0.4
                ])
            else:
                # High confidence - minimal neutral probability
                probabilities_3class = np.array([
                    down_prob,
                    0.01,  # Small neutral
                    up_prob
                ])

            # Normalize to sum to 1.0
            probabilities_3class /= probabilities_3class.sum()

            # Get direction and confidence
            predicted_class = np.argmax(probabilities_3class)
            confidence = float(probabilities_3class[predicted_class])

            direction_map = {0: 'down', 1: 'neutral', 2: 'up'}
            direction = direction_map[predicted_class]

            return {
                'direction': direction,
                'confidence': confidence,
                'probabilities': probabilities_3class.tolist(),
                'raw_output': logits.cpu().numpy()[0].tolist()
            }

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            import traceback
            traceback.print_exc()
            return self._neutral_prediction()

    def _prepare_features(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Prepare 58 features from raw OHLCV data using REAL feature engineering
        """

        if len(df) < 200:  # Need enough data for long-period indicators (200-day SMA)
            logger.warning(f"Insufficient data: {len(df)} rows (need 200+)")
            return None

        try:
            # Import the production feature engine
            import sys
            from pathlib import Path
            project_root = Path(__file__).parent.parent.parent
            sys.path.insert(0, str(project_root))

            from real_features_production import ProductionFeatureEngine

            # Initialize feature engine
            engine = ProductionFeatureEngine()

            # Generate all 58 features
            df_with_features = engine.generate_features(df)

            # Extract features in correct order
            features = df_with_features[engine.features_58].values

            # Apply per-stock z-score normalization (matches training)
            features_normalized = engine.normalize_features(features)

            logger.info(f"Generated {features_normalized.shape} REAL features (not placeholder)")

            return features_normalized

        except Exception as e:
            logger.error(f"Feature generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _neutral_prediction(self) -> Dict:
        """Return neutral prediction when model can't predict"""
        return {
            'direction': 'neutral',
            'confidence': 0.33,
            'probabilities': [0.33, 0.34, 0.33],
            'raw_output': [0.0, 0.0]
        }


__all__ = ['ProductionLSTMPredictor', 'ProductionLSTMModel']
