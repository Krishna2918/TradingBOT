"""
Test the Model Ensemble with trained models
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ai.model_ensemble import ModelEnsemble
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def create_sample_data(n_rows: int = 500) -> pd.DataFrame:
    """Create sample OHLCV data for testing"""

    np.random.seed(42)

    # Generate realistic price data
    base_price = 100.0
    prices = [base_price]

    for _ in range(n_rows - 1):
        change = np.random.normal(0, 0.02)  # 2% std dev
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)

    prices = np.array(prices)

    # Create OHLCV data
    df = pd.DataFrame({
        'open': prices * np.random.uniform(0.98, 1.02, n_rows),
        'high': prices * np.random.uniform(1.00, 1.05, n_rows),
        'low': prices * np.random.uniform(0.95, 1.00, n_rows),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, n_rows)
    })

    return df


def main():
    """Test the ensemble"""

    logger.info("="*60)
    logger.info("TESTING MODEL ENSEMBLE")
    logger.info("="*60)

    # Initialize ensemble
    logger.info("\n1. Initializing ensemble...")
    ensemble = ModelEnsemble(
        lstm_model_path="models/lstm_best.pth",
        gru_model_path="models/gru_transformer_10h.pth",
        lstm_weight=0.4,
        gru_weight=0.6,
        voting_method="confidence_weighted"
    )

    logger.info(f"\n{ensemble}")

    # Create sample data
    logger.info("\n2. Creating sample market data...")
    df = create_sample_data(n_rows=500)
    logger.info(f"Sample data shape: {df.shape}")
    logger.info(f"Last 5 rows:\n{df.tail()}")

    # Test single prediction
    logger.info("\n3. Testing single prediction...")
    prediction = ensemble.predict(df, symbol="TEST")

    logger.info("\n" + "="*60)
    logger.info("ENSEMBLE PREDICTION RESULT:")
    logger.info("="*60)
    logger.info(f"Direction: {prediction['direction'].upper()}")
    logger.info(f"Confidence: {prediction['confidence']:.2%}")
    logger.info(f"Probabilities: {prediction['probabilities']}")
    logger.info(f"Voting Method: {prediction['voting_method']}")
    logger.info(f"Model Agreement: {prediction['model_agreement']}")

    if prediction.get('lstm_prediction'):
        logger.info(f"\nLSTM Prediction:")
        logger.info(f"  Direction: {prediction['lstm_prediction']['direction']}")
        logger.info(f"  Confidence: {prediction['lstm_prediction']['confidence']:.2%}")

    if prediction.get('gru_prediction'):
        logger.info(f"\nGRU Prediction:")
        logger.info(f"  Direction: {prediction['gru_prediction']['direction']}")
        logger.info(f"  Confidence: {prediction['gru_prediction']['confidence']:.2%}")

    # Test batch prediction
    logger.info("\n4. Testing batch predictions...")
    symbols = ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL']
    data_dict = {symbol: create_sample_data() for symbol in symbols}

    batch_predictions = ensemble.batch_predict(symbols, data_dict)

    logger.info(f"\nBatch prediction results ({len(batch_predictions)} symbols):")
    for symbol, pred in batch_predictions.items():
        logger.info(
            f"  {symbol}: {pred['direction'].upper()} "
            f"(conf: {pred['confidence']:.2%}, "
            f"agreement: {pred['model_agreement']})"
        )

    # Get top predictions
    logger.info("\n5. Getting top predictions...")
    top_up = ensemble.get_top_predictions(
        batch_predictions,
        top_n=3,
        min_confidence=0.4,
        direction_filter='up'
    )

    logger.info("\nTop 3 BUY signals:")
    for symbol, pred in top_up:
        logger.info(f"  {symbol}: {pred['confidence']:.2%} confidence")

    # Test different voting methods
    logger.info("\n6. Testing different voting methods...")
    voting_methods = ['weighted_average', 'majority_vote', 'confidence_weighted']

    for method in voting_methods:
        ensemble.voting_method = method
        pred = ensemble.predict(df, symbol="TEST")
        logger.info(
            f"  {method:25s}: {pred['direction']:7s} "
            f"(confidence: {pred['confidence']:.2%})"
        )

    # Performance summary
    logger.info("\n7. Performance summary...")
    performance = ensemble.get_performance_summary()
    logger.info(f"\n{performance}")

    logger.info("\n" + "="*60)
    logger.info("ENSEMBLE TEST COMPLETED SUCCESSFULLY!")
    logger.info("="*60)

    return ensemble, prediction


if __name__ == "__main__":
    ensemble, prediction = main()
