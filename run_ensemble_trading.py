"""
Real-time Trading with Model Ensemble
Uses trained LSTM + GRU models to generate trading signals
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import List, Dict

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ai.model_ensemble import ModelEnsemble
import yfinance as yf

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class EnsembleTradingSystem:
    """Real-time trading system using Model Ensemble"""

    def __init__(self, voting_method: str = "confidence_weighted", min_confidence: float = 0.55):
        logger.info("="*70)
        logger.info("ENSEMBLE TRADING SYSTEM")
        logger.info("="*70)

        # Initialize ensemble
        logger.info("\nInitializing model ensemble...")
        self.ensemble = ModelEnsemble(
            lstm_model_path="models/lstm_best.pth",
            gru_model_path="models/gru_transformer_10h.pth",
            lstm_weight=0.4,
            gru_weight=0.6,
            voting_method=voting_method
        )

        self.min_confidence = min_confidence
        logger.info(f"Minimum confidence threshold: {self.min_confidence:.1%}")

        logger.info(f"\n{self.ensemble}")

    def fetch_market_data(self, symbol: str, period: str = "6mo") -> pd.DataFrame:
        """Fetch market data from Yahoo Finance"""
        try:
            logger.info(f"Fetching data for {symbol}...")
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval='1d')

            if df.empty:
                logger.warning(f"No data available for {symbol}")
                return None

            # Rename columns to lowercase
            df.columns = [col.lower() for col in df.columns]

            logger.info(f"  Fetched {len(df)} rows for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            return None

    def generate_trading_signals(self, symbols: List[str]) -> Dict[str, Dict]:
        """Generate trading signals for multiple symbols"""

        logger.info("\n" + "="*70)
        logger.info("GENERATING TRADING SIGNALS")
        logger.info("="*70)

        signals = {}

        for symbol in symbols:
            # Fetch data
            df = self.fetch_market_data(symbol)

            if df is None or len(df) < 100:
                logger.warning(f"Skipping {symbol} - insufficient data")
                continue

            # Generate prediction
            prediction = self.ensemble.predict(df, symbol=symbol)

            # Determine trading action
            direction = prediction['direction']
            confidence = prediction['confidence']

            if confidence < self.min_confidence:
                action = 'HOLD'
                reason = f"Low confidence ({confidence:.1%})"
            elif direction == 'up':
                action = 'BUY'
                reason = f"Bullish signal ({confidence:.1%})"
            elif direction == 'down':
                action = 'SELL'
                reason = f"Bearish signal ({confidence:.1%})"
            else:
                action = 'HOLD'
                reason = "Neutral signal"

            # Get current price
            current_price = df['close'].iloc[-1]

            # Store signal
            signals[symbol] = {
                'action': action,
                'direction': direction,
                'confidence': confidence,
                'reason': reason,
                'current_price': current_price,
                'prediction': prediction,
                'timestamp': datetime.now()
            }

        return signals

    def display_signals(self, signals: Dict[str, Dict]):
        """Display trading signals in a formatted table"""

        logger.info("\n" + "="*70)
        logger.info("TRADING SIGNALS SUMMARY")
        logger.info("="*70)

        # Separate by action
        buys = []
        sells = []
        holds = []

        for symbol, signal in signals.items():
            if signal['action'] == 'BUY':
                buys.append((symbol, signal))
            elif signal['action'] == 'SELL':
                sells.append((symbol, signal))
            else:
                holds.append((symbol, signal))

        # Display BUY signals
        if buys:
            logger.info("\n🟢 BUY SIGNALS:")
            logger.info("-" * 70)
            buys.sort(key=lambda x: x[1]['confidence'], reverse=True)

            for symbol, signal in buys:
                pred = signal['prediction']
                logger.info(
                    f"  {symbol:6s} ${signal['current_price']:8.2f} | "
                    f"Confidence: {signal['confidence']:5.1%} | "
                    f"LSTM: {pred['lstm_prediction']['direction']:7s} ({pred['lstm_prediction']['confidence']:.1%}) | "
                    f"GRU: {pred['gru_prediction']['direction']:7s} ({pred['gru_prediction']['confidence']:.1%}) | "
                    f"Agree: {'YES' if pred['model_agreement'] else 'NO'}"
                )

        # Display SELL signals
        if sells:
            logger.info("\n🔴 SELL SIGNALS:")
            logger.info("-" * 70)
            sells.sort(key=lambda x: x[1]['confidence'], reverse=True)

            for symbol, signal in sells:
                pred = signal['prediction']
                logger.info(
                    f"  {symbol:6s} ${signal['current_price']:8.2f} | "
                    f"Confidence: {signal['confidence']:5.1%} | "
                    f"LSTM: {pred['lstm_prediction']['direction']:7s} ({pred['lstm_prediction']['confidence']:.1%}) | "
                    f"GRU: {pred['gru_prediction']['direction']:7s} ({pred['gru_prediction']['confidence']:.1%}) | "
                    f"Agree: {'YES' if pred['model_agreement'] else 'NO'}"
                )

        # Display HOLD signals (only top 5)
        if holds:
            logger.info("\n⚪ HOLD SIGNALS (Top 5):")
            logger.info("-" * 70)
            holds.sort(key=lambda x: x[1]['confidence'], reverse=True)

            for symbol, signal in holds[:5]:
                logger.info(
                    f"  {symbol:6s} ${signal['current_price']:8.2f} | "
                    f"Reason: {signal['reason']}"
                )

        # Summary statistics
        logger.info("\n" + "="*70)
        logger.info("SUMMARY")
        logger.info("="*70)
        logger.info(f"  Total symbols analyzed: {len(signals)}")
        logger.info(f"  BUY signals:  {len(buys)}")
        logger.info(f"  SELL signals: {len(sells)}")
        logger.info(f"  HOLD signals: {len(holds)}")

        # Model agreement stats
        agreements = [s['prediction']['model_agreement'] for s in signals.values() if s['prediction']['model_agreement'] is not None]
        if agreements:
            agreement_rate = sum(agreements) / len(agreements)
            logger.info(f"  Model agreement rate: {agreement_rate:.1%}")

        return buys, sells, holds


def main():
    """Main entry point"""

    # Popular stocks to analyze
    symbols = [
        # Tech giants
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',

        # Canadian stocks
        'SHOP.TO', 'TD.TO', 'RY.TO', 'ENB.TO', 'CNQ.TO',

        # Other popular stocks
        'SPY', 'QQQ', 'DIA'
    ]

    # Initialize trading system
    trading_system = EnsembleTradingSystem(
        voting_method="confidence_weighted",
        min_confidence=0.55  # Only trade on signals > 55% confidence
    )

    # Generate signals
    signals = trading_system.generate_trading_signals(symbols)

    # Display signals
    buys, sells, holds = trading_system.display_signals(signals)

    # Show top 3 BUY recommendations
    logger.info("\n" + "="*70)
    logger.info("TOP 3 BUY RECOMMENDATIONS")
    logger.info("="*70)

    if buys:
        for symbol, signal in buys[:3]:
            logger.info(f"\n{symbol}:")
            logger.info(f"  Price: ${signal['current_price']:.2f}")
            logger.info(f"  Confidence: {signal['confidence']:.1%}")
            logger.info(f"  Reason: {signal['reason']}")
            logger.info(f"  Model Agreement: {'YES' if signal['prediction']['model_agreement'] else 'NO'}")
    else:
        logger.info("  No BUY signals above confidence threshold")

    logger.info("\n" + "="*70)
    logger.info("ENSEMBLE TRADING SYSTEM COMPLETE")
    logger.info("="*70)

    return signals


if __name__ == "__main__":
    signals = main()
