"""
AUTOMATED DAILY TRADING SIGNALS
Runs every morning, generates buy/sell signals, sends you email/SMS
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ai.model_ensemble import ModelEnsemble
import yfinance as yf

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('daily_signals.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DailyTradingSignals:
    """Automated daily trading signal generator"""

    def __init__(self, min_confidence=0.60, require_agreement=True):
        self.min_confidence = min_confidence
        self.require_agreement = require_agreement

        logger.info("="*70)
        logger.info("DAILY TRADING SIGNALS - AUTOMATED SYSTEM")
        logger.info("="*70)

        # Initialize ensemble
        self.ensemble = ModelEnsemble(
            lstm_model_path="models/lstm_best.pth",
            gru_model_path="models/gru_transformer_10h.pth",
            voting_method="confidence_weighted"
        )

        # Top stocks to monitor
        self.watchlist = [
            # Tech giants
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
            # Canadian blue chips
            'SHOP.TO', 'TD.TO', 'RY.TO', 'ENB.TO', 'CNQ.TO',
            # ETFs
            'SPY', 'QQQ', 'DIA', 'IWM',
            # Others
            'AMD', 'NFLX', 'CRM', 'ORCL'
        ]

    def fetch_data(self, symbol, period='1y'):
        """Fetch market data"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval='1d')

            if df.empty:
                return None

            df.columns = [col.lower() for col in df.columns]
            return df
        except Exception as e:
            logger.error(f"Failed to fetch {symbol}: {e}")
            return None

    def generate_signals(self):
        """Generate trading signals for all watchlist stocks"""

        logger.info(f"\n{'='*70}")
        logger.info(f"GENERATING SIGNALS FOR {len(self.watchlist)} STOCKS")
        logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        logger.info(f"Min Confidence: {self.min_confidence:.0%}")
        logger.info(f"Require Agreement: {self.require_agreement}")
        logger.info(f"{'='*70}\n")

        signals = {
            'BUY': [],
            'SELL': [],
            'HOLD': []
        }

        for symbol in self.watchlist:
            df = self.fetch_data(symbol)

            if df is None or len(df) < 100:
                continue

            # Generate prediction
            pred = self.ensemble.predict(df, symbol=symbol)

            # Get current price
            current_price = df['close'].iloc[-1]

            # Determine action
            direction = pred['direction']
            confidence = pred['confidence']
            agreement = pred.get('model_agreement', False)

            signal = {
                'symbol': symbol,
                'price': current_price,
                'direction': direction,
                'confidence': confidence,
                'agreement': agreement,
                'lstm': pred['lstm_prediction'],
                'gru': pred['gru_prediction']
            }

            # Filter by criteria
            if confidence < self.min_confidence:
                signals['HOLD'].append(signal)
            elif self.require_agreement and not agreement:
                signals['HOLD'].append(signal)
            elif direction == 'up':
                signals['BUY'].append(signal)
            elif direction == 'down':
                signals['SELL'].append(signal)
            else:
                signals['HOLD'].append(signal)

        # Sort by confidence
        for action in signals:
            signals[action].sort(key=lambda x: x['confidence'], reverse=True)

        return signals

    def format_report(self, signals):
        """Format signals into readable report"""

        report = []
        report.append("="*70)
        report.append("DAILY TRADING SIGNALS REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report.append("="*70)
        report.append("")

        # BUY signals
        if signals['BUY']:
            report.append("BUY SIGNALS (High Confidence - Both Models Agree)")
            report.append("-"*70)
            for s in signals['BUY']:
                report.append(
                    f"  {s['symbol']:8s} @ ${s['price']:8.2f} | "
                    f"Confidence: {s['confidence']:5.1%} | "
                    f"Models Agree: {'YES' if s['agreement'] else 'NO'}"
                )
                report.append(
                    f"           LSTM: {s['lstm']['direction']:7s} ({s['lstm']['confidence']:.1%}) | "
                    f"GRU: {s['gru']['direction']:7s} ({s['gru']['confidence']:.1%})"
                )
                report.append("")
        else:
            report.append("BUY SIGNALS: None above threshold")
            report.append("")

        # SELL signals
        if signals['SELL']:
            report.append("SELL SIGNALS (High Confidence - Both Models Agree)")
            report.append("-"*70)
            for s in signals['SELL']:
                report.append(
                    f"  {s['symbol']:8s} @ ${s['price']:8.2f} | "
                    f"Confidence: {s['confidence']:5.1%} | "
                    f"Models Agree: {'YES' if s['agreement'] else 'NO'}"
                )
                report.append(
                    f"           LSTM: {s['lstm']['direction']:7s} ({s['lstm']['confidence']:.1%}) | "
                    f"GRU: {s['gru']['direction']:7s} ({s['gru']['confidence']:.1%})"
                )
                report.append("")
        else:
            report.append("SELL SIGNALS: None above threshold")
            report.append("")

        # Summary
        report.append("="*70)
        report.append("SUMMARY")
        report.append("="*70)
        report.append(f"Total stocks analyzed: {len(self.watchlist)}")
        report.append(f"BUY signals:  {len(signals['BUY'])}")
        report.append(f"SELL signals: {len(signals['SELL'])}")
        report.append(f"HOLD signals: {len(signals['HOLD'])}")
        report.append("")

        # Top 3 recommendations
        if signals['BUY']:
            report.append("TOP 3 BUY RECOMMENDATIONS:")
            for i, s in enumerate(signals['BUY'][:3], 1):
                report.append(f"  {i}. {s['symbol']} - {s['confidence']:.1%} confidence")

        report.append("")
        report.append("="*70)
        report.append("DISCLAIMER: AI predictions, not financial advice. Trade at your own risk.")
        report.append("="*70)

        return "\n".join(report)

    def save_report(self, signals, report_text):
        """Save report to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save JSON
        json_file = f"signals/daily_signals_{timestamp}.json"
        Path("signals").mkdir(exist_ok=True)

        with open(json_file, 'w') as f:
            json.dump(signals, f, indent=2, default=str)

        # Save text report
        txt_file = f"signals/daily_report_{timestamp}.txt"
        with open(txt_file, 'w') as f:
            f.write(report_text)

        logger.info(f"Report saved: {txt_file}")
        logger.info(f"Signals saved: {json_file}")

    def send_email(self, report_text, to_email="your-email@example.com"):
        """Send email with signals (configure SMTP settings)"""

        # TODO: Configure your email settings
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        from_email = "your-email@gmail.com"
        password = "your-app-password"  # Use app password, not regular password

        try:
            msg = MIMEMultipart()
            msg['From'] = from_email
            msg['To'] = to_email
            msg['Subject'] = f"Daily Trading Signals - {datetime.now().strftime('%Y-%m-%d')}"

            msg.attach(MIMEText(report_text, 'plain'))

            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(from_email, password)
            server.send_message(msg)
            server.quit()

            logger.info(f"Email sent to {to_email}")
        except Exception as e:
            logger.error(f"Failed to send email: {e}")

    def run(self):
        """Main execution"""

        # Generate signals
        signals = self.generate_signals()

        # Format report
        report_text = self.format_report(signals)

        # Print to console
        print(report_text)

        # Save to file
        self.save_report(signals, report_text)

        # Send email (uncomment and configure)
        # self.send_email(report_text, "your-email@example.com")

        logger.info("\nDaily signal generation complete!")

        return signals


def main():
    """Run daily signals"""

    # Create signal generator
    signal_generator = DailyTradingSignals(
        min_confidence=0.60,      # Only signals > 60% confidence
        require_agreement=True    # Both models must agree
    )

    # Generate signals
    signals = signal_generator.run()

    return signals


if __name__ == "__main__":
    main()
