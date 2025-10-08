"""
Start Demo Trading Mode - 1 Week AI Trading Trial
Real Canadian market data + Simulated AI trading
"""

import os
import sys
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from src.demo.demo_trading_engine import DemoTradingEngine

# Setup logging
os.makedirs('logs', exist_ok=True)
log_file = f"logs/demo_trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    print("=" * 80)
    print("🎮 DEMO TRADING MODE - 1 WEEK AI TRADING TRIAL")
    print("=" * 80)
    print("📊 Features:")
    print("   • Real Canadian market data (TSX/TSXV)")
    print("   • AI-controlled trading (5 strategies)")
    print("   • Simulated order execution")
    print("   • Risk management & stop loss/take profit")
    print("   • Real-time P&L tracking")
    print("=" * 80)
    print("💰 Starting Capital: $50,000 CAD")
    print("⏱️  Duration: 7 days")
    print("🤖 AI Strategies: Momentum, News-Vol, Gamma/OI, Arbitrage, AI/ML")
    print("=" * 80)
    print("🌐 Dashboard: Starting on http://localhost:8051...")
    print("📝 Logs: " + log_file)
    print("=" * 80)
    print()
    
    try:
        # Initialize demo engine
        engine = DemoTradingEngine()
        
        print("✅ Demo engine initialized")
        print("🚀 Starting AI trading...")
        print("   Press Ctrl+C to stop\n")
        
        # Run demo (will run continuously until stopped or demo period ends)
        engine.run_demo()
        
    except KeyboardInterrupt:
        print("\n⏸️  Demo stopped by user")
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        print(f"\n❌ Error occurred: {e}")
    
    print("\n" + "=" * 80)
    print("Demo trading session ended")
    print("=" * 80)

