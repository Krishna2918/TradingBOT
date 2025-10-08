"""
Trading Bot Dashboard Launcher
Start the Groww-style trading dashboard
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from dashboard.app import app

if __name__ == '__main__':
    print("=" * 70)
    print("🚀 Trading Bot Dashboard Starting...")
    print("=" * 70)
    print()
    print("📊 Dashboard Features:")
    print("   • Real-time Portfolio Monitoring")
    print("   • AI Trading Signals")
    print("   • Live Trade Execution")
    print("   • Strategy Performance")
    print("   • P&L Tracking")
    print()
    print("🌐 Access URL: http://localhost:8050")
    print("🎨 UI Style: Groww-inspired design")
    print()
    print("=" * 70)
    print("⚡ Dashboard is now running...")
    print("   Press Ctrl+C to stop")
    print("=" * 70)
    print()
    
    app.run(debug=True, host='0.0.0.0', port=8050)

