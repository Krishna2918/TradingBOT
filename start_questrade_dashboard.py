"""
Start Trading Bot Dashboard with Real Questrade Data
"""
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(__file__))

# Direct import to avoid __init__.py conflicts
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'dashboard'))
from questrade_dashboard import app

if __name__ == '__main__':
    print("=" * 70)
    print("🚀 Trading Bot Dashboard Starting...")
    print("=" * 70)
    print("📊 Dashboard Features:")
    print("   • Real-time Questrade Portfolio Data")
    print("   • Live Account Balances")
    print("   • Position Tracking")
    print("   • P&L Monitoring")
    print("=" * 70)
    print("🔐 Questrade API Configuration:")
    print("   • Mode: READ-ONLY (Retail Account)")
    print("   • Data: Market data & account info")
    print("   • Trading: Manual only (compliance)")
    print("=" * 70)
    print("🌐 Access URL: http://localhost:8050")
    print("🎨 UI Style: Groww-inspired design")
    print("=" * 70)
    print("⚡ Dashboard is now running...")
    print("   Press Ctrl+C to stop")
    print("=" * 70)
    print()

    app.run(debug=True, host='0.0.0.0', port=8050)

