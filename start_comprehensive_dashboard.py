"""
Start Comprehensive Trading Dashboard

Launches the full-featured dashboard with:
- Market data analysis
- Technical analysis
- Options analysis
- Macro analysis
- Risk management
- Backtesting
- AI analysis
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.dashboard.comprehensive_dashboard import app

if __name__ == '__main__':
    print("=" * 80)
    print("🚀 Starting Comprehensive Trading Dashboard")
    print("=" * 80)
    print()
    print("📊 Dashboard Features:")
    print("   • Market Data Analysis")
    print("   • Technical Analysis")
    print("   • Options Data")
    print("   • Macro Analysis")
    print("   • Risk Management")
    print("   • Backtesting")
    print("   • AI Analysis")
    print()
    print("🌐 Dashboard will be available at: http://localhost:8051")
    print()
    print("Press Ctrl+C to stop the dashboard")
    print("=" * 80)
    print()
    
    app.run_server(debug=False, port=8051, host='0.0.0.0')
