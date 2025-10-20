#!/usr/bin/env python3
"""
Launch the REAL Trading Dashboard
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    print("🚀 Starting REAL AI Trading Dashboard...")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("real_trading_dashboard.py").exists():
        print("❌ real_trading_dashboard.py not found!")
        print("Please run this from the TradingBOT directory")
        return
    
    try:
        # Start the dashboard
        subprocess.run([sys.executable, "real_trading_dashboard.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")

if __name__ == "__main__":
    main()
