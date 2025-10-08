"""
Start Complete Demo System
Dashboard + AI Trading Engine
"""

import os
import sys
import subprocess
import time

sys.path.append(os.path.dirname(__file__))

print("=" * 80)
print("🎮 STARTING COMPLETE DEMO TRADING SYSTEM")
print("=" * 80)
print("📊 Components:")
print("   1. Demo Trading Dashboard (http://localhost:8051)")
print("   2. AI Trading Engine (Background)")
print("=" * 80)
print()

# Start dashboard
print("🚀 Starting demo dashboard...")
dashboard_process = subprocess.Popen(
    [sys.executable, "src/dashboard/demo_dashboard.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

time.sleep(3)

print("✅ Demo system started!")
print()
print("=" * 80)
print("🌐 Access Dashboard: http://localhost:8051")
print("=" * 80)
print("📊 Features:")
print("   • Real-time Canadian market data")
print("   • AI-controlled trading (5 strategies)")
print("   • Live portfolio tracking")
print("   • Trade history")
print("   • P&L monitoring")
print("=" * 80)
print("💰 Starting Capital: $50,000 CAD")
print("🤖 AI will trade automatically")
print("⏱️  Demo runs for 7 days or until stopped")
print("=" * 80)
print()
print("Press Ctrl+C to stop demo system")
print()

try:
    # Keep running
    dashboard_process.wait()
except KeyboardInterrupt:
    print("\n⏸️  Stopping demo system...")
    dashboard_process.terminate()
    print("✅ Demo system stopped")

