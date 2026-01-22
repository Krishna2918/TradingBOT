#!/usr/bin/env python3
"""
Paper Trading Dashboard Launcher
Starts the comprehensive trading dashboard with demo trading engine
"""
import sys
import os

# Set the project root
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Change to project directory
os.chdir(project_root)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

print(f"Project root: {project_root}")
print(f"Python path: {sys.path[:3]}")

# Now import and run the comprehensive dashboard
print("\n" + "="*50)
print("    PAPER TRADING DASHBOARD")
print("="*50)
print(f"\nMode: DEMO (Paper Trading)")
print(f"Capital: $100,000 CAD")
print(f"Market: Canadian (TSX)")
print("\nStarting dashboard...")
print("Access at: http://localhost:8052")
print("Press Ctrl+C to stop\n")

# Import the app from comprehensive_dashboard
from Final_dashboards.comprehensive_dashboard import app

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8052)
