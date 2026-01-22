#!/usr/bin/env python3
"""
Dashboard Launcher - Properly sets Python path and runs the dashboard
"""
import sys
import os

# Set the project root
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Change to project directory
os.chdir(project_root)

print(f"Project root: {project_root}")
print(f"Python path: {sys.path[:3]}")

# Now import and run the dashboard
print("\nStarting Risk Dashboard...")
print("Access at: http://localhost:8053")
print("Press Ctrl+C to stop\n")

# Import the app from risk_dashboard
from Final_dashboards.risk_dashboard import app

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8053)
