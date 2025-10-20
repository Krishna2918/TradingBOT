#!/usr/bin/env python3
"""
Start Modern Trading Dashboard - Default Implementation
This script launches the modern dashboard as the default interface
"""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path

def main():
    """Start the modern trading dashboard"""
    print("🚀 Starting Modern AI Trading Dashboard...")
    print("=" * 60)
    
    # Check if the modern dashboard file exists
    dashboard_file = Path("modern_trading_dashboard.py")
    if not dashboard_file.exists():
        print("❌ Error: modern_trading_dashboard.py not found!")
        print("Please ensure the file exists in the current directory.")
        return 1
    
    try:
        # Start the dashboard
        print("📊 Launching Modern Dashboard on port 8052...")
        print("🌐 Dashboard will open at: http://localhost:8052")
        print("⏹️  Press Ctrl+C to stop the server")
        print("=" * 60)
        
        # Start the dashboard process
        process = subprocess.Popen([
            sys.executable, "modern_trading_dashboard.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Wait a moment for the server to start
        time.sleep(3)
        
        # Open the browser
        try:
            webbrowser.open("http://localhost:8052")
            print("✅ Browser opened successfully!")
        except Exception as e:
            print(f"⚠️  Could not open browser automatically: {e}")
            print("   Please manually open: http://localhost:8052")
        
        # Wait for the process to complete
        try:
            stdout, stderr = process.communicate()
            if stdout:
                print("STDOUT:", stdout)
            if stderr:
                print("STDERR:", stderr)
        except KeyboardInterrupt:
            print("\n🛑 Shutting down dashboard...")
            process.terminate()
            process.wait()
            print("✅ Dashboard stopped successfully!")
            
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
