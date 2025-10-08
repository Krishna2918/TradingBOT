#!/usr/bin/env python3
"""
Test Runner Script
Easy way to run all tests and verify setup
"""

import sys
import os
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """Run a command and return success status"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed:")
            print(f"   Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} failed with exception: {e}")
        return False

def main():
    """Main test runner function"""
    print("🧪 Trading Bot Test Runner")
    print("=" * 50)
    
    # Change to project directory
    project_dir = Path(__file__).parent.parent
    os.chdir(project_dir)
    
    # Check if virtual environment exists
    if not Path("venv").exists():
        print("❌ Virtual environment not found. Please run setup first.")
        return 1
    
    # Activate virtual environment
    if os.name == 'nt':  # Windows
        python_path = "venv\\Scripts\\python.exe"
    else:  # Unix/Linux
        python_path = "venv/bin/python"
    
    if not Path(python_path).exists():
        print(f"❌ Python not found at {python_path}")
        return 1
    
    print(f"🐍 Using Python: {python_path}")
    
    # Test setup
    tests = [
        (f"{python_path} scripts/test_setup.py", "Setup Verification"),
        (f"{python_path} -m pytest tests/unit/ -v", "Unit Tests"),
        (f"{python_path} src/main.py", "Main Application Test")
    ]
    
    passed = 0
    total = len(tests)
    
    for cmd, description in tests:
        if run_command(cmd, description):
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Setup is complete and working.")
        print("\n📋 Next steps:")
        print("1. Review configuration files in config/")
        print("2. Update broker API credentials")
        print("3. Start trading bot: python src/main.py")
        return 0
    else:
        print("⚠️ Some tests failed. Please review and fix issues.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

