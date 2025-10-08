#!/usr/bin/env python3
"""
Test Setup Script
Verifies that all components are working correctly
"""

import sys
import os
import subprocess
import importlib
import redis
import yaml
from pathlib import Path

def test_python_environment():
    """Test Python environment and dependencies"""
    print("🐍 Testing Python environment...")
    
    # Check Python version
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        print("✅ Python 3.11+ detected")
    else:
        print(f"❌ Python 3.11+ required, found {version.major}.{version.minor}")
        return False
    
    # Test key dependencies
    dependencies = [
        'pandas', 'numpy', 'redis', 'influxdb_client',
        'yaml', 'requests', 'websocket_client'
    ]
    
    for dep in dependencies:
        try:
            importlib.import_module(dep)
            print(f"✅ {dep} imported successfully")
        except ImportError as e:
            print(f"❌ {dep} import failed: {e}")
            return False
    
    return True

def test_redis_connection():
    """Test Redis connection"""
    print("🔴 Testing Redis connection...")
    
    try:
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        print("✅ Redis connection successful")
        return True
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return False

def test_influxdb_connection():
    """Test InfluxDB connection"""
    print("📊 Testing InfluxDB connection...")
    
    try:
        from influxdb_client import InfluxDBClient
        
        client = InfluxDBClient(
            url="http://localhost:8086",
            token="MOCK_TOKEN_123",
            org="trading_org"
        )
        
        # Test basic connectivity
        health = client.health()
        print("✅ InfluxDB connection successful")
        return True
    except Exception as e:
        print(f"❌ InfluxDB connection failed: {e}")
        return False

def test_project_structure():
    """Test project directory structure"""
    print("📁 Testing project structure...")
    
    required_dirs = [
        'src', 'tests', 'config', 'scripts', 'docs', 'logs', 'data',
        'src/strategies', 'src/risk_management', 'src/data_pipeline',
        'src/execution', 'src/monitoring'
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path} exists")
        else:
            print(f"❌ {dir_path} missing")
            return False
    
    return True

def test_configuration_files():
    """Test configuration files"""
    print("⚙️ Testing configuration files...")
    
    config_files = [
        'config/trading_config.yaml',
        'config/risk_config.yaml',
        'config/broker_config.yaml'
    ]
    
    for config_file in config_files:
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    yaml.safe_load(f)
                print(f"✅ {config_file} is valid YAML")
            except Exception as e:
                print(f"❌ {config_file} YAML parsing failed: {e}")
                return False
        else:
            print(f"❌ {config_file} missing")
            return False
    
    return True

def test_risk_modules():
    """Test risk management modules"""
    print("🛡️ Testing risk management modules...")
    
    try:
        # Add src to path
        sys.path.append('src')
        
        from risk_management.capital_allocation import CapitalAllocator
        from risk_management.leverage_governance import LeverageGovernor
        from risk_management.kill_switches import KillSwitchManager
        
        print("✅ Risk management modules imported successfully")
        return True
    except Exception as e:
        print(f"❌ Risk management modules import failed: {e}")
        return False

def run_unit_tests():
    """Run unit tests"""
    print("🧪 Running unit tests...")
    
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 'tests/unit/', '-v'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Unit tests passed")
            return True
        else:
            print(f"❌ Unit tests failed: {result.stdout}")
            return False
    except Exception as e:
        print(f"❌ Unit test execution failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Trading Bot Setup Test Suite")
    print("=" * 50)
    
    tests = [
        test_python_environment,
        test_redis_connection,
        test_influxdb_connection,
        test_project_structure,
        test_configuration_files,
        test_risk_modules,
        run_unit_tests
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Setup is complete.")
        return 0
    else:
        print("⚠️ Some tests failed. Please review and fix issues.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

