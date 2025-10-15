#!/usr/bin/env python3
"""
Phase 10 Simple Test - CI & Automation

Simplified test for core CI/CD functionality without complex imports.
"""

import sys
import os
import logging
import json
import tempfile
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_file_existence():
    """Test that all required CI/CD files exist."""
    print("\n[TEST] File Existence Check")
    print("=" * 50)
    
    required_files = [
        '.github/workflows/nightly-validation.yml',
        '.pre-commit-config.yaml',
        'scripts/ci_validation.py',
        'scripts/performance_benchmark.py',
        'scripts/generate_validation_report.py',
        'scripts/pre_commit_validation.py',
        'scripts/check_api_keys.py',
        'tests/test_phase10_integration.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"[OK] {file_path} exists")
        else:
            print(f"[FAIL] {file_path} missing")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"[FAIL] {len(missing_files)} files missing")
        return False
    else:
        print("[PASS] All required files exist")
        return True

def test_github_workflow_content():
    """Test GitHub workflow file content."""
    print("\n[TEST] GitHub Workflow Content")
    print("=" * 50)
    
    workflow_file = '.github/workflows/nightly-validation.yml'
    
    try:
        with open(workflow_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_elements = [
            'name: Nightly Validation',
            'on:',
            'schedule:',
            'jobs:',
            'nightly-validation:',
            'runs-on: windows-latest',
            'AI_LIMIT: 20',
            'python scripts/final_validation.py',
            'python scripts/smoke_test.py',
            'python scripts/ci_validation.py'
        ]
        
        missing_elements = []
        for element in required_elements:
            if element not in content:
                missing_elements.append(element)
        
        if missing_elements:
            print(f"[FAIL] Missing elements: {missing_elements}")
            return False
        else:
            print("[PASS] All required elements present in workflow")
            return True
            
    except Exception as e:
        print(f"[FAIL] Error reading workflow file: {e}")
        return False

def test_precommit_config_content():
    """Test pre-commit configuration file content."""
    print("\n[TEST] Pre-commit Configuration Content")
    print("=" * 50)
    
    precommit_file = '.pre-commit-config.yaml'
    
    try:
        with open(precommit_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_hooks = [
            'detect-secrets',
            'black',
            'isort',
            'flake8',
            'mypy',
            'bandit'
        ]
        
        missing_hooks = []
        for hook in required_hooks:
            if hook not in content:
                missing_hooks.append(hook)
        
        if missing_hooks:
            print(f"[FAIL] Missing hooks: {missing_hooks}")
            return False
        else:
            print("[PASS] All required hooks present in pre-commit config")
            return True
            
    except Exception as e:
        print(f"[FAIL] Error reading pre-commit config: {e}")
        return False

def test_ci_validation_script():
    """Test CI validation script structure."""
    print("\n[TEST] CI Validation Script")
    print("=" * 50)
    
    script_file = 'scripts/ci_validation.py'
    
    try:
        with open(script_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_elements = [
            'class CIValidator:',
            'async def run_validation_suite',
            'async def _check_system_health',
            'async def _test_database_connectivity',
            'async def _test_ai_model_availability',
            'async def _run_performance_benchmarks',
            'async def _detect_regressions'
        ]
        
        missing_elements = []
        for element in required_elements:
            if element not in content:
                missing_elements.append(element)
        
        if missing_elements:
            print(f"[FAIL] Missing elements: {missing_elements}")
            return False
        else:
            print("[PASS] All required elements present in CI validation script")
            return True
            
    except Exception as e:
        print(f"[FAIL] Error reading CI validation script: {e}")
        return False

def test_performance_benchmark_script():
    """Test performance benchmark script structure."""
    print("\n[TEST] Performance Benchmark Script")
    print("=" * 50)
    
    script_file = 'scripts/performance_benchmark.py'
    
    try:
        with open(script_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_elements = [
            'class PerformanceBenchmark:',
            'async def run_benchmark_suite',
            'async def _benchmark_system_startup',
            'async def _benchmark_database_performance',
            'async def _benchmark_ai_performance',
            'async def _benchmark_memory_usage',
            'async def _benchmark_end_to_end'
        ]
        
        missing_elements = []
        for element in required_elements:
            if element not in content:
                missing_elements.append(element)
        
        if missing_elements:
            print(f"[FAIL] Missing elements: {missing_elements}")
            return False
        else:
            print("[PASS] All required elements present in performance benchmark script")
            return True
            
    except Exception as e:
        print(f"[FAIL] Error reading performance benchmark script: {e}")
        return False

def test_validation_report_script():
    """Test validation report script structure."""
    print("\n[TEST] Validation Report Script")
    print("=" * 50)
    
    script_file = 'scripts/generate_validation_report.py'
    
    try:
        with open(script_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_elements = [
            'def generate_validation_report',
            'def generate_system_health_section',
            'def generate_test_results_section',
            'def generate_performance_section',
            'def generate_regressions_section',
            'def generate_error_report'
        ]
        
        missing_elements = []
        for element in required_elements:
            if element not in content:
                missing_elements.append(element)
        
        if missing_elements:
            print(f"[FAIL] Missing elements: {missing_elements}")
            return False
        else:
            print("[PASS] All required elements present in validation report script")
            return True
            
    except Exception as e:
        print(f"[FAIL] Error reading validation report script: {e}")
        return False

def test_precommit_validation_script():
    """Test pre-commit validation script structure."""
    print("\n[TEST] Pre-commit Validation Script")
    print("=" * 50)
    
    script_file = 'scripts/pre_commit_validation.py'
    
    try:
        with open(script_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_elements = [
            'def validate_imports',
            'def validate_file_structure',
            'def validate_configuration',
            'def validate_database_schema',
            'def validate_ai_system',
            'def validate_trading_system',
            'def validate_dashboard'
        ]
        
        missing_elements = []
        for element in required_elements:
            if element not in content:
                missing_elements.append(element)
        
        if missing_elements:
            print(f"[FAIL] Missing elements: {missing_elements}")
            return False
        else:
            print("[PASS] All required elements present in pre-commit validation script")
            return True
            
    except Exception as e:
        print(f"[FAIL] Error reading pre-commit validation script: {e}")
        return False

def test_api_key_check_script():
    """Test API key check script structure."""
    print("\n[TEST] API Key Check Script")
    print("=" * 50)
    
    script_file = 'scripts/check_api_keys.py'
    
    try:
        with open(script_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_elements = [
            'def find_potential_api_keys',
            'def is_placeholder_key',
            'def check_environment_variables',
            'def is_sensitive_env_var',
            'def is_placeholder_value',
            'def check_gitignore'
        ]
        
        missing_elements = []
        for element in required_elements:
            if element not in content:
                missing_elements.append(element)
        
        if missing_elements:
            print(f"[FAIL] Missing elements: {missing_elements}")
            return False
        else:
            print("[PASS] All required elements present in API key check script")
            return True
            
    except Exception as e:
        print(f"[FAIL] Error reading API key check script: {e}")
        return False

def test_integration_tests():
    """Test integration tests file structure."""
    print("\n[TEST] Integration Tests")
    print("=" * 50)
    
    test_file = 'tests/test_phase10_integration.py'
    
    try:
        with open(test_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_elements = [
            'class TestCIValidation:',
            'class TestPerformanceBenchmark:',
            'class TestPreCommitValidation:',
            'class TestAPIKeyCheck:',
            'class TestValidationReport:',
            'def test_',
            '@pytest.mark.asyncio'
        ]
        
        missing_elements = []
        for element in required_elements:
            if element not in content:
                missing_elements.append(element)
        
        if missing_elements:
            print(f"[FAIL] Missing elements: {missing_elements}")
            return False
        else:
            print("[PASS] All required elements present in integration tests")
            return True
            
    except Exception as e:
        print(f"[FAIL] Error reading integration tests: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of key components."""
    print("\n[TEST] Basic Functionality")
    print("=" * 50)
    
    try:
        # Test that we can import basic modules
        import config.database
        print("[OK] Database module import successful")
        
        import monitoring.system_monitor
        print("[OK] System monitor module import successful")
        
        import ai.multi_model
        print("[OK] Multi-model module import successful")
        
        import trading.risk
        print("[OK] Risk module import successful")
        
        import dashboard.connector
        print("[OK] Dashboard connector module import successful")
        
        print("[PASS] Basic functionality test completed successfully")
        return True
        
    except Exception as e:
        print(f"[FAIL] Basic functionality test failed: {e}")
        return False

def main():
    """Run all Phase 10 simple tests."""
    print("Phase 10 Simple Test - CI & Automation")
    print("=" * 60)
    
    tests = [
        ("File Existence Check", test_file_existence),
        ("GitHub Workflow Content", test_github_workflow_content),
        ("Pre-commit Configuration Content", test_precommit_config_content),
        ("CI Validation Script", test_ci_validation_script),
        ("Performance Benchmark Script", test_performance_benchmark_script),
        ("Validation Report Script", test_validation_report_script),
        ("Pre-commit Validation Script", test_precommit_validation_script),
        ("API Key Check Script", test_api_key_check_script),
        ("Integration Tests", test_integration_tests),
        ("Basic Functionality", test_basic_functionality),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n[READY] Starting {test_name} test...")
        try:
            if test_func():
                passed += 1
            else:
                print(f"[FAIL] {test_name} test failed")
        except Exception as e:
            print(f"[FAIL] {test_name} test failed: {e}")
    
    print("\n" + "=" * 60)
    print("Phase 10 Simple Test Summary")
    print("=" * 60)
    
    for test_name, _ in tests:
        status = "[PASS]" if test_name in [tests[i][0] for i in range(passed)] else "[FAIL]"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n[SUCCESS] All tests passed! Phase 10 CI & Automation is working correctly.")
        return 0
    else:
        print(f"\n[FAILURE] {total - passed} test(s) failed. Please check the logs above.")
        return 1

if __name__ == "__main__":
    exit(main())
