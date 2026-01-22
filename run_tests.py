#!/usr/bin/env python3
"""
Test runner for the continuous data collection system.
Provides different test suites for various testing scenarios.
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path


def run_command(command, description):
    """Run a command and handle the result."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(command)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"FAILED: {description}")
        print(f"Exit code: {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False


def run_unit_tests():
    """Run unit tests."""
    command = [
        sys.executable, "-m", "pytest",
        "tests/unit/",
        "-v",
        "--tb=short",
        "--durations=10"
    ]
    return run_command(command, "Unit Tests")


def run_integration_tests():
    """Run integration tests."""
    command = [
        sys.executable, "-m", "pytest",
        "tests/integration/",
        "-v",
        "--tb=short",
        "--durations=10"
    ]
    return run_command(command, "Integration Tests")


def run_reliability_tests():
    """Run reliability and stress tests."""
    command = [
        sys.executable, "-m", "pytest",
        "tests/reliability/",
        "-v",
        "--tb=short",
        "--durations=10",
        "-s"  # Show print output for reliability tests
    ]
    return run_command(command, "Reliability and Stress Tests")


def run_performance_tests():
    """Run performance-specific tests."""
    command = [
        sys.executable, "-m", "pytest",
        "tests/integration/test_performance_load.py",
        "tests/reliability/test_long_running_stability.py",
        "-v",
        "--tb=short",
        "--durations=10",
        "-s"
    ]
    return run_command(command, "Performance Tests")


def run_quick_tests():
    """Run a quick subset of tests for development."""
    command = [
        sys.executable, "-m", "pytest",
        "tests/unit/test_data_collection_components.py",
        "tests/unit/test_queue_management.py",
        "tests/integration/test_end_to_end_workflow.py",
        "-v",
        "--tb=short"
    ]
    return run_command(command, "Quick Test Suite")


def run_all_tests():
    """Run all tests."""
    command = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "--tb=short",
        "--durations=20"
    ]
    return run_command(command, "All Tests")


def run_coverage_tests():
    """Run tests with coverage reporting."""
    command = [
        sys.executable, "-m", "pytest",
        "tests/",
        "--cov=continuous_data_collection",
        "--cov-report=html",
        "--cov-report=term-missing",
        "-v"
    ]
    return run_command(command, "Tests with Coverage")


def run_specific_test(test_path):
    """Run a specific test file or test function."""
    command = [
        sys.executable, "-m", "pytest",
        test_path,
        "-v",
        "--tb=short",
        "-s"
    ]
    return run_command(command, f"Specific Test: {test_path}")


def check_test_environment():
    """Check if the test environment is properly set up."""
    print("Checking test environment...")
    
    # Check if pytest is installed
    try:
        import pytest
        print(f"✓ pytest version: {pytest.__version__}")
    except ImportError:
        print("✗ pytest not installed. Run: pip install pytest")
        return False
    
    # Check if test directories exist
    test_dirs = ["tests/unit", "tests/integration", "tests/reliability"]
    for test_dir in test_dirs:
        if Path(test_dir).exists():
            print(f"✓ {test_dir} directory exists")
        else:
            print(f"✗ {test_dir} directory missing")
            return False
    
    # Check if main module can be imported
    try:
        sys.path.insert(0, os.path.abspath('.'))
        import continuous_data_collection
        print("✓ continuous_data_collection module can be imported")
    except ImportError as e:
        print(f"✗ Cannot import continuous_data_collection: {e}")
        return False
    
    print("✓ Test environment is ready")
    return True


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Test runner for continuous data collection system")
    parser.add_argument(
        "suite",
        nargs="?",
        choices=["unit", "integration", "reliability", "performance", "quick", "all", "coverage"],
        default="quick",
        help="Test suite to run (default: quick)"
    )
    parser.add_argument(
        "--specific",
        help="Run a specific test file or test function"
    )
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Check test environment setup"
    )
    
    args = parser.parse_args()
    
    if args.check_env:
        if not check_test_environment():
            sys.exit(1)
        return
    
    if args.specific:
        success = run_specific_test(args.specific)
        sys.exit(0 if success else 1)
    
    # Map suite names to functions
    suite_functions = {
        "unit": run_unit_tests,
        "integration": run_integration_tests,
        "reliability": run_reliability_tests,
        "performance": run_performance_tests,
        "quick": run_quick_tests,
        "all": run_all_tests,
        "coverage": run_coverage_tests
    }
    
    if args.suite not in suite_functions:
        print(f"Unknown test suite: {args.suite}")
        sys.exit(1)
    
    # Check environment before running tests
    if not check_test_environment():
        print("Test environment check failed. Use --check-env for details.")
        sys.exit(1)
    
    # Run the selected test suite
    success = suite_functions[args.suite]()
    
    if success:
        print(f"\n✓ {args.suite.title()} tests completed successfully!")
    else:
        print(f"\n✗ {args.suite.title()} tests failed!")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()