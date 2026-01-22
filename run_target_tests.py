#!/usr/bin/env python3
"""
Test runner for target labeling surgical fixes test suite.

Runs all tests related to the target creation and validation system.
"""

import unittest
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def run_target_tests():
    """Run all target-related tests."""
    
    # Discover and run unit tests
    print("=" * 60)
    print("RUNNING UNIT TESTS")
    print("=" * 60)
    
    unit_loader = unittest.TestLoader()
    unit_suite = unit_loader.discover('tests/unit', pattern='test_*.py')
    unit_runner = unittest.TextTestRunner(verbosity=2)
    unit_result = unit_runner.run(unit_suite)
    
    print("\n" + "=" * 60)
    print("RUNNING INTEGRATION TESTS")
    print("=" * 60)
    
    # Run only our target pipeline integration tests
    integration_loader = unittest.TestLoader()
    integration_suite = integration_loader.loadTestsFromName('tests.integration.test_target_pipeline')
    integration_runner = unittest.TextTestRunner(verbosity=2)
    integration_result = integration_runner.run(integration_suite)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    unit_tests_run = unit_result.testsRun
    unit_failures = len(unit_result.failures)
    unit_errors = len(unit_result.errors)
    unit_passed = unit_tests_run - unit_failures - unit_errors
    
    integration_tests_run = integration_result.testsRun
    integration_failures = len(integration_result.failures)
    integration_errors = len(integration_result.errors)
    integration_passed = integration_tests_run - integration_failures - integration_errors
    
    total_tests = unit_tests_run + integration_tests_run
    total_passed = unit_passed + integration_passed
    total_failed = unit_failures + integration_failures + unit_errors + integration_errors
    
    print(f"Unit Tests:        {unit_passed}/{unit_tests_run} passed")
    print(f"Integration Tests: {integration_passed}/{integration_tests_run} passed")
    print(f"Total:             {total_passed}/{total_tests} passed")
    
    if total_failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nTarget labeling surgical fixes test suite validation:")
        print("  ✅ ensure_direction_1d function unit tests")
        print("  ✅ Validation system unit tests")
        print("  ✅ End-to-end pipeline integration tests")
        print("  ✅ Data leakage prevention verification")
        print("  ✅ Error handling and edge cases")
        print("  ✅ Performance with large datasets")
        return True
    else:
        print(f"\n❌ {total_failed} TESTS FAILED")
        return False

if __name__ == "__main__":
    success = run_target_tests()
    sys.exit(0 if success else 1)