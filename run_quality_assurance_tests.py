#!/usr/bin/env python3
"""
Quality Assurance Test Runner for Target Labeling System.

This script runs comprehensive quality assurance tests including:
- A/B testing framework tests
- Data leakage prevention tests  
- Production monitoring tests
- Integration tests with real data patterns
"""

import sys
import unittest
import logging
from pathlib import Path
import argparse
import time
from typing import List, Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def discover_and_run_tests(test_directory: str, pattern: str = "test_*.py") -> unittest.TestResult:
    """
    Discover and run tests in the specified directory.
    
    Args:
        test_directory: Directory containing test files
        pattern: Pattern to match test files
        
    Returns:
        TestResult object
    """
    loader = unittest.TestLoader()
    start_dir = Path(__file__).parent / test_directory
    
    if not start_dir.exists():
        logger.error(f"Test directory not found: {start_dir}")
        return None
    
    suite = loader.discover(str(start_dir), pattern=pattern)
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    
    logger.info(f"Running tests from {start_dir} with pattern {pattern}")
    result = runner.run(suite)
    
    return result


def run_ab_testing_tests() -> unittest.TestResult:
    """Run A/B testing framework tests."""
    logger.info("=" * 60)
    logger.info("RUNNING A/B TESTING FRAMEWORK TESTS")
    logger.info("=" * 60)
    
    return discover_and_run_tests("tests/quality_assurance", "test_ab_testing_framework.py")


def run_data_leakage_tests() -> unittest.TestResult:
    """Run data leakage prevention tests."""
    logger.info("=" * 60)
    logger.info("RUNNING DATA LEAKAGE PREVENTION TESTS")
    logger.info("=" * 60)
    
    return discover_and_run_tests("tests/quality_assurance", "test_data_leakage_prevention.py")


def run_production_monitoring_tests() -> unittest.TestResult:
    """Run production monitoring tests."""
    logger.info("=" * 60)
    logger.info("RUNNING PRODUCTION MONITORING TESTS")
    logger.info("=" * 60)
    
    return discover_and_run_tests("tests/quality_assurance", "test_production_monitoring.py")


def run_all_quality_assurance_tests() -> Dict[str, unittest.TestResult]:
    """Run all quality assurance tests."""
    logger.info("=" * 80)
    logger.info("STARTING COMPREHENSIVE QUALITY ASSURANCE TEST SUITE")
    logger.info("=" * 80)
    
    start_time = time.time()
    results = {}
    
    # Run A/B testing tests
    results['ab_testing'] = run_ab_testing_tests()
    
    # Run data leakage tests
    results['data_leakage'] = run_data_leakage_tests()
    
    # Run production monitoring tests
    results['production_monitoring'] = run_production_monitoring_tests()
    
    end_time = time.time()
    duration = end_time - start_time
    
    logger.info("=" * 80)
    logger.info("QUALITY ASSURANCE TEST SUITE COMPLETED")
    logger.info(f"Total Duration: {duration:.2f} seconds")
    logger.info("=" * 80)
    
    return results


def generate_test_report(results: Dict[str, unittest.TestResult]) -> str:
    """
    Generate a comprehensive test report.
    
    Args:
        results: Dictionary of test results by category
        
    Returns:
        Formatted test report string
    """
    report = []
    report.append("=" * 80)
    report.append("QUALITY ASSURANCE TEST REPORT")
    report.append("=" * 80)
    report.append("")
    
    total_tests = 0
    total_failures = 0
    total_errors = 0
    total_skipped = 0
    
    for category, result in results.items():
        if result is None:
            report.append(f"{category.upper()}: FAILED TO RUN")
            continue
            
        tests_run = result.testsRun
        failures = len(result.failures)
        errors = len(result.errors)
        skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
        
        total_tests += tests_run
        total_failures += failures
        total_errors += errors
        total_skipped += skipped
        
        status = "PASSED" if (failures == 0 and errors == 0) else "FAILED"
        
        report.append(f"{category.upper()}: {status}")
        report.append(f"  Tests Run: {tests_run}")
        report.append(f"  Failures: {failures}")
        report.append(f"  Errors: {errors}")
        report.append(f"  Skipped: {skipped}")
        
        # Add failure details
        if failures > 0:
            report.append("  FAILURES:")
            for test, traceback in result.failures:
                report.append(f"    - {test}: {traceback.split('AssertionError:')[-1].strip()}")
        
        if errors > 0:
            report.append("  ERRORS:")
            for test, traceback in result.errors:
                report.append(f"    - {test}: {traceback.split('Exception:')[-1].strip()}")
        
        report.append("")
    
    # Summary
    overall_status = "PASSED" if (total_failures == 0 and total_errors == 0) else "FAILED"
    report.append("OVERALL SUMMARY:")
    report.append(f"Status: {overall_status}")
    report.append(f"Total Tests: {total_tests}")
    report.append(f"Total Failures: {total_failures}")
    report.append(f"Total Errors: {total_errors}")
    report.append(f"Total Skipped: {total_skipped}")
    
    if total_tests > 0:
        success_rate = ((total_tests - total_failures - total_errors) / total_tests) * 100
        report.append(f"Success Rate: {success_rate:.1f}%")
    
    report.append("")
    report.append("=" * 80)
    
    return "\n".join(report)


def run_integration_demo() -> None:
    """Run integration demo showing quality assurance in action."""
    logger.info("=" * 60)
    logger.info("RUNNING QUALITY ASSURANCE INTEGRATION DEMO")
    logger.info("=" * 60)
    
    try:
        import pandas as pd
        import numpy as np
        from ai.data.targets import ensure_direction_1d
        from ai.monitoring.target_quality_monitor import setup_production_monitoring
        from tests.quality_assurance.test_ab_testing_framework import ABTestingFramework
        
        # Create sample data
        np.random.seed(42)
        prices = [100.0]
        for _ in range(199):
            change = np.random.normal(0.001, 0.02)
            prices.append(prices[-1] * (1 + change))
        
        sample_data = pd.DataFrame({
            'close': prices,
            'volume': np.random.randint(1000, 5000, 200)
        })
        
        logger.info("1. Testing A/B Framework with different neutral bands...")
        
        # A/B test different neutral bands
        ab_framework = ABTestingFramework(random_seed=42)
        neutral_bands = [0.003, 0.004, 0.005]
        
        ab_results = ab_framework.test_neutral_band_impact(
            sample_data, 
            neutral_bands, 
            "DEMO_SYMBOL"
        )
        
        logger.info(f"   A/B test completed: {len(ab_results)} comparisons")
        for result in ab_results:
            logger.info(f"   {result.method_a_name} vs {result.method_b_name}: "
                       f"{'Identical' if result.identical_results else f'{result.differences_count} differences'}")
        
        logger.info("2. Testing production monitoring...")
        
        # Setup production monitoring
        monitor = setup_production_monitoring()
        
        # Create targets and monitor
        df_with_targets = ensure_direction_1d(sample_data, neutral_band=0.004, symbol="DEMO_SYMBOL")
        metrics = monitor.monitor_target_creation(df_with_targets, "DEMO_SYMBOL", 0.004)
        
        logger.info(f"   Quality Score: {metrics.quality_score:.1f}/100")
        logger.info(f"   Distribution: DOWN={metrics.down_percentage:.1f}% "
                   f"FLAT={metrics.flat_percentage:.1f}% UP={metrics.up_percentage:.1f}%")
        logger.info(f"   Alerts: {len(metrics.alerts)} ({'None' if len(metrics.alerts) == 0 else ', '.join(metrics.alerts)})")
        
        logger.info("3. Testing data leakage prevention...")
        
        # Test temporal consistency
        from tests.quality_assurance.test_data_leakage_prevention import DataLeakageDetector
        
        leakage_results = DataLeakageDetector.detect_future_information_usage(
            df_with_targets, 
            target_col='direction_1d'
        )
        
        logger.info(f"   Leakage Detection: {'PASSED' if not leakage_results['has_leakage'] else 'FAILED'}")
        logger.info(f"   Temporal Consistency: {'PASSED' if leakage_results['temporal_consistency'] else 'FAILED'}")
        
        if leakage_results['leakage_indicators']:
            logger.warning(f"   Indicators: {', '.join(leakage_results['leakage_indicators'])}")
        
        logger.info("4. Integration demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Integration demo failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function to run quality assurance tests."""
    parser = argparse.ArgumentParser(description="Quality Assurance Test Runner")
    parser.add_argument('--test-type', choices=['all', 'ab', 'leakage', 'monitoring', 'demo'], 
                       default='all', help='Type of tests to run')
    parser.add_argument('--report-file', type=str, help='File to save test report')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run tests based on selection
    if args.test_type == 'all':
        results = run_all_quality_assurance_tests()
        report = generate_test_report(results)
    elif args.test_type == 'ab':
        results = {'ab_testing': run_ab_testing_tests()}
        report = generate_test_report(results)
    elif args.test_type == 'leakage':
        results = {'data_leakage': run_data_leakage_tests()}
        report = generate_test_report(results)
    elif args.test_type == 'monitoring':
        results = {'production_monitoring': run_production_monitoring_tests()}
        report = generate_test_report(results)
    elif args.test_type == 'demo':
        run_integration_demo()
        return
    
    # Print report
    print(report)
    
    # Save report if requested
    if args.report_file:
        with open(args.report_file, 'w') as f:
            f.write(report)
        logger.info(f"Test report saved to {args.report_file}")
    
    # Exit with appropriate code
    if args.test_type != 'demo':
        total_failures = sum(len(r.failures) + len(r.errors) for r in results.values() if r is not None)
        sys.exit(0 if total_failures == 0 else 1)


if __name__ == "__main__":
    main()