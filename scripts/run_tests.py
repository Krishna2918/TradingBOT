#!/usr/bin/env python3
"""
Comprehensive test runner for the Trading Bot system.
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestRunner:
    """Comprehensive test runner with multiple test categories and reporting."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.test_dir = self.project_root / "tests"
        self.reports_dir = self.project_root / "tests" / "reports"
        self.ensure_directories()
    
    def ensure_directories(self):
        """Ensure required directories exist."""
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def run_command(self, command: List[str], description: str) -> Dict[str, Any]:
        """Run a command and return results."""
        print(f"\n{'='*60}")
        print(f"Running: {description}")
        print(f"Command: {' '.join(command)}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                command,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            duration = time.time() - start_time
            
            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration": duration,
                "description": description
            }
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return {
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": f"Command timed out after {duration:.2f} seconds",
                "duration": duration,
                "description": description
            }
        except Exception as e:
            duration = time.time() - start_time
            return {
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": str(e),
                "duration": duration,
                "description": description
            }
    
    def run_unit_tests(self) -> Dict[str, Any]:
        """Run unit tests."""
        command = [
            "python", "-m", "pytest",
            "tests/unit/",
            "-m", "unit",
            "--tb=short"
        ]
        return self.run_command(command, "Unit Tests")
    
    def run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests."""
        command = [
            "python", "-m", "pytest",
            "tests/integration/",
            "-m", "integration",
            "--tb=short"
        ]
        return self.run_command(command, "Integration Tests")
    
    def run_smoke_tests(self) -> Dict[str, Any]:
        """Run smoke tests."""
        command = [
            "python", "-m", "pytest",
            "tests/smoke/",
            "-m", "smoke",
            "--tb=short"
        ]
        return self.run_command(command, "Smoke Tests")
    
    def run_regression_tests(self) -> Dict[str, Any]:
        """Run regression tests."""
        command = [
            "python", "-m", "pytest",
            "tests/regression/",
            "-m", "regression",
            "--tb=short"
        ]
        return self.run_command(command, "Regression Tests")
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests with coverage."""
        command = [
            "python", "-m", "pytest",
            "tests/",
            "--cov=src",
            "--cov-report=html:tests/reports/coverage",
            "--cov-report=xml:tests/reports/coverage.xml",
            "--cov-report=term-missing",
            "--junitxml=tests/reports/junit.xml",
            "--html=tests/reports/report.html",
            "--self-contained-html",
            "--tb=short"
        ]
        return self.run_command(command, "All Tests with Coverage")
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance tests."""
        command = [
            "python", "-m", "pytest",
            "tests/",
            "-m", "performance",
            "--tb=short",
            "--benchmark-only"
        ]
        return self.run_command(command, "Performance Tests")
    
    def run_security_tests(self) -> Dict[str, Any]:
        """Run security tests."""
        command = [
            "python", "-m", "pytest",
            "tests/",
            "-m", "security",
            "--tb=short"
        ]
        return self.run_command(command, "Security Tests")
    
    def run_code_quality_checks(self) -> Dict[str, Any]:
        """Run code quality checks."""
        results = []
        
        # Black formatting check
        black_result = self.run_command(
            ["python", "-m", "black", "--check", "src/", "tests/"],
            "Black Formatting Check"
        )
        results.append(black_result)
        
        # Flake8 linting
        flake8_result = self.run_command(
            ["python", "-m", "flake8", "src/", "tests/"],
            "Flake8 Linting"
        )
        results.append(flake8_result)
        
        # MyPy type checking
        mypy_result = self.run_command(
            ["python", "-m", "mypy", "src/"],
            "MyPy Type Checking"
        )
        results.append(mypy_result)
        
        # Bandit security scan
        bandit_result = self.run_command(
            ["python", "-m", "bandit", "-r", "src/"],
            "Bandit Security Scan"
        )
        results.append(bandit_result)
        
        return {
            "success": all(r["success"] for r in results),
            "results": results,
            "description": "Code Quality Checks"
        }
    
    def run_phase_tests(self) -> Dict[str, Any]:
        """Run phase-specific tests."""
        results = []
        
        # Phase 4 tests
        phase4_result = self.run_command(
            ["python", "scripts/phase4_smoke_test.py"],
            "Phase 4 Smoke Test"
        )
        results.append(phase4_result)
        
        # Phase 6 tests
        phase6_result = self.run_command(
            ["python", "scripts/phase6_smoke_test.py"],
            "Phase 6 Smoke Test"
        )
        results.append(phase6_result)
        
        return {
            "success": all(r["success"] for r in results),
            "results": results,
            "description": "Phase Tests"
        }
    
    def generate_test_report(self, results: List[Dict[str, Any]]) -> str:
        """Generate comprehensive test report."""
        report = []
        report.append("# Trading Bot Test Report")
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r["success"])
        failed_tests = total_tests - passed_tests
        
        report.append("## Summary")
        report.append(f"- Total Test Suites: {total_tests}")
        report.append(f"- Passed: {passed_tests}")
        report.append(f"- Failed: {failed_tests}")
        report.append(f"- Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        report.append("")
        
        # Detailed Results
        report.append("## Detailed Results")
        report.append("")
        
        for result in results:
            status = "✅ PASS" if result["success"] else "❌ FAIL"
            report.append(f"### {result['description']} - {status}")
            report.append(f"- Duration: {result['duration']:.2f}s")
            report.append(f"- Return Code: {result['returncode']}")
            
            if not result["success"] and result["stderr"]:
                report.append(f"- Error: {result['stderr']}")
            
            report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        if failed_tests > 0:
            report.append("- Fix failing tests before deployment")
            report.append("- Review error messages and logs")
            report.append("- Consider adding more test coverage")
        else:
            report.append("- All tests passed! System is ready for deployment")
            report.append("- Consider adding performance benchmarks")
            report.append("- Review test coverage for new features")
        
        return "\n".join(report)
    
    def run_tests(self, test_types: List[str], generate_report: bool = True) -> Dict[str, Any]:
        """Run specified test types."""
        results = []
        
        if "unit" in test_types:
            results.append(self.run_unit_tests())
        
        if "integration" in test_types:
            results.append(self.run_integration_tests())
        
        if "smoke" in test_types:
            results.append(self.run_smoke_tests())
        
        if "regression" in test_types:
            results.append(self.run_regression_tests())
        
        if "performance" in test_types:
            results.append(self.run_performance_tests())
        
        if "security" in test_types:
            results.append(self.run_security_tests())
        
        if "quality" in test_types:
            results.append(self.run_code_quality_checks())
        
        if "phases" in test_types:
            results.append(self.run_phase_tests())
        
        if "all" in test_types:
            results.append(self.run_all_tests())
        
        # Generate report
        if generate_report:
            report_content = self.generate_test_report(results)
            report_file = self.reports_dir / "test_report.md"
            report_file.write_text(report_content)
            print(f"\nTest report generated: {report_file}")
        
        # Summary
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r["success"])
        failed_tests = total_tests - passed_tests
        
        print(f"\n{'='*60}")
        print(f"TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total Test Suites: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            print(f"\n❌ {failed_tests} test suite(s) failed!")
            return {"success": False, "results": results}
        else:
            print(f"\n✅ All test suites passed!")
            return {"success": True, "results": results}

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Trading Bot Test Runner")
    parser.add_argument(
        "--types",
        nargs="+",
        choices=["unit", "integration", "smoke", "regression", "performance", "security", "quality", "phases", "all"],
        default=["all"],
        help="Test types to run"
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip generating test report"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    print("Trading Bot Test Runner")
    print("=" * 60)
    print(f"Test types: {', '.join(args.types)}")
    print(f"Generate report: {not args.no_report}")
    print(f"Verbose: {args.verbose}")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        result = runner.run_tests(args.types, not args.no_report)
        
        total_duration = time.time() - start_time
        print(f"\nTotal execution time: {total_duration:.2f}s")
        
        if result["success"]:
            print("\n🎉 All tests completed successfully!")
            sys.exit(0)
        else:
            print("\n💥 Some tests failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Test execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test runner error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()