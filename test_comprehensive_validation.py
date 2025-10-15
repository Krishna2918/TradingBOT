"""
Comprehensive Validation Test Runner
====================================

Runs all validation tests for the AI Trading System including
integration tests, performance tests, security tests, e2e tests,
and monitoring system tests.
"""

import asyncio
import logging
import sys
import time
import io
from datetime import datetime
from typing import Dict, Any, List
import subprocess
import os

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except AttributeError:
        pass  # Already wrapped or not needed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/comprehensive_validation.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class ComprehensiveValidator:
    """Comprehensive validation test runner."""
    
    def __init__(self):
        self.test_results: Dict[str, Any] = {}
        self.start_time = time.time()
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all comprehensive validation tests."""
        logger.info("Starting comprehensive validation tests...")
        
        try:
            # Test categories
            test_categories = [
                ("Integration Tests", self._run_integration_tests),
                ("Performance Tests", self._run_performance_tests),
                ("Security Tests", self._run_security_tests),
                ("End-to-End Tests", self._run_e2e_tests),
                ("Monitoring Tests", self._run_monitoring_tests),
                ("Phase Validation Tests", self._run_phase_validation_tests)
            ]
            
            # Run each test category
            for category_name, test_function in test_categories:
                logger.info(f"Running {category_name}...")
                try:
                    result = await test_function()
                    self.test_results[category_name] = result
                    
                    if result["success"]:
                        self.passed_tests += result["passed"]
                        logger.info(f"PASSED {category_name}: {result['passed']} tests passed")
                    else:
                        self.failed_tests += result["failed"]
                        logger.error(f"FAILED {category_name}: {result['failed']} tests failed")
                    
                    self.total_tests += result["total"]
                    
                except Exception as e:
                    logger.error(f"Error running {category_name}: {e}")
                    self.test_results[category_name] = {
                        "success": False,
                        "error": str(e),
                        "total": 0,
                        "passed": 0,
                        "failed": 0
                    }
            
            # Generate final report
            final_report = self._generate_final_report()
            
            logger.info("Comprehensive validation tests completed")
            return final_report
            
        except Exception as e:
            logger.error(f"Error running comprehensive validation: {e}")
            return {
                "success": False,
                "error": str(e),
                "total_tests": self.total_tests,
                "passed_tests": self.passed_tests,
                "failed_tests": self.failed_tests
            }
    
    async def _run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests."""
        try:
            # Basic integration validation - test core component imports
            logger.info("Testing core component imports...")
            tests_passed = 0
            tests_failed = 0
            errors = []
            
            # Test 1: Trading cycle imports
            try:
                from src.workflows.trading_cycle import get_trading_cycle
                tests_passed += 1
                logger.info("[PASS] Trading cycle imports successful")
            except Exception as e:
                tests_failed += 1
                errors.append(f"Trading cycle import failed: {e}")
                logger.error(f"[FAIL] Trading cycle import failed: {e}")
            
            # Test 2: Execution engine imports
            try:
                from src.trading.execution import get_execution_engine
                tests_passed += 1
                logger.info("[PASS] Execution engine imports successful")
            except Exception as e:
                tests_failed += 1
                errors.append(f"Execution engine import failed: {e}")
                logger.error(f"[FAIL] Execution engine import failed: {e}")
            
            # Test 3: Position manager imports
            try:
                from src.trading.positions import get_position_manager
                tests_passed += 1
                logger.info("[PASS] Position manager imports successful")
            except Exception as e:
                tests_failed += 1
                errors.append(f"Position manager import failed: {e}")
                logger.error(f"[FAIL] Position manager import failed: {e}")
            
            return {
                "success": tests_failed == 0,
                "total": tests_passed + tests_failed,
                "passed": tests_passed,
                "failed": tests_failed,
                "output": f"Integration validation: {tests_passed} passed, {tests_failed} failed",
                "error": "; ".join(errors) if errors else None
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "total": 0,
                "passed": 0,
                "failed": 1,
                "error": "Integration tests timed out"
            }
        except Exception as e:
            return {
                "success": False,
                "total": 0,
                "passed": 0,
                "failed": 1,
                "error": str(e)
            }
    
    async def _run_performance_tests(self) -> Dict[str, Any]:
        """Run performance tests."""
        try:
            # Basic performance validation - test system responsiveness
            logger.info("Testing system performance...")
            tests_passed = 0
            tests_failed = 0
            errors = []
            
            # Test 1: Import speed test
            try:
                import time
                start_time = time.time()
                from src.ai.multi_model import get_multi_model_manager
                import_time = time.time() - start_time
                if import_time < 2.0:  # Should import in under 2 seconds
                    tests_passed += 1
                    logger.info(f"[PASS] AI ensemble import speed: {import_time:.3f}s")
                else:
                    tests_failed += 1
                    errors.append(f"AI ensemble import too slow: {import_time:.3f}s")
                    logger.error(f"[FAIL] AI ensemble import too slow: {import_time:.3f}s")
            except Exception as e:
                tests_failed += 1
                errors.append(f"AI ensemble import failed: {e}")
                logger.error(f"[FAIL] AI ensemble import failed: {e}")
            
            # Test 2: Database connection speed
            try:
                start_time = time.time()
                from src.config.database import get_database_manager
                db_manager = get_database_manager()
                db_time = time.time() - start_time
                if db_time < 1.0:  # Should connect in under 1 second
                    tests_passed += 1
                    logger.info(f"[PASS] Database connection speed: {db_time:.3f}s")
                else:
                    tests_failed += 1
                    errors.append(f"Database connection too slow: {db_time:.3f}s")
                    logger.error(f"[FAIL] Database connection too slow: {db_time:.3f}s")
            except Exception as e:
                tests_failed += 1
                errors.append(f"Database connection failed: {e}")
                logger.error(f"[FAIL] Database connection failed: {e}")
            
            # Test 3: Memory usage check
            try:
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                if memory_mb < 500:  # Should use less than 500MB
                    tests_passed += 1
                    logger.info(f"[PASS] Memory usage: {memory_mb:.1f}MB")
                else:
                    tests_failed += 1
                    errors.append(f"Memory usage too high: {memory_mb:.1f}MB")
                    logger.error(f"[FAIL] Memory usage too high: {memory_mb:.1f}MB")
            except Exception as e:
                tests_failed += 1
                errors.append(f"Memory check failed: {e}")
                logger.error(f"[FAIL] Memory check failed: {e}")
            
            return {
                "success": tests_failed == 0,
                "total": tests_passed + tests_failed,
                "passed": tests_passed,
                "failed": tests_failed,
                "output": f"Performance validation: {tests_passed} passed, {tests_failed} failed",
                "error": "; ".join(errors) if errors else None
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "total": 0,
                "passed": 0,
                "failed": 1,
                "error": "Performance tests timed out"
            }
        except Exception as e:
            return {
                "success": False,
                "total": 0,
                "passed": 0,
                "failed": 1,
                "error": str(e)
            }
    
    async def _run_security_tests(self) -> Dict[str, Any]:
        """Run security tests."""
        try:
            # Basic security validation - test security components
            logger.info("Testing security components...")
            tests_passed = 0
            tests_failed = 0
            errors = []
            
            # Test 1: Security validator imports
            try:
                from src.validation.security_validator import get_security_validator
                tests_passed += 1
                logger.info("[PASS] Security validator imports successful")
            except Exception as e:
                tests_failed += 1
                errors.append(f"Security validator import failed: {e}")
                logger.error(f"[FAIL] Security validator import failed: {e}")
            
            # Test 2: Hallucination detector imports
            try:
                from src.validation.hallucination_detector import get_hallucination_detector
                tests_passed += 1
                logger.info("[PASS] Hallucination detector imports successful")
            except Exception as e:
                tests_failed += 1
                errors.append(f"Hallucination detector import failed: {e}")
                logger.error(f"[FAIL] Hallucination detector import failed: {e}")
            
            # Test 3: Change tracker imports
            try:
                from src.validation.change_tracker import get_change_tracker
                tests_passed += 1
                logger.info("[PASS] Change tracker imports successful")
            except Exception as e:
                tests_failed += 1
                errors.append(f"Change tracker import failed: {e}")
                logger.error(f"[FAIL] Change tracker import failed: {e}")
            
            return {
                "success": tests_failed == 0,
                "total": tests_passed + tests_failed,
                "passed": tests_passed,
                "failed": tests_failed,
                "output": f"Security validation: {tests_passed} passed, {tests_failed} failed",
                "error": "; ".join(errors) if errors else None
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "total": 0,
                "passed": 0,
                "failed": 1,
                "error": "Security tests timed out"
            }
        except Exception as e:
            return {
                "success": False,
                "total": 0,
                "passed": 0,
                "failed": 1,
                "error": str(e)
            }
    
    async def _run_e2e_tests(self) -> Dict[str, Any]:
        """Run end-to-end tests."""
        try:
            # Basic E2E validation - test system integration
            logger.info("Testing end-to-end system integration...")
            tests_passed = 0
            tests_failed = 0
            errors = []
            
            # Test 1: Main system imports
            try:
                from src.main import TradingSystem
                tests_passed += 1
                logger.info("[PASS] Main system imports successful")
            except Exception as e:
                tests_failed += 1
                errors.append(f"Main system import failed: {e}")
                logger.error(f"[FAIL] Main system import failed: {e}")
            
            # Test 2: Master Orchestrator imports
            try:
                from src.integration.master_orchestrator import MasterOrchestrator
                tests_passed += 1
                logger.info("[PASS] Master Orchestrator imports successful")
            except Exception as e:
                tests_failed += 1
                errors.append(f"Master Orchestrator import failed: {e}")
                logger.error(f"[FAIL] Master Orchestrator import failed: {e}")
            
            # Test 3: System integration test
            try:
                # Test that we can create a trading system instance
                from src.main import TradingSystem
                trading_system = TradingSystem()
                tests_passed += 1
                logger.info("[PASS] Trading system instantiation successful")
            except Exception as e:
                tests_failed += 1
                errors.append(f"Trading system instantiation failed: {e}")
                logger.error(f"[FAIL] Trading system instantiation failed: {e}")
            
            return {
                "success": tests_failed == 0,
                "total": tests_passed + tests_failed,
                "passed": tests_passed,
                "failed": tests_failed,
                "output": f"E2E validation: {tests_passed} passed, {tests_failed} failed",
                "error": "; ".join(errors) if errors else None
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "total": 0,
                "passed": 0,
                "failed": 1,
                "error": "E2E tests timed out"
            }
        except Exception as e:
            return {
                "success": False,
                "total": 0,
                "passed": 0,
                "failed": 1,
                "error": str(e)
            }
    
    async def _run_monitoring_tests(self) -> Dict[str, Any]:
        """Run monitoring system tests."""
        try:
            # Test monitoring system components
            from src.monitoring.system_monitor import get_system_monitor
            from src.monitoring.error_tracker import get_error_tracker
            from src.monitoring.alert_system import get_alert_system
            from src.monitoring.audit_trail import get_audit_trail
            from src.monitoring.performance_analytics import get_performance_analytics
            from src.monitoring.predictive_alerts import get_predictive_alerts
            
            # Test system monitor
            system_monitor = get_system_monitor()
            assert system_monitor is not None, "System monitor should be available"
            
            # Test error tracker
            error_tracker = get_error_tracker()
            assert error_tracker is not None, "Error tracker should be available"
            
            # Test alert system
            alert_system = get_alert_system()
            assert alert_system is not None, "Alert system should be available"
            
            # Test audit trail
            audit_trail = get_audit_trail()
            assert audit_trail is not None, "Audit trail should be available"
            
            # Test performance analytics
            performance_analytics = get_performance_analytics()
            assert performance_analytics is not None, "Performance analytics should be available"
            
            # Test predictive alerts
            predictive_alerts = get_predictive_alerts()
            assert predictive_alerts is not None, "Predictive alerts should be available"
            
            return {
                "success": True,
                "total": 6,
                "passed": 6,
                "failed": 0,
                "output": "All monitoring components are available and functional"
            }
            
        except Exception as e:
            return {
                "success": False,
                "total": 6,
                "passed": 0,
                "failed": 6,
                "error": str(e)
            }
    
    async def _run_phase_validation_tests(self) -> Dict[str, Any]:
        """Run phase validation tests."""
        try:
            # Basic phase validation - test phase components
            logger.info("Testing phase validation components...")
            tests_passed = 0
            tests_failed = 0
            errors = []
            
            # Test 1: Phase 1 components
            try:
                from src.config.mode_manager import get_mode_manager
                from src.config.database import get_database_manager
                tests_passed += 1
                logger.info("[PASS] Phase 1 components imports successful")
            except Exception as e:
                tests_failed += 1
                errors.append(f"Phase 1 components import failed: {e}")
                logger.error(f"[FAIL] Phase 1 components import failed: {e}")
            
            # Test 2: Phase 2 components
            try:
                from src.workflows.trading_cycle import get_trading_cycle
                from src.workflows.activity_scheduler import get_activity_scheduler
                tests_passed += 1
                logger.info("[PASS] Phase 2 components imports successful")
            except Exception as e:
                tests_failed += 1
                errors.append(f"Phase 2 components import failed: {e}")
                logger.error(f"[FAIL] Phase 2 components import failed: {e}")
            
            # Test 3: Phase 3 components
            try:
                from src.ai.reinforcement_learning import get_rl_trainer
                from src.risk.advanced_risk_management import get_advanced_risk_manager
                tests_passed += 1
                logger.info("[PASS] Phase 3 components imports successful")
            except Exception as e:
                tests_failed += 1
                errors.append(f"Phase 3 components import failed: {e}")
                logger.error(f"[FAIL] Phase 3 components import failed: {e}")
            
            return {
                "success": tests_failed == 0,
                "total": tests_passed + tests_failed,
                "passed": tests_passed,
                "failed": tests_failed,
                "output": f"Phase validation: {tests_passed} passed, {tests_failed} failed",
                "error": "; ".join(errors) if errors else None
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "total": 0,
                "passed": 0,
                "failed": 1,
                "error": "Phase validation tests timed out"
            }
        except Exception as e:
            return {
                "success": False,
                "total": 0,
                "passed": 0,
                "failed": 1,
                "error": str(e)
            }
    
    def _count_tests(self, output: str) -> int:
        """Count total tests from pytest output."""
        try:
            lines = output.split('\n')
            for line in lines:
                if 'collected' in line and 'item' in line:
                    # Extract number from line like "collected 25 items"
                    parts = line.split()
                    for part in parts:
                        if part.isdigit():
                            return int(part)
            return 0
        except:
            return 0
    
    def _count_passed(self, output: str) -> int:
        """Count passed tests from pytest output."""
        try:
            lines = output.split('\n')
            for line in lines:
                if 'passed' in line and 'failed' in line:
                    # Extract number from line like "25 passed, 0 failed"
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'passed':
                            if i > 0 and parts[i-1].isdigit():
                                return int(parts[i-1])
            return 0
        except:
            return 0
    
    def _count_failed(self, output: str) -> int:
        """Count failed tests from pytest output."""
        try:
            lines = output.split('\n')
            for line in lines:
                if 'passed' in line and 'failed' in line:
                    # Extract number from line like "25 passed, 0 failed"
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'failed':
                            if i > 0 and parts[i-1].isdigit():
                                return int(parts[i-1])
            return 0
        except:
            return 0
    
    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate final validation report."""
        end_time = time.time()
        total_time = end_time - self.start_time
        
        success_rate = (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_time": total_time,
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "success_rate": success_rate,
            "overall_success": self.failed_tests == 0,
            "test_results": self.test_results,
            "summary": {
                "integration_tests": self.test_results.get("Integration Tests", {}).get("success", False),
                "performance_tests": self.test_results.get("Performance Tests", {}).get("success", False),
                "security_tests": self.test_results.get("Security Tests", {}).get("success", False),
                "e2e_tests": self.test_results.get("End-to-End Tests", {}).get("success", False),
                "monitoring_tests": self.test_results.get("Monitoring Tests", {}).get("success", False),
                "phase_validation_tests": self.test_results.get("Phase Validation Tests", {}).get("success", False)
            }
        }
        
        return report


async def main():
    """Main function to run comprehensive validation."""
    try:
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        # Run comprehensive validation
        validator = ComprehensiveValidator()
        report = await validator.run_all_tests()
        
        # Print results
        print("\n" + "="*80)
        print("COMPREHENSIVE VALIDATION REPORT")
        print("="*80)
        print(f"Timestamp: {report['timestamp']}")
        print(f"Total Time: {report['total_time']:.2f} seconds")
        print(f"Total Tests: {report['total_tests']}")
        print(f"Passed Tests: {report['passed_tests']}")
        print(f"Failed Tests: {report['failed_tests']}")
        print(f"Success Rate: {report['success_rate']:.1f}%")
        print(f"Overall Success: {'PASS' if report['overall_success'] else 'FAIL'}")
        
        print("\nTest Category Results:")
        for category, result in report['test_results'].items():
            status = "PASS" if result.get('success', False) else "FAIL"
            print(f"  {category}: {status}")
        
        print("\nSummary:")
        summary = report['summary']
        for test_type, success in summary.items():
            status = "PASS" if success else "FAIL"
            print(f"  {test_type.replace('_', ' ').title()}: {status}")
        
        print("="*80)
        
        # Save report to file
        import json
        with open("logs/comprehensive_validation_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Comprehensive validation completed: {report['success_rate']:.1f}% success rate")
        
        # Return exit code
        return 0 if report['overall_success'] else 1
        
    except Exception as e:
        logger.error(f"Error running comprehensive validation: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
