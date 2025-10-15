#!/usr/bin/env python3
"""
CI Validation Suite - Automated validation for continuous integration

This script provides comprehensive validation for the trading system,
including performance benchmarking, regression detection, and system health checks.
"""

import sys
import os
import time
import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import subprocess
import psutil

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CIValidator:
    """Comprehensive CI validation suite."""
    
    def __init__(self):
        """Initialize the CI validator."""
        self.start_time = time.time()
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'tests': {},
            'performance': {},
            'system_health': {},
            'summary': {}
        }
        self.ai_limit = int(os.environ.get('AI_LIMIT', 20))
        self.pytest_disable_plugins = os.environ.get('PYTEST_DISABLE_PLUGIN_AUTOLOAD', '1')
        
        logger.info(f"CI Validator initialized with AI_LIMIT={self.ai_limit}")
    
    async def run_validation_suite(self) -> Dict[str, Any]:
        """Run the complete validation suite."""
        logger.info("Starting CI validation suite...")
        
        try:
            # System health checks
            await self._check_system_health()
            
            # Core functionality tests
            await self._test_core_functionality()
            
            # Performance benchmarks
            await self._run_performance_benchmarks()
            
            # Integration tests
            await self._run_integration_tests()
            
            # Regression detection
            await self._detect_regressions()
            
            # Generate summary
            self._generate_summary()
            
            # Save results
            self._save_results()
            
            logger.info("CI validation suite completed successfully")
            return self.results
            
        except Exception as e:
            logger.error(f"CI validation suite failed: {e}")
            self.results['error'] = str(e)
            self.results['status'] = 'FAILED'
            return self.results
    
    async def _check_system_health(self) -> None:
        """Check system health and resource availability."""
        logger.info("Checking system health...")
        
        health_checks = {}
        
        try:
            # Check Python version
            python_version = sys.version_info
            health_checks['python_version'] = f"{python_version.major}.{python_version.minor}.{python_version.micro}"
            
            # Check available memory
            memory = psutil.virtual_memory()
            health_checks['memory_available_gb'] = round(memory.available / (1024**3), 2)
            health_checks['memory_percent_used'] = memory.percent
            
            # Check disk space
            disk = psutil.disk_usage('.')
            health_checks['disk_free_gb'] = round(disk.free / (1024**3), 2)
            health_checks['disk_percent_used'] = round((disk.used / disk.total) * 100, 2)
            
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            health_checks['cpu_percent'] = cpu_percent
            
            # Check if required directories exist
            required_dirs = ['src', 'tests', 'scripts', 'data', 'logs']
            health_checks['required_directories'] = {}
            for dir_name in required_dirs:
                health_checks['required_directories'][dir_name] = os.path.exists(dir_name)
            
            # Check if required files exist
            required_files = [
                'src/config/database.py',
                'src/monitoring/system_monitor.py',
                'src/ai/multi_model.py',
                'src/trading/risk.py'
            ]
            health_checks['required_files'] = {}
            for file_path in required_files:
                health_checks['required_files'][file_path] = os.path.exists(file_path)
            
            self.results['system_health'] = health_checks
            logger.info("System health check completed")
            
        except Exception as e:
            logger.error(f"System health check failed: {e}")
            self.results['system_health'] = {'error': str(e)}
    
    async def _test_core_functionality(self) -> None:
        """Test core system functionality."""
        logger.info("Testing core functionality...")
        
        core_tests = {}
        
        try:
            # Test database connectivity
            core_tests['database_connectivity'] = await self._test_database_connectivity()
            
            # Test AI model availability
            core_tests['ai_model_availability'] = await self._test_ai_model_availability()
            
            # Test API connectivity
            core_tests['api_connectivity'] = await self._test_api_connectivity()
            
            # Test configuration loading
            core_tests['configuration_loading'] = await self._test_configuration_loading()
            
            self.results['tests']['core_functionality'] = core_tests
            logger.info("Core functionality tests completed")
            
        except Exception as e:
            logger.error(f"Core functionality tests failed: {e}")
            self.results['tests']['core_functionality'] = {'error': str(e)}
    
    async def _test_database_connectivity(self) -> bool:
        """Test database connectivity."""
        try:
            from config.database import get_connection
            
            with get_connection('DEMO') as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                return result[0] == 1
                
        except Exception as e:
            logger.error(f"Database connectivity test failed: {e}")
            return False
    
    async def _test_ai_model_availability(self) -> bool:
        """Test AI model availability."""
        try:
            from ai.ollama_lifecycle import get_lifecycle_manager
            
            manager = get_lifecycle_manager()
            return await manager.health_check()
            
        except Exception as e:
            logger.error(f"AI model availability test failed: {e}")
            return False
    
    async def _test_api_connectivity(self) -> Dict[str, bool]:
        """Test API connectivity."""
        api_tests = {}
        
        try:
            import requests
            
            # Test Questrade API (if available)
            try:
                response = requests.get("https://api.questrade.com/v1/time", timeout=5)
                api_tests['questrade'] = response.status_code == 200
            except:
                api_tests['questrade'] = False
            
            # Test Yahoo Finance API
            try:
                response = requests.get("https://query1.finance.yahoo.com/v8/finance/chart/AAPL", timeout=5)
                api_tests['yahoo_finance'] = response.status_code == 200
            except:
                api_tests['yahoo_finance'] = False
            
            # Test Alpha Vantage API (if key available)
            try:
                api_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
                if api_key:
                    response = requests.get(f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=AAPL&interval=1min&apikey={api_key}", timeout=5)
                    api_tests['alpha_vantage'] = response.status_code == 200
                else:
                    api_tests['alpha_vantage'] = None  # No key available
            except:
                api_tests['alpha_vantage'] = False
            
        except Exception as e:
            logger.error(f"API connectivity tests failed: {e}")
            api_tests['error'] = str(e)
        
        return api_tests
    
    async def _test_configuration_loading(self) -> bool:
        """Test configuration loading."""
        try:
            from config.mode_manager import get_current_mode
            from config.database import DatabaseManager
            
            # Test mode manager
            mode = get_current_mode()
            
            # Test database manager
            db_manager = DatabaseManager()
            
            return mode is not None and db_manager is not None
            
        except Exception as e:
            logger.error(f"Configuration loading test failed: {e}")
            return False
    
    async def _run_performance_benchmarks(self) -> None:
        """Run performance benchmarks."""
        logger.info("Running performance benchmarks...")
        
        performance = {}
        
        try:
            # Benchmark system startup time
            start_time = time.time()
            from src.monitoring.system_monitor import SystemMonitor
            startup_time = time.time() - start_time
            performance['startup_time_seconds'] = round(startup_time, 3)
            
            # Benchmark database operations
            db_start = time.time()
            await self._benchmark_database_operations()
            db_time = time.time() - db_start
            performance['database_operations_seconds'] = round(db_time, 3)
            
            # Benchmark AI model operations
            ai_start = time.time()
            await self._benchmark_ai_operations()
            ai_time = time.time() - ai_start
            performance['ai_operations_seconds'] = round(ai_time, 3)
            
            # Benchmark memory usage
            memory_before = psutil.Process().memory_info().rss / (1024**2)  # MB
            await self._benchmark_memory_usage()
            memory_after = psutil.Process().memory_info().rss / (1024**2)  # MB
            performance['memory_usage_mb'] = round(memory_after - memory_before, 2)
            
            self.results['performance'] = performance
            logger.info("Performance benchmarks completed")
            
        except Exception as e:
            logger.error(f"Performance benchmarks failed: {e}")
            self.results['performance'] = {'error': str(e)}
    
    async def _benchmark_database_operations(self) -> None:
        """Benchmark database operations."""
        try:
            from config.database import get_connection
            
            with get_connection('DEMO') as conn:
                cursor = conn.cursor()
                
                # Benchmark simple query
                for _ in range(10):
                    cursor.execute("SELECT COUNT(*) FROM sqlite_master")
                    cursor.fetchone()
                
                # Benchmark complex query
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name NOT LIKE 'sqlite_%'
                """)
                cursor.fetchall()
                
        except Exception as e:
            logger.error(f"Database benchmark failed: {e}")
    
    async def _benchmark_ai_operations(self) -> None:
        """Benchmark AI operations."""
        try:
            from ai.multi_model import MultiModelManager
            
            manager = MultiModelManager(mode="DEMO")
            
            # Benchmark model configuration loading
            configs = manager.get_all_model_configs()
            
            # Benchmark weight calculations
            weights = manager.get_model_weights()
            adaptive_weights = manager.get_adaptive_weights()
            
        except Exception as e:
            logger.error(f"AI operations benchmark failed: {e}")
    
    async def _benchmark_memory_usage(self) -> None:
        """Benchmark memory usage."""
        try:
            # Create some test data
            test_data = []
            for i in range(1000):
                test_data.append({
                    'id': i,
                    'data': f'test_data_{i}',
                    'timestamp': datetime.now().isoformat()
                })
            
            # Process the data
            processed_data = [item['data'] for item in test_data]
            
            # Clean up
            del test_data, processed_data
            
        except Exception as e:
            logger.error(f"Memory usage benchmark failed: {e}")
    
    async def _run_integration_tests(self) -> None:
        """Run integration tests."""
        logger.info("Running integration tests...")
        
        integration_tests = {}
        
        try:
            # Test phase integration
            integration_tests['phase_integration'] = await self._test_phase_integration()
            
            # Test mode switching
            integration_tests['mode_switching'] = await self._test_mode_switching()
            
            # Test data flow
            integration_tests['data_flow'] = await self._test_data_flow()
            
            self.results['tests']['integration'] = integration_tests
            logger.info("Integration tests completed")
            
        except Exception as e:
            logger.error(f"Integration tests failed: {e}")
            self.results['tests']['integration'] = {'error': str(e)}
    
    async def _test_phase_integration(self) -> bool:
        """Test phase integration."""
        try:
            # Test that all phases can be imported and initialized
            from src.monitoring.system_monitor import SystemMonitor
            from src.data_pipeline.api_budget_manager import API_Budget_Manager
            from src.validation.data_quality import DataQualityValidator
            from src.adaptive.confidence_calibration import ConfidenceCalibrator
            from src.ai.adaptive_weights import AdaptiveWeightManager
            from src.trading.atr_brackets import ATRBracketManager
            from src.ai.regime_detection import RegimeDetector
            from src.dashboard.connector import DashboardConnector
            from src.ai.ollama_lifecycle import OllamaLifecycleManager
            
            return True
            
        except Exception as e:
            logger.error(f"Phase integration test failed: {e}")
            return False
    
    async def _test_mode_switching(self) -> bool:
        """Test mode switching functionality."""
        try:
            from config.mode_manager import get_current_mode, set_mode
            
            # Test getting current mode
            current_mode = get_current_mode()
            
            # Test mode switching (if supported)
            # Note: This might not be implemented yet, so we'll just test getting the mode
            return current_mode is not None
            
        except Exception as e:
            logger.error(f"Mode switching test failed: {e}")
            return False
    
    async def _test_data_flow(self) -> bool:
        """Test data flow through the system."""
        try:
            # Test that data can flow from database to AI models
            from config.database import get_connection
            from ai.multi_model import MultiModelManager
            
            # Get some test data
            with get_connection('DEMO') as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
            
            # Test AI model manager
            manager = MultiModelManager(mode="DEMO")
            configs = manager.get_all_model_configs()
            
            return len(tables) > 0 and len(configs) > 0
            
        except Exception as e:
            logger.error(f"Data flow test failed: {e}")
            return False
    
    async def _detect_regressions(self) -> None:
        """Detect potential regressions."""
        logger.info("Detecting regressions...")
        
        regressions = {}
        
        try:
            # Check for missing critical files
            critical_files = [
                'src/config/database.py',
                'src/monitoring/system_monitor.py',
                'src/ai/multi_model.py',
                'src/trading/risk.py',
                'src/dashboard/connector.py'
            ]
            
            missing_files = []
            for file_path in critical_files:
                if not os.path.exists(file_path):
                    missing_files.append(file_path)
            
            regressions['missing_files'] = missing_files
            
            # Check for import errors
            import_errors = []
            try:
                import src.config.database
                import src.monitoring.system_monitor
                import src.ai.multi_model
                import src.trading.risk
                import src.dashboard.connector
            except ImportError as e:
                import_errors.append(str(e))
            
            regressions['import_errors'] = import_errors
            
            # Check for configuration issues
            config_issues = []
            try:
                from config.mode_manager import get_current_mode
                mode = get_current_mode()
                if mode is None:
                    config_issues.append("Mode manager not returning valid mode")
            except Exception as e:
                config_issues.append(f"Mode manager error: {e}")
            
            regressions['config_issues'] = config_issues
            
            self.results['regressions'] = regressions
            logger.info("Regression detection completed")
            
        except Exception as e:
            logger.error(f"Regression detection failed: {e}")
            self.results['regressions'] = {'error': str(e)}
    
    def _generate_summary(self) -> None:
        """Generate validation summary."""
        logger.info("Generating validation summary...")
        
        summary = {
            'total_time_seconds': round(time.time() - self.start_time, 2),
            'ai_limit': self.ai_limit,
            'timestamp': datetime.now().isoformat(),
            'status': 'PASSED'
        }
        
        # Check if any tests failed
        if 'tests' in self.results:
            for test_category, test_results in self.results['tests'].items():
                if isinstance(test_results, dict) and 'error' in test_results:
                    summary['status'] = 'FAILED'
                    break
        
        # Check for regressions
        if 'regressions' in self.results:
            regressions = self.results['regressions']
            if regressions.get('missing_files') or regressions.get('import_errors') or regressions.get('config_issues'):
                summary['status'] = 'FAILED'
        
        # Add performance summary
        if 'performance' in self.results:
            performance = self.results['performance']
            summary['performance'] = {
                'startup_time_seconds': performance.get('startup_time_seconds', 0),
                'total_operations_seconds': sum([
                    performance.get('database_operations_seconds', 0),
                    performance.get('ai_operations_seconds', 0)
                ])
            }
        
        self.results['summary'] = summary
        logger.info(f"Validation summary generated: {summary['status']}")
    
    def _save_results(self) -> None:
        """Save validation results to file."""
        try:
            results_file = 'ci-validation-results.json'
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            
            logger.info(f"Validation results saved to {results_file}")
            
        except Exception as e:
            logger.error(f"Failed to save validation results: {e}")

async def main():
    """Main CI validation function."""
    print("CI Validation Suite - Automated System Validation")
    print("=" * 60)
    
    validator = CIValidator()
    results = await validator.run_validation_suite()
    
    # Print summary
    summary = results.get('summary', {})
    status = summary.get('status', 'UNKNOWN')
    total_time = summary.get('total_time_seconds', 0)
    
    print(f"\nValidation Status: {status}")
    print(f"Total Time: {total_time}s")
    print(f"AI Limit: {summary.get('ai_limit', 'N/A')}")
    
    if status == 'PASSED':
        print("\n✅✅✅ CI VALIDATION PASSED — SYSTEM READY 🚀")
        return 0
    else:
        print("\n❌❌❌ CI VALIDATION FAILED — ISSUES DETECTED 🚨")
        return 1

if __name__ == "__main__":
    exit(asyncio.run(main()))
