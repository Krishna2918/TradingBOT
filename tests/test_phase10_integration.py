"""
Phase 10 Integration Tests - CI & Automation

Tests the CI/CD pipeline, automation features, and validation systems.
"""

import pytest
import asyncio
import json
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Add src to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestCIValidation:
    """Test CI validation suite."""
    
    @pytest.fixture
    def ci_validator(self):
        """Create a CI validator instance for testing."""
        from scripts.ci_validation import CIValidator
        return CIValidator()
    
    @pytest.mark.asyncio
    async def test_system_health_check(self, ci_validator):
        """Test system health check functionality."""
        health = await ci_validator._check_system_health()
        
        assert 'system_health' in ci_validator.results
        assert 'python_version' in health
        assert 'memory_available_gb' in health
        assert 'cpu_percent' in health
    
    @pytest.mark.asyncio
    async def test_database_connectivity(self, ci_validator):
        """Test database connectivity validation."""
        with patch('config.database.get_connection') as mock_get_conn:
            mock_conn = MagicMock()
            mock_conn.__enter__ = MagicMock(return_value=mock_conn)
            mock_conn.__exit__ = MagicMock(return_value=None)
            mock_cursor = MagicMock()
            mock_cursor.fetchone.return_value = (1,)
            mock_conn.cursor.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn
            
            result = await ci_validator._test_database_connectivity()
            
            assert result is True
            mock_cursor.execute.assert_called_once_with("SELECT 1")
    
    @pytest.mark.asyncio
    async def test_ai_model_availability(self, ci_validator):
        """Test AI model availability validation."""
        with patch('ai.ollama_lifecycle.get_lifecycle_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_manager.health_check = Mock(return_value=True)
            mock_get_manager.return_value = mock_manager
            
            result = await ci_validator._test_ai_model_availability()
            
            assert result is True
            mock_manager.health_check.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_api_connectivity(self, ci_validator):
        """Test API connectivity validation."""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response
            
            result = await ci_validator._test_api_connectivity()
            
            assert 'yahoo_finance' in result
            assert result['yahoo_finance'] is True
    
    @pytest.mark.asyncio
    async def test_configuration_loading(self, ci_validator):
        """Test configuration loading validation."""
        with patch('config.mode_manager.get_current_mode') as mock_get_mode:
            mock_get_mode.return_value = 'DEMO'
            
            with patch('config.database.DatabaseManager') as mock_db_manager:
                mock_db_manager.return_value = Mock()
                
                result = await ci_validator._test_configuration_loading()
                
                assert result is True
    
    @pytest.mark.asyncio
    async def test_performance_benchmarks(self, ci_validator):
        """Test performance benchmark functionality."""
        with patch('src.monitoring.system_monitor.SystemMonitor') as mock_monitor:
            mock_monitor.return_value = Mock()
            
            await ci_validator._run_performance_benchmarks()
            
            assert 'performance' in ci_validator.results
            assert 'startup_time_seconds' in ci_validator.results['performance']
    
    @pytest.mark.asyncio
    async def test_integration_tests(self, ci_validator):
        """Test integration test functionality."""
        await ci_validator._run_integration_tests()
        
        assert 'tests' in ci_validator.results
        assert 'integration' in ci_validator.results['tests']
    
    @pytest.mark.asyncio
    async def test_regression_detection(self, ci_validator):
        """Test regression detection functionality."""
        await ci_validator._detect_regressions()
        
        assert 'regressions' in ci_validator.results
        assert 'missing_files' in ci_validator.results['regressions']
        assert 'import_errors' in ci_validator.results['regressions']
        assert 'config_issues' in ci_validator.results['regressions']
    
    def test_summary_generation(self, ci_validator):
        """Test summary generation."""
        ci_validator.results['tests'] = {'test1': {'error': 'test error'}}
        ci_validator.results['regressions'] = {'missing_files': ['test.py']}
        
        ci_validator._generate_summary()
        
        assert 'summary' in ci_validator.results
        assert ci_validator.results['summary']['status'] == 'FAILED'
    
    def test_results_saving(self, ci_validator):
        """Test results saving functionality."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_file = f.name
        
        try:
            with patch('builtins.open', create=True) as mock_open:
                mock_file = MagicMock()
                mock_open.return_value.__enter__.return_value = mock_file
                
                ci_validator._save_results()
                
                mock_open.assert_called_once()
                mock_file.write.assert_called_once()
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

class TestPerformanceBenchmark:
    """Test performance benchmark suite."""
    
    @pytest.fixture
    def performance_benchmark(self):
        """Create a performance benchmark instance for testing."""
        from scripts.performance_benchmark import PerformanceBenchmark
        return PerformanceBenchmark()
    
    @pytest.mark.asyncio
    async def test_system_startup_benchmark(self, performance_benchmark):
        """Test system startup benchmark."""
        with patch('src.monitoring.system_monitor.SystemMonitor') as mock_monitor:
            mock_monitor.return_value = Mock()
            
            await performance_benchmark._benchmark_system_startup()
            
            assert 'benchmarks' in performance_benchmark.results
            assert 'system_startup' in performance_benchmark.results['benchmarks']
            assert 'avg_time_seconds' in performance_benchmark.results['benchmarks']['system_startup']
    
    @pytest.mark.asyncio
    async def test_database_performance_benchmark(self, performance_benchmark):
        """Test database performance benchmark."""
        with patch('config.database.get_connection') as mock_get_conn:
            mock_conn = MagicMock()
            mock_conn.__enter__ = MagicMock(return_value=mock_conn)
            mock_conn.__exit__ = MagicMock(return_value=None)
            mock_cursor = MagicMock()
            mock_cursor.fetchone.return_value = (1,)
            mock_cursor.fetchall.return_value = [('table1',), ('table2',)]
            mock_conn.cursor.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn
            
            await performance_benchmark._benchmark_database_performance()
            
            assert 'benchmarks' in performance_benchmark.results
            assert 'database_performance' in performance_benchmark.results['benchmarks']
    
    @pytest.mark.asyncio
    async def test_ai_performance_benchmark(self, performance_benchmark):
        """Test AI performance benchmark."""
        with patch('ai.multi_model.MultiModelManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager.get_all_model_configs.return_value = {'model1': Mock()}
            mock_manager.get_model_weights.return_value = {'model1': 0.5}
            mock_manager.get_adaptive_weights.return_value = {'model1': 0.6}
            mock_manager.check_model_availability.return_value = {'model1': True}
            mock_manager_class.return_value = mock_manager
            
            await performance_benchmark._benchmark_ai_performance()
            
            assert 'benchmarks' in performance_benchmark.results
            assert 'ai_performance' in performance_benchmark.results['benchmarks']
    
    @pytest.mark.asyncio
    async def test_memory_usage_benchmark(self, performance_benchmark):
        """Test memory usage benchmark."""
        with patch('psutil.Process') as mock_process:
            mock_process.return_value.memory_info.return_value.rss = 100 * 1024 * 1024  # 100MB
            
            await performance_benchmark._benchmark_memory_usage()
            
            assert 'benchmarks' in performance_benchmark.results
            assert 'memory_usage' in performance_benchmark.results['benchmarks']
    
    @pytest.mark.asyncio
    async def test_end_to_end_benchmark(self, performance_benchmark):
        """Test end-to-end benchmark."""
        with patch('src.monitoring.system_monitor.SystemMonitor') as mock_monitor:
            with patch('config.database.get_connection') as mock_get_conn:
                with patch('ai.multi_model.MultiModelManager') as mock_manager_class:
                    with patch('src.trading.risk.RiskManager') as mock_risk:
                        with patch('src.dashboard.connector.DashboardConnector') as mock_connector:
                            mock_conn = MagicMock()
                            mock_conn.__enter__ = MagicMock(return_value=mock_conn)
                            mock_conn.__exit__ = MagicMock(return_value=None)
                            mock_cursor = MagicMock()
                            mock_cursor.fetchone.return_value = (1,)
                            mock_conn.cursor.return_value = mock_cursor
                            mock_get_conn.return_value = mock_conn
                            
                            mock_manager = Mock()
                            mock_manager.get_all_model_configs.return_value = {'model1': Mock()}
                            mock_manager.get_model_weights.return_value = {'model1': 0.5}
                            mock_manager_class.return_value = mock_manager
                            
                            await performance_benchmark._benchmark_end_to_end()
                            
                            assert 'benchmarks' in performance_benchmark.results
                            assert 'end_to_end' in performance_benchmark.results['benchmarks']
    
    def test_summary_generation(self, performance_benchmark):
        """Test summary generation."""
        performance_benchmark.results['benchmarks'] = {
            'test1': {'avg_time_seconds': 1.0},
            'test2': {'error': 'test error'}
        }
        
        performance_benchmark._generate_summary()
        
        assert 'summary' in performance_benchmark.results
        assert performance_benchmark.results['summary']['successful_benchmarks'] == 1
        assert performance_benchmark.results['summary']['failed_benchmarks'] == 1

class TestPreCommitValidation:
    """Test pre-commit validation functionality."""
    
    def test_validate_imports(self):
        """Test import validation."""
        from scripts.pre_commit_validation import validate_imports
        
        with patch('builtins.__import__') as mock_import:
            mock_import.side_effect = ImportError("Test error")
            
            errors = validate_imports()
            
            assert len(errors) > 0
            assert any("Failed to import" in error for error in errors)
    
    def test_validate_file_structure(self):
        """Test file structure validation."""
        from scripts.pre_commit_validation import validate_file_structure
        
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = False
            
            errors = validate_file_structure()
            
            assert len(errors) > 0
            assert any("Required directory missing" in error for error in errors)
    
    def test_validate_configuration(self):
        """Test configuration validation."""
        from scripts.pre_commit_validation import validate_configuration
        
        with patch('config.mode_manager.get_current_mode') as mock_get_mode:
            mock_get_mode.return_value = None
            
            errors = validate_configuration()
            
            assert len(errors) > 0
            assert any("Mode manager not returning valid mode" in error for error in errors)
    
    def test_validate_database_schema(self):
        """Test database schema validation."""
        from scripts.pre_commit_validation import validate_database_schema
        
        with patch('config.database.get_connection') as mock_get_conn:
            mock_conn = MagicMock()
            mock_conn.__enter__ = MagicMock(return_value=mock_conn)
            mock_conn.__exit__ = MagicMock(return_value=None)
            mock_cursor = MagicMock()
            mock_cursor.fetchone.return_value = (2,)  # Wrong result
            mock_conn.cursor.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn
            
            errors = validate_database_schema()
            
            assert len(errors) > 0
            assert any("Database connectivity test failed" in error for error in errors)
    
    def test_validate_ai_system(self):
        """Test AI system validation."""
        from scripts.pre_commit_validation import validate_ai_system
        
        with patch('ai.multi_model.MultiModelManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager.get_all_model_configs.return_value = {}
            mock_manager_class.return_value = mock_manager
            
            errors = validate_ai_system()
            
            assert len(errors) > 0
            assert any("No AI model configurations found" in error for error in errors)
    
    def test_validate_trading_system(self):
        """Test trading system validation."""
        from scripts.pre_commit_validation import validate_trading_system
        
        with patch('src.trading.risk.RiskManager') as mock_risk_class:
            mock_risk_class.side_effect = Exception("Test error")
            
            errors = validate_trading_system()
            
            assert len(errors) > 0
            assert any("Trading system validation failed" in error for error in errors)
    
    def test_validate_dashboard(self):
        """Test dashboard validation."""
        from scripts.pre_commit_validation import validate_dashboard
        
        with patch('src.dashboard.connector.DashboardConnector') as mock_connector_class:
            mock_connector_class.side_effect = Exception("Test error")
            
            errors = validate_dashboard()
            
            assert len(errors) > 0
            assert any("Dashboard validation failed" in error for error in errors)

class TestAPIKeyCheck:
    """Test API key check functionality."""
    
    def test_find_potential_api_keys(self):
        """Test finding potential API keys."""
        from scripts.check_api_keys import find_potential_api_keys
        
        with patch('os.walk') as mock_walk:
            mock_walk.return_value = [
                ('.', [], ['test.py'])
            ]
            
            with patch('builtins.open', create=True) as mock_open:
                mock_file = MagicMock()
                mock_file.read.return_value = 'api_key = "sk-1234567890abcdef"'
                mock_open.return_value.__enter__.return_value = mock_file
                
                potential_keys = find_potential_api_keys()
                
                assert len(potential_keys) > 0
                assert potential_keys[0]['key'] == 'sk-1234567890abcdef'
    
    def test_is_placeholder_key(self):
        """Test placeholder key detection."""
        from scripts.check_api_keys import is_placeholder_key
        
        assert is_placeholder_key('your_api_key') is True
        assert is_placeholder_key('example_key') is True
        assert is_placeholder_key('placeholder') is True
        assert is_placeholder_key('real_api_key_12345') is False
    
    def test_check_environment_variables(self):
        """Test environment variable checking."""
        from scripts.check_api_keys import check_environment_variables
        
        with patch('os.walk') as mock_walk:
            mock_walk.return_value = [
                ('.', [], ['test.py'])
            ]
            
            with patch('builtins.open', create=True) as mock_open:
                mock_file = MagicMock()
                mock_file.read.return_value = 'os.environ["API_KEY"] = "real_secret_key"'
                mock_open.return_value.__enter__.return_value = mock_file
                
                issues = check_environment_variables()
                
                assert len(issues) > 0
                assert any('API_KEY' in issue for issue in issues)
    
    def test_is_sensitive_env_var(self):
        """Test sensitive environment variable detection."""
        from scripts.check_api_keys import is_sensitive_env_var
        
        assert is_sensitive_env_var('API_KEY') is True
        assert is_sensitive_env_var('SECRET_TOKEN') is True
        assert is_sensitive_env_var('DATABASE_URL') is False
        assert is_sensitive_env_var('DEBUG_MODE') is False
    
    def test_is_placeholder_value(self):
        """Test placeholder value detection."""
        from scripts.check_api_keys import is_placeholder_value
        
        assert is_placeholder_value('your_api_key') is True
        assert is_placeholder_value('placeholder') is True
        assert is_placeholder_value('real_secret_value') is False
    
    def test_check_gitignore(self):
        """Test gitignore checking."""
        from scripts.check_api_keys import check_gitignore
        
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = True
            
            with patch('builtins.open', create=True) as mock_open:
                mock_file = MagicMock()
                mock_file.read.return_value = '# No sensitive patterns'
                mock_open.return_value.__enter__.return_value = mock_file
                
                issues = check_gitignore()
                
                assert len(issues) > 0
                assert any('Missing pattern' in issue for issue in issues)

class TestValidationReport:
    """Test validation report generation."""
    
    def test_generate_validation_report(self):
        """Test validation report generation."""
        from scripts.generate_validation_report import generate_validation_report
        
        # Test with valid results
        test_results = {
            'timestamp': '2024-01-01T00:00:00',
            'summary': {
                'status': 'PASSED',
                'total_time_seconds': 10.5,
                'ai_limit': 20
            },
            'system_health': {
                'python_version': '3.11.0',
                'memory_available_gb': 8.0,
                'cpu_percent': 25.0
            },
            'tests': {
                'core_functionality': {
                    'database_connectivity': True,
                    'ai_model_availability': True
                }
            },
            'performance': {
                'startup_time_seconds': 1.5,
                'database_operations_seconds': 0.5
            },
            'regressions': {
                'missing_files': [],
                'import_errors': [],
                'config_issues': []
            }
        }
        
        with patch('builtins.open', create=True) as mock_open:
            mock_file = MagicMock()
            mock_file.read.return_value = json.dumps(test_results)
            mock_open.return_value.__enter__.return_value = mock_file
            
            html = generate_validation_report()
            
            assert '<!DOCTYPE html>' in html
            assert 'Trading System Validation Report' in html
            assert 'PASSED' in html
    
    def test_generate_error_report(self):
        """Test error report generation."""
        from scripts.generate_validation_report import generate_error_report
        
        html = generate_error_report("Test error message")
        
        assert '<!DOCTYPE html>' in html
        assert 'Validation Report Error' in html
        assert 'Test error message' in html

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
