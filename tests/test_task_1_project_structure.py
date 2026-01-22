"""
Comprehensive tests for Task 1: Set up project structure and core interfaces

This test suite validates all components created in Task 1 including:
- Project structure and imports
- Configuration management
- Core interfaces
- Exception handling
- Utility functions
- Resource management
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import shutil
import time
import threading
from pathlib import Path
from unittest.mock import Mock, patch

# Import all components from Task 1
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from portfolio_optimization.config.settings import (
    ConfigManager, APILimits, OptimizationSettings, 
    RiskConstraints, PerformanceSettings, DatabaseConfig
)
from portfolio_optimization.interfaces.optimization_algorithm import IOptimizationAlgorithm
from portfolio_optimization.interfaces.data_provider import IDataProvider
from portfolio_optimization.interfaces.risk_model import IRiskModel
from portfolio_optimization.exceptions.optimization_errors import (
    OptimizationError, DataError, ConstraintViolationError, 
    ConvergenceError, ResourceError, ValidationError
)
from portfolio_optimization.utils.logger import get_logger, setup_logging, OptimizationLogger
from portfolio_optimization.utils.cache_manager import CacheManager, get_cache_manager
from portfolio_optimization.utils.resource_monitor import ResourceMonitor, get_resource_monitor
from portfolio_optimization.utils.math_utils import (
    ensure_positive_definite, validate_correlation_matrix, normalize_weights,
    calculate_portfolio_metrics, calculate_risk_contribution, calculate_var,
    calculate_expected_shortfall, calculate_maximum_drawdown
)


class TestProjectStructure:
    """Test project structure and imports"""
    
    def test_package_imports(self):
        """Test that all main package components can be imported"""
        import portfolio_optimization
        
        # Test version and metadata
        assert hasattr(portfolio_optimization, '__version__')
        assert hasattr(portfolio_optimization, '__author__')
        
        # Test core components are available
        assert hasattr(portfolio_optimization, 'PortfolioOptimizer')
        assert hasattr(portfolio_optimization, 'MarketDataProcessor')
        assert hasattr(portfolio_optimization, 'CorrelationAnalyzer')
        
        # Test interfaces are available
        assert hasattr(portfolio_optimization, 'IOptimizationAlgorithm')
        assert hasattr(portfolio_optimization, 'IDataProvider')
        assert hasattr(portfolio_optimization, 'IRiskModel')
        
        # Test exceptions are available
        assert hasattr(portfolio_optimization, 'OptimizationError')
        assert hasattr(portfolio_optimization, 'DataError')
    
    def test_directory_structure(self):
        """Test that required directories exist"""
        base_path = Path('src/portfolio_optimization')
        
        required_dirs = [
            'config',
            'interfaces', 
            'exceptions',
            'utils'
        ]
        
        for dir_name in required_dirs:
            dir_path = base_path / dir_name
            assert dir_path.exists(), f"Directory {dir_path} should exist"
            assert (dir_path / '__init__.py').exists(), f"__init__.py missing in {dir_path}"


class TestConfigurationManager:
    """Test configuration management system"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, 'test_config.json')
    
    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_config_manager_initialization(self):
        """Test ConfigManager initialization"""
        config = ConfigManager(self.config_file)
        
        # Test that default configuration is created
        assert isinstance(config.api_limits, APILimits)
        assert isinstance(config.optimization, OptimizationSettings)
        assert isinstance(config.risk_constraints, RiskConstraints)
        assert isinstance(config.performance, PerformanceSettings)
        assert isinstance(config.database, DatabaseConfig)
    
    def test_api_limits_configuration(self):
        """Test API limits configuration"""
        config = ConfigManager(self.config_file)
        api_limits = config.api_limits
        
        assert api_limits.alpha_vantage_calls_per_minute == 74
        assert api_limits.max_concurrent_requests == 5
        assert api_limits.request_timeout == 30
        assert api_limits.retry_attempts == 3
        assert api_limits.backoff_factor == 1.5
    
    def test_risk_constraints_configuration(self):
        """Test risk constraints configuration"""
        config = ConfigManager(self.config_file)
        risk_constraints = config.risk_constraints
        
        # Test position limits
        assert risk_constraints.max_position_size == 0.05
        assert risk_constraints.max_sector_concentration == 0.25
        assert risk_constraints.max_leverage == 1.0
        
        # Test 4-bucket allocation
        assert risk_constraints.penny_stocks_allocation == 0.02
        assert risk_constraints.futures_options_allocation == 0.05
        assert risk_constraints.core_allocation == 0.90
        assert risk_constraints.sip_allocation == 0.01
        
        # Verify allocations sum to 1
        total_allocation = (
            risk_constraints.penny_stocks_allocation +
            risk_constraints.futures_options_allocation +
            risk_constraints.core_allocation +
            risk_constraints.sip_allocation
        )
        assert abs(total_allocation - 1.0) < 1e-6
    
    def test_config_persistence(self):
        """Test configuration save and load"""
        config = ConfigManager(self.config_file)
        
        # Modify a setting
        config.update_setting('optimization', 'risk_aversion', 5.0)
        config.save_config()
        
        # Create new config manager and verify setting persisted
        config2 = ConfigManager(self.config_file)
        assert config2.get_setting('optimization', 'risk_aversion') == 5.0


class TestInterfaces:
    """Test core interfaces"""
    
    def test_optimization_algorithm_interface(self):
        """Test IOptimizationAlgorithm interface"""
        
        class MockOptimizer(IOptimizationAlgorithm):
            @property
            def algorithm_name(self):
                return "mock_optimizer"
            
            @property
            def requires_covariance_matrix(self):
                return True
            
            @property
            def supports_factor_model(self):
                return False
            
            def optimize(self, returns, constraints, covariance_matrix=None, **kwargs):
                return np.ones(len(returns)) / len(returns)
            
            def get_optimization_info(self):
                return {"algorithm_name": "mock_optimizer", "convergence_status": "success"}
            
            def validate_inputs(self, returns, constraints, covariance_matrix=None):
                return True
            
            def get_supported_constraints(self):
                return ["position_limits", "sector_limits"]
        
        optimizer = MockOptimizer()
        
        # Test interface methods
        returns = np.array([0.1, 0.08, 0.12])
        constraints = {"position_limits": 0.5}
        
        weights = optimizer.optimize(returns, constraints)
        assert len(weights) == len(returns)
        assert abs(np.sum(weights) - 1.0) < 1e-6
        
        info = optimizer.get_optimization_info()
        assert "algorithm_name" in info
        assert "convergence_status" in info
        
        assert optimizer.validate_inputs(returns, constraints)
        assert "position_limits" in optimizer.get_supported_constraints()
    
    def test_data_provider_interface(self):
        """Test IDataProvider interface"""
        
        class MockDataProvider(IDataProvider):
            @property
            def provider_name(self):
                return "mock_provider"
            
            @property
            def rate_limit_per_minute(self):
                return 60
            
            @property
            def supported_frequencies(self):
                return ["daily", "hourly"]
            
            def get_market_data(self, symbols, start_date=None, end_date=None, frequency='daily'):
                dates = pd.date_range('2023-01-01', periods=10, freq='D')
                data = pd.DataFrame({
                    'close': np.random.randn(10) * 0.02 + 100
                }, index=dates)
                return data
            
            def get_factor_data(self, start_date=None, end_date=None):
                dates = pd.date_range('2023-01-01', periods=10, freq='D')
                return pd.DataFrame({
                    'momentum': np.random.randn(10) * 0.01,
                    'value': np.random.randn(10) * 0.01
                }, index=dates)
            
            def get_risk_free_rate(self, start_date=None, end_date=None):
                dates = pd.date_range('2023-01-01', periods=10, freq='D')
                return pd.Series(0.02 / 252, index=dates)
            
            def get_sector_data(self, symbols):
                return {symbol: "Technology" for symbol in symbols}
            
            def get_market_cap_data(self, symbols):
                return {symbol: 1e9 for symbol in symbols}
            
            def get_trading_volume_data(self, symbols, lookback_days=30):
                return {symbol: 1e6 for symbol in symbols}
            
            def is_data_stale(self, symbol, max_age_minutes=60):
                return False
            
            def get_data_quality_score(self, symbols):
                return {symbol: 0.95 for symbol in symbols}
        
        provider = MockDataProvider()
        
        # Test interface methods
        symbols = ['AAPL', 'GOOGL', 'MSFT']
        
        market_data = provider.get_market_data(symbols)
        assert isinstance(market_data, pd.DataFrame)
        assert 'close' in market_data.columns
        
        factor_data = provider.get_factor_data()
        assert isinstance(factor_data, pd.DataFrame)
        assert 'momentum' in factor_data.columns
        
        sector_data = provider.get_sector_data(symbols)
        assert len(sector_data) == len(symbols)
        
        assert not provider.is_data_stale('AAPL')
    
    def test_risk_model_interface(self):
        """Test IRiskModel interface"""
        
        class MockRiskModel(IRiskModel):
            @property
            def model_name(self):
                return "mock_risk_model"
            
            @property
            def supported_metrics(self):
                return ["volatility", "var", "sharpe_ratio"]
            
            @property
            def requires_historical_data(self):
                return True
            
            def calculate_risk(self, weights, returns=None, covariance_matrix=None):
                if covariance_matrix is not None:
                    return np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
                return 0.15  # Default risk
            
            def get_risk_decomposition(self, weights, asset_names=None):
                return {
                    "individual_risk": weights * 0.1,
                    "marginal_risk": np.ones(len(weights)) * 0.05,
                    "diversification_ratio": 1.2
                }
            
            def calculate_var(self, weights, confidence_level=0.95, time_horizon=1):
                return -0.05  # 5% VaR
            
            def calculate_expected_shortfall(self, weights, confidence_level=0.95, time_horizon=1):
                return -0.08  # 8% ES
            
            def calculate_beta(self, weights, market_returns=None):
                return 1.1
            
            def calculate_tracking_error(self, weights, benchmark_weights):
                return 0.03
            
            def validate_risk_constraints(self, weights, constraints):
                violations = []
                if np.max(weights) > constraints.get('max_position_size', 1.0):
                    violations.append("Position size limit exceeded")
                return len(violations) == 0, violations
            
            def estimate_covariance_matrix(self, returns, method='sample'):
                n_assets = returns.shape[1]
                return np.eye(n_assets) * 0.01  # Simple diagonal matrix
        
        risk_model = MockRiskModel()
        
        # Test interface methods
        weights = np.array([0.4, 0.3, 0.3])
        covariance_matrix = np.eye(3) * 0.01
        
        risk = risk_model.calculate_risk(weights, covariance_matrix=covariance_matrix)
        assert risk > 0
        
        risk_decomp = risk_model.get_risk_decomposition(weights)
        assert "individual_risk" in risk_decomp
        assert "diversification_ratio" in risk_decomp
        
        var = risk_model.calculate_var(weights)
        assert var < 0  # VaR should be negative
        
        constraints = {"max_position_size": 0.5}
        is_valid, violations = risk_model.validate_risk_constraints(weights, constraints)
        assert isinstance(is_valid, bool)
        assert isinstance(violations, list)


class TestExceptions:
    """Test custom exception classes"""
    
    def test_optimization_error_base(self):
        """Test base OptimizationError class"""
        error = OptimizationError(
            "Test error", 
            error_code="TEST_001",
            context={"param": "value"}
        )
        
        assert str(error) == "[TEST_001] Test error Context: {'param': 'value'}"
        assert error.message == "Test error"
        assert error.error_code == "TEST_001"
        assert error.context == {"param": "value"}
    
    def test_data_error(self):
        """Test DataError exception"""
        error = DataError(
            "Data not available",
            data_source="alpha_vantage",
            symbols=["AAPL", "GOOGL"]
        )
        
        assert error.data_source == "alpha_vantage"
        assert error.symbols == ["AAPL", "GOOGL"]
        assert "alpha_vantage" in str(error)
    
    def test_constraint_violation_error(self):
        """Test ConstraintViolationError exception"""
        error = ConstraintViolationError(
            "Position limit exceeded",
            constraint_type="position_limit",
            violated_constraints=["AAPL: 0.6 > 0.5"]
        )
        
        assert error.constraint_type == "position_limit"
        assert len(error.violated_constraints) == 1
    
    def test_convergence_error(self):
        """Test ConvergenceError exception"""
        error = ConvergenceError(
            "Optimization failed to converge",
            algorithm_name="mean_variance",
            iterations=100,
            objective_value=0.5
        )
        
        assert error.algorithm_name == "mean_variance"
        assert error.iterations == 100
        assert error.objective_value == 0.5


class TestUtilities:
    """Test utility functions"""
    
    def test_logger_setup(self):
        """Test logging setup"""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, 'test.log')
            setup_logging(log_level='INFO', log_file=log_file, console_output=False)
            
            logger = get_logger('test_component')
            logger.info("Test message")
            
            # Verify log file was created and contains message
            assert os.path.exists(log_file)
            with open(log_file, 'r') as f:
                content = f.read()
                assert "Test message" in content
    
    def test_optimization_logger(self):
        """Test OptimizationLogger functionality"""
        logger = OptimizationLogger('test_component')
        
        # Test operation timing
        logger.start_operation('test_operation', param1='value1')
        time.sleep(0.01)  # Small delay
        logger.end_operation('test_operation', success=True, result='success')
        
        # Test performance metrics logging
        logger.log_performance_metrics({'sharpe_ratio': 1.5, 'volatility': 0.15})
        
        # Test API call logging
        logger.log_api_call('alpha_vantage', 'daily', ['AAPL'], True)
    
    def test_cache_manager(self):
        """Test cache management"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = CacheManager(temp_dir)
            
            # Test basic caching
            cache.set('test_key', 'test_value', ttl_seconds=60)
            assert cache.get('test_key') == 'test_value'
            
            # Test cache miss
            assert cache.get('nonexistent_key', 'default') == 'default'
            
            # Test cache deletion
            assert cache.delete('test_key')
            assert cache.get('test_key') is None
            
            # Test cache statistics
            stats = cache.get_stats()
            assert 'hits' in stats
            assert 'misses' in stats
            assert 'hit_rate' in stats
    
    def test_cache_decorator(self):
        """Test cache decorator functionality"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = CacheManager(temp_dir)
            
            call_count = 0
            
            @cache.cached(ttl_seconds=60)
            def expensive_function(x, y):
                nonlocal call_count
                call_count += 1
                return x + y
            
            # First call should execute function
            result1 = expensive_function(1, 2)
            assert result1 == 3
            assert call_count == 1
            
            # Second call should use cache
            result2 = expensive_function(1, 2)
            assert result2 == 3
            assert call_count == 1  # Should not increment
    
    def test_resource_monitor(self):
        """Test resource monitoring"""
        monitor = ResourceMonitor(monitoring_interval=1)
        
        # Test API call recording
        monitor.record_api_call('alpha_vantage', 'daily')
        assert monitor.can_make_api_call('alpha_vantage')
        
        # Test resource usage
        usage = monitor.get_resource_usage()
        assert 'cpu_percent' in usage
        assert 'memory_mb' in usage
        assert 'api_calls_per_minute' in usage
        
        # Test resource availability
        assert monitor.is_resource_available('cpu')
        assert monitor.is_resource_available('memory')
        
        # Test recommendations
        recommendations = monitor.get_optimization_recommendations()
        assert isinstance(recommendations, dict)
    
    def test_math_utilities(self):
        """Test mathematical utility functions"""
        
        # Test positive definite matrix
        matrix = np.array([[1, 0.5], [0.5, 1]])
        pd_matrix = ensure_positive_definite(matrix)
        eigenvals = np.linalg.eigvals(pd_matrix)
        assert np.all(eigenvals > 0)
        
        # Test correlation matrix validation
        corr_matrix = np.array([[1, 0.5], [0.5, 1]])
        is_valid, message = validate_correlation_matrix(corr_matrix)
        assert is_valid
        assert "Valid" in message
        
        # Test weight normalization
        weights = np.array([0.3, 0.5, 0.2])
        normalized = normalize_weights(weights)
        assert abs(np.sum(normalized) - 1.0) < 1e-6
        
        # Test portfolio metrics calculation
        weights = np.array([0.4, 0.3, 0.3])
        returns = np.array([0.1, 0.08, 0.12])
        covariance = np.eye(3) * 0.01
        
        metrics = calculate_portfolio_metrics(weights, returns, covariance)
        assert 'expected_return' in metrics
        assert 'volatility' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'diversification_ratio' in metrics
        
        # Test risk contribution
        risk_contrib = calculate_risk_contribution(weights, covariance)
        assert len(risk_contrib) == len(weights)
        assert abs(np.sum(risk_contrib) - 1.0) < 1e-6
        
        # Test VaR calculation
        returns_series = np.random.normal(0.001, 0.02, 1000)
        var = calculate_var(returns_series, confidence_level=0.95)
        assert var < 0  # VaR should be negative
        
        # Test Expected Shortfall
        es = calculate_expected_shortfall(returns_series, confidence_level=0.95)
        assert es < var  # ES should be more negative than VaR
        
        # Test maximum drawdown
        returns_series = np.array([0.01, -0.02, 0.015, -0.03, 0.02])
        max_dd = calculate_maximum_drawdown(returns_series)
        assert max_dd <= 0  # Drawdown should be negative or zero


class TestIntegration:
    """Integration tests for Task 1 components"""
    
    def test_component_integration(self):
        """Test that all components work together"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize components
            config = ConfigManager(os.path.join(temp_dir, 'config.json'))
            cache = CacheManager(os.path.join(temp_dir, 'cache'))
            monitor = ResourceMonitor()
            
            # Test configuration affects other components
            assert config.api_limits.alpha_vantage_calls_per_minute == 74
            
            # Test cache and resource monitor work together
            cache.set('test_data', {'prices': [100, 101, 102]})
            monitor.record_api_call('alpha_vantage', 'daily')
            
            # Verify data integrity
            cached_data = cache.get('test_data')
            assert cached_data['prices'] == [100, 101, 102]
            
            # Test resource monitoring
            assert monitor.can_make_api_call('alpha_vantage')
    
    def test_error_handling_integration(self):
        """Test error handling across components"""
        
        # Test that exceptions propagate correctly
        with pytest.raises(DataError):
            raise DataError("Test data error", data_source="test")
        
        with pytest.raises(ConstraintViolationError):
            raise ConstraintViolationError("Test constraint error")
        
        with pytest.raises(ConvergenceError):
            raise ConvergenceError("Test convergence error")
    
    def test_thread_safety(self):
        """Test thread safety of shared components"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = CacheManager(temp_dir)
            monitor = ResourceMonitor()
            
            def worker_function(worker_id):
                for i in range(10):
                    cache.set(f'worker_{worker_id}_key_{i}', f'value_{i}')
                    monitor.record_api_call('test_provider', 'endpoint')
                    time.sleep(0.001)
            
            # Create multiple threads
            threads = []
            for i in range(5):
                thread = threading.Thread(target=worker_function, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            # Verify no data corruption
            stats = cache.get_stats()
            assert stats['memory_entries'] > 0


if __name__ == '__main__':
    # Run the tests
    pytest.main([__file__, '-v', '--tb=short'])