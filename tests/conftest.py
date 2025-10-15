"""
Pytest configuration and shared fixtures for the Trading Bot test suite.
"""

import pytest
import sys
import os
import tempfile
import sqlite3
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from typing import Dict, List, Any, Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Test configuration
pytest_plugins = ["pytest_mock"]

# Test markers
pytestmark = [
    pytest.mark.unit,
]

# Global test configuration
TEST_CONFIG = {
    "database_path": ":memory:",
    "trading_mode": "DEMO",
    "log_level": "DEBUG",
    "test_data_dir": "tests/fixtures/data"
}

@pytest.fixture(scope="session")
def test_config():
    """Global test configuration."""
    return TEST_CONFIG

@pytest.fixture(scope="session")
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def mock_database():
    """Mock database connection for testing."""
    mock_conn = Mock()
    mock_cursor = Mock()
    mock_conn.cursor.return_value = mock_cursor
    mock_conn.__enter__ = Mock(return_value=mock_conn)
    mock_conn.__exit__ = Mock(return_value=None)
    return mock_conn

@pytest.fixture
def temp_database():
    """Create temporary SQLite database for testing."""
    db_fd, db_path = tempfile.mkstemp()
    conn = sqlite3.connect(db_path)
    
    # Create test schema
    conn.execute("""
        CREATE TABLE positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            entry_price REAL NOT NULL,
            entry_date TEXT NOT NULL,
            exit_price REAL,
            exit_date TEXT,
            pnl REAL,
            status TEXT NOT NULL DEFAULT 'OPEN',
            mode TEXT NOT NULL DEFAULT 'DEMO'
        )
    """)
    
    conn.execute("""
        CREATE TABLE market_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            atr REAL,
            mode TEXT NOT NULL DEFAULT 'DEMO'
        )
    """)
    
    conn.execute("""
        CREATE TABLE predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT NOT NULL,
            symbol TEXT NOT NULL,
            prediction_date TEXT NOT NULL,
            confidence REAL NOT NULL,
            prediction_type TEXT NOT NULL,
            mode TEXT NOT NULL DEFAULT 'DEMO'
        )
    """)
    
    conn.execute("""
        CREATE TABLE confidence_calibration (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_date TEXT NOT NULL,
            model TEXT NOT NULL,
            symbol TEXT NOT NULL,
            raw_confidence REAL NOT NULL,
            calibrated_confidence REAL NOT NULL,
            outcome TEXT,
            pnl REAL,
            exit_date TEXT,
            window_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            mode TEXT NOT NULL DEFAULT 'DEMO'
        )
    """)
    
    conn.commit()
    
    yield conn
    
    conn.close()
    os.unlink(db_path)

@pytest.fixture
def sample_market_data():
    """Sample market data for testing."""
    base_date = datetime.now() - timedelta(days=30)
    data = []
    
    for i in range(30):
        date = base_date + timedelta(days=i)
        data.append({
            "symbol": "AAPL",
            "date": date.strftime("%Y-%m-%d"),
            "open": 100.0 + i * 0.5,
            "high": 102.0 + i * 0.5,
            "low": 98.0 + i * 0.5,
            "close": 100.5 + i * 0.5,
            "volume": 1000000 + i * 10000,
            "atr": 2.0 + i * 0.1,
            "mode": "DEMO"
        })
    
    return data

@pytest.fixture
def sample_positions():
    """Sample position data for testing."""
    return [
        {
            "symbol": "AAPL",
            "quantity": 100,
            "entry_price": 150.0,
            "entry_date": datetime.now().isoformat(),
            "status": "OPEN",
            "mode": "DEMO"
        },
        {
            "symbol": "TSLA",
            "quantity": 50,
            "entry_price": 200.0,
            "entry_date": datetime.now().isoformat(),
            "status": "OPEN",
            "mode": "DEMO"
        }
    ]

@pytest.fixture
def sample_predictions():
    """Sample prediction data for testing."""
    return [
        {
            "model": "trend_following",
            "symbol": "AAPL",
            "prediction_date": datetime.now().isoformat(),
            "confidence": 0.8,
            "prediction_type": "BUY",
            "mode": "DEMO"
        },
        {
            "model": "mean_reversion",
            "symbol": "TSLA",
            "prediction_date": datetime.now().isoformat(),
            "confidence": 0.7,
            "prediction_type": "SELL",
            "mode": "DEMO"
        }
    ]

@pytest.fixture
def sample_calibration_data():
    """Sample calibration data for testing."""
    base_date = datetime.now() - timedelta(days=30)
    data = []
    
    for i in range(20):
        trade_date = base_date + timedelta(days=i)
        data.append({
            "trade_date": trade_date.isoformat(),
            "model": "test_model",
            "symbol": "AAPL",
            "raw_confidence": 0.8,
            "calibrated_confidence": 0.75,
            "outcome": "WIN" if i % 3 != 0 else "LOSS",
            "pnl": 100.0 if i % 3 != 0 else -50.0,
            "exit_date": (trade_date + timedelta(days=1)).isoformat(),
            "window_id": f"20250101_20250131",
            "created_at": trade_date.isoformat(),
            "updated_at": trade_date.isoformat(),
            "mode": "DEMO"
        })
    
    return data

@pytest.fixture
def mock_api_response():
    """Mock API response for testing."""
    return {
        "status": "success",
        "data": {
            "AAPL": {
                "price": 150.0,
                "volume": 1000000,
                "change": 2.5,
                "change_percent": 1.69
            },
            "TSLA": {
                "price": 200.0,
                "volume": 500000,
                "change": -5.0,
                "change_percent": -2.44
            }
        },
        "timestamp": datetime.now().isoformat()
    }

@pytest.fixture
def mock_questrade_client():
    """Mock Questrade client for testing."""
    client = Mock()
    client.get_account_info.return_value = {
        "account_id": "test_account",
        "balance": 100000.0,
        "currency": "CAD"
    }
    client.get_positions.return_value = []
    client.get_market_data.return_value = {
        "AAPL": {"price": 150.0, "volume": 1000000}
    }
    client.place_order.return_value = {
        "order_id": "test_order_123",
        "status": "submitted"
    }
    return client

@pytest.fixture
def mock_risk_manager():
    """Mock risk manager for testing."""
    risk_manager = Mock()
    risk_manager.calculate_position_size.return_value = 1000.0
    risk_manager.calculate_daily_drawdown.return_value = {
        "daily_drawdown": 0.02,
        "within_limit": True,
        "can_trade": True
    }
    risk_manager.calculate_drawdown_scale.return_value = 0.8
    risk_manager.calculate_kelly_fraction.return_value = 0.15
    return risk_manager

@pytest.fixture
def mock_ai_models():
    """Mock AI models for testing."""
    models = {
        "trend_following": Mock(),
        "mean_reversion": Mock(),
        "momentum": Mock()
    }
    
    for model in models.values():
        model.predict.return_value = 0.8
        model.get_confidence.return_value = 0.75
    
    return models

@pytest.fixture
def mock_system_monitor():
    """Mock system monitor for testing."""
    monitor = Mock()
    monitor.get_system_metrics.return_value = {
        "cpu_percent": 45.0,
        "memory_percent": 60.0,
        "disk_percent": 30.0,
        "status": "healthy"
    }
    monitor.start_phase_timer.return_value = None
    monitor.end_phase_timer.return_value = 0.5
    return monitor

@pytest.fixture
def mock_logger():
    """Mock logger for testing."""
    logger = Mock()
    logger.info.return_value = None
    logger.warning.return_value = None
    logger.error.return_value = None
    logger.debug.return_value = None
    return logger

@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    config = Mock()
    config.trading_mode = "DEMO"
    config.database_path = ":memory:"
    config.log_level = "DEBUG"
    config.max_daily_drawdown = 0.05
    config.max_portfolio_risk = 0.20
    config.max_positions = 10
    return config

@pytest.fixture
def mock_feature_flags():
    """Mock feature flags for testing."""
    flags = Mock()
    flags.is_enabled.return_value = True
    flags.get_all_flags.return_value = {
        "advanced_ml_models": True,
        "new_trading_strategy": False,
        "real_time_analytics": True
    }
    return flags

@pytest.fixture
def mock_regime_detector():
    """Mock regime detector for testing."""
    detector = Mock()
    detector.detect_current_regime.return_value = Mock(
        regime="TREND_UP",
        trend_direction="UP",
        volatility_level="MEDIUM",
        confidence=0.8
    )
    return detector

@pytest.fixture
def mock_confidence_calibrator():
    """Mock confidence calibrator for testing."""
    calibrator = Mock()
    calibrator.calibrate_confidence.return_value = 0.75
    calibrator.add_trade_outcome.return_value = None
    calibrator.get_calibration_summary.return_value = {
        "total_trades": 100,
        "wins": 60,
        "losses": 40,
        "calibration_quality": 0.85
    }
    return calibrator

@pytest.fixture
def mock_atr_bracket_manager():
    """Mock ATR bracket manager for testing."""
    manager = Mock()
    manager.calculate_brackets.return_value = Mock(
        entry_price=150.0,
        stop_loss=144.0,
        take_profit=157.5,
        atr=3.0,
        r_multiple=1.5
    )
    return manager

@pytest.fixture
def mock_position_manager():
    """Mock position manager for testing."""
    manager = Mock()
    manager.get_open_positions.return_value = []
    manager.track_position.return_value = None
    manager.close_position.return_value = None
    manager.get_portfolio_summary.return_value = {
        "total_value": 100000.0,
        "total_pnl": 5000.0,
        "daily_pnl": 250.0
    }
    return manager

@pytest.fixture
def mock_execution_engine():
    """Mock execution engine for testing."""
    engine = Mock()
    engine.execute_order.return_value = Mock(
        order_id="test_order_123",
        status="filled",
        filled_price=150.0,
        filled_quantity=100
    )
    return engine

@pytest.fixture
def mock_data_quality_validator():
    """Mock data quality validator for testing."""
    validator = Mock()
    validator.validate_market_data.return_value = {
        "is_valid": True,
        "quality_score": 0.95,
        "issues": []
    }
    validator.validate_indicators.return_value = {
        "is_valid": True,
        "quality_score": 0.90,
        "issues": []
    }
    return validator

@pytest.fixture
def mock_performance_analytics():
    """Mock performance analytics for testing."""
    analytics = Mock()
    analytics.calculate_returns.return_value = {
        "total_return": 0.15,
        "daily_returns": [0.01, -0.02, 0.03, 0.01, -0.01],
        "monthly_returns": [0.05, 0.03, 0.02, 0.04, 0.01]
    }
    analytics.calculate_sharpe_ratio.return_value = 1.25
    analytics.calculate_max_drawdown.return_value = 0.08
    analytics.generate_performance_report.return_value = {
        "total_return": 0.15,
        "sharpe_ratio": 1.25,
        "max_drawdown": 0.08,
        "win_rate": 0.60
    }
    return analytics

@pytest.fixture
def mock_error_tracker():
    """Mock error tracker for testing."""
    tracker = Mock()
    tracker.log_error.return_value = None
    tracker.get_error_summary.return_value = {
        "total_errors": 5,
        "critical_errors": 1,
        "warning_errors": 4,
        "recent_errors": []
    }
    tracker.get_recent_errors.return_value = []
    return tracker

@pytest.fixture
def mock_api_budget_manager():
    """Mock API budget manager for testing."""
    manager = Mock()
    manager.check_budget.return_value = True
    manager.consume_budget.return_value = None
    manager.get_cache.return_value = None
    manager.set_cache.return_value = None
    manager.get_budget_status.return_value = {
        "questrade": {"used": 100, "limit": 1000, "remaining": 900},
        "yahoo_finance": {"used": 50, "limit": 2000, "remaining": 1950}
    }
    return manager

@pytest.fixture
def mock_adaptive_weight_manager():
    """Mock adaptive weight manager for testing."""
    manager = Mock()
    manager.update_weights.return_value = {
        "trend_following": 0.4,
        "mean_reversion": 0.3,
        "momentum": 0.3
    }
    manager.get_current_weights.return_value = {
        "trend_following": 0.4,
        "mean_reversion": 0.3,
        "momentum": 0.3
    }
    manager.calculate_brier_score.return_value = 0.15
    return manager

@pytest.fixture
def mock_regime_policy_manager():
    """Mock regime policy manager for testing."""
    manager = Mock()
    manager.get_policy.return_value = {
        "kelly_sensitivity": 0.8,
        "ensemble_weights": {
            "trend_following": 0.5,
            "mean_reversion": 0.3,
            "momentum": 0.2
        },
        "sl_multiplier": 2.0,
        "tp_multiplier": 1.5
    }
    manager.get_ensemble_weights.return_value = {
        "trend_following": 0.5,
        "mean_reversion": 0.3,
        "momentum": 0.2
    }
    manager.get_kelly_sensitivity.return_value = 0.8
    return manager

@pytest.fixture
def mock_ollama_lifecycle_manager():
    """Mock Ollama lifecycle manager for testing."""
    manager = Mock()
    manager.ensure_models_ready.return_value = True
    manager.pre_warm_models.return_value = None
    manager.cleanup_memory_if_needed.return_value = None
    manager.get_system_status.return_value = {
        "gpu_available": True,
        "gpu_memory_used": 2048,
        "gpu_memory_total": 8192,
        "system_memory_used": 4096,
        "system_memory_total": 16384,
        "models_loaded": ["llama2", "codellama"]
    }
    return manager

@pytest.fixture
def mock_feature_flag_manager():
    """Mock feature flag manager for testing."""
    manager = Mock()
    manager.is_enabled.return_value = True
    manager.set_flag.return_value = None
    manager.get_all_flags.return_value = {
        "advanced_ml_models": True,
        "new_trading_strategy": False,
        "real_time_analytics": True,
        "gpu_acceleration": True
    }
    return manager

@pytest.fixture
def mock_safety_controls():
    """Mock safety controls for testing."""
    controls = Mock()
    controls.check_slo_compliance.return_value = {
        "compliant": True,
        "metrics": {
            "uptime": 0.999,
            "response_time": 0.5,
            "error_rate": 0.001
        }
    }
    controls.get_rollback_history.return_value = []
    controls.can_rollback.return_value = True
    controls.rollback_feature.return_value = True
    return controls

# Test data generators
@pytest.fixture
def generate_market_data():
    """Generate synthetic market data for testing."""
    def _generate(symbols: List[str], days: int = 30) -> Dict[str, List[Dict]]:
        data = {}
        base_date = datetime.now() - timedelta(days=days)
        
        for symbol in symbols:
            data[symbol] = []
            price = 100.0
            
            for i in range(days):
                date = base_date + timedelta(days=i)
                price += (i % 3 - 1) * 0.5  # Simple price movement
                volume = 1000000 + i * 10000
                
                data[symbol].append({
                    "date": date.strftime("%Y-%m-%d"),
                    "price": price,
                    "volume": volume,
                    "open": price - 0.5,
                    "high": price + 1.0,
                    "low": price - 1.0,
                    "close": price
                })
        
        return data
    
    return _generate

@pytest.fixture
def generate_trade_outcomes():
    """Generate synthetic trade outcomes for testing."""
    def _generate(count: int = 100) -> List[Dict]:
        outcomes = []
        symbols = ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN"]
        
        for i in range(count):
            outcomes.append({
                "symbol": symbols[i % len(symbols)],
                "confidence": 0.5 + (i % 5) * 0.1,
                "outcome": "WIN" if i % 3 != 0 else "LOSS",
                "pnl": 100.0 if i % 3 != 0 else -50.0,
                "trade_date": (datetime.now() - timedelta(days=i)).isoformat()
            })
        
        return outcomes
    
    return _generate

# Test utilities
@pytest.fixture
def assert_almost_equal():
    """Utility for approximate equality assertions."""
    def _assert_almost_equal(actual, expected, tolerance=1e-6):
        assert abs(actual - expected) < tolerance
    
    return _assert_almost_equal

@pytest.fixture
def assert_dict_contains():
    """Utility for dictionary containment assertions."""
    def _assert_dict_contains(actual_dict, expected_dict):
        for key, value in expected_dict.items():
            assert key in actual_dict
            assert actual_dict[key] == value
    
    return _assert_dict_contains

# Test markers
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "smoke: mark test as a smoke test"
    )
    config.addinivalue_line(
        "markers", "regression: mark test as a regression test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "api: mark test as requiring API access"
    )
    config.addinivalue_line(
        "markers", "database: mark test as requiring database"
    )

# Test collection hooks
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "smoke" in str(item.fspath):
            item.add_marker(pytest.mark.smoke)
        elif "regression" in str(item.fspath):
            item.add_marker(pytest.mark.regression)
        
        # Add slow marker for tests that take longer than 1 second
        if "performance" in str(item.fspath) or "load" in str(item.fspath):
            item.add_marker(pytest.mark.slow)

# Test reporting
def pytest_html_report_title(report):
    """Set custom title for HTML test report."""
    report.title = "Trading Bot Test Report"

def pytest_configure(config):
    """Configure pytest HTML report."""
    if hasattr(config, '_html'):
        config._html.logfile = "reports/test_report.html"