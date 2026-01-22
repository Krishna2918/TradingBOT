# Testing Framework Documentation

## Overview

This document describes the comprehensive testing framework for the Trading Bot system, including test organization, coverage analysis, and automated testing pipelines.

## Test Structure

```
tests/
├── unit/                    # Unit tests for individual components
│   ├── config/             # Configuration module tests
│   ├── data_pipeline/      # Data pipeline tests
│   ├── ai/                 # AI/ML module tests
│   ├── trading/            # Trading engine tests
│   ├── monitoring/         # Monitoring system tests
│   └── utils/              # Utility function tests
├── integration/            # Integration tests
│   ├── api_integration/    # External API integration tests
│   ├── database/           # Database integration tests
│   ├── workflow/           # End-to-end workflow tests
│   └── performance/        # Performance integration tests
├── smoke/                  # Smoke tests for quick validation
├── regression/             # Regression tests
├── fixtures/               # Test data and fixtures
├── helpers/                # Test helper functions
└── reports/                # Test reports and coverage
```

## Test Categories

### 1. Unit Tests

**Purpose**: Test individual components in isolation

**Coverage**:
- All public methods and functions
- Error handling and edge cases
- Input validation
- Business logic

**Example**:
```python
# tests/unit/trading/test_risk.py
import pytest
from trading.risk import RiskManager

class TestRiskManager:
    def test_calculate_position_size(self):
        risk_manager = RiskManager()
        size = risk_manager.calculate_position_size("AAPL", 150.0, 0.8)
        assert size > 0
        assert size <= 10000  # Max position limit
    
    def test_daily_drawdown_calculation(self):
        risk_manager = RiskManager()
        drawdown = risk_manager.calculate_daily_drawdown()
        assert "daily_drawdown" in drawdown
        assert 0 <= drawdown["daily_drawdown"] <= 1
```

### 2. Integration Tests

**Purpose**: Test component interactions and data flow

**Coverage**:
- API integrations
- Database operations
- Workflow execution
- Cross-module communication

**Example**:
```python
# tests/integration/test_trading_workflow.py
import pytest
from workflows.trading_cycle import TradingCycle

class TestTradingWorkflow:
    def test_complete_trading_cycle(self):
        cycle = TradingCycle()
        result = cycle.execute_cycle()
        assert result["status"] == "success"
        assert "positions" in result
        assert "performance" in result
```

### 3. Smoke Tests

**Purpose**: Quick validation that system is working

**Coverage**:
- System startup
- Basic functionality
- Critical paths
- Health checks

**Example**:
```python
# tests/smoke/test_system_health.py
import pytest
from monitoring.system_monitor import SystemMonitor

class TestSystemHealth:
    def test_system_startup(self):
        monitor = SystemMonitor()
        health = monitor.get_system_metrics()
        assert health["status"] == "healthy"
        assert health["cpu_percent"] < 90
        assert health["memory_percent"] < 90
```

### 4. Regression Tests

**Purpose**: Ensure fixes don't break existing functionality

**Coverage**:
- Previously fixed bugs
- Critical user workflows
- Performance benchmarks
- Data integrity

## Test Execution

### Running Tests

#### All Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/trading/test_risk.py

# Run specific test class
pytest tests/unit/trading/test_risk.py::TestRiskManager

# Run specific test method
pytest tests/unit/trading/test_risk.py::TestRiskManager::test_calculate_position_size
```

#### By Category
```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# Smoke tests only
pytest tests/smoke/

# Regression tests only
pytest tests/regression/
```

#### By Module
```bash
# Test specific module
pytest tests/unit/trading/
pytest tests/unit/ai/
pytest tests/unit/config/
```

### Test Configuration

#### pytest.ini
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --strict-markers
    --strict-config
    --verbose
    --tb=short
    --cov=src
    --cov-report=term-missing
    --cov-report=html:reports/coverage
markers =
    unit: Unit tests
    integration: Integration tests
    smoke: Smoke tests
    regression: Regression tests
    slow: Slow running tests
    api: Tests requiring API access
    database: Tests requiring database
```

#### conftest.py
```python
import pytest
import sys
import os
from unittest.mock import Mock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

@pytest.fixture
def mock_database():
    """Mock database for testing"""
    return Mock()

@pytest.fixture
def sample_market_data():
    """Sample market data for testing"""
    return {
        "AAPL": {
            "price": 150.0,
            "volume": 1000000,
            "atr": 3.0
        }
    }

@pytest.fixture
def risk_manager():
    """Risk manager instance for testing"""
    from trading.risk import RiskManager
    return RiskManager()
```

## Test Data Management

### Fixtures

#### Database Fixtures
```python
# tests/fixtures/database.py
import sqlite3
import tempfile
import os

@pytest.fixture
def temp_database():
    """Create temporary database for testing"""
    db_fd, db_path = tempfile.mkstemp()
    conn = sqlite3.connect(db_path)
    
    # Create test schema
    conn.execute("""
        CREATE TABLE positions (
            id INTEGER PRIMARY KEY,
            symbol TEXT,
            quantity INTEGER,
            entry_price REAL
        )
    """)
    
    yield conn
    conn.close()
    os.unlink(db_path)
```

#### Market Data Fixtures
```python
# tests/fixtures/market_data.py
@pytest.fixture
def sample_positions():
    """Sample position data"""
    return [
        {"symbol": "AAPL", "quantity": 100, "entry_price": 150.0},
        {"symbol": "TSLA", "quantity": 50, "entry_price": 200.0}
    ]

@pytest.fixture
def mock_api_response():
    """Mock API response"""
    return {
        "status": "success",
        "data": {
            "AAPL": {"price": 150.0, "volume": 1000000}
        }
    }
```

### Test Data Generation

#### Data Generators
```python
# tests/helpers/data_generators.py
import random
from datetime import datetime, timedelta

def generate_market_data(symbols, days=30):
    """Generate synthetic market data"""
    data = {}
    base_date = datetime.now() - timedelta(days=days)
    
    for symbol in symbols:
        data[symbol] = []
        price = 100.0
        
        for i in range(days):
            date = base_date + timedelta(days=i)
            price += random.uniform(-5, 5)
            volume = random.randint(100000, 1000000)
            
            data[symbol].append({
                "date": date,
                "price": price,
                "volume": volume
            })
    
    return data

def generate_trade_outcomes(count=100):
    """Generate synthetic trade outcomes"""
    outcomes = []
    for i in range(count):
        outcomes.append({
            "symbol": random.choice(["AAPL", "TSLA", "MSFT"]),
            "confidence": random.uniform(0.5, 0.9),
            "outcome": random.choice(["WIN", "LOSS"]),
            "pnl": random.uniform(-100, 200)
        })
    return outcomes
```

## Coverage Analysis

### Coverage Configuration

#### .coveragerc
```ini
[run]
source = src
omit = 
    */tests/*
    */venv/*
    */env/*
    setup.py
    */migrations/*

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
    if TYPE_CHECKING:
```

### Coverage Reports

#### HTML Report
```bash
# Generate HTML coverage report
pytest --cov=src --cov-report=html:reports/coverage

# View report
open reports/coverage/index.html
```

#### Terminal Report
```bash
# Generate terminal coverage report
pytest --cov=src --cov-report=term-missing
```

#### XML Report
```bash
# Generate XML coverage report for CI
pytest --cov=src --cov-report=xml:reports/coverage.xml
```

### Coverage Targets

- **Overall Coverage**: 90%+
- **Critical Modules**: 95%+
- **New Code**: 100%
- **Public APIs**: 100%

## Automated Testing

### CI/CD Pipeline

#### GitHub Actions
```yaml
# .github/workflows/test.yml
name: Test Suite

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.11, 3.12]

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run unit tests
      run: |
        pytest tests/unit/ --cov=src --cov-report=xml
    
    - name: Run integration tests
      run: |
        pytest tests/integration/ --cov=src --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

### Pre-commit Hooks

#### .pre-commit-config.yaml
```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files

  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3.11

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8

  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: pytest
        language: system
        pass_filenames: false
        always_run: true
        args: [tests/unit/]
```

## Performance Testing

### Load Testing

#### Performance Test Example
```python
# tests/performance/test_load.py
import pytest
import time
from concurrent.futures import ThreadPoolExecutor

class TestPerformance:
    def test_concurrent_api_calls(self):
        """Test system under concurrent load"""
        def make_api_call():
            # Simulate API call
            time.sleep(0.1)
            return {"status": "success"}
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_api_call) for _ in range(100)]
            results = [future.result() for future in futures]
        
        end_time = time.time()
        duration = end_time - start_time
        
        assert duration < 5.0  # Should complete within 5 seconds
        assert all(r["status"] == "success" for r in results)
    
    def test_memory_usage(self):
        """Test memory usage under load"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Perform memory-intensive operations
        data = []
        for i in range(10000):
            data.append({"id": i, "data": "x" * 1000})
        
        peak_memory = process.memory_info().rss
        memory_increase = peak_memory - initial_memory
        
        # Should not use more than 100MB
        assert memory_increase < 100 * 1024 * 1024
```

### Benchmark Testing

#### Benchmark Configuration
```python
# tests/performance/benchmarks.py
import pytest
import time
from trading.risk import RiskManager

class TestBenchmarks:
    @pytest.mark.benchmark
    def test_position_size_calculation_speed(self, benchmark):
        """Benchmark position size calculation"""
        risk_manager = RiskManager()
        
        def calculate_position_size():
            return risk_manager.calculate_position_size("AAPL", 150.0, 0.8)
        
        result = benchmark(calculate_position_size)
        assert result > 0
    
    @pytest.mark.benchmark
    def test_database_query_speed(self, benchmark):
        """Benchmark database queries"""
        from config.database import execute_query
        
        def query_positions():
            return execute_query("SELECT * FROM positions LIMIT 100", (), "DEMO")
        
        result = benchmark(query_positions)
        assert len(result) <= 100
```

## Test Maintenance

### Test Organization

#### Naming Conventions
- Test files: `test_*.py`
- Test classes: `Test*`
- Test methods: `test_*`
- Fixtures: `*_fixture` or `*_data`

#### Test Documentation
```python
class TestRiskManager:
    """Test suite for RiskManager class"""
    
    def test_calculate_position_size(self):
        """
        Test position size calculation
        
        Given: Symbol, entry price, and confidence
        When: Calculating position size
        Then: Should return valid position size within limits
        """
        # Test implementation
        pass
```

### Test Data Management

#### Test Data Cleanup
```python
@pytest.fixture(autouse=True)
def cleanup_test_data():
    """Automatically cleanup test data after each test"""
    yield
    # Cleanup code here
    pass
```

#### Test Isolation
```python
@pytest.fixture
def isolated_database():
    """Create isolated database for each test"""
    # Create temporary database
    yield
    # Cleanup
```

## Best Practices

### Writing Tests

1. **Test One Thing**: Each test should verify one specific behavior
2. **Descriptive Names**: Test names should clearly describe what they test
3. **Arrange-Act-Assert**: Structure tests with clear setup, execution, and verification
4. **Independent Tests**: Tests should not depend on each other
5. **Fast Tests**: Keep tests fast and focused

### Test Data

1. **Use Fixtures**: Create reusable test data with fixtures
2. **Minimal Data**: Use only the data necessary for the test
3. **Realistic Data**: Use realistic test data that matches production
4. **Clean Data**: Ensure test data is clean and consistent

### Error Testing

1. **Test Error Cases**: Include tests for error conditions
2. **Test Edge Cases**: Test boundary conditions and edge cases
3. **Test Validation**: Test input validation and error handling
4. **Test Recovery**: Test error recovery and fallback mechanisms

### Performance Testing

1. **Baseline Performance**: Establish performance baselines
2. **Monitor Trends**: Track performance over time
3. **Test Under Load**: Test system behavior under various loads
4. **Resource Monitoring**: Monitor CPU, memory, and I/O usage

---

## Test Reports

### Test Results

#### JUnit XML
```bash
# Generate JUnit XML report
pytest --junitxml=reports/junit.xml
```

#### HTML Report
```bash
# Generate HTML test report
pytest --html=reports/report.html --self-contained-html
```

### Coverage Reports

#### Coverage Summary
```bash
# Generate coverage summary
pytest --cov=src --cov-report=term-missing
```

#### Coverage HTML
```bash
# Generate coverage HTML report
pytest --cov=src --cov-report=html:reports/coverage
```

---

**Document Version**: 1.0.0
**Last Updated**: 2025-10-13
**Compatibility**: pytest 7.0+, Python 3.11+

