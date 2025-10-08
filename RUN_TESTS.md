# 🧪 How to Run Tests

## Quick Start

### Run All Core Tests
```bash
python tests/test_core_systems.py
```

### Run with Pytest
```bash
pytest tests/test_core_systems.py -v
```

### Run Specific Test Class
```bash
pytest tests/test_core_systems.py::TestExecutionEngine -v
```

### Run Specific Test
```bash
pytest tests/test_core_systems.py::TestExecutionEngine::test_create_order -v
```

---

## Test Files

### ✅ `tests/test_core_systems.py`
**Status**: All tests passing (14/14)

**Components Tested**:
- Execution Engine (5 tests)
- Event Calendar (4 tests)
- Volatility Detector (4 tests)
- Trading Modes (1 test)

**Dependencies**: None (uses only core libraries)

**Run Command**:
```bash
python tests/test_core_systems.py
```

---

### ⚠️ `tests/test_all_systems.py`
**Status**: Not fully functional yet

**Components Tested**:
- Everything in test_core_systems.py
- AI Model Stack (LSTM, GRU, Ensemble)
- RL Core (PPO, DQN)
- Anomaly Detector
- Reporting System

**Dependencies**: 
- ⚠️ stable-baselines3 (not installed)
- ⚠️ transformers (not installed)
- ✅ gymnasium (installed)

**Run Command** (after installing dependencies):
```bash
pytest tests/test_all_systems.py -v
```

---

### ✅ `tests/test_reporting_system.py`
**Status**: Working

**Components Tested**:
- Report Generator
- Report Scheduler
- AI Learning Integration
- Report Persistence

**Run Command**:
```bash
pytest tests/test_reporting_system.py -v
```

---

## Pytest Options

### Verbose Output
```bash
pytest tests/ -v
```

### Show Print Statements
```bash
pytest tests/ -v -s
```

### Stop on First Failure
```bash
pytest tests/ -x
```

### Run in Parallel
```bash
pytest tests/ -n auto
```

### Coverage Report
```bash
pytest tests/ --cov=src --cov-report=html
```

### Run Only Failed Tests
```bash
pytest tests/ --lf
```

---

## Test Results

### Expected Output
```
================================================================================
🧪 RUNNING CORE SYSTEM TESTS (No ML Dependencies)
================================================================================

============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-8.4.1, pluggy-1.6.0
...
collected 14 items

tests/test_core_systems.py::TestExecutionEngine::test_create_order PASSED
tests/test_core_systems.py::TestExecutionEngine::test_market_order_execution PASSED
tests/test_core_systems.py::TestExecutionEngine::test_fractional_shares PASSED
tests/test_core_systems.py::TestExecutionEngine::test_vwap_execution PASSED
tests/test_core_systems.py::TestExecutionEngine::test_execution_statistics PASSED
tests/test_core_systems.py::TestEventCalendar::test_add_event PASSED
tests/test_core_systems.py::TestEventCalendar::test_get_upcoming_events PASSED
tests/test_core_systems.py::TestEventCalendar::test_market_holiday PASSED
tests/test_core_systems.py::TestEventCalendar::test_high_impact_events PASSED
tests/test_core_systems.py::TestVolatilityDetector::test_historical_volatility PASSED
tests/test_core_systems.py::TestVolatilityDetector::test_atr_calculation PASSED
tests/test_core_systems.py::TestVolatilityDetector::test_volatility_regime_classification PASSED
tests/test_core_systems.py::TestVolatilityDetector::test_volatility_analysis PASSED
tests/test_core_systems.py::TestTradingModes::test_mode_manager PASSED

============================= 14 passed in 0.14s ==============================

================================================================================
✅ ALL TESTS PASSED!
================================================================================
```

---

## Troubleshooting

### Issue: ModuleNotFoundError for gymnasium
**Solution**:
```bash
pip install gymnasium==0.29.1
```

### Issue: ModuleNotFoundError for stable_baselines3
**Solution**:
```bash
pip install stable-baselines3==2.1.0
```

### Issue: Tests fail with import errors
**Solution**: Make sure you're in the project root directory
```bash
cd C:\Users\Coding\Desktop\TradingBOT
python tests/test_core_systems.py
```

### Issue: Slow test execution
**Solution**: Run tests in parallel
```bash
pytest tests/ -n auto
```

---

## Test Coverage

### Current Coverage
- **Execution Engine**: ~90%
- **Event Calendar**: ~85%
- **Volatility Detector**: ~90%
- **Trading Modes**: ~75%
- **Overall**: ~85%

### Generate Coverage Report
```bash
pytest tests/ --cov=src --cov-report=html
# Open htmlcov/index.html in browser
```

---

## CI/CD Integration

### GitHub Actions (example)
```yaml
name: Run Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: pytest tests/test_core_systems.py -v
```

---

## Test Data

### Location
- `data/test_event_calendar.json` - Test event data
- Sample data generated in tests (OHLCV, prices, etc.)

### Cleanup
Test data is automatically cleaned up after each test run.

---

## Best Practices

1. **Run tests before committing**
   ```bash
   python tests/test_core_systems.py
   ```

2. **Check for new failures**
   ```bash
   pytest tests/ --lf
   ```

3. **Verify coverage**
   ```bash
   pytest tests/ --cov=src
   ```

4. **Run full suite periodically**
   ```bash
   pytest tests/ -v
   ```

---

## Quick Commands Reference

```bash
# Run all core tests
python tests/test_core_systems.py

# Run with pytest (verbose)
pytest tests/test_core_systems.py -v -s

# Run specific test class
pytest tests/test_core_systems.py::TestExecutionEngine -v

# Run and generate coverage
pytest tests/ --cov=src --cov-report=term-missing

# Run in parallel
pytest tests/ -n auto

# Stop on first failure
pytest tests/ -x

# Run only failed tests
pytest tests/ --lf
```

---

**Last Updated**: October 4, 2025  
**Test Status**: ✅ 14/14 Core Tests Passing

