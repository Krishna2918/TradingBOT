# Trading Bot API Reference

## Overview

This document provides comprehensive API reference for all modules in the Trading Bot system.

## Table of Contents

1. [Core Modules](#core-modules)
2. [Data Pipeline](#data-pipeline)
3. [AI & Machine Learning](#ai--machine-learning)
4. [Trading Engine](#trading-engine)
5. [Risk Management](#risk-management)
6. [Monitoring & Analytics](#monitoring--analytics)
7. [Configuration](#configuration)
8. [Utilities](#utilities)

---

## Core Modules

### Database Management

#### `config.database`

**Purpose**: Centralized database connection and schema management.

**Key Functions**:
- `get_connection(mode: str) -> sqlite3.Connection`
- `execute_query(query: str, params: tuple, mode: str) -> List[sqlite3.Row]`
- `create_schema(mode: str) -> None`
- `migrate_schema(mode: str) -> None`

**Example Usage**:
```python
from config.database import get_connection, execute_query

# Get database connection
with get_connection("DEMO") as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM positions")

# Execute query with parameters
result = execute_query(
    "SELECT * FROM positions WHERE symbol = ?", 
    ("AAPL",), 
    "DEMO"
)
```

#### `config.mode_manager`

**Purpose**: Manages trading modes (LIVE/DEMO) and mode-specific configurations.

**Key Functions**:
- `get_current_mode() -> str`
- `set_mode(mode: str) -> None`
- `get_mode_config(mode: str) -> Dict`

**Example Usage**:
```python
from config.mode_manager import get_current_mode, set_mode

# Get current mode
current_mode = get_current_mode()  # Returns "DEMO" or "LIVE"

# Set mode
set_mode("LIVE")
```

---

## Data Pipeline

### API Budget Management

#### `data_pipeline.api_budget_manager`

**Purpose**: Manages API rate limits, budgets, and caching for external API calls.

**Key Classes**:
- `API_Budget_Manager`

**Key Methods**:
- `check_budget(api_name: str) -> bool`
- `consume_budget(api_name: str, cost: int) -> None`
- `get_cache(key: str) -> Optional[Any]`
- `set_cache(key: str, value: Any, ttl: int) -> None`

**Example Usage**:
```python
from data_pipeline.api_budget_manager import API_Budget_Manager

budget_manager = API_Budget_Manager()

# Check if API call is allowed
if budget_manager.check_budget("questrade"):
    # Make API call
    data = make_api_call()
    budget_manager.consume_budget("questrade", 1)

# Use caching
cached_data = budget_manager.get_cache("market_data_AAPL")
if not cached_data:
    cached_data = fetch_market_data("AAPL")
    budget_manager.set_cache("market_data_AAPL", cached_data, 300)
```

### Market Data Collection

#### `data_pipeline.questrade_client`

**Purpose**: Questrade API client for market data and trading operations.

**Key Classes**:
- `QuestradeClient`

**Key Methods**:
- `get_account_info() -> Dict`
- `get_positions() -> List[Dict]`
- `get_market_data(symbols: List[str]) -> Dict`
- `place_order(order_data: Dict) -> Dict`

**Example Usage**:
```python
from data_pipeline.questrade_client import QuestradeClient

client = QuestradeClient()

# Get account information
account_info = client.get_account_info()

# Get market data
market_data = client.get_market_data(["AAPL", "TSLA"])

# Place order
order_result = client.place_order({
      "symbol": "AAPL",
      "quantity": 100,
    "order_type": "buy",
    "price": 150.00
})
```

---

## AI & Machine Learning

### Confidence Calibration

#### `adaptive.confidence_calibration`

**Purpose**: Bayesian confidence calibration for AI model predictions.

**Key Classes**:
- `ConfidenceCalibrator`

**Key Methods**:
- `calibrate_confidence(model: str, confidence: float, trade_date: datetime) -> float`
- `add_trade_outcome(model: str, symbol: str, confidence: float, outcome: str, pnl: float, trade_date: datetime) -> None`
- `get_calibration_summary(model: str) -> Dict`

**Example Usage**:
```python
from adaptive.confidence_calibration import ConfidenceCalibrator
from datetime import datetime

calibrator = ConfidenceCalibrator()

# Calibrate confidence
raw_confidence = 0.8
calibrated_confidence = calibrator.calibrate_confidence(
    "test_model", 
    raw_confidence, 
    datetime.now()
)

# Add trade outcome for learning
calibrator.add_trade_outcome(
    "test_model",
    "AAPL",
    0.8,
    "WIN",
    100.0,
    datetime.now()
)
```

### Adaptive Ensemble Weights

#### `ai.adaptive_weights`

**Purpose**: Dynamic model weighting based on rolling performance metrics.

**Key Classes**:
- `AdaptiveWeightManager`

**Key Methods**:
- `update_weights(predictions: Dict[str, float], outcomes: Dict[str, str]) -> Dict[str, float]`
- `get_current_weights() -> Dict[str, float]`
- `calculate_brier_score(predictions: List[float], outcomes: List[str]) -> float`

**Example Usage**:
```python
from ai.adaptive_weights import AdaptiveWeightManager

weight_manager = AdaptiveWeightManager()

# Update weights based on performance
predictions = {"model1": 0.8, "model2": 0.7}
outcomes = {"model1": "WIN", "model2": "LOSS"}
new_weights = weight_manager.update_weights(predictions, outcomes)

# Get current weights
current_weights = weight_manager.get_current_weights()
```

### Regime Detection

#### `ai.regime_detection`

**Purpose**: Market regime detection using ATR/VIX-based analysis.

**Key Classes**:
- `RegimeDetector`

**Key Methods**:
- `detect_current_regime(symbol: str = "SPY", mode: str = None) -> RegimeState`
- `get_regime_metrics(symbol: str) -> RegimeMetrics`
- `classify_regime(metrics: RegimeMetrics) -> str`

**Example Usage**:
```python
from ai.regime_detection import RegimeDetector

detector = RegimeDetector()

# Detect current market regime
regime_state = detector.detect_current_regime("SPY")

print(f"Current regime: {regime_state.regime}")
print(f"Trend direction: {regime_state.trend_direction}")
print(f"Volatility level: {regime_state.volatility_level}")
```

---

## Trading Engine

### Risk Management

#### `trading.risk`

**Purpose**: Comprehensive risk management including position sizing and drawdown control.

**Key Classes**:
- `RiskManager`

**Key Methods**:
- `calculate_position_size(symbol: str, entry_price: float, confidence: float) -> float`
- `calculate_daily_drawdown(mode: str = None) -> Dict[str, Any]`
- `calculate_drawdown_scale(mode: str = None) -> float`
- `calculate_kelly_fraction(win_probability: float, avg_win: float, avg_loss: float) -> float`

**Example Usage**:
```python
from trading.risk import RiskManager

risk_manager = RiskManager()

# Calculate position size
position_size = risk_manager.calculate_position_size(
    "AAPL", 
    150.00, 
    0.8
)

# Check daily drawdown
drawdown_info = risk_manager.calculate_daily_drawdown()
print(f"Daily drawdown: {drawdown_info['daily_drawdown']:.2%}")

# Calculate Kelly fraction
kelly_fraction = risk_manager.calculate_kelly_fraction(0.6, 100.0, 50.0)
```

### ATR Brackets

#### `trading.atr_brackets`

**Purpose**: ATR-based stop loss and take profit bracket management.

**Key Classes**:
- `ATRBracketManager`

**Key Methods**:
- `calculate_brackets(symbol: str, entry_price: float, atr: float, regime: str = None) -> BracketParameters`
- `log_bracket_parameters(params: BracketParameters) -> None`
- `get_historical_performance(symbol: str) -> Dict`

**Example Usage**:
```python
from trading.atr_brackets import ATRBracketManager

bracket_manager = ATRBracketManager()

# Calculate brackets
brackets = bracket_manager.calculate_brackets(
    "AAPL",
    150.00,
    3.0,
    "TREND_UP"
)

print(f"Stop Loss: ${brackets.stop_loss:.2f}")
print(f"Take Profit: ${brackets.take_profit:.2f}")
print(f"R-Multiple: {brackets.r_multiple:.2f}")
```

### Position Management

#### `trading.positions`

**Purpose**: Position tracking and management.

**Key Classes**:
- `PositionManager`

**Key Methods**:
- `get_open_positions(mode: str = None) -> List[Position]`
- `track_position(position: Position) -> None`
- `close_position(position_id: str, exit_price: float, exit_reason: str) -> None`
- `get_portfolio_summary(mode: str = None) -> Dict`

**Example Usage**:
```python
from trading.positions import PositionManager, Position

position_manager = PositionManager()

# Get open positions
open_positions = position_manager.get_open_positions()

# Create new position
position = Position(
    symbol="AAPL",
    quantity=100,
    entry_price=150.00,
    entry_date=datetime.now()
)
position_manager.track_position(position)

# Get portfolio summary
summary = position_manager.get_portfolio_summary()
print(f"Total value: ${summary['total_value']:.2f}")
```

---

## Monitoring & Analytics

### System Monitoring

#### `monitoring.system_monitor`

**Purpose**: System health monitoring and performance tracking.

**Key Classes**:
- `SystemMonitor`

**Key Methods**:
- `start_phase_timer(phase_name: str) -> None`
- `end_phase_timer(phase_name: str) -> float`
- `get_system_metrics() -> Dict`
- `log_phase_duration(phase_name: str, duration: float) -> None`

**Example Usage**:
```python
from monitoring.system_monitor import SystemMonitor

monitor = SystemMonitor()

# Time a phase
monitor.start_phase_timer("data_collection")
# ... do work ...
duration = monitor.end_phase_timer("data_collection")

# Get system metrics
metrics = monitor.get_system_metrics()
print(f"CPU usage: {metrics['cpu_percent']:.1f}%")
print(f"Memory usage: {metrics['memory_percent']:.1f}%")
```

### Performance Analytics

#### `monitoring.performance_analytics`

**Purpose**: Trading performance analysis and reporting.

**Key Classes**:
- `PerformanceAnalytics`

**Key Methods**:
- `calculate_returns(positions: List[Position]) -> Dict`
- `calculate_sharpe_ratio(returns: List[float]) -> float`
- `calculate_max_drawdown(equity_curve: List[float]) -> float`
- `generate_performance_report(mode: str) -> Dict`

**Example Usage**:
```python
from monitoring.performance_analytics import PerformanceAnalytics

analytics = PerformanceAnalytics()

# Calculate returns
returns = analytics.calculate_returns(positions)

# Calculate Sharpe ratio
sharpe = analytics.calculate_sharpe_ratio(returns['daily_returns'])

# Generate performance report
report = analytics.generate_performance_report("DEMO")
print(f"Total return: {report['total_return']:.2%}")
print(f"Sharpe ratio: {report['sharpe_ratio']:.2f}")
```

---

## Configuration

### Feature Flags

#### `config.feature_flags`

**Purpose**: Runtime feature flag management for safe deployments.

**Key Classes**:
- `FeatureFlagManager`

**Key Methods**:
- `is_enabled(flag_name: str) -> bool`
- `set_flag(flag_name: str, enabled: bool) -> None`
- `get_all_flags() -> Dict[str, bool]`

**Example Usage**:
```python
from config.feature_flags import FeatureFlagManager

flag_manager = FeatureFlagManager()

# Check if feature is enabled
if flag_manager.is_enabled("advanced_ml_models"):
    # Use advanced ML models
    pass

# Enable/disable feature
flag_manager.set_flag("new_trading_strategy", True)
```

### Regime Policies

#### `config.regime_policy_manager`

**Purpose**: Manages regime-specific trading policies and parameters.

**Key Classes**:
- `RegimePolicyManager`

**Key Methods**:
- `get_policy(regime: str) -> Dict`
- `get_ensemble_weights(regime: str) -> Dict[str, float]`
- `get_kelly_sensitivity(regime: str) -> float`

**Example Usage**:
```python
from config.regime_policy_manager import RegimePolicyManager

policy_manager = RegimePolicyManager()

# Get regime-specific policy
policy = policy_manager.get_policy("TREND_UP")
print(f"Kelly sensitivity: {policy['kelly_sensitivity']}")

# Get ensemble weights for regime
weights = policy_manager.get_ensemble_weights("HIGH_VOLATILITY")
print(f"Model weights: {weights}")
```

---

## Utilities

### Data Quality Validation

#### `validation.data_quality`

**Purpose**: Data quality validation and integrity checks.

**Key Classes**:
- `DataQualityValidator`

**Key Methods**:
- `validate_market_data(data: Dict) -> Dict`
- `validate_indicators(indicators: Dict) -> Dict`
- `check_data_completeness(data: Dict) -> bool`

**Example Usage**:
```python
from validation.data_quality import DataQualityValidator

validator = DataQualityValidator()

# Validate market data
validation_result = validator.validate_market_data(market_data)
if validation_result['is_valid']:
    print("Data quality: PASS")
else:
    print(f"Data quality issues: {validation_result['issues']}")
```

### Error Tracking

#### `monitoring.error_tracker`

**Purpose**: Error tracking and analysis.

**Key Classes**:
- `ErrorTracker`

**Key Methods**:
- `log_error(error: Exception, context: Dict) -> None`
- `get_error_summary() -> Dict`
- `get_recent_errors(limit: int = 10) -> List[Dict]`

**Example Usage**:
```python
from monitoring.error_tracker import ErrorTracker

error_tracker = ErrorTracker()

try:
    # Some operation
    risky_operation()
except Exception as e:
    error_tracker.log_error(e, {"operation": "data_fetch", "symbol": "AAPL"})

# Get error summary
summary = error_tracker.get_error_summary()
print(f"Total errors: {summary['total_errors']}")
```

---

## Error Handling

### Common Error Patterns

1. **Database Connection Errors**
   ```python
   try:
       with get_connection("DEMO") as conn:
           # Database operations
   except sqlite3.Error as e:
       logger.error(f"Database error: {e}")
   ```

2. **API Rate Limiting**
   ```python
   try:
       data = api_call()
   except requests.exceptions.HTTPError as e:
       if e.response.status_code == 429:
           logger.warning("Rate limit exceeded, backing off")
           time.sleep(60)
   ```

3. **Data Validation Errors**
   ```python
   try:
       validated_data = validator.validate_market_data(data)
   except ValidationError as e:
       logger.error(f"Data validation failed: {e}")
   ```

---

## Best Practices

### 1. Error Handling
- Always use try-catch blocks for external API calls
- Log errors with sufficient context
- Implement graceful degradation

### 2. Performance
- Use caching for expensive operations
- Implement connection pooling for database operations
- Monitor memory usage for large data processing

### 3. Security
- Never log sensitive information (API keys, passwords)
- Validate all input data
- Use parameterized queries for database operations

### 4. Testing
- Write unit tests for all public methods
- Use mocking for external dependencies
- Test error conditions and edge cases

---

## Version Information

- **API Version**: 1.0.0
- **Last Updated**: 2025-10-13
- **Compatibility**: Python 3.11+

---

## Support

For questions or issues with the API:
1. Check the troubleshooting guide
2. Review the error logs
3. Contact the development team

---

**Document Version**: 1.0.0
**Last Updated**: 2025-10-13