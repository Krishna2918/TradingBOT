# AI Trading System - Developer Guide

## Overview

This developer guide provides comprehensive information for developers working on the AI Trading System, including development environment setup, code structure, architecture patterns, and contribution guidelines.

## Development Environment Setup

### Prerequisites
- **Python 3.11+**: [Download from python.org](https://www.python.org/downloads/)
- **Git**: [Download from git-scm.com](https://git-scm.com/downloads)
- **Docker** (Optional): [Download from docker.com](https://www.docker.com/products/docker-desktop)
- **IDE**: VS Code, PyCharm, or similar

### Local Development Setup

#### 1. Clone Repository
```bash
git clone <repository-url>
cd TradingBOT
```

#### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies
```

#### 4. Environment Configuration
```bash
# Copy environment template
cp .env.template .env

# Edit environment variables
nano .env
```

#### 5. Database Setup
```bash
# Initialize database
python -c "from src.config.database import get_database_manager; get_database_manager().initialize_database()"

# Run migrations
python scripts/migrate.py
```

#### 6. Start Development Server
```bash
# Start the system
python src/main.py

# Or start with hot reload
python -m watchdog src/main.py
```

## Code Structure and Organization

### Project Structure
```
TradingBOT/
├── src/                    # Source code
│   ├── config/            # Configuration management
│   ├── trading/           # Core trading components
│   ├── ai/                # AI and machine learning
│   ├── workflows/         # Workflow orchestration
│   ├── monitoring/        # Monitoring and logging
│   ├── validation/        # Security and validation
│   ├── adaptive/          # Adaptive learning
│   ├── performance/       # Performance optimization
│   ├── risk/              # Risk management
│   ├── data_pipeline/     # Data processing
│   ├── execution/         # Order execution
│   ├── dashboard/         # Web dashboard
│   └── main.py           # Main entry point
├── tests/                 # Test suite
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   ├── performance/      # Performance tests
│   ├── security/         # Security tests
│   └── e2e/              # End-to-end tests
├── docs/                  # Documentation
├── config/                # Configuration files
├── scripts/               # Utility scripts
├── data/                  # Data storage
├── logs/                  # Log files
└── requirements.txt       # Dependencies
```

### Architecture Patterns

#### 1. Modular Design
Each component is designed as a self-contained module with clear interfaces:

```python
# Example: Position Manager
class PositionManager:
    def __init__(self, mode: str = None):
        self.mode = mode or get_current_mode()
        self.db_manager = get_database_manager()
    
    def create_position(self, symbol: str, quantity: int, price: float) -> Position:
        """Create a new position with validation"""
        pass
    
    def get_position(self, position_id: str) -> Optional[Position]:
        """Retrieve position by ID"""
        pass
    
    def update_position(self, position_id: str, updates: dict) -> bool:
        """Update position with new data"""
        pass
```

#### 2. Dependency Injection
Components use dependency injection for better testability:

```python
# Example: Trading Cycle
class TradingCycle:
    def __init__(self, 
                 position_manager: PositionManager = None,
                 execution_engine: ExecutionEngine = None,
                 risk_manager: RiskManager = None):
        self.position_manager = position_manager or get_position_manager()
        self.execution_engine = execution_engine or get_execution_engine()
        self.risk_manager = risk_manager or get_risk_manager()
```

#### 3. Factory Pattern
Use factory functions for component creation:

```python
# Example: Component Factories
def get_position_manager(mode: str = None) -> PositionManager:
    """Factory function for PositionManager"""
    return PositionManager(mode)

def get_execution_engine(mode: str = None) -> ExecutionEngine:
    """Factory function for ExecutionEngine"""
    return ExecutionEngine(mode)
```

#### 4. Observer Pattern
Use observers for event-driven architecture:

```python
# Example: Event System
class EventObserver:
    def update(self, event: Event):
        """Handle event updates"""
        pass

class TradingEventManager:
    def __init__(self):
        self.observers = []
    
    def subscribe(self, observer: EventObserver):
        self.observers.append(observer)
    
    def notify(self, event: Event):
        for observer in self.observers:
            observer.update(event)
```

## Development Workflow

### Git Workflow

#### Branch Strategy
- **main**: Production-ready code
- **develop**: Integration branch for features
- **feature/**: Feature development branches
- **hotfix/**: Critical bug fixes
- **release/**: Release preparation branches

#### Commit Convention
```
type(scope): description

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test additions/changes
- `chore`: Build process or auxiliary tool changes

**Examples:**
```
feat(trading): add position sizing algorithm
fix(ai): resolve model loading timeout issue
docs(api): update API documentation
test(integration): add trading cycle tests
```

#### Pull Request Process
1. Create feature branch from `develop`
2. Implement changes with tests
3. Update documentation
4. Create pull request
5. Code review and approval
6. Merge to `develop`

### Testing Strategy

#### Unit Tests
```python
# Example: Unit test
import unittest
from unittest.mock import Mock, patch
from src.trading.positions import PositionManager

class TestPositionManager(unittest.TestCase):
    def setUp(self):
        self.position_manager = PositionManager('DEMO')
    
    def test_create_position(self):
        position = self.position_manager.create_position('AAPL', 100, 150.0)
        self.assertIsNotNone(position)
        self.assertEqual(position.symbol, 'AAPL')
        self.assertEqual(position.quantity, 100)
    
    @patch('src.config.database.get_connection')
    def test_create_position_database_error(self, mock_get_connection):
        mock_get_connection.side_effect = Exception("Database error")
        with self.assertRaises(Exception):
            self.position_manager.create_position('AAPL', 100, 150.0)
```

#### Integration Tests
```python
# Example: Integration test
import unittest
from src.workflows.trading_cycle import TradingCycle
from src.trading.positions import PositionManager
from src.trading.execution import ExecutionEngine

class TestTradingCycleIntegration(unittest.TestCase):
    def setUp(self):
        self.trading_cycle = TradingCycle()
    
    def test_complete_trading_cycle(self):
        # Test complete trading workflow
        result = self.trading_cycle.execute_complete_cycle()
        self.assertTrue(result.success)
        self.assertIsNotNone(result.positions_created)
```

#### Performance Tests
```python
# Example: Performance test
import unittest
import time
from src.ai.multi_model import MultiModelManager

class TestPerformance(unittest.TestCase):
    def test_ai_decision_latency(self):
        manager = MultiModelManager()
        start_time = time.time()
        
        result = manager.get_analysis('AAPL', 'entry_signal')
        
        end_time = time.time()
        latency = end_time - start_time
        
        self.assertLess(latency, 2.0)  # Should complete in under 2 seconds
        self.assertIsNotNone(result)
```

### Code Quality Standards

#### Code Style
Follow PEP 8 with these additional guidelines:

```python
# Good example
class TradingEngine:
    """Trading engine for executing orders and managing positions."""
    
    def __init__(self, mode: str = 'DEMO'):
        """Initialize trading engine with specified mode.
        
        Args:
            mode: Trading mode ('LIVE' or 'DEMO')
        """
        self.mode = mode
        self._initialize_components()
    
    def execute_order(self, order: Order) -> ExecutionResult:
        """Execute a trading order.
        
        Args:
            order: Order to execute
            
        Returns:
            ExecutionResult with execution details
            
        Raises:
            InsufficientFundsError: If account has insufficient funds
            InvalidOrderError: If order parameters are invalid
        """
        if not self._validate_order(order):
            raise InvalidOrderError("Invalid order parameters")
        
        return self._process_order(order)
```

#### Type Hints
Always use type hints for better code clarity:

```python
from typing import List, Dict, Optional, Union
from dataclasses import dataclass

@dataclass
class Position:
    symbol: str
    quantity: int
    entry_price: float
    current_price: float

def get_positions(mode: str = None) -> List[Position]:
    """Get all positions for specified mode."""
    pass

def calculate_pnl(position: Position) -> float:
    """Calculate profit/loss for position."""
    return (position.current_price - position.entry_price) * position.quantity
```

#### Documentation
Use docstrings for all public methods and classes:

```python
def calculate_position_size(
    account_balance: float,
    risk_percent: float,
    stock_price: float,
    stop_loss_percent: float
) -> int:
    """Calculate optimal position size based on risk management.
    
    Args:
        account_balance: Total account balance
        risk_percent: Maximum risk percentage (0.01 = 1%)
        stock_price: Current stock price
        stop_loss_percent: Stop loss percentage (0.05 = 5%)
    
    Returns:
        Number of shares to purchase
        
    Example:
        >>> calculate_position_size(10000, 0.02, 100, 0.05)
        40
    """
    risk_amount = account_balance * risk_percent
    stop_loss_amount = stock_price * stop_loss_percent
    position_size = risk_amount / stop_loss_amount
    return int(position_size)
```

## API Development

### REST API Design

#### Endpoint Structure
```
GET    /api/v1/positions          # List positions
GET    /api/v1/positions/{id}     # Get position details
POST   /api/v1/positions          # Create position
PUT    /api/v1/positions/{id}     # Update position
DELETE /api/v1/positions/{id}     # Delete position
```

#### Request/Response Format
```python
# Example: Position API
from flask import Flask, request, jsonify
from src.trading.positions import get_position_manager

app = Flask(__name__)

@app.route('/api/v1/positions', methods=['GET'])
def get_positions():
    """Get all positions."""
    try:
        position_manager = get_position_manager()
        positions = position_manager.get_all_positions()
        
        return jsonify({
            'success': True,
            'data': [position.to_dict() for position in positions],
            'count': len(positions)
        }), 200
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/v1/positions', methods=['POST'])
def create_position():
    """Create a new position."""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['symbol', 'quantity', 'price']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        position_manager = get_position_manager()
        position = position_manager.create_position(
            symbol=data['symbol'],
            quantity=data['quantity'],
            price=data['price']
        )
        
        return jsonify({
            'success': True,
            'data': position.to_dict()
        }), 201
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
```

### WebSocket API

#### Real-time Data Streaming
```python
# Example: WebSocket handler
import asyncio
import websockets
import json
from src.data_pipeline.questrade_client import QuestradeClient

class MarketDataWebSocket:
    def __init__(self):
        self.clients = set()
        self.questrade_client = QuestradeClient()
    
    async def register_client(self, websocket):
        """Register new WebSocket client."""
        self.clients.add(websocket)
        print(f"Client connected. Total clients: {len(self.clients)}")
    
    async def unregister_client(self, websocket):
        """Unregister WebSocket client."""
        self.clients.remove(websocket)
        print(f"Client disconnected. Total clients: {len(self.clients)}")
    
    async def broadcast_market_data(self, symbol: str):
        """Broadcast market data to all clients."""
        if not self.clients:
            return
        
        try:
            quote = self.questrade_client.get_quote(symbol)
            message = json.dumps({
                'type': 'market_data',
                'symbol': symbol,
                'data': quote
            })
            
            # Send to all connected clients
            disconnected = set()
            for client in self.clients:
                try:
                    await client.send(message)
                except websockets.exceptions.ConnectionClosed:
                    disconnected.add(client)
            
            # Remove disconnected clients
            self.clients -= disconnected
            
        except Exception as e:
            print(f"Error broadcasting market data: {e}")
    
    async def handle_client_message(self, websocket, message):
        """Handle incoming client messages."""
        try:
            data = json.loads(message)
            
            if data['type'] == 'subscribe':
                symbol = data['symbol']
                # Start broadcasting data for this symbol
                asyncio.create_task(self.broadcast_market_data(symbol))
            
        except Exception as e:
            print(f"Error handling client message: {e}")

# WebSocket server
async def websocket_handler(websocket, path):
    market_data_ws = MarketDataWebSocket()
    await market_data_ws.register_client(websocket)
    
    try:
        async for message in websocket:
            await market_data_ws.handle_client_message(websocket, message)
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        await market_data_ws.unregister_client(websocket)

# Start WebSocket server
start_server = websockets.serve(websocket_handler, "localhost", 8765)
asyncio.get_event_loop().run_until_complete(start_server)
```

## Database Development

### Database Schema Design

#### Entity Relationship Diagram
```sql
-- Positions table
CREATE TABLE positions (
    id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    entry_price REAL NOT NULL,
    current_price REAL,
    entry_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'open',
    mode TEXT NOT NULL
);

-- Orders table
CREATE TABLE orders (
    id TEXT PRIMARY KEY,
    position_id TEXT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    order_type TEXT NOT NULL,
    price REAL,
    status TEXT DEFAULT 'submitted',
    created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    filled_time TIMESTAMP,
    mode TEXT NOT NULL,
    FOREIGN KEY (position_id) REFERENCES positions(id)
);

-- Trades table
CREATE TABLE trades (
    id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    price REAL NOT NULL,
    commission REAL DEFAULT 0,
    trade_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    mode TEXT NOT NULL,
    FOREIGN KEY (order_id) REFERENCES orders(id)
);
```

#### Database Migrations
```python
# Example: Database migration
from src.config.database import get_connection

def migrate_to_v2():
    """Migrate database to version 2."""
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # Add new columns
        cursor.execute("""
            ALTER TABLE positions 
            ADD COLUMN stop_loss_price REAL
        """)
        
        cursor.execute("""
            ALTER TABLE positions 
            ADD COLUMN take_profit_price REAL
        """)
        
        # Create new table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS risk_metrics (
                id TEXT PRIMARY KEY,
                position_id TEXT NOT NULL,
                var_1d REAL,
                var_5d REAL,
                beta REAL,
                calculated_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (position_id) REFERENCES positions(id)
            )
        """)
        
        conn.commit()
        print("Migration to v2 completed successfully")

def rollback_from_v2():
    """Rollback migration from version 2."""
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # Drop new columns
        cursor.execute("""
            ALTER TABLE positions 
            DROP COLUMN stop_loss_price
        """)
        
        cursor.execute("""
            ALTER TABLE positions 
            DROP COLUMN take_profit_price
        """)
        
        # Drop new table
        cursor.execute("DROP TABLE IF EXISTS risk_metrics")
        
        conn.commit()
        print("Rollback from v2 completed successfully")
```

## AI Development

### Model Integration

#### Adding New AI Models
```python
# Example: Custom AI model
from abc import ABC, abstractmethod
from typing import Dict, Any
from src.ai.multi_model import ModelConfig, ModelOpinion

class CustomAIModel(ABC):
    """Base class for custom AI models."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.name = config.name
        self.model_id = config.model_id
    
    @abstractmethod
    def analyze(self, symbol: str, analysis_type: str, context: Dict[str, Any]) -> ModelOpinion:
        """Analyze stock and return opinion."""
        pass
    
    @abstractmethod
    def get_confidence(self, analysis: ModelOpinion) -> float:
        """Calculate confidence score for analysis."""
        pass

class TechnicalAnalysisModel(CustomAIModel):
    """Technical analysis AI model."""
    
    def analyze(self, symbol: str, analysis_type: str, context: Dict[str, Any]) -> ModelOpinion:
        """Perform technical analysis."""
        # Get historical data
        historical_data = context.get('historical_data', [])
        
        # Perform technical analysis
        signal = self._calculate_technical_signals(historical_data)
        confidence = self._calculate_confidence(signal)
        
        return ModelOpinion(
            model_id=self.model_id,
            symbol=symbol,
            signal=signal,
            confidence=confidence,
            reasoning="Technical analysis based on moving averages and RSI",
            timestamp=datetime.now()
        )
    
    def _calculate_technical_signals(self, data: List[Dict]) -> str:
        """Calculate technical trading signals."""
        # Implementation of technical analysis
        pass
    
    def _calculate_confidence(self, signal: str) -> float:
        """Calculate confidence based on signal strength."""
        # Implementation of confidence calculation
        pass
```

#### Model Performance Monitoring
```python
# Example: Model performance tracking
from dataclasses import dataclass
from typing import List
import statistics

@dataclass
class ModelPerformance:
    model_id: str
    accuracy: float
    latency: float
    confidence: float
    total_predictions: int
    correct_predictions: int

class ModelPerformanceTracker:
    def __init__(self):
        self.performance_history = {}
        self.prediction_history = {}
    
    def record_prediction(self, model_id: str, prediction: str, actual: str, latency: float):
        """Record model prediction for performance tracking."""
        if model_id not in self.prediction_history:
            self.prediction_history[model_id] = []
        
        self.prediction_history[model_id].append({
            'prediction': prediction,
            'actual': actual,
            'latency': latency,
            'timestamp': datetime.now()
        })
    
    def calculate_performance(self, model_id: str) -> ModelPerformance:
        """Calculate model performance metrics."""
        if model_id not in self.prediction_history:
            return None
        
        predictions = self.prediction_history[model_id]
        total = len(predictions)
        correct = sum(1 for p in predictions if p['prediction'] == p['actual'])
        accuracy = correct / total if total > 0 else 0
        
        latencies = [p['latency'] for p in predictions]
        avg_latency = statistics.mean(latencies) if latencies else 0
        
        confidences = [p.get('confidence', 0) for p in predictions]
        avg_confidence = statistics.mean(confidences) if confidences else 0
        
        return ModelPerformance(
            model_id=model_id,
            accuracy=accuracy,
            latency=avg_latency,
            confidence=avg_confidence,
            total_predictions=total,
            correct_predictions=correct
        )
```

## Security Development

### Security Best Practices

#### Input Validation
```python
# Example: Input validation
from typing import Any
import re

class InputValidator:
    """Input validation for security."""
    
    @staticmethod
    def validate_symbol(symbol: str) -> bool:
        """Validate stock symbol format."""
        if not symbol or not isinstance(symbol, str):
            return False
        
        # Check symbol format (alphanumeric, 1-10 characters)
        pattern = r'^[A-Za-z0-9]{1,10}$'
        return bool(re.match(pattern, symbol))
    
    @staticmethod
    def validate_quantity(quantity: Any) -> bool:
        """Validate order quantity."""
        try:
            qty = int(quantity)
            return 1 <= qty <= 10000  # Reasonable quantity limits
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_price(price: Any) -> bool:
        """Validate stock price."""
        try:
            p = float(price)
            return 0.01 <= p <= 10000.0  # Reasonable price limits
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def sanitize_string(input_str: str) -> str:
        """Sanitize string input to prevent injection attacks."""
        if not isinstance(input_str, str):
            return ""
        
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>"\';\\]', '', input_str)
        return sanitized.strip()
```

#### API Security
```python
# Example: API security middleware
from functools import wraps
from flask import request, jsonify
import jwt
import time

def require_auth(f):
    """Decorator to require authentication for API endpoints."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        
        if not token:
            return jsonify({'error': 'No token provided'}), 401
        
        try:
            # Remove 'Bearer ' prefix
            token = token.split(' ')[1]
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            current_user = data['user_id']
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
        
        return f(current_user, *args, **kwargs)
    
    return decorated_function

def rate_limit(max_requests: int, window: int):
    """Rate limiting decorator."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Implementation of rate limiting
            pass
        return decorated_function
    return decorator

# Usage
@app.route('/api/v1/positions', methods=['POST'])
@require_auth
@rate_limit(max_requests=100, window=60)
def create_position(user_id, *args, **kwargs):
    """Create position with authentication and rate limiting."""
    pass
```

## Performance Optimization

### Caching Strategies
```python
# Example: Redis caching
import redis
import json
from typing import Any, Optional
from functools import wraps

class CacheManager:
    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0):
        self.redis_client = redis.Redis(host=host, port=port, db=db)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            value = self.redis_client.get(key)
            return json.loads(value) if value else None
        except Exception:
            return None
    
    def set(self, key: str, value: Any, ttl: int = 300):
        """Set value in cache with TTL."""
        try:
            self.redis_client.setex(key, ttl, json.dumps(value))
        except Exception:
            pass
    
    def delete(self, key: str):
        """Delete value from cache."""
        try:
            self.redis_client.delete(key)
        except Exception:
            pass

def cache_result(ttl: int = 300):
    """Decorator to cache function results."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_manager = CacheManager()
            
            # Generate cache key
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_manager.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator

# Usage
@cache_result(ttl=60)
def get_market_quote(symbol: str) -> Dict[str, Any]:
    """Get market quote with caching."""
    # Implementation
    pass
```

### Database Optimization
```python
# Example: Database connection pooling
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager

class DatabaseManager:
    def __init__(self, database_url: str):
        self.engine = create_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=20,
            max_overflow=30,
            pool_timeout=30,
            pool_recycle=3600,
            echo=False
        )
    
    @contextmanager
    def get_connection(self):
        """Get database connection with proper cleanup."""
        connection = self.engine.connect()
        try:
            yield connection
        finally:
            connection.close()
    
    def execute_query(self, query: str, params: dict = None):
        """Execute query with connection pooling."""
        with self.get_connection() as conn:
            result = conn.execute(query, params or {})
            return result.fetchall()
    
    def execute_update(self, query: str, params: dict = None):
        """Execute update with connection pooling."""
        with self.get_connection() as conn:
            result = conn.execute(query, params or {})
            conn.commit()
            return result.rowcount
```

## Testing and Quality Assurance

### Test Automation
```python
# Example: Test automation setup
import pytest
import asyncio
from unittest.mock import Mock, patch
from src.workflows.trading_cycle import TradingCycle

class TestTradingCycle:
    @pytest.fixture
    def trading_cycle(self):
        """Create trading cycle instance for testing."""
        return TradingCycle()
    
    @pytest.fixture
    def mock_position_manager(self):
        """Mock position manager."""
        return Mock()
    
    @pytest.fixture
    def mock_execution_engine(self):
        """Mock execution engine."""
        return Mock()
    
    def test_trading_cycle_initialization(self, trading_cycle):
        """Test trading cycle initialization."""
        assert trading_cycle is not None
        assert trading_cycle.position_manager is not None
        assert trading_cycle.execution_engine is not None
    
    @patch('src.trading.positions.get_position_manager')
    def test_create_position(self, mock_get_position_manager, trading_cycle):
        """Test position creation."""
        mock_position_manager = Mock()
        mock_get_position_manager.return_value = mock_position_manager
        
        result = trading_cycle.create_position('AAPL', 100, 150.0)
        
        mock_position_manager.create_position.assert_called_once_with('AAPL', 100, 150.0)
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_async_trading_cycle(self, trading_cycle):
        """Test asynchronous trading cycle operations."""
        result = await trading_cycle.async_execute_cycle()
        assert result.success is True
```

### Continuous Integration
```yaml
# Example: GitHub Actions CI/CD
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.11
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run tests
      run: |
        pytest tests/ --cov=src --cov-report=xml
    
    - name: Run linting
      run: |
        flake8 src/
        black --check src/
        mypy src/
    
    - name: Upload coverage
      uses: codecov/codecov-action@v1
      with:
        file: ./coverage.xml
```

## Deployment and DevOps

### Docker Configuration
```dockerfile
# Example: Multi-stage Dockerfile
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY . .

# Create non-root user
RUN useradd -m -u 1000 tradingbot
RUN chown -R tradingbot:tradingbot /app
USER tradingbot

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8050/health || exit 1

# Expose port
EXPOSE 8050

# Run application
CMD ["python", "src/main.py"]
```

### Kubernetes Deployment
```yaml
# Example: Kubernetes deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tradingbot
  labels:
    app: tradingbot
spec:
  replicas: 3
  selector:
    matchLabels:
      app: tradingbot
  template:
    metadata:
      labels:
        app: tradingbot
    spec:
      containers:
      - name: tradingbot
        image: tradingbot:latest
        ports:
        - containerPort: 8050
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: tradingbot-secrets
              key: database-url
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8050
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8050
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: tradingbot-service
spec:
  selector:
    app: tradingbot
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8050
  type: LoadBalancer
```

## Contributing Guidelines

### Code Review Process
1. **Self Review**: Review your own code before submitting
2. **Peer Review**: At least one peer review required
3. **Automated Checks**: All CI/CD checks must pass
4. **Documentation**: Update documentation for new features
5. **Testing**: Add tests for new functionality

### Contribution Checklist
- [ ] Code follows style guidelines
- [ ] Tests are added/updated
- [ ] Documentation is updated
- [ ] No breaking changes (or properly documented)
- [ ] Security considerations addressed
- [ ] Performance impact assessed

### Issue Reporting
When reporting issues, please include:
1. **Description**: Clear description of the issue
2. **Steps to Reproduce**: Detailed steps to reproduce
3. **Expected Behavior**: What should happen
4. **Actual Behavior**: What actually happens
5. **Environment**: System and software versions
6. **Logs**: Relevant log files and error messages

## Conclusion

This developer guide provides comprehensive information for developing and contributing to the AI Trading System. Follow the guidelines and best practices to ensure code quality, security, and maintainability.

For additional information or questions, please refer to the documentation or contact the development team.
