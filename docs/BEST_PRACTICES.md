# AI Trading System - Best Practices

## Overview

This document outlines best practices for using, developing, and maintaining the AI Trading System. Following these practices ensures optimal performance, security, and reliability.

## Trading Best Practices

### Risk Management

#### Position Sizing
```python
# Good: Risk-based position sizing
def calculate_position_size(account_balance: float, risk_percent: float, 
                          stock_price: float, stop_loss_percent: float) -> int:
    """Calculate position size based on risk management."""
    risk_amount = account_balance * risk_percent
    stop_loss_amount = stock_price * stop_loss_percent
    position_size = risk_amount / stop_loss_amount
    return int(position_size)

# Example usage
account_balance = 10000.0
risk_percent = 0.02  # 2% risk per trade
stock_price = 100.0
stop_loss_percent = 0.05  # 5% stop loss

position_size = calculate_position_size(
    account_balance, risk_percent, stock_price, stop_loss_percent
)
# Result: 40 shares (2% of $10,000 = $200 risk / $5 stop loss = 40 shares)
```

#### Portfolio Diversification
```python
# Good: Diversified portfolio management
class PortfolioManager:
    def __init__(self, max_positions: int = 10, max_sector_exposure: float = 0.3):
        self.max_positions = max_positions
        self.max_sector_exposure = max_sector_exposure
        self.positions = {}
        self.sector_exposure = {}
    
    def can_add_position(self, symbol: str, sector: str, size: float) -> bool:
        """Check if position can be added without violating diversification rules."""
        # Check position count limit
        if len(self.positions) >= self.max_positions:
            return False
        
        # Check sector exposure limit
        current_sector_exposure = self.sector_exposure.get(sector, 0)
        if current_sector_exposure + size > self.max_sector_exposure:
            return False
        
        return True
```

#### Stop Loss and Take Profit
```python
# Good: Automated stop loss and take profit
class PositionManager:
    def __init__(self):
        self.positions = {}
    
    def set_stop_loss(self, position_id: str, stop_loss_percent: float):
        """Set stop loss for position."""
        position = self.positions.get(position_id)
        if position:
            stop_loss_price = position.entry_price * (1 - stop_loss_percent)
            position.stop_loss_price = stop_loss_price
    
    def set_take_profit(self, position_id: str, take_profit_percent: float):
        """Set take profit for position."""
        position = self.positions.get(position_id)
        if position:
            take_profit_price = position.entry_price * (1 + take_profit_percent)
            position.take_profit_price = take_profit_price
    
    def check_exit_conditions(self, position_id: str, current_price: float):
        """Check if position should be exited based on stop loss or take profit."""
        position = self.positions.get(position_id)
        if not position:
            return None
        
        if position.stop_loss_price and current_price <= position.stop_loss_price:
            return "stop_loss"
        
        if position.take_profit_price and current_price >= position.take_profit_price:
            return "take_profit"
        
        return None
```

### Market Analysis

#### Technical Analysis
```python
# Good: Comprehensive technical analysis
import pandas as pd
import numpy as np
from typing import Dict, List

class TechnicalAnalyzer:
    def __init__(self, lookback_period: int = 20):
        self.lookback_period = lookback_period
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate technical indicators."""
        indicators = {}
        
        # Moving averages
        indicators['sma_20'] = data['close'].rolling(20).mean().iloc[-1]
        indicators['sma_50'] = data['close'].rolling(50).mean().iloc[-1]
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        indicators['rsi'] = 100 - (100 / (1 + rs.iloc[-1]))
        
        # MACD
        ema_12 = data['close'].ewm(span=12).mean()
        ema_26 = data['close'].ewm(span=26).mean()
        indicators['macd'] = ema_12.iloc[-1] - ema_26.iloc[-1]
        
        # Bollinger Bands
        sma_20 = data['close'].rolling(20).mean()
        std_20 = data['close'].rolling(20).std()
        indicators['bb_upper'] = (sma_20 + 2 * std_20).iloc[-1]
        indicators['bb_lower'] = (sma_20 - 2 * std_20).iloc[-1]
        indicators['bb_position'] = (data['close'].iloc[-1] - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower'])
        
        return indicators
    
    def generate_signal(self, indicators: Dict[str, float]) -> str:
        """Generate trading signal based on technical indicators."""
        signals = []
        
        # Moving average crossover
        if indicators['sma_20'] > indicators['sma_50']:
            signals.append('buy')
        else:
            signals.append('sell')
        
        # RSI signals
        if indicators['rsi'] < 30:
            signals.append('buy')  # Oversold
        elif indicators['rsi'] > 70:
            signals.append('sell')  # Overbought
        
        # MACD signals
        if indicators['macd'] > 0:
            signals.append('buy')
        else:
            signals.append('sell')
        
        # Bollinger Bands
        if indicators['bb_position'] < 0.2:
            signals.append('buy')  # Near lower band
        elif indicators['bb_position'] > 0.8:
            signals.append('sell')  # Near upper band
        
        # Majority vote
        buy_signals = signals.count('buy')
        sell_signals = signals.count('sell')
        
        if buy_signals > sell_signals:
            return 'buy'
        elif sell_signals > buy_signals:
            return 'sell'
        else:
            return 'hold'
```

#### Fundamental Analysis
```python
# Good: Fundamental analysis integration
class FundamentalAnalyzer:
    def __init__(self):
        self.metrics = {}
    
    def analyze_fundamentals(self, symbol: str, data: Dict) -> Dict[str, float]:
        """Analyze fundamental metrics."""
        analysis = {}
        
        # Price-to-Earnings ratio
        if 'pe_ratio' in data:
            analysis['pe_ratio'] = data['pe_ratio']
            analysis['pe_signal'] = 'buy' if data['pe_ratio'] < 15 else 'sell'
        
        # Price-to-Book ratio
        if 'pb_ratio' in data:
            analysis['pb_ratio'] = data['pb_ratio']
            analysis['pb_signal'] = 'buy' if data['pb_ratio'] < 1.5 else 'sell'
        
        # Debt-to-Equity ratio
        if 'debt_to_equity' in data:
            analysis['debt_to_equity'] = data['debt_to_equity']
            analysis['debt_signal'] = 'buy' if data['debt_to_equity'] < 0.5 else 'sell'
        
        # Return on Equity
        if 'roe' in data:
            analysis['roe'] = data['roe']
            analysis['roe_signal'] = 'buy' if data['roe'] > 0.15 else 'sell'
        
        # Revenue Growth
        if 'revenue_growth' in data:
            analysis['revenue_growth'] = data['revenue_growth']
            analysis['growth_signal'] = 'buy' if data['revenue_growth'] > 0.1 else 'sell'
        
        return analysis
```

## AI Model Best Practices

### Model Selection and Management

#### Model Performance Monitoring
```python
# Good: Model performance tracking
from dataclasses import dataclass
from typing import List, Dict
import statistics

@dataclass
class ModelPerformance:
    model_id: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    latency: float
    confidence: float

class ModelPerformanceTracker:
    def __init__(self):
        self.performance_history = {}
        self.prediction_history = {}
    
    def record_prediction(self, model_id: str, prediction: str, actual: str, 
                         confidence: float, latency: float):
        """Record model prediction for performance tracking."""
        if model_id not in self.prediction_history:
            self.prediction_history[model_id] = []
        
        self.prediction_history[model_id].append({
            'prediction': prediction,
            'actual': actual,
            'confidence': confidence,
            'latency': latency,
            'timestamp': datetime.now()
        })
    
    def calculate_performance_metrics(self, model_id: str) -> ModelPerformance:
        """Calculate comprehensive performance metrics."""
        if model_id not in self.prediction_history:
            return None
        
        predictions = self.prediction_history[model_id]
        
        # Calculate accuracy
        correct = sum(1 for p in predictions if p['prediction'] == p['actual'])
        total = len(predictions)
        accuracy = correct / total if total > 0 else 0
        
        # Calculate precision, recall, F1-score
        true_positives = sum(1 for p in predictions 
                           if p['prediction'] == 'buy' and p['actual'] == 'buy')
        false_positives = sum(1 for p in predictions 
                            if p['prediction'] == 'buy' and p['actual'] != 'buy')
        false_negatives = sum(1 for p in predictions 
                            if p['prediction'] != 'buy' and p['actual'] == 'buy')
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate average latency and confidence
        latencies = [p['latency'] for p in predictions]
        confidences = [p['confidence'] for p in predictions]
        
        avg_latency = statistics.mean(latencies) if latencies else 0
        avg_confidence = statistics.mean(confidences) if confidences else 0
        
        return ModelPerformance(
            model_id=model_id,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            latency=avg_latency,
            confidence=avg_confidence
        )
```

#### Model Ensemble Management
```python
# Good: Ensemble model management
class ModelEnsemble:
    def __init__(self):
        self.models = {}
        self.weights = {}
        self.performance_tracker = ModelPerformanceTracker()
    
    def add_model(self, model_id: str, model: Any, initial_weight: float = 1.0):
        """Add model to ensemble."""
        self.models[model_id] = model
        self.weights[model_id] = initial_weight
    
    def update_weights(self):
        """Update model weights based on performance."""
        for model_id in self.models:
            performance = self.performance_tracker.calculate_performance_metrics(model_id)
            if performance:
                # Weight based on accuracy and confidence
                self.weights[model_id] = performance.accuracy * performance.confidence
    
    def get_ensemble_prediction(self, symbol: str, context: Dict) -> Dict:
        """Get ensemble prediction with weighted voting."""
        predictions = {}
        total_weight = sum(self.weights.values())
        
        for model_id, model in self.models.items():
            try:
                prediction = model.analyze(symbol, context)
                predictions[model_id] = {
                    'prediction': prediction.signal,
                    'confidence': prediction.confidence,
                    'weight': self.weights[model_id] / total_weight
                }
            except Exception as e:
                print(f"Error in model {model_id}: {e}")
                continue
        
        # Weighted voting
        weighted_votes = {'buy': 0, 'sell': 0, 'hold': 0}
        for pred in predictions.values():
            weighted_votes[pred['prediction']] += pred['weight'] * pred['confidence']
        
        # Determine final prediction
        final_prediction = max(weighted_votes, key=weighted_votes.get)
        final_confidence = weighted_votes[final_prediction]
        
        return {
            'signal': final_prediction,
            'confidence': final_confidence,
            'individual_predictions': predictions,
            'reasoning': f"Ensemble prediction based on {len(predictions)} models"
        }
```

### Model Training and Validation

#### Cross-Validation
```python
# Good: Cross-validation for model training
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

class ModelValidator:
    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits
        self.tscv = TimeSeriesSplit(n_splits=n_splits)
    
    def validate_model(self, model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Perform time series cross-validation."""
        scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': []
        }
        
        for train_idx, test_idx in self.tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            scores['accuracy'].append(accuracy_score(y_test, y_pred))
            scores['precision'].append(precision_score(y_test, y_pred, average='weighted'))
            scores['recall'].append(recall_score(y_test, y_pred, average='weighted'))
            scores['f1_score'].append(f1_score(y_test, y_pred, average='weighted'))
        
        # Return average scores
        return {metric: np.mean(values) for metric, values in scores.items()}
```

#### Model Retraining
```python
# Good: Automated model retraining
class ModelRetrainer:
    def __init__(self, retrain_threshold: float = 0.05):
        self.retrain_threshold = retrain_threshold
        self.last_performance = {}
    
    def should_retrain(self, model_id: str, current_performance: float) -> bool:
        """Determine if model should be retrained."""
        if model_id not in self.last_performance:
            self.last_performance[model_id] = current_performance
            return False
        
        performance_degradation = self.last_performance[model_id] - current_performance
        return performance_degradation > self.retrain_threshold
    
    def retrain_model(self, model_id: str, new_data: pd.DataFrame):
        """Retrain model with new data."""
        model = self.get_model(model_id)
        
        # Prepare training data
        X, y = self.prepare_training_data(new_data)
        
        # Retrain model
        model.fit(X, y)
        
        # Validate retrained model
        validator = ModelValidator()
        performance = validator.validate_model(model, X, y)
        
        # Update performance tracking
        self.last_performance[model_id] = performance['accuracy']
        
        return model, performance
```

## System Performance Best Practices

### Database Optimization

#### Connection Pooling
```python
# Good: Database connection pooling
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
    
    def execute_batch_query(self, queries: List[str], params: List[Dict]):
        """Execute multiple queries in batch for better performance."""
        with self.get_connection() as conn:
            for query, param in zip(queries, params):
                conn.execute(query, param)
            conn.commit()
```

#### Query Optimization
```python
# Good: Optimized database queries
class OptimizedQueries:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def get_positions_with_performance(self, mode: str) -> List[Dict]:
        """Get positions with performance metrics in single query."""
        query = """
        SELECT 
            p.id,
            p.symbol,
            p.quantity,
            p.entry_price,
            p.current_price,
            (p.current_price - p.entry_price) * p.quantity as unrealized_pnl,
            ((p.current_price - p.entry_price) / p.entry_price) * 100 as return_percent,
            p.entry_time,
            p.status
        FROM positions p
        WHERE p.mode = :mode AND p.status = 'open'
        ORDER BY p.entry_time DESC
        """
        
        with self.db_manager.get_connection() as conn:
            result = conn.execute(query, {'mode': mode})
            return [dict(row) for row in result]
    
    def get_portfolio_summary(self, mode: str) -> Dict:
        """Get portfolio summary with aggregated data."""
        query = """
        SELECT 
            COUNT(*) as position_count,
            SUM(quantity * current_price) as total_value,
            SUM((current_price - entry_price) * quantity) as total_pnl,
            AVG((current_price - entry_price) / entry_price) * 100 as avg_return
        FROM positions
        WHERE mode = :mode AND status = 'open'
        """
        
        with self.db_manager.get_connection() as conn:
            result = conn.execute(query, {'mode': mode})
            return dict(result.fetchone())
```

### Caching Strategies

#### Multi-Level Caching
```python
# Good: Multi-level caching implementation
import redis
import json
from typing import Any, Optional
from functools import wraps

class MultiLevelCache:
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=0)
        self.local_cache = {}
        self.cache_stats = {'hits': 0, 'misses': 0}
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache (local first, then Redis)."""
        # Try local cache first
        if key in self.local_cache:
            self.cache_stats['hits'] += 1
            return self.local_cache[key]
        
        # Try Redis cache
        try:
            value = self.redis_client.get(key)
            if value:
                data = json.loads(value)
                # Store in local cache for faster access
                self.local_cache[key] = data
                self.cache_stats['hits'] += 1
                return data
        except Exception:
            pass
        
        self.cache_stats['misses'] += 1
        return None
    
    def set(self, key: str, value: Any, ttl: int = 300):
        """Set value in both local and Redis cache."""
        # Store in local cache
        self.local_cache[key] = value
        
        # Store in Redis cache
        try:
            self.redis_client.setex(key, ttl, json.dumps(value))
        except Exception:
            pass
    
    def invalidate(self, key: str):
        """Invalidate cache entry."""
        if key in self.local_cache:
            del self.local_cache[key]
        
        try:
            self.redis_client.delete(key)
        except Exception:
            pass

def cache_result(cache: MultiLevelCache, ttl: int = 300):
    """Decorator to cache function results."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator
```

### Async Processing

#### Async Task Management
```python
# Good: Async task management
import asyncio
import aiohttp
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

class AsyncTaskManager:
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def fetch_multiple_quotes(self, symbols: List[str]) -> Dict[str, Dict]:
        """Fetch multiple stock quotes concurrently."""
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_quote(session, symbol) for symbol in symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            quotes = {}
            for symbol, result in zip(symbols, results):
                if isinstance(result, Exception):
                    print(f"Error fetching quote for {symbol}: {result}")
                else:
                    quotes[symbol] = result
            
            return quotes
    
    async def fetch_quote(self, session: aiohttp.ClientSession, symbol: str) -> Dict:
        """Fetch single stock quote."""
        url = f"https://api.questrade.com/v1/markets/quotes/{symbol}"
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                return data
            else:
                raise Exception(f"HTTP {response.status}")
    
    async def process_trading_signals(self, symbols: List[str]) -> List[Dict]:
        """Process trading signals for multiple symbols concurrently."""
        tasks = [self.analyze_symbol(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        signals = []
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                print(f"Error analyzing {symbol}: {result}")
            else:
                signals.append(result)
        
        return signals
    
    async def analyze_symbol(self, symbol: str) -> Dict:
        """Analyze single symbol for trading signals."""
        # Run CPU-intensive analysis in thread pool
        loop = asyncio.get_event_loop()
        analysis = await loop.run_in_executor(
            self.executor, 
            self._cpu_intensive_analysis, 
            symbol
        )
        return analysis
    
    def _cpu_intensive_analysis(self, symbol: str) -> Dict:
        """CPU-intensive analysis (runs in thread pool)."""
        # Implementation of analysis
        pass
```

## Security Best Practices

### Input Validation and Sanitization

#### Comprehensive Input Validation
```python
# Good: Comprehensive input validation
import re
from typing import Any, Union
from decimal import Decimal, InvalidOperation

class InputValidator:
    """Comprehensive input validation for security."""
    
    @staticmethod
    def validate_symbol(symbol: Any) -> bool:
        """Validate stock symbol format."""
        if not isinstance(symbol, str):
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
        """Validate stock price with decimal precision."""
        try:
            p = Decimal(str(price))
            return Decimal('0.01') <= p <= Decimal('10000.00')
        except (InvalidOperation, ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_percentage(percentage: Any) -> bool:
        """Validate percentage values."""
        try:
            pct = float(percentage)
            return 0.0 <= pct <= 100.0
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def sanitize_string(input_str: Any) -> str:
        """Sanitize string input to prevent injection attacks."""
        if not isinstance(input_str, str):
            return ""
        
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>"\';\\]', '', input_str)
        return sanitized.strip()
    
    @staticmethod
    def validate_json_input(data: Any) -> bool:
        """Validate JSON input structure."""
        if not isinstance(data, dict):
            return False
        
        # Check for required fields
        required_fields = ['symbol', 'quantity', 'price']
        for field in required_fields:
            if field not in data:
                return False
        
        # Validate each field
        return (InputValidator.validate_symbol(data['symbol']) and
                InputValidator.validate_quantity(data['quantity']) and
                InputValidator.validate_price(data['price']))
```

#### API Security
```python
# Good: API security implementation
from functools import wraps
from flask import request, jsonify
import jwt
import time
import hashlib
import hmac

class APISecurity:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.rate_limit_storage = {}
    
    def require_auth(self, f):
        """Decorator to require authentication for API endpoints."""
        @wraps(f)
        def decorated_function(*args, **kwargs):
            token = request.headers.get('Authorization')
            
            if not token:
                return jsonify({'error': 'No token provided'}), 401
            
            try:
                # Remove 'Bearer ' prefix
                token = token.split(' ')[1]
                data = jwt.decode(token, self.secret_key, algorithms=['HS256'])
                current_user = data['user_id']
                
                # Check token expiration
                if data['exp'] < time.time():
                    return jsonify({'error': 'Token expired'}), 401
                
            except jwt.ExpiredSignatureError:
                return jsonify({'error': 'Token expired'}), 401
            except jwt.InvalidTokenError:
                return jsonify({'error': 'Invalid token'}), 401
            
            return f(current_user, *args, **kwargs)
        
        return decorated_function
    
    def rate_limit(self, max_requests: int, window: int):
        """Rate limiting decorator."""
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                client_ip = request.remote_addr
                current_time = time.time()
                
                # Clean old entries
                self.rate_limit_storage = {
                    ip: requests for ip, requests in self.rate_limit_storage.items()
                    if any(req_time > current_time - window for req_time in requests)
                }
                
                # Check rate limit
                if client_ip in self.rate_limit_storage:
                    requests = self.rate_limit_storage[client_ip]
                    recent_requests = [req_time for req_time in requests if req_time > current_time - window]
                    
                    if len(recent_requests) >= max_requests:
                        return jsonify({'error': 'Rate limit exceeded'}), 429
                    
                    recent_requests.append(current_time)
                    self.rate_limit_storage[client_ip] = recent_requests
                else:
                    self.rate_limit_storage[client_ip] = [current_time]
                
                return f(*args, **kwargs)
            
            return decorated_function
        return decorator
    
    def validate_request_signature(self, f):
        """Validate request signature for API integrity."""
        @wraps(f)
        def decorated_function(*args, **kwargs):
            signature = request.headers.get('X-Signature')
            timestamp = request.headers.get('X-Timestamp')
            
            if not signature or not timestamp:
                return jsonify({'error': 'Missing signature or timestamp'}), 400
            
            # Check timestamp (prevent replay attacks)
            current_time = time.time()
            if abs(current_time - int(timestamp)) > 300:  # 5 minutes
                return jsonify({'error': 'Request too old'}), 400
            
            # Validate signature
            expected_signature = hmac.new(
                self.secret_key.encode(),
                f"{request.method}{request.path}{timestamp}".encode(),
                hashlib.sha256
            ).hexdigest()
            
            if not hmac.compare_digest(signature, expected_signature):
                return jsonify({'error': 'Invalid signature'}), 401
            
            return f(*args, **kwargs)
        
        return decorated_function
```

### Data Protection

#### Encryption and Hashing
```python
# Good: Data encryption and hashing
from cryptography.fernet import Fernet
import hashlib
import secrets
from typing import Union

class DataProtection:
    def __init__(self, encryption_key: bytes = None):
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        return self.cipher.decrypt(encrypted_data.encode()).decode()
    
    def hash_password(self, password: str, salt: str = None) -> tuple:
        """Hash password with salt."""
        if salt is None:
            salt = secrets.token_hex(32)
        
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode(),
            salt.encode(),
            100000  # iterations
        )
        
        return password_hash.hex(), salt
    
    def verify_password(self, password: str, password_hash: str, salt: str) -> bool:
        """Verify password against hash."""
        computed_hash, _ = self.hash_password(password, salt)
        return computed_hash == password_hash
    
    def mask_sensitive_data(self, data: str, mask_char: str = '*') -> str:
        """Mask sensitive data for logging."""
        if len(data) <= 4:
            return mask_char * len(data)
        
        return data[:2] + mask_char * (len(data) - 4) + data[-2:]
```

## Monitoring and Logging Best Practices

### Comprehensive Logging

#### Structured Logging
```python
# Good: Structured logging implementation
import logging
import json
from datetime import datetime
from typing import Dict, Any
from dataclasses import dataclass, asdict

@dataclass
class LogEntry:
    timestamp: str
    level: str
    component: str
    message: str
    data: Dict[str, Any] = None
    user_id: str = None
    session_id: str = None
    request_id: str = None

class StructuredLogger:
    def __init__(self, name: str, level: str = 'INFO'):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create handler
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def log(self, level: str, message: str, component: str, 
            data: Dict[str, Any] = None, **kwargs):
        """Log structured message."""
        log_entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            level=level.upper(),
            component=component,
            message=message,
            data=data,
            **kwargs
        )
        
        # Convert to JSON for structured logging
        log_message = json.dumps(asdict(log_entry))
        
        # Log with appropriate level
        getattr(self.logger, level.lower())(log_message)
    
    def log_trade(self, trade_data: Dict[str, Any], user_id: str = None):
        """Log trading activity."""
        self.log(
            level='INFO',
            message='Trade executed',
            component='trading',
            data=trade_data,
            user_id=user_id
        )
    
    def log_error(self, error: Exception, component: str, context: Dict[str, Any] = None):
        """Log error with context."""
        self.log(
            level='ERROR',
            message=str(error),
            component=component,
            data={
                'error_type': type(error).__name__,
                'context': context or {}
            }
        )
    
    def log_performance(self, operation: str, duration: float, metrics: Dict[str, Any] = None):
        """Log performance metrics."""
        self.log(
            level='INFO',
            message=f'Performance metric: {operation}',
            component='performance',
            data={
                'operation': operation,
                'duration': duration,
                'metrics': metrics or {}
            }
        )
```

### Performance Monitoring

#### Real-time Performance Tracking
```python
# Good: Real-time performance monitoring
import time
import psutil
import threading
from typing import Dict, List
from collections import deque
from dataclasses import dataclass

@dataclass
class PerformanceMetric:
    timestamp: float
    cpu_percent: float
    memory_percent: float
    disk_io: Dict[str, float]
    network_io: Dict[str, float]
    active_connections: int
    response_time: float

class PerformanceMonitor:
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics_history = deque(maxlen=max_history)
        self.is_monitoring = False
        self.monitor_thread = None
        self.response_times = deque(maxlen=100)
    
    def start_monitoring(self, interval: float = 1.0):
        """Start performance monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self, interval: float):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                metric = self._collect_metrics()
                self.metrics_history.append(metric)
                time.sleep(interval)
            except Exception as e:
                print(f"Error in performance monitoring: {e}")
    
    def _collect_metrics(self) -> PerformanceMetric:
        """Collect current performance metrics."""
        # CPU and memory
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        disk_io_dict = {
            'read_bytes': disk_io.read_bytes if disk_io else 0,
            'write_bytes': disk_io.write_bytes if disk_io else 0
        }
        
        # Network I/O
        network_io = psutil.net_io_counters()
        network_io_dict = {
            'bytes_sent': network_io.bytes_sent,
            'bytes_recv': network_io.bytes_recv
        }
        
        # Active connections
        connections = len(psutil.net_connections())
        
        # Average response time
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        
        return PerformanceMetric(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            disk_io=disk_io_dict,
            network_io=network_io_dict,
            active_connections=connections,
            response_time=avg_response_time
        )
    
    def record_response_time(self, response_time: float):
        """Record API response time."""
        self.response_times.append(response_time)
    
    def get_performance_summary(self) -> Dict[str, float]:
        """Get performance summary."""
        if not self.metrics_history:
            return {}
        
        recent_metrics = list(self.metrics_history)[-10:]  # Last 10 metrics
        
        return {
            'avg_cpu': sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics),
            'avg_memory': sum(m.memory_percent for m in recent_metrics) / len(recent_metrics),
            'avg_response_time': sum(m.response_time for m in recent_metrics) / len(recent_metrics),
            'max_cpu': max(m.cpu_percent for m in recent_metrics),
            'max_memory': max(m.memory_percent for m in recent_metrics),
            'active_connections': recent_metrics[-1].active_connections
        }
```

## Conclusion

Following these best practices ensures optimal performance, security, and reliability of the AI Trading System. Regular review and updates of these practices help maintain system quality and adapt to changing requirements.

Remember to:
- Regularly review and update practices
- Monitor system performance and adjust accordingly
- Stay informed about new security threats and mitigation strategies
- Continuously improve based on system metrics and user feedback
- Document any deviations from standard practices
