# 🚀 Unified Trading API Documentation

## Overview

The **Unified Trading API** is a comprehensive, single-interface API that brings together all components of your TradingBOT system. It provides a complete REST API for:

- **AI Trading System** (MasterOrchestrator integration)
- **Order Execution** (Live/Demo modes)
- **Portfolio Management**
- **Risk Management**
- **Market Data**
- **Performance Analytics**
- **System Monitoring**
- **Session Management**

## 🚀 Quick Start

### 1. Start the API Server
```bash
python unified_trading_api.py
```

### 2. Access the API
- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Root Page**: http://localhost:8000/

### 3. Test the API
```bash
# Check system status
curl http://localhost:8000/api/status

# Start a trading session
curl -X POST http://localhost:8000/api/session/start \
  -H "Content-Type: application/json" \
  -d '{"capital": 10000, "mode": "DEMO"}'
```

## 📊 API Endpoints

### System Management

#### `GET /api/status`
Get comprehensive system status including all components.

**Response:**
```json
{
  "status": "OPERATIONAL",
  "timestamp": "2025-01-15T10:30:00",
  "mode": "DEMO",
  "health": "HEALTHY",
  "components": {
    "master_orchestrator": "ACTIVE",
    "order_executor": "ACTIVE",
    "position_manager": "ACTIVE",
    "risk_manager": "ACTIVE",
    "ai_engine": "ACTIVE",
    "state_manager": "ACTIVE",
    "system_monitor": "ACTIVE"
  },
  "current_session": {
    "session_id": "abc123",
    "capital": 10000,
    "mode": "DEMO",
    "is_active": true
  },
  "performance_metrics": {
    "total_decisions": 150,
    "successful_predictions": 120,
    "average_confidence": 0.85,
    "active_positions": 5,
    "total_trades": 25,
    "current_pnl": 1250.50,
    "system_uptime": "24h 15m 30s"
  }
}
```

#### `GET /api/health`
Get system health status.

**Response:**
```json
{
  "success": true,
  "health": {
    "status": "HEALTHY",
    "components": {
      "database": "OK",
      "ai_models": "OK",
      "market_data": "OK",
      "order_execution": "OK"
    }
  },
  "timestamp": "2025-01-15T10:30:00"
}
```

### Session Management

#### `POST /api/session/start`
Start a new trading session.

**Request:**
```json
{
  "capital": 10000,
  "mode": "DEMO"
}
```

**Response:**
```json
{
  "success": true,
  "session_id": "abc123",
  "capital": 10000,
  "mode": "DEMO",
  "timestamp": "2025-01-15T10:30:00"
}
```

#### `POST /api/session/stop`
Stop current trading session.

**Response:**
```json
{
  "success": true,
  "session_id": "abc123",
  "timestamp": "2025-01-15T10:30:00"
}
```

#### `GET /api/session/status`
Get current session status.

**Response:**
```json
{
  "session_id": "abc123",
  "capital": 10000,
  "mode": "DEMO",
  "is_active": true,
  "start_time": "2025-01-15T10:30:00",
  "current_capital": 10250.50,
  "total_pnl": 250.50,
  "holdings_count": 3
}
```

### AI Trading System

#### `POST /api/ai/start`
Start AI trading system.

**Response:**
```json
{
  "success": true,
  "message": "AI trading started",
  "timestamp": "2025-01-15T10:30:00"
}
```

#### `POST /api/ai/stop`
Stop AI trading system.

**Response:**
```json
{
  "success": true,
  "message": "AI trading stopped",
  "timestamp": "2025-01-15T10:30:00"
}
```

#### `POST /api/ai/analyze`
Run AI analysis on a specific symbol.

**Request:**
```json
{
  "symbol": "AAPL",
  "market_data": {
    "Date": ["2024-01-01", "2024-01-02"],
    "Close": [150.0, 152.0]
  }
}
```

**Response:**
```json
{
  "success": true,
  "symbol": "AAPL",
  "decision": {
    "action": "buy",
    "confidence": 0.85,
    "position_size": 100,
    "reasoning": ["Strong momentum", "Good fundamentals"],
    "model_consensus": {
      "lstm": 0.8,
      "transformer": 0.9,
      "ensemble": 0.85
    },
    "risk_assessment": {
      "risk_level": "medium",
      "stop_loss": 145.0,
      "take_profit": 160.0
    },
    "execution_recommendations": ["Use limit order", "Set stop loss"],
    "timestamp": "2025-01-15T10:30:00"
  }
}
```

### Trading Operations

#### `POST /api/orders/place`
Place a trading order.

**Request:**
```json
{
  "symbol": "AAPL",
  "quantity": 100,
  "order_type": "LIMIT",
  "side": "BUY",
  "price": 150.0
}
```

**Response:**
```json
{
  "success": true,
  "order": {
    "id": 1,
    "order_id": "ORD_20250115_103000_AAPL",
    "symbol": "AAPL",
    "quantity": 100,
    "order_type": "LIMIT",
    "side": "BUY",
    "price": 150.0,
    "status": "FILLED",
    "filled_quantity": 100,
    "filled_price": 149.95,
    "created_at": "2025-01-15T10:30:00",
    "executed_at": "2025-01-15T10:30:05"
  },
  "timestamp": "2025-01-15T10:30:00"
}
```

#### `GET /api/orders`
Get orders with optional filtering.

**Parameters:**
- `status` (optional): Filter by order status
- `symbol` (optional): Filter by symbol

**Response:**
```json
{
  "success": true,
  "orders": [
    {
      "id": 1,
      "order_id": "ORD_20250115_103000_AAPL",
      "symbol": "AAPL",
      "quantity": 100,
      "order_type": "LIMIT",
      "side": "BUY",
      "price": 150.0,
      "status": "FILLED",
      "filled_quantity": 100,
      "filled_price": 149.95,
      "created_at": "2025-01-15T10:30:00"
    }
  ],
  "count": 1,
  "timestamp": "2025-01-15T10:30:00"
}
```

### Portfolio Management

#### `GET /api/portfolio`
Get current portfolio.

**Response:**
```json
{
  "success": true,
  "portfolio": [
    {
      "symbol": "AAPL",
      "quantity": 100,
      "avg_price": 149.95,
      "current_price": 152.0,
      "market_value": 15200.0,
      "unrealized_pnl": 205.0,
      "unrealized_pnl_pct": 1.37,
      "created_at": "2025-01-15T10:30:00"
    }
  ],
  "summary": {
    "total_positions": 1,
    "total_value": 15200.0,
    "total_pnl": 205.0,
    "total_pnl_pct": 1.37
  },
  "timestamp": "2025-01-15T10:30:00"
}
```

#### `GET /api/positions`
Get positions with optional symbol filtering.

**Parameters:**
- `symbol` (optional): Filter by symbol

**Response:**
```json
{
  "success": true,
  "positions": [
    {
      "id": 1,
      "symbol": "AAPL",
      "quantity": 100,
      "avg_price": 149.95,
      "current_price": 152.0,
      "market_value": 15200.0,
      "unrealized_pnl": 205.0,
      "unrealized_pnl_pct": 1.37,
      "created_at": "2025-01-15T10:30:00"
    }
  ],
  "count": 1,
  "timestamp": "2025-01-15T10:30:00"
}
```

### Risk Management

#### `GET /api/risk/metrics`
Get current risk metrics.

**Response:**
```json
{
  "success": true,
  "risk_metrics": {
    "portfolio_risk": 0.15,
    "max_drawdown": 0.05,
    "var_95": 0.08,
    "sharpe_ratio": 1.25,
    "beta": 0.95,
    "correlation_risk": 0.3
  },
  "timestamp": "2025-01-15T10:30:00"
}
```

#### `POST /api/risk/check`
Check if a trade would violate risk limits.

**Parameters:**
- `symbol`: Stock symbol
- `quantity`: Number of shares
- `price`: Price per share

**Response:**
```json
{
  "success": true,
  "risk_check": {
    "allowed": true,
    "risk_level": "medium",
    "position_size_pct": 0.08,
    "warnings": [],
    "recommendations": ["Consider reducing position size"]
  },
  "timestamp": "2025-01-15T10:30:00"
}
```

### Market Data

#### `GET /api/market/data/{symbol}`
Get market data for a symbol.

**Parameters:**
- `symbol`: Stock symbol
- `period` (optional): Time period (default: "1d")

**Response:**
```json
{
  "success": true,
  "symbol": "AAPL",
  "period": "1d",
  "data": [
    {
      "Date": "2024-01-01",
      "Open": 150.0,
      "High": 152.0,
      "Low": 149.0,
      "Close": 151.5,
      "Volume": 1000000
    }
  ],
  "timestamp": "2025-01-15T10:30:00"
}
```

#### `GET /api/market/price/{symbol}`
Get current price for a symbol.

**Response:**
```json
{
  "success": true,
  "symbol": "AAPL",
  "price": 152.0,
  "timestamp": "2025-01-15T10:30:00"
}
```

### Analytics & Reporting

#### `GET /api/analytics/performance`
Get comprehensive performance analytics.

**Response:**
```json
{
  "success": true,
  "analytics": {
    "total_trades": 25,
    "winning_trades": 18,
    "win_rate": 72.0,
    "total_pnl": 1250.50,
    "average_trade_pnl": 50.02,
    "best_trade": {
      "symbol": "AAPL",
      "pnl": 250.0,
      "pnl_pct": 1.67
    },
    "worst_trade": {
      "symbol": "TSLA",
      "pnl": -75.0,
      "pnl_pct": -0.5
    }
  },
  "timestamp": "2025-01-15T10:30:00"
}
```

#### `GET /api/logs/ai`
Get AI system logs.

**Parameters:**
- `limit` (optional): Number of logs to return (default: 50)

**Response:**
```json
{
  "success": true,
  "logs": {
    "activity": [
      {
        "timestamp": "2025-01-15T10:30:00",
        "component": "AI Engine",
        "action": "Started",
        "details": {"session_id": "abc123"}
      }
    ],
    "trading_decisions": [
      {
        "timestamp": "2025-01-15T10:30:00",
        "symbol": "AAPL",
        "action": "buy",
        "confidence": 0.85,
        "reasoning": ["Strong momentum"]
      }
    ],
    "performance_metrics": {
      "total_decisions": 150,
      "average_confidence": 0.85,
      "success_rate": 0.8
    }
  },
  "timestamp": "2025-01-15T10:30:00"
}
```

## 🔧 Integration Examples

### Python Client Example

```python
import requests
import json

# API Base URL
BASE_URL = "http://localhost:8000"

# Start a trading session
def start_session(capital=10000, mode="DEMO"):
    response = requests.post(
        f"{BASE_URL}/api/session/start",
        json={"capital": capital, "mode": mode}
    )
    return response.json()

# Place an order
def place_order(symbol, quantity, order_type, side, price=None):
    data = {
        "symbol": symbol,
        "quantity": quantity,
        "order_type": order_type,
        "side": side
    }
    if price:
        data["price"] = price
    
    response = requests.post(
        f"{BASE_URL}/api/orders/place",
        json=data
    )
    return response.json()

# Get portfolio
def get_portfolio():
    response = requests.get(f"{BASE_URL}/api/portfolio")
    return response.json()

# Run AI analysis
def run_ai_analysis(symbol):
    response = requests.post(
        f"{BASE_URL}/api/ai/analyze",
        json={"symbol": symbol}
    )
    return response.json()

# Example usage
if __name__ == "__main__":
    # Start session
    session = start_session(10000, "DEMO")
    print(f"Session started: {session}")
    
    # Run AI analysis
    analysis = run_ai_analysis("AAPL")
    print(f"AI Analysis: {analysis}")
    
    # Place order based on AI decision
    if analysis["success"] and analysis["decision"]["action"] == "buy":
        order = place_order("AAPL", 100, "LIMIT", "BUY", 150.0)
        print(f"Order placed: {order}")
    
    # Get portfolio
    portfolio = get_portfolio()
    print(f"Portfolio: {portfolio}")
```

### JavaScript/Node.js Client Example

```javascript
const axios = require('axios');

const BASE_URL = 'http://localhost:8000';

// Start a trading session
async function startSession(capital = 10000, mode = 'DEMO') {
    try {
        const response = await axios.post(`${BASE_URL}/api/session/start`, {
            capital,
            mode
        });
        return response.data;
    } catch (error) {
        console.error('Error starting session:', error.response.data);
        throw error;
    }
}

// Place an order
async function placeOrder(symbol, quantity, orderType, side, price = null) {
    try {
        const data = {
            symbol,
            quantity,
            order_type: orderType,
            side
        };
        if (price) data.price = price;
        
        const response = await axios.post(`${BASE_URL}/api/orders/place`, data);
        return response.data;
    } catch (error) {
        console.error('Error placing order:', error.response.data);
        throw error;
    }
}

// Get portfolio
async function getPortfolio() {
    try {
        const response = await axios.get(`${BASE_URL}/api/portfolio`);
        return response.data;
    } catch (error) {
        console.error('Error getting portfolio:', error.response.data);
        throw error;
    }
}

// Example usage
async function main() {
    try {
        // Start session
        const session = await startSession(10000, 'DEMO');
        console.log('Session started:', session);
        
        // Place order
        const order = await placeOrder('AAPL', 100, 'LIMIT', 'BUY', 150.0);
        console.log('Order placed:', order);
        
        // Get portfolio
        const portfolio = await getPortfolio();
        console.log('Portfolio:', portfolio);
    } catch (error) {
        console.error('Error:', error.message);
    }
}

main();
```

### cURL Examples

```bash
# Start a trading session
curl -X POST http://localhost:8000/api/session/start \
  -H "Content-Type: application/json" \
  -d '{"capital": 10000, "mode": "DEMO"}'

# Place an order
curl -X POST http://localhost:8000/api/orders/place \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "quantity": 100,
    "order_type": "LIMIT",
    "side": "BUY",
    "price": 150.0
  }'

# Get portfolio
curl http://localhost:8000/api/portfolio

# Run AI analysis
curl -X POST http://localhost:8000/api/ai/analyze \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL"}'

# Get system status
curl http://localhost:8000/api/status
```

## 🚀 Advanced Features

### 1. Real-time WebSocket Support (Future)
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Real-time update:', data);
};
```

### 2. Batch Operations (Future)
```python
# Place multiple orders at once
orders = [
    {"symbol": "AAPL", "quantity": 100, "order_type": "LIMIT", "side": "BUY", "price": 150.0},
    {"symbol": "GOOGL", "quantity": 50, "order_type": "LIMIT", "side": "BUY", "price": 2800.0}
]

response = requests.post(f"{BASE_URL}/api/orders/batch", json={"orders": orders})
```

### 3. Custom AI Models (Future)
```python
# Upload custom AI model
with open('my_model.pkl', 'rb') as f:
    files = {'model': f}
    response = requests.post(f"{BASE_URL}/api/ai/models/upload", files=files)
```

## 🔒 Security & Authentication

### API Key Authentication (Future)
```python
headers = {
    'Authorization': 'Bearer your-api-key-here',
    'Content-Type': 'application/json'
}

response = requests.get(f"{BASE_URL}/api/portfolio", headers=headers)
```

### Rate Limiting
- **Default**: 100 requests per minute per IP
- **AI Analysis**: 10 requests per minute per session
- **Order Placement**: 5 requests per minute per session

## 📊 Monitoring & Logging

### Health Checks
```bash
# Check API health
curl http://localhost:8000/api/health

# Check system status
curl http://localhost:8000/api/status
```

### Logging
- **API Logs**: `/logs/api.log`
- **AI Logs**: `/logs/ai.log`
- **Trading Logs**: `/logs/trading.log`
- **Error Logs**: `/logs/error.log`

## 🚀 Deployment

### Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "unified_trading_api.py"]
```

### Production Configuration
```bash
# Environment variables
export API_HOST=0.0.0.0
export API_PORT=8000
export LOG_LEVEL=INFO
export DATABASE_URL=sqlite:///data/trading.db
export REDIS_URL=redis://localhost:6379
```

## 📈 Performance

### Benchmarks
- **API Response Time**: < 100ms average
- **AI Analysis**: < 2 seconds
- **Order Execution**: < 500ms
- **Concurrent Users**: 100+ supported

### Optimization Tips
1. Use connection pooling for database connections
2. Enable Redis caching for market data
3. Implement request batching for multiple operations
4. Use WebSocket for real-time updates

## 🆘 Troubleshooting

### Common Issues

#### 1. "No active session" Error
```bash
# Start a session first
curl -X POST http://localhost:8000/api/session/start \
  -H "Content-Type: application/json" \
  -d '{"capital": 10000, "mode": "DEMO"}'
```

#### 2. "AI Engine not initialized" Error
```bash
# Check system status
curl http://localhost:8000/api/status

# Restart AI engine
curl -X POST http://localhost:8000/api/ai/stop
curl -X POST http://localhost:8000/api/ai/start
```

#### 3. Database Connection Issues
```bash
# Check database files exist
ls -la data/
# Should see: trading_demo.db, trading_live.db
```

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python unified_trading_api.py
```

## 📞 Support

For issues and questions:
1. Check the API documentation at `/docs`
2. Review the logs in the `/logs` directory
3. Check system status at `/api/status`
4. Verify all dependencies are installed

---

**Version**: 1.0.0  
**Last Updated**: 2025-01-15  
**Compatible with**: TradingBOT v2.0+
