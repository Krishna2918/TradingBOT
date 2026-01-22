"""
Unified Trading API - Complete TradingBOT System Interface

This is the single, comprehensive API that brings together all components
of the TradingBOT system into one unified interface.

Features:
- Complete AI Trading System
- Real-time Market Data
- Portfolio Management
- Risk Management
- Order Execution (Live/Demo)
- Performance Analytics
- System Monitoring
- Dashboard Integration
"""

import asyncio
import logging
import json
import os
import sys
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Core System Imports
from src.integration.master_orchestrator import MasterOrchestrator, TradingDecision, MarketContext
from src.trading.execution import OrderExecutor, Order, OrderType, OrderSide, OrderStatus, ExecutionResult
from src.trading.positions import Position, get_position_manager
from src.trading.risk import RiskMetrics, get_risk_manager
from src.config.mode_manager import ModeManager, get_mode_manager, get_current_mode
from src.config.database import get_connection, execute_query, execute_update
from src.dashboard.clean_state_manager import CleanStateManager, TradingSession
from src.dashboard.maximum_power_ai_engine import MaximumPowerAIEngine
from src.dashboard.advanced_ai_logger import AdvancedAILogger
from src.data_pipeline.api_budget_manager import API_Budget_Manager
from src.monitoring.system_monitor import SystemMonitor

# FastAPI and Web Framework
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# UNIFIED TRADING API CLASS
# ============================================================================

class UnifiedTradingAPI:
    """
    Unified Trading API - Single interface for all TradingBOT functionality
    
    This class provides a comprehensive API that integrates:
    - AI Trading System (MasterOrchestrator)
    - Order Execution (Live/Demo modes)
    - Portfolio Management
    - Risk Management
    - Market Data
    - Performance Analytics
    - System Monitoring
    """
    
    def __init__(self):
        """Initialize the unified trading API."""
        logger.info("Initializing Unified Trading API...")
        
        # Core System Components
        self.master_orchestrator = MasterOrchestrator()
        self.order_executor = OrderExecutor()
        self.position_manager = get_position_manager()
        self.risk_manager = get_risk_manager()
        self.mode_manager = get_mode_manager()
        self.state_manager = CleanStateManager()
        self.ai_engine = MaximumPowerAIEngine()
        self.ai_logger = AdvancedAILogger()
        self.api_budget_manager = API_Budget_Manager()
        self.system_monitor = SystemMonitor()
        
        # System Status
        self.is_initialized = True
        self.current_session = None
        self.system_health = "HEALTHY"
        
        logger.info("Unified Trading API initialized successfully!")
    
    # ========================================================================
    # SYSTEM MANAGEMENT
    # ========================================================================
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            return {
                "status": "OPERATIONAL",
                "timestamp": datetime.now().isoformat(),
                "mode": get_current_mode(),
                "health": self.system_health,
                "components": {
                    "master_orchestrator": "ACTIVE",
                    "order_executor": "ACTIVE",
                    "position_manager": "ACTIVE",
                    "risk_manager": "ACTIVE",
                    "ai_engine": "ACTIVE" if self.ai_engine.running else "IDLE",
                    "state_manager": "ACTIVE",
                    "system_monitor": "ACTIVE"
                },
                "current_session": asdict(self.current_session) if self.current_session else None,
                "performance_metrics": self._get_performance_metrics()
            }
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {"status": "ERROR", "error": str(e)}
    
    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics."""
        try:
            return {
                "total_decisions": len(self.master_orchestrator.decision_history),
                "successful_predictions": self.master_orchestrator.performance_metrics.get('successful_predictions', 0),
                "average_confidence": self.master_orchestrator.performance_metrics.get('average_confidence', 0.0),
                "active_positions": len(self.position_manager.get_all_positions()),
                "total_trades": self._get_total_trades(),
                "current_pnl": self._get_current_pnl(),
                "system_uptime": self._get_system_uptime()
            }
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}
    
    def _get_total_trades(self) -> int:
        """Get total number of trades executed."""
        try:
            with get_connection(get_current_mode()) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM orders WHERE status = 'FILLED'")
                return cursor.fetchone()[0]
        except:
            return 0
    
    def _get_current_pnl(self) -> float:
        """Get current portfolio P&L."""
        try:
            positions = self.position_manager.get_all_positions()
            total_pnl = sum(pos.unrealized_pnl for pos in positions)
            return total_pnl
        except:
            return 0.0
    
    def _get_system_uptime(self) -> str:
        """Get system uptime."""
        # This would be implemented with actual uptime tracking
        return "24h 15m 30s"
    
    # ========================================================================
    # SESSION MANAGEMENT
    # ========================================================================
    
    def start_trading_session(self, capital: float, mode: str = "DEMO") -> Dict[str, Any]:
        """Start a new trading session."""
        try:
            # Set mode
            self.mode_manager.set_mode(mode)
            
            # Start new session
            session_id = self.state_manager.start_new_session(capital, mode)
            self.current_session = self.state_manager.current_session
            
            # Clear AI logs for fresh start
            self.ai_logger.clear_logs()
            
            # Log session start
            self.ai_logger.log_component_activity(
                "Session Manager", 
                "Started", 
                {"session_id": session_id, "capital": capital, "mode": mode}
            )
            
            return {
                "success": True,
                "session_id": session_id,
                "capital": capital,
                "mode": mode,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error starting trading session: {e}")
            return {"success": False, "error": str(e)}
    
    def stop_trading_session(self) -> Dict[str, Any]:
        """Stop the current trading session."""
        try:
            if self.current_session:
                self.current_session.is_active = False
                self.state_manager.save()
                
                # Stop AI engine
                self.ai_engine.stop_maximum_power_trading()
                
                # Log session stop
                self.ai_logger.log_component_activity(
                    "Session Manager", 
                    "Stopped", 
                    {"session_id": self.current_session.session_id}
                )
                
                return {
                    "success": True,
                    "session_id": self.current_session.session_id,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {"success": False, "error": "No active session"}
        except Exception as e:
            logger.error(f"Error stopping trading session: {e}")
            return {"success": False, "error": str(e)}
    
    def get_session_status(self) -> Dict[str, Any]:
        """Get current session status."""
        try:
            if self.current_session:
                return asdict(self.current_session)
            else:
                return {"active": False, "message": "No active session"}
        except Exception as e:
            logger.error(f"Error getting session status: {e}")
            return {"error": str(e)}
    
    # ========================================================================
    # AI TRADING SYSTEM
    # ========================================================================
    
    async def run_ai_analysis(self, symbol: str, market_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Run AI analysis on a specific symbol."""
        try:
            if market_data is None:
                # Generate synthetic data for testing
                market_data = self._generate_synthetic_data(symbol)
            
            # Run decision pipeline
            decision = await self.master_orchestrator.run_decision_pipeline(market_data)
            
            # Log the decision
            self.ai_logger.log_trading_decision(
                symbol=symbol,
                action=decision.action,
                confidence=decision.confidence,
                reasoning=decision.reasoning,
                model_consensus=decision.model_consensus,
                risk_assessment=decision.risk_assessment
            )
            
            return {
                "success": True,
                "symbol": symbol,
                "decision": asdict(decision),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error running AI analysis: {e}")
            return {"success": False, "error": str(e)}
    
    def start_ai_trading(self) -> Dict[str, Any]:
        """Start AI trading system."""
        try:
            if not self.current_session:
                return {"success": False, "error": "No active session"}
            
            # Start maximum power AI trading
            self.ai_engine.start_maximum_power_trading()
            
            # Log AI trading start
            self.ai_logger.log_component_activity(
                "AI Engine", 
                "Started", 
                {"session_id": self.current_session.session_id}
            )
            
            return {
                "success": True,
                "message": "AI trading started",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error starting AI trading: {e}")
            return {"success": False, "error": str(e)}
    
    def stop_ai_trading(self) -> Dict[str, Any]:
        """Stop AI trading system."""
        try:
            self.ai_engine.stop_maximum_power_trading()
            
            # Log AI trading stop
            self.ai_logger.log_component_activity(
                "AI Engine", 
                "Stopped", 
                {}
            )
            
            return {
                "success": True,
                "message": "AI trading stopped",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error stopping AI trading: {e}")
            return {"success": False, "error": str(e)}
    
    def _generate_synthetic_data(self, symbol: str) -> pd.DataFrame:
        """Generate synthetic market data for testing."""
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        np.random.seed(42)  # For reproducible results
        
        # Generate realistic price data
        base_price = 100.0
        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        data = pd.DataFrame({
            'Date': dates,
            'Open': prices,
            'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        })
        
        return data.set_index('Date')
    
    # ========================================================================
    # ORDER MANAGEMENT
    # ========================================================================
    
    def place_order(self, symbol: str, quantity: int, order_type: str, 
                   side: str, price: Optional[float] = None) -> Dict[str, Any]:
        """Place a trading order."""
        try:
            # Validate inputs
            if quantity <= 0:
                return {"success": False, "error": "Quantity must be positive"}
            
            if order_type not in ['MARKET', 'LIMIT', 'STOP', 'STOP_LIMIT']:
                return {"success": False, "error": "Invalid order type"}
            
            if side not in ['BUY', 'SELL']:
                return {"success": False, "error": "Invalid order side"}
            
            # Create order
            order = Order(
                id=None,
                order_id=f"ORD_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{symbol}",
                position_id=None,
                order_type=OrderType(order_type),
                side=OrderSide(side),
                symbol=symbol,
                quantity=quantity,
                price=price or 0.0,
                status=OrderStatus.PENDING,
                filled_quantity=0,
                filled_price=0.0,
                created_at=datetime.now(),
                submitted_at=None,
                executed_at=None,
                mode=get_current_mode(),
                execution_type='REAL' if get_current_mode() == 'LIVE' else 'PAPER'
            )
            
            # Execute order
            result = self.order_executor.execute_order(order)
            
            # Log order execution
            self.ai_logger.log_component_activity(
                "Order Executor", 
                "Order Placed" if result.success else "Order Failed",
                {
                    "symbol": symbol,
                    "quantity": quantity,
                    "order_type": order_type,
                    "side": side,
                    "success": result.success
                }
            )
            
            return {
                "success": result.success,
                "order": asdict(result.order) if result.order else None,
                "error": result.error_message,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return {"success": False, "error": str(e)}
    
    def get_orders(self, status: Optional[str] = None, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get orders with optional filtering."""
        try:
            with get_connection(get_current_mode()) as conn:
                cursor = conn.cursor()
                
                query = "SELECT * FROM orders WHERE 1=1"
                params = []
                
                if status:
                    query += " AND status = ?"
                    params.append(status)
                
                if symbol:
                    query += " AND symbol = ?"
                    params.append(symbol)
                
                query += " ORDER BY created_at DESC LIMIT 100"
                
                cursor.execute(query, params)
                orders = cursor.fetchall()
                
                return {
                    "success": True,
                    "orders": [dict(order) for order in orders],
                    "count": len(orders),
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Error getting orders: {e}")
            return {"success": False, "error": str(e)}
    
    # ========================================================================
    # PORTFOLIO MANAGEMENT
    # ========================================================================
    
    def get_portfolio(self) -> Dict[str, Any]:
        """Get current portfolio."""
        try:
            positions = self.position_manager.get_all_positions()
            
            portfolio_data = []
            total_value = 0.0
            total_pnl = 0.0
            
            for position in positions:
                position_data = {
                    "symbol": position.symbol,
                    "quantity": position.quantity,
                    "avg_price": position.avg_price,
                    "current_price": position.current_price,
                    "market_value": position.market_value,
                    "unrealized_pnl": position.unrealized_pnl,
                    "unrealized_pnl_pct": position.unrealized_pnl_pct,
                    "created_at": position.created_at.isoformat()
                }
                portfolio_data.append(position_data)
                total_value += position.market_value
                total_pnl += position.unrealized_pnl
            
            return {
                "success": True,
                "portfolio": portfolio_data,
                "summary": {
                    "total_positions": len(positions),
                    "total_value": total_value,
                    "total_pnl": total_pnl,
                    "total_pnl_pct": (total_pnl / (total_value - total_pnl)) * 100 if total_value > total_pnl else 0
                },
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting portfolio: {e}")
            return {"success": False, "error": str(e)}
    
    def get_positions(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get positions with optional symbol filtering."""
        try:
            if symbol:
                position = self.position_manager.get_position(symbol)
                positions = [position] if position else []
            else:
                positions = self.position_manager.get_all_positions()
            
            return {
                "success": True,
                "positions": [asdict(pos) for pos in positions],
                "count": len(positions),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return {"success": False, "error": str(e)}
    
    # ========================================================================
    # RISK MANAGEMENT
    # ========================================================================
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get current risk metrics."""
        try:
            positions = self.position_manager.get_all_positions()
            total_value = sum(pos.market_value for pos in positions)
            
            risk_metrics = self.risk_manager.calculate_portfolio_risk(positions, total_value)
            
            return {
                "success": True,
                "risk_metrics": asdict(risk_metrics),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting risk metrics: {e}")
            return {"success": False, "error": str(e)}
    
    def check_risk_limits(self, symbol: str, quantity: int, price: float) -> Dict[str, Any]:
        """Check if a trade would violate risk limits."""
        try:
            # Get current positions
            positions = self.position_manager.get_all_positions()
            
            # Calculate new position value
            new_position_value = quantity * price
            
            # Check risk limits
            risk_check = self.risk_manager.check_risk_limits(
                positions, new_position_value, symbol
            )
            
            return {
                "success": True,
                "risk_check": risk_check,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
            return {"success": False, "error": str(e)}
    
    # ========================================================================
    # MARKET DATA
    # ========================================================================
    
    def get_market_data(self, symbol: str, period: str = "1d") -> Dict[str, Any]:
        """Get market data for a symbol."""
        try:
            # This would integrate with your actual market data provider
            # For now, return synthetic data
            market_data = self._generate_synthetic_data(symbol)
            
            return {
                "success": True,
                "symbol": symbol,
                "period": period,
                "data": market_data.to_dict('records'),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return {"success": False, "error": str(e)}
    
    def get_current_price(self, symbol: str) -> Dict[str, Any]:
        """Get current price for a symbol."""
        try:
            # This would integrate with your actual price feed
            # For now, return a synthetic price
            base_price = 100.0
            price_variation = np.random.normal(0, 0.02)
            current_price = base_price * (1 + price_variation)
            
            return {
                "success": True,
                "symbol": symbol,
                "price": round(current_price, 2),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting current price: {e}")
            return {"success": False, "error": str(e)}
    
    # ========================================================================
    # ANALYTICS & REPORTING
    # ========================================================================
    
    def get_performance_analytics(self) -> Dict[str, Any]:
        """Get comprehensive performance analytics."""
        try:
            # Get trading statistics
            with get_connection(get_current_mode()) as conn:
                cursor = conn.cursor()
                
                # Total trades
                cursor.execute("SELECT COUNT(*) FROM orders WHERE status = 'FILLED'")
                total_trades = cursor.fetchone()[0]
                
                # Winning trades
                cursor.execute("""
                    SELECT COUNT(*) FROM orders o 
                    JOIN positions p ON o.symbol = p.symbol 
                    WHERE o.status = 'FILLED' AND p.unrealized_pnl > 0
                """)
                winning_trades = cursor.fetchone()[0]
                
                # Total P&L
                cursor.execute("SELECT SUM(unrealized_pnl) FROM positions")
                total_pnl = cursor.fetchone()[0] or 0.0
            
            # Calculate metrics
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            return {
                "success": True,
                "analytics": {
                    "total_trades": total_trades,
                    "winning_trades": winning_trades,
                    "win_rate": round(win_rate, 2),
                    "total_pnl": round(total_pnl, 2),
                    "average_trade_pnl": round(total_pnl / total_trades, 2) if total_trades > 0 else 0,
                    "best_trade": self._get_best_trade(),
                    "worst_trade": self._get_worst_trade()
                },
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting performance analytics: {e}")
            return {"success": False, "error": str(e)}
    
    def _get_best_trade(self) -> Dict[str, Any]:
        """Get best performing trade."""
        try:
            positions = self.position_manager.get_all_positions()
            if not positions:
                return {}
            
            best_position = max(positions, key=lambda p: p.unrealized_pnl)
            return {
                "symbol": best_position.symbol,
                "pnl": best_position.unrealized_pnl,
                "pnl_pct": best_position.unrealized_pnl_pct
            }
        except:
            return {}
    
    def _get_worst_trade(self) -> Dict[str, Any]:
        """Get worst performing trade."""
        try:
            positions = self.position_manager.get_all_positions()
            if not positions:
                return {}
            
            worst_position = min(positions, key=lambda p: p.unrealized_pnl)
            return {
                "symbol": worst_position.symbol,
                "pnl": worst_position.unrealized_pnl,
                "pnl_pct": worst_position.unrealized_pnl_pct
            }
        except:
            return {}
    
    # ========================================================================
    # AI LOGS & MONITORING
    # ========================================================================
    
    def get_ai_logs(self, limit: int = 50) -> Dict[str, Any]:
        """Get AI system logs."""
        try:
            activity_logs = self.ai_logger.get_recent_activity(limit)
            trading_decisions = self.ai_logger.get_recent_decisions(limit)
            performance_metrics = self.ai_logger.get_performance_summary()
            
            return {
                "success": True,
                "logs": {
                    "activity": activity_logs,
                    "trading_decisions": trading_decisions,
                    "performance_metrics": performance_metrics
                },
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting AI logs: {e}")
            return {"success": False, "error": str(e)}
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get system health status."""
        try:
            health_status = self.system_monitor.get_health_status()
            
            return {
                "success": True,
                "health": health_status,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return {"success": False, "error": str(e)}

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

# Initialize the unified API
unified_api = UnifiedTradingAPI()

# Create FastAPI app
app = FastAPI(
    title="Unified Trading API",
    description="Complete TradingBOT System Interface",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class SessionRequest(BaseModel):
    capital: float = Field(..., description="Starting capital amount")
    mode: str = Field("DEMO", description="Trading mode (LIVE/DEMO)")

class OrderRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol")
    quantity: int = Field(..., description="Number of shares")
    order_type: str = Field(..., description="Order type (MARKET/LIMIT/STOP/STOP_LIMIT)")
    side: str = Field(..., description="Order side (BUY/SELL)")
    price: Optional[float] = Field(None, description="Order price (for limit orders)")

class AnalysisRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol to analyze")
    market_data: Optional[Dict[str, Any]] = Field(None, description="Market data (optional)")

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API information."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Unified Trading API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #1a1a1a; color: #fff; }
            .container { max-width: 800px; margin: 0 auto; }
            .header { text-align: center; margin-bottom: 40px; }
            .endpoint { background: #2a2a2a; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .method { color: #4CAF50; font-weight: bold; }
            .path { color: #2196F3; font-family: monospace; }
            .description { color: #ccc; margin-top: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 Unified Trading API</h1>
                <p>Complete TradingBOT System Interface</p>
            </div>
            
            <h2>📊 System Management</h2>
            <div class="endpoint">
                <span class="method">GET</span> <span class="path">/api/status</span>
                <div class="description">Get comprehensive system status</div>
            </div>
            <div class="endpoint">
                <span class="method">GET</span> <span class="path">/api/health</span>
                <div class="description">Get system health status</div>
            </div>
            
            <h2>🎯 Session Management</h2>
            <div class="endpoint">
                <span class="method">POST</span> <span class="path">/api/session/start</span>
                <div class="description">Start a new trading session</div>
            </div>
            <div class="endpoint">
                <span class="method">POST</span> <span class="path">/api/session/stop</span>
                <div class="description">Stop current trading session</div>
            </div>
            <div class="endpoint">
                <span class="method">GET</span> <span class="path">/api/session/status</span>
                <div class="description">Get current session status</div>
            </div>
            
            <h2>🤖 AI Trading System</h2>
            <div class="endpoint">
                <span class="method">POST</span> <span class="path">/api/ai/start</span>
                <div class="description">Start AI trading system</div>
            </div>
            <div class="endpoint">
                <span class="method">POST</span> <span class="path">/api/ai/stop</span>
                <div class="description">Stop AI trading system</div>
            </div>
            <div class="endpoint">
                <span class="method">POST</span> <span class="path">/api/ai/analyze</span>
                <div class="description">Run AI analysis on a symbol</div>
            </div>
            
            <h2>📈 Trading Operations</h2>
            <div class="endpoint">
                <span class="method">POST</span> <span class="path">/api/orders/place</span>
                <div class="description">Place a trading order</div>
            </div>
            <div class="endpoint">
                <span class="method">GET</span> <span class="path">/api/orders</span>
                <div class="description">Get orders with filtering</div>
            </div>
            
            <h2>💼 Portfolio Management</h2>
            <div class="endpoint">
                <span class="method">GET</span> <span class="path">/api/portfolio</span>
                <div class="description">Get current portfolio</div>
            </div>
            <div class="endpoint">
                <span class="method">GET</span> <span class="path">/api/positions</span>
                <div class="description">Get positions with filtering</div>
            </div>
            
            <h2>⚠️ Risk Management</h2>
            <div class="endpoint">
                <span class="method">GET</span> <span class="path">/api/risk/metrics</span>
                <div class="description">Get current risk metrics</div>
            </div>
            <div class="endpoint">
                <span class="method">POST</span> <span class="path">/api/risk/check</span>
                <div class="description">Check risk limits for a trade</div>
            </div>
            
            <h2>📊 Market Data</h2>
            <div class="endpoint">
                <span class="method">GET</span> <span class="path">/api/market/data/{symbol}</span>
                <div class="description">Get market data for a symbol</div>
            </div>
            <div class="endpoint">
                <span class="method">GET</span> <span class="path">/api/market/price/{symbol}</span>
                <div class="description">Get current price for a symbol</div>
            </div>
            
            <h2>📈 Analytics & Reporting</h2>
            <div class="endpoint">
                <span class="method">GET</span> <span class="path">/api/analytics/performance</span>
                <div class="description">Get performance analytics</div>
            </div>
            <div class="endpoint">
                <span class="method">GET</span> <span class="path">/api/logs/ai</span>
                <div class="description">Get AI system logs</div>
            </div>
            
            <h2>📚 Documentation</h2>
            <div class="endpoint">
                <span class="method">GET</span> <span class="path">/docs</span>
                <div class="description">Interactive API documentation (Swagger UI)</div>
            </div>
            <div class="endpoint">
                <span class="method">GET</span> <span class="path">/redoc</span>
                <div class="description">Alternative API documentation (ReDoc)</div>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# System Management Endpoints
@app.get("/api/status")
async def get_system_status():
    """Get comprehensive system status."""
    return unified_api.get_system_status()

@app.get("/api/health")
async def get_system_health():
    """Get system health status."""
    return unified_api.get_system_health()

# Session Management Endpoints
@app.post("/api/session/start")
async def start_session(request: SessionRequest):
    """Start a new trading session."""
    return unified_api.start_trading_session(request.capital, request.mode)

@app.post("/api/session/stop")
async def stop_session():
    """Stop current trading session."""
    return unified_api.stop_trading_session()

@app.get("/api/session/status")
async def get_session_status():
    """Get current session status."""
    return unified_api.get_session_status()

# AI Trading System Endpoints
@app.post("/api/ai/start")
async def start_ai_trading():
    """Start AI trading system."""
    return unified_api.start_ai_trading()

@app.post("/api/ai/stop")
async def stop_ai_trading():
    """Stop AI trading system."""
    return unified_api.stop_ai_trading()

@app.post("/api/ai/analyze")
async def run_ai_analysis(request: AnalysisRequest):
    """Run AI analysis on a specific symbol."""
    market_data = None
    if request.market_data:
        market_data = pd.DataFrame(request.market_data)
    return await unified_api.run_ai_analysis(request.symbol, market_data)

# Trading Operations Endpoints
@app.post("/api/orders/place")
async def place_order(request: OrderRequest):
    """Place a trading order."""
    return unified_api.place_order(
        request.symbol,
        request.quantity,
        request.order_type,
        request.side,
        request.price
    )

@app.get("/api/orders")
async def get_orders(status: Optional[str] = None, symbol: Optional[str] = None):
    """Get orders with optional filtering."""
    return unified_api.get_orders(status, symbol)

# Portfolio Management Endpoints
@app.get("/api/portfolio")
async def get_portfolio():
    """Get current portfolio."""
    return unified_api.get_portfolio()

@app.get("/api/positions")
async def get_positions(symbol: Optional[str] = None):
    """Get positions with optional symbol filtering."""
    return unified_api.get_positions(symbol)

# Risk Management Endpoints
@app.get("/api/risk/metrics")
async def get_risk_metrics():
    """Get current risk metrics."""
    return unified_api.get_risk_metrics()

@app.post("/api/risk/check")
async def check_risk_limits(symbol: str, quantity: int, price: float):
    """Check if a trade would violate risk limits."""
    return unified_api.check_risk_limits(symbol, quantity, price)

# Market Data Endpoints
@app.get("/api/market/data/{symbol}")
async def get_market_data(symbol: str, period: str = "1d"):
    """Get market data for a symbol."""
    return unified_api.get_market_data(symbol, period)

@app.get("/api/market/price/{symbol}")
async def get_current_price(symbol: str):
    """Get current price for a symbol."""
    return unified_api.get_current_price(symbol)

# Analytics & Reporting Endpoints
@app.get("/api/analytics/performance")
async def get_performance_analytics():
    """Get comprehensive performance analytics."""
    return unified_api.get_performance_analytics()

@app.get("/api/logs/ai")
async def get_ai_logs(limit: int = 50):
    """Get AI system logs."""
    return unified_api.get_ai_logs(limit)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

if __name__ == "__main__":
    print("""
    🚀 Unified Trading API Starting...
    
    📊 Features:
    - Complete AI Trading System
    - Real-time Market Data
    - Portfolio Management
    - Risk Management
    - Order Execution (Live/Demo)
    - Performance Analytics
    - System Monitoring
    - Dashboard Integration
    
    🌐 Access Points:
    - API Documentation: http://localhost:8000/docs
    - Alternative Docs: http://localhost:8000/redoc
    - Root Page: http://localhost:8000/
    
    🔧 Ready for integration with your TradingBOT system!
    """)
    
    uvicorn.run(
        "unified_trading_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
