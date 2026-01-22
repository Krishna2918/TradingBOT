"""
Order Execution Engine - Dual Mode Order Management

This module manages order execution for both LIVE and DEMO modes,
providing paper trading for DEMO and real trading for LIVE mode.
"""

import logging
import uuid
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from .positions import Position, get_position_manager
from .risk import RiskMetrics, get_risk_manager
from .atr_brackets import get_atr_bracket_manager, BracketParameters
from src.config.mode_manager import get_mode_manager, get_current_mode, is_real_trading, is_paper_trading
from src.config.database import get_connection, execute_query, execute_update

logger = logging.getLogger(__name__)

class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"

class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "BUY"
    SELL = "SELL"

@dataclass
class Order:
    """Represents a trading order."""
    id: Optional[int]
    order_id: str
    position_id: Optional[int]
    order_type: OrderType
    side: OrderSide
    symbol: str
    quantity: int
    price: float
    status: OrderStatus
    filled_quantity: int
    filled_price: float
    created_at: datetime
    submitted_at: Optional[datetime]
    executed_at: Optional[datetime]
    mode: str
    execution_type: str  # 'REAL' or 'PAPER'
    # Phase 6: ATR bracket parameters
    atr: Optional[float] = None
    atr_multiplier: Optional[float] = None
    r_multiple: Optional[float] = None
    bracket_type: Optional[str] = None
    volatility_percent: Optional[float] = None
    
    def __post_init__(self):
        """Validate order data."""
        if self.quantity <= 0:
            raise ValueError("Quantity must be positive")
        if self.price <= 0:
            raise ValueError("Price must be positive")
        if self.mode not in ['LIVE', 'DEMO']:
            raise ValueError("Invalid mode")
        if self.execution_type not in ['REAL', 'PAPER']:
            raise ValueError("Invalid execution type")

@dataclass
class ExecutionResult:
    """Result of order execution."""
    success: bool
    order: Optional[Order]
    error_message: Optional[str]
    execution_time: datetime
    mode: str

class ExecutionEngine:
    """Execution engine (alias for OrderExecutor)."""
    pass

class OrderExecutor(ExecutionEngine):
    """Order executor class for compatibility."""
    """Manages order execution for dual mode operation."""
    
    def __init__(self, mode: Optional[str] = None):
        """Initialize Execution Engine."""
        self.mode_manager = get_mode_manager()
        self.position_manager = get_position_manager()
        self.risk_manager = get_risk_manager()
        self.mode = mode  # Store mode parameter for compatibility
        
        logger.info("Execution Engine initialized")
    
    def execute_buy_order(self, symbol: str, quantity: int, price: float,
                         order_type: OrderType = OrderType.MARKET,
                         position_id: Optional[int] = None,
                         mode: Optional[str] = None) -> ExecutionResult:
        """Execute a buy order."""
        if mode is None:
            mode = get_current_mode()
        
        try:
            # Validate order parameters
            if not self._validate_buy_order(symbol, quantity, price, mode):
                return ExecutionResult(
                    success=False,
                    order=None,
                    error_message="Order validation failed",
                    execution_time=datetime.now(),
                    mode=mode
                )
            
            # Phase 6: Calculate ATR brackets for the order
            bracket_params = None
            try:
                # Get ATR value (simplified - in real implementation, get from market data)
                atr = self._get_atr_value(symbol, mode)
                
                # Calculate ATR brackets
                bracket_manager = get_atr_bracket_manager()
                bracket_params = bracket_manager.calculate_atr_brackets(
                    symbol=symbol,
                    entry_price=price,
                    atr=atr,
                    mode=mode
                )
                
                logger.debug(f"ATR brackets calculated for {symbol}: "
                           f"SL={bracket_params.stop_loss:.2f}, TP={bracket_params.take_profit:.2f}")
                
            except Exception as e:
                logger.warning(f"Failed to calculate ATR brackets for {symbol}: {e}")
                # Continue without brackets - will use default risk management
            
            # Create order with bracket parameters
            order = self._create_order(
                symbol=symbol,
                quantity=quantity,
                price=price,
                side=OrderSide.BUY,
                order_type=order_type,
                position_id=position_id,
                mode=mode,
                bracket_params=bracket_params
            )
            
            # Execute order based on mode
            if is_real_trading():
                execution_result = self._execute_real_order(order)
            else:
                execution_result = self._execute_paper_order(order)
            
            # Log the trade
            if execution_result.success:
                self._log_trade(order, mode)
                logger.info(f"Buy order executed: {symbol} {quantity} shares @ ${price:.2f} in {mode} mode")
            
            return execution_result
            
        except Exception as e:
            logger.error(f"Buy order execution error: {e}")
            return ExecutionResult(
                success=False,
                order=None,
                error_message=str(e),
                execution_time=datetime.now(),
                mode=mode
            )
    
    def execute_sell_order(self, position: Position, exit_price: float,
                          order_type: OrderType = OrderType.MARKET,
                          mode: Optional[str] = None) -> ExecutionResult:
        """Execute a sell order."""
        if mode is None:
            mode = get_current_mode()
        
        try:
            # Validate sell order
            if not self._validate_sell_order(position, exit_price, mode):
                return ExecutionResult(
                    success=False,
                    order=None,
                    error_message="Sell order validation failed",
                    execution_time=datetime.now(),
                    mode=mode
                )
            
            # Create sell order
            order = self._create_order(
                symbol=position.symbol,
                quantity=position.quantity,
                price=exit_price,
                side=OrderSide.SELL,
                order_type=order_type,
                position_id=position.id,
                mode=mode
            )
            
            # Execute order based on mode
            if is_real_trading():
                execution_result = self._execute_real_order(order)
            else:
                execution_result = self._execute_paper_order(order)
            
            # Log the trade
            if execution_result.success:
                self._log_trade(order, mode)
                logger.info(f"Sell order executed: {position.symbol} {position.quantity} shares @ ${exit_price:.2f} in {mode} mode")
            
            return execution_result
            
        except Exception as e:
            logger.error(f"Sell order execution error: {e}")
            return ExecutionResult(
                success=False,
                order=None,
                error_message=str(e),
                execution_time=datetime.now(),
                mode=mode
            )
    
    def _create_order(self, symbol: str, quantity: int, price: float,
                     side: OrderSide, order_type: OrderType,
                     position_id: Optional[int] = None,
                     mode: str = None, bracket_params: Optional[BracketParameters] = None) -> Order:
        """Create an order object."""
        order_id = str(uuid.uuid4())
        execution_type = "REAL" if is_real_trading() else "PAPER"
        
        # Insert order into database with bracket parameters
        query = """
            INSERT INTO orders 
            (position_id, order_type, side, symbol, quantity, price, status, mode,
             atr, atr_multiplier, r_multiple, bracket_type, volatility_percent)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        # Extract bracket parameters if available
        atr = bracket_params.atr if bracket_params else None
        atr_multiplier = bracket_params.atr_multiplier if bracket_params else None
        r_multiple = bracket_params.r_multiple if bracket_params else None
        bracket_type = bracket_params.bracket_type.value if bracket_params else None
        volatility_percent = bracket_params.volatility_percent if bracket_params else None
        
        with get_connection(mode) as conn:
            cursor = conn.cursor()
            cursor.execute(query, (position_id, order_type.value, side.value, symbol, quantity, price, 
                                 OrderStatus.PENDING.value, mode, atr, atr_multiplier, r_multiple, 
                                 bracket_type, volatility_percent))
            order_db_id = cursor.lastrowid
            conn.commit()
        
        return Order(
            id=order_db_id,
            order_id=order_id,
            position_id=position_id,
            order_type=order_type,
            side=side,
            symbol=symbol,
            quantity=quantity,
            price=price,
            status=OrderStatus.PENDING,
            filled_quantity=0,
            filled_price=0.0,
            created_at=datetime.now(),
            submitted_at=None,
            executed_at=None,
            mode=mode,
            execution_type=execution_type,
            # Phase 6: ATR bracket parameters
            atr=atr,
            atr_multiplier=atr_multiplier,
            r_multiple=r_multiple,
            bracket_type=bracket_type,
            volatility_percent=volatility_percent
        )
    
    def _execute_real_order(self, order: Order) -> ExecutionResult:
        """Execute order in real trading mode (Questrade API)."""
        try:
            # In production, this would integrate with Questrade API
            # For now, simulate real execution with some latency and potential failures
            
            # Simulate API call delay
            import time
            time.sleep(0.1)  # 100ms delay
            
            # Simulate execution (90% success rate for demo)
            import random
            if random.random() < 0.9:
                # Successful execution
                order.status = OrderStatus.FILLED
                order.filled_quantity = order.quantity
                order.filled_price = order.price
                order.submitted_at = datetime.now()
                order.executed_at = datetime.now()
                
                # Update order in database
                self._update_order_status(order)
                
                return ExecutionResult(
                    success=True,
                    order=order,
                    error_message=None,
                    execution_time=datetime.now(),
                    mode=order.mode
                )
            else:
                # Simulated failure
                order.status = OrderStatus.REJECTED
                self._update_order_status(order)
                
                return ExecutionResult(
                    success=False,
                    order=order,
                    error_message="Order rejected by broker",
                    execution_time=datetime.now(),
                    mode=order.mode
                )
                
        except Exception as e:
            logger.error(f"Real order execution error: {e}")
            order.status = OrderStatus.REJECTED
            self._update_order_status(order)
            
            return ExecutionResult(
                success=False,
                order=order,
                error_message=str(e),
                execution_time=datetime.now(),
                mode=order.mode
            )
    
    def _execute_paper_order(self, order: Order) -> ExecutionResult:
        """Execute order in paper trading mode."""
        try:
            # Paper trading always succeeds (simulated)
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.filled_price = order.price
            order.submitted_at = datetime.now()
            order.executed_at = datetime.now()
            
            # Update order in database
            self._update_order_status(order)
            
            return ExecutionResult(
                success=True,
                order=order,
                error_message=None,
                execution_time=datetime.now(),
                mode=order.mode
            )
            
        except Exception as e:
            logger.error(f"Paper order execution error: {e}")
            order.status = OrderStatus.REJECTED
            self._update_order_status(order)
            
            return ExecutionResult(
                success=False,
                order=order,
                error_message=str(e),
                execution_time=datetime.now(),
                mode=order.mode
            )
    
    def _update_order_status(self, order: Order) -> None:
        """Update order status in database."""
        query = """
            UPDATE orders 
            SET status = ?, filled_quantity = ?, filled_price = ?, 
                submitted_at = ?, executed_at = ?
            WHERE id = ?
        """
        
        execute_update(query, (
            order.status.value,
            order.filled_quantity,
            order.filled_price,
            order.submitted_at,
            order.executed_at,
            order.id
        ), order.mode)
    
    def _validate_buy_order(self, symbol: str, quantity: int, price: float, mode: str) -> bool:
        """Validate buy order parameters."""
        try:
            # Check portfolio limits
            portfolio_limits = self.risk_manager.check_portfolio_limits(mode)
            if not portfolio_limits['can_open_new_position']:
                logger.warning("Cannot open new position - portfolio limits reached")
                return False
            
            # Check if position already exists
            existing_position = self.position_manager.get_position_by_symbol(symbol, mode)
            if existing_position:
                logger.warning(f"Position already exists for {symbol}")
                return False
            
            # Validate quantity and price
            if quantity <= 0 or price <= 0:
                logger.error("Invalid quantity or price")
                return False
            
            # Check position size limits
            account_balance = 10000.0  # Would be from database in production
            position_value = quantity * price
            max_position_value = self.risk_manager.get_max_position_size(symbol, 0.02, account_balance, mode)
            
            if position_value > max_position_value:
                logger.warning(f"Position size exceeds limit: ${position_value:.2f} > ${max_position_value:.2f}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Buy order validation error: {e}")
            return False
    
    def _validate_sell_order(self, position: Position, exit_price: float, mode: str) -> bool:
        """Validate sell order parameters."""
        try:
            # Check if position exists and is open
            if position.status != 'OPEN':
                logger.error(f"Position {position.symbol} is not open")
                return False
            
            # Validate exit price
            if exit_price <= 0:
                logger.error("Invalid exit price")
                return False
            
            # Check if position belongs to current mode
            if position.mode != mode:
                logger.error(f"Position mode mismatch: {position.mode} != {mode}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Sell order validation error: {e}")
            return False
    
    def _log_trade(self, order: Order, mode: str) -> None:
        """Log trade execution."""
        try:
            # Log to system logs
            query = """
                INSERT INTO system_logs (log_level, component, message, mode, data)
                VALUES (?, ?, ?, ?, ?)
            """
            
            log_data = {
                "order_id": order.order_id,
                "symbol": order.symbol,
                "side": order.side.value,
                "quantity": order.quantity,
                "price": order.price,
                "execution_type": order.execution_type
            }
            
            execute_update(query, (
                "INFO",
                "execution_engine",
                f"Trade executed: {order.side.value} {order.quantity} {order.symbol} @ ${order.price:.2f}",
                mode,
                str(log_data)
            ), mode)
            
        except Exception as e:
            logger.error(f"Trade logging error: {e}")
    
    def track_order_status(self, order_id: str, mode: Optional[str] = None) -> Optional[Order]:
        """Track order status by order ID."""
        if mode is None:
            mode = get_current_mode()
        
        query = "SELECT * FROM orders WHERE order_id = ? AND mode = ?"
        rows = execute_query(query, (order_id, mode), mode)
        
        if not rows:
            return None
        
        row = rows[0]
        return Order(
            id=row['id'],
            order_id=row.get('order_id', ''),
            position_id=row['position_id'],
            order_type=OrderType(row['order_type']),
            side=OrderSide.BUY if row.get('side', 'BUY') == 'BUY' else OrderSide.SELL,
            symbol=row['symbol'],
            quantity=row['quantity'],
            price=row['price'],
            status=OrderStatus(row['status']),
            filled_quantity=row.get('filled_quantity', 0),
            filled_price=row.get('filled_price', 0.0),
            created_at=datetime.fromisoformat(row['created_at']),
            submitted_at=datetime.fromisoformat(row['submitted_at']) if row.get('submitted_at') else None,
            executed_at=datetime.fromisoformat(row['executed_at']) if row.get('executed_at') else None,
            mode=row['mode'],
            execution_type="REAL" if is_real_trading() else "PAPER"
        )
    
    def get_order_history(self, symbol: Optional[str] = None, mode: Optional[str] = None) -> List[Order]:
        """Get order history for a symbol or all orders."""
        if mode is None:
            mode = get_current_mode()
        
        if symbol:
            query = "SELECT * FROM orders WHERE symbol = ? AND mode = ? ORDER BY created_at DESC"
            params = (symbol, mode)
        else:
            query = "SELECT * FROM orders WHERE mode = ? ORDER BY created_at DESC"
            params = (mode,)
        
        rows = execute_query(query, params, mode)
        orders = []
        
        for row in rows:
            order = Order(
                id=row['id'],
                order_id=row.get('order_id', ''),
                position_id=row['position_id'],
                order_type=OrderType(row['order_type']),
                side=OrderSide.BUY if row.get('side', 'BUY') == 'BUY' else OrderSide.SELL,
                symbol=row['symbol'],
                quantity=row['quantity'],
                price=row['price'],
                status=OrderStatus(row['status']),
                filled_quantity=row.get('filled_quantity', 0),
                filled_price=row.get('filled_price', 0.0),
                created_at=datetime.fromisoformat(row['created_at']),
                submitted_at=datetime.fromisoformat(row['submitted_at']) if row.get('submitted_at') else None,
                executed_at=datetime.fromisoformat(row['executed_at']) if row.get('executed_at') else None,
                mode=row['mode'],
                execution_type="REAL" if is_real_trading() else "PAPER"
            )
            orders.append(order)
        
        return orders

    def submit_orders(self, selections: list) -> List[ExecutionResult]:
        """Submit orders for selected symbols (paper trading in offline mode)."""
        results = []
        for selection in selections:
            try:
                # Handle both position_size and shares attribute names
                shares = getattr(selection, 'shares', None) or getattr(selection, 'position_size', 0)
                result = self.execute_buy_order(
                    symbol=selection.symbol,
                    quantity=int(shares),
                    price=selection.entry_price,
                    order_type=OrderType.LIMIT,
                )
                results.append(result)
                if result.success:
                    logger.info(f"Order submitted: {selection.symbol} x{shares} @ ${selection.entry_price:.2f}")
                else:
                    logger.warning(f"Order failed for {selection.symbol}: {result.message}")
            except Exception as e:
                logger.error(f"Failed to submit order for {selection.symbol}: {e}")
        return results

    def cancel_order(self, order_id: str, mode: Optional[str] = None) -> bool:
        """Cancel a pending order."""
        if mode is None:
            mode = get_current_mode()
        
        try:
            # Check if order exists and is cancellable
            order = self.track_order_status(order_id, mode)
            if not order:
                logger.error(f"Order {order_id} not found")
                return False
            
            if order.status not in [OrderStatus.PENDING, OrderStatus.SUBMITTED]:
                logger.error(f"Order {order_id} cannot be cancelled (status: {order.status.value})")
                return False
            
            # Update order status
            query = "UPDATE orders SET status = ? WHERE order_id = ? AND mode = ?"
            updated_rows = execute_update(query, (OrderStatus.CANCELLED.value, order_id, mode), mode)
            
            if updated_rows > 0:
                logger.info(f"Order {order_id} cancelled successfully")
                return True
            else:
                logger.error(f"Failed to cancel order {order_id}")
                return False
                
        except Exception as e:
            logger.error(f"Order cancellation error: {e}")
            return False
    
    def get_execution_summary(self, mode: Optional[str] = None) -> Dict[str, Any]:
        """Get execution engine summary."""
        if mode is None:
            mode = get_current_mode()
        
        # Get order statistics
        query = """
            SELECT 
                COUNT(*) as total_orders,
                SUM(CASE WHEN status = 'FILLED' THEN 1 ELSE 0 END) as filled_orders,
                SUM(CASE WHEN status = 'PENDING' THEN 1 ELSE 0 END) as pending_orders,
                SUM(CASE WHEN status = 'REJECTED' THEN 1 ELSE 0 END) as rejected_orders,
                SUM(CASE WHEN COALESCE(side, 'BUY') = 'BUY' THEN 1 ELSE 0 END) as buy_orders,
                SUM(CASE WHEN COALESCE(side, 'BUY') = 'SELL' THEN 1 ELSE 0 END) as sell_orders
            FROM orders 
            WHERE mode = ?
        """
        
        result = execute_query(query, (mode,), mode)
        stats = result[0] if result else {}
        
        # Get recent orders
        recent_orders = self.get_order_history(mode=mode)[:10]
        
        # Convert stats to dict if it's a Row object
        if hasattr(stats, 'keys'):
            stats_dict = dict(stats)
        else:
            stats_dict = stats
        
        summary = {
            "mode": mode,
            "execution_type": "REAL" if is_real_trading() else "PAPER",
            "order_statistics": {
                "total_orders": stats_dict.get('total_orders', 0),
                "filled_orders": stats_dict.get('filled_orders', 0),
                "pending_orders": stats_dict.get('pending_orders', 0),
                "rejected_orders": stats_dict.get('rejected_orders', 0),
                "buy_orders": stats_dict.get('buy_orders', 0),
                "sell_orders": stats_dict.get('sell_orders', 0),
                "success_rate": (stats_dict.get('filled_orders', 0) / max(stats_dict.get('total_orders', 0), 1)) * 100 if stats_dict.get('total_orders', 0) > 0 else 0
            },
            "recent_orders": [
                {
                    "order_id": order.order_id,
                    "symbol": order.symbol,
                    "side": order.side.value,
                    "quantity": order.quantity,
                    "price": order.price,
                    "status": order.status.value,
                    "created_at": order.created_at
                }
                for order in recent_orders
            ],
            "execution_capabilities": {
                "real_trading": is_real_trading(),
                "paper_trading": is_paper_trading(),
                "order_types": [ot.value for ot in OrderType],
                "order_sides": [os.value for os in OrderSide]
            }
        }
        
        return summary
    
    def _get_atr_value(self, symbol: str, mode: str) -> float:
        """
        Get ATR value for a symbol.
        
        In a real implementation, this would fetch ATR from market data.
        For now, returns a simplified ATR calculation.
        """
        try:
            # Simplified ATR calculation - in production, use real market data
            # This is a placeholder that returns a reasonable ATR value
            base_atr = 0.02  # 2% base ATR
            
            # Adjust based on symbol volatility (simplified)
            if symbol in ['AAPL', 'MSFT', 'GOOGL']:
                return base_atr * 0.8  # Large cap stocks - lower volatility
            elif symbol in ['TSLA', 'NVDA', 'AMD']:
                return base_atr * 1.5  # High volatility stocks
            else:
                return base_atr  # Default ATR
            
        except Exception as e:
            logger.error(f"Error getting ATR value for {symbol}: {e}")
            return 0.02  # Default fallback ATR

# Global execution engine instance
_execution_engine: Optional[ExecutionEngine] = None

def get_execution_engine() -> ExecutionEngine:
    """Get the global execution engine instance."""
    global _execution_engine
    if _execution_engine is None:
        _execution_engine = ExecutionEngine()
    return _execution_engine

def execute_buy_order(symbol: str, quantity: int, price: float,
                     order_type: OrderType = OrderType.MARKET,
                     position_id: Optional[int] = None,
                     mode: Optional[str] = None) -> ExecutionResult:
    """Execute a buy order."""
    return get_execution_engine().execute_buy_order(symbol, quantity, price, order_type, position_id, mode)

def execute_sell_order(position: Position, exit_price: float,
                      order_type: OrderType = OrderType.MARKET,
                      mode: Optional[str] = None) -> ExecutionResult:
    """Execute a sell order."""
    return get_execution_engine().execute_sell_order(position, exit_price, order_type, mode)

def track_order_status(order_id: str, mode: Optional[str] = None) -> Optional[Order]:
    """Track order status."""
    return get_execution_engine().track_order_status(order_id, mode)

def get_order_history(symbol: Optional[str] = None, mode: Optional[str] = None) -> List[Order]:
    """Get order history."""
    return get_execution_engine().get_order_history(symbol, mode)

def get_execution_summary(mode: Optional[str] = None) -> Dict[str, Any]:
    """Get execution summary."""
    return get_execution_engine().get_execution_summary(mode)