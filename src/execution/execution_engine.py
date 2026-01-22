"""
Advanced Execution Engine

Handles order execution with:
- VWAP (Volume Weighted Average Price) execution
- Partial fills
- Fractional shares
- Slippage modeling
- Commission calculation
- Order types (Market, Limit, Stop, IOC)
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from enum import Enum
import numpy as np
import pandas as pd

# Import new components
from .smart_order_router import SmartOrderRouter as SOR, OrderType as SOROrderType
from .slippage_latency_simulator import SlippageLatencySimulator
from .fractional_shares_handler import FractionalSharesHandler
from .intraday_trading_engine import IntradayTradingEngine

if TYPE_CHECKING:
    from .brokers.base_broker import BrokerAPI

logger = logging.getLogger(__name__)

class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    IOC = "immediate_or_cancel"  # Immediate or Cancel
    FOK = "fill_or_kill"  # Fill or Kill

class OrderSide(Enum):
    """Order side"""
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    """Order status"""
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class Order:
    """Trading order"""
    
    def __init__(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        allow_fractional: bool = True,
        time_in_force: str = "GTC"  # GTC, DAY, IOC, FOK
    ):
        self.order_id = str(uuid.uuid4())
        self.client_order_id = f"{symbol}_{side.value}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.order_type = order_type
        self.limit_price = limit_price
        self.stop_price = stop_price
        self.allow_fractional = allow_fractional
        self.time_in_force = time_in_force
        
        self.status = OrderStatus.PENDING
        self.filled_quantity = 0.0
        self.remaining_quantity = quantity
        self.average_fill_price = 0.0
        self.fills = []
        
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        
        logger.debug(f" Order created: {self.order_id} - {side.value} {quantity} {symbol}")
    
    def add_fill(self, quantity: float, price: float, commission: float = 0.0):
        """Add a fill to the order"""
        self.fills.append({
            'quantity': quantity,
            'price': price,
            'commission': commission,
            'timestamp': datetime.now()
        })
        
        # Update filled quantity
        self.filled_quantity += quantity
        self.remaining_quantity = self.quantity - self.filled_quantity
        
        # Update average fill price
        total_value = sum(f['quantity'] * f['price'] for f in self.fills)
        self.average_fill_price = total_value / self.filled_quantity if self.filled_quantity > 0 else 0.0
        
        # Update status
        if self.remaining_quantity <= 0.001:  # Small threshold for floating point
            self.status = OrderStatus.FILLED
        elif self.filled_quantity > 0:
            self.status = OrderStatus.PARTIALLY_FILLED
        else:
            self.status = OrderStatus.OPEN
        
        self.updated_at = datetime.now()
        
        logger.debug(
            f" Fill added to {self.order_id}: {quantity} @ ${price:.2f} "
            f"(Total: {self.filled_quantity}/{self.quantity})"
        )
    
    def cancel(self):
        """Cancel the order"""
        if self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
            logger.warning(f" Cannot cancel order {self.order_id} - status: {self.status.value}")
            return False
        
        self.status = OrderStatus.CANCELLED
        self.updated_at = datetime.now()
        logger.info(f" Order cancelled: {self.order_id}")
        return True
    
    def to_dict(self) -> Dict:
        """Convert order to dictionary"""
        return {
            'order_id': self.order_id,
            'client_order_id': self.client_order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'order_type': self.order_type.value,
            'limit_price': self.limit_price,
            'stop_price': self.stop_price,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'remaining_quantity': self.remaining_quantity,
            'average_fill_price': self.average_fill_price,
            'fills': self.fills,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

class ExecutionEngine:
    """
    Advanced execution engine with VWAP, partial fills, and fractional shares
    
    Features:
    - Multiple order types
    - VWAP execution algorithm
    - Partial fill simulation
    - Fractional share support
    - Realistic slippage modeling
    - Commission calculation
    - Order management
    """
    
    def __init__(
        self,
        commission_rate: float = 0.001,  # 0.1% commission
        min_commission: float = 1.0,  # $1 minimum
        slippage_model: str = "proportional",  # proportional, fixed, adaptive
        slippage_bps: float = 5.0,  # 5 basis points default
        allow_fractional: bool = True,
        max_order_size_pct: float = 0.10,  # Max 10% of daily volume
        config: Optional[Dict] = None,
        broker: Optional["BrokerAPI"] = None,
    ):
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.slippage_model = slippage_model
        self.slippage_bps = slippage_bps
        self.allow_fractional = allow_fractional
        self.max_order_size_pct = max_order_size_pct
        self.config = config or {}
        self.broker = broker
        
        # Initialize Smart Order Router
        try:
            self.smart_router = SOR(self.config)
            logger.info("Smart Order Router initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Smart Order Router: {e}")
            self.smart_router = None
        
        # Initialize Slippage & Latency Simulator
        try:
            self.slippage_simulator = SlippageLatencySimulator(self.config)
            logger.info("Slippage & Latency Simulator initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Slippage Simulator: {e}")
            self.slippage_simulator = None
        
        # Initialize Fractional Shares Handler
        try:
            self.fractional_handler = FractionalSharesHandler(self.config)
            logger.info("Fractional Shares Handler initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Fractional Shares Handler: {e}")
            self.fractional_handler = None
        
        # Initialize Intraday Trading Engine
        try:
            self.intraday_engine = IntradayTradingEngine(self.config)
            logger.info("Intraday Trading Engine initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Intraday Trading Engine: {e}")
            self.intraday_engine = None
        
        self.orders: Dict[str, Order] = {}
        self.execution_history: List[Dict] = []
        
        logger.info(" Execution Engine initialized with Fractional & Intraday support")
        if self.broker:
            logger.info(" Broker connected: %s", type(self.broker).__name__)
    
    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> Order:
        """Create a new order"""
        
        # Validate quantity
        if not self.allow_fractional and quantity != int(quantity):
            logger.warning(f" Fractional shares not allowed, rounding {quantity} to {int(quantity)}")
            quantity = int(quantity)
        
        if quantity <= 0:
            raise ValueError(f"Invalid quantity: {quantity}")
        
        # Create order
        order = Order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            allow_fractional=self.allow_fractional
        )
        
        # Store order
        self.orders[order.order_id] = order
        
        logger.info(f" Order created: {order.order_id} - {side.value} {quantity} {symbol}")
        
        return order

    def execute_broker_order(
        self,
        *,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        time_in_force: str = "Day",
    ) -> Dict:
        """Place an order via connected broker (paper/live)."""
        if not self.broker:
            raise RuntimeError("No broker configured for live/paper execution")

        action = "Buy" if side == OrderSide.BUY else "Sell"
        ord_type = "Market"
        if order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT):
            ord_type = "Limit"

        result = self.broker.place_order(
            symbol=symbol,
            quantity=quantity,
            action=action,
            order_type=ord_type,
            limit_price=limit_price,
            time_in_force=time_in_force,
        )

        status = {
            "submitted": bool(result) and not (isinstance(result, dict) and result.get("error")),
            "broker_result": result,
            "symbol": symbol,
            "side": action,
            "quantity": quantity,
            "order_type": ord_type,
            "limit_price": limit_price,
        }

        if status["submitted"]:
            logger.info(
                " Submitted broker order: %s %s x%s (%s)",
                action,
                symbol,
                quantity,
                ord_type,
            )
        else:
            logger.error(" Broker order rejected: %s", result)

        return status
    
    def execute_market_order(
        self,
        order: Order,
        current_price: float,
        volume: float,
        market_data: Optional[pd.DataFrame] = None
    ) -> bool:
        """
        Execute market order with realistic simulation
        
        Args:
            order: Order to execute
            current_price: Current market price
            volume: Current market volume
            market_data: Historical market data for VWAP calculation
        
        Returns:
            True if order executed successfully
        """
        
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
            logger.warning(f" Order {order.order_id} already {order.status.value}")
            return False
        
        # Check order size vs market volume
        order_value = order.remaining_quantity * current_price
        max_order_value = volume * current_price * self.max_order_size_pct
        
        if order_value > max_order_value:
            # Partial fill
            fill_quantity = (max_order_value / current_price)
            if not self.allow_fractional:
                fill_quantity = int(fill_quantity)
            
            logger.info(
                f" Large order - partial fill: {fill_quantity}/{order.remaining_quantity}"
            )
        else:
            fill_quantity = order.remaining_quantity
        
        # Calculate slippage
        slippage = self._calculate_slippage(
            quantity=fill_quantity,
            price=current_price,
            volume=volume,
            side=order.side
        )
        
        # Calculate execution price with slippage
        if order.side == OrderSide.BUY:
            execution_price = current_price * (1 + slippage)
        else:
            execution_price = current_price * (1 - slippage)
        
        # Calculate commission
        commission = self._calculate_commission(fill_quantity, execution_price)
        
        # Add fill
        order.add_fill(fill_quantity, execution_price, commission)
        
        # Record execution
        self._record_execution(order, fill_quantity, execution_price, commission, slippage)
        
        logger.info(
            f" Market order executed: {fill_quantity} {order.symbol} @ ${execution_price:.4f} "
            f"(slippage: {slippage:.4%}, commission: ${commission:.2f})"
        )
        
        return True
    
    def execute_vwap_order(
        self,
        order: Order,
        market_data: pd.DataFrame,
        time_window_minutes: int = 30
    ) -> bool:
        """
        Execute order using VWAP (Volume Weighted Average Price) algorithm
        
        Splits large order into smaller chunks to minimize market impact
        
        Args:
            order: Order to execute
            market_data: Market data with OHLCV
            time_window_minutes: Time window for VWAP calculation
        
        Returns:
            True if order executed successfully
        """
        
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
            return False
        
        logger.info(f" Executing VWAP order: {order.order_id}")
        
        # Calculate VWAP from historical data
        if 'vwap' in market_data.columns:
            vwap = market_data['vwap'].iloc[-1]
        else:
            # Calculate VWAP if not available
            typical_price = (market_data['high'] + market_data['low'] + market_data['close']) / 3
            vwap = (typical_price * market_data['volume']).sum() / market_data['volume'].sum()
        
        # Get current volume profile
        recent_data = market_data.tail(time_window_minutes)
        total_volume = recent_data['volume'].sum()
        
        # Split order into chunks based on volume profile
        num_chunks = min(10, len(recent_data))
        chunk_size = order.remaining_quantity / num_chunks
        
        if not self.allow_fractional:
            chunk_size = max(1, int(chunk_size))
        
        executed_chunks = 0
        
        for i in range(num_chunks):
            if order.remaining_quantity <= 0:
                break
            
            # Determine chunk size based on volume profile
            volume_weight = recent_data['volume'].iloc[-i-1] / total_volume if i < len(recent_data) else 1/num_chunks
            current_chunk = min(chunk_size * volume_weight * 2, order.remaining_quantity)
            
            if not self.allow_fractional:
                current_chunk = max(1, int(current_chunk))
            
            # Calculate execution price with VWAP-based slippage
            price_variance = np.random.normal(0, 0.001)  # Small random variance
            execution_price = vwap * (1 + price_variance)
            
            # Add direction-based adjustment
            if order.side == OrderSide.BUY:
                execution_price *= 1.0005  # Slight premium for buying
            else:
                execution_price *= 0.9995  # Slight discount for selling
            
            # Calculate commission
            commission = self._calculate_commission(current_chunk, execution_price)
            
            # Add fill
            slippage = abs(execution_price - vwap) / vwap
            order.add_fill(current_chunk, execution_price, commission)
            
            # Record execution
            self._record_execution(order, current_chunk, execution_price, commission, slippage)
            
            executed_chunks += 1
        
        logger.info(
            f" VWAP order executed in {executed_chunks} chunks: "
            f"{order.filled_quantity}/{order.quantity} @ avg ${order.average_fill_price:.4f}"
        )
        
        return True
    
    def execute_limit_order(
        self,
        order: Order,
        current_price: float,
        volume: float
    ) -> bool:
        """Execute limit order if price condition is met"""
        
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
            return False
        
        if order.limit_price is None:
            logger.error(f" Limit order {order.order_id} has no limit price")
            return False
        
        # Check if limit price condition is met
        can_execute = False
        if order.side == OrderSide.BUY and current_price <= order.limit_price:
            can_execute = True
        elif order.side == OrderSide.SELL and current_price >= order.limit_price:
            can_execute = True
        
        if not can_execute:
            logger.debug(
                f"⏳ Limit order {order.order_id} waiting: "
                f"current ${current_price:.2f} vs limit ${order.limit_price:.2f}"
            )
            order.status = OrderStatus.OPEN
            return False
        
        # Execute at limit price (or better)
        execution_price = order.limit_price
        
        # Calculate commission
        commission = self._calculate_commission(order.remaining_quantity, execution_price)
        
        # Add fill
        order.add_fill(order.remaining_quantity, execution_price, commission)
        
        # Record execution
        slippage = 0.0  # No slippage on limit orders (price improvement not modeled)
        self._record_execution(order, order.filled_quantity, execution_price, commission, slippage)
        
        logger.info(
            f" Limit order executed: {order.filled_quantity} {order.symbol} @ ${execution_price:.2f}"
        )
        
        return True
    
    def _calculate_slippage(
        self,
        quantity: float,
        price: float,
        volume: float,
        side: OrderSide
    ) -> float:
        """
        Calculate slippage based on order size and market conditions
        
        Returns:
            Slippage as decimal (e.g., 0.0005 for 5 bps)
        """
        
        if self.slippage_model == "fixed":
            # Fixed slippage
            return self.slippage_bps / 10000
        
        elif self.slippage_model == "proportional":
            # Proportional to order size relative to volume
            order_value = quantity * price
            volume_value = volume * price
            
            if volume_value > 0:
                order_pct = order_value / volume_value
                slippage = self.slippage_bps / 10000 * (1 + order_pct * 10)
            else:
                slippage = self.slippage_bps / 10000
            
            return min(slippage, 0.01)  # Cap at 1%
        
        elif self.slippage_model == "adaptive":
            # Adaptive slippage based on volatility and liquidity
            order_pct = (quantity * price) / (volume * price) if volume > 0 else 1.0
            
            # Base slippage
            base_slippage = self.slippage_bps / 10000
            
            # Liquidity adjustment
            liquidity_factor = min(order_pct * 5, 2.0)
            
            # Random volatility component
            volatility_factor = 1 + np.random.normal(0, 0.2)
            
            slippage = base_slippage * liquidity_factor * volatility_factor
            
            return min(max(slippage, 0), 0.02)  # Between 0% and 2%
        
        else:
            return self.slippage_bps / 10000
    
    def _calculate_commission(self, quantity: float, price: float) -> float:
        """Calculate commission for trade"""
        commission = quantity * price * self.commission_rate
        return max(commission, self.min_commission)
    
    def _record_execution(
        self,
        order: Order,
        quantity: float,
        price: float,
        commission: float,
        slippage: float
    ):
        """Record execution in history"""
        execution = {
            'timestamp': datetime.now(),
            'order_id': order.order_id,
            'symbol': order.symbol,
            'side': order.side.value,
            'quantity': quantity,
            'price': price,
            'commission': commission,
            'slippage': slippage,
            'order_type': order.order_type.value
        }
        
        self.execution_history.append(execution)
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        return self.orders.get(order_id)
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all open orders, optionally filtered by symbol"""
        open_orders = [
            order for order in self.orders.values()
            if order.status in [OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED]
        ]
        
        if symbol:
            open_orders = [o for o in open_orders if o.symbol == symbol]
        
        return open_orders
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        order = self.get_order(order_id)
        if order:
            return order.cancel()
        return False
    
    def get_execution_statistics(self) -> Dict:
        """Get execution statistics"""
        if not self.execution_history:
            return {}
        
        total_executions = len(self.execution_history)
        total_volume = sum(e['quantity'] for e in self.execution_history)
        total_value = sum(e['quantity'] * e['price'] for e in self.execution_history)
        total_commission = sum(e['commission'] for e in self.execution_history)
        avg_slippage = np.mean([e['slippage'] for e in self.execution_history])
        
        return {
            'total_executions': total_executions,
            'total_volume': total_volume,
            'total_value': total_value,
            'total_commission': total_commission,
            'average_slippage': avg_slippage,
            'commission_rate': total_commission / total_value if total_value > 0 else 0
        }
    
    def execute_smart_order(self, symbol: str, side: str, quantity: int, 
                          order_type: str = 'VWAP', market_data: Dict = None) -> Dict:
        """
        Execute order using Smart Order Router with advanced algorithms
        
        Args:
            symbol: Stock symbol
            side: 'BUY' or 'SELL'
            quantity: Order quantity
            order_type: 'VWAP', 'TWAP', 'POV', 'MARKET', 'LIMIT'
            market_data: Current market data
        
        Returns:
            Execution results
        """
        try:
            if not self.smart_router:
                logger.warning("Smart Order Router not available, using basic execution")
                return self.execute_basic_order(symbol, side, quantity, market_data)
            
            # Convert order type to SOR format
            sor_order_type = SOROrderType.MARKET
            if order_type.upper() == 'VWAP':
                sor_order_type = SOROrderType.VWAP
            elif order_type.upper() == 'TWAP':
                sor_order_type = SOROrderType.TWAP
            elif order_type.upper() == 'POV':
                sor_order_type = SOROrderType.POV
            elif order_type.upper() == 'LIMIT':
                sor_order_type = SOROrderType.LIMIT
            
            # Create SOR order
            sor_order = SOR.Order(
                order_id=f"sor_{uuid.uuid4().hex[:8]}",
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=sor_order_type,
                price=market_data.get('price', 100.0) if market_data else 100.0
            )
            
            # Route order
            executions = self.smart_router.route_order(sor_order, market_data or {})
            
            # Process executions
            total_filled = 0
            total_value = 0.0
            execution_records = []
            
            for execution in executions:
                # Calculate commission
                commission = self._calculate_commission(execution.quantity, execution.price)
                
                # Create execution record
                exec_record = {
                    'execution_id': execution.execution_id,
                    'order_id': execution.order_id,
                    'symbol': execution.symbol,
                    'side': execution.side,
                    'quantity': execution.quantity,
                    'price': execution.price,
                    'commission': commission,
                    'timestamp': execution.timestamp,
                    'venue': execution.venue,
                    'order_type': order_type
                }
                
                execution_records.append(exec_record)
                total_filled += execution.quantity
                total_value += execution.quantity * execution.price
            
            # Store execution history
            self.execution_history.extend(execution_records)
            
            # Calculate average price
            avg_price = total_value / total_filled if total_filled > 0 else 0
            
            result = {
                'order_id': sor_order.order_id,
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'filled_quantity': total_filled,
                'average_price': avg_price,
                'total_value': total_value,
                'total_commission': sum(exec['commission'] for exec in execution_records),
                'executions': execution_records,
                'order_type': order_type,
                'status': 'FILLED' if total_filled >= quantity else 'PARTIAL'
            }
            
            logger.info(f"Smart order executed: {symbol} {side} {total_filled}/{quantity} @ {avg_price:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Error executing smart order: {e}")
            return {}
    
    def execute_basic_order(self, symbol: str, side: str, quantity: int, 
                          market_data: Dict = None) -> Dict:
        """Execute basic order without smart routing"""
        try:
            current_price = market_data.get('price', 100.0) if market_data else 100.0
            
            # Create basic order
            order = self.create_order(symbol, side, quantity, OrderType.MARKET)
            
            # Execute immediately
            success = self.execute_market_order(order, current_price, 10000, market_data)
            
            if success:
                return {
                    'order_id': order.order_id,
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity,
                    'filled_quantity': quantity,
                    'average_price': order.average_price,
                    'total_value': quantity * order.average_price,
                    'total_commission': order.commission,
                    'status': 'FILLED'
                }
            else:
                return {}
                
        except Exception as e:
            logger.error(f"Error executing basic order: {e}")
            return {}
    
    # Fractional Shares Methods
    
    def execute_dollar_based_order(self, symbol: str, side: str, dollar_amount: float, 
                                   current_price: float) -> Dict:
        """
        Execute order based on dollar amount (e.g., buy $500 worth of stock).
        
        Args:
            symbol: Stock symbol
            side: 'BUY' or 'SELL'
            dollar_amount: Dollar amount to invest
            current_price: Current stock price
        
        Returns:
            Execution result
        
        Example:
            >>> engine.execute_dollar_based_order('AAPL', 'BUY', 500.0, 150.0)
            >>> # Buys $500 worth of AAPL (3.333333 shares @ $150)
        """
        try:
            if not self.fractional_handler:
                logger.warning("Fractional handler not available, using whole shares")
                quantity = int(dollar_amount / current_price)
                return self.execute_basic_order(symbol, side, quantity, {'price': current_price})
            
            # Create fractional order
            order = self.fractional_handler.create_dollar_based_order(
                symbol, side, dollar_amount, current_price
            )
            
            # Execute order
            result = self.fractional_handler.execute_fractional_order(order, current_price)
            
            logger.info(f"Executed dollar-based order: ${dollar_amount} of {symbol} = {result['shares']} shares")
            return result
            
        except Exception as e:
            logger.error(f"Error executing dollar-based order: {e}")
            return {}
    
    def execute_fractional_order(self, symbol: str, side: str, shares: float, 
                                current_price: float) -> Dict:
        """
        Execute order with fractional shares (e.g., buy 15.5 shares).
        
        Args:
            symbol: Stock symbol
            side: 'BUY' or 'SELL'
            shares: Number of shares (can be fractional)
            current_price: Current stock price
        
        Returns:
            Execution result
        
        Example:
            >>> engine.execute_fractional_order('AAPL', 'BUY', 15.5, 150.0)
            >>> # Buys 15.5 shares of AAPL @ $150 = $2,325
        """
        try:
            if not self.fractional_handler:
                logger.warning("Fractional handler not available, rounding to whole shares")
                quantity = int(shares)
                return self.execute_basic_order(symbol, side, quantity, {'price': current_price})
            
            # Create fractional order
            order = self.fractional_handler.create_share_based_order(symbol, side, shares)
            
            # Execute order
            result = self.fractional_handler.execute_fractional_order(order, current_price)
            
            logger.info(f"Executed fractional order: {shares} shares of {symbol} @ ${current_price:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Error executing fractional order: {e}")
            return {}
    
    def get_fractional_position(self, symbol: str) -> Optional[Dict]:
        """Get fractional position for a symbol"""
        if not self.fractional_handler:
            return None
        return self.fractional_handler.get_position(symbol)
    
    def get_all_fractional_positions(self) -> List[Dict]:
        """Get all fractional positions"""
        if not self.fractional_handler:
            return []
        return self.fractional_handler.get_all_positions()
    
    # Intraday Trading Methods
    
    def open_intraday_position(self, symbol: str, side: str, shares: float, entry_price: float,
                              stop_loss: Optional[float] = None, take_profit: Optional[float] = None) -> Dict:
        """
        Open an intraday trading position.
        
        Args:
            symbol: Stock symbol
            side: 'LONG' or 'SHORT'
            shares: Number of shares
            entry_price: Entry price
            stop_loss: Optional stop loss price
            take_profit: Optional take profit price
        
        Returns:
            Result dictionary
        
        Example:
            >>> engine.open_intraday_position('AAPL', 'LONG', 10, 150.0, stop_loss=148.0, take_profit=155.0)
        """
        try:
            if not self.intraday_engine:
                logger.warning("Intraday engine not available")
                return {'success': False, 'reason': 'Intraday engine not initialized'}
            
            result = self.intraday_engine.open_intraday_position(
                symbol, side, shares, entry_price, stop_loss, take_profit
            )
            
            if result['success']:
                logger.info(f"Opened intraday position: {symbol} {side} {shares} shares @ ${entry_price:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error opening intraday position: {e}")
            return {'success': False, 'reason': str(e)}
    
    def close_intraday_position(self, position_id: str, exit_price: float, reason: str = "") -> Dict:
        """
        Close an intraday position.
        
        Args:
            position_id: Position ID to close
            exit_price: Exit price
            reason: Reason for closing
        
        Returns:
            Result dictionary
        """
        try:
            if not self.intraday_engine:
                logger.warning("Intraday engine not available")
                return {'success': False, 'reason': 'Intraday engine not initialized'}
            
            result = self.intraday_engine.close_intraday_position(position_id, exit_price, reason)
            
            if result['success']:
                logger.info(f"Closed intraday position: {result['symbol']} P&L: ${result['pnl']:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error closing intraday position: {e}")
            return {'success': False, 'reason': str(e)}
    
    def update_intraday_positions(self, market_data: Dict[str, float]):
        """
        Update all intraday positions with current market prices.
        Automatically triggers stop loss, take profit, and trailing stops.
        
        Args:
            market_data: Dictionary of {symbol: current_price}
        """
        try:
            if not self.intraday_engine:
                return
            
            self.intraday_engine.update_positions(market_data)
            
            # Auto-close if near market close
            if self.intraday_engine.should_auto_close():
                logger.info("Near market close - auto-closing all intraday positions")
                self.intraday_engine.auto_close_all_positions(market_data)
            
        except Exception as e:
            logger.error(f"Error updating intraday positions: {e}")
    
    def get_open_intraday_positions(self) -> List[Dict]:
        """Get all open intraday positions"""
        if not self.intraday_engine:
            return []
        return self.intraday_engine.get_open_positions()
    
    def get_pdt_status(self) -> Dict:
        """Get Pattern Day Trader status"""
        if not self.intraday_engine:
            return {}
        return self.intraday_engine.get_pdt_status()
    
    def get_intraday_statistics(self) -> Dict:
        """Get intraday trading statistics"""
        if not self.intraday_engine:
            return {}
        return self.intraday_engine.get_statistics()

# Global execution engine instance
_execution_engine_instance = None

def get_execution_engine(**kwargs) -> ExecutionEngine:
    """Get global execution engine instance"""
    global _execution_engine_instance
    if _execution_engine_instance is None:
        _execution_engine_instance = ExecutionEngine(**kwargs)
    return _execution_engine_instance

