"""
Fractional/Partial Shares Trading Handler

Enables trading of fractional shares for Canadian stocks.
Supports:
- Dollar-based position sizing
- Partial share execution
- Fractional share tracking
- Cost basis calculation
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP

logger = logging.getLogger(__name__)


@dataclass
class FractionalPosition:
    """Represents a fractional share position"""
    symbol: str
    shares: Decimal  # Can be fractional (e.g., 15.5 shares)
    average_cost: Decimal
    total_cost: Decimal
    timestamp: datetime
    position_id: str


@dataclass
class FractionalOrder:
    """Represents a fractional share order"""
    order_id: str
    symbol: str
    side: str  # BUY or SELL
    dollar_amount: Optional[float] = None  # Dollar-based order
    shares: Optional[Decimal] = None  # Share-based order
    limit_price: Optional[float] = None
    filled_shares: Decimal = Decimal('0')
    average_price: Decimal = Decimal('0')
    status: str = 'PENDING'  # PENDING, FILLED, PARTIAL, CANCELLED
    timestamp: datetime = None


class FractionalSharesHandler:
    """
    Handles fractional/partial share trading for Canadian stocks.
    
    Key Features:
    - Dollar-based position sizing ($500 worth of stock)
    - Fractional share execution (buy 15.5 shares)
    - Portfolio tracking with fractional shares
    - Cost basis calculation for tax purposes
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.fractional_enabled = self.config.get('fractional_shares_enabled', True)
        self.min_fractional_amount = self.config.get('min_fractional_amount', 1.0)  # $1 minimum
        self.max_fractional_amount = self.config.get('max_fractional_amount', 50000.0)  # $50k max
        self.fractional_precision = self.config.get('fractional_precision', 6)  # 6 decimal places
        
        # Position tracking
        self.positions: Dict[str, FractionalPosition] = {}
        self.order_history: List[FractionalOrder] = []
        
        logger.info(f"Fractional Shares Handler initialized (enabled={self.fractional_enabled})")
    
    def create_dollar_based_order(self, symbol: str, side: str, dollar_amount: float,
                                  current_price: float, limit_price: Optional[float] = None) -> FractionalOrder:
        """
        Create an order based on dollar amount instead of shares.
        
        Args:
            symbol: Stock symbol
            side: 'BUY' or 'SELL'
            dollar_amount: Dollar amount to invest (e.g., $500)
            current_price: Current stock price
            limit_price: Optional limit price
        
        Returns:
            FractionalOrder object
        
        Example:
            >>> handler.create_dollar_based_order('AAPL', 'BUY', 500.0, 150.0)
            >>> # Buys $500 worth of AAPL at $150 = 3.333333 shares
        """
        try:
            if not self.fractional_enabled:
                logger.warning("Fractional shares not enabled, rounding to whole shares")
                shares = Decimal(int(dollar_amount / current_price))
            else:
                # Calculate fractional shares
                shares = Decimal(str(dollar_amount / current_price)).quantize(
                    Decimal(f'0.{"0" * self.fractional_precision}'),
                    rounding=ROUND_DOWN
                )
            
            # Validate amount
            if dollar_amount < self.min_fractional_amount:
                raise ValueError(f"Order amount ${dollar_amount:.2f} below minimum ${self.min_fractional_amount:.2f}")
            
            if dollar_amount > self.max_fractional_amount:
                raise ValueError(f"Order amount ${dollar_amount:.2f} exceeds maximum ${self.max_fractional_amount:.2f}")
            
            # Create order
            order = FractionalOrder(
                order_id=f"frac_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                symbol=symbol,
                side=side,
                dollar_amount=dollar_amount,
                shares=shares,
                limit_price=limit_price,
                timestamp=datetime.now()
            )
            
            logger.info(f"Created dollar-based order: {symbol} {side} ${dollar_amount:.2f} = {shares} shares @ ${current_price:.2f}")
            return order
            
        except Exception as e:
            logger.error(f"Error creating dollar-based order: {e}")
            raise
    
    def create_share_based_order(self, symbol: str, side: str, shares: float,
                                 limit_price: Optional[float] = None) -> FractionalOrder:
        """
        Create an order based on fractional shares.
        
        Args:
            symbol: Stock symbol
            side: 'BUY' or 'SELL'
            shares: Number of shares (can be fractional, e.g., 15.5)
            limit_price: Optional limit price
        
        Returns:
            FractionalOrder object
        
        Example:
            >>> handler.create_share_based_order('AAPL', 'BUY', 15.5, 150.0)
            >>> # Buys 15.5 shares of AAPL at $150 = $2,325
        """
        try:
            # Convert to Decimal for precise calculation
            if not self.fractional_enabled:
                shares_decimal = Decimal(int(shares))
            else:
                shares_decimal = Decimal(str(shares)).quantize(
                    Decimal(f'0.{"0" * self.fractional_precision}'),
                    rounding=ROUND_DOWN
                )
            
            # Create order
            order = FractionalOrder(
                order_id=f"frac_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                symbol=symbol,
                side=side,
                shares=shares_decimal,
                limit_price=limit_price,
                timestamp=datetime.now()
            )
            
            logger.info(f"Created share-based order: {symbol} {side} {shares_decimal} shares")
            return order
            
        except Exception as e:
            logger.error(f"Error creating share-based order: {e}")
            raise
    
    def execute_fractional_order(self, order: FractionalOrder, execution_price: float) -> Dict:
        """
        Execute a fractional share order.
        
        Args:
            order: FractionalOrder to execute
            execution_price: Actual execution price
        
        Returns:
            Execution result dictionary
        """
        try:
            # Calculate actual cost
            execution_price_decimal = Decimal(str(execution_price))
            total_cost = order.shares * execution_price_decimal
            
            # Update order
            order.filled_shares = order.shares
            order.average_price = execution_price_decimal
            order.status = 'FILLED'
            
            # Update position
            if order.side == 'BUY':
                self._add_to_position(order.symbol, order.shares, execution_price_decimal, total_cost)
            else:  # SELL
                self._reduce_position(order.symbol, order.shares, execution_price_decimal)
            
            # Store order history
            self.order_history.append(order)
            
            result = {
                'order_id': order.order_id,
                'symbol': order.symbol,
                'side': order.side,
                'shares': float(order.shares),
                'price': float(execution_price_decimal),
                'total_cost': float(total_cost),
                'status': 'FILLED',
                'timestamp': order.timestamp.isoformat()
            }
            
            logger.info(f"Executed fractional order: {order.symbol} {order.side} {order.shares} @ ${execution_price:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Error executing fractional order: {e}")
            order.status = 'FAILED'
            raise
    
    def _add_to_position(self, symbol: str, shares: Decimal, price: Decimal, cost: Decimal):
        """Add shares to position with cost basis tracking"""
        if symbol in self.positions:
            # Update existing position
            position = self.positions[symbol]
            total_shares = position.shares + shares
            total_cost = position.total_cost + cost
            
            position.shares = total_shares
            position.total_cost = total_cost
            position.average_cost = total_cost / total_shares if total_shares > 0 else Decimal('0')
            position.timestamp = datetime.now()
        else:
            # Create new position
            self.positions[symbol] = FractionalPosition(
                symbol=symbol,
                shares=shares,
                average_cost=price,
                total_cost=cost,
                timestamp=datetime.now(),
                position_id=f"pos_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        
        logger.debug(f"Updated position: {symbol} = {self.positions[symbol].shares} shares @ avg ${self.positions[symbol].average_cost:.2f}")
    
    def _reduce_position(self, symbol: str, shares: Decimal, price: Decimal):
        """Reduce position by selling shares"""
        if symbol not in self.positions:
            raise ValueError(f"No position found for {symbol}")
        
        position = self.positions[symbol]
        
        if position.shares < shares:
            raise ValueError(f"Insufficient shares: have {position.shares}, trying to sell {shares}")
        
        # Calculate cost reduction (proportional to shares sold)
        cost_reduction = (shares / position.shares) * position.total_cost
        
        position.shares -= shares
        position.total_cost -= cost_reduction
        position.timestamp = datetime.now()
        
        # Remove position if fully closed
        if position.shares <= Decimal('0.000001'):  # Account for rounding
            del self.positions[symbol]
            logger.info(f"Position closed: {symbol}")
        else:
            logger.debug(f"Reduced position: {symbol} = {position.shares} shares remaining")
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get current fractional position for a symbol"""
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        return {
            'symbol': symbol,
            'shares': float(position.shares),
            'average_cost': float(position.average_cost),
            'total_cost': float(position.total_cost),
            'timestamp': position.timestamp.isoformat()
        }
    
    def get_all_positions(self) -> List[Dict]:
        """Get all fractional positions"""
        return [self.get_position(symbol) for symbol in self.positions.keys()]
    
    def calculate_position_value(self, symbol: str, current_price: float) -> Dict:
        """Calculate current value and P&L for a position"""
        if symbol not in self.positions:
            return {'error': f'No position found for {symbol}'}
        
        position = self.positions[symbol]
        current_value = position.shares * Decimal(str(current_price))
        unrealized_pnl = current_value - position.total_cost
        unrealized_pnl_pct = (unrealized_pnl / position.total_cost * 100) if position.total_cost > 0 else Decimal('0')
        
        return {
            'symbol': symbol,
            'shares': float(position.shares),
            'average_cost': float(position.average_cost),
            'current_price': current_price,
            'cost_basis': float(position.total_cost),
            'current_value': float(current_value),
            'unrealized_pnl': float(unrealized_pnl),
            'unrealized_pnl_pct': float(unrealized_pnl_pct)
        }
    
    def get_order_history(self, symbol: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Get fractional order history"""
        orders = self.order_history[-limit:]
        
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        
        return [{
            'order_id': o.order_id,
            'symbol': o.symbol,
            'side': o.side,
            'shares': float(o.shares) if o.shares else None,
            'dollar_amount': o.dollar_amount,
            'filled_shares': float(o.filled_shares),
            'average_price': float(o.average_price),
            'status': o.status,
            'timestamp': o.timestamp.isoformat()
        } for o in orders]
    
    def get_statistics(self) -> Dict:
        """Get fractional trading statistics"""
        total_positions = len(self.positions)
        total_shares = sum(float(p.shares) for p in self.positions.values())
        total_value = sum(float(p.total_cost) for p in self.positions.values())
        
        total_orders = len(self.order_history)
        buy_orders = sum(1 for o in self.order_history if o.side == 'BUY')
        sell_orders = sum(1 for o in self.order_history if o.side == 'SELL')
        
        return {
            'fractional_enabled': self.fractional_enabled,
            'total_positions': total_positions,
            'total_shares_held': total_shares,
            'total_position_value': total_value,
            'total_orders': total_orders,
            'buy_orders': buy_orders,
            'sell_orders': sell_orders,
            'fractional_precision': self.fractional_precision
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize handler
    handler = FractionalSharesHandler({
        'fractional_shares_enabled': True,
        'fractional_precision': 6
    })
    
    # Example 1: Dollar-based order
    print("\n1. Dollar-based order:")
    order1 = handler.create_dollar_based_order('AAPL', 'BUY', 500.0, 150.0)
    print(f"   Order: ${order1.dollar_amount} = {order1.shares} shares")
    
    # Execute order
    result1 = handler.execute_fractional_order(order1, 150.25)
    print(f"   Executed: {result1}")
    
    # Example 2: Share-based order
    print("\n2. Share-based order:")
    order2 = handler.create_share_based_order('AAPL', 'BUY', 15.5)
    print(f"   Order: {order2.shares} shares")
    
    result2 = handler.execute_fractional_order(order2, 150.50)
    print(f"   Executed: {result2}")
    
    # Check position
    print("\n3. Position:")
    position = handler.get_position('AAPL')
    print(f"   {position}")
    
    # Calculate P&L
    print("\n4. Position Value:")
    pnl = handler.calculate_position_value('AAPL', 155.0)
    print(f"   {pnl}")
    
    # Statistics
    print("\n5. Statistics:")
    stats = handler.get_statistics()
    print(f"   {stats}")

