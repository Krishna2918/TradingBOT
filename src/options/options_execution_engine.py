"""
Options Execution Engine

Handles options order execution with:
- Single leg orders (calls, puts)
- Multi-leg strategies (spreads, straddles, etc.)
- Options order management
- Fill simulation with realistic pricing
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class OptionOrderType(Enum):
    """Option order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OptionOrderSide(Enum):
    """Option order side"""
    BUY_TO_OPEN = "buy_to_open"  # Open long position
    SELL_TO_CLOSE = "sell_to_close"  # Close long position
    SELL_TO_OPEN = "sell_to_open"  # Open short position (write)
    BUY_TO_CLOSE = "buy_to_close"  # Close short position


class OptionOrderStatus(Enum):
    """Option order status"""
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class OptionLeg:
    """Represents one leg of an options order"""
    symbol: str  # Option symbol
    side: OptionOrderSide
    quantity: int
    option_type: str  # 'CALL' or 'PUT'
    strike: float
    expiry: datetime
    limit_price: Optional[float] = None


@dataclass
class OptionOrder:
    """Represents an options order (single or multi-leg)"""
    order_id: str
    legs: List[OptionLeg]
    order_type: OptionOrderType
    strategy_type: str  # 'SINGLE', 'VERTICAL', 'IRON_CONDOR', 'STRADDLE', etc.
    status: OptionOrderStatus = OptionOrderStatus.PENDING
    net_debit_credit: float = 0.0  # Positive = debit, Negative = credit
    filled_quantity: int = 0
    average_fill_price: float = 0.0
    commission: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    
    def get_max_loss(self) -> float:
        """Calculate maximum loss for the strategy"""
        if self.strategy_type == 'SINGLE':
            # Single option
            if self.legs[0].side in [OptionOrderSide.BUY_TO_OPEN, OptionOrderSide.BUY_TO_CLOSE]:
                # Long option - max loss is premium paid
                return self.net_debit_credit * self.legs[0].quantity * 100
            else:
                # Short option - theoretically unlimited
                return float('inf')
        
        elif self.strategy_type in ['VERTICAL_CALL_SPREAD', 'VERTICAL_PUT_SPREAD']:
            # Vertical spread - max loss is difference in strikes minus net credit
            strikes = sorted([leg.strike for leg in self.legs])
            strike_diff = strikes[1] - strikes[0]
            return (strike_diff - abs(self.net_debit_credit)) * 100
        
        elif self.strategy_type == 'IRON_CONDOR':
            # Iron condor - max loss is width of widest spread minus net credit
            call_strikes = sorted([leg.strike for leg in self.legs if leg.option_type == 'CALL'])
            put_strikes = sorted([leg.strike for leg in self.legs if leg.option_type == 'PUT'])
            call_width = call_strikes[1] - call_strikes[0]
            put_width = put_strikes[1] - put_strikes[0]
            max_width = max(call_width, put_width)
            return (max_width - abs(self.net_debit_credit)) * 100
        
        else:
            # Conservative estimate
            return abs(self.net_debit_credit) * 100
    
    def get_max_profit(self) -> float:
        """Calculate maximum profit for the strategy"""
        if self.strategy_type == 'SINGLE':
            # Single option
            if self.legs[0].side in [OptionOrderSide.SELL_TO_OPEN, OptionOrderSide.SELL_TO_CLOSE]:
                # Short option - max profit is premium received
                return abs(self.net_debit_credit) * self.legs[0].quantity * 100
            else:
                # Long option - theoretically unlimited for calls, strike for puts
                return float('inf')
        
        elif self.strategy_type in ['VERTICAL_CALL_SPREAD', 'VERTICAL_PUT_SPREAD']:
            # Vertical spread - max profit is net credit received
            return abs(self.net_debit_credit) * 100
        
        elif self.strategy_type == 'IRON_CONDOR':
            # Iron condor - max profit is net credit received
            return abs(self.net_debit_credit) * 100
        
        else:
            # Conservative estimate
            return abs(self.net_debit_credit) * 100


class OptionsExecutionEngine:
    """
    Options execution engine for single and multi-leg strategies.
    
    Features:
    - Single leg orders (buy/sell calls/puts)
    - Multi-leg spreads (verticals, iron condors, butterflies)
    - Straddles and strangles
    - Risk calculations
    - Commission modeling
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Commission structure
        self.base_commission = self.config.get('base_commission', 1.0)  # $1 per trade
        self.per_contract_commission = self.config.get('per_contract_commission', 0.65)  # $0.65 per contract
        self.min_commission = self.config.get('min_commission', 1.0)
        
        # Risk limits
        self.max_naked_position = self.config.get('max_naked_position', 10)  # Max 10 naked contracts
        self.max_total_risk = self.config.get('max_total_risk', 5000.0)  # Max $5k risk per trade
        
        # Order tracking
        self.orders: Dict[str, OptionOrder] = {}
        self.positions: Dict[str, int] = {}  # {option_symbol: quantity}
        self.execution_history: List[Dict] = []
        
        logger.info("Options Execution Engine initialized")
    
    def create_single_option_order(self, symbol: str, side: OptionOrderSide, quantity: int,
                                   option_type: str, strike: float, expiry: datetime,
                                   order_type: OptionOrderType = OptionOrderType.MARKET,
                                   limit_price: Optional[float] = None) -> OptionOrder:
        """
        Create a single option order.
        
        Args:
            symbol: Option symbol
            side: Order side (BUY_TO_OPEN, SELL_TO_OPEN, etc.)
            quantity: Number of contracts
            option_type: 'CALL' or 'PUT'
            strike: Strike price
            expiry: Expiration date
            order_type: Order type
            limit_price: Limit price (if applicable)
        
        Returns:
            OptionOrder object
        """
        try:
            leg = OptionLeg(
                symbol=symbol,
                side=side,
                quantity=quantity,
                option_type=option_type,
                strike=strike,
                expiry=expiry,
                limit_price=limit_price
            )
            
            order = OptionOrder(
                order_id=f"opt_{uuid.uuid4().hex[:8]}",
                legs=[leg],
                order_type=order_type,
                strategy_type='SINGLE'
            )
            
            self.orders[order.order_id] = order
            
            logger.info(f"Created single option order: {side.value} {quantity} {option_type} ${strike} exp {expiry.strftime('%Y-%m-%d')}")
            return order
            
        except Exception as e:
            logger.error(f"Error creating single option order: {e}")
            raise
    
    def create_vertical_spread(self, underlying: str, spread_type: str, 
                              long_strike: float, short_strike: float, expiry: datetime,
                              quantity: int = 1, limit_price: Optional[float] = None) -> OptionOrder:
        """
        Create a vertical spread order.
        
        Args:
            underlying: Underlying symbol
            spread_type: 'CALL' or 'PUT'
            long_strike: Strike of long leg
            short_strike: Strike of short leg
            expiry: Expiration date
            quantity: Number of spreads
            limit_price: Net debit/credit limit
        
        Returns:
            OptionOrder object
        
        Example:
            >>> # Bull call spread: Buy $50 call, Sell $55 call
            >>> engine.create_vertical_spread('AAPL', 'CALL', 50.0, 55.0, expiry, 1)
        """
        try:
            # Create legs
            long_leg = OptionLeg(
                symbol=f"{underlying}_{spread_type}_{long_strike}_{expiry.strftime('%Y%m%d')}",
                side=OptionOrderSide.BUY_TO_OPEN,
                quantity=quantity,
                option_type=spread_type,
                strike=long_strike,
                expiry=expiry
            )
            
            short_leg = OptionLeg(
                symbol=f"{underlying}_{spread_type}_{short_strike}_{expiry.strftime('%Y%m%d')}",
                side=OptionOrderSide.SELL_TO_OPEN,
                quantity=quantity,
                option_type=spread_type,
                strike=short_strike,
                expiry=expiry
            )
            
            strategy_name = f"VERTICAL_{spread_type}_SPREAD"
            
            order = OptionOrder(
                order_id=f"opt_{uuid.uuid4().hex[:8]}",
                legs=[long_leg, short_leg],
                order_type=OptionOrderType.LIMIT if limit_price else OptionOrderType.MARKET,
                strategy_type=strategy_name
            )
            
            self.orders[order.order_id] = order
            
            logger.info(f"Created vertical {spread_type} spread: Long ${long_strike}, Short ${short_strike}")
            return order
            
        except Exception as e:
            logger.error(f"Error creating vertical spread: {e}")
            raise
    
    def create_iron_condor(self, underlying: str, call_long_strike: float, call_short_strike: float,
                          put_short_strike: float, put_long_strike: float, expiry: datetime,
                          quantity: int = 1, limit_price: Optional[float] = None) -> OptionOrder:
        """
        Create an iron condor order.
        
        Args:
            underlying: Underlying symbol
            call_long_strike: Long call strike (highest)
            call_short_strike: Short call strike
            put_short_strike: Short put strike
            put_long_strike: Long put strike (lowest)
            expiry: Expiration date
            quantity: Number of condors
            limit_price: Net credit limit
        
        Returns:
            OptionOrder object
        
        Example:
            >>> # Iron condor: Sell $55 call, Buy $60 call, Sell $45 put, Buy $40 put
            >>> engine.create_iron_condor('AAPL', 60.0, 55.0, 45.0, 40.0, expiry, 1)
        """
        try:
            legs = [
                # Call spread
                OptionLeg(
                    symbol=f"{underlying}_CALL_{call_short_strike}_{expiry.strftime('%Y%m%d')}",
                    side=OptionOrderSide.SELL_TO_OPEN,
                    quantity=quantity,
                    option_type='CALL',
                    strike=call_short_strike,
                    expiry=expiry
                ),
                OptionLeg(
                    symbol=f"{underlying}_CALL_{call_long_strike}_{expiry.strftime('%Y%m%d')}",
                    side=OptionOrderSide.BUY_TO_OPEN,
                    quantity=quantity,
                    option_type='CALL',
                    strike=call_long_strike,
                    expiry=expiry
                ),
                # Put spread
                OptionLeg(
                    symbol=f"{underlying}_PUT_{put_short_strike}_{expiry.strftime('%Y%m%d')}",
                    side=OptionOrderSide.SELL_TO_OPEN,
                    quantity=quantity,
                    option_type='PUT',
                    strike=put_short_strike,
                    expiry=expiry
                ),
                OptionLeg(
                    symbol=f"{underlying}_PUT_{put_long_strike}_{expiry.strftime('%Y%m%d')}",
                    side=OptionOrderSide.BUY_TO_OPEN,
                    quantity=quantity,
                    option_type='PUT',
                    strike=put_long_strike,
                    expiry=expiry
                )
            ]
            
            order = OptionOrder(
                order_id=f"opt_{uuid.uuid4().hex[:8]}",
                legs=legs,
                order_type=OptionOrderType.LIMIT if limit_price else OptionOrderType.MARKET,
                strategy_type='IRON_CONDOR'
            )
            
            self.orders[order.order_id] = order
            
            logger.info(f"Created iron condor: Calls ${call_short_strike}/${call_long_strike}, Puts ${put_short_strike}/${put_long_strike}")
            return order
            
        except Exception as e:
            logger.error(f"Error creating iron condor: {e}")
            raise
    
    def create_straddle(self, underlying: str, strike: float, expiry: datetime,
                       side: str = 'BUY', quantity: int = 1, limit_price: Optional[float] = None) -> OptionOrder:
        """
        Create a straddle order (buy/sell call and put at same strike).
        
        Args:
            underlying: Underlying symbol
            strike: Strike price
            expiry: Expiration date
            side: 'BUY' or 'SELL'
            quantity: Number of straddles
            limit_price: Net debit/credit limit
        
        Returns:
            OptionOrder object
        """
        try:
            order_side = OptionOrderSide.BUY_TO_OPEN if side == 'BUY' else OptionOrderSide.SELL_TO_OPEN
            
            legs = [
                OptionLeg(
                    symbol=f"{underlying}_CALL_{strike}_{expiry.strftime('%Y%m%d')}",
                    side=order_side,
                    quantity=quantity,
                    option_type='CALL',
                    strike=strike,
                    expiry=expiry
                ),
                OptionLeg(
                    symbol=f"{underlying}_PUT_{strike}_{expiry.strftime('%Y%m%d')}",
                    side=order_side,
                    quantity=quantity,
                    option_type='PUT',
                    strike=strike,
                    expiry=expiry
                )
            ]
            
            order = OptionOrder(
                order_id=f"opt_{uuid.uuid4().hex[:8]}",
                legs=legs,
                order_type=OptionOrderType.LIMIT if limit_price else OptionOrderType.MARKET,
                strategy_type='STRADDLE'
            )
            
            self.orders[order.order_id] = order
            
            logger.info(f"Created {side} straddle: Strike ${strike}")
            return order
            
        except Exception as e:
            logger.error(f"Error creating straddle: {e}")
            raise
    
    def execute_option_order(self, order_id: str, market_prices: Dict[str, Tuple[float, float]]) -> Dict:
        """
        Execute an options order.
        
        Args:
            order_id: Order ID to execute
            market_prices: Dict of {symbol: (bid, ask)} for each leg
        
        Returns:
            Execution result dictionary
        """
        try:
            if order_id not in self.orders:
                return {'success': False, 'reason': 'Order not found'}
            
            order = self.orders[order_id]
            
            # Calculate net debit/credit
            net_price = 0.0
            for leg in order.legs:
                if leg.symbol not in market_prices:
                    return {'success': False, 'reason': f'No market price for {leg.symbol}'}
                
                bid, ask = market_prices[leg.symbol]
                
                # Determine execution price based on side
                if leg.side in [OptionOrderSide.BUY_TO_OPEN, OptionOrderSide.BUY_TO_CLOSE]:
                    price = ask  # Pay the ask when buying
                    net_price += price * leg.quantity
                else:
                    price = bid  # Receive the bid when selling
                    net_price -= price * leg.quantity
            
            # Calculate commission
            total_contracts = sum(leg.quantity for leg in order.legs)
            commission = self.base_commission + (self.per_contract_commission * total_contracts * len(order.legs))
            commission = max(commission, self.min_commission)
            
            # Update order
            order.status = OptionOrderStatus.FILLED
            order.net_debit_credit = net_price
            order.filled_quantity = order.legs[0].quantity
            order.average_fill_price = net_price / total_contracts if total_contracts > 0 else 0
            order.commission = commission
            order.filled_at = datetime.now()
            
            # Update positions
            for leg in order.legs:
                if leg.side in [OptionOrderSide.BUY_TO_OPEN, OptionOrderSide.SELL_TO_CLOSE]:
                    delta = leg.quantity if leg.side == OptionOrderSide.BUY_TO_OPEN else -leg.quantity
                else:
                    delta = -leg.quantity if leg.side == OptionOrderSide.SELL_TO_OPEN else leg.quantity
                
                self.positions[leg.symbol] = self.positions.get(leg.symbol, 0) + delta
            
            # Store execution history
            self.execution_history.append({
                'order_id': order_id,
                'strategy_type': order.strategy_type,
                'net_debit_credit': net_price,
                'commission': commission,
                'max_loss': order.get_max_loss(),
                'max_profit': order.get_max_profit(),
                'timestamp': datetime.now().isoformat()
            })
            
            result = {
                'success': True,
                'order_id': order_id,
                'strategy_type': order.strategy_type,
                'net_debit_credit': net_price,
                'commission': commission,
                'max_loss': order.get_max_loss(),
                'max_profit': order.get_max_profit(),
                'status': 'FILLED'
            }
            
            logger.info(f"Executed option order: {order.strategy_type} | Net: ${net_price:.2f} | Commission: ${commission:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Error executing option order: {e}")
            return {'success': False, 'reason': str(e)}
    
    def get_position(self, symbol: str) -> int:
        """Get current position for an option symbol"""
        return self.positions.get(symbol, 0)
    
    def get_all_positions(self) -> Dict[str, int]:
        """Get all option positions"""
        return self.positions.copy()
    
    def get_order(self, order_id: str) -> Optional[OptionOrder]:
        """Get order by ID"""
        return self.orders.get(order_id)
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        if order.status in [OptionOrderStatus.FILLED, OptionOrderStatus.CANCELLED]:
            return False
        
        order.status = OptionOrderStatus.CANCELLED
        logger.info(f"Cancelled option order: {order_id}")
        return True
    
    def get_statistics(self) -> Dict:
        """Get execution statistics"""
        total_orders = len(self.orders)
        filled_orders = sum(1 for o in self.orders.values() if o.status == OptionOrderStatus.FILLED)
        total_commissions = sum(o.commission for o in self.orders.values() if o.status == OptionOrderStatus.FILLED)
        
        return {
            'total_orders': total_orders,
            'filled_orders': filled_orders,
            'cancelled_orders': sum(1 for o in self.orders.values() if o.status == OptionOrderStatus.CANCELLED),
            'open_positions': len([p for p in self.positions.values() if p != 0]),
            'total_contracts': sum(abs(p) for p in self.positions.values()),
            'total_commissions': total_commissions
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize engine
    engine = OptionsExecutionEngine()
    
    # Example 1: Single call option
    print("\n1. Creating single call order:")
    order1 = engine.create_single_option_order(
        symbol='AAPL_CALL_150_20250117',
        side=OptionOrderSide.BUY_TO_OPEN,
        quantity=1,
        option_type='CALL',
        strike=150.0,
        expiry=datetime(2025, 1, 17)
    )
    print(f"   Order ID: {order1.order_id}")
    
    # Execute with market prices
    market_prices = {
        'AAPL_CALL_150_20250117': (3.50, 3.55)  # bid, ask
    }
    result1 = engine.execute_option_order(order1.order_id, market_prices)
    print(f"   Result: {result1}")
    
    # Example 2: Vertical spread
    print("\n2. Creating vertical call spread:")
    order2 = engine.create_vertical_spread(
        underlying='AAPL',
        spread_type='CALL',
        long_strike=150.0,
        short_strike=155.0,
        expiry=datetime(2025, 1, 17),
        quantity=1
    )
    print(f"   Order ID: {order2.order_id}")
    print(f"   Max Loss: ${order2.get_max_loss():.2f}")
    print(f"   Max Profit: ${order2.get_max_profit():.2f}")
    
    # Example 3: Iron Condor
    print("\n3. Creating iron condor:")
    order3 = engine.create_iron_condor(
        underlying='AAPL',
        call_long_strike=160.0,
        call_short_strike=155.0,
        put_short_strike=145.0,
        put_long_strike=140.0,
        expiry=datetime(2025, 1, 17),
        quantity=1
    )
    print(f"   Order ID: {order3.order_id}")
    print(f"   Max Loss: ${order3.get_max_loss():.2f}")
    print(f"   Max Profit: ${order3.get_max_profit():.2f}")
    
    # Statistics
    print("\n4. Statistics:")
    stats = engine.get_statistics()
    print(f"   {stats}")

