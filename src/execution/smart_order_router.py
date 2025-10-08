"""
Smart Order Router (VWAP/TWAP/POV)
Implements advanced order routing algorithms for optimal execution
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from enum import Enum
import asyncio
import time

logger = logging.getLogger(__name__)

class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    VWAP = "vwap"
    TWAP = "twap"
    POV = "pov"  # Percentage of Volume
    ICEBERG = "iceberg"

class OrderStatus(Enum):
    """Order status"""
    PENDING = "pending"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

@dataclass
class Order:
    """Order representation"""
    order_id: str
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: int
    order_type: OrderType
    price: Optional[float] = None
    limit_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    average_price: float = 0.0
    timestamp: datetime = None
    expiry: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class Execution:
    """Execution record"""
    execution_id: str
    order_id: str
    symbol: str
    side: str
    quantity: int
    price: float
    timestamp: datetime
    venue: str = "simulated"

class VWAPRouter:
    """Volume Weighted Average Price router"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.execution_history = []
        self.volume_profile = {}
        
        logger.info("VWAP Router initialized")
    
    def calculate_vwap_target(self, symbol: str, duration_minutes: int = 60) -> float:
        """Calculate VWAP target for a symbol"""
        try:
            # Get historical volume data (simulated)
            # In practice, this would come from market data
            base_price = 100.0  # This would be current market price
            
            # Simulate volume-weighted price calculation
            # Real implementation would use actual tick data
            vwap_target = base_price * (1 + np.random.normal(0, 0.01))
            
            return vwap_target
            
        except Exception as e:
            logger.error(f"Error calculating VWAP target: {e}")
            return 0.0
    
    def route_vwap_order(self, order: Order, market_data: Dict) -> List[Execution]:
        """Route VWAP order with optimal execution"""
        try:
            executions = []
            remaining_quantity = order.quantity
            start_time = order.timestamp
            duration = timedelta(minutes=60)  # Default 1-hour VWAP
            
            # Calculate VWAP target
            vwap_target = self.calculate_vwap_target(order.symbol)
            
            # Simulate execution over time
            current_time = start_time
            while remaining_quantity > 0 and current_time < start_time + duration:
                # Calculate participation rate based on market volume
                participation_rate = self._calculate_participation_rate(
                    order.symbol, remaining_quantity, market_data
                )
                
                # Execute portion of order
                execution_quantity = min(
                    remaining_quantity,
                    int(order.quantity * participation_rate)
                )
                
                if execution_quantity > 0:
                    # Calculate execution price (close to VWAP target)
                    execution_price = vwap_target * (1 + np.random.normal(0, 0.005))
                    
                    execution = Execution(
                        execution_id=f"exec_{len(self.execution_history) + 1}",
                        order_id=order.order_id,
                        symbol=order.symbol,
                        side=order.side,
                        quantity=execution_quantity,
                        price=execution_price,
                        timestamp=current_time
                    )
                    
                    executions.append(execution)
                    remaining_quantity -= execution_quantity
                
                # Move to next time slice
                current_time += timedelta(minutes=5)  # 5-minute intervals
                
                # Simulate processing delay
                time.sleep(0.01)
            
            return executions
            
        except Exception as e:
            logger.error(f"Error routing VWAP order: {e}")
            return []
    
    def _calculate_participation_rate(self, symbol: str, remaining_quantity: int, 
                                    market_data: Dict) -> float:
        """Calculate optimal participation rate"""
        try:
            # Base participation rate
            base_rate = 0.1  # 10% of remaining quantity
            
            # Adjust based on market conditions
            if 'volatility' in market_data:
                volatility = market_data['volatility']
                if volatility > 0.02:  # High volatility
                    base_rate *= 0.5  # Reduce participation
                elif volatility < 0.01:  # Low volatility
                    base_rate *= 1.5  # Increase participation
            
            # Adjust based on remaining quantity
            if remaining_quantity > 10000:  # Large order
                base_rate *= 0.7  # Reduce participation for large orders
            
            return min(base_rate, 0.3)  # Cap at 30%
            
        except Exception as e:
            logger.error(f"Error calculating participation rate: {e}")
            return 0.1

class TWAPRouter:
    """Time Weighted Average Price router"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.execution_history = []
        
        logger.info("TWAP Router initialized")
    
    def route_twap_order(self, order: Order, duration_minutes: int = 60) -> List[Execution]:
        """Route TWAP order with time-based execution"""
        try:
            executions = []
            remaining_quantity = order.quantity
            start_time = order.timestamp
            duration = timedelta(minutes=duration_minutes)
            
            # Calculate time slices
            num_slices = max(1, duration_minutes // 5)  # 5-minute slices
            quantity_per_slice = remaining_quantity // num_slices
            
            current_time = start_time
            slice_duration = duration / num_slices
            
            for slice_num in range(num_slices):
                if remaining_quantity <= 0:
                    break
                
                # Execute quantity for this time slice
                execution_quantity = min(quantity_per_slice, remaining_quantity)
                
                if execution_quantity > 0:
                    # Calculate execution price (simulated)
                    execution_price = order.price or 100.0
                    execution_price *= (1 + np.random.normal(0, 0.002))  # Small price variation
                    
                    execution = Execution(
                        execution_id=f"exec_{len(self.execution_history) + 1}",
                        order_id=order.order_id,
                        symbol=order.symbol,
                        side=order.side,
                        quantity=execution_quantity,
                        price=execution_price,
                        timestamp=current_time
                    )
                    
                    executions.append(execution)
                    remaining_quantity -= execution_quantity
                
                # Move to next time slice
                current_time += slice_duration
                
                # Simulate processing delay
                time.sleep(0.01)
            
            return executions
            
        except Exception as e:
            logger.error(f"Error routing TWAP order: {e}")
            return []

class POVRouter:
    """Percentage of Volume router"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.execution_history = []
        self.volume_tracker = {}
        
        logger.info("POV Router initialized")
    
    def route_pov_order(self, order: Order, pov_percentage: float = 0.1, 
                       market_data: Dict = None) -> List[Execution]:
        """Route POV order based on market volume"""
        try:
            executions = []
            remaining_quantity = order.quantity
            start_time = order.timestamp
            duration = timedelta(minutes=60)  # Default 1-hour execution
            
            # Track volume for this symbol
            if order.symbol not in self.volume_tracker:
                self.volume_tracker[order.symbol] = []
            
            current_time = start_time
            while remaining_quantity > 0 and current_time < start_time + duration:
                # Get current market volume (simulated)
                current_volume = self._get_market_volume(order.symbol, current_time)
                
                # Calculate target participation
                target_participation = current_volume * pov_percentage
                
                # Execute based on volume participation
                execution_quantity = min(
                    remaining_quantity,
                    int(target_participation)
                )
                
                if execution_quantity > 0:
                    # Calculate execution price
                    execution_price = order.price or 100.0
                    execution_price *= (1 + np.random.normal(0, 0.003))
                    
                    execution = Execution(
                        execution_id=f"exec_{len(self.execution_history) + 1}",
                        order_id=order.order_id,
                        symbol=order.symbol,
                        side=order.side,
                        quantity=execution_quantity,
                        price=execution_price,
                        timestamp=current_time
                    )
                    
                    executions.append(execution)
                    remaining_quantity -= execution_quantity
                
                # Move to next time slice
                current_time += timedelta(minutes=2)  # 2-minute intervals
                
                # Simulate processing delay
                time.sleep(0.01)
            
            return executions
            
        except Exception as e:
            logger.error(f"Error routing POV order: {e}")
            return []
    
    def _get_market_volume(self, symbol: str, timestamp: datetime) -> int:
        """Get market volume for a symbol at a given time"""
        try:
            # Simulate market volume (in practice, this would be real data)
            base_volume = 10000  # Base volume
            
            # Add time-based variation
            hour_factor = 1.0
            if 9 <= timestamp.hour <= 16:  # Market hours
                hour_factor = 1.5
            elif timestamp.hour < 9 or timestamp.hour > 16:  # Off hours
                hour_factor = 0.3
            
            # Add random variation
            volume = int(base_volume * hour_factor * (1 + np.random.normal(0, 0.2)))
            
            return max(volume, 100)  # Minimum volume
            
        except Exception as e:
            logger.error(f"Error getting market volume: {e}")
            return 1000

class SmartOrderRouter:
    """Main smart order router that coordinates all routing algorithms"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.vwap_router = VWAPRouter(config)
        self.twap_router = TWAPRouter(config)
        self.pov_router = POVRouter(config)
        self.active_orders = {}
        self.execution_history = []
        
        logger.info("Smart Order Router initialized")
    
    def route_order(self, order: Order, market_data: Dict = None) -> List[Execution]:
        """Route order using appropriate algorithm"""
        try:
            logger.info(f"Routing order {order.order_id}: {order.order_type.value}")
            
            # Store active order
            self.active_orders[order.order_id] = order
            
            # Route based on order type
            if order.order_type == OrderType.VWAP:
                executions = self.vwap_router.route_vwap_order(order, market_data or {})
            elif order.order_type == OrderType.TWAP:
                executions = self.twap_router.route_twap_order(order)
            elif order.order_type == OrderType.POV:
                pov_percentage = self.config.get('default_pov_percentage', 0.1)
                executions = self.pov_router.route_pov_order(order, pov_percentage, market_data)
            elif order.order_type == OrderType.MARKET:
                executions = self._route_market_order(order)
            elif order.order_type == OrderType.LIMIT:
                executions = self._route_limit_order(order)
            else:
                logger.error(f"Unsupported order type: {order.order_type}")
                return []
            
            # Update order status
            if executions:
                total_filled = sum(exec.quantity for exec in executions)
                if total_filled >= order.quantity:
                    order.status = OrderStatus.FILLED
                else:
                    order.status = OrderStatus.PARTIAL
                
                order.filled_quantity = total_filled
                order.average_price = sum(exec.price * exec.quantity for exec in executions) / total_filled
            
            # Store executions
            self.execution_history.extend(executions)
            
            logger.info(f"Order {order.order_id} executed: {len(executions)} executions")
            return executions
            
        except Exception as e:
            logger.error(f"Error routing order: {e}")
            return []
    
    def _route_market_order(self, order: Order) -> List[Execution]:
        """Route market order for immediate execution"""
        try:
            # Market orders execute immediately at current market price
            execution_price = order.price or 100.0  # Would be current market price
            
            execution = Execution(
                execution_id=f"exec_{len(self.execution_history) + 1}",
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=execution_price,
                timestamp=datetime.now()
            )
            
            return [execution]
            
        except Exception as e:
            logger.error(f"Error routing market order: {e}")
            return []
    
    def _route_limit_order(self, order: Order) -> List[Execution]:
        """Route limit order"""
        try:
            # Limit orders execute when price is favorable
            # For simulation, we'll execute immediately if price is good
            if order.limit_price and order.price:
                if order.side == 'BUY' and order.price <= order.limit_price:
                    return self._route_market_order(order)
                elif order.side == 'SELL' and order.price >= order.limit_price:
                    return self._route_market_order(order)
            
            # If price not favorable, order remains pending
            return []
            
        except Exception as e:
            logger.error(f"Error routing limit order: {e}")
            return []
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an active order"""
        try:
            if order_id in self.active_orders:
                order = self.active_orders[order_id]
                order.status = OrderStatus.CANCELLED
                del self.active_orders[order_id]
                logger.info(f"Order {order_id} cancelled")
                return True
            else:
                logger.warning(f"Order {order_id} not found for cancellation")
                return False
                
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False
    
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get status of an order"""
        return self.active_orders.get(order_id)
    
    def get_execution_summary(self, order_id: str) -> Dict:
        """Get execution summary for an order"""
        try:
            executions = [exec for exec in self.execution_history if exec.order_id == order_id]
            
            if not executions:
                return {}
            
            total_quantity = sum(exec.quantity for exec in executions)
            total_value = sum(exec.price * exec.quantity for exec in executions)
            average_price = total_value / total_quantity if total_quantity > 0 else 0
            
            return {
                'order_id': order_id,
                'total_executions': len(executions),
                'total_quantity': total_quantity,
                'total_value': total_value,
                'average_price': average_price,
                'executions': executions
            }
            
        except Exception as e:
            logger.error(f"Error getting execution summary: {e}")
            return {}
    
    def get_router_performance(self) -> Dict:
        """Get router performance metrics"""
        try:
            if not self.execution_history:
                return {}
            
            # Calculate performance metrics
            total_executions = len(self.execution_history)
            total_volume = sum(exec.quantity for exec in self.execution_history)
            total_value = sum(exec.price * exec.quantity for exec in self.execution_history)
            
            # Calculate average execution time
            execution_times = []
            for exec in self.execution_history:
                if exec.order_id in self.active_orders:
                    order = self.active_orders[exec.order_id]
                    exec_time = (exec.timestamp - order.timestamp).total_seconds()
                    execution_times.append(exec_time)
            
            avg_execution_time = np.mean(execution_times) if execution_times else 0
            
            return {
                'total_executions': total_executions,
                'total_volume': total_volume,
                'total_value': total_value,
                'average_execution_time': avg_execution_time,
                'active_orders': len(self.active_orders),
                'router_types': {
                    'vwap': len([exec for exec in self.execution_history if 'vwap' in exec.order_id.lower()]),
                    'twap': len([exec for exec in self.execution_history if 'twap' in exec.order_id.lower()]),
                    'pov': len([exec for exec in self.execution_history if 'pov' in exec.order_id.lower()]),
                    'market': len([exec for exec in self.execution_history if 'market' in exec.order_id.lower()])
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting router performance: {e}")
            return {}
    
    def optimize_routing(self, symbol: str, quantity: int, side: str, 
                        market_data: Dict) -> OrderType:
        """Recommend optimal routing algorithm based on market conditions"""
        try:
            # Analyze market conditions
            volatility = market_data.get('volatility', 0.02)
            volume = market_data.get('volume', 10000)
            spread = market_data.get('spread', 0.01)
            
            # Decision logic
            if quantity > volume * 0.1:  # Large order (>10% of volume)
                if volatility > 0.03:  # High volatility
                    return OrderType.POV  # Use POV to minimize market impact
                else:
                    return OrderType.VWAP  # Use VWAP for large orders
            elif volatility > 0.02:  # Medium volatility
                return OrderType.TWAP  # Use TWAP for time-based execution
            else:
                return OrderType.MARKET  # Use market order for small orders
            
        except Exception as e:
            logger.error(f"Error optimizing routing: {e}")
            return OrderType.MARKET
