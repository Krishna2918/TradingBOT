"""
Execution Agent

CRITICAL priority agent that handles intelligent order execution with
optimization, slippage control, and execution quality monitoring.

Responsibilities:
- Intelligent order execution
- Slippage control and optimization
- Execution quality monitoring
- Order routing optimization
- Fill rate tracking
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass

from .base_agent import BaseAgent, AgentPriority, AgentStatus, ResourceRequirements

logger = logging.getLogger(__name__)


@dataclass
class ExecutionMetrics:
    """Execution performance metrics"""
    timestamp: datetime
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: int
    requested_price: float
    executed_price: float
    slippage: float
    execution_time_ms: float
    fill_rate: float
    order_type: str


@dataclass
class ExecutionQuality:
    """Execution quality assessment"""
    symbol: str
    avg_slippage: float
    avg_execution_time: float
    fill_rate: float
    total_orders: int
    successful_orders: int
    quality_score: float  # 0-100


class ExecutionAgent(BaseAgent):
    """
    Execution Agent - CRITICAL priority
    
    Handles intelligent order execution with optimization and quality monitoring.
    """
    
    def __init__(self, mode: str = 'DEMO'):
        super().__init__(
            agent_id='execution_agent',
            name='Order Execution Agent',
            priority=AgentPriority.CRITICAL,  # Must always run
            resource_requirements=ResourceRequirements(
                min_cpu_percent=5.0,
                min_memory_mb=120.0,
                max_cpu_percent=20.0,
                max_memory_mb=400.0
            )
        )
        
        self.mode = mode
        self.order_executor = None
        self.execution_history: List[ExecutionMetrics] = []
        self.quality_metrics: Dict[str, ExecutionQuality] = {}
        
        # Performance tracking
        self.orders_executed = 0
        self.orders_failed = 0
        self.total_slippage = 0.0
        self.total_execution_time = 0.0
        
        # Execution settings
        self.max_slippage_percent = 0.5  # 0.5% max slippage
        self.max_execution_time_ms = 5000  # 5 seconds max
        self.retry_attempts = 3
    
    async def initialize(self) -> bool:
        """Initialize the Order Executor"""
        try:
            from src.trading.execution import OrderExecutor
            self.order_executor = OrderExecutor(mode=self.mode)
            self.status = AgentStatus.IDLE
            logger.info(f"Execution Agent initialized in {self.mode} mode")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Order Executor: {e}")
            self.status = AgentStatus.ERROR
            return False
    
    async def shutdown(self) -> bool:
        """Gracefully shutdown"""
        logger.info("Execution Agent shutting down")
        self.status = AgentStatus.STOPPED
        return True
    
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process execution tasks.
        
        Task types:
        - 'execute_order': Execute a trading order
        - 'get_execution_quality': Get execution quality metrics
        - 'optimize_execution': Analyze and optimize execution parameters
        - 'get_fill_rates': Get fill rate statistics
        - 'cancel_order': Cancel pending order
        """
        task_type = task.get('type')
        
        if task_type == 'execute_order':
            return await self._execute_order(task)
        elif task_type == 'get_execution_quality':
            return await self._get_execution_quality(task.get('symbol'))
        elif task_type == 'optimize_execution':
            return await self._optimize_execution_parameters()
        elif task_type == 'get_fill_rates':
            return await self._get_fill_rates()
        elif task_type == 'cancel_order':
            return await self._cancel_order(task.get('order_id'))
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    async def _execute_order(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a trading order with optimization"""
        if not self.order_executor:
            raise RuntimeError("Order Executor not initialized")
        
        symbol = task.get('symbol')
        side = task.get('side')  # 'buy' or 'sell'
        quantity = task.get('quantity')
        price = task.get('price')
        order_type = task.get('order_type', 'MARKET')
        
        start_time = datetime.now()
        
        try:
            # Pre-execution validation
            validation_result = await self._validate_order(symbol, side, quantity, price)
            if not validation_result['valid']:
                self.orders_failed += 1
                return {
                    'success': False,
                    'error': validation_result['reason'],
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity
                }
            
            # Execute order based on side
            if side.lower() == 'buy':
                result = self.order_executor.execute_buy_order(
                    symbol=symbol,
                    quantity=quantity,
                    price=price,
                    order_type=order_type,
                    position_id=None,
                    mode=self.mode
                )
            else:  # sell
                # Get position for sell order
                from src.trading.positions import PositionManager
                position_manager = PositionManager(mode=self.mode)
                position = position_manager.get_position(symbol)
                
                if not position:
                    self.orders_failed += 1
                    return {
                        'success': False,
                        'error': f'No position found for {symbol}',
                        'symbol': symbol,
                        'side': side,
                        'quantity': quantity
                    }
                
                result = self.order_executor.execute_sell_order(
                    position=position,
                    price=price,
                    order_type=order_type,
                    mode=self.mode
                )
            
            # Calculate execution metrics
            execution_time = (datetime.now() - start_time).total_seconds() * 1000  # ms
            executed_price = result.price if hasattr(result, 'price') else price
            slippage = abs(executed_price - price) / price * 100 if price > 0 else 0
            
            # Store execution metrics
            metrics = ExecutionMetrics(
                timestamp=datetime.now(),
                symbol=symbol,
                side=side,
                quantity=quantity,
                requested_price=price,
                executed_price=executed_price,
                slippage=slippage,
                execution_time_ms=execution_time,
                fill_rate=1.0 if result.success else 0.0,
                order_type=order_type
            )
            
            self.execution_history.append(metrics)
            if len(self.execution_history) > 1000:  # Keep last 1000 executions
                self.execution_history.pop(0)
            
            # Update performance tracking
            if result.success:
                self.orders_executed += 1
                self.total_slippage += slippage
                self.total_execution_time += execution_time
            else:
                self.orders_failed += 1
            
            # Update quality metrics
            self._update_quality_metrics(symbol, metrics)
            
            return {
                'success': result.success,
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'requested_price': price,
                'executed_price': executed_price,
                'slippage_percent': slippage,
                'execution_time_ms': execution_time,
                'fill_rate': metrics.fill_rate,
                'order_id': result.order_id if hasattr(result, 'order_id') else None,
                'message': result.message if hasattr(result, 'message') else 'Order executed'
            }
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            self.orders_failed += 1
            
            logger.error(f"Order execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'execution_time_ms': execution_time
            }
    
    async def _validate_order(self, symbol: str, side: str, quantity: int, price: float) -> Dict[str, Any]:
        """Validate order before execution"""
        # Basic validation
        if not symbol or not side or not quantity or not price:
            return {'valid': False, 'reason': 'Missing required order parameters'}
        
        if quantity <= 0:
            return {'valid': False, 'reason': 'Quantity must be positive'}
        
        if price <= 0:
            return {'valid': False, 'reason': 'Price must be positive'}
        
        if side.lower() not in ['buy', 'sell']:
            return {'valid': False, 'reason': 'Side must be buy or sell'}
        
        # Check execution quality for this symbol
        if symbol in self.quality_metrics:
            quality = self.quality_metrics[symbol]
            if quality.quality_score < 50:  # Low quality
                return {'valid': False, 'reason': f'Poor execution quality for {symbol} (score: {quality.quality_score})'}
        
        return {'valid': True, 'reason': 'Order validation passed'}
    
    def _update_quality_metrics(self, symbol: str, metrics: ExecutionMetrics):
        """Update execution quality metrics for symbol"""
        if symbol not in self.quality_metrics:
            self.quality_metrics[symbol] = ExecutionQuality(
                symbol=symbol,
                avg_slippage=0.0,
                avg_execution_time=0.0,
                fill_rate=0.0,
                total_orders=0,
                successful_orders=0,
                quality_score=0.0
            )
        
        quality = self.quality_metrics[symbol]
        quality.total_orders += 1
        
        if metrics.fill_rate > 0:
            quality.successful_orders += 1
        
        # Update averages
        quality.avg_slippage = (quality.avg_slippage * (quality.total_orders - 1) + metrics.slippage) / quality.total_orders
        quality.avg_execution_time = (quality.avg_execution_time * (quality.total_orders - 1) + metrics.execution_time_ms) / quality.total_orders
        quality.fill_rate = quality.successful_orders / quality.total_orders
        
        # Calculate quality score (0-100)
        slippage_score = max(0, 100 - (quality.avg_slippage * 100))  # Lower slippage = higher score
        time_score = max(0, 100 - (quality.avg_execution_time / 100))  # Lower time = higher score
        fill_score = quality.fill_rate * 100  # Higher fill rate = higher score
        
        quality.quality_score = (slippage_score + time_score + fill_score) / 3
    
    async def _get_execution_quality(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get execution quality metrics"""
        if symbol:
            if symbol in self.quality_metrics:
                quality = self.quality_metrics[symbol]
                return {
                    'symbol': symbol,
                    'avg_slippage': quality.avg_slippage,
                    'avg_execution_time': quality.avg_execution_time,
                    'fill_rate': quality.fill_rate,
                    'total_orders': quality.total_orders,
                    'successful_orders': quality.successful_orders,
                    'quality_score': quality.quality_score
                }
            else:
                return {'error': f'No execution data for {symbol}'}
        else:
            # Return all symbols
            return {
                'symbols': {
                    sym: {
                        'avg_slippage': q.avg_slippage,
                        'avg_execution_time': q.avg_execution_time,
                        'fill_rate': q.fill_rate,
                        'total_orders': q.total_orders,
                        'quality_score': q.quality_score
                    }
                    for sym, q in self.quality_metrics.items()
                },
                'overall_metrics': {
                    'total_orders_executed': self.orders_executed,
                    'total_orders_failed': self.orders_failed,
                    'success_rate': self.orders_executed / (self.orders_executed + self.orders_failed) if (self.orders_executed + self.orders_failed) > 0 else 0,
                    'avg_slippage': self.total_slippage / self.orders_executed if self.orders_executed > 0 else 0,
                    'avg_execution_time': self.total_execution_time / self.orders_executed if self.orders_executed > 0 else 0
                }
            }
    
    async def _optimize_execution_parameters(self) -> Dict[str, Any]:
        """Analyze execution patterns and suggest optimizations"""
        if not self.execution_history:
            return {'error': 'No execution history available'}
        
        # Analyze recent executions (last 100)
        recent_executions = self.execution_history[-100:] if len(self.execution_history) >= 100 else self.execution_history
        
        # Calculate averages
        avg_slippage = sum(e.slippage for e in recent_executions) / len(recent_executions)
        avg_execution_time = sum(e.execution_time_ms for e in recent_executions) / len(recent_executions)
        avg_fill_rate = sum(e.fill_rate for e in recent_executions) / len(recent_executions)
        
        optimizations = []
        
        # Slippage optimization
        if avg_slippage > 0.3:  # 0.3% threshold
            optimizations.append({
                'category': 'slippage',
                'priority': 'high' if avg_slippage > 0.5 else 'medium',
                'issue': f'High average slippage: {avg_slippage:.2f}%',
                'recommendation': 'Consider using limit orders or reducing order size',
                'current_avg': avg_slippage,
                'target': 0.2
            })
        
        # Execution time optimization
        if avg_execution_time > 2000:  # 2 seconds
            optimizations.append({
                'category': 'execution_time',
                'priority': 'high' if avg_execution_time > 5000 else 'medium',
                'issue': f'Slow execution: {avg_execution_time:.0f}ms average',
                'recommendation': 'Check network latency or broker connection',
                'current_avg': avg_execution_time,
                'target': 1000
            })
        
        # Fill rate optimization
        if avg_fill_rate < 0.95:  # 95% threshold
            optimizations.append({
                'category': 'fill_rate',
                'priority': 'high' if avg_fill_rate < 0.9 else 'medium',
                'issue': f'Low fill rate: {avg_fill_rate:.1%}',
                'recommendation': 'Review order types and market conditions',
                'current_avg': avg_fill_rate,
                'target': 0.98
            })
        
        return {
            'analysis_timestamp': datetime.now().isoformat(),
            'sample_size': len(recent_executions),
            'current_metrics': {
                'avg_slippage': avg_slippage,
                'avg_execution_time': avg_execution_time,
                'avg_fill_rate': avg_fill_rate
            },
            'optimizations': optimizations,
            'total_optimizations': len(optimizations)
        }
    
    async def _get_fill_rates(self) -> Dict[str, Any]:
        """Get fill rate statistics by symbol and time period"""
        if not self.execution_history:
            return {'error': 'No execution history available'}
        
        # Group by symbol
        symbol_stats = {}
        for execution in self.execution_history:
            if execution.symbol not in symbol_stats:
                symbol_stats[execution.symbol] = {'total': 0, 'filled': 0}
            
            symbol_stats[execution.symbol]['total'] += 1
            if execution.fill_rate > 0:
                symbol_stats[execution.symbol]['filled'] += 1
        
        # Calculate fill rates
        fill_rates = {}
        for symbol, stats in symbol_stats.items():
            fill_rates[symbol] = {
                'total_orders': stats['total'],
                'filled_orders': stats['filled'],
                'fill_rate': stats['filled'] / stats['total'] if stats['total'] > 0 else 0
            }
        
        # Overall statistics
        total_orders = sum(stats['total'] for stats in symbol_stats.values())
        total_filled = sum(stats['filled'] for stats in symbol_stats.values())
        overall_fill_rate = total_filled / total_orders if total_orders > 0 else 0
        
        return {
            'overall': {
                'total_orders': total_orders,
                'filled_orders': total_filled,
                'fill_rate': overall_fill_rate
            },
            'by_symbol': fill_rates,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    async def _cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel a pending order"""
        # This would integrate with your broker's cancel order functionality
        # For now, return a placeholder response
        return {
            'success': True,
            'order_id': order_id,
            'message': 'Order cancellation requested',
            'note': 'Integration with broker cancel API needed'
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status with execution-specific metrics"""
        base_status = super().get_status()
        
        # Add execution-specific metrics
        base_status['execution_metrics'] = {
            'mode': self.mode,
            'orders_executed': self.orders_executed,
            'orders_failed': self.orders_failed,
            'success_rate': self.orders_executed / (self.orders_executed + self.orders_failed) if (self.orders_executed + self.orders_failed) > 0 else 0,
            'avg_slippage': self.total_slippage / self.orders_executed if self.orders_executed > 0 else 0,
            'avg_execution_time': self.total_execution_time / self.orders_executed if self.orders_executed > 0 else 0,
            'execution_history_length': len(self.execution_history),
            'symbols_tracked': len(self.quality_metrics),
            'max_slippage_percent': self.max_slippage_percent,
            'max_execution_time_ms': self.max_execution_time_ms
        }
        
        return base_status
