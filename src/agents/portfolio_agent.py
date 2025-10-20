"""
Portfolio Agent

IMPORTANT priority agent that manages portfolio optimization, rebalancing,
and position sizing based on market conditions and risk parameters.

Responsibilities:
- Portfolio optimization
- Position rebalancing
- Asset allocation management
- Performance tracking
- Risk-adjusted returns analysis
"""

import logging
import asyncio
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from .base_agent import BaseAgent, AgentPriority, AgentStatus, ResourceRequirements

logger = logging.getLogger(__name__)


@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics"""
    timestamp: datetime
    total_value: float
    cash_balance: float
    invested_value: float
    daily_pnl: float
    total_pnl: float
    daily_return: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    volatility: float


@dataclass
class PositionAllocation:
    """Position allocation data"""
    symbol: str
    quantity: int
    current_price: float
    market_value: float
    allocation_percent: float
    target_allocation: float
    rebalance_needed: bool
    risk_score: float


@dataclass
class RebalancingRecommendation:
    """Portfolio rebalancing recommendation"""
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    current_allocation: float
    target_allocation: float
    quantity_change: int
    estimated_cost: float
    priority: str  # 'high', 'medium', 'low'
    reason: str


class PortfolioAgent(BaseAgent):
    """
    Portfolio Agent - IMPORTANT priority
    
    Manages portfolio optimization and rebalancing based on market conditions.
    """
    
    def __init__(self, mode: str = 'DEMO'):
        super().__init__(
            agent_id='portfolio_agent',
            name='Portfolio Management Agent',
            priority=AgentPriority.IMPORTANT,  # Important but not critical
            resource_requirements=ResourceRequirements(
                min_cpu_percent=8.0,
                min_memory_mb=150.0,
                max_cpu_percent=25.0,
                max_memory_mb=500.0
            )
        )
        
        self.mode = mode
        self.position_manager = None
        self.portfolio_history: List[PortfolioMetrics] = []
        self.rebalancing_history: List[RebalancingRecommendation] = []
        
        # Portfolio settings
        self.target_allocations = {
            'AAPL': 0.20,  # 20% Apple
            'MSFT': 0.15,  # 15% Microsoft
            'GOOGL': 0.15, # 15% Google
            'TSLA': 0.10,  # 10% Tesla
            'SPY': 0.25,   # 25% S&P 500 ETF
            'CASH': 0.15   # 15% Cash
        }
        
        self.rebalance_threshold = 0.05  # 5% deviation triggers rebalancing
        self.max_position_size = 0.30    # 30% max per position
        self.min_position_size = 0.02    # 2% min per position
        
        # Performance tracking
        self.rebalancing_events = 0
        self.optimization_suggestions = 0
        self.performance_analyses = 0
    
    async def initialize(self) -> bool:
        """Initialize portfolio management systems"""
        try:
            from src.trading.positions import PositionManager
            self.position_manager = PositionManager(mode=self.mode)
            self.status = AgentStatus.IDLE
            logger.info(f"Portfolio Agent initialized in {self.mode} mode")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Position Manager: {e}")
            self.status = AgentStatus.ERROR
            return False
    
    async def shutdown(self) -> bool:
        """Gracefully shutdown"""
        logger.info("Portfolio Agent shutting down")
        self.status = AgentStatus.STOPPED
        return True
    
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process portfolio management tasks.
        
        Task types:
        - 'analyze_portfolio': Analyze current portfolio performance
        - 'rebalance_portfolio': Generate rebalancing recommendations
        - 'optimize_allocation': Optimize asset allocation
        - 'get_performance_metrics': Get portfolio performance metrics
        - 'update_targets': Update target allocations
        - 'risk_assessment': Assess portfolio risk
        """
        task_type = task.get('type')
        
        if task_type == 'analyze_portfolio':
            return await self._analyze_portfolio()
        elif task_type == 'rebalance_portfolio':
            return await self._rebalance_portfolio()
        elif task_type == 'optimize_allocation':
            return await self._optimize_allocation()
        elif task_type == 'get_performance_metrics':
            return await self._get_performance_metrics()
        elif task_type == 'update_targets':
            return await self._update_target_allocations(task.get('allocations'))
        elif task_type == 'risk_assessment':
            return await self._assess_portfolio_risk()
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    async def _analyze_portfolio(self) -> Dict[str, Any]:
        """Analyze current portfolio composition and performance"""
        if not self.position_manager:
            raise RuntimeError("Position Manager not initialized")
        
        try:
            # Get current positions
            positions = self.position_manager.get_all_positions()
            
            # Calculate portfolio metrics
            total_value = 0.0
            cash_balance = 100000.0  # Placeholder - would get from account
            position_allocations = []
            
            for position in positions:
                current_price = position.current_price
                market_value = position.quantity * current_price
                total_value += market_value
                
                # Calculate allocation
                allocation = market_value / (total_value + cash_balance) if (total_value + cash_balance) > 0 else 0
                target_allocation = self.target_allocations.get(position.symbol, 0.0)
                
                position_allocations.append(PositionAllocation(
                    symbol=position.symbol,
                    quantity=position.quantity,
                    current_price=current_price,
                    market_value=market_value,
                    allocation_percent=allocation,
                    target_allocation=target_allocation,
                    rebalance_needed=abs(allocation - target_allocation) > self.rebalance_threshold,
                    risk_score=self._calculate_position_risk(position)
                ))
            
            # Add cash allocation
            total_portfolio_value = total_value + cash_balance
            cash_allocation = cash_balance / total_portfolio_value if total_portfolio_value > 0 else 0
            
            position_allocations.append(PositionAllocation(
                symbol='CASH',
                quantity=0,
                current_price=1.0,
                market_value=cash_balance,
                allocation_percent=cash_allocation,
                target_allocation=self.target_allocations.get('CASH', 0.15),
                rebalance_needed=abs(cash_allocation - self.target_allocations.get('CASH', 0.15)) > self.rebalance_threshold,
                risk_score=0.0  # Cash has no risk
            ))
            
            # Calculate portfolio metrics
            portfolio_metrics = self._calculate_portfolio_metrics(total_portfolio_value, cash_balance, total_value)
            
            # Store in history
            self.portfolio_history.append(portfolio_metrics)
            if len(self.portfolio_history) > 1000:  # Keep last 1000 snapshots
                self.portfolio_history.pop(0)
            
            self.performance_analyses += 1
            
            return {
                'timestamp': datetime.now().isoformat(),
                'portfolio_value': total_portfolio_value,
                'cash_balance': cash_balance,
                'invested_value': total_value,
                'positions': [
                    {
                        'symbol': pos.symbol,
                        'quantity': pos.quantity,
                        'current_price': pos.current_price,
                        'market_value': pos.market_value,
                        'allocation_percent': pos.allocation_percent,
                        'target_allocation': pos.target_allocation,
                        'rebalance_needed': pos.rebalance_needed,
                        'risk_score': pos.risk_score
                    }
                    for pos in position_allocations
                ],
                'metrics': {
                    'daily_pnl': portfolio_metrics.daily_pnl,
                    'total_pnl': portfolio_metrics.total_pnl,
                    'daily_return': portfolio_metrics.daily_return,
                    'total_return': portfolio_metrics.total_return,
                    'sharpe_ratio': portfolio_metrics.sharpe_ratio,
                    'max_drawdown': portfolio_metrics.max_drawdown,
                    'volatility': portfolio_metrics.volatility
                },
                'rebalancing_needed': any(pos.rebalance_needed for pos in position_allocations)
            }
            
        except Exception as e:
            logger.error(f"Portfolio analysis failed: {e}")
            return {'error': str(e)}
    
    def _calculate_position_risk(self, position) -> float:
        """Calculate risk score for a position (0-1 scale)"""
        # Simple risk calculation based on position size and volatility
        # In a real implementation, this would use historical volatility data
        
        position_size = position.quantity * position.current_price
        portfolio_value = 100000.0  # Placeholder
        
        size_risk = min(position_size / portfolio_value, 1.0)  # Larger positions = higher risk
        
        # Add volatility component (placeholder)
        volatility_risk = 0.3  # Would be calculated from historical data
        
        return (size_risk + volatility_risk) / 2
    
    def _calculate_portfolio_metrics(self, total_value: float, cash_balance: float, invested_value: float) -> PortfolioMetrics:
        """Calculate portfolio performance metrics"""
        # Placeholder calculations - would use historical data in real implementation
        
        # Calculate returns (placeholder)
        daily_pnl = np.random.normal(0, 100)  # Random daily P&L
        total_pnl = daily_pnl * 30  # Placeholder total P&L
        
        daily_return = daily_pnl / total_value if total_value > 0 else 0
        total_return = total_pnl / total_value if total_value > 0 else 0
        
        # Calculate Sharpe ratio (placeholder)
        sharpe_ratio = total_return / 0.15 if 0.15 > 0 else 0  # Assuming 15% volatility
        
        # Calculate max drawdown (placeholder)
        max_drawdown = abs(min(0, total_return - 0.1))  # Placeholder
        
        # Calculate volatility (placeholder)
        volatility = 0.15  # 15% annual volatility
        
        return PortfolioMetrics(
            timestamp=datetime.now(),
            total_value=total_value,
            cash_balance=cash_balance,
            invested_value=invested_value,
            daily_pnl=daily_pnl,
            total_pnl=total_pnl,
            daily_return=daily_return,
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            volatility=volatility
        )
    
    async def _rebalance_portfolio(self) -> Dict[str, Any]:
        """Generate portfolio rebalancing recommendations"""
        try:
            # Get current portfolio analysis
            analysis = await self._analyze_portfolio()
            if 'error' in analysis:
                return analysis
            
            recommendations = []
            total_portfolio_value = analysis['portfolio_value']
            
            for position in analysis['positions']:
                symbol = position['symbol']
                current_allocation = position['allocation_percent']
                target_allocation = position['target_allocation']
                
                if position['rebalance_needed']:
                    # Calculate required changes
                    target_value = total_portfolio_value * target_allocation
                    current_value = position['market_value']
                    value_difference = target_value - current_value
                    
                    # Determine action
                    if value_difference > 0:
                        action = 'buy'
                        quantity_change = int(value_difference / position['current_price'])
                        priority = 'high' if abs(current_allocation - target_allocation) > 0.1 else 'medium'
                    elif value_difference < 0:
                        action = 'sell'
                        quantity_change = int(abs(value_difference) / position['current_price'])
                        priority = 'high' if abs(current_allocation - target_allocation) > 0.1 else 'medium'
                    else:
                        action = 'hold'
                        quantity_change = 0
                        priority = 'low'
                    
                    if action != 'hold':
                        recommendation = RebalancingRecommendation(
                            symbol=symbol,
                            action=action,
                            current_allocation=current_allocation,
                            target_allocation=target_allocation,
                            quantity_change=quantity_change,
                            estimated_cost=abs(value_difference),
                            priority=priority,
                            reason=f'Rebalance to target allocation of {target_allocation:.1%}'
                        )
                        
                        recommendations.append(recommendation)
            
            # Store recommendations
            self.rebalancing_history.extend(recommendations)
            if len(self.rebalancing_history) > 100:  # Keep last 100 recommendations
                self.rebalancing_history = self.rebalancing_history[-100:]
            
            self.rebalancing_events += 1
            
            return {
                'timestamp': datetime.now().isoformat(),
                'rebalancing_needed': len(recommendations) > 0,
                'recommendations': [
                    {
                        'symbol': rec.symbol,
                        'action': rec.action,
                        'current_allocation': rec.current_allocation,
                        'target_allocation': rec.target_allocation,
                        'quantity_change': rec.quantity_change,
                        'estimated_cost': rec.estimated_cost,
                        'priority': rec.priority,
                        'reason': rec.reason
                    }
                    for rec in recommendations
                ],
                'total_recommendations': len(recommendations),
                'high_priority_count': len([r for r in recommendations if r.priority == 'high'])
            }
            
        except Exception as e:
            logger.error(f"Portfolio rebalancing analysis failed: {e}")
            return {'error': str(e)}
    
    async def _optimize_allocation(self) -> Dict[str, Any]:
        """Optimize asset allocation based on performance and risk"""
        try:
            # Get current portfolio analysis
            analysis = await self._analyze_portfolio()
            if 'error' in analysis:
                return analysis
            
            # Simple optimization logic (in real implementation, would use more sophisticated algorithms)
            current_allocations = {pos['symbol']: pos['allocation_percent'] for pos in analysis['positions']}
            
            optimizations = []
            
            # Check for over-concentration
            for symbol, allocation in current_allocations.items():
                if allocation > self.max_position_size:
                    optimizations.append({
                        'symbol': symbol,
                        'issue': f'Over-concentrated: {allocation:.1%} > {self.max_position_size:.1%}',
                        'recommendation': f'Reduce allocation to {self.max_position_size:.1%}',
                        'priority': 'high'
                    })
                elif allocation < self.min_position_size and allocation > 0:
                    optimizations.append({
                        'symbol': symbol,
                        'issue': f'Under-sized position: {allocation:.1%} < {self.min_position_size:.1%}',
                        'recommendation': f'Increase to {self.min_position_size:.1%} or close position',
                        'priority': 'medium'
                    })
            
            # Check cash allocation
            cash_allocation = current_allocations.get('CASH', 0)
            target_cash = self.target_allocations.get('CASH', 0.15)
            
            if cash_allocation < target_cash * 0.5:  # Less than half target
                optimizations.append({
                    'symbol': 'CASH',
                    'issue': f'Low cash allocation: {cash_allocation:.1%}',
                    'recommendation': f'Increase cash to {target_cash:.1%} for liquidity',
                    'priority': 'high'
                })
            
            self.optimization_suggestions += len(optimizations)
            
            return {
                'timestamp': datetime.now().isoformat(),
                'optimizations': optimizations,
                'total_optimizations': len(optimizations),
                'high_priority_count': len([opt for opt in optimizations if opt['priority'] == 'high'])
            }
            
        except Exception as e:
            logger.error(f"Allocation optimization failed: {e}")
            return {'error': str(e)}
    
    async def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive portfolio performance metrics"""
        if not self.portfolio_history:
            return {'error': 'No portfolio history available'}
        
        latest = self.portfolio_history[-1]
        
        # Calculate trends (last 30 days)
        recent_metrics = self.portfolio_history[-30:] if len(self.portfolio_history) >= 30 else self.portfolio_history
        
        # Calculate performance statistics
        returns = [m.daily_return for m in recent_metrics]
        pnls = [m.daily_pnl for m in recent_metrics]
        
        avg_daily_return = np.mean(returns) if returns else 0
        avg_daily_pnl = np.mean(pnls) if pnls else 0
        volatility = np.std(returns) if len(returns) > 1 else 0
        
        # Calculate Sharpe ratio
        sharpe_ratio = avg_daily_return / volatility if volatility > 0 else 0
        
        # Calculate max drawdown
        cumulative_returns = np.cumprod([1 + r for r in returns]) if returns else [1]
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
        
        return {
            'current': {
                'total_value': latest.total_value,
                'cash_balance': latest.cash_balance,
                'invested_value': latest.invested_value,
                'daily_pnl': latest.daily_pnl,
                'total_pnl': latest.total_pnl,
                'daily_return': latest.daily_return,
                'total_return': latest.total_return
            },
            'performance': {
                'avg_daily_return': avg_daily_return,
                'avg_daily_pnl': avg_daily_pnl,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'analysis_period_days': len(recent_metrics)
            },
            'history_length': len(self.portfolio_history),
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    async def _update_target_allocations(self, new_allocations: Dict[str, float]) -> Dict[str, Any]:
        """Update target asset allocations"""
        if not new_allocations:
            return {'error': 'No allocations provided'}
        
        # Validate allocations
        total_allocation = sum(new_allocations.values())
        if abs(total_allocation - 1.0) > 0.01:  # Allow 1% tolerance
            return {'error': f'Allocations must sum to 1.0, got {total_allocation:.3f}'}
        
        # Update targets
        old_targets = self.target_allocations.copy()
        self.target_allocations.update(new_allocations)
        
        return {
            'success': True,
            'old_targets': old_targets,
            'new_targets': self.target_allocations,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _assess_portfolio_risk(self) -> Dict[str, Any]:
        """Assess overall portfolio risk"""
        try:
            analysis = await self._analyze_portfolio()
            if 'error' in analysis:
                return analysis
            
            # Calculate portfolio risk metrics
            positions = analysis['positions']
            
            # Concentration risk
            max_allocation = max(pos['allocation_percent'] for pos in positions)
            concentration_risk = 'high' if max_allocation > 0.3 else 'medium' if max_allocation > 0.2 else 'low'
            
            # Diversification score
            non_cash_positions = [pos for pos in positions if pos['symbol'] != 'CASH' and pos['allocation_percent'] > 0]
            diversification_score = len(non_cash_positions) / 10.0  # Normalize to 0-1
            
            # Cash allocation risk
            cash_allocation = next((pos['allocation_percent'] for pos in positions if pos['symbol'] == 'CASH'), 0)
            liquidity_risk = 'high' if cash_allocation < 0.05 else 'medium' if cash_allocation < 0.1 else 'low'
            
            # Overall risk score
            risk_factors = {
                'concentration': 0.8 if concentration_risk == 'high' else 0.5 if concentration_risk == 'medium' else 0.2,
                'diversification': 1.0 - diversification_score,
                'liquidity': 0.8 if liquidity_risk == 'high' else 0.5 if liquidity_risk == 'medium' else 0.2
            }
            
            overall_risk_score = np.mean(list(risk_factors.values()))
            risk_level = 'high' if overall_risk_score > 0.7 else 'medium' if overall_risk_score > 0.4 else 'low'
            
            return {
                'timestamp': datetime.now().isoformat(),
                'overall_risk_level': risk_level,
                'overall_risk_score': overall_risk_score,
                'risk_factors': {
                    'concentration_risk': concentration_risk,
                    'max_allocation': max_allocation,
                    'diversification_score': diversification_score,
                    'position_count': len(non_cash_positions),
                    'liquidity_risk': liquidity_risk,
                    'cash_allocation': cash_allocation
                },
                'recommendations': self._generate_risk_recommendations(risk_factors, positions)
            }
            
        except Exception as e:
            logger.error(f"Portfolio risk assessment failed: {e}")
            return {'error': str(e)}
    
    def _generate_risk_recommendations(self, risk_factors: Dict[str, float], positions: List[Dict]) -> List[str]:
        """Generate risk mitigation recommendations"""
        recommendations = []
        
        if risk_factors['concentration'] > 0.7:
            max_pos = max(positions, key=lambda p: p['allocation_percent'])
            recommendations.append(f"Reduce concentration in {max_pos['symbol']} (currently {max_pos['allocation_percent']:.1%})")
        
        if risk_factors['diversification'] > 0.7:
            recommendations.append("Increase portfolio diversification by adding more positions")
        
        if risk_factors['liquidity'] > 0.7:
            recommendations.append("Increase cash allocation for better liquidity")
        
        return recommendations
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status with portfolio-specific metrics"""
        base_status = super().get_status()
        
        # Add portfolio-specific metrics
        base_status['portfolio_metrics'] = {
            'mode': self.mode,
            'rebalancing_events': self.rebalancing_events,
            'optimization_suggestions': self.optimization_suggestions,
            'performance_analyses': self.performance_analyses,
            'portfolio_history_length': len(self.portfolio_history),
            'rebalancing_history_length': len(self.rebalancing_history),
            'target_allocations': self.target_allocations,
            'rebalance_threshold': self.rebalance_threshold,
            'max_position_size': self.max_position_size,
            'min_position_size': self.min_position_size
        }
        
        return base_status
