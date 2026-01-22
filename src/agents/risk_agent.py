"""
Risk Management Agent

Wraps the existing RiskManager with agent interface for dynamic resource management.
This agent is CRITICAL priority - it must always run to protect capital.
"""

import logging
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime

from .base_agent import BaseAgent, AgentPriority, AgentStatus, ResourceRequirements
from src.trading.risk import RiskManager, RiskMetrics

logger = logging.getLogger(__name__)


class RiskManagementAgent(BaseAgent):
    """
    Risk Management Agent - CRITICAL priority
    
    Responsibilities:
    - Validate all trading decisions against risk parameters
    - Calculate position sizes based on risk tolerance
    - Monitor portfolio risk metrics
    - Reject trades that exceed risk limits
    - Provide risk assessments for decisions
    """
    
    def __init__(self, mode: str = 'DEMO'):
        super().__init__(
            agent_id='risk_agent',
            name='Risk Management Agent',
            priority=AgentPriority.CRITICAL,  # Must always run
            resource_requirements=ResourceRequirements(
                min_cpu_percent=5.0,
                min_memory_mb=100.0,
                max_cpu_percent=15.0,
                max_memory_mb=300.0
            )
        )
        
        self.mode = mode
        self.risk_manager: Optional[RiskManager] = None
        self.risk_assessments_today = 0
        self.trades_rejected_today = 0
        self.trades_approved_today = 0
    
    async def initialize(self) -> bool:
        """Initialize the Risk Manager"""
        try:
            self.risk_manager = RiskManager(mode=self.mode)
            self.status = AgentStatus.IDLE
            logger.info(f"Risk Management Agent initialized in {self.mode} mode")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Risk Manager: {e}")
            self.status = AgentStatus.ERROR
            return False
    
    async def shutdown(self) -> bool:
        """Gracefully shutdown"""
        logger.info("Risk Management Agent shutting down")
        self.status = AgentStatus.STOPPED
        return True
    
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process risk-related tasks.
        
        Task types:
        - 'assess_risk': Assess risk for a trading decision
        - 'calculate_position_size': Calculate safe position size
        - 'validate_trade': Validate if trade meets risk criteria
        - 'get_risk_metrics': Get current portfolio risk metrics
        """
        task_type = task.get('type')
        
        if task_type == 'assess_risk':
            return await self._assess_risk(task)
        elif task_type == 'calculate_position_size':
            return await self._calculate_position_size(task)
        elif task_type == 'validate_trade':
            return await self._validate_trade(task)
        elif task_type == 'get_risk_metrics':
            return await self._get_risk_metrics(task)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    async def _assess_risk(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk for a trading decision"""
        if not self.risk_manager:
            raise RuntimeError("Risk Manager not initialized")
        
        symbol = task.get('symbol')
        action = task.get('action')  # 'buy' or 'sell'
        confidence = task.get('confidence', 0.5)
        price = task.get('price')
        
        self.risk_assessments_today += 1
        
        # Calculate risk metrics
        try:
            risk_metrics = self.risk_manager.calculate_position_size(
                price=price,
                confidence=confidence,
                signal_strength=confidence
            )
            
            # Determine if trade is acceptable
            risk_level = "low"
            if risk_metrics.position_size < 50:
                risk_level = "high"  # Very small position = high risk
            elif risk_metrics.position_size > 500:
                risk_level = "very_low"  # Large position = low risk
            else:
                risk_level = "medium"
            
            approved = risk_metrics.position_size > 0
            
            if approved:
                self.trades_approved_today += 1
            else:
                self.trades_rejected_today += 1
            
            return {
                'symbol': symbol,
                'action': action,
                'approved': approved,
                'risk_level': risk_level,
                'position_size': risk_metrics.position_size,
                'risk_metrics': {
                    'position_size': risk_metrics.position_size,
                    'max_loss': risk_metrics.max_loss,
                    'max_gain': risk_metrics.max_gain,
                    'risk_reward_ratio': risk_metrics.risk_reward_ratio,
                    'confidence_adjusted_size': confidence * risk_metrics.position_size
                },
                'recommendation': self._generate_recommendation(risk_metrics, confidence)
            }
        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            self.trades_rejected_today += 1
            return {
                'symbol': symbol,
                'action': action,
                'approved': False,
                'risk_level': 'error',
                'error': str(e)
            }
    
    async def _calculate_position_size(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate safe position size"""
        if not self.risk_manager:
            raise RuntimeError("Risk Manager not initialized")
        
        price = task.get('price')
        confidence = task.get('confidence', 0.5)
        
        risk_metrics = self.risk_manager.calculate_position_size(
            price=price,
            confidence=confidence,
            signal_strength=confidence
        )
        
        return {
            'position_size': risk_metrics.position_size,
            'max_loss': risk_metrics.max_loss,
            'max_gain': risk_metrics.max_gain,
            'risk_reward_ratio': risk_metrics.risk_reward_ratio
        }
    
    async def _validate_trade(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Validate if a trade meets risk criteria"""
        if not self.risk_manager:
            raise RuntimeError("Risk Manager not initialized")
        
        symbol = task.get('symbol')
        quantity = task.get('quantity')
        price = task.get('price')
        action = task.get('action')
        
        # Simple validation: check if quantity is reasonable
        max_allowed = self.risk_manager.calculate_position_size(
            price=price,
            confidence=1.0,  # Max confidence
            signal_strength=1.0
        ).position_size
        
        is_valid = quantity <= max_allowed
        
        if is_valid:
            self.trades_approved_today += 1
        else:
            self.trades_rejected_today += 1
        
        return {
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'action': action,
            'is_valid': is_valid,
            'max_allowed_quantity': max_allowed,
            'reason': 'Valid trade' if is_valid else f'Quantity exceeds max allowed ({max_allowed})'
        }
    
    async def _get_risk_metrics(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Get current portfolio risk metrics"""
        if not self.risk_manager:
            raise RuntimeError("Risk Manager not initialized")
        
        # Return current risk statistics
        return {
            'assessments_today': self.risk_assessments_today,
            'trades_approved': self.trades_approved_today,
            'trades_rejected': self.trades_rejected_today,
            'approval_rate': (
                self.trades_approved_today / self.risk_assessments_today 
                if self.risk_assessments_today > 0 else 0.0
            ),
            'rejection_rate': (
                self.trades_rejected_today / self.risk_assessments_today 
                if self.risk_assessments_today > 0 else 0.0
            )
        }
    
    def _generate_recommendation(self, risk_metrics: RiskMetrics, confidence: float) -> str:
        """Generate human-readable risk recommendation"""
        if risk_metrics.position_size == 0:
            return "REJECT: Position size is zero - trade does not meet risk criteria"
        
        if risk_metrics.risk_reward_ratio < 2.0:
            return f"CAUTION: Low risk/reward ratio ({risk_metrics.risk_reward_ratio:.2f})"
        
        if confidence < 0.6:
            return f"CAUTION: Low confidence ({confidence:.2f}) - consider smaller position"
        
        if risk_metrics.position_size < 100:
            return f"PROCEED: Small position ({risk_metrics.position_size}) - limited risk"
        
        return f"PROCEED: Good risk profile (R/R: {risk_metrics.risk_reward_ratio:.2f}, Confidence: {confidence:.2f})"
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status with risk-specific metrics"""
        base_status = super().get_status()
        
        # Add risk-specific metrics
        base_status['risk_metrics'] = {
            'assessments_today': self.risk_assessments_today,
            'trades_approved': self.trades_approved_today,
            'trades_rejected': self.trades_rejected_today,
            'approval_rate': (
                self.trades_approved_today / self.risk_assessments_today 
                if self.risk_assessments_today > 0 else 0.0
            ),
            'mode': self.mode
        }
        
        return base_status


