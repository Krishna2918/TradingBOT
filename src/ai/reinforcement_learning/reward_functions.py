"""
Reward Functions for Reinforcement Learning Trading

This module implements various reward functions for training
reinforcement learning agents in trading environments.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class RewardFunction(ABC):
    """
    Abstract base class for reward functions.
    """
    
    def __init__(self, name: str = "base_reward"):
        """
        Initialize reward function.
        
        Args:
            name: Name of the reward function
        """
        self.name = name
        self.history = []
        
        logger.info(f"Initialized reward function: {name}")
    
    @abstractmethod
    def calculate(self, **kwargs) -> float:
        """
        Calculate reward based on current state and action.
        
        Returns:
            Reward value
        """
        pass
    
    def update_history(self, reward: float, **kwargs) -> None:
        """
        Update reward history.
        
        Args:
            reward: Reward value
            **kwargs: Additional context
        """
        self.history.append({
            'reward': reward,
            'timestamp': datetime.now(),
            **kwargs
        })
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Get reward statistics.
        
        Returns:
            Dictionary with reward statistics
        """
        if not self.history:
            return {}
        
        rewards = [h['reward'] for h in self.history]
        
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'total_rewards': len(rewards)
        }

class SharpeReward(RewardFunction):
    """
    Sharpe ratio-based reward function.
    """
    
    def __init__(
        self,
        risk_free_rate: float = 0.02,
        lookback_window: int = 20,
        name: str = "sharpe_reward"
    ):
        """
        Initialize Sharpe reward function.
        
        Args:
            risk_free_rate: Risk-free rate (annual)
            lookback_window: Window for calculating Sharpe ratio
            name: Name of the reward function
        """
        super().__init__(name)
        self.risk_free_rate = risk_free_rate
        self.lookback_window = lookback_window
        self.returns_history = []
        
        logger.info(f"Initialized Sharpe Reward: risk_free_rate={risk_free_rate}, window={lookback_window}")
    
    def calculate(
        self,
        portfolio_value: float,
        previous_portfolio_value: float,
        **kwargs
    ) -> float:
        """
        Calculate Sharpe ratio-based reward.
        
        Args:
            portfolio_value: Current portfolio value
            previous_portfolio_value: Previous portfolio value
            
        Returns:
            Sharpe ratio-based reward
        """
        # Calculate return
        if previous_portfolio_value > 0:
            return_rate = (portfolio_value - previous_portfolio_value) / previous_portfolio_value
        else:
            return_rate = 0.0
        
        # Add to returns history
        self.returns_history.append(return_rate)
        
        # Keep only recent returns
        if len(self.returns_history) > self.lookback_window:
            self.returns_history = self.returns_history[-self.lookback_window:]
        
        # Calculate Sharpe ratio if we have enough data
        if len(self.returns_history) >= 2:
            returns_array = np.array(self.returns_history)
            excess_returns = returns_array - (self.risk_free_rate / 252)  # Daily risk-free rate
            
            if np.std(excess_returns) > 0:
                sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0
        
        # Scale reward
        reward = sharpe_ratio * 0.1  # Scale down for stability
        
        # Update history
        self.update_history(reward, portfolio_value=portfolio_value, sharpe_ratio=sharpe_ratio)
        
        return reward

class RiskAdjustedReward(RewardFunction):
    """
    Risk-adjusted reward function considering multiple risk metrics.
    """
    
    def __init__(
        self,
        return_weight: float = 1.0,
        volatility_weight: float = -0.5,
        drawdown_weight: float = -1.0,
        sharpe_weight: float = 0.5,
        lookback_window: int = 20,
        name: str = "risk_adjusted_reward"
    ):
        """
        Initialize risk-adjusted reward function.
        
        Args:
            return_weight: Weight for return component
            volatility_weight: Weight for volatility component
            drawdown_weight: Weight for drawdown component
            sharpe_weight: Weight for Sharpe ratio component
            lookback_window: Window for calculating metrics
            name: Name of the reward function
        """
        super().__init__(name)
        self.return_weight = return_weight
        self.volatility_weight = volatility_weight
        self.drawdown_weight = drawdown_weight
        self.sharpe_weight = sharpe_weight
        self.lookback_window = lookback_window
        
        self.returns_history = []
        self.portfolio_values = []
        
        logger.info(f"Initialized Risk Adjusted Reward: return={return_weight}, "
                   f"volatility={volatility_weight}, drawdown={drawdown_weight}, "
                   f"sharpe={sharpe_weight}")
    
    def calculate(
        self,
        portfolio_value: float,
        previous_portfolio_value: float,
        **kwargs
    ) -> float:
        """
        Calculate risk-adjusted reward.
        
        Args:
            portfolio_value: Current portfolio value
            previous_portfolio_value: Previous portfolio value
            
        Returns:
            Risk-adjusted reward
        """
        # Calculate return
        if previous_portfolio_value > 0:
            return_rate = (portfolio_value - previous_portfolio_value) / previous_portfolio_value
        else:
            return_rate = 0.0
        
        # Add to history
        self.returns_history.append(return_rate)
        self.portfolio_values.append(portfolio_value)
        
        # Keep only recent data
        if len(self.returns_history) > self.lookback_window:
            self.returns_history = self.returns_history[-self.lookback_window:]
            self.portfolio_values = self.portfolio_values[-self.lookback_window:]
        
        # Calculate components
        reward = 0.0
        
        # Return component
        if self.return_weight != 0:
            reward += self.return_weight * return_rate
        
        # Volatility component
        if self.volatility_weight != 0 and len(self.returns_history) >= 2:
            volatility = np.std(self.returns_history)
            reward += self.volatility_weight * volatility
        
        # Drawdown component
        if self.drawdown_weight != 0 and len(self.portfolio_values) >= 2:
            peak = np.max(self.portfolio_values)
            current_drawdown = (peak - portfolio_value) / peak if peak > 0 else 0
            reward += self.drawdown_weight * current_drawdown
        
        # Sharpe ratio component
        if self.sharpe_weight != 0 and len(self.returns_history) >= 2:
            if np.std(self.returns_history) > 0:
                sharpe_ratio = np.mean(self.returns_history) / np.std(self.returns_history) * np.sqrt(252)
            else:
                sharpe_ratio = 0.0
            reward += self.sharpe_weight * sharpe_ratio * 0.1  # Scale down
        
        # Update history
        self.update_history(reward, portfolio_value=portfolio_value, return_rate=return_rate)
        
        return reward

class ProfitLossReward(RewardFunction):
    """
    Simple profit/loss reward function.
    """
    
    def __init__(
        self,
        scale_factor: float = 1.0,
        name: str = "profit_loss_reward"
    ):
        """
        Initialize profit/loss reward function.
        
        Args:
            scale_factor: Scaling factor for rewards
            name: Name of the reward function
        """
        super().__init__(name)
        self.scale_factor = scale_factor
        
        logger.info(f"Initialized Profit/Loss Reward: scale_factor={scale_factor}")
    
    def calculate(
        self,
        portfolio_value: float,
        previous_portfolio_value: float,
        **kwargs
    ) -> float:
        """
        Calculate profit/loss reward.
        
        Args:
            portfolio_value: Current portfolio value
            previous_portfolio_value: Previous portfolio value
            
        Returns:
            Profit/loss reward
        """
        # Calculate P&L
        pnl = portfolio_value - previous_portfolio_value
        
        # Scale reward
        reward = pnl * self.scale_factor
        
        # Update history
        self.update_history(reward, portfolio_value=portfolio_value, pnl=pnl)
        
        return reward

class TransactionCostReward(RewardFunction):
    """
    Reward function that considers transaction costs.
    """
    
    def __init__(
        self,
        transaction_cost_rate: float = 0.001,
        base_reward_weight: float = 1.0,
        cost_penalty_weight: float = -1.0,
        name: str = "transaction_cost_reward"
    ):
        """
        Initialize transaction cost reward function.
        
        Args:
            transaction_cost_rate: Transaction cost rate
            base_reward_weight: Weight for base reward
            cost_penalty_weight: Weight for cost penalty
            name: Name of the reward function
        """
        super().__init__(name)
        self.transaction_cost_rate = transaction_cost_rate
        self.base_reward_weight = base_reward_weight
        self.cost_penalty_weight = cost_penalty_weight
        
        logger.info(f"Initialized Transaction Cost Reward: cost_rate={transaction_cost_rate}, "
                   f"base_weight={base_reward_weight}, penalty_weight={cost_penalty_weight}")
    
    def calculate(
        self,
        portfolio_value: float,
        previous_portfolio_value: float,
        transaction_cost: float = 0.0,
        base_reward: float = 0.0,
        **kwargs
    ) -> float:
        """
        Calculate transaction cost-adjusted reward.
        
        Args:
            portfolio_value: Current portfolio value
            previous_portfolio_value: Previous portfolio value
            transaction_cost: Transaction cost incurred
            base_reward: Base reward from other sources
            
        Returns:
            Transaction cost-adjusted reward
        """
        # Calculate base reward component
        base_component = self.base_reward_weight * base_reward
        
        # Calculate cost penalty component
        cost_penalty = self.cost_penalty_weight * transaction_cost
        
        # Total reward
        reward = base_component + cost_penalty
        
        # Update history
        self.update_history(reward, portfolio_value=portfolio_value, 
                          transaction_cost=transaction_cost, base_reward=base_reward)
        
        return reward

class MomentumReward(RewardFunction):
    """
    Momentum-based reward function.
    """
    
    def __init__(
        self,
        momentum_window: int = 5,
        momentum_weight: float = 0.5,
        return_weight: float = 0.5,
        name: str = "momentum_reward"
    ):
        """
        Initialize momentum reward function.
        
        Args:
            momentum_window: Window for momentum calculation
            momentum_weight: Weight for momentum component
            return_weight: Weight for return component
            name: Name of the reward function
        """
        super().__init__(name)
        self.momentum_window = momentum_window
        self.momentum_weight = momentum_weight
        self.return_weight = return_weight
        
        self.returns_history = []
        
        logger.info(f"Initialized Momentum Reward: window={momentum_window}, "
                   f"momentum_weight={momentum_weight}, return_weight={return_weight}")
    
    def calculate(
        self,
        portfolio_value: float,
        previous_portfolio_value: float,
        **kwargs
    ) -> float:
        """
        Calculate momentum-based reward.
        
        Args:
            portfolio_value: Current portfolio value
            previous_portfolio_value: Previous portfolio value
            
        Returns:
            Momentum-based reward
        """
        # Calculate return
        if previous_portfolio_value > 0:
            return_rate = (portfolio_value - previous_portfolio_value) / previous_portfolio_value
        else:
            return_rate = 0.0
        
        # Add to history
        self.returns_history.append(return_rate)
        
        # Keep only recent returns
        if len(self.returns_history) > self.momentum_window:
            self.returns_history = self.returns_history[-self.momentum_window:]
        
        # Calculate momentum
        if len(self.returns_history) >= self.momentum_window:
            momentum = np.mean(self.returns_history[-self.momentum_window:])
        else:
            momentum = return_rate
        
        # Calculate reward
        return_component = self.return_weight * return_rate
        momentum_component = self.momentum_weight * momentum
        
        reward = return_component + momentum_component
        
        # Update history
        self.update_history(reward, portfolio_value=portfolio_value, 
                          return_rate=return_rate, momentum=momentum)
        
        return reward

class CompositeReward(RewardFunction):
    """
    Composite reward function combining multiple reward functions.
    """
    
    def __init__(
        self,
        reward_functions: List[RewardFunction],
        weights: Optional[List[float]] = None,
        name: str = "composite_reward"
    ):
        """
        Initialize composite reward function.
        
        Args:
            reward_functions: List of reward functions to combine
            weights: Weights for each reward function
            name: Name of the reward function
        """
        super().__init__(name)
        self.reward_functions = reward_functions
        
        if weights is None:
            self.weights = [1.0 / len(reward_functions)] * len(reward_functions)
        else:
            self.weights = weights
        
        if len(self.weights) != len(reward_functions):
            raise ValueError("Number of weights must match number of reward functions")
        
        logger.info(f"Initialized Composite Reward with {len(reward_functions)} components")
    
    def calculate(self, **kwargs) -> float:
        """
        Calculate composite reward.
        
        Args:
            **kwargs: Arguments to pass to reward functions
            
        Returns:
            Composite reward
        """
        total_reward = 0.0
        component_rewards = {}
        
        for i, (reward_func, weight) in enumerate(zip(self.reward_functions, self.weights)):
            try:
                component_reward = reward_func.calculate(**kwargs)
                total_reward += weight * component_reward
                component_rewards[f'component_{i}_{reward_func.name}'] = component_reward
            except Exception as e:
                logger.warning(f"Error calculating reward from {reward_func.name}: {e}")
                component_rewards[f'component_{i}_{reward_func.name}'] = 0.0
        
        # Update history
        self.update_history(total_reward, **component_rewards)
        
        return total_reward
    
    def get_component_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for each component reward function.
        
        Returns:
            Dictionary with statistics for each component
        """
        stats = {}
        for reward_func in self.reward_functions:
            stats[reward_func.name] = reward_func.get_statistics()
        return stats

class AdaptiveReward(RewardFunction):
    """
    Adaptive reward function that adjusts based on performance.
    """
    
    def __init__(
        self,
        base_reward_function: RewardFunction,
        adaptation_window: int = 100,
        performance_threshold: float = 0.1,
        adaptation_rate: float = 0.1,
        name: str = "adaptive_reward"
    ):
        """
        Initialize adaptive reward function.
        
        Args:
            base_reward_function: Base reward function to adapt
            adaptation_window: Window for performance evaluation
            performance_threshold: Threshold for performance evaluation
            adaptation_rate: Rate of adaptation
            name: Name of the reward function
        """
        super().__init__(name)
        self.base_reward_function = base_reward_function
        self.adaptation_window = adaptation_window
        self.performance_threshold = performance_threshold
        self.adaptation_rate = adaptation_rate
        
        self.adaptation_factor = 1.0
        self.performance_history = []
        
        logger.info(f"Initialized Adaptive Reward: window={adaptation_window}, "
                   f"threshold={performance_threshold}, rate={adaptation_rate}")
    
    def calculate(self, **kwargs) -> float:
        """
        Calculate adaptive reward.
        
        Args:
            **kwargs: Arguments to pass to base reward function
            
        Returns:
            Adaptive reward
        """
        # Calculate base reward
        base_reward = self.base_reward_function.calculate(**kwargs)
        
        # Apply adaptation factor
        adapted_reward = base_reward * self.adaptation_factor
        
        # Update performance history
        self.performance_history.append(base_reward)
        if len(self.performance_history) > self.adaptation_window:
            self.performance_history = self.performance_history[-self.adaptation_window:]
        
        # Adapt based on performance
        if len(self.performance_history) >= self.adaptation_window:
            avg_performance = np.mean(self.performance_history)
            
            if avg_performance > self.performance_threshold:
                # Good performance, increase adaptation factor
                self.adaptation_factor = min(2.0, self.adaptation_factor + self.adaptation_rate)
            elif avg_performance < -self.performance_threshold:
                # Poor performance, decrease adaptation factor
                self.adaptation_factor = max(0.1, self.adaptation_factor - self.adaptation_rate)
        
        # Update history
        self.update_history(adapted_reward, base_reward=base_reward, 
                          adaptation_factor=self.adaptation_factor)
        
        return adapted_reward
    
    def get_adaptation_info(self) -> Dict[str, Any]:
        """
        Get adaptation information.
        
        Returns:
            Dictionary with adaptation information
        """
        return {
            'adaptation_factor': self.adaptation_factor,
            'performance_history_length': len(self.performance_history),
            'avg_performance': np.mean(self.performance_history) if self.performance_history else 0,
            'base_reward_stats': self.base_reward_function.get_statistics()
        }

