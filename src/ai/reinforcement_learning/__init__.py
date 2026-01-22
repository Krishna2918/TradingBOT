"""
Reinforcement Learning Models for Adaptive Trading Strategies

This module contains reinforcement learning algorithms for creating
adaptive trading strategies that learn from market feedback.
"""

from .trading_environment import TradingEnvironment, MarketSimulator
from .dqn_agent import DQNAgent, DQNTrainer
from .ppo_agent import PPOAgent, PPOTrainer
from .a3c_agent import A3CAgent, A3CTrainer
from .sac_agent import SACAgent, SACTrainer
from .reward_functions import RewardFunction, SharpeReward, RiskAdjustedReward
from .rl_manager import ReinforcementLearningManager

# Global RL manager instance
_rl_manager = None

def get_rl_trainer():
    """Get the global RL trainer instance."""
    global _rl_manager
    if _rl_manager is None:
        _rl_manager = ReinforcementLearningManager()
    return _rl_manager

__all__ = [
    'TradingEnvironment',
    'MarketSimulator',
    'DQNAgent',
    'DQNTrainer',
    'PPOAgent',
    'PPOTrainer',
    'A3CAgent',
    'A3CTrainer',
    'SACAgent',
    'SACTrainer',
    'RewardFunction',
    'SharpeReward',
    'RiskAdjustedReward',
    'ReinforcementLearningManager',
    'get_rl_trainer'
]

