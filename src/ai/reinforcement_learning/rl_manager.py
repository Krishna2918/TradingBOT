"""
Reinforcement Learning Manager for Trading

This module provides a unified interface for managing, training,
and deploying reinforcement learning agents in the trading system.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from pathlib import Path

from .trading_environment import TradingEnvironment, MarketSimulator
from .dqn_agent import DQNAgent, DQNTrainer
from .ppo_agent import PPOAgent, PPOTrainer
from .a3c_agent import A3CAgent, A3CTrainer
from .sac_agent import SACAgent, SACTrainer
from .reward_functions import (
    RewardFunction, SharpeReward, RiskAdjustedReward, 
    ProfitLossReward, TransactionCostReward, MomentumReward,
    CompositeReward, AdaptiveReward
)

logger = logging.getLogger(__name__)

class ReinforcementLearningManager:
    """
    Manager for reinforcement learning agents in the trading system.
    
    This class provides a unified interface for managing multiple
    RL agents, including training, inference, and deployment.
    """
    
    def __init__(
        self,
        model_dir: str = "models/reinforcement_learning",
        config_file: Optional[str] = None
    ):
        """
        Initialize Reinforcement Learning Manager.
        
        Args:
            model_dir: Directory for storing models
            config_file: Path to configuration file
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.agents = {}
        self.agent_configs = {}
        self.agent_metadata = {}
        self.environments = {}
        self.reward_functions = {}
        
        # Load configuration
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)
        else:
            self._load_default_config()
        
        logger.info(f"Initialized Reinforcement Learning Manager: {model_dir}")
    
    def _load_default_config(self) -> None:
        """Load default agent configurations."""
        self.agent_configs = {
            'dqn': {
                'class': DQNAgent,
                'trainer_class': DQNTrainer,
                'params': {
                    'state_size': 100,  # Will be updated based on environment
                    'action_size': 3,
                    'hidden_sizes': [128, 64, 32],
                    'learning_rate': 0.001,
                    'gamma': 0.95,
                    'epsilon': 1.0,
                    'epsilon_min': 0.01,
                    'epsilon_decay': 0.995,
                    'buffer_size': 10000,
                    'batch_size': 32,
                    'target_update_freq': 100
                }
            },
            'ppo': {
                'class': PPOAgent,
                'trainer_class': PPOTrainer,
                'params': {
                    'state_size': 100,  # Will be updated based on environment
                    'action_size': 3,
                    'hidden_sizes': [128, 64],
                    'learning_rate': 3e-4,
                    'gamma': 0.99,
                    'lambda_gae': 0.95,
                    'clip_ratio': 0.2,
                    'value_coef': 0.5,
                    'entropy_coef': 0.01,
                    'max_grad_norm': 0.5,
                    'buffer_size': 2048,
                    'batch_size': 64,
                    'n_epochs': 10
                }
            },
            'a3c': {
                'class': A3CAgent,
                'trainer_class': A3CTrainer,
                'params': {
                    'state_size': 100,  # Will be updated based on environment
                    'action_size': 3,
                    'hidden_sizes': [128, 64],
                    'learning_rate': 3e-4,
                    'gamma': 0.99,
                    'beta': 0.01,
                    'max_grad_norm': 0.5
                }
            },
            'sac': {
                'class': SACAgent,
                'trainer_class': SACTrainer,
                'params': {
                    'state_size': 100,  # Will be updated based on environment
                    'action_size': 3,
                    'hidden_sizes': [128, 64],
                    'learning_rate': 3e-4,
                    'gamma': 0.99,
                    'tau': 0.005,
                    'alpha': 0.2,
                    'buffer_size': 100000,
                    'batch_size': 256
                }
            }
        }
        
        # Default reward functions
        self.reward_functions = {
            'sharpe': SharpeReward(),
            'risk_adjusted': RiskAdjustedReward(),
            'profit_loss': ProfitLossReward(),
            'transaction_cost': TransactionCostReward(),
            'momentum': MomentumReward()
        }
    
    def load_config(self, config_file: str) -> None:
        """Load agent configurations from file."""
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        self.agent_configs = config_data.get('agent_configs', {})
        logger.info(f"Loaded configuration from {config_file}")
    
    def save_config(self, config_file: str) -> None:
        """Save agent configurations to file."""
        config_data = {
            'agent_configs': self.agent_configs,
            'metadata': self.agent_metadata,
            'last_updated': datetime.now().isoformat()
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"Saved configuration to {config_file}")
    
    def create_environment(
        self,
        data: pd.DataFrame,
        env_name: str = "default",
        **kwargs
    ) -> TradingEnvironment:
        """
        Create a trading environment.
        
        Args:
            data: Market data DataFrame
            env_name: Name for the environment
            **kwargs: Additional environment parameters
            
        Returns:
            Trading environment instance
        """
        environment = TradingEnvironment(data, **kwargs)
        self.environments[env_name] = environment
        
        logger.info(f"Created environment: {env_name}")
        
        return environment
    
    def create_agent(
        self,
        agent_name: str,
        environment: TradingEnvironment,
        **kwargs
    ) -> Any:
        """
        Create a new RL agent.
        
        Args:
            agent_name: Name of the agent to create
            environment: Trading environment
            **kwargs: Additional parameters for agent creation
            
        Returns:
            Agent instance
        """
        if agent_name not in self.agent_configs:
            raise ValueError(f"Unknown agent type: {agent_name}")
        
        config = self.agent_configs[agent_name]
        agent_class = config['class']
        params = config['params'].copy()
        
        # Update state size based on environment
        params['state_size'] = environment.observation_space.shape[0]
        
        # Update parameters with kwargs
        params.update(kwargs)
        
        # Create agent instance
        agent = agent_class(**params)
        
        # Store agent
        self.agents[agent_name] = agent
        
        # Store metadata
        self.agent_metadata[agent_name] = {
            'created_at': datetime.now().isoformat(),
            'agent_type': agent_name,
            'parameters': params,
            'is_trained': False,
            'environment': environment
        }
        
        logger.info(f"Created agent: {agent_name}")
        
        return agent
    
    def get_agent(self, agent_name: str) -> Any:
        """
        Get an existing agent instance.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Agent instance
        """
        if agent_name not in self.agents:
            raise ValueError(f"Agent not found: {agent_name}")
        
        return self.agents[agent_name]
    
    def train_agent(
        self,
        agent_name: str,
        environment: TradingEnvironment,
        n_episodes: int = 1000,
        **kwargs
    ) -> List[Dict[str, float]]:
        """
        Train an RL agent.
        
        Args:
            agent_name: Name of the agent to train
            environment: Trading environment
            n_episodes: Number of episodes to train
            **kwargs: Additional training parameters
            
        Returns:
            Training statistics
        """
        if agent_name not in self.agents:
            raise ValueError(f"Agent not found: {agent_name}")
        
        agent = self.agents[agent_name]
        config = self.agent_configs[agent_name]
        trainer_class = config['trainer_class']
        
        # Create trainer
        trainer = trainer_class(agent, environment, **kwargs)
        
        logger.info(f"Training agent: {agent_name} for {n_episodes} episodes")
        
        # Train agent
        training_stats = trainer.train(n_episodes)
        
        # Update metadata
        self.agent_metadata[agent_name]['is_trained'] = True
        self.agent_metadata[agent_name]['trained_at'] = datetime.now().isoformat()
        self.agent_metadata[agent_name]['training_episodes'] = n_episodes
        self.agent_metadata[agent_name]['training_stats'] = training_stats
        
        logger.info(f"Agent {agent_name} training completed")
        
        return training_stats
    
    def evaluate_agent(
        self,
        agent_name: str,
        environment: TradingEnvironment,
        n_episodes: int = 10
    ) -> Dict[str, float]:
        """
        Evaluate an RL agent's performance.
        
        Args:
            agent_name: Name of the agent to evaluate
            environment: Trading environment
            n_episodes: Number of episodes to evaluate
            
        Returns:
            Evaluation statistics
        """
        if agent_name not in self.agents:
            raise ValueError(f"Agent not found: {agent_name}")
        
        agent = self.agents[agent_name]
        config = self.agent_configs[agent_name]
        trainer_class = config['trainer_class']
        
        # Create trainer
        trainer = trainer_class(agent, environment)
        
        logger.info(f"Evaluating agent: {agent_name} for {n_episodes} episodes")
        
        # Evaluate agent
        evaluation_stats = trainer.evaluate(n_episodes)
        
        # Update metadata
        self.agent_metadata[agent_name]['evaluation_stats'] = evaluation_stats
        self.agent_metadata[agent_name]['evaluated_at'] = datetime.now().isoformat()
        
        logger.info(f"Agent {agent_name} evaluation completed")
        
        return evaluation_stats
    
    def predict(
        self,
        agent_name: str,
        state: np.ndarray
    ) -> int:
        """
        Make a prediction using a trained agent.
        
        Args:
            agent_name: Name of the agent
            state: Current state
            
        Returns:
            Predicted action
        """
        if agent_name not in self.agents:
            raise ValueError(f"Agent not found: {agent_name}")
        
        agent = self.agents[agent_name]
        
        if not self.agent_metadata[agent_name].get('is_trained', False):
            raise ValueError(f"Agent {agent_name} is not trained")
        
        # Make prediction
        action = agent.act(state, training=False)
        
        return action
    
    def get_agent_performance(self, agent_name: str) -> Dict[str, Any]:
        """
        Get performance metrics for an agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Performance metrics
        """
        if agent_name not in self.agent_metadata:
            raise ValueError(f"Agent metadata not found: {agent_name}")
        
        metadata = self.agent_metadata[agent_name]
        
        performance = {
            'agent_name': agent_name,
            'agent_type': metadata['agent_type'],
            'is_trained': metadata['is_trained'],
            'created_at': metadata['created_at'],
            'parameters': metadata['parameters']
        }
        
        if 'trained_at' in metadata:
            performance['trained_at'] = metadata['trained_at']
            performance['training_episodes'] = metadata.get('training_episodes', 0)
        
        if 'evaluation_stats' in metadata:
            performance['evaluation_stats'] = metadata['evaluation_stats']
        
        if 'training_stats' in metadata:
            training_stats = metadata['training_stats']
            if training_stats:
                performance['final_training_reward'] = training_stats[-1].get('episode_reward', 0)
                performance['avg_training_reward'] = np.mean([s.get('episode_reward', 0) for s in training_stats])
        
        return performance
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """
        List all available agents.
        
        Returns:
            List of agent information
        """
        agents_info = []
        
        for agent_name in self.agent_configs.keys():
            info = {
                'name': agent_name,
                'class': self.agent_configs[agent_name]['class'].__name__,
                'parameters': self.agent_configs[agent_name]['params'],
                'is_created': agent_name in self.agents,
                'is_trained': self.agent_metadata.get(agent_name, {}).get('is_trained', False)
            }
            
            if agent_name in self.agent_metadata:
                info.update({
                    'created_at': self.agent_metadata[agent_name].get('created_at'),
                    'trained_at': self.agent_metadata[agent_name].get('trained_at')
                })
            
            agents_info.append(info)
        
        return agents_info
    
    def save_agent(self, agent_name: str) -> None:
        """
        Save a trained agent to disk.
        
        Args:
            agent_name: Name of the agent to save
        """
        if agent_name not in self.agents:
            raise ValueError(f"Agent not found: {agent_name}")
        
        agent = self.agents[agent_name]
        agent_path = self.model_dir / agent_name
        
        # Create agent directory
        agent_path.mkdir(exist_ok=True)
        
        # Save agent
        model_file = agent_path / f"{agent_name}.pth"
        agent.save(str(model_file))
        
        # Save metadata
        metadata_file = agent_path / f"{agent_name}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.agent_metadata[agent_name], f, indent=2)
        
        logger.info(f"Saved agent {agent_name} to {agent_path}")
    
    def load_agent(self, agent_name: str) -> None:
        """
        Load a trained agent from disk.
        
        Args:
            agent_name: Name of the agent to load
        """
        if agent_name not in self.agent_configs:
            raise ValueError(f"Unknown agent type: {agent_name}")
        
        agent_path = self.model_dir / agent_name
        
        if not agent_path.exists():
            raise ValueError(f"Agent directory not found: {agent_path}")
        
        # Create agent instance
        config = self.agent_configs[agent_name]
        agent_class = config['class']
        params = config['params'].copy()
        
        agent = agent_class(**params)
        
        # Load trained agent
        model_file = agent_path / f"{agent_name}.pth"
        agent.load(str(model_file))
        
        # Load metadata
        metadata_file = agent_path / f"{agent_name}_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self.agent_metadata[agent_name] = json.load(f)
        
        # Store agent
        self.agents[agent_name] = agent
        
        logger.info(f"Loaded agent {agent_name} from {agent_path}")
    
    def save_all_agents(self) -> None:
        """Save all trained agents to disk."""
        for agent_name in self.agents.keys():
            if self.agent_metadata.get(agent_name, {}).get('is_trained', False):
                self.save_agent(agent_name)
        
        logger.info("Saved all trained agents")
    
    def load_all_agents(self) -> None:
        """Load all available agents from disk."""
        for agent_name in self.agent_configs.keys():
            agent_path = self.model_dir / agent_name
            if agent_path.exists():
                try:
                    self.load_agent(agent_name)
                except Exception as e:
                    logger.warning(f"Failed to load agent {agent_name}: {e}")
        
        logger.info("Loaded all available agents")
    
    def delete_agent(self, agent_name: str) -> None:
        """
        Delete an agent and its files.
        
        Args:
            agent_name: Name of the agent to delete
        """
        # Remove from memory
        if agent_name in self.agents:
            del self.agents[agent_name]
        
        if agent_name in self.agent_metadata:
            del self.agent_metadata[agent_name]
        
        # Remove from disk
        agent_path = self.model_dir / agent_name
        if agent_path.exists():
            import shutil
            shutil.rmtree(agent_path)
        
        logger.info(f"Deleted agent: {agent_name}")
    
    def get_agent_status(self) -> Dict[str, Any]:
        """
        Get status of all agents.
        
        Returns:
            Dictionary with agent status information
        """
        status = {
            'total_agents': len(self.agent_configs),
            'created_agents': len(self.agents),
            'trained_agents': sum(1 for m in self.agent_metadata.values() if m.get('is_trained', False)),
            'agents': {}
        }
        
        for agent_name in self.agent_configs.keys():
            agent_status = {
                'is_created': agent_name in self.agents,
                'is_trained': self.agent_metadata.get(agent_name, {}).get('is_trained', False),
                'agent_type': self.agent_configs[agent_name]['class'].__name__
            }
            
            if agent_name in self.agent_metadata:
                agent_status.update({
                    'created_at': self.agent_metadata[agent_name].get('created_at'),
                    'trained_at': self.agent_metadata[agent_name].get('trained_at')
                })
            
            status['agents'][agent_name] = agent_status
        
        return status
    
    def create_ensemble_prediction(
        self,
        state: np.ndarray,
        agent_names: Optional[List[str]] = None,
        weights: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Create ensemble prediction using multiple RL agents.
        
        Args:
            state: Current state
            agent_names: List of agent names to use (if None, use all trained agents)
            weights: Weights for each agent (if None, use equal weights)
            
        Returns:
            Ensemble prediction results
        """
        if agent_names is None:
            # Use all trained agents
            agent_names = [
                name for name, metadata in self.agent_metadata.items()
                if metadata.get('is_trained', False)
            ]
        
        if not agent_names:
            raise ValueError("No trained agents available for ensemble prediction")
        
        if weights is None:
            # Use equal weights
            weights = [1.0 / len(agent_names)] * len(agent_names)
        
        if len(weights) != len(agent_names):
            raise ValueError("Number of weights must match number of agents")
        
        # Get predictions from all agents
        predictions = []
        confidences = []
        
        for agent_name in agent_names:
            try:
                action = self.predict(agent_name, state)
                predictions.append(action)
                confidences.append(0.8)  # Default confidence for RL agents
            except Exception as e:
                logger.warning(f"Failed to get prediction from {agent_name}: {e}")
        
        if not predictions:
            raise ValueError("No valid predictions obtained from agents")
        
        # Calculate weighted ensemble prediction
        # For discrete actions, use majority voting
        action_counts = {}
        for action, weight in zip(predictions, weights):
            if action not in action_counts:
                action_counts[action] = 0
            action_counts[action] += weight
        
        ensemble_action = max(action_counts, key=action_counts.get)
        ensemble_confidence = max(action_counts.values()) / sum(action_counts.values())
        
        return {
            'ensemble_action': ensemble_action,
            'ensemble_confidence': ensemble_confidence,
            'individual_predictions': dict(zip(agent_names, predictions)),
            'individual_confidences': dict(zip(agent_names, confidences)),
            'weights': dict(zip(agent_names, weights)),
            'agent_count': len(agent_names),
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_synthetic_data(
        self,
        n_periods: int = 1000,
        **kwargs
    ) -> pd.DataFrame:
        """
        Generate synthetic market data for training.
        
        Args:
            n_periods: Number of periods to generate
            **kwargs: Additional parameters for data generation
            
        Returns:
            Generated market data
        """
        simulator = MarketSimulator(n_periods=n_periods, **kwargs)
        return simulator.generate_data()
    
    def get_reward_function(self, name: str) -> RewardFunction:
        """
        Get a reward function by name.
        
        Args:
            name: Name of the reward function
            
        Returns:
            Reward function instance
        """
        if name not in self.reward_functions:
            raise ValueError(f"Reward function not found: {name}")
        
        return self.reward_functions[name]
    
    def add_reward_function(self, name: str, reward_function: RewardFunction) -> None:
        """
        Add a custom reward function.
        
        Args:
            name: Name for the reward function
            reward_function: Reward function instance
        """
        self.reward_functions[name] = reward_function
        logger.info(f"Added reward function: {name}")
    
    def create_composite_reward(
        self,
        reward_names: List[str],
        weights: Optional[List[float]] = None,
        name: str = "composite_reward"
    ) -> CompositeReward:
        """
        Create a composite reward function.
        
        Args:
            reward_names: List of reward function names
            weights: Weights for each reward function
            name: Name for the composite reward function
            
        Returns:
            Composite reward function instance
        """
        reward_functions = [self.get_reward_function(name) for name in reward_names]
        composite_reward = CompositeReward(reward_functions, weights, name)
        
        self.reward_functions[name] = composite_reward
        
        logger.info(f"Created composite reward function: {name}")
        
        return composite_reward

