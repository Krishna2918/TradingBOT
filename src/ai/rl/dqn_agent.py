"""
DQN (Deep Q-Network) Agent for Trading

Uses Stable-Baselines3 DQN for learning optimal trading policies
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import gymnasium as gym

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from .trading_environment import TradingEnvironment

logger = logging.getLogger(__name__)

class DQNTradingAgent:
    """
    DQN Agent for trading
    
    Advantages of DQN:
    - Works well with discrete action spaces
    - Experience replay for sample efficiency
    - Target network for stability
    - Good for environments with clear actions
    """
    
    def __init__(
        self,
        model_path: str = "models/dqn_trading_agent",
        learning_rate: float = 0.0001,
        buffer_size: int = 100000,
        learning_starts: int = 1000,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: int = 4,
        gradient_steps: int = 1,
        target_update_interval: int = 1000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05
    ):
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.target_update_interval = target_update_interval
        self.exploration_fraction = exploration_fraction
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        
        self.model = None
        self.env = None
        
        logger.info(" DQN Trading Agent initialized")
    
    def create_environment(self, df, initial_capital: float = 100000.0):
        """Create and wrap trading environment"""
        
        # Create base environment
        env = TradingEnvironment(
            df=df,
            initial_capital=initial_capital
        )
        
        # Wrap with Monitor for logging
        env = Monitor(env)
        
        # Wrap with DummyVecEnv for stable-baselines3
        env = DummyVecEnv([lambda: env])
        
        self.env = env
        
        logger.info(f" Trading environment created with ${initial_capital:,.2f} capital")
        
        return env
    
    def create_model(self, env=None):
        """Create DQN model"""
        
        if env is None:
            env = self.env
        
        if env is None:
            raise ValueError("Environment must be created or provided")
        
        self.model = DQN(
            policy="MlpPolicy",
            env=env,
            learning_rate=self.learning_rate,
            buffer_size=self.buffer_size,
            learning_starts=self.learning_starts,
            batch_size=self.batch_size,
            tau=self.tau,
            gamma=self.gamma,
            train_freq=self.train_freq,
            gradient_steps=self.gradient_steps,
            target_update_interval=self.target_update_interval,
            exploration_fraction=self.exploration_fraction,
            exploration_initial_eps=self.exploration_initial_eps,
            exploration_final_eps=self.exploration_final_eps,
            verbose=1,
            tensorboard_log="./logs/dqn_tensorboard/"
        )
        
        logger.info(" DQN model created")
        
        return self.model
    
    def train(
        self,
        df,
        total_timesteps: int = 100000,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        initial_capital: float = 100000.0
    ):
        """
        Train DQN agent
        
        Args:
            df: Market data DataFrame
            total_timesteps: Total training timesteps
            eval_freq: Evaluation frequency
            n_eval_episodes: Number of evaluation episodes
            initial_capital: Starting capital
        """
        
        logger.info(f" Training DQN agent for {total_timesteps:,} timesteps...")
        
        # Create environment
        if self.env is None:
            self.create_environment(df, initial_capital)
        
        # Create model
        if self.model is None:
            self.create_model()
        
        # Create eval environment
        eval_env = TradingEnvironment(df=df, initial_capital=initial_capital)
        eval_env = Monitor(eval_env)
        eval_env = DummyVecEnv([lambda: eval_env])
        
        # Setup callbacks
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(self.model_path.parent / "best_model"),
            log_path=str(self.model_path.parent / "eval_logs"),
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            render=False
        )
        
        checkpoint_callback = CheckpointCallback(
            save_freq=eval_freq,
            save_path=str(self.model_path.parent / "checkpoints"),
            name_prefix="dqn_trading"
        )
        
        # Train
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, checkpoint_callback]
        )
        
        # Save final model
        self.save_model()
        
        logger.info(" DQN training completed")
        
        return self.model
    
    def predict(self, observation, deterministic: bool = True):
        """
        Predict action for given observation
        
        Args:
            observation: Current environment state
            deterministic: Use deterministic policy
        
        Returns:
            action
        """
        
        if self.model is None:
            raise ValueError("Model not loaded. Train or load a model first.")
        
        action, _states = self.model.predict(observation, deterministic=deterministic)
        
        return action
    
    def evaluate(
        self,
        df,
        n_episodes: int = 10,
        initial_capital: float = 100000.0
    ) -> Dict[str, float]:
        """
        Evaluate agent performance
        
        Args:
            df: Market data for evaluation
            n_episodes: Number of evaluation episodes
            initial_capital: Starting capital
        
        Returns:
            Performance metrics
        """
        
        logger.info(f" Evaluating DQN agent over {n_episodes} episodes...")
        
        # Create evaluation environment
        eval_env = TradingEnvironment(df=df, initial_capital=initial_capital)
        
        episode_rewards = []
        episode_metrics = []
        
        for episode in range(n_episodes):
            obs = eval_env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action = self.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action[0])
                episode_reward += reward
            
            # Get performance metrics
            metrics = eval_env.get_performance_metrics()
            episode_metrics.append(metrics)
            episode_rewards.append(episode_reward)
            
            logger.info(
                f"Episode {episode + 1}/{n_episodes}: "
                f"Reward={episode_reward:.2f}, "
                f"Return={metrics['total_return']:.2%}, "
                f"Sharpe={metrics['sharpe_ratio']:.2f}"
            )
        
        # Calculate aggregate metrics
        avg_metrics = {
            'avg_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'avg_return': np.mean([m['total_return'] for m in episode_metrics]),
            'avg_sharpe': np.mean([m['sharpe_ratio'] for m in episode_metrics]),
            'avg_max_drawdown': np.mean([m['max_drawdown'] for m in episode_metrics]),
            'avg_win_rate': np.mean([m['win_rate'] for m in episode_metrics]),
            'avg_num_trades': np.mean([m['num_trades'] for m in episode_metrics])
        }
        
        logger.info(" Evaluation completed")
        logger.info(f" Average Return: {avg_metrics['avg_return']:.2%}")
        logger.info(f" Average Sharpe Ratio: {avg_metrics['avg_sharpe']:.2f}")
        logger.info(f" Average Win Rate: {avg_metrics['avg_win_rate']:.2%}")
        
        return avg_metrics
    
    def save_model(self):
        """Save trained model"""
        if self.model is None:
            logger.warning(" No model to save")
            return
        
        self.model.save(str(self.model_path))
        logger.info(f" Model saved to {self.model_path}")
    
    def load_model(self, env=None):
        """Load trained model"""
        if not self.model_path.with_suffix('.zip').exists():
            logger.warning(f" Model file not found: {self.model_path}")
            return False
        
        if env is None:
            env = self.env
        
        if env is None:
            logger.warning(" No environment provided for loading model")
            return False
        
        self.model = DQN.load(str(self.model_path), env=env)
        logger.info(f" Model loaded from {self.model_path}")
        
        return True
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        if self.model is None:
            return {'status': 'not_initialized'}
        
        return {
            'status': 'initialized',
            'learning_rate': self.learning_rate,
            'buffer_size': self.buffer_size,
            'batch_size': self.batch_size,
            'gamma': self.gamma,
            'model_path': str(self.model_path)
        }

def create_dqn_agent(**kwargs) -> DQNTradingAgent:
    """Factory function to create DQN agent"""
    return DQNTradingAgent(**kwargs)

