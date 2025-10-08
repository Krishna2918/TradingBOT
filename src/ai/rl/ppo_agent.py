"""
PPO (Proximal Policy Optimization) Agent for Trading

Uses Stable-Baselines3 PPO for learning optimal trading policies
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from .trading_environment import TradingEnvironment

logger = logging.getLogger(__name__)

class PPOTradingAgent:
    """
    PPO Agent for trading
    
    Advantages of PPO:
    - Stable training
    - Good sample efficiency
    - Works well with continuous action spaces
    - Less sensitive to hyperparameters
    """
    
    def __init__(
        self,
        model_path: str = "models/ppo_trading_agent",
        learning_rate: float = 0.0003,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5
    ):
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        
        self.model = None
        self.env = None
        
        logger.info(" PPO Trading Agent initialized")
    
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
        """Create PPO model"""
        
        if env is None:
            env = self.env
        
        if env is None:
            raise ValueError("Environment must be created or provided")
        
        self.model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=self.learning_rate,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_range=self.clip_range,
            ent_coef=self.ent_coef,
            vf_coef=self.vf_coef,
            max_grad_norm=self.max_grad_norm,
            verbose=1,
            tensorboard_log="./logs/ppo_tensorboard/"
        )
        
        logger.info(" PPO model created")
        
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
        Train PPO agent
        
        Args:
            df: Market data DataFrame
            total_timesteps: Total training timesteps
            eval_freq: Evaluation frequency
            n_eval_episodes: Number of evaluation episodes
            initial_capital: Starting capital
        """
        
        logger.info(f" Training PPO agent for {total_timesteps:,} timesteps...")
        
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
            name_prefix="ppo_trading"
        )
        
        # Train
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, checkpoint_callback]
        )
        
        # Save final model
        self.save_model()
        
        logger.info(" PPO training completed")
        
        return self.model
    
    def predict(self, observation, deterministic: bool = True):
        """
        Predict action for given observation
        
        Args:
            observation: Current environment state
            deterministic: Use deterministic policy
        
        Returns:
            action, value prediction
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
        
        logger.info(f" Evaluating PPO agent over {n_episodes} episodes...")
        
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
        
        self.model = PPO.load(str(self.model_path), env=env)
        logger.info(f" Model loaded from {self.model_path}")
        
        return True
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        if self.model is None:
            return {'status': 'not_initialized'}
        
        return {
            'status': 'initialized',
            'learning_rate': self.learning_rate,
            'n_steps': self.n_steps,
            'batch_size': self.batch_size,
            'gamma': self.gamma,
            'model_path': str(self.model_path)
        }

def create_ppo_agent(**kwargs) -> PPOTradingAgent:
    """Factory function to create PPO agent"""
    return PPOTradingAgent(**kwargs)

