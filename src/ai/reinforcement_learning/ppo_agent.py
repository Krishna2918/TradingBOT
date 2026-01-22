"""
Proximal Policy Optimization (PPO) Agent for Trading

This module implements a PPO agent for reinforcement learning-based trading.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import os
from collections import deque

logger = logging.getLogger(__name__)

class PPONetwork(nn.Module):
    """
    PPO network with actor and critic heads.
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int = 3,
        hidden_sizes: List[int] = [128, 64],
        dropout_rate: float = 0.2
    ):
        """
        Initialize PPO network.
        
        Args:
            state_size: Size of input state
            action_size: Number of actions
            hidden_sizes: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
        """
        super(PPONetwork, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        
        # Shared layers
        self.shared_layers = nn.ModuleList()
        prev_size = state_size
        
        for hidden_size in hidden_sizes:
            self.shared_layers.append(nn.Linear(prev_size, hidden_size))
            self.shared_layers.append(nn.ReLU())
            self.shared_layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Actor head (policy)
        self.actor = nn.Linear(prev_size, action_size)
        
        # Critic head (value function)
        self.critic = nn.Linear(prev_size, 1)
        
        logger.info(f"Initialized PPO Network: {state_size} -> {hidden_sizes} -> Actor({action_size}), Critic(1)")
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            state: Input state tensor
            
        Returns:
            Tuple of (action_logits, value)
        """
        x = state
        
        # Shared layers
        for layer in self.shared_layers:
            x = layer(x)
        
        # Actor and critic heads
        action_logits = self.actor(x)
        value = self.critic(x)
        
        return action_logits, value
    
    def get_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action from policy.
        
        Args:
            state: Input state tensor
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        action_logits, value = self.forward(state)
        
        # Create action distribution
        action_dist = torch.distributions.Categorical(logits=action_logits)
        
        # Sample action
        action = action_dist.sample()
        
        # Get log probability
        log_prob = action_dist.log_prob(action)
        
        return action, log_prob, value
    
    def evaluate_action(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate action for given state.
        
        Args:
            state: Input state tensor
            action: Action to evaluate
            
        Returns:
            Tuple of (log_prob, value, entropy)
        """
        action_logits, value = self.forward(state)
        
        # Create action distribution
        action_dist = torch.distributions.Categorical(logits=action_logits)
        
        # Get log probability and entropy
        log_prob = action_dist.log_prob(action)
        entropy = action_dist.entropy()
        
        return log_prob, value, entropy

class PPOBuffer:
    """
    Buffer for storing PPO experiences.
    """
    
    def __init__(self, buffer_size: int, state_size: int, device: str = 'cpu'):
        """
        Initialize PPO buffer.
        
        Args:
            buffer_size: Maximum buffer size
            state_size: Size of state space
            device: Device to use
        """
        self.buffer_size = buffer_size
        self.state_size = state_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Initialize buffers
        self.states = torch.zeros(buffer_size, state_size, device=self.device)
        self.actions = torch.zeros(buffer_size, dtype=torch.long, device=self.device)
        self.rewards = torch.zeros(buffer_size, device=self.device)
        self.values = torch.zeros(buffer_size, device=self.device)
        self.log_probs = torch.zeros(buffer_size, device=self.device)
        self.advantages = torch.zeros(buffer_size, device=self.device)
        self.returns = torch.zeros(buffer_size, device=self.device)
        
        self.ptr = 0
        self.size = 0
        
        logger.info(f"Initialized PPO Buffer with size {buffer_size}")
    
    def add(self, state: np.ndarray, action: int, reward: float, 
            value: float, log_prob: float) -> None:
        """
        Add experience to buffer.
        
        Args:
            state: State observation
            action: Action taken
            reward: Reward received
            value: Value estimate
            log_prob: Log probability of action
        """
        self.states[self.ptr] = torch.FloatTensor(state).to(self.device)
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    
    def compute_advantages_and_returns(self, gamma: float = 0.99, 
                                     lambda_gae: float = 0.95) -> None:
        """
        Compute advantages and returns using GAE.
        
        Args:
            gamma: Discount factor
            lambda_gae: GAE parameter
        """
        # Compute returns and advantages
        advantages = torch.zeros_like(self.rewards)
        returns = torch.zeros_like(self.rewards)
        
        # Compute returns (discounted cumulative rewards)
        running_return = 0
        for t in reversed(range(self.size)):
            running_return = self.rewards[t] + gamma * running_return
            returns[t] = running_return
        
        # Compute advantages using GAE
        running_advantage = 0
        for t in reversed(range(self.size)):
            if t == self.size - 1:
                next_value = 0
            else:
                next_value = self.values[t + 1]
            
            delta = self.rewards[t] + gamma * next_value - self.values[t]
            running_advantage = delta + gamma * lambda_gae * running_advantage
            advantages[t] = running_advantage
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        self.advantages = advantages
        self.returns = returns
    
    def get_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Get a batch of experiences.
        
        Args:
            batch_size: Size of batch to return
            
        Returns:
            Dictionary with batch data
        """
        indices = torch.randperm(self.size)[:batch_size]
        
        return {
            'states': self.states[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'values': self.values[indices],
            'log_probs': self.log_probs[indices],
            'advantages': self.advantages[indices],
            'returns': self.returns[indices]
        }
    
    def clear(self) -> None:
        """Clear the buffer."""
        self.ptr = 0
        self.size = 0

class PPOAgent:
    """
    Proximal Policy Optimization agent for trading.
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int = 3,
        hidden_sizes: List[int] = [128, 64],
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        lambda_gae: float = 0.95,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        buffer_size: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        device: str = 'cpu'
    ):
        """
        Initialize PPO agent.
        
        Args:
            state_size: Size of state space
            action_size: Number of actions
            hidden_sizes: Hidden layer sizes
            learning_rate: Learning rate
            gamma: Discount factor
            lambda_gae: GAE parameter
            clip_ratio: PPO clip ratio
            value_coef: Value function loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            buffer_size: Buffer size
            batch_size: Training batch size
            n_epochs: Number of training epochs per update
            device: Device to use
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Initialize network
        self.network = PPONetwork(state_size, action_size, hidden_sizes).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Initialize buffer
        self.buffer = PPOBuffer(buffer_size, state_size, device)
        
        # Training statistics
        self.episode_rewards = []
        self.episode_losses = []
        self.policy_losses = []
        self.value_losses = []
        
        logger.info(f"Initialized PPO Agent on {self.device}")
    
    def act(self, state: np.ndarray, training: bool = True) -> Tuple[int, float, float]:
        """
        Choose action using current policy.
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, value = self.network.get_action(state_tensor)
        
        return action.item(), log_prob.item(), value.item()
    
    def store_experience(self, state: np.ndarray, action: int, reward: float,
                        value: float, log_prob: float) -> None:
        """
        Store experience in buffer.
        
        Args:
            state: State observation
            action: Action taken
            reward: Reward received
            value: Value estimate
            log_prob: Log probability of action
        """
        self.buffer.add(state, action, reward, value, log_prob)
    
    def update(self) -> Dict[str, float]:
        """
        Update the policy using PPO.
        
        Returns:
            Dictionary with training statistics
        """
        if self.buffer.size < self.batch_size:
            return {}
        
        # Compute advantages and returns
        self.buffer.compute_advantages_and_returns(self.gamma, self.lambda_gae)
        
        # Training statistics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        total_loss = 0
        
        # Train for multiple epochs
        for epoch in range(self.n_epochs):
            # Get batch
            batch = self.buffer.get_batch(self.batch_size)
            
            # Forward pass
            log_probs, values, entropy = self.network.evaluate_action(
                batch['states'], batch['actions']
            )
            
            # Compute policy loss (PPO)
            ratio = torch.exp(log_probs - batch['log_probs'])
            surr1 = ratio * batch['advantages']
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch['advantages']
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Compute value loss
            value_loss = F.mse_loss(values.squeeze(), batch['returns'])
            
            # Compute entropy loss
            entropy_loss = -entropy.mean()
            
            # Total loss
            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            # Accumulate statistics
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy_loss += entropy_loss.item()
            total_loss += loss.item()
        
        # Average statistics
        avg_policy_loss = total_policy_loss / self.n_epochs
        avg_value_loss = total_value_loss / self.n_epochs
        avg_entropy_loss = total_entropy_loss / self.n_epochs
        avg_loss = total_loss / self.n_epochs
        
        # Store statistics
        self.policy_losses.append(avg_policy_loss)
        self.value_losses.append(avg_value_loss)
        self.episode_losses.append(avg_loss)
        
        # Clear buffer
        self.buffer.clear()
        
        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy_loss': avg_entropy_loss,
            'total_loss': avg_loss
        }
    
    def save(self, filepath: str) -> None:
        """
        Save the agent's state.
        
        Args:
            filepath: Path to save the model
        """
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'episode_losses': self.episode_losses,
            'policy_losses': self.policy_losses,
            'value_losses': self.value_losses
        }, filepath)
        
        logger.info(f"Saved PPO agent to {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Load the agent's state.
        
        Args:
            filepath: Path to load the model from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_losses = checkpoint['episode_losses']
        self.policy_losses = checkpoint['policy_losses']
        self.value_losses = checkpoint['value_losses']
        
        logger.info(f"Loaded PPO agent from {filepath}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """
        Get training statistics.
        
        Returns:
            Dictionary with training statistics
        """
        return {
            'episode_count': len(self.episode_rewards),
            'avg_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
            'avg_loss': np.mean(self.episode_losses[-100:]) if self.episode_losses else 0,
            'avg_policy_loss': np.mean(self.policy_losses[-100:]) if self.policy_losses else 0,
            'avg_value_loss': np.mean(self.value_losses[-100:]) if self.value_losses else 0
        }

class PPOTrainer:
    """
    Trainer for PPO agent.
    """
    
    def __init__(
        self,
        agent: PPOAgent,
        environment,
        save_dir: str = "models/ppo",
        log_interval: int = 100,
        update_interval: int = 2048
    ):
        """
        Initialize PPO trainer.
        
        Args:
            agent: PPO agent to train
            environment: Trading environment
            save_dir: Directory to save models
            log_interval: Logging interval
            update_interval: Update interval (steps)
        """
        self.agent = agent
        self.environment = environment
        self.save_dir = save_dir
        self.log_interval = log_interval
        self.update_interval = update_interval
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Training statistics
        self.episode_count = 0
        self.total_steps = 0
        
        logger.info("Initialized PPO Trainer")
    
    def train_episode(self) -> Dict[str, float]:
        """
        Train the agent for one episode.
        
        Returns:
            Episode statistics
        """
        state = self.environment.reset()
        episode_reward = 0
        step_count = 0
        
        done = False
        while not done:
            # Choose action
            action, log_prob, value = self.agent.act(state, training=True)
            
            # Execute action
            next_state, reward, done, info = self.environment.step(action)
            
            # Store experience
            self.agent.store_experience(state, action, reward, value, log_prob)
            
            # Update if buffer is full
            if self.agent.buffer.size >= self.update_interval:
                update_stats = self.agent.update()
            
            # Update statistics
            episode_reward += reward
            step_count += 1
            self.total_steps += 1
            
            state = next_state
        
        # Record episode statistics
        self.agent.episode_rewards.append(episode_reward)
        self.episode_count += 1
        
        # Log progress
        if self.episode_count % self.log_interval == 0:
            self._log_progress()
        
        return {
            'episode_reward': episode_reward,
            'episode_steps': step_count
        }
    
    def train(self, n_episodes: int) -> List[Dict[str, float]]:
        """
        Train the agent for multiple episodes.
        
        Args:
            n_episodes: Number of episodes to train
            
        Returns:
            List of episode statistics
        """
        logger.info(f"Starting PPO training for {n_episodes} episodes")
        
        episode_stats = []
        
        for episode in range(n_episodes):
            stats = self.train_episode()
            episode_stats.append(stats)
            
            # Save model periodically
            if episode % (n_episodes // 10) == 0:
                self.save_model(f"ppo_episode_{episode}")
        
        # Save final model
        self.save_model("ppo_final")
        
        logger.info("PPO training completed")
        
        return episode_stats
    
    def evaluate(self, n_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate the agent's performance.
        
        Args:
            n_episodes: Number of episodes to evaluate
            
        Returns:
            Evaluation statistics
        """
        logger.info(f"Evaluating PPO agent for {n_episodes} episodes")
        
        episode_rewards = []
        episode_metrics = []
        
        for episode in range(n_episodes):
            state = self.environment.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # Choose action (no exploration during evaluation)
                action, _, _ = self.agent.act(state, training=False)
                
                # Execute action
                state, reward, done, info = self.environment.step(action)
                episode_reward += reward
            
            episode_rewards.append(episode_reward)
            
            # Get environment metrics
            metrics = self.environment.calculate_metrics()
            episode_metrics.append(metrics)
        
        # Calculate average metrics
        avg_reward = np.mean(episode_rewards)
        avg_metrics = {}
        
        if episode_metrics:
            for key in episode_metrics[0].keys():
                avg_metrics[f'avg_{key}'] = np.mean([m[key] for m in episode_metrics if key in m])
        
        evaluation_stats = {
            'avg_episode_reward': avg_reward,
            'std_episode_reward': np.std(episode_rewards),
            **avg_metrics
        }
        
        logger.info(f"Evaluation completed: {evaluation_stats}")
        
        return evaluation_stats
    
    def save_model(self, filename: str) -> None:
        """
        Save the trained model.
        
        Args:
            filename: Filename to save the model
        """
        filepath = os.path.join(self.save_dir, f"{filename}.pth")
        self.agent.save(filepath)
    
    def load_model(self, filename: str) -> None:
        """
        Load a trained model.
        
        Args:
            filename: Filename to load the model from
        """
        filepath = os.path.join(self.save_dir, f"{filename}.pth")
        self.agent.load(filepath)
    
    def _log_progress(self) -> None:
        """Log training progress."""
        stats = self.agent.get_training_stats()
        
        logger.info(f"Episode {self.episode_count}: "
                   f"Avg Reward: {stats['avg_reward']:.4f}, "
                   f"Avg Loss: {stats['avg_loss']:.4f}, "
                   f"Policy Loss: {stats['avg_policy_loss']:.4f}, "
                   f"Value Loss: {stats['avg_value_loss']:.4f}")
    
    def get_training_history(self) -> Dict[str, List[float]]:
        """
        Get training history.
        
        Returns:
            Dictionary with training history
        """
        return {
            'episode_rewards': self.agent.episode_rewards,
            'episode_losses': self.agent.episode_losses,
            'policy_losses': self.agent.policy_losses,
            'value_losses': self.agent.value_losses,
            'episode_count': self.episode_count,
            'total_steps': self.total_steps
        }

