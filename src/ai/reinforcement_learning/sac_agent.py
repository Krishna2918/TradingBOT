"""
Soft Actor-Critic (SAC) Agent for Trading

This module implements a SAC agent for reinforcement learning-based trading.
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
import random

logger = logging.getLogger(__name__)

class SACNetwork(nn.Module):
    """
    SAC network with actor and critic heads.
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int = 3,
        hidden_sizes: List[int] = [128, 64],
        dropout_rate: float = 0.2
    ):
        """
        Initialize SAC network.
        
        Args:
            state_size: Size of input state
            action_size: Number of actions
            hidden_sizes: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
        """
        super(SACNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        
        # Build network layers
        layers = []
        prev_size = state_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, action_size))
        
        self.network = nn.Sequential(*layers)
        
        logger.info(f"Initialized SAC Network: {state_size} -> {hidden_sizes} -> {action_size}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Network output
        """
        return self.network(x)

class Actor(nn.Module):
    """
    Actor network for SAC.
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int = 3,
        hidden_sizes: List[int] = [128, 64],
        dropout_rate: float = 0.2
    ):
        """
        Initialize Actor network.
        
        Args:
            state_size: Size of input state
            action_size: Number of actions
            hidden_sizes: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
        """
        super(Actor, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        
        # Build network layers
        layers = []
        prev_size = state_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer for action logits
        layers.append(nn.Linear(prev_size, action_size))
        
        self.network = nn.Sequential(*layers)
        
        logger.info(f"Initialized Actor Network: {state_size} -> {hidden_sizes} -> {action_size}")
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            state: Input state tensor
            
        Returns:
            Action logits
        """
        return self.network(state)
    
    def get_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get action from policy.
        
        Args:
            state: Input state tensor
            
        Returns:
            Tuple of (action, log_prob)
        """
        action_logits = self.forward(state)
        
        # Create action distribution
        action_dist = torch.distributions.Categorical(logits=action_logits)
        
        # Sample action
        action = action_dist.sample()
        
        # Get log probability
        log_prob = action_dist.log_prob(action)
        
        return action, log_prob

class Critic(nn.Module):
    """
    Critic network for SAC.
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int = 3,
        hidden_sizes: List[int] = [128, 64],
        dropout_rate: float = 0.2
    ):
        """
        Initialize Critic network.
        
        Args:
            state_size: Size of input state
            action_size: Number of actions
            hidden_sizes: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
        """
        super(Critic, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        
        # Build network layers
        layers = []
        prev_size = state_size + action_size  # State + action input
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer for Q-value
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
        
        logger.info(f"Initialized Critic Network: {state_size + action_size} -> {hidden_sizes} -> 1")
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            state: Input state tensor
            action: Input action tensor
            
        Returns:
            Q-value
        """
        # Concatenate state and action
        x = torch.cat([state, action], dim=-1)
        return self.network(x)

class SACBuffer:
    """
    Replay buffer for SAC.
    """
    
    def __init__(self, capacity: int = 100000):
        """
        Initialize SAC buffer.
        
        Args:
            capacity: Maximum buffer size
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        
        logger.info(f"Initialized SAC Buffer with capacity {capacity}")
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool) -> None:
        """
        Add experience to buffer.
        
        Args:
            state: State observation
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample batch from buffer.
        
        Args:
            batch_size: Number of samples
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """Return buffer size."""
        return len(self.buffer)

class SACAgent:
    """
    Soft Actor-Critic agent for trading.
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int = 3,
        hidden_sizes: List[int] = [128, 64],
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        buffer_size: int = 100000,
        batch_size: int = 256,
        device: str = 'cpu'
    ):
        """
        Initialize SAC agent.
        
        Args:
            state_size: Size of state space
            action_size: Number of actions
            hidden_sizes: Hidden layer sizes
            learning_rate: Learning rate
            gamma: Discount factor
            tau: Soft update parameter
            alpha: Entropy regularization coefficient
            buffer_size: Replay buffer size
            batch_size: Training batch size
            device: Device to use
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Initialize networks
        self.actor = Actor(state_size, action_size, hidden_sizes).to(self.device)
        self.critic1 = Critic(state_size, action_size, hidden_sizes).to(self.device)
        self.critic2 = Critic(state_size, action_size, hidden_sizes).to(self.device)
        
        # Target networks
        self.target_critic1 = Critic(state_size, action_size, hidden_sizes).to(self.device)
        self.target_critic2 = Critic(state_size, action_size, hidden_sizes).to(self.device)
        
        # Copy weights to target networks
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate)
        
        # Initialize replay buffer
        self.memory = SACBuffer(buffer_size)
        
        # Training statistics
        self.episode_rewards = []
        self.episode_losses = []
        self.actor_losses = []
        self.critic_losses = []
        
        logger.info(f"Initialized SAC Agent on {self.device}")
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """
        Choose action using current policy.
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Selected action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, _ = self.actor.get_action(state_tensor)
        
        return action.item()
    
    def remember(self, state: np.ndarray, action: int, reward: float,
                next_state: np.ndarray, done: bool) -> None:
        """
        Store experience in replay buffer.
        
        Args:
            state: State observation
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.memory.push(state, action, reward, next_state, done)
    
    def update(self) -> Dict[str, float]:
        """
        Update the agent using SAC.
        
        Returns:
            Dictionary with training statistics
        """
        if len(self.memory) < self.batch_size:
            return {}
        
        # Sample batch from memory
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Update critics
        critic1_loss, critic2_loss = self._update_critics(states, actions, rewards, next_states, dones)
        
        # Update actor
        actor_loss = self._update_actor(states)
        
        # Soft update target networks
        self._soft_update(self.target_critic1, self.critic1)
        self._soft_update(self.target_critic2, self.critic2)
        
        # Store statistics
        self.actor_losses.append(actor_loss)
        self.critic_losses.append((critic1_loss + critic2_loss) / 2)
        self.episode_losses.append((actor_loss + critic1_loss + critic2_loss) / 3)
        
        return {
            'actor_loss': actor_loss,
            'critic1_loss': critic1_loss,
            'critic2_loss': critic2_loss,
            'total_loss': (actor_loss + critic1_loss + critic2_loss) / 3
        }
    
    def _update_critics(self, states: torch.Tensor, actions: torch.Tensor, 
                       rewards: torch.Tensor, next_states: torch.Tensor, 
                       dones: torch.Tensor) -> Tuple[float, float]:
        """
        Update critic networks.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of done flags
            
        Returns:
            Tuple of (critic1_loss, critic2_loss)
        """
        with torch.no_grad():
            # Get next actions and log probabilities
            next_actions, next_log_probs = self.actor.get_action(next_states)
            
            # Get target Q-values
            target_q1 = self.target_critic1(next_states, next_actions)
            target_q2 = self.target_critic2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs.unsqueeze(1)
            target_q = rewards.unsqueeze(1) + self.gamma * target_q * (~dones).unsqueeze(1)
        
        # Current Q-values
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        
        # Compute losses
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        # Update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        return critic1_loss.item(), critic2_loss.item()
    
    def _update_actor(self, states: torch.Tensor) -> float:
        """
        Update actor network.
        
        Args:
            states: Batch of states
            
        Returns:
            Actor loss
        """
        # Get actions and log probabilities
        actions, log_probs = self.actor.get_action(states)
        
        # Get Q-values
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        q = torch.min(q1, q2)
        
        # Compute loss
        actor_loss = (self.alpha * log_probs.unsqueeze(1) - q).mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return actor_loss.item()
    
    def _soft_update(self, target_network: nn.Module, source_network: nn.Module) -> None:
        """
        Soft update target network.
        
        Args:
            target_network: Target network
            source_network: Source network
        """
        for target_param, source_param in zip(target_network.parameters(), source_network.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1 - self.tau) * target_param.data)
    
    def save(self, filepath: str) -> None:
        """
        Save the agent's state.
        
        Args:
            filepath: Path to save the model
        """
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'target_critic1_state_dict': self.target_critic1.state_dict(),
            'target_critic2_state_dict': self.target_critic2.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'episode_losses': self.episode_losses,
            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses
        }, filepath)
        
        logger.info(f"Saved SAC agent to {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Load the agent's state.
        
        Args:
            filepath: Path to load the model from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.target_critic1.load_state_dict(checkpoint['target_critic1_state_dict'])
        self.target_critic2.load_state_dict(checkpoint['target_critic2_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_losses = checkpoint['episode_losses']
        self.actor_losses = checkpoint['actor_losses']
        self.critic_losses = checkpoint['critic_losses']
        
        logger.info(f"Loaded SAC agent from {filepath}")
    
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
            'avg_actor_loss': np.mean(self.actor_losses[-100:]) if self.actor_losses else 0,
            'avg_critic_loss': np.mean(self.critic_losses[-100:]) if self.critic_losses else 0,
            'buffer_size': len(self.memory)
        }

class SACTrainer:
    """
    Trainer for SAC agent.
    """
    
    def __init__(
        self,
        agent: SACAgent,
        environment,
        save_dir: str = "models/sac",
        log_interval: int = 100
    ):
        """
        Initialize SAC trainer.
        
        Args:
            agent: SAC agent to train
            environment: Trading environment
            save_dir: Directory to save models
            log_interval: Logging interval
        """
        self.agent = agent
        self.environment = environment
        self.save_dir = save_dir
        self.log_interval = log_interval
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Training statistics
        self.episode_count = 0
        self.total_steps = 0
        
        logger.info("Initialized SAC Trainer")
    
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
            action = self.agent.act(state, training=True)
            
            # Execute action
            next_state, reward, done, info = self.environment.step(action)
            
            # Store experience
            self.agent.remember(state, action, reward, next_state, done)
            
            # Train agent
            self.agent.update()
            
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
        logger.info(f"Starting SAC training for {n_episodes} episodes")
        
        episode_stats = []
        
        for episode in range(n_episodes):
            stats = self.train_episode()
            episode_stats.append(stats)
            
            # Save model periodically
            if episode % (n_episodes // 10) == 0:
                self.save_model(f"sac_episode_{episode}")
        
        # Save final model
        self.save_model("sac_final")
        
        logger.info("SAC training completed")
        
        return episode_stats
    
    def evaluate(self, n_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate the agent's performance.
        
        Args:
            n_episodes: Number of episodes to evaluate
            
        Returns:
            Evaluation statistics
        """
        logger.info(f"Evaluating SAC agent for {n_episodes} episodes")
        
        episode_rewards = []
        episode_metrics = []
        
        for episode in range(n_episodes):
            state = self.environment.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # Choose action (no exploration during evaluation)
                action = self.agent.act(state, training=False)
                
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
                   f"Actor Loss: {stats['avg_actor_loss']:.4f}, "
                   f"Critic Loss: {stats['avg_critic_loss']:.4f}")
    
    def get_training_history(self) -> Dict[str, List[float]]:
        """
        Get training history.
        
        Returns:
            Dictionary with training history
        """
        return {
            'episode_rewards': self.agent.episode_rewards,
            'episode_losses': self.agent.episode_losses,
            'actor_losses': self.agent.actor_losses,
            'critic_losses': self.agent.critic_losses,
            'episode_count': self.episode_count,
            'total_steps': self.total_steps
        }

