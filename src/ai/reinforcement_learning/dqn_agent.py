"""
Deep Q-Network (DQN) Agent for Trading

This module implements a DQN agent for reinforcement learning-based trading.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class DQNNetwork(nn.Module):
    """
    Deep Q-Network architecture for trading.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int] = [128, 64, 32],
        output_size: int = 3,
        dropout_rate: float = 0.2
    ):
        """
        Initialize DQN network.
        
        Args:
            input_size: Size of input state
            hidden_sizes: List of hidden layer sizes
            output_size: Number of actions
            dropout_rate: Dropout rate for regularization
        """
        super(DQNNetwork, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        
        # Build network layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
        logger.info(f"Initialized DQN Network: {input_size} -> {hidden_sizes} -> {output_size}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Q-values for each action
        """
        return self.network(x)

class ReplayBuffer:
    """
    Experience replay buffer for DQN training.
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        
        logger.info(f"Initialized Replay Buffer with capacity {capacity}")
    
    def push(self, experience: Experience) -> None:
        """
        Add experience to buffer.
        
        Args:
            experience: Experience tuple
        """
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """
        Sample batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            List of sampled experiences
        """
        return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)

class DQNAgent:
    """
    Deep Q-Network agent for trading.
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int = 3,
        hidden_sizes: List[int] = [128, 64, 32],
        learning_rate: float = 0.001,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 32,
        target_update_freq: int = 100,
        device: str = 'cpu'
    ):
        """
        Initialize DQN agent.
        
        Args:
            state_size: Size of state space
            action_size: Number of actions
            hidden_sizes: Hidden layer sizes
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Epsilon decay rate
            buffer_size: Replay buffer size
            batch_size: Training batch size
            target_update_freq: Target network update frequency
            device: Device to use ('cpu' or 'cuda')
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Initialize networks
        self.q_network = DQNNetwork(state_size, hidden_sizes, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, hidden_sizes, action_size).to(self.device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Initialize replay buffer
        self.memory = ReplayBuffer(buffer_size)
        
        # Training statistics
        self.training_step = 0
        self.episode_rewards = []
        self.episode_losses = []
        
        logger.info(f"Initialized DQN Agent on {self.device}")
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """
        Choose action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Selected action
        """
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool) -> None:
        """
        Store experience in replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        experience = Experience(state, action, reward, next_state, done)
        self.memory.push(experience)
    
    def replay(self) -> float:
        """
        Train the agent on a batch of experiences.
        
        Returns:
            Average loss for the batch
        """
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch from memory
        experiences = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
        actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.device)
        dones = torch.BoolTensor([e.done for e in experiences]).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def save(self, filepath: str) -> None:
        """
        Save the agent's state.
        
        Args:
            filepath: Path to save the model
        """
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'episode_rewards': self.episode_rewards,
            'episode_losses': self.episode_losses
        }, filepath)
        
        logger.info(f"Saved DQN agent to {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Load the agent's state.
        
        Args:
            filepath: Path to load the model from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_losses = checkpoint['episode_losses']
        
        logger.info(f"Loaded DQN agent from {filepath}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """
        Get training statistics.
        
        Returns:
            Dictionary with training statistics
        """
        return {
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'buffer_size': len(self.memory),
            'episode_count': len(self.episode_rewards),
            'avg_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
            'avg_loss': np.mean(self.episode_losses[-100:]) if self.episode_losses else 0
        }

class DQNTrainer:
    """
    Trainer for DQN agent.
    """
    
    def __init__(
        self,
        agent: DQNAgent,
        environment,
        save_dir: str = "models/dqn",
        log_interval: int = 100
    ):
        """
        Initialize DQN trainer.
        
        Args:
            agent: DQN agent to train
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
        
        logger.info("Initialized DQN Trainer")
    
    def train_episode(self) -> Dict[str, float]:
        """
        Train the agent for one episode.
        
        Returns:
            Episode statistics
        """
        state = self.environment.reset()
        episode_reward = 0
        episode_loss = 0
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
            loss = self.agent.replay()
            episode_loss += loss
            
            # Update statistics
            episode_reward += reward
            step_count += 1
            self.total_steps += 1
            
            state = next_state
        
        # Record episode statistics
        self.agent.episode_rewards.append(episode_reward)
        self.agent.episode_losses.append(episode_loss / step_count if step_count > 0 else 0)
        self.episode_count += 1
        
        # Log progress
        if self.episode_count % self.log_interval == 0:
            self._log_progress()
        
        return {
            'episode_reward': episode_reward,
            'episode_loss': episode_loss / step_count if step_count > 0 else 0,
            'episode_steps': step_count,
            'epsilon': self.agent.epsilon
        }
    
    def train(self, n_episodes: int) -> List[Dict[str, float]]:
        """
        Train the agent for multiple episodes.
        
        Args:
            n_episodes: Number of episodes to train
            
        Returns:
            List of episode statistics
        """
        logger.info(f"Starting DQN training for {n_episodes} episodes")
        
        episode_stats = []
        
        for episode in range(n_episodes):
            stats = self.train_episode()
            episode_stats.append(stats)
            
            # Save model periodically
            if episode % (n_episodes // 10) == 0:
                self.save_model(f"dqn_episode_{episode}")
        
        # Save final model
        self.save_model("dqn_final")
        
        logger.info("DQN training completed")
        
        return episode_stats
    
    def evaluate(self, n_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate the agent's performance.
        
        Args:
            n_episodes: Number of episodes to evaluate
            
        Returns:
            Evaluation statistics
        """
        logger.info(f"Evaluating DQN agent for {n_episodes} episodes")
        
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
                   f"Epsilon: {stats['epsilon']:.4f}")
    
    def get_training_history(self) -> Dict[str, List[float]]:
        """
        Get training history.
        
        Returns:
            Dictionary with training history
        """
        return {
            'episode_rewards': self.agent.episode_rewards,
            'episode_losses': self.agent.episode_losses,
            'episode_count': self.episode_count,
            'total_steps': self.total_steps
        }

