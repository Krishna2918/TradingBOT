"""
Asynchronous Advantage Actor-Critic (A3C) Agent for Trading

This module implements an A3C agent for reinforcement learning-based trading.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import threading
import time
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import os
from collections import deque

logger = logging.getLogger(__name__)

class A3CNetwork(nn.Module):
    """
    A3C network with actor and critic heads.
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int = 3,
        hidden_sizes: List[int] = [128, 64],
        dropout_rate: float = 0.2
    ):
        """
        Initialize A3C network.
        
        Args:
            state_size: Size of input state
            action_size: Number of actions
            hidden_sizes: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
        """
        super(A3CNetwork, self).__init__()
        
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
        
        logger.info(f"Initialized A3C Network: {state_size} -> {hidden_sizes} -> Actor({action_size}), Critic(1)")
    
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

class A3CAgent:
    """
    Asynchronous Advantage Actor-Critic agent for trading.
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int = 3,
        hidden_sizes: List[int] = [128, 64],
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        beta: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = 'cpu'
    ):
        """
        Initialize A3C agent.
        
        Args:
            state_size: Size of state space
            action_size: Number of actions
            hidden_sizes: Hidden layer sizes
            learning_rate: Learning rate
            gamma: Discount factor
            beta: Entropy regularization coefficient
            max_grad_norm: Maximum gradient norm for clipping
            device: Device to use
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.beta = beta
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Initialize network
        self.network = A3CNetwork(state_size, action_size, hidden_sizes).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Training statistics
        self.episode_rewards = []
        self.episode_losses = []
        self.policy_losses = []
        self.value_losses = []
        
        logger.info(f"Initialized A3C Agent on {self.device}")
    
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
    
    def compute_returns_and_advantages(self, rewards: List[float], values: List[float], 
                                     dones: List[bool]) -> Tuple[List[float], List[float]]:
        """
        Compute returns and advantages.
        
        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags
            
        Returns:
            Tuple of (returns, advantages)
        """
        returns = []
        advantages = []
        
        # Compute returns
        running_return = 0
        for i in reversed(range(len(rewards))):
            if dones[i]:
                running_return = 0
            running_return = rewards[i] + self.gamma * running_return
            returns.insert(0, running_return)
        
        # Compute advantages
        for i in range(len(rewards)):
            advantage = returns[i] - values[i]
            advantages.append(advantage)
        
        return returns, advantages
    
    def update(self, states: List[np.ndarray], actions: List[int], rewards: List[float],
               values: List[float], log_probs: List[float], dones: List[bool]) -> Dict[str, float]:
        """
        Update the policy using A3C.
        
        Args:
            states: List of states
            actions: List of actions
            rewards: List of rewards
            values: List of value estimates
            log_probs: List of log probabilities
            dones: List of done flags
            
        Returns:
            Dictionary with training statistics
        """
        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        values_tensor = torch.FloatTensor(values).to(self.device)
        log_probs_tensor = torch.FloatTensor(log_probs).to(self.device)
        
        # Compute returns and advantages
        returns, advantages = self.compute_returns_and_advantages(rewards, values, dones)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        
        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        # Forward pass
        log_probs_new, values_new, entropy = self.network.evaluate_action(states_tensor, actions_tensor)
        
        # Compute losses
        policy_loss = -(log_probs_new * advantages_tensor).mean()
        value_loss = F.mse_loss(values_new.squeeze(), returns_tensor)
        entropy_loss = -entropy.mean()
        
        # Total loss
        loss = policy_loss + 0.5 * value_loss + self.beta * entropy_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # Store statistics
        self.policy_losses.append(policy_loss.item())
        self.value_losses.append(value_loss.item())
        self.episode_losses.append(loss.item())
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': loss.item()
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
        
        logger.info(f"Saved A3C agent to {filepath}")
    
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
        
        logger.info(f"Loaded A3C agent from {filepath}")
    
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

class A3CWorker:
    """
    A3C worker thread for asynchronous training.
    """
    
    def __init__(
        self,
        worker_id: int,
        global_agent: A3CAgent,
        environment,
        max_steps: int = 20,
        lock: Optional[threading.Lock] = None
    ):
        """
        Initialize A3C worker.
        
        Args:
            worker_id: Worker ID
            global_agent: Global agent to update
            environment: Trading environment
            max_steps: Maximum steps per update
            lock: Thread lock for synchronization
        """
        self.worker_id = worker_id
        self.global_agent = global_agent
        self.environment = environment
        self.max_steps = max_steps
        self.lock = lock or threading.Lock()
        
        # Local agent (copy of global agent)
        self.local_agent = A3CAgent(
            state_size=global_agent.state_size,
            action_size=global_agent.action_size,
            learning_rate=global_agent.learning_rate,
            gamma=global_agent.gamma,
            beta=global_agent.beta,
            max_grad_norm=global_agent.max_grad_norm,
            device=global_agent.device
        )
        
        # Copy weights from global agent
        self.local_agent.network.load_state_dict(global_agent.network.state_dict())
        
        logger.info(f"Initialized A3C Worker {worker_id}")
    
    def work(self) -> Dict[str, float]:
        """
        Worker main loop.
        
        Returns:
            Worker statistics
        """
        # Reset environment
        state = self.environment.reset()
        
        # Experience buffers
        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        dones = []
        
        episode_reward = 0
        step_count = 0
        
        # Collect experience
        for step in range(self.max_steps):
            # Choose action
            action, log_prob, value = self.local_agent.act(state, training=True)
            
            # Execute action
            next_state, reward, done, info = self.environment.step(action)
            
            # Store experience
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            dones.append(done)
            
            episode_reward += reward
            step_count += 1
            
            state = next_state
            
            if done:
                break
        
        # Update global agent
        with self.lock:
            # Copy weights from global agent
            self.local_agent.network.load_state_dict(self.global_agent.network.state_dict())
            
            # Update global agent
            update_stats = self.global_agent.update(states, actions, rewards, values, log_probs, dones)
            
            # Record episode reward
            self.global_agent.episode_rewards.append(episode_reward)
        
        return {
            'worker_id': self.worker_id,
            'episode_reward': episode_reward,
            'episode_steps': step_count,
            **update_stats
        }

class A3CTrainer:
    """
    Trainer for A3C agent.
    """
    
    def __init__(
        self,
        agent: A3CAgent,
        environment,
        n_workers: int = 4,
        save_dir: str = "models/a3c",
        log_interval: int = 100,
        max_steps_per_update: int = 20
    ):
        """
        Initialize A3C trainer.
        
        Args:
            agent: A3C agent to train
            environment: Trading environment
            n_workers: Number of worker threads
            save_dir: Directory to save models
            log_interval: Logging interval
            max_steps_per_update: Maximum steps per update
        """
        self.agent = agent
        self.environment = environment
        self.n_workers = n_workers
        self.save_dir = save_dir
        self.log_interval = log_interval
        self.max_steps_per_update = max_steps_per_update
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Thread synchronization
        self.lock = threading.Lock()
        
        # Training statistics
        self.episode_count = 0
        self.total_steps = 0
        
        logger.info(f"Initialized A3C Trainer with {n_workers} workers")
    
    def train_episode(self) -> Dict[str, float]:
        """
        Train the agent for one episode using multiple workers.
        
        Returns:
            Episode statistics
        """
        # Create workers
        workers = []
        for i in range(self.n_workers):
            worker = A3CWorker(
                worker_id=i,
                global_agent=self.agent,
                environment=self.environment,
                max_steps=self.max_steps_per_update,
                lock=self.lock
            )
            workers.append(worker)
        
        # Start workers
        threads = []
        results = []
        
        for worker in workers:
            thread = threading.Thread(target=lambda w=worker: results.append(w.work()))
            threads.append(thread)
            thread.start()
        
        # Wait for all workers to complete
        for thread in threads:
            thread.join()
        
        # Aggregate results
        avg_reward = np.mean([r['episode_reward'] for r in results])
        avg_steps = np.mean([r['episode_steps'] for r in results])
        
        self.episode_count += 1
        self.total_steps += sum([r['episode_steps'] for r in results])
        
        # Log progress
        if self.episode_count % self.log_interval == 0:
            self._log_progress()
        
        return {
            'episode_reward': avg_reward,
            'episode_steps': avg_steps,
            'n_workers': self.n_workers
        }
    
    def train(self, n_episodes: int) -> List[Dict[str, float]]:
        """
        Train the agent for multiple episodes.
        
        Args:
            n_episodes: Number of episodes to train
            
        Returns:
            List of episode statistics
        """
        logger.info(f"Starting A3C training for {n_episodes} episodes with {self.n_workers} workers")
        
        episode_stats = []
        
        for episode in range(n_episodes):
            stats = self.train_episode()
            episode_stats.append(stats)
            
            # Save model periodically
            if episode % (n_episodes // 10) == 0:
                self.save_model(f"a3c_episode_{episode}")
        
        # Save final model
        self.save_model("a3c_final")
        
        logger.info("A3C training completed")
        
        return episode_stats
    
    def evaluate(self, n_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate the agent's performance.
        
        Args:
            n_episodes: Number of episodes to evaluate
            
        Returns:
            Evaluation statistics
        """
        logger.info(f"Evaluating A3C agent for {n_episodes} episodes")
        
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

