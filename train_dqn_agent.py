"""
PRODUCTION DQN AGENT TRAINER - Optimized for RTX 4080
Trains DQN (Deep Q-Network) agent for trading

Features:
- GPU-accelerated training
- Experience replay buffer
- Target network with soft updates
- Double DQN to reduce overestimation
- ε-greedy exploration with decay
- Prioritized experience replay (optional)
- TensorBoard logging

Usage:
    # Training from scratch
    python train_dqn_agent.py --data TrainingData/features/*.parquet --episodes 1000

    # Resume from checkpoint
    python train_dqn_agent.py --resume models/dqn_checkpoint.pth

    # Quick test
    python train_dqn_agent.py --test-mode --episodes 100

Hardware Requirements:
    GPU: RTX 4080 (16GB VRAM)
    RAM: 32GB recommended

Author: Trading Bot Team
Date: October 28, 2025
"""

import os
import sys
import time
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from collections import deque
import random

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# Import custom environment
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.ai.rl.enhanced_trading_environment import EnhancedTradingEnvironment

# GPU Configuration
torch.backends.cudnn.benchmark = True
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


class DQNNetwork(nn.Module):
    """Deep Q-Network for trading"""

    def __init__(self, obs_shape: Tuple, action_dim: int, hidden_dim: int = 256):
        super(DQNNetwork, self).__init__()

        self.obs_shape = obs_shape  # (lookback_window, features)
        input_dim = obs_shape[0] * obs_shape[1]

        # Network layers
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                nn.init.constant_(module.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            obs: Observation tensor [batch, lookback_window, features]

        Returns:
            q_values: [batch, action_dim]
        """
        # Flatten observation
        batch_size = obs.shape[0]
        obs_flat = obs.view(batch_size, -1)

        # Get Q-values
        q_values = self.network(obs_flat)

        return q_values


class ReplayBuffer:
    """Experience replay buffer for DQN"""

    def __init__(self, capacity: int, obs_shape: Tuple):
        self.capacity = capacity
        self.obs_shape = obs_shape

        # Storage
        self.observations = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_observations = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        """Add experience to buffer"""
        idx = self.ptr % self.capacity

        self.observations[idx] = obs
        self.next_observations[idx] = next_obs
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = done

        self.ptr += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        """Sample batch of experiences"""
        indices = np.random.randint(0, self.size, size=batch_size)

        return (
            self.observations[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_observations[indices],
            self.dones[indices]
        )

    def __len__(self):
        return self.size


class DQNTrainer:
    """DQN trainer with GPU optimization"""

    def __init__(self, q_network, target_network, device, config):
        self.q_network = q_network.to(device)
        self.target_network = target_network.to(device)
        self.device = device
        self.config = config

        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(
            q_network.parameters(),
            lr=config['learning_rate']
        )

        # Epsilon for exploration
        self.epsilon = config['epsilon_start']

        # Tracking
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.best_avg_reward = -float('inf')
        self.total_steps = 0

        # TensorBoard
        log_dir = Path('logs/tensorboard_dqn') / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(log_dir)
        print(f"TensorBoard logging to: {log_dir}")

    def select_action(self, obs: np.ndarray, eval_mode: bool = False) -> int:
        """Select action using ε-greedy policy"""
        if eval_mode or random.random() > self.epsilon:
            # Exploit
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(obs_tensor)
                action = q_values.argmax(dim=1).item()
        else:
            # Explore
            action = random.randint(0, self.config['action_dim'] - 1)

        return action

    def update_epsilon(self):
        """Decay epsilon"""
        self.epsilon = max(
            self.config['epsilon_end'],
            self.epsilon * self.config['epsilon_decay']
        )

    def update_target_network(self):
        """Soft update target network"""
        tau = self.config['tau']
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def compute_loss(self, batch):
        """Compute DQN loss (Double DQN)"""
        obs, actions, rewards, next_obs, dones = batch

        # Convert to tensors
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_obs = torch.tensor(next_obs, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Current Q-values
        q_values = self.q_network(obs)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Clip rewards to prevent extreme Q-values
        # This stabilizes learning with large-scale trading rewards
        rewards = torch.clamp(rewards, min=-10.0, max=10.0)

        # Double DQN: use Q-network to select actions, target network to evaluate
        with torch.no_grad():
            next_q_values_online = self.q_network(next_obs)
            next_actions = next_q_values_online.argmax(dim=1)

            next_q_values_target = self.target_network(next_obs)
            next_q_values = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)

            # Clip Q-values to prevent explosion
            next_q_values = torch.clamp(next_q_values, min=-100.0, max=100.0)

            target_q_values = rewards + self.config['gamma'] * next_q_values * (1 - dones)
            target_q_values = torch.clamp(target_q_values, min=-100.0, max=100.0)

        # Use Huber loss (smooth L1) instead of MSE for robustness to outliers
        # This prevents loss explosion from extreme Q-value errors
        loss = F.smooth_l1_loss(q_values, target_q_values)

        return loss

    def train_step(self, replay_buffer, batch_size):
        """Perform one training step"""
        if len(replay_buffer) < batch_size:
            return None

        # Sample batch
        batch = replay_buffer.sample(batch_size)

        # Compute loss
        loss = self.compute_loss(batch)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config['max_grad_norm'])
        self.optimizer.step()

        # Update target network
        if self.total_steps % self.config['target_update_freq'] == 0:
            self.update_target_network()

        return loss.item()

    def train(self, env, total_episodes):
        """Main training loop"""
        print("\n" + "=" * 80)
        print("STARTING DQN TRAINING")
        print("=" * 80)

        replay_buffer = ReplayBuffer(
            self.config['buffer_size'],
            env.observation_space.shape
        )

        episode = 0
        patience_counter = 0

        while episode < total_episodes:
            start_time = time.time()

            # Reset environment
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            episode_loss = 0
            loss_count = 0

            done = False
            while not done:
                # Select action
                action = self.select_action(obs)

                # Step environment
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Store in replay buffer
                replay_buffer.add(obs, action, reward, next_obs, done)

                # Train
                if self.total_steps >= self.config['learning_starts']:
                    loss = self.train_step(replay_buffer, self.config['batch_size'])
                    if loss is not None:
                        episode_loss += loss
                        loss_count += 1

                episode_reward += reward
                episode_length += 1
                self.total_steps += 1

                obs = next_obs

            # Decay epsilon
            self.update_epsilon()

            # Track episode
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            episode += 1

            # Logging
            if episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards)
                avg_length = np.mean(self.episode_lengths)
                avg_loss = episode_loss / loss_count if loss_count > 0 else 0

                print(f"\nEpisode {episode}/{total_episodes} ({time.time() - start_time:.1f}s)")
                print(f"  Avg Reward (100): {avg_reward:.2f}")
                print(f"  Avg Length: {avg_length:.1f}")
                print(f"  Avg Loss: {avg_loss:.4f}")
                print(f"  Epsilon: {self.epsilon:.4f}")
                print(f"  Buffer Size: {len(replay_buffer)}/{self.config['buffer_size']}")
                print(f"  Total Steps: {self.total_steps}")

                # TensorBoard
                self.writer.add_scalar('Reward/Average', avg_reward, episode)
                self.writer.add_scalar('Loss/TD', avg_loss, episode)
                self.writer.add_scalar('Epsilon', self.epsilon, episode)
                self.writer.add_scalar('Buffer/Size', len(replay_buffer), episode)

                # Save best model
                if avg_reward > self.best_avg_reward:
                    self.best_avg_reward = avg_reward
                    patience_counter = 0

                    checkpoint = {
                        'episode': episode,
                        'q_network_state_dict': self.q_network.state_dict(),
                        'target_network_state_dict': self.target_network.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'epsilon': self.epsilon,
                        'total_steps': self.total_steps,
                        'avg_reward': avg_reward,
                        'config': self.config
                    }

                    save_path = Path('models') / 'dqn_best.pth'
                    save_path.parent.mkdir(exist_ok=True)
                    torch.save(checkpoint, save_path)
                    print(f"  [SAVED] Best model saved: {save_path} (Reward: {avg_reward:.2f})")
                else:
                    patience_counter += 1

                # Early stopping
                if patience_counter >= self.config['patience'] // 10:  # Check every 10 episodes
                    print(f"\nEarly stopping! No improvement for {patience_counter * 10} episodes.")
                    break

            # Save checkpoint every 100 episodes
            if episode % 100 == 0:
                checkpoint_path = Path('models') / f'dqn_episode_{episode}.pth'
                torch.save({
                    'episode': episode,
                    'q_network_state_dict': self.q_network.state_dict(),
                    'target_network_state_dict': self.target_network.state_dict(),
                    'epsilon': self.epsilon,
                    'total_steps': self.total_steps,
                    'avg_reward': np.mean(self.episode_rewards) if len(self.episode_rewards) > 0 else 0
                }, checkpoint_path)
                print(f"  Checkpoint saved: {checkpoint_path}")

        print("\n" + "=" * 80)
        print("DQN TRAINING COMPLETE")
        print(f"Best Average Reward: {self.best_avg_reward:.2f}")
        print("=" * 80)

        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train DQN agent on trading')

    # Data arguments
    parser.add_argument('--data', type=str, default='TrainingData/features/*.parquet',
                       help='Path pattern to feature files')
    parser.add_argument('--test-mode', action='store_true',
                       help='Use only 10%% of data for quick testing')

    # Network arguments
    parser.add_argument('--hidden-dim', type=int, default=256,
                       help='Hidden layer dimension')

    # Training arguments
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--buffer-size', type=int, default=100000,
                       help='Replay buffer size')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--epsilon-start', type=float, default=1.0,
                       help='Initial epsilon')
    parser.add_argument('--epsilon-end', type=float, default=0.01,
                       help='Final epsilon')
    parser.add_argument('--epsilon-decay', type=float, default=0.995,
                       help='Epsilon decay rate')
    parser.add_argument('--tau', type=float, default=0.005,
                       help='Target network soft update rate')
    parser.add_argument('--target-update-freq', type=int, default=1,
                       help='Target network update frequency')
    parser.add_argument('--learning-starts', type=int, default=1000,
                       help='Steps before learning starts')
    parser.add_argument('--max-grad-norm', type=float, default=10.0,
                       help='Max gradient norm')
    parser.add_argument('--patience', type=int, default=500,
                       help='Early stopping patience')

    # Environment arguments
    parser.add_argument('--initial-capital', type=float, default=100000.0,
                       help='Initial trading capital')
    parser.add_argument('--lookback', type=int, default=60,
                       help='Lookback window for observations')

    # Checkpoint arguments
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')

    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    from glob import glob
    data_files = [Path(f) for f in glob(args.data)]

    if len(data_files) == 0:
        print(f"ERROR: No data files found matching pattern: {args.data}")
        return

    print(f"Found {len(data_files)} data files")

    # Select random stock for training
    if args.test_mode:
        data_files = data_files[:max(1, len(data_files) // 10)]

    selected_file = np.random.choice(data_files)
    print(f"Training on: {selected_file.name}")

    # Load data
    df = pd.read_parquet(selected_file)
    print(f"Loaded {len(df)} timesteps with {df.shape[1]} features")

    # Create environment
    print("\n" + "=" * 80)
    print("CREATING ENVIRONMENT")
    print("=" * 80)

    env = EnhancedTradingEnvironment(
        df=df,
        initial_capital=args.initial_capital,
        lookback_window=args.lookback
    )

    # Create networks
    print("\n" + "=" * 80)
    print("CREATING DQN NETWORKS")
    print("=" * 80)

    obs_shape = env.observation_space.shape
    action_dim = env.action_space.n

    print(f"Observation shape: {obs_shape}")
    print(f"Action dimension: {action_dim}")

    q_network = DQNNetwork(obs_shape, action_dim, args.hidden_dim)
    target_network = DQNNetwork(obs_shape, action_dim, args.hidden_dim)

    total_params = sum(p.numel() for p in q_network.parameters())
    print(f"Total parameters: {total_params:,}")

    # Training config
    config = {
        'action_dim': action_dim,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'buffer_size': args.buffer_size,
        'gamma': args.gamma,
        'epsilon_start': args.epsilon_start,
        'epsilon_end': args.epsilon_end,
        'epsilon_decay': args.epsilon_decay,
        'tau': args.tau,
        'target_update_freq': args.target_update_freq,
        'learning_starts': args.learning_starts,
        'max_grad_norm': args.max_grad_norm,
        'patience': args.patience
    }

    # Create trainer
    trainer = DQNTrainer(q_network, target_network, device, config)

    # Resume if specified
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        checkpoint = torch.load(args.resume)
        q_network.load_state_dict(checkpoint['q_network_state_dict'])
        target_network.load_state_dict(checkpoint['target_network_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.epsilon = checkpoint['epsilon']
        trainer.total_steps = checkpoint['total_steps']
        print(f"Resumed from episode {checkpoint['episode']}")

    # Train
    trainer.train(env, args.episodes)

    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"Best Average Reward: {trainer.best_avg_reward:.2f}")
    print(f"Model saved to: models/dqn_best.pth")
    print("\nNext steps:")
    print("  1. Evaluate: python evaluate_dqn.py")
    print("  2. Compare models: python evaluate_all_models.py")
    print("  3. TensorBoard: tensorboard --logdir logs/tensorboard_dqn")
    print("=" * 80)


if __name__ == '__main__':
    main()
