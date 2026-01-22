"""
PRODUCTION PPO AGENT TRAINER - Optimized for RTX 4080
Trains PPO (Proximal Policy Optimization) agent for trading

Features:
- GPU-accelerated training
- Rollout buffer with GAE (Generalized Advantage Estimation)
- Clipped surrogate objective
- Value function learning
- Entropy regularization
- Early stopping based on episode rewards
- TensorBoard logging

Usage:
    # Training from scratch
    python train_ppo_agent.py --data TrainingData/features/*.parquet --episodes 1000

    # Resume from checkpoint
    python train_ppo_agent.py --resume models/ppo_checkpoint.pth

    # Quick test
    python train_ppo_agent.py --test-mode --episodes 100

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

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

# Import custom environment
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.ai.rl.enhanced_trading_environment import EnhancedTradingEnvironment

# GPU Configuration
torch.backends.cudnn.benchmark = True
torch.manual_seed(42)
np.random.seed(42)


class PPONetwork(nn.Module):
    """Actor-Critic network for PPO"""

    def __init__(self, obs_shape: Tuple, action_dim: int, hidden_dim: int = 256):
        super(PPONetwork, self).__init__()

        # Calculate input dimension (flatten observation)
        self.obs_shape = obs_shape  # (lookback_window, features)
        input_dim = obs_shape[0] * obs_shape[1]

        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Policy head (actor)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1)
        )

        # Value head (critic)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            obs: Observation tensor [batch, lookback_window, features]

        Returns:
            action_probs: [batch, action_dim]
            state_value: [batch, 1]
        """
        # Flatten observation
        batch_size = obs.shape[0]
        obs_flat = obs.view(batch_size, -1)

        # Extract features
        features = self.feature_extractor(obs_flat)

        # Get action probabilities and state value
        action_probs = self.policy_head(features)
        state_value = self.value_head(features)

        return action_probs, state_value

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        """Sample action from policy"""
        action_probs, state_value = self.forward(obs)

        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
        else:
            dist = Categorical(action_probs)
            action = dist.sample()

        log_prob = torch.log(action_probs.gather(1, action.unsqueeze(1)) + 1e-8)

        return action, log_prob, state_value


class RolloutBuffer:
    """Storage for PPO rollouts with GAE calculation"""

    def __init__(self, buffer_size: int, obs_shape: Tuple, device: torch.device):
        self.buffer_size = buffer_size
        self.obs_shape = obs_shape
        self.device = device

        # Storage
        self.observations = torch.zeros((buffer_size, *obs_shape), dtype=torch.float32)
        self.actions = torch.zeros(buffer_size, dtype=torch.long)
        self.log_probs = torch.zeros(buffer_size, dtype=torch.float32)
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32)
        self.values = torch.zeros(buffer_size, dtype=torch.float32)
        self.dones = torch.zeros(buffer_size, dtype=torch.float32)
        self.advantages = torch.zeros(buffer_size, dtype=torch.float32)
        self.returns = torch.zeros(buffer_size, dtype=torch.float32)

        self.ptr = 0
        self.path_start = 0

    def add(self, obs, action, log_prob, reward, value, done):
        """Add experience to buffer"""
        if self.ptr < self.buffer_size:
            self.observations[self.ptr] = torch.tensor(obs, dtype=torch.float32)
            self.actions[self.ptr] = action
            self.log_probs[self.ptr] = log_prob
            self.rewards[self.ptr] = reward
            self.values[self.ptr] = value
            self.dones[self.ptr] = done
            self.ptr += 1

    def compute_gae(self, last_value: float, gamma: float = 0.99, gae_lambda: float = 0.95):
        """Compute Generalized Advantage Estimation"""
        last_gae = 0

        for t in reversed(range(self.ptr)):
            if t == self.ptr - 1:
                next_value = last_value
                next_done = 0.0
            else:
                next_value = self.values[t + 1]
                next_done = self.dones[t + 1]

            delta = self.rewards[t] + gamma * next_value * (1 - next_done) - self.values[t]
            last_gae = delta + gamma * gae_lambda * (1 - next_done) * last_gae
            self.advantages[t] = last_gae

        # Calculate returns (for value loss)
        self.returns = self.advantages + self.values[:self.ptr]

        # Normalize advantages
        self.advantages[:self.ptr] = (
            (self.advantages[:self.ptr] - self.advantages[:self.ptr].mean()) /
            (self.advantages[:self.ptr].std() + 1e-8)
        )

    def get(self):
        """Get all data from buffer"""
        return (
            self.observations[:self.ptr].to(self.device),
            self.actions[:self.ptr].to(self.device),
            self.log_probs[:self.ptr].to(self.device),
            self.advantages[:self.ptr].to(self.device),
            self.returns[:self.ptr].to(self.device)
        )

    def clear(self):
        """Clear buffer"""
        self.ptr = 0
        self.path_start = 0


class PPOTrainer:
    """PPO trainer with GPU optimization"""

    def __init__(self, network, device, config):
        self.network = network.to(device)
        self.device = device
        self.config = config

        # Optimizer
        self.optimizer = optim.Adam(
            network.parameters(),
            lr=config['learning_rate'],
            eps=1e-5
        )

        # Tracking
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.best_avg_reward = -float('inf')

        # TensorBoard
        log_dir = Path('logs/tensorboard_ppo') / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(log_dir)
        print(f"TensorBoard logging to: {log_dir}")

    def collect_rollouts(self, env, buffer, n_steps):
        """Collect rollouts from environment"""
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0

        for step in range(n_steps):
            # Get action
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)

            with torch.no_grad():
                action, log_prob, value = self.network.get_action(obs_tensor)

            action_item = action.item()
            log_prob_item = log_prob.item()
            value_item = value.item()

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action_item)
            done = terminated or truncated

            # Store in buffer
            buffer.add(obs, action_item, log_prob_item, reward, value_item, float(done))

            episode_reward += reward
            episode_length += 1

            # Episode done
            if done:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)

                obs, _ = env.reset()
                episode_reward = 0
                episode_length = 0
            else:
                obs = next_obs

        # Compute advantages
        if done:
            last_value = 0.0
        else:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, _, last_value_tensor = self.network.get_action(obs_tensor)
                last_value = last_value_tensor.item()

        buffer.compute_gae(last_value, self.config['gamma'], self.config['gae_lambda'])

        return buffer.ptr

    def update_policy(self, buffer, n_epochs):
        """Update policy using collected rollouts"""
        obs, actions, old_log_probs, advantages, returns = buffer.get()

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0

        for epoch in range(n_epochs):
            # Forward pass
            action_probs, values = self.network.forward(obs)
            values = values.squeeze()

            # Calculate log probs
            dist = Categorical(action_probs)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            # Policy loss (clipped)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config['clip_epsilon'], 1 + self.config['clip_epsilon']) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = nn.MSELoss()(values, returns)

            # Total loss
            loss = (
                policy_loss +
                self.config['value_coef'] * value_loss -
                self.config['entropy_coef'] * entropy
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.config['max_grad_norm'])
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()

        return (
            total_policy_loss / n_epochs,
            total_value_loss / n_epochs,
            total_entropy / n_epochs
        )

    def train(self, env, total_episodes):
        """Main training loop"""
        print("\n" + "=" * 80)
        print("STARTING PPO TRAINING")
        print("=" * 80)

        buffer = RolloutBuffer(
            self.config['n_steps'],
            env.observation_space.shape,
            self.device
        )

        episode = 0
        total_steps = 0
        patience_counter = 0

        while episode < total_episodes:
            start_time = time.time()

            # Collect rollouts
            steps_collected = self.collect_rollouts(env, buffer, self.config['n_steps'])
            total_steps += steps_collected

            # Update policy
            policy_loss, value_loss, entropy = self.update_policy(buffer, self.config['n_epochs'])

            # Clear buffer
            buffer.clear()

            # Update episode count
            episode = len(self.episode_rewards)

            # Logging
            if len(self.episode_rewards) > 0:
                avg_reward = np.mean(self.episode_rewards)
                avg_length = np.mean(self.episode_lengths)

                print(f"\nEpisode {episode}/{total_episodes} ({time.time() - start_time:.1f}s)")
                print(f"  Avg Reward: {avg_reward:.2f}")
                print(f"  Avg Length: {avg_length:.1f}")
                print(f"  Policy Loss: {policy_loss:.4f}")
                print(f"  Value Loss: {value_loss:.4f}")
                print(f"  Entropy: {entropy:.4f}")

                # TensorBoard
                self.writer.add_scalar('Reward/Average', avg_reward, episode)
                self.writer.add_scalar('Loss/Policy', policy_loss, episode)
                self.writer.add_scalar('Loss/Value', value_loss, episode)
                self.writer.add_scalar('Entropy', entropy, episode)

                # Save best model
                if avg_reward > self.best_avg_reward:
                    self.best_avg_reward = avg_reward
                    patience_counter = 0

                    checkpoint = {
                        'episode': episode,
                        'network_state_dict': self.network.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'avg_reward': avg_reward,
                        'config': self.config
                    }

                    save_path = Path('models') / 'ppo_best.pth'
                    save_path.parent.mkdir(exist_ok=True)
                    torch.save(checkpoint, save_path)
                    print(f"  ✓ Best model saved: {save_path} (Reward: {avg_reward:.2f})")
                else:
                    patience_counter += 1

                # Early stopping
                if patience_counter >= self.config['patience']:
                    print(f"\nEarly stopping! No improvement for {self.config['patience']} episodes.")
                    break

            # Save checkpoint every 100 episodes
            if episode % 100 == 0 and episode > 0:
                checkpoint_path = Path('models') / f'ppo_episode_{episode}.pth'
                torch.save({
                    'episode': episode,
                    'network_state_dict': self.network.state_dict(),
                    'avg_reward': np.mean(self.episode_rewards) if len(self.episode_rewards) > 0 else 0
                }, checkpoint_path)
                print(f"  Checkpoint saved: {checkpoint_path}")

        print("\n" + "=" * 80)
        print("PPO TRAINING COMPLETE")
        print(f"Best Average Reward: {self.best_avg_reward:.2f}")
        print("=" * 80)

        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train PPO agent on trading')

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
    parser.add_argument('--n-steps', type=int, default=2048,
                       help='Steps per rollout')
    parser.add_argument('--n-epochs', type=int, default=10,
                       help='Epochs per update')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                       help='GAE lambda')
    parser.add_argument('--clip-epsilon', type=float, default=0.2,
                       help='PPO clipping parameter')
    parser.add_argument('--value-coef', type=float, default=0.5,
                       help='Value loss coefficient')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                       help='Entropy coefficient')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                       help='Max gradient norm')
    parser.add_argument('--patience', type=int, default=50,
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

    # Create network
    print("\n" + "=" * 80)
    print("CREATING PPO NETWORK")
    print("=" * 80)

    obs_shape = env.observation_space.shape
    action_dim = env.action_space.n

    print(f"Observation shape: {obs_shape}")
    print(f"Action dimension: {action_dim}")

    network = PPONetwork(obs_shape, action_dim, args.hidden_dim)

    total_params = sum(p.numel() for p in network.parameters())
    print(f"Total parameters: {total_params:,}")

    # Training config
    config = {
        'learning_rate': args.learning_rate,
        'n_steps': args.n_steps,
        'n_epochs': args.n_epochs,
        'gamma': args.gamma,
        'gae_lambda': args.gae_lambda,
        'clip_epsilon': args.clip_epsilon,
        'value_coef': args.value_coef,
        'entropy_coef': args.entropy_coef,
        'max_grad_norm': args.max_grad_norm,
        'patience': args.patience
    }

    # Create trainer
    trainer = PPOTrainer(network, device, config)

    # Resume if specified
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        checkpoint = torch.load(args.resume)
        network.load_state_dict(checkpoint['network_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Resumed from episode {checkpoint['episode']}")

    # Train
    trainer.train(env, args.episodes)

    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"Best Average Reward: {trainer.best_avg_reward:.2f}")
    print(f"Model saved to: models/ppo_best.pth")
    print("\nNext steps:")
    print("  1. Evaluate: python evaluate_ppo.py")
    print("  2. Compare models: python evaluate_all_models.py")
    print("  3. TensorBoard: tensorboard --logdir logs/tensorboard_ppo")
    print("=" * 80)


if __name__ == '__main__':
    main()
