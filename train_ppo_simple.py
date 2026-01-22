"""
SIMPLIFIED PPO TRAINING - 40% Resource Usage
Fixed initialization issues, cleaner implementation

Usage:
    python train_ppo_simple.py --episodes 500
"""

import os
import sys
import time
import psutil
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
from gymnasium import spaces

print("=" * 80)
print("SIMPLIFIED PPO TRAINER - 40% RESOURCES")
print("=" * 80)

# Set resource limits (40% of available)
import resource_control
TOTAL_RAM_GB = psutil.virtual_memory().total / (1024**3)
RAM_LIMIT_GB = TOTAL_RAM_GB * 0.4
print(f"RAM Limit: {RAM_LIMIT_GB:.1f}GB / {TOTAL_RAM_GB:.1f}GB (40%)")

# GPU setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

if torch.cuda.is_available():
    gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    gpu_limit = gpu_total * 0.4
    print(f"GPU Limit: {gpu_limit:.1f}GB / {gpu_total:.1f}GB (40%)")
    # Set PyTorch memory limit
    torch.cuda.set_per_process_memory_fraction(0.4, 0)


class SimpleTradingEnv(gym.Env):
    """Simplified trading environment - no complex features"""

    def __init__(self, df, initial_capital=100000, lookback=30):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.initial_capital = initial_capital
        self.lookback = lookback

        # Observation: [lookback x features] flattened
        self.n_features = min(df.shape[1], 10)  # Use only first 10 features
        obs_size = self.lookback * self.n_features

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_size,), dtype=np.float32
        )

        # Actions: 0=hold, 1=buy, 2=sell
        self.action_space = spaces.Discrete(3)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = self.lookback
        self.cash = self.initial_capital
        self.shares = 0
        self.portfolio_value = self.initial_capital

        return self._get_observation(), {}

    def _get_observation(self):
        """Get current observation"""
        # Get last lookback rows
        start_idx = max(0, self.current_step - self.lookback)
        end_idx = self.current_step

        window = self.df.iloc[start_idx:end_idx, :self.n_features].values

        # Normalize
        if window.std() > 0:
            window = (window - window.mean()) / (window.std() + 1e-8)

        # Pad if needed
        if window.shape[0] < self.lookback:
            padding = np.zeros((self.lookback - window.shape[0], self.n_features))
            window = np.vstack([padding, window])

        return window.flatten().astype(np.float32)

    def step(self, action):
        """Execute one step"""
        prev_value = self.portfolio_value

        # Get current price (assume 'close' is column 3 or last)
        if 'close' in self.df.columns:
            price = self.df.iloc[self.current_step]['close']
        else:
            price = self.df.iloc[self.current_step, 3]  # Column 3

        # Execute action
        if action == 1:  # Buy
            shares_to_buy = int(self.cash / price) if price > 0 else 0
            cost = shares_to_buy * price
            self.cash -= cost
            self.shares += shares_to_buy

        elif action == 2:  # Sell
            revenue = self.shares * price
            self.cash += revenue
            self.shares = 0

        # Update portfolio value
        self.portfolio_value = self.cash + (self.shares * price)

        # Calculate reward (simple return)
        reward = (self.portfolio_value - prev_value) / prev_value if prev_value > 0 else 0

        # Move to next step
        self.current_step += 1

        # Check if done
        terminated = (self.current_step >= len(self.df) - 1)
        truncated = False

        obs = self._get_observation() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)

        info = {
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'shares': self.shares
        }

        return obs, reward, terminated, truncated, info


class PPONetwork(nn.Module):
    """Simple PPO network"""

    def __init__(self, obs_dim, action_dim, hidden=128):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )

        self.actor = nn.Sequential(
            nn.Linear(hidden, action_dim),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        features = self.shared(x)
        action_probs = self.actor(features)
        value = self.critic(features)
        return action_probs, value


def train_ppo(env, network, optimizer, episodes=500, steps_per_episode=2048):
    """Train PPO agent"""

    print("\n" + "=" * 80)
    print("STARTING PPO TRAINING")
    print(f"Episodes: {episodes}")
    print(f"Steps per episode: {steps_per_episode}")
    print("=" * 80 + "\n")

    best_reward = -float('inf')
    episode_rewards = []

    for episode in range(episodes):
        obs, _ = env.reset()
        obs = torch.FloatTensor(obs).to(device)

        # Collect rollout
        observations = []
        actions = []
        log_probs = []
        rewards = []
        values = []

        episode_reward = 0

        for step in range(steps_per_episode):
            # Get action
            action_probs, value = network(obs.unsqueeze(0))
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            # Step environment
            obs_np, reward, terminated, truncated, info = env.step(action.item())

            # Store
            observations.append(obs)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)

            episode_reward += reward

            if terminated or truncated:
                break

            obs = torch.FloatTensor(obs_np).to(device)

        episode_rewards.append(episode_reward)

        # Calculate returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + 0.99 * R
            returns.insert(0, R)

        returns = torch.FloatTensor(returns).to(device)

        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Update policy
        observations = torch.stack(observations)
        actions = torch.stack(actions)
        log_probs = torch.stack(log_probs)
        values = torch.cat([v for v in values])

        advantages = returns - values.detach()

        # PPO update
        action_probs, new_values = network(observations)
        dist = Categorical(action_probs)
        new_log_probs = dist.log_prob(actions)

        ratio = (new_log_probs - log_probs).exp()
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages

        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = (returns - new_values.squeeze()).pow(2).mean()

        loss = actor_loss + 0.5 * critic_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), 0.5)
        optimizer.step()

        # Logging
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode+1}/{episodes} | Avg Reward: {avg_reward:.4f} | "
                  f"Portfolio: ${info['portfolio_value']:,.0f}")

            # Save best model
            if avg_reward > best_reward:
                best_reward = avg_reward
                torch.save({
                    'episode': episode,
                    'model_state_dict': network.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'reward': best_reward
                }, 'models/ppo_best.pth')
                print(f"  [SAVED] Best model (reward: {best_reward:.4f})")

        # Check RAM usage
        ram_used = psutil.virtual_memory().used / (1024**3)
        if ram_used > RAM_LIMIT_GB:
            print(f"  [WARNING] RAM usage {ram_used:.1f}GB exceeds limit {RAM_LIMIT_GB:.1f}GB")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print(f"Best Reward: {best_reward:.4f}")
    print("=" * 80)

    return episode_rewards


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--data', type=str, default='TrainingData/features')
    args = parser.parse_args()

    # Load data
    print("\nLoading data...")
    from glob import glob
    data_files = list(Path(args.data).glob('*.parquet'))

    if len(data_files) == 0:
        print(f"ERROR: No data files found in {args.data}")
        sys.exit(1)

    # Select random stock
    selected_file = np.random.choice(data_files)
    print(f"Training on: {selected_file.name}")

    df = pd.read_parquet(selected_file)
    print(f"Loaded: {len(df)} rows, {df.shape[1]} features")

    # Create environment
    print("\nCreating environment...")
    env = SimpleTradingEnv(df, initial_capital=100000, lookback=30)
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.n}")

    # Create network
    print("\nCreating network...")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    network = PPONetwork(obs_dim, action_dim, hidden=128).to(device)

    total_params = sum(p.numel() for p in network.parameters())
    print(f"Parameters: {total_params:,}")

    # Create optimizer
    optimizer = optim.Adam(network.parameters(), lr=3e-4)

    # Train
    rewards = train_ppo(env, network, optimizer, episodes=args.episodes)

    print(f"\nTraining complete! Best model saved to: models/ppo_best.pth")
