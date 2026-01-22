"""
ADVANCED PPO TRAINING - Full Power with Resource Limits
40% VRAM/GPU, 20% RAM minimum
Removes TensorBoard to avoid TensorFlow conflicts

Usage:
    python train_ppo_no_tf_gpu.py --episodes 500
"""

import os

print("=" * 80)
print("FULL-POWERED PPO TRAINER - 40% GPU, 20% RAM MIN")
print("=" * 80)

import sys
import time
import argparse
import warnings
import psutil
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

# Resource limits
TOTAL_RAM_GB = psutil.virtual_memory().total / (1024**3)
RAM_MIN_GB = TOTAL_RAM_GB * 0.2  # 20% minimum
print(f"RAM Target: {RAM_MIN_GB:.1f}GB minimum / {TOTAL_RAM_GB:.1f}GB (20% min)")

# GPU Configuration (PyTorch only) - 60% VRAM
torch.backends.cudnn.benchmark = True
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"PyTorch Device: {device}")

if torch.cuda.is_available():
    gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    gpu_limit = gpu_total * 0.4  # 40% VRAM
    print(f"GPU VRAM Limit: {gpu_limit:.1f}GB / {gpu_total:.1f}GB (40%)")
    torch.cuda.set_per_process_memory_fraction(0.4, 0)

# NOW import the environment (with __init__.py disabled to avoid TensorFlow)
print("\nImporting trading environment...")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.ai.rl.enhanced_trading_environment import EnhancedTradingEnvironment
import gymnasium as gym
print("Environment loaded successfully!")


# TimeLimit wrapper to ensure episodes terminate
class TimeLimitWrapper(gym.Wrapper):
    def __init__(self, env, max_episode_steps=1000):
        super().__init__(env)
        self.max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._elapsed_steps += 1

        # Truncate if max steps reached
        if self._elapsed_steps >= self.max_episode_steps:
            truncated = True

        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

    def __getattr__(self, name):
        # Delegate all other attributes to the wrapped environment
        return getattr(self.env, name)


class PPONetwork(nn.Module):
    """Actor-Critic network for PPO"""

    def __init__(self, obs_shape: Tuple, action_dim: int, hidden_dim: int = 256):
        super(PPONetwork, self).__init__()

        self.obs_shape = obs_shape
        input_dim = obs_shape[0] * obs_shape[1]

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Policy head outputs logits (no softmax here)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

        # Policy head final layer needs smaller initialization for entropy
        # This prevents extreme logit values that cause zero entropy
        for module in self.policy_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = obs.shape[0]
        obs_flat = obs.view(batch_size, -1)
        features = self.feature_extractor(obs_flat)
        action_logits = self.policy_head(features)

        # Clip logits to prevent extreme values that cause zero entropy
        # This maintains exploration by keeping the distribution reasonable
        action_logits = torch.clamp(action_logits, min=-10.0, max=10.0)

        state_value = self.value_head(features)
        return action_logits, state_value

    def get_action(self, obs: torch.Tensor, deterministic: bool = False, exploration_noise: float = 0.0):
        action_logits, state_value = self.forward(obs)

        # Add exploration noise to logits (safer than temperature scaling)
        if exploration_noise > 0 and not deterministic:
            noise = torch.randn_like(action_logits) * exploration_noise
            action_logits = action_logits + noise

        dist = Categorical(logits=action_logits)
        if deterministic:
            action = torch.argmax(action_logits, dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, state_value


class RolloutBuffer:
    def __init__(self, buffer_size: int, obs_shape: Tuple, device: torch.device):
        self.buffer_size = buffer_size
        self.obs_shape = obs_shape
        self.device = device
        # CPU-only allocation (35% faster, no deadlocks)
        # Networks stay on GPU for computation
        self.observations = torch.zeros((buffer_size, *obs_shape), dtype=torch.float32)
        self.actions = torch.zeros(buffer_size, dtype=torch.long)
        self.log_probs = torch.zeros(buffer_size, dtype=torch.float32)
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32)
        self.values = torch.zeros(buffer_size, dtype=torch.float32)
        self.dones = torch.zeros(buffer_size, dtype=torch.float32)
        self.advantages = torch.zeros(buffer_size, dtype=torch.float32)
        self.returns = torch.zeros(buffer_size, dtype=torch.float32)
        self.ptr = 0

    def add(self, obs, action, log_prob, reward, value, done):
        if self.ptr < self.buffer_size:
            self.observations[self.ptr] = torch.tensor(obs, dtype=torch.float32)
            self.actions[self.ptr] = action
            self.log_probs[self.ptr] = log_prob
            self.rewards[self.ptr] = reward
            self.values[self.ptr] = value
            self.dones[self.ptr] = done
            self.ptr += 1

    def compute_gae(self, last_value: float, gamma: float = 0.99, gae_lambda: float = 0.95):
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
        self.returns = self.advantages + self.values[:self.ptr]
        self.advantages[:self.ptr] = (
            (self.advantages[:self.ptr] - self.advantages[:self.ptr].mean()) /
            (self.advantages[:self.ptr].std() + 1e-8)
        )

    def get(self):
        # Transfer to GPU when needed for training
        return (
            self.observations[:self.ptr].to(self.device),
            self.actions[:self.ptr].to(self.device),
            self.log_probs[:self.ptr].to(self.device),
            self.advantages[:self.ptr].to(self.device),
            self.returns[:self.ptr].to(self.device)
        )

    def clear(self):
        self.ptr = 0


class PPOTrainer:
    def __init__(self, network, device, config):
        self.network = network.to(device)
        self.device = device
        self.config = config
        self.optimizer = optim.Adam(network.parameters(), lr=config['learning_rate'], eps=1e-5)
        self.episode_rewards = deque(maxlen=100)
        self.best_avg_reward = -float('inf')

        # Exploration noise scheduling (decay from 1.0 to 0.1)
        self.exploration_noise_start = 1.0
        self.exploration_noise_end = 0.1
        self.current_exploration_noise = self.exploration_noise_start

        # Adaptive entropy coefficient
        self.entropy_target = 0.5  # Target entropy value
        self.current_entropy_coef = config['entropy_coef']

        # Logging to file instead of TensorBoard
        self.log_file = Path('logs') / f"ppo_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.log_file.parent.mkdir(exist_ok=True)
        print(f"Log file: {self.log_file}")

        # Verify GPU usage
        if device.type == 'cuda':
            print(f"Network on GPU: {next(network.parameters()).is_cuda}")
            print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f}GB")

    def collect_rollouts(self, env, buffer, n_steps):
        obs, _ = env.reset()
        episode_reward = 0
        steps_collected = 0
        for step in range(n_steps):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            with torch.no_grad():
                # Use current exploration noise for action selection
                action, log_prob, value = self.network.get_action(
                    obs_tensor,
                    exploration_noise=self.current_exploration_noise
                )
            action_np = action.item()
            log_prob_np = log_prob.item()
            value_np = value.item()
            next_obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated
            buffer.add(obs, action_np, log_prob_np, reward, value_np, done)
            steps_collected += 1
            episode_reward += reward
            obs = next_obs
            if done:
                self.episode_rewards.append(episode_reward)
                obs, _ = env.reset()
                episode_reward = 0
        return steps_collected

    def update_policy(self, buffer):
        observations, actions, old_log_probs, advantages, returns = buffer.get()

        total_entropy = 0.0
        num_updates = 0

        for epoch in range(self.config['n_epochs']):
            action_logits, values = self.network(observations)

            # No temperature or noise in update (causes NaN)
            dist = Categorical(logits=action_logits)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            total_entropy += entropy.item()
            num_updates += 1

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config['clip_epsilon'], 1 + self.config['clip_epsilon']) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Clip value loss to prevent explosion
            value_pred_clipped = values.squeeze()
            value_loss_unclipped = (returns - value_pred_clipped).pow(2)
            value_loss = torch.clamp(value_loss_unclipped, max=1000.0).mean()

            # Use adaptive entropy coefficient
            loss = policy_loss + self.config['value_coef'] * value_loss - self.current_entropy_coef * entropy
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config['max_grad_norm'])
            self.optimizer.step()

        avg_entropy = total_entropy / num_updates

        # Adapt entropy coefficient based on target
        if avg_entropy < self.entropy_target:
            self.current_entropy_coef = min(self.current_entropy_coef * 1.01, 0.2)
        else:
            self.current_entropy_coef = max(self.current_entropy_coef * 0.99, 0.001)

        return policy_loss.item(), value_loss.item(), avg_entropy

    def train(self, env, total_episodes):
        print("\n" + "=" * 80)
        print("STARTING FULL-POWERED PPO TRAINING")
        print(f"Target: {total_episodes} episodes")
        print("=" * 80)

        buffer = RolloutBuffer(self.config['n_steps'], env.observation_space.shape, self.device)

        # Use iteration-based training (avoids episode counting bug)
        # Each iteration = 1 rollout collection + policy update
        # Estimate ~3-5 iterations per episode on average
        max_iterations = total_episodes * 5

        iteration = 0
        total_steps = 0
        episodes_completed = 0

        # Write CSV header
        with open(self.log_file, 'w') as f:
            f.write("iteration,episodes_completed,avg_reward,policy_loss,value_loss,entropy,ram_gb,gpu_util_pct\n")

        print(f"Starting {max_iterations} iterations (estimated for {total_episodes} episodes)\n")

        while iteration < max_iterations and episodes_completed < total_episodes:
            iteration += 1
            start_time = time.time()

            # Collect rollouts
            steps_collected = self.collect_rollouts(env, buffer, self.config['n_steps'])
            total_steps += steps_collected

            # Track episodes completed
            episodes_before = episodes_completed
            episodes_completed = len([r for r in self.episode_rewards if r is not None])
            new_episodes = episodes_completed - episodes_before

            # Compute advantages
            with torch.no_grad():
                last_obs = torch.FloatTensor(env._get_observation()).unsqueeze(0).to(self.device)
                _, _, last_value = self.network.get_action(last_obs)
                buffer.compute_gae(last_value.item(), self.config['gamma'], self.config['gae_lambda'])

            # Update policy
            policy_loss, value_loss, entropy = self.update_policy(buffer)
            buffer.clear()

            # Decay exploration noise (linear decay)
            progress = iteration / max_iterations
            self.current_exploration_noise = self.exploration_noise_start - (self.exploration_noise_start - self.exploration_noise_end) * progress

            # Progress reporting every 10 iterations
            if iteration % 10 == 0:
                elapsed = time.time() - start_time
                avg_reward = np.mean(self.episode_rewards) if len(self.episode_rewards) > 0 else 0.0

                # Resource monitoring
                ram_used = psutil.virtual_memory().used / (1024**3)
                ram_pct = (ram_used / TOTAL_RAM_GB) * 100

                gpu_util = 0.0
                if torch.cuda.is_available():
                    gpu_mem_used = torch.cuda.memory_allocated(0) / (1024**3)
                    gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    gpu_util = (gpu_mem_used / gpu_mem_total) * 100

                print(f"\nIteration {iteration}/{max_iterations} | Episodes: {episodes_completed}/{total_episodes} ({elapsed:.1f}s)")
                print(f"  Avg Reward (last 100): {avg_reward:.2f}")
                print(f"  Loss - Policy: {policy_loss:.4f}, Value: {value_loss:.4f}, Entropy: {entropy:.4f}")
                print(f"  Exploration: {self.current_exploration_noise:.3f} | Entropy Coef: {self.current_entropy_coef:.4f}")
                print(f"  RAM: {ram_used:.1f}GB ({ram_pct:.0f}%) | GPU VRAM: {gpu_util:.0f}%")

                # Log to CSV
                with open(self.log_file, 'a') as f:
                    f.write(f"{iteration},{episodes_completed},{avg_reward:.4f},{policy_loss:.6f},{value_loss:.6f},{entropy:.6f},{ram_used:.2f},{gpu_util:.1f}\n")

                # Save best model
                if avg_reward > self.best_avg_reward and len(self.episode_rewards) >= 10:
                    self.best_avg_reward = avg_reward
                    Path('models').mkdir(exist_ok=True)
                    torch.save({
                        'iteration': iteration,
                        'episodes': episodes_completed,
                        'model_state_dict': self.network.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'reward': avg_reward
                    }, 'models/ppo_best.pth')
                    print(f"  [SAVED] Best model (reward: {avg_reward:.2f})")

            # Checkpoint every 100 iterations (more reliable than episodes)
            if iteration % 100 == 0 and iteration > 0:
                Path('models').mkdir(exist_ok=True)
                torch.save({
                    'iteration': iteration,
                    'episodes': episodes_completed,
                    'model_state_dict': self.network.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()
                }, f'models/ppo_iter_{iteration}.pth')
                print(f"  [CHECKPOINT] Saved at iteration {iteration}")

            # Also checkpoint every 100 episodes if they complete
            if episodes_completed % 100 == 0 and episodes_completed > 0 and new_episodes > 0:
                Path('models').mkdir(exist_ok=True)
                torch.save({
                    'iteration': iteration,
                    'episodes': episodes_completed,
                    'model_state_dict': self.network.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()
                }, f'models/ppo_episode_{episodes_completed}.pth')
                print(f"  [CHECKPOINT] Saved at {episodes_completed} episodes")

        print("\n" + "=" * 80)
        print(f"TRAINING COMPLETE")
        print(f"  Iterations: {iteration}")
        print(f"  Episodes: {episodes_completed}")
        print(f"  Best Avg Reward: {self.best_avg_reward:.2f}")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='TrainingData/features')
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--n-steps', type=int, default=2048)
    parser.add_argument('--n-epochs', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--clip-epsilon', type=float, default=0.2)
    parser.add_argument('--value-coef', type=float, default=0.5)
    parser.add_argument('--entropy-coef', type=float, default=0.05)  # Increased for better exploration
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--initial-capital', type=float, default=100000.0)
    parser.add_argument('--lookback', type=int, default=60)
    args = parser.parse_args()

    from glob import glob
    data_files = [Path(f) for f in glob(f"{args.data}/*.parquet")]
    if len(data_files) == 0:
        print(f"ERROR: No data files in {args.data}")
        return
    print(f"Found {len(data_files)} data files")
    selected_file = np.random.choice(data_files)
    print(f"Training on: {selected_file.name}")
    df = pd.read_parquet(selected_file)
    print(f"Loaded {len(df)} timesteps, {df.shape[1]} features")

    print("\n" + "=" * 80)
    print("CREATING ENHANCED TRADING ENVIRONMENT")
    print("=" * 80)
    base_env = EnhancedTradingEnvironment(df=df, initial_capital=args.initial_capital, lookback_window=args.lookback)
    # Wrap with TimeLimit to ensure episodes terminate after 1000 steps
    env = TimeLimitWrapper(base_env, max_episode_steps=1000)
    print("Applied TimeLimit wrapper (max 1000 steps per episode)")

    print("\n" + "=" * 80)
    print("CREATING PPO NETWORK")
    print("=" * 80)
    obs_shape = env.observation_space.shape
    action_dim = env.action_space.n
    print(f"Observation: {obs_shape}, Actions: {action_dim}")
    network = PPONetwork(obs_shape, action_dim, hidden_dim=256)
    print(f"Parameters: {sum(p.numel() for p in network.parameters()):,}")

    config = {
        'n_steps': args.n_steps, 'n_epochs': args.n_epochs, 'learning_rate': args.learning_rate,
        'gamma': args.gamma, 'gae_lambda': args.gae_lambda, 'clip_epsilon': args.clip_epsilon,
        'value_coef': args.value_coef, 'entropy_coef': args.entropy_coef, 'max_grad_norm': args.max_grad_norm
    }

    trainer = PPOTrainer(network, device, config)
    trainer.train(env, args.episodes)
    print(f"\nBest model: models/ppo_best.pth")


if __name__ == '__main__':
    main()
