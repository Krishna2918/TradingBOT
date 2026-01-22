"""
Stable-Baselines3 PPO Training for TradingBOT
==============================================

Replaces custom PPO implementation with battle-tested SB3 for:
- Proper return normalization
- Robust entropy handling
- Verified PPO algorithm implementation
- Better debugging and monitoring

Usage:
    python train_sb3_ppo.py --timesteps 1000000
    python train_sb3_ppo.py --timesteps 500000 --eval-freq 10000

Author: Trading Bot Team
Date: December 2025
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Stable-Baselines3 imports
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
    from stable_baselines3.common.callbacks import (
        EvalCallback,
        CheckpointCallback,
        CallbackList,
        BaseCallback
    )
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.utils import set_random_seed
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("WARNING: stable-baselines3 not installed. Install with: pip install stable-baselines3[extra]")

# Import our trading environment
from src.ai.rl.enhanced_trading_environment import EnhancedTradingEnvironment

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TensorboardCallback(BaseCallback):
    """Custom callback for additional TensorBoard logging"""

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Log additional metrics
        if self.n_calls % 1000 == 0:
            # Log reward statistics
            if len(self.model.ep_info_buffer) > 0:
                ep_rewards = [ep_info['r'] for ep_info in self.model.ep_info_buffer]
                self.logger.record('rollout/ep_rew_mean_custom', np.mean(ep_rewards))
                self.logger.record('rollout/ep_rew_std', np.std(ep_rewards))
        return True


def make_env(data_file: str, rank: int = 0, seed: int = 42, log_dir: Optional[str] = None):
    """Create a wrapped trading environment"""
    def _init():
        # Load data
        df = pd.read_parquet(data_file)

        # Create environment
        env = EnhancedTradingEnvironment(df)

        # Wrap with Monitor for logging
        if log_dir:
            env = Monitor(env, os.path.join(log_dir, f"monitor_{rank}"))
        else:
            env = Monitor(env)

        return env

    set_random_seed(seed + rank)
    return _init


def load_training_data(data_dir: Path, max_files: Optional[int] = None) -> list:
    """Load training data files"""
    feature_files = list(data_dir.glob("*_features.parquet"))

    if not feature_files:
        raise FileNotFoundError(f"No feature files found in {data_dir}")

    logger.info(f"Found {len(feature_files)} feature files")

    if max_files:
        feature_files = feature_files[:max_files]
        logger.info(f"Using {len(feature_files)} files (limited)")

    return feature_files


def create_ppo_model(env, device: str = "auto", tensorboard_log: str = None) -> PPO:
    """Create PPO model with optimized hyperparameters for trading"""

    # Optimized hyperparameters for trading environment
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,  # Standard PPO learning rate
        n_steps=2048,  # Steps per update
        batch_size=64,  # Minibatch size
        n_epochs=10,  # Optimization epochs per update
        gamma=0.99,  # Discount factor
        gae_lambda=0.95,  # GAE lambda
        clip_range=0.2,  # PPO clip range
        clip_range_vf=None,  # Value function clip range (None = no clipping)
        normalize_advantage=True,  # Normalize advantages
        ent_coef=0.05,  # Entropy coefficient (INCREASED for exploration)
        vf_coef=0.5,  # Value function coefficient
        max_grad_norm=0.5,  # Gradient clipping
        use_sde=False,  # State-dependent exploration
        sde_sample_freq=-1,
        target_kl=None,  # Target KL divergence
        stats_window_size=100,  # Window for logging stats
        tensorboard_log=tensorboard_log,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),  # Separate networks
            activation_fn=torch.nn.ReLU,
            ortho_init=True,  # Orthogonal initialization
        ),
        verbose=1,
        device=device,
    )

    return model


def train_sb3_ppo(args):
    """Main training function"""

    if not SB3_AVAILABLE:
        logger.error("stable-baselines3 not installed!")
        return

    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(f"logs/sb3_ppo/{timestamp}")
    log_dir.mkdir(parents=True, exist_ok=True)

    model_dir = Path("models/sb3_ppo")
    model_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = model_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    tensorboard_dir = str(log_dir / "tensorboard")

    logger.info("=" * 80)
    logger.info("STABLE-BASELINES3 PPO TRAINING")
    logger.info("=" * 80)
    logger.info(f"Log directory: {log_dir}")
    logger.info(f"Model directory: {model_dir}")
    logger.info(f"Total timesteps: {args.timesteps:,}")
    logger.info(f"Device: {args.device}")

    # Load training data
    data_dir = Path(args.data_dir)
    feature_files = load_training_data(data_dir, max_files=args.max_files)

    # Select a representative file for training (or use multiple with SubprocVecEnv)
    # For simplicity, we use the first file - extend to multi-file training later
    train_file = str(feature_files[0])
    logger.info(f"Training on: {train_file}")

    # Create vectorized environment
    logger.info("Creating training environment...")
    if args.n_envs > 1:
        # Use multiple parallel environments
        env = SubprocVecEnv([
            make_env(train_file, rank=i, seed=args.seed, log_dir=str(log_dir))
            for i in range(args.n_envs)
        ])
    else:
        # Single environment
        env = DummyVecEnv([make_env(train_file, seed=args.seed, log_dir=str(log_dir))])

    # Normalize observations and rewards (critical for PPO stability)
    env = VecNormalize(
        env,
        norm_obs=True,  # Normalize observations
        norm_reward=True,  # Normalize rewards
        clip_obs=10.0,  # Clip observations
        clip_reward=10.0,  # Clip rewards
        gamma=0.99,  # Discount for reward normalization
        epsilon=1e-8,
    )

    logger.info(f"Environment created with {args.n_envs} parallel envs")
    logger.info(f"Observation space: {env.observation_space}")
    logger.info(f"Action space: {env.action_space}")

    # Create evaluation environment
    eval_env = None
    if args.eval_freq > 0 and len(feature_files) > 1:
        eval_file = str(feature_files[1])  # Use different file for eval
        logger.info(f"Evaluation on: {eval_file}")
        eval_env = DummyVecEnv([make_env(eval_file, seed=args.seed + 100)])
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        # Use same normalization stats as training env
        eval_env.obs_rms = env.obs_rms

    # Create model
    logger.info("Creating PPO model...")
    model = create_ppo_model(env, device=args.device, tensorboard_log=tensorboard_dir)

    total_params = sum(p.numel() for p in model.policy.parameters())
    logger.info(f"Total parameters: {total_params:,}")

    # Setup callbacks
    callbacks = []

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(checkpoint_dir),
        name_prefix="ppo",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)

    # Evaluation callback
    if eval_env is not None:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(model_dir / "best"),
            log_path=str(log_dir / "eval"),
            eval_freq=args.eval_freq,
            n_eval_episodes=5,
            deterministic=True,
            render=False,
        )
        callbacks.append(eval_callback)

    # Custom TensorBoard callback
    callbacks.append(TensorboardCallback())

    callback = CallbackList(callbacks)

    # Train the model
    logger.info("=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80)
    logger.info(f"TensorBoard: tensorboard --logdir {tensorboard_dir}")

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            log_interval=10,
            progress_bar=True,
            reset_num_timesteps=True,
        )
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")

    # Save final model
    final_model_path = model_dir / "ppo_final"
    model.save(str(final_model_path))
    logger.info(f"Final model saved to: {final_model_path}")

    # Save VecNormalize statistics
    vec_normalize_path = model_dir / "vec_normalize.pkl"
    env.save(str(vec_normalize_path))
    logger.info(f"VecNormalize saved to: {vec_normalize_path}")

    # Cleanup
    env.close()
    if eval_env:
        eval_env.close()

    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Model: {final_model_path}")
    logger.info(f"Normalization: {vec_normalize_path}")
    logger.info(f"Logs: {log_dir}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent with Stable-Baselines3")

    parser.add_argument(
        "--timesteps",
        type=int,
        default=1_000_000,
        help="Total training timesteps (default: 1,000,000)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="TrainingData/features",
        help="Directory containing feature parquet files"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of data files to use (default: all)"
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="Number of parallel environments (default: 1)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cuda, cpu)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=10000,
        help="Evaluation frequency (0 to disable)"
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=50000,
        help="Checkpoint save frequency"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to model to resume training from"
    )

    args = parser.parse_args()

    # Check for SB3
    if not SB3_AVAILABLE:
        print("\nERROR: stable-baselines3 is not installed.")
        print("Install with: pip install stable-baselines3[extra]")
        sys.exit(1)

    # Run training
    train_sb3_ppo(args)


if __name__ == "__main__":
    main()
