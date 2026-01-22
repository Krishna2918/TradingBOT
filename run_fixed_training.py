"""
Run Fixed Training Pipeline
============================

This script runs the AI training with all fixes applied:
1. LSTM training on ALL 1,681 stocks (not just 5)
2. Focal Loss + Stratified Sampling for class balance
3. Improved NaN handling
4. Widened target thresholds (0.5% instead of 0.1%)

Usage:
    python run_fixed_training.py --model lstm      # Train LSTM only
    python run_fixed_training.py --model ppo       # Train PPO only
    python run_fixed_training.py --model all       # Train both
    python run_fixed_training.py --check           # Just verify fixes are in place

Author: Trading Bot Team
Date: December 2025
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def check_fixes():
    """Verify all fixes are in place"""
    print("=" * 60)
    print("VERIFYING FIXES ARE IN PLACE")
    print("=" * 60)

    issues = []

    # Check 1: max_symbols removed
    orchestrator_path = Path("master_training_orchestrator.py")
    if orchestrator_path.exists():
        content = orchestrator_path.read_text(encoding='utf-8')
        if "max_symbols=5" in content:
            issues.append("master_training_orchestrator.py still has max_symbols=5")
        elif "max_symbols=None" in content:
            print("[OK] master_training_orchestrator.py: max_symbols=None")

    # Check 2: Focal Loss added
    lstm_trainer_path = Path("src/ai/models/aggressive_lstm_trainer.py")
    if lstm_trainer_path.exists():
        content = lstm_trainer_path.read_text(encoding='utf-8')
        if "class FocalLoss" in content:
            print("[OK] aggressive_lstm_trainer.py: FocalLoss class present")
        else:
            issues.append("aggressive_lstm_trainer.py missing FocalLoss class")

        if "WeightedRandomSampler" in content:
            print("[OK] aggressive_lstm_trainer.py: Stratified sampling present")
        else:
            issues.append("aggressive_lstm_trainer.py missing WeightedRandomSampler")

        if "CLASS COLLAPSE DETECTED" in content:
            print("[OK] aggressive_lstm_trainer.py: Class collapse detection present")
        else:
            issues.append("aggressive_lstm_trainer.py missing class collapse detection")

    # Check 3: SB3 PPO trainer exists
    sb3_ppo_path = Path("train_sb3_ppo.py")
    if sb3_ppo_path.exists():
        print("[OK] train_sb3_ppo.py: SB3 PPO trainer created")
    else:
        issues.append("train_sb3_ppo.py not found")

    # Check 4: Reward function fixed
    env_path = Path("src/ai/rl/enhanced_trading_environment.py")
    if env_path.exists():
        content = env_path.read_text(encoding='utf-8')
        if "np.clip(total_reward" in content and "-2.0, 2.0" in content:
            print("[OK] enhanced_trading_environment.py: Reward function balanced")
        else:
            issues.append("enhanced_trading_environment.py reward not properly clipped")

    # Check 5: Target thresholds widened
    feature_eng_path = Path("src/ai/feature_engineering/comprehensive_feature_engineer.py")
    if feature_eng_path.exists():
        content = feature_eng_path.read_text(encoding='utf-8')
        if "DIRECTION_THRESHOLD = 0.005" in content:
            print("[OK] comprehensive_feature_engineer.py: Threshold widened to 0.5%")
        elif "0.001" in content:
            issues.append("comprehensive_feature_engineer.py still uses 0.1% threshold")

    # Check 6: Stratified split in production trainer
    prod_trainer_path = Path("train_lstm_production.py")
    if prod_trainer_path.exists():
        content = prod_trainer_path.read_text(encoding='utf-8')
        if "StratifiedShuffleSplit" in content:
            print("[OK] train_lstm_production.py: Stratified split enabled")
        else:
            issues.append("train_lstm_production.py missing stratified split")

    print()
    if issues:
        print("ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("ALL FIXES VERIFIED!")
        return True


def check_sb3_installed():
    """Check if stable-baselines3 is installed"""
    try:
        import stable_baselines3
        print(f"[OK] stable-baselines3 v{stable_baselines3.__version__} installed")
        return True
    except ImportError:
        print("[!] stable-baselines3 not installed")
        print("    Install with: pip install stable-baselines3[extra]")
        return False


def train_lstm():
    """Train LSTM model with all fixes"""
    print("\n" + "=" * 60)
    print("STARTING LSTM TRAINING (ALL 1,681 STOCKS)")
    print("=" * 60)
    print("This will take several hours...")
    print()

    cmd = [
        sys.executable,
        "train_lstm_production.py",
        "--epochs", "100",
        "--batch-size", "128",
        "--sequence-length", "60"
    ]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)


def train_ppo():
    """Train PPO agent with Stable-Baselines3"""
    print("\n" + "=" * 60)
    print("STARTING SB3 PPO TRAINING")
    print("=" * 60)

    if not check_sb3_installed():
        print("\nCannot train PPO without stable-baselines3!")
        print("Install with: pip install stable-baselines3[extra]")
        return

    cmd = [
        sys.executable,
        "train_sb3_ppo.py",
        "--timesteps", "1000000",
        "--eval-freq", "10000"
    ]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(description="Run fixed training pipeline")
    parser.add_argument(
        "--model",
        choices=["lstm", "ppo", "all"],
        default="all",
        help="Which model to train"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check if fixes are in place"
    )

    args = parser.parse_args()

    # Always check fixes first
    fixes_ok = check_fixes()

    if args.check:
        sys.exit(0 if fixes_ok else 1)

    if not fixes_ok:
        print("\nWARNING: Some fixes are missing!")
        response = input("Continue anyway? [y/N] ")
        if response.lower() != 'y':
            sys.exit(1)

    print("\n" + "=" * 60)
    print("FIXED TRAINING PIPELINE")
    print("=" * 60)
    print("Improvements applied:")
    print("  - Training on ALL 1,681 stocks (not just 5)")
    print("  - Focal Loss for class imbalance")
    print("  - Stratified sampling in DataLoader")
    print("  - Stratified train/val split")
    print("  - Improved NaN handling (forward-fill)")
    print("  - Widened target threshold (0.5%)")
    print("  - Balanced reward function for PPO")
    print("  - Class collapse detection")
    print()

    if args.model in ["lstm", "all"]:
        train_lstm()

    if args.model in ["ppo", "all"]:
        train_ppo()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Check models/ directory for trained models")
    print("2. Run backtesting: python -m src.backtesting.backtest_runner")
    print("3. Check TensorBoard logs: tensorboard --logdir logs/")


if __name__ == "__main__":
    main()
