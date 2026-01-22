"""
TRAINING ORCHESTRATOR
Coordinate sequential training of all models (LSTM → Transformer → PPO → DQN)

Features:
- Sequential training with dependency management
- Progress tracking and logging
- Automatic checkpoint management
- Resource allocation
- Training time estimation
- Comprehensive summary report

Usage:
    # Full training pipeline
    python orchestrate_training.py

    # Test mode (quick validation)
    python orchestrate_training.py --test-mode

    # Skip specific models
    python orchestrate_training.py --skip-ppo --skip-dqn

Author: Trading Bot Team
Date: October 28, 2025
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List
import json


class TrainingOrchestrator:
    """Orchestrate multi-model training pipeline"""

    def __init__(self, test_mode: bool = False):
        self.test_mode = test_mode
        self.results = {}
        self.start_time = None
        self.log_file = None

    def log(self, message: str):
        """Log message to console and file"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        print(log_message)

        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(log_message + '\n')

    def run_command(self, cmd: List[str], stage_name: str) -> Dict:
        """Run training command and track results"""

        self.log(f"\n{'='*80}")
        self.log(f"Starting: {stage_name}")
        self.log(f"Command: {' '.join(cmd)}")
        self.log(f"{'='*80}\n")

        stage_start = time.time()

        try:
            # Run command
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=False,  # Show output in real-time
                text=True
            )

            stage_duration = time.time() - stage_start
            success = True
            error_message = None

        except subprocess.CalledProcessError as e:
            stage_duration = time.time() - stage_start
            success = False
            error_message = str(e)
            self.log(f"\nERROR in {stage_name}: {error_message}")

        except KeyboardInterrupt:
            stage_duration = time.time() - stage_start
            success = False
            error_message = "Interrupted by user"
            self.log(f"\n{stage_name} interrupted by user")
            raise

        # Record results
        stage_result = {
            'stage': stage_name,
            'success': success,
            'duration': stage_duration,
            'duration_str': str(timedelta(seconds=int(stage_duration))),
            'error': error_message,
            'timestamp': datetime.now().isoformat()
        }

        self.results[stage_name] = stage_result

        self.log(f"\n{'='*80}")
        self.log(f"Completed: {stage_name}")
        self.log(f"Duration: {stage_result['duration_str']}")
        self.log(f"Status: {'SUCCESS' if success else 'FAILED'}")
        self.log(f"{'='*80}\n")

        return stage_result

    def train_lstm(self) -> bool:
        """Train LSTM model"""

        cmd = ['python', 'train_lstm_production.py']

        if self.test_mode:
            cmd.extend(['--test-mode', '--epochs', '5'])
        else:
            cmd.extend(['--epochs', '100', '--batch-size', '128'])

        result = self.run_command(cmd, 'LSTM Training')
        return result['success']

    def train_transformer(self) -> bool:
        """Train Transformer model"""

        cmd = ['python', 'train_transformer_production.py']

        if self.test_mode:
            cmd.extend(['--test-mode', '--epochs', '5'])
        else:
            cmd.extend(['--epochs', '100', '--batch-size', '64'])

        result = self.run_command(cmd, 'Transformer Training')
        return result['success']

    def train_ppo(self) -> bool:
        """Train PPO agent"""

        cmd = ['python', 'train_ppo_agent.py']

        if self.test_mode:
            cmd.extend(['--test-mode', '--episodes', '50'])
        else:
            cmd.extend(['--episodes', '1000'])

        result = self.run_command(cmd, 'PPO Training')
        return result['success']

    def train_dqn(self) -> bool:
        """Train DQN agent"""

        cmd = ['python', 'train_dqn_agent.py']

        if self.test_mode:
            cmd.extend(['--test-mode', '--episodes', '50'])
        else:
            cmd.extend(['--episodes', '1000'])

        result = self.run_command(cmd, 'DQN Training')
        return result['success']

    def evaluate_models(self) -> bool:
        """Evaluate and compare all models"""

        cmd = ['python', 'evaluate_all_models.py']

        if self.test_mode:
            cmd.append('--test-mode')

        result = self.run_command(cmd, 'Model Evaluation')
        return result['success']

    def generate_summary_report(self):
        """Generate comprehensive summary report"""

        report_path = Path('results/training_summary.txt')
        report_path.parent.mkdir(parents=True, exist_ok=True)

        total_duration = time.time() - self.start_time

        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("TRAINING ORCHESTRATION SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Start Time: {datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Duration: {str(timedelta(seconds=int(total_duration)))}\n")
            f.write(f"Test Mode: {self.test_mode}\n\n")

            f.write("-" * 80 + "\n")
            f.write("STAGE RESULTS\n")
            f.write("-" * 80 + "\n\n")

            success_count = 0
            failed_count = 0

            for stage_name, result in self.results.items():
                status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
                f.write(f"{stage_name}:\n")
                f.write(f"  Status: {status}\n")
                f.write(f"  Duration: {result['duration_str']}\n")

                if result['error']:
                    f.write(f"  Error: {result['error']}\n")

                f.write("\n")

                if result['success']:
                    success_count += 1
                else:
                    failed_count += 1

            f.write("-" * 80 + "\n")
            f.write("SUMMARY\n")
            f.write("-" * 80 + "\n\n")

            f.write(f"Total Stages: {len(self.results)}\n")
            f.write(f"Successful: {success_count}\n")
            f.write(f"Failed: {failed_count}\n\n")

            # Model checkpoints
            f.write("-" * 80 + "\n")
            f.write("MODEL CHECKPOINTS\n")
            f.write("-" * 80 + "\n\n")

            models_dir = Path('models')
            if models_dir.exists():
                for checkpoint in models_dir.glob('*.pth'):
                    size_mb = checkpoint.stat().st_size / (1024 * 1024)
                    f.write(f"  {checkpoint.name} ({size_mb:.1f} MB)\n")
            else:
                f.write("  No checkpoints found\n")

            f.write("\n")

            # Next steps
            f.write("-" * 80 + "\n")
            f.write("NEXT STEPS\n")
            f.write("-" * 80 + "\n\n")

            f.write("1. Review model comparison: results/model_comparison.txt\n")
            f.write("2. View TensorBoard logs:\n")
            f.write("   - LSTM: tensorboard --logdir logs/tensorboard\n")
            f.write("   - Transformer: tensorboard --logdir logs/tensorboard_transformer\n")
            f.write("   - PPO: tensorboard --logdir logs/tensorboard_ppo\n")
            f.write("   - DQN: tensorboard --logdir logs/tensorboard_dqn\n")
            f.write("3. Run backtests on individual stocks\n")
            f.write("4. Deploy best performing model\n\n")

            f.write("=" * 80 + "\n")

        self.log(f"\nSummary report saved to: {report_path}")

        # Also save JSON version
        json_path = Path('results/training_summary.json')
        with open(json_path, 'w') as f:
            json.dump({
                'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_duration_seconds': total_duration,
                'test_mode': self.test_mode,
                'results': self.results,
                'success_count': success_count,
                'failed_count': failed_count
            }, f, indent=2)

        self.log(f"JSON summary saved to: {json_path}")

    def run_full_pipeline(self, skip_lstm: bool = False, skip_transformer: bool = False,
                         skip_ppo: bool = False, skip_dqn: bool = False,
                         skip_eval: bool = False):
        """Run full training pipeline"""

        self.start_time = time.time()

        # Setup logging
        log_dir = Path('logs/orchestration')
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        self.log("=" * 80)
        self.log("TRAINING ORCHESTRATION STARTED")
        self.log("=" * 80)
        self.log(f"Test Mode: {self.test_mode}")
        self.log(f"Log File: {self.log_file}")

        try:
            # Stage 1: LSTM
            if not skip_lstm:
                if not self.train_lstm():
                    self.log("\nWARNING: LSTM training failed, continuing with next stages...")
            else:
                self.log("\nSkipping LSTM training")

            # Stage 2: Transformer
            if not skip_transformer:
                if not self.train_transformer():
                    self.log("\nWARNING: Transformer training failed, continuing with next stages...")
            else:
                self.log("\nSkipping Transformer training")

            # Stage 3: PPO
            if not skip_ppo:
                if not self.train_ppo():
                    self.log("\nWARNING: PPO training failed, continuing with next stages...")
            else:
                self.log("\nSkipping PPO training")

            # Stage 4: DQN
            if not skip_dqn:
                if not self.train_dqn():
                    self.log("\nWARNING: DQN training failed, continuing with next stages...")
            else:
                self.log("\nSkipping DQN training")

            # Stage 5: Evaluation
            if not skip_eval:
                self.evaluate_models()
            else:
                self.log("\nSkipping model evaluation")

        except KeyboardInterrupt:
            self.log("\n\nTraining pipeline interrupted by user!")

        finally:
            # Generate summary
            self.generate_summary_report()

            total_time = time.time() - self.start_time
            self.log("\n" + "=" * 80)
            self.log("TRAINING ORCHESTRATION COMPLETE")
            self.log(f"Total Time: {str(timedelta(seconds=int(total_time)))}")
            self.log("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Orchestrate training of all models')

    parser.add_argument('--test-mode', action='store_true',
                       help='Run in test mode (quick validation)')
    parser.add_argument('--skip-lstm', action='store_true',
                       help='Skip LSTM training')
    parser.add_argument('--skip-transformer', action='store_true',
                       help='Skip Transformer training')
    parser.add_argument('--skip-ppo', action='store_true',
                       help='Skip PPO training')
    parser.add_argument('--skip-dqn', action='store_true',
                       help='Skip DQN training')
    parser.add_argument('--skip-eval', action='store_true',
                       help='Skip model evaluation')

    args = parser.parse_args()

    # Create orchestrator
    orchestrator = TrainingOrchestrator(test_mode=args.test_mode)

    # Run pipeline
    orchestrator.run_full_pipeline(
        skip_lstm=args.skip_lstm,
        skip_transformer=args.skip_transformer,
        skip_ppo=args.skip_ppo,
        skip_dqn=args.skip_dqn,
        skip_eval=args.skip_eval
    )


if __name__ == '__main__':
    main()
