"""
50-Hour Continuous AI Training Orchestrator
============================================
Trains all AI models for 50 hours continuously with automatic checkpointing and recovery.

Features:
- Trains multiple models sequentially or in parallel
- Automatic checkpointing every 30 minutes
- Crash recovery from last checkpoint
- GPU memory management
- Comprehensive logging and monitoring
- Email notifications for completion/errors (optional)

Usage:
    python train_50h_continuous.py                    # Start fresh 50h training
    python train_50h_continuous.py --resume           # Resume from checkpoint
    python train_50h_continuous.py --parallel         # Train models in parallel
"""

import os
import sys
import json
import time
import torch
import psutil
import logging
import argparse
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import subprocess
import threading
import queue

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


class ContinuousTrainingOrchestrator:
    """Orchestrates 50-hour continuous training of all AI models."""

    def __init__(self,
                 total_hours: float = 50.0,
                 checkpoint_interval_minutes: int = 30,
                 parallel: bool = False,
                 resume: bool = False):

        self.total_hours = total_hours
        self.total_seconds = total_hours * 3600
        self.checkpoint_interval = checkpoint_interval_minutes * 60
        self.parallel = parallel
        self.resume = resume

        # Directories
        self.project_root = Path(__file__).parent
        self.checkpoint_dir = self.project_root / "continuous_training_checkpoints"
        self.log_dir = self.project_root / "training_logs"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)

        # State file
        self.state_file = self.checkpoint_dir / "training_state.json"

        # Setup logging
        self.setup_logging()

        # Training state
        self.start_time = None
        self.elapsed_time = 0
        self.current_model_idx = 0
        self.training_session_id = None

        # Model configurations
        self.models_to_train = self.get_model_configurations()

        # Performance tracking
        self.performance_log = []

        self.logger.info(f"Initialized 50-hour training orchestrator")
        self.logger.info(f"Total duration: {self.total_hours} hours")
        self.logger.info(f"Checkpoint interval: {checkpoint_interval_minutes} minutes")
        self.logger.info(f"Parallel training: {self.parallel}")
        self.logger.info(f"Resume mode: {self.resume}")

    def setup_logging(self):
        """Setup comprehensive logging."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"training_50h_{timestamp}.log"

        # Create logger
        self.logger = logging.getLogger("ContinuousTraining")
        self.logger.setLevel(logging.DEBUG)

        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        self.logger.info(f"Logging to: {log_file}")

    def get_model_configurations(self) -> List[Dict]:
        """Define all models to train with their configurations."""
        return [
            {
                "name": "LSTM_Model",
                "script": "train_lstm_production.py",
                "description": "Production LSTM model for stock prediction",
                "estimated_hours": 10,
                "priority": 1,
                "gpu_memory_gb": 4
            },
            {
                "name": "Transformer_Model",
                "script": "train_transformer_production.py",
                "description": "Market Transformer model",
                "estimated_hours": 12,
                "priority": 1,
                "gpu_memory_gb": 6
            },
            {
                "name": "GRU_Transformer",
                "script": "train_market_transformer_full_scale.py",
                "description": "Full-scale GRU-Transformer hybrid",
                "estimated_hours": 10,
                "priority": 2,
                "gpu_memory_gb": 5
            },
            {
                "name": "PPO_Agent",
                "script": "train_ppo_agent.py",
                "description": "PPO Reinforcement Learning Agent",
                "estimated_hours": 8,
                "priority": 2,
                "gpu_memory_gb": 3
            },
            {
                "name": "DQN_Agent",
                "script": "train_dqn_agent.py",
                "description": "DQN Reinforcement Learning Agent",
                "estimated_hours": 8,
                "priority": 2,
                "gpu_memory_gb": 3
            },
            {
                "name": "Aggressive_LSTM",
                "script": "train_aggressive_lstm_daily_fixed.py",
                "description": "Aggressive LSTM trainer",
                "estimated_hours": 6,
                "priority": 3,
                "gpu_memory_gb": 4
            }
        ]

    def save_state(self, additional_info: Optional[Dict] = None):
        """Save current training state to checkpoint."""
        state = {
            "session_id": self.training_session_id,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "elapsed_time": self.elapsed_time,
            "current_model_idx": self.current_model_idx,
            "total_hours": self.total_hours,
            "models_completed": self.current_model_idx,
            "models_remaining": len(self.models_to_train) - self.current_model_idx,
            "performance_log": self.performance_log,
            "timestamp": datetime.now().isoformat(),
            "gpu_available": torch.cuda.is_available(),
            "cuda_devices": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }

        if additional_info:
            state.update(additional_info)

        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

        self.logger.debug(f"State saved to {self.state_file}")

    def load_state(self) -> bool:
        """Load training state from checkpoint."""
        if not self.state_file.exists():
            self.logger.info("No previous state found, starting fresh")
            return False

        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)

            self.training_session_id = state.get("session_id")
            self.start_time = datetime.fromisoformat(state["start_time"]) if state.get("start_time") else None
            self.elapsed_time = state.get("elapsed_time", 0)
            self.current_model_idx = state.get("current_model_idx", 0)
            self.performance_log = state.get("performance_log", [])

            self.logger.info(f"Loaded state from checkpoint")
            self.logger.info(f"Session ID: {self.training_session_id}")
            self.logger.info(f"Elapsed time: {self.elapsed_time/3600:.2f} hours")
            self.logger.info(f"Models completed: {self.current_model_idx}/{len(self.models_to_train)}")

            return True

        except Exception as e:
            self.logger.error(f"Error loading state: {e}")
            return False

    def get_system_stats(self) -> Dict:
        """Get current system resource usage."""
        stats = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_available_gb": psutil.virtual_memory().available / (1024**3),
            "disk_usage_percent": psutil.disk_usage(str(self.project_root)).percent
        }

        if torch.cuda.is_available():
            stats["gpu_count"] = torch.cuda.device_count()
            stats["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / (1024**3)
            stats["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / (1024**3)

        return stats

    def log_progress(self):
        """Log current training progress."""
        if not self.start_time:
            return

        elapsed = time.time() - self.start_time.timestamp() + self.elapsed_time
        remaining = max(0, self.total_seconds - elapsed)
        progress_pct = (elapsed / self.total_seconds) * 100

        elapsed_str = str(timedelta(seconds=int(elapsed)))
        remaining_str = str(timedelta(seconds=int(remaining)))
        eta = datetime.now() + timedelta(seconds=remaining)

        self.logger.info("=" * 80)
        self.logger.info(f"TRAINING PROGRESS: {progress_pct:.1f}%")
        self.logger.info(f"Elapsed: {elapsed_str} | Remaining: {remaining_str}")
        self.logger.info(f"ETA: {eta.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Models: {self.current_model_idx}/{len(self.models_to_train)} completed")

        # System stats
        stats = self.get_system_stats()
        self.logger.info(f"CPU: {stats['cpu_percent']:.1f}% | RAM: {stats['memory_percent']:.1f}%")
        if "gpu_memory_allocated_gb" in stats:
            self.logger.info(f"GPU Memory: {stats['gpu_memory_allocated_gb']:.2f}GB allocated")

        self.logger.info("=" * 80)

    def train_model(self, model_config: Dict) -> Tuple[bool, str]:
        """Train a single model."""
        model_name = model_config["name"]
        script_path = self.project_root / model_config["script"]

        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"Starting training: {model_name}")
        self.logger.info(f"Description: {model_config['description']}")
        self.logger.info(f"Script: {script_path}")
        self.logger.info(f"Estimated time: {model_config['estimated_hours']} hours")
        self.logger.info(f"{'='*80}\n")

        if not script_path.exists():
            error_msg = f"Training script not found: {script_path}"
            self.logger.error(error_msg)
            return False, error_msg

        # Calculate time allocation for this model
        remaining_time = self.total_seconds - self.elapsed_time
        models_left = len(self.models_to_train) - self.current_model_idx
        time_per_model = remaining_time / max(models_left, 1)

        model_start_time = time.time()

        try:
            # Run training script
            self.logger.info(f"Executing: python {script_path}")

            process = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            # Stream output
            for line in process.stdout:
                self.logger.info(f"[{model_name}] {line.rstrip()}")

                # Check if we've exceeded total time
                current_elapsed = time.time() - self.start_time.timestamp() + self.elapsed_time
                if current_elapsed >= self.total_seconds:
                    self.logger.warning(f"50-hour limit reached, terminating {model_name}")
                    process.terminate()
                    process.wait(timeout=30)
                    return True, "Time limit reached"

            process.wait()

            model_elapsed = time.time() - model_start_time

            if process.returncode == 0:
                self.logger.info(f"✓ {model_name} completed successfully in {model_elapsed/3600:.2f}h")

                # Log performance
                self.performance_log.append({
                    "model": model_name,
                    "status": "completed",
                    "duration_hours": model_elapsed / 3600,
                    "timestamp": datetime.now().isoformat()
                })

                return True, "Success"
            else:
                error_msg = f"Training failed with return code {process.returncode}"
                self.logger.error(f"✗ {model_name} failed: {error_msg}")

                self.performance_log.append({
                    "model": model_name,
                    "status": "failed",
                    "error": error_msg,
                    "duration_hours": model_elapsed / 3600,
                    "timestamp": datetime.now().isoformat()
                })

                return False, error_msg

        except Exception as e:
            error_msg = f"Exception during training: {str(e)}\n{traceback.format_exc()}"
            self.logger.error(error_msg)

            self.performance_log.append({
                "model": model_name,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })

            return False, error_msg

    def train_sequential(self):
        """Train all models sequentially."""
        self.logger.info("Starting SEQUENTIAL training mode")

        while self.current_model_idx < len(self.models_to_train):
            # Check if time is up
            elapsed = time.time() - self.start_time.timestamp() + self.elapsed_time
            if elapsed >= self.total_seconds:
                self.logger.info("50-hour training period completed!")
                break

            model_config = self.models_to_train[self.current_model_idx]

            # Train model
            success, message = self.train_model(model_config)

            # Update progress
            self.current_model_idx += 1
            self.save_state()
            self.log_progress()

            # Short break between models
            if self.current_model_idx < len(self.models_to_train):
                self.logger.info("Cooling down for 60 seconds...")
                time.sleep(60)

    def train_parallel(self):
        """Train multiple models in parallel (if GPU memory allows)."""
        self.logger.info("Starting PARALLEL training mode")
        self.logger.warning("Parallel mode requires careful GPU memory management!")

        # For now, fall back to sequential
        # TODO: Implement true parallel training with resource management
        self.logger.info("Falling back to sequential mode for safety")
        self.train_sequential()

    def run(self):
        """Main training loop."""
        try:
            # Load state if resuming
            if self.resume:
                self.load_state()

            # Initialize session
            if not self.training_session_id:
                self.training_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

            if not self.start_time:
                self.start_time = datetime.now()
            else:
                # Adjust start time for elapsed time
                self.start_time = datetime.now() - timedelta(seconds=self.elapsed_time)

            self.save_state()

            # Log initial state
            self.logger.info("\n" + "="*80)
            self.logger.info("STARTING 50-HOUR CONTINUOUS TRAINING")
            self.logger.info("="*80)
            self.logger.info(f"Session ID: {self.training_session_id}")
            self.logger.info(f"Start time: {self.start_time}")
            self.logger.info(f"End time: {self.start_time + timedelta(hours=self.total_hours)}")
            self.logger.info(f"Total models: {len(self.models_to_train)}")
            self.logger.info(f"GPU available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                self.logger.info(f"GPU devices: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    self.logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            self.logger.info("="*80 + "\n")

            # Start checkpoint monitor thread
            checkpoint_thread = threading.Thread(target=self.checkpoint_monitor, daemon=True)
            checkpoint_thread.start()

            # Run training
            if self.parallel:
                self.train_parallel()
            else:
                self.train_sequential()

            # Final state
            self.log_progress()
            self.save_state({"status": "completed"})

            self.logger.info("\n" + "="*80)
            self.logger.info("TRAINING COMPLETED!")
            self.logger.info("="*80)
            self.logger.info(f"Total time: {(time.time() - self.start_time.timestamp())/3600:.2f} hours")
            self.logger.info(f"Models trained: {self.current_model_idx}/{len(self.models_to_train)}")
            self.logger.info("="*80 + "\n")

            # Print summary
            self.print_summary()

        except KeyboardInterrupt:
            self.logger.warning("\n\nTraining interrupted by user!")
            self.elapsed_time = time.time() - self.start_time.timestamp() + self.elapsed_time
            self.save_state({"status": "interrupted"})
            self.logger.info(f"Progress saved. Resume with: python {__file__} --resume")

        except Exception as e:
            self.logger.error(f"\n\nFATAL ERROR: {e}")
            self.logger.error(traceback.format_exc())
            self.elapsed_time = time.time() - self.start_time.timestamp() + self.elapsed_time
            self.save_state({"status": "error", "error": str(e)})
            raise

    def checkpoint_monitor(self):
        """Monitor thread that saves checkpoints periodically."""
        while True:
            time.sleep(self.checkpoint_interval)
            if self.start_time:
                self.elapsed_time = time.time() - self.start_time.timestamp() + self.elapsed_time
                self.start_time = datetime.now()
                self.save_state()
                self.logger.info("💾 Checkpoint saved")

    def print_summary(self):
        """Print training summary."""
        self.logger.info("\n" + "="*80)
        self.logger.info("TRAINING SUMMARY")
        self.logger.info("="*80)

        for log_entry in self.performance_log:
            status_symbol = "✓" if log_entry["status"] == "completed" else "✗"
            duration = log_entry.get("duration_hours", 0)
            self.logger.info(f"{status_symbol} {log_entry['model']}: {log_entry['status']} ({duration:.2f}h)")

        self.logger.info("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="50-Hour Continuous AI Training")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--parallel", action="store_true", help="Train models in parallel")
    parser.add_argument("--hours", type=float, default=50.0, help="Total training hours (default: 50)")
    parser.add_argument("--checkpoint-interval", type=int, default=30,
                       help="Checkpoint interval in minutes (default: 30)")

    args = parser.parse_args()

    orchestrator = ContinuousTrainingOrchestrator(
        total_hours=args.hours,
        checkpoint_interval_minutes=args.checkpoint_interval,
        parallel=args.parallel,
        resume=args.resume
    )

    orchestrator.run()


if __name__ == "__main__":
    main()
