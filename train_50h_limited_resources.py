"""
50-Hour Continuous AI Training with 50% Resource Limit
=======================================================
Trains all AI models for 50 hours with resource limits:
- Maximum 50% GPU memory usage
- Maximum 50% RAM usage
- Leaves 50% resources free for other processes

Usage:
    python train_50h_limited_resources.py                    # Start fresh 50h training
    python train_50h_limited_resources.py --resume           # Resume from checkpoint
"""

import os
import sys
import gc
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

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# ============================
# RESOURCE LIMITS - 50% MAX
# ============================
GPU_MEMORY_FRACTION = 0.50  # 50% of GPU memory
RAM_MEMORY_FRACTION = 0.50  # 50% of system RAM
MONITORING_INTERVAL = 10    # Check every 10 seconds


class ResourceLimiter:
    """Enforces strict 50% resource limits."""

    def __init__(self):
        self.total_ram_gb = psutil.virtual_memory().total / (1024**3)
        self.max_ram_gb = self.total_ram_gb * RAM_MEMORY_FRACTION

        if torch.cuda.is_available():
            self.total_gpu_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            self.max_gpu_gb = self.total_gpu_gb * GPU_MEMORY_FRACTION
        else:
            self.total_gpu_gb = 0
            self.max_gpu_gb = 0

        self.setup_gpu_limits()

    def setup_gpu_limits(self):
        """Configure PyTorch to use only 40% of GPU memory."""
        if not torch.cuda.is_available():
            return

        # Set memory fraction
        torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION, device=0)

        # Enable memory efficient settings
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

        print(f"[OK] GPU Memory Limited to {self.max_gpu_gb:.2f}GB ({GPU_MEMORY_FRACTION*100:.0f}%)")

    def get_current_usage(self) -> Dict:
        """Get current resource usage."""
        ram_used = psutil.virtual_memory().used / (1024**3)
        ram_percent = (ram_used / self.total_ram_gb) * 100

        usage = {
            'ram_used_gb': ram_used,
            'ram_total_gb': self.total_ram_gb,
            'ram_percent': ram_percent,
            'ram_limit_gb': self.max_ram_gb,
            'ram_within_limit': ram_used <= self.max_ram_gb
        }

        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated(0) / (1024**3)
            gpu_reserved = torch.cuda.memory_reserved(0) / (1024**3)
            gpu_percent = (gpu_allocated / self.total_gpu_gb) * 100

            usage.update({
                'gpu_allocated_gb': gpu_allocated,
                'gpu_reserved_gb': gpu_reserved,
                'gpu_total_gb': self.total_gpu_gb,
                'gpu_percent': gpu_percent,
                'gpu_limit_gb': self.max_gpu_gb,
                'gpu_within_limit': gpu_allocated <= self.max_gpu_gb
            })

        return usage

    def enforce_limits(self) -> Tuple[bool, str]:
        """Check if resource usage is within limits."""
        usage = self.get_current_usage()

        # Check RAM
        if not usage['ram_within_limit']:
            return False, f"RAM limit exceeded: {usage['ram_used_gb']:.2f}GB / {usage['ram_limit_gb']:.2f}GB"

        # Check GPU
        if torch.cuda.is_available() and not usage['gpu_within_limit']:
            return False, f"GPU limit exceeded: {usage['gpu_allocated_gb']:.2f}GB / {usage['gpu_limit_gb']:.2f}GB"

        return True, "Within limits"

    def cleanup_memory(self):
        """Aggressive memory cleanup."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


class LimitedTrainingOrchestrator:
    """Orchestrates 50-hour continuous training with 40% resource limits."""

    def __init__(self,
                 total_hours: float = 50.0,
                 checkpoint_interval_minutes: int = 30,
                 resume: bool = False):

        self.total_hours = total_hours
        self.total_seconds = total_hours * 3600
        self.checkpoint_interval = checkpoint_interval_minutes * 60
        self.resume = resume

        # Resource limiter
        self.resource_limiter = ResourceLimiter()

        # Directories
        self.project_root = Path(__file__).parent
        self.checkpoint_dir = self.project_root / "continuous_training_checkpoints"
        self.log_dir = self.project_root / "training_logs"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)

        # State file
        self.state_file = self.checkpoint_dir / "training_state_limited.json"

        # Setup logging
        self.setup_logging()

        # Training state
        self.start_time = None
        self.elapsed_time = 0
        self.current_model_idx = 0
        self.training_session_id = None

        # Model configurations with REDUCED batch sizes for 40% memory
        self.models_to_train = self.get_model_configurations()

        # Performance tracking
        self.performance_log = []

        self.logger.info(f"Initialized 50-HOUR TRAINING - 50% GPU/RAM UTILIZATION")
        self.logger.info(f"Total duration: {self.total_hours} hours")
        self.logger.info(f"Resource limits: {RAM_MEMORY_FRACTION*100:.0f}% RAM, {GPU_MEMORY_FRACTION*100:.0f}% GPU")
        self.logger.info(f"RAM Limit: {self.resource_limiter.max_ram_gb:.2f}GB / {self.resource_limiter.total_ram_gb:.2f}GB")
        if torch.cuda.is_available():
            self.logger.info(f"GPU Limit: {self.resource_limiter.max_gpu_gb:.2f}GB / {self.resource_limiter.total_gpu_gb:.2f}GB")

    def setup_logging(self):
        """Setup comprehensive logging."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"training_50h_limited_{timestamp}.log"

        # Create logger
        self.logger = logging.getLogger("LimitedTraining")
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
        """Define models optimized for 50% GPU/RAM with extended epochs for 50h runtime."""
        return [
            {
                "name": "LSTM_Model",
                "script": "train_lstm_production.py",
                "description": "Production LSTM (50% resources, 1000 epochs)",
                "estimated_hours": 16,
                "priority": 1,
                "batch_size_override": 64,  # Increased for better GPU utilization
                "env_vars": {
                    "CUDA_VISIBLE_DEVICES": "0",
                    "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:256"
                }
            },
            {
                "name": "Transformer_Model",
                "script": "train_transformer_production.py",
                "description": "Market Transformer (50% resources, 800 epochs)",
                "estimated_hours": 16,
                "priority": 1,
                "batch_size_override": 32,  # Increased from 16
                "env_vars": {
                    "CUDA_VISIBLE_DEVICES": "0",
                    "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:256"
                }
            },
            {
                "name": "LSTM_Round2",
                "script": "train_lstm_production.py",
                "description": "LSTM Extended Training (50% resources)",
                "estimated_hours": 18,
                "priority": 2,
                "batch_size_override": 64,
                "env_vars": {
                    "CUDA_VISIBLE_DEVICES": "0",
                    "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:256"
                }
            }
        ]

    def save_state(self, additional_info: Optional[Dict] = None):
        """Save current training state."""
        state = {
            "session_id": self.training_session_id,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "elapsed_time": self.elapsed_time,
            "current_model_idx": self.current_model_idx,
            "total_hours": self.total_hours,
            "resource_limits": {
                "ram_fraction": RAM_MEMORY_FRACTION,
                "gpu_fraction": GPU_MEMORY_FRACTION,
                "ram_limit_gb": self.resource_limiter.max_ram_gb,
                "gpu_limit_gb": self.resource_limiter.max_gpu_gb
            },
            "performance_log": self.performance_log,
            "timestamp": datetime.now().isoformat()
        }

        if additional_info:
            state.update(additional_info)

        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

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
            self.logger.info(f"Elapsed: {self.elapsed_time/3600:.2f}h, Models: {self.current_model_idx}/{len(self.models_to_train)}")

            return True

        except Exception as e:
            self.logger.error(f"Error loading state: {e}")
            return False

    def monitor_resources(self):
        """Background thread to monitor resource usage."""
        while True:
            time.sleep(MONITORING_INTERVAL)

            usage = self.resource_limiter.get_current_usage()
            within_limits, msg = self.resource_limiter.enforce_limits()

            if not within_limits:
                self.logger.warning(f"[WARNING] RESOURCE LIMIT EXCEEDED: {msg}")
                self.logger.warning("Performing aggressive memory cleanup...")
                self.resource_limiter.cleanup_memory()
            else:
                self.logger.debug(f"Resources: RAM {usage['ram_used_gb']:.1f}/{usage['ram_limit_gb']:.1f}GB, " +
                                (f"GPU {usage.get('gpu_allocated_gb', 0):.1f}/{usage.get('gpu_limit_gb', 0):.1f}GB"
                                 if torch.cuda.is_available() else ""))

    def train_model(self, model_config: Dict) -> Tuple[bool, str]:
        """Train a single model with resource limits."""
        model_name = model_config["name"]
        script_path = self.project_root / model_config["script"]

        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"Starting training: {model_name}")
        self.logger.info(f"Description: {model_config['description']}")
        self.logger.info(f"{'='*80}\n")

        if not script_path.exists():
            error_msg = f"Training script not found: {script_path}"
            self.logger.error(error_msg)
            return False, error_msg

        # Memory cleanup before starting
        self.resource_limiter.cleanup_memory()

        # Check resources before starting
        usage = self.resource_limiter.get_current_usage()
        self.logger.info(f"Pre-training resources:")
        self.logger.info(f"  RAM: {usage['ram_used_gb']:.2f}GB / {usage['ram_limit_gb']:.2f}GB")
        if torch.cuda.is_available():
            self.logger.info(f"  GPU: {usage.get('gpu_allocated_gb', 0):.2f}GB / {usage.get('gpu_limit_gb', 0):.2f}GB")

        model_start_time = time.time()

        try:
            # Setup environment variables
            env = os.environ.copy()
            if "env_vars" in model_config:
                env.update(model_config["env_vars"])

            # Run training script
            self.logger.info(f"Executing: python {script_path}")

            process = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env
            )

            # Stream output
            for line in process.stdout:
                self.logger.info(f"[{model_name}] {line.rstrip()}")

                # Check time limit
                current_elapsed = time.time() - self.start_time.timestamp() + self.elapsed_time
                if current_elapsed >= self.total_seconds:
                    self.logger.warning(f"50-hour limit reached, terminating {model_name}")
                    process.terminate()
                    process.wait(timeout=30)
                    return True, "Time limit reached"

            process.wait()

            model_elapsed = time.time() - model_start_time

            if process.returncode == 0:
                self.logger.info(f"[SUCCESS] {model_name} completed in {model_elapsed/3600:.2f}h")

                self.performance_log.append({
                    "model": model_name,
                    "status": "completed",
                    "duration_hours": model_elapsed / 3600,
                    "timestamp": datetime.now().isoformat()
                })

                return True, "Success"
            else:
                error_msg = f"Training failed with return code {process.returncode}"
                self.logger.error(f"[FAILED] {model_name} failed: {error_msg}")

                self.performance_log.append({
                    "model": model_name,
                    "status": "failed",
                    "error": error_msg,
                    "timestamp": datetime.now().isoformat()
                })

                return False, error_msg

        except Exception as e:
            error_msg = f"Exception: {str(e)}"
            self.logger.error(error_msg)

            self.performance_log.append({
                "model": model_name,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })

            return False, error_msg
        finally:
            # Cleanup after training
            self.resource_limiter.cleanup_memory()

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
                self.start_time = datetime.now() - timedelta(seconds=self.elapsed_time)

            self.save_state()

            # Log initial state
            self.logger.info("\n" + "="*80)
            self.logger.info("50-HOUR TRAINING WITH 50% GPU/RAM UTILIZATION")
            self.logger.info("="*80)
            self.logger.info(f"Session: {self.training_session_id}")
            self.logger.info(f"Start: {self.start_time}")
            self.logger.info(f"End: {self.start_time + timedelta(hours=self.total_hours)}")
            self.logger.info(f"RAM Limit: {self.resource_limiter.max_ram_gb:.2f}GB (50%)")
            if torch.cuda.is_available():
                self.logger.info(f"GPU Limit: {self.resource_limiter.max_gpu_gb:.2f}GB (50%)")
            self.logger.info("="*80 + "\n")

            # Start resource monitor thread
            monitor_thread = threading.Thread(target=self.monitor_resources, daemon=True)
            monitor_thread.start()

            # Start checkpoint monitor thread
            checkpoint_thread = threading.Thread(target=self.checkpoint_monitor, daemon=True)
            checkpoint_thread.start()

            # Train all models
            while self.current_model_idx < len(self.models_to_train):
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

                # Cooldown between models
                if self.current_model_idx < len(self.models_to_train):
                    self.logger.info("Cooling down for 60 seconds...")
                    self.resource_limiter.cleanup_memory()
                    time.sleep(60)

            # Final state
            self.save_state({"status": "completed"})

            self.logger.info("\n" + "="*80)
            self.logger.info("TRAINING COMPLETED!")
            self.logger.info("="*80)
            self.print_summary()

        except KeyboardInterrupt:
            self.logger.warning("\n\nTraining interrupted!")
            self.elapsed_time = time.time() - self.start_time.timestamp() + self.elapsed_time
            self.save_state({"status": "interrupted"})
            self.logger.info(f"Resume with: python {__file__} --resume")

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
                self.logger.info("[CHECKPOINT] Saved")

    def print_summary(self):
        """Print training summary."""
        self.logger.info("TRAINING SUMMARY")
        self.logger.info("="*80)

        for log_entry in self.performance_log:
            status_symbol = "[OK]" if log_entry["status"] == "completed" else "[FAIL]"
            duration = log_entry.get("duration_hours", 0)
            self.logger.info(f"{status_symbol} {log_entry['model']}: {duration:.2f}h")

        self.logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(description="50-Hour Training with 40% Resource Limits")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--hours", type=float, default=50.0, help="Training hours (default: 50)")
    parser.add_argument("--checkpoint-interval", type=int, default=30,
                       help="Checkpoint interval in minutes (default: 30)")

    args = parser.parse_args()

    orchestrator = LimitedTrainingOrchestrator(
        total_hours=args.hours,
        checkpoint_interval_minutes=args.checkpoint_interval,
        resume=args.resume
    )

    orchestrator.run()


if __name__ == "__main__":
    main()
