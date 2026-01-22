"""
MASTER AI MODEL TRAINING ORCHESTRATOR
======================================

Trains all AI models for the Trading Bot in sequence or parallel:
1. LSTM - Sequential pattern recognition
2. Transformer - Advanced attention-based learning
3. PPO - Reinforcement learning (policy-based)
4. DQN - Reinforcement learning (value-based)

Features:
- GPU resource management (RTX 4080 optimized)
- Sequential or parallel training
- Automated checkpointing
- Performance tracking & comparison
- Failure recovery
- Training logs & metrics
- Model ensemble preparation

Usage:
    # Train all models sequentially
    python train_all_models.py --mode sequential

    # Train all models in parallel (if enough VRAM)
    python train_all_models.py --mode parallel

    # Train specific models only
    python train_all_models.py --models lstm transformer

    # Quick test mode
    python train_all_models.py --test-mode --epochs 5

    # Resume from last checkpoint
    python train_all_models.py --resume

Hardware Requirements:
    GPU: RTX 4080 (12GB VRAM minimum)
    RAM: 32GB recommended
    Storage: 50GB for all models

Author: Trading Bot Team
Date: October 29, 2025
"""

import os
import sys
import time
import json
import subprocess
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import psutil
import GPUtil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/training_master_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ModelConfig:
    """Configuration for each model"""
    def __init__(
        self,
        name: str,
        script: str,
        priority: int,
        estimated_time_hours: float,
        gpu_memory_gb: float,
        default_epochs: int
    ):
        self.name = name
        self.script = script
        self.priority = priority
        self.estimated_time_hours = estimated_time_hours
        self.gpu_memory_gb = gpu_memory_gb
        self.default_epochs = default_epochs


# Model configurations
MODELS = {
    'lstm': ModelConfig(
        name='LSTM',
        script='train_lstm_production.py',
        priority=1,
        estimated_time_hours=4.0,
        gpu_memory_gb=6.0,
        default_epochs=50
    ),
    'transformer': ModelConfig(
        name='Transformer',
        script='train_transformer_production.py',
        priority=2,
        estimated_time_hours=8.0,
        gpu_memory_gb=8.0,
        default_epochs=100
    ),
    'ppo': ModelConfig(
        name='PPO Agent',
        script='train_ppo_agent.py',
        priority=3,
        estimated_time_hours=6.0,
        gpu_memory_gb=5.0,
        default_epochs=1000  # episodes
    ),
    'dqn': ModelConfig(
        name='DQN Agent',
        script='train_dqn_agent.py',
        priority=4,
        estimated_time_hours=5.0,
        gpu_memory_gb=4.5,
        default_epochs=1000  # episodes
    )
}


class TrainingOrchestrator:
    """Orchestrates training of all AI models"""

    def __init__(
        self,
        models_to_train: List[str],
        mode: str = 'sequential',
        test_mode: bool = False,
        resume: bool = False,
        epochs_override: Optional[int] = None,
        batch_size: int = 256,
        stride: int = 5
    ):
        self.models_to_train = models_to_train
        self.mode = mode
        self.test_mode = test_mode
        self.resume = resume
        self.epochs_override = epochs_override
        self.batch_size = batch_size
        self.stride = stride

        self.results = {}
        self.start_time = None
        self.project_root = Path.cwd()

        logger.info("=" * 80)
        logger.info("MASTER AI MODEL TRAINING ORCHESTRATOR")
        logger.info("=" * 80)
        logger.info(f"Mode: {mode.upper()}")
        logger.info(f"Models to train: {', '.join(models_to_train)}")
        logger.info(f"Test mode: {test_mode}")
        logger.info(f"Resume: {resume}")
        logger.info("=" * 80)

    def check_gpu_availability(self) -> Dict:
        """Check GPU status and availability"""
        try:
            gpus = GPUtil.getGPUs()
            if not gpus:
                logger.error("No GPU detected!")
                return {'available': False}

            gpu = gpus[0]  # RTX 4080
            return {
                'available': True,
                'name': gpu.name,
                'memory_total': gpu.memoryTotal,
                'memory_free': gpu.memoryFree,
                'memory_used': gpu.memoryUsed,
                'utilization': gpu.load * 100,
                'temperature': gpu.temperature
            }
        except Exception as e:
            logger.warning(f"Could not get GPU info: {e}")
            return {'available': False}

    def check_system_resources(self) -> Dict:
        """Check system RAM and disk space"""
        ram = psutil.virtual_memory()
        disk = psutil.disk_usage('.')

        return {
            'ram_total_gb': ram.total / (1024**3),
            'ram_available_gb': ram.available / (1024**3),
            'ram_percent': ram.percent,
            'disk_free_gb': disk.free / (1024**3)
        }

    def estimate_total_time(self) -> float:
        """Estimate total training time"""
        total_hours = sum(
            MODELS[model].estimated_time_hours
            for model in self.models_to_train
        )

        if self.test_mode:
            total_hours *= 0.1  # Test mode is ~10% of full training

        if self.mode == 'parallel':
            # Parallel mode depends on longest model
            total_hours = max(
                MODELS[model].estimated_time_hours
                for model in self.models_to_train
            )

        return total_hours

    def build_training_command(self, model_key: str) -> List[str]:
        """Build command line for training a model"""
        model = MODELS[model_key]
        cmd = ['python', model.script]

        # Add common arguments
        if model_key in ['lstm', 'transformer']:
            epochs = self.epochs_override or (5 if self.test_mode else model.default_epochs)
            cmd.extend(['--epochs', str(epochs)])
            cmd.extend(['--batch-size', str(self.batch_size)])
            if model_key == 'lstm':
                cmd.extend(['--stride', str(self.stride)])
        elif model_key in ['ppo', 'dqn']:
            episodes = self.epochs_override or (100 if self.test_mode else model.default_epochs)
            cmd.extend(['--episodes', str(episodes)])

        # Test mode flag
        if self.test_mode:
            cmd.append('--test-mode')

        # Resume flag
        if self.resume:
            cmd.append('--resume')

        return cmd

    def train_model(self, model_key: str) -> Dict:
        """Train a single model"""
        model = MODELS[model_key]

        logger.info("\n" + "=" * 80)
        logger.info(f"TRAINING: {model.name}")
        logger.info("=" * 80)
        logger.info(f"Script: {model.script}")
        logger.info(f"Estimated time: {model.estimated_time_hours:.1f} hours")
        logger.info(f"GPU memory needed: {model.gpu_memory_gb:.1f} GB")
        logger.info("=" * 80)

        # Check GPU availability
        gpu_info = self.check_gpu_availability()
        if gpu_info['available']:
            logger.info(f"GPU: {gpu_info['name']}")
            logger.info(f"VRAM Free: {gpu_info['memory_free']:.1f} MB")

            if gpu_info['memory_free'] < model.gpu_memory_gb * 1024:
                logger.warning(f"Low VRAM! May need {model.gpu_memory_gb:.1f} GB")

        # Build command
        cmd = self.build_training_command(model_key)
        logger.info(f"Command: {' '.join(cmd)}\n")

        # Execute training
        result = {
            'model': model.name,
            'start_time': datetime.now(),
            'command': ' '.join(cmd),
            'success': False,
            'error': None
        }

        try:
            start = time.time()

            # Run training subprocess
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )

            # Stream output
            for line in process.stdout:
                print(line, end='')
                sys.stdout.flush()

            # Wait for completion
            process.wait()

            if process.returncode == 0:
                result['success'] = True
                result['duration_hours'] = (time.time() - start) / 3600
                logger.info(f"\n[SUCCESS] {model.name} training completed!")
                logger.info(f"Duration: {result['duration_hours']:.2f} hours")
            else:
                result['error'] = f"Process exited with code {process.returncode}"
                logger.error(f"\n[FAILED] {model.name} training failed!")
                logger.error(f"Error: {result['error']}")

        except KeyboardInterrupt:
            logger.warning(f"\n[INTERRUPTED] {model.name} training interrupted by user")
            result['error'] = "Interrupted by user"
            if 'process' in locals():
                process.terminate()
            raise

        except Exception as e:
            logger.error(f"\n[ERROR] {model.name} training error: {e}")
            result['error'] = str(e)

        result['end_time'] = datetime.now()
        return result

    def train_sequential(self) -> Dict:
        """Train all models sequentially"""
        logger.info("\n" + "=" * 80)
        logger.info("SEQUENTIAL TRAINING MODE")
        logger.info("=" * 80)

        results = {}

        # Sort by priority
        models_sorted = sorted(
            self.models_to_train,
            key=lambda m: MODELS[m].priority
        )

        for i, model_key in enumerate(models_sorted, 1):
            logger.info(f"\nModel {i}/{len(models_sorted)}: {MODELS[model_key].name}")

            # Train model
            result = self.train_model(model_key)
            results[model_key] = result

            # Stop if failed (unless continuing on error)
            if not result['success']:
                logger.warning("Model training failed. Continuing with next model...")
                # Could add option to stop on first failure

        return results

    def train_parallel(self) -> Dict:
        """Train models in parallel (if enough resources)"""
        logger.info("\n" + "=" * 80)
        logger.info("PARALLEL TRAINING MODE")
        logger.info("=" * 80)
        logger.warning("Parallel mode requires significant GPU memory!")
        logger.warning("Recommended only for multi-GPU setups or small models")
        logger.info("=" * 80)

        # Check total GPU memory needed
        total_gpu_memory = sum(
            MODELS[model].gpu_memory_gb
            for model in self.models_to_train
        )

        gpu_info = self.check_gpu_availability()
        if gpu_info['available']:
            available_gb = gpu_info['memory_free'] / 1024
            if total_gpu_memory > available_gb:
                logger.error(f"Insufficient VRAM: Need {total_gpu_memory:.1f} GB, have {available_gb:.1f} GB")
                logger.info("Falling back to sequential mode...")
                return self.train_sequential()

        # Start all processes
        processes = {}
        results = {}

        for model_key in self.models_to_train:
            cmd = self.build_training_command(model_key)
            logger.info(f"Starting {MODELS[model_key].name}: {' '.join(cmd)}")

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            processes[model_key] = process

        # Wait for all to complete
        for model_key, process in processes.items():
            logger.info(f"Waiting for {MODELS[model_key].name} to complete...")
            process.wait()

            results[model_key] = {
                'model': MODELS[model_key].name,
                'success': process.returncode == 0,
                'return_code': process.returncode
            }

        return results

    def run(self) -> Dict:
        """Run the training orchestration"""
        self.start_time = datetime.now()

        # Print system info
        logger.info("\nSYSTEM STATUS:")
        gpu_info = self.check_gpu_availability()
        if gpu_info['available']:
            logger.info(f"  GPU: {gpu_info['name']}")
            logger.info(f"  VRAM: {gpu_info['memory_free']:.0f} MB free / {gpu_info['memory_total']:.0f} MB total")

        sys_info = self.check_system_resources()
        logger.info(f"  RAM: {sys_info['ram_available_gb']:.1f} GB free / {sys_info['ram_total_gb']:.1f} GB total")
        logger.info(f"  Disk: {sys_info['disk_free_gb']:.1f} GB free")

        # Estimate time
        estimated_hours = self.estimate_total_time()
        logger.info(f"\nEstimated training time: {estimated_hours:.1f} hours")
        logger.info(f"Expected completion: {datetime.now().replace(hour=(datetime.now().hour + int(estimated_hours)) % 24).strftime('%H:%M')}")

        # Execute training
        if self.mode == 'sequential':
            self.results = self.train_sequential()
        elif self.mode == 'parallel':
            self.results = self.train_parallel()
        else:
            logger.error(f"Unknown mode: {self.mode}")
            return {}

        # Generate summary
        self.print_summary()
        self.save_results()

        return self.results

    def print_summary(self):
        """Print training summary"""
        total_time = (datetime.now() - self.start_time).total_seconds() / 3600

        logger.info("\n" + "=" * 80)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total time: {total_time:.2f} hours")
        logger.info(f"Models trained: {len(self.results)}")

        successful = sum(1 for r in self.results.values() if r.get('success', False))
        failed = len(self.results) - successful

        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info("\nModel Results:")
        logger.info("-" * 80)

        for model_key, result in self.results.items():
            status = "[SUCCESS]" if result.get('success') else "[FAILED]"
            duration = result.get('duration_hours', 0)
            logger.info(f"{status} {MODELS[model_key].name:15} - {duration:.2f}h")

        logger.info("=" * 80)

    def save_results(self):
        """Save training results to file"""
        results_file = f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        results_data = {
            'timestamp': self.start_time.isoformat(),
            'mode': self.mode,
            'test_mode': self.test_mode,
            'models_trained': self.models_to_train,
            'results': {
                k: {
                    'model': v.get('model'),
                    'success': v.get('success'),
                    'duration_hours': v.get('duration_hours'),
                    'error': v.get('error')
                }
                for k, v in self.results.items()
            }
        }

        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)

        logger.info(f"\nResults saved to: {results_file}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Train all AI models for Trading Bot',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--mode',
        choices=['sequential', 'parallel'],
        default='sequential',
        help='Training mode (default: sequential)'
    )

    parser.add_argument(
        '--models',
        nargs='+',
        choices=list(MODELS.keys()),
        default=list(MODELS.keys()),
        help='Models to train (default: all)'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        help='Override epochs for all models'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=256,
        help='Batch size (default: 256)'
    )

    parser.add_argument(
        '--stride',
        type=int,
        default=5,
        help='Stride for LSTM (default: 5)'
    )

    parser.add_argument(
        '--test-mode',
        action='store_true',
        help='Quick test mode with reduced data'
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from checkpoints'
    )

    args = parser.parse_args()

    try:
        orchestrator = TrainingOrchestrator(
            models_to_train=args.models,
            mode=args.mode,
            test_mode=args.test_mode,
            resume=args.resume,
            epochs_override=args.epochs,
            batch_size=args.batch_size,
            stride=args.stride
        )

        results = orchestrator.run()

        # Exit code based on success
        all_successful = all(r.get('success', False) for r in results.values())
        sys.exit(0 if all_successful else 1)

    except KeyboardInterrupt:
        logger.warning("\n\nTraining interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
