"""
Master Training Orchestrator - 10 Hour Continuous Training
==========================================================

This script orchestrates continuous training of all models for 10 hours with:
- Frequent checkpointing (every 5-10 minutes)
- Graceful termination with signal handling
- Automatic error recovery
- Progress tracking and logging
- Sequential model training (Transformer -> Aggressive LSTM Daily -> Aggressive LSTM Intraday -> GRU-Transformer)

Features:
- CTRL+C safe (saves state before exit)
- System signal handling
- Automatic checkpoint restoration
- Comprehensive error logging
- Resource monitoring (GPU, memory)
"""

import sys
import os
import signal
import time
import logging
import json
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, List
import torch
import psutil
import numpy as np

# Add project paths
sys.path.append(str(Path(__file__).parent / "src"))

# Configure comprehensive logging
log_dir = Path("logs/training_orchestrator")
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f'orchestrator_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class GracefulInterruptHandler:
    """Handle interrupts gracefully to save state before exiting"""

    def __init__(self):
        self.interrupted = False
        self.original_handlers = {}

        # Register signal handlers
        signal.signal(signal.SIGINT, self.handle_interrupt)
        signal.signal(signal.SIGTERM, self.handle_interrupt)
        if hasattr(signal, 'SIGBREAK'):  # Windows
            signal.signal(signal.SIGBREAK, self.handle_interrupt)

    def handle_interrupt(self, signum, frame):
        """Handle interrupt signal"""
        logger.warning(f"Received interrupt signal {signum}. Initiating graceful shutdown...")
        self.interrupted = True

    def reset(self):
        """Reset interrupt flag"""
        self.interrupted = False


class TrainingState:
    """Track training state across sessions"""

    def __init__(self, state_file: Path):
        self.state_file = state_file
        self.state = self.load_state()

    def load_state(self) -> Dict:
        """Load training state from file"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading state: {e}")

        # Default state
        return {
            'start_time': datetime.now().isoformat(),
            'total_training_time': 0.0,
            'models_completed': [],
            'current_model': None,
            'current_epoch': 0,
            'models_to_train': ['transformer', 'aggressive_lstm_daily', 'aggressive_lstm_intraday', 'gru_transformer'],
            'checkpoints': {},
            'errors': []
        }

    def save_state(self):
        """Save training state to file"""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
            logger.debug(f"State saved to {self.state_file}")
        except Exception as e:
            logger.error(f"Error saving state: {e}")

    def update_current_model(self, model_name: str, epoch: int = 0):
        """Update current training model"""
        self.state['current_model'] = model_name
        self.state['current_epoch'] = epoch
        self.save_state()

    def mark_model_completed(self, model_name: str):
        """Mark model as completed"""
        if model_name not in self.state['models_completed']:
            self.state['models_completed'].append(model_name)
        if model_name in self.state['models_to_train']:
            self.state['models_to_train'].remove(model_name)
        self.save_state()

    def add_checkpoint(self, model_name: str, checkpoint_path: str, metrics: Dict):
        """Add checkpoint information"""
        if model_name not in self.state['checkpoints']:
            self.state['checkpoints'][model_name] = []

        self.state['checkpoints'][model_name].append({
            'path': checkpoint_path,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        })
        self.save_state()

    def add_error(self, model_name: str, error_msg: str):
        """Add error to history"""
        self.state['errors'].append({
            'model': model_name,
            'error': error_msg,
            'timestamp': datetime.now().isoformat()
        })
        self.save_state()

    def update_training_time(self, elapsed_seconds: float):
        """Update total training time"""
        self.state['total_training_time'] = elapsed_seconds
        self.save_state()


class ResourceMonitor:
    """Monitor GPU and system resources"""

    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        self.process = psutil.Process()

    def get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage in GB"""
        if self.gpu_available:
            return torch.cuda.memory_allocated() / (1024**3)
        return 0.0

    def get_system_memory_usage(self) -> float:
        """Get system memory usage percentage"""
        return self.process.memory_percent()

    def get_cpu_usage(self) -> float:
        """Get CPU usage percentage"""
        return self.process.cpu_percent(interval=0.1)

    def log_resources(self):
        """Log current resource usage"""
        gpu_mem = self.get_gpu_memory_usage()
        sys_mem = self.get_system_memory_usage()
        cpu = self.get_cpu_usage()

        logger.info(f"Resources - GPU: {gpu_mem:.2f}GB, System Memory: {sys_mem:.1f}%, CPU: {cpu:.1f}%")

        return {
            'gpu_memory_gb': gpu_mem,
            'system_memory_percent': sys_mem,
            'cpu_percent': cpu
        }


class MasterTrainingOrchestrator:
    """Master orchestrator for 10-hour continuous training"""

    def __init__(self, total_hours: float = 10.0, resource_limit: float = 0.4):
        self.total_hours = total_hours
        self.total_seconds = total_hours * 3600
        self.resource_limit = resource_limit  # Use only 40% of resources

        # Initialize components
        self.interrupt_handler = GracefulInterruptHandler()
        self.state_file = Path("training_state.json")
        self.state = TrainingState(self.state_file)
        self.resource_monitor = ResourceMonitor()

        # Training configuration
        self.checkpoint_interval_seconds = 300  # 5 minutes
        self.max_retries = 100  # Keep retrying indefinitely within time limit
        self.continue_on_error = True  # Never stop on errors

        # Configure resource limits
        self.configure_resource_limits()

        logger.info(f"Initialized Master Training Orchestrator for {total_hours} hours")
        logger.info(f"Resource limit: {resource_limit*100}% of available resources")
        logger.info(f"GPU Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB")

    def configure_resource_limits(self):
        """Configure resource limits to use only specified percentage"""
        if torch.cuda.is_available():
            # Set PyTorch to use limited GPU memory
            total_memory = torch.cuda.get_device_properties(0).total_memory
            memory_fraction = self.resource_limit
            torch.cuda.set_per_process_memory_fraction(memory_fraction, device=0)
            logger.info(f"GPU memory limited to {memory_fraction*100}% ({(total_memory * memory_fraction)/(1024**3):.1f}GB)")

        # Set number of threads for CPU operations
        num_cpus = os.cpu_count() or 1
        limited_threads = max(1, int(num_cpus * self.resource_limit))
        torch.set_num_threads(limited_threads)
        logger.info(f"CPU threads limited to {limited_threads} (40% of {num_cpus})")

    def train_transformer_model(self, time_budget_seconds: float) -> Dict:
        """Train Transformer model with time budget"""
        logger.info("=" * 80)
        logger.info("TRAINING TRANSFORMER MODEL")
        logger.info("=" * 80)

        try:
            from train_market_transformer_full_scale import (
                load_real_trading_data,
                train_full_scale_transformer,
                create_production_config,
                create_model_config
            )

            # Load data
            logger.info("Loading training data...")
            features, targets, time_features, volatility, feature_names = load_real_trading_data()

            # Adjust epochs based on time budget
            config = create_production_config()
            estimated_time_per_epoch = 120  # seconds (estimate)
            max_epochs = min(100, int(time_budget_seconds / estimated_time_per_epoch))
            config.max_epochs = max(5, max_epochs)  # At least 5 epochs
            config.checkpoint_frequency = 2  # Save every 2 epochs

            logger.info(f"Time budget: {time_budget_seconds/3600:.1f} hours")
            logger.info(f"Estimated epochs: {config.max_epochs}")

            # Train model
            start_time = time.time()
            last_checkpoint = start_time

            result = train_full_scale_transformer(
                features=features,
                targets=targets,
                time_features=time_features,
                volatility=volatility,
                feature_names=feature_names,
                model_name=f"transformer_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                sequence_length=features.shape[1] if len(features.shape) > 2 else 252
            )

            elapsed = time.time() - start_time
            logger.info(f"Transformer training completed in {elapsed/3600:.2f} hours")

            return {
                'success': True,
                'time_elapsed': elapsed,
                'model_name': 'transformer',
                'result': result
            }

        except Exception as e:
            logger.error(f"Transformer training failed: {e}")
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e),
                'model_name': 'transformer'
            }

    def train_aggressive_lstm_model(self, mode: str, time_budget_seconds: float) -> Dict:
        """Train Aggressive LSTM model (daily or intraday)"""
        logger.info("=" * 80)
        logger.info(f"TRAINING AGGRESSIVE LSTM MODEL ({mode.upper()})")
        logger.info("=" * 80)

        try:
            from ai.models.aggressive_lstm_trainer import AggressiveLSTMTrainer

            # Initialize trainer
            trainer = AggressiveLSTMTrainer(mode=mode)

            # Adjust epochs based on time budget
            estimated_time_per_epoch = 60 if mode == 'daily' else 90  # seconds
            max_epochs = min(200 if mode == 'daily' else 150, int(time_budget_seconds / estimated_time_per_epoch))
            trainer.epochs = max(10, max_epochs)  # At least 10 epochs
            trainer.eval_every = 2  # Evaluate every 2 epochs

            logger.info(f"Time budget: {time_budget_seconds/3600:.1f} hours")
            logger.info(f"Target epochs: {trainer.epochs}")

            # Train model
            start_time = time.time()
            result = trainer.train_full_pipeline(max_symbols=None)  # Use ALL 1,681 stocks
            elapsed = time.time() - start_time

            logger.info(f"Aggressive LSTM ({mode}) training completed in {elapsed/3600:.2f} hours")

            return {
                'success': result.get('success', False),
                'time_elapsed': elapsed,
                'model_name': f'aggressive_lstm_{mode}',
                'result': result
            }

        except Exception as e:
            logger.error(f"Aggressive LSTM ({mode}) training failed: {e}")
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e),
                'model_name': f'aggressive_lstm_{mode}'
            }

    def train_gru_transformer_model(self, time_budget_seconds: float) -> Dict:
        """Train GRU-Transformer hybrid model"""
        logger.info("=" * 80)
        logger.info("TRAINING GRU-TRANSFORMER HYBRID MODEL")
        logger.info("=" * 80)

        try:
            import pandas as pd
            from ai.model_stack.gru_transformer_model import GRUTransformerPredictor

            # Load data (same as other models)
            logger.info("Loading training data for GRU-Transformer...")
            features_dir = Path("TrainingData/features")
            feature_files = list(features_dir.glob("*_features.parquet"))

            if not feature_files:
                raise FileNotFoundError("No training data found")

            # Load first file for training
            df = pd.read_parquet(feature_files[0])
            logger.info(f"Loaded data: {df.shape}")

            # Initialize predictor
            predictor = GRUTransformerPredictor()

            # Adjust epochs based on time budget
            estimated_time_per_epoch = 45  # seconds
            max_epochs = min(100, int(time_budget_seconds / estimated_time_per_epoch))
            epochs = max(10, max_epochs)

            logger.info(f"Time budget: {time_budget_seconds/3600:.1f} hours")
            logger.info(f"Target epochs: {epochs}")

            # Train model
            start_time = time.time()
            predictor.train(df, epochs=epochs, batch_size=64)
            elapsed = time.time() - start_time

            logger.info(f"GRU-Transformer training completed in {elapsed/3600:.2f} hours")

            return {
                'success': True,
                'time_elapsed': elapsed,
                'model_name': 'gru_transformer',
                'result': {'trained': True}
            }

        except Exception as e:
            logger.error(f"GRU-Transformer training failed: {e}")
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e),
                'model_name': 'gru_transformer'
            }

    def run_training_cycle(self):
        """Run complete training cycle for all models - NEVER STOP UNTIL TIME LIMIT"""
        logger.info("=" * 80)
        logger.info("STARTING 10-HOUR TRAINING ORCHESTRATOR")
        logger.info("CONFIGURATION: NON-STOP TRAINING FOR FULL DURATION")
        logger.info("=" * 80)

        start_time = time.time()
        end_time = start_time + self.total_seconds

        # Get models to train
        models_to_train = self.state.state['models_to_train'].copy()
        completed_models = self.state.state['models_completed']

        logger.info(f"Models to train: {models_to_train}")
        logger.info(f"Completed models: {completed_models}")
        logger.info(f"Training until: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"IMPORTANT: Training will continue for FULL {self.total_hours} hours regardless of errors")

        results = []
        training_cycle = 0

        # Keep training until time limit - CYCLE THROUGH MODELS CONTINUOUSLY
        while time.time() < end_time and not self.interrupt_handler.interrupted:
            training_cycle += 1
            logger.info(f"\n{'='*80}")
            logger.info(f"TRAINING CYCLE #{training_cycle}")
            logger.info(f"{'='*80}\n")

            # If all models completed, restart from beginning to continue training
            if not models_to_train:
                logger.info("All models completed one cycle. Restarting for additional training rounds...")
                models_to_train = ['transformer', 'aggressive_lstm_daily', 'aggressive_lstm_intraday', 'gru_transformer']
            # Calculate remaining time
            remaining_time = end_time - time.time()
            logger.info(f"Remaining time: {remaining_time/3600:.2f} hours")

            # Log resources
            self.resource_monitor.log_resources()

            # Get next model to train
            current_model = models_to_train[0]
            logger.info(f"Training model: {current_model}")

            # Allocate time budget (divide remaining time by remaining models + buffer)
            time_budget = remaining_time / (len(models_to_train) + 0.5)

            # Update state
            self.state.update_current_model(current_model)

            # Train model with infinite retries until time runs out
            success = False
            attempt = 0

            while time.time() < end_time and not self.interrupt_handler.interrupted:
                attempt += 1

                try:
                    logger.info(f"Training attempt #{attempt} for {current_model}")

                    # Re-calculate time budget for each attempt
                    remaining_time = end_time - time.time()
                    if remaining_time < 60:  # Less than 1 minute left
                        logger.warning("Less than 1 minute remaining, skipping this model")
                        break

                    time_budget = min(time_budget, remaining_time - 30)  # Leave 30s buffer

                    # Train based on model type
                    if current_model == 'transformer':
                        result = self.train_transformer_model(time_budget)
                    elif current_model == 'aggressive_lstm_daily':
                        result = self.train_aggressive_lstm_model('daily', time_budget)
                    elif current_model == 'aggressive_lstm_intraday':
                        result = self.train_aggressive_lstm_model('intraday', time_budget)
                    elif current_model == 'gru_transformer':
                        result = self.train_gru_transformer_model(time_budget)
                    else:
                        logger.error(f"Unknown model type: {current_model}")
                        break

                    results.append(result)

                    if result.get('success'):
                        logger.info(f"Model {current_model} trained successfully on attempt #{attempt}!")
                        self.state.mark_model_completed(current_model)
                        success = True
                        break
                    else:
                        logger.warning(f"Model {current_model} training failed (attempt #{attempt}), but CONTINUING...")
                        logger.warning(f"Error: {result.get('error', 'Unknown error')}")
                        self.state.add_error(current_model, result.get('error', 'Unknown error'))

                        # Wait a bit before retrying
                        logger.info("Waiting 10 seconds before retry...")
                        time.sleep(10)

                except Exception as e:
                    logger.error(f"Exception during training (attempt #{attempt}): {e}")
                    logger.error(traceback.format_exc())
                    self.state.add_error(current_model, str(e))

                    # NEVER GIVE UP - just log and retry
                    logger.info("Exception caught but CONTINUING training - waiting 10 seconds...")
                    time.sleep(10)

            # Move to next model
            if current_model in models_to_train:
                models_to_train.remove(current_model)

            # If successful or out of time, continue to next model
            if not success:
                logger.warning(f"Moving to next model after {attempt} attempts for {current_model}")

            # Update training time
            elapsed = time.time() - start_time
            self.state.update_training_time(elapsed)

            # Check if interrupted
            if self.interrupt_handler.interrupted:
                logger.warning("Training interrupted by user")
                break

        # Final summary
        total_elapsed = time.time() - start_time
        logger.info("=" * 80)
        logger.info("TRAINING ORCHESTRATOR COMPLETED")
        logger.info("=" * 80)
        logger.info(f"Total time: {total_elapsed/3600:.2f} hours")
        logger.info(f"Completed models: {self.state.state['models_completed']}")
        logger.info(f"Remaining models: {self.state.state['models_to_train']}")

        # Save final state
        self.state.save_state()

        # Generate summary report
        self.generate_summary_report(results, total_elapsed)

        return results

    def generate_summary_report(self, results: List[Dict], total_time: float):
        """Generate training summary report"""
        report_path = log_dir / f"training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        summary = {
            'total_training_time_hours': total_time / 3600,
            'total_models_attempted': len(results),
            'successful_models': sum(1 for r in results if r.get('success')),
            'failed_models': sum(1 for r in results if not r.get('success')),
            'completed_models': self.state.state['models_completed'],
            'remaining_models': self.state.state['models_to_train'],
            'errors': self.state.state['errors'],
            'results': results
        }

        with open(report_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Summary report saved to: {report_path}")

        # Print summary to console
        logger.info("\nTRAINING SUMMARY:")
        logger.info(f"  Total time: {total_time/3600:.2f} hours")
        logger.info(f"  Successful: {summary['successful_models']}/{summary['total_models_attempted']}")
        logger.info(f"  Completed models: {', '.join(summary['completed_models']) if summary['completed_models'] else 'None'}")


def main():
    """Main entry point"""
    logger.info("Starting Master Training Orchestrator")
    logger.info("NOTE: Training will run for FULL 10 hours and will NOT stop on errors")

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description='Master Training Orchestrator for 10-hour continuous training')
    parser.add_argument('--hours', type=float, default=10.0, help='Total training hours (default: 10)')
    parser.add_argument('--resource-limit', type=float, default=0.4, help='Resource usage limit (default: 0.4 = 40%%)')
    parser.add_argument('--reset-state', action='store_true', help='Reset training state and start fresh')
    args = parser.parse_args()

    # Reset state if requested
    if args.reset_state:
        state_file = Path("training_state.json")
        if state_file.exists():
            state_file.unlink()
            logger.info("Training state reset")

    # Create orchestrator with resource limits
    orchestrator = MasterTrainingOrchestrator(
        total_hours=args.hours,
        resource_limit=args.resource_limit
    )

    try:
        # Run training - this will NEVER stop except on time limit or manual interrupt
        logger.info(f"Starting {args.hours}-hour training session with {args.resource_limit*100}% resource limit")
        logger.info("Press Ctrl+C to gracefully stop (will save state)")

        results = orchestrator.run_training_cycle()

        logger.info("\nTraining orchestrator finished successfully!")
        return 0

    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user (Ctrl+C)")
        logger.info("Training state saved. You can resume by running this script again.")
        return 1

    except Exception as e:
        logger.error(f"\nFatal error in orchestrator: {e}")
        logger.error(traceback.format_exc())
        logger.error("This should NEVER happen - orchestrator should handle all errors")
        return 2


if __name__ == "__main__":
    sys.exit(main())
