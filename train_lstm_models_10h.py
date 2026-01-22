"""
10-Hour LSTM Training Session
==============================

Train LSTM models continuously for 10 hours with:
- Aggressive LSTM Daily (200 epochs target)
- Aggressive LSTM Intraday (150 epochs target)
- Multiple training cycles to maximize model refinement
- Automatic checkpointing and error recovery
- 40% resource usage limit
"""

import sys
import os
import time
import signal
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).parent / "src"))

from ai.models.aggressive_lstm_trainer import AggressiveLSTMTrainer

# Setup logging
log_dir = Path("logs/lstm_training_10h")
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f'lstm_10h_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class GracefulKiller:
    """Handle graceful shutdown"""
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        logger.warning(f"Received signal {signum}, initiating graceful shutdown...")
        self.kill_now = True


def configure_resources(resource_limit=0.4):
    """Configure resource limits"""
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(resource_limit, device=0)
        logger.info(f"GPU memory limited to {resource_limit*100}% ({(total_memory * resource_limit)/(1024**3):.1f}GB)")

    num_cpus = os.cpu_count() or 1
    limited_threads = max(1, int(num_cpus * resource_limit))
    torch.set_num_threads(limited_threads)
    logger.info(f"CPU threads limited to {limited_threads} ({resource_limit*100}% of {num_cpus})")


def train_lstm_model(mode: str, epochs_target: int, time_budget_seconds: float, killer: GracefulKiller):
    """Train LSTM model with time budget"""
    logger.info("=" * 80)
    logger.info(f"TRAINING AGGRESSIVE LSTM MODEL ({mode.upper()})")
    logger.info(f"Target Epochs: {epochs_target}")
    logger.info(f"Time Budget: {time_budget_seconds/3600:.1f} hours")
    logger.info("=" * 80)

    try:
        # Initialize trainer
        trainer = AggressiveLSTMTrainer(mode=mode)

        # Set training parameters
        trainer.epochs = epochs_target
        trainer.eval_every = 5  # Evaluate every 5 epochs
        trainer.patience = 20  # Be patient for better results

        logger.info(f"Starting training with {trainer.epochs} epochs target")
        logger.info(f"Model: {trainer.hidden_size}x{trainer.num_layers} layers")
        logger.info(f"Sequence length: {trainer.sequence_length}")
        logger.info(f"Batch size: {trainer.batch_size}")
        logger.info(f"Learning rate: {trainer.learning_rate} -> {trainer.min_lr}")

        # Train model
        start_time = time.time()
        result = trainer.train_full_pipeline(max_symbols=10)  # Use more symbols
        elapsed = time.time() - start_time

        logger.info(f"Training completed in {elapsed/3600:.2f} hours")

        if result.get('success'):
            logger.info(f"SUCCESS - Accuracy: {result['accuracy']:.4f}")
            logger.info(f"Trained on {result['training_samples']:,} samples")
            logger.info(f"Model parameters: {result['model_parameters']:,}")
        else:
            logger.error(f"Training failed: {result.get('error', 'Unknown error')}")

        return result

    except Exception as e:
        logger.error(f"Exception during training: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }


def main():
    """Main 10-hour training session"""
    logger.info("=" * 80)
    logger.info("10-HOUR LSTM MODEL TRAINING SESSION")
    logger.info("=" * 80)

    # Configuration
    total_hours = 10.0
    resource_limit = 0.4

    # Configure resources
    configure_resources(resource_limit)

    # Setup graceful shutdown
    killer = GracefulKiller()

    # Calculate end time
    start_time = time.time()
    end_time = start_time + (total_hours * 3600)

    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"End time: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Resource limit: {resource_limit*100}%")
    logger.info(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # Training plan
    training_plan = [
        {'mode': 'daily', 'epochs': 200, 'priority': 1},
        {'mode': 'intraday', 'epochs': 150, 'priority': 2},
    ]

    results = []
    cycle = 0

    # Train continuously for 10 hours
    while time.time() < end_time and not killer.kill_now:
        cycle += 1
        remaining_time = end_time - time.time()

        logger.info("\n" + "=" * 80)
        logger.info(f"TRAINING CYCLE #{cycle}")
        logger.info(f"Remaining time: {remaining_time/3600:.2f} hours")
        logger.info("=" * 80 + "\n")

        if remaining_time < 300:  # Less than 5 minutes
            logger.info("Less than 5 minutes remaining, stopping training")
            break

        # Train each model in the plan
        for plan in training_plan:
            if killer.kill_now:
                break

            remaining_time = end_time - time.time()
            if remaining_time < 300:
                break

            # Allocate time budget (split remaining time among models)
            time_budget = remaining_time / len(training_plan)
            time_budget = min(time_budget, remaining_time - 60)  # Leave 1 minute buffer

            logger.info(f"\nTraining {plan['mode']} model (Cycle {cycle})")
            logger.info(f"Time budget: {time_budget/3600:.2f} hours")

            # Adjust epochs based on remaining time
            estimated_time_per_epoch = 30 if plan['mode'] == 'daily' else 45
            max_epochs = int(time_budget / estimated_time_per_epoch)
            epochs_to_run = min(plan['epochs'], max(5, max_epochs))

            logger.info(f"Running {epochs_to_run} epochs")

            # Train model
            result = train_lstm_model(
                mode=plan['mode'],
                epochs_target=epochs_to_run,
                time_budget_seconds=time_budget,
                killer=killer
            )

            results.append({
                'cycle': cycle,
                'model': plan['mode'],
                'result': result,
                'timestamp': datetime.now().isoformat()
            })

            # Save intermediate results
            results_file = log_dir / 'training_results.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)

    # Final summary
    total_elapsed = time.time() - start_time
    logger.info("\n" + "=" * 80)
    logger.info("10-HOUR TRAINING SESSION COMPLETED")
    logger.info("=" * 80)
    logger.info(f"Total time: {total_elapsed/3600:.2f} hours")
    logger.info(f"Training cycles: {cycle}")
    logger.info(f"Models trained: {len(results)}")

    # Count successes
    successful = sum(1 for r in results if r['result'].get('success', False))
    logger.info(f"Successful trainings: {successful}/{len(results)}")

    # Save final results
    final_results = {
        'total_time_hours': total_elapsed / 3600,
        'cycles': cycle,
        'total_trainings': len(results),
        'successful_trainings': successful,
        'results': results
    }

    final_file = log_dir / f'final_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(final_file, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)

    logger.info(f"\nResults saved to: {final_file}")
    logger.info("\nTraining session completed successfully!")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)
