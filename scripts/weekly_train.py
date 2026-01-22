"""
Weekly "Max-Power" Training Orchestrator

Runs all AI retrains sequentially every Saturday 18:00 America/Toronto for up to 60 min.
Uses frozen data snapshot, trains, tests on recent holdout, promotes only if beats champ.
"""

import os
import time
import json
import shutil
import pathlib
import datetime as dt
import subprocess
import sys
import yaml
import logging

CFG = "configs/lstm_weekly.yaml"
PY = sys.executable

# Setup UTF-8 logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/weekly_train.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run(cmd):
    """Run command with logging"""
    logger.info(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, check=False, capture_output=True, text=True)
    if result.stdout:
        logger.info(f"STDOUT: {result.stdout}")
    if result.stderr:
        logger.error(f"STDERR: {result.stderr}")
    return result

def ensure_dirs():
    """Ensure all required directories exist"""
    dirs = ["data/snapshots", "artifacts/lstm", "logs", "scripts", "artifacts/lstm/champion"]
    for p in dirs:
        pathlib.Path(p).mkdir(parents=True, exist_ok=True)
    logger.info("Directory structure verified")

def load_config():
    """Load configuration from YAML"""
    with open(CFG, 'r') as f:
        config = yaml.safe_load(f)
    return config

def make_snapshot():
    """Create data snapshot for training"""
    snap = dt.datetime.now().strftime("%Y-%m-%d")
    out = f"data/snapshots/{snap}"
    
    logger.info(f"Creating snapshot: {out}")
    result = run(f'{PY} scripts/make_snapshot.py --out "{out}"')
    
    if result.returncode == 0:
        logger.info(f"Snapshot created successfully: {out}")
        return out
    else:
        logger.error(f"Snapshot creation failed with code {result.returncode}")
        return None

def train_lstm(snapshot_dir, config):
    """Train LSTM model with given snapshot"""
    logger.info(f"Starting LSTM training with snapshot: {snapshot_dir}")
    
    cmd = (f'{PY} train_ai_model_with_real_data.py '
           f'--config {CFG} --snapshot "{snapshot_dir}" '
           f'--utf8-logs --timebox')
    
    result = run(cmd)
    
    if result.returncode == 0:
        logger.info("LSTM training completed successfully")
    else:
        logger.error(f"LSTM training failed with code {result.returncode}")
    
    return result.returncode

def gate_and_promote(config):
    """Gate and promote model if it beats champion"""
    logger.info("Starting model promotion evaluation")
    
    # Reads AI_TRAINING_REPORT.md / JSON sidecar produced by trainer
    result = run(f'{PY} scripts/promote_if_better.py --config {CFG}')
    
    if result.returncode == 0:
        logger.info("Model promotion completed successfully")
    else:
        logger.error(f"Model promotion failed with code {result.returncode}")
    
    return result.returncode

def main():
    """Main weekly training orchestrator"""
    logger.info("=" * 80)
    logger.info("WEEKLY MAX-POWER TRAINING STARTED")
    logger.info("=" * 80)
    
    start_time = time.time()
    config = load_config()
    
    # Ensure directory structure
    ensure_dirs()
    
    # Acquire GPU lock
    lock_file = pathlib.Path(config['runtime']['gpu_lockfile'])
    
    try:
        # Create exclusive GPU lock
        lock_file.write_text(f"locked_by_weekly_train_{dt.datetime.now().isoformat()}")
        logger.info("GPU lock acquired")
        
        # Step 1: Create data snapshot
        snapshot_dir = make_snapshot()
        if not snapshot_dir:
            logger.error("Failed to create snapshot, aborting")
            return 1
        
        # Step 2: Train LSTM
        train_rc = train_lstm(snapshot_dir, config)
        
        # Step 3: Gate and promote if training succeeded
        if train_rc == 0:
            promote_rc = gate_and_promote(config)
            if promote_rc == 0:
                logger.info("Training and promotion completed successfully")
            else:
                logger.warning("Training succeeded but promotion failed")
        else:
            logger.error("Training failed, skipping promotion")
        
        # Log final status
        elapsed_minutes = (time.time() - start_time) / 60
        timebox_minutes = config['runtime']['timebox_minutes']
        
        logger.info("=" * 80)
        logger.info("WEEKLY TRAINING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total time: {elapsed_minutes:.1f} minutes (limit: {timebox_minutes})")
        logger.info(f"Training result: {'SUCCESS' if train_rc == 0 else 'FAILED'}")
        logger.info(f"Snapshot: {snapshot_dir}")
        
        if elapsed_minutes > timebox_minutes:
            logger.warning(f"Training exceeded timebox by {elapsed_minutes - timebox_minutes:.1f} minutes")
        
        return train_rc
        
    except Exception as e:
        logger.error(f"Weekly training failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    finally:
        # Release GPU lock
        if lock_file.exists():
            lock_file.unlink()
            logger.info("GPU lock released")
        
        elapsed_minutes = (time.time() - start_time) / 60
        logger.info(f"Weekly run finished in {elapsed_minutes:.1f} min")

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)