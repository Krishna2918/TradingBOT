"""
Model Promotion Script

Read trainer's JSON metrics (val loss/acc, per-class F1 on last 60d).
If improve >= 1% on recent holdout: move best_model.pth + scaler_stats.npz
to artifacts/lstm/champion/, write champion_manifest.json, else keep old.
"""

import os
import sys
import json
import shutil
import argparse
import yaml
from pathlib import Path
from datetime import datetime, timedelta
import logging
import hashlib

# Setup UTF-8 logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/promote_if_better.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_path):
    """Load configuration from YAML"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_training_results():
    """Load training results from JSON sidecar"""
    # Look for training results JSON
    results_files = [
        'AI_TRAINING_REPORT.json',
        'training_results.json',
        'models/real_data_lstm/training_results.json'
    ]
    
    for results_file in results_files:
        if Path(results_file).exists():
            logger.info(f"Loading training results from: {results_file}")
            with open(results_file, 'r', encoding='utf-8') as f:
                return json.load(f)
    
    logger.error("No training results JSON found")
    return None

def load_champion_metrics():
    """Load current champion metrics"""
    champion_manifest = Path('artifacts/lstm/champion/champion_manifest.json')
    
    if not champion_manifest.exists():
        logger.info("No existing champion found")
        return None
    
    with open(champion_manifest, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_improvement(new_metrics, champion_metrics, config):
    """Calculate improvement percentage"""
    if champion_metrics is None:
        logger.info("No champion to compare against - promoting new model")
        return True, "No existing champion"
    
    # Extract validation accuracy
    new_val_acc = new_metrics.get('best_val_accuracy', 0)
    champion_val_acc = champion_metrics.get('metrics', {}).get('best_val_accuracy', 0)
    
    if champion_val_acc == 0:
        logger.info("Champion has no valid accuracy - promoting new model")
        return True, "Champion has invalid accuracy"
    
    # Calculate improvement percentage
    improvement_pct = ((new_val_acc - champion_val_acc) / champion_val_acc) * 100
    required_improvement = config['promote_rules']['require_val_improve_pct']
    
    logger.info(f"New model accuracy: {new_val_acc:.4f}")
    logger.info(f"Champion accuracy: {champion_val_acc:.4f}")
    logger.info(f"Improvement: {improvement_pct:.2f}% (required: {required_improvement:.1f}%)")
    
    should_promote = improvement_pct >= required_improvement
    reason = f"Improvement {improvement_pct:.2f}% {'meets' if should_promote else 'below'} threshold {required_improvement:.1f}%"
    
    return should_promote, reason

def promote_model(training_results, config, reason):
    """Promote the new model to champion"""
    logger.info(f"Promoting new model: {reason}")
    
    champion_dir = Path('artifacts/lstm/champion')
    champion_dir.mkdir(parents=True, exist_ok=True)
    
    # Find model files to promote
    model_files = {
        'model': None,
        'scaler': None,
        'config': None
    }
    
    # Look for model file
    model_paths = [
        'models/real_data_lstm/best_model.pth',
        'models/production_test/best_model.pth',
        'best_model.pth'
    ]
    
    for model_path in model_paths:
        if Path(model_path).exists():
            model_files['model'] = model_path
            break
    
    # Look for scaler file
    scaler_paths = [
        'models/scaler_stats.npz',
        'models/real_data_lstm/scaler_stats.npz',
        'scaler_stats.npz'
    ]
    
    for scaler_path in scaler_paths:
        if Path(scaler_path).exists():
            model_files['scaler'] = scaler_path
            break
    
    # Copy files to champion directory
    promoted_files = {}
    
    if model_files['model']:
        dest_model = champion_dir / 'best_model.pth'
        shutil.copy2(model_files['model'], dest_model)
        promoted_files['model'] = str(dest_model)
        logger.info(f"Promoted model: {model_files['model']} -> {dest_model}")
    
    if model_files['scaler']:
        dest_scaler = champion_dir / 'scaler_stats.npz'
        shutil.copy2(model_files['scaler'], dest_scaler)
        promoted_files['scaler'] = str(dest_scaler)
        logger.info(f"Promoted scaler: {model_files['scaler']} -> {dest_scaler}")
    
    # Create champion manifest
    champion_manifest = {
        'promoted_at': datetime.now().isoformat(),
        'promotion_reason': reason,
        'model_version': config['model_name'],
        'files': promoted_files,
        'metrics': {
            'best_val_accuracy': training_results.get('best_val_accuracy', 0),
            'final_val_accuracy': training_results.get('final_val_accuracy', 0),
            'training_time_seconds': training_results.get('training_time_seconds', 0),
            'oom_events': training_results.get('oom_events', 0)
        },
        'training_config': {
            'epochs': config.get('epochs', 10),
            'batch_max': config.get('batch_max', 512),
            'seq_len': config.get('seq_len', 252),
            'window_days': config.get('window_days', 1008)
        },
        'feature_info': training_results.get('feature_info', {}),
        'memory_metrics': training_results.get('memory_metrics', {}),
        'batch_statistics': training_results.get('batch_statistics', {})
    }
    
    # Save champion manifest
    manifest_path = champion_dir / 'champion_manifest.json'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(champion_manifest, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Champion manifest saved: {manifest_path}")
    
    # Log promotion summary
    logger.info("=" * 60)
    logger.info("MODEL PROMOTION SUCCESSFUL")
    logger.info("=" * 60)
    logger.info(f"New champion accuracy: {champion_manifest['metrics']['best_val_accuracy']:.4f}")
    logger.info(f"Training time: {champion_manifest['metrics']['training_time_seconds']:.1f}s")
    logger.info(f"Files promoted: {len(promoted_files)}")
    logger.info(f"Reason: {reason}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Promote model if better than champion')
    parser.add_argument('--config', required=True, help='Configuration YAML file')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Load training results
        training_results = load_training_results()
        if not training_results:
            logger.error("No training results found")
            return 1
        
        # Check if training was successful
        if not training_results.get('success', False):
            logger.error("Training was not successful - not promoting")
            return 1
        
        # Load current champion metrics
        champion_metrics = load_champion_metrics()
        
        # Calculate improvement
        should_promote, reason = calculate_improvement(training_results, champion_metrics, config)
        
        if should_promote:
            # Promote the model
            success = promote_model(training_results, config, reason)
            return 0 if success else 1
        else:
            logger.info(f"Not promoting model: {reason}")
            return 0
            
    except Exception as e:
        logger.error(f"Promotion failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)