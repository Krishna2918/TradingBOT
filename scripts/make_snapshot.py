"""
Data Snapshot Creator

Copy/serialize latest approved feature parquet/csv into a dated folder (no future rows).
Ensures monotonic timestamps, ffill/bfill, drop >1% NaN cols, write manifest.json
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import json
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import logging
import hashlib

# Add the src directory to the path to import our feature consistency system
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from ai.data.feature_consistency import FeatureConsistencyManager, FeatureConsistencyConfig

# Setup UTF-8 logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/make_snapshot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def calculate_file_hash(filepath):
    """Calculate SHA256 hash of file"""
    hash_sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

# The clean_and_validate_data function has been replaced by the FeatureConsistencyManager
# which provides more comprehensive and consistent data processing

def create_snapshot(source_dir, output_dir, cutoff_date=None):
    """Create data snapshot from source directory using the new FeatureConsistencyManager"""
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    if not source_path.exists():
        logger.error(f"Source directory not found: {source_dir}")
        return False
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set cutoff date (no future data)
    if cutoff_date is None:
        cutoff_date = datetime.now().date()
    else:
        cutoff_date = datetime.strptime(cutoff_date, "%Y-%m-%d").date()
    
    logger.info(f"Creating snapshot with cutoff date: {cutoff_date}")
    
    # Find all feature files
    feature_files = list(source_path.glob("*_features.parquet"))
    if not feature_files:
        feature_files = list(source_path.glob("*_features.csv"))
    
    if not feature_files:
        logger.error(f"No feature files found in {source_dir}")
        return False
    
    logger.info(f"Found {len(feature_files)} feature files")
    
    # Initialize the new FeatureConsistencyManager
    logger.info("=" * 80)
    logger.info("INITIALIZING FEATURE CONSISTENCY SYSTEM")
    logger.info("=" * 80)
    
    # Create configuration for snapshot creation
    config = FeatureConsistencyConfig(
        warmup_trim_days=200,
        nan_drop_threshold_per_symbol=0.05,  # 5% threshold
        global_feature_keep_ratio=0.95,      # Keep features in 95% of symbols
        min_symbol_feature_coverage=0.90,    # Skip symbols with <90% coverage
        use_missingness_mask=True,
        imputation_strategy="zero",
        manifest_path=str(output_path / "FEATURE_MANIFEST.json"),
        detailed_logging=True
    )
    
    # Initialize the feature consistency manager
    feature_manager = FeatureConsistencyManager(config)
    
    logger.info("Feature consistency system initialized")
    logger.info(f"Configuration: warmup_trim={config.warmup_trim_days}, "
               f"nan_threshold={config.nan_drop_threshold_per_symbol:.1%}, "
               f"global_keep_ratio={config.global_feature_keep_ratio:.1%}")
    
    # Load all symbol data
    logger.info("=" * 80)
    logger.info("LOADING SYMBOL DATA")
    logger.info("=" * 80)
    
    symbol_dataframes = {}
    loading_errors = []
    
    for file_path in feature_files:
        try:
            symbol = file_path.stem.replace('_features', '')
            logger.info(f"Loading {symbol}...")
            
            # Load data
            if file_path.suffix == '.parquet':
                df = pd.read_parquet(file_path)
            else:
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            
            # Filter out future data
            if hasattr(df.index, 'date'):
                original_rows = len(df)
                df = df[df.index.date <= cutoff_date]
                filtered_rows = len(df)
                if filtered_rows < original_rows:
                    logger.info(f"  Filtered future data: {original_rows} -> {filtered_rows} rows")
            
            if df.empty:
                logger.warning(f"  {symbol}: No data remaining after cutoff date filter")
                loading_errors.append(f"{symbol}: no_data_after_cutoff")
                continue
            
            symbol_dataframes[symbol] = df
            logger.info(f"  {symbol}: Loaded {len(df)} rows, {len(df.columns)} columns")
            
        except Exception as e:
            error_msg = f"{symbol}: loading_error_{str(e)[:50]}"
            logger.error(f"Error loading {file_path}: {e}")
            loading_errors.append(error_msg)
            continue
    
    if not symbol_dataframes:
        logger.error("No symbol data could be loaded")
        return False
    
    logger.info(f"Successfully loaded {len(symbol_dataframes)} symbols")
    if loading_errors:
        logger.warning(f"Failed to load {len(loading_errors)} symbols: {loading_errors[:5]}{'...' if len(loading_errors) > 5 else ''}")
    
    # Process symbols with the complete feature consistency pipeline
    logger.info("=" * 80)
    logger.info("PROCESSING WITH FEATURE CONSISTENCY PIPELINE")
    logger.info("=" * 80)
    
    try:
        # Use the complete pipeline that includes all functionality:
        # - Global coverage analysis and feature manifest generation
        # - Per-symbol feature selection enforcement
        # - Updated NaN threshold application
        # - Missingness mask generation and imputation
        # - Symbol coverage validation
        processed_symbols = feature_manager.process_symbols_with_complete_pipeline(symbol_dataframes)
        
        if not processed_symbols:
            logger.error("No symbols survived the feature consistency pipeline")
            return False
        
        # Validate tensor shapes
        logger.info("=" * 80)
        logger.info("TENSOR SHAPE VALIDATION")
        logger.info("=" * 80)
        
        is_valid, validation_report = feature_manager.validate_tensor_shapes(processed_symbols)
        
        if not is_valid:
            logger.error("Tensor shape validation failed!")
            logger.error("This indicates inconsistent feature processing - aborting snapshot creation")
            return False
        
        logger.info("✓ Tensor shape validation passed - all symbols have consistent shapes")
        
        # Detect and log feature drift
        if feature_manager.global_analysis_result:
            drift_analysis = feature_manager.detect_and_log_feature_drift(
                feature_manager.global_analysis_result.stable_features
            )
        
        # Save processed data
        logger.info("=" * 80)
        logger.info("SAVING PROCESSED DATA")
        logger.info("=" * 80)
        
        manifest = {
            'created_at': datetime.now().isoformat(),
            'cutoff_date': cutoff_date.isoformat(),
            'source_dir': str(source_path.absolute()),
            'feature_consistency_config': {
                'warmup_trim_days': config.warmup_trim_days,
                'nan_drop_threshold_per_symbol': config.nan_drop_threshold_per_symbol,
                'global_feature_keep_ratio': config.global_feature_keep_ratio,
                'min_symbol_feature_coverage': config.min_symbol_feature_coverage,
                'use_missingness_mask': config.use_missingness_mask,
                'imputation_strategy': config.imputation_strategy
            },
            'tensor_validation': validation_report['summary'],
            'files': {},
            'summary': {
                'total_files': 0,
                'total_rows': 0,
                'total_features': 0,
                'symbols': [],
                'excluded_symbols': []
            }
        }
        
        processed_files = 0
        total_rows = 0
        
        # Save each processed symbol
        for symbol, df in processed_symbols.items():
            try:
                # Save processed data
                output_file = output_path / f"{symbol}_features.parquet"
                df.to_parquet(output_file)
                
                # Calculate hash
                file_hash = calculate_file_hash(output_file)
                
                # Analyze final structure
                essential_columns = {'symbol', 'target', 'date', 'timestamp', 'direction_1d', 'close'}
                feature_cols = [c for c in df.columns if c not in essential_columns and not c.endswith('_isnan')]
                mask_cols = [c for c in df.columns if c.endswith('_isnan')]
                
                # Update manifest
                manifest['files'][symbol] = {
                    'filename': output_file.name,
                    'rows': len(df),
                    'columns': len(df.columns),
                    'feature_columns': len(feature_cols),
                    'mask_columns': len(mask_cols),
                    'hash': file_hash,
                    'date_range': {
                        'start': str(df.index.min()) if hasattr(df.index, 'min') else 'unknown',
                        'end': str(df.index.max()) if hasattr(df.index, 'max') else 'unknown'
                    }
                }
                
                processed_files += 1
                total_rows += len(df)
                manifest['summary']['symbols'].append(symbol)
                
                logger.info(f"Saved {symbol}: {len(df)} rows, {len(df.columns)} columns "
                           f"({len(feature_cols)} features, {len(mask_cols)} masks)")
                
            except Exception as e:
                logger.error(f"Error saving {symbol}: {e}")
                continue
        
        # Update summary
        manifest['summary']['total_files'] = processed_files
        manifest['summary']['total_rows'] = total_rows
        
        if feature_manager.global_analysis_result:
            manifest['summary']['total_features'] = len(feature_manager.global_analysis_result.stable_features)
            manifest['feature_manifest_path'] = config.manifest_path
        
        # Add exclusion information
        processing_summary = feature_manager.get_processing_summary()
        if processing_summary.get('symbols_excluded', 0) > 0:
            # Get excluded symbols from the difference
            all_loaded_symbols = set(symbol_dataframes.keys())
            processed_symbol_names = set(processed_symbols.keys())
            excluded_symbol_names = all_loaded_symbols - processed_symbol_names
            
            manifest['summary']['excluded_symbols'] = [
                f"{symbol}: excluded_during_processing" for symbol in excluded_symbol_names
            ]
        
        # Save manifest
        manifest_path = output_path / 'manifest.json'
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        
        # Log comprehensive summary
        feature_manager.log_comprehensive_pipeline_summary(processed_symbols)
        
        logger.info("=" * 80)
        logger.info("SNAPSHOT CREATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Output directory: {output_path}")
        logger.info(f"Files processed: {processed_files}")
        logger.info(f"Symbols excluded: {len(manifest['summary']['excluded_symbols'])}")
        logger.info(f"Total rows: {total_rows:,}")
        
        if feature_manager.global_analysis_result:
            logger.info(f"Stable features: {len(feature_manager.global_analysis_result.stable_features)}")
        
        logger.info(f"Symbols: {', '.join(manifest['summary']['symbols'][:10])}{'...' if len(manifest['summary']['symbols']) > 10 else ''}")
        logger.info(f"Manifest saved: {manifest_path}")
        logger.info(f"Feature manifest saved: {config.manifest_path}")
        
        return processed_files > 0
        
    except Exception as e:
        logger.error(f"Error in feature consistency pipeline: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Create data snapshot for training')
    parser.add_argument('--source', default='TrainingData/features', 
                       help='Source directory containing feature files')
    parser.add_argument('--out', required=True,
                       help='Output directory for snapshot')
    parser.add_argument('--cutoff-date', 
                       help='Cutoff date (YYYY-MM-DD), defaults to today')
    
    args = parser.parse_args()
    
    success = create_snapshot(args.source, args.out, args.cutoff_date)
    
    if success:
        logger.info("Snapshot creation successful")
        sys.exit(0)
    else:
        logger.error("Snapshot creation failed")
        sys.exit(1)

if __name__ == "__main__":
    main()