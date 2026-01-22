"""
Rebuild direction_1d targets for all symbols with consistent methodology.

This script addresses class imbalance by using a wider neutral band and 
ensures all symbols use the same target creation logic.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
import argparse
from typing import List, Dict

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from ai.target_engineering import (
    rebuild_targets_for_symbol, 
    analyze_target_distribution,
    build_direction_targets
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def rebuild_all_targets(features_dir: str = "TrainingData/features",
                       horizon: int = 1,
                       neutral_band: float = 0.005,
                       min_flat_pct: float = 15.0,
                       dry_run: bool = False) -> Dict:
    """
    Rebuild targets for all symbols in the features directory.
    
    Args:
        features_dir: Directory containing feature parquet files
        horizon: Forward horizon in days
        neutral_band: Neutral band threshold (±0.5% = 0.005)
        min_flat_pct: Minimum percentage of FLAT samples required
        dry_run: If True, don't save changes, just analyze
        
    Returns:
        Dictionary with rebuild statistics
    """
    features_path = Path(features_dir)
    if not features_path.exists():
        raise ValueError(f"Features directory not found: {features_path}")
    
    # Find all parquet files
    parquet_files = list(features_path.glob("*_features.parquet"))
    logger.info(f"Found {len(parquet_files)} feature files to process")
    
    results = {
        'processed_symbols': [],
        'skipped_symbols': [],
        'total_stats': {'down': 0, 'flat': 0, 'up': 0},
        'symbol_stats': {},
        'config': {
            'horizon': horizon,
            'neutral_band': neutral_band,
            'min_flat_pct': min_flat_pct
        }
    }
    
    for parquet_file in parquet_files:
        symbol = parquet_file.stem.replace('_features', '')
        logger.info(f"\nProcessing {symbol}...")
        
        try:
            # Load data
            df = pd.read_parquet(parquet_file)
            logger.info(f"Loaded {len(df)} rows for {symbol}")
            
            if 'close' not in df.columns:
                logger.warning(f"Skipping {symbol}: no 'close' column")
                results['skipped_symbols'].append(symbol)
                continue
            
            # Rebuild targets
            df_updated = rebuild_targets_for_symbol(
                df, horizon=horizon, neutral_band=neutral_band, symbol=symbol
            )
            
            # Analyze new distribution
            new_targets = df_updated['direction_1d'].dropna()
            stats = analyze_target_distribution(new_targets, symbol)
            
            # Check if FLAT percentage meets minimum
            if stats['flat_pct'] < min_flat_pct:
                logger.warning(f"{symbol}: FLAT class only {stats['flat_pct']:.1f}% "
                             f"(minimum: {min_flat_pct}%). Consider wider neutral band.")
            
            # Update totals
            results['total_stats']['down'] += stats['down_count']
            results['total_stats']['flat'] += stats['flat_count']
            results['total_stats']['up'] += stats['up_count']
            results['symbol_stats'][symbol] = stats
            
            # Save updated data (unless dry run)
            if not dry_run:
                df_updated.to_parquet(parquet_file)
                logger.info(f"Updated {symbol} targets saved to {parquet_file}")
            else:
                logger.info(f"DRY RUN: Would update {symbol} targets")
            
            results['processed_symbols'].append(symbol)
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            results['skipped_symbols'].append(symbol)
    
    # Log overall statistics
    total_samples = sum(results['total_stats'].values())
    if total_samples > 0:
        logger.info(f"\n=== OVERALL STATISTICS ===")
        logger.info(f"Total samples: {total_samples:,}")
        logger.info(f"DOWN: {results['total_stats']['down']:,} "
                   f"({results['total_stats']['down']/total_samples*100:.1f}%)")
        logger.info(f"FLAT: {results['total_stats']['flat']:,} "
                   f"({results['total_stats']['flat']/total_samples*100:.1f}%)")
        logger.info(f"UP: {results['total_stats']['up']:,} "
                   f"({results['total_stats']['up']/total_samples*100:.1f}%)")
        
        flat_pct = results['total_stats']['flat']/total_samples*100
        if flat_pct < min_flat_pct:
            logger.warning(f"Overall FLAT percentage ({flat_pct:.1f}%) is below minimum ({min_flat_pct}%)")
            logger.warning(f"Consider increasing neutral_band from {neutral_band} to {neutral_band * 1.5:.4f}")
    
    logger.info(f"\nProcessed: {len(results['processed_symbols'])} symbols")
    logger.info(f"Skipped: {len(results['skipped_symbols'])} symbols")
    
    return results

def test_different_bands(features_dir: str = "TrainingData/features",
                        test_symbols: List[str] = None,
                        bands_to_test: List[float] = None) -> Dict:
    """
    Test different neutral bands to find optimal class balance.
    
    Args:
        features_dir: Directory containing feature parquet files
        test_symbols: List of symbols to test (default: first 3 found)
        bands_to_test: List of neutral bands to test
        
    Returns:
        Dictionary with test results
    """
    if bands_to_test is None:
        bands_to_test = [0.003, 0.005, 0.007, 0.010]  # 0.3%, 0.5%, 0.7%, 1.0%
    
    features_path = Path(features_dir)
    parquet_files = list(features_path.glob("*_features.parquet"))
    
    if test_symbols is None:
        test_symbols = [f.stem.replace('_features', '') for f in parquet_files[:3]]
    
    logger.info(f"Testing neutral bands {bands_to_test} on symbols {test_symbols}")
    
    results = {}
    
    for band in bands_to_test:
        logger.info(f"\n=== Testing neutral band: ±{band*100:.1f}% ===")
        band_results = {'symbols': {}, 'totals': {'down': 0, 'flat': 0, 'up': 0}}
        
        for symbol in test_symbols:
            parquet_file = features_path / f"{symbol}_features.parquet"
            if not parquet_file.exists():
                logger.warning(f"File not found: {parquet_file}")
                continue
            
            try:
                df = pd.read_parquet(parquet_file)
                if 'close' not in df.columns:
                    continue
                
                # Test this band
                targets = build_direction_targets(df['close'], horizon=1, neutral_band=band)
                stats = analyze_target_distribution(targets.dropna(), f"{symbol} (±{band*100:.1f}%)")
                
                band_results['symbols'][symbol] = stats
                band_results['totals']['down'] += stats['down_count']
                band_results['totals']['flat'] += stats['flat_count']
                band_results['totals']['up'] += stats['up_count']
                
            except Exception as e:
                logger.error(f"Error testing {symbol} with band {band}: {e}")
        
        # Calculate overall percentages
        total = sum(band_results['totals'].values())
        if total > 0:
            flat_pct = band_results['totals']['flat'] / total * 100
            logger.info(f"Band ±{band*100:.1f}%: Overall FLAT = {flat_pct:.1f}%")
            band_results['overall_flat_pct'] = flat_pct
        
        results[band] = band_results
    
    # Find best band
    best_band = None
    best_flat_pct = 0
    target_flat_pct = 20.0  # Target around 20% FLAT
    
    for band, result in results.items():
        flat_pct = result.get('overall_flat_pct', 0)
        if abs(flat_pct - target_flat_pct) < abs(best_flat_pct - target_flat_pct):
            best_band = band
            best_flat_pct = flat_pct
    
    logger.info(f"\n=== RECOMMENDATION ===")
    logger.info(f"Best neutral band: ±{best_band*100:.1f}% (FLAT: {best_flat_pct:.1f}%)")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Rebuild direction_1d targets")
    parser.add_argument("--features-dir", default="TrainingData/features",
                       help="Features directory path")
    parser.add_argument("--horizon", type=int, default=1,
                       help="Forward horizon in days")
    parser.add_argument("--neutral-band", type=float, default=0.005,
                       help="Neutral band threshold (0.005 = ±0.5%)")
    parser.add_argument("--min-flat-pct", type=float, default=15.0,
                       help="Minimum FLAT percentage required")
    parser.add_argument("--dry-run", action="store_true",
                       help="Don't save changes, just analyze")
    parser.add_argument("--test-bands", action="store_true",
                       help="Test different neutral bands first")
    
    args = parser.parse_args()
    
    try:
        if args.test_bands:
            logger.info("Testing different neutral bands...")
            test_results = test_different_bands(args.features_dir)
            return
        
        logger.info("Rebuilding targets for all symbols...")
        results = rebuild_all_targets(
            features_dir=args.features_dir,
            horizon=args.horizon,
            neutral_band=args.neutral_band,
            min_flat_pct=args.min_flat_pct,
            dry_run=args.dry_run
        )
        
        if not args.dry_run:
            logger.info("✅ Target rebuild complete!")
        else:
            logger.info("✅ Dry run complete - no files modified")
            
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()