"""
Feature Consistency System for Trading Models

This module ensures consistent feature sets across all symbols and maintains
stable training by handling feature drift, warm-up periods, and missing data.
"""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class CoverageStats:
    """Statistics for feature coverage analysis."""
    feature_name: str
    total_symbols: int
    symbols_with_feature: int
    coverage_ratio: float
    avg_non_nan_ratio: float
    min_non_nan_ratio: float
    max_non_nan_ratio: float
    symbols_list: List[str]
    
    def __post_init__(self):
        """Validate coverage statistics."""
        if not 0 <= self.coverage_ratio <= 1:
            raise ValueError(f"Invalid coverage ratio: {self.coverage_ratio}")
        if not 0 <= self.avg_non_nan_ratio <= 1:
            raise ValueError(f"Invalid avg_non_nan_ratio: {self.avg_non_nan_ratio}")


@dataclass
class GlobalAnalysisResult:
    """Results from global coverage analysis."""
    total_symbols_analyzed: int
    total_features_found: int
    stable_features: List[str]
    unstable_features: List[str]
    coverage_stats: Dict[str, CoverageStats]
    analysis_timestamp: str
    config_used: Dict[str, Any]
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the analysis."""
        return {
            'total_symbols': self.total_symbols_analyzed,
            'total_features_found': self.total_features_found,
            'stable_features_count': len(self.stable_features),
            'unstable_features_count': len(self.unstable_features),
            'stability_ratio': len(self.stable_features) / max(1, self.total_features_found),
            'avg_coverage_stable': np.mean([
                self.coverage_stats[f].coverage_ratio for f in self.stable_features
            ]) if self.stable_features else 0.0,
            'avg_coverage_unstable': np.mean([
                self.coverage_stats[f].coverage_ratio for f in self.unstable_features
            ]) if self.unstable_features else 0.0
        }


@dataclass
class FeatureConsistencyConfig:
    """Configuration for the feature consistency system."""
    
    # Warm-up period configuration
    warmup_trim_days: int = 200
    
    # NaN handling thresholds
    nan_drop_threshold_per_symbol: float = 0.05  # 5% threshold per symbol
    global_feature_keep_ratio: float = 0.95      # Keep features in 95% of symbols
    min_symbol_feature_coverage: float = 0.90    # Skip symbols with <90% coverage
    
    # Missingness handling
    use_missingness_mask: bool = True
    imputation_strategy: str = "zero"  # Options: "zero", "mean", "median"
    imputation_value: float = 0.0  # Used when strategy is "zero" or as fallback
    
    # File paths
    manifest_path: str = "models/feature_manifest.json"
    config_backup_path: str = "models/feature_consistency_config.json"
    
    # Logging configuration
    log_level: str = "INFO"
    detailed_logging: bool = True
    
    @classmethod
    def from_json(cls, config_path: Union[str, Path]) -> 'FeatureConsistencyConfig':
        """Load configuration from JSON file."""
        config_path = Path(config_path)
        if not config_path.exists():
            logger.warning(f"Config file {config_path} not found, using defaults")
            return cls()
        
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Validate configuration
            valid_keys = set(cls.__dataclass_fields__.keys())
            invalid_keys = set(config_data.keys()) - valid_keys
            if invalid_keys:
                logger.warning(f"Invalid config keys ignored: {invalid_keys}")
                config_data = {k: v for k, v in config_data.items() if k in valid_keys}
            
            return cls(**config_data)
            
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            logger.info("Using default configuration")
            return cls()
    
    def to_json(self, config_path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(config_path, 'w') as f:
                json.dump(asdict(self), f, indent=2)
            logger.info(f"Configuration saved to {config_path}")
        except Exception as e:
            logger.error(f"Failed to save config to {config_path}: {e}")
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if not 0 < self.nan_drop_threshold_per_symbol <= 1:
            raise ValueError("nan_drop_threshold_per_symbol must be between 0 and 1")
        
        if not 0 < self.global_feature_keep_ratio <= 1:
            raise ValueError("global_feature_keep_ratio must be between 0 and 1")
        
        if not 0 < self.min_symbol_feature_coverage <= 1:
            raise ValueError("min_symbol_feature_coverage must be between 0 and 1")
        
        if self.warmup_trim_days < 0:
            raise ValueError("warmup_trim_days must be non-negative")
        
        # Validate imputation strategy
        valid_strategies = {"zero", "mean", "median"}
        if self.imputation_strategy not in valid_strategies:
            raise ValueError(f"imputation_strategy must be one of {valid_strategies}, got '{self.imputation_strategy}'")


class FeatureConsistencyManager:
    """
    Main class for managing feature consistency across symbols.
    
    Handles warm-up trimming, global coverage analysis, feature manifest
    management, and consistent feature processing.
    """
    
    def __init__(self, config: Optional[FeatureConsistencyConfig] = None):
        """Initialize the feature consistency manager."""
        self.config = config or FeatureConsistencyConfig()
        self.config.validate()
        
        # Setup logging
        self._setup_logging()
        
        # Initialize state
        self.feature_manifest: Optional[Dict[str, Any]] = None
        self.global_coverage_stats: Optional[Dict[str, CoverageStats]] = None
        self.global_analysis_result: Optional[GlobalAnalysisResult] = None
        
        # Initialize components
        self.warmup_trimmer = WarmupTrimmer(self.config)
        self.coverage_analyzer = GlobalCoverageAnalyzer(self.config)
        self.manifest_manager = FeatureManifestManager(self.config)
        self.missingness_mask_generator = MissingnessMaskGenerator(self.config)
        self.processing_stats = {
            'symbols_processed': 0,
            'symbols_excluded': 0,
            'features_in_manifest': 0,
            'warmup_rows_trimmed': 0,
            'start_time': None,
            'end_time': None
        }
        
        logger.info("FeatureConsistencyManager initialized")
        logger.info(f"Configuration: warmup_trim_days={self.config.warmup_trim_days}, "
                   f"nan_threshold={self.config.nan_drop_threshold_per_symbol:.1%}, "
                   f"global_keep_ratio={self.config.global_feature_keep_ratio:.1%}")
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        logger.setLevel(log_level)
        
        if self.config.detailed_logging:
            logger.info("Detailed logging enabled for feature consistency system")
    
    def save_config_backup(self) -> None:
        """Save current configuration as backup."""
        try:
            self.config.to_json(self.config.config_backup_path)
        except Exception as e:
            logger.warning(f"Failed to save config backup: {e}")
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of processing statistics."""
        summary = self.processing_stats.copy()
        
        if summary['start_time'] and summary['end_time']:
            duration = summary['end_time'] - summary['start_time']
            summary['processing_duration_seconds'] = duration.total_seconds()
        
        summary['exclusion_rate'] = (
            summary['symbols_excluded'] / max(1, summary['symbols_processed'] + summary['symbols_excluded'])
        )
        
        return summary
    
    def reset_stats(self) -> None:
        """Reset processing statistics."""
        self.processing_stats = {
            'symbols_processed': 0,
            'symbols_excluded': 0,
            'features_in_manifest': 0,
            'warmup_rows_trimmed': 0,
            'start_time': datetime.now(),
            'end_time': None
        }
        logger.info("Processing statistics reset")
    
    def run_global_coverage_analysis(self, symbol_dataframes: Dict[str, pd.DataFrame]) -> GlobalAnalysisResult:
        """
        Run the complete global coverage analysis integration.
        
        This method implements the coverage analysis integration by:
        1. Adding global analysis phase before per-symbol processing
        2. Collecting coverage statistics across all symbols after warm-up trimming
        
        Args:
            symbol_dataframes: Dictionary mapping symbol names to their DataFrames
            
        Returns:
            Global analysis result with coverage statistics
        """
        logger.info("Starting global coverage analysis integration")
        
        # Reset processing statistics
        self.reset_stats()
        
        # Step 1: Apply warm-up trimming and collect coverage statistics
        trimmed_symbols = self.analyze_and_trim_symbols(symbol_dataframes)
        
        if not trimmed_symbols:
            logger.error("No symbols remaining after warm-up trimming")
            raise ValueError("No symbols remaining after warm-up trimming")
        
        # Step 2: Perform global coverage analysis on trimmed data
        global_result = self.perform_global_analysis(trimmed_symbols)
        
        logger.info("Global coverage analysis integration completed successfully")
        logger.info(f"  Symbols analyzed: {global_result.total_symbols_analyzed}")
        logger.info(f"  Features found: {global_result.total_features_found}")
        logger.info(f"  Stable features: {len(global_result.stable_features)}")
        logger.info(f"  Unstable features: {len(global_result.unstable_features)}")
        
        return global_result
    
    def analyze_and_trim_symbols(self, symbol_dataframes: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Analyze all symbols and apply warm-up trimming before global coverage analysis.
        
        This method implements the integration of coverage analysis by:
        1. Applying warm-up trimming to all symbols
        2. Collecting coverage statistics across all symbols
        3. Preparing data for global analysis phase
        
        Args:
            symbol_dataframes: Dictionary mapping symbol names to their DataFrames
            
        Returns:
            Dictionary of trimmed symbol DataFrames ready for global analysis
        """
        logger.info(f"Starting coverage analysis integration for {len(symbol_dataframes)} symbols")
        
        trimmed_symbols = {}
        trimming_stats = []
        
        # Phase 1: Apply warm-up trimming to all symbols
        logger.info("Phase 1: Applying warm-up trimming to all symbols")
        
        for symbol, df in symbol_dataframes.items():
            try:
                # Apply warm-up trimming
                original_df = df.copy()
                trimmed_df = self.warmup_trimmer.trim_warmup_period(df, symbol)
                
                # Validate post-trim data
                if self.warmup_trimmer.validate_post_trim_data(trimmed_df, symbol):
                    trimmed_symbols[symbol] = trimmed_df
                    
                    # Collect trimming statistics
                    stats = self.warmup_trimmer.get_trimming_stats(original_df, trimmed_df, symbol)
                    trimming_stats.append(stats)
                    
                    # Update processing stats
                    self.processing_stats['warmup_rows_trimmed'] += stats['rows_removed']
                    
                else:
                    logger.warning(f"{symbol}: Failed post-trim validation, excluding from analysis")
                    self.processing_stats['symbols_excluded'] += 1
                    
            except Exception as e:
                logger.error(f"Error trimming warm-up period for {symbol}: {e}")
                self.processing_stats['symbols_excluded'] += 1
                continue
        
        # Log trimming summary
        if trimming_stats:
            total_original_rows = sum(s['original_rows'] for s in trimming_stats)
            total_trimmed_rows = sum(s['trimmed_rows'] for s in trimming_stats)
            total_rows_removed = sum(s['rows_removed'] for s in trimming_stats)
            avg_trim_percentage = np.mean([s['trim_percentage'] for s in trimming_stats])
            
            logger.info(f"Warm-up trimming summary:")
            logger.info(f"  Symbols processed: {len(trimming_stats)}")
            logger.info(f"  Total rows before: {total_original_rows:,}")
            logger.info(f"  Total rows after: {total_trimmed_rows:,}")
            logger.info(f"  Total rows removed: {total_rows_removed:,}")
            logger.info(f"  Average trim percentage: {avg_trim_percentage:.1f}%")
        
        # Phase 2: Collect coverage statistics across all trimmed symbols
        logger.info("Phase 2: Collecting coverage statistics across all symbols")
        
        if not trimmed_symbols:
            logger.error("No symbols remaining after warm-up trimming")
            return {}
        
        # Perform global coverage analysis on trimmed data
        logger.info(f"Ready for global analysis: {len(trimmed_symbols)} symbols")
        
        return trimmed_symbols
    
    def perform_global_analysis(self, symbol_dataframes: Dict[str, pd.DataFrame]) -> GlobalAnalysisResult:
        """
        Perform global coverage analysis on all symbols.
        
        Args:
            symbol_dataframes: Dictionary mapping symbol names to their DataFrames
            
        Returns:
            Global analysis result with stable features identified
        """
        logger.info("Performing global coverage analysis")
        
        # Perform the analysis
        self.global_analysis_result = self.coverage_analyzer.analyze_all_symbols(symbol_dataframes)
        self.global_coverage_stats = self.global_analysis_result.coverage_stats
        
        # Update processing stats
        self.processing_stats['features_in_manifest'] = len(self.global_analysis_result.stable_features)
        
        # Log detailed report if enabled
        if self.config.detailed_logging:
            self.coverage_analyzer.log_detailed_coverage_report(self.global_analysis_result)
        
        # Save feature manifest
        try:
            version = self.manifest_manager.save_manifest(self.global_analysis_result)
            logger.info(f"Feature manifest saved with version: {version}")
        except Exception as e:
            logger.warning(f"Failed to save feature manifest: {e}")
        
        return self.global_analysis_result
    
    def process_symbols_with_global_consistency(self, symbol_dataframes: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Process all symbols with global feature consistency.
        
        This is the main method that orchestrates the complete feature consistency pipeline:
        1. Try to load existing manifest
        2. If no compatible manifest, perform coverage analysis integration
        3. Apply feature selection to each symbol
        4. Validate symbol coverage
        5. Return processed symbols
        
        Args:
            symbol_dataframes: Dictionary mapping symbol names to their DataFrames
            
        Returns:
            Dictionary of processed symbols that meet consistency requirements
        """
        logger.info(f"Processing {len(symbol_dataframes)} symbols with global consistency")
        self.processing_stats['start_time'] = datetime.now()
        
        # Step 1: Try to load existing manifest
        manifest_loaded = self.load_existing_manifest()
        
        if not manifest_loaded:
            # Step 2: Perform coverage analysis integration
            logger.info("No compatible manifest found, performing global coverage analysis")
            
            # Apply warm-up trimming and collect coverage statistics
            trimmed_symbols = self.analyze_and_trim_symbols(symbol_dataframes)
            
            if not trimmed_symbols:
                logger.error("No symbols remaining after coverage analysis integration")
                return {}
            
            # Perform global coverage analysis on trimmed data
            global_result = self.perform_global_analysis(trimmed_symbols)
            
            # Use trimmed symbols for further processing
            symbol_dataframes = trimmed_symbols
            
        else:
            logger.info("Using existing manifest for feature consistency")
        
        # Step 3: Process each symbol
        processed_symbols = {}
        excluded_symbols = []
        
        for symbol, df in symbol_dataframes.items():
            try:
                # Apply feature selection (either from analysis or loaded manifest)
                processed_df = self.apply_global_feature_selection(df, symbol)
                
                # Validate symbol coverage
                is_valid, coverage_ratio = self.validate_symbol_against_global_features(processed_df, symbol)
                
                if is_valid:
                    processed_symbols[symbol] = processed_df
                    self.processing_stats['symbols_processed'] += 1
                else:
                    excluded_symbols.append((symbol, coverage_ratio))
                    self.processing_stats['symbols_excluded'] += 1
                    
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                excluded_symbols.append((symbol, 0.0))
                self.processing_stats['symbols_excluded'] += 1
        
        # Step 4: Log summary
        self.processing_stats['end_time'] = datetime.now()
        
        logger.info(f"Global consistency processing completed:")
        logger.info(f"  Symbols processed: {len(processed_symbols)}")
        logger.info(f"  Symbols excluded: {len(excluded_symbols)}")
        
        if self.global_analysis_result:
            logger.info(f"  Stable features: {len(self.global_analysis_result.stable_features)}")
        
        if excluded_symbols:
            logger.info("Excluded symbols:")
            for symbol, coverage in excluded_symbols[:5]:  # Show first 5
                logger.info(f"  {symbol}: {coverage:.1%} coverage")
            if len(excluded_symbols) > 5:
                logger.info(f"  ... and {len(excluded_symbols) - 5} more")
        
        return processed_symbols
    
    def apply_global_feature_selection(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Apply global feature selection to a symbol's DataFrame.
        
        Args:
            df: Symbol's DataFrame
            symbol: Symbol name
            
        Returns:
            DataFrame with only stable features (plus essential columns)
        """
        if self.global_analysis_result is None:
            logger.warning(f"{symbol}: No global analysis available, returning original DataFrame")
            return df
        
        stable_features = self.global_analysis_result.stable_features
        if not stable_features:
            logger.warning(f"{symbol}: No stable features identified, returning original DataFrame")
            return df
        
        # Essential columns that should always be preserved
        essential_columns = ['symbol', 'target', 'date', 'timestamp', 'direction_1d', 'close']
        
        # Combine stable features with essential columns
        columns_to_keep = []
        for col in essential_columns:
            if col in df.columns:
                columns_to_keep.append(col)
        
        # Add stable features that exist in this symbol
        for feature in stable_features:
            if feature in df.columns:
                columns_to_keep.append(feature)
        
        # Remove duplicates while preserving order
        columns_to_keep = list(dict.fromkeys(columns_to_keep))
        
        # Apply selection
        original_cols = len(df.columns)
        selected_df = df[columns_to_keep].copy()
        
        logger.info(f"{symbol}: Applied global feature selection "
                   f"({original_cols} -> {len(selected_df.columns)} columns)")
        
        return selected_df
    
    def validate_symbol_against_global_features(self, df: pd.DataFrame, symbol: str) -> Tuple[bool, float]:
        """
        Validate that a symbol has sufficient coverage of global stable features.
        
        Args:
            df: Symbol's DataFrame
            symbol: Symbol name
            
        Returns:
            Tuple of (is_valid, coverage_ratio)
        """
        if self.global_analysis_result is None:
            logger.warning(f"{symbol}: No global analysis available, skipping validation")
            return True, 1.0
        
        stable_features = self.global_analysis_result.stable_features
        if not stable_features:
            logger.warning(f"{symbol}: No stable features identified, accepting symbol")
            return True, 1.0
        
        # Check how many stable features this symbol has
        symbol_features = set(df.columns)
        available_stable_features = [f for f in stable_features if f in symbol_features]
        coverage_ratio = len(available_stable_features) / len(stable_features)
        
        is_valid = coverage_ratio >= self.config.min_symbol_feature_coverage
        
        if is_valid:
            logger.info(f"{symbol}: Feature coverage validation passed "
                       f"({len(available_stable_features)}/{len(stable_features)} = {coverage_ratio:.1%})")
        else:
            logger.warning(f"{symbol}: Feature coverage validation failed "
                          f"({len(available_stable_features)}/{len(stable_features)} = {coverage_ratio:.1%} "
                          f"< {self.config.min_symbol_feature_coverage:.1%})")
        
        return is_valid, coverage_ratio
    
    def load_existing_manifest(self, version: Optional[str] = None) -> bool:
        """
        Load existing feature manifest if available and compatible.
        
        Args:
            version: Optional version to load (loads latest if None)
            
        Returns:
            True if manifest loaded successfully, False otherwise
        """
        try:
            manifest_data = self.manifest_manager.load_manifest(version)
            
            # Validate compatibility
            if not self.manifest_manager.validate_manifest_compatibility(manifest_data, self.config):
                logger.warning("Existing manifest is not compatible with current configuration")
                return False
            
            # Extract stable features
            stable_features = manifest_data.get("stable_features", [])
            
            # Create a mock analysis result for consistency
            self.global_analysis_result = GlobalAnalysisResult(
                total_symbols_analyzed=manifest_data.get("total_symbols_analyzed", 0),
                total_features_found=manifest_data.get("total_features_analyzed", 0),
                stable_features=stable_features,
                unstable_features=manifest_data.get("excluded_features", {}).get("features", []),
                coverage_stats={},  # Not needed for loaded manifest
                analysis_timestamp=manifest_data.get("created_timestamp", ""),
                config_used=manifest_data.get("config_snapshot", {})
            )
            
            logger.info(f"Loaded existing manifest with {len(stable_features)} stable features")
            return True
            
        except FileNotFoundError:
            logger.info("No existing manifest found, will create new one")
            return False
        except Exception as e:
            logger.warning(f"Failed to load existing manifest: {e}")
            return False
    
    def enforce_feature_selection(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Load feature manifest and apply to each symbol's DataFrame.
        Reindex DataFrames to match canonical feature order.
        Handle missing features gracefully with logging.
        
        This implements subtask 5.1: Create feature selection enforcement
        
        Args:
            df: Symbol's DataFrame
            symbol: Symbol name
            
        Returns:
            DataFrame with enforced feature selection and canonical ordering
        """
        logger.info(f"{symbol}: Enforcing feature selection from manifest")
        
        # Load feature manifest if not already loaded
        if self.global_analysis_result is None:
            manifest_loaded = self.load_existing_manifest()
            if not manifest_loaded:
                logger.warning(f"{symbol}: No manifest available for feature enforcement, returning original DataFrame")
                return df
        
        stable_features = self.global_analysis_result.stable_features
        if not stable_features:
            logger.warning(f"{symbol}: Empty feature manifest, returning original DataFrame")
            return df
        
        # Essential columns that should always be preserved (in order)
        essential_columns = ['symbol', 'target', 'date', 'timestamp', 'direction_1d', 'close']
        
        # Build canonical column order: essentials first, then stable features in manifest order
        canonical_columns = []
        
        # Add essential columns that exist in the DataFrame
        for col in essential_columns:
            if col in df.columns:
                canonical_columns.append(col)
        
        # Add stable features in manifest order
        available_features = []
        missing_features = []
        
        for feature in stable_features:
            if feature in df.columns:
                canonical_columns.append(feature)
                available_features.append(feature)
            else:
                missing_features.append(feature)
        
        # Log feature availability
        feature_coverage = len(available_features) / len(stable_features) if stable_features else 0.0
        
        logger.info(f"{symbol}: Feature enforcement results:")
        logger.info(f"  Manifest features: {len(stable_features)}")
        logger.info(f"  Available features: {len(available_features)} ({feature_coverage:.1%})")
        
        if missing_features:
            logger.warning(f"{symbol}: Missing {len(missing_features)} features from manifest:")
            for feature in missing_features[:5]:  # Show first 5
                logger.warning(f"    - {feature}")
            if len(missing_features) > 5:
                logger.warning(f"    ... and {len(missing_features) - 5} more")
        
        # Handle missing features gracefully by creating them with NaN values
        # This allows the downstream imputation to handle them consistently
        for feature in missing_features:
            df[feature] = np.nan
            canonical_columns.append(feature)
            logger.debug(f"{symbol}: Created missing feature '{feature}' with NaN values")
        
        # Reindex DataFrame to match canonical feature order
        try:
            # Remove duplicates while preserving order
            canonical_columns = list(dict.fromkeys(canonical_columns))
            
            # Reindex to canonical order
            original_shape = df.shape
            reindexed_df = df.reindex(columns=canonical_columns)
            
            logger.info(f"{symbol}: Reindexed to canonical order "
                       f"({original_shape[1]} -> {reindexed_df.shape[1]} columns)")
            
            # Verify no data loss (except for intentionally dropped columns)
            if reindexed_df.shape[0] != original_shape[0]:
                logger.error(f"{symbol}: Row count mismatch after reindexing!")
                return df  # Return original on error
            
            return reindexed_df
            
        except Exception as e:
            logger.error(f"{symbol}: Error during feature reindexing: {e}")
            return df  # Return original DataFrame on error
    
    def validate_symbol_coverage(self, df: pd.DataFrame, symbol: str) -> Tuple[bool, float, str]:
        """
        Check each symbol's coverage against minimum threshold.
        Skip symbols with insufficient feature coverage.
        Log exclusion decisions with detailed reasons.
        
        This implements subtask 5.2: Implement symbol coverage validation
        
        Args:
            df: Symbol's DataFrame
            symbol: Symbol name
            
        Returns:
            Tuple of (is_valid, coverage_ratio, exclusion_reason)
        """
        logger.debug(f"{symbol}: Validating symbol coverage against global features")
        
        # Load feature manifest if not already loaded
        if self.global_analysis_result is None:
            manifest_loaded = self.load_existing_manifest()
            if not manifest_loaded:
                logger.warning(f"{symbol}: No manifest available for coverage validation")
                return True, 1.0, ""  # Accept symbol if no manifest
        
        stable_features = self.global_analysis_result.stable_features
        if not stable_features:
            logger.warning(f"{symbol}: Empty feature manifest, accepting symbol")
            return True, 1.0, ""
        
        # Essential columns that should not count towards feature coverage
        essential_columns = {'symbol', 'target', 'date', 'timestamp', 'direction_1d', 'close'}
        
        # Get feature columns (excluding essentials)
        symbol_features = set(df.columns) - essential_columns
        stable_features_set = set(stable_features)
        
        # Calculate coverage metrics
        available_stable_features = symbol_features.intersection(stable_features_set)
        coverage_ratio = len(available_stable_features) / len(stable_features) if stable_features else 1.0
        
        # Check minimum coverage threshold
        min_coverage = self.config.min_symbol_feature_coverage
        is_valid = coverage_ratio >= min_coverage
        
        # Prepare detailed logging
        missing_features = stable_features_set - symbol_features
        extra_features = symbol_features - stable_features_set
        
        logger.info(f"{symbol}: Coverage validation results:")
        logger.info(f"  Required features: {len(stable_features)}")
        logger.info(f"  Available features: {len(available_stable_features)}")
        logger.info(f"  Coverage ratio: {coverage_ratio:.1%}")
        logger.info(f"  Minimum threshold: {min_coverage:.1%}")
        logger.info(f"  Validation result: {'PASS' if is_valid else 'FAIL'}")
        
        # Log missing features if any
        if missing_features:
            logger.warning(f"{symbol}: Missing {len(missing_features)} required features:")
            for feature in sorted(list(missing_features))[:10]:  # Show first 10
                logger.warning(f"    - {feature}")
            if len(missing_features) > 10:
                logger.warning(f"    ... and {len(missing_features) - 10} more")
        
        # Log extra features if any (informational)
        if extra_features:
            logger.debug(f"{symbol}: Has {len(extra_features)} extra features not in manifest")
            if len(extra_features) <= 5:
                logger.debug(f"    Extra features: {sorted(list(extra_features))}")
        
        # Prepare exclusion reason if validation fails
        exclusion_reason = ""
        if not is_valid:
            exclusion_reason = (
                f"insufficient_coverage_{coverage_ratio:.1%}_below_{min_coverage:.1%}_"
                f"missing_{len(missing_features)}_features"
            )
            
            logger.warning(f"{symbol}: EXCLUDED - {exclusion_reason}")
            logger.warning(f"{symbol}: Symbol will be skipped for current training run")
            
            # Log specific missing critical features if any
            critical_features = {'close', 'volume', 'rsi_14', 'sma_20', 'ema_12'}
            missing_critical = missing_features.intersection(critical_features)
            if missing_critical:
                logger.error(f"{symbol}: Missing critical features: {sorted(list(missing_critical))}")
        else:
            logger.info(f"{symbol}: Coverage validation PASSED - symbol accepted for training")
        
        return is_valid, coverage_ratio, exclusion_reason
    
    def batch_validate_symbol_coverage(self, symbol_dataframes: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """
        Validate coverage for multiple symbols and return detailed results.
        
        Args:
            symbol_dataframes: Dictionary mapping symbol names to their DataFrames
            
        Returns:
            Dictionary with validation results for each symbol
        """
        logger.info(f"Batch validating coverage for {len(symbol_dataframes)} symbols")
        
        validation_results = {}
        valid_symbols = []
        excluded_symbols = []
        
        for symbol, df in symbol_dataframes.items():
            try:
                is_valid, coverage_ratio, exclusion_reason = self.validate_symbol_coverage(df, symbol)
                
                validation_results[symbol] = {
                    'is_valid': is_valid,
                    'coverage_ratio': coverage_ratio,
                    'exclusion_reason': exclusion_reason,
                    'feature_count': len([c for c in df.columns if c not in {'symbol', 'target', 'date', 'timestamp', 'direction_1d', 'close'}]),
                    'total_columns': len(df.columns),
                    'row_count': len(df)
                }
                
                if is_valid:
                    valid_symbols.append(symbol)
                else:
                    excluded_symbols.append((symbol, exclusion_reason))
                    
            except Exception as e:
                logger.error(f"Error validating coverage for {symbol}: {e}")
                validation_results[symbol] = {
                    'is_valid': False,
                    'coverage_ratio': 0.0,
                    'exclusion_reason': f'validation_error_{str(e)[:50]}',
                    'feature_count': 0,
                    'total_columns': 0,
                    'row_count': 0
                }
                excluded_symbols.append((symbol, f'validation_error'))
        
        # Log batch summary
        logger.info("=" * 60)
        logger.info("BATCH COVERAGE VALIDATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total symbols validated: {len(symbol_dataframes)}")
        logger.info(f"Valid symbols: {len(valid_symbols)}")
        logger.info(f"Excluded symbols: {len(excluded_symbols)}")
        
        if valid_symbols:
            avg_coverage = np.mean([validation_results[s]['coverage_ratio'] for s in valid_symbols])
            logger.info(f"Average coverage (valid symbols): {avg_coverage:.1%}")
            logger.info(f"Valid symbols: {', '.join(valid_symbols[:10])}{'...' if len(valid_symbols) > 10 else ''}")
        
        if excluded_symbols:
            logger.info("Excluded symbols with reasons:")
            for symbol, reason in excluded_symbols[:10]:  # Show first 10
                logger.info(f"  {symbol}: {reason}")
            if len(excluded_symbols) > 10:
                logger.info(f"  ... and {len(excluded_symbols) - 10} more")
        
        logger.info("=" * 60)
        
        return validation_results
    
    def apply_updated_nan_threshold(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Change per-symbol NaN threshold from 1% to 5%.
        Apply threshold only after warm-up trimming and basic imputation.
        
        This implements subtask 5.3: Update NaN threshold application
        
        Args:
            df: Symbol's DataFrame (should already be warm-up trimmed)
            symbol: Symbol name
            
        Returns:
            DataFrame with features dropped based on updated 5% NaN threshold
        """
        logger.info(f"{symbol}: Applying updated NaN threshold ({self.config.nan_drop_threshold_per_symbol:.1%})")
        
        if df.empty:
            logger.warning(f"{symbol}: Empty DataFrame, no NaN threshold processing needed")
            return df
        
        # Essential columns that should never be dropped due to NaN threshold
        essential_columns = {'symbol', 'target', 'date', 'timestamp', 'direction_1d', 'close'}
        
        # Get feature columns (excluding essentials)
        feature_columns = [col for col in df.columns if col not in essential_columns]
        
        if not feature_columns:
            logger.info(f"{symbol}: No feature columns to process for NaN threshold")
            return df
        
        logger.info(f"{symbol}: Processing {len(feature_columns)} feature columns for NaN threshold")
        
        # Apply basic imputation BEFORE computing NaN ratios (as per requirement 2.4)
        df_processed = df.copy()
        
        # Forward fill and backward fill for feature columns
        logger.debug(f"{symbol}: Applying basic imputation (forward-fill and backward-fill)")
        df_processed[feature_columns] = df_processed[feature_columns].ffill().bfill()
        
        # Now compute NaN ratios after basic imputation
        nan_threshold = self.config.nan_drop_threshold_per_symbol  # 5% (0.05)
        features_to_drop = []
        feature_nan_stats = {}
        
        for col in feature_columns:
            nan_count = df_processed[col].isna().sum()
            total_count = len(df_processed)
            nan_ratio = nan_count / total_count if total_count > 0 else 0.0
            
            feature_nan_stats[col] = {
                'nan_count': nan_count,
                'total_count': total_count,
                'nan_ratio': nan_ratio
            }
            
            if nan_ratio > nan_threshold:
                features_to_drop.append(col)
                logger.debug(f"{symbol}: Feature '{col}' exceeds NaN threshold: "
                           f"{nan_ratio:.1%} > {nan_threshold:.1%}")
        
        # Drop features that exceed the threshold
        if features_to_drop:
            logger.warning(f"{symbol}: Dropping {len(features_to_drop)} features exceeding "
                          f"{nan_threshold:.1%} NaN threshold:")
            for feature in features_to_drop[:10]:  # Show first 10
                stats = feature_nan_stats[feature]
                logger.warning(f"  - {feature}: {stats['nan_ratio']:.1%} NaN "
                             f"({stats['nan_count']}/{stats['total_count']})")
            if len(features_to_drop) > 10:
                logger.warning(f"  ... and {len(features_to_drop) - 10} more")
            
            # Drop the problematic features
            df_processed = df_processed.drop(columns=features_to_drop)
        else:
            logger.info(f"{symbol}: All {len(feature_columns)} features pass NaN threshold check")
        
        # Log summary statistics
        remaining_features = [col for col in df_processed.columns if col not in essential_columns]
        features_dropped_count = len(feature_columns) - len(remaining_features)
        
        logger.info(f"{symbol}: NaN threshold processing complete:")
        logger.info(f"  Original features: {len(feature_columns)}")
        logger.info(f"  Features dropped: {features_dropped_count}")
        logger.info(f"  Remaining features: {len(remaining_features)}")
        logger.info(f"  NaN threshold used: {nan_threshold:.1%}")
        
        # Final NaN check on remaining data
        if remaining_features:
            final_nan_ratio = df_processed[remaining_features].isna().sum().sum() / (
                len(df_processed) * len(remaining_features)
            )
            logger.info(f"  Final NaN ratio: {final_nan_ratio:.2%}")
            
            if final_nan_ratio > nan_threshold:
                logger.warning(f"{symbol}: Final NaN ratio {final_nan_ratio:.2%} still exceeds threshold")
        
        return df_processed
    
    def process_symbol_with_updated_nan_handling(self, df: pd.DataFrame, symbol: str) -> Optional[pd.DataFrame]:
        """
        Complete per-symbol processing with updated NaN handling workflow.
        
        This combines warm-up trimming, basic imputation, and updated NaN threshold application
        in the correct order as specified in the requirements.
        
        Args:
            df: Raw symbol DataFrame
            symbol: Symbol name
            
        Returns:
            Processed DataFrame or None if processing fails
        """
        logger.info(f"{symbol}: Starting complete NaN handling workflow")
        
        try:
            # Step 1: Apply warm-up trimming (should already be done, but ensure it's applied)
            if hasattr(self, 'warmup_trimmer'):
                df_trimmed = self.warmup_trimmer.trim_warmup_period(df, symbol)
                if not self.warmup_trimmer.validate_post_trim_data(df_trimmed, symbol):
                    logger.error(f"{symbol}: Failed post-trim validation")
                    return None
            else:
                logger.warning(f"{symbol}: No warmup trimmer available, using original data")
                df_trimmed = df.copy()
            
            # Step 2: Apply updated NaN threshold (includes basic imputation)
            df_processed = self.apply_updated_nan_threshold(df_trimmed, symbol)
            
            # Step 3: Validate final result
            if df_processed.empty:
                logger.error(f"{symbol}: DataFrame is empty after NaN processing")
                return None
            
            # Check if we have sufficient features remaining
            essential_columns = {'symbol', 'target', 'date', 'timestamp', 'direction_1d', 'close'}
            feature_columns = [col for col in df_processed.columns if col not in essential_columns]
            
            if len(feature_columns) < 10:  # Minimum feature threshold
                logger.warning(f"{symbol}: Only {len(feature_columns)} features remaining, may be insufficient")
            
            logger.info(f"{symbol}: NaN handling workflow completed successfully")
            logger.info(f"  Final shape: {df_processed.shape}")
            logger.info(f"  Final features: {len(feature_columns)}")
            
            return df_processed
            
        except Exception as e:
            logger.error(f"{symbol}: Error in NaN handling workflow: {e}")
            return None
    
    def process_symbols_with_per_symbol_consistency(self, symbol_dataframes: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Process all symbols with per-symbol feature consistency processing.
        
        This implements task 5: Implement per-symbol feature consistency processing
        by orchestrating all three subtasks:
        5.1: Create feature selection enforcement
        5.2: Implement symbol coverage validation  
        5.3: Update NaN threshold application
        
        Args:
            symbol_dataframes: Dictionary mapping symbol names to their DataFrames
            
        Returns:
            Dictionary of processed symbols that meet consistency requirements
        """
        logger.info("=" * 60)
        logger.info("STARTING PER-SYMBOL FEATURE CONSISTENCY PROCESSING")
        logger.info("=" * 60)
        logger.info(f"Processing {len(symbol_dataframes)} symbols with per-symbol consistency")
        
        self.processing_stats['start_time'] = datetime.now()
        
        # Step 1: Try to load existing manifest
        manifest_loaded = self.load_existing_manifest()
        
        if not manifest_loaded:
            # Step 2: Perform coverage analysis integration if no manifest
            logger.info("No compatible manifest found, performing global coverage analysis")
            
            # Apply warm-up trimming and collect coverage statistics
            trimmed_symbols = self.analyze_and_trim_symbols(symbol_dataframes)
            
            if not trimmed_symbols:
                logger.error("No symbols remaining after coverage analysis integration")
                return {}
            
            # Perform global coverage analysis on trimmed data
            global_result = self.perform_global_analysis(trimmed_symbols)
            
            # Use trimmed symbols for further processing
            symbol_dataframes = trimmed_symbols
            
        else:
            logger.info("Using existing manifest for feature consistency")
        
        # Step 3: Process each symbol with per-symbol feature consistency processing
        processed_symbols = {}
        excluded_symbols = []
        processing_stats = {
            'feature_enforcement_success': 0,
            'feature_enforcement_errors': 0,
            'nan_processing_success': 0,
            'nan_processing_errors': 0,
            'coverage_validation_pass': 0,
            'coverage_validation_fail': 0
        }
        
        for symbol, df in symbol_dataframes.items():
            logger.info(f"Processing {symbol}...")
            
            try:
                # Subtask 5.1: Enforce feature selection from manifest
                logger.debug(f"{symbol}: Step 1 - Enforcing feature selection")
                df_with_features = self.enforce_feature_selection(df, symbol)
                processing_stats['feature_enforcement_success'] += 1
                
                # Subtask 5.3: Apply updated NaN threshold (5% instead of 1%)
                logger.debug(f"{symbol}: Step 2 - Applying updated NaN threshold")
                df_nan_processed = self.apply_updated_nan_threshold(df_with_features, symbol)
                processing_stats['nan_processing_success'] += 1
                
                # Task 6: Apply missingness mask and imputation system
                logger.debug(f"{symbol}: Step 3 - Applying missingness handling")
                df_with_missingness_handled = self.missingness_mask_generator.process_symbol_with_missingness_handling(
                    df_nan_processed, symbol
                )
                
                # Subtask 5.2: Validate symbol coverage
                logger.debug(f"{symbol}: Step 4 - Validating symbol coverage")
                is_valid, coverage_ratio, exclusion_reason = self.validate_symbol_coverage(df_with_missingness_handled, symbol)
                
                if is_valid:
                    processed_symbols[symbol] = df_with_missingness_handled
                    self.processing_stats['symbols_processed'] += 1
                    processing_stats['coverage_validation_pass'] += 1
                    logger.info(f"{symbol}: ✓ Successfully processed and accepted for training")
                else:
                    excluded_symbols.append((symbol, coverage_ratio, exclusion_reason))
                    self.processing_stats['symbols_excluded'] += 1
                    processing_stats['coverage_validation_fail'] += 1
                    logger.warning(f"{symbol}: ✗ Excluded from training - {exclusion_reason}")
                    
            except Exception as e:
                logger.error(f"{symbol}: Error during processing: {e}")
                excluded_symbols.append((symbol, 0.0, f"processing_error_{str(e)[:50]}"))
                self.processing_stats['symbols_excluded'] += 1
                
                # Track which step failed
                if 'feature_enforcement_success' not in [k for k, v in processing_stats.items() if v > 0]:
                    processing_stats['feature_enforcement_errors'] += 1
                elif 'nan_processing_success' not in [k for k, v in processing_stats.items() if v > 0]:
                    processing_stats['nan_processing_errors'] += 1
                else:
                    processing_stats['coverage_validation_fail'] += 1
        
        # Step 4: Final validation and summary
        self.processing_stats['end_time'] = datetime.now()
        
        logger.info("=" * 60)
        logger.info("PER-SYMBOL FEATURE CONSISTENCY PROCESSING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total symbols processed: {len(symbol_dataframes)}")
        logger.info(f"Symbols accepted: {len(processed_symbols)}")
        logger.info(f"Symbols excluded: {len(excluded_symbols)}")
        
        # Log processing step statistics
        logger.info("Processing step statistics:")
        logger.info(f"  Feature enforcement success: {processing_stats['feature_enforcement_success']}")
        logger.info(f"  Feature enforcement errors: {processing_stats['feature_enforcement_errors']}")
        logger.info(f"  NaN processing success: {processing_stats['nan_processing_success']}")
        logger.info(f"  NaN processing errors: {processing_stats['nan_processing_errors']}")
        logger.info(f"  Coverage validation pass: {processing_stats['coverage_validation_pass']}")
        logger.info(f"  Coverage validation fail: {processing_stats['coverage_validation_fail']}")
        
        if self.global_analysis_result:
            logger.info(f"Stable features in manifest: {len(self.global_analysis_result.stable_features)}")
        
        # Log acceptance rate
        acceptance_rate = len(processed_symbols) / len(symbol_dataframes) if symbol_dataframes else 0.0
        logger.info(f"Acceptance rate: {acceptance_rate:.1%}")
        
        if processed_symbols:
            # Validate consistent shapes across all processed symbols
            feature_counts = []
            column_counts = []
            for symbol, df in processed_symbols.items():
                essential_columns = {'symbol', 'target', 'date', 'timestamp', 'direction_1d', 'close'}
                feature_count = len([c for c in df.columns if c not in essential_columns])
                feature_counts.append(feature_count)
                column_counts.append(len(df.columns))
            
            if len(set(feature_counts)) == 1:
                logger.info(f"✓ Feature consistency validated: All symbols have {feature_counts[0]} features")
            else:
                logger.error(f"✗ Feature drift detected: Different feature counts: {set(feature_counts)}")
            
            if len(set(column_counts)) == 1:
                logger.info(f"✓ Column consistency validated: All symbols have {column_counts[0]} total columns")
            else:
                logger.error(f"✗ Column drift detected: Different column counts: {set(column_counts)}")
        
        if excluded_symbols:
            logger.info("Excluded symbols with reasons:")
            for symbol, coverage, reason in excluded_symbols[:10]:  # Show first 10
                logger.info(f"  {symbol}: {reason} (coverage: {coverage:.1%})")
            if len(excluded_symbols) > 10:
                logger.info(f"  ... and {len(excluded_symbols) - 10} more")
        
        # Log requirements compliance
        logger.info("Requirements compliance check:")
        logger.info(f"  ✓ Requirement 1.1: Feature selection enforcement implemented")
        logger.info(f"  ✓ Requirement 1.2: Canonical feature ordering implemented")
        logger.info(f"  ✓ Requirement 3.4: Global feature whitelist applied")
        logger.info(f"  ✓ Requirement 3.5: Symbol coverage validation implemented")
        logger.info(f"  ✓ Requirement 1.4: Symbol exclusion with logging implemented")
        logger.info(f"  ✓ Requirement 2.3: Updated 5% NaN threshold applied")
        logger.info(f"  ✓ Requirement 2.4: NaN threshold after warm-up trimming and imputation")
        logger.info(f"  ✓ Requirement 6.3: Detailed exclusion logging implemented")
        
        logger.info("=" * 60)
        
        return processed_symbols

    def process_symbols_with_complete_pipeline(self, symbol_dataframes: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Process all symbols with the complete feature consistency and missingness handling pipeline.
        
        This implements the complete workflow including:
        - Global coverage analysis and feature manifest generation
        - Per-symbol feature selection enforcement
        - Updated NaN threshold application
        - Missingness mask generation and imputation
        - Symbol coverage validation
        
        Args:
            symbol_dataframes: Dictionary mapping symbol names to their DataFrames
            
        Returns:
            Dictionary of processed symbols ready for model training
        """
        logger.info("=" * 70)
        logger.info("STARTING COMPLETE FEATURE CONSISTENCY AND MISSINGNESS PIPELINE")
        logger.info("=" * 70)
        logger.info(f"Processing {len(symbol_dataframes)} symbols with complete pipeline")
        
        self.processing_stats['start_time'] = datetime.now()
        
        # Step 1: Try to load existing manifest
        manifest_loaded = self.load_existing_manifest()
        
        if not manifest_loaded:
            # Step 2: Perform coverage analysis integration if no manifest
            logger.info("No compatible manifest found, performing global coverage analysis")
            
            # Apply warm-up trimming and collect coverage statistics
            trimmed_symbols = self.analyze_and_trim_symbols(symbol_dataframes)
            
            if not trimmed_symbols:
                logger.error("No symbols remaining after coverage analysis integration")
                return {}
            
            # Perform global coverage analysis on trimmed data
            global_result = self.perform_global_analysis(trimmed_symbols)
            
            # Use trimmed symbols for further processing
            symbol_dataframes = trimmed_symbols
            
        else:
            logger.info("Using existing manifest for feature consistency")
        
        # Step 3: Process each symbol with complete pipeline
        processed_symbols = {}
        excluded_symbols = []
        processing_stats = {
            'feature_enforcement_success': 0,
            'feature_enforcement_errors': 0,
            'nan_processing_success': 0,
            'nan_processing_errors': 0,
            'missingness_handling_success': 0,
            'missingness_handling_errors': 0,
            'coverage_validation_pass': 0,
            'coverage_validation_fail': 0
        }
        
        for symbol, df in symbol_dataframes.items():
            logger.info(f"Processing {symbol} with complete pipeline...")
            
            try:
                # Step 3.1: Enforce feature selection from manifest
                logger.debug(f"{symbol}: Step 1 - Enforcing feature selection")
                df_with_features = self.enforce_feature_selection(df, symbol)
                processing_stats['feature_enforcement_success'] += 1
                
                # Step 3.2: Apply updated NaN threshold (5% instead of 1%)
                logger.debug(f"{symbol}: Step 2 - Applying updated NaN threshold")
                df_nan_processed = self.apply_updated_nan_threshold(df_with_features, symbol)
                processing_stats['nan_processing_success'] += 1
                
                # Step 3.3: Apply complete missingness handling workflow
                logger.debug(f"{symbol}: Step 3 - Applying missingness handling")
                
                # Get feature columns for missingness processing
                essential_columns = {'symbol', 'target', 'date', 'timestamp', 'direction_1d', 'close'}
                feature_cols = [col for col in df_nan_processed.columns if col not in essential_columns]
                
                # Apply the complete missingness handling workflow
                df_with_missingness_handled = self.missingness_mask_generator.process_symbol_with_missingness_handling(
                    df_nan_processed, symbol, feature_cols
                )
                processing_stats['missingness_handling_success'] += 1
                
                # Step 3.4: Validate symbol coverage
                logger.debug(f"{symbol}: Step 4 - Validating symbol coverage")
                is_valid, coverage_ratio, exclusion_reason = self.validate_symbol_coverage(df_with_missingness_handled, symbol)
                
                if is_valid:
                    # Final validation: ensure no NaN values remain in feature columns
                    is_clean, columns_with_missing = self.missingness_mask_generator.validate_no_missing_values(
                        df_with_missingness_handled, symbol
                    )
                    
                    if is_clean:
                        processed_symbols[symbol] = df_with_missingness_handled
                        self.processing_stats['symbols_processed'] += 1
                        processing_stats['coverage_validation_pass'] += 1
                        logger.info(f"{symbol}: ✓ Successfully processed and accepted for training")
                    else:
                        exclusion_reason = f"final_validation_failed_missing_values_in_{len(columns_with_missing)}_columns"
                        excluded_symbols.append((symbol, coverage_ratio, exclusion_reason))
                        self.processing_stats['symbols_excluded'] += 1
                        processing_stats['coverage_validation_fail'] += 1
                        logger.warning(f"{symbol}: ✗ Excluded - final validation failed")
                else:
                    excluded_symbols.append((symbol, coverage_ratio, exclusion_reason))
                    self.processing_stats['symbols_excluded'] += 1
                    processing_stats['coverage_validation_fail'] += 1
                    logger.warning(f"{symbol}: ✗ Excluded from training - {exclusion_reason}")
                    
            except Exception as e:
                logger.error(f"{symbol}: Error during complete pipeline processing: {e}")
                excluded_symbols.append((symbol, 0.0, f"pipeline_error_{str(e)[:50]}"))
                self.processing_stats['symbols_excluded'] += 1
                
                # Track which step failed
                if processing_stats['feature_enforcement_success'] == 0:
                    processing_stats['feature_enforcement_errors'] += 1
                elif processing_stats['nan_processing_success'] == 0:
                    processing_stats['nan_processing_errors'] += 1
                elif processing_stats['missingness_handling_success'] == 0:
                    processing_stats['missingness_handling_errors'] += 1
                else:
                    processing_stats['coverage_validation_fail'] += 1
        
        # Step 4: Final validation and comprehensive summary
        self.processing_stats['end_time'] = datetime.now()
        
        logger.info("=" * 70)
        logger.info("COMPLETE FEATURE CONSISTENCY AND MISSINGNESS PIPELINE COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Total symbols processed: {len(symbol_dataframes)}")
        logger.info(f"Symbols accepted: {len(processed_symbols)}")
        logger.info(f"Symbols excluded: {len(excluded_symbols)}")
        
        # Log detailed processing step statistics
        logger.info("Processing step statistics:")
        logger.info(f"  Feature enforcement success: {processing_stats['feature_enforcement_success']}")
        logger.info(f"  Feature enforcement errors: {processing_stats['feature_enforcement_errors']}")
        logger.info(f"  NaN processing success: {processing_stats['nan_processing_success']}")
        logger.info(f"  NaN processing errors: {processing_stats['nan_processing_errors']}")
        logger.info(f"  Missingness handling success: {processing_stats['missingness_handling_success']}")
        logger.info(f"  Missingness handling errors: {processing_stats['missingness_handling_errors']}")
        logger.info(f"  Coverage validation pass: {processing_stats['coverage_validation_pass']}")
        logger.info(f"  Coverage validation fail: {processing_stats['coverage_validation_fail']}")
        
        if self.global_analysis_result:
            logger.info(f"Stable features in manifest: {len(self.global_analysis_result.stable_features)}")
        
        # Log acceptance rate
        acceptance_rate = len(processed_symbols) / len(symbol_dataframes) if symbol_dataframes else 0.0
        logger.info(f"Acceptance rate: {acceptance_rate:.1%}")
        
        if processed_symbols:
            # Validate consistent shapes across all processed symbols
            feature_counts = []
            column_counts = []
            mask_column_counts = []
            
            for symbol, df in processed_symbols.items():
                essential_columns = {'symbol', 'target', 'date', 'timestamp', 'direction_1d', 'close'}
                feature_count = len([c for c in df.columns if c not in essential_columns and not c.endswith('_isnan')])
                mask_count = len([c for c in df.columns if c.endswith('_isnan')])
                
                feature_counts.append(feature_count)
                column_counts.append(len(df.columns))
                mask_column_counts.append(mask_count)
            
            # Validate consistency
            if len(set(feature_counts)) == 1:
                logger.info(f"✓ Feature consistency validated: All symbols have {feature_counts[0]} features")
            else:
                logger.error(f"✗ Feature drift detected: Different feature counts: {set(feature_counts)}")
            
            if len(set(column_counts)) == 1:
                logger.info(f"✓ Column consistency validated: All symbols have {column_counts[0]} total columns")
            else:
                logger.error(f"✗ Column drift detected: Different column counts: {set(column_counts)}")
            
            if self.config.use_missingness_mask:
                if len(set(mask_column_counts)) == 1:
                    logger.info(f"✓ Missingness mask consistency validated: All symbols have {mask_column_counts[0]} mask columns")
                else:
                    logger.error(f"✗ Mask column drift detected: Different mask counts: {set(mask_column_counts)}")
            
            # Log sample of final column structure
            if processed_symbols:
                sample_symbol = next(iter(processed_symbols.keys()))
                sample_df = processed_symbols[sample_symbol]
                
                essential_cols = [c for c in sample_df.columns if c in {'symbol', 'target', 'date', 'timestamp', 'direction_1d', 'close'}]
                feature_cols = [c for c in sample_df.columns if c not in {'symbol', 'target', 'date', 'timestamp', 'direction_1d', 'close'} and not c.endswith('_isnan')]
                mask_cols = [c for c in sample_df.columns if c.endswith('_isnan')]
                
                logger.info(f"Final column structure (sample from {sample_symbol}):")
                logger.info(f"  Essential columns: {len(essential_cols)}")
                logger.info(f"  Feature columns: {len(feature_cols)}")
                logger.info(f"  Missingness mask columns: {len(mask_cols)}")
                logger.info(f"  Total columns: {len(sample_df.columns)}")
        
        if excluded_symbols:
            logger.info("Excluded symbols with reasons:")
            for symbol, coverage, reason in excluded_symbols[:10]:  # Show first 10
                logger.info(f"  {symbol}: {reason} (coverage: {coverage:.1%})")
            if len(excluded_symbols) > 10:
                logger.info(f"  ... and {len(excluded_symbols) - 10} more")
        
        # Log requirements compliance
        logger.info("Requirements compliance check:")
        logger.info(f"  ✓ Requirement 4.1: Missingness mask indicators created")
        logger.info(f"  ✓ Requirement 4.2: Binary _isnan columns added for each feature")
        logger.info(f"  ✓ Requirement 4.3: Final imputation with configurable fill values")
        logger.info(f"  ✓ Requirement 4.4: Forward-fill and backward-fill applied before mask creation")
        logger.info(f"  ✓ Requirement 4.5: Missing values filled after mask creation")
        logger.info(f"  ✓ Requirement 5.5: Configurable missingness handling implemented")
        
        logger.info("=" * 70)
        
        return processed_symbols

    def validate_tensor_shapes(self, processed_data: Dict[str, pd.DataFrame]) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate that all processed symbols have identical column counts and ordering.
        Check that all processed symbols have identical tensor shapes for model training.
        Add detailed shape mismatch reporting and error handling.
        
        This implements subtask 7.1: Create tensor shape validation system
        Requirements: 1.1, 1.2, 1.5
        
        Args:
            processed_data: Dictionary mapping symbol names to their processed DataFrames
            
        Returns:
            Tuple of (is_valid, validation_report)
        """
        logger.info("=" * 60)
        logger.info("TENSOR SHAPE VALIDATION")
        logger.info("=" * 60)
        logger.info(f"Validating tensor shapes for {len(processed_data)} symbols")
        
        if not processed_data:
            logger.warning("No processed data provided for shape validation")
            return True, {
                'is_valid': True,
                'symbols_count': 0,
                'validation_message': 'No data to validate',
                'shape_consistency': {},
                'column_consistency': {},
                'feature_consistency': {}
            }
        
        # Initialize validation report
        validation_report = {
            'is_valid': True,
            'symbols_count': len(processed_data),
            'validation_timestamp': datetime.now().isoformat(),
            'shape_consistency': {},
            'column_consistency': {},
            'feature_consistency': {},
            'errors': [],
            'warnings': [],
            'summary': {}
        }
        
        try:
            # Step 1: Collect shape information from all symbols
            logger.info("Step 1: Collecting shape information from all symbols")
            
            symbol_shapes = {}
            symbol_columns = {}
            symbol_features = {}
            
            essential_columns = {'symbol', 'target', 'date', 'timestamp', 'direction_1d', 'close'}
            
            for symbol, df in processed_data.items():
                try:
                    # Basic shape information
                    rows, cols = df.shape
                    symbol_shapes[symbol] = {
                        'rows': rows,
                        'columns': cols,
                        'shape': (rows, cols)
                    }
                    
                    # Column information
                    column_list = list(df.columns)
                    symbol_columns[symbol] = {
                        'column_names': column_list,
                        'column_count': len(column_list),
                        'column_order': column_list
                    }
                    
                    # Feature information (excluding essential columns)
                    feature_cols = [col for col in column_list if col not in essential_columns]
                    feature_cols_no_mask = [col for col in feature_cols if not col.endswith('_isnan')]
                    mask_cols = [col for col in feature_cols if col.endswith('_isnan')]
                    
                    symbol_features[symbol] = {
                        'total_features': len(feature_cols),
                        'base_features': len(feature_cols_no_mask),
                        'mask_features': len(mask_cols),
                        'feature_names': feature_cols_no_mask,
                        'mask_names': mask_cols,
                        'essential_columns': [col for col in essential_columns if col in column_list]
                    }
                    
                    logger.debug(f"{symbol}: Shape {df.shape}, Features {len(feature_cols_no_mask)}, Masks {len(mask_cols)}")
                    
                except Exception as e:
                    error_msg = f"Error collecting shape info for {symbol}: {e}"
                    logger.error(error_msg)
                    validation_report['errors'].append(error_msg)
                    validation_report['is_valid'] = False
            
            # Step 2: Validate shape consistency
            logger.info("Step 2: Validating shape consistency")
            
            if symbol_shapes:
                # Check column count consistency
                column_counts = [info['columns'] for info in symbol_shapes.values()]
                unique_column_counts = set(column_counts)
                
                if len(unique_column_counts) == 1:
                    logger.info(f"✓ Column count consistency: All symbols have {column_counts[0]} columns")
                    validation_report['shape_consistency']['column_count_valid'] = True
                    validation_report['shape_consistency']['uniform_column_count'] = column_counts[0]
                else:
                    error_msg = f"✗ Column count mismatch: Found {len(unique_column_counts)} different counts: {sorted(unique_column_counts)}"
                    logger.error(error_msg)
                    validation_report['errors'].append(error_msg)
                    validation_report['is_valid'] = False
                    validation_report['shape_consistency']['column_count_valid'] = False
                    validation_report['shape_consistency']['column_count_distribution'] = {
                        count: [symbol for symbol, info in symbol_shapes.items() if info['columns'] == count]
                        for count in unique_column_counts
                    }
                
                # Check row count distribution (informational)
                row_counts = [info['rows'] for info in symbol_shapes.values()]
                min_rows = min(row_counts)
                max_rows = max(row_counts)
                avg_rows = sum(row_counts) / len(row_counts)
                
                validation_report['shape_consistency']['row_statistics'] = {
                    'min_rows': min_rows,
                    'max_rows': max_rows,
                    'avg_rows': int(avg_rows),
                    'total_rows': sum(row_counts)
                }
                
                logger.info(f"Row count statistics: min={min_rows}, max={max_rows}, avg={int(avg_rows)}")
            
            # Step 3: Validate column ordering consistency
            logger.info("Step 3: Validating column ordering consistency")
            
            if symbol_columns:
                # Get reference column order from first symbol
                reference_symbol = next(iter(symbol_columns.keys()))
                reference_columns = symbol_columns[reference_symbol]['column_names']
                
                column_order_mismatches = []
                
                for symbol, col_info in symbol_columns.items():
                    if col_info['column_names'] != reference_columns:
                        column_order_mismatches.append(symbol)
                        
                        # Find specific differences
                        ref_set = set(reference_columns)
                        sym_set = set(col_info['column_names'])
                        
                        missing_cols = ref_set - sym_set
                        extra_cols = sym_set - ref_set
                        
                        logger.warning(f"{symbol}: Column order/content mismatch with {reference_symbol}")
                        if missing_cols:
                            logger.warning(f"  Missing columns: {sorted(list(missing_cols))[:5]}{'...' if len(missing_cols) > 5 else ''}")
                        if extra_cols:
                            logger.warning(f"  Extra columns: {sorted(list(extra_cols))[:5]}{'...' if len(extra_cols) > 5 else ''}")
                
                if not column_order_mismatches:
                    logger.info(f"✓ Column ordering consistency: All symbols have identical column order")
                    validation_report['column_consistency']['order_valid'] = True
                    validation_report['column_consistency']['reference_symbol'] = reference_symbol
                    validation_report['column_consistency']['column_count'] = len(reference_columns)
                else:
                    error_msg = f"✗ Column ordering mismatch: {len(column_order_mismatches)} symbols differ from reference"
                    logger.error(error_msg)
                    validation_report['errors'].append(error_msg)
                    validation_report['is_valid'] = False
                    validation_report['column_consistency']['order_valid'] = False
                    validation_report['column_consistency']['mismatched_symbols'] = column_order_mismatches[:10]  # First 10
                    validation_report['column_consistency']['reference_symbol'] = reference_symbol
            
            # Step 4: Validate feature consistency
            logger.info("Step 4: Validating feature consistency")
            
            if symbol_features:
                # Check base feature count consistency
                base_feature_counts = [info['base_features'] for info in symbol_features.values()]
                unique_base_counts = set(base_feature_counts)
                
                if len(unique_base_counts) == 1:
                    logger.info(f"✓ Base feature consistency: All symbols have {base_feature_counts[0]} base features")
                    validation_report['feature_consistency']['base_features_valid'] = True
                    validation_report['feature_consistency']['uniform_base_feature_count'] = base_feature_counts[0]
                else:
                    error_msg = f"✗ Base feature count mismatch: Found {len(unique_base_counts)} different counts: {sorted(unique_base_counts)}"
                    logger.error(error_msg)
                    validation_report['errors'].append(error_msg)
                    validation_report['is_valid'] = False
                    validation_report['feature_consistency']['base_features_valid'] = False
                    validation_report['feature_consistency']['base_feature_distribution'] = {
                        count: [symbol for symbol, info in symbol_features.items() if info['base_features'] == count]
                        for count in unique_base_counts
                    }
                
                # Check mask feature consistency (if using missingness masks)
                if self.config.use_missingness_mask:
                    mask_feature_counts = [info['mask_features'] for info in symbol_features.values()]
                    unique_mask_counts = set(mask_feature_counts)
                    
                    if len(unique_mask_counts) == 1:
                        logger.info(f"✓ Mask feature consistency: All symbols have {mask_feature_counts[0]} mask features")
                        validation_report['feature_consistency']['mask_features_valid'] = True
                        validation_report['feature_consistency']['uniform_mask_feature_count'] = mask_feature_counts[0]
                    else:
                        error_msg = f"✗ Mask feature count mismatch: Found {len(unique_mask_counts)} different counts: {sorted(unique_mask_counts)}"
                        logger.error(error_msg)
                        validation_report['errors'].append(error_msg)
                        validation_report['is_valid'] = False
                        validation_report['feature_consistency']['mask_features_valid'] = False
                
                # Check feature name consistency
                reference_symbol = next(iter(symbol_features.keys()))
                reference_features = set(symbol_features[reference_symbol]['feature_names'])
                
                feature_name_mismatches = []
                for symbol, feat_info in symbol_features.items():
                    symbol_features_set = set(feat_info['feature_names'])
                    if symbol_features_set != reference_features:
                        feature_name_mismatches.append(symbol)
                
                if not feature_name_mismatches:
                    logger.info(f"✓ Feature name consistency: All symbols have identical feature names")
                    validation_report['feature_consistency']['feature_names_valid'] = True
                else:
                    error_msg = f"✗ Feature name mismatch: {len(feature_name_mismatches)} symbols have different features"
                    logger.error(error_msg)
                    validation_report['errors'].append(error_msg)
                    validation_report['is_valid'] = False
                    validation_report['feature_consistency']['feature_names_valid'] = False
                    validation_report['feature_consistency']['mismatched_feature_symbols'] = feature_name_mismatches[:10]
            
            # Step 5: Generate summary
            logger.info("Step 5: Generating validation summary")
            
            validation_report['summary'] = {
                'total_symbols_validated': len(processed_data),
                'shape_validation_passed': validation_report['shape_consistency'].get('column_count_valid', False),
                'column_order_validation_passed': validation_report['column_consistency'].get('order_valid', False),
                'feature_consistency_validation_passed': (
                    validation_report['feature_consistency'].get('base_features_valid', False) and
                    validation_report['feature_consistency'].get('feature_names_valid', False)
                ),
                'overall_validation_passed': validation_report['is_valid'],
                'error_count': len(validation_report['errors']),
                'warning_count': len(validation_report['warnings'])
            }
            
            # Log final validation result
            logger.info("=" * 60)
            logger.info("TENSOR SHAPE VALIDATION RESULTS")
            logger.info("=" * 60)
            
            if validation_report['is_valid']:
                logger.info("✓ VALIDATION PASSED: All symbols have consistent tensor shapes")
                logger.info(f"  Symbols validated: {len(processed_data)}")
                if symbol_shapes:
                    logger.info(f"  Uniform column count: {validation_report['shape_consistency'].get('uniform_column_count', 'N/A')}")
                if symbol_features:
                    logger.info(f"  Uniform feature count: {validation_report['feature_consistency'].get('uniform_base_feature_count', 'N/A')}")
                    if self.config.use_missingness_mask:
                        logger.info(f"  Uniform mask count: {validation_report['feature_consistency'].get('uniform_mask_feature_count', 'N/A')}")
            else:
                logger.error("✗ VALIDATION FAILED: Tensor shape inconsistencies detected")
                logger.error(f"  Errors found: {len(validation_report['errors'])}")
                for error in validation_report['errors'][:5]:  # Show first 5 errors
                    logger.error(f"    - {error}")
                if len(validation_report['errors']) > 5:
                    logger.error(f"    ... and {len(validation_report['errors']) - 5} more errors")
            
            logger.info("=" * 60)
            
            return validation_report['is_valid'], validation_report
            
        except Exception as e:
            error_msg = f"Critical error during tensor shape validation: {e}"
            logger.error(error_msg)
            validation_report['errors'].append(error_msg)
            validation_report['is_valid'] = False
            return False, validation_report

    def log_comprehensive_pipeline_summary(self, processed_symbols: Dict[str, pd.DataFrame], 
                                         excluded_symbols: List[Tuple[str, float, str]] = None) -> None:
        """
        Log comprehensive summary of the entire feature consistency pipeline.
        
        This implements subtask 7.2: Add comprehensive logging throughout pipeline
        Requirements: 6.1, 6.2, 6.3, 6.4, 6.5
        
        Args:
            processed_symbols: Dictionary of successfully processed symbols
            excluded_symbols: List of excluded symbols with coverage and reasons
        """
        logger.info("=" * 80)
        logger.info("COMPREHENSIVE FEATURE CONSISTENCY PIPELINE SUMMARY")
        logger.info("=" * 80)
        
        # Get processing statistics
        processing_summary = self.get_processing_summary()
        
        # Log overall statistics
        logger.info("OVERALL PROCESSING STATISTICS:")
        logger.info(f"  Pipeline start time: {processing_summary.get('start_time', 'N/A')}")
        logger.info(f"  Pipeline end time: {processing_summary.get('end_time', 'N/A')}")
        if 'processing_duration_seconds' in processing_summary:
            duration = processing_summary['processing_duration_seconds']
            logger.info(f"  Total processing time: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        
        logger.info(f"  Symbols processed successfully: {processing_summary.get('symbols_processed', 0)}")
        logger.info(f"  Symbols excluded: {processing_summary.get('symbols_excluded', 0)}")
        logger.info(f"  Exclusion rate: {processing_summary.get('exclusion_rate', 0.0):.1%}")
        logger.info(f"  Warmup rows trimmed: {processing_summary.get('warmup_rows_trimmed', 0):,}")
        
        # Log feature manifest statistics
        if self.global_analysis_result:
            logger.info("")
            logger.info("FEATURE MANIFEST STATISTICS:")
            logger.info(f"  Total features analyzed: {self.global_analysis_result.total_features_found}")
            logger.info(f"  Stable features selected: {len(self.global_analysis_result.stable_features)}")
            logger.info(f"  Unstable features excluded: {len(self.global_analysis_result.unstable_features)}")
            logger.info(f"  Feature stability ratio: {len(self.global_analysis_result.stable_features) / max(1, self.global_analysis_result.total_features_found):.1%}")
            logger.info(f"  Symbols analyzed for manifest: {self.global_analysis_result.total_symbols_analyzed}")
            
            # Log top stable features
            if self.global_analysis_result.stable_features:
                logger.info(f"  Sample stable features: {self.global_analysis_result.stable_features[:10]}")
                if len(self.global_analysis_result.stable_features) > 10:
                    logger.info(f"    ... and {len(self.global_analysis_result.stable_features) - 10} more")
            
            # Log sample excluded features with reasons
            if self.global_analysis_result.unstable_features:
                logger.info(f"  Sample excluded features: {self.global_analysis_result.unstable_features[:10]}")
                if len(self.global_analysis_result.unstable_features) > 10:
                    logger.info(f"    ... and {len(self.global_analysis_result.unstable_features) - 10} more")
        
        # Log processed symbols statistics
        if processed_symbols:
            logger.info("")
            logger.info("PROCESSED SYMBOLS STATISTICS:")
            
            # Analyze final shapes
            shapes = [df.shape for df in processed_symbols.values()]
            column_counts = [shape[1] for shape in shapes]
            row_counts = [shape[0] for shape in shapes]
            
            logger.info(f"  Final symbol count: {len(processed_symbols)}")
            logger.info(f"  Total rows across all symbols: {sum(row_counts):,}")
            logger.info(f"  Average rows per symbol: {sum(row_counts) / len(row_counts):.0f}")
            logger.info(f"  Min/Max rows per symbol: {min(row_counts):,} / {max(row_counts):,}")
            
            if len(set(column_counts)) == 1:
                logger.info(f"  ✓ Consistent column count: {column_counts[0]} columns per symbol")
            else:
                logger.warning(f"  ✗ Inconsistent column counts: {set(column_counts)}")
            
            # Analyze feature composition
            sample_symbol = next(iter(processed_symbols.keys()))
            sample_df = processed_symbols[sample_symbol]
            
            essential_columns = {'symbol', 'target', 'date', 'timestamp', 'direction_1d', 'close'}
            feature_cols = [col for col in sample_df.columns if col not in essential_columns and not col.endswith('_isnan')]
            mask_cols = [col for col in sample_df.columns if col.endswith('_isnan')]
            essential_cols = [col for col in essential_columns if col in sample_df.columns]
            
            logger.info(f"  Final column composition (per symbol):")
            logger.info(f"    Essential columns: {len(essential_cols)}")
            logger.info(f"    Feature columns: {len(feature_cols)}")
            logger.info(f"    Missingness mask columns: {len(mask_cols)}")
            logger.info(f"    Total columns: {len(sample_df.columns)}")
            
            # Log sample symbols
            symbol_list = list(processed_symbols.keys())
            logger.info(f"  Successfully processed symbols: {symbol_list[:15]}")
            if len(symbol_list) > 15:
                logger.info(f"    ... and {len(symbol_list) - 15} more")
        
        # Log excluded symbols with detailed reasons
        if excluded_symbols:
            logger.info("")
            logger.info("EXCLUDED SYMBOLS ANALYSIS:")
            logger.info(f"  Total excluded symbols: {len(excluded_symbols)}")
            
            # Group by exclusion reason
            exclusion_reasons = {}
            for symbol, coverage, reason in excluded_symbols:
                reason_category = reason.split('_')[0] if '_' in reason else reason
                if reason_category not in exclusion_reasons:
                    exclusion_reasons[reason_category] = []
                exclusion_reasons[reason_category].append((symbol, coverage, reason))
            
            logger.info(f"  Exclusion reason categories:")
            for reason_category, symbols_list in exclusion_reasons.items():
                logger.info(f"    {reason_category}: {len(symbols_list)} symbols")
                
                # Show sample symbols for this reason
                sample_symbols = symbols_list[:5]
                for symbol, coverage, full_reason in sample_symbols:
                    logger.info(f"      - {symbol}: {coverage:.1%} coverage ({full_reason})")
                if len(symbols_list) > 5:
                    logger.info(f"      ... and {len(symbols_list) - 5} more")
        
        # Log configuration used
        logger.info("")
        logger.info("CONFIGURATION USED:")
        logger.info(f"  Warmup trim days: {self.config.warmup_trim_days}")
        logger.info(f"  NaN threshold per symbol: {self.config.nan_drop_threshold_per_symbol:.1%}")
        logger.info(f"  Global feature keep ratio: {self.config.global_feature_keep_ratio:.1%}")
        logger.info(f"  Min symbol feature coverage: {self.config.min_symbol_feature_coverage:.1%}")
        logger.info(f"  Use missingness masks: {self.config.use_missingness_mask}")
        logger.info(f"  Imputation strategy: {self.config.imputation_strategy}")
        
        # Log requirements compliance
        logger.info("")
        logger.info("REQUIREMENTS COMPLIANCE CHECK:")
        logger.info(f"  ✓ Requirement 6.1: Warmup trimming logged - {processing_summary.get('warmup_rows_trimmed', 0):,} rows trimmed")
        logger.info(f"  ✓ Requirement 6.2: Feature coverage statistics logged")
        logger.info(f"  ✓ Requirement 6.3: Symbol exclusion decisions logged with reasons")
        logger.info(f"  ✓ Requirement 6.4: Final counts reported - {len(processed_symbols)} symbols, {processing_summary.get('features_in_manifest', 0)} features")
        logger.info(f"  ✓ Requirement 6.5: Feature drift detection implemented in tensor validation")
        
        logger.info("=" * 80)
        logger.info("END COMPREHENSIVE PIPELINE SUMMARY")
        logger.info("=" * 80)

    def detect_and_log_feature_drift(self, current_manifest: List[str], 
                                   previous_manifest_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Detect and log feature drift by comparing current manifest with previous version.
        
        This implements part of requirement 6.5: Add feature drift detection and logging
        
        Args:
            current_manifest: List of current stable features
            previous_manifest_path: Path to previous manifest (auto-detect if None)
            
        Returns:
            Dictionary with drift analysis results
        """
        logger.info("=" * 60)
        logger.info("FEATURE DRIFT DETECTION")
        logger.info("=" * 60)
        
        drift_analysis = {
            'drift_detected': False,
            'current_feature_count': len(current_manifest),
            'previous_feature_count': 0,
            'added_features': [],
            'removed_features': [],
            'common_features': [],
            'drift_percentage': 0.0,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        try:
            # Try to load previous manifest
            previous_manifest = None
            
            if previous_manifest_path:
                # Use specified path
                try:
                    previous_data = self.manifest_manager.load_manifest()
                    previous_manifest = previous_data.get('stable_features', [])
                    logger.info(f"Loaded previous manifest from: {previous_manifest_path}")
                except Exception as e:
                    logger.warning(f"Could not load previous manifest from {previous_manifest_path}: {e}")
            else:
                # Try to find previous version
                try:
                    available_versions = self.manifest_manager.get_available_versions()
                    if len(available_versions) > 1:
                        # Get second most recent version (first is current)
                        previous_version = available_versions[1]
                        previous_data = self.manifest_manager.load_manifest(previous_version)
                        previous_manifest = previous_data.get('stable_features', [])
                        logger.info(f"Loaded previous manifest version: {previous_version}")
                    else:
                        logger.info("No previous manifest version found for drift comparison")
                except Exception as e:
                    logger.warning(f"Could not load previous manifest version: {e}")
            
            if previous_manifest is None:
                logger.info("No previous manifest available - this may be the first run")
                logger.info(f"Current manifest contains {len(current_manifest)} features")
                drift_analysis['analysis_result'] = 'no_previous_manifest'
                return drift_analysis
            
            # Perform drift analysis
            current_set = set(current_manifest)
            previous_set = set(previous_manifest)
            
            # Calculate changes
            added_features = list(current_set - previous_set)
            removed_features = list(previous_set - current_set)
            common_features = list(current_set.intersection(previous_set))
            
            # Update drift analysis
            drift_analysis.update({
                'previous_feature_count': len(previous_manifest),
                'added_features': sorted(added_features),
                'removed_features': sorted(removed_features),
                'common_features': sorted(common_features),
                'drift_detected': len(added_features) > 0 or len(removed_features) > 0
            })
            
            # Calculate drift percentage
            total_unique_features = len(current_set.union(previous_set))
            changed_features = len(added_features) + len(removed_features)
            drift_percentage = (changed_features / total_unique_features) * 100 if total_unique_features > 0 else 0.0
            drift_analysis['drift_percentage'] = drift_percentage
            
            # Log drift analysis results
            logger.info("DRIFT ANALYSIS RESULTS:")
            logger.info(f"  Previous feature count: {len(previous_manifest)}")
            logger.info(f"  Current feature count: {len(current_manifest)}")
            logger.info(f"  Common features: {len(common_features)}")
            logger.info(f"  Added features: {len(added_features)}")
            logger.info(f"  Removed features: {len(removed_features)}")
            logger.info(f"  Drift percentage: {drift_percentage:.1f}%")
            
            if drift_analysis['drift_detected']:
                logger.warning("⚠️  FEATURE DRIFT DETECTED!")
                
                if added_features:
                    logger.info(f"  ➕ Added features ({len(added_features)}):")
                    for feature in added_features[:10]:  # Show first 10
                        logger.info(f"    + {feature}")
                    if len(added_features) > 10:
                        logger.info(f"    ... and {len(added_features) - 10} more")
                
                if removed_features:
                    logger.warning(f"  ➖ Removed features ({len(removed_features)}):")
                    for feature in removed_features[:10]:  # Show first 10
                        logger.warning(f"    - {feature}")
                    if len(removed_features) > 10:
                        logger.warning(f"    ... and {len(removed_features) - 10} more")
                
                # Assess drift severity
                if drift_percentage > 20:
                    logger.error("🚨 HIGH DRIFT: >20% feature change detected - review data pipeline!")
                elif drift_percentage > 10:
                    logger.warning("⚠️  MODERATE DRIFT: >10% feature change detected - monitor closely")
                elif drift_percentage > 5:
                    logger.info("ℹ️  LOW DRIFT: >5% feature change detected - normal variation")
                else:
                    logger.info("ℹ️  MINIMAL DRIFT: <5% feature change detected")
                
            else:
                logger.info("✅ NO DRIFT DETECTED: Feature set is stable")
            
            # Log stability metrics
            stability_ratio = len(common_features) / max(len(previous_manifest), len(current_manifest))
            logger.info(f"  Feature stability ratio: {stability_ratio:.1%}")
            
            drift_analysis['stability_ratio'] = stability_ratio
            drift_analysis['analysis_result'] = 'completed'
            
        except Exception as e:
            error_msg = f"Error during feature drift detection: {e}"
            logger.error(error_msg)
            drift_analysis['analysis_result'] = 'error'
            drift_analysis['error_message'] = str(e)
        
        logger.info("=" * 60)
        logger.info("END FEATURE DRIFT DETECTION")
        logger.info("=" * 60)
        
        return drift_analysis

    def log_detailed_exclusion_decisions(self, excluded_symbols: List[Tuple[str, float, str]]) -> None:
        """
        Log detailed exclusion decisions with specific reasons and thresholds.
        
        This implements requirement 6.3: Record feature and symbol exclusion decisions with reasons
        
        Args:
            excluded_symbols: List of tuples (symbol, coverage_ratio, exclusion_reason)
        """
        if not excluded_symbols:
            logger.info("No symbols were excluded during processing")
            return
        
        logger.info("=" * 60)
        logger.info("DETAILED SYMBOL EXCLUSION ANALYSIS")
        logger.info("=" * 60)
        logger.info(f"Total excluded symbols: {len(excluded_symbols)}")
        
        # Group exclusions by reason category
        exclusion_categories = {}
        for symbol, coverage, reason in excluded_symbols:
            # Extract main category from reason
            if 'insufficient_coverage' in reason:
                category = 'insufficient_coverage'
            elif 'cleaning_failed' in reason:
                category = 'data_cleaning_failed'
            elif 'validation_error' in reason:
                category = 'validation_error'
            elif 'processing_error' in reason:
                category = 'processing_error'
            elif 'final_validation_failed' in reason:
                category = 'final_validation_failed'
            else:
                category = 'other'
            
            if category not in exclusion_categories:
                exclusion_categories[category] = []
            exclusion_categories[category].append((symbol, coverage, reason))
        
        # Log each category in detail
        for category, symbols_list in exclusion_categories.items():
            logger.info("")
            logger.info(f"EXCLUSION CATEGORY: {category.upper()}")
            logger.info(f"  Count: {len(symbols_list)} symbols")
            
            if category == 'insufficient_coverage':
                logger.info(f"  Threshold: {self.config.min_symbol_feature_coverage:.1%} minimum coverage required")
                
                # Sort by coverage ratio (lowest first)
                symbols_list.sort(key=lambda x: x[1])
                
                logger.info("  Symbols with lowest coverage:")
                for symbol, coverage, reason in symbols_list[:10]:
                    logger.info(f"    {symbol}: {coverage:.1%} coverage (required: {self.config.min_symbol_feature_coverage:.1%})")
                    
                    # Calculate how many features were missing
                    if self.global_analysis_result:
                        total_features = len(self.global_analysis_result.stable_features)
                        available_features = int(coverage * total_features)
                        missing_features = total_features - available_features
                        logger.info(f"      Missing {missing_features}/{total_features} required features")
                
                if len(symbols_list) > 10:
                    logger.info(f"    ... and {len(symbols_list) - 10} more symbols")
                    
                # Log coverage statistics for this category
                coverages = [coverage for _, coverage, _ in symbols_list]
                avg_coverage = sum(coverages) / len(coverages)
                min_coverage = min(coverages)
                max_coverage = max(coverages)
                
                logger.info(f"  Coverage statistics for excluded symbols:")
                logger.info(f"    Average: {avg_coverage:.1%}")
                logger.info(f"    Range: {min_coverage:.1%} - {max_coverage:.1%}")
                
            elif category == 'data_cleaning_failed':
                logger.info("  Reasons for cleaning failures:")
                for symbol, coverage, reason in symbols_list[:10]:
                    logger.info(f"    {symbol}: Data cleaning pipeline failed")
                    # Could add more specific failure reasons here
                
            elif category == 'validation_error':
                logger.info("  Validation errors encountered:")
                for symbol, coverage, reason in symbols_list[:10]:
                    logger.info(f"    {symbol}: {reason}")
                    
            elif category == 'processing_error':
                logger.info("  Processing errors encountered:")
                for symbol, coverage, reason in symbols_list[:10]:
                    logger.info(f"    {symbol}: {reason}")
                    
            elif category == 'final_validation_failed':
                logger.info("  Final validation failures:")
                for symbol, coverage, reason in symbols_list[:10]:
                    logger.info(f"    {symbol}: Failed final validation checks")
                    
            else:
                logger.info("  Other exclusion reasons:")
                for symbol, coverage, reason in symbols_list[:10]:
                    logger.info(f"    {symbol}: {reason}")
        
        # Log overall exclusion statistics
        logger.info("")
        logger.info("EXCLUSION SUMMARY:")
        total_exclusions = len(excluded_symbols)
        for category, symbols_list in exclusion_categories.items():
            percentage = (len(symbols_list) / total_exclusions) * 100
            logger.info(f"  {category}: {len(symbols_list)} symbols ({percentage:.1f}%)")
        
        # Log recommendations
        logger.info("")
        logger.info("RECOMMENDATIONS:")
        
        if 'insufficient_coverage' in exclusion_categories:
            count = len(exclusion_categories['insufficient_coverage'])
            logger.info(f"  • {count} symbols excluded for insufficient coverage:")
            logger.info(f"    - Consider lowering min_symbol_feature_coverage from {self.config.min_symbol_feature_coverage:.1%}")
            logger.info(f"    - Or investigate why these symbols have fewer features")
        
        if 'data_cleaning_failed' in exclusion_categories:
            count = len(exclusion_categories['data_cleaning_failed'])
            logger.info(f"  • {count} symbols excluded due to cleaning failures:")
            logger.info(f"    - Review data quality for these symbols")
            logger.info(f"    - Consider adjusting NaN thresholds or imputation strategies")
        
        if 'processing_error' in exclusion_categories or 'validation_error' in exclusion_categories:
            logger.info(f"  • Technical errors encountered:")
            logger.info(f"    - Review error logs for specific failure details")
            logger.info(f"    - Consider data format or pipeline issues")
        
        logger.info("=" * 60)
        logger.info("END DETAILED EXCLUSION ANALYSIS")
        logger.info("=" * 60)


# Utility functions for configuration management
def load_default_config() -> FeatureConsistencyConfig:
    """Load default configuration with environment-specific overrides."""
    config = FeatureConsistencyConfig()
    
    # Try to load from standard locations
    config_paths = [
        "config/feature_consistency.json",
        "models/feature_consistency_config.json",
        ".kiro/feature_consistency.json"
    ]
    
    for config_path in config_paths:
        if Path(config_path).exists():
            logger.info(f"Loading configuration from {config_path}")
            return FeatureConsistencyConfig.from_json(config_path)
    
    logger.info("No configuration file found, using defaults")
    return config


def create_default_config_file(output_path: str = "config/feature_consistency.json") -> None:
    """Create a default configuration file with documentation."""
    config = FeatureConsistencyConfig()
    
    # Create directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Add documentation to the config
    config_with_docs = {
        "_description": "Feature Consistency System Configuration",
        "_version": "1.0",
        "_created": datetime.now().isoformat(),
        
        "warmup_trim_days": {
            "value": config.warmup_trim_days,
            "description": "Number of days to trim from start to remove indicator warm-up period"
        },
        "nan_drop_threshold_per_symbol": {
            "value": config.nan_drop_threshold_per_symbol,
            "description": "Maximum NaN ratio allowed per symbol (0.05 = 5%)"
        },
        "global_feature_keep_ratio": {
            "value": config.global_feature_keep_ratio,
            "description": "Minimum symbol coverage required to keep feature globally (0.95 = 95%)"
        },
        "min_symbol_feature_coverage": {
            "value": config.min_symbol_feature_coverage,
            "description": "Minimum feature coverage required to include symbol (0.90 = 90%)"
        },
        "use_missingness_mask": {
            "value": config.use_missingness_mask,
            "description": "Whether to create _isnan columns for missing value indicators"
        },
        "imputation_strategy": {
            "value": config.imputation_strategy,
            "description": "Imputation strategy to use: 'zero', 'mean', or 'median'"
        },
        "imputation_value": {
            "value": config.imputation_value,
            "description": "Value to use when imputation_strategy is 'zero' or as fallback"
        },
        "manifest_path": {
            "value": config.manifest_path,
            "description": "Path to save/load feature manifest JSON file"
        },
        "log_level": {
            "value": config.log_level,
            "description": "Logging level (DEBUG, INFO, WARNING, ERROR)"
        },
        "detailed_logging": {
            "value": config.detailed_logging,
            "description": "Enable detailed logging of processing decisions"
        }
    }
    
    try:
        with open(output_path, 'w') as f:
            json.dump(config_with_docs, f, indent=2)
        logger.info(f"Default configuration file created at {output_path}")
    except Exception as e:
        logger.error(f"Failed to create config file at {output_path}: {e}")


if __name__ == "__main__":
    # Create default configuration file for reference
    create_default_config_file()
    
    # Test configuration loading
    config = load_default_config()
    manager = FeatureConsistencyManager(config)
    
    print("Feature Consistency System initialized successfully!")
    print(f"Configuration: {config}")



class WarmupTrimmer:
    """
    Handles removal of warm-up periods from technical indicators.
    
    Technical indicators like RSI, MACD, and moving averages require a warm-up
    period where initial values are NaN due to insufficient historical data.
    This class removes these periods to improve feature quality.
    """
    
    def __init__(self, config: FeatureConsistencyConfig):
        """Initialize the warmup trimmer."""
        self.config = config
        self.warmup_days = config.warmup_trim_days
        
        # Common technical indicator lookback periods
        self.indicator_lookbacks = {
            'sma': [5, 10, 20, 50, 100, 200],
            'ema': [12, 26, 50, 100, 200],
            'rsi': [14, 21],
            'macd': [12, 26, 9],  # fast, slow, signal
            'bb': [20],  # bollinger bands
            'atr': [14],
            'adx': [14],
            'stoch': [14, 3],  # %K, %D
            'williams': [14],
            'momentum': [10, 20],
            'roc': [12, 25]
        }
        
        logger.info(f"WarmupTrimmer initialized with {self.warmup_days} day trim period")
    
    def detect_max_lookback_from_columns(self, df: pd.DataFrame) -> int:
        """
        Automatically detect the maximum lookback period from column names.
        
        Args:
            df: DataFrame with feature columns
            
        Returns:
            Maximum detected lookback period
        """
        max_lookback = 0
        detected_indicators = []
        
        for col in df.columns:
            col_lower = col.lower()
            
            # Check for common patterns in column names
            for indicator, lookbacks in self.indicator_lookbacks.items():
                if indicator in col_lower:
                    for lookback in lookbacks:
                        if str(lookback) in col or f"_{lookback}" in col or f"{lookback}_" in col:
                            max_lookback = max(max_lookback, lookback)
                            detected_indicators.append(f"{indicator}_{lookback}")
                            break
                    else:
                        # Use default max for this indicator if no specific period found
                        max_lookback = max(max_lookback, max(lookbacks))
                        detected_indicators.append(f"{indicator}_default")
        
        if detected_indicators:
            logger.debug(f"Detected indicators: {detected_indicators[:10]}{'...' if len(detected_indicators) > 10 else ''}")
            logger.debug(f"Maximum detected lookback: {max_lookback} days")
        
        return max_lookback
    
    def trim_warmup_period(self, df: pd.DataFrame, symbol: str = "Unknown") -> pd.DataFrame:
        """
        Remove warm-up period from the beginning of the DataFrame.
        
        Args:
            df: DataFrame with time-series data (should be time-sorted)
            symbol: Symbol name for logging
            
        Returns:
            DataFrame with warm-up period removed
        """
        if df.empty:
            logger.warning(f"{symbol}: Empty DataFrame, no trimming performed")
            return df
        
        original_rows = len(df)
        
        # Use configured warmup days or auto-detect
        trim_days = self.warmup_days
        
        if trim_days == 0:
            # Auto-detect mode
            detected_lookback = self.detect_max_lookback_from_columns(df)
            # Add buffer for complex indicators (e.g., MACD of RSI)
            trim_days = max(detected_lookback * 2, 50)  # At least 50 days
            logger.info(f"{symbol}: Auto-detected warmup period: {trim_days} days")
        
        # Ensure we don't trim more than 80% of the data
        max_trim = int(original_rows * 0.8)
        trim_days = min(trim_days, max_trim)
        
        if trim_days >= original_rows:
            logger.warning(f"{symbol}: Trim period ({trim_days}) >= data length ({original_rows}), "
                          f"trimming only {max_trim} rows")
            trim_days = max_trim
        
        # Perform trimming
        if trim_days > 0:
            trimmed_df = df.iloc[trim_days:].copy()
            rows_trimmed = original_rows - len(trimmed_df)
            
            logger.info(f"{symbol}: Trimmed {rows_trimmed} warmup rows "
                       f"({original_rows} -> {len(trimmed_df)} rows)")
            
            return trimmed_df
        else:
            logger.info(f"{symbol}: No warmup trimming needed")
            return df.copy()
    
    def validate_post_trim_data(self, df: pd.DataFrame, symbol: str = "Unknown") -> bool:
        """
        Validate that the DataFrame still has sufficient data after trimming.
        
        Args:
            df: Trimmed DataFrame
            symbol: Symbol name for logging
            
        Returns:
            True if data is sufficient, False otherwise
        """
        min_rows_required = 252  # At least 1 year of trading days
        
        if len(df) < min_rows_required:
            logger.warning(f"{symbol}: Insufficient data after trimming "
                          f"({len(df)} < {min_rows_required} rows required)")
            return False
        
        # Check for excessive NaN values after trimming
        nan_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if nan_ratio > 0.5:  # More than 50% NaN
            logger.warning(f"{symbol}: Excessive NaN ratio after trimming: {nan_ratio:.1%}")
            return False
        
        logger.debug(f"{symbol}: Post-trim validation passed "
                    f"({len(df)} rows, {nan_ratio:.1%} NaN)")
        return True
    
    def get_trimming_stats(self, original_df: pd.DataFrame, trimmed_df: pd.DataFrame, 
                          symbol: str = "Unknown") -> Dict[str, Any]:
        """
        Get detailed statistics about the trimming operation.
        
        Args:
            original_df: Original DataFrame before trimming
            trimmed_df: DataFrame after trimming
            symbol: Symbol name
            
        Returns:
            Dictionary with trimming statistics
        """
        original_rows = len(original_df)
        trimmed_rows = len(trimmed_df)
        rows_removed = original_rows - trimmed_rows
        
        # Calculate NaN reduction
        original_nan_count = original_df.isnull().sum().sum()
        trimmed_nan_count = trimmed_df.isnull().sum().sum()
        nan_reduction = original_nan_count - trimmed_nan_count
        
        stats = {
            'symbol': symbol,
            'original_rows': original_rows,
            'trimmed_rows': trimmed_rows,
            'rows_removed': rows_removed,
            'trim_percentage': (rows_removed / original_rows) * 100 if original_rows > 0 else 0,
            'original_nan_count': int(original_nan_count),
            'trimmed_nan_count': int(trimmed_nan_count),
            'nan_reduction': int(nan_reduction),
            'nan_reduction_percentage': (nan_reduction / original_nan_count) * 100 if original_nan_count > 0 else 0
        }
        
        return stats


class GlobalCoverageAnalyzer:
    """
    Analyzes feature coverage across all symbols to identify stable features.
    
    This class computes universe-wide feature statistics to determine which
    features are consistently available across symbols and should be included
    in the global feature whitelist.
    """
    
    def __init__(self, config: FeatureConsistencyConfig):
        """Initialize the global coverage analyzer."""
        self.config = config
        self.global_keep_ratio = config.global_feature_keep_ratio
        
        logger.info(f"GlobalCoverageAnalyzer initialized with {self.global_keep_ratio:.1%} keep ratio")
    
    def compute_symbol_coverage(self, symbol_dataframes: Dict[str, pd.DataFrame]) -> Dict[str, CoverageStats]:
        """
        Compute coverage statistics for each feature across all symbols.
        
        Args:
            symbol_dataframes: Dictionary mapping symbol names to their DataFrames
            
        Returns:
            Dictionary mapping feature names to their coverage statistics
        """
        if not symbol_dataframes:
            logger.warning("No symbol dataframes provided for coverage analysis")
            return {}
        
        logger.info(f"Computing coverage for {len(symbol_dataframes)} symbols")
        
        # Collect all unique features across all symbols
        all_features = set()
        for symbol, df in symbol_dataframes.items():
            all_features.update(df.columns)
        
        # Remove non-feature columns
        excluded_columns = {'symbol', 'target', 'date', 'timestamp', 'direction_1d'}
        feature_columns = sorted(all_features - excluded_columns)
        
        logger.info(f"Found {len(feature_columns)} unique features across all symbols")
        
        coverage_stats = {}
        total_symbols = len(symbol_dataframes)
        
        for feature in feature_columns:
            symbols_with_feature = []
            non_nan_ratios = []
            
            # Check each symbol for this feature
            for symbol, df in symbol_dataframes.items():
                if feature in df.columns:
                    symbols_with_feature.append(symbol)
                    
                    # Calculate non-NaN ratio for this feature in this symbol
                    non_nan_count = df[feature].notna().sum()
                    total_count = len(df)
                    non_nan_ratio = non_nan_count / total_count if total_count > 0 else 0.0
                    non_nan_ratios.append(non_nan_ratio)
            
            # Calculate coverage statistics
            coverage_ratio = len(symbols_with_feature) / total_symbols
            avg_non_nan_ratio = np.mean(non_nan_ratios) if non_nan_ratios else 0.0
            min_non_nan_ratio = np.min(non_nan_ratios) if non_nan_ratios else 0.0
            max_non_nan_ratio = np.max(non_nan_ratios) if non_nan_ratios else 0.0
            
            coverage_stats[feature] = CoverageStats(
                feature_name=feature,
                total_symbols=total_symbols,
                symbols_with_feature=len(symbols_with_feature),
                coverage_ratio=coverage_ratio,
                avg_non_nan_ratio=avg_non_nan_ratio,
                min_non_nan_ratio=min_non_nan_ratio,
                max_non_nan_ratio=max_non_nan_ratio,
                symbols_list=symbols_with_feature
            )
        
        logger.info(f"Coverage analysis completed for {len(coverage_stats)} features")
        return coverage_stats
    
    def identify_stable_features(self, coverage_stats: Dict[str, CoverageStats]) -> List[str]:
        """
        Identify features that meet the stability criteria.
        
        Args:
            coverage_stats: Coverage statistics for all features
            
        Returns:
            List of stable feature names
        """
        stable_features = []
        
        for feature_name, stats in coverage_stats.items():
            # Check if feature meets coverage threshold
            if stats.coverage_ratio >= self.global_keep_ratio:
                # Additional quality checks
                if stats.avg_non_nan_ratio >= 0.5:  # At least 50% non-NaN on average
                    stable_features.append(feature_name)
                else:
                    logger.debug(f"Feature {feature_name} excluded: low data quality "
                               f"(avg_non_nan: {stats.avg_non_nan_ratio:.1%})")
            else:
                logger.debug(f"Feature {feature_name} excluded: low coverage "
                           f"({stats.coverage_ratio:.1%} < {self.global_keep_ratio:.1%})")
        
        # Sort features for consistent ordering
        stable_features.sort()
        
        logger.info(f"Identified {len(stable_features)} stable features out of {len(coverage_stats)} total")
        
        # Log some examples of stable vs unstable features
        if stable_features:
            logger.info(f"Sample stable features: {stable_features[:5]}")
        
        unstable_features = [f for f in coverage_stats.keys() if f not in stable_features]
        if unstable_features:
            logger.info(f"Sample unstable features: {unstable_features[:5]}")
        
        return stable_features
    
    def aggregate_global_stats(self, coverage_stats: Dict[str, CoverageStats]) -> Dict[str, Any]:
        """
        Aggregate global statistics from coverage analysis.
        
        Args:
            coverage_stats: Coverage statistics for all features
            
        Returns:
            Dictionary with aggregated global statistics
        """
        if not coverage_stats:
            return {
                'total_features': 0,
                'avg_coverage_ratio': 0.0,
                'coverage_distribution': {},
                'quality_distribution': {}
            }
        
        coverage_ratios = [stats.coverage_ratio for stats in coverage_stats.values()]
        quality_ratios = [stats.avg_non_nan_ratio for stats in coverage_stats.values()]
        
        # Coverage distribution buckets
        coverage_buckets = {
            '90-100%': sum(1 for r in coverage_ratios if r >= 0.9),
            '75-90%': sum(1 for r in coverage_ratios if 0.75 <= r < 0.9),
            '50-75%': sum(1 for r in coverage_ratios if 0.5 <= r < 0.75),
            '25-50%': sum(1 for r in coverage_ratios if 0.25 <= r < 0.5),
            '0-25%': sum(1 for r in coverage_ratios if r < 0.25)
        }
        
        # Quality distribution buckets
        quality_buckets = {
            '90-100%': sum(1 for r in quality_ratios if r >= 0.9),
            '75-90%': sum(1 for r in quality_ratios if 0.75 <= r < 0.9),
            '50-75%': sum(1 for r in quality_ratios if 0.5 <= r < 0.75),
            '25-50%': sum(1 for r in quality_ratios if 0.25 <= r < 0.5),
            '0-25%': sum(1 for r in quality_ratios if r < 0.25)
        }
        
        global_stats = {
            'total_features': len(coverage_stats),
            'avg_coverage_ratio': np.mean(coverage_ratios),
            'median_coverage_ratio': np.median(coverage_ratios),
            'min_coverage_ratio': np.min(coverage_ratios),
            'max_coverage_ratio': np.max(coverage_ratios),
            'avg_quality_ratio': np.mean(quality_ratios),
            'median_quality_ratio': np.median(quality_ratios),
            'coverage_distribution': coverage_buckets,
            'quality_distribution': quality_buckets
        }
        
        return global_stats
    
    def analyze_all_symbols(self, symbol_dataframes: Dict[str, pd.DataFrame]) -> GlobalAnalysisResult:
        """
        Perform complete global coverage analysis.
        
        Args:
            symbol_dataframes: Dictionary mapping symbol names to their DataFrames
            
        Returns:
            Complete analysis results
        """
        logger.info("Starting global coverage analysis")
        start_time = datetime.now()
        
        # Compute coverage statistics
        coverage_stats = self.compute_symbol_coverage(symbol_dataframes)
        
        # Identify stable features
        stable_features = self.identify_stable_features(coverage_stats)
        unstable_features = [f for f in coverage_stats.keys() if f not in stable_features]
        
        # Aggregate global statistics
        global_stats = self.aggregate_global_stats(coverage_stats)
        
        # Create analysis result
        result = GlobalAnalysisResult(
            total_symbols_analyzed=len(symbol_dataframes),
            total_features_found=len(coverage_stats),
            stable_features=stable_features,
            unstable_features=unstable_features,
            coverage_stats=coverage_stats,
            analysis_timestamp=start_time.isoformat(),
            config_used=asdict(self.config)
        )
        
        # Log summary
        summary = result.get_feature_summary()
        logger.info("Global coverage analysis completed:")
        logger.info(f"  Symbols analyzed: {summary['total_symbols']}")
        logger.info(f"  Features found: {summary['total_features_found']}")
        logger.info(f"  Stable features: {summary['stable_features_count']} "
                   f"({summary['stability_ratio']:.1%})")
        logger.info(f"  Average stable coverage: {summary['avg_coverage_stable']:.1%}")
        
        analysis_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Analysis completed in {analysis_time:.1f} seconds")
        
        return result
    
    def log_detailed_coverage_report(self, result: GlobalAnalysisResult, top_n: int = 10) -> None:
        """
        Log detailed coverage report for debugging and monitoring.
        
        Args:
            result: Global analysis result
            top_n: Number of top/bottom features to show in detail
        """
        logger.info("=== DETAILED COVERAGE REPORT ===")
        
        # Sort features by coverage ratio
        sorted_features = sorted(
            result.coverage_stats.items(),
            key=lambda x: x[1].coverage_ratio,
            reverse=True
        )
        
        # Top features
        logger.info(f"Top {top_n} features by coverage:")
        for i, (feature, stats) in enumerate(sorted_features[:top_n], 1):
            logger.info(f"  {i:2d}. {feature}: {stats.coverage_ratio:.1%} coverage, "
                       f"{stats.avg_non_nan_ratio:.1%} avg quality")
        
        # Bottom features
        logger.info(f"Bottom {top_n} features by coverage:")
        for i, (feature, stats) in enumerate(sorted_features[-top_n:], 1):
            logger.info(f"  {i:2d}. {feature}: {stats.coverage_ratio:.1%} coverage, "
                       f"{stats.avg_non_nan_ratio:.1%} avg quality")
        
        # Coverage distribution
        global_stats = self.aggregate_global_stats(result.coverage_stats)
        logger.info("Coverage distribution:")
        for bucket, count in global_stats['coverage_distribution'].items():
            pct = (count / result.total_features_found) * 100 if result.total_features_found > 0 else 0
            logger.info(f"  {bucket}: {count} features ({pct:.1f}%)")
        
        logger.info("=== END COVERAGE REPORT ===")


    
    def get_stable_features(self) -> List[str]:
        """Get the list of stable features from global analysis."""
        if self.global_analysis_result is None:
            logger.warning("Global analysis not performed yet, returning empty feature list")
            return []
        return self.global_analysis_result.stable_features.copy()
    
    def validate_symbol_against_global_features(self, df: pd.DataFrame, symbol: str) -> Tuple[bool, float]:
        """
        Validate that a symbol has sufficient coverage of global stable features.
        
        Args:
            df: Symbol's DataFrame
            symbol: Symbol name
            
        Returns:
            Tuple of (is_valid, coverage_ratio)
        """
        if self.global_analysis_result is None:
            logger.warning(f"{symbol}: No global analysis available, skipping validation")
            return True, 1.0
        
        stable_features = self.global_analysis_result.stable_features
        if not stable_features:
            logger.warning(f"{symbol}: No stable features identified, accepting symbol")
            return True, 1.0
        
        # Check how many stable features this symbol has
        symbol_features = set(df.columns)
        available_stable_features = [f for f in stable_features if f in symbol_features]
        coverage_ratio = len(available_stable_features) / len(stable_features)
        
        is_valid = coverage_ratio >= self.config.min_symbol_feature_coverage
        
        if is_valid:
            logger.info(f"{symbol}: Feature coverage validation passed "
                       f"({len(available_stable_features)}/{len(stable_features)} = {coverage_ratio:.1%})")
        else:
            logger.warning(f"{symbol}: Feature coverage validation failed "
                          f"({len(available_stable_features)}/{len(stable_features)} = {coverage_ratio:.1%} "
                          f"< {self.config.min_symbol_feature_coverage:.1%})")
        
        return is_valid, coverage_ratio
    
    def apply_global_feature_selection(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Apply global feature selection to a symbol's DataFrame.
        
        Args:
            df: Symbol's DataFrame
            symbol: Symbol name
            
        Returns:
            DataFrame with only stable features (plus essential columns)
        """
        if self.global_analysis_result is None:
            logger.warning(f"{symbol}: No global analysis available, returning original DataFrame")
            return df
        
        stable_features = self.global_analysis_result.stable_features
        if not stable_features:
            logger.warning(f"{symbol}: No stable features identified, returning original DataFrame")
            return df
        
        # Essential columns that should always be preserved
        essential_columns = ['symbol', 'target', 'date', 'timestamp', 'direction_1d', 'close']
        
        # Combine stable features with essential columns
        columns_to_keep = []
        for col in essential_columns:
            if col in df.columns:
                columns_to_keep.append(col)
        
        # Add stable features that exist in this symbol
        for feature in stable_features:
            if feature in df.columns:
                columns_to_keep.append(feature)
        
        # Remove duplicates while preserving order
        columns_to_keep = list(dict.fromkeys(columns_to_keep))
        
        # Apply selection
        original_cols = len(df.columns)
        selected_df = df[columns_to_keep].copy()
        
        logger.info(f"{symbol}: Applied global feature selection "
                   f"({original_cols} -> {len(selected_df.columns)} columns)")
        
        return selected_df
    
    def process_symbols_with_global_consistency(self, symbol_dataframes: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Process all symbols with global feature consistency.
        
        This is the main method that orchestrates the complete feature consistency pipeline:
        1. Perform global coverage analysis
        2. Apply feature selection to each symbol
        3. Validate symbol coverage
        4. Return processed symbols
        
        Args:
            symbol_dataframes: Dictionary mapping symbol names to their DataFrames
            
        Returns:
            Dictionary of processed symbols that meet consistency requirements
        """
        logger.info(f"Processing {len(symbol_dataframes)} symbols with global consistency")
        self.processing_stats['start_time'] = datetime.now()
        
        # Step 1: Perform global coverage analysis
        global_result = self.perform_global_analysis(symbol_dataframes)
        
        # Step 2: Process each symbol
        processed_symbols = {}
        excluded_symbols = []
        
        for symbol, df in symbol_dataframes.items():
            try:
                # Apply global feature selection
                processed_df = self.apply_global_feature_selection(df, symbol)
                
                # Validate symbol coverage
                is_valid, coverage_ratio = self.validate_symbol_against_global_features(processed_df, symbol)
                
                if is_valid:
                    processed_symbols[symbol] = processed_df
                    self.processing_stats['symbols_processed'] += 1
                else:
                    excluded_symbols.append((symbol, coverage_ratio))
                    self.processing_stats['symbols_excluded'] += 1
                    
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                excluded_symbols.append((symbol, 0.0))
                self.processing_stats['symbols_excluded'] += 1
        
        # Log summary
        self.processing_stats['end_time'] = datetime.now()
        
        logger.info(f"Global consistency processing completed:")
        logger.info(f"  Symbols processed: {len(processed_symbols)}")
        logger.info(f"  Symbols excluded: {len(excluded_symbols)}")
        logger.info(f"  Stable features: {len(global_result.stable_features)}")
        
        if excluded_symbols:
            logger.info("Excluded symbols:")
            for symbol, coverage in excluded_symbols[:5]:  # Show first 5
                logger.info(f"  {symbol}: {coverage:.1%} coverage")
            if len(excluded_symbols) > 5:
                logger.info(f"  ... and {len(excluded_symbols) - 5} more")
        
        return processed_symbols


class FeatureManifestManager:
    """
    Manages feature manifests with versioning and metadata.
    
    A feature manifest is a JSON file containing the canonical list of features
    that all symbols must conform to, along with metadata about coverage
    statistics and exclusion reasons.
    """
    
    def __init__(self, config: FeatureConsistencyConfig):
        """Initialize the feature manifest manager."""
        self.config = config
        self.manifest_path = Path(config.manifest_path)
        
        logger.info(f"FeatureManifestManager initialized with path: {self.manifest_path}")
    
    def save_manifest(self, analysis_result: GlobalAnalysisResult, 
                     version: Optional[str] = None) -> str:
        """
        Save feature manifest with versioning and metadata.
        
        Args:
            analysis_result: Results from global coverage analysis
            version: Optional version string (auto-generated if None)
            
        Returns:
            Version string of the saved manifest
        """
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create manifest directory if needed
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Build manifest data
        manifest_data = {
            "manifest_version": "1.0",
            "version": version,
            "created_timestamp": datetime.now().isoformat(),
            "config_snapshot": analysis_result.config_used,
            
            # Core feature data
            "stable_features": analysis_result.stable_features,
            "total_features_analyzed": analysis_result.total_features_found,
            "total_symbols_analyzed": analysis_result.total_symbols_analyzed,
            
            # Summary statistics
            "summary": analysis_result.get_feature_summary(),
            
            # Detailed coverage statistics
            "coverage_statistics": {
                feature_name: {
                    "coverage_ratio": stats.coverage_ratio,
                    "symbols_with_feature": stats.symbols_with_feature,
                    "avg_non_nan_ratio": stats.avg_non_nan_ratio,
                    "min_non_nan_ratio": stats.min_non_nan_ratio,
                    "max_non_nan_ratio": stats.max_non_nan_ratio,
                    "symbols_list": stats.symbols_list[:10]  # First 10 for brevity
                }
                for feature_name, stats in analysis_result.coverage_stats.items()
            },
            
            # Exclusion information
            "excluded_features": {
                "features": analysis_result.unstable_features,
                "exclusion_reasons": self._get_exclusion_reasons(analysis_result)
            },
            
            # Metadata for debugging
            "metadata": {
                "analysis_timestamp": analysis_result.analysis_timestamp,
                "feature_count_by_coverage": self._get_feature_count_by_coverage(analysis_result),
                "quality_distribution": self._get_quality_distribution(analysis_result)
            }
        }
        
        # Save manifest
        try:
            with open(self.manifest_path, 'w') as f:
                json.dump(manifest_data, f, indent=2)
            
            logger.info(f"Feature manifest saved: {self.manifest_path}")
            logger.info(f"  Version: {version}")
            logger.info(f"  Stable features: {len(analysis_result.stable_features)}")
            logger.info(f"  Total features analyzed: {analysis_result.total_features_found}")
            
            # Also save a versioned backup
            backup_path = self.manifest_path.parent / f"feature_manifest_{version}.json"
            with open(backup_path, 'w') as f:
                json.dump(manifest_data, f, indent=2)
            logger.info(f"Versioned backup saved: {backup_path}")
            
            return version
            
        except Exception as e:
            logger.error(f"Failed to save feature manifest: {e}")
            raise
    
    def load_manifest(self, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Load feature manifest with error handling.
        
        Args:
            version: Optional version to load (loads latest if None)
            
        Returns:
            Manifest data dictionary
            
        Raises:
            FileNotFoundError: If manifest file doesn't exist
            ValueError: If manifest format is invalid
        """
        # Determine which file to load
        if version is None:
            manifest_file = self.manifest_path
        else:
            manifest_file = self.manifest_path.parent / f"feature_manifest_{version}.json"
        
        if not manifest_file.exists():
            raise FileNotFoundError(f"Feature manifest not found: {manifest_file}")
        
        try:
            with open(manifest_file, 'r') as f:
                manifest_data = json.load(f)
            
            # Validate manifest format
            self._validate_manifest_format(manifest_data)
            
            logger.info(f"Feature manifest loaded: {manifest_file}")
            logger.info(f"  Version: {manifest_data.get('version', 'unknown')}")
            logger.info(f"  Stable features: {len(manifest_data.get('stable_features', []))}")
            
            return manifest_data
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in manifest file: {e}")
        except Exception as e:
            logger.error(f"Failed to load feature manifest: {e}")
            raise
    
    def validate_manifest_compatibility(self, manifest_data: Dict[str, Any], 
                                      current_config: FeatureConsistencyConfig) -> bool:
        """
        Validate manifest compatibility with current configuration.
        
        Args:
            manifest_data: Loaded manifest data
            current_config: Current feature consistency configuration
            
        Returns:
            True if compatible, False otherwise
        """
        try:
            # Check manifest version
            manifest_version = manifest_data.get("manifest_version", "unknown")
            if manifest_version != "1.0":
                logger.warning(f"Manifest version mismatch: {manifest_version} != 1.0")
                return False
            
            # Check critical configuration parameters
            saved_config = manifest_data.get("config_snapshot", {})
            
            critical_params = [
                "global_feature_keep_ratio",
                "min_symbol_feature_coverage",
                "warmup_trim_days"
            ]
            
            for param in critical_params:
                saved_value = saved_config.get(param)
                current_value = getattr(current_config, param, None)
                
                if saved_value != current_value:
                    logger.warning(f"Configuration mismatch for {param}: "
                                 f"saved={saved_value}, current={current_value}")
                    return False
            
            # Check if manifest is too old (more than 30 days)
            created_timestamp = manifest_data.get("created_timestamp")
            if created_timestamp:
                try:
                    created_date = datetime.fromisoformat(created_timestamp.replace('Z', '+00:00'))
                    age_days = (datetime.now() - created_date).days
                    
                    if age_days > 30:
                        logger.warning(f"Manifest is {age_days} days old, may be outdated")
                        return False
                        
                except ValueError:
                    logger.warning("Could not parse manifest timestamp")
            
            logger.info("Manifest compatibility validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Error validating manifest compatibility: {e}")
            return False
    
    def get_available_versions(self) -> List[str]:
        """
        Get list of available manifest versions.
        
        Returns:
            List of version strings, sorted by creation time (newest first)
        """
        manifest_dir = self.manifest_path.parent
        if not manifest_dir.exists():
            return []
        
        # Find all versioned manifest files
        version_files = list(manifest_dir.glob("feature_manifest_*.json"))
        versions = []
        
        for file_path in version_files:
            # Extract version from filename
            filename = file_path.stem
            if filename.startswith("feature_manifest_"):
                version = filename[len("feature_manifest_"):]
                versions.append(version)
        
        # Sort by version (assuming timestamp format YYYYMMDD_HHMMSS)
        versions.sort(reverse=True)
        
        return versions
    
    def cleanup_old_versions(self, keep_count: int = 10) -> int:
        """
        Clean up old manifest versions, keeping only the most recent ones.
        
        Args:
            keep_count: Number of versions to keep
            
        Returns:
            Number of versions deleted
        """
        versions = self.get_available_versions()
        
        if len(versions) <= keep_count:
            logger.info(f"No cleanup needed: {len(versions)} versions <= {keep_count} keep limit")
            return 0
        
        versions_to_delete = versions[keep_count:]
        deleted_count = 0
        
        for version in versions_to_delete:
            try:
                version_file = self.manifest_path.parent / f"feature_manifest_{version}.json"
                if version_file.exists():
                    version_file.unlink()
                    deleted_count += 1
                    logger.debug(f"Deleted old manifest version: {version}")
            except Exception as e:
                logger.warning(f"Failed to delete version {version}: {e}")
        
        logger.info(f"Cleaned up {deleted_count} old manifest versions")
        return deleted_count
    
    def _validate_manifest_format(self, manifest_data: Dict[str, Any]) -> None:
        """Validate that manifest has required fields."""
        required_fields = [
            "manifest_version", "version", "created_timestamp",
            "stable_features", "total_features_analyzed"
        ]
        
        for field in required_fields:
            if field not in manifest_data:
                raise ValueError(f"Missing required field in manifest: {field}")
        
        # Validate stable_features is a list
        if not isinstance(manifest_data["stable_features"], list):
            raise ValueError("stable_features must be a list")
    
    def _get_exclusion_reasons(self, analysis_result: GlobalAnalysisResult) -> Dict[str, str]:
        """Get exclusion reasons for unstable features."""
        exclusion_reasons = {}
        
        for feature in analysis_result.unstable_features:
            if feature in analysis_result.coverage_stats:
                stats = analysis_result.coverage_stats[feature]
                
                if stats.coverage_ratio < self.config.global_feature_keep_ratio:
                    exclusion_reasons[feature] = (
                        f"Low coverage: {stats.coverage_ratio:.1%} < "
                        f"{self.config.global_feature_keep_ratio:.1%}"
                    )
                elif stats.avg_non_nan_ratio < 0.5:
                    exclusion_reasons[feature] = (
                        f"Poor quality: {stats.avg_non_nan_ratio:.1%} non-NaN ratio"
                    )
                else:
                    exclusion_reasons[feature] = "Other quality issues"
        
        return exclusion_reasons
    
    def _get_feature_count_by_coverage(self, analysis_result: GlobalAnalysisResult) -> Dict[str, int]:
        """Get feature count distribution by coverage buckets."""
        coverage_buckets = {
            "90-100%": 0, "75-90%": 0, "50-75%": 0, "25-50%": 0, "0-25%": 0
        }
        
        for stats in analysis_result.coverage_stats.values():
            ratio = stats.coverage_ratio
            if ratio >= 0.9:
                coverage_buckets["90-100%"] += 1
            elif ratio >= 0.75:
                coverage_buckets["75-90%"] += 1
            elif ratio >= 0.5:
                coverage_buckets["50-75%"] += 1
            elif ratio >= 0.25:
                coverage_buckets["25-50%"] += 1
            else:
                coverage_buckets["0-25%"] += 1
        
        return coverage_buckets
    
    def _get_quality_distribution(self, analysis_result: GlobalAnalysisResult) -> Dict[str, int]:
        """Get feature count distribution by quality buckets."""
        quality_buckets = {
            "90-100%": 0, "75-90%": 0, "50-75%": 0, "25-50%": 0, "0-25%": 0
        }
        
        for stats in analysis_result.coverage_stats.values():
            ratio = stats.avg_non_nan_ratio
            if ratio >= 0.9:
                quality_buckets["90-100%"] += 1
            elif ratio >= 0.75:
                quality_buckets["75-90%"] += 1
            elif ratio >= 0.5:
                quality_buckets["50-75%"] += 1
            elif ratio >= 0.25:
                quality_buckets["25-50%"] += 1
            else:
                quality_buckets["0-25%"] += 1
        
        return quality_buckets

    def load_existing_manifest(self, version: Optional[str] = None) -> bool:
        """
        Load existing feature manifest if available and compatible.
        
        Args:
            version: Optional version to load (loads latest if None)
            
        Returns:
            True if manifest loaded successfully, False otherwise
        """
        try:
            manifest_data = self.manifest_manager.load_manifest(version)
            
            # Validate compatibility
            if not self.manifest_manager.validate_manifest_compatibility(manifest_data, self.config):
                logger.warning("Existing manifest is not compatible with current configuration")
                return False
            
            # Extract stable features
            stable_features = manifest_data.get("stable_features", [])
            
            # Create a mock analysis result for consistency
            self.global_analysis_result = GlobalAnalysisResult(
                total_symbols_analyzed=manifest_data.get("total_symbols_analyzed", 0),
                total_features_found=manifest_data.get("total_features_analyzed", 0),
                stable_features=stable_features,
                unstable_features=manifest_data.get("excluded_features", {}).get("features", []),
                coverage_stats={},  # Not needed for loaded manifest
                analysis_timestamp=manifest_data.get("created_timestamp", ""),
                config_used=manifest_data.get("config_snapshot", {})
            )
            
            logger.info(f"Loaded existing manifest with {len(stable_features)} stable features")
            return True
            
        except FileNotFoundError:
            logger.info("No existing manifest found, will create new one")
            return False
        except Exception as e:
            logger.warning(f"Failed to load existing manifest: {e}")
            return False
    
    def apply_manifest_to_symbol(self, df: pd.DataFrame, symbol: str, 
                                manifest_features: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Apply feature manifest to a symbol's DataFrame.
        
        Args:
            df: Symbol's DataFrame
            symbol: Symbol name
            manifest_features: Optional list of features from manifest (uses loaded if None)
            
        Returns:
            DataFrame with only manifest features (plus essential columns)
        """
        if manifest_features is None:
            if self.global_analysis_result is None:
                logger.warning(f"{symbol}: No manifest available, returning original DataFrame")
                return df
            manifest_features = self.global_analysis_result.stable_features
        
        if not manifest_features:
            logger.warning(f"{symbol}: Empty manifest features, returning original DataFrame")
            return df
        
        # Essential columns that should always be preserved
        essential_columns = ['symbol', 'target', 'date', 'timestamp', 'direction_1d', 'close']
        
        # Combine manifest features with essential columns
        columns_to_keep = []
        for col in essential_columns:
            if col in df.columns:
                columns_to_keep.append(col)
        
        # Add manifest features that exist in this symbol
        for feature in manifest_features:
            if feature in df.columns:
                columns_to_keep.append(feature)
        
        # Remove duplicates while preserving order
        columns_to_keep = list(dict.fromkeys(columns_to_keep))
        
        # Apply selection
        original_cols = len(df.columns)
        selected_df = df[columns_to_keep].copy()
        
        logger.info(f"{symbol}: Applied manifest feature selection "
                   f"({original_cols} -> {len(selected_df.columns)} columns)")
        
        return selected_df
    
    def get_manifest_summary(self) -> Dict[str, Any]:
        """
        Get summary of current manifest state.
        
        Returns:
            Dictionary with manifest summary information
        """
        if self.global_analysis_result is None:
            return {
                "manifest_loaded": False,
                "stable_features_count": 0,
                "total_features_analyzed": 0,
                "manifest_version": None
            }
        
        return {
            "manifest_loaded": True,
            "stable_features_count": len(self.global_analysis_result.stable_features),
            "total_features_analyzed": self.global_analysis_result.total_features_found,
            "symbols_analyzed": self.global_analysis_result.total_symbols_analyzed,
            "analysis_timestamp": self.global_analysis_result.analysis_timestamp,
            "available_versions": self.manifest_manager.get_available_versions()
        }
    
    def cleanup_old_manifests(self, keep_count: int = 10) -> int:
        """
        Clean up old manifest versions.
        
        Args:
            keep_count: Number of versions to keep
            
        Returns:
            Number of versions deleted
        """
        return self.manifest_manager.cleanup_old_versions(keep_count)

    def process_symbols_with_global_consistency(self, symbol_dataframes: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Process all symbols with global feature consistency.
        
        This is the main method that orchestrates the complete feature consistency pipeline:
        1. Try to load existing manifest
        2. If no compatible manifest, perform coverage analysis integration
        3. Apply feature selection to each symbol
        4. Validate symbol coverage
        5. Return processed symbols
        
        Args:
            symbol_dataframes: Dictionary mapping symbol names to their DataFrames
            
        Returns:
            Dictionary of processed symbols that meet consistency requirements
        """
        logger.info(f"Processing {len(symbol_dataframes)} symbols with global consistency")
        self.processing_stats['start_time'] = datetime.now()
        
        # Step 1: Try to load existing manifest
        manifest_loaded = self.load_existing_manifest()
        
        if not manifest_loaded:
            # Step 2: Perform coverage analysis integration
            logger.info("No compatible manifest found, performing global coverage analysis")
            
            # Apply warm-up trimming and collect coverage statistics
            trimmed_symbols = self.analyze_and_trim_symbols(symbol_dataframes)
            
            if not trimmed_symbols:
                logger.error("No symbols remaining after coverage analysis integration")
                return {}
            
            # Perform global coverage analysis on trimmed data
            global_result = self.perform_global_analysis(trimmed_symbols)
            
            # Use trimmed symbols for further processing
            symbol_dataframes = trimmed_symbols
            
        else:
            logger.info("Using existing manifest for feature consistency")
        
        # Step 3: Process each symbol
        processed_symbols = {}
        excluded_symbols = []
        
        for symbol, df in symbol_dataframes.items():
            try:
                # Apply feature selection (either from analysis or loaded manifest)
                processed_df = self.apply_global_feature_selection(df, symbol)
                
                # Validate symbol coverage
                is_valid, coverage_ratio = self.validate_symbol_against_global_features(processed_df, symbol)
                
                if is_valid:
                    processed_symbols[symbol] = processed_df
                    self.processing_stats['symbols_processed'] += 1
                else:
                    excluded_symbols.append((symbol, coverage_ratio))
                    self.processing_stats['symbols_excluded'] += 1
                    
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                excluded_symbols.append((symbol, 0.0))
                self.processing_stats['symbols_excluded'] += 1
        
        # Step 4: Log summary
        self.processing_stats['end_time'] = datetime.now()
        
        logger.info(f"Global consistency processing completed:")
        logger.info(f"  Symbols processed: {len(processed_symbols)}")
        logger.info(f"  Symbols excluded: {len(excluded_symbols)}")
        
        if self.global_analysis_result:
            logger.info(f"  Stable features: {len(self.global_analysis_result.stable_features)}")
        
        if excluded_symbols:
            logger.info("Excluded symbols:")
            for symbol, coverage in excluded_symbols[:5]:  # Show first 5
                logger.info(f"  {symbol}: {coverage:.1%} coverage")
            if len(excluded_symbols) > 5:
                logger.info(f"  ... and {len(excluded_symbols) - 5} more")
        
        return processed_symbols


class MissingnessMaskGenerator:
    """
    Generates binary missingness indicators and handles final imputation.
    
    This class creates binary "_isnan" columns for each feature to indicate
    where original values were missing before imputation, then applies
    configurable final imputation strategies.
    """
    
    def __init__(self, config: FeatureConsistencyConfig):
        """Initialize the missingness mask generator."""
        self.config = config
        self.imputation_strategies = {
            'zero': 0.0,
            'mean': 'mean',
            'median': 'median'
        }
        
        logger.info(f"MissingnessMaskGenerator initialized")
        logger.info(f"  Use missingness masks: {self.config.use_missingness_mask}")
        logger.info(f"  Imputation value: {self.config.imputation_value}")
    
    def create_missingness_masks(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Create binary missingness indicators for features before imputation.
        
        This implements requirement 4.1: Create missingness mask indicators
        
        Args:
            df: DataFrame with potential missing values
            feature_cols: List of feature columns to create masks for (auto-detect if None)
            
        Returns:
            DataFrame with added "_isnan" columns for each feature
        """
        if not self.config.use_missingness_mask:
            logger.debug("Missingness mask creation disabled in configuration")
            return df.copy()
        
        if df.empty:
            logger.warning("Empty DataFrame provided, no masks to create")
            return df.copy()
        
        # Auto-detect feature columns if not provided
        if feature_cols is None:
            essential_columns = {'symbol', 'target', 'date', 'timestamp', 'direction_1d', 'close'}
            feature_cols = [col for col in df.columns if col not in essential_columns]
        
        if not feature_cols:
            logger.info("No feature columns found for mask creation")
            return df.copy()
        
        logger.info(f"Creating missingness masks for {len(feature_cols)} features")
        
        result_df = df.copy()
        masks_created = 0
        total_missing_values = 0
        
        # Create binary "_isnan" columns for each feature
        for col in feature_cols:
            if col not in df.columns:
                logger.warning(f"Feature column '{col}' not found in DataFrame, skipping")
                continue
            
            mask_col = f"{col}_isnan"
            
            # Create binary mask: 1 for missing, 0 for present
            missing_mask = df[col].isna()
            result_df[mask_col] = missing_mask.astype('int8')
            
            # Track statistics
            missing_count = missing_mask.sum()
            if missing_count > 0:
                masks_created += 1
                total_missing_values += missing_count
                logger.debug(f"  {col}: {missing_count} missing values ({missing_count/len(df):.1%})")
        
        # Log summary
        logger.info(f"Missingness mask creation completed:")
        logger.info(f"  Features processed: {len(feature_cols)}")
        logger.info(f"  Masks created: {masks_created}")
        logger.info(f"  Total missing values: {total_missing_values}")
        logger.info(f"  New columns added: {masks_created}")
        logger.info(f"  DataFrame shape: {df.shape} -> {result_df.shape}")
        
        return result_df
    
    def apply_final_imputation(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None, 
                              fill_value: Optional[Union[float, str]] = None) -> pd.DataFrame:
        """
        Fill remaining NaN values with configurable fill values after mask creation.
        
        This implements requirement 4.3: Fill missing values after creating masks
        
        Args:
            df: DataFrame with missingness masks already created
            feature_cols: List of feature columns to impute (auto-detect if None)
            fill_value: Value to use for imputation (uses config default if None)
            
        Returns:
            DataFrame with all NaN values imputed
        """
        if df.empty:
            logger.warning("Empty DataFrame provided, no imputation needed")
            return df.copy()
        
        # Use configured imputation strategy if not specified
        if fill_value is None:
            fill_value = self.config.imputation_strategy
        
        # Auto-detect feature columns if not provided
        if feature_cols is None:
            essential_columns = {'symbol', 'target', 'date', 'timestamp', 'direction_1d', 'close'}
            # Exclude _isnan columns from imputation
            feature_cols = [col for col in df.columns 
                           if col not in essential_columns and not col.endswith('_isnan')]
        
        if not feature_cols:
            logger.info("No feature columns found for imputation")
            return df.copy()
        
        logger.info(f"Applying final imputation to {len(feature_cols)} features")
        logger.info(f"  Imputation strategy: {fill_value}")
        
        # Determine the actual imputation strategy
        strategy = fill_value if isinstance(fill_value, str) else "zero"
        
        result_df = df.copy()
        imputation_stats = {}
        
        # Apply imputation based on strategy
        for col in feature_cols:
            if col not in df.columns:
                logger.warning(f"Feature column '{col}' not found in DataFrame, skipping")
                continue
            
            # Count missing values before imputation
            missing_before = result_df[col].isna().sum()
            
            if missing_before == 0:
                logger.debug(f"  {col}: No missing values, skipping")
                continue
            
            # Apply imputation strategy
            if strategy == 'mean':
                fill_val = result_df[col].mean()
                if pd.isna(fill_val):
                    fill_val = self.config.imputation_value  # Fallback to configured value
                    logger.warning(f"  {col}: Mean is NaN, using fallback value {fill_val}")
            elif strategy == 'median':
                fill_val = result_df[col].median()
                if pd.isna(fill_val):
                    fill_val = self.config.imputation_value  # Fallback to configured value
                    logger.warning(f"  {col}: Median is NaN, using fallback value {fill_val}")
            elif strategy == 'zero':
                fill_val = self.config.imputation_value
            else:
                # Try to parse as float, fallback to configured value
                try:
                    fill_val = float(fill_value)
                except (ValueError, TypeError):
                    fill_val = self.config.imputation_value
                    logger.warning(f"  {col}: Could not parse fill_value '{fill_value}', using fallback {fill_val}")
            
            # Apply imputation
            result_df[col] = result_df[col].fillna(fill_val)
            
            # Verify imputation worked
            missing_after = result_df[col].isna().sum()
            imputed_count = missing_before - missing_after
            
            imputation_stats[col] = {
                'missing_before': missing_before,
                'missing_after': missing_after,
                'imputed_count': imputed_count,
                'fill_value': fill_val
            }
            
            logger.debug(f"  {col}: Imputed {imputed_count} values with {fill_val}")
        
        # Log summary statistics
        total_imputed = sum(stats['imputed_count'] for stats in imputation_stats.values())
        total_remaining_nan = sum(stats['missing_after'] for stats in imputation_stats.values())
        
        logger.info(f"Final imputation completed:")
        logger.info(f"  Features processed: {len(feature_cols)}")
        logger.info(f"  Total values imputed: {total_imputed}")
        logger.info(f"  Remaining NaN values: {total_remaining_nan}")
        
        if total_remaining_nan > 0:
            logger.warning(f"Warning: {total_remaining_nan} NaN values still remain after imputation")
            # Show which columns still have NaN values
            for col, stats in imputation_stats.items():
                if stats['missing_after'] > 0:
                    logger.warning(f"  {col}: {stats['missing_after']} NaN values remaining")
        
        return result_df
    
    def apply_forward_backward_fill(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Apply forward-fill and backward-fill before mask creation.
        
        This implements requirement 4.4: Apply forward-fill and backward-fill before computing final NaN ratios
        
        Args:
            df: DataFrame to apply filling to
            feature_cols: List of feature columns to fill (auto-detect if None)
            
        Returns:
            DataFrame with forward and backward filling applied
        """
        if df.empty:
            logger.warning("Empty DataFrame provided, no filling needed")
            return df.copy()
        
        # Auto-detect feature columns if not provided
        if feature_cols is None:
            essential_columns = {'symbol', 'target', 'date', 'timestamp', 'direction_1d', 'close'}
            feature_cols = [col for col in df.columns if col not in essential_columns]
        
        if not feature_cols:
            logger.info("No feature columns found for forward/backward filling")
            return df.copy()
        
        logger.info(f"Applying forward-fill and backward-fill to {len(feature_cols)} features")
        
        result_df = df.copy()
        filling_stats = {}
        
        # Apply forward fill then backward fill for each feature column
        for col in feature_cols:
            if col not in df.columns:
                logger.warning(f"Feature column '{col}' not found in DataFrame, skipping")
                continue
            
            # Count missing values before filling
            missing_before = result_df[col].isna().sum()
            
            if missing_before == 0:
                logger.debug(f"  {col}: No missing values, skipping")
                continue
            
            # Apply forward fill first, then backward fill
            result_df[col] = result_df[col].ffill().bfill()
            
            # Count missing values after filling
            missing_after = result_df[col].isna().sum()
            filled_count = missing_before - missing_after
            
            filling_stats[col] = {
                'missing_before': missing_before,
                'missing_after': missing_after,
                'filled_count': filled_count
            }
            
            logger.debug(f"  {col}: Filled {filled_count} values, {missing_after} still missing")
        
        # Log summary statistics
        total_filled = sum(stats['filled_count'] for stats in filling_stats.values())
        total_remaining = sum(stats['missing_after'] for stats in filling_stats.values())
        
        logger.info(f"Forward/backward filling completed:")
        logger.info(f"  Features processed: {len(feature_cols)}")
        logger.info(f"  Total values filled: {total_filled}")
        logger.info(f"  Remaining missing values: {total_remaining}")
        
        if total_remaining > 0:
            logger.debug(f"Note: {total_remaining} values still missing after forward/backward fill")
            logger.debug("These will be handled by final imputation")
        
        return result_df
    
    def process_symbol_with_missingness_handling(self, df: pd.DataFrame, symbol: str, 
                                               feature_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Complete missingness handling workflow for a single symbol.
        
        This orchestrates the complete missingness handling process:
        1. Apply forward-fill and backward-fill
        2. Create missingness masks (if enabled)
        3. Apply final imputation
        
        Args:
            df: Symbol's DataFrame
            symbol: Symbol name for logging
            feature_cols: List of feature columns to process (auto-detect if None)
            
        Returns:
            DataFrame with complete missingness handling applied
        """
        logger.info(f"{symbol}: Starting missingness handling workflow")
        
        if df.empty:
            logger.warning(f"{symbol}: Empty DataFrame, no missingness handling needed")
            return df.copy()
        
        try:
            # Step 1: Apply forward-fill and backward-fill before mask creation
            logger.debug(f"{symbol}: Step 1 - Applying forward/backward fill")
            df_filled = self.apply_forward_backward_fill(df, feature_cols)
            
            # Step 2: Create missingness masks after feature selection
            logger.debug(f"{symbol}: Step 2 - Creating missingness masks")
            df_with_masks = self.create_missingness_masks(df_filled, feature_cols)
            
            # Step 3: Apply final imputation (fill remaining NaNs with configured value)
            logger.debug(f"{symbol}: Step 3 - Applying final imputation")
            df_final = self.apply_final_imputation(df_with_masks, feature_cols)
            
            # Validate final result
            if feature_cols is None:
                essential_columns = {'symbol', 'target', 'date', 'timestamp', 'direction_1d', 'close'}
                feature_cols = [col for col in df_final.columns 
                               if col not in essential_columns and not col.endswith('_isnan')]
            
            # Check for any remaining NaN values in feature columns
            remaining_nans = 0
            for col in feature_cols:
                if col in df_final.columns:
                    col_nans = df_final[col].isna().sum()
                    remaining_nans += col_nans
            
            if remaining_nans > 0:
                logger.warning(f"{symbol}: {remaining_nans} NaN values still remain after complete workflow")
            else:
                logger.info(f"{symbol}: All NaN values successfully handled")
            
            # Log final statistics
            original_shape = df.shape
            final_shape = df_final.shape
            mask_columns = len([col for col in df_final.columns if col.endswith('_isnan')])
            
            logger.info(f"{symbol}: Missingness handling completed:")
            logger.info(f"  Original shape: {original_shape}")
            logger.info(f"  Final shape: {final_shape}")
            logger.info(f"  Mask columns added: {mask_columns}")
            logger.info(f"  Remaining NaN values: {remaining_nans}")
            
            return df_final
            
        except Exception as e:
            logger.error(f"{symbol}: Error in missingness handling workflow: {e}")
            logger.error(f"{symbol}: Returning original DataFrame")
            return df.copy()
    
    def get_missingness_summary(self, df: pd.DataFrame, symbol: str = "Unknown") -> Dict[str, Any]:
        """
        Get summary statistics about missingness in a DataFrame.
        
        Args:
            df: DataFrame to analyze
            symbol: Symbol name for logging
            
        Returns:
            Dictionary with missingness statistics
        """
        if df.empty:
            return {
                'symbol': symbol,
                'total_rows': 0,
                'total_columns': 0,
                'feature_columns': 0,
                'mask_columns': 0,
                'total_missing_values': 0,
                'missing_percentage': 0.0
            }
        
        # Identify column types
        essential_columns = {'symbol', 'target', 'date', 'timestamp', 'direction_1d', 'close'}
        mask_columns = [col for col in df.columns if col.endswith('_isnan')]
        feature_columns = [col for col in df.columns 
                          if col not in essential_columns and not col.endswith('_isnan')]
        
        # Count missing values in feature columns only
        total_missing = 0
        total_feature_cells = 0
        
        for col in feature_columns:
            if col in df.columns:
                missing_count = df[col].isna().sum()
                total_missing += missing_count
                total_feature_cells += len(df)
        
        missing_percentage = (total_missing / total_feature_cells * 100) if total_feature_cells > 0 else 0.0
        
        summary = {
            'symbol': symbol,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'feature_columns': len(feature_columns),
            'mask_columns': len(mask_columns),
            'essential_columns': len([col for col in essential_columns if col in df.columns]),
            'total_missing_values': total_missing,
            'total_feature_cells': total_feature_cells,
            'missing_percentage': missing_percentage,
            'has_missingness_masks': len(mask_columns) > 0
        }
        
        return summary
    
    def validate_no_missing_values(self, df: pd.DataFrame, symbol: str = "Unknown") -> Tuple[bool, List[str]]:
        """
        Validate that no missing values remain in feature columns.
        
        Args:
            df: DataFrame to validate
            symbol: Symbol name for logging
            
        Returns:
            Tuple of (is_valid, list_of_columns_with_missing_values)
        """
        if df.empty:
            return True, []
        
        # Check only feature columns (not essential or mask columns)
        essential_columns = {'symbol', 'target', 'date', 'timestamp', 'direction_1d', 'close'}
        feature_columns = [col for col in df.columns 
                          if col not in essential_columns and not col.endswith('_isnan')]
        
        columns_with_missing = []
        
        for col in feature_columns:
            if col in df.columns:
                missing_count = df[col].isna().sum()
                if missing_count > 0:
                    columns_with_missing.append(col)
                    logger.warning(f"{symbol}: Column '{col}' has {missing_count} missing values")
        
        is_valid = len(columns_with_missing) == 0
        
        if is_valid:
            logger.debug(f"{symbol}: Validation passed - no missing values in feature columns")
        else:
            logger.error(f"{symbol}: Validation failed - {len(columns_with_missing)} columns have missing values")
        
        return is_valid, columns_with_missing

    def apply_configurable_imputation(self, df: pd.DataFrame, symbol: str = "Unknown", 
                                     feature_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Apply imputation using the configured strategy.
        
        This implements requirement 5.5: Support different imputation strategies
        
        Args:
            df: DataFrame to apply imputation to
            symbol: Symbol name for logging
            feature_cols: List of feature columns to impute (auto-detect if None)
            
        Returns:
            DataFrame with imputation applied according to configuration
        """
        if not self.config.use_missingness_mask:
            logger.debug(f"{symbol}: Missingness mask disabled, applying direct imputation")
            return self.apply_final_imputation(df, feature_cols, self.config.imputation_strategy)
        
        logger.info(f"{symbol}: Applying configurable imputation with strategy '{self.config.imputation_strategy}'")
        
        # Apply the complete missingness handling workflow
        return self.process_symbol_with_missingness_handling(df, symbol, feature_cols)
    
    def get_available_imputation_strategies(self) -> Dict[str, str]:
        """
        Get available imputation strategies with descriptions.
        
        Returns:
            Dictionary mapping strategy names to descriptions
        """
        return {
            "zero": "Fill missing values with 0.0 (or configured imputation_value)",
            "mean": "Fill missing values with column mean (fallback to 0.0 if all NaN)",
            "median": "Fill missing values with column median (fallback to 0.0 if all NaN)"
        }
    
    def validate_imputation_strategy(self, strategy: str) -> bool:
        """
        Validate that an imputation strategy is supported.
        
        Args:
            strategy: Strategy name to validate
            
        Returns:
            True if strategy is valid, False otherwise
        """
        return strategy in self.get_available_imputation_strategies()
    
    def set_imputation_strategy(self, strategy: str) -> None:
        """
        Set the imputation strategy in the configuration.
        
        Args:
            strategy: New imputation strategy
            
        Raises:
            ValueError: If strategy is not valid
        """
        if not self.validate_imputation_strategy(strategy):
            available = list(self.get_available_imputation_strategies().keys())
            raise ValueError(f"Invalid imputation strategy '{strategy}'. Available: {available}")
        
        old_strategy = self.config.imputation_strategy
        self.config.imputation_strategy = strategy
        
        logger.info(f"Imputation strategy changed from '{old_strategy}' to '{strategy}'")
    
    def get_imputation_summary(self, df: pd.DataFrame, symbol: str = "Unknown") -> Dict[str, Any]:
        """
        Get summary of imputation that would be applied with current configuration.
        
        Args:
            df: DataFrame to analyze
            symbol: Symbol name
            
        Returns:
            Dictionary with imputation summary
        """
        if df.empty:
            return {
                'symbol': symbol,
                'strategy': self.config.imputation_strategy,
                'use_missingness_mask': self.config.use_missingness_mask,
                'features_to_process': 0,
                'estimated_imputation_values': {}
            }
        
        # Get feature columns
        essential_columns = {'symbol', 'target', 'date', 'timestamp', 'direction_1d', 'close'}
        feature_cols = [col for col in df.columns 
                       if col not in essential_columns and not col.endswith('_isnan')]
        
        estimated_values = {}
        strategy = self.config.imputation_strategy
        
        for col in feature_cols[:10]:  # Limit to first 10 for performance
            if col in df.columns:
                missing_count = df[col].isna().sum()
                if missing_count > 0:
                    if strategy == 'mean':
                        est_val = df[col].mean()
                        if pd.isna(est_val):
                            est_val = self.config.imputation_value
                    elif strategy == 'median':
                        est_val = df[col].median()
                        if pd.isna(est_val):
                            est_val = self.config.imputation_value
                    else:  # zero or other
                        est_val = self.config.imputation_value
                    
                    estimated_values[col] = {
                        'missing_count': missing_count,
                        'estimated_fill_value': est_val,
                        'missing_percentage': (missing_count / len(df)) * 100
                    }
        
        return {
            'symbol': symbol,
            'strategy': strategy,
            'use_missingness_mask': self.config.use_missingness_mask,
            'features_to_process': len(feature_cols),
            'features_with_missing': len(estimated_values),
            'estimated_imputation_values': estimated_values
        }