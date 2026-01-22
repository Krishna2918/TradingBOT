"""
Integration module for Feature Consistency Monitoring

This module provides integration between the monitoring system and the
existing feature consistency implementation.
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from .feature_consistency_monitor import FeatureConsistencyMonitor, setup_feature_consistency_monitoring

logger = logging.getLogger(__name__)


class MonitoredFeatureConsistencyManager:
    """
    Wrapper for FeatureConsistencyManager that adds monitoring capabilities.
    
    This class wraps the existing FeatureConsistencyManager and automatically
    collects metrics and triggers alerts during processing.
    """
    
    def __init__(self, feature_consistency_manager, monitoring_config: Optional[Dict[str, Any]] = None):
        """
        Initialize monitored feature consistency manager.
        
        Args:
            feature_consistency_manager: Instance of FeatureConsistencyManager
            monitoring_config: Configuration for monitoring system
        """
        self.manager = feature_consistency_manager
        self.monitor = setup_feature_consistency_monitoring(monitoring_config)
        self.processing_active = False
        
        # Store previous manifest for drift detection
        self.previous_manifest_features: Optional[List[str]] = None
        
        logger.info("MonitoredFeatureConsistencyManager initialized")
    
    def __getattr__(self, name):
        """Delegate attribute access to the wrapped manager."""
        return getattr(self.manager, name)
    
    def process_symbols_with_monitoring(self, symbol_data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process symbols with full monitoring and alerting.
        
        Args:
            symbol_data_dict: Dictionary of symbol data
            
        Returns:
            Processing results with monitoring data
        """
        # Start monitoring
        self.monitor.start_processing_monitoring()
        self.processing_active = True
        
        try:
            # Load previous manifest for drift detection
            self._load_previous_manifest()
            
            # Process symbols using the wrapped manager
            results = self._process_with_monitoring(symbol_data_dict)
            
            # End monitoring and get final metrics
            final_metrics = self.monitor.end_processing_monitoring()
            self.processing_active = False
            
            # Add monitoring data to results
            results['monitoring_metrics'] = final_metrics
            results['monitoring_dashboard_data'] = final_metrics.to_dashboard_format()
            
            return results
            
        except Exception as e:
            logger.error(f"Error during monitored processing: {e}")
            if self.processing_active:
                self.monitor.end_processing_monitoring()
                self.processing_active = False
            raise
    
    def _load_previous_manifest(self):
        """Load previous manifest for drift detection."""
        try:
            manifest_path = Path(self.manager.config.manifest_path)
            if manifest_path.exists():
                manifest_data = self.manager.manifest_manager.load_manifest()
                if manifest_data and 'features' in manifest_data:
                    self.previous_manifest_features = manifest_data['features']
                    logger.info(f"Loaded previous manifest with {len(self.previous_manifest_features)} features")
        except Exception as e:
            logger.warning(f"Could not load previous manifest for drift detection: {e}")
            self.previous_manifest_features = None
    
    def _process_with_monitoring(self, symbol_data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Process symbols with monitoring integration."""
        
        # Step 1: Global coverage analysis with monitoring
        logger.info("Starting global coverage analysis...")
        global_analysis_result = self.manager.analyze_global_coverage(symbol_data_dict)
        
        # Record feature analysis metrics
        total_features = len(global_analysis_result.coverage_stats)
        stable_features = global_analysis_result.stable_features
        unstable_features = global_analysis_result.excluded_features
        
        self.monitor.record_feature_analysis(total_features, stable_features, unstable_features)
        
        # Check for feature drift
        if self.previous_manifest_features:
            self.monitor.record_feature_drift(self.previous_manifest_features, stable_features)
        
        # Step 2: Process individual symbols with monitoring
        logger.info("Processing individual symbols...")
        processed_symbols = {}
        total_rows_before = 0
        total_rows_after = 0
        nan_ratios_before = []
        nan_ratios_after = []
        
        for symbol, symbol_df in symbol_data_dict.items():
            try:
                symbol_start_time = logger.time() if hasattr(logger, 'time') else 0
                
                # Process symbol
                result = self.manager.process_single_symbol(symbol, symbol_df, global_analysis_result)
                
                symbol_processing_time = (logger.time() - symbol_start_time) if hasattr(logger, 'time') else 0
                
                # Calculate metrics for this symbol
                rows_before = len(symbol_df)
                rows_after = len(result.processed_data) if result.processed_data is not None else 0
                coverage = result.coverage_ratio
                included = result.processing_status == 'success'
                
                # Record symbol processing metrics
                self.monitor.record_symbol_processing(
                    symbol, included, coverage, rows_before, rows_after, symbol_processing_time
                )
                
                # Accumulate data quality metrics
                total_rows_before += rows_before
                total_rows_after += rows_after
                
                # Calculate NaN ratios (simplified)
                if symbol_df is not None and len(symbol_df) > 0:
                    nan_ratio_before = symbol_df.isnull().sum().sum() / (len(symbol_df) * len(symbol_df.columns))
                    nan_ratios_before.append(nan_ratio_before)
                
                if result.processed_data is not None and len(result.processed_data) > 0:
                    nan_ratio_after = result.processed_data.isnull().sum().sum() / (len(result.processed_data) * len(result.processed_data.columns))
                    nan_ratios_after.append(nan_ratio_after)
                
                processed_symbols[symbol] = result
                
                # Log processing result
                status_msg = "included" if included else f"excluded ({result.error_message})"
                logger.info(f"Symbol {symbol}: {status_msg}, coverage={coverage:.1%}, "
                           f"rows={rows_before}→{rows_after}")
                
            except Exception as e:
                logger.error(f"Error processing symbol {symbol}: {e}")
                # Record failed symbol
                self.monitor.record_symbol_processing(symbol, False, 0.0, len(symbol_df), 0, 0)
                continue
        
        # Record overall data quality metrics
        avg_nan_before = sum(nan_ratios_before) / len(nan_ratios_before) if nan_ratios_before else 0
        avg_nan_after = sum(nan_ratios_after) / len(nan_ratios_after) if nan_ratios_after else 0
        self.monitor.record_data_quality_metrics(avg_nan_before, avg_nan_after)
        
        # Step 3: Validate tensor shapes
        logger.info("Validating tensor shapes...")
        processed_data_dict = {symbol: result.processed_data 
                              for symbol, result in processed_symbols.items() 
                              if result.processed_data is not None}
        
        shape_validation_result = self.manager.validate_tensor_shapes(processed_data_dict)
        
        # Prepare results
        results = {
            'global_analysis_result': global_analysis_result,
            'processed_symbols': processed_symbols,
            'shape_validation_result': shape_validation_result,
            'processing_summary': {
                'total_symbols': len(symbol_data_dict),
                'symbols_included': len([r for r in processed_symbols.values() if r.processing_status == 'success']),
                'symbols_excluded': len([r for r in processed_symbols.values() if r.processing_status != 'success']),
                'stable_features_count': len(stable_features),
                'unstable_features_count': len(unstable_features),
                'total_rows_before': total_rows_before,
                'total_rows_after': total_rows_after
            }
        }
        
        return results
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status."""
        return self.monitor.get_current_status()
    
    def add_custom_alert_rule(self, rule_name: str, metric_path: str, threshold: float, 
                             comparison: str = "gt", severity: str = "warning"):
        """Add a custom alert rule."""
        from .feature_consistency_monitor import AlertRule
        
        rule = AlertRule(
            name=rule_name,
            description=f"Custom alert for {metric_path}",
            metric_path=metric_path,
            threshold=threshold,
            comparison=comparison,
            severity=severity
        )
        
        self.monitor.add_custom_alert_rule(rule)
        logger.info(f"Added custom alert rule: {rule_name}")
    
    def disable_alert(self, rule_name: str) -> bool:
        """Disable an alert rule."""
        return self.monitor.disable_alert_rule(rule_name)
    
    def enable_alert(self, rule_name: str) -> bool:
        """Enable an alert rule."""
        return self.monitor.enable_alert_rule(rule_name)


def create_monitored_feature_consistency_manager(feature_consistency_config, 
                                                monitoring_config: Optional[Dict[str, Any]] = None):
    """
    Create a monitored feature consistency manager.
    
    Args:
        feature_consistency_config: Configuration for FeatureConsistencyManager
        monitoring_config: Configuration for monitoring system
        
    Returns:
        MonitoredFeatureConsistencyManager instance
    """
    # Import here to avoid circular imports
    from ..data.feature_consistency import FeatureConsistencyManager
    
    # Create the base manager
    base_manager = FeatureConsistencyManager(feature_consistency_config)
    
    # Wrap with monitoring
    monitored_manager = MonitoredFeatureConsistencyManager(base_manager, monitoring_config)
    
    return monitored_manager


def setup_monitoring_for_existing_manager(manager, monitoring_config: Optional[Dict[str, Any]] = None):
    """
    Add monitoring to an existing FeatureConsistencyManager.
    
    Args:
        manager: Existing FeatureConsistencyManager instance
        monitoring_config: Configuration for monitoring system
        
    Returns:
        MonitoredFeatureConsistencyManager instance
    """
    return MonitoredFeatureConsistencyManager(manager, monitoring_config)


# Enhanced logging functions for dashboard-ready output
def log_symbol_processing(symbol: str, included: bool, coverage: float, 
                         feature_count: int, processing_time: float):
    """Log symbol processing with structured data for dashboard."""
    extra_data = {
        'symbol': symbol,
        'included': included,
        'coverage': coverage,
        'feature_count': feature_count,
        'processing_time': processing_time
    }
    
    if included:
        logger.info(f"Symbol {symbol} included in training", extra=extra_data)
    else:
        logger.warning(f"Symbol {symbol} excluded from training", extra=extra_data)


def log_feature_analysis(total_features: int, stable_count: int, unstable_count: int):
    """Log feature analysis results with structured data."""
    extra_data = {
        'total_features': total_features,
        'stable_features': stable_count,
        'unstable_features': unstable_count,
        'stability_rate': stable_count / total_features if total_features > 0 else 0
    }
    
    logger.info(f"Feature analysis complete: {stable_count}/{total_features} stable features", 
                extra=extra_data)


def log_processing_summary(symbols_processed: int, symbols_included: int, 
                          processing_time: float, feature_count: int):
    """Log processing summary with structured data."""
    exclusion_rate = (symbols_processed - symbols_included) / symbols_processed if symbols_processed > 0 else 0
    
    extra_data = {
        'symbols_processed': symbols_processed,
        'symbols_included': symbols_included,
        'symbols_excluded': symbols_processed - symbols_included,
        'exclusion_rate': exclusion_rate,
        'processing_time': processing_time,
        'feature_count': feature_count
    }
    
    logger.info(f"Processing complete: {symbols_included}/{symbols_processed} symbols included, "
                f"{feature_count} features, {processing_time:.1f}s", extra=extra_data)


if __name__ == "__main__":
    # Example usage
    from ..data.simple_config_loader import load_config_for_feature_consistency_manager
    
    # Load configuration
    config = load_config_for_feature_consistency_manager('development')
    
    # Create monitored manager
    monitoring_config = {
        'enable_dashboard_logging': True,
        'dashboard_output_path': 'test_monitoring/dashboard.json',
        'metrics_output_path': 'test_monitoring/metrics.json',
        'alert_output_path': 'test_monitoring/alerts.json'
    }
    
    monitored_manager = create_monitored_feature_consistency_manager(config, monitoring_config)
    
    print("Monitored Feature Consistency Manager created successfully!")
    print(f"Monitoring status: {monitored_manager.get_monitoring_status()}")