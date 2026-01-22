"""
Feature Consistency Monitoring and Alerting System

This module provides comprehensive monitoring, metrics collection, and alerting
capabilities for the feature consistency system.
"""

import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import pandas as pd
import numpy as np
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

@dataclass
class FeatureConsistencyMetrics:
    """Metrics collected during feature consistency processing."""
    
    # Processing metrics
    total_symbols_processed: int = 0
    symbols_included: int = 0
    symbols_excluded: int = 0
    exclusion_rate: float = 0.0
    
    # Feature metrics
    total_features_analyzed: int = 0
    stable_features_count: int = 0
    unstable_features_count: int = 0
    feature_stability_rate: float = 0.0
    
    # Coverage metrics
    avg_symbol_coverage: float = 0.0
    min_symbol_coverage: float = 0.0
    max_symbol_coverage: float = 0.0
    coverage_std: float = 0.0
    
    # Processing time metrics
    processing_start_time: Optional[str] = None
    processing_end_time: Optional[str] = None
    total_processing_time_seconds: float = 0.0
    avg_time_per_symbol: float = 0.0
    
    # Data quality metrics
    total_rows_before_trimming: int = 0
    total_rows_after_trimming: int = 0
    rows_trimmed_ratio: float = 0.0
    avg_nan_ratio_before: float = 0.0
    avg_nan_ratio_after: float = 0.0
    
    # Drift detection metrics
    feature_drift_detected: bool = False
    features_added: List[str] = None
    features_removed: List[str] = None
    feature_count_change: int = 0
    
    # Alert metrics
    alerts_triggered: List[str] = None
    
    def __post_init__(self):
        """Initialize list fields if None."""
        if self.features_added is None:
            self.features_added = []
        if self.features_removed is None:
            self.features_removed = []
        if self.alerts_triggered is None:
            self.alerts_triggered = []
    
    def calculate_derived_metrics(self):
        """Calculate derived metrics from base metrics."""
        if self.total_symbols_processed > 0:
            self.exclusion_rate = self.symbols_excluded / self.total_symbols_processed
            
        if self.total_features_analyzed > 0:
            self.feature_stability_rate = self.stable_features_count / self.total_features_analyzed
            
        if self.total_rows_before_trimming > 0:
            self.rows_trimmed_ratio = (self.total_rows_before_trimming - self.total_rows_after_trimming) / self.total_rows_before_trimming
            
        if self.symbols_included > 0 and self.total_processing_time_seconds > 0:
            self.avg_time_per_symbol = self.total_processing_time_seconds / self.symbols_included
    
    def to_dashboard_format(self) -> Dict[str, Any]:
        """Convert metrics to dashboard-friendly format."""
        return {
            'timestamp': datetime.now().isoformat(),
            'processing_summary': {
                'symbols_processed': self.total_symbols_processed,
                'symbols_included': self.symbols_included,
                'symbols_excluded': self.symbols_excluded,
                'exclusion_rate_pct': round(self.exclusion_rate * 100, 2),
                'processing_time_minutes': round(self.total_processing_time_seconds / 60, 2)
            },
            'feature_summary': {
                'total_features': self.total_features_analyzed,
                'stable_features': self.stable_features_count,
                'unstable_features': self.unstable_features_count,
                'stability_rate_pct': round(self.feature_stability_rate * 100, 2)
            },
            'coverage_summary': {
                'avg_coverage_pct': round(self.avg_symbol_coverage * 100, 2),
                'min_coverage_pct': round(self.min_symbol_coverage * 100, 2),
                'max_coverage_pct': round(self.max_symbol_coverage * 100, 2),
                'coverage_std_pct': round(self.coverage_std * 100, 2)
            },
            'data_quality': {
                'rows_trimmed_pct': round(self.rows_trimmed_ratio * 100, 2),
                'nan_ratio_before_pct': round(self.avg_nan_ratio_before * 100, 2),
                'nan_ratio_after_pct': round(self.avg_nan_ratio_after * 100, 2)
            },
            'drift_detection': {
                'drift_detected': self.feature_drift_detected,
                'features_added': len(self.features_added),
                'features_removed': len(self.features_removed),
                'net_feature_change': self.feature_count_change
            },
            'alerts': {
                'total_alerts': len(self.alerts_triggered),
                'alert_types': self.alerts_triggered
            }
        }


@dataclass
class AlertRule:
    """Configuration for an alert rule."""
    name: str
    description: str
    metric_path: str  # dot-separated path to metric (e.g., "exclusion_rate")
    threshold: float
    comparison: str  # "gt", "lt", "eq", "gte", "lte"
    severity: str  # "info", "warning", "error", "critical"
    enabled: bool = True
    cooldown_minutes: int = 60  # Minimum time between alerts
    
    def evaluate(self, metrics: FeatureConsistencyMetrics) -> bool:
        """Evaluate if this alert rule should trigger."""
        if not self.enabled:
            return False
            
        # Get metric value using dot notation
        value = self._get_nested_value(metrics, self.metric_path)
        if value is None:
            return False
        
        # Evaluate comparison
        if self.comparison == "gt":
            return value > self.threshold
        elif self.comparison == "lt":
            return value < self.threshold
        elif self.comparison == "gte":
            return value >= self.threshold
        elif self.comparison == "lte":
            return value <= self.threshold
        elif self.comparison == "eq":
            return value == self.threshold
        else:
            logger.warning(f"Unknown comparison operator: {self.comparison}")
            return False
    
    def _get_nested_value(self, obj: Any, path: str) -> Any:
        """Get nested attribute value using dot notation."""
        try:
            for attr in path.split('.'):
                obj = getattr(obj, attr)
            return obj
        except AttributeError:
            return None


class FeatureConsistencyMonitor:
    """
    Monitoring and alerting system for feature consistency processing.
    
    Collects metrics, detects anomalies, and triggers alerts based on
    configurable rules.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the monitoring system."""
        self.config = config or {}
        self.metrics_history: deque = deque(maxlen=100)  # Keep last 100 runs
        self.alert_history: deque = deque(maxlen=1000)   # Keep last 1000 alerts
        self.last_alert_times: Dict[str, datetime] = {}
        
        # Initialize alert rules
        self.alert_rules = self._initialize_default_alert_rules()
        
        # Metrics collection state
        self.current_metrics: Optional[FeatureConsistencyMetrics] = None
        self.processing_start_time: Optional[datetime] = None
        
        # Dashboard output configuration
        self.dashboard_output_path = self.config.get('dashboard_output_path', 'monitoring/feature_consistency_dashboard.json')
        self.metrics_output_path = self.config.get('metrics_output_path', 'monitoring/feature_consistency_metrics.json')
        
        logger.info("FeatureConsistencyMonitor initialized")
    
    def _initialize_default_alert_rules(self) -> List[AlertRule]:
        """Initialize default alert rules."""
        return [
            AlertRule(
                name="high_exclusion_rate",
                description="High symbol exclusion rate detected",
                metric_path="exclusion_rate",
                threshold=0.20,  # 20%
                comparison="gt",
                severity="warning",
                cooldown_minutes=30
            ),
            AlertRule(
                name="critical_exclusion_rate", 
                description="Critical symbol exclusion rate detected",
                metric_path="exclusion_rate",
                threshold=0.50,  # 50%
                comparison="gt",
                severity="critical",
                cooldown_minutes=15
            ),
            AlertRule(
                name="low_feature_stability",
                description="Low feature stability rate detected",
                metric_path="feature_stability_rate",
                threshold=0.80,  # 80%
                comparison="lt",
                severity="warning",
                cooldown_minutes=60
            ),
            AlertRule(
                name="feature_drift_detected",
                description="Feature drift detected in processing",
                metric_path="feature_drift_detected",
                threshold=1,  # True
                comparison="eq",
                severity="warning",
                cooldown_minutes=120
            ),
            AlertRule(
                name="low_average_coverage",
                description="Low average symbol coverage detected",
                metric_path="avg_symbol_coverage",
                threshold=0.85,  # 85%
                comparison="lt",
                severity="info",
                cooldown_minutes=60
            ),
            AlertRule(
                name="processing_time_exceeded",
                description="Processing time exceeded expected duration",
                metric_path="total_processing_time_seconds",
                threshold=1800,  # 30 minutes
                comparison="gt",
                severity="warning",
                cooldown_minutes=60
            )
        ]
    
    def start_processing_monitoring(self) -> None:
        """Start monitoring a new processing run."""
        self.processing_start_time = datetime.now()
        self.current_metrics = FeatureConsistencyMetrics()
        self.current_metrics.processing_start_time = self.processing_start_time.isoformat()
        
        logger.info("Started feature consistency processing monitoring")
    
    def end_processing_monitoring(self) -> FeatureConsistencyMetrics:
        """End monitoring and finalize metrics."""
        if not self.current_metrics or not self.processing_start_time:
            logger.warning("No active monitoring session to end")
            return FeatureConsistencyMetrics()
        
        end_time = datetime.now()
        self.current_metrics.processing_end_time = end_time.isoformat()
        self.current_metrics.total_processing_time_seconds = (end_time - self.processing_start_time).total_seconds()
        
        # Calculate derived metrics
        self.current_metrics.calculate_derived_metrics()
        
        # Add to history
        self.metrics_history.append(self.current_metrics)
        
        # Check alerts
        self._check_alerts(self.current_metrics)
        
        # Output dashboard data
        self._output_dashboard_data(self.current_metrics)
        
        # Save metrics
        self._save_metrics(self.current_metrics)
        
        logger.info(f"Completed feature consistency processing monitoring in {self.current_metrics.total_processing_time_seconds:.2f} seconds")
        
        return self.current_metrics
    
    def record_symbol_processing(self, symbol: str, included: bool, coverage: float, 
                                rows_before: int, rows_after: int, processing_time: float) -> None:
        """Record metrics for a single symbol processing."""
        if not self.current_metrics:
            logger.warning("No active monitoring session")
            return
        
        self.current_metrics.total_symbols_processed += 1
        
        if included:
            self.current_metrics.symbols_included += 1
        else:
            self.current_metrics.symbols_excluded += 1
        
        # Update coverage statistics
        if self.current_metrics.symbols_included == 1:
            self.current_metrics.min_symbol_coverage = coverage
            self.current_metrics.max_symbol_coverage = coverage
            self.current_metrics.avg_symbol_coverage = coverage
        else:
            self.current_metrics.min_symbol_coverage = min(self.current_metrics.min_symbol_coverage, coverage)
            self.current_metrics.max_symbol_coverage = max(self.current_metrics.max_symbol_coverage, coverage)
            
            # Update running average
            n = self.current_metrics.symbols_included
            self.current_metrics.avg_symbol_coverage = ((n - 1) * self.current_metrics.avg_symbol_coverage + coverage) / n
        
        # Update row statistics
        self.current_metrics.total_rows_before_trimming += rows_before
        self.current_metrics.total_rows_after_trimming += rows_after
    
    def record_feature_analysis(self, total_features: int, stable_features: List[str], 
                               unstable_features: List[str]) -> None:
        """Record feature analysis metrics."""
        if not self.current_metrics:
            logger.warning("No active monitoring session")
            return
        
        self.current_metrics.total_features_analyzed = total_features
        self.current_metrics.stable_features_count = len(stable_features)
        self.current_metrics.unstable_features_count = len(unstable_features)
    
    def record_feature_drift(self, previous_features: List[str], current_features: List[str]) -> None:
        """Record feature drift detection results."""
        if not self.current_metrics:
            logger.warning("No active monitoring session")
            return
        
        previous_set = set(previous_features)
        current_set = set(current_features)
        
        added_features = list(current_set - previous_set)
        removed_features = list(previous_set - current_set)
        
        if added_features or removed_features:
            self.current_metrics.feature_drift_detected = True
            self.current_metrics.features_added = added_features
            self.current_metrics.features_removed = removed_features
            self.current_metrics.feature_count_change = len(current_features) - len(previous_features)
            
            logger.warning(f"Feature drift detected: +{len(added_features)} features, -{len(removed_features)} features")
    
    def record_data_quality_metrics(self, avg_nan_before: float, avg_nan_after: float) -> None:
        """Record data quality metrics."""
        if not self.current_metrics:
            logger.warning("No active monitoring session")
            return
        
        self.current_metrics.avg_nan_ratio_before = avg_nan_before
        self.current_metrics.avg_nan_ratio_after = avg_nan_after
    
    def _check_alerts(self, metrics: FeatureConsistencyMetrics) -> None:
        """Check all alert rules against current metrics."""
        current_time = datetime.now()
        
        for rule in self.alert_rules:
            if rule.evaluate(metrics):
                # Check cooldown
                last_alert_time = self.last_alert_times.get(rule.name)
                if last_alert_time:
                    time_since_last = current_time - last_alert_time
                    if time_since_last.total_seconds() < rule.cooldown_minutes * 60:
                        continue  # Still in cooldown
                
                # Trigger alert
                self._trigger_alert(rule, metrics)
                self.last_alert_times[rule.name] = current_time
    
    def _trigger_alert(self, rule: AlertRule, metrics: FeatureConsistencyMetrics) -> None:
        """Trigger an alert."""
        alert_data = {
            'timestamp': datetime.now().isoformat(),
            'rule_name': rule.name,
            'description': rule.description,
            'severity': rule.severity,
            'metric_path': rule.metric_path,
            'threshold': rule.threshold,
            'actual_value': rule._get_nested_value(metrics, rule.metric_path),
            'comparison': rule.comparison
        }
        
        self.alert_history.append(alert_data)
        metrics.alerts_triggered.append(rule.name)
        
        # Log alert
        log_level = {
            'info': logging.INFO,
            'warning': logging.WARNING,
            'error': logging.ERROR,
            'critical': logging.CRITICAL
        }.get(rule.severity, logging.WARNING)
        
        logger.log(log_level, f"ALERT [{rule.severity.upper()}]: {rule.description} - "
                             f"{rule.metric_path}={alert_data['actual_value']} {rule.comparison} {rule.threshold}")
        
        # Output alert to dashboard format
        self._output_alert(alert_data)
    
    def _output_alert(self, alert_data: Dict[str, Any]) -> None:
        """Output alert in dashboard-ready format."""
        alert_output_path = Path(self.config.get('alert_output_path', 'monitoring/alerts.json'))
        alert_output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing alerts
        alerts = []
        if alert_output_path.exists():
            try:
                with open(alert_output_path, 'r') as f:
                    alerts = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load existing alerts: {e}")
        
        # Add new alert
        alerts.append(alert_data)
        
        # Keep only recent alerts (last 100)
        alerts = alerts[-100:]
        
        # Save alerts
        try:
            with open(alert_output_path, 'w') as f:
                json.dump(alerts, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save alert data: {e}")
    
    def _output_dashboard_data(self, metrics: FeatureConsistencyMetrics) -> None:
        """Output metrics in dashboard-ready format."""
        dashboard_path = Path(self.dashboard_output_path)
        dashboard_path.parent.mkdir(parents=True, exist_ok=True)
        
        dashboard_data = {
            'last_updated': datetime.now().isoformat(),
            'current_run': metrics.to_dashboard_format(),
            'recent_alerts': list(self.alert_history)[-10:],  # Last 10 alerts
            'historical_trends': self._get_historical_trends()
        }
        
        try:
            with open(dashboard_path, 'w') as f:
                json.dump(dashboard_data, f, indent=2)
            logger.info(f"Dashboard data saved to {dashboard_path}")
        except Exception as e:
            logger.error(f"Could not save dashboard data: {e}")
    
    def _save_metrics(self, metrics: FeatureConsistencyMetrics) -> None:
        """Save detailed metrics to file."""
        metrics_path = Path(self.metrics_output_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing metrics
        all_metrics = []
        if metrics_path.exists():
            try:
                with open(metrics_path, 'r') as f:
                    all_metrics = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load existing metrics: {e}")
        
        # Add current metrics
        all_metrics.append(asdict(metrics))
        
        # Keep only recent metrics (last 100 runs)
        all_metrics = all_metrics[-100:]
        
        try:
            with open(metrics_path, 'w') as f:
                json.dump(all_metrics, f, indent=2)
            logger.info(f"Metrics saved to {metrics_path}")
        except Exception as e:
            logger.error(f"Could not save metrics: {e}")
    
    def _get_historical_trends(self) -> Dict[str, Any]:
        """Get historical trends for dashboard."""
        if len(self.metrics_history) < 2:
            return {}
        
        recent_metrics = list(self.metrics_history)[-10:]  # Last 10 runs
        
        # Calculate trends
        exclusion_rates = [m.exclusion_rate for m in recent_metrics]
        stability_rates = [m.feature_stability_rate for m in recent_metrics]
        processing_times = [m.total_processing_time_seconds for m in recent_metrics]
        
        return {
            'exclusion_rate_trend': {
                'current': exclusion_rates[-1] if exclusion_rates else 0,
                'average': np.mean(exclusion_rates) if exclusion_rates else 0,
                'trend': 'increasing' if len(exclusion_rates) > 1 and exclusion_rates[-1] > exclusion_rates[-2] else 'stable'
            },
            'stability_rate_trend': {
                'current': stability_rates[-1] if stability_rates else 0,
                'average': np.mean(stability_rates) if stability_rates else 0,
                'trend': 'increasing' if len(stability_rates) > 1 and stability_rates[-1] > stability_rates[-2] else 'stable'
            },
            'processing_time_trend': {
                'current': processing_times[-1] if processing_times else 0,
                'average': np.mean(processing_times) if processing_times else 0,
                'trend': 'increasing' if len(processing_times) > 1 and processing_times[-1] > processing_times[-2] else 'stable'
            }
        }
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current monitoring status."""
        return {
            'monitoring_active': self.current_metrics is not None,
            'total_runs_monitored': len(self.metrics_history),
            'total_alerts_triggered': len(self.alert_history),
            'active_alert_rules': len([r for r in self.alert_rules if r.enabled]),
            'last_run_metrics': self.metrics_history[-1].to_dashboard_format() if self.metrics_history else None
        }
    
    def add_custom_alert_rule(self, rule: AlertRule) -> None:
        """Add a custom alert rule."""
        self.alert_rules.append(rule)
        logger.info(f"Added custom alert rule: {rule.name}")
    
    def disable_alert_rule(self, rule_name: str) -> bool:
        """Disable an alert rule by name."""
        for rule in self.alert_rules:
            if rule.name == rule_name:
                rule.enabled = False
                logger.info(f"Disabled alert rule: {rule_name}")
                return True
        return False
    
    def enable_alert_rule(self, rule_name: str) -> bool:
        """Enable an alert rule by name."""
        for rule in self.alert_rules:
            if rule.name == rule_name:
                rule.enabled = True
                logger.info(f"Enabled alert rule: {rule_name}")
                return True
        return False


def create_dashboard_ready_log_formatter() -> logging.Formatter:
    """Create a log formatter that outputs dashboard-ready JSON logs."""
    
    class DashboardJSONFormatter(logging.Formatter):
        """Custom formatter for dashboard-ready JSON logs."""
        
        def format(self, record):
            log_data = {
                'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                'level': record.levelname,
                'logger': record.name,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno
            }
            
            # Add extra fields if present
            if hasattr(record, 'symbol'):
                log_data['symbol'] = record.symbol
            if hasattr(record, 'feature_count'):
                log_data['feature_count'] = record.feature_count
            if hasattr(record, 'coverage'):
                log_data['coverage'] = record.coverage
            if hasattr(record, 'processing_time'):
                log_data['processing_time'] = record.processing_time
            
            return json.dumps(log_data)
    
    return DashboardJSONFormatter()


# Convenience function for easy integration
def setup_feature_consistency_monitoring(config: Optional[Dict[str, Any]] = None) -> FeatureConsistencyMonitor:
    """
    Set up feature consistency monitoring with default configuration.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured FeatureConsistencyMonitor instance
    """
    monitor = FeatureConsistencyMonitor(config)
    
    # Set up dashboard-ready logging if requested
    if config and config.get('enable_dashboard_logging', False):
        log_path = Path(config.get('dashboard_log_path', 'monitoring/feature_consistency.log'))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        dashboard_handler = logging.FileHandler(str(log_path))
        dashboard_handler.setFormatter(create_dashboard_ready_log_formatter())
        
        # Add handler to feature consistency logger
        fc_logger = logging.getLogger('ai.data.feature_consistency')
        fc_logger.addHandler(dashboard_handler)
        fc_logger.setLevel(logging.INFO)
    
    return monitor


if __name__ == "__main__":
    # Example usage and testing
    monitor = setup_feature_consistency_monitoring({
        'enable_dashboard_logging': True,
        'dashboard_output_path': 'test_monitoring/dashboard.json',
        'metrics_output_path': 'test_monitoring/metrics.json'
    })
    
    # Simulate a processing run
    monitor.start_processing_monitoring()
    
    # Simulate symbol processing
    monitor.record_symbol_processing('AAPL', True, 0.95, 5000, 4800, 2.5)
    monitor.record_symbol_processing('GOOGL', True, 0.92, 4800, 4600, 2.1)
    monitor.record_symbol_processing('TSLA', False, 0.75, 3000, 2800, 1.8)  # Excluded
    
    # Simulate feature analysis
    monitor.record_feature_analysis(100, ['feature_' + str(i) for i in range(85)], 
                                   ['unstable_' + str(i) for i in range(15)])
    
    # Simulate data quality metrics
    monitor.record_data_quality_metrics(0.12, 0.03)
    
    # End monitoring
    final_metrics = monitor.end_processing_monitoring()
    
    print("Monitoring test completed!")
    print(f"Final metrics: {final_metrics.to_dashboard_format()}")