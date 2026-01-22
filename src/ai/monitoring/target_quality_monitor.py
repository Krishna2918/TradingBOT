"""
Production Monitoring for Target Quality.

This module provides:
- Target distribution monitoring and alerting
- Dashboard-ready logging output for target quality metrics
- Guardrails for automatic reversion if target quality degrades
"""

import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import warnings
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TargetQualityMetrics:
    """Metrics for target quality monitoring."""
    timestamp: str
    symbol: str
    total_samples: int
    down_count: int
    flat_count: int
    up_count: int
    down_percentage: float
    flat_percentage: float
    up_percentage: float
    neutral_band_used: float
    class_balance_score: float  # 1.0 = perfectly balanced, 0.0 = completely imbalanced
    missing_classes: List[str]
    quality_score: float  # Overall quality score 0-100
    alerts: List[str]


@dataclass
class QualityThresholds:
    """Thresholds for target quality monitoring."""
    min_flat_percentage: float = 15.0
    max_flat_percentage: float = 50.0
    min_samples: int = 100
    min_quality_score: float = 70.0
    max_class_imbalance: float = 0.8  # Max percentage for any single class
    alert_on_missing_classes: bool = True


class TargetQualityMonitor:
    """Production monitoring system for target quality."""
    
    def __init__(self, 
                 log_file: Optional[str] = None,
                 dashboard_output: Optional[str] = None,
                 thresholds: Optional[QualityThresholds] = None):
        """
        Initialize target quality monitor.
        
        Args:
            log_file: Path to log file for monitoring data
            dashboard_output: Path to dashboard JSON output
            thresholds: Quality thresholds for alerting
        """
        self.log_file = log_file
        self.dashboard_output = dashboard_output
        self.thresholds = thresholds or QualityThresholds()
        self.metrics_history: List[TargetQualityMetrics] = []
        
        # Setup logging
        if self.log_file:
            self._setup_file_logging()
    
    def monitor_target_creation(self, 
                              df: pd.DataFrame,
                              symbol: str,
                              neutral_band: float,
                              target_col: str = 'direction_1d') -> TargetQualityMetrics:
        """
        Monitor target creation and generate quality metrics.
        
        Args:
            df: DataFrame with created targets
            symbol: Symbol name
            neutral_band: Neutral band used for target creation
            target_col: Name of target column
            
        Returns:
            TargetQualityMetrics object
        """
        timestamp = datetime.now().isoformat()
        
        if target_col not in df.columns:
            logger.error(f"Target column '{target_col}' not found for {symbol}")
            return self._create_error_metrics(symbol, timestamp, f"Missing {target_col} column")
        
        # Extract targets and remove NaN values
        targets = df[target_col].dropna()
        
        if len(targets) == 0:
            logger.error(f"No valid targets found for {symbol}")
            return self._create_error_metrics(symbol, timestamp, "No valid targets")
        
        # Calculate distribution
        unique_vals, counts = np.unique(targets.values, return_counts=True)
        total_samples = len(targets)
        
        # Initialize counts
        down_count = flat_count = up_count = 0
        
        for val, count in zip(unique_vals, counts):
            if val == -1:
                down_count = count
            elif val == 0:
                flat_count = count
            elif val == 1:
                up_count = count
        
        # Calculate percentages
        down_pct = (down_count / total_samples) * 100
        flat_pct = (flat_count / total_samples) * 100
        up_pct = (up_count / total_samples) * 100
        
        # Calculate class balance score
        percentages = np.array([down_pct, flat_pct, up_pct]) / 100
        ideal_balance = 1.0 / 3  # Ideal would be 33.33% each
        balance_deviation = np.sum(np.abs(percentages - ideal_balance))
        class_balance_score = max(0.0, 1.0 - balance_deviation)
        
        # Check for missing classes
        expected_classes = {-1: "DOWN", 0: "FLAT", 1: "UP"}
        present_classes = set(unique_vals)
        missing_classes = []
        
        for class_val, class_name in expected_classes.items():
            if class_val not in present_classes:
                missing_classes.append(class_name)
        
        # Calculate overall quality score
        quality_score = self._calculate_quality_score(
            total_samples, flat_pct, class_balance_score, missing_classes
        )
        
        # Generate alerts
        alerts = self._generate_alerts(
            symbol, total_samples, down_pct, flat_pct, up_pct, 
            missing_classes, quality_score
        )
        
        # Create metrics object
        metrics = TargetQualityMetrics(
            timestamp=timestamp,
            symbol=symbol,
            total_samples=total_samples,
            down_count=down_count,
            flat_count=flat_count,
            up_count=up_count,
            down_percentage=down_pct,
            flat_percentage=flat_pct,
            up_percentage=up_pct,
            neutral_band_used=neutral_band,
            class_balance_score=class_balance_score,
            missing_classes=missing_classes,
            quality_score=quality_score,
            alerts=alerts
        )
        
        # Store metrics
        self.metrics_history.append(metrics)
        
        # Log metrics
        self._log_metrics(metrics)
        
        # Output to dashboard if configured
        if self.dashboard_output:
            self._output_to_dashboard(metrics)
        
        return metrics
    
    def monitor_global_targets(self, 
                             df: pd.DataFrame,
                             symbols: List[str],
                             neutral_band: float,
                             target_col: str = 'direction_1d') -> TargetQualityMetrics:
        """
        Monitor global target quality across all symbols.
        
        Args:
            df: Combined DataFrame with all symbols
            symbols: List of symbol names
            neutral_band: Neutral band used
            target_col: Name of target column
            
        Returns:
            TargetQualityMetrics for global dataset
        """
        return self.monitor_target_creation(
            df, f"GLOBAL_{len(symbols)}_symbols", neutral_band, target_col
        )
    
    def check_quality_degradation(self, 
                                current_metrics: TargetQualityMetrics,
                                lookback_periods: int = 5) -> Dict[str, Any]:
        """
        Check for quality degradation compared to historical performance.
        
        Args:
            current_metrics: Current quality metrics
            lookback_periods: Number of historical periods to compare against
            
        Returns:
            Dictionary with degradation analysis
        """
        if len(self.metrics_history) < lookback_periods + 1:
            return {
                'degradation_detected': False,
                'reason': 'Insufficient historical data',
                'recommendation': 'Continue monitoring'
            }
        
        # Get historical metrics for same symbol
        symbol_history = [
            m for m in self.metrics_history[-lookback_periods-1:-1] 
            if m.symbol == current_metrics.symbol
        ]
        
        if len(symbol_history) < 2:
            return {
                'degradation_detected': False,
                'reason': 'Insufficient symbol history',
                'recommendation': 'Continue monitoring'
            }
        
        # Calculate historical averages
        hist_quality_scores = [m.quality_score for m in symbol_history]
        hist_flat_percentages = [m.flat_percentage for m in symbol_history]
        
        avg_historical_quality = np.mean(hist_quality_scores)
        avg_historical_flat = np.mean(hist_flat_percentages)
        
        # Check for degradation
        quality_drop = avg_historical_quality - current_metrics.quality_score
        flat_change = abs(avg_historical_flat - current_metrics.flat_percentage)
        
        degradation_detected = False
        reasons = []
        
        if quality_drop > 10.0:  # Quality dropped by more than 10 points
            degradation_detected = True
            reasons.append(f"Quality score dropped by {quality_drop:.1f} points")
        
        if flat_change > 15.0:  # FLAT percentage changed by more than 15%
            degradation_detected = True
            reasons.append(f"FLAT percentage changed by {flat_change:.1f}%")
        
        if current_metrics.quality_score < self.thresholds.min_quality_score:
            degradation_detected = True
            reasons.append(f"Quality score below threshold ({self.thresholds.min_quality_score})")
        
        recommendation = "Continue monitoring"
        if degradation_detected:
            recommendation = "Consider reverting to previous neutral band or investigating data quality"
        
        return {
            'degradation_detected': degradation_detected,
            'reasons': reasons,
            'current_quality': current_metrics.quality_score,
            'historical_average': avg_historical_quality,
            'quality_drop': quality_drop,
            'recommendation': recommendation
        }
    
    def get_dashboard_data(self, 
                          symbol: Optional[str] = None,
                          hours_back: int = 24) -> Dict[str, Any]:
        """
        Get dashboard-ready data for target quality monitoring.
        
        Args:
            symbol: Specific symbol to filter (None for all)
            hours_back: Hours of history to include
            
        Returns:
            Dashboard data dictionary
        """
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        # Filter metrics
        filtered_metrics = []
        for m in self.metrics_history:
            metric_time = datetime.fromisoformat(m.timestamp)
            if metric_time >= cutoff_time:
                if symbol is None or m.symbol == symbol:
                    filtered_metrics.append(m)
        
        if not filtered_metrics:
            return {
                'status': 'no_data',
                'message': f'No data available for the last {hours_back} hours'
            }
        
        # Aggregate data
        symbols = list(set(m.symbol for m in filtered_metrics))
        latest_metrics = {}
        
        for sym in symbols:
            sym_metrics = [m for m in filtered_metrics if m.symbol == sym]
            if sym_metrics:
                latest_metrics[sym] = sym_metrics[-1]  # Most recent
        
        # Calculate summary statistics
        all_quality_scores = [m.quality_score for m in filtered_metrics]
        all_flat_percentages = [m.flat_percentage for m in filtered_metrics]
        
        summary = {
            'total_symbols': len(symbols),
            'avg_quality_score': np.mean(all_quality_scores),
            'min_quality_score': np.min(all_quality_scores),
            'max_quality_score': np.max(all_quality_scores),
            'avg_flat_percentage': np.mean(all_flat_percentages),
            'total_alerts': sum(len(m.alerts) for m in filtered_metrics)
        }
        
        # Recent alerts
        recent_alerts = []
        for m in filtered_metrics[-10:]:  # Last 10 metrics
            for alert in m.alerts:
                recent_alerts.append({
                    'timestamp': m.timestamp,
                    'symbol': m.symbol,
                    'alert': alert
                })
        
        return {
            'status': 'success',
            'summary': summary,
            'latest_metrics': {k: asdict(v) for k, v in latest_metrics.items()},
            'recent_alerts': recent_alerts,
            'time_range': {
                'start': cutoff_time.isoformat(),
                'end': datetime.now().isoformat()
            }
        } 
   
    def _calculate_quality_score(self, 
                                total_samples: int,
                                flat_percentage: float,
                                class_balance_score: float,
                                missing_classes: List[str]) -> float:
        """Calculate overall quality score (0-100)."""
        score = 100.0
        
        # Penalize for insufficient samples
        if total_samples < self.thresholds.min_samples:
            sample_penalty = (1.0 - total_samples / self.thresholds.min_samples) * 30
            score -= sample_penalty
        
        # Penalize for poor FLAT percentage
        if flat_percentage < self.thresholds.min_flat_percentage:
            flat_penalty = (self.thresholds.min_flat_percentage - flat_percentage) * 0.5
            score -= flat_penalty
        elif flat_percentage > self.thresholds.max_flat_percentage:
            flat_penalty = (flat_percentage - self.thresholds.max_flat_percentage) * 0.3
            score -= flat_penalty
        
        # Penalize for class imbalance
        balance_penalty = (1.0 - class_balance_score) * 25
        score -= balance_penalty
        
        # Penalize for missing classes
        missing_penalty = len(missing_classes) * 15
        score -= missing_penalty
        
        return max(0.0, min(100.0, score))
    
    def _generate_alerts(self, 
                        symbol: str,
                        total_samples: int,
                        down_pct: float,
                        flat_pct: float,
                        up_pct: float,
                        missing_classes: List[str],
                        quality_score: float) -> List[str]:
        """Generate alerts based on quality metrics."""
        alerts = []
        
        # Sample count alerts
        if total_samples < self.thresholds.min_samples:
            alerts.append(f"LOW_SAMPLES: Only {total_samples} samples (min: {self.thresholds.min_samples})")
        
        # FLAT percentage alerts
        if flat_pct < self.thresholds.min_flat_percentage:
            alerts.append(f"FLAT_TOO_LOW: {flat_pct:.1f}% (min: {self.thresholds.min_flat_percentage}%)")
        elif flat_pct > self.thresholds.max_flat_percentage:
            alerts.append(f"FLAT_TOO_HIGH: {flat_pct:.1f}% (max: {self.thresholds.max_flat_percentage}%)")
        
        # Class imbalance alerts
        max_class_pct = max(down_pct, flat_pct, up_pct)
        if max_class_pct > self.thresholds.max_class_imbalance * 100:
            alerts.append(f"CLASS_IMBALANCE: Max class {max_class_pct:.1f}% (threshold: {self.thresholds.max_class_imbalance*100}%)")
        
        # Missing class alerts
        if missing_classes and self.thresholds.alert_on_missing_classes:
            alerts.append(f"MISSING_CLASSES: {', '.join(missing_classes)}")
        
        # Overall quality alerts
        if quality_score < self.thresholds.min_quality_score:
            alerts.append(f"LOW_QUALITY: Score {quality_score:.1f} (min: {self.thresholds.min_quality_score})")
        
        return alerts
    
    def _log_metrics(self, metrics: TargetQualityMetrics) -> None:
        """Log metrics in structured format."""
        # Standard logging
        logger.info(f"TARGET_QUALITY_METRICS: {metrics.symbol}")
        logger.info(f"  Samples: {metrics.total_samples:,}")
        logger.info(f"  Distribution: DOWN={metrics.down_percentage:.1f}% FLAT={metrics.flat_percentage:.1f}% UP={metrics.up_percentage:.1f}%")
        logger.info(f"  Quality Score: {metrics.quality_score:.1f}/100")
        logger.info(f"  Neutral Band: ±{metrics.neutral_band_used*100:.1f}%")
        
        if metrics.alerts:
            logger.warning(f"  ALERTS: {'; '.join(metrics.alerts)}")
        
        # Structured logging for monitoring systems
        structured_log = {
            'event_type': 'target_quality_metrics',
            'timestamp': metrics.timestamp,
            'symbol': metrics.symbol,
            'metrics': self._convert_to_json_serializable(asdict(metrics))
        }
        
        logger.info(f"STRUCTURED_METRICS: {json.dumps(structured_log)}")
    
    def _output_to_dashboard(self, metrics: TargetQualityMetrics) -> None:
        """Output metrics to dashboard file."""
        try:
            dashboard_data = {
                'last_updated': metrics.timestamp,
                'metrics': self._convert_to_json_serializable(asdict(metrics))
            }
            
            dashboard_path = Path(self.dashboard_output)
            dashboard_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(dashboard_path, 'w') as f:
                json.dump(dashboard_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to write dashboard output: {e}")
    
    def _create_error_metrics(self, 
                            symbol: str, 
                            timestamp: str, 
                            error_msg: str) -> TargetQualityMetrics:
        """Create error metrics object."""
        return TargetQualityMetrics(
            timestamp=timestamp,
            symbol=symbol,
            total_samples=0,
            down_count=0,
            flat_count=0,
            up_count=0,
            down_percentage=0.0,
            flat_percentage=0.0,
            up_percentage=0.0,
            neutral_band_used=0.0,
            class_balance_score=0.0,
            missing_classes=["DOWN", "FLAT", "UP"],
            quality_score=0.0,
            alerts=[f"ERROR: {error_msg}"]
        )
    
    def _convert_to_json_serializable(self, obj):
        """Convert numpy types to JSON serializable types."""
        if isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(v) for v in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def _setup_file_logging(self) -> None:
        """Setup file logging for monitoring data."""
        try:
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create file handler
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(logging.INFO)
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            
            # Add handler to logger
            logger.addHandler(file_handler)
            
        except Exception as e:
            logger.error(f"Failed to setup file logging: {e}")


class TargetQualityGuardrails:
    """Guardrails for automatic reversion if target quality degrades."""
    
    def __init__(self, 
                 monitor: TargetQualityMonitor,
                 reversion_threshold: float = 60.0,
                 consecutive_failures: int = 3):
        """
        Initialize guardrails.
        
        Args:
            monitor: TargetQualityMonitor instance
            reversion_threshold: Quality score threshold for reversion
            consecutive_failures: Number of consecutive failures before reversion
        """
        self.monitor = monitor
        self.reversion_threshold = reversion_threshold
        self.consecutive_failures = consecutive_failures
        self.failure_count = 0
        self.last_good_config = None
    
    def check_and_revert(self, 
                        current_metrics: TargetQualityMetrics,
                        current_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check quality and recommend reversion if needed.
        
        Args:
            current_metrics: Current quality metrics
            current_config: Current configuration (neutral_band, etc.)
            
        Returns:
            Dictionary with reversion recommendation
        """
        result = {
            'should_revert': False,
            'reason': '',
            'recommended_config': None,
            'current_quality': current_metrics.quality_score
        }
        
        # Check if quality is below threshold
        if current_metrics.quality_score < self.reversion_threshold:
            self.failure_count += 1
            logger.warning(f"Quality failure {self.failure_count}/{self.consecutive_failures} for {current_metrics.symbol}")
            
            if self.failure_count >= self.consecutive_failures:
                result['should_revert'] = True
                result['reason'] = f"Quality score {current_metrics.quality_score:.1f} below threshold {self.reversion_threshold} for {self.consecutive_failures} consecutive periods"
                result['recommended_config'] = self.last_good_config
                
                # Reset failure count after reversion
                self.failure_count = 0
        else:
            # Quality is good, reset failure count and update last good config
            self.failure_count = 0
            self.last_good_config = current_config.copy()
        
        return result
    
    def get_reversion_recommendations(self, 
                                    symbol: str,
                                    lookback_periods: int = 10) -> Dict[str, Any]:
        """
        Get recommendations for reverting to better configuration.
        
        Args:
            symbol: Symbol to analyze
            lookback_periods: Number of periods to look back
            
        Returns:
            Dictionary with reversion recommendations
        """
        # Get historical metrics for symbol
        symbol_history = [
            m for m in self.monitor.metrics_history[-lookback_periods:]
            if m.symbol == symbol
        ]
        
        if len(symbol_history) < 2:
            return {
                'recommendation': 'insufficient_data',
                'message': 'Not enough historical data for reversion analysis'
            }
        
        # Find best performing configuration
        best_metrics = max(symbol_history, key=lambda m: m.quality_score)
        
        if best_metrics.quality_score > self.reversion_threshold:
            return {
                'recommendation': 'revert_to_best',
                'best_config': {
                    'neutral_band': best_metrics.neutral_band_used,
                    'quality_score': best_metrics.quality_score,
                    'timestamp': best_metrics.timestamp
                },
                'message': f'Revert to neutral band {best_metrics.neutral_band_used} (quality: {best_metrics.quality_score:.1f})'
            }
        else:
            return {
                'recommendation': 'investigate_data',
                'message': 'No recent good configurations found - investigate data quality issues'
            }


# Integration functions for existing codebase
def monitor_target_creation_with_logging(df: pd.DataFrame,
                                       symbol: str,
                                       neutral_band: float,
                                       monitor: Optional[TargetQualityMonitor] = None) -> TargetQualityMetrics:
    """
    Convenience function to monitor target creation with logging.
    
    Args:
        df: DataFrame with created targets
        symbol: Symbol name
        neutral_band: Neutral band used
        monitor: Optional monitor instance (creates default if None)
        
    Returns:
        TargetQualityMetrics
    """
    if monitor is None:
        monitor = TargetQualityMonitor()
    
    return monitor.monitor_target_creation(df, symbol, neutral_band)


def setup_production_monitoring(log_file: str = "logs/target_quality.log",
                              dashboard_file: str = "dashboard/target_quality.json") -> TargetQualityMonitor:
    """
    Setup production monitoring with default configuration.
    
    Args:
        log_file: Path to log file
        dashboard_file: Path to dashboard JSON file
        
    Returns:
        Configured TargetQualityMonitor
    """
    thresholds = QualityThresholds(
        min_flat_percentage=15.0,
        max_flat_percentage=50.0,
        min_samples=100,
        min_quality_score=70.0,
        max_class_imbalance=0.75
    )
    
    return TargetQualityMonitor(
        log_file=log_file,
        dashboard_output=dashboard_file,
        thresholds=thresholds
    )


def create_quality_alerts_summary(monitor: TargetQualityMonitor,
                                hours_back: int = 24) -> str:
    """
    Create a summary of quality alerts for the specified time period.
    
    Args:
        monitor: TargetQualityMonitor instance
        hours_back: Hours to look back
        
    Returns:
        Formatted alert summary string
    """
    dashboard_data = monitor.get_dashboard_data(hours_back=hours_back)
    
    if dashboard_data['status'] != 'success':
        return f"No monitoring data available: {dashboard_data.get('message', 'Unknown error')}"
    
    summary = dashboard_data['summary']
    recent_alerts = dashboard_data['recent_alerts']
    
    report = [
        "=" * 50,
        "TARGET QUALITY MONITORING SUMMARY",
        "=" * 50,
        f"Time Period: Last {hours_back} hours",
        f"Symbols Monitored: {summary['total_symbols']}",
        f"Average Quality Score: {summary['avg_quality_score']:.1f}/100",
        f"Quality Range: {summary['min_quality_score']:.1f} - {summary['max_quality_score']:.1f}",
        f"Average FLAT Percentage: {summary['avg_flat_percentage']:.1f}%",
        f"Total Alerts: {summary['total_alerts']}",
        ""
    ]
    
    if recent_alerts:
        report.append("RECENT ALERTS:")
        report.append("-" * 20)
        for alert in recent_alerts[-10:]:  # Show last 10 alerts
            timestamp = datetime.fromisoformat(alert['timestamp']).strftime('%H:%M:%S')
            report.append(f"{timestamp} [{alert['symbol']}] {alert['alert']}")
    else:
        report.append("No recent alerts.")
    
    return "\n".join(report)