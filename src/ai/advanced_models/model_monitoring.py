"""
Model Monitoring and Performance Tracking for Advanced AI Models

This module provides comprehensive monitoring and performance tracking
for all advanced AI models with real-time metrics and alerting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import logging
from datetime import datetime, timedelta
import threading
import time
from collections import defaultdict, deque
import warnings

logger = logging.getLogger(__name__)

class ModelMonitoring:
    """
    Comprehensive model monitoring system for advanced AI models.
    """
    
    def __init__(
        self,
        monitoring_name: str = "model_monitoring",
        monitoring_interval: int = 60,
        alert_thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize model monitoring system.
        
        Args:
            monitoring_name: Name for the monitoring system
            monitoring_interval: Monitoring interval in seconds
            alert_thresholds: Thresholds for triggering alerts
        """
        self.monitoring_name = monitoring_name
        self.monitoring_interval = monitoring_interval
        self.alert_thresholds = alert_thresholds or {
            'accuracy_drop': 0.1,
            'latency_increase': 2.0,
            'error_rate': 0.05,
            'memory_usage': 0.9,
            'cpu_usage': 0.9
        }
        
        # Model registry
        self.monitored_models = {}
        
        # Performance metrics
        self.performance_metrics = defaultdict(lambda: {
            'predictions': deque(maxlen=1000),
            'latencies': deque(maxlen=1000),
            'errors': deque(maxlen=1000),
            'memory_usage': deque(maxlen=1000),
            'cpu_usage': deque(maxlen=1000),
            'accuracy_scores': deque(maxlen=1000)
        })
        
        # Alert system
        self.alerts = deque(maxlen=1000)
        self.alert_callbacks = []
        
        # Monitoring control
        self.monitoring_active = False
        self.monitoring_thread = None
        
        logger.info(f"Model Monitoring initialized: {monitoring_name}")
    
    def register_model(
        self,
        model_name: str,
        model: Any,
        model_type: str = 'unknown',
        performance_metrics: Optional[List[str]] = None
    ) -> None:
        """
        Register a model for monitoring.
        
        Args:
            model_name: Name for the model
            model: Model object
            model_type: Type of model
            performance_metrics: List of metrics to track
        """
        self.monitored_models[model_name] = {
            'model': model,
            'type': model_type,
            'metrics': performance_metrics or ['latency', 'accuracy', 'memory'],
            'registered_at': datetime.now(),
            'total_predictions': 0,
            'total_errors': 0
        }
        
        logger.info(f"Registered model '{model_name}' for monitoring")
    
    def unregister_model(self, model_name: str) -> None:
        """Unregister a model from monitoring."""
        if model_name in self.monitored_models:
            del self.monitored_models[model_name]
            if model_name in self.performance_metrics:
                del self.performance_metrics[model_name]
            logger.info(f"Unregistered model '{model_name}' from monitoring")
        else:
            logger.warning(f"Model '{model_name}' not found in monitoring")
    
    def start_monitoring(self) -> None:
        """Start continuous model monitoring."""
        if self.monitoring_active:
            logger.warning("Monitoring is already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Model monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop continuous model monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.info("Model monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Monitor each registered model
                for model_name, model_info in self.monitored_models.items():
                    self._monitor_model(model_name, model_info)
                
                # Check for alerts
                self._check_alerts()
                
                # Sleep for monitoring interval
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _monitor_model(self, model_name: str, model_info: Dict[str, Any]) -> None:
        """Monitor a specific model."""
        try:
            # Collect system metrics
            import psutil
            process = psutil.Process()
            
            # Memory usage
            memory_info = process.memory_info()
            memory_usage = memory_info.rss / (1024 * 1024)  # MB
            
            # CPU usage
            cpu_usage = process.cpu_percent()
            
            # Store metrics
            self.performance_metrics[model_name]['memory_usage'].append(memory_usage)
            self.performance_metrics[model_name]['cpu_usage'].append(cpu_usage)
            
        except Exception as e:
            logger.error(f"Error monitoring model '{model_name}': {e}")
    
    def _check_alerts(self) -> None:
        """Check for alert conditions."""
        try:
            for model_name, metrics in self.performance_metrics.items():
                # Check accuracy drop
                if len(metrics['accuracy_scores']) > 10:
                    recent_accuracy = np.mean(list(metrics['accuracy_scores'])[-5:])
                    historical_accuracy = np.mean(list(metrics['accuracy_scores'])[-20:-5])
                    
                    if historical_accuracy - recent_accuracy > self.alert_thresholds['accuracy_drop']:
                        self._trigger_alert(
                            model_name,
                            'accuracy_drop',
                            f"Accuracy dropped from {historical_accuracy:.3f} to {recent_accuracy:.3f}"
                        )
                
                # Check latency increase
                if len(metrics['latencies']) > 10:
                    recent_latency = np.mean(list(metrics['latencies'])[-5:])
                    historical_latency = np.mean(list(metrics['latencies'])[-20:-5])
                    
                    if recent_latency > historical_latency * self.alert_thresholds['latency_increase']:
                        self._trigger_alert(
                            model_name,
                            'latency_increase',
                            f"Latency increased from {historical_latency:.3f}s to {recent_latency:.3f}s"
                        )
                
                # Check error rate
                if len(metrics['errors']) > 10:
                    recent_errors = list(metrics['errors'])[-10:]
                    error_rate = sum(recent_errors) / len(recent_errors)
                    
                    if error_rate > self.alert_thresholds['error_rate']:
                        self._trigger_alert(
                            model_name,
                            'high_error_rate',
                            f"Error rate is {error_rate:.3f}, threshold is {self.alert_thresholds['error_rate']}"
                        )
                
                # Check memory usage
                if len(metrics['memory_usage']) > 0:
                    recent_memory = list(metrics['memory_usage'])[-1]
                    if recent_memory > self.alert_thresholds['memory_usage'] * 1024:  # Convert to MB
                        self._trigger_alert(
                            model_name,
                            'high_memory_usage',
                            f"Memory usage is {recent_memory:.1f}MB, threshold is {self.alert_thresholds['memory_usage'] * 1024:.1f}MB"
                        )
                
                # Check CPU usage
                if len(metrics['cpu_usage']) > 0:
                    recent_cpu = list(metrics['cpu_usage'])[-1]
                    if recent_cpu > self.alert_thresholds['cpu_usage'] * 100:  # Convert to percentage
                        self._trigger_alert(
                            model_name,
                            'high_cpu_usage',
                            f"CPU usage is {recent_cpu:.1f}%, threshold is {self.alert_thresholds['cpu_usage'] * 100:.1f}%"
                        )
        
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
    
    def _trigger_alert(
        self,
        model_name: str,
        alert_type: str,
        message: str
    ) -> None:
        """Trigger an alert."""
        alert = {
            'model_name': model_name,
            'alert_type': alert_type,
            'message': message,
            'timestamp': datetime.now(),
            'severity': self._get_alert_severity(alert_type)
        }
        
        self.alerts.append(alert)
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
        
        logger.warning(f"ALERT [{alert_type}] {model_name}: {message}")
    
    def _get_alert_severity(self, alert_type: str) -> str:
        """Get alert severity level."""
        severity_map = {
            'accuracy_drop': 'high',
            'latency_increase': 'medium',
            'high_error_rate': 'high',
            'high_memory_usage': 'medium',
            'high_cpu_usage': 'low'
        }
        return severity_map.get(alert_type, 'medium')
    
    def add_alert_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add an alert callback function."""
        self.alert_callbacks.append(callback)
        logger.info("Added alert callback")
    
    def remove_alert_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Remove an alert callback function."""
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)
            logger.info("Removed alert callback")
    
    def log_prediction(
        self,
        model_name: str,
        prediction: Any,
        actual: Optional[Any] = None,
        latency: Optional[float] = None,
        error: Optional[str] = None
    ) -> None:
        """
        Log a model prediction for monitoring.
        
        Args:
            model_name: Name of the model
            prediction: Model prediction
            actual: Actual target value (for accuracy calculation)
            latency: Prediction latency in seconds
            error: Error message if prediction failed
        """
        if model_name not in self.monitored_models:
            logger.warning(f"Model '{model_name}' not registered for monitoring")
            return
        
        try:
            # Update model statistics
            model_info = self.monitored_models[model_name]
            model_info['total_predictions'] += 1
            
            if error:
                model_info['total_errors'] += 1
                self.performance_metrics[model_name]['errors'].append(1)
            else:
                self.performance_metrics[model_name]['errors'].append(0)
            
            # Log prediction
            self.performance_metrics[model_name]['predictions'].append({
                'prediction': prediction,
                'actual': actual,
                'timestamp': datetime.now(),
                'error': error
            })
            
            # Log latency
            if latency is not None:
                self.performance_metrics[model_name]['latencies'].append(latency)
            
            # Calculate accuracy if actual value provided
            if actual is not None and error is None:
                if isinstance(prediction, (int, float)) and isinstance(actual, (int, float)):
                    # Regression accuracy (inverted error)
                    error_value = abs(prediction - actual)
                    accuracy = 1.0 / (1.0 + error_value)
                else:
                    # Classification accuracy
                    accuracy = 1.0 if prediction == actual else 0.0
                
                self.performance_metrics[model_name]['accuracy_scores'].append(accuracy)
        
        except Exception as e:
            logger.error(f"Error logging prediction for '{model_name}': {e}")
    
    def get_model_performance(self, model_name: str) -> Dict[str, Any]:
        """Get performance metrics for a specific model."""
        if model_name not in self.monitored_models:
            return {'error': f'Model {model_name} not found'}
        
        metrics = self.performance_metrics[model_name]
        model_info = self.monitored_models[model_name]
        
        # Calculate performance statistics
        performance_stats = {
            'model_name': model_name,
            'model_type': model_info['type'],
            'total_predictions': model_info['total_predictions'],
            'total_errors': model_info['total_errors'],
            'error_rate': model_info['total_errors'] / max(model_info['total_predictions'], 1),
            'registered_at': model_info['registered_at'].isoformat()
        }
        
        # Latency statistics
        if metrics['latencies']:
            latencies = list(metrics['latencies'])
            performance_stats['latency'] = {
                'mean': np.mean(latencies),
                'std': np.std(latencies),
                'min': np.min(latencies),
                'max': np.max(latencies),
                'p95': np.percentile(latencies, 95),
                'p99': np.percentile(latencies, 99)
            }
        
        # Accuracy statistics
        if metrics['accuracy_scores']:
            accuracies = list(metrics['accuracy_scores'])
            performance_stats['accuracy'] = {
                'mean': np.mean(accuracies),
                'std': np.std(accuracies),
                'min': np.min(accuracies),
                'max': np.max(accuracies),
                'recent_trend': np.mean(accuracies[-10:]) if len(accuracies) >= 10 else np.mean(accuracies)
            }
        
        # Memory statistics
        if metrics['memory_usage']:
            memory_usage = list(metrics['memory_usage'])
            performance_stats['memory'] = {
                'current_mb': memory_usage[-1] if memory_usage else 0,
                'mean_mb': np.mean(memory_usage),
                'max_mb': np.max(memory_usage)
            }
        
        # CPU statistics
        if metrics['cpu_usage']:
            cpu_usage = list(metrics['cpu_usage'])
            performance_stats['cpu'] = {
                'current_percent': cpu_usage[-1] if cpu_usage else 0,
                'mean_percent': np.mean(cpu_usage),
                'max_percent': np.max(cpu_usage)
            }
        
        return performance_stats
    
    def get_all_performance(self) -> Dict[str, Any]:
        """Get performance metrics for all monitored models."""
        return {
            model_name: self.get_model_performance(model_name)
            for model_name in self.monitored_models.keys()
        }
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent alerts within specified time window."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            alert for alert in self.alerts
            if alert['timestamp'] >= cutoff_time
        ]
    
    def get_monitoring_statistics(self) -> Dict[str, Any]:
        """Get overall monitoring statistics."""
        return {
            'monitoring_name': self.monitoring_name,
            'monitoring_active': self.monitoring_active,
            'monitoring_interval': self.monitoring_interval,
            'total_models': len(self.monitored_models),
            'model_names': list(self.monitored_models.keys()),
            'total_alerts': len(self.alerts),
            'recent_alerts_24h': len(self.get_recent_alerts(24)),
            'alert_thresholds': self.alert_thresholds,
            'alert_callbacks': len(self.alert_callbacks)
        }


class PerformanceTracker:
    """
    Performance tracking for individual models and predictions.
    """
    
    def __init__(self, tracker_name: str = "performance_tracker"):
        """
        Initialize performance tracker.
        
        Args:
            tracker_name: Name for the tracker
        """
        self.tracker_name = tracker_name
        
        # Performance history
        self.performance_history = deque(maxlen=10000)
        
        # Model performance
        self.model_performance = defaultdict(lambda: {
            'predictions': 0,
            'correct_predictions': 0,
            'total_latency': 0.0,
            'min_latency': float('inf'),
            'max_latency': 0.0,
            'errors': 0
        })
        
        logger.info(f"Performance Tracker initialized: {tracker_name}")
    
    def track_prediction(
        self,
        model_name: str,
        prediction: Any,
        actual: Optional[Any] = None,
        latency: float = 0.0,
        error: Optional[str] = None
    ) -> None:
        """
        Track a model prediction.
        
        Args:
            model_name: Name of the model
            prediction: Model prediction
            actual: Actual target value
            latency: Prediction latency
            error: Error message if any
        """
        try:
            # Create performance record
            record = {
                'model_name': model_name,
                'prediction': prediction,
                'actual': actual,
                'latency': latency,
                'error': error,
                'timestamp': datetime.now(),
                'correct': False
            }
            
            # Determine if prediction is correct
            if actual is not None and error is None:
                if isinstance(prediction, (int, float)) and isinstance(actual, (int, float)):
                    # Regression: consider correct if within 5% error
                    error_percent = abs(prediction - actual) / abs(actual) if actual != 0 else 0
                    record['correct'] = error_percent <= 0.05
                else:
                    # Classification: exact match
                    record['correct'] = prediction == actual
            
            # Store in history
            self.performance_history.append(record)
            
            # Update model performance
            model_perf = self.model_performance[model_name]
            model_perf['predictions'] += 1
            model_perf['total_latency'] += latency
            model_perf['min_latency'] = min(model_perf['min_latency'], latency)
            model_perf['max_latency'] = max(model_perf['max_latency'], latency)
            
            if error:
                model_perf['errors'] += 1
            elif record['correct']:
                model_perf['correct_predictions'] += 1
        
        except Exception as e:
            logger.error(f"Error tracking prediction: {e}")
    
    def get_model_statistics(self, model_name: str) -> Dict[str, Any]:
        """Get statistics for a specific model."""
        if model_name not in self.model_performance:
            return {'error': f'Model {model_name} not found'}
        
        perf = self.model_performance[model_name]
        
        return {
            'model_name': model_name,
            'total_predictions': perf['predictions'],
            'correct_predictions': perf['correct_predictions'],
            'accuracy': perf['correct_predictions'] / max(perf['predictions'], 1),
            'error_rate': perf['errors'] / max(perf['predictions'], 1),
            'average_latency': perf['total_latency'] / max(perf['predictions'], 1),
            'min_latency': perf['min_latency'] if perf['min_latency'] != float('inf') else 0,
            'max_latency': perf['max_latency']
        }
    
    def get_all_statistics(self) -> Dict[str, Any]:
        """Get statistics for all models."""
        return {
            model_name: self.get_model_statistics(model_name)
            for model_name in self.model_performance.keys()
        }
    
    def get_performance_trend(
        self,
        model_name: str,
        hours: int = 24,
        metric: str = 'accuracy'
    ) -> List[float]:
        """Get performance trend for a model."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Filter records for the model and time window
        model_records = [
            record for record in self.performance_history
            if (record['model_name'] == model_name and 
                record['timestamp'] >= cutoff_time and
                record['error'] is None)
        ]
        
        if not model_records:
            return []
        
        # Calculate metric values
        if metric == 'accuracy':
            # Calculate rolling accuracy
            window_size = max(10, len(model_records) // 10)
            trend = []
            
            for i in range(window_size, len(model_records) + 1):
                window_records = model_records[i-window_size:i]
                accuracy = sum(1 for r in window_records if r['correct']) / len(window_records)
                trend.append(accuracy)
            
            return trend
        
        elif metric == 'latency':
            # Calculate rolling average latency
            window_size = max(10, len(model_records) // 10)
            trend = []
            
            for i in range(window_size, len(model_records) + 1):
                window_records = model_records[i-window_size:i]
                avg_latency = np.mean([r['latency'] for r in window_records])
                trend.append(avg_latency)
            
            return trend
        
        return []
    
    def get_tracker_statistics(self) -> Dict[str, Any]:
        """Get overall tracker statistics."""
        return {
            'tracker_name': self.tracker_name,
            'total_records': len(self.performance_history),
            'total_models': len(self.model_performance),
            'model_names': list(self.model_performance.keys()),
            'date_range': {
                'start': self.performance_history[0]['timestamp'].isoformat() if self.performance_history else None,
                'end': self.performance_history[-1]['timestamp'].isoformat() if self.performance_history else None
            }
        }

