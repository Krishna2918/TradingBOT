"""
Resource monitoring for the Portfolio Optimization Engine.

Monitors system resources to prevent over-usage and ensure efficient
operation alongside existing trading system components.
"""

import psutil
import time
import threading
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from collections import deque

from ..config.settings import get_config
from .logger import get_logger


class ResourceMonitor:
    """
    System resource monitor for the optimization engine.
    
    Tracks CPU, memory, and API usage to prevent resource conflicts
    and ensure efficient operation.
    """
    
    def __init__(self, monitoring_interval: int = 30):
        self.config = get_config()
        self.logger = get_logger('resource_monitor')
        self.monitoring_interval = monitoring_interval
        
        # Resource usage history
        self._cpu_history = deque(maxlen=100)
        self._memory_history = deque(maxlen=100)
        self._api_call_history = deque(maxlen=1000)
        
        # Current resource usage
        self._current_stats = {
            'cpu_percent': 0.0,
            'memory_mb': 0.0,
            'memory_percent': 0.0,
            'api_calls_per_minute': 0
        }
        
        # Resource limits and thresholds
        self._limits = {
            'max_memory_mb': self.config.performance.memory_limit_mb,
            'max_cpu_percent': 80.0,
            'max_api_calls_per_minute': self.config.api_limits.alpha_vantage_calls_per_minute
        }
        
        # Alert callbacks
        self._alert_callbacks = []
        
        # Monitoring thread
        self._monitoring_active = False
        self._monitoring_thread = None
        
        # Lock for thread safety
        self._lock = threading.RLock()
    
    def start_monitoring(self) -> None:
        """Start resource monitoring"""
        if not self._monitoring_active:
            self._monitoring_active = True
            self._monitoring_thread = threading.Thread(
                target=self._monitoring_loop, 
                daemon=True
            )
            self._monitoring_thread.start()
            self.logger.info("Resource monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop resource monitoring"""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        self.logger.info("Resource monitoring stopped")
    
    def record_api_call(self, provider: str, endpoint: str) -> None:
        """
        Record an API call for rate limiting tracking.
        
        Args:
            provider: API provider name
            endpoint: API endpoint called
        """
        with self._lock:
            self._api_call_history.append({
                'timestamp': time.time(),
                'provider': provider,
                'endpoint': endpoint
            })
    
    def can_make_api_call(self, provider: str = 'alpha_vantage') -> bool:
        """
        Check if we can make an API call without exceeding rate limits.
        
        Args:
            provider: API provider to check
            
        Returns:
            True if API call is allowed, False otherwise
        """
        with self._lock:
            current_time = time.time()
            minute_ago = current_time - 60
            
            # Count API calls in the last minute for this provider
            recent_calls = sum(
                1 for call in self._api_call_history
                if call['timestamp'] > minute_ago and call['provider'] == provider
            )
            
            if provider == 'alpha_vantage':
                limit = self._limits['max_api_calls_per_minute']
            else:
                limit = 60  # Default limit
            
            return recent_calls < limit
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """
        Get current resource usage statistics.
        
        Returns:
            Dictionary with current resource usage
        """
        with self._lock:
            return self._current_stats.copy()
    
    def get_resource_history(self, minutes: int = 30) -> Dict[str, Any]:
        """
        Get resource usage history.
        
        Args:
            minutes: Number of minutes of history to return
            
        Returns:
            Dictionary with resource usage history
        """
        cutoff_time = time.time() - (minutes * 60)
        
        with self._lock:
            cpu_history = [
                entry for entry in self._cpu_history
                if entry['timestamp'] > cutoff_time
            ]
            memory_history = [
                entry for entry in self._memory_history
                if entry['timestamp'] > cutoff_time
            ]
            api_history = [
                entry for entry in self._api_call_history
                if entry['timestamp'] > cutoff_time
            ]
            
            return {
                'cpu_history': cpu_history,
                'memory_history': memory_history,
                'api_history': api_history
            }
    
    def is_resource_available(self, resource_type: str) -> bool:
        """
        Check if a specific resource is available.
        
        Args:
            resource_type: Type of resource ('cpu', 'memory', 'api')
            
        Returns:
            True if resource is available, False otherwise
        """
        with self._lock:
            if resource_type == 'cpu':
                return self._current_stats['cpu_percent'] < self._limits['max_cpu_percent']
            elif resource_type == 'memory':
                return self._current_stats['memory_mb'] < self._limits['max_memory_mb']
            elif resource_type == 'api':
                return self.can_make_api_call()
            else:
                return True
    
    def wait_for_resource(
        self, 
        resource_type: str, 
        timeout: int = 60,
        check_interval: int = 5
    ) -> bool:
        """
        Wait for a resource to become available.
        
        Args:
            resource_type: Type of resource to wait for
            timeout: Maximum time to wait in seconds
            check_interval: How often to check in seconds
            
        Returns:
            True if resource became available, False if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.is_resource_available(resource_type):
                return True
            
            self.logger.debug(f"Waiting for {resource_type} resource to become available")
            time.sleep(check_interval)
        
        self.logger.warning(f"Timeout waiting for {resource_type} resource")
        return False
    
    def add_alert_callback(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """
        Add callback for resource alerts.
        
        Args:
            callback: Function to call when resource limits are exceeded
        """
        self._alert_callbacks.append(callback)
    
    def get_optimization_recommendations(self) -> Dict[str, str]:
        """
        Get recommendations for optimizing resource usage.
        
        Returns:
            Dictionary with optimization recommendations
        """
        recommendations = {}
        
        with self._lock:
            # CPU recommendations
            if self._current_stats['cpu_percent'] > 70:
                recommendations['cpu'] = "Consider reducing parallel workers or optimization frequency"
            
            # Memory recommendations
            if self._current_stats['memory_percent'] > 80:
                recommendations['memory'] = "Consider reducing cache size or clearing old data"
            
            # API recommendations
            api_rate = self._current_stats['api_calls_per_minute']
            if api_rate > self._limits['max_api_calls_per_minute'] * 0.8:
                recommendations['api'] = "Consider increasing data caching or reducing update frequency"
        
        return recommendations
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self._monitoring_active:
            try:
                self._update_resource_stats()
                self._check_resource_limits()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                self.logger.error(f"Error in resource monitoring: {e}")
                time.sleep(self.monitoring_interval)
    
    def _update_resource_stats(self) -> None:
        """Update current resource statistics"""
        current_time = time.time()
        
        # Get CPU and memory usage
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        process = psutil.Process()
        process_memory_mb = process.memory_info().rss / (1024 * 1024)
        
        # Calculate API calls per minute
        minute_ago = current_time - 60
        api_calls_per_minute = sum(
            1 for call in self._api_call_history
            if call['timestamp'] > minute_ago
        )
        
        with self._lock:
            # Update current stats
            self._current_stats.update({
                'cpu_percent': cpu_percent,
                'memory_mb': process_memory_mb,
                'memory_percent': memory_info.percent,
                'api_calls_per_minute': api_calls_per_minute
            })
            
            # Add to history
            self._cpu_history.append({
                'timestamp': current_time,
                'value': cpu_percent
            })
            
            self._memory_history.append({
                'timestamp': current_time,
                'value': process_memory_mb
            })
    
    def _check_resource_limits(self) -> None:
        """Check if resource limits are exceeded and trigger alerts"""
        alerts = []
        
        with self._lock:
            # Check CPU limit
            if self._current_stats['cpu_percent'] > self._limits['max_cpu_percent']:
                alerts.append({
                    'type': 'cpu_limit_exceeded',
                    'current': self._current_stats['cpu_percent'],
                    'limit': self._limits['max_cpu_percent']
                })
            
            # Check memory limit
            if self._current_stats['memory_mb'] > self._limits['max_memory_mb']:
                alerts.append({
                    'type': 'memory_limit_exceeded',
                    'current': self._current_stats['memory_mb'],
                    'limit': self._limits['max_memory_mb']
                })
            
            # Check API rate limit
            if self._current_stats['api_calls_per_minute'] > self._limits['max_api_calls_per_minute']:
                alerts.append({
                    'type': 'api_limit_exceeded',
                    'current': self._current_stats['api_calls_per_minute'],
                    'limit': self._limits['max_api_calls_per_minute']
                })
        
        # Trigger alert callbacks
        for alert in alerts:
            self.logger.warning(f"Resource limit exceeded: {alert}")
            for callback in self._alert_callbacks:
                try:
                    callback(alert['type'], alert)
                except Exception as e:
                    self.logger.error(f"Error in alert callback: {e}")


# Global resource monitor instance
_resource_monitor = None


def get_resource_monitor() -> ResourceMonitor:
    """Get the global resource monitor instance"""
    global _resource_monitor
    if _resource_monitor is None:
        _resource_monitor = ResourceMonitor()
        _resource_monitor.start_monitoring()
    return _resource_monitor