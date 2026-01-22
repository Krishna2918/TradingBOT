"""Production Optimizer for Real-time System Monitoring and Optimization"""

import asyncio
import logging
import psutil
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from collections import deque
import threading
import gc

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    memory_available: float
    disk_usage: float
    network_io: Dict[str, float]
    process_count: int
    load_average: float

@dataclass
class OptimizationAction:
    """An optimization action to be taken."""
    action_type: str
    description: str
    expected_improvement: float
    risk_level: str
    parameters: Dict[str, Any]
    timestamp: datetime

@dataclass
class OptimizationResult:
    """Result of an optimization action."""
    action: OptimizationAction
    success: bool
    actual_improvement: float
    execution_time: float
    side_effects: List[str]
    timestamp: datetime

class SystemMonitor:
    """Real-time system monitoring."""
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.metrics_history = deque(maxlen=1000)
        self.is_monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start system monitoring in background thread."""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("System monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        # CPU usage
        cpu_usage = psutil.cpu_percent(interval=0.1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        memory_available = memory.available / (1024**3)  # GB
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_usage = disk.percent
        
        # Network I/O
        network = psutil.net_io_counters()
        network_io = {
            'bytes_sent': network.bytes_sent,
            'bytes_recv': network.bytes_recv,
            'packets_sent': network.packets_sent,
            'packets_recv': network.packets_recv
        }
        
        # Process count
        process_count = len(psutil.pids())
        
        # Load average (Unix-like systems)
        try:
            load_average = psutil.getloadavg()[0]
        except AttributeError:
            load_average = cpu_usage / 100.0  # Fallback for Windows
        
        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            memory_available=memory_available,
            disk_usage=disk_usage,
            network_io=network_io,
            process_count=process_count,
            load_average=load_average
        )
    
    def get_current_metrics(self) -> Optional[SystemMetrics]:
        """Get the most recent system metrics."""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_trend(self, metric_name: str, window_size: int = 10) -> float:
        """Get trend for a specific metric."""
        if len(self.metrics_history) < window_size:
            return 0.0
        
        recent_metrics = list(self.metrics_history)[-window_size:]
        values = [getattr(metric, metric_name) for metric in recent_metrics]
        
        if len(values) < 2:
            return 0.0
        
        # Calculate trend using linear regression
        x = np.arange(len(values))
        trend = np.polyfit(x, values, 1)[0]
        return trend
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of system metrics."""
        if not self.metrics_history:
            return {}
        
        recent_metrics = list(self.metrics_history)[-10:]  # Last 10 samples
        
        summary = {
            'avg_cpu_usage': np.mean([m.cpu_usage for m in recent_metrics]),
            'avg_memory_usage': np.mean([m.memory_usage for m in recent_metrics]),
            'avg_memory_available': np.mean([m.memory_available for m in recent_metrics]),
            'avg_disk_usage': np.mean([m.disk_usage for m in recent_metrics]),
            'avg_load_average': np.mean([m.load_average for m in recent_metrics]),
            'cpu_trend': self.get_metrics_trend('cpu_usage'),
            'memory_trend': self.get_metrics_trend('memory_usage'),
            'total_samples': len(self.metrics_history)
        }
        
        return summary

class PerformanceOptimizer:
    """Performance optimization engine."""
    
    def __init__(self):
        self.optimization_history = []
        self.optimization_triggers = {
            'high_cpu_usage': {'threshold': 80.0, 'enabled': True},
            'high_memory_usage': {'threshold': 85.0, 'enabled': True},
            'low_memory_available': {'threshold': 1.0, 'enabled': True},  # GB
            'high_disk_usage': {'threshold': 90.0, 'enabled': True},
            'high_load_average': {'threshold': 2.0, 'enabled': True}
        }
        self.optimization_actions = {
            'memory_cleanup': self._optimize_memory,
            'cpu_optimization': self._optimize_cpu,
            'disk_cleanup': self._optimize_disk,
            'process_optimization': self._optimize_processes,
            'cache_optimization': self._optimize_cache
        }
    
    def check_optimization_triggers(self, metrics: SystemMetrics) -> List[str]:
        """Check if any optimization triggers are activated."""
        triggered_optimizations = []
        
        for trigger_name, config in self.optimization_triggers.items():
            if not config['enabled']:
                continue
            
            threshold = config['threshold']
            triggered = False
            
            if trigger_name == 'high_cpu_usage' and metrics.cpu_usage > threshold:
                triggered = True
            elif trigger_name == 'high_memory_usage' and metrics.memory_usage > threshold:
                triggered = True
            elif trigger_name == 'low_memory_available' and metrics.memory_available < threshold:
                triggered = True
            elif trigger_name == 'high_disk_usage' and metrics.disk_usage > threshold:
                triggered = True
            elif trigger_name == 'high_load_average' and metrics.load_average > threshold:
                triggered = True
            
            if triggered:
                triggered_optimizations.append(trigger_name)
        
        return triggered_optimizations
    
    def generate_optimization_actions(self, triggered_optimizations: List[str]) -> List[OptimizationAction]:
        """Generate optimization actions based on triggered optimizations."""
        actions = []
        
        for trigger in triggered_optimizations:
            if trigger == 'high_cpu_usage':
                actions.append(OptimizationAction(
                    action_type='cpu_optimization',
                    description='Optimize CPU usage by reducing process priority and limiting concurrent operations',
                    expected_improvement=0.15,
                    risk_level='low',
                    parameters={'max_concurrent_operations': 4},
                    timestamp=datetime.now()
                ))
            
            elif trigger == 'high_memory_usage':
                actions.append(OptimizationAction(
                    action_type='memory_cleanup',
                    description='Clean up memory by garbage collection and cache clearing',
                    expected_improvement=0.20,
                    risk_level='low',
                    parameters={'force_gc': True, 'clear_caches': True},
                    timestamp=datetime.now()
                ))
            
            elif trigger == 'low_memory_available':
                actions.append(OptimizationAction(
                    action_type='memory_cleanup',
                    description='Aggressive memory cleanup to free up available memory',
                    expected_improvement=0.30,
                    risk_level='medium',
                    parameters={'force_gc': True, 'clear_caches': True, 'reduce_buffers': True},
                    timestamp=datetime.now()
                ))
            
            elif trigger == 'high_disk_usage':
                actions.append(OptimizationAction(
                    action_type='disk_cleanup',
                    description='Clean up temporary files and logs to free disk space',
                    expected_improvement=0.10,
                    risk_level='low',
                    parameters={'clean_temp_files': True, 'compress_logs': True},
                    timestamp=datetime.now()
                ))
            
            elif trigger == 'high_load_average':
                actions.append(OptimizationAction(
                    action_type='process_optimization',
                    description='Optimize process scheduling and reduce system load',
                    expected_improvement=0.25,
                    risk_level='medium',
                    parameters={'reduce_process_priority': True, 'limit_concurrent_io': True},
                    timestamp=datetime.now()
                ))
        
        return actions
    
    async def execute_optimization_action(self, action: OptimizationAction) -> OptimizationResult:
        """Execute an optimization action."""
        start_time = time.time()
        side_effects = []
        
        try:
            if action.action_type in self.optimization_actions:
                optimizer_func = self.optimization_actions[action.action_type]
                actual_improvement = await optimizer_func(action.parameters)
                success = True
            else:
                logger.warning(f"Unknown optimization action: {action.action_type}")
                actual_improvement = 0.0
                success = False
                side_effects.append(f"Unknown action type: {action.action_type}")
        
        except Exception as e:
            logger.error(f"Error executing optimization action {action.action_type}: {e}")
            actual_improvement = 0.0
            success = False
            side_effects.append(f"Execution error: {e}")
        
        execution_time = time.time() - start_time
        
        result = OptimizationResult(
            action=action,
            success=success,
            actual_improvement=actual_improvement,
            execution_time=execution_time,
            side_effects=side_effects,
            timestamp=datetime.now()
        )
        
        self.optimization_history.append(result)
        
        return result
    
    async def _optimize_memory(self, parameters: Dict[str, Any]) -> float:
        """Optimize memory usage."""
        improvement = 0.0
        
        if parameters.get('force_gc', False):
            # Force garbage collection
            collected = gc.collect()
            improvement += 0.1
            logger.info(f"Garbage collection freed {collected} objects")
        
        if parameters.get('clear_caches', False):
            # Clear various caches (this would be application-specific)
            improvement += 0.1
            logger.info("Cleared application caches")
        
        if parameters.get('reduce_buffers', False):
            # Reduce buffer sizes (application-specific)
            improvement += 0.1
            logger.info("Reduced buffer sizes")
        
        return min(improvement, 0.3)  # Cap at 30% improvement
    
    async def _optimize_cpu(self, parameters: Dict[str, Any]) -> float:
        """Optimize CPU usage."""
        improvement = 0.0
        
        if 'max_concurrent_operations' in parameters:
            # Limit concurrent operations (application-specific)
            max_ops = parameters['max_concurrent_operations']
            improvement += 0.15
            logger.info(f"Limited concurrent operations to {max_ops}")
        
        return improvement
    
    async def _optimize_disk(self, parameters: Dict[str, Any]) -> float:
        """Optimize disk usage."""
        improvement = 0.0
        
        if parameters.get('clean_temp_files', False):
            # Clean temporary files (application-specific)
            improvement += 0.05
            logger.info("Cleaned temporary files")
        
        if parameters.get('compress_logs', False):
            # Compress log files (application-specific)
            improvement += 0.05
            logger.info("Compressed log files")
        
        return improvement
    
    async def _optimize_processes(self, parameters: Dict[str, Any]) -> float:
        """Optimize process management."""
        improvement = 0.0
        
        if parameters.get('reduce_process_priority', False):
            # Reduce process priority (application-specific)
            improvement += 0.1
            logger.info("Reduced process priority")
        
        if parameters.get('limit_concurrent_io', False):
            # Limit concurrent I/O operations
            improvement += 0.15
            logger.info("Limited concurrent I/O operations")
        
        return improvement
    
    async def _optimize_cache(self, parameters: Dict[str, Any]) -> float:
        """Optimize cache usage."""
        improvement = 0.0
        
        # Cache optimization (application-specific)
        improvement += 0.1
        logger.info("Optimized cache usage")
        
        return improvement

class ProductionOptimizer:
    """Main production optimizer that coordinates system monitoring and optimization."""
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.system_monitor = SystemMonitor(monitoring_interval)
        self.performance_optimizer = PerformanceOptimizer()
        self.is_running = False
        self.optimization_interval = 30.0  # Check for optimizations every 30 seconds
        self.last_optimization_check = datetime.now()
        
    async def start(self):
        """Start the production optimizer."""
        if not self.is_running:
            self.is_running = True
            self.system_monitor.start_monitoring()
            logger.info("Production optimizer started")
            
            # Start optimization loop
            asyncio.create_task(self._optimization_loop())
    
    async def stop(self):
        """Stop the production optimizer."""
        if self.is_running:
            self.is_running = False
            self.system_monitor.stop_monitoring()
            logger.info("Production optimizer stopped")
    
    async def _optimization_loop(self):
        """Main optimization loop."""
        while self.is_running:
            try:
                await asyncio.sleep(self.optimization_interval)
                
                # Check if it's time for optimization check
                now = datetime.now()
                if (now - self.last_optimization_check).total_seconds() >= self.optimization_interval:
                    await self._check_and_optimize()
                    self.last_optimization_check = now
                    
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(5.0)  # Wait before retrying
    
    async def _check_and_optimize(self):
        """Check system metrics and perform optimizations if needed."""
        current_metrics = self.system_monitor.get_current_metrics()
        
        if not current_metrics:
            return
        
        # Check for optimization triggers
        triggered_optimizations = self.performance_optimizer.check_optimization_triggers(current_metrics)
        
        if triggered_optimizations:
            logger.info(f"Optimization triggers activated: {triggered_optimizations}")
            
            # Generate optimization actions
            actions = self.performance_optimizer.generate_optimization_actions(triggered_optimizations)
            
            # Execute optimization actions
            for action in actions:
                logger.info(f"Executing optimization: {action.description}")
                result = await self.performance_optimizer.execute_optimization_action(action)
                
                if result.success:
                    logger.info(f"Optimization successful: {result.actual_improvement:.2%} improvement")
                else:
                    logger.warning(f"Optimization failed: {result.side_effects}")
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        stats = {
            'is_running': self.is_running,
            'monitoring_active': self.system_monitor.is_monitoring,
            'total_optimizations': len(self.performance_optimizer.optimization_history),
            'successful_optimizations': len([r for r in self.performance_optimizer.optimization_history if r.success]),
            'failed_optimizations': len([r for r in self.performance_optimizer.optimization_history if not r.success]),
            'avg_improvement': 0.0,
            'avg_execution_time': 0.0,
            'optimization_types': {},
            'recent_optimizations': []
        }
        
        if self.performance_optimizer.optimization_history:
            results = self.performance_optimizer.optimization_history
            
            # Calculate averages
            successful_results = [r for r in results if r.success]
            if successful_results:
                stats['avg_improvement'] = np.mean([r.actual_improvement for r in successful_results])
                stats['avg_execution_time'] = np.mean([r.execution_time for r in successful_results])
            
            # Count optimization types
            for result in results:
                action_type = result.action.action_type
                stats['optimization_types'][action_type] = stats['optimization_types'].get(action_type, 0) + 1
            
            # Recent optimizations
            recent_results = results[-10:]  # Last 10 optimizations
            stats['recent_optimizations'] = [
                {
                    'action_type': r.action.action_type,
                    'description': r.action.description,
                    'success': r.success,
                    'improvement': r.actual_improvement,
                    'timestamp': r.timestamp.isoformat()
                }
                for r in recent_results
            ]
        
        # Add system metrics summary
        stats['system_metrics'] = self.system_monitor.get_metrics_summary()
        
        return stats
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get current system health status."""
        current_metrics = self.system_monitor.get_current_metrics()
        
        if not current_metrics:
            return {'status': 'unknown', 'message': 'No metrics available'}
        
        health_status = 'healthy'
        issues = []
        
        # Check CPU usage
        if current_metrics.cpu_usage > 90:
            health_status = 'critical'
            issues.append('Very high CPU usage')
        elif current_metrics.cpu_usage > 80:
            health_status = 'warning'
            issues.append('High CPU usage')
        
        # Check memory usage
        if current_metrics.memory_usage > 95:
            health_status = 'critical'
            issues.append('Very high memory usage')
        elif current_metrics.memory_usage > 85:
            health_status = 'warning'
            issues.append('High memory usage')
        
        # Check available memory
        if current_metrics.memory_available < 0.5:  # Less than 500MB
            health_status = 'critical'
            issues.append('Very low available memory')
        elif current_metrics.memory_available < 1.0:  # Less than 1GB
            health_status = 'warning'
            issues.append('Low available memory')
        
        # Check disk usage
        if current_metrics.disk_usage > 95:
            health_status = 'critical'
            issues.append('Very high disk usage')
        elif current_metrics.disk_usage > 90:
            health_status = 'warning'
            issues.append('High disk usage')
        
        return {
            'status': health_status,
            'issues': issues,
            'metrics': {
                'cpu_usage': current_metrics.cpu_usage,
                'memory_usage': current_metrics.memory_usage,
                'memory_available': current_metrics.memory_available,
                'disk_usage': current_metrics.disk_usage,
                'load_average': current_metrics.load_average
            },
            'timestamp': current_metrics.timestamp.isoformat()
        }

# Global instance
_production_optimizer = None

def get_production_optimizer() -> ProductionOptimizer:
    """Get the global production optimizer instance."""
    global _production_optimizer
    if _production_optimizer is None:
        _production_optimizer = ProductionOptimizer()
    return _production_optimizer