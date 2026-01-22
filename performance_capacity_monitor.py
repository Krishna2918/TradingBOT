#!/usr/bin/env python3
"""
Performance Monitoring and Capacity Planning Tools for Continuous Data Collection System

This script provides comprehensive performance monitoring, capacity planning,
and resource optimization recommendations for production deployment.
"""

import asyncio
import psutil
import time
import json
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import statistics
import argparse
import threading
import numpy as np
from pathlib import Path


@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    process_count: int
    thread_count: int
    collection_rate: float
    error_rate: float
    api_response_time: float
    queue_size: int
    worker_utilization: float


@dataclass
class CapacityRecommendation:
    """Capacity planning recommendation."""
    resource: str
    current_usage: float
    projected_usage: float
    recommendation: str
    priority: str  # 'low', 'medium', 'high', 'critical'
    estimated_timeline: str
    cost_impact: str


class PerformanceCapacityMonitor:
    """
    Comprehensive performance monitoring and capacity planning system.
    
    Features:
    - Real-time performance metrics collection
    - Historical trend analysis
    - Capacity planning with projections
    - Resource optimization recommendations
    - Automated alerting for capacity issues
    - Performance bottleneck identification
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize performance and capacity monitor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        
        # Monitoring settings
        self.collection_interval = self.config.get('collection_interval', 30)  # seconds
        self.history_size = self.config.get('history_size', 2880)  # 24 hours at 30s intervals
        self.db_path = self.config.get('db_path', 'performance_metrics.db')
        
        # Capacity planning settings
        self.projection_days = self.config.get('projection_days', 30)
        self.growth_analysis_days = self.config.get('growth_analysis_days', 7)
        
        # Thresholds
        self.thresholds = self.config.get('thresholds', {
            'cpu_warning': 70.0,
            'cpu_critical': 85.0,
            'memory_warning': 75.0,
            'memory_critical': 90.0,
            'disk_warning': 80.0,
            'disk_critical': 95.0,
            'collection_rate_min': 30.0,
            'error_rate_max': 0.1,
            'api_response_time_max': 5.0
        })
        
        # Data storage
        self.metrics_history: deque = deque(maxlen=self.history_size)
        self.performance_trends = defaultdict(lambda: deque(maxlen=100))
        
        # State
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Statistics
        self.stats = {
            'monitoring_start_time': None,
            'total_samples': 0,
            'last_capacity_analysis': None,
            'recommendations_generated': 0
        }
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize database
        self._init_database()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'collection_interval': 30,
            'history_size': 2880,
            'db_path': 'performance_metrics.db',
            'projection_days': 30,
            'growth_analysis_days': 7,
            'thresholds': {
                'cpu_warning': 70.0,
                'cpu_critical': 85.0,
                'memory_warning': 75.0,
                'memory_critical': 90.0,
                'disk_warning': 80.0,
                'disk_critical': 95.0,
                'collection_rate_min': 30.0,
                'error_rate_max': 0.1,
                'api_response_time_max': 5.0
            }
        }
        
    def _init_database(self) -> None:
        """Initialize SQLite database for metrics storage."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    cpu_percent REAL,
                    memory_percent REAL,
                    disk_usage_percent REAL,
                    network_bytes_sent INTEGER,
                    network_bytes_recv INTEGER,
                    process_count INTEGER,
                    thread_count INTEGER,
                    collection_rate REAL,
                    error_rate REAL,
                    api_response_time REAL,
                    queue_size INTEGER,
                    worker_utilization REAL
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS capacity_recommendations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    resource TEXT NOT NULL,
                    current_usage REAL,
                    projected_usage REAL,
                    recommendation TEXT,
                    priority TEXT,
                    estimated_timeline TEXT,
                    cost_impact TEXT
                )
            ''')
            
            # Create indexes for better query performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON performance_metrics(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_recommendations_timestamp ON capacity_recommendations(timestamp)')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            
    def start_monitoring(self) -> None:
        """Start performance monitoring."""
        if self.is_monitoring:
            self.logger.warning("Performance monitoring is already running")
            return
            
        self.is_monitoring = True
        self.stats['monitoring_start_time'] = datetime.utcnow()
        
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitor_thread.start()
        
        self.logger.info("Started performance monitoring")
        
    def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        if not self.is_monitoring:
            return
            
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=30)
            
        self.logger.info("Stopped performance monitoring")
        
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                metrics = self._collect_metrics()
                self._store_metrics(metrics)
                self._update_trends(metrics)
                
                # Perform capacity analysis periodically (every hour)
                if (self.stats['total_samples'] % (3600 // self.collection_interval)) == 0:
                    asyncio.run(self._perform_capacity_analysis())
                    
                time.sleep(self.collection_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.collection_interval)
                
    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('.')
            network = psutil.net_io_counters()
            
            # Process metrics
            process_count = len(psutil.pids())
            thread_count = sum(p.num_threads() for p in psutil.process_iter(['num_threads']) 
                             if p.info['num_threads'] is not None)
            
            # Application-specific metrics (would be provided by the main system)
            collection_rate = self._get_collection_rate()
            error_rate = self._get_error_rate()
            api_response_time = self._get_api_response_time()
            queue_size = self._get_queue_size()
            worker_utilization = self._get_worker_utilization()
            
            metrics = PerformanceMetrics(
                timestamp=datetime.utcnow(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_usage_percent=(disk.used / disk.total) * 100,
                network_bytes_sent=network.bytes_sent,
                network_bytes_recv=network.bytes_recv,
                process_count=process_count,
                thread_count=thread_count,
                collection_rate=collection_rate,
                error_rate=error_rate,
                api_response_time=api_response_time,
                queue_size=queue_size,
                worker_utilization=worker_utilization
            )
            
            self.metrics_history.append(metrics)
            self.stats['total_samples'] += 1
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error collecting metrics: {e}")
            # Return default metrics on error
            return PerformanceMetrics(
                timestamp=datetime.utcnow(),
                cpu_percent=0.0, memory_percent=0.0, disk_usage_percent=0.0,
                network_bytes_sent=0, network_bytes_recv=0,
                process_count=0, thread_count=0,
                collection_rate=0.0, error_rate=0.0, api_response_time=0.0,
                queue_size=0, worker_utilization=0.0
            )
            
    def _get_collection_rate(self) -> float:
        """Get current collection rate (placeholder - would integrate with main system)."""
        # This would integrate with the actual collection system
        return 45.0  # Default placeholder value
        
    def _get_error_rate(self) -> float:
        """Get current error rate (placeholder)."""
        return 0.05  # Default placeholder value
        
    def _get_api_response_time(self) -> float:
        """Get current API response time (placeholder)."""
        return 2.5  # Default placeholder value
        
    def _get_queue_size(self) -> int:
        """Get current queue size (placeholder)."""
        return 150  # Default placeholder value
        
    def _get_worker_utilization(self) -> float:
        """Get current worker utilization (placeholder)."""
        return 0.75  # Default placeholder value
        
    def _store_metrics(self, metrics: PerformanceMetrics) -> None:
        """Store metrics in database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO performance_metrics (
                    timestamp, cpu_percent, memory_percent, disk_usage_percent,
                    network_bytes_sent, network_bytes_recv, process_count, thread_count,
                    collection_rate, error_rate, api_response_time, queue_size, worker_utilization
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics.timestamp.isoformat(),
                metrics.cpu_percent,
                metrics.memory_percent,
                metrics.disk_usage_percent,
                metrics.network_bytes_sent,
                metrics.network_bytes_recv,
                metrics.process_count,
                metrics.thread_count,
                metrics.collection_rate,
                metrics.error_rate,
                metrics.api_response_time,
                metrics.queue_size,
                metrics.worker_utilization
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error storing metrics: {e}")
            
    def _update_trends(self, metrics: PerformanceMetrics) -> None:
        """Update performance trends."""
        self.performance_trends['cpu'].append((metrics.timestamp, metrics.cpu_percent))
        self.performance_trends['memory'].append((metrics.timestamp, metrics.memory_percent))
        self.performance_trends['disk'].append((metrics.timestamp, metrics.disk_usage_percent))
        self.performance_trends['collection_rate'].append((metrics.timestamp, metrics.collection_rate))
        self.performance_trends['error_rate'].append((metrics.timestamp, metrics.error_rate))
        self.performance_trends['api_response_time'].append((metrics.timestamp, metrics.api_response_time))
        
    async def _perform_capacity_analysis(self) -> None:
        """Perform comprehensive capacity analysis."""
        try:
            self.logger.info("Performing capacity analysis...")
            
            recommendations = []
            
            # Analyze each resource
            for resource in ['cpu', 'memory', 'disk']:
                recommendation = await self._analyze_resource_capacity(resource)
                if recommendation:
                    recommendations.append(recommendation)
                    
            # Store recommendations
            for rec in recommendations:
                self._store_recommendation(rec)
                
            self.stats['last_capacity_analysis'] = datetime.utcnow()
            self.stats['recommendations_generated'] += len(recommendations)
            
            self.logger.info(f"Capacity analysis completed - {len(recommendations)} recommendations generated")
            
        except Exception as e:
            self.logger.error(f"Error in capacity analysis: {e}")
            
    async def _analyze_resource_capacity(self, resource: str) -> Optional[CapacityRecommendation]:
        """Analyze capacity for a specific resource."""
        if resource not in self.performance_trends or len(self.performance_trends[resource]) < 10:
            return None
            
        try:
            # Get recent data
            recent_data = list(self.performance_trends[resource])[-100:]  # Last 100 samples
            values = [value for _, value in recent_data]
            
            if not values:
                return None
                
            # Calculate current usage
            current_usage = statistics.mean(values[-10:])  # Average of last 10 samples
            
            # Calculate growth trend
            if len(values) >= 20:
                # Linear regression for trend analysis
                x = np.arange(len(values))
                y = np.array(values)
                
                # Calculate slope (growth rate)
                slope, intercept = np.polyfit(x, y, 1)
                
                # Project future usage
                future_samples = self.projection_days * (24 * 3600 // self.collection_interval)
                projected_usage = current_usage + (slope * future_samples)
                
                # Determine recommendation
                recommendation = self._generate_capacity_recommendation(
                    resource, current_usage, projected_usage, slope
                )
                
                return recommendation
                
        except Exception as e:
            self.logger.error(f"Error analyzing {resource} capacity: {e}")
            
        return None
        
    def _generate_capacity_recommendation(self, resource: str, current: float, 
                                        projected: float, growth_rate: float) -> CapacityRecommendation:
        """Generate capacity recommendation based on analysis."""
        # Determine thresholds
        warning_threshold = self.thresholds.get(f'{resource}_warning', 80.0)
        critical_threshold = self.thresholds.get(f'{resource}_critical', 90.0)
        
        # Determine priority and recommendation
        if projected >= critical_threshold:
            priority = 'critical'
            if growth_rate > 0:
                timeline = self._calculate_timeline_to_threshold(current, critical_threshold, growth_rate)
                recommendation = f"Immediate {resource} capacity expansion required. " \
                               f"Will reach critical threshold in {timeline}."
                cost_impact = 'high'
            else:
                timeline = 'immediate'
                recommendation = f"Current {resource} usage is approaching critical levels."
                cost_impact = 'medium'
                
        elif projected >= warning_threshold:
            priority = 'high'
            timeline = self._calculate_timeline_to_threshold(current, warning_threshold, growth_rate)
            recommendation = f"Plan {resource} capacity expansion. " \
                           f"Will reach warning threshold in {timeline}."
            cost_impact = 'medium'
            
        elif growth_rate > 0.1:  # Growing more than 0.1% per sample
            priority = 'medium'
            timeline = self._calculate_timeline_to_threshold(current, warning_threshold, growth_rate)
            recommendation = f"Monitor {resource} usage closely. " \
                           f"Growing trend detected, will reach warning threshold in {timeline}."
            cost_impact = 'low'
            
        else:
            priority = 'low'
            timeline = 'stable'
            recommendation = f"{resource.capitalize()} usage is stable and within acceptable limits."
            cost_impact = 'none'
            
        return CapacityRecommendation(
            resource=resource,
            current_usage=current,
            projected_usage=projected,
            recommendation=recommendation,
            priority=priority,
            estimated_timeline=timeline,
            cost_impact=cost_impact
        )
        
    def _calculate_timeline_to_threshold(self, current: float, threshold: float, 
                                       growth_rate: float) -> str:
        """Calculate timeline to reach threshold."""
        if growth_rate <= 0:
            return 'stable'
            
        # Calculate samples needed to reach threshold
        samples_to_threshold = (threshold - current) / growth_rate
        
        # Convert to time
        seconds_to_threshold = samples_to_threshold * self.collection_interval
        
        if seconds_to_threshold < 3600:  # Less than 1 hour
            return f"{int(seconds_to_threshold / 60)} minutes"
        elif seconds_to_threshold < 86400:  # Less than 1 day
            return f"{int(seconds_to_threshold / 3600)} hours"
        else:
            return f"{int(seconds_to_threshold / 86400)} days"
            
    def _store_recommendation(self, recommendation: CapacityRecommendation) -> None:
        """Store capacity recommendation in database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO capacity_recommendations (
                    timestamp, resource, current_usage, projected_usage,
                    recommendation, priority, estimated_timeline, cost_impact
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.utcnow().isoformat(),
                recommendation.resource,
                recommendation.current_usage,
                recommendation.projected_usage,
                recommendation.recommendation,
                recommendation.priority,
                recommendation.estimated_timeline,
                recommendation.cost_impact
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error storing recommendation: {e}")
            
    def get_current_performance(self) -> Dict[str, Any]:
        """Get current performance snapshot."""
        if not self.metrics_history:
            return {'error': 'No metrics available'}
            
        latest = self.metrics_history[-1]
        
        return {
            'timestamp': latest.timestamp.isoformat(),
            'cpu_percent': latest.cpu_percent,
            'memory_percent': latest.memory_percent,
            'disk_usage_percent': latest.disk_usage_percent,
            'collection_rate': latest.collection_rate,
            'error_rate': latest.error_rate,
            'api_response_time': latest.api_response_time,
            'queue_size': latest.queue_size,
            'worker_utilization': latest.worker_utilization,
            'process_count': latest.process_count,
            'thread_count': latest.thread_count,
            'network_io': {
                'bytes_sent': latest.network_bytes_sent,
                'bytes_recv': latest.network_bytes_recv
            }
        }
        
    def get_performance_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance trends for specified time period."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        trends = {}
        
        for metric_name, history in self.performance_trends.items():
            recent_data = [(ts, value) for ts, value in history if ts >= cutoff_time]
            
            if len(recent_data) < 2:
                trends[metric_name] = {'status': 'insufficient_data'}
                continue
                
            values = [value for _, value in recent_data]
            
            # Calculate trend statistics
            current = values[-1]
            average = statistics.mean(values)
            minimum = min(values)
            maximum = max(values)
            
            # Calculate trend direction
            if len(values) >= 10:
                recent_avg = statistics.mean(values[-10:])
                older_avg = statistics.mean(values[-20:-10]) if len(values) >= 20 else average
                
                if recent_avg > older_avg * 1.05:
                    trend_direction = 'increasing'
                elif recent_avg < older_avg * 0.95:
                    trend_direction = 'decreasing'
                else:
                    trend_direction = 'stable'
            else:
                trend_direction = 'unknown'
                
            trends[metric_name] = {
                'current': current,
                'average': average,
                'minimum': minimum,
                'maximum': maximum,
                'trend_direction': trend_direction,
                'samples': len(values),
                'time_span_hours': hours
            }
            
        return trends
        
    def get_capacity_recommendations(self, priority: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get capacity recommendations."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if priority:
                cursor.execute('''
                    SELECT * FROM capacity_recommendations 
                    WHERE priority = ? 
                    ORDER BY timestamp DESC 
                    LIMIT 50
                ''', (priority,))
            else:
                cursor.execute('''
                    SELECT * FROM capacity_recommendations 
                    ORDER BY timestamp DESC 
                    LIMIT 50
                ''')
                
            rows = cursor.fetchall()
            conn.close()
            
            recommendations = []
            for row in rows:
                recommendations.append({
                    'id': row[0],
                    'timestamp': row[1],
                    'resource': row[2],
                    'current_usage': row[3],
                    'projected_usage': row[4],
                    'recommendation': row[5],
                    'priority': row[6],
                    'estimated_timeline': row[7],
                    'cost_impact': row[8]
                })
                
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error getting recommendations: {e}")
            return []
            
    def generate_capacity_report(self) -> Dict[str, Any]:
        """Generate comprehensive capacity planning report."""
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'monitoring_stats': self.stats,
            'current_performance': self.get_current_performance(),
            'performance_trends': self.get_performance_trends(24),
            'capacity_recommendations': {
                'critical': self.get_capacity_recommendations('critical'),
                'high': self.get_capacity_recommendations('high'),
                'medium': self.get_capacity_recommendations('medium'),
                'low': self.get_capacity_recommendations('low')
            },
            'resource_utilization': self._calculate_resource_utilization(),
            'optimization_opportunities': self._identify_optimization_opportunities()
        }
        
        return report
        
    def _calculate_resource_utilization(self) -> Dict[str, Any]:
        """Calculate resource utilization statistics."""
        if not self.metrics_history:
            return {}
            
        # Get recent metrics (last hour)
        recent_metrics = [m for m in self.metrics_history 
                         if (datetime.utcnow() - m.timestamp).total_seconds() < 3600]
        
        if not recent_metrics:
            return {}
            
        utilization = {}
        
        # CPU utilization
        cpu_values = [m.cpu_percent for m in recent_metrics]
        utilization['cpu'] = {
            'average': statistics.mean(cpu_values),
            'peak': max(cpu_values),
            'minimum': min(cpu_values),
            'efficiency': self._calculate_efficiency(cpu_values, 'cpu')
        }
        
        # Memory utilization
        memory_values = [m.memory_percent for m in recent_metrics]
        utilization['memory'] = {
            'average': statistics.mean(memory_values),
            'peak': max(memory_values),
            'minimum': min(memory_values),
            'efficiency': self._calculate_efficiency(memory_values, 'memory')
        }
        
        # Collection performance
        collection_rates = [m.collection_rate for m in recent_metrics]
        utilization['collection'] = {
            'average_rate': statistics.mean(collection_rates),
            'peak_rate': max(collection_rates),
            'minimum_rate': min(collection_rates),
            'consistency': 1.0 - (statistics.stdev(collection_rates) / statistics.mean(collection_rates))
        }
        
        return utilization
        
    def _calculate_efficiency(self, values: List[float], resource: str) -> float:
        """Calculate resource efficiency score."""
        if not values:
            return 0.0
            
        # Efficiency is based on consistent utilization without peaks
        average = statistics.mean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0
        
        # Ideal utilization ranges
        ideal_ranges = {
            'cpu': (60, 80),
            'memory': (50, 75),
            'disk': (40, 70)
        }
        
        ideal_min, ideal_max = ideal_ranges.get(resource, (50, 80))
        
        # Calculate efficiency score
        if ideal_min <= average <= ideal_max:
            base_score = 1.0
        else:
            # Penalize for being outside ideal range
            if average < ideal_min:
                base_score = average / ideal_min
            else:
                base_score = ideal_max / average
                
        # Penalize for high variability
        variability_penalty = min(stdev / 20, 0.5)  # Max 50% penalty for high variability
        
        efficiency = max(0.0, base_score - variability_penalty)
        
        return efficiency
        
    def _identify_optimization_opportunities(self) -> List[Dict[str, Any]]:
        """Identify performance optimization opportunities."""
        opportunities = []
        
        if not self.metrics_history:
            return opportunities
            
        recent_metrics = list(self.metrics_history)[-100:]  # Last 100 samples
        
        # Check for CPU optimization opportunities
        cpu_values = [m.cpu_percent for m in recent_metrics]
        if cpu_values:
            avg_cpu = statistics.mean(cpu_values)
            max_cpu = max(cpu_values)
            
            if max_cpu > 90 and avg_cpu < 60:
                opportunities.append({
                    'type': 'cpu_spike_optimization',
                    'description': 'CPU spikes detected with low average usage',
                    'recommendation': 'Consider load balancing or process scheduling optimization',
                    'potential_impact': 'medium'
                })
                
            elif avg_cpu < 30:
                opportunities.append({
                    'type': 'cpu_underutilization',
                    'description': 'CPU is underutilized',
                    'recommendation': 'Consider increasing worker count or processing batch sizes',
                    'potential_impact': 'high'
                })
                
        # Check for memory optimization opportunities
        memory_values = [m.memory_percent for m in recent_metrics]
        if memory_values:
            avg_memory = statistics.mean(memory_values)
            
            if avg_memory > 85:
                opportunities.append({
                    'type': 'memory_pressure',
                    'description': 'High memory usage detected',
                    'recommendation': 'Consider memory optimization or capacity increase',
                    'potential_impact': 'high'
                })
                
        # Check for collection rate optimization
        collection_rates = [m.collection_rate for m in recent_metrics if m.collection_rate > 0]
        if collection_rates:
            avg_rate = statistics.mean(collection_rates)
            
            if avg_rate < self.thresholds.get('collection_rate_min', 30):
                opportunities.append({
                    'type': 'collection_rate_optimization',
                    'description': 'Collection rate below target',
                    'recommendation': 'Investigate API performance, network issues, or worker efficiency',
                    'potential_impact': 'high'
                })
                
        # Check for error rate optimization
        error_rates = [m.error_rate for m in recent_metrics if m.error_rate >= 0]
        if error_rates:
            avg_error_rate = statistics.mean(error_rates)
            
            if avg_error_rate > self.thresholds.get('error_rate_max', 0.1):
                opportunities.append({
                    'type': 'error_rate_optimization',
                    'description': 'High error rate detected',
                    'recommendation': 'Review error patterns and implement better retry mechanisms',
                    'potential_impact': 'high'
                })
                
        return opportunities
        
    def export_metrics(self, hours: int = 24, format: str = 'json') -> str:
        """Export metrics data."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM performance_metrics 
                WHERE timestamp >= ? 
                ORDER BY timestamp
            ''', (cutoff_time.isoformat(),))
            
            rows = cursor.fetchall()
            conn.close()
            
            if format == 'json':
                metrics_data = []
                for row in rows:
                    metrics_data.append({
                        'timestamp': row[1],
                        'cpu_percent': row[2],
                        'memory_percent': row[3],
                        'disk_usage_percent': row[4],
                        'network_bytes_sent': row[5],
                        'network_bytes_recv': row[6],
                        'process_count': row[7],
                        'thread_count': row[8],
                        'collection_rate': row[9],
                        'error_rate': row[10],
                        'api_response_time': row[11],
                        'queue_size': row[12],
                        'worker_utilization': row[13]
                    })
                    
                return json.dumps({
                    'export_time': datetime.utcnow().isoformat(),
                    'time_period_hours': hours,
                    'total_samples': len(metrics_data),
                    'metrics': metrics_data
                }, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error exporting metrics: {e}")
            return json.dumps({'error': str(e)})


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Performance and Capacity Monitor')
    parser.add_argument('--action', choices=[
        'start', 'status', 'trends', 'recommendations', 'report', 'export'
    ], default='status', help='Action to perform')
    parser.add_argument('--hours', type=int, default=24, help='Time period in hours')
    parser.add_argument('--priority', choices=['critical', 'high', 'medium', 'low'],
                       help='Filter recommendations by priority')
    parser.add_argument('--output', help='Output file path')
    parser.add_argument('--config', help='Configuration file path')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    config = None
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = json.load(f)
            
    # Create monitor
    monitor = PerformanceCapacityMonitor(config)
    
    try:
        if args.action == 'start':
            monitor.start_monitoring()
            print("Started performance monitoring. Press Ctrl+C to stop...")
            try:
                while monitor.is_monitoring:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nStopping monitoring...")
                monitor.stop_monitoring()
            return
            
        elif args.action == 'status':
            results = monitor.get_current_performance()
        elif args.action == 'trends':
            results = monitor.get_performance_trends(args.hours)
        elif args.action == 'recommendations':
            results = monitor.get_capacity_recommendations(args.priority)
        elif args.action == 'report':
            results = monitor.generate_capacity_report()
        elif args.action == 'export':
            results = monitor.export_metrics(args.hours)
        else:
            results = {'error': f'Unknown action: {args.action}'}
            
        # Output results
        if args.output:
            with open(args.output, 'w') as f:
                if isinstance(results, str):
                    f.write(results)
                else:
                    json.dump(results, f, indent=2)
            print(f"Results saved to {args.output}")
        else:
            if isinstance(results, str):
                print(results)
            else:
                print(json.dumps(results, indent=2))
                
    except Exception as e:
        logging.error(f"Error: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    exit(main())