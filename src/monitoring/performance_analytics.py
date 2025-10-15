"""
Performance Analytics
=====================

Advanced performance tracking and analytics for the AI Trading System.
Includes AI performance, execution performance, optimization performance,
and comprehensive dashboards.
"""

import asyncio
import logging
import statistics
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import time

logger = logging.getLogger(__name__)


class PerformanceMetric(Enum):
    """Performance metric types."""
    AI_ACCURACY = "ai_accuracy"
    AI_LATENCY = "ai_latency"
    EXECUTION_SPEED = "execution_speed"
    EXECUTION_SLIPPAGE = "execution_slippage"
    OPTIMIZATION_IMPROVEMENT = "optimization_improvement"
    PARAMETER_CHANGE = "parameter_change"
    LEARNING_PROGRESS = "learning_progress"


@dataclass
class AIPerformanceMetrics:
    """AI performance metrics data structure."""
    timestamp: datetime
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    latency: float
    throughput: float
    error_rate: float
    confidence_score: float


@dataclass
class ExecutionPerformanceMetrics:
    """Execution performance metrics data structure."""
    timestamp: datetime
    order_type: str
    execution_time: float
    slippage: float
    fill_rate: float
    rejection_rate: float
    average_size: float
    total_volume: float


@dataclass
class OptimizationPerformanceMetrics:
    """Optimization performance metrics data structure."""
    timestamp: datetime
    optimization_type: str
    improvement_percentage: float
    parameter_changes: Dict[str, Any]
    performance_before: float
    performance_after: float
    optimization_time: float
    convergence_iterations: int


@dataclass
class PerformanceDashboard:
    """Performance dashboard data structure."""
    timestamp: datetime
    overall_score: float
    ai_performance: Dict[str, Any]
    execution_performance: Dict[str, Any]
    optimization_performance: Dict[str, Any]
    trends: Dict[str, Any]
    recommendations: List[str]


class PerformanceAnalytics:
    """Advanced performance tracking and analytics."""
    
    def __init__(self):
        self.ai_metrics: List[AIPerformanceMetrics] = []
        self.execution_metrics: List[ExecutionPerformanceMetrics] = []
        self.optimization_metrics: List[OptimizationPerformanceMetrics] = []
        self.max_history_size = 10000  # Keep last 10,000 entries
        
        # Phase 1: Real-time phase duration tracking
        self.phase_durations: Dict[str, List[float]] = {}
        self.current_phase_timers: Dict[str, float] = {}
        self.phase_step_labels = [
            "ingest", "features", "factors", "scoring", "ensemble", 
            "sizing", "orders", "persist", "dashboard"
        ]
    
    def start_phase_timer(self, phase_name: str) -> None:
        """Start timing a phase."""
        self.current_phase_timers[phase_name] = time.time()
        logger.debug(f"Started phase timer: {phase_name}")
    
    def end_phase_timer(self, phase_name: str) -> float:
        """End timing a phase and return duration in seconds."""
        if phase_name not in self.current_phase_timers:
            logger.warning(f"Phase timer not found for: {phase_name}")
            return 0.0
        
        start_time = self.current_phase_timers[phase_name]
        duration = time.time() - start_time
        
        # Store duration
        if phase_name not in self.phase_durations:
            self.phase_durations[phase_name] = []
        
        self.phase_durations[phase_name].append(duration)
        
        # Trim to max history size
        if len(self.phase_durations[phase_name]) > self.max_history_size:
            self.phase_durations[phase_name] = self.phase_durations[phase_name][-self.max_history_size:]
        
        # Remove from current timers
        del self.current_phase_timers[phase_name]
        
        logger.debug(f"Ended phase timer: {phase_name} - {duration:.3f}s")
        return duration
    
    def get_phase_duration_stats(self, phase_name: str) -> Dict[str, Any]:
        """Get statistics for a specific phase duration."""
        if phase_name not in self.phase_durations or not self.phase_durations[phase_name]:
            return {
                "phase_name": phase_name,
                "count": 0,
                "average": 0.0,
                "min": 0.0,
                "max": 0.0,
                "p95": 0.0,
                "p99": 0.0
            }
        
        durations = self.phase_durations[phase_name]
        
        return {
            "phase_name": phase_name,
            "count": len(durations),
            "average": statistics.mean(durations),
            "min": min(durations),
            "max": max(durations),
            "p95": self._percentile(durations, 95),
            "p99": self._percentile(durations, 99),
            "recent_trend": self._calculate_trend(durations[-10:]) if len(durations) >= 10 else "insufficient_data"
        }
    
    def get_all_phase_stats(self) -> Dict[str, Any]:
        """Get statistics for all phases."""
        all_stats = {}
        
        for phase_name in self.phase_durations:
            all_stats[phase_name] = self.get_phase_duration_stats(phase_name)
        
        # Add overall statistics
        all_durations = []
        for durations in self.phase_durations.values():
            all_durations.extend(durations)
        
        if all_durations:
            all_stats["overall"] = {
                "total_phases": len(self.phase_durations),
                "total_executions": len(all_durations),
                "average_duration": statistics.mean(all_durations),
                "min_duration": min(all_durations),
                "max_duration": max(all_durations),
                "p95_duration": self._percentile(all_durations, 95),
                "p99_duration": self._percentile(all_durations, 99)
            }
        
        return all_stats
    
    async def track_phase_performance(self, phase_name: str, step_label: str = None) -> Dict[str, Any]:
        """Track performance of a specific phase with optional step."""
        try:
            # Start phase timer if not already started
            if phase_name not in self.current_phase_timers:
                self.start_phase_timer(phase_name)
            
            # Get current phase stats
            phase_stats = self.get_phase_duration_stats(phase_name)
            
            # Add step information if provided
            if step_label and step_label in self.phase_step_labels:
                phase_stats["current_step"] = step_label
                phase_stats["step_in_progress"] = True
            
            logger.info(f"Phase performance tracked: {phase_name} - {phase_stats['count']} executions, avg: {phase_stats['average']:.3f}s")
            return phase_stats
            
        except Exception as e:
            logger.error(f"Error tracking phase performance: {e}")
            return {"error": str(e)}
    
    async def generate_phase_performance_dashboard(self) -> Dict[str, Any]:
        """Generate comprehensive phase performance dashboard."""
        try:
            all_stats = self.get_all_phase_stats()
            
            # Calculate performance insights
            insights = []
            
            # Find slowest phases
            phase_averages = {
                phase: stats["average"] 
                for phase, stats in all_stats.items() 
                if phase != "overall" and stats["count"] > 0
            }
            
            if phase_averages:
                slowest_phase = max(phase_averages, key=phase_averages.get)
                fastest_phase = min(phase_averages, key=phase_averages.get)
                
                insights.append(f"Slowest phase: {slowest_phase} ({phase_averages[slowest_phase]:.3f}s avg)")
                insights.append(f"Fastest phase: {fastest_phase} ({phase_averages[fastest_phase]:.3f}s avg)")
            
            # Find phases with high variability
            for phase, stats in all_stats.items():
                if phase != "overall" and stats["count"] > 5:
                    variability = (stats["max"] - stats["min"]) / stats["average"] if stats["average"] > 0 else 0
                    if variability > 2.0:  # High variability threshold
                        insights.append(f"High variability in {phase}: {variability:.1f}x range")
            
            dashboard = {
                "timestamp": datetime.now(),
                "phase_statistics": all_stats,
                "insights": insights,
                "active_phases": list(self.current_phase_timers.keys()),
                "total_phases_tracked": len(self.phase_durations)
            }
            
            logger.info(f"Phase performance dashboard generated: {len(all_stats)} phases tracked")
            return dashboard
            
        except Exception as e:
            logger.error(f"Error generating phase performance dashboard: {e}")
            return {"error": str(e)}
        
    async def track_ai_performance(self) -> Dict[str, Any]:
        """Track AI performance metrics."""
        try:
            # Mock AI performance data (in real implementation, this would come from actual AI models)
            ai_metrics = AIPerformanceMetrics(
                timestamp=datetime.now(),
                model_name="ensemble_model",
                accuracy=0.85,
                precision=0.82,
                recall=0.88,
                f1_score=0.85,
                latency=0.15,  # 150ms
                throughput=100.0,  # 100 predictions per second
                error_rate=0.02,  # 2% error rate
                confidence_score=0.87
            )
            
            # Store metrics
            self.ai_metrics.append(ai_metrics)
            self._trim_ai_metrics()
            
            # Calculate performance summary
            performance_summary = {
                "timestamp": ai_metrics.timestamp,
                "model_name": ai_metrics.model_name,
                "accuracy": ai_metrics.accuracy,
                "precision": ai_metrics.precision,
                "recall": ai_metrics.recall,
                "f1_score": ai_metrics.f1_score,
                "latency": ai_metrics.latency,
                "throughput": ai_metrics.throughput,
                "error_rate": ai_metrics.error_rate,
                "confidence_score": ai_metrics.confidence_score,
                "performance_score": self._calculate_ai_performance_score(ai_metrics)
            }
            
            logger.info(f"AI performance tracked: {ai_metrics.model_name} - Accuracy: {ai_metrics.accuracy:.2%}")
            return performance_summary
            
        except Exception as e:
            logger.error(f"Error tracking AI performance: {e}")
            return {"error": str(e)}
    
    def _calculate_ai_performance_score(self, metrics: AIPerformanceMetrics) -> float:
        """Calculate overall AI performance score."""
        try:
            # Weighted performance score
            weights = {
                "accuracy": 0.3,
                "precision": 0.2,
                "recall": 0.2,
                "f1_score": 0.2,
                "latency": 0.1
            }
            
            # Normalize latency (lower is better)
            latency_score = max(0, 1 - (metrics.latency / 1.0))  # Normalize to 0-1
            
            score = (
                weights["accuracy"] * metrics.accuracy +
                weights["precision"] * metrics.precision +
                weights["recall"] * metrics.recall +
                weights["f1_score"] * metrics.f1_score +
                weights["latency"] * latency_score
            )
            
            return min(1.0, max(0.0, score))
            
        except Exception as e:
            logger.error(f"Error calculating AI performance score: {e}")
            return 0.0
    
    async def track_model_accuracy(self) -> Dict[str, Any]:
        """Track model accuracy metrics."""
        try:
            # Get recent AI metrics
            recent_metrics = self.ai_metrics[-100:] if self.ai_metrics else []
            
            if not recent_metrics:
                return {"error": "No AI metrics available"}
            
            # Calculate accuracy statistics
            accuracies = [m.accuracy for m in recent_metrics]
            precisions = [m.precision for m in recent_metrics]
            recalls = [m.recall for m in recent_metrics]
            f1_scores = [m.f1_score for m in recent_metrics]
            
            accuracy_summary = {
                "timestamp": datetime.now(),
                "sample_size": len(recent_metrics),
                "average_accuracy": statistics.mean(accuracies),
                "accuracy_std": statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
                "min_accuracy": min(accuracies),
                "max_accuracy": max(accuracies),
                "average_precision": statistics.mean(precisions),
                "average_recall": statistics.mean(recalls),
                "average_f1_score": statistics.mean(f1_scores),
                "accuracy_trend": self._calculate_trend(accuracies)
            }
            
            logger.info(f"Model accuracy tracked: {accuracy_summary['average_accuracy']:.2%}")
            return accuracy_summary
            
        except Exception as e:
            logger.error(f"Error tracking model accuracy: {e}")
            return {"error": str(e)}
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a list of values."""
        try:
            if len(values) < 2:
                return "insufficient_data"
            
            # Simple linear trend calculation
            first_half = values[:len(values)//2]
            second_half = values[len(values)//2:]
            
            first_avg = statistics.mean(first_half)
            second_avg = statistics.mean(second_half)
            
            if second_avg > first_avg * 1.05:
                return "improving"
            elif second_avg < first_avg * 0.95:
                return "declining"
            else:
                return "stable"
                
        except Exception as e:
            logger.error(f"Error calculating trend: {e}")
            return "unknown"
    
    async def track_decision_quality(self) -> Dict[str, Any]:
        """Track decision quality metrics."""
        try:
            # Mock decision quality data (in real implementation, this would come from actual trading decisions)
            decision_quality = {
                "timestamp": datetime.now(),
                "total_decisions": 1000,
                "correct_decisions": 850,
                "incorrect_decisions": 150,
                "accuracy": 0.85,
                "confidence_distribution": {
                    "high_confidence": 0.6,
                    "medium_confidence": 0.3,
                    "low_confidence": 0.1
                },
                "decision_types": {
                    "buy": 0.4,
                    "sell": 0.3,
                    "hold": 0.3
                },
                "average_confidence": 0.82,
                "decision_latency": 0.12  # 120ms average
            }
            
            logger.info(f"Decision quality tracked: {decision_quality['accuracy']:.2%} accuracy")
            return decision_quality
            
        except Exception as e:
            logger.error(f"Error tracking decision quality: {e}")
            return {"error": str(e)}
    
    async def track_execution_metrics(self) -> Dict[str, Any]:
        """Track execution performance metrics."""
        try:
            # Mock execution metrics (in real implementation, this would come from actual order execution)
            execution_metrics = ExecutionPerformanceMetrics(
                timestamp=datetime.now(),
                order_type="market",
                execution_time=0.05,  # 50ms
                slippage=0.001,  # 0.1%
                fill_rate=0.98,  # 98%
                rejection_rate=0.02,  # 2%
                average_size=100.0,
                total_volume=10000.0
            )
            
            # Store metrics
            self.execution_metrics.append(execution_metrics)
            self._trim_execution_metrics()
            
            # Calculate execution summary
            execution_summary = {
                "timestamp": execution_metrics.timestamp,
                "order_type": execution_metrics.order_type,
                "execution_time": execution_metrics.execution_time,
                "slippage": execution_metrics.slippage,
                "fill_rate": execution_metrics.fill_rate,
                "rejection_rate": execution_metrics.rejection_rate,
                "average_size": execution_metrics.average_size,
                "total_volume": execution_metrics.total_volume,
                "execution_score": self._calculate_execution_score(execution_metrics)
            }
            
            logger.info(f"Execution metrics tracked: {execution_metrics.execution_time:.3f}s execution time")
            return execution_summary
            
        except Exception as e:
            logger.error(f"Error tracking execution metrics: {e}")
            return {"error": str(e)}
    
    def _calculate_execution_score(self, metrics: ExecutionPerformanceMetrics) -> float:
        """Calculate execution performance score."""
        try:
            # Weighted execution score
            weights = {
                "execution_time": 0.3,
                "slippage": 0.3,
                "fill_rate": 0.2,
                "rejection_rate": 0.2
            }
            
            # Normalize metrics (lower is better for time, slippage, rejection rate)
            time_score = max(0, 1 - (metrics.execution_time / 1.0))  # Normalize to 0-1
            slippage_score = max(0, 1 - (metrics.slippage / 0.01))  # Normalize to 0-1
            fill_rate_score = metrics.fill_rate  # Already 0-1
            rejection_score = max(0, 1 - (metrics.rejection_rate / 0.1))  # Normalize to 0-1
            
            score = (
                weights["execution_time"] * time_score +
                weights["slippage"] * slippage_score +
                weights["fill_rate"] * fill_rate_score +
                weights["rejection_rate"] * rejection_score
            )
            
            return min(1.0, max(0.0, score))
            
        except Exception as e:
            logger.error(f"Error calculating execution score: {e}")
            return 0.0
    
    async def track_slippage(self) -> Dict[str, Any]:
        """Track execution slippage metrics."""
        try:
            # Get recent execution metrics
            recent_metrics = self.execution_metrics[-100:] if self.execution_metrics else []
            
            if not recent_metrics:
                return {"error": "No execution metrics available"}
            
            # Calculate slippage statistics
            slippages = [m.slippage for m in recent_metrics]
            
            slippage_summary = {
                "timestamp": datetime.now(),
                "sample_size": len(recent_metrics),
                "average_slippage": statistics.mean(slippages),
                "slippage_std": statistics.stdev(slippages) if len(slippages) > 1 else 0.0,
                "min_slippage": min(slippages),
                "max_slippage": max(slippages),
                "slippage_trend": self._calculate_trend(slippages),
                "slippage_percentiles": {
                    "p50": statistics.median(slippages),
                    "p90": self._percentile(slippages, 90),
                    "p95": self._percentile(slippages, 95),
                    "p99": self._percentile(slippages, 99)
                }
            }
            
            logger.info(f"Slippage tracked: {slippage_summary['average_slippage']:.4f} average")
            return slippage_summary
            
        except Exception as e:
            logger.error(f"Error tracking slippage: {e}")
            return {"error": str(e)}
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile of a list of values."""
        try:
            if not values:
                return 0.0
            
            sorted_values = sorted(values)
            index = int((percentile / 100) * len(sorted_values))
            index = min(index, len(sorted_values) - 1)
            
            return sorted_values[index]
            
        except Exception as e:
            logger.error(f"Error calculating percentile: {e}")
            return 0.0
    
    async def track_latency(self) -> Dict[str, Any]:
        """Track system latency metrics."""
        try:
            # Get recent AI and execution metrics
            recent_ai = self.ai_metrics[-100:] if self.ai_metrics else []
            recent_execution = self.execution_metrics[-100:] if self.execution_metrics else []
            
            # Calculate latency statistics
            ai_latencies = [m.latency for m in recent_ai]
            execution_times = [m.execution_time for m in recent_execution]
            
            latency_summary = {
                "timestamp": datetime.now(),
                "ai_latency": {
                    "average": statistics.mean(ai_latencies) if ai_latencies else 0.0,
                    "min": min(ai_latencies) if ai_latencies else 0.0,
                    "max": max(ai_latencies) if ai_latencies else 0.0,
                    "p95": self._percentile(ai_latencies, 95) if ai_latencies else 0.0
                },
                "execution_latency": {
                    "average": statistics.mean(execution_times) if execution_times else 0.0,
                    "min": min(execution_times) if execution_times else 0.0,
                    "max": max(execution_times) if execution_times else 0.0,
                    "p95": self._percentile(execution_times, 95) if execution_times else 0.0
                },
                "total_latency": {
                    "average": (statistics.mean(ai_latencies) + statistics.mean(execution_times)) if ai_latencies and execution_times else 0.0,
                    "trend": self._calculate_trend(ai_latencies + execution_times) if ai_latencies or execution_times else "unknown"
                }
            }
            
            logger.info(f"Latency tracked: AI {latency_summary['ai_latency']['average']:.3f}s, Execution {latency_summary['execution_latency']['average']:.3f}s")
            return latency_summary
            
        except Exception as e:
            logger.error(f"Error tracking latency: {e}")
            return {"error": str(e)}
    
    async def track_optimization_results(self) -> Dict[str, Any]:
        """Track optimization performance results."""
        try:
            # Mock optimization results (in real implementation, this would come from actual optimization)
            optimization_metrics = OptimizationPerformanceMetrics(
                timestamp=datetime.now(),
                optimization_type="parameter_tuning",
                improvement_percentage=0.15,  # 15% improvement
                parameter_changes={
                    "learning_rate": 0.001,
                    "batch_size": 32,
                    "epochs": 100
                },
                performance_before=0.75,
                performance_after=0.86,
                optimization_time=300.0,  # 5 minutes
                convergence_iterations=50
            )
            
            # Store metrics
            self.optimization_metrics.append(optimization_metrics)
            self._trim_optimization_metrics()
            
            # Calculate optimization summary
            optimization_summary = {
                "timestamp": optimization_metrics.timestamp,
                "optimization_type": optimization_metrics.optimization_type,
                "improvement_percentage": optimization_metrics.improvement_percentage,
                "parameter_changes": optimization_metrics.parameter_changes,
                "performance_before": optimization_metrics.performance_before,
                "performance_after": optimization_metrics.performance_after,
                "optimization_time": optimization_metrics.optimization_time,
                "convergence_iterations": optimization_metrics.convergence_iterations,
                "optimization_score": self._calculate_optimization_score(optimization_metrics)
            }
            
            logger.info(f"Optimization results tracked: {optimization_metrics.improvement_percentage:.2%} improvement")
            return optimization_summary
            
        except Exception as e:
            logger.error(f"Error tracking optimization results: {e}")
            return {"error": str(e)}
    
    def _calculate_optimization_score(self, metrics: OptimizationPerformanceMetrics) -> float:
        """Calculate optimization performance score."""
        try:
            # Weighted optimization score
            weights = {
                "improvement": 0.5,
                "time_efficiency": 0.3,
                "convergence": 0.2
            }
            
            # Normalize metrics
            improvement_score = min(1.0, metrics.improvement_percentage / 0.5)  # Normalize to 0-1
            time_score = max(0, 1 - (metrics.optimization_time / 3600))  # Normalize to 0-1 (1 hour max)
            convergence_score = max(0, 1 - (metrics.convergence_iterations / 100))  # Normalize to 0-1
            
            score = (
                weights["improvement"] * improvement_score +
                weights["time_efficiency"] * time_score +
                weights["convergence"] * convergence_score
            )
            
            return min(1.0, max(0.0, score))
            
        except Exception as e:
            logger.error(f"Error calculating optimization score: {e}")
            return 0.0
    
    async def track_parameter_changes(self) -> Dict[str, Any]:
        """Track parameter changes over time."""
        try:
            # Get recent optimization metrics
            recent_metrics = self.optimization_metrics[-50:] if self.optimization_metrics else []
            
            if not recent_metrics:
                return {"error": "No optimization metrics available"}
            
            # Analyze parameter changes
            parameter_changes = {}
            for metric in recent_metrics:
                for param, value in metric.parameter_changes.items():
                    if param not in parameter_changes:
                        parameter_changes[param] = []
                    parameter_changes[param].append(value)
            
            # Calculate parameter statistics
            parameter_summary = {}
            for param, values in parameter_changes.items():
                parameter_summary[param] = {
                    "current_value": values[-1] if values else 0.0,
                    "average_value": statistics.mean(values),
                    "min_value": min(values),
                    "max_value": max(values),
                    "change_count": len(values),
                    "trend": self._calculate_trend(values)
                }
            
            logger.info(f"Parameter changes tracked: {len(parameter_summary)} parameters")
            return {
                "timestamp": datetime.now(),
                "parameters": parameter_summary,
                "total_changes": len(recent_metrics)
            }
            
        except Exception as e:
            logger.error(f"Error tracking parameter changes: {e}")
            return {"error": str(e)}
    
    async def track_learning_progress(self) -> Dict[str, Any]:
        """Track learning progress metrics."""
        try:
            # Mock learning progress data (in real implementation, this would come from actual learning)
            learning_progress = {
                "timestamp": datetime.now(),
                "total_training_samples": 10000,
                "current_epoch": 50,
                "total_epochs": 100,
                "training_accuracy": 0.92,
                "validation_accuracy": 0.88,
                "training_loss": 0.15,
                "validation_loss": 0.22,
                "learning_rate": 0.001,
                "convergence_rate": 0.85,
                "overfitting_risk": 0.05
            }
            
            logger.info(f"Learning progress tracked: {learning_progress['training_accuracy']:.2%} training accuracy")
            return learning_progress
            
        except Exception as e:
            logger.error(f"Error tracking learning progress: {e}")
            return {"error": str(e)}
    
    async def generate_performance_dashboard(self) -> Dict[str, Any]:
        """Generate comprehensive performance dashboard."""
        try:
            # Get all performance metrics
            ai_performance = await self.track_ai_performance()
            execution_performance = await self.track_execution_metrics()
            optimization_performance = await self.track_optimization_results()
            
            # Calculate overall performance score
            overall_score = self._calculate_overall_performance_score(
                ai_performance, execution_performance, optimization_performance
            )
            
            # Generate trends
            trends = await self._generate_performance_trends()
            
            # Generate recommendations
            recommendations = await self._generate_performance_recommendations(
                ai_performance, execution_performance, optimization_performance
            )
            
            dashboard = {
                "timestamp": datetime.now(),
                "overall_score": overall_score,
                "ai_performance": ai_performance,
                "execution_performance": execution_performance,
                "optimization_performance": optimization_performance,
                "trends": trends,
                "recommendations": recommendations
            }
            
            logger.info(f"Performance dashboard generated: Overall score {overall_score:.2%}")
            return dashboard
            
        except Exception as e:
            logger.error(f"Error generating performance dashboard: {e}")
            return {"error": str(e)}
    
    def _calculate_overall_performance_score(
        self,
        ai_performance: Dict[str, Any],
        execution_performance: Dict[str, Any],
        optimization_performance: Dict[str, Any]
    ) -> float:
        """Calculate overall performance score."""
        try:
            # Weighted overall score
            weights = {
                "ai": 0.4,
                "execution": 0.4,
                "optimization": 0.2
            }
            
            ai_score = ai_performance.get("performance_score", 0.0)
            execution_score = execution_performance.get("execution_score", 0.0)
            optimization_score = optimization_performance.get("optimization_score", 0.0)
            
            overall_score = (
                weights["ai"] * ai_score +
                weights["execution"] * execution_score +
                weights["optimization"] * optimization_score
            )
            
            return min(1.0, max(0.0, overall_score))
            
        except Exception as e:
            logger.error(f"Error calculating overall performance score: {e}")
            return 0.0
    
    async def _generate_performance_trends(self) -> Dict[str, Any]:
        """Generate performance trends."""
        try:
            trends = {
                "ai_accuracy_trend": "improving",
                "execution_speed_trend": "stable",
                "optimization_improvement_trend": "improving",
                "overall_performance_trend": "improving"
            }
            
            return trends
            
        except Exception as e:
            logger.error(f"Error generating performance trends: {e}")
            return {}
    
    async def _generate_performance_recommendations(
        self,
        ai_performance: Dict[str, Any],
        execution_performance: Dict[str, Any],
        optimization_performance: Dict[str, Any]
    ) -> List[str]:
        """Generate performance recommendations."""
        try:
            recommendations = []
            
            # AI performance recommendations
            if ai_performance.get("accuracy", 0) < 0.8:
                recommendations.append("Consider retraining AI models to improve accuracy")
            
            if ai_performance.get("latency", 0) > 0.2:
                recommendations.append("Optimize AI model inference to reduce latency")
            
            # Execution performance recommendations
            if execution_performance.get("slippage", 0) > 0.005:
                recommendations.append("Review execution strategy to reduce slippage")
            
            if execution_performance.get("rejection_rate", 0) > 0.05:
                recommendations.append("Investigate order rejection causes")
            
            # Optimization performance recommendations
            if optimization_performance.get("improvement_percentage", 0) < 0.05:
                recommendations.append("Consider different optimization strategies")
            
            if not recommendations:
                recommendations.append("System performance is optimal")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating performance recommendations: {e}")
            return ["Error generating recommendations"]
    
    async def generate_comparison_report(self, mode1: str, mode2: str) -> Dict[str, Any]:
        """Generate performance comparison report between two modes."""
        try:
            # Mock comparison data (in real implementation, this would compare actual mode performance)
            comparison_report = {
                "timestamp": datetime.now(),
                "mode1": mode1,
                "mode2": mode2,
                "comparison_period": "24h",
                "metrics_comparison": {
                    "ai_accuracy": {
                        "mode1": 0.85,
                        "mode2": 0.82,
                        "difference": 0.03,
                        "better_mode": mode1
                    },
                    "execution_speed": {
                        "mode1": 0.05,
                        "mode2": 0.08,
                        "difference": -0.03,
                        "better_mode": mode1
                    },
                    "slippage": {
                        "mode1": 0.001,
                        "mode2": 0.002,
                        "difference": -0.001,
                        "better_mode": mode1
                    }
                },
                "overall_winner": mode1,
                "recommendations": [
                    f"{mode1} shows better overall performance",
                    f"Consider using {mode1} for production trading"
                ]
            }
            
            logger.info(f"Comparison report generated: {mode1} vs {mode2}")
            return comparison_report
            
        except Exception as e:
            logger.error(f"Error generating comparison report: {e}")
            return {"error": str(e)}
    
    def _trim_ai_metrics(self) -> None:
        """Trim AI metrics to maximum size."""
        if len(self.ai_metrics) > self.max_history_size:
            self.ai_metrics = self.ai_metrics[-self.max_history_size:]
    
    def _trim_execution_metrics(self) -> None:
        """Trim execution metrics to maximum size."""
        if len(self.execution_metrics) > self.max_history_size:
            self.execution_metrics = self.execution_metrics[-self.max_history_size:]
    
    def _trim_optimization_metrics(self) -> None:
        """Trim optimization metrics to maximum size."""
        if len(self.optimization_metrics) > self.max_history_size:
            self.optimization_metrics = self.optimization_metrics[-self.max_history_size:]


# Global performance analytics instance
_performance_analytics: Optional[PerformanceAnalytics] = None


def get_performance_analytics() -> PerformanceAnalytics:
    """Get global performance analytics instance."""
    global _performance_analytics
    if _performance_analytics is None:
        _performance_analytics = PerformanceAnalytics()
    return _performance_analytics


async def track_ai_performance() -> Dict[str, Any]:
    """Track AI performance metrics."""
    analytics = get_performance_analytics()
    return await analytics.track_ai_performance()


async def track_model_accuracy() -> Dict[str, Any]:
    """Track model accuracy metrics."""
    analytics = get_performance_analytics()
    return await analytics.track_model_accuracy()


async def track_decision_quality() -> Dict[str, Any]:
    """Track decision quality metrics."""
    analytics = get_performance_analytics()
    return await analytics.track_decision_quality()


async def track_execution_metrics() -> Dict[str, Any]:
    """Track execution performance metrics."""
    analytics = get_performance_analytics()
    return await analytics.track_execution_metrics()


async def track_slippage() -> Dict[str, Any]:
    """Track execution slippage metrics."""
    analytics = get_performance_analytics()
    return await analytics.track_slippage()


async def track_latency() -> Dict[str, Any]:
    """Track system latency metrics."""
    analytics = get_performance_analytics()
    return await analytics.track_latency()


async def track_optimization_results() -> Dict[str, Any]:
    """Track optimization performance results."""
    analytics = get_performance_analytics()
    return await analytics.track_optimization_results()


async def track_parameter_changes() -> Dict[str, Any]:
    """Track parameter changes over time."""
    analytics = get_performance_analytics()
    return await analytics.track_parameter_changes()


async def track_learning_progress() -> Dict[str, Any]:
    """Track learning progress metrics."""
    analytics = get_performance_analytics()
    return await analytics.track_learning_progress()


async def generate_performance_dashboard() -> Dict[str, Any]:
    """Generate comprehensive performance dashboard."""
    analytics = get_performance_analytics()
    return await analytics.generate_performance_dashboard()


async def generate_comparison_report(mode1: str, mode2: str) -> Dict[str, Any]:
    """Generate performance comparison report between two modes."""
    analytics = get_performance_analytics()
    return await analytics.generate_comparison_report(mode1, mode2)


# Phase 1: Global helper functions for phase performance tracking
def start_phase_timer(phase_name: str) -> None:
    """Start timing a phase."""
    analytics = get_performance_analytics()
    analytics.start_phase_timer(phase_name)


def end_phase_timer(phase_name: str) -> float:
    """End timing a phase and return duration in seconds."""
    analytics = get_performance_analytics()
    return analytics.end_phase_timer(phase_name)


def get_phase_duration_stats(phase_name: str) -> Dict[str, Any]:
    """Get statistics for a specific phase duration."""
    analytics = get_performance_analytics()
    return analytics.get_phase_duration_stats(phase_name)


def get_all_phase_stats() -> Dict[str, Any]:
    """Get statistics for all phases."""
    analytics = get_performance_analytics()
    return analytics.get_all_phase_stats()


async def track_phase_performance(phase_name: str, step_label: str = None) -> Dict[str, Any]:
    """Track performance of a specific phase with optional step."""
    analytics = get_performance_analytics()
    return await analytics.track_phase_performance(phase_name, step_label)


async def generate_phase_performance_dashboard() -> Dict[str, Any]:
    """Generate comprehensive phase performance dashboard."""
    analytics = get_performance_analytics()
    return await analytics.generate_phase_performance_dashboard()
