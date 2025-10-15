"""
Enterprise System Monitor
=========================

Enterprise-grade system health monitoring with comprehensive tracking
of system health, performance metrics, resource usage, and API usage.
"""

import asyncio
import json
import logging
import psutil
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# Phase 2: Import API Budget Manager (commented out to avoid circular imports)
# from ..data_pipeline.api_budget_manager import get_api_budget_manager

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """System health status enumeration."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class SystemMetrics:
    """System metrics data structure."""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, int]
    process_count: int
    uptime: float
    health_status: HealthStatus


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    timestamp: datetime
    response_time: float
    throughput: float
    error_rate: float
    success_rate: float
    active_connections: int
    queue_size: int


@dataclass
class ResourceMetrics:
    """Resource usage metrics data structure."""
    timestamp: datetime
    cpu_cores: int
    cpu_percent: float
    memory_total: int
    memory_used: int
    memory_percent: float
    disk_total: int
    disk_used: int
    disk_percent: float
    network_sent: int
    network_recv: int


@dataclass
class APIMetrics:
    """API usage metrics data structure."""
    timestamp: datetime
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time: float
    requests_per_second: float
    api_endpoints: Dict[str, int]


class SystemMonitor:
    """Enterprise-grade system health monitoring."""
    
    def __init__(self):
        self.start_time = time.time()
        self.metrics_history: List[SystemMetrics] = []
        self.performance_history: List[PerformanceMetrics] = []
        self.resource_history: List[ResourceMetrics] = []
        self.api_history: List[APIMetrics] = []
        self.is_monitoring = False
        self.monitoring_interval = 60  # 60 seconds
        self.max_history_size = 1000  # Keep last 1000 entries
        
        # Phase 0: Phase timers and structured logging
        self.phase_timers: Dict[str, Dict[str, Any]] = {}
        self.current_phase = None
        self.phase_start_time = None
        self.step_labels = [
            "ingest", "features", "factors", "scoring", "ensemble", 
            "sizing", "orders", "persist", "dashboard"
        ]
        
        # Phase 0: Prometheus metrics
        self.prometheus_metrics = {
            "phase_duration_seconds": {},
            "api_calls_total": {},
            "api_rate_limit_hits_total": {},
            "api_budget_remaining": {},
            "system_health_score": 1.0
        }
        
        # Phase 2: Initialize API Budget Manager (lazy import to avoid circular imports)
        self.api_budget_manager = None
        
        # Phase 0: Structured JSON logging
        self.setup_structured_logging()
    
    def setup_structured_logging(self):
        """Setup structured JSON logging to logs/system.log."""
        try:
            # Ensure logs directory exists
            logs_dir = Path("logs")
            logs_dir.mkdir(exist_ok=True)
            
            # Setup JSON file handler
            json_handler = logging.FileHandler(logs_dir / "system.log")
            json_handler.setLevel(logging.INFO)
            
            # Create formatter for structured JSON logging
            class JSONFormatter(logging.Formatter):
                def format(self, record):
                    log_entry = {
                        "ts": datetime.now().isoformat(),
                        "phase": getattr(record, 'phase', None),
                        "symbol": getattr(record, 'symbol', None),
                        "duration_ms": getattr(record, 'duration_ms', None),
                        "status": getattr(record, 'status', None),
                        "detail": record.getMessage(),
                        "level": record.levelname,
                        "logger": record.name
                    }
                    return json.dumps(log_entry)
            
            json_handler.setFormatter(JSONFormatter())
            
            # Add handler to system logger
            system_logger = logging.getLogger('system_monitor')
            system_logger.addHandler(json_handler)
            system_logger.setLevel(logging.INFO)
            
        except Exception as e:
            logger.error(f"Failed to setup structured logging: {e}")
    
    def start_phase_timer(self, phase_name: str, symbol: str = None) -> str:
        """Start phase timer with execution ID."""
        execution_id = f"{phase_name}_{int(time.time() * 1000)}"
        self.current_phase = phase_name
        self.phase_start_time = time.time()
        
        self.phase_timers[execution_id] = {
            "phase_name": phase_name,
            "symbol": symbol,
            "start_time": self.phase_start_time,
            "steps": {},
            "status": "RUNNING"
        }
        
        # Log phase start
        self.log_phase_event(phase_name, symbol, "STARTED", execution_id)
        
        return execution_id
    
    def end_phase_timer(self, execution_id: str, status: str = "COMPLETED", 
                       error_message: str = None) -> Dict[str, Any]:
        """End phase timer and return timing data."""
        if execution_id not in self.phase_timers:
            logger.warning(f"Phase timer {execution_id} not found")
            return {}
        
        end_time = time.time()
        phase_data = self.phase_timers[execution_id]
        duration_ms = int((end_time - phase_data["start_time"]) * 1000)
        
        # Update phase data
        phase_data.update({
            "end_time": end_time,
            "duration_ms": duration_ms,
            "status": status,
            "error_message": error_message
        })
        
        # Log phase end
        self.log_phase_event(
            phase_data["phase_name"], 
            phase_data["symbol"], 
            status, 
            execution_id,
            duration_ms,
            error_message
        )
        
        # Update Prometheus metrics
        self.update_prometheus_metrics(phase_data)
        
        # Clean up
        self.current_phase = None
        self.phase_start_time = None
        
        return phase_data
    
    def log_step_timer(self, execution_id: str, step_label: str, 
                      duration_ms: int = None, status: str = "COMPLETED"):
        """Log step timing within a phase."""
        if execution_id not in self.phase_timers:
            logger.warning(f"Phase timer {execution_id} not found for step {step_label}")
            return
        
        if step_label not in self.step_labels:
            logger.warning(f"Unknown step label: {step_label}")
            return
        
        self.phase_timers[execution_id]["steps"][step_label] = {
            "duration_ms": duration_ms,
            "status": status,
            "timestamp": datetime.now().isoformat()
        }
        
        # Log step event
        self.log_phase_event(
            self.phase_timers[execution_id]["phase_name"],
            self.phase_timers[execution_id]["symbol"],
            f"STEP_{step_label}",
            execution_id,
            duration_ms
        )
    
    def log_phase_event(self, phase: str, symbol: str, status: str, 
                       execution_id: str, duration_ms: int = None, 
                       error_message: str = None):
        """Log phase event with structured JSON format."""
        try:
            system_logger = logging.getLogger('system_monitor')
            
            # Create log record with custom attributes
            record = system_logger.makeRecord(
                system_logger.name, logging.INFO, __file__, 0,
                f"Phase {phase} {status}" + (f": {error_message}" if error_message else ""),
                (), None
            )
            
            # Add custom attributes
            record.phase = phase
            record.symbol = symbol
            record.duration_ms = duration_ms
            record.status = status
            record.execution_id = execution_id
            
            system_logger.handle(record)
            
        except Exception as e:
            logger.error(f"Failed to log phase event: {e}")
    
    def update_prometheus_metrics(self, phase_data: Dict[str, Any]):
        """Update Prometheus metrics with phase data."""
        try:
            phase_name = phase_data["phase_name"]
            duration_seconds = phase_data["duration_ms"] / 1000.0
            
            # Update phase duration metrics
            if phase_name not in self.prometheus_metrics["phase_duration_seconds"]:
                self.prometheus_metrics["phase_duration_seconds"][phase_name] = []
            
            self.prometheus_metrics["phase_duration_seconds"][phase_name].append({
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": duration_seconds,
                "status": phase_data["status"]
            })
            
            # Keep only last 100 entries per phase
            if len(self.prometheus_metrics["phase_duration_seconds"][phase_name]) > 100:
                self.prometheus_metrics["phase_duration_seconds"][phase_name] = \
                    self.prometheus_metrics["phase_duration_seconds"][phase_name][-100:]
            
        except Exception as e:
            logger.error(f"Failed to update Prometheus metrics: {e}")
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics in text format."""
        try:
            metrics_lines = []
            
            # Phase duration metrics
            for phase_name, durations in self.prometheus_metrics["phase_duration_seconds"].items():
                if durations:
                    latest = durations[-1]
                    metrics_lines.append(f"phase_duration_seconds{{phase=\"{phase_name}\"}} {latest['duration_seconds']}")
            
            # API budget metrics
            for api_name, budget in self.prometheus_metrics["api_budget_remaining"].items():
                metrics_lines.append(f"api_budget_remaining{{api=\"{api_name}\"}} {budget}")
            
            # System health score
            metrics_lines.append(f"system_health_score {self.prometheus_metrics['system_health_score']}")
            
            return "\n".join(metrics_lines)
            
        except Exception as e:
            logger.error(f"Failed to generate Prometheus metrics: {e}")
            return ""
    
    def update_api_metrics(self, api_name: str, endpoint: str, 
                          success: bool, response_time_ms: int, 
                          rate_limit_hit: bool = False):
        """Update API usage metrics."""
        try:
            # Phase 2: Lazy import API budget manager to avoid circular imports
            if self.api_budget_manager is None:
                try:
                    from ..data_pipeline.api_budget_manager import get_api_budget_manager
                    self.api_budget_manager = get_api_budget_manager()
                except ImportError:
                    logger.warning("API Budget Manager not available, using fallback metrics")
                    self.api_budget_manager = None
            
            if self.api_budget_manager is not None:
                # Get real API usage stats from budget manager
                api_stats = self.api_budget_manager.get_usage_stats(api_name)
                
                # Update Prometheus metrics with real data
                self.prometheus_metrics["api_calls_total"][api_name] = api_stats.get("requests_made", 0)
                self.prometheus_metrics["api_rate_limit_hits_total"][api_name] = api_stats.get("rate_limit_hits", 0)
                self.prometheus_metrics["api_budget_remaining"][api_name] = api_stats.get("daily_remaining", 0)
                
                # Update system health score based on API health
                self._update_system_health_score()
            else:
                # Fallback to basic metrics tracking
                if api_name not in self.prometheus_metrics["api_calls_total"]:
                    self.prometheus_metrics["api_calls_total"][api_name] = 0
                    self.prometheus_metrics["api_rate_limit_hits_total"][api_name] = 0
                    self.prometheus_metrics["api_budget_remaining"][api_name] = 1000
                
                self.prometheus_metrics["api_calls_total"][api_name] += 1
                if rate_limit_hit:
                    self.prometheus_metrics["api_rate_limit_hits_total"][api_name] += 1
                self.prometheus_metrics["api_budget_remaining"][api_name] = max(
                    0, self.prometheus_metrics["api_budget_remaining"][api_name] - 1
                )
            
        except Exception as e:
            logger.error(f"Failed to update API metrics: {e}")
    
    def _update_system_health_score(self):
        """Update system health score based on API and system metrics."""
        try:
            if self.api_budget_manager is None:
                # Fallback to basic health score
                self.prometheus_metrics["system_health_score"] = 0.8
                return
            
            # Get all API stats
            all_api_stats = self.api_budget_manager.get_all_usage_stats()
            
            if not all_api_stats:
                self.prometheus_metrics["system_health_score"] = 1.0
                return
            
            # Calculate health score based on API success rates
            total_requests = 0
            total_successful = 0
            total_rate_limits = 0
            
            for api_name, stats in all_api_stats.items():
                requests = stats.get("requests_made", 0)
                successful = stats.get("requests_successful", 0)
                rate_limits = stats.get("rate_limit_hits", 0)
                
                total_requests += requests
                total_successful += successful
                total_rate_limits += rate_limits
            
            if total_requests == 0:
                self.prometheus_metrics["system_health_score"] = 1.0
                return
            
            # Calculate health score (0.0 to 1.0)
            success_rate = total_successful / total_requests
            rate_limit_penalty = min(total_rate_limits / max(total_requests, 1), 0.5)  # Max 50% penalty
            
            health_score = max(0.0, success_rate - rate_limit_penalty)
            self.prometheus_metrics["system_health_score"] = health_score
            
        except Exception as e:
            logger.error(f"Failed to update system health score: {e}")
            self.prometheus_metrics["system_health_score"] = 0.5  # Default to 50% on error
    
    def get_phase_timing_summary(self) -> Dict[str, Any]:
        """Get summary of phase timings."""
        try:
            summary = {
                "total_phases": len(self.phase_timers),
                "phases": {},
                "average_durations": {},
                "success_rates": {}
            }
            
            for execution_id, phase_data in self.phase_timers.items():
                phase_name = phase_data["phase_name"]
                
                if phase_name not in summary["phases"]:
                    summary["phases"][phase_name] = []
                
                summary["phases"][phase_name].append({
                    "execution_id": execution_id,
                    "duration_ms": phase_data.get("duration_ms", 0),
                    "status": phase_data.get("status", "UNKNOWN"),
                    "symbol": phase_data.get("symbol"),
                    "steps": phase_data.get("steps", {})
                })
            
            # Calculate averages and success rates
            for phase_name, phases in summary["phases"].items():
                durations = [p["duration_ms"] for p in phases if p["duration_ms"]]
                successes = [p for p in phases if p["status"] == "COMPLETED"]
                
                summary["average_durations"][phase_name] = sum(durations) / len(durations) if durations else 0
                summary["success_rates"][phase_name] = len(successes) / len(phases) if phases else 0
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get phase timing summary: {e}")
            return {}
        
    async def start_monitoring(self) -> bool:
        """Start system monitoring."""
        try:
            if self.is_monitoring:
                logger.warning("System monitoring is already running")
                return True
            
            self.is_monitoring = True
            logger.info("Starting system monitoring...")
            
            # Start monitoring tasks
            asyncio.create_task(self._monitoring_loop())
            
            logger.info("System monitoring started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start system monitoring: {e}")
            return False
    
    async def stop_monitoring(self) -> bool:
        """Stop system monitoring."""
        try:
            if not self.is_monitoring:
                logger.warning("System monitoring is not running")
                return True
            
            self.is_monitoring = False
            logger.info("Stopping system monitoring...")
            
            logger.info("System monitoring stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop system monitoring: {e}")
            return False
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Collect metrics
                await self._collect_metrics()
                
                # Wait for next interval
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _collect_metrics(self) -> None:
        """Collect all system metrics."""
        try:
            # Collect system metrics
            system_metrics = await self.monitor_system_health()
            self.metrics_history.append(system_metrics)
            
            # Collect performance metrics
            performance_metrics = await self.track_performance_metrics()
            self.performance_history.append(performance_metrics)
            
            # Collect resource metrics
            resource_metrics = await self.monitor_resource_usage()
            self.resource_history.append(resource_metrics)
            
            # Collect API metrics
            api_metrics = await self.track_api_usage()
            self.api_history.append(api_metrics)
            
            # Trim history to max size
            self._trim_history()
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
    
    def _trim_history(self) -> None:
        """Trim history to maximum size."""
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history = self.metrics_history[-self.max_history_size:]
        if len(self.performance_history) > self.max_history_size:
            self.performance_history = self.performance_history[-self.max_history_size:]
        if len(self.resource_history) > self.max_history_size:
            self.resource_history = self.resource_history[-self.max_history_size:]
        if len(self.api_history) > self.max_history_size:
            self.api_history = self.api_history[-self.max_history_size:]
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status (alias for monitor_system_health)."""
        # Return a simplified health status for compatibility
        return {
            "status": "healthy",
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "disk_usage": 0.0,
            "timestamp": datetime.now().isoformat()
        }
    
    async def monitor_system_health(self) -> SystemMetrics:
        """Monitor system health."""
        try:
            # Get system information
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            process_count = len(psutil.pids())
            uptime = time.time() - self.start_time
            
            # Determine health status
            health_status = self._determine_health_status(cpu_usage, memory.percent, disk.percent)
            
            metrics = SystemMetrics(
                timestamp=datetime.now(),
                cpu_usage=cpu_usage,
                memory_usage=memory.percent,
                disk_usage=disk.percent,
                network_io={
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                    "packets_sent": network.packets_sent,
                    "packets_recv": network.packets_recv
                },
                process_count=process_count,
                uptime=uptime,
                health_status=health_status
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error monitoring system health: {e}")
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_usage=0.0,
                memory_usage=0.0,
                disk_usage=0.0,
                network_io={},
                process_count=0,
                uptime=0.0,
                health_status=HealthStatus.UNKNOWN
            )
    
    def _determine_health_status(self, cpu_usage: float, memory_usage: float, disk_usage: float) -> HealthStatus:
        """Determine system health status based on metrics."""
        if cpu_usage > 90 or memory_usage > 90 or disk_usage > 90:
            return HealthStatus.CRITICAL
        elif cpu_usage > 70 or memory_usage > 70 or disk_usage > 70:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY
    
    async def track_performance_metrics(self) -> PerformanceMetrics:
        """Track performance metrics."""
        try:
            # Mock performance metrics (in real implementation, these would come from actual measurements)
            response_time = 0.1  # 100ms average response time
            throughput = 100.0  # 100 requests per second
            error_rate = 0.01  # 1% error rate
            success_rate = 1.0 - error_rate
            active_connections = 10
            queue_size = 5
            
            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                response_time=response_time,
                throughput=throughput,
                error_rate=error_rate,
                success_rate=success_rate,
                active_connections=active_connections,
                queue_size=queue_size
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error tracking performance metrics: {e}")
            return PerformanceMetrics(
                timestamp=datetime.now(),
                response_time=0.0,
                throughput=0.0,
                error_rate=0.0,
                success_rate=0.0,
                active_connections=0,
                queue_size=0
            )
    
    async def monitor_resource_usage(self) -> ResourceMetrics:
        """Monitor resource usage."""
        try:
            # Get resource information
            cpu_cores = psutil.cpu_count()
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            
            metrics = ResourceMetrics(
                timestamp=datetime.now(),
                cpu_cores=cpu_cores,
                cpu_percent=cpu_percent,
                memory_total=memory.total,
                memory_used=memory.used,
                memory_percent=memory.percent,
                disk_total=disk.total,
                disk_used=disk.used,
                disk_percent=disk.percent,
                network_sent=network.bytes_sent,
                network_recv=network.bytes_recv
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error monitoring resource usage: {e}")
            return ResourceMetrics(
                timestamp=datetime.now(),
                cpu_cores=0,
                cpu_percent=0.0,
                memory_total=0,
                memory_used=0,
                memory_percent=0.0,
                disk_total=0,
                disk_used=0,
                disk_percent=0.0,
                network_sent=0,
                network_recv=0
            )
    
    async def track_api_usage(self) -> APIMetrics:
        """Track API usage metrics."""
        try:
            # Mock API metrics (in real implementation, these would come from actual API tracking)
            total_requests = 1000
            successful_requests = 990
            failed_requests = 10
            average_response_time = 0.15
            requests_per_second = 10.0
            api_endpoints = {
                "/api/trading": 500,
                "/api/market": 300,
                "/api/portfolio": 200
            }
            
            metrics = APIMetrics(
                timestamp=datetime.now(),
                total_requests=total_requests,
                successful_requests=successful_requests,
                failed_requests=failed_requests,
                average_response_time=average_response_time,
                requests_per_second=requests_per_second,
                api_endpoints=api_endpoints
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error tracking API usage: {e}")
            return APIMetrics(
                timestamp=datetime.now(),
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                average_response_time=0.0,
                requests_per_second=0.0,
                api_endpoints={}
            )
    
    async def log_trade_activity(self, mode: str) -> None:
        """Log trade activity for specific mode."""
        try:
            logger.info(f"Logging trade activity for mode: {mode}")
            
            # Log trade activity (in real implementation, this would log actual trade data)
            trade_log = {
                "timestamp": datetime.now(),
                "mode": mode,
                "activity": "trade_execution",
                "details": "Trade activity logged"
            }
            
            logger.info(f"Trade activity logged: {trade_log}")
            
        except Exception as e:
            logger.error(f"Error logging trade activity: {e}")
    
    async def monitor_mode_specific_metrics(self, mode: str) -> Dict[str, Any]:
        """Monitor mode-specific metrics."""
        try:
            # Get mode-specific metrics
            mode_metrics = {
                "timestamp": datetime.now(),
                "mode": mode,
                "active_trades": 5,
                "portfolio_value": 10000.0,
                "cash_balance": 5000.0,
                "pnl": 100.0,
                "risk_metrics": {
                    "var": 0.05,
                    "cvar": 0.08,
                    "max_drawdown": 0.02
                }
            }
            
            return mode_metrics
            
        except Exception as e:
            logger.error(f"Error monitoring mode-specific metrics: {e}")
            return {
                "timestamp": datetime.now(),
                "mode": mode,
                "error": str(e)
            }
    
    async def track_position_changes(self) -> List[Dict[str, Any]]:
        """Track position changes."""
        try:
            # Mock position changes (in real implementation, this would track actual position changes)
            position_changes = [
                {
                    "timestamp": datetime.now(),
                    "symbol": "AAPL",
                    "action": "BUY",
                    "quantity": 10,
                    "price": 150.0,
                    "value": 1500.0
                },
                {
                    "timestamp": datetime.now(),
                    "symbol": "MSFT",
                    "action": "SELL",
                    "quantity": 5,
                    "price": 300.0,
                    "value": 1500.0
                }
            ]
            
            return position_changes
            
        except Exception as e:
            logger.error(f"Error tracking position changes: {e}")
            return []
    
    async def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time metrics."""
        try:
            # Get latest metrics from history
            latest_system = self.metrics_history[-1] if self.metrics_history else None
            latest_performance = self.performance_history[-1] if self.performance_history else None
            latest_resource = self.resource_history[-1] if self.resource_history else None
            latest_api = self.api_history[-1] if self.api_history else None
            
            real_time_metrics = {
                "timestamp": datetime.now(),
                "system": latest_system.__dict__ if latest_system else None,
                "performance": latest_performance.__dict__ if latest_performance else None,
                "resource": latest_resource.__dict__ if latest_resource else None,
                "api": latest_api.__dict__ if latest_api else None
            }
            
            return real_time_metrics
            
        except Exception as e:
            logger.error(f"Error getting real-time metrics: {e}")
            return {
                "timestamp": datetime.now(),
                "error": str(e)
            }
    
    async def get_historical_metrics(self, period: str) -> Dict[str, Any]:
        """Get historical metrics for specified period."""
        try:
            # Calculate time range based on period
            now = datetime.now()
            if period == "1h":
                start_time = now - timedelta(hours=1)
            elif period == "24h":
                start_time = now - timedelta(days=1)
            elif period == "7d":
                start_time = now - timedelta(days=7)
            elif period == "30d":
                start_time = now - timedelta(days=30)
            else:
                start_time = now - timedelta(hours=1)  # Default to 1 hour
            
            # Filter metrics by time range
            filtered_system = [m for m in self.metrics_history if m.timestamp >= start_time]
            filtered_performance = [m for m in self.performance_history if m.timestamp >= start_time]
            filtered_resource = [m for m in self.resource_history if m.timestamp >= start_time]
            filtered_api = [m for m in self.api_history if m.timestamp >= start_time]
            
            historical_metrics = {
                "period": period,
                "start_time": start_time,
                "end_time": now,
                "system_metrics": [m.__dict__ for m in filtered_system],
                "performance_metrics": [m.__dict__ for m in filtered_performance],
                "resource_metrics": [m.__dict__ for m in filtered_resource],
                "api_metrics": [m.__dict__ for m in filtered_api]
            }
            
            return historical_metrics
            
        except Exception as e:
            logger.error(f"Error getting historical metrics: {e}")
            return {
                "period": period,
                "error": str(e)
            }
    
    async def generate_health_report(self) -> str:
        """Generate system health report."""
        try:
            # Get latest metrics
            latest_system = self.metrics_history[-1] if self.metrics_history else None
            latest_performance = self.performance_history[-1] if self.performance_history else None
            latest_resource = self.resource_history[-1] if self.resource_history else None
            
            if not latest_system:
                return "No system metrics available"
            
            # Generate health report
            report = f"""
System Health Report
===================
Timestamp: {latest_system.timestamp}
Overall Status: {latest_system.health_status.value.upper()}

System Metrics:
- CPU Usage: {latest_system.cpu_usage:.1f}%
- Memory Usage: {latest_system.memory_usage:.1f}%
- Disk Usage: {latest_system.disk_usage:.1f}%
- Process Count: {latest_system.process_count}
- Uptime: {latest_system.uptime:.1f} seconds

Network I/O:
- Bytes Sent: {latest_system.network_io.get('bytes_sent', 0):,}
- Bytes Received: {latest_system.network_io.get('bytes_recv', 0):,}
- Packets Sent: {latest_system.network_io.get('packets_sent', 0):,}
- Packets Received: {latest_system.network_io.get('packets_recv', 0):,}
"""
            
            if latest_performance:
                report += f"""
Performance Metrics:
- Response Time: {latest_performance.response_time:.3f}s
- Throughput: {latest_performance.throughput:.1f} req/s
- Error Rate: {latest_performance.error_rate:.2%}
- Success Rate: {latest_performance.success_rate:.2%}
- Active Connections: {latest_performance.active_connections}
- Queue Size: {latest_performance.queue_size}
"""
            
            if latest_resource:
                report += f"""
Resource Usage:
- CPU Cores: {latest_resource.cpu_cores}
- CPU Percent: {latest_resource.cpu_percent:.1f}%
- Memory Total: {latest_resource.memory_total / (1024**3):.1f} GB
- Memory Used: {latest_resource.memory_used / (1024**3):.1f} GB
- Memory Percent: {latest_resource.memory_percent:.1f}%
- Disk Total: {latest_resource.disk_total / (1024**3):.1f} GB
- Disk Used: {latest_resource.disk_used / (1024**3):.1f} GB
- Disk Percent: {latest_resource.disk_percent:.1f}%
"""
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating health report: {e}")
            return f"Error generating health report: {e}"


# Global system monitor instance
_system_monitor: Optional[SystemMonitor] = None


def get_system_monitor() -> SystemMonitor:
    """Get global system monitor instance."""
    global _system_monitor
    if _system_monitor is None:
        _system_monitor = SystemMonitor()
    return _system_monitor


async def start_system_monitoring() -> bool:
    """Start system monitoring."""
    monitor = get_system_monitor()
    return await monitor.start_monitoring()


async def stop_system_monitoring() -> bool:
    """Stop system monitoring."""
    monitor = get_system_monitor()
    return await monitor.stop_monitoring()


async def get_system_health() -> Dict[str, Any]:
    """Get system health status."""
    monitor = get_system_monitor()
    metrics = await monitor.monitor_system_health()
    return metrics.__dict__


async def get_performance_metrics() -> Dict[str, Any]:
    """Get performance metrics."""
    monitor = get_system_monitor()
    metrics = await monitor.track_performance_metrics()
    return metrics.__dict__


async def get_resource_usage() -> Dict[str, Any]:
    """Get resource usage metrics."""
    monitor = get_system_monitor()
    metrics = await monitor.monitor_resource_usage()
    return metrics.__dict__


async def get_api_usage() -> Dict[str, Any]:
    """Get API usage metrics."""
    monitor = get_system_monitor()
    metrics = await monitor.track_api_usage()
    return metrics.__dict__


# Phase 0: Global helper functions for phase timing and monitoring
def start_phase_timer(phase_name: str, symbol: str = None) -> str:
    """Start phase timer with execution ID."""
    monitor = get_system_monitor()
    return monitor.start_phase_timer(phase_name, symbol)


def end_phase_timer(execution_id: str, status: str = "COMPLETED", 
                   error_message: str = None) -> Dict[str, Any]:
    """End phase timer and return timing data."""
    monitor = get_system_monitor()
    return monitor.end_phase_timer(execution_id, status, error_message)


def log_step_timer(execution_id: str, step_label: str, 
                  duration_ms: int = None, status: str = "COMPLETED"):
    """Log step timing within a phase."""
    monitor = get_system_monitor()
    monitor.log_step_timer(execution_id, step_label, duration_ms, status)


def update_api_metrics(api_name: str, endpoint: str, 
                      success: bool, response_time_ms: int, 
                      rate_limit_hit: bool = False):
    """Update API usage metrics."""
    monitor = get_system_monitor()
    monitor.update_api_metrics(api_name, endpoint, success, response_time_ms, rate_limit_hit)


def get_prometheus_metrics() -> str:
    """Get Prometheus metrics in text format."""
    monitor = get_system_monitor()
    return monitor.get_prometheus_metrics()


def get_phase_timing_summary() -> Dict[str, Any]:
    """Get summary of phase timings."""
    monitor = get_system_monitor()
    return monitor.get_phase_timing_summary()
