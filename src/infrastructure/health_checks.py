"""
Health Check Endpoints Module
=============================

Provides comprehensive health check capabilities for the trading system:
- Overall system health
- Broker connection status
- Data feed status
- Database connectivity
- Circuit breaker states
- Component-level health

Usage:
    from src.infrastructure.health_checks import (
        HealthCheckManager,
        get_health_check_manager,
    )

    manager = get_health_check_manager()

    # Register components
    manager.register_check("broker", broker_health_check)

    # Get health status
    status = manager.check_health()

    # Get specific component health
    broker_health = manager.check_component("broker")
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger('trading.health')


# =============================================================================
# Enums and Data Classes
# =============================================================================

class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status of a single component."""
    name: str
    status: HealthStatus
    message: str = ""
    latency_ms: float = 0.0
    last_check: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "latency_ms": self.latency_ms,
            "last_check": self.last_check.isoformat(),
            "details": self.details,
        }


@dataclass
class SystemHealth:
    """Overall system health status."""
    status: HealthStatus
    timestamp: datetime
    components: Dict[str, ComponentHealth]
    uptime_seconds: float
    version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "uptime_seconds": self.uptime_seconds,
            "version": self.version,
            "components": {
                name: comp.to_dict() for name, comp in self.components.items()
            },
        }


# =============================================================================
# Health Check Functions
# =============================================================================

HealthCheckFunc = Callable[[], ComponentHealth]
AsyncHealthCheckFunc = Callable[[], "asyncio.Future[ComponentHealth]"]


def create_health_result(
    name: str,
    healthy: bool,
    message: str = "",
    latency_ms: float = 0.0,
    details: Optional[Dict[str, Any]] = None,
) -> ComponentHealth:
    """Helper to create a health check result."""
    return ComponentHealth(
        name=name,
        status=HealthStatus.HEALTHY if healthy else HealthStatus.UNHEALTHY,
        message=message,
        latency_ms=latency_ms,
        last_check=datetime.now(),
        details=details or {},
    )


# =============================================================================
# Built-in Health Checks
# =============================================================================

def check_broker_health(broker: Optional[Any] = None) -> ComponentHealth:
    """
    Check broker connection health.

    Args:
        broker: Broker instance to check

    Returns:
        ComponentHealth with broker status
    """
    start = time.time()

    if broker is None:
        return ComponentHealth(
            name="broker",
            status=HealthStatus.UNKNOWN,
            message="No broker configured",
            latency_ms=0.0,
            last_check=datetime.now(),
        )

    try:
        # Check if broker has a health check method
        if hasattr(broker, 'is_connected'):
            connected = broker.is_connected()
        elif hasattr(broker, 'health_check'):
            connected = broker.health_check()
        elif hasattr(broker, 'get_account'):
            # Try to get account as connectivity test
            broker.get_account()
            connected = True
        else:
            connected = True  # Assume healthy if no check available

        latency = (time.time() - start) * 1000

        if connected:
            return ComponentHealth(
                name="broker",
                status=HealthStatus.HEALTHY,
                message="Broker connected",
                latency_ms=latency,
                last_check=datetime.now(),
            )
        else:
            return ComponentHealth(
                name="broker",
                status=HealthStatus.UNHEALTHY,
                message="Broker disconnected",
                latency_ms=latency,
                last_check=datetime.now(),
            )

    except Exception as e:
        latency = (time.time() - start) * 1000
        return ComponentHealth(
            name="broker",
            status=HealthStatus.UNHEALTHY,
            message=f"Broker check failed: {str(e)}",
            latency_ms=latency,
            last_check=datetime.now(),
        )


def check_data_feed_health(data_client: Optional[Any] = None) -> ComponentHealth:
    """
    Check data feed health.

    Args:
        data_client: Data client instance to check

    Returns:
        ComponentHealth with data feed status
    """
    start = time.time()

    if data_client is None:
        return ComponentHealth(
            name="data_feed",
            status=HealthStatus.UNKNOWN,
            message="No data client configured",
            latency_ms=0.0,
            last_check=datetime.now(),
        )

    try:
        # Check if data client has health check
        if hasattr(data_client, 'health_check'):
            healthy = data_client.health_check()
        elif hasattr(data_client, 'is_connected'):
            healthy = data_client.is_connected()
        elif hasattr(data_client, 'get_quote'):
            # Try to get a quote as connectivity test
            data_client.get_quote("SPY")
            healthy = True
        else:
            healthy = True

        latency = (time.time() - start) * 1000

        return ComponentHealth(
            name="data_feed",
            status=HealthStatus.HEALTHY if healthy else HealthStatus.UNHEALTHY,
            message="Data feed operational" if healthy else "Data feed unavailable",
            latency_ms=latency,
            last_check=datetime.now(),
        )

    except Exception as e:
        latency = (time.time() - start) * 1000
        return ComponentHealth(
            name="data_feed",
            status=HealthStatus.UNHEALTHY,
            message=f"Data feed check failed: {str(e)}",
            latency_ms=latency,
            last_check=datetime.now(),
        )


def check_database_health(db_connection: Optional[Any] = None) -> ComponentHealth:
    """
    Check database connectivity.

    Args:
        db_connection: Database connection to check

    Returns:
        ComponentHealth with database status
    """
    start = time.time()

    if db_connection is None:
        return ComponentHealth(
            name="database",
            status=HealthStatus.UNKNOWN,
            message="No database configured",
            latency_ms=0.0,
            last_check=datetime.now(),
        )

    try:
        # Try to execute a simple query
        if hasattr(db_connection, 'execute'):
            db_connection.execute("SELECT 1")
        elif hasattr(db_connection, 'ping'):
            db_connection.ping()
        elif hasattr(db_connection, 'is_connected'):
            if not db_connection.is_connected():
                raise ConnectionError("Database not connected")

        latency = (time.time() - start) * 1000

        return ComponentHealth(
            name="database",
            status=HealthStatus.HEALTHY,
            message="Database connected",
            latency_ms=latency,
            last_check=datetime.now(),
        )

    except Exception as e:
        latency = (time.time() - start) * 1000
        return ComponentHealth(
            name="database",
            status=HealthStatus.UNHEALTHY,
            message=f"Database check failed: {str(e)}",
            latency_ms=latency,
            last_check=datetime.now(),
        )


def check_circuit_breakers_health() -> ComponentHealth:
    """
    Check circuit breaker states.

    Returns:
        ComponentHealth with circuit breaker status
    """
    start = time.time()

    try:
        from src.utils.circuit_breaker import get_circuit_breaker_registry

        registry = get_circuit_breaker_registry()
        status = registry.get_status()

        latency = (time.time() - start) * 1000

        # Check if any breakers are open
        open_breakers = [
            name for name, info in status.items()
            if info.get("state") == "OPEN"
        ]

        half_open_breakers = [
            name for name, info in status.items()
            if info.get("state") == "HALF_OPEN"
        ]

        if open_breakers:
            return ComponentHealth(
                name="circuit_breakers",
                status=HealthStatus.DEGRADED,
                message=f"Open breakers: {', '.join(open_breakers)}",
                latency_ms=latency,
                last_check=datetime.now(),
                details={
                    "open": open_breakers,
                    "half_open": half_open_breakers,
                    "total": len(status),
                },
            )
        elif half_open_breakers:
            return ComponentHealth(
                name="circuit_breakers",
                status=HealthStatus.DEGRADED,
                message=f"Recovering breakers: {', '.join(half_open_breakers)}",
                latency_ms=latency,
                last_check=datetime.now(),
                details={
                    "open": [],
                    "half_open": half_open_breakers,
                    "total": len(status),
                },
            )
        else:
            return ComponentHealth(
                name="circuit_breakers",
                status=HealthStatus.HEALTHY,
                message=f"All {len(status)} breakers closed",
                latency_ms=latency,
                last_check=datetime.now(),
                details={
                    "open": [],
                    "half_open": [],
                    "total": len(status),
                },
            )

    except ImportError:
        return ComponentHealth(
            name="circuit_breakers",
            status=HealthStatus.UNKNOWN,
            message="Circuit breaker module not available",
            latency_ms=0.0,
            last_check=datetime.now(),
        )
    except Exception as e:
        latency = (time.time() - start) * 1000
        return ComponentHealth(
            name="circuit_breakers",
            status=HealthStatus.UNHEALTHY,
            message=f"Circuit breaker check failed: {str(e)}",
            latency_ms=latency,
            last_check=datetime.now(),
        )


def check_memory_health(threshold_mb: float = 1024.0) -> ComponentHealth:
    """
    Check memory usage.

    Args:
        threshold_mb: Memory threshold in MB for degraded status

    Returns:
        ComponentHealth with memory status
    """
    start = time.time()

    try:
        import psutil

        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)

        latency = (time.time() - start) * 1000

        if memory_mb > threshold_mb * 1.5:
            status = HealthStatus.UNHEALTHY
            message = f"High memory usage: {memory_mb:.1f}MB"
        elif memory_mb > threshold_mb:
            status = HealthStatus.DEGRADED
            message = f"Elevated memory usage: {memory_mb:.1f}MB"
        else:
            status = HealthStatus.HEALTHY
            message = f"Memory usage: {memory_mb:.1f}MB"

        return ComponentHealth(
            name="memory",
            status=status,
            message=message,
            latency_ms=latency,
            last_check=datetime.now(),
            details={
                "rss_mb": memory_mb,
                "threshold_mb": threshold_mb,
            },
        )

    except ImportError:
        return ComponentHealth(
            name="memory",
            status=HealthStatus.UNKNOWN,
            message="psutil not available",
            latency_ms=0.0,
            last_check=datetime.now(),
        )
    except Exception as e:
        latency = (time.time() - start) * 1000
        return ComponentHealth(
            name="memory",
            status=HealthStatus.UNHEALTHY,
            message=f"Memory check failed: {str(e)}",
            latency_ms=latency,
            last_check=datetime.now(),
        )


def check_disk_health(path: str = ".", threshold_percent: float = 90.0) -> ComponentHealth:
    """
    Check disk usage.

    Args:
        path: Path to check disk usage for
        threshold_percent: Disk usage threshold for degraded status

    Returns:
        ComponentHealth with disk status
    """
    start = time.time()

    try:
        import psutil

        disk = psutil.disk_usage(path)
        usage_percent = disk.percent

        latency = (time.time() - start) * 1000

        if usage_percent > 95:
            status = HealthStatus.UNHEALTHY
            message = f"Critical disk usage: {usage_percent:.1f}%"
        elif usage_percent > threshold_percent:
            status = HealthStatus.DEGRADED
            message = f"High disk usage: {usage_percent:.1f}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"Disk usage: {usage_percent:.1f}%"

        return ComponentHealth(
            name="disk",
            status=status,
            message=message,
            latency_ms=latency,
            last_check=datetime.now(),
            details={
                "usage_percent": usage_percent,
                "free_gb": disk.free / (1024 ** 3),
                "total_gb": disk.total / (1024 ** 3),
            },
        )

    except ImportError:
        return ComponentHealth(
            name="disk",
            status=HealthStatus.UNKNOWN,
            message="psutil not available",
            latency_ms=0.0,
            last_check=datetime.now(),
        )
    except Exception as e:
        latency = (time.time() - start) * 1000
        return ComponentHealth(
            name="disk",
            status=HealthStatus.UNHEALTHY,
            message=f"Disk check failed: {str(e)}",
            latency_ms=latency,
            last_check=datetime.now(),
        )


# =============================================================================
# Health Check Manager
# =============================================================================

class HealthCheckManager:
    """
    Manages health checks for all system components.

    Provides:
    - Component registration
    - Periodic health checking
    - Health status aggregation
    - Health history tracking
    """

    def __init__(
        self,
        check_interval_seconds: float = 30.0,
        history_size: int = 100,
    ):
        """
        Initialize health check manager.

        Args:
            check_interval_seconds: Interval between automatic checks
            history_size: Number of health results to keep in history
        """
        self.check_interval = check_interval_seconds
        self.history_size = history_size

        # Component health checks
        self._checks: Dict[str, HealthCheckFunc] = {}
        self._async_checks: Dict[str, AsyncHealthCheckFunc] = {}

        # Health results
        self._current_health: Dict[str, ComponentHealth] = {}
        self._health_history: List[SystemHealth] = []

        # Timing
        self._start_time = datetime.now()
        self._last_check: Optional[datetime] = None

        # Threading
        self._lock = threading.Lock()
        self._running = False
        self._check_thread: Optional[threading.Thread] = None

        # Callbacks
        self._on_health_change: Optional[Callable[[ComponentHealth], None]] = None
        self._on_system_unhealthy: Optional[Callable[[SystemHealth], None]] = None

        logger.info(f"Health Check Manager initialized (interval={check_interval_seconds}s)")

    # -------------------------------------------------------------------------
    # Registration
    # -------------------------------------------------------------------------

    def register_check(
        self,
        name: str,
        check_func: HealthCheckFunc,
        replace: bool = False,
    ) -> None:
        """
        Register a health check function.

        Args:
            name: Component name
            check_func: Function that returns ComponentHealth
            replace: Whether to replace existing check
        """
        with self._lock:
            if name in self._checks and not replace:
                raise ValueError(f"Health check '{name}' already registered")
            self._checks[name] = check_func

        logger.debug(f"Registered health check: {name}")

    def register_async_check(
        self,
        name: str,
        check_func: AsyncHealthCheckFunc,
        replace: bool = False,
    ) -> None:
        """
        Register an async health check function.

        Args:
            name: Component name
            check_func: Async function that returns ComponentHealth
            replace: Whether to replace existing check
        """
        with self._lock:
            if name in self._async_checks and not replace:
                raise ValueError(f"Async health check '{name}' already registered")
            self._async_checks[name] = check_func

        logger.debug(f"Registered async health check: {name}")

    def unregister_check(self, name: str) -> bool:
        """
        Unregister a health check.

        Args:
            name: Component name

        Returns:
            True if check was removed
        """
        with self._lock:
            removed = False
            if name in self._checks:
                del self._checks[name]
                removed = True
            if name in self._async_checks:
                del self._async_checks[name]
                removed = True
            if name in self._current_health:
                del self._current_health[name]

        return removed

    def get_registered_checks(self) -> List[str]:
        """Get list of registered check names."""
        with self._lock:
            return list(set(self._checks.keys()) | set(self._async_checks.keys()))

    # -------------------------------------------------------------------------
    # Health Checking
    # -------------------------------------------------------------------------

    def check_component(self, name: str) -> Optional[ComponentHealth]:
        """
        Check health of a specific component.

        Args:
            name: Component name

        Returns:
            ComponentHealth or None if not registered
        """
        with self._lock:
            check_func = self._checks.get(name)

        if check_func is None:
            return None

        try:
            result = check_func()

            with self._lock:
                old_health = self._current_health.get(name)
                self._current_health[name] = result

                # Notify on status change
                if (
                    old_health is not None
                    and old_health.status != result.status
                    and self._on_health_change
                ):
                    self._on_health_change(result)

            return result

        except Exception as e:
            logger.error(f"Health check '{name}' failed: {e}")
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {str(e)}",
                last_check=datetime.now(),
            )

    def check_health(self) -> SystemHealth:
        """
        Check health of all registered components.

        Returns:
            SystemHealth with all component statuses
        """
        components: Dict[str, ComponentHealth] = {}

        # Run all sync checks
        with self._lock:
            checks = dict(self._checks)

        for name, check_func in checks.items():
            try:
                result = check_func()
                components[name] = result
            except Exception as e:
                logger.error(f"Health check '{name}' failed: {e}")
                components[name] = ComponentHealth(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check failed: {str(e)}",
                    last_check=datetime.now(),
                )

        # Determine overall status
        overall_status = self._determine_overall_status(components)

        # Create system health
        uptime = (datetime.now() - self._start_time).total_seconds()
        system_health = SystemHealth(
            status=overall_status,
            timestamp=datetime.now(),
            components=components,
            uptime_seconds=uptime,
        )

        # Update state
        with self._lock:
            self._current_health = components
            self._last_check = datetime.now()

            # Add to history
            self._health_history.append(system_health)
            if len(self._health_history) > self.history_size:
                self._health_history = self._health_history[-self.history_size:]

            # Notify if unhealthy
            if overall_status == HealthStatus.UNHEALTHY and self._on_system_unhealthy:
                self._on_system_unhealthy(system_health)

        return system_health

    async def check_health_async(self) -> SystemHealth:
        """
        Check health of all components including async checks.

        Returns:
            SystemHealth with all component statuses
        """
        components: Dict[str, ComponentHealth] = {}

        # Run sync checks
        with self._lock:
            sync_checks = dict(self._checks)
            async_checks = dict(self._async_checks)

        for name, check_func in sync_checks.items():
            try:
                result = check_func()
                components[name] = result
            except Exception as e:
                components[name] = ComponentHealth(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check failed: {str(e)}",
                    last_check=datetime.now(),
                )

        # Run async checks
        for name, check_func in async_checks.items():
            try:
                result = await check_func()
                components[name] = result
            except Exception as e:
                components[name] = ComponentHealth(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Async check failed: {str(e)}",
                    last_check=datetime.now(),
                )

        overall_status = self._determine_overall_status(components)
        uptime = (datetime.now() - self._start_time).total_seconds()

        return SystemHealth(
            status=overall_status,
            timestamp=datetime.now(),
            components=components,
            uptime_seconds=uptime,
        )

    def _determine_overall_status(
        self, components: Dict[str, ComponentHealth]
    ) -> HealthStatus:
        """Determine overall health status from components."""
        if not components:
            return HealthStatus.UNKNOWN

        statuses = [c.status for c in components.values()]

        if any(s == HealthStatus.UNHEALTHY for s in statuses):
            return HealthStatus.UNHEALTHY
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            return HealthStatus.DEGRADED
        elif any(s == HealthStatus.UNKNOWN for s in statuses):
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY

    # -------------------------------------------------------------------------
    # Background Checking
    # -------------------------------------------------------------------------

    def start_background_checks(self) -> None:
        """Start background health checking thread."""
        if self._running:
            return

        self._running = True
        self._check_thread = threading.Thread(
            target=self._background_check_loop,
            daemon=True,
        )
        self._check_thread.start()
        logger.info("Background health checks started")

    def stop_background_checks(self) -> None:
        """Stop background health checking."""
        self._running = False
        if self._check_thread:
            self._check_thread.join(timeout=5.0)
            self._check_thread = None
        logger.info("Background health checks stopped")

    def _background_check_loop(self) -> None:
        """Background thread loop for periodic health checks."""
        while self._running:
            try:
                self.check_health()
            except Exception as e:
                logger.error(f"Background health check failed: {e}")

            # Sleep in small increments for responsive shutdown
            for _ in range(int(self.check_interval * 10)):
                if not self._running:
                    break
                time.sleep(0.1)

    # -------------------------------------------------------------------------
    # Status and History
    # -------------------------------------------------------------------------

    def get_current_health(self) -> Optional[SystemHealth]:
        """Get most recent health status."""
        with self._lock:
            if not self._health_history:
                return None
            return self._health_history[-1]

    def get_component_health(self, name: str) -> Optional[ComponentHealth]:
        """Get current health of a specific component."""
        with self._lock:
            return self._current_health.get(name)

    def get_health_history(
        self,
        limit: Optional[int] = None,
    ) -> List[SystemHealth]:
        """Get health check history."""
        with self._lock:
            history = list(self._health_history)

        if limit:
            history = history[-limit:]
        return history

    def get_uptime(self) -> timedelta:
        """Get system uptime."""
        return datetime.now() - self._start_time

    def is_healthy(self) -> bool:
        """Check if system is currently healthy."""
        current = self.get_current_health()
        if current is None:
            return True  # No checks = healthy
        return current.status == HealthStatus.HEALTHY

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def set_callbacks(
        self,
        on_health_change: Optional[Callable[[ComponentHealth], None]] = None,
        on_system_unhealthy: Optional[Callable[[SystemHealth], None]] = None,
    ) -> None:
        """Set callback functions for health events."""
        self._on_health_change = on_health_change
        self._on_system_unhealthy = on_system_unhealthy

    # -------------------------------------------------------------------------
    # Default Checks Setup
    # -------------------------------------------------------------------------

    def register_default_checks(
        self,
        broker: Optional[Any] = None,
        data_client: Optional[Any] = None,
        db_connection: Optional[Any] = None,
    ) -> None:
        """
        Register default health checks.

        Args:
            broker: Broker instance
            data_client: Data client instance
            db_connection: Database connection
        """
        # Always register circuit breaker check
        self.register_check(
            "circuit_breakers",
            check_circuit_breakers_health,
        )

        # Always register memory check
        self.register_check(
            "memory",
            check_memory_health,
        )

        # Always register disk check
        self.register_check(
            "disk",
            check_disk_health,
        )

        # Conditional checks
        if broker is not None:
            self.register_check(
                "broker",
                lambda: check_broker_health(broker),
            )

        if data_client is not None:
            self.register_check(
                "data_feed",
                lambda: check_data_feed_health(data_client),
            )

        if db_connection is not None:
            self.register_check(
                "database",
                lambda: check_database_health(db_connection),
            )

        logger.info("Default health checks registered")


# =============================================================================
# Global Instance
# =============================================================================

_manager_instance: Optional[HealthCheckManager] = None


def get_health_check_manager(
    check_interval_seconds: float = 30.0,
) -> HealthCheckManager:
    """Get global health check manager instance."""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = HealthCheckManager(
            check_interval_seconds=check_interval_seconds,
        )
    return _manager_instance


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "HealthStatus",
    # Data classes
    "ComponentHealth",
    "SystemHealth",
    # Classes
    "HealthCheckManager",
    # Functions
    "get_health_check_manager",
    "create_health_result",
    # Built-in checks
    "check_broker_health",
    "check_data_feed_health",
    "check_database_health",
    "check_circuit_breakers_health",
    "check_memory_health",
    "check_disk_health",
]
