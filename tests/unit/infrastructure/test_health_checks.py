"""
Unit tests for Health Check Endpoints module.

Tests cover:
- HealthStatus enum
- ComponentHealth and SystemHealth dataclasses
- Built-in health check functions
- HealthCheckManager class
- Background checking
- Callbacks
- Global instance
"""

import asyncio
import threading
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.infrastructure.health_checks import (
    HealthStatus,
    ComponentHealth,
    SystemHealth,
    HealthCheckManager,
    get_health_check_manager,
    create_health_result,
    check_broker_health,
    check_data_feed_health,
    check_database_health,
    check_circuit_breakers_health,
    check_memory_health,
    check_disk_health,
)


# =============================================================================
# Test HealthStatus Enum
# =============================================================================

class TestHealthStatus:
    """Tests for HealthStatus enum."""

    def test_healthy_value(self):
        """Test HEALTHY status value."""
        assert HealthStatus.HEALTHY.value == "healthy"

    def test_degraded_value(self):
        """Test DEGRADED status value."""
        assert HealthStatus.DEGRADED.value == "degraded"

    def test_unhealthy_value(self):
        """Test UNHEALTHY status value."""
        assert HealthStatus.UNHEALTHY.value == "unhealthy"

    def test_unknown_value(self):
        """Test UNKNOWN status value."""
        assert HealthStatus.UNKNOWN.value == "unknown"

    def test_all_statuses_exist(self):
        """Test all expected statuses exist."""
        statuses = [s.value for s in HealthStatus]
        assert "healthy" in statuses
        assert "degraded" in statuses
        assert "unhealthy" in statuses
        assert "unknown" in statuses


# =============================================================================
# Test ComponentHealth Dataclass
# =============================================================================

class TestComponentHealth:
    """Tests for ComponentHealth dataclass."""

    def test_basic_creation(self):
        """Test basic ComponentHealth creation."""
        health = ComponentHealth(
            name="test",
            status=HealthStatus.HEALTHY,
        )
        assert health.name == "test"
        assert health.status == HealthStatus.HEALTHY
        assert health.message == ""
        assert health.latency_ms == 0.0

    def test_full_creation(self):
        """Test ComponentHealth with all fields."""
        now = datetime.now()
        health = ComponentHealth(
            name="broker",
            status=HealthStatus.DEGRADED,
            message="Slow response",
            latency_ms=150.5,
            last_check=now,
            details={"retries": 2},
        )
        assert health.name == "broker"
        assert health.status == HealthStatus.DEGRADED
        assert health.message == "Slow response"
        assert health.latency_ms == 150.5
        assert health.last_check == now
        assert health.details == {"retries": 2}

    def test_to_dict(self):
        """Test ComponentHealth to_dict conversion."""
        health = ComponentHealth(
            name="test",
            status=HealthStatus.HEALTHY,
            message="OK",
            latency_ms=10.5,
            details={"key": "value"},
        )
        d = health.to_dict()

        assert d["name"] == "test"
        assert d["status"] == "healthy"
        assert d["message"] == "OK"
        assert d["latency_ms"] == 10.5
        assert "last_check" in d
        assert d["details"] == {"key": "value"}

    def test_to_dict_serialization(self):
        """Test to_dict produces JSON-serializable output."""
        import json
        health = ComponentHealth(
            name="test",
            status=HealthStatus.UNHEALTHY,
        )
        # Should not raise
        json_str = json.dumps(health.to_dict())
        assert "test" in json_str


# =============================================================================
# Test SystemHealth Dataclass
# =============================================================================

class TestSystemHealth:
    """Tests for SystemHealth dataclass."""

    def test_basic_creation(self):
        """Test basic SystemHealth creation."""
        health = SystemHealth(
            status=HealthStatus.HEALTHY,
            timestamp=datetime.now(),
            components={},
            uptime_seconds=100.0,
        )
        assert health.status == HealthStatus.HEALTHY
        assert health.uptime_seconds == 100.0
        assert health.version == "1.0.0"

    def test_with_components(self):
        """Test SystemHealth with components."""
        comp1 = ComponentHealth(name="c1", status=HealthStatus.HEALTHY)
        comp2 = ComponentHealth(name="c2", status=HealthStatus.DEGRADED)

        health = SystemHealth(
            status=HealthStatus.DEGRADED,
            timestamp=datetime.now(),
            components={"c1": comp1, "c2": comp2},
            uptime_seconds=50.0,
        )
        assert len(health.components) == 2
        assert "c1" in health.components
        assert "c2" in health.components

    def test_to_dict(self):
        """Test SystemHealth to_dict conversion."""
        comp = ComponentHealth(name="test", status=HealthStatus.HEALTHY)
        health = SystemHealth(
            status=HealthStatus.HEALTHY,
            timestamp=datetime.now(),
            components={"test": comp},
            uptime_seconds=100.0,
            version="2.0.0",
        )
        d = health.to_dict()

        assert d["status"] == "healthy"
        assert d["uptime_seconds"] == 100.0
        assert d["version"] == "2.0.0"
        assert "timestamp" in d
        assert "components" in d
        assert "test" in d["components"]

    def test_to_dict_serialization(self):
        """Test to_dict produces JSON-serializable output."""
        import json
        comp = ComponentHealth(name="test", status=HealthStatus.HEALTHY)
        health = SystemHealth(
            status=HealthStatus.HEALTHY,
            timestamp=datetime.now(),
            components={"test": comp},
            uptime_seconds=100.0,
        )
        # Should not raise
        json_str = json.dumps(health.to_dict())
        assert "healthy" in json_str


# =============================================================================
# Test create_health_result Helper
# =============================================================================

class TestCreateHealthResult:
    """Tests for create_health_result helper."""

    def test_healthy_result(self):
        """Test creating healthy result."""
        result = create_health_result(
            name="test",
            healthy=True,
            message="All good",
        )
        assert result.name == "test"
        assert result.status == HealthStatus.HEALTHY
        assert result.message == "All good"

    def test_unhealthy_result(self):
        """Test creating unhealthy result."""
        result = create_health_result(
            name="test",
            healthy=False,
            message="Error occurred",
        )
        assert result.status == HealthStatus.UNHEALTHY
        assert result.message == "Error occurred"

    def test_with_latency(self):
        """Test result with latency."""
        result = create_health_result(
            name="test",
            healthy=True,
            latency_ms=50.5,
        )
        assert result.latency_ms == 50.5

    def test_with_details(self):
        """Test result with details."""
        result = create_health_result(
            name="test",
            healthy=True,
            details={"count": 10},
        )
        assert result.details == {"count": 10}


# =============================================================================
# Test check_broker_health
# =============================================================================

class TestCheckBrokerHealth:
    """Tests for check_broker_health function."""

    def test_no_broker(self):
        """Test with no broker configured."""
        result = check_broker_health(None)
        assert result.name == "broker"
        assert result.status == HealthStatus.UNKNOWN
        assert "No broker" in result.message

    def test_broker_is_connected(self):
        """Test broker with is_connected method."""
        broker = Mock()
        broker.is_connected.return_value = True

        result = check_broker_health(broker)
        assert result.status == HealthStatus.HEALTHY
        assert "connected" in result.message.lower()

    def test_broker_disconnected(self):
        """Test broker that is disconnected."""
        broker = Mock()
        broker.is_connected.return_value = False

        result = check_broker_health(broker)
        assert result.status == HealthStatus.UNHEALTHY
        assert "disconnected" in result.message.lower()

    def test_broker_health_check_method(self):
        """Test broker with health_check method."""
        broker = Mock(spec=["health_check"])
        broker.health_check.return_value = True

        result = check_broker_health(broker)
        assert result.status == HealthStatus.HEALTHY

    def test_broker_get_account_method(self):
        """Test broker with get_account method as connectivity test."""
        broker = Mock(spec=["get_account"])
        broker.get_account.return_value = {"balance": 1000}

        result = check_broker_health(broker)
        assert result.status == HealthStatus.HEALTHY

    def test_broker_exception(self):
        """Test broker that raises exception."""
        broker = Mock()
        broker.is_connected.side_effect = Exception("Connection error")

        result = check_broker_health(broker)
        assert result.status == HealthStatus.UNHEALTHY
        assert "failed" in result.message.lower()

    def test_latency_measured(self):
        """Test that latency is measured."""
        broker = Mock()
        broker.is_connected.return_value = True

        result = check_broker_health(broker)
        assert result.latency_ms >= 0


# =============================================================================
# Test check_data_feed_health
# =============================================================================

class TestCheckDataFeedHealth:
    """Tests for check_data_feed_health function."""

    def test_no_client(self):
        """Test with no data client."""
        result = check_data_feed_health(None)
        assert result.name == "data_feed"
        assert result.status == HealthStatus.UNKNOWN

    def test_client_health_check(self):
        """Test client with health_check method."""
        client = Mock()
        client.health_check.return_value = True

        result = check_data_feed_health(client)
        assert result.status == HealthStatus.HEALTHY

    def test_client_unhealthy(self):
        """Test unhealthy client."""
        client = Mock()
        client.health_check.return_value = False

        result = check_data_feed_health(client)
        assert result.status == HealthStatus.UNHEALTHY

    def test_client_is_connected(self):
        """Test client with is_connected method."""
        client = Mock(spec=["is_connected"])
        client.is_connected.return_value = True

        result = check_data_feed_health(client)
        assert result.status == HealthStatus.HEALTHY

    def test_client_get_quote(self):
        """Test client with get_quote method."""
        client = Mock(spec=["get_quote"])
        client.get_quote.return_value = {"price": 100}

        result = check_data_feed_health(client)
        assert result.status == HealthStatus.HEALTHY

    def test_client_exception(self):
        """Test client that raises exception."""
        client = Mock()
        client.health_check.side_effect = Exception("API error")

        result = check_data_feed_health(client)
        assert result.status == HealthStatus.UNHEALTHY


# =============================================================================
# Test check_database_health
# =============================================================================

class TestCheckDatabaseHealth:
    """Tests for check_database_health function."""

    def test_no_connection(self):
        """Test with no database connection."""
        result = check_database_health(None)
        assert result.name == "database"
        assert result.status == HealthStatus.UNKNOWN

    def test_db_execute(self):
        """Test database with execute method."""
        db = Mock()
        db.execute.return_value = None

        result = check_database_health(db)
        assert result.status == HealthStatus.HEALTHY
        db.execute.assert_called_once_with("SELECT 1")

    def test_db_ping(self):
        """Test database with ping method."""
        db = Mock(spec=["ping"])
        db.ping.return_value = None

        result = check_database_health(db)
        assert result.status == HealthStatus.HEALTHY
        db.ping.assert_called_once()

    def test_db_is_connected(self):
        """Test database with is_connected method."""
        db = Mock(spec=["is_connected"])
        db.is_connected.return_value = True

        result = check_database_health(db)
        assert result.status == HealthStatus.HEALTHY

    def test_db_not_connected(self):
        """Test database that is not connected."""
        db = Mock(spec=["is_connected"])
        db.is_connected.return_value = False

        result = check_database_health(db)
        assert result.status == HealthStatus.UNHEALTHY

    def test_db_exception(self):
        """Test database that raises exception."""
        db = Mock()
        db.execute.side_effect = Exception("Connection lost")

        result = check_database_health(db)
        assert result.status == HealthStatus.UNHEALTHY


# =============================================================================
# Test check_circuit_breakers_health
# =============================================================================

class TestCheckCircuitBreakersHealth:
    """Tests for check_circuit_breakers_health function."""

    def _get_cb_module(self):
        """Get the actual circuit_breaker module."""
        import importlib
        return importlib.import_module("src.utils.circuit_breaker")

    def test_all_closed(self):
        """Test with all breakers closed."""
        cb_module = self._get_cb_module()

        mock_registry = Mock()
        mock_registry.get_status.return_value = {
            "api": {"state": "CLOSED"},
            "db": {"state": "CLOSED"},
        }

        with patch.object(
            cb_module,
            "get_circuit_breaker_registry",
            return_value=mock_registry,
        ):
            result = check_circuit_breakers_health()
            assert result.status == HealthStatus.HEALTHY
            assert "2 breakers closed" in result.message

    def test_open_breakers(self):
        """Test with open breakers."""
        cb_module = self._get_cb_module()

        mock_registry = Mock()
        mock_registry.get_status.return_value = {
            "api": {"state": "OPEN"},
            "db": {"state": "CLOSED"},
        }

        with patch.object(
            cb_module,
            "get_circuit_breaker_registry",
            return_value=mock_registry,
        ):
            result = check_circuit_breakers_health()
            assert result.status == HealthStatus.DEGRADED
            assert "api" in result.message
            assert result.details["open"] == ["api"]

    def test_half_open_breakers(self):
        """Test with half-open breakers."""
        cb_module = self._get_cb_module()

        mock_registry = Mock()
        mock_registry.get_status.return_value = {
            "api": {"state": "HALF_OPEN"},
        }

        with patch.object(
            cb_module,
            "get_circuit_breaker_registry",
            return_value=mock_registry,
        ):
            result = check_circuit_breakers_health()
            assert result.status == HealthStatus.DEGRADED
            assert "Recovering" in result.message
            assert result.details["half_open"] == ["api"]

    def test_import_error(self):
        """Test when circuit breaker module not available."""
        # Simulate import error by making the module unavailable
        import sys
        original_module = sys.modules.get("src.utils.circuit_breaker")

        # Remove the module temporarily
        if "src.utils.circuit_breaker" in sys.modules:
            del sys.modules["src.utils.circuit_breaker"]

        # Patch to raise ImportError
        with patch.dict("sys.modules", {"src.utils.circuit_breaker": None}):
            result = check_circuit_breakers_health()
            assert result.status == HealthStatus.UNKNOWN

        # Restore
        if original_module:
            sys.modules["src.utils.circuit_breaker"] = original_module

    def test_exception(self):
        """Test when registry throws exception."""
        cb_module = self._get_cb_module()

        mock_registry = Mock()
        mock_registry.get_status.side_effect = Exception("Error")

        with patch.object(
            cb_module,
            "get_circuit_breaker_registry",
            return_value=mock_registry,
        ):
            result = check_circuit_breakers_health()
            assert result.status == HealthStatus.UNHEALTHY


# =============================================================================
# Test check_memory_health
# =============================================================================

class TestCheckMemoryHealth:
    """Tests for check_memory_health function."""

    def test_healthy_memory(self):
        """Test with healthy memory usage."""
        mock_process = Mock()
        mock_process.memory_info.return_value.rss = 500 * 1024 * 1024  # 500MB

        with patch("psutil.Process", return_value=mock_process):
            result = check_memory_health(threshold_mb=1024.0)
            assert result.status == HealthStatus.HEALTHY
            assert result.details["rss_mb"] == pytest.approx(500.0, rel=0.01)

    def test_degraded_memory(self):
        """Test with elevated memory usage."""
        mock_process = Mock()
        mock_process.memory_info.return_value.rss = 1100 * 1024 * 1024  # 1100MB

        with patch("psutil.Process", return_value=mock_process):
            result = check_memory_health(threshold_mb=1024.0)
            assert result.status == HealthStatus.DEGRADED
            assert "Elevated" in result.message

    def test_unhealthy_memory(self):
        """Test with high memory usage."""
        mock_process = Mock()
        mock_process.memory_info.return_value.rss = 2000 * 1024 * 1024  # 2000MB

        with patch("psutil.Process", return_value=mock_process):
            result = check_memory_health(threshold_mb=1024.0)
            assert result.status == HealthStatus.UNHEALTHY
            assert "High memory" in result.message

    def test_psutil_not_available(self):
        """Test when psutil is not available."""
        # Mock psutil to raise ImportError when accessed
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "psutil":
                raise ImportError("No module named 'psutil'")
            return real_import(name, *args, **kwargs)

        # This test verifies the error handling path exists
        # Since psutil IS available in our test environment, we test the exception path
        with patch("psutil.Process", side_effect=Exception("psutil error")):
            result = check_memory_health()
            assert result.status == HealthStatus.UNHEALTHY

    def test_exception(self):
        """Test when memory check fails."""
        with patch("psutil.Process", side_effect=Exception("Error")):
            result = check_memory_health()
            assert result.status == HealthStatus.UNHEALTHY


# =============================================================================
# Test check_disk_health
# =============================================================================

class TestCheckDiskHealth:
    """Tests for check_disk_health function."""

    def test_healthy_disk(self):
        """Test with healthy disk usage."""
        mock_disk = Mock()
        mock_disk.percent = 50.0
        mock_disk.free = 100 * 1024 ** 3  # 100GB
        mock_disk.total = 200 * 1024 ** 3  # 200GB

        with patch("psutil.disk_usage", return_value=mock_disk):
            result = check_disk_health(threshold_percent=90.0)
            assert result.status == HealthStatus.HEALTHY
            assert result.details["usage_percent"] == 50.0

    def test_degraded_disk(self):
        """Test with high disk usage."""
        mock_disk = Mock()
        mock_disk.percent = 92.0
        mock_disk.free = 16 * 1024 ** 3
        mock_disk.total = 200 * 1024 ** 3

        with patch("psutil.disk_usage", return_value=mock_disk):
            result = check_disk_health(threshold_percent=90.0)
            assert result.status == HealthStatus.DEGRADED
            assert "High disk" in result.message

    def test_unhealthy_disk(self):
        """Test with critical disk usage."""
        mock_disk = Mock()
        mock_disk.percent = 97.0
        mock_disk.free = 6 * 1024 ** 3
        mock_disk.total = 200 * 1024 ** 3

        with patch("psutil.disk_usage", return_value=mock_disk):
            result = check_disk_health()
            assert result.status == HealthStatus.UNHEALTHY
            assert "Critical" in result.message

    def test_custom_threshold(self):
        """Test with custom threshold."""
        mock_disk = Mock()
        mock_disk.percent = 85.0
        mock_disk.free = 30 * 1024 ** 3
        mock_disk.total = 200 * 1024 ** 3

        with patch("psutil.disk_usage", return_value=mock_disk):
            result = check_disk_health(threshold_percent=80.0)
            assert result.status == HealthStatus.DEGRADED

    def test_psutil_not_available(self):
        """Test when psutil is not available."""
        with patch("psutil.disk_usage", side_effect=ImportError):
            result = check_disk_health()
            assert result.status == HealthStatus.UNKNOWN


# =============================================================================
# Test HealthCheckManager
# =============================================================================

class TestHealthCheckManager:
    """Tests for HealthCheckManager class."""

    @pytest.fixture
    def manager(self):
        """Create fresh manager for each test."""
        return HealthCheckManager(check_interval_seconds=1.0)

    def test_initialization(self, manager):
        """Test manager initialization."""
        assert manager.check_interval == 1.0
        assert manager.history_size == 100
        assert len(manager.get_registered_checks()) == 0

    def test_register_check(self, manager):
        """Test registering a health check."""
        def check_func():
            return ComponentHealth(name="test", status=HealthStatus.HEALTHY)

        manager.register_check("test", check_func)
        assert "test" in manager.get_registered_checks()

    def test_register_duplicate_check_fails(self, manager):
        """Test registering duplicate check fails."""
        def check_func():
            return ComponentHealth(name="test", status=HealthStatus.HEALTHY)

        manager.register_check("test", check_func)

        with pytest.raises(ValueError, match="already registered"):
            manager.register_check("test", check_func)

    def test_register_duplicate_with_replace(self, manager):
        """Test registering duplicate with replace=True."""
        def check_func1():
            return ComponentHealth(name="test", status=HealthStatus.HEALTHY)

        def check_func2():
            return ComponentHealth(name="test", status=HealthStatus.DEGRADED)

        manager.register_check("test", check_func1)
        manager.register_check("test", check_func2, replace=True)

        result = manager.check_component("test")
        assert result.status == HealthStatus.DEGRADED

    def test_unregister_check(self, manager):
        """Test unregistering a health check."""
        def check_func():
            return ComponentHealth(name="test", status=HealthStatus.HEALTHY)

        manager.register_check("test", check_func)
        result = manager.unregister_check("test")

        assert result is True
        assert "test" not in manager.get_registered_checks()

    def test_unregister_nonexistent(self, manager):
        """Test unregistering non-existent check."""
        result = manager.unregister_check("nonexistent")
        assert result is False

    def test_check_component(self, manager):
        """Test checking single component."""
        def check_func():
            return ComponentHealth(
                name="test",
                status=HealthStatus.HEALTHY,
                message="OK",
            )

        manager.register_check("test", check_func)
        result = manager.check_component("test")

        assert result is not None
        assert result.status == HealthStatus.HEALTHY

    def test_check_nonexistent_component(self, manager):
        """Test checking non-existent component."""
        result = manager.check_component("nonexistent")
        assert result is None

    def test_check_component_exception(self, manager):
        """Test component that throws exception."""
        def check_func():
            raise Exception("Check failed")

        manager.register_check("test", check_func)
        result = manager.check_component("test")

        assert result.status == HealthStatus.UNHEALTHY
        assert "failed" in result.message.lower()

    def test_check_health_all_healthy(self, manager):
        """Test check_health with all healthy components."""
        manager.register_check(
            "c1",
            lambda: ComponentHealth(name="c1", status=HealthStatus.HEALTHY),
        )
        manager.register_check(
            "c2",
            lambda: ComponentHealth(name="c2", status=HealthStatus.HEALTHY),
        )

        result = manager.check_health()

        assert result.status == HealthStatus.HEALTHY
        assert len(result.components) == 2

    def test_check_health_one_unhealthy(self, manager):
        """Test check_health with one unhealthy component."""
        manager.register_check(
            "c1",
            lambda: ComponentHealth(name="c1", status=HealthStatus.HEALTHY),
        )
        manager.register_check(
            "c2",
            lambda: ComponentHealth(name="c2", status=HealthStatus.UNHEALTHY),
        )

        result = manager.check_health()

        assert result.status == HealthStatus.UNHEALTHY

    def test_check_health_one_degraded(self, manager):
        """Test check_health with one degraded component."""
        manager.register_check(
            "c1",
            lambda: ComponentHealth(name="c1", status=HealthStatus.HEALTHY),
        )
        manager.register_check(
            "c2",
            lambda: ComponentHealth(name="c2", status=HealthStatus.DEGRADED),
        )

        result = manager.check_health()

        assert result.status == HealthStatus.DEGRADED

    def test_check_health_unknown_degrades(self, manager):
        """Test that UNKNOWN status causes DEGRADED overall."""
        manager.register_check(
            "c1",
            lambda: ComponentHealth(name="c1", status=HealthStatus.HEALTHY),
        )
        manager.register_check(
            "c2",
            lambda: ComponentHealth(name="c2", status=HealthStatus.UNKNOWN),
        )

        result = manager.check_health()

        assert result.status == HealthStatus.DEGRADED

    def test_check_health_no_components(self, manager):
        """Test check_health with no components."""
        result = manager.check_health()
        assert result.status == HealthStatus.UNKNOWN

    def test_uptime_tracking(self, manager):
        """Test uptime is tracked."""
        time.sleep(0.1)
        uptime = manager.get_uptime()
        assert uptime.total_seconds() >= 0.1

    def test_is_healthy_with_no_checks(self, manager):
        """Test is_healthy with no checks returns True."""
        assert manager.is_healthy() is True

    def test_is_healthy_after_check(self, manager):
        """Test is_healthy after health check."""
        manager.register_check(
            "test",
            lambda: ComponentHealth(name="test", status=HealthStatus.HEALTHY),
        )
        manager.check_health()

        assert manager.is_healthy() is True

    def test_health_history(self, manager):
        """Test health history is maintained."""
        manager.register_check(
            "test",
            lambda: ComponentHealth(name="test", status=HealthStatus.HEALTHY),
        )

        manager.check_health()
        manager.check_health()
        manager.check_health()

        history = manager.get_health_history()
        assert len(history) == 3

    def test_health_history_limit(self, manager):
        """Test health history respects limit."""
        history = manager.get_health_history(limit=2)
        # Should not error even with empty history
        assert isinstance(history, list)

    def test_get_current_health(self, manager):
        """Test get_current_health."""
        manager.register_check(
            "test",
            lambda: ComponentHealth(name="test", status=HealthStatus.HEALTHY),
        )

        # Before any check
        assert manager.get_current_health() is None

        manager.check_health()

        current = manager.get_current_health()
        assert current is not None
        assert current.status == HealthStatus.HEALTHY

    def test_get_component_health(self, manager):
        """Test get_component_health."""
        manager.register_check(
            "test",
            lambda: ComponentHealth(name="test", status=HealthStatus.HEALTHY),
        )

        manager.check_health()

        comp = manager.get_component_health("test")
        assert comp is not None
        assert comp.name == "test"


# =============================================================================
# Test HealthCheckManager Async
# =============================================================================

class TestHealthCheckManagerAsync:
    """Tests for async health check functionality."""

    @pytest.fixture
    def manager(self):
        """Create fresh manager for each test."""
        return HealthCheckManager()

    def test_register_async_check(self, manager):
        """Test registering async health check."""
        async def async_check():
            return ComponentHealth(name="async_test", status=HealthStatus.HEALTHY)

        manager.register_async_check("async_test", async_check)
        assert "async_test" in manager.get_registered_checks()

    def test_register_duplicate_async_check_fails(self, manager):
        """Test registering duplicate async check fails."""
        async def async_check():
            return ComponentHealth(name="test", status=HealthStatus.HEALTHY)

        manager.register_async_check("test", async_check)

        with pytest.raises(ValueError, match="already registered"):
            manager.register_async_check("test", async_check)

    @pytest.mark.asyncio
    async def test_check_health_async(self, manager):
        """Test async health check."""
        manager.register_check(
            "sync",
            lambda: ComponentHealth(name="sync", status=HealthStatus.HEALTHY),
        )

        async def async_check():
            return ComponentHealth(name="async", status=HealthStatus.HEALTHY)

        manager.register_async_check("async", async_check)

        result = await manager.check_health_async()

        assert result.status == HealthStatus.HEALTHY
        assert "sync" in result.components
        assert "async" in result.components

    @pytest.mark.asyncio
    async def test_check_health_async_exception(self, manager):
        """Test async check that raises exception."""
        async def failing_check():
            raise Exception("Async error")

        manager.register_async_check("failing", failing_check)

        result = await manager.check_health_async()

        assert result.components["failing"].status == HealthStatus.UNHEALTHY


# =============================================================================
# Test HealthCheckManager Background Checks
# =============================================================================

class TestHealthCheckManagerBackground:
    """Tests for background health checking."""

    @pytest.fixture
    def manager(self):
        """Create manager with short interval."""
        return HealthCheckManager(check_interval_seconds=0.1)

    def test_start_stop_background_checks(self, manager):
        """Test starting and stopping background checks."""
        check_count = []

        def counting_check():
            check_count.append(1)
            return ComponentHealth(name="test", status=HealthStatus.HEALTHY)

        manager.register_check("test", counting_check)

        manager.start_background_checks()
        time.sleep(0.35)  # Allow ~3 checks
        manager.stop_background_checks()

        # Should have run multiple times
        assert len(check_count) >= 2

    def test_start_twice_is_safe(self, manager):
        """Test starting twice doesn't create duplicate threads."""
        manager.start_background_checks()
        manager.start_background_checks()  # Should be no-op

        # Should only have one thread
        assert manager._running is True
        manager.stop_background_checks()

    def test_stop_without_start(self, manager):
        """Test stopping without starting is safe."""
        manager.stop_background_checks()  # Should not error


# =============================================================================
# Test HealthCheckManager Callbacks
# =============================================================================

class TestHealthCheckManagerCallbacks:
    """Tests for health check callbacks."""

    @pytest.fixture
    def manager(self):
        """Create fresh manager."""
        return HealthCheckManager()

    def test_on_health_change_callback(self, manager):
        """Test on_health_change callback is called."""
        changes = []
        manager.set_callbacks(on_health_change=lambda h: changes.append(h))

        status_toggle = [HealthStatus.HEALTHY, HealthStatus.UNHEALTHY]
        call_count = [0]

        def toggling_check():
            status = status_toggle[call_count[0] % 2]
            call_count[0] += 1
            return ComponentHealth(name="toggle", status=status)

        manager.register_check("toggle", toggling_check)

        manager.check_component("toggle")  # HEALTHY (no prev, no callback)
        manager.check_component("toggle")  # UNHEALTHY (change, callback)
        manager.check_component("toggle")  # HEALTHY (change, callback)

        assert len(changes) == 2

    def test_on_system_unhealthy_callback(self, manager):
        """Test on_system_unhealthy callback is called."""
        unhealthy_events = []
        manager.set_callbacks(on_system_unhealthy=lambda h: unhealthy_events.append(h))

        manager.register_check(
            "unhealthy",
            lambda: ComponentHealth(name="unhealthy", status=HealthStatus.UNHEALTHY),
        )

        manager.check_health()

        assert len(unhealthy_events) == 1
        assert unhealthy_events[0].status == HealthStatus.UNHEALTHY


# =============================================================================
# Test HealthCheckManager Default Checks
# =============================================================================

class TestHealthCheckManagerDefaultChecks:
    """Tests for default checks registration."""

    def test_register_default_checks_minimal(self):
        """Test registering default checks with no components."""
        manager = HealthCheckManager()

        with patch("src.infrastructure.health_checks.check_circuit_breakers_health"), \
             patch("src.infrastructure.health_checks.check_memory_health"), \
             patch("src.infrastructure.health_checks.check_disk_health"):
            manager.register_default_checks()

        checks = manager.get_registered_checks()
        assert "circuit_breakers" in checks
        assert "memory" in checks
        assert "disk" in checks

    def test_register_default_checks_with_broker(self):
        """Test registering default checks with broker."""
        manager = HealthCheckManager()
        broker = Mock()

        with patch("src.infrastructure.health_checks.check_circuit_breakers_health"), \
             patch("src.infrastructure.health_checks.check_memory_health"), \
             patch("src.infrastructure.health_checks.check_disk_health"):
            manager.register_default_checks(broker=broker)

        assert "broker" in manager.get_registered_checks()

    def test_register_default_checks_with_data_client(self):
        """Test registering default checks with data client."""
        manager = HealthCheckManager()
        client = Mock()

        with patch("src.infrastructure.health_checks.check_circuit_breakers_health"), \
             patch("src.infrastructure.health_checks.check_memory_health"), \
             patch("src.infrastructure.health_checks.check_disk_health"):
            manager.register_default_checks(data_client=client)

        assert "data_feed" in manager.get_registered_checks()

    def test_register_default_checks_with_database(self):
        """Test registering default checks with database."""
        manager = HealthCheckManager()
        db = Mock()

        with patch("src.infrastructure.health_checks.check_circuit_breakers_health"), \
             patch("src.infrastructure.health_checks.check_memory_health"), \
             patch("src.infrastructure.health_checks.check_disk_health"):
            manager.register_default_checks(db_connection=db)

        assert "database" in manager.get_registered_checks()


# =============================================================================
# Test Global Instance
# =============================================================================

class TestGlobalInstance:
    """Tests for global instance management."""

    def test_get_health_check_manager(self):
        """Test getting global instance."""
        # Reset global
        import src.infrastructure.health_checks as hc
        hc._manager_instance = None

        manager1 = get_health_check_manager()
        manager2 = get_health_check_manager()

        assert manager1 is manager2

    def test_custom_interval(self):
        """Test getting manager with custom interval."""
        import src.infrastructure.health_checks as hc
        hc._manager_instance = None

        manager = get_health_check_manager(check_interval_seconds=60.0)
        assert manager.check_interval == 60.0


# =============================================================================
# Test Thread Safety
# =============================================================================

class TestThreadSafety:
    """Tests for thread safety of HealthCheckManager."""

    def test_concurrent_check_registration(self):
        """Test concurrent check registration."""
        manager = HealthCheckManager()
        errors = []

        def register_checks(prefix):
            try:
                for i in range(10):
                    def check(name=f"{prefix}_{i}"):
                        return ComponentHealth(name=name, status=HealthStatus.HEALTHY)
                    manager.register_check(f"{prefix}_{i}", check)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=register_checks, args=(f"t{i}",))
            for i in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(manager.get_registered_checks()) == 50

    def test_concurrent_health_checks(self):
        """Test concurrent health checks."""
        manager = HealthCheckManager()
        results = []

        for i in range(5):
            manager.register_check(
                f"c{i}",
                lambda: ComponentHealth(name="test", status=HealthStatus.HEALTHY),
            )

        def run_check():
            result = manager.check_health()
            results.append(result)

        threads = [threading.Thread(target=run_check) for _ in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 10
        for r in results:
            assert r.status == HealthStatus.HEALTHY


# =============================================================================
# Test Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_history_limit(self):
        """Test history with 0 limit."""
        manager = HealthCheckManager()
        history = manager.get_health_history(limit=0)
        assert history == []

    def test_history_overflow(self):
        """Test history size overflow."""
        manager = HealthCheckManager(history_size=5)
        manager.register_check(
            "test",
            lambda: ComponentHealth(name="test", status=HealthStatus.HEALTHY),
        )

        for _ in range(10):
            manager.check_health()

        history = manager.get_health_history()
        assert len(history) == 5

    def test_check_with_slow_component(self):
        """Test check with slow component."""
        manager = HealthCheckManager()

        def slow_check():
            time.sleep(0.1)
            return ComponentHealth(name="slow", status=HealthStatus.HEALTHY)

        manager.register_check("slow", slow_check)
        result = manager.check_health()

        assert result.status == HealthStatus.HEALTHY

    def test_unhealthy_priority_over_degraded(self):
        """Test UNHEALTHY takes priority over DEGRADED."""
        manager = HealthCheckManager()

        manager.register_check(
            "degraded",
            lambda: ComponentHealth(name="degraded", status=HealthStatus.DEGRADED),
        )
        manager.register_check(
            "unhealthy",
            lambda: ComponentHealth(name="unhealthy", status=HealthStatus.UNHEALTHY),
        )

        result = manager.check_health()
        assert result.status == HealthStatus.UNHEALTHY

    def test_uptime_accuracy(self):
        """Test uptime is reasonably accurate."""
        manager = HealthCheckManager()
        time.sleep(0.2)
        uptime = manager.get_uptime()

        # Should be at least 0.2 seconds but less than 1 second
        assert 0.2 <= uptime.total_seconds() < 1.0

    def test_component_health_update_tracking(self):
        """Test component health is updated correctly."""
        manager = HealthCheckManager()

        status_list = [HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNHEALTHY]
        index = [0]

        def changing_check():
            status = status_list[index[0] % 3]
            index[0] += 1
            return ComponentHealth(name="changing", status=status)

        manager.register_check("changing", changing_check)

        # First check - HEALTHY
        manager.check_health()
        assert manager.get_component_health("changing").status == HealthStatus.HEALTHY

        # Second check - DEGRADED
        manager.check_health()
        assert manager.get_component_health("changing").status == HealthStatus.DEGRADED

        # Third check - UNHEALTHY
        manager.check_health()
        assert manager.get_component_health("changing").status == HealthStatus.UNHEALTHY
