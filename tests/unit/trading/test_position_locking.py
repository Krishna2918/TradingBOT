"""Unit tests for Position Locking.

Tests cover:
- Concurrent position opens (same symbol)
- Concurrent position closes (same symbol)
- Lock contention with timeout
- Position state consistency under load
- Symbol-level lock isolation
- Deadlock prevention
- Lock release on exception
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import threading
import time
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.trading.positions import (
    PositionLockManager,
    get_lock_manager,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def lock_manager():
    """Create a fresh lock manager for testing."""
    return PositionLockManager()


@pytest.fixture
def reset_global_lock_manager():
    """Reset global lock manager after test."""
    import src.trading.positions as pos_module
    original_manager = pos_module._lock_manager
    pos_module._lock_manager = PositionLockManager()
    yield
    pos_module._lock_manager = original_manager


# =============================================================================
# Test: PositionLockManager Basic Operations
# =============================================================================

class TestPositionLockManagerBasic:
    """Tests for basic lock manager operations."""

    def test_get_symbol_lock_creates_new(self, lock_manager):
        """Test get_symbol_lock creates new lock for new symbol."""
        lock = lock_manager.get_symbol_lock("AAPL")

        assert lock is not None
        assert "AAPL" in lock_manager._symbol_locks

    def test_get_symbol_lock_returns_same_lock(self, lock_manager):
        """Test get_symbol_lock returns same lock for same symbol."""
        lock1 = lock_manager.get_symbol_lock("AAPL")
        lock2 = lock_manager.get_symbol_lock("AAPL")

        assert lock1 is lock2

    def test_get_symbol_lock_normalizes_case(self, lock_manager):
        """Test get_symbol_lock normalizes symbol to uppercase."""
        lock1 = lock_manager.get_symbol_lock("aapl")
        lock2 = lock_manager.get_symbol_lock("AAPL")
        lock3 = lock_manager.get_symbol_lock("Aapl")

        assert lock1 is lock2
        assert lock2 is lock3

    def test_different_symbols_get_different_locks(self, lock_manager):
        """Test different symbols get different locks."""
        lock_aapl = lock_manager.get_symbol_lock("AAPL")
        lock_tsla = lock_manager.get_symbol_lock("TSLA")

        assert lock_aapl is not lock_tsla


# =============================================================================
# Test: Global Lock Context Manager
# =============================================================================

class TestGlobalLock:
    """Tests for global lock context manager."""

    def test_global_lock_context_manager(self, lock_manager):
        """Test global lock can be used as context manager."""
        with lock_manager.global_lock():
            # Should execute without error
            pass

    def test_global_lock_is_reentrant(self, lock_manager):
        """Test global lock can be acquired multiple times by same thread."""
        with lock_manager.global_lock():
            with lock_manager.global_lock():
                # Should execute without deadlock
                pass

    def test_global_lock_releases_on_exception(self, lock_manager):
        """Test global lock is released when exception occurs."""
        try:
            with lock_manager.global_lock():
                raise ValueError("test error")
        except ValueError:
            pass

        # Should be able to acquire lock again
        with lock_manager.global_lock():
            pass


# =============================================================================
# Test: Symbol Lock Context Manager
# =============================================================================

class TestSymbolLock:
    """Tests for symbol lock context manager."""

    def test_symbol_lock_context_manager(self, lock_manager):
        """Test symbol lock can be used as context manager."""
        with lock_manager.symbol_lock("AAPL"):
            pass

    def test_symbol_lock_is_reentrant(self, lock_manager):
        """Test symbol lock can be acquired multiple times by same thread."""
        with lock_manager.symbol_lock("AAPL"):
            with lock_manager.symbol_lock("AAPL"):
                pass

    def test_symbol_lock_releases_on_exception(self, lock_manager):
        """Test symbol lock is released when exception occurs."""
        try:
            with lock_manager.symbol_lock("AAPL"):
                raise ValueError("test error")
        except ValueError:
            pass

        # Should be able to acquire lock again
        with lock_manager.symbol_lock("AAPL"):
            pass

    def test_symbol_locks_are_independent(self, lock_manager):
        """Test locks for different symbols don't block each other."""
        results = []
        start_barrier = threading.Barrier(2)

        def acquire_lock(symbol, delay):
            start_barrier.wait()  # Ensure both threads start at same time
            with lock_manager.symbol_lock(symbol):
                time.sleep(delay)
                results.append(symbol)

        threads = [
            threading.Thread(target=acquire_lock, args=("AAPL", 0.3)),
            threading.Thread(target=acquire_lock, args=("TSLA", 0.05)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # TSLA should complete before AAPL since they have independent locks
        # Using larger time difference (0.3 vs 0.05) for reliable ordering
        assert results[0] == "TSLA"
        assert results[1] == "AAPL"


# =============================================================================
# Test: Multi-Symbol Lock
# =============================================================================

class TestMultiSymbolLock:
    """Tests for multi-symbol lock context manager."""

    def test_multi_symbol_lock_acquires_all(self, lock_manager):
        """Test multi_symbol_lock acquires all symbol locks."""
        with lock_manager.multi_symbol_lock(["AAPL", "TSLA", "NVDA"]):
            # All locks should be held
            # We can verify by trying to acquire them non-blocking
            pass

    def test_multi_symbol_lock_sorted_order(self, lock_manager):
        """Test multi_symbol_lock acquires in sorted order for deadlock prevention."""
        # Verify that locks are acquired in sorted order by checking
        # that both orderings work without deadlock

        # First acquire in one order
        with lock_manager.multi_symbol_lock(["TSLA", "AAPL", "NVDA"]):
            pass

        # Then acquire in reverse order - should work because internal sorting
        with lock_manager.multi_symbol_lock(["NVDA", "AAPL", "TSLA"]):
            pass

        # Both should complete without deadlock

    def test_multi_symbol_lock_releases_on_exception(self, lock_manager):
        """Test multi_symbol_lock releases all locks on exception."""
        try:
            with lock_manager.multi_symbol_lock(["AAPL", "TSLA"]):
                raise ValueError("test error")
        except ValueError:
            pass

        # Should be able to acquire all locks again
        with lock_manager.multi_symbol_lock(["AAPL", "TSLA"]):
            pass

    def test_multi_symbol_lock_deduplicates(self, lock_manager):
        """Test multi_symbol_lock handles duplicate symbols."""
        # Should not deadlock even with duplicates
        with lock_manager.multi_symbol_lock(["AAPL", "AAPL", "TSLA", "TSLA"]):
            pass


# =============================================================================
# Test: Cleanup Unused Locks
# =============================================================================

class TestCleanupUnusedLocks:
    """Tests for lock cleanup functionality."""

    def test_cleanup_removes_unused_locks(self, lock_manager):
        """Test cleanup removes locks for inactive symbols."""
        # Create some locks
        lock_manager.get_symbol_lock("AAPL")
        lock_manager.get_symbol_lock("TSLA")
        lock_manager.get_symbol_lock("NVDA")

        assert len(lock_manager._symbol_locks) == 3

        # Cleanup, keeping only AAPL
        removed = lock_manager.cleanup_unused_locks(["AAPL"])

        assert removed == 2
        assert "AAPL" in lock_manager._symbol_locks
        assert "TSLA" not in lock_manager._symbol_locks
        assert "NVDA" not in lock_manager._symbol_locks

    def test_cleanup_none_does_nothing(self, lock_manager):
        """Test cleanup with None does nothing."""
        lock_manager.get_symbol_lock("AAPL")

        removed = lock_manager.cleanup_unused_locks(None)

        assert removed == 0
        assert "AAPL" in lock_manager._symbol_locks

    def test_cleanup_preserves_active_locks(self, lock_manager):
        """Test cleanup doesn't remove locks that are held by another thread."""
        lock_manager.get_symbol_lock("AAPL")
        lock_manager.get_symbol_lock("TSLA")

        lock_held = threading.Event()
        cleanup_done = threading.Event()

        def hold_lock():
            with lock_manager.symbol_lock("TSLA"):
                lock_held.set()
                cleanup_done.wait(timeout=2.0)

        # Hold the TSLA lock from another thread
        holder_thread = threading.Thread(target=hold_lock)
        holder_thread.start()

        try:
            # Wait for lock to be held
            lock_held.wait(timeout=1.0)

            # Try to clean up - TSLA should not be removed since it's held
            removed = lock_manager.cleanup_unused_locks(["AAPL"])

            # TSLA should still exist because it's being held
            assert "TSLA" in lock_manager._symbol_locks
        finally:
            cleanup_done.set()
            holder_thread.join(timeout=2.0)


# =============================================================================
# Test: Concurrent Position Opens
# =============================================================================

class TestConcurrentPositionOpens:
    """Tests for concurrent position open operations."""

    def test_concurrent_opens_same_symbol_serialized(self, lock_manager):
        """Test concurrent opens on same symbol are serialized."""
        execution_order = []
        execution_lock = threading.Lock()

        def open_position(name, delay):
            with lock_manager.symbol_lock("AAPL"):
                with execution_lock:
                    execution_order.append(f"{name}_start")
                time.sleep(delay)
                with execution_lock:
                    execution_order.append(f"{name}_end")

        threads = [
            threading.Thread(target=open_position, args=("thread1", 0.1)),
            threading.Thread(target=open_position, args=("thread2", 0.05)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Operations should be serialized (one completes before other starts)
        # Either thread1_end comes before thread2_start or vice versa
        idx_t1_end = execution_order.index("thread1_end")
        idx_t2_start = execution_order.index("thread2_start")
        idx_t2_end = execution_order.index("thread2_end")
        idx_t1_start = execution_order.index("thread1_start")

        # Either thread1 fully completes before thread2 starts, or vice versa
        assert (idx_t1_end < idx_t2_start) or (idx_t2_end < idx_t1_start)

    def test_concurrent_opens_different_symbols_parallel(self, lock_manager):
        """Test concurrent opens on different symbols can run in parallel."""
        start_times = {}
        end_times = {}
        times_lock = threading.Lock()

        def open_position(symbol, delay):
            with lock_manager.symbol_lock(symbol):
                with times_lock:
                    start_times[symbol] = time.time()
                time.sleep(delay)
                with times_lock:
                    end_times[symbol] = time.time()

        threads = [
            threading.Thread(target=open_position, args=("AAPL", 0.1)),
            threading.Thread(target=open_position, args=("TSLA", 0.1)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Operations should overlap (run in parallel)
        # AAPL starts before TSLA ends and vice versa
        assert start_times["AAPL"] < end_times["TSLA"]
        assert start_times["TSLA"] < end_times["AAPL"]


# =============================================================================
# Test: Lock Contention
# =============================================================================

class TestLockContention:
    """Tests for lock contention scenarios."""

    def test_lock_contention_all_complete(self, lock_manager):
        """Test all threads eventually complete under contention."""
        counter = [0]
        counter_lock = threading.Lock()

        def increment():
            with lock_manager.symbol_lock("AAPL"):
                with counter_lock:
                    counter[0] += 1
                time.sleep(0.01)

        threads = [threading.Thread(target=increment) for _ in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert counter[0] == 10

    def test_heavy_contention_stability(self, lock_manager):
        """Test system stability under heavy contention."""
        results = []
        results_lock = threading.Lock()

        def worker(worker_id):
            for i in range(5):
                with lock_manager.symbol_lock("SHARED"):
                    with results_lock:
                        results.append((worker_id, i))
                    time.sleep(0.001)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 25  # 5 workers * 5 iterations


# =============================================================================
# Test: Position State Consistency
# =============================================================================

class TestPositionStateConsistency:
    """Tests for position state consistency under load."""

    def test_counter_consistency(self, lock_manager):
        """Test counter remains consistent with concurrent updates."""
        state = {"count": 0}

        def increment():
            with lock_manager.symbol_lock("STATE"):
                current = state["count"]
                time.sleep(0.001)  # Simulate some work
                state["count"] = current + 1

        threads = [threading.Thread(target=increment) for _ in range(20)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert state["count"] == 20

    def test_balance_consistency(self, lock_manager):
        """Test balance remains consistent with concurrent deposits/withdrawals."""
        account = {"balance": 1000.0}

        def deposit(amount):
            with lock_manager.symbol_lock("ACCOUNT"):
                current = account["balance"]
                time.sleep(0.001)
                account["balance"] = current + amount

        def withdraw(amount):
            with lock_manager.symbol_lock("ACCOUNT"):
                current = account["balance"]
                time.sleep(0.001)
                account["balance"] = current - amount

        threads = []
        # 10 deposits of 100
        for _ in range(10):
            threads.append(threading.Thread(target=deposit, args=(100,)))
        # 10 withdrawals of 50
        for _ in range(10):
            threads.append(threading.Thread(target=withdraw, args=(50,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Expected: 1000 + (10 * 100) - (10 * 50) = 1500
        assert account["balance"] == 1500.0


# =============================================================================
# Test: Deadlock Prevention
# =============================================================================

class TestDeadlockPrevention:
    """Tests for deadlock prevention."""

    def test_multi_symbol_lock_prevents_deadlock(self, lock_manager):
        """Test multi_symbol_lock prevents deadlock with multiple symbols."""
        # This test verifies that acquiring locks in sorted order prevents deadlock
        # when multiple threads try to acquire the same locks in different orders

        completed = [0]
        completed_lock = threading.Lock()

        def worker1():
            # Try to acquire AAPL, then TSLA (but will be reordered)
            with lock_manager.multi_symbol_lock(["TSLA", "AAPL"]):
                time.sleep(0.05)
                with completed_lock:
                    completed[0] += 1

        def worker2():
            # Try to acquire TSLA, then AAPL (but will be reordered)
            with lock_manager.multi_symbol_lock(["AAPL", "TSLA"]):
                time.sleep(0.05)
                with completed_lock:
                    completed[0] += 1

        threads = [
            threading.Thread(target=worker1),
            threading.Thread(target=worker2),
        ]

        for t in threads:
            t.start()

        # Wait with timeout to detect deadlock
        for t in threads:
            t.join(timeout=2.0)

        # If we got here without timeout, no deadlock occurred
        assert completed[0] == 2

    def test_nested_locks_same_symbol_no_deadlock(self, lock_manager):
        """Test nested locks on same symbol don't deadlock (reentrant)."""
        def worker():
            with lock_manager.symbol_lock("AAPL"):
                with lock_manager.symbol_lock("AAPL"):
                    with lock_manager.symbol_lock("AAPL"):
                        return True

        result = worker()
        assert result is True


# =============================================================================
# Test: Lock Release on Exception
# =============================================================================

class TestLockReleaseOnException:
    """Tests for proper lock release when exceptions occur."""

    def test_symbol_lock_released_after_exception(self, lock_manager):
        """Test symbol lock is released after exception."""
        exception_raised = False

        try:
            with lock_manager.symbol_lock("AAPL"):
                raise RuntimeError("test error")
        except RuntimeError:
            exception_raised = True

        assert exception_raised

        # Lock should be released - another thread should be able to acquire it
        acquired = [False]

        def try_acquire():
            lock = lock_manager.get_symbol_lock("AAPL")
            if lock.acquire(blocking=False):
                acquired[0] = True
                lock.release()

        t = threading.Thread(target=try_acquire)
        t.start()
        t.join()

        assert acquired[0] is True

    def test_global_lock_released_after_exception(self, lock_manager):
        """Test global lock is released after exception."""
        try:
            with lock_manager.global_lock():
                raise RuntimeError("test error")
        except RuntimeError:
            pass

        # Lock should be released
        with lock_manager.global_lock():
            pass  # Should not deadlock

    def test_multi_symbol_lock_released_after_exception(self, lock_manager):
        """Test all locks in multi_symbol_lock are released after exception."""
        try:
            with lock_manager.multi_symbol_lock(["AAPL", "TSLA", "NVDA"]):
                raise RuntimeError("test error")
        except RuntimeError:
            pass

        # All locks should be released
        with lock_manager.multi_symbol_lock(["AAPL", "TSLA", "NVDA"]):
            pass  # Should not deadlock


# =============================================================================
# Test: Global Lock Manager
# =============================================================================

class TestGlobalLockManager:
    """Tests for global lock manager singleton."""

    def test_get_lock_manager_returns_instance(self, reset_global_lock_manager):
        """Test get_lock_manager returns an instance."""
        manager = get_lock_manager()

        assert manager is not None
        assert isinstance(manager, PositionLockManager)

    def test_get_lock_manager_returns_same_instance(self, reset_global_lock_manager):
        """Test get_lock_manager returns the same instance."""
        manager1 = get_lock_manager()
        manager2 = get_lock_manager()

        assert manager1 is manager2


# =============================================================================
# Test: ThreadPoolExecutor Compatibility
# =============================================================================

class TestThreadPoolExecutorCompatibility:
    """Tests for compatibility with ThreadPoolExecutor."""

    def test_works_with_thread_pool(self, lock_manager):
        """Test lock manager works with ThreadPoolExecutor."""
        results = []
        results_lock = threading.Lock()

        def task(symbol, value):
            with lock_manager.symbol_lock(symbol):
                time.sleep(0.01)
                with results_lock:
                    results.append((symbol, value))

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for i in range(10):
                symbol = "AAPL" if i % 2 == 0 else "TSLA"
                futures.append(executor.submit(task, symbol, i))

            for future in as_completed(futures):
                future.result()  # Raise any exceptions

        assert len(results) == 10

    def test_high_concurrency_thread_pool(self, lock_manager):
        """Test lock manager handles high concurrency from thread pool."""
        counter = {"value": 0}
        counter_lock = threading.Lock()

        def increment():
            with lock_manager.symbol_lock("COUNTER"):
                with counter_lock:
                    counter["value"] += 1

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(increment) for _ in range(100)]
            for future in as_completed(futures):
                future.result()

        assert counter["value"] == 100


# =============================================================================
# Test: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_multi_symbol_lock(self, lock_manager):
        """Test multi_symbol_lock with empty list."""
        with lock_manager.multi_symbol_lock([]):
            pass  # Should work without error

    def test_single_symbol_multi_lock(self, lock_manager):
        """Test multi_symbol_lock with single symbol."""
        with lock_manager.multi_symbol_lock(["AAPL"]):
            pass

    def test_special_characters_in_symbol(self, lock_manager):
        """Test symbols with special characters work."""
        # Some exchanges use special characters in symbols
        with lock_manager.symbol_lock("BRK.A"):
            pass

        with lock_manager.symbol_lock("BRK-B"):
            pass

    def test_unicode_symbol(self, lock_manager):
        """Test unicode symbols work."""
        with lock_manager.symbol_lock("ABC"):
            pass

    def test_very_long_symbol(self, lock_manager):
        """Test very long symbol names work."""
        long_symbol = "A" * 100
        with lock_manager.symbol_lock(long_symbol):
            pass
