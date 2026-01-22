"""
Phase 1 Metrics Smoke Test
==========================

Quick smoke test to verify Prometheus metrics and phase duration tracking.
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]


async def test_prometheus_metrics():
    """Test Prometheus metrics functionality."""
    try:
        # Add src to path
        src_path = str(ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        from monitoring.system_monitor import get_system_monitor
        
        system_monitor = get_system_monitor()
        
        # Test basic metrics with proper phase data
        phase_data = {
            "phase_name": "test_phase",
            "duration_ms": 100,
            "status": "completed"
        }
        system_monitor.update_prometheus_metrics(phase_data)
        
        # Get metrics
        metrics = system_monitor.get_prometheus_metrics()
        
        # Verify metrics exist
        assert "test_phase" in metrics, "Test phase should be in metrics"
        
        # Test API metrics
        system_monitor.update_api_metrics("test_api", "test_endpoint", True, 100)
        
        # Verify API metrics were updated (no getter method available)
        logger.info("API metrics updated successfully")
        
        logger.info("Prometheus metrics test: PASS")
        return True
        
    except Exception as e:
        logger.error(f"Prometheus metrics test: FAIL - {e}")
        return False


async def test_phase_duration_tracking():
    """Test phase duration tracking functionality."""
    try:
        # Add src to path
        src_path = str(ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        from monitoring.performance_analytics import (
            start_phase_timer, 
            end_phase_timer, 
            get_phase_duration_stats,
            get_all_phase_stats
        )
        
        # Test phase timing
        start_phase_timer("test_phase")
        await asyncio.sleep(0.1)  # 100ms
        duration = end_phase_timer("test_phase")
        
        assert duration >= 0.1, f"Expected duration >= 0.1s, got {duration:.3f}s"
        
        # Test phase stats
        stats = get_phase_duration_stats("test_phase")
        assert stats["count"] == 1, f"Expected 1 execution, got {stats['count']}"
        assert stats["average"] >= 0.1, f"Expected average >= 0.1s, got {stats['average']:.3f}s"
        
        # Test multiple executions
        for i in range(3):
            start_phase_timer("test_phase")
            await asyncio.sleep(0.05)  # 50ms
            end_phase_timer("test_phase")
        
        # Verify stats updated
        stats = get_phase_duration_stats("test_phase")
        assert stats["count"] == 4, f"Expected 4 executions, got {stats['count']}"
        
        # Test all phase stats
        all_stats = get_all_phase_stats()
        assert "test_phase" in all_stats, "Test phase should be in all stats"
        
        logger.info(f"Phase duration tracking test: PASS - {stats['count']} executions, avg: {stats['average']:.3f}s")
        return True
        
    except Exception as e:
        logger.error(f"Phase duration tracking test: FAIL - {e}")
        return False


async def test_structured_logging():
    """Test structured JSON logging functionality."""
    try:
        # Add src to path
        src_path = str(ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        from monitoring.system_monitor import get_system_monitor
        
        system_monitor = get_system_monitor()
        
        # Test structured logging
        system_monitor.log_phase_event("test_phase", "DEMO", "started", "test_execution_1")
        system_monitor.log_phase_event("test_phase", "DEMO", "completed", "test_execution_1", 100)
        
        # Check if logs were written
        log_file = ROOT / "logs" / "system.log"
        assert log_file.exists(), "System log file should exist"
        
        # Read recent lines to verify structured logging
        with open(log_file, 'r') as f:
            lines = f.readlines()
            recent_lines = lines[-10:]  # Last 10 lines
        
        # Look for structured log entries
        structured_logs = [line for line in recent_lines if "Phase test_phase" in line]
        assert len(structured_logs) >= 2, f"Expected at least 2 structured log entries, found {len(structured_logs)}"
        
        # Verify log entries contain expected information
        for log_line in structured_logs:
            assert "test_phase" in log_line, "Log should contain test_phase"
            assert "DEMO" in log_line, "Log should contain DEMO mode"
        
        logger.info(f"Structured logging test: PASS - {len(structured_logs)} structured log entries")
        return True
        
    except Exception as e:
        logger.error(f"Structured logging test: FAIL - {e}")
        return False


async def test_database_metrics():
    """Test database metrics functionality."""
    try:
        # Add src to path
        src_path = str(ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        from config.database import get_database_manager
        
        db_manager = get_database_manager()
        mode = "DEMO"
        
        # Test basic database connectivity
        with db_manager.get_connection(mode) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            assert result is not None, "Database query should return result"
        
        # Test phase execution tracking
        execution_id = f"test_execution_{int(time.time())}"
        db_manager.log_phase_execution(
            "test_phase", execution_id, datetime.now(), 
            datetime.now(), 1000, "completed", True, mode
        )
        
        # Test API validation logging
        db_manager.log_api_validation(
            "test_api", "connectivity", True, 50, 200, mode
        )
        
        # Test API usage metrics
        db_manager.log_api_usage_metrics(
            "test_api", "test_endpoint", 1, 1, 0, 50, 0, 
            datetime.now(), datetime.now(), datetime.now(), mode
        )
        
        # Verify data was stored
        phase_history = db_manager.get_phase_execution_history(limit=5, mode=mode)
        assert len(phase_history) > 0, "Phase execution history should contain entries"
        
        api_history = db_manager.get_api_validation_history(limit=5, mode=mode)
        assert len(api_history) > 0, "API validation history should contain entries"
        
        usage_metrics = db_manager.get_api_usage_metrics(mode=mode)
        assert len(usage_metrics) > 0, "API usage metrics should contain entries"
        
        logger.info(f"Database metrics test: PASS - {len(phase_history)} phase entries, {len(api_history)} API entries")
        return True
        
    except Exception as e:
        logger.error(f"Database metrics test: FAIL - {e}")
        return False


async def main():
    """Run all metrics smoke tests."""
    results = {}
    
    logger.info("Starting Phase 1 Metrics Smoke Test...")
    start_time = time.time()
    
    # Run all tests
    logger.info("Testing Prometheus Metrics...")
    results["Prometheus Metrics"] = await test_prometheus_metrics()
    
    logger.info("Testing Phase Duration Tracking...")
    results["Phase Duration Tracking"] = await test_phase_duration_tracking()
    
    logger.info("Testing Structured Logging...")
    results["Structured Logging"] = await test_structured_logging()
    
    logger.info("Testing Database Metrics...")
    results["Database Metrics"] = await test_database_metrics()
    
    # Calculate results
    total_tests = len(results)
    passed_tests = sum(1 for r in results.values() if r)
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    
    logger.info("==================================================")
    logger.info("PHASE 1 METRICS SMOKE TEST SUMMARY")
    logger.info("==================================================")
    logger.info(f"Duration: {time.time() - start_time:.1f}s")
    logger.info(f"Tests: {passed_tests}/{total_tests} passed")
    logger.info(f"Success Rate: {success_rate:.1f}%")
    for test_name, status in results.items():
        logger.info(f"{test_name}: {'PASS' if status else 'FAIL'}")
    
    # Save results
    results_file = ROOT / "logs" / "phase1_metrics_smoke_test_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {results_file}")
    
    if success_rate == 100:
        logger.info("PHASE 1 METRICS SMOKE TEST PASSED")
        return 0
    else:
        logger.error("PHASE 1 METRICS SMOKE TEST FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
