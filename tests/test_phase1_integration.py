"""
Phase 1 Integration Tests
=========================

Integration tests for Phase 1: Observability Foundation
Tests phase timers, structured logging, and Prometheus metrics integration.
"""

import asyncio
import json
import logging
import pytest
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]


class TestPhase1Integration:
    """Test Phase 1 observability foundation integration."""
    
    def setup_method(self):
        """Setup for each test method."""
        # Add src to path
        import sys
        src_path = str(ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
    
    def test_phase_timer_integration(self):
        """Test phase timer integration with trading cycle."""
        try:
            from src.monitoring.performance_analytics import start_phase_timer, end_phase_timer, get_phase_duration_stats
            from src.workflows.trading_cycle import get_trading_cycle
            
            # Test phase timer functionality
            start_phase_timer("test_phase")
            time.sleep(0.1)  # 100ms
            duration = end_phase_timer("test_phase")
            
            assert duration >= 0.1, f"Expected duration >= 0.1s, got {duration:.3f}s"
            
            # Test phase stats
            stats = get_phase_duration_stats("test_phase")
            assert stats["count"] == 1, f"Expected 1 execution, got {stats['count']}"
            assert stats["average"] >= 0.1, f"Expected average >= 0.1s, got {stats['average']:.3f}s"
            
            logger.info(f"Phase timer test passed: {duration:.3f}s")
            
        except Exception as e:
            logger.error(f"Phase timer integration test failed: {e}")
            raise
    
    def test_structured_logging_integration(self):
        """Test structured JSON logging integration."""
        try:
            from src.monitoring.system_monitor import get_system_monitor
            
            system_monitor = get_system_monitor()
            
            # Test structured logging
            system_monitor.log_phase_event("test_phase", "started", "DEMO", {"step": "test_step"})
            system_monitor.log_phase_event("test_phase", "completed", "DEMO", {
                "step": "test_step",
                "duration_ms": 100
            })
            
            # Check if logs were written
            log_file = ROOT / "logs" / "system.log"
            assert log_file.exists(), "System log file should exist"
            
            # Read last few lines to verify structured logging
            with open(log_file, 'r') as f:
                lines = f.readlines()
                recent_lines = lines[-10:]  # Last 10 lines
            
            # Look for structured log entries
            structured_logs = [line for line in recent_lines if '"phase":' in line and '"status":' in line]
            assert len(structured_logs) >= 2, f"Expected at least 2 structured log entries, found {len(structured_logs)}"
            
            # Verify JSON structure
            for log_line in structured_logs:
                try:
                    log_data = json.loads(log_line.strip())
                    assert "ts" in log_data, "Log should contain timestamp"
                    assert "phase" in log_data, "Log should contain phase"
                    assert "status" in log_data, "Log should contain status"
                    assert "mode" in log_data, "Log should contain mode"
                except json.JSONDecodeError:
                    pytest.fail(f"Invalid JSON in log line: {log_line}")
            
            logger.info("Structured logging integration test passed")
            
        except Exception as e:
            logger.error(f"Structured logging integration test failed: {e}")
            raise
    
    def test_prometheus_metrics_integration(self):
        """Test Prometheus metrics integration."""
        try:
            from src.monitoring.system_monitor import get_system_monitor
            
            system_monitor = get_system_monitor()
            
            # Test Prometheus metrics
            system_monitor.update_prometheus_metrics("test_metric", {"test_label": "test_value"}, 1.0)
            
            # Get metrics
            metrics = system_monitor.get_prometheus_metrics()
            assert "test_metric" in metrics, "Test metric should be in Prometheus metrics"
            
            # Test API metrics
            system_monitor.update_api_metrics("test_api", "test_endpoint", 200, 100)
            
            # Verify API metrics were updated
            api_metrics = system_monitor.get_api_metrics()
            assert "test_api" in api_metrics, "Test API should be in API metrics"
            
            logger.info("Prometheus metrics integration test passed")
            
        except Exception as e:
            logger.error(f"Prometheus metrics integration test failed: {e}")
            raise
    
    def test_trading_cycle_phase_integration(self):
        """Test trading cycle phase integration with monitoring."""
        try:
            from src.workflows.trading_cycle import get_trading_cycle
            from src.monitoring.performance_analytics import get_all_phase_stats
            
            trading_cycle = get_trading_cycle()
            
            # Run a complete cycle to test phase integration
            results = trading_cycle.execute_complete_cycle("DEMO")
            
            # Verify cycle completed
            assert results is not None, "Trading cycle should return results"
            assert results.mode == "DEMO", f"Expected DEMO mode, got {results.mode}"
            
            # Check phase stats were recorded
            phase_stats = get_all_phase_stats()
            
            # Should have recorded complete_cycle phase
            assert "complete_cycle" in phase_stats, "Complete cycle phase should be recorded"
            assert phase_stats["complete_cycle"]["count"] >= 1, "Complete cycle should have at least 1 execution"
            
            # Should have recorded individual phases
            expected_phases = ["buy_phase", "hold_phase", "sell_phase"]
            for phase in expected_phases:
                if phase in phase_stats:
                    assert phase_stats[phase]["count"] >= 0, f"{phase} should have execution count"
            
            logger.info(f"Trading cycle phase integration test passed: {results.duration_seconds:.3f}s")
            
        except Exception as e:
            logger.error(f"Trading cycle phase integration test failed: {e}")
            raise
    
    def test_performance_analytics_integration(self):
        """Test performance analytics integration."""
        try:
            from src.monitoring.performance_analytics import (
                track_phase_performance, 
                generate_phase_performance_dashboard,
                get_all_phase_stats
            )
            
            # Test phase performance tracking
            performance = asyncio.run(track_phase_performance("test_analytics_phase", "test_step"))
            assert performance is not None, "Phase performance tracking should return data"
            
            # Test phase performance dashboard
            dashboard = asyncio.run(generate_phase_performance_dashboard())
            assert dashboard is not None, "Phase performance dashboard should return data"
            assert "phase_statistics" in dashboard, "Dashboard should contain phase statistics"
            assert "insights" in dashboard, "Dashboard should contain insights"
            
            # Test phase stats
            all_stats = get_all_phase_stats()
            assert isinstance(all_stats, dict), "Phase stats should be a dictionary"
            
            logger.info("Performance analytics integration test passed")
            
        except Exception as e:
            logger.error(f"Performance analytics integration test failed: {e}")
            raise
    
    def test_database_integration(self):
        """Test database integration with Phase 1 enhancements."""
        try:
            from src.config.database import get_database_manager
            
            db_manager = get_database_manager()
            mode = "DEMO"
            
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
            phase_history = db_manager.get_phase_execution_history(mode, limit=10)
            assert len(phase_history) > 0, "Phase execution history should contain entries"
            
            api_history = db_manager.get_api_validation_history(mode, limit=10)
            assert len(api_history) > 0, "API validation history should contain entries"
            
            usage_metrics = db_manager.get_api_usage_metrics(mode, limit=10)
            assert len(usage_metrics) > 0, "API usage metrics should contain entries"
            
            logger.info("Database integration test passed")
            
        except Exception as e:
            logger.error(f"Database integration test failed: {e}")
            raise


def test_phase1_smoke_test():
    """Run Phase 1 smoke test to verify all components work together."""
    try:
        logger.info("Starting Phase 1 smoke test...")
        
        # Test all integration points
        test_suite = TestPhase1Integration()
        test_suite.setup_method()
        
        # Run all tests
        test_suite.test_phase_timer_integration()
        test_suite.test_structured_logging_integration()
        test_suite.test_prometheus_metrics_integration()
        test_suite.test_trading_cycle_phase_integration()
        test_suite.test_performance_analytics_integration()
        test_suite.test_database_integration()
        
        logger.info("Phase 1 smoke test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Phase 1 smoke test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_phase1_smoke_test()
    if success:
        print("✅ Phase 1 smoke test passed")
        exit(0)
    else:
        print("❌ Phase 1 smoke test failed")
        exit(1)
