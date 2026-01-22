"""
Combined Phases Test (Phases 0-3)
=================================

Comprehensive test to verify all implemented phases (0-3) are working together properly.
Tests system validation, observability, API budgets, and data quality validation.
"""

import asyncio
import json
import logging
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]


def test_phase0_system_validation():
    """Test Phase 0: System validation and baseline."""
    try:
        # Add src to path
        src_path = str(ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        # Test database connectivity
        from config.database import get_database_manager, validate_database_integrity
        
        db_manager = get_database_manager()
        assert db_manager is not None, "Database manager should be initialized"
        
        # Test database integrity for both modes
        live_integrity = validate_database_integrity("LIVE")
        demo_integrity = validate_database_integrity("DEMO")
        
        assert live_integrity, "LIVE database should be valid"
        assert demo_integrity, "DEMO database should be valid"
        
        # Test database stats
        live_stats = db_manager.get_database_stats("LIVE")
        demo_stats = db_manager.get_database_stats("DEMO")
        
        assert isinstance(live_stats, dict), "LIVE stats should be a dictionary"
        assert isinstance(demo_stats, dict), "DEMO stats should be a dictionary"
        
        logger.info("Phase 0 system validation test: PASS")
        return True
        
    except Exception as e:
        logger.error(f"Phase 0 system validation test: FAIL - {e}")
        return False


def test_phase1_observability():
    """Test Phase 1: Observability foundation."""
    try:
        # Add src to path
        src_path = str(ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        from monitoring.system_monitor import get_system_monitor
        from monitoring.performance_analytics import get_performance_analytics
        
        # Test system monitor
        system_monitor = get_system_monitor()
        assert system_monitor is not None, "System monitor should be initialized"
        
        # Test phase timing
        execution_id = system_monitor.start_phase_timer("test_phase", "TEST")
        time.sleep(0.1)  # Small delay
        phase_data = system_monitor.end_phase_timer(execution_id)
        
        assert phase_data["duration_ms"] > 0, "Phase duration should be positive"
        
        # Test structured logging
        system_monitor.log_phase_event("test_phase", "DEMO", "started", "test_execution_1")
        
        # Test Prometheus metrics
        metrics = system_monitor.get_prometheus_metrics()
        assert "phase_duration_seconds" in metrics, "Should have phase duration metrics"
        assert "system_health_score" in metrics, "Should have system health score"
        
        # Test performance analytics
        perf_analytics = get_performance_analytics()
        assert perf_analytics is not None, "Performance analytics should be initialized"
        
        # Test phase duration tracking
        perf_analytics.start_phase_timer("test_phase")
        time.sleep(0.1)
        duration = perf_analytics.end_phase_timer("test_phase")
        
        assert duration > 0, "Performance analytics phase duration should be positive"
        
        logger.info("Phase 1 observability test: PASS")
        return True
        
    except Exception as e:
        logger.error(f"Phase 1 observability test: FAIL - {e}")
        return False


def test_phase2_api_budgets():
    """Test Phase 2: API budgets, backoff, and caching."""
    try:
        # Add src to path
        src_path = str(ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        from data_pipeline.api_budget_manager import get_api_budget_manager
        
        # Test API budget manager
        budget_manager = get_api_budget_manager()
        assert budget_manager is not None, "API budget manager should be initialized"
        
        # Test budget configuration
        assert len(budget_manager.budgets) > 0, "Should have API budgets configured"
        assert "alpha_vantage" in budget_manager.budgets, "Should have Alpha Vantage budget"
        assert "finnhub" in budget_manager.budgets, "Should have Finnhub budget"
        
        # Test budget status checking
        status = budget_manager.get_budget_status("alpha_vantage")
        assert status.value in ["healthy", "rate_limited", "error", "budget_exhausted"], "Should have valid status"
        
        # Test request recording
        budget_manager.record_request("alpha_vantage", True, 0.1)
        stats = budget_manager.get_usage_stats("alpha_vantage")
        assert stats["requests_made"] > 0, "Should record requests"
        
        # Test caching
        cache_key = budget_manager.get_cache_key("alpha_vantage", "quote", {"symbol": "AAPL"})
        assert cache_key is not None, "Should generate cache key"
        
        # Test cache operations
        test_data = {"price": 150.0, "symbol": "AAPL"}
        budget_manager.cache_response(cache_key, test_data, 60)
        cached_response = budget_manager.get_cached_response(cache_key)
        assert cached_response == test_data, "Should retrieve cached data"
        
        logger.info("Phase 2 API budgets test: PASS")
        return True
        
    except Exception as e:
        logger.error(f"Phase 2 API budgets test: FAIL - {e}")
        return False


def test_phase3_data_quality():
    """Test Phase 3: Data contracts and quality gates."""
    try:
        # Add src to path
        src_path = str(ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        from validation.data_quality import get_data_quality_validator
        from config.database import get_database_manager
        
        # Test data quality validator
        validator = get_data_quality_validator()
        assert validator is not None, "Data quality validator should be initialized"
        
        # Test clean data validation
        clean_data = pd.DataFrame({
            'open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'high': [101.0, 102.0, 103.0, 104.0, 105.0],
            'low': [99.0, 100.0, 101.0, 102.0, 103.0],
            'close': [100.5, 101.5, 102.5, 103.5, 104.5],
            'volume': [1000000, 1100000, 1200000, 1300000, 1400000],
            'rsi': [45.0, 50.0, 55.0, 60.0, 65.0],
            'adx': [25.0, 30.0, 35.0, 40.0, 45.0],
            'atr': [1.5, 1.6, 1.7, 1.8, 1.9]
        })
        
        report = validator.validate_dataframe(clean_data, "TEST")
        assert report.overall_score >= 0.8, f"Clean data should have high quality, got {report.overall_score}"
        assert len(report.violations) == 0, f"Clean data should have no violations, got {len(report.violations)}"
        
        # Test dirty data validation
        dirty_data = pd.DataFrame({
            'open': [100.0, -50.0, 102.0, np.nan, 104.0],
            'high': [101.0, 102.0, 50.0, 104.0, 105.0],
            'low': [99.0, 100.0, 101.0, 102.0, 103.0],
            'close': [100.5, 101.5, 102.5, 103.5, 104.5],
            'volume': [1000000, -500000, 1200000, 1300000, 1400000],
            'rsi': [150.0, 50.0, 55.0, 60.0, 65.0],
            'adx': [25.0, 30.0, 35.0, 40.0, 45.0],
            'atr': [1.5, 1.6, 1.7, 1.8, 1.9]
        })
        
        dirty_report = validator.validate_dataframe(dirty_data, "TEST")
        assert dirty_report.overall_score < 0.7, f"Dirty data should have low quality, got {dirty_report.overall_score}"
        assert len(dirty_report.violations) > 0, f"Dirty data should have violations, got {len(dirty_report.violations)}"
        
        # Test should skip sizing logic
        should_skip = validator.should_skip_sizing(dirty_report, threshold=0.7)
        assert should_skip, "Should skip sizing for poor quality data"
        
        # Test database provenance tracking
        db_manager = get_database_manager()
        provenance_id = db_manager.log_data_provenance(
            symbol="TEST",
            data_type="price_data",
            source="alpha_vantage",
            source_metadata={"test": "data"},
            quality_score=0.95,
            quality_level="excellent"
        )
        assert provenance_id > 0, "Should return valid provenance ID"
        
        logger.info("Phase 3 data quality test: PASS")
        return True
        
    except Exception as e:
        logger.error(f"Phase 3 data quality test: FAIL - {e}")
        return False


def test_integration_between_phases():
    """Test integration between all phases."""
    try:
        # Add src to path
        src_path = str(ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        from monitoring.system_monitor import get_system_monitor
        from data_pipeline.api_budget_manager import get_api_budget_manager
        from validation.data_quality import get_data_quality_validator
        from config.database import get_database_manager
        
        # Test that all components work together
        system_monitor = get_system_monitor()
        budget_manager = get_api_budget_manager()
        quality_validator = get_data_quality_validator()
        db_manager = get_database_manager()
        
        # Test integrated workflow
        execution_id = system_monitor.start_phase_timer("integration_test", "INTEGRATION_TEST")
        
        # Simulate API request with budget management
        if budget_manager.can_make_request("alpha_vantage"):
            budget_manager.record_request("alpha_vantage", True, 0.1)
        
        # Simulate data quality validation
        test_data = pd.DataFrame({
            'open': [100.0, 101.0, 102.0],
            'high': [101.0, 102.0, 103.0],
            'low': [99.0, 100.0, 101.0],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000000, 1100000, 1200000]
        })
        
        quality_report = quality_validator.validate_dataframe(test_data, "INTEGRATION_TEST")
        
        # Log quality event to system monitor
        system_monitor.log_phase_event("integration_test", "INTEGRATION_TEST", "quality_validated", execution_id, 
                                     duration_ms=100)
        
        # Log to database
        db_manager.log_data_provenance(
            symbol="INTEGRATION_TEST",
            data_type="test_data",
            source="integration_test",
            quality_score=quality_report.overall_score,
            quality_level=quality_report.quality_level.value
        )
        
        phase_data = system_monitor.end_phase_timer(execution_id)
        assert phase_data["duration_ms"] > 0, "Integration test should complete successfully"
        
        logger.info("Integration between phases test: PASS")
        return True
        
    except Exception as e:
        logger.error(f"Integration between phases test: FAIL - {e}")
        return False


def main():
    """Run all combined phase tests."""
    results = {}
    
    logger.info("Starting Combined Phases Test (Phases 0-3)...")
    start_time = time.time()
    
    # Run all tests
    logger.info("Testing Phase 0: System Validation...")
    results["Phase 0: System Validation"] = test_phase0_system_validation()
    
    logger.info("Testing Phase 1: Observability Foundation...")
    results["Phase 1: Observability Foundation"] = test_phase1_observability()
    
    logger.info("Testing Phase 2: API Budgets & Caching...")
    results["Phase 2: API Budgets & Caching"] = test_phase2_api_budgets()
    
    logger.info("Testing Phase 3: Data Quality & Gates...")
    results["Phase 3: Data Quality & Gates"] = test_phase3_data_quality()
    
    logger.info("Testing Integration Between Phases...")
    results["Integration Between Phases"] = test_integration_between_phases()
    
    # Calculate results
    total_tests = len(results)
    passed_tests = sum(1 for r in results.values() if r)
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    
    logger.info("==================================================")
    logger.info("COMBINED PHASES TEST SUMMARY (Phases 0-3)")
    logger.info("==================================================")
    logger.info(f"Duration: {time.time() - start_time:.1f}s")
    logger.info(f"Tests: {passed_tests}/{total_tests} passed")
    logger.info(f"Success Rate: {success_rate:.1f}%")
    for test_name, status in results.items():
        logger.info(f"{test_name}: {'PASS' if status else 'FAIL'}")
    
    # Save results
    results_file = ROOT / "logs" / "combined_phases_test_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {results_file}")
    
    if success_rate == 100:
        logger.info("✅✅✅ ALL PHASES 0-3 TESTS PASSED — SYSTEM READY FOR PHASE 4 🚀")
        return 0
    else:
        logger.error("❌ SOME TESTS FAILED — FIXING ISSUES BEFORE PROCEEDING TO PHASE 4")
        return 1


if __name__ == "__main__":
    sys.exit(main())
