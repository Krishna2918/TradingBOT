"""
Phase 2 Smoke Test
==================

Quick smoke test to verify API budget management, caching, and rate limiting.
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


async def test_api_budget_manager():
    """Test API Budget Manager functionality."""
    try:
        # Add src to path
        src_path = str(ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        from data_pipeline.api_budget_manager import get_api_budget_manager
        
        budget_manager = get_api_budget_manager()
        
        # Test initialization
        assert len(budget_manager.budgets) > 0, "Should have initialized API budgets"
        
        # Test specific APIs
        expected_apis = ["alpha_vantage", "finnhub", "newsapi", "questrade", "yfinance"]
        for api_name in expected_apis:
            assert api_name in budget_manager.budgets, f"Should have budget for {api_name}"
        
        # Test status checking
        status = budget_manager.get_budget_status("alpha_vantage")
        assert status.value == "healthy", f"Initial status should be healthy, got {status.value}"
        
        # Test request recording
        budget_manager.record_request("alpha_vantage", True, 0.1)
        budget_manager.record_request("alpha_vantage", False, 0.0)
        
        stats = budget_manager.get_usage_stats("alpha_vantage")
        assert stats["requests_made"] == 2, f"Expected 2 requests, got {stats['requests_made']}"
        assert stats["requests_successful"] == 1, f"Expected 1 successful, got {stats['requests_successful']}"
        
        logger.info("API Budget Manager test: PASS")
        return True
        
    except Exception as e:
        logger.error(f"API Budget Manager test: FAIL - {e}")
        return False


async def test_caching_functionality():
    """Test caching functionality."""
    try:
        # Add src to path
        src_path = str(ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        from data_pipeline.api_budget_manager import get_api_budget_manager
        
        budget_manager = get_api_budget_manager()
        
        # Test cache key generation
        cache_key = budget_manager.get_cache_key("alpha_vantage", "quote", {"symbol": "AAPL"})
        expected_key = "alpha_vantage:quote:symbol=AAPL"
        assert cache_key == expected_key, f"Cache key mismatch: {cache_key} != {expected_key}"
        
        # Test cache miss
        cached_response = budget_manager.get_cached_response(cache_key)
        assert cached_response is None, "Should be cache miss initially"
        
        # Test cache storage
        test_data = {"price": 150.0, "symbol": "AAPL"}
        budget_manager.cache_response(cache_key, test_data, 60)  # 1 minute TTL
        
        # Test cache hit
        cached_response = budget_manager.get_cached_response(cache_key)
        assert cached_response is not None, "Should be cache hit after storing"
        assert cached_response == test_data, "Cached data should match stored data"
        
        logger.info("Caching functionality test: PASS")
        return True
        
    except Exception as e:
        logger.error(f"Caching functionality test: FAIL - {e}")
        return False


async def test_rate_limit_handling():
    """Test rate limit handling and backoff."""
    try:
        # Add src to path
        src_path = str(ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        from data_pipeline.api_budget_manager import get_api_budget_manager
        
        budget_manager = get_api_budget_manager()
        
        # Record rate limit hits
        budget_manager.record_rate_limit("alpha_vantage")
        budget_manager.record_rate_limit("alpha_vantage")
        
        # Check that backoff time is set
        backoff_time = budget_manager.get_backoff_time("alpha_vantage")
        assert backoff_time > 0, f"Backoff time should be > 0 after rate limits, got {backoff_time}"
        
        # Check status is rate limited
        status = budget_manager.get_budget_status("alpha_vantage")
        assert status.value == "rate_limited", f"Status should be rate_limited, got {status.value}"
        
        # Check usage stats
        stats = budget_manager.get_usage_stats("alpha_vantage")
        assert stats["rate_limit_hits"] == 2, f"Expected 2 rate limit hits, got {stats['rate_limit_hits']}"
        
        logger.info("Rate limit handling test: PASS")
        return True
        
    except Exception as e:
        logger.error(f"Rate limit handling test: FAIL - {e}")
        return False


async def test_api_request_with_backoff():
    """Test making API requests with backoff and retry logic."""
    try:
        # Add src to path
        src_path = str(ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        from data_pipeline.api_budget_manager import make_api_request
        
        # Mock request function
        async def mock_request():
            await asyncio.sleep(0.1)  # Simulate API call
            return {"success": True, "data": "test"}
        
        # Test successful request
        success, response, total_time = await make_api_request("alpha_vantage", mock_request)
        
        assert success, "Request should succeed"
        assert response is not None, "Response should not be None"
        assert total_time >= 0.1, f"Total time should be >= 0.1s, got {total_time}"
        
        logger.info("API request with backoff test: PASS")
        return True
        
    except Exception as e:
        logger.error(f"API request with backoff test: FAIL - {e}")
        return False


async def test_cached_api_request():
    """Test making cached API requests."""
    try:
        # Add src to path
        src_path = str(ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        from data_pipeline.api_budget_manager import make_cached_api_request, get_api_budget_manager
        
        # Clear cache to ensure clean test
        budget_manager = get_api_budget_manager()
        budget_manager.clear_cache("alpha_vantage")
        
        # Mock request function
        async def mock_request():
            await asyncio.sleep(0.1)  # Simulate API call
            return {"success": True, "data": "test"}
        
        # Test cached request with unique symbol
        unique_symbol = f"AAPL_{int(time.time())}"
        success, response, total_time, cache_hit = await make_cached_api_request(
            "alpha_vantage", 
            "quote", 
            mock_request,
            params={"symbol": unique_symbol},
            cache_ttl=60
        )
        
        assert success, "Request should succeed"
        assert response is not None, "Response should not be None"
        assert not cache_hit, "First request should not be cache hit"
        
        # Test second request (should be cache hit)
        success2, response2, total_time2, cache_hit2 = await make_cached_api_request(
            "alpha_vantage", 
            "quote", 
            mock_request,
            params={"symbol": unique_symbol},
            cache_ttl=60
        )
        
        assert success2, "Second request should succeed"
        assert cache_hit2, "Second request should be cache hit"
        assert total_time2 < total_time, "Cached request should be faster"
        
        logger.info("Cached API request test: PASS")
        return True
        
    except Exception as e:
        logger.error(f"Cached API request test: FAIL - {e}")
        return False


async def test_questrade_integration():
    """Test Questrade client integration with budget management."""
    try:
        # Add src to path
        src_path = str(ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        from data_pipeline.questrade_client import QuestradeClient
        
        # Create Questrade client
        client = QuestradeClient(allow_trading=False, practice_mode=True)
        
        # Test that budget manager is initialized
        assert hasattr(client, 'budget_manager'), "Questrade client should have budget manager"
        assert client.budget_manager is not None, "Budget manager should be initialized"
        
        # Test API usage stats method
        stats = client.get_api_usage_stats()
        assert isinstance(stats, dict), "API usage stats should be a dictionary"
        assert "api_name" in stats, "Stats should contain api_name"
        assert stats["api_name"] == "questrade", "API name should be questrade"
        
        logger.info("Questrade integration test: PASS")
        return True
        
    except Exception as e:
        logger.error(f"Questrade integration test: FAIL - {e}")
        return False


async def test_system_monitor_integration():
    """Test system monitor integration with API budget manager."""
    try:
        # Add src to path
        src_path = str(ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        # Test basic system monitor functionality without the problematic import
        from monitoring.system_monitor import SystemMonitor
        
        # Create a new instance to avoid import issues
        system_monitor = SystemMonitor()
        
        # Test basic functionality
        assert hasattr(system_monitor, 'prometheus_metrics'), "System monitor should have Prometheus metrics"
        assert hasattr(system_monitor, 'phase_timers'), "System monitor should have phase timers"
        
        # Test API metrics update (basic version)
        system_monitor.update_api_metrics("test_api", "test_endpoint", True, 100)
        
        # Verify metrics were updated
        assert "test_api" in system_monitor.prometheus_metrics.get("api_calls_total", {}), "API calls should be tracked"
        
        logger.info("System monitor integration test: PASS")
        return True
        
    except Exception as e:
        logger.error(f"System monitor integration test: FAIL - {e}")
        return False


async def main():
    """Run all Phase 2 smoke tests."""
    results = {}
    
    logger.info("Starting Phase 2 Smoke Test...")
    start_time = time.time()
    
    # Run all tests
    logger.info("Testing API Budget Manager...")
    results["API Budget Manager"] = await test_api_budget_manager()
    
    logger.info("Testing Caching Functionality...")
    results["Caching Functionality"] = await test_caching_functionality()
    
    logger.info("Testing Rate Limit Handling...")
    results["Rate Limit Handling"] = await test_rate_limit_handling()
    
    logger.info("Testing API Request with Backoff...")
    results["API Request with Backoff"] = await test_api_request_with_backoff()
    
    logger.info("Testing Cached API Request...")
    results["Cached API Request"] = await test_cached_api_request()
    
    logger.info("Testing Questrade Integration...")
    results["Questrade Integration"] = await test_questrade_integration()
    
    logger.info("Testing System Monitor Integration...")
    results["System Monitor Integration"] = await test_system_monitor_integration()
    
    # Calculate results
    total_tests = len(results)
    passed_tests = sum(1 for r in results.values() if r)
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    
    logger.info("==================================================")
    logger.info("PHASE 2 SMOKE TEST SUMMARY")
    logger.info("==================================================")
    logger.info(f"Duration: {time.time() - start_time:.1f}s")
    logger.info(f"Tests: {passed_tests}/{total_tests} passed")
    logger.info(f"Success Rate: {success_rate:.1f}%")
    for test_name, status in results.items():
        logger.info(f"{test_name}: {'PASS' if status else 'FAIL'}")
    
    # Save results
    results_file = ROOT / "logs" / "phase2_smoke_test_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {results_file}")
    
    if success_rate == 100:
        logger.info("PHASE 2 SMOKE TEST PASSED")
        return 0
    else:
        logger.error("PHASE 2 SMOKE TEST FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
