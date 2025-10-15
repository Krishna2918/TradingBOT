"""
Phase 2 Integration Tests
=========================

Integration tests for Phase 2: API Budgets, Backoff, and Caching
Tests API budget management, retry logic, caching, and rate limiting.
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


class TestPhase2Integration:
    """Test Phase 2 API budget management and caching integration."""
    
    def setup_method(self):
        """Setup for each test method."""
        # Add src to path
        import sys
        src_path = str(ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
    
    def test_api_budget_manager_initialization(self):
        """Test API Budget Manager initialization and configuration."""
        try:
            from src.data_pipeline.api_budget_manager import get_api_budget_manager
            
            budget_manager = get_api_budget_manager()
            
            # Test that budgets are initialized
            assert len(budget_manager.budgets) > 0, "Should have initialized API budgets"
            
            # Test specific API budgets
            expected_apis = ["alpha_vantage", "finnhub", "newsapi", "questrade", "yfinance"]
            for api_name in expected_apis:
                assert api_name in budget_manager.budgets, f"Should have budget for {api_name}"
                assert api_name in budget_manager.usage, f"Should have usage tracking for {api_name}"
            
            # Test budget configuration
            av_budget = budget_manager.budgets["alpha_vantage"]
            assert av_budget.qps_limit > 0, "Alpha Vantage should have QPS limit"
            assert av_budget.daily_limit > 0, "Alpha Vantage should have daily limit"
            assert av_budget.timeout_seconds > 0, "Alpha Vantage should have timeout"
            
            logger.info("API Budget Manager initialization test passed")
            
        except Exception as e:
            logger.error(f"API Budget Manager initialization test failed: {e}")
            raise
    
    def test_api_budget_status_checking(self):
        """Test API budget status checking and rate limiting."""
        try:
            from src.data_pipeline.api_budget_manager import get_api_budget_manager
            
            budget_manager = get_api_budget_manager()
            
            # Test initial status (should be healthy)
            status = budget_manager.get_budget_status("alpha_vantage")
            assert status.value == "healthy", f"Initial status should be healthy, got {status.value}"
            
            # Test can_make_request
            can_request = budget_manager.can_make_request("alpha_vantage")
            assert can_request, "Should be able to make request initially"
            
            # Test backoff time (should be 0 initially)
            backoff_time = budget_manager.get_backoff_time("alpha_vantage")
            assert backoff_time == 0.0, f"Initial backoff should be 0, got {backoff_time}"
            
            logger.info("API budget status checking test passed")
            
        except Exception as e:
            logger.error(f"API budget status checking test failed: {e}")
            raise
    
    def test_api_request_recording(self):
        """Test API request recording and statistics."""
        try:
            from src.data_pipeline.api_budget_manager import get_api_budget_manager
            
            budget_manager = get_api_budget_manager()
            
            # Record some test requests
            budget_manager.record_request("alpha_vantage", True, 0.1)
            budget_manager.record_request("alpha_vantage", True, 0.2)
            budget_manager.record_request("alpha_vantage", False, 0.0)
            
            # Get usage stats
            stats = budget_manager.get_usage_stats("alpha_vantage")
            
            assert stats["requests_made"] == 3, f"Expected 3 requests, got {stats['requests_made']}"
            assert stats["requests_successful"] == 2, f"Expected 2 successful, got {stats['requests_successful']}"
            assert stats["requests_failed"] == 1, f"Expected 1 failed, got {stats['requests_failed']}"
            assert stats["daily_requests"] == 3, f"Expected 3 daily requests, got {stats['daily_requests']}"
            
            # Test success rate calculation
            expected_success_rate = 2 / 3
            assert abs(stats["success_rate"] - expected_success_rate) < 0.01, f"Success rate mismatch"
            
            logger.info("API request recording test passed")
            
        except Exception as e:
            logger.error(f"API request recording test failed: {e}")
            raise
    
    def test_rate_limit_handling(self):
        """Test rate limit handling and backoff calculation."""
        try:
            from src.data_pipeline.api_budget_manager import get_api_budget_manager
            
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
            assert stats["consecutive_failures"] == 2, f"Expected 2 consecutive failures, got {stats['consecutive_failures']}"
            
            logger.info("Rate limit handling test passed")
            
        except Exception as e:
            logger.error(f"Rate limit handling test failed: {e}")
            raise
    
    def test_caching_functionality(self):
        """Test API response caching functionality."""
        try:
            from src.data_pipeline.api_budget_manager import get_api_budget_manager
            
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
            
            logger.info("Caching functionality test passed")
            
        except Exception as e:
            logger.error(f"Caching functionality test failed: {e}")
            raise
    
    def test_questrade_budget_integration(self):
        """Test Questrade client integration with budget management."""
        try:
            from src.data_pipeline.questrade_client import QuestradeClient
            
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
            
            logger.info("Questrade budget integration test passed")
            
        except Exception as e:
            logger.error(f"Questrade budget integration test failed: {e}")
            raise
    
    def test_canadian_market_collector_integration(self):
        """Test Canadian market collector integration with budget management."""
        try:
            from src.data_pipeline.collectors.canadian_market_collector import CanadianMarketCollector
            
            # Create collector (this might fail if config file doesn't exist, which is OK)
            try:
                collector = CanadianMarketCollector("config/data_sources.yaml")
                
                # Test that budget manager is initialized
                assert hasattr(collector, 'budget_manager'), "Collector should have budget manager"
                assert collector.budget_manager is not None, "Budget manager should be initialized"
                
                logger.info("Canadian market collector budget integration test passed")
                
            except FileNotFoundError:
                logger.info("Config file not found, skipping Canadian market collector test")
            
        except Exception as e:
            logger.error(f"Canadian market collector integration test failed: {e}")
            raise
    
    def test_system_monitor_budget_integration(self):
        """Test system monitor integration with API budget manager."""
        try:
            from src.monitoring.system_monitor import get_system_monitor
            
            system_monitor = get_system_monitor()
            
            # Test that API budget manager is initialized
            assert hasattr(system_monitor, 'api_budget_manager'), "System monitor should have API budget manager"
            assert system_monitor.api_budget_manager is not None, "API budget manager should be initialized"
            
            # Test API metrics update
            system_monitor.update_api_metrics("test_api", "test_endpoint", True, 100)
            
            # Test system health score update
            system_monitor._update_system_health_score()
            
            # Verify health score is set
            health_score = system_monitor.prometheus_metrics["system_health_score"]
            assert 0.0 <= health_score <= 1.0, f"Health score should be between 0 and 1, got {health_score}"
            
            logger.info("System monitor budget integration test passed")
            
        except Exception as e:
            logger.error(f"System monitor budget integration test failed: {e}")
            raise
    
    def test_make_api_request_with_backoff(self):
        """Test making API requests with backoff and retry logic."""
        try:
            from src.data_pipeline.api_budget_manager import make_api_request
            
            # Mock request function
            async def mock_request():
                await asyncio.sleep(0.1)  # Simulate API call
                return {"success": True, "data": "test"}
            
            # Test successful request
            success, response, total_time = asyncio.run(
                make_api_request("alpha_vantage", mock_request)
            )
            
            assert success, "Request should succeed"
            assert response is not None, "Response should not be None"
            assert total_time >= 0.1, f"Total time should be >= 0.1s, got {total_time}"
            
            logger.info("API request with backoff test passed")
            
        except Exception as e:
            logger.error(f"API request with backoff test failed: {e}")
            raise
    
    def test_make_cached_api_request(self):
        """Test making cached API requests."""
        try:
            from src.data_pipeline.api_budget_manager import make_cached_api_request
            
            # Mock request function
            async def mock_request():
                await asyncio.sleep(0.1)  # Simulate API call
                return {"success": True, "data": "test"}
            
            # Test cached request
            success, response, total_time, cache_hit = asyncio.run(
                make_cached_api_request(
                    "alpha_vantage", 
                    "quote", 
                    mock_request,
                    params={"symbol": "AAPL"},
                    cache_ttl=60
                )
            )
            
            assert success, "Request should succeed"
            assert response is not None, "Response should not be None"
            assert not cache_hit, "First request should not be cache hit"
            
            # Test second request (should be cache hit)
            success2, response2, total_time2, cache_hit2 = asyncio.run(
                make_cached_api_request(
                    "alpha_vantage", 
                    "quote", 
                    mock_request,
                    params={"symbol": "AAPL"},
                    cache_ttl=60
                )
            )
            
            assert success2, "Second request should succeed"
            assert cache_hit2, "Second request should be cache hit"
            assert total_time2 < total_time, "Cached request should be faster"
            
            logger.info("Cached API request test passed")
            
        except Exception as e:
            logger.error(f"Cached API request test failed: {e}")
            raise


def test_phase2_smoke_test():
    """Run Phase 2 smoke test to verify all components work together."""
    try:
        logger.info("Starting Phase 2 smoke test...")
        
        # Test all integration points
        test_suite = TestPhase2Integration()
        test_suite.setup_method()
        
        # Run all tests
        test_suite.test_api_budget_manager_initialization()
        test_suite.test_api_budget_status_checking()
        test_suite.test_api_request_recording()
        test_suite.test_rate_limit_handling()
        test_suite.test_caching_functionality()
        test_suite.test_questrade_budget_integration()
        test_suite.test_canadian_market_collector_integration()
        test_suite.test_system_monitor_budget_integration()
        test_suite.test_make_api_request_with_backoff()
        test_suite.test_make_cached_api_request()
        
        logger.info("Phase 2 smoke test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Phase 2 smoke test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_phase2_smoke_test()
    if success:
        print("✅ Phase 2 smoke test passed")
        exit(0)
    else:
        print("❌ Phase 2 smoke test failed")
        exit(1)
