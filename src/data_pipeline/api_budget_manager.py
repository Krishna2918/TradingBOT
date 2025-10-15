"""
API Budget Manager
==================

Manages API budgets, rate limiting, exponential backoff, and caching for external APIs.
Provides centralized budget tracking, retry logic, and performance monitoring.
"""

import asyncio
import logging
import os
import random
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class APIStatus(Enum):
    """API status enumeration."""
    HEALTHY = "healthy"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"
    BUDGET_EXHAUSTED = "budget_exhausted"


@dataclass
class APIBudget:
    """API budget configuration."""
    name: str
    qps_limit: int  # Queries per second
    daily_limit: int  # Daily request limit
    timeout_seconds: int  # Hard timeout per request
    backoff_base: float = 2.0  # Exponential backoff base
    backoff_max: float = 60.0  # Maximum backoff time
    jitter: bool = True  # Add jitter to backoff


@dataclass
class APIUsage:
    """Current API usage tracking."""
    requests_made: int = 0
    requests_successful: int = 0
    requests_failed: int = 0
    rate_limit_hits: int = 0
    last_request_time: Optional[datetime] = None
    daily_requests: int = 0
    daily_reset_time: datetime = None
    current_backoff: float = 0.0
    consecutive_failures: int = 0


class APIBudgetManager:
    """Manages API budgets, rate limiting, and retry logic."""
    
    def __init__(self):
        """Initialize API Budget Manager."""
        self.budgets: Dict[str, APIBudget] = {}
        self.usage: Dict[str, APIUsage] = {}
        self.cache: Dict[str, Dict[str, Any]] = {}
        
        # Initialize default budgets from environment variables
        self._initialize_default_budgets()
        
        # Cache configuration
        self.cache_ttl = {
            "finnhub_metrics": 6 * 3600,  # 6 hours
            "news_sentiment": 4 * 3600,   # 4 hours
            "alpha_vantage": 12 * 3600,   # 12 hours
            "questrade_quotes": 300,      # 5 minutes
            "questrade_positions": 60,    # 1 minute
        }
        
        logger.info("API Budget Manager initialized")
    
    def _initialize_default_budgets(self):
        """Initialize default API budgets from environment variables."""
        # Alpha Vantage
        av_qps = int(os.getenv("AV_QPS", "4"))  # 25/day limit, 4 QPS max
        self.add_budget("alpha_vantage", APIBudget(
            name="alpha_vantage",
            qps_limit=av_qps,
            daily_limit=25,
            timeout_seconds=30,
            backoff_base=2.0,
            backoff_max=60.0
        ))
        
        # Finnhub
        fh_qps = int(os.getenv("FH_QPS", "1"))  # 60/min limit, 1 QPS max
        self.add_budget("finnhub", APIBudget(
            name="finnhub",
            qps_limit=fh_qps,
            daily_limit=1440,  # 60/min * 24h
            timeout_seconds=15,
            backoff_base=1.5,
            backoff_max=30.0
        ))
        
        # NewsAPI
        self.add_budget("newsapi", APIBudget(
            name="newsapi",
            qps_limit=1,  # 100/day limit
            daily_limit=100,
            timeout_seconds=20,
            backoff_base=2.0,
            backoff_max=60.0
        ))
        
        # Questrade
        self.add_budget("questrade", APIBudget(
            name="questrade",
            qps_limit=10,  # Higher limit for trading API
            daily_limit=10000,
            timeout_seconds=10,
            backoff_base=1.5,
            backoff_max=30.0
        ))
        
        # Yahoo Finance (no official limits, but be respectful)
        self.add_budget("yfinance", APIBudget(
            name="yfinance",
            qps_limit=2,  # Conservative limit
            daily_limit=2000,  # Conservative daily limit
            timeout_seconds=15,
            backoff_base=2.0,
            backoff_max=60.0
        ))
        
        logger.info(f"Initialized {len(self.budgets)} API budgets")
    
    def add_budget(self, api_name: str, budget: APIBudget):
        """Add or update an API budget."""
        self.budgets[api_name] = budget
        if api_name not in self.usage:
            self.usage[api_name] = APIUsage()
        logger.info(f"Added budget for {api_name}: {budget.qps_limit} QPS, {budget.daily_limit} daily")
    
    def get_budget_status(self, api_name: str) -> APIStatus:
        """Get current status of an API budget."""
        if api_name not in self.budgets:
            return APIStatus.ERROR
        
        budget = self.budgets[api_name]
        usage = self.usage[api_name]
        
        # Check daily limit
        if usage.daily_requests >= budget.daily_limit:
            return APIStatus.BUDGET_EXHAUSTED
        
        # Check rate limiting
        if usage.current_backoff > 0:
            return APIStatus.RATE_LIMITED
        
        # Check consecutive failures
        if usage.consecutive_failures >= 5:
            return APIStatus.ERROR
        
        return APIStatus.HEALTHY
    
    def can_make_request(self, api_name: str) -> bool:
        """Check if we can make a request to the API."""
        status = self.get_budget_status(api_name)
        return status == APIStatus.HEALTHY
    
    def get_backoff_time(self, api_name: str) -> float:
        """Get backoff time for an API (alias for calculate_backoff)."""
        return self.calculate_backoff(api_name)
    
    def calculate_backoff(self, api_name: str, attempt: int = None) -> float:
        """Get current backoff time for an API."""
        # If attempt number is provided, calculate backoff based on attempt
        if attempt is not None and attempt > 0:
            budget = self.budgets.get(api_name)
            if budget:
                backoff = min(budget.backoff_base ** attempt, budget.backoff_max)
                if budget.jitter:
                    backoff *= (0.5 + random.random() * 0.5)  # Add jitter
                return backoff
            else:
                # Default backoff for unknown APIs
                backoff = min(2.0 ** attempt, 30.0)  # 2^attempt, max 30 seconds
                return backoff
        
        if api_name not in self.usage:
            return 0.0
        
        usage = self.usage[api_name]
        
        if usage.current_backoff <= 0:
            return 0.0
        
        # Check if backoff period has expired
        if usage.last_request_time:
            time_since_last = (datetime.now() - usage.last_request_time).total_seconds()
            if time_since_last >= usage.current_backoff:
                usage.current_backoff = 0.0
                return 0.0
        
        return usage.current_backoff
    
    def wait_for_backoff(self, api_name: str) -> float:
        """Wait for backoff period to expire and return actual wait time."""
        backoff_time = self.get_backoff_time(api_name)
        if backoff_time > 0:
            logger.info(f"Waiting {backoff_time:.1f}s for {api_name} backoff")
            time.sleep(backoff_time)
            self.usage[api_name].current_backoff = 0.0
        return backoff_time
    
    def record_request(self, api_name: str, success: bool, response_time: float = 0.0):
        """Record a request attempt."""
        if api_name not in self.usage:
            self.usage[api_name] = APIUsage()
        
        usage = self.usage[api_name]
        usage.requests_made += 1
        usage.last_request_time = datetime.now()
        
        # Reset daily counter if needed
        if usage.daily_reset_time is None or datetime.now() >= usage.daily_reset_time:
            usage.daily_requests = 0
            usage.daily_reset_time = datetime.now() + timedelta(days=1)
        
        usage.daily_requests += 1
        
        if success:
            usage.requests_successful += 1
            usage.consecutive_failures = 0
        else:
            usage.requests_failed += 1
            usage.consecutive_failures += 1
    
    def record_rate_limit(self, api_name: str):
        """Record a rate limit hit."""
        if api_name not in self.usage:
            self.usage[api_name] = APIUsage()
        
        usage = self.usage[api_name]
        usage.rate_limit_hits += 1
        
        # Calculate exponential backoff
        budget = self.budgets[api_name]
        backoff_time = min(
            budget.backoff_base ** usage.consecutive_failures,
            budget.backoff_max
        )
        
        # Add jitter if enabled
        if budget.jitter:
            jitter = random.uniform(0.1, 0.3) * backoff_time
            backoff_time += jitter
        
        usage.current_backoff = backoff_time
        usage.consecutive_failures += 1
        
        logger.warning(f"Rate limit hit for {api_name}, backoff: {backoff_time:.1f}s")
    
    async def make_request_with_backoff(
        self, 
        api_name: str, 
        request_func: Callable,
        *args, 
        **kwargs
    ) -> Tuple[bool, Any, float]:
        """
        Make a request with automatic backoff and retry logic.
        
        Returns:
            Tuple of (success, response_data, total_time)
        """
        if api_name not in self.budgets:
            logger.error(f"No budget configured for {api_name}")
            return False, None, 0.0
        
        budget = self.budgets[api_name]
        start_time = time.time()
        
        # Wait for any existing backoff
        wait_time = self.wait_for_backoff(api_name)
        
        # Check if we can make the request
        if not self.can_make_request(api_name):
            status = self.get_budget_status(api_name)
            logger.warning(f"Cannot make request to {api_name}: {status.value}")
            return False, None, time.time() - start_time
        
        # Make the request with timeout
        try:
            if asyncio.iscoroutinefunction(request_func):
                response = await asyncio.wait_for(
                    request_func(*args, **kwargs),
                    timeout=budget.timeout_seconds
                )
            else:
                # For sync functions, run in thread pool
                loop = asyncio.get_event_loop()
                response = await asyncio.wait_for(
                    loop.run_in_executor(None, request_func, *args, **kwargs),
                    timeout=budget.timeout_seconds
                )
            
            # Record successful request
            response_time = time.time() - start_time
            self.record_request(api_name, True, response_time)
            
            logger.debug(f"Request to {api_name} successful in {response_time:.3f}s")
            return True, response, response_time
            
        except asyncio.TimeoutError:
            logger.error(f"Request to {api_name} timed out after {budget.timeout_seconds}s")
            self.record_request(api_name, False)
            return False, None, time.time() - start_time
            
        except Exception as e:
            logger.error(f"Request to {api_name} failed: {e}")
            self.record_request(api_name, False)
            
            # Check if it's a rate limit error
            if "429" in str(e) or "rate limit" in str(e).lower():
                self.record_rate_limit(api_name)
            
            return False, None, time.time() - start_time
    
    def get_cache_key(self, api_name: str, endpoint: str, params: Dict[str, Any]) -> str:
        """Generate cache key for API request."""
        # Sort params for consistent keys
        sorted_params = sorted(params.items()) if params else []
        param_str = "&".join(f"{k}={v}" for k, v in sorted_params)
        return f"{api_name}:{endpoint}:{param_str}"
    
    def get_cached_response(self, cache_key: str) -> Optional[Any]:
        """Get cached response if available and not expired."""
        if cache_key not in self.cache:
            return None
        
        cached_data = self.cache[cache_key]
        if datetime.now() >= cached_data["expires_at"]:
            del self.cache[cache_key]
            return None
        
        logger.debug(f"Cache hit for {cache_key}")
        return cached_data["data"]
    
    def cache_response(self, cache_key: str, data: Any, ttl_seconds: int):
        """Cache API response with TTL."""
        expires_at = datetime.now() + timedelta(seconds=ttl_seconds)
        self.cache[cache_key] = {
            "data": data,
            "expires_at": expires_at,
            "cached_at": datetime.now()
        }
        logger.debug(f"Cached response for {cache_key}, expires in {ttl_seconds}s")
    
    async def make_cached_request(
        self,
        api_name: str,
        endpoint: str,
        request_func: Callable,
        params: Dict[str, Any] = None,
        cache_ttl: Optional[int] = None,
        *args,
        **kwargs
    ) -> Tuple[bool, Any, float, bool]:
        """
        Make a cached request with automatic backoff.
        
        Returns:
            Tuple of (success, response_data, total_time, cache_hit)
        """
        params = params or {}
        cache_key = self.get_cache_key(api_name, endpoint, params)
        
        # Check cache first
        cached_response = self.get_cached_response(cache_key)
        if cached_response is not None:
            return True, cached_response, 0.0, True
        
        # Make request with backoff
        success, response, total_time = await self.make_request_with_backoff(
            api_name, request_func, *args, **kwargs
        )
        
        # Cache successful response
        if success and response is not None:
            ttl = cache_ttl or self.cache_ttl.get(api_name, 3600)  # Default 1 hour
            self.cache_response(cache_key, response, ttl)
        
        return success, response, total_time, False
    
    def get_usage_stats(self, api_name: str) -> Dict[str, Any]:
        """Get usage statistics for an API."""
        if api_name not in self.usage:
            return {}
        
        usage = self.usage[api_name]
        budget = self.budgets.get(api_name)
        
        stats = {
            "api_name": api_name,
            "requests_made": usage.requests_made,
            "requests_successful": usage.requests_successful,
            "requests_failed": usage.requests_failed,
            "rate_limit_hits": usage.rate_limit_hits,
            "daily_requests": usage.daily_requests,
            "consecutive_failures": usage.consecutive_failures,
            "current_backoff": usage.current_backoff,
            "status": self.get_budget_status(api_name).value
        }
        
        if budget:
            stats.update({
                "qps_limit": budget.qps_limit,
                "daily_limit": budget.daily_limit,
                "daily_remaining": budget.daily_limit - usage.daily_requests,
                "success_rate": usage.requests_successful / max(usage.requests_made, 1)
            })
        
        return stats
    
    def get_all_usage_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get usage statistics for all APIs."""
        return {api_name: self.get_usage_stats(api_name) for api_name in self.usage}
    
    def reset_daily_limits(self):
        """Reset daily request limits for all APIs."""
        for usage in self.usage.values():
            usage.daily_requests = 0
            usage.daily_reset_time = datetime.now() + timedelta(days=1)
        logger.info("Reset daily limits for all APIs")
    
    def clear_cache(self, api_name: Optional[str] = None):
        """Clear cache for specific API or all APIs."""
        if api_name:
            keys_to_remove = [key for key in self.cache.keys() if key.startswith(f"{api_name}:")]
            for key in keys_to_remove:
                del self.cache[key]
            logger.info(f"Cleared cache for {api_name}")
        else:
            self.cache.clear()
            logger.info("Cleared all API caches")


# Global API Budget Manager instance
_api_budget_manager: Optional[APIBudgetManager] = None


def get_api_budget_manager() -> APIBudgetManager:
    """Get global API Budget Manager instance."""
    global _api_budget_manager
    if _api_budget_manager is None:
        _api_budget_manager = APIBudgetManager()
    return _api_budget_manager


# Convenience functions
async def make_api_request(api_name: str, request_func: Callable, *args, **kwargs) -> Tuple[bool, Any, float]:
    """Make an API request with budget management."""
    manager = get_api_budget_manager()
    return await manager.make_request_with_backoff(api_name, request_func, *args, **kwargs)


async def make_cached_api_request(
    api_name: str, 
    endpoint: str, 
    request_func: Callable, 
    params: Dict[str, Any] = None,
    cache_ttl: Optional[int] = None,
    *args, 
    **kwargs
) -> Tuple[bool, Any, float, bool]:
    """Make a cached API request with budget management."""
    manager = get_api_budget_manager()
    return await manager.make_cached_request(
        api_name, endpoint, request_func, params, cache_ttl, *args, **kwargs
    )


def get_api_usage_stats(api_name: str = None) -> Dict[str, Any]:
    """Get API usage statistics."""
    manager = get_api_budget_manager()
    if api_name:
        return manager.get_usage_stats(api_name)
    return manager.get_all_usage_stats()


def can_make_api_request(api_name: str) -> bool:
    """Check if we can make a request to the API."""
    manager = get_api_budget_manager()
    return manager.can_make_request(api_name)
