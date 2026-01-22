"""
Enhanced Rate Limiter - Intelligent API Quota Management

Manages API request limits across multiple sources with intelligent scheduling,
quota prediction, and automatic optimization for ML data collection.
"""

import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque
import threading
import queue
import json
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class APIQuota:
    """API quota configuration and tracking"""
    name: str
    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    burst_limit: int = 0  # Maximum burst requests
    reset_time: Optional[datetime] = None
    
    # Tracking
    minute_requests: deque = field(default_factory=deque)
    hour_requests: deque = field(default_factory=deque)
    day_requests: deque = field(default_factory=deque)
    
    # Performance metrics
    avg_response_time: float = 0.0
    success_rate: float = 1.0
    last_request_time: float = 0.0

@dataclass
class RateLimitRequest:
    """Request for rate limiting"""
    api_name: str
    request_type: str
    priority: int = 5  # 1 = highest priority
    symbol: str = ""
    timeout: float = 30.0
    created_at: float = field(default_factory=time.time)

class RateLimiter:
    """Enhanced rate limiter with intelligent scheduling"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.quotas = self._load_quotas(config_path)
        self.request_queue = queue.PriorityQueue()
        self.active_requests = {}
        self.performance_history = {}
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Statistics
        self.total_requests = 0
        self.denied_requests = 0
        self.queue_wait_times = []
        
        logger.info(f"🚀 Rate Limiter initialized with {len(self.quotas)} API quotas")
    
    def _load_quotas(self, config_path: Optional[str]) -> Dict[str, APIQuota]:
        """Load API quota configurations"""
        
        # Default quotas for known APIs
        default_quotas = {
            "yahoo_finance": APIQuota(
                name="yahoo_finance",
                requests_per_minute=60,
                requests_per_hour=2000,
                requests_per_day=20000,
                burst_limit=10
            ),
            "alpha_vantage": APIQuota(
                name="alpha_vantage",
                requests_per_minute=5,
                requests_per_hour=500,
                requests_per_day=500,
                burst_limit=2
            ),
            "news_api": APIQuota(
                name="news_api",
                requests_per_minute=10,
                requests_per_hour=1000,
                requests_per_day=1000,
                burst_limit=5
            ),
            "finnhub": APIQuota(
                name="finnhub",
                requests_per_minute=60,
                requests_per_hour=3600,
                requests_per_day=86400,
                burst_limit=10
            ),
            "reddit": APIQuota(
                name="reddit",
                requests_per_minute=100,
                requests_per_hour=6000,
                requests_per_day=144000,
                burst_limit=20
            )
        }
        
        # Load custom config if provided
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                for api_name, quota_config in config.get("quotas", {}).items():
                    if api_name in default_quotas:
                        # Update existing quota
                        quota = default_quotas[api_name]
                        quota.requests_per_minute = quota_config.get("requests_per_minute", quota.requests_per_minute)
                        quota.requests_per_hour = quota_config.get("requests_per_hour", quota.requests_per_hour)
                        quota.requests_per_day = quota_config.get("requests_per_day", quota.requests_per_day)
                        quota.burst_limit = quota_config.get("burst_limit", quota.burst_limit)
                    else:
                        # Create new quota
                        default_quotas[api_name] = APIQuota(
                            name=api_name,
                            requests_per_minute=quota_config.get("requests_per_minute", 60),
                            requests_per_hour=quota_config.get("requests_per_hour", 3600),
                            requests_per_day=quota_config.get("requests_per_day", 86400),
                            burst_limit=quota_config.get("burst_limit", 10)
                        )
                
                logger.info(f"📄 Loaded rate limit config from {config_path}")
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to load rate limit config: {e}")
        
        return default_quotas
    
    def acquire_permit(self, api_name: str, request_type: str = "default", 
                      priority: int = 5, symbol: str = "", timeout: float = 30.0) -> bool:
        """Acquire a permit to make an API request"""
        
        with self.lock:
            self.total_requests += 1
            
            # Check if API is configured
            if api_name not in self.quotas:
                logger.warning(f"⚠️ Unknown API: {api_name}")
                return False
            
            quota = self.quotas[api_name]
            
            # Clean old requests
            self._clean_old_requests(quota)
            
            # Check immediate availability
            if self._can_make_request_now(quota):
                self._record_request(quota)
                logger.debug(f"✅ Immediate permit granted for {api_name}")
                return True
            
            # Check if we should queue or deny
            if self._should_queue_request(quota, priority):
                # Add to queue
                request = RateLimitRequest(
                    api_name=api_name,
                    request_type=request_type,
                    priority=priority,
                    symbol=symbol,
                    timeout=timeout
                )
                
                self.request_queue.put((priority, time.time(), request))
                logger.debug(f"📋 Queued request for {api_name} (priority: {priority})")
                
                # Wait for permit
                return self._wait_for_permit(request, timeout)
            else:
                # Deny request
                self.denied_requests += 1
                logger.debug(f"❌ Request denied for {api_name} (quota exceeded)")
                return False
    
    def _can_make_request_now(self, quota: APIQuota) -> bool:
        """Check if a request can be made immediately"""
        
        now = time.time()
        
        # Check minute limit
        minute_count = len([t for t in quota.minute_requests if now - t < 60])
        if minute_count >= quota.requests_per_minute:
            return False
        
        # Check hour limit
        hour_count = len([t for t in quota.hour_requests if now - t < 3600])
        if hour_count >= quota.requests_per_hour:
            return False
        
        # Check day limit
        day_count = len([t for t in quota.day_requests if now - t < 86400])
        if day_count >= quota.requests_per_day:
            return False
        
        # Check minimum delay between requests
        min_delay = 60.0 / quota.requests_per_minute if quota.requests_per_minute > 0 else 1.0
        if now - quota.last_request_time < min_delay:
            return False
        
        return True
    
    def _should_queue_request(self, quota: APIQuota, priority: int) -> bool:
        """Determine if request should be queued or denied"""
        
        # Always queue high priority requests (1-2)
        if priority <= 2:
            return True
        
        # Check queue size
        queue_size = self.request_queue.qsize()
        max_queue_size = quota.requests_per_minute * 2  # Allow 2 minutes worth of requests
        
        if queue_size >= max_queue_size:
            return False
        
        # Check if we'll have capacity soon
        return self._estimate_wait_time(quota) < 300  # Max 5 minute wait
    
    def _wait_for_permit(self, request: RateLimitRequest, timeout: float) -> bool:
        """Wait for a permit to become available"""
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self.lock:
                quota = self.quotas[request.api_name]
                self._clean_old_requests(quota)
                
                if self._can_make_request_now(quota):
                    self._record_request(quota)
                    
                    # Record wait time
                    wait_time = time.time() - request.created_at
                    self.queue_wait_times.append(wait_time)
                    
                    logger.debug(f"✅ Permit granted for {request.api_name} after {wait_time:.1f}s wait")
                    return True
            
            # Wait a bit before checking again
            time.sleep(0.1)
        
        # Timeout
        logger.debug(f"⏰ Permit timeout for {request.api_name}")
        return False
    
    def _record_request(self, quota: APIQuota):
        """Record a request timestamp"""
        
        now = time.time()
        
        quota.minute_requests.append(now)
        quota.hour_requests.append(now)
        quota.day_requests.append(now)
        quota.last_request_time = now
    
    def _clean_old_requests(self, quota: APIQuota):
        """Remove old request timestamps"""
        
        now = time.time()
        
        # Clean minute requests (keep last 60 seconds)
        while quota.minute_requests and now - quota.minute_requests[0] >= 60:
            quota.minute_requests.popleft()
        
        # Clean hour requests (keep last 3600 seconds)
        while quota.hour_requests and now - quota.hour_requests[0] >= 3600:
            quota.hour_requests.popleft()
        
        # Clean day requests (keep last 86400 seconds)
        while quota.day_requests and now - quota.day_requests[0] >= 86400:
            quota.day_requests.popleft()
    
    def get_remaining_quota(self, api_name: str) -> Dict[str, int]:
        """Get remaining quota for an API"""
        
        if api_name not in self.quotas:
            return {"error": "Unknown API"}
        
        with self.lock:
            quota = self.quotas[api_name]
            self._clean_old_requests(quota)
            
            now = time.time()
            
            minute_used = len([t for t in quota.minute_requests if now - t < 60])
            hour_used = len([t for t in quota.hour_requests if now - t < 3600])
            day_used = len([t for t in quota.day_requests if now - t < 86400])
            
            return {
                "minute_remaining": max(0, quota.requests_per_minute - minute_used),
                "hour_remaining": max(0, quota.requests_per_hour - hour_used),
                "day_remaining": max(0, quota.requests_per_day - day_used),
                "minute_used": minute_used,
                "hour_used": hour_used,
                "day_used": day_used
            }
    
    def estimate_completion_time(self, total_requests: int, api_name: str) -> timedelta:
        """Estimate time to complete a number of requests"""
        
        if api_name not in self.quotas:
            return timedelta(hours=24)  # Unknown API, assume long time
        
        quota = self.quotas[api_name]
        
        # Calculate effective rate (considering current usage)
        remaining = self.get_remaining_quota(api_name)
        
        # Use the most restrictive limit
        minute_rate = remaining["minute_remaining"] / 1.0  # requests per minute
        hour_rate = remaining["hour_remaining"] / 60.0    # requests per minute
        day_rate = remaining["day_remaining"] / 1440.0    # requests per minute
        
        effective_rate = min(minute_rate, hour_rate, day_rate)
        
        if effective_rate <= 0:
            # Need to wait for quota reset
            return timedelta(minutes=1)  # Minimum wait
        
        # Calculate completion time
        minutes_needed = total_requests / effective_rate
        return timedelta(minutes=minutes_needed)
    
    def _estimate_wait_time(self, quota: APIQuota) -> float:
        """Estimate wait time for next available slot"""
        
        now = time.time()
        
        # Find when the oldest request in the minute window will expire
        if quota.minute_requests:
            oldest_minute = quota.minute_requests[0]
            minute_wait = max(0, 60 - (now - oldest_minute))
        else:
            minute_wait = 0
        
        # Find when the oldest request in the hour window will expire
        if quota.hour_requests:
            oldest_hour = quota.hour_requests[0]
            hour_wait = max(0, 3600 - (now - oldest_hour))
        else:
            hour_wait = 0
        
        # Return the minimum wait time
        return min(minute_wait, hour_wait)
    
    def optimize_request_schedule(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize request schedule to minimize wait times"""
        
        # Sort by priority first, then by API efficiency
        def sort_key(request):
            api_name = request.get("api_name", "unknown")
            priority = request.get("priority", 5)
            
            # Get API efficiency (requests per minute)
            if api_name in self.quotas:
                efficiency = self.quotas[api_name].requests_per_minute
            else:
                efficiency = 1
            
            # Higher priority (lower number) and higher efficiency first
            return (priority, -efficiency)
        
        optimized = sorted(requests, key=sort_key)
        
        logger.info(f"📊 Optimized schedule for {len(requests)} requests")
        return optimized
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        
        stats = {
            "total_requests": self.total_requests,
            "denied_requests": self.denied_requests,
            "denial_rate": self.denied_requests / max(1, self.total_requests),
            "queue_size": self.request_queue.qsize(),
            "avg_wait_time": sum(self.queue_wait_times) / max(1, len(self.queue_wait_times)),
            "quotas": {}
        }
        
        # Add per-API stats
        for api_name, quota in self.quotas.items():
            remaining = self.get_remaining_quota(api_name)
            stats["quotas"][api_name] = {
                "remaining": remaining,
                "avg_response_time": quota.avg_response_time,
                "success_rate": quota.success_rate,
                "last_request": quota.last_request_time
            }
        
        return stats
    
    def update_performance_metrics(self, api_name: str, response_time: float, success: bool):
        """Update performance metrics for an API"""
        
        if api_name not in self.quotas:
            return
        
        quota = self.quotas[api_name]
        
        # Update average response time (exponential moving average)
        alpha = 0.1  # Smoothing factor
        quota.avg_response_time = (alpha * response_time + 
                                 (1 - alpha) * quota.avg_response_time)
        
        # Update success rate (exponential moving average)
        success_value = 1.0 if success else 0.0
        quota.success_rate = (alpha * success_value + 
                            (1 - alpha) * quota.success_rate)
    
    def reset_quota(self, api_name: str):
        """Reset quota for an API (for testing or manual reset)"""
        
        if api_name in self.quotas:
            quota = self.quotas[api_name]
            quota.minute_requests.clear()
            quota.hour_requests.clear()
            quota.day_requests.clear()
            quota.last_request_time = 0.0
            
            logger.info(f"🔄 Reset quota for {api_name}")

# Convenience functions
def create_rate_limiter(config_path: Optional[str] = None) -> RateLimiter:
    """Create and return a rate limiter instance"""
    return RateLimiter(config_path)

# For testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test the rate limiter
    limiter = RateLimiter()
    
    # Test quota checking
    print("📊 Testing Rate Limiter...")
    
    for api_name in ["yahoo_finance", "alpha_vantage"]:
        remaining = limiter.get_remaining_quota(api_name)
        print(f"   {api_name}: {remaining}")
        
        # Test permit acquisition
        permit = limiter.acquire_permit(api_name, priority=1)
        print(f"   Permit for {api_name}: {'✅' if permit else '❌'}")
    
    # Test performance stats
    stats = limiter.get_performance_stats()
    print(f"\n📈 Performance Stats:")
    print(f"   Total requests: {stats['total_requests']}")
    print(f"   Denial rate: {stats['denial_rate']:.1%}")
    print(f"   Queue size: {stats['queue_size']}")
    
    # Test completion time estimation
    completion_time = limiter.estimate_completion_time(100, "yahoo_finance")
    print(f"\n⏱️ Estimated time for 100 requests: {completion_time}")