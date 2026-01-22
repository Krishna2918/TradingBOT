"""
API Orchestrator - Multi-Source Data Collection Coordinator

Coordinates data collection across multiple API sources with intelligent fallback,
rate limiting, and quality-based source selection for ML data extraction.
"""

import logging
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import os
import yaml
from pathlib import Path

from .enhanced_collectors import MultiSourceDataCollector, EnhancedYahooFinanceCollector
from .alpha_vantage_collector import AlphaVantageCollector
from .symbol_manager import SymbolManager
from .progress_tracker import ProgressTracker

logger = logging.getLogger(__name__)

@dataclass
class APISourceConfig:
    """Configuration for an API data source"""
    name: str
    enabled: bool
    priority: int  # 1 = highest priority
    rate_limit_per_minute: int
    timeout_seconds: int
    retry_attempts: int
    quality_weight: float  # Weight for quality-based selection

@dataclass
class CollectionRequest:
    """Data collection request"""
    symbol: str
    timeframe: str
    period: str
    priority: int = 5  # 1 = highest priority
    required_quality: float = 0.8
    max_age_hours: int = 24

@dataclass
class CollectionResult:
    """Result of data collection"""
    symbol: str
    timeframe: str
    success: bool
    source: str
    data_rows: int
    quality_score: float
    collection_time: float
    error_message: Optional[str] = None

class APIOrchestrator:
    """Coordinates data collection across multiple API sources"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.symbol_manager = SymbolManager()
        self.progress_tracker = ProgressTracker()
        
        # Track API health and performance (initialize before collectors)
        self.api_health = {}
        self.api_performance = {}
        self.last_health_check = {}
        
        # Initialize collectors
        self.collectors = self._initialize_collectors()
        
        # Rate limiting
        self.request_timestamps = {source: [] for source in self.collectors.keys()}
        
        logger.info(f"🚀 API Orchestrator initialized with {len(self.collectors)} sources")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        
        default_config = {
            "api_sources": {
                "yahoo_finance": {
                    "enabled": True,
                    "priority": 1,
                    "rate_limit_per_minute": 60,
                    "timeout_seconds": 30,
                    "retry_attempts": 3,
                    "quality_weight": 1.0
                },
                "alpha_vantage": {
                    "enabled": True,
                    "priority": 2,
                    "rate_limit_per_minute": 5,
                    "timeout_seconds": 30,
                    "retry_attempts": 2,
                    "quality_weight": 0.9
                }
            },
            "collection": {
                "max_concurrent_requests": 5,
                "health_check_interval_minutes": 15,
                "quality_threshold": 0.8,
                "fallback_enabled": True
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                # Merge with defaults
                default_config.update(user_config)
                logger.info(f"📄 Loaded configuration from {config_path}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load config from {config_path}: {e}")
        
        return default_config
    
    def _initialize_collectors(self) -> Dict[str, Any]:
        """Initialize available data collectors"""
        
        collectors = {}
        
        # Yahoo Finance collector
        if self.config["api_sources"]["yahoo_finance"]["enabled"]:
            try:
                collectors["yahoo_finance"] = EnhancedYahooFinanceCollector()
                self.api_health["yahoo_finance"] = True
                logger.info("✅ Yahoo Finance collector initialized")
            except Exception as e:
                logger.warning(f"⚠️ Yahoo Finance collector failed: {e}")
                self.api_health["yahoo_finance"] = False
        
        # Alpha Vantage collector
        if self.config["api_sources"]["alpha_vantage"]["enabled"]:
            try:
                collectors["alpha_vantage"] = AlphaVantageCollector()
                self.api_health["alpha_vantage"] = True
                logger.info("✅ Alpha Vantage collector initialized")
            except Exception as e:
                logger.warning(f"⚠️ Alpha Vantage collector failed: {e}")
                self.api_health["alpha_vantage"] = False
        
        return collectors
    
    def detect_available_apis(self) -> Dict[str, bool]:
        """Detect which APIs are currently available"""
        
        logger.info("🔍 Detecting available APIs...")
        
        availability = {}
        
        for source_name, collector in self.collectors.items():
            try:
                if source_name == "yahoo_finance":
                    # Test with a simple connectivity check
                    available = collector.test_connectivity()
                elif source_name == "alpha_vantage":
                    # Test Alpha Vantage connectivity
                    available = collector.test_connectivity()
                else:
                    available = False
                
                availability[source_name] = available
                self.api_health[source_name] = available
                
                status = "✅ Available" if available else "❌ Unavailable"
                logger.info(f"   {source_name}: {status}")
                
            except Exception as e:
                logger.error(f"❌ Error testing {source_name}: {e}")
                availability[source_name] = False
                self.api_health[source_name] = False
        
        return availability
    
    def _get_available_sources(self, request: CollectionRequest) -> List[str]:
        """Get available sources sorted by priority and health"""
        
        sources = []
        
        for source_name in self.collectors.keys():
            config = self.config["api_sources"].get(source_name, {})
            
            # Check if source is enabled and healthy
            if (config.get("enabled", False) and 
                self.api_health.get(source_name, False) and
                self._check_rate_limit(source_name)):
                
                sources.append(source_name)
        
        # Sort by priority (lower number = higher priority)
        sources.sort(key=lambda x: self.config["api_sources"][x]["priority"])
        
        return sources
    
    def _check_rate_limit(self, source_name: str) -> bool:
        """Check if source is within rate limits"""
        
        config = self.config["api_sources"].get(source_name, {})
        rate_limit = config.get("rate_limit_per_minute", 60)
        
        now = time.time()
        timestamps = self.request_timestamps[source_name]
        
        # Remove timestamps older than 1 minute
        timestamps[:] = [ts for ts in timestamps if now - ts < 60]
        
        # Check if we're under the rate limit
        return len(timestamps) < rate_limit
    
    def _record_request(self, source_name: str):
        """Record a request timestamp for rate limiting"""
        self.request_timestamps[source_name].append(time.time())
    
    def collect_market_data(self, symbols: List[str], timeframe: str = "1d", 
                          period: str = "max") -> Dict[str, CollectionResult]:
        """Collect market data for multiple symbols with intelligent source selection"""
        
        logger.info(f"📊 Collecting market data for {len(symbols)} symbols ({timeframe})")
        
        # Create collection requests
        requests = [
            CollectionRequest(
                symbol=symbol,
                timeframe=timeframe,
                period=period,
                priority=self._get_symbol_priority(symbol)
            )
            for symbol in symbols
        ]
        
        # Sort by priority
        requests.sort(key=lambda x: x.priority)
        
        results = {}
        
        # Use ThreadPoolExecutor for concurrent collection
        max_workers = self.config["collection"]["max_concurrent_requests"]
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit collection tasks
            future_to_request = {
                executor.submit(self._collect_single_symbol, request): request
                for request in requests
            }
            
            # Process completed tasks
            for future in as_completed(future_to_request):
                request = future_to_request[future]
                
                try:
                    result = future.result()
                    results[request.symbol] = result
                    
                    # Log result
                    if result.success:
                        logger.info(f"✅ {result.symbol}: {result.data_rows} rows from {result.source}")
                    else:
                        logger.warning(f"❌ {result.symbol}: {result.error_message}")
                        
                except Exception as e:
                    logger.error(f"❌ Error collecting {request.symbol}: {e}")
                    results[request.symbol] = CollectionResult(
                        symbol=request.symbol,
                        timeframe=request.timeframe,
                        success=False,
                        source="none",
                        data_rows=0,
                        quality_score=0.0,
                        collection_time=0.0,
                        error_message=str(e)
                    )
        
        # Log summary
        successful = sum(1 for r in results.values() if r.success)
        logger.info(f"📊 Collection complete: {successful}/{len(symbols)} successful")
        
        return results
    
    def _collect_single_symbol(self, request: CollectionRequest) -> CollectionResult:
        """Collect data for a single symbol with fallback logic"""
        
        start_time = time.time()
        available_sources = self._get_available_sources(request)
        
        if not available_sources:
            return CollectionResult(
                symbol=request.symbol,
                timeframe=request.timeframe,
                success=False,
                source="none",
                data_rows=0,
                quality_score=0.0,
                collection_time=time.time() - start_time,
                error_message="No available data sources"
            )
        
        # Try each source in priority order
        for source_name in available_sources:
            try:
                # Check rate limit before making request
                if not self._check_rate_limit(source_name):
                    logger.debug(f"⏳ Rate limit reached for {source_name}, skipping")
                    continue
                
                # Record request
                self._record_request(source_name)
                
                # Collect data
                collector = self.collectors[source_name]
                
                if source_name == "yahoo_finance":
                    data = collector.fetch_data(request.symbol, request.period, request.timeframe)
                elif source_name == "alpha_vantage":
                    if request.timeframe == "1d":
                        data = collector.fetch_daily_data(request.symbol, "full")
                    else:
                        data = collector.fetch_intraday_data(request.symbol, request.timeframe)
                else:
                    continue
                
                if data is not None and not data.empty:
                    # Calculate quality score (simplified)
                    quality_score = self._calculate_quality_score(data)
                    
                    # Check if quality meets requirements
                    if quality_score >= request.required_quality:
                        return CollectionResult(
                            symbol=request.symbol,
                            timeframe=request.timeframe,
                            success=True,
                            source=source_name,
                            data_rows=len(data),
                            quality_score=quality_score,
                            collection_time=time.time() - start_time
                        )
                    else:
                        logger.debug(f"⚠️ {source_name} quality too low for {request.symbol}: {quality_score:.3f}")
                
            except Exception as e:
                logger.debug(f"❌ {source_name} failed for {request.symbol}: {e}")
                # Mark source as temporarily unhealthy
                self.api_health[source_name] = False
                continue
        
        # All sources failed
        return CollectionResult(
            symbol=request.symbol,
            timeframe=request.timeframe,
            success=False,
            source="all_failed",
            data_rows=0,
            quality_score=0.0,
            collection_time=time.time() - start_time,
            error_message="All data sources failed"
        )
    
    def _get_symbol_priority(self, symbol: str) -> int:
        """Get priority for a symbol (lower number = higher priority)"""
        
        priority_groups = self.symbol_manager.get_symbols_by_priority()
        
        if symbol in priority_groups.get("HIGHEST", []):
            return 1
        elif symbol in priority_groups.get("HIGH", []):
            return 2
        elif symbol in priority_groups.get("MEDIUM", []):
            return 3
        else:
            return 5  # Default priority
    
    def _calculate_quality_score(self, data) -> float:
        """Calculate data quality score (simplified version)"""
        
        if data is None or data.empty:
            return 0.0
        
        score = 1.0
        
        # Check completeness
        completeness = data.count().sum() / (len(data) * len(data.columns))
        score *= completeness
        
        # Check for basic OHLC consistency
        if all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
            consistent_hl = (data['High'] >= data['Low']).mean()
            consistent_close = ((data['Close'] >= data['Low']) & (data['Close'] <= data['High'])).mean()
            consistency = (consistent_hl + consistent_close) / 2
            score *= consistency
        
        return min(score, 1.0)
    
    def get_api_health_status(self) -> Dict[str, Dict[str, Any]]:
        """Get comprehensive API health status"""
        
        status = {}
        
        for source_name in self.collectors.keys():
            config = self.config["api_sources"].get(source_name, {})
            
            # Calculate recent performance metrics
            recent_requests = len([
                ts for ts in self.request_timestamps[source_name]
                if time.time() - ts < 300  # Last 5 minutes
            ])
            
            status[source_name] = {
                "enabled": config.get("enabled", False),
                "healthy": self.api_health.get(source_name, False),
                "priority": config.get("priority", 999),
                "rate_limit": config.get("rate_limit_per_minute", 0),
                "recent_requests": recent_requests,
                "within_rate_limit": self._check_rate_limit(source_name),
                "last_check": self.last_health_check.get(source_name, "Never")
            }
        
        return status
    
    def refresh_api_health(self):
        """Refresh health status of all APIs"""
        
        logger.info("🔄 Refreshing API health status...")
        
        availability = self.detect_available_apis()
        
        for source_name, available in availability.items():
            self.last_health_check[source_name] = datetime.now().isoformat()
            
            if available != self.api_health.get(source_name):
                status = "✅ RECOVERED" if available else "❌ FAILED"
                logger.info(f"🔄 {source_name} status changed: {status}")
    
    def get_collection_statistics(self) -> Dict[str, Any]:
        """Get collection statistics and performance metrics"""
        
        stats = {
            "total_sources": len(self.collectors),
            "healthy_sources": sum(self.api_health.values()),
            "total_symbols": len(self.symbol_manager.get_all_symbols()),
            "api_health": self.api_health.copy(),
            "request_counts": {
                source: len(timestamps) 
                for source, timestamps in self.request_timestamps.items()
            }
        }
        
        return stats

# Convenience function for easy usage
def create_orchestrator(config_path: Optional[str] = None) -> APIOrchestrator:
    """Create and return an API orchestrator instance"""
    return APIOrchestrator(config_path)

# For testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test the orchestrator
    orchestrator = APIOrchestrator()
    
    # Test API detection
    availability = orchestrator.detect_available_apis()
    print(f"📊 API Availability: {availability}")
    
    # Test health status
    health = orchestrator.get_api_health_status()
    print(f"🏥 API Health: {health}")
    
    # Test small collection
    test_symbols = ["RY.TO", "TD.TO"]
    results = orchestrator.collect_market_data(test_symbols, timeframe="1d", period="5d")
    
    print(f"\n📊 Collection Results:")
    for symbol, result in results.items():
        status = "✅" if result.success else "❌"
        print(f"   {symbol}: {status} {result.data_rows} rows from {result.source}")