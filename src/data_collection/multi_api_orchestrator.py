"""
Multi-API Data Orchestrator

Efficiently coordinates data collection across all available APIs with intelligent
fallback, rate limiting, and optimization strategies.
"""

import logging
import asyncio
import aiohttp
import time
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
from pathlib import Path

# Import existing collectors
from .alpha_vantage_collector import AlphaVantageCollector
from .storage_manager import StorageManager
from .progress_tracker import ProgressTracker

logger = logging.getLogger(__name__)

@dataclass
class APISource:
    """Configuration for an API data source"""
    name: str
    priority: int  # Lower number = higher priority
    rate_limit: int  # Requests per minute
    cost_per_request: float  # Cost in credits/dollars
    reliability_score: float  # 0.0 to 1.0
    data_quality_score: float  # 0.0 to 1.0
    enabled: bool = True
    api_key_env: Optional[str] = None
    base_url: Optional[str] = None

@dataclass
class DataRequest:
    """Request for market data"""
    symbol: str
    timeframe: str
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    priority: int = 5  # 1=highest, 10=lowest

class MultiAPIOrchestrator:
    """Orchestrates data collection across multiple APIs efficiently"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.api_sources = self._initialize_api_sources()
        self.rate_limiters = {}
        self.api_health = {}
        self.storage_manager = StorageManager()
        self.progress_tracker = ProgressTracker()
        
        # Performance tracking
        self.request_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'fallback_usage': 0,
            'cost_savings': 0.0
        }
        
        logger.info("🚀 Multi-API Orchestrator initialized")
        self._log_available_apis()
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from YAML file with sensible defaults"""
        default_config = {
            'max_concurrent_requests': 5,
            'request_timeout': 30,
            'retry_attempts': 3,
            'fallback_enabled': True,
            'cost_optimization': True,
            'quality_threshold': 0.8,
            'cache_duration_hours': 24
        }

        # Determine config path
        if config_path is None:
            # Try default locations
            possible_paths = [
                'config/api_config.yaml',
                '../config/api_config.yaml',
                Path(__file__).parent.parent.parent / 'config' / 'api_config.yaml'
            ]
            for path in possible_paths:
                if Path(path).exists():
                    config_path = str(path)
                    break

        # Load from YAML file if provided/found
        if config_path and Path(config_path).exists():
            try:
                import yaml
                with open(config_path, 'r') as file:
                    yaml_config = yaml.safe_load(file)

                if yaml_config:
                    # Merge orchestrator settings
                    orchestrator_config = yaml_config.get('orchestrator', {})
                    for key, value in orchestrator_config.items():
                        default_config[key] = value

                    # Store API sources config for later use
                    self._yaml_api_sources = yaml_config.get('api_sources', {})
                    self._symbol_priorities = yaml_config.get('symbol_priorities', {})
                    self._data_quality_config = yaml_config.get('data_quality', {})
                    self._rate_limiting_config = yaml_config.get('rate_limiting', {})
                    self._caching_config = yaml_config.get('caching', {})

                    logger.info(f"✅ Configuration loaded from {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
                self._yaml_api_sources = {}
                self._symbol_priorities = {}
        else:
            self._yaml_api_sources = {}
            self._symbol_priorities = {}
            if config_path:
                logger.warning(f"Config file not found: {config_path}, using defaults")

        return default_config
    
    def _initialize_api_sources(self) -> Dict[str, APISource]:
        """Initialize all available API sources from YAML config or defaults"""
        sources = {}

        # Check if YAML config was loaded
        yaml_sources = getattr(self, '_yaml_api_sources', {})

        if yaml_sources:
            # Initialize from YAML configuration
            for source_id, config in yaml_sources.items():
                api_key_env = config.get('api_key_env')

                # Determine if enabled based on config and API key availability
                enabled = config.get('enabled', True)
                if api_key_env and enabled:
                    if source_id == 'reddit_api':
                        enabled = self._check_reddit_credentials()
                    else:
                        enabled = self._check_api_key(api_key_env)

                sources[source_id] = APISource(
                    name=config.get('name', source_id),
                    priority=config.get('priority', 5),
                    rate_limit=config.get('rate_limit', 60),
                    cost_per_request=config.get('cost_per_request', 0.0),
                    reliability_score=config.get('reliability_score', 0.8),
                    data_quality_score=config.get('data_quality_score', 0.8),
                    enabled=enabled,
                    api_key_env=api_key_env,
                    base_url=config.get('base_url')
                )

            logger.info(f"Initialized {len(sources)} API sources from YAML config")
        else:
            # Use default configuration
            sources = {
                'yahoo_finance': APISource(
                    name='Yahoo Finance',
                    priority=1,  # Highest priority - free and reliable
                    rate_limit=600,  # 10 per second = 600 per minute
                    cost_per_request=0.0,
                    reliability_score=0.85,  # Sometimes has rate limits
                    data_quality_score=0.90,
                    enabled=True
                ),

                'alpha_vantage': APISource(
                    name='Alpha Vantage',
                    priority=2,  # Second choice - professional grade
                    rate_limit=5,  # Free tier: 5 per minute
                    cost_per_request=0.0,  # Free tier
                    reliability_score=0.95,
                    data_quality_score=0.95,
                    enabled=self._check_api_key('ALPHA_VANTAGE_API_KEY'),
                    api_key_env='ALPHA_VANTAGE_API_KEY',
                    base_url='https://www.alphavantage.co/query'
                ),

                'finnhub': APISource(
                    name='Finnhub',
                    priority=3,  # Good for news and basic data
                    rate_limit=60,  # Free tier: 60 per minute
                    cost_per_request=0.0,  # Free tier
                    reliability_score=0.90,
                    data_quality_score=0.85,
                    enabled=self._check_api_key('FINNHUB_API_KEY'),
                    api_key_env='FINNHUB_API_KEY',
                    base_url='https://finnhub.io/api/v1'
                ),

                'news_api': APISource(
                    name='News API',
                    priority=4,  # For sentiment data only
                    rate_limit=1000,  # Per day, not per minute
                    cost_per_request=0.0,  # Free tier
                    reliability_score=0.80,
                    data_quality_score=0.75,
                    enabled=self._check_api_key('NEWS_API_KEY'),
                    api_key_env='NEWS_API_KEY',
                    base_url='https://newsapi.org/v2'
                ),

                'reddit_api': APISource(
                    name='Reddit API',
                    priority=5,  # For social sentiment
                    rate_limit=100,  # 100 per minute
                    cost_per_request=0.0,
                    reliability_score=0.75,
                    data_quality_score=0.70,
                    enabled=self._check_reddit_credentials(),
                    api_key_env='REDDIT_CLIENT_ID'
                )
            }
        
        # Initialize rate limiters
        for source_id, source in sources.items():
            if source.enabled:
                self.rate_limiters[source_id] = RateLimiter(
                    source.rate_limit, 
                    source.name
                )
                self.api_health[source_id] = {
                    'status': 'unknown',
                    'last_check': None,
                    'success_rate': 1.0,
                    'avg_response_time': 0.0
                }
        
        return sources
    
    def _check_api_key(self, env_var: str) -> bool:
        """Check if API key is available and not a placeholder"""
        key = os.getenv(env_var, '')
        return key and key not in ['demo', 'YOUR_API_KEY', 'placeholder']
    
    def _check_reddit_credentials(self) -> bool:
        """Check if Reddit API credentials are available"""
        client_id = os.getenv('REDDIT_CLIENT_ID', '')
        client_secret = os.getenv('REDDIT_CLIENT_SECRET', '')
        return (client_id and client_secret and 
                client_id not in ['demo', 'YOUR_CLIENT_ID'] and
                client_secret not in ['demo', 'YOUR_CLIENT_SECRET'])
    
    def _log_available_apis(self):
        """Log which APIs are available"""
        enabled_apis = [s.name for s in self.api_sources.values() if s.enabled]
        disabled_apis = [s.name for s in self.api_sources.values() if not s.enabled]
        
        logger.info(f"✅ Available APIs ({len(enabled_apis)}): {', '.join(enabled_apis)}")
        if disabled_apis:
            logger.info(f"❌ Disabled APIs ({len(disabled_apis)}): {', '.join(disabled_apis)}")
    
    async def collect_market_data(self, symbols: List[str], timeframe: str = '1d') -> Dict[str, pd.DataFrame]:
        """Collect market data for multiple symbols using optimal API strategy"""
        
        logger.info(f"📊 Starting data collection for {len(symbols)} symbols ({timeframe})")
        
        # Create data requests with priority
        requests = []
        for symbol in symbols:
            priority = self._get_symbol_priority(symbol)
            requests.append(DataRequest(symbol, timeframe, priority=priority))
        
        # Sort by priority (lower number = higher priority)
        requests.sort(key=lambda x: x.priority)
        
        # Collect data with intelligent API selection
        results = {}
        failed_symbols = []
        
        # Use ThreadPoolExecutor for concurrent requests
        with ThreadPoolExecutor(max_workers=self.config['max_concurrent_requests']) as executor:
            # Submit all requests
            future_to_request = {}
            for request in requests:
                future = executor.submit(self._collect_single_symbol, request)
                future_to_request[future] = request
            
            # Process completed requests
            for future in as_completed(future_to_request):
                request = future_to_request[future]
                try:
                    data = future.result()
                    if data is not None and not data.empty:
                        results[request.symbol] = data
                        logger.info(f"✅ {request.symbol}: {len(data)} rows collected")
                    else:
                        failed_symbols.append(request.symbol)
                        logger.warning(f"❌ {request.symbol}: No data collected")
                except Exception as e:
                    failed_symbols.append(request.symbol)
                    logger.error(f"❌ {request.symbol}: Error - {e}")
        
        # Log collection summary
        success_rate = len(results) / len(symbols) * 100
        logger.info(f"📊 Collection complete: {len(results)}/{len(symbols)} symbols ({success_rate:.1f}% success)")
        
        if failed_symbols:
            logger.warning(f"⚠️ Failed symbols: {', '.join(failed_symbols[:10])}")
        
        return results
    
    def _collect_single_symbol(self, request: DataRequest) -> Optional[pd.DataFrame]:
        """Collect data for a single symbol using best available API"""
        
        # Get ordered list of APIs to try
        api_order = self._get_optimal_api_order(request)
        
        for api_id in api_order:
            try:
                # Check rate limit
                if not self.rate_limiters[api_id].acquire():
                    logger.debug(f"⏳ Rate limit hit for {api_id}, trying next API")
                    continue
                
                # Collect data from this API
                data = self._collect_from_api(api_id, request)
                
                if data is not None and not data.empty:
                    # Update success stats
                    self._update_api_health(api_id, True, time.time())
                    self.request_stats['successful_requests'] += 1
                    
                    # Log which API was used
                    api_name = self.api_sources[api_id].name
                    logger.debug(f"📡 {request.symbol} collected from {api_name}")
                    
                    return data
                else:
                    # API returned no data, try next
                    self._update_api_health(api_id, False, time.time())
                    continue
                    
            except Exception as e:
                # API failed, try next
                self._update_api_health(api_id, False, time.time())
                logger.debug(f"❌ {api_id} failed for {request.symbol}: {e}")
                continue
        
        # All APIs failed
        self.request_stats['failed_requests'] += 1
        logger.warning(f"❌ All APIs failed for {request.symbol}")
        return None
    
    def _get_optimal_api_order(self, request: DataRequest) -> List[str]:
        """Get optimal order of APIs to try for this request"""
        
        # Get enabled APIs sorted by priority
        available_apis = [
            (api_id, source) for api_id, source in self.api_sources.items() 
            if source.enabled
        ]
        
        # Sort by multiple factors
        def api_score(api_tuple):
            api_id, source = api_tuple
            health = self.api_health.get(api_id, {})
            
            # Factors: priority (lower=better), success rate, cost, response time
            priority_score = source.priority
            success_score = health.get('success_rate', 1.0)
            cost_score = source.cost_per_request
            speed_score = 1.0 / (health.get('avg_response_time', 1.0) + 1.0)
            
            # Weighted combination (lower is better)
            total_score = (
                priority_score * 0.4 +  # Priority is most important
                (1.0 - success_score) * 0.3 +  # Success rate
                cost_score * 0.2 +  # Cost
                (1.0 - speed_score) * 0.1  # Speed
            )
            
            return total_score
        
        # Sort and return API IDs
        sorted_apis = sorted(available_apis, key=api_score)
        return [api_id for api_id, _ in sorted_apis]
    
    def _collect_from_api(self, api_id: str, request: DataRequest) -> Optional[pd.DataFrame]:
        """Collect data from specific API"""
        
        start_time = time.time()
        
        try:
            if api_id == 'yahoo_finance':
                return self._collect_yahoo_finance(request)
            elif api_id == 'alpha_vantage':
                return self._collect_alpha_vantage(request)
            elif api_id == 'finnhub':
                return self._collect_finnhub(request)
            else:
                logger.warning(f"Unknown API: {api_id}")
                return None
                
        finally:
            # Update response time
            response_time = time.time() - start_time
            self._update_response_time(api_id, response_time)
    
    def _collect_yahoo_finance(self, request: DataRequest) -> Optional[pd.DataFrame]:
        """Collect data from Yahoo Finance with improved error handling"""
        import yfinance as yf
        
        try:
            ticker = yf.Ticker(request.symbol)
            
            # Determine period based on timeframe
            if request.timeframe == '1d':
                period = 'max'  # Get all available data
            elif request.timeframe in ['1m', '5m', '15m', '30m']:
                period = '7d'  # Intraday data limited to 7 days
            else:
                period = '2y'  # Other timeframes
            
            # Fetch data with timeout
            data = ticker.history(
                period=period,
                interval=request.timeframe,
                timeout=self.config['request_timeout']
            )
            
            if data.empty:
                return None
            
            # Clean and standardize data
            data = self._standardize_data(data, request.symbol, 'yahoo_finance')
            return data
            
        except Exception as e:
            logger.debug(f"Yahoo Finance error for {request.symbol}: {e}")
            return None
    
    def _collect_alpha_vantage(self, request: DataRequest) -> Optional[pd.DataFrame]:
        """Collect data from Alpha Vantage"""
        try:
            collector = AlphaVantageCollector()
            
            if request.timeframe == '1d':
                data = collector.fetch_daily_data(request.symbol, outputsize='full')
            elif request.timeframe in ['1m', '5m', '15m', '30m']:
                # Map timeframe to Alpha Vantage format
                av_interval = request.timeframe.replace('m', 'min')
                data = collector.fetch_intraday_data(request.symbol, interval=av_interval)
            else:
                # Alpha Vantage doesn't support other timeframes directly
                return None
            
            if data is not None and not data.empty:
                data = self._standardize_data(data, request.symbol, 'alpha_vantage')
            
            return data
            
        except Exception as e:
            logger.debug(f"Alpha Vantage error for {request.symbol}: {e}")
            return None
    
    def _collect_finnhub(self, request: DataRequest) -> Optional[pd.DataFrame]:
        """Collect data from Finnhub (basic implementation)"""
        try:
            # This is a placeholder - Finnhub integration would go here
            # For now, return None to indicate not implemented
            return None
            
        except Exception as e:
            logger.debug(f"Finnhub error for {request.symbol}: {e}")
            return None
    
    def _standardize_data(self, data: pd.DataFrame, symbol: str, source: str) -> pd.DataFrame:
        """Standardize data format across all sources"""
        
        # Ensure required columns exist
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Rename columns if needed (different APIs use different names)
        column_mapping = {
            'Adj Close': 'Close',  # Use adjusted close if available
            'Adj_Close': 'Close'
        }
        
        data = data.rename(columns=column_mapping)
        
        # Ensure all required columns exist
        for col in required_columns:
            if col not in data.columns:
                logger.warning(f"Missing column {col} in {symbol} data from {source}")
                return pd.DataFrame()  # Return empty if missing critical columns
        
        # Select only required columns
        data = data[required_columns].copy()
        
        # Add metadata
        data.attrs['symbol'] = symbol
        data.attrs['source'] = source
        data.attrs['collected_at'] = datetime.now()
        
        return data
    
    def _get_symbol_priority(self, symbol: str) -> int:
        """Get priority for symbol (lower number = higher priority)"""
        
        # High priority symbols (major Canadian banks and tech)
        high_priority = ['RY.TO', 'TD.TO', 'BNS.TO', 'BMO.TO', 'CM.TO', 'SHOP.TO']
        
        # Medium priority symbols (other TSX 60)
        medium_priority = ['CNQ.TO', 'SU.TO', 'ENB.TO', 'TRP.TO', 'CVE.TO']
        
        if symbol in high_priority:
            return 1
        elif symbol in medium_priority:
            return 2
        else:
            return 3
    
    def _update_api_health(self, api_id: str, success: bool, response_time: float):
        """Update API health metrics"""
        if api_id not in self.api_health:
            self.api_health[api_id] = {
                'status': 'unknown',
                'last_check': None,
                'success_rate': 1.0,
                'avg_response_time': 0.0,
                'total_requests': 0,
                'successful_requests': 0
            }
        
        health = self.api_health[api_id]
        health['last_check'] = datetime.now()
        health['total_requests'] += 1
        
        if success:
            health['successful_requests'] += 1
            health['status'] = 'healthy'
        
        # Update success rate (exponential moving average)
        current_success_rate = health['successful_requests'] / health['total_requests']
        health['success_rate'] = (
            0.8 * health['success_rate'] + 0.2 * current_success_rate
        )
        
        # Update status based on success rate
        if health['success_rate'] > 0.8:
            health['status'] = 'healthy'
        elif health['success_rate'] > 0.5:
            health['status'] = 'degraded'
        else:
            health['status'] = 'unhealthy'
    
    def _update_response_time(self, api_id: str, response_time: float):
        """Update average response time for API"""
        if api_id in self.api_health:
            current_avg = self.api_health[api_id]['avg_response_time']
            # Exponential moving average
            self.api_health[api_id]['avg_response_time'] = (
                0.8 * current_avg + 0.2 * response_time
            )
    
    def get_api_status_report(self) -> Dict[str, Any]:
        """Get comprehensive API status report"""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_apis': len(self.api_sources),
            'enabled_apis': len([s for s in self.api_sources.values() if s.enabled]),
            'healthy_apis': len([h for h in self.api_health.values() if h.get('status') == 'healthy']),
            'request_stats': self.request_stats.copy(),
            'api_details': {}
        }
        
        for api_id, source in self.api_sources.items():
            health = self.api_health.get(api_id, {})
            
            report['api_details'][api_id] = {
                'name': source.name,
                'enabled': source.enabled,
                'priority': source.priority,
                'rate_limit': source.rate_limit,
                'cost_per_request': source.cost_per_request,
                'reliability_score': source.reliability_score,
                'status': health.get('status', 'unknown'),
                'success_rate': health.get('success_rate', 0.0),
                'avg_response_time': health.get('avg_response_time', 0.0),
                'last_check': health.get('last_check')
            }
        
        return report
    
    def optimize_api_usage(self) -> Dict[str, Any]:
        """Analyze and optimize API usage patterns"""
        
        recommendations = []
        cost_savings = 0.0
        
        # Analyze API performance
        for api_id, health in self.api_health.items():
            source = self.api_sources[api_id]
            
            if health.get('success_rate', 0) < 0.5:
                recommendations.append(f"Consider disabling {source.name} (low success rate: {health.get('success_rate', 0):.1%})")
            
            if health.get('avg_response_time', 0) > 10.0:
                recommendations.append(f"{source.name} is slow (avg: {health.get('avg_response_time', 0):.1f}s)")
        
        # Calculate potential cost savings
        total_requests = self.request_stats['successful_requests']
        if total_requests > 0:
            # Estimate savings from using free APIs vs paid alternatives
            free_api_usage = sum(1 for api_id in self.api_health.keys() 
                               if self.api_sources[api_id].cost_per_request == 0.0)
            
            if free_api_usage > 0:
                cost_savings = total_requests * 0.01  # Assume $0.01 per request for paid APIs
        
        return {
            'recommendations': recommendations,
            'estimated_cost_savings': cost_savings,
            'optimization_score': self._calculate_optimization_score(),
            'next_actions': self._get_optimization_actions()
        }
    
    def _calculate_optimization_score(self) -> float:
        """Calculate how well optimized the current API usage is"""
        
        if not self.api_health:
            return 0.0
        
        # Factors: success rate, cost efficiency, speed
        total_score = 0.0
        total_weight = 0.0
        
        for api_id, health in self.api_health.items():
            source = self.api_sources[api_id]
            
            success_score = health.get('success_rate', 0.0)
            cost_score = 1.0 if source.cost_per_request == 0.0 else 0.5
            speed_score = min(1.0, 5.0 / (health.get('avg_response_time', 5.0) + 1.0))
            
            api_score = (success_score * 0.5 + cost_score * 0.3 + speed_score * 0.2)
            weight = source.priority  # Higher priority APIs matter more
            
            total_score += api_score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _get_optimization_actions(self) -> List[str]:
        """Get specific actions to optimize API usage"""
        
        actions = []
        
        # Check for missing API keys
        for api_id, source in self.api_sources.items():
            if not source.enabled and source.api_key_env:
                actions.append(f"Add {source.api_key_env} environment variable to enable {source.name}")
        
        # Check for underperforming APIs
        for api_id, health in self.api_health.items():
            if health.get('success_rate', 0) < 0.3:
                actions.append(f"Investigate {self.api_sources[api_id].name} connectivity issues")
        
        # Suggest rate limit optimizations
        if self.request_stats['failed_requests'] > self.request_stats['successful_requests'] * 0.1:
            actions.append("Consider reducing concurrent requests to avoid rate limits")
        
        return actions


class RateLimiter:
    """Simple rate limiter for API requests"""
    
    def __init__(self, requests_per_minute: int, name: str):
        self.requests_per_minute = requests_per_minute
        self.name = name
        self.requests = []
        self.lock = asyncio.Lock() if asyncio.iscoroutinefunction(self.acquire) else None
    
    def acquire(self) -> bool:
        """Try to acquire a request permit"""
        now = time.time()
        
        # Remove old requests (older than 1 minute)
        self.requests = [req_time for req_time in self.requests if now - req_time < 60]
        
        # Check if we can make another request
        if len(self.requests) < self.requests_per_minute:
            self.requests.append(now)
            return True
        
        return False
    
    def get_wait_time(self) -> float:
        """Get time to wait before next request is allowed"""
        if not self.requests:
            return 0.0
        
        oldest_request = min(self.requests)
        wait_time = 60 - (time.time() - oldest_request)
        return max(0.0, wait_time)


# For testing
if __name__ == "__main__":
    import asyncio
    
    logging.basicConfig(level=logging.INFO)
    
    async def test_orchestrator():
        orchestrator = MultiAPIOrchestrator()
        
        # Test with a few symbols
        test_symbols = ['RY.TO', 'TD.TO', 'SHOP.TO']
        
        print("🧪 Testing Multi-API Orchestrator...")
        
        # Collect data
        results = await orchestrator.collect_market_data(test_symbols, '1d')
        
        print(f"📊 Results: {len(results)} symbols collected")
        for symbol, data in results.items():
            print(f"   {symbol}: {len(data)} rows")
        
        # Get status report
        status = orchestrator.get_api_status_report()
        print(f"\n📈 API Status:")
        print(f"   Enabled APIs: {status['enabled_apis']}/{status['total_apis']}")
        print(f"   Healthy APIs: {status['healthy_apis']}")
        print(f"   Success Rate: {status['request_stats']['successful_requests']}/{status['request_stats']['total_requests']}")
        
        # Get optimization recommendations
        optimization = orchestrator.optimize_api_usage()
        print(f"\n🎯 Optimization Score: {optimization['optimization_score']:.2f}")
        if optimization['recommendations']:
            print("   Recommendations:")
            for rec in optimization['recommendations']:
                print(f"     - {rec}")
    
    # Run test
    asyncio.run(test_orchestrator())