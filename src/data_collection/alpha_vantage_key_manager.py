"""
Alpha Vantage API Key Manager for Free Tier Keys

Manages three free tier Alpha Vantage keys with 25 requests/day each (75 total).
Implements intelligent rotation, usage tracking, and optimal scheduling.
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

@dataclass
class KeyUsage:
    """Track usage for a single API key"""
    key_name: str
    api_key: str
    daily_limit: int
    requests_used_today: int
    last_request_time: Optional[datetime]
    last_reset_date: str
    is_available: bool
    purpose: str  # 'market_data', 'sentiment', 'backup'
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        if self.last_request_time:
            data['last_request_time'] = self.last_request_time.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'KeyUsage':
        """Create from dictionary"""
        if data.get('last_request_time'):
            data['last_request_time'] = datetime.fromisoformat(data['last_request_time'])
        return cls(**data)

class AlphaVantageKeyManager:
    """
    Manages multiple Alpha Vantage free tier API keys
    
    Features:
    - Round-robin key rotation
    - Daily usage tracking (25 requests per key per day)
    - Automatic daily reset at midnight UTC
    - Intelligent key selection based on purpose
    - Usage optimization and monitoring
    - Hourly usage reports
    """
    
    def __init__(self, usage_file: str = 'data/alpha_vantage_usage.json'):
        self.usage_file = Path(usage_file)
        self.usage_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize keys from environment
        self.keys = {
            'premium': KeyUsage(
                key_name='Premium',
                api_key=os.getenv('AV_PREMIUM_KEY', ''),
                daily_limit=4500,  # 75 RPM * 60 minutes = 4500/hour, conservative daily estimate
                requests_used_today=0,
                last_request_time=None,
                last_reset_date=datetime.now().strftime('%Y-%m-%d'),
                is_available=True,
                purpose='premium_realtime'
            ),
            'primary': KeyUsage(
                key_name='Primary',
                api_key=os.getenv('ALPHA_VANTAGE_API_KEY', ''),
                daily_limit=25,
                requests_used_today=0,
                last_request_time=None,
                last_reset_date=datetime.now().strftime('%Y-%m-%d'),
                is_available=True,
                purpose='market_data'
            ),
            'secondary': KeyUsage(
                key_name='Secondary',
                api_key=os.getenv('ALPHA_VANTAGE_API_KEY_SECONDARY', ''),
                daily_limit=25,
                requests_used_today=0,
                last_request_time=None,
                last_reset_date=datetime.now().strftime('%Y-%m-%d'),
                is_available=True,
                purpose='backup'
            ),
            'sentiment': KeyUsage(
                key_name='Sentiment',
                api_key=os.getenv('AV_SENTIMENT_KEY', ''),
                daily_limit=25,
                requests_used_today=0,
                last_request_time=None,
                last_reset_date=datetime.now().strftime('%Y-%m-%d'),
                is_available=True,
                purpose='sentiment'
            )
        }
        
        # Load existing usage data
        self._load_usage_data()
        
        # Check for daily reset
        self._check_daily_reset()
        
        logger.info(f"Alpha Vantage Key Manager initialized with {len(self.keys)} keys")
        self._log_current_status()
    
    def _load_usage_data(self):
        """Load usage data from file"""
        if self.usage_file.exists():
            try:
                with open(self.usage_file, 'r') as f:
                    data = json.load(f)
                
                for key_id, key_data in data.items():
                    if key_id in self.keys:
                        # Update existing key with saved data
                        saved_key = KeyUsage.from_dict(key_data)
                        self.keys[key_id].requests_used_today = saved_key.requests_used_today
                        self.keys[key_id].last_request_time = saved_key.last_request_time
                        self.keys[key_id].last_reset_date = saved_key.last_reset_date
                        self.keys[key_id].is_available = saved_key.is_available
                
                logger.info("Loaded existing usage data")
            except Exception as e:
                logger.error(f"Error loading usage data: {e}")
    
    def _save_usage_data(self):
        """Save usage data to file"""
        try:
            data = {key_id: key.to_dict() for key_id, key in self.keys.items()}
            with open(self.usage_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving usage data: {e}")
    
    def _check_daily_reset(self):
        """Check if we need to reset daily counters"""
        today = datetime.now().strftime('%Y-%m-%d')
        
        for key in self.keys.values():
            if key.last_reset_date != today:
                logger.info(f"Resetting daily counter for {key.key_name} key")
                key.requests_used_today = 0
                key.last_reset_date = today
                key.is_available = True
        
        self._save_usage_data()
    
    def _log_current_status(self):
        """Log current status of all keys"""
        total_used = sum(key.requests_used_today for key in self.keys.values())
        total_available = sum(key.daily_limit - key.requests_used_today for key in self.keys.values())
        
        logger.info(f"Daily Usage Summary:")
        logger.info(f"  Total used today: {total_used}/75 requests")
        logger.info(f"  Total available: {total_available} requests")
        
        for key in self.keys.values():
            remaining = key.daily_limit - key.requests_used_today
            logger.info(f"  {key.key_name} ({key.purpose}): {key.requests_used_today}/{key.daily_limit} used, {remaining} remaining")
    
    def get_best_key(self, purpose: str = 'any') -> Optional[Tuple[str, str]]:
        """
        Get the best available key for a specific purpose
        
        Args:
            purpose: 'market_data', 'sentiment', 'backup', or 'any'
        
        Returns:
            Tuple of (key_name, api_key) or None if no keys available
        """
        self._check_daily_reset()
        
        # Filter keys by purpose
        if purpose == 'premium_realtime':
            candidates = [k for k in self.keys.values() if k.purpose == 'premium_realtime' and k.is_available]
        elif purpose == 'sentiment':
            candidates = [k for k in self.keys.values() if k.purpose == 'sentiment' and k.is_available]
        elif purpose == 'market_data':
            # Prefer premium for market data, fallback to free keys
            candidates = [k for k in self.keys.values() if k.purpose in ['premium_realtime', 'market_data', 'backup'] and k.is_available]
        else:
            candidates = [k for k in self.keys.values() if k.is_available]
        
        # Filter by availability (not at daily limit)
        available_keys = [k for k in candidates if k.requests_used_today < k.daily_limit]
        
        if not available_keys:
            logger.warning(f"No available keys for purpose: {purpose}")
            return None
        
        # Sort by priority: premium first (if available), then by usage
        available_keys.sort(key=lambda k: (k.purpose != 'premium_realtime', k.requests_used_today / k.daily_limit))
        
        best_key = available_keys[0]
        logger.debug(f"Selected {best_key.key_name} key for {purpose} (used: {best_key.requests_used_today}/{best_key.daily_limit})")
        
        return best_key.key_name, best_key.api_key
    
    def record_request(self, api_key: str, success: bool = True):
        """
        Record a request made with a specific API key
        
        Args:
            api_key: The API key that was used
            success: Whether the request was successful
        """
        # Find the key
        key_obj = None
        for key in self.keys.values():
            if key.api_key == api_key:
                key_obj = key
                break
        
        if not key_obj:
            logger.error(f"Unknown API key used: {api_key[:8]}...")
            return
        
        # Record the request
        key_obj.requests_used_today += 1
        key_obj.last_request_time = datetime.now()
        
        # Check if key is now at limit
        if key_obj.requests_used_today >= key_obj.daily_limit:
            key_obj.is_available = False
            logger.warning(f"{key_obj.key_name} key has reached daily limit ({key_obj.daily_limit})")
        
        # Save updated usage
        self._save_usage_data()
        
        logger.debug(f"Recorded request for {key_obj.key_name} key: {key_obj.requests_used_today}/{key_obj.daily_limit}")
    
    def make_request(self, function: str, params: Dict[str, str], purpose: str = 'any') -> Optional[Dict]:
        """
        Make an Alpha Vantage API request using the best available key
        
        Args:
            function: Alpha Vantage function name (e.g., 'GLOBAL_QUOTE')
            params: Additional parameters (e.g., {'symbol': 'AAPL'})
            purpose: Request purpose for key selection
        
        Returns:
            API response data or None if failed
        """
        key_info = self.get_best_key(purpose)
        if not key_info:
            logger.error(f"No available keys for {purpose} request")
            return None
        
        key_name, api_key = key_info
        
        # Build URL
        url_params = {'function': function, 'apikey': api_key}
        url_params.update(params)
        
        # Add delayed entitlement for premium key if not specified
        if api_key == os.getenv('AV_PREMIUM_KEY') and 'entitlement' not in url_params:
            url_params['entitlement'] = 'delayed'
        
        url = 'https://www.alphavantage.co/query?' + '&'.join(f'{k}={v}' for k, v in url_params.items())
        
        try:
            logger.debug(f"Making {function} request with {key_name} key")
            response = requests.get(url, timeout=15)
            data = response.json()
            
            # Check for errors
            if 'Error Message' in data:
                logger.error(f"API Error: {data['Error Message']}")
                self.record_request(api_key, success=False)
                return None
            elif 'Note' in data and ('25 requests per day' in data['Note'] or 'rate limit' in data['Note'].lower()):
                logger.warning(f"Rate/Daily limit reached for {key_name} key: {data['Note']}")
                self.record_request(api_key, success=False)
                return None
            elif 'Information' in data and ('25 requests per day' in data['Information'] or 'rate limit' in data['Information'].lower()):
                logger.warning(f"Rate/Daily limit reached for {key_name} key: {data['Information']}")
                self.record_request(api_key, success=False)
                return None
            else:
                # Success - handle both regular and delayed data formats
                success_indicators = [
                    'Global Quote',
                    'Global Quote - DATA DELAYED BY 15 MINUTES',
                    'Time Series (1min)',
                    'Time Series (5min)',
                    'Time Series (15min)',
                    'Time Series (30min)',
                    'Time Series (60min)',
                    'Time Series (Daily)',
                    'Technical Analysis: RSI',
                    'Technical Analysis: SMA',
                    'Technical Analysis: EMA',
                    'Technical Analysis: MACD',
                    'Technical Analysis: STOCH',
                    'Technical Analysis: ADX',
                    'Technical Analysis: CCI',
                    'Technical Analysis: AROON',
                    'Technical Analysis: BBANDS',
                    'items'  # For news/sentiment
                ]
                
                if any(indicator in data for indicator in success_indicators):
                    self.record_request(api_key, success=True)
                    logger.debug(f"Successful {function} request with {key_name} key")
                    return data
                else:
                    logger.warning(f"Unexpected response format from {key_name} key: {list(data.keys())}")
                    self.record_request(api_key, success=False)
                    return None
        
        except Exception as e:
            logger.error(f"Request failed: {e}")
            self.record_request(api_key, success=False)
            return None
    
    def get_usage_summary(self) -> Dict:
        """Get comprehensive usage summary"""
        self._check_daily_reset()
        
        total_used = sum(key.requests_used_today for key in self.keys.values())
        total_limit = sum(key.daily_limit for key in self.keys.values())
        total_remaining = total_limit - total_used
        
        key_details = []
        for key in self.keys.values():
            remaining = key.daily_limit - key.requests_used_today
            key_details.append({
                'name': key.key_name,
                'purpose': key.purpose,
                'used': key.requests_used_today,
                'limit': key.daily_limit,
                'remaining': remaining,
                'available': key.is_available,
                'last_used': key.last_request_time.isoformat() if key.last_request_time else None
            })
        
        return {
            'total_used': total_used,
            'total_limit': total_limit,
            'total_remaining': total_remaining,
            'usage_percentage': (total_used / total_limit) * 100,
            'keys': key_details,
            'reset_date': datetime.now().strftime('%Y-%m-%d'),
            'next_reset': (datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)).isoformat()
        }
    
    def get_hourly_usage_plan(self) -> Dict:
        """
        Generate an optimal hourly usage plan for remaining requests
        
        Returns plan for distributing remaining requests throughout the day
        """
        summary = self.get_usage_summary()
        remaining = summary['total_remaining']
        
        if remaining <= 0:
            return {
                'remaining_requests': 0,
                'hours_left_today': 0,
                'recommended_rate': 0,
                'plan': 'All requests used for today'
            }
        
        # Calculate hours left in the day
        now = datetime.now()
        end_of_day = now.replace(hour=23, minute=59, second=59, microsecond=0)
        hours_left = (end_of_day - now).total_seconds() / 3600
        
        if hours_left <= 0:
            hours_left = 24  # Next day
        
        # Recommended rate
        recommended_rate = remaining / hours_left if hours_left > 0 else 0
        
        # Create hourly plan
        plan_details = []
        if remaining > 0 and hours_left > 0:
            requests_per_hour = max(1, int(remaining / hours_left))
            
            # Market hours get priority (9:30 AM - 4:00 PM EST)
            market_hours = []
            current_hour = now.hour
            
            for hour in range(24):
                if 9 <= hour <= 16:  # Market hours
                    priority = 'high'
                    suggested_requests = min(requests_per_hour * 2, remaining // 8)  # Distribute more during market hours
                elif 6 <= hour <= 8 or 17 <= hour <= 20:  # Pre/post market
                    priority = 'medium'
                    suggested_requests = requests_per_hour
                else:  # Off hours
                    priority = 'low'
                    suggested_requests = max(1, requests_per_hour // 2)
                
                plan_details.append({
                    'hour': f'{hour:02d}:00',
                    'priority': priority,
                    'suggested_requests': suggested_requests
                })
        
        return {
            'remaining_requests': remaining,
            'hours_left_today': round(hours_left, 1),
            'recommended_rate': round(recommended_rate, 2),
            'hourly_plan': plan_details,
            'strategy': 'Prioritize market hours (9:30 AM - 4:00 PM EST) for real-time data'
        }

# Global instance
_key_manager = None

def get_alpha_vantage_key_manager() -> AlphaVantageKeyManager:
    """Get the global Alpha Vantage key manager instance"""
    global _key_manager
    if _key_manager is None:
        _key_manager = AlphaVantageKeyManager()
    return _key_manager

# Convenience functions
def make_alpha_vantage_request(function: str, params: Dict[str, str], purpose: str = 'any') -> Optional[Dict]:
    """Make an Alpha Vantage request using the key manager"""
    return get_alpha_vantage_key_manager().make_request(function, params, purpose)

def get_alpha_vantage_usage() -> Dict:
    """Get Alpha Vantage usage summary"""
    return get_alpha_vantage_key_manager().get_usage_summary()

def get_alpha_vantage_hourly_plan() -> Dict:
    """Get Alpha Vantage hourly usage plan"""
    return get_alpha_vantage_key_manager().get_hourly_usage_plan()

if __name__ == '__main__':
    # Test the key manager
    logging.basicConfig(level=logging.INFO)
    
    manager = AlphaVantageKeyManager()
    
    print("=== Alpha Vantage Key Manager Test ===")
    
    # Test usage summary
    summary = manager.get_usage_summary()
    print(f"\nUsage Summary:")
    print(f"Total: {summary['total_used']}/{summary['total_limit']} ({summary['usage_percentage']:.1f}%)")
    
    for key in summary['keys']:
        print(f"  {key['name']} ({key['purpose']}): {key['used']}/{key['limit']} - {'Available' if key['available'] else 'Exhausted'}")
    
    # Test hourly plan
    plan = manager.get_hourly_usage_plan()
    print(f"\nHourly Plan:")
    print(f"Remaining: {plan['remaining_requests']} requests")
    print(f"Hours left: {plan['hours_left_today']}")
    print(f"Recommended rate: {plan['recommended_rate']} requests/hour")
    
    # Test request (if keys available)
    if summary['total_remaining'] > 0:
        print(f"\nTesting API request...")
        data = manager.make_request('GLOBAL_QUOTE', {'symbol': 'AAPL'}, 'market_data')
        if data:
            print("✅ Test request successful")
        else:
            print("❌ Test request failed")
    else:
        print("\n⚠️  No requests remaining for testing")