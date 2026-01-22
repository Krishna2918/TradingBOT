"""
Market Calendar - Smart Market Hours and Trading Day Detection

Handles TSX/TSXV market hours, holidays, and weekend detection to optimize
data collection timing and avoid unnecessary API calls during market closures.
"""

import logging
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Tuple
import pytz
import pandas as pd

logger = logging.getLogger(__name__)

class MarketCalendar:
    """Smart market calendar for TSX/TSXV trading hours and holidays"""
    
    def __init__(self):
        # Toronto timezone (TSX/TSXV timezone)
        self.tz = pytz.timezone('America/Toronto')
        
        # TSX trading hours (9:30 AM - 4:00 PM ET)
        self.market_open = time(9, 30)
        self.market_close = time(16, 0)
        
        # Known holidays (add more as needed)
        self.holidays_2025 = [
            "2025-01-01",  # New Year's Day
            "2025-02-17",  # Family Day
            "2025-04-18",  # Good Friday
            "2025-05-19",  # Victoria Day
            "2025-07-01",  # Canada Day
            "2025-08-04",  # Civic Holiday
            "2025-09-01",  # Labour Day
            "2025-10-14",  # Thanksgiving
            "2025-12-25",  # Christmas Day
            "2025-12-26",  # Boxing Day
        ]
        
        logger.info("📅 Market Calendar initialized for TSX/TSXV")
    
    def is_market_open_now(self) -> bool:
        """Check if market is currently open"""
        now = datetime.now(self.tz)
        return self.is_market_open_at(now)
    
    def is_market_open_at(self, dt: datetime) -> bool:
        """Check if market is open at specific datetime"""
        
        # Convert to Toronto timezone if needed
        if dt.tzinfo is None:
            dt = self.tz.localize(dt)
        else:
            dt = dt.astimezone(self.tz)
        
        # Check if it's a weekend
        if dt.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Check if it's a holiday
        if dt.date().isoformat() in self.holidays_2025:
            return False
        
        # Check if it's within trading hours
        current_time = dt.time()
        return self.market_open <= current_time <= self.market_close
    
    def is_trading_day(self, date: datetime = None) -> bool:
        """Check if given date is a trading day (ignores time)"""
        if date is None:
            date = datetime.now(self.tz)
        
        # Convert to Toronto timezone if needed
        if date.tzinfo is None:
            date = self.tz.localize(date)
        else:
            date = date.astimezone(self.tz)
        
        # Check if it's a weekend
        if date.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Check if it's a holiday
        if date.date().isoformat() in self.holidays_2025:
            return False
        
        return True
    
    def get_last_trading_day(self) -> datetime:
        """Get the most recent trading day"""
        current = datetime.now(self.tz).replace(hour=12, minute=0, second=0, microsecond=0)
        
        # Go back until we find a trading day
        while not self.is_trading_day(current):
            current -= timedelta(days=1)
        
        return current
    
    def get_next_trading_day(self) -> datetime:
        """Get the next trading day"""
        current = datetime.now(self.tz).replace(hour=12, minute=0, second=0, microsecond=0)
        current += timedelta(days=1)  # Start from tomorrow
        
        # Go forward until we find a trading day
        while not self.is_trading_day(current):
            current += timedelta(days=1)
        
        return current
    
    def get_market_status(self) -> Dict[str, any]:
        """Get comprehensive market status"""
        now = datetime.now(self.tz)
        
        status = {
            "current_time": now.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "is_trading_day": self.is_trading_day(now),
            "is_market_open": self.is_market_open_now(),
            "last_trading_day": self.get_last_trading_day().strftime("%Y-%m-%d"),
            "next_trading_day": self.get_next_trading_day().strftime("%Y-%m-%d"),
            "market_hours": f"{self.market_open.strftime('%H:%M')} - {self.market_close.strftime('%H:%M')} ET"
        }
        
        # Add context message
        if not status["is_trading_day"]:
            if now.weekday() == 5:  # Saturday
                status["message"] = "Markets closed - Saturday"
            elif now.weekday() == 6:  # Sunday
                status["message"] = "Markets closed - Sunday"
            else:
                status["message"] = "Markets closed - Holiday"
        elif not status["is_market_open"]:
            if now.time() < self.market_open:
                status["message"] = "Markets closed - Before opening"
            else:
                status["message"] = "Markets closed - After closing"
        else:
            status["message"] = "Markets open - Trading hours"
        
        return status
    
    def should_collect_data_now(self) -> Tuple[bool, str]:
        """Determine if we should collect data now and why"""
        
        status = self.get_market_status()
        
        # Always allow historical data collection (markets closed doesn't affect historical data)
        if not status["is_trading_day"]:
            return True, f"Historical data collection OK - {status['message']}"
        
        # During trading hours, be more careful with API calls
        if status["is_market_open"]:
            return True, "Live data collection - Markets open"
        
        # After hours on trading days
        return True, "Historical data collection - Markets closed but trading day"
    
    def get_optimal_collection_strategy(self) -> Dict[str, any]:
        """Get optimal data collection strategy based on market status"""
        
        status = self.get_market_status()
        should_collect, reason = self.should_collect_data_now()
        
        strategy = {
            "should_collect": should_collect,
            "reason": reason,
            "recommended_approach": "",
            "rate_limit_strategy": "",
            "data_freshness_expectation": ""
        }
        
        if not status["is_trading_day"]:
            # Weekend or holiday - focus on historical data
            strategy["recommended_approach"] = "Historical data collection - no new intraday data expected"
            strategy["rate_limit_strategy"] = "Conservative - use full rate limits for historical data"
            strategy["data_freshness_expectation"] = f"Latest data from {status['last_trading_day']}"
            
        elif status["is_market_open"]:
            # During trading hours - be careful with rate limits
            strategy["recommended_approach"] = "Live data collection - new data available"
            strategy["rate_limit_strategy"] = "Aggressive - prioritize real-time data"
            strategy["data_freshness_expectation"] = "Real-time data available"
            
        else:
            # After hours on trading day
            strategy["recommended_approach"] = "End-of-day data collection - final prices available"
            strategy["rate_limit_strategy"] = "Moderate - collect end-of-day data"
            strategy["data_freshness_expectation"] = "End-of-day data available"
        
        return strategy
    
    def log_market_status(self):
        """Log current market status for debugging"""
        status = self.get_market_status()
        strategy = self.get_optimal_collection_strategy()
        
        logger.info("📅 MARKET STATUS")
        logger.info(f"   Current Time: {status['current_time']}")
        logger.info(f"   Trading Day: {'✅' if status['is_trading_day'] else '❌'}")
        logger.info(f"   Market Open: {'✅' if status['is_market_open'] else '❌'}")
        logger.info(f"   Status: {status['message']}")
        logger.info(f"   Last Trading Day: {status['last_trading_day']}")
        logger.info(f"   Next Trading Day: {status['next_trading_day']}")
        logger.info(f"   Collection Strategy: {strategy['recommended_approach']}")
        logger.info(f"   Data Freshness: {strategy['data_freshness_expectation']}")

# For testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    calendar = MarketCalendar()
    
    print("📅 MARKET CALENDAR TEST")
    print("=" * 50)
    
    # Show current status
    calendar.log_market_status()
    
    # Test specific dates
    test_dates = [
        datetime(2025, 10, 25),  # Today (Saturday)
        datetime(2025, 10, 24),  # Yesterday (Friday)
        datetime(2025, 10, 28),  # Monday
        datetime(2025, 12, 25),  # Christmas
    ]
    
    print("\n🧪 Testing specific dates:")
    for test_date in test_dates:
        is_trading = calendar.is_trading_day(test_date)
        day_name = test_date.strftime("%A")
        print(f"   {test_date.strftime('%Y-%m-%d')} ({day_name}): {'✅ Trading Day' if is_trading else '❌ Non-Trading Day'}")
    
    # Show strategy
    strategy = calendar.get_optimal_collection_strategy()
    print(f"\n💡 Recommended Strategy:")
    print(f"   Should Collect: {'✅' if strategy['should_collect'] else '❌'}")
    print(f"   Reason: {strategy['reason']}")
    print(f"   Approach: {strategy['recommended_approach']}")
    print(f"   Rate Limiting: {strategy['rate_limit_strategy']}")
    print(f"   Data Freshness: {strategy['data_freshness_expectation']}")