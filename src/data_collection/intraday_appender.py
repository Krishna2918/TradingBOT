"""
Intraday Appender - Real-Time Session Data Collection

Collects intraday data during TSX market hours with intelligent polling.
Appends to existing files without duplication for continuous data collection.
"""

import logging
import time
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set
import pandas as pd
import yfinance as yf
from pathlib import Path

from .symbol_manager import SymbolManager
from .progress_tracker import ProgressTracker
from .storage_manager import StorageManager

logger = logging.getLogger(__name__)

class IntradayAppender:
    """Real-time intraday data collection during market hours"""
    
    def __init__(self):
        self.symbol_manager = SymbolManager()
        self.progress_tracker = ProgressTracker()
        self.storage_manager = StorageManager()
        
        # Market hours (TSX: 9:30 AM - 4:00 PM EDT)
        self.market_open_hour = 9
        self.market_open_minute = 30
        self.market_close_hour = 16
        self.market_close_minute = 0
        
        # Polling configuration
        self.polling_intervals = {
            "1m": 60,    # Poll every 1 minute for 1-minute data
            "5m": 300,   # Poll every 5 minutes for 5-minute data
            "15m": 900,  # Poll every 15 minutes for 15-minute data
            "30m": 1800  # Poll every 30 minutes for 30-minute data
        }
        
        # Control flags
        self.is_running = False
        self.polling_threads = {}
        
        logger.info("📡 Intraday Appender initialized")
    
    def is_market_open(self) -> bool:
        """Check if TSX market is currently open"""
        now = datetime.now()
        
        # Check if it's a weekday (Monday=0, Sunday=6)
        if now.weekday() >= 5:  # Saturday or Sunday
            return False
        
        # Check market hours (9:30 AM - 4:00 PM EDT)
        market_open = now.replace(hour=self.market_open_hour, minute=self.market_open_minute, second=0, microsecond=0)
        market_close = now.replace(hour=self.market_close_hour, minute=self.market_close_minute, second=0, microsecond=0)
        
        return market_open <= now <= market_close
    
    def get_market_status(self) -> Dict[str, any]:
        """Get detailed market status information"""
        now = datetime.now()
        
        # Calculate next market open/close
        if now.weekday() >= 5:  # Weekend
            # Next Monday
            days_until_monday = 7 - now.weekday()
            next_open = (now + timedelta(days=days_until_monday)).replace(
                hour=self.market_open_hour, minute=self.market_open_minute, second=0, microsecond=0
            )
        else:
            # Today or next business day
            today_open = now.replace(hour=self.market_open_hour, minute=self.market_open_minute, second=0, microsecond=0)
            today_close = now.replace(hour=self.market_close_hour, minute=self.market_close_minute, second=0, microsecond=0)
            
            if now < today_open:
                next_open = today_open
            elif now > today_close:
                # Next business day
                next_open = (now + timedelta(days=1)).replace(
                    hour=self.market_open_hour, minute=self.market_open_minute, second=0, microsecond=0
                )
            else:
                next_open = today_close  # Market closes today
        
        return {
            "is_open": self.is_market_open(),
            "current_time": now.isoformat(),
            "next_event": next_open.isoformat(),
            "next_event_type": "open" if not self.is_market_open() else "close"
        }
    
    def fetch_recent_intraday_data(self, symbol: str, interval: str, period: str = "1d") -> Optional[pd.DataFrame]:
        """Fetch recent intraday data for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            
            logger.debug(f"📥 Fetching {symbol} {interval} data (period: {period})")
            data = ticker.history(period=period, interval=interval, auto_adjust=True, prepost=False)
            
            if data.empty:
                logger.warning(f"⚠️ No intraday data returned for {symbol} {interval}")
                return None
            
            # Clean column names
            data.columns = data.columns.str.strip()
            
            # Ensure required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_cols):
                logger.error(f"❌ Missing required columns for {symbol}: {data.columns.tolist()}")
                return None
            
            logger.debug(f"✅ Fetched {symbol} {interval}: {len(data)} rows")
            return data
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch intraday data for {symbol} {interval}: {e}")
            return None
    
    def append_intraday_data(self, symbol: str, interval: str) -> bool:
        """Append new intraday data for a symbol"""
        try:
            # Fetch recent data
            new_data = self.fetch_recent_intraday_data(symbol, interval, period="1d")
            
            if new_data is None or new_data.empty:
                return False
            
            # Append to existing file (handles deduplication)
            success = self.storage_manager.append_to_parquet(new_data, symbol, interval)
            
            if success:
                # Calculate data size
                file_path = self.storage_manager.get_file_path(symbol, interval)
                data_size_mb = file_path.stat().st_size / (1024 * 1024) if file_path.exists() else 0.0
                
                # Log progress
                self.progress_tracker.log_progress(
                    symbol, interval, len(new_data), data_size_mb, 0, 0.90
                )
                
                logger.info(f"📈 Appended {symbol} {interval}: {len(new_data)} new rows")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to append intraday data for {symbol} {interval}: {e}")
            return False
    
    def poll_symbols_for_interval(self, symbols: List[str], interval: str):
        """Poll symbols for a specific interval"""
        thread_name = f"poll_{interval}"
        
        while self.is_running:
            try:
                if not self.is_market_open():
                    logger.info(f"📴 Market closed, pausing {interval} polling")
                    time.sleep(300)  # Check every 5 minutes
                    continue
                
                logger.info(f"📡 Polling {len(symbols)} symbols for {interval} data")
                
                success_count = 0
                for symbol in symbols:
                    if not self.is_running:  # Check if we should stop
                        break
                    
                    success = self.append_intraday_data(symbol, interval)
                    if success:
                        success_count += 1
                    
                    # Small delay between symbols
                    time.sleep(1)
                
                logger.info(f"✅ {interval} polling complete: {success_count}/{len(symbols)} successful")
                
                # Wait for next polling cycle
                polling_interval = self.polling_intervals.get(interval, 300)
                time.sleep(polling_interval)
                
            except Exception as e:
                logger.error(f"Error in {interval} polling thread: {e}")
                time.sleep(60)  # Wait before retrying
    
    def start_polling(self, symbols: Optional[List[str]] = None, intervals: Optional[List[str]] = None):
        """Start intraday polling for specified symbols and intervals"""
        
        if self.is_running:
            logger.warning("Intraday polling is already running")
            return
        
        # Default symbols (use high-priority symbols for intraday)
        if symbols is None:
            priority_symbols = self.symbol_manager.get_symbols_by_priority()
            symbols = priority_symbols.get("HIGHEST", [])[:20]  # Limit to top 20 for intraday
        
        # Default intervals
        if intervals is None:
            intervals = ["1m", "5m", "15m", "30m"]
        
        logger.info(f"🚀 Starting intraday polling for {len(symbols)} symbols, {len(intervals)} intervals")
        
        self.is_running = True
        
        # Start polling threads for each interval
        for interval in intervals:
            if interval in self.polling_intervals:
                thread = threading.Thread(
                    target=self.poll_symbols_for_interval,
                    args=(symbols, interval),
                    name=f"intraday_poll_{interval}",
                    daemon=True
                )
                thread.start()
                self.polling_threads[interval] = thread
                
                logger.info(f"📡 Started {interval} polling thread")
        
        # Start session tracking
        session_id = self.progress_tracker.start_session("intraday_polling")
        logger.info(f"📊 Started intraday session {session_id}")
    
    def stop_polling(self):
        """Stop intraday polling"""
        if not self.is_running:
            logger.warning("Intraday polling is not running")
            return
        
        logger.info("🛑 Stopping intraday polling...")
        
        self.is_running = False
        
        # Wait for threads to finish
        for interval, thread in self.polling_threads.items():
            if thread.is_alive():
                logger.info(f"⏳ Waiting for {interval} thread to stop...")
                thread.join(timeout=30)  # Wait up to 30 seconds
        
        self.polling_threads.clear()
        
        logger.info("✅ Intraday polling stopped")
    
    def get_polling_status(self) -> Dict[str, any]:
        """Get current polling status"""
        status = {
            "is_running": self.is_running,
            "market_status": self.get_market_status(),
            "active_threads": {},
            "polling_intervals": self.polling_intervals
        }
        
        for interval, thread in self.polling_threads.items():
            status["active_threads"][interval] = {
                "is_alive": thread.is_alive(),
                "name": thread.name
            }
        
        return status
    
    def collect_session_data(self, symbols: Optional[List[str]] = None, 
                           duration_minutes: int = 60) -> Dict[str, any]:
        """Collect intraday data for a specific session duration"""
        
        if symbols is None:
            priority_symbols = self.symbol_manager.get_symbols_by_priority()
            symbols = priority_symbols.get("HIGHEST", [])[:10]  # Limit for session
        
        logger.info(f"📊 Starting {duration_minutes}-minute intraday collection session")
        
        # Start polling
        self.start_polling(symbols, intervals=["1m", "5m"])
        
        # Wait for specified duration
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        try:
            while datetime.now() < end_time and self.is_running:
                remaining = (end_time - datetime.now()).total_seconds() / 60
                logger.info(f"⏰ Session running... {remaining:.1f} minutes remaining")
                time.sleep(60)  # Check every minute
        
        finally:
            # Stop polling
            self.stop_polling()
        
        # Get collection summary
        summary = {
            "session_duration_minutes": duration_minutes,
            "symbols_collected": symbols,
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "storage_summary": self.storage_manager.get_storage_summary()
        }
        
        logger.info(f"✅ Intraday session complete: {len(symbols)} symbols")
        
        return summary

# For testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    appender = IntradayAppender()
    
    # Test market status
    status = appender.get_market_status()
    print(f"📊 Market Status:")
    print(f"   Open: {status['is_open']}")
    print(f"   Current time: {status['current_time']}")
    print(f"   Next event: {status['next_event']} ({status['next_event_type']})")
    
    # Test data fetching (without starting full polling)
    test_symbol = "RY.TO"
    print(f"\n🧪 Testing intraday data fetch for {test_symbol}")
    
    data = appender.fetch_recent_intraday_data(test_symbol, "5m", period="1d")
    if data is not None and not data.empty:
        print(f"✅ Fetched {len(data)} rows of 5-minute data")
        print(f"   Date range: {data.index.min()} to {data.index.max()}")
    else:
        print("❌ No data fetched")
    
    # Test append (if data was fetched)
    if data is not None and not data.empty:
        success = appender.append_intraday_data(test_symbol, "5m")
        print(f"💾 Append test: {'✅ Success' if success else '❌ Failed'}")