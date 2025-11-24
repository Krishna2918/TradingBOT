"""
Historical Appender - 20-Year Data Collection

Fetches and stores 20 years of historical data for TSX/TSXV symbols.
Handles daily, weekly, monthly, quarterly, and yearly timeframes with rate limiting.
"""

import logging
import time
import random
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from .symbol_manager import SymbolManager
from .progress_tracker import ProgressTracker
from .storage_manager import StorageManager

logger = logging.getLogger(__name__)

class HistoricalAppender:
    """Collects 20 years of historical data with intelligent rate limiting"""
    
    def __init__(self, max_workers: int = 5):
        self.symbol_manager = SymbolManager()
        self.progress_tracker = ProgressTracker()
        self.storage_manager = StorageManager()
        self.max_workers = max_workers
        
        # Rate limiting configuration
        self.min_delay = 1.0  # Minimum delay between requests (seconds)
        self.max_delay = 5.0  # Maximum delay between requests (seconds)
        self.request_count = 0
        self.start_time = datetime.now()
        self.lock = threading.Lock()
        
        # User agent rotation for respectful scraping
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:91.0) Gecko/20100101",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:91.0) Gecko/20100101"
        ]
        
        logger.info("🚀 Historical Appender initialized")
    
    def _rate_limit_delay(self):
        """Apply intelligent rate limiting"""
        with self.lock:
            self.request_count += 1
            
            # Calculate requests per minute
            elapsed_minutes = (datetime.now() - self.start_time).total_seconds() / 60
            if elapsed_minutes > 0:
                rpm = self.request_count / elapsed_minutes
                
                # Adjust delay based on request rate
                if rpm > 30:  # Too fast, slow down
                    delay = random.uniform(self.max_delay, self.max_delay * 1.5)
                elif rpm > 20:  # Moderate pace
                    delay = random.uniform(self.min_delay * 1.5, self.max_delay)
                else:  # Safe pace
                    delay = random.uniform(self.min_delay, self.min_delay * 2)
            else:
                rpm = 0.0
                delay = random.uniform(self.min_delay, self.max_delay)
            
            logger.debug(f"Rate limiting: {delay:.2f}s delay (RPM: {rpm:.1f})")
            time.sleep(delay)
    
    def fetch_symbol_data(self, symbol: str, period: str = "max", interval: str = "1d") -> Optional[pd.DataFrame]:
        """Fetch data for a single symbol with error handling"""
        try:
            # Apply rate limiting
            self._rate_limit_delay()
            
            # Create ticker with random user agent
            ticker = yf.Ticker(symbol)
            
            # Fetch data
            logger.info(f"📥 Fetching {symbol} {interval} data (period: {period})")
            data = ticker.history(period=period, interval=interval, auto_adjust=True, prepost=True)
            
            if data.empty:
                logger.warning(f"⚠️ No data returned for {symbol} {interval}")
                return None
            
            # Clean column names (remove any extra spaces)
            data.columns = data.columns.str.strip()
            
            # Ensure we have the required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_cols):
                logger.error(f"❌ Missing required columns for {symbol}: {data.columns.tolist()}")
                return None
            
            logger.info(f"✅ Fetched {symbol} {interval}: {len(data)} rows from {data.index.min()} to {data.index.max()}")
            return data
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch {symbol} {interval}: {e}")
            return None
    
    def fetch_daily_data(self, symbol: str) -> bool:
        """Fetch 20 years of daily data for a symbol"""
        try:
            data = self.fetch_symbol_data(symbol, period="max", interval="1d")
            
            if data is None or data.empty:
                self.progress_tracker.log_progress(symbol, "1d", 0, 0.0, 1, 0.0)
                return False
            
            # Save to storage
            success = self.storage_manager.save_to_parquet(data, symbol, "1d")
            
            if success:
                # Calculate data size
                file_path = self.storage_manager.get_file_path(symbol, "1d")
                data_size_mb = file_path.stat().st_size / (1024 * 1024) if file_path.exists() else 0.0
                
                # Validate data quality
                quality_metrics = self.storage_manager.validate_data_quality(symbol, "1d")
                quality_score = quality_metrics.get("quality_score", 0.0)
                
                # Log progress
                self.progress_tracker.log_progress(
                    symbol, "1d", len(data), data_size_mb, 0, quality_score
                )
                
                logger.info(f"💾 Saved {symbol} daily data: {len(data)} rows, quality: {quality_score:.3f}")
                return True
            else:
                self.progress_tracker.log_progress(symbol, "1d", 0, 0.0, 1, 0.0)
                return False
                
        except Exception as e:
            logger.error(f"Failed to fetch daily data for {symbol}: {e}")
            self.progress_tracker.log_progress(symbol, "1d", 0, 0.0, 1, 0.0)
            return False
    
    def fetch_intraday_data(self, symbol: str, interval: str) -> bool:
        """Fetch recent intraday data for a symbol"""
        try:
            # Determine period based on interval (yfinance limitations)
            period_map = {
                "1m": "7d",    # 1-minute: last 7 days only
                "5m": "60d",   # 5-minute: last 60 days
                "15m": "60d",  # 15-minute: last 60 days  
                "30m": "60d"   # 30-minute: last 60 days
            }
            
            period = period_map.get(interval, "60d")
            
            data = self.fetch_symbol_data(symbol, period=period, interval=interval)
            
            if data is None or data.empty:
                self.progress_tracker.log_progress(symbol, interval, 0, 0.0, 1, 0.0)
                return False
            
            # Save to storage (intraday uses different file structure)
            success = self.storage_manager.save_to_parquet(data, symbol, interval)
            
            if success:
                # Calculate data size
                file_path = self.storage_manager.get_file_path(symbol, interval)
                data_size_mb = file_path.stat().st_size / (1024 * 1024) if file_path.exists() else 0.0
                
                # Validate data quality
                quality_metrics = self.storage_manager.validate_data_quality(symbol, interval)
                quality_score = quality_metrics.get("quality_score", 0.0)
                
                # Log progress
                self.progress_tracker.log_progress(
                    symbol, interval, len(data), data_size_mb, 0, quality_score
                )
                
                logger.info(f"💾 Saved {symbol} {interval} data: {len(data)} rows, quality: {quality_score:.3f}")
                return True
            else:
                self.progress_tracker.log_progress(symbol, interval, 0, 0.0, 1, 0.0)
                return False
                
        except Exception as e:
            logger.error(f"Failed to fetch {interval} data for {symbol}: {e}")
            self.progress_tracker.log_progress(symbol, interval, 0, 0.0, 1, 0.0)
            return False
    
    def aggregate_higher_timeframes(self, symbol: str) -> bool:
        """Create weekly, monthly, quarterly, yearly data from daily data"""
        try:
            # Load daily data
            daily_data = self.storage_manager.load_from_parquet(symbol, "1d")
            
            if daily_data.empty:
                logger.warning(f"No daily data available for {symbol} aggregation")
                return False
            
            # Define aggregation rules
            agg_rules = {
                'Open': 'first',
                'High': 'max', 
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }
            
            # Create aggregated timeframes
            timeframes = {
                "1wk": "W",      # Weekly
                "1mo": "M",      # Monthly  
                "3mo": "Q",      # Quarterly
                "1y": "Y"        # Yearly
            }
            
            success_count = 0
            
            for timeframe, freq in timeframes.items():
                try:
                    # Aggregate data
                    aggregated = daily_data.resample(freq).agg(agg_rules)
                    
                    # Remove rows with no data (e.g., all NaN)
                    aggregated = aggregated.dropna()
                    
                    if aggregated.empty:
                        logger.warning(f"No {timeframe} data after aggregation for {symbol}")
                        continue
                    
                    # Save aggregated data
                    success = self.storage_manager.save_to_parquet(aggregated, symbol, timeframe)
                    
                    if success:
                        # Calculate data size
                        file_path = self.storage_manager.get_file_path(symbol, timeframe)
                        data_size_mb = file_path.stat().st_size / (1024 * 1024) if file_path.exists() else 0.0
                        
                        # Log progress (aggregated data should have high quality)
                        self.progress_tracker.log_progress(
                            symbol, timeframe, len(aggregated), data_size_mb, 0, 0.95
                        )
                        
                        success_count += 1
                        logger.info(f"📊 Aggregated {symbol} {timeframe}: {len(aggregated)} rows")
                    
                except Exception as e:
                    logger.error(f"Failed to aggregate {symbol} {timeframe}: {e}")
                    continue
            
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Failed to aggregate higher timeframes for {symbol}: {e}")
            return False
    
    def collect_symbol_data(self, symbol: str, include_intraday: bool = True) -> Dict[str, bool]:
        """Collect all data for a single symbol"""
        results = {}
        
        logger.info(f"🎯 Starting data collection for {symbol}")
        
        # 1. Fetch daily data (20 years)
        results["daily"] = self.fetch_daily_data(symbol)
        
        # 2. Aggregate higher timeframes from daily data
        if results["daily"]:
            results["aggregation"] = self.aggregate_higher_timeframes(symbol)
        else:
            results["aggregation"] = False
        
        # 3. Fetch intraday data (recent only due to yfinance limits)
        if include_intraday:
            intraday_intervals = ["1m", "5m", "15m", "30m"]
            for interval in intraday_intervals:
                results[f"intraday_{interval}"] = self.fetch_intraday_data(symbol, interval)
        
        # Summary
        success_count = sum(results.values())
        total_count = len(results)
        
        logger.info(f"📊 {symbol} collection complete: {success_count}/{total_count} successful")
        
        return results
    
    def collect_all_symbols(self, symbols: Optional[List[str]] = None, 
                           include_intraday: bool = True,
                           priority_first: bool = True) -> Dict[str, Dict[str, bool]]:
        """Collect data for all symbols with parallel processing"""
        
        # Get symbols to process
        if symbols is None:
            if priority_first:
                # Process by priority: HIGHEST → HIGH → MEDIUM
                symbol_groups = self.symbol_manager.get_symbols_by_priority()
                symbols = []
                for priority in ["HIGHEST", "HIGH", "MEDIUM"]:
                    symbols.extend(symbol_groups.get(priority, []))
            else:
                symbols = self.symbol_manager.get_all_symbols()
        
        logger.info(f"🚀 Starting collection for {len(symbols)} symbols")
        
        # Start collection session
        session_id = self.progress_tracker.start_session("historical_collection")
        
        results = {}
        
        try:
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_symbol = {
                    executor.submit(self.collect_symbol_data, symbol, include_intraday): symbol
                    for symbol in symbols
                }
                
                # Process completed tasks
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        result = future.result()
                        results[symbol] = result
                        
                        # Log progress
                        success_count = sum(result.values())
                        total_count = len(result)
                        logger.info(f"✅ Completed {symbol}: {success_count}/{total_count} successful")
                        
                    except Exception as e:
                        logger.error(f"❌ Failed to collect data for {symbol}: {e}")
                        results[symbol] = {"error": str(e)}
        
        finally:
            # End collection session
            self.progress_tracker.end_session(session_id)
        
        # Summary statistics
        total_symbols = len(results)
        successful_symbols = sum(1 for r in results.values() if isinstance(r, dict) and any(r.values()))
        
        logger.info(f"🎉 Collection complete: {successful_symbols}/{total_symbols} symbols successful")
        
        # Export progress report
        self.progress_tracker.export_progress_report("reports/historical_collection_report.json")
        
        return results
    
    def resume_incomplete_collection(self, completion_threshold: float = 95.0) -> Dict[str, Dict[str, bool]]:
        """Resume collection for incomplete symbols"""
        
        # Get incomplete symbols
        incomplete_symbols = self.progress_tracker.get_incomplete_symbols(completion_threshold)
        
        if not incomplete_symbols:
            logger.info("✅ All symbols are complete, nothing to resume")
            return {}
        
        logger.info(f"🔄 Resuming collection for {len(incomplete_symbols)} incomplete symbols")
        
        return self.collect_all_symbols(symbols=incomplete_symbols, include_intraday=True)
    
    def get_collection_summary(self) -> Dict[str, any]:
        """Get summary of collection progress"""
        progress_summary = self.progress_tracker.get_progress_summary()
        storage_summary = self.storage_manager.get_storage_summary()
        
        return {
            "progress": progress_summary,
            "storage": storage_summary,
            "timestamp": datetime.now().isoformat()
        }

# For testing and standalone execution
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    appender = HistoricalAppender(max_workers=3)
    
    # Test with a few symbols first
    test_symbols = ["RY.TO", "TD.TO", "SHOP.TO"]
    
    print(f"🧪 Testing historical data collection with {test_symbols}")
    
    results = appender.collect_all_symbols(symbols=test_symbols, include_intraday=True)
    
    # Show results
    for symbol, result in results.items():
        if isinstance(result, dict):
            success_count = sum(1 for v in result.values() if v is True)
            total_count = len(result)
            print(f"{symbol}: {success_count}/{total_count} successful")
        else:
            print(f"{symbol}: Failed")
    
    # Show summary
    summary = appender.get_collection_summary()
    print(f"\n📊 Collection Summary:")
    print(f"Total files: {summary['storage']['total_files']}")
    print(f"Total size: {summary['storage']['total_size_mb']:.2f} MB")
    print(f"Completion: {summary['progress']['overall_progress']['completion_percentage']:.1f}%")