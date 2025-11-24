#!/usr/bin/env python3
"""
Start Data Collection - Persistent Collection System

Continuously attempts data collection with retry mechanisms and fallback strategies.
Monitors progress and keeps trying until successful or user stops.
"""

import logging
import sys
import time
import json
from pathlib import Path
from datetime import datetime, timedelta
import requests
from typing import Dict, List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_collection import (
    SymbolManager, 
    ProgressTracker, 
    StorageManager, 
    HistoricalAppender,
    DataValidator
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_collection_live.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class PersistentDataCollector:
    """Persistent data collection with retry mechanisms"""
    
    def __init__(self):
        self.symbol_manager = SymbolManager()
        self.progress_tracker = ProgressTracker()
        self.storage_manager = StorageManager()
        self.historical_appender = HistoricalAppender(max_workers=3)
        self.validator = DataValidator()
        
        self.retry_count = 0
        self.max_retries = 10
        self.retry_delay = 300  # 5 minutes between retries
        
        self.is_running = True
        
        logger.info("🚀 Persistent Data Collector initialized")
    
    def check_internet_connectivity(self) -> bool:
        """Check basic internet connectivity"""
        try:
            # Try multiple endpoints
            test_urls = [
                "https://www.google.com",
                "https://finance.yahoo.com", 
                "https://httpbin.org/get"
            ]
            
            for url in test_urls:
                try:
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        logger.info(f"✅ Internet connectivity confirmed via {url}")
                        return True
                except:
                    continue
            
            logger.warning("❌ No internet connectivity detected")
            return False
            
        except Exception as e:
            logger.error(f"Connectivity check failed: {e}")
            return False
    
    def test_yfinance_with_retry(self) -> bool:
        """Test yfinance with different strategies"""
        try:
            import yfinance as yf
            
            # Try different symbols and approaches
            test_strategies = [
                ("AAPL", "1d"),
                ("^GSPC", "1d"),  # S&P 500 index
                ("EURUSD=X", "1d"),  # Currency
                ("RY.TO", "1d"),  # Canadian stock
            ]
            
            for symbol, period in test_strategies:
                try:
                    logger.info(f"🧪 Testing yfinance with {symbol}")
                    
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period=period, timeout=30)
                    
                    if not data.empty:
                        logger.info(f"✅ yfinance working with {symbol}: {len(data)} rows")
                        return True
                    else:
                        logger.warning(f"⚠️ Empty data for {symbol}")
                        
                except Exception as e:
                    logger.warning(f"❌ Failed {symbol}: {e}")
                    continue
                
                # Small delay between attempts
                time.sleep(2)
            
            logger.error("❌ All yfinance test strategies failed")
            return False
            
        except Exception as e:
            logger.error(f"yfinance test failed: {e}")
            return False
    
    def attempt_data_collection(self) -> Dict[str, any]:
        """Attempt to collect data with current conditions"""
        
        logger.info("🎯 Starting data collection attempt...")
        
        # Start with high-priority symbols only
        priority_symbols = self.symbol_manager.get_symbols_by_priority()
        test_symbols = priority_symbols.get("HIGHEST", [])[:5]  # Just 5 symbols first
        
        logger.info(f"📊 Testing with {len(test_symbols)} high-priority symbols: {test_symbols}")
        
        try:
            # Attempt collection
            results = self.historical_appender.collect_all_symbols(
                symbols=test_symbols, 
                include_intraday=False  # Skip intraday for now
            )
            
            # Analyze results
            successful_symbols = []
            failed_symbols = []
            
            for symbol, result in results.items():
                if isinstance(result, dict) and any(result.values()):
                    successful_symbols.append(symbol)
                else:
                    failed_symbols.append(symbol)
            
            success_rate = len(successful_symbols) / len(test_symbols) if test_symbols else 0
            
            collection_summary = {
                "timestamp": datetime.now().isoformat(),
                "test_symbols": test_symbols,
                "successful_symbols": successful_symbols,
                "failed_symbols": failed_symbols,
                "success_rate": success_rate,
                "total_results": len(results)
            }
            
            logger.info(f"📊 Collection attempt results:")
            logger.info(f"   Success rate: {success_rate:.1%}")
            logger.info(f"   Successful: {successful_symbols}")
            logger.info(f"   Failed: {failed_symbols}")
            
            return collection_summary
            
        except Exception as e:
            logger.error(f"❌ Collection attempt failed: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "success_rate": 0.0
            }
    
    def monitor_and_retry_loop(self):
        """Main monitoring and retry loop"""
        
        logger.info("🔄 Starting persistent data collection loop...")
        logger.info(f"   Max retries: {self.max_retries}")
        logger.info(f"   Retry delay: {self.retry_delay} seconds")
        
        while self.is_running and self.retry_count < self.max_retries:
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"🔄 ATTEMPT {self.retry_count + 1}/{self.max_retries}")
                logger.info(f"{'='*60}")
                
                # Check internet connectivity
                if not self.check_internet_connectivity():
                    logger.warning("⚠️ No internet connectivity, waiting...")
                    time.sleep(60)  # Wait 1 minute for connectivity
                    continue
                
                # Test yfinance
                if not self.test_yfinance_with_retry():
                    logger.warning("⚠️ yfinance not responding, will retry...")
                    self.retry_count += 1
                    
                    if self.retry_count < self.max_retries:
                        logger.info(f"⏳ Waiting {self.retry_delay} seconds before retry...")
                        time.sleep(self.retry_delay)
                    continue
                
                # Attempt data collection
                results = self.attempt_data_collection()
                
                if results.get("success_rate", 0) > 0:
                    logger.info("🎉 Data collection successful!")
                    
                    # If partial success, try to continue with more symbols
                    if results.get("success_rate", 0) >= 0.6:  # 60% success rate
                        logger.info("📈 Good success rate, expanding to more symbols...")
                        self.expand_collection()
                    
                    break  # Success, exit retry loop
                else:
                    logger.warning("⚠️ Data collection failed, will retry...")
                    self.retry_count += 1
                    
                    if self.retry_count < self.max_retries:
                        logger.info(f"⏳ Waiting {self.retry_delay} seconds before retry...")
                        time.sleep(self.retry_delay)
                
            except KeyboardInterrupt:
                logger.info("🛑 User interrupted, stopping...")
                self.is_running = False
                break
                
            except Exception as e:
                logger.error(f"❌ Unexpected error in retry loop: {e}")
                self.retry_count += 1
                
                if self.retry_count < self.max_retries:
                    time.sleep(self.retry_delay)
        
        # Final status
        if self.retry_count >= self.max_retries:
            logger.error(f"❌ Max retries ({self.max_retries}) reached, giving up")
        elif not self.is_running:
            logger.info("🛑 Collection stopped by user")
        else:
            logger.info("✅ Collection completed successfully")
    
    def expand_collection(self):
        """Expand collection to more symbols after initial success"""
        
        logger.info("🚀 Expanding collection to all symbols...")
        
        try:
            # Get all symbols
            all_symbols = self.symbol_manager.get_all_symbols()
            
            # Start full collection
            results = self.historical_appender.collect_all_symbols(
                symbols=all_symbols,
                include_intraday=True,
                priority_first=True
            )
            
            # Get final summary
            summary = self.historical_appender.get_collection_summary()
            
            logger.info("🎉 Full collection completed!")
            logger.info(f"📊 Final Summary:")
            logger.info(f"   Total files: {summary['storage']['total_files']}")
            logger.info(f"   Total size: {summary['storage']['total_size_mb']:.2f} MB")
            logger.info(f"   Completion: {summary['progress']['overall_progress']['completion_percentage']:.1f}%")
            
        except Exception as e:
            logger.error(f"❌ Full collection failed: {e}")
    
    def get_current_status(self) -> Dict[str, any]:
        """Get current collection status"""
        
        try:
            progress_summary = self.progress_tracker.get_progress_summary()
            storage_summary = self.storage_manager.get_storage_summary()
            
            return {
                "timestamp": datetime.now().isoformat(),
                "retry_count": self.retry_count,
                "max_retries": self.max_retries,
                "is_running": self.is_running,
                "progress": progress_summary,
                "storage": storage_summary
            }
            
        except Exception as e:
            logger.error(f"Failed to get status: {e}")
            return {"error": str(e)}

def main():
    """Main entry point"""
    
    print("🚀 STARTING PERSISTENT DATA COLLECTION SYSTEM")
    print("="*80)
    print("This system will continuously attempt to collect historical data")
    print("until successful or maximum retries reached.")
    print("Press Ctrl+C to stop at any time.")
    print("="*80)
    
    collector = PersistentDataCollector()
    
    try:
        # Show initial status
        status = collector.get_current_status()
        print(f"\n📊 Initial Status:")
        print(f"   Progress: {status.get('progress', {}).get('overall_progress', {}).get('completion_percentage', 0):.1f}%")
        print(f"   Storage: {status.get('storage', {}).get('total_files', 0)} files")
        
        # Start monitoring loop
        collector.monitor_and_retry_loop()
        
        # Show final status
        final_status = collector.get_current_status()
        print(f"\n📊 Final Status:")
        print(f"   Progress: {final_status.get('progress', {}).get('overall_progress', {}).get('completion_percentage', 0):.1f}%")
        print(f"   Storage: {final_status.get('storage', {}).get('total_files', 0)} files, {final_status.get('storage', {}).get('total_size_mb', 0):.2f} MB")
        
    except KeyboardInterrupt:
        print("\n🛑 Collection stopped by user")
    except Exception as e:
        print(f"\n❌ System error: {e}")
    
    print("\n✅ Data collection system shutdown complete")

if __name__ == "__main__":
    main()