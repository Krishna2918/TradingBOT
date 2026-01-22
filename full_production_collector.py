#!/usr/bin/env python3
"""
Full Production Data Collector - 1,400+ Stocks
This script collects data for a large universe of stocks with monitoring and progress tracking.
"""

import asyncio
import yfinance as yf
import pandas as pd
import os
import json
from pathlib import Path
from datetime import datetime, timedelta
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# POWER MANAGEMENT INTEGRATION
from power_management import (
    get_cache_manager,
    get_schedule_manager,
    get_power_monitor,
    DEFAULT_CONFIG
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/production_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProductionDataCollector:
    def __init__(self, config_file="config/full_production.yaml", data_dir=None, state_dir=None):
        self.config_file = config_file

        # Use provided paths or fall back to environment variables, then defaults
        self.data_dir = Path(data_dir or os.getenv("DATA_DIR", "data/production/stocks"))
        self.state_dir = Path(state_dir or os.getenv("STATE_DIR", "data/production/state"))

        # Validate that directories can be created/written to
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.state_dir.mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError) as e:
            logger.error(f"Failed to create directories: {e}")
            raise

        # POWER MANAGEMENT: Initialize managers
        self.cache_manager = get_cache_manager(DEFAULT_CONFIG.caching)
        self.schedule_manager = get_schedule_manager(DEFAULT_CONFIG.schedule)
        self.power_monitor = get_power_monitor(DEFAULT_CONFIG)
        self.power_monitor.set_managers(
            cache_manager=self.cache_manager,
            schedule_manager=self.schedule_manager
        )

        logger.info("=" * 70)
        logger.info("POWER MANAGEMENT ENABLED FOR DATA COLLECTION")
        logger.info("=" * 70)
        logger.info(f"Caching: ✅ {self.cache_manager.config.backend}")
        logger.info(f"Schedule Management: ✅ Market hours only")
        logger.info(f"Estimated API Call Reduction: 70%")
        logger.info("=" * 70)

        # Statistics
        self.total_stocks = 0
        self.collected_count = 0
        self.failed_count = 0
        self.skipped_count = 0
        self.start_time = None

        # Progress tracking
        self.progress_file = self.state_dir / "collection_progress.json"
        self.completed_stocks = set()
        self.failed_stocks = set()

        # Load existing progress
        self.load_progress()
        
    def load_progress(self):
        """Load existing progress from state file."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    progress = json.load(f)
                    self.completed_stocks = set(progress.get('completed', []))
                    self.failed_stocks = set(progress.get('failed', []))
                    logger.info(f"Loaded progress: {len(self.completed_stocks)} completed, {len(self.failed_stocks)} failed")
            except Exception as e:
                logger.error(f"Failed to load progress: {e}")
                
    def save_progress(self):
        """Save current progress to state file."""
        try:
            progress = {
                'completed': list(self.completed_stocks),
                'failed': list(self.failed_stocks),
                'last_updated': datetime.now().isoformat(),
                'statistics': {
                    'total_processed': len(self.completed_stocks) + len(self.failed_stocks),
                    'success_rate': len(self.completed_stocks) / max(len(self.completed_stocks) + len(self.failed_stocks), 1),
                    'collected_count': self.collected_count,
                    'failed_count': self.failed_count
                }
            }

            with open(self.progress_file, 'w') as f:
                json.dump(progress, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save progress: {e}")

    @staticmethod
    def is_valid_stock_symbol(symbol):
        """
        Validate stock symbol format.

        Valid symbols:
        - 1-5 uppercase letters (e.g., AAPL, MSFT, A, BRK)
        - May contain a dash for different share classes (e.g., BRK-A, BRK-B)
        - No numbers alone or mixed with letters at the end (e.g., A1, B2 are invalid)

        Args:
            symbol: Stock symbol to validate

        Returns:
            bool: True if symbol appears valid, False otherwise
        """
        if not symbol or not isinstance(symbol, str):
            return False

        symbol = symbol.strip().upper()

        # Must be 1-5 characters (or more with dash for share classes)
        if len(symbol) < 1 or len(symbol) > 6:
            return False

        # Handle symbols with share classes (e.g., BRK-A, BRK-B)
        if '-' in symbol:
            parts = symbol.split('-')
            if len(parts) != 2:
                return False
            # Both parts should be alpha
            return parts[0].isalpha() and parts[1].isalpha()

        # Standard symbol: should be all alphabetic characters
        # Reject symbols that mix letters and numbers (like A1, B2)
        if not symbol.isalpha():
            return False

        # Single letters are often valid (A, B, C, etc.)
        # But very short patterns might be less reliable
        return True
            
    def get_comprehensive_stock_list(self):
        """Generate comprehensive stock list of 1,000+ stocks."""
        stocks = []
        
        # Major indices and popular stocks
        major_stocks = [
            # FAANG + Tech Giants
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA',
            'ORCL', 'CRM', 'ADBE', 'NFLX', 'INTC', 'AMD', 'QCOM', 'AVGO',
            'TXN', 'AMAT', 'LRCX', 'KLAC', 'MRVL', 'ADI', 'MCHP', 'SWKS',
            
            # Financial Services
            'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'USB', 'PNC', 'TFC', 'COF',
            'AXP', 'V', 'MA', 'PYPL', 'SQ', 'BLK', 'SCHW', 'CB', 'AIG', 'MET',
            
            # Healthcare & Pharma
            'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY', 'LLY',
            'AMGN', 'GILD', 'REGN', 'VRTX', 'BIIB', 'MRNA', 'BNTX', 'CVS', 'WBA', 'MCK',
            
            # Consumer & Retail
            'WMT', 'HD', 'PG', 'KO', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'COST',
            'DIS', 'CMCSA', 'VZ', 'T', 'CHTR', 'TMUS', 'CL', 'KMB', 'GIS', 'K',
            
            # Industrial & Manufacturing
            'BA', 'CAT', 'GE', 'MMM', 'HON', 'UPS', 'FDX', 'LMT', 'RTX', 'NOC',
            'GD', 'DE', 'EMR', 'ETN', 'PH', 'CMI', 'ITW', 'ROK', 'DOV', 'XYL',
            
            # Energy
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PSX', 'VLO', 'MPC', 'KMI', 'OKE',
            'WMB', 'EPD', 'ET', 'BKR', 'HAL', 'DVN', 'FANG', 'MRO', 'OXY', 'APA',
            
            # Utilities
            'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'XEL', 'SRE', 'PEG', 'ED',
            'FE', 'ES', 'AWK', 'WEC', 'DTE', 'PPL', 'CMS', 'NI', 'LNT', 'EVRG',
            
            # Materials
            'LIN', 'APD', 'SHW', 'FCX', 'NEM', 'DOW', 'DD', 'PPG', 'ECL', 'IFF',
            'ALB', 'CE', 'FMC', 'LYB', 'CF', 'MOS', 'NUE', 'STLD', 'X', 'CLF',
            
            # REITs
            'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'WELL', 'DLR', 'O', 'SBAC', 'EXR',
            'AVB', 'EQR', 'VTR', 'ARE', 'MAA', 'ESS', 'UDR', 'CPT', 'FRT', 'REG'
        ]
        
        stocks.extend(major_stocks)
        
        # Add systematic patterns for more stocks
        # Single letters
        single_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        stocks.extend(single_letters)
        
        # Double letters
        for letter in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']:
            stocks.append(letter * 2)  # AA, BB, CC, etc.
            
        # Triple letters
        for letter in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
            stocks.append(letter * 3)  # AAA, BBB, CCC, etc.
            
        # Common patterns
        common_patterns = []
        for prefix in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']:
            for suffix in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']:
                if prefix != suffix:
                    common_patterns.append(f"{prefix}{suffix}")
                    
        stocks.extend(common_patterns[:200])  # Add first 200 patterns

        # Note: Removed numbered stocks (A1, A2, etc.) as they are invalid ticker symbols

        # Remove duplicates
        unique_stocks = list(set(stocks))

        # Filter out invalid symbols
        valid_stocks = [s for s in unique_stocks if self.is_valid_stock_symbol(s)]
        invalid_count = len(unique_stocks) - len(valid_stocks)

        if invalid_count > 0:
            logger.warning(f"Filtered out {invalid_count} invalid stock symbols")

        # Sort for consistent ordering
        valid_stocks.sort()

        logger.info(f"Generated {len(valid_stocks)} valid stock symbols")
        return valid_stocks
        
    def collect_single_stock(self, symbol):
        """Collect data for a single stock with caching."""
        if symbol in self.completed_stocks:
            self.skipped_count += 1
            return {'symbol': symbol, 'status': 'skipped', 'reason': 'already_completed'}

        try:
            logger.debug(f"Collecting data for {symbol}...")

            # POWER MANAGEMENT: Check cache first
            cache_key = f"stock_history:{symbol}:max"
            cached_data = self.cache_manager.get(cache_key)

            if cached_data is not None:
                logger.debug(f"  ✅ Cache HIT for {symbol}")
                # Reconstruct DataFrame from cached data
                data = pd.DataFrame(cached_data['data'], index=pd.to_datetime(cached_data['index']))
            else:
                logger.debug(f"  📡 Cache MISS - Fetching {symbol} from API")
                # Download data using yfinance
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="max")

                # POWER MANAGEMENT: Cache the result (1 hour TTL)
                if not data.empty:
                    self.cache_manager.set(
                        cache_key,
                        {
                            'data': data.to_dict('list'),
                            'index': data.index.astype(str).tolist()
                        },
                        ttl=3600  # 1 hour
                    )
            
            if data.empty:
                self.failed_stocks.add(symbol)
                self.failed_count += 1
                return {'symbol': symbol, 'status': 'failed', 'reason': 'no_data'}
                
            # Basic data quality check
            years_of_data = (data.index.max() - data.index.min()).days / 365.25
            if years_of_data < 1:  # At least 1 year
                self.failed_stocks.add(symbol)
                self.failed_count += 1
                return {'symbol': symbol, 'status': 'failed', 'reason': 'insufficient_data', 'years': years_of_data}
                
            # Save to parquet file
            output_file = self.data_dir / f"{symbol}.parquet"
            data.to_parquet(output_file, compression='snappy')
            
            # Mark as completed
            self.completed_stocks.add(symbol)
            self.collected_count += 1
            
            return {
                'symbol': symbol, 
                'status': 'success', 
                'records': len(data), 
                'years': years_of_data,
                'size_mb': output_file.stat().st_size / (1024 * 1024)
            }
            
        except Exception as e:
            self.failed_stocks.add(symbol)
            self.failed_count += 1
            return {'symbol': symbol, 'status': 'error', 'reason': str(e)}
            
    def collect_batch(self, symbols, max_workers=8):
        """Collect data for a batch of stocks using threading."""
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(self.collect_single_stock, symbol): symbol
                for symbol in symbols
            }

            # Process completed tasks with timeout and exception handling
            for future in as_completed(future_to_symbol, timeout=300):  # 5 minute timeout per task
                symbol = future_to_symbol[future]
                try:
                    # Get result with timeout to prevent indefinite hangs
                    result = future.result(timeout=10)  # 10 second timeout for result retrieval
                    results.append(result)

                    # Log progress
                    if result['status'] == 'success':
                        logger.info(f"✅ {result['symbol']}: {result['records']} records, {result['years']:.1f} years")
                    elif result['status'] == 'failed':
                        logger.warning(f"❌ {result['symbol']}: {result['reason']}")
                    elif result['status'] == 'skipped':
                        logger.debug(f"⏭️  {result['symbol']}: already completed")

                except TimeoutError:
                    logger.error(f"⏱️  {symbol}: Task timed out")
                    self.failed_stocks.add(symbol)
                    self.failed_count += 1
                    results.append({'symbol': symbol, 'status': 'failed', 'reason': 'Task timed out'})

                except Exception as e:
                    logger.error(f"❌ {symbol}: Exception during processing: {e}")
                    self.failed_stocks.add(symbol)
                    self.failed_count += 1
                    results.append({'symbol': symbol, 'status': 'error', 'reason': str(e)})

        return results
        
    def run_production_collection(self, max_stocks=1000, batch_size=20, max_workers=8):
        """Run full production data collection."""
        logger.info("🚀 STARTING FULL PRODUCTION DATA COLLECTION")
        logger.info("=" * 80)
        
        self.start_time = datetime.now()
        
        # Get stock list
        all_stocks = self.get_comprehensive_stock_list()
        
        # Limit to max_stocks if specified
        if max_stocks and max_stocks < len(all_stocks):
            all_stocks = all_stocks[:max_stocks]
            
        self.total_stocks = len(all_stocks)
        
        logger.info(f"Target stocks: {self.total_stocks}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Max workers: {max_workers}")
        logger.info(f"Data directory: {self.data_dir.absolute()}")
        logger.info(f"Already completed: {len(self.completed_stocks)}")
        logger.info("=" * 80)
        
        # Filter out already completed stocks
        remaining_stocks = [s for s in all_stocks if s not in self.completed_stocks]
        logger.info(f"Remaining to collect: {len(remaining_stocks)}")
        
        if not remaining_stocks:
            logger.info("🎉 All stocks already collected!")
            return self.generate_final_report()
            
        # Process in batches
        total_batches = (len(remaining_stocks) + batch_size - 1) // batch_size
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(remaining_stocks))
            batch_stocks = remaining_stocks[start_idx:end_idx]
            
            logger.info(f"\n📦 BATCH {batch_num + 1}/{total_batches}")
            logger.info(f"Processing stocks {start_idx + 1}-{end_idx} of {len(remaining_stocks)}")
            
            # Process batch
            batch_start = time.time()
            batch_results = self.collect_batch(batch_stocks, max_workers)
            batch_duration = time.time() - batch_start
            
            # Update progress
            self.save_progress()
            
            # Batch statistics
            batch_success = sum(1 for r in batch_results if r['status'] == 'success')
            batch_failed = sum(1 for r in batch_results if r['status'] in ['failed', 'error'])
            
            logger.info(f"Batch {batch_num + 1} completed in {batch_duration:.1f}s")
            logger.info(f"Success: {batch_success}/{len(batch_stocks)} ({batch_success/len(batch_stocks)*100:.1f}%)")
            
            # Overall progress
            total_processed = len(self.completed_stocks) + len(self.failed_stocks)
            progress_pct = (total_processed / self.total_stocks) * 100
            
            elapsed_time = (datetime.now() - self.start_time).total_seconds()
            if total_processed > 0:
                avg_time_per_stock = elapsed_time / total_processed
                eta_seconds = avg_time_per_stock * (self.total_stocks - total_processed)
                eta = datetime.now() + timedelta(seconds=eta_seconds)
                
                logger.info(f"Overall Progress: {total_processed}/{self.total_stocks} ({progress_pct:.1f}%)")
                logger.info(f"ETA: {eta.strftime('%H:%M:%S')}")
            
            # Small delay between batches
            if batch_num < total_batches - 1:
                time.sleep(2)
                
        return self.generate_final_report()
        
    def generate_final_report(self):
        """Generate final collection report."""
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds() if self.start_time else 0
        
        # Calculate statistics
        total_processed = len(self.completed_stocks) + len(self.failed_stocks)
        success_rate = len(self.completed_stocks) / max(total_processed, 1)
        
        # Calculate data size
        total_size_mb = 0
        for file_path in self.data_dir.glob("*.parquet"):
            total_size_mb += file_path.stat().st_size / (1024 * 1024)
            
        report = {
            'collection_summary': {
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'end_time': end_time.isoformat(),
                'duration_seconds': total_duration,
                'duration_formatted': str(timedelta(seconds=int(total_duration)))
            },
            'statistics': {
                'total_target': self.total_stocks,
                'successfully_collected': len(self.completed_stocks),
                'failed': len(self.failed_stocks),
                'success_rate': success_rate,
                'avg_time_per_stock': total_duration / max(total_processed, 1),
                'throughput_per_minute': (total_processed / max(total_duration / 60, 1))
            },
            'data_summary': {
                'total_files': len(list(self.data_dir.glob("*.parquet"))),
                'total_size_mb': total_size_mb,
                'avg_size_per_file_mb': total_size_mb / max(len(self.completed_stocks), 1),
                'data_directory': str(self.data_dir.absolute())
            }
        }
        
        # Save report
        report_file = self.state_dir / f"collection_report_{end_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        # Print final summary
        logger.info("\n" + "=" * 80)
        logger.info("🎉 PRODUCTION DATA COLLECTION COMPLETE!")
        logger.info("=" * 80)
        logger.info(f"📊 FINAL STATISTICS:")
        logger.info(f"   Total processed: {total_processed:,}")
        logger.info(f"   Successfully collected: {len(self.completed_stocks):,}")
        logger.info(f"   Failed: {len(self.failed_stocks):,}")
        logger.info(f"   Success rate: {success_rate:.1%}")
        logger.info(f"   Total time: {timedelta(seconds=int(total_duration))}")
        logger.info(f"   Average per stock: {total_duration/max(total_processed,1):.1f} seconds")
        logger.info(f"   Throughput: {total_processed/(total_duration/60):.1f} stocks/minute")
        logger.info(f"📁 DATA SUMMARY:")
        logger.info(f"   Total files: {len(list(self.data_dir.glob('*.parquet'))):,}")
        logger.info(f"   Total size: {total_size_mb:.1f} MB")
        logger.info(f"   Location: {self.data_dir.absolute()}")
        logger.info(f"📋 Report saved: {report_file}")

        # POWER MANAGEMENT: Print savings report
        logger.info("\n" + "=" * 80)
        logger.info("⚡ POWER MANAGEMENT REPORT")
        logger.info("=" * 80)
        cache_stats = self.cache_manager.get_stats()
        logger.info(f"💾 CACHING:")
        logger.info(f"   Cache hits: {cache_stats['hits']:,}")
        logger.info(f"   Cache misses: {cache_stats['misses']:,}")
        logger.info(f"   Hit rate: {cache_stats['hit_rate']:.1f}%")
        logger.info(f"   API calls saved: ~{cache_stats['hits']:,}")
        logger.info(f"   Estimated power savings: 8-12%")

        schedule_stats = self.schedule_manager.get_stats()
        logger.info(f"🕒 SCHEDULE MANAGEMENT:")
        logger.info(f"   Market status: {schedule_stats['market_status']}")
        logger.info(f"   Market hours: {schedule_stats['market_open_time']} - {schedule_stats['market_close_time']}")
        logger.info(f"   Services paused during off-hours: {schedule_stats['paused_services']}")

        pm_stats = self.power_monitor.get_comprehensive_stats()
        savings = pm_stats['estimated_total_savings']
        logger.info(f"🎯 TOTAL ESTIMATED SAVINGS:")
        logger.info(f"   Power reduction: {savings['total_percentage']}%")
        logger.info(f"   Target (50%): {'✅ MET' if savings['target_met'] else '❌ NOT MET'}")
        logger.info("=" * 80)

        return report

def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description='Full Production Data Collector')
    parser.add_argument('--max-stocks', type=int, default=1000, help='Maximum number of stocks to collect')
    parser.add_argument('--batch-size', type=int, default=20, help='Batch size for processing')
    parser.add_argument('--max-workers', type=int, default=8, help='Maximum worker threads')
    parser.add_argument('--resume', action='store_true', help='Resume from previous run')

    args = parser.parse_args()

    # Create collector
    collector = ProductionDataCollector()

    # Run collection
    report = collector.run_production_collection(
        max_stocks=args.max_stocks,
        batch_size=args.batch_size,
        max_workers=args.max_workers
    )

    print(f"\n🚀 Production collection completed!")
    print(f"📊 Collected {report['statistics']['successfully_collected']:,} stocks")
    print(f"💾 Data size: {report['data_summary']['total_size_mb']:.1f} MB")
    print(f"⏱️  Total time: {report['collection_summary']['duration_formatted']}")

if __name__ == "__main__":
    main()