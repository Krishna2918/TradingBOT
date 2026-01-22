#!/usr/bin/env python3
"""
Fixed Production Data Collector - Clean Output, No Unicode Errors
This version fixes all Unicode display issues and provides clean terminal output.
"""

import asyncio
import yfinance as yf
import pandas as pd
import os
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Fix Unicode issues for Windows
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

# Setup logging with UTF-8 encoding
class UTF8FileHandler(logging.FileHandler):
    def __init__(self, filename, mode='a', encoding='utf-8', delay=False):
        super().__init__(filename, mode, encoding, delay)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        UTF8FileHandler('logs/production_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProductionDataCollectorFixed:
    """
    Fixed production data collector with clean output and better success rates.
    """
    
    def __init__(self, config_file="config/full_production.yaml"):
        self.config_file = config_file
        self.data_dir = Path("data/production/stocks")
        self.state_dir = Path("data/production/state")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
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
                with open(self.progress_file, 'r', encoding='utf-8') as f:
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
            
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Failed to save progress: {e}")
            
    def get_high_quality_stock_list(self):
        """Generate high-quality stock list with better success rates."""
        # Focus on real, well-known stocks for higher success rate
        stocks = []
        
        # Major S&P 500 and popular stocks (high success rate expected)
        major_stocks = [
            # FAANG + Tech Giants
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA',
            'ORCL', 'CRM', 'ADBE', 'NFLX', 'INTC', 'AMD', 'QCOM', 'AVGO',
            'TXN', 'AMAT', 'LRCX', 'KLAC', 'MRVL', 'ADI', 'MCHP', 'SWKS',
            'CSCO', 'IBM', 'INTU', 'NOW', 'SNOW', 'PLTR', 'TWLO', 'OKTA',
            
            # Financial Services
            'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'USB', 'PNC', 'TFC', 'COF',
            'AXP', 'V', 'MA', 'PYPL', 'SQ', 'BLK', 'SCHW', 'CB', 'AIG', 'MET',
            'ALL', 'TRV', 'PGR', 'AON', 'MMC', 'SPGI', 'MCO', 'ICE', 'CME', 'NDAQ',
            
            # Healthcare & Pharma
            'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY', 'LLY',
            'AMGN', 'GILD', 'REGN', 'VRTX', 'BIIB', 'MRNA', 'BNTX', 'CVS', 'WBA', 'MCK',
            'CI', 'HUM', 'ANTM', 'CNC', 'MOH', 'ELV', 'CVH', 'HCA', 'UHS', 'THC',
            
            # Consumer & Retail
            'WMT', 'HD', 'PG', 'KO', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'COST',
            'DIS', 'CMCSA', 'VZ', 'T', 'CHTR', 'TMUS', 'CL', 'KMB', 'GIS', 'K',
            'PEP', 'MDLZ', 'KHC', 'HSY', 'CPB', 'SJM', 'CAG', 'TSN', 'HRL', 'MKC',
            
            # Industrial & Manufacturing
            'BA', 'CAT', 'GE', 'MMM', 'HON', 'UPS', 'FDX', 'LMT', 'RTX', 'NOC',
            'GD', 'DE', 'EMR', 'ETN', 'PH', 'CMI', 'ITW', 'ROK', 'DOV', 'XYL',
            'IR', 'OTIS', 'CARR', 'PWR', 'FLR', 'JCI', 'PCAR', 'WAB', 'NSC', 'UNP',
            
            # Energy
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PSX', 'VLO', 'MPC', 'KMI', 'OKE',
            'WMB', 'EPD', 'ET', 'BKR', 'HAL', 'DVN', 'FANG', 'MRO', 'OXY', 'APA',
            'HES', 'PXD', 'CXO', 'EQT', 'KNTK', 'AR', 'SM', 'NOV', 'FTI', 'RIG',
            
            # Utilities
            'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'XEL', 'SRE', 'PEG', 'ED',
            'FE', 'ES', 'AWK', 'WEC', 'DTE', 'PPL', 'CMS', 'NI', 'LNT', 'EVRG',
            'ATO', 'CNP', 'ETR', 'AES', 'NRG', 'VST', 'CEG', 'PCG', 'EIX', 'PNW',
            
            # Materials
            'LIN', 'APD', 'SHW', 'FCX', 'NEM', 'DOW', 'DD', 'PPG', 'ECL', 'IFF',
            'ALB', 'CE', 'FMC', 'LYB', 'CF', 'MOS', 'NUE', 'STLD', 'X', 'CLF',
            'VMC', 'MLM', 'EMN', 'RPM', 'SEE', 'AVY', 'BLL', 'CCK', 'PKG', 'IP',
            
            # REITs (Real Estate)
            'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'WELL', 'DLR', 'O', 'SBAC', 'EXR',
            'AVB', 'EQR', 'VTR', 'ARE', 'MAA', 'ESS', 'UDR', 'CPT', 'FRT', 'REG',
            'BXP', 'VNO', 'KIM', 'SPG', 'SLG', 'HST', 'RHP', 'ELS', 'UMH', 'SUI',
            
            # Communication Services
            'NFLX', 'DIS', 'CMCSA', 'VZ', 'T', 'CHTR', 'TMUS', 'DISH', 'SIRI', 'LBRDA',
            'LBRDK', 'FWONA', 'FWONK', 'BATRK', 'BATRA', 'WBD', 'PARA', 'FOX', 'FOXA', 'NYT'
        ]
        
        stocks.extend(major_stocks)
        
        # Add some systematic high-probability stocks (avoid single letters that often fail)
        # Focus on 2-4 letter combinations that are more likely to be real stocks
        high_prob_patterns = [
            'AA', 'AAL', 'AAP', 'AAON', 'AAWW', 'AAXJ', 'AB', 'ABB', 'ABBV', 'ABC',
            'ABCB', 'ABCL', 'ABCM', 'ABEO', 'ABG', 'ABIO', 'ABM', 'ABNB', 'ABOS', 'ABR',
            'ABSI', 'ABT', 'ABUS', 'ABVC', 'AC', 'ACA', 'ACAD', 'ACB', 'ACCD', 'ACCO',
            'ACEL', 'ACER', 'ACES', 'ACET', 'ACGL', 'ACH', 'ACHC', 'ACHL', 'ACHR', 'ACHV',
            'ACI', 'ACIC', 'ACIU', 'ACIW', 'ACLS', 'ACM', 'ACMR', 'ACN', 'ACNB', 'ACON'
        ]
        
        stocks.extend(high_prob_patterns)
        
        # Remove duplicates and sort
        unique_stocks = sorted(list(set(stocks)))
        
        logger.info(f"Generated {len(unique_stocks)} high-quality stock symbols")
        return unique_stocks
        
    def collect_single_stock(self, symbol):
        """Collect data for a single stock with better error handling."""
        if symbol in self.completed_stocks:
            self.skipped_count += 1
            return {'symbol': symbol, 'status': 'skipped', 'reason': 'already_completed'}
            
        try:
            # Download data using yfinance with timeout
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="max", timeout=15)  # 15 second timeout
            
            if data.empty:
                self.failed_stocks.add(symbol)
                self.failed_count += 1
                return {'symbol': symbol, 'status': 'failed', 'reason': 'no_data'}
                
            # More lenient data quality check for better success rate
            years_of_data = (data.index.max() - data.index.min()).days / 365.25
            if years_of_data < 0.5:  # At least 6 months (more lenient)
                self.failed_stocks.add(symbol)
                self.failed_count += 1
                return {'symbol': symbol, 'status': 'failed', 'reason': 'insufficient_data', 'years': years_of_data}
                
            # Check for reasonable data (not all zeros or NaN)
            if data['Close'].isna().all() or (data['Close'] == 0).all():
                self.failed_stocks.add(symbol)
                self.failed_count += 1
                return {'symbol': symbol, 'status': 'failed', 'reason': 'invalid_data'}
                
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
            return {'symbol': symbol, 'status': 'error', 'reason': str(e)[:100]}  # Truncate long errors
            
    def collect_batch(self, symbols, max_workers=8):
        """Collect data for a batch of stocks using threading."""
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(self.collect_single_stock, symbol): symbol 
                for symbol in symbols
            }
            
            # Process completed tasks
            for future in as_completed(future_to_symbol):
                result = future.result()
                results.append(result)
                
                # Clean log output (no Unicode emojis)
                if result['status'] == 'success':
                    logger.info(f"SUCCESS {result['symbol']}: {result['records']} records, {result['years']:.1f} years")
                elif result['status'] == 'failed':
                    logger.warning(f"FAILED {result['symbol']}: {result['reason']}")
                elif result['status'] == 'error':
                    logger.error(f"ERROR {result['symbol']}: {result['reason']}")
                    
        return results
        
    def run_production_collection(self, max_stocks=500, batch_size=25, max_workers=8):
        """Run production data collection with clean output."""
        print("=" * 80)
        print("PRODUCTION DATA COLLECTION SYSTEM - STARTING")
        print("=" * 80)
        
        self.start_time = datetime.now()
        
        # Get high-quality stock list
        all_stocks = self.get_high_quality_stock_list()
        
        # Limit to max_stocks if specified
        if max_stocks and max_stocks < len(all_stocks):
            all_stocks = all_stocks[:max_stocks]
            
        self.total_stocks = len(all_stocks)
        
        print(f"Target stocks: {self.total_stocks}")
        print(f"Batch size: {batch_size}")
        print(f"Max workers: {max_workers}")
        print(f"Data directory: {self.data_dir.absolute()}")
        print(f"Already completed: {len(self.completed_stocks)}")
        print("=" * 80)
        
        # Filter out already completed stocks
        remaining_stocks = [s for s in all_stocks if s not in self.completed_stocks]
        print(f"Remaining to collect: {len(remaining_stocks)}")
        
        if not remaining_stocks:
            print("All stocks already collected!")
            return self.generate_final_report()
            
        # Process in batches
        total_batches = (len(remaining_stocks) + batch_size - 1) // batch_size
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(remaining_stocks))
            batch_stocks = remaining_stocks[start_idx:end_idx]
            
            print(f"\nBATCH {batch_num + 1}/{total_batches}")
            print(f"Processing stocks {start_idx + 1}-{end_idx} of {len(remaining_stocks)}")
            
            # Process batch
            batch_start = time.time()
            batch_results = self.collect_batch(batch_stocks, max_workers)
            batch_duration = time.time() - batch_start
            
            # Update progress
            self.save_progress()
            
            # Batch statistics
            batch_success = sum(1 for r in batch_results if r['status'] == 'success')
            batch_failed = sum(1 for r in batch_results if r['status'] in ['failed', 'error'])
            
            print(f"Batch {batch_num + 1} completed in {batch_duration:.1f}s")
            print(f"Success: {batch_success}/{len(batch_stocks)} ({batch_success/len(batch_stocks)*100:.1f}%)")
            
            # Overall progress
            total_processed = len(self.completed_stocks) + len(self.failed_stocks)
            progress_pct = (total_processed / self.total_stocks) * 100
            
            elapsed_time = (datetime.now() - self.start_time).total_seconds()
            if total_processed > 0:
                avg_time_per_stock = elapsed_time / total_processed
                eta_seconds = avg_time_per_stock * (self.total_stocks - total_processed)
                eta = datetime.now() + timedelta(seconds=eta_seconds)
                
                print(f"Overall Progress: {total_processed}/{self.total_stocks} ({progress_pct:.1f}%)")
                print(f"ETA: {eta.strftime('%H:%M:%S')}")
            
            # Small delay between batches
            if batch_num < total_batches - 1:
                time.sleep(2)
                
        return self.generate_final_report()
        
    def generate_final_report(self):
        """Generate final collection report with clean output."""
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
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        # Print final summary (clean output)
        print("\n" + "=" * 80)
        print("PRODUCTION DATA COLLECTION COMPLETE!")
        print("=" * 80)
        print(f"FINAL STATISTICS:")
        print(f"   Total processed: {total_processed:,}")
        print(f"   Successfully collected: {len(self.completed_stocks):,}")
        print(f"   Failed: {len(self.failed_stocks):,}")
        print(f"   Success rate: {success_rate:.1%}")
        print(f"   Total time: {timedelta(seconds=int(total_duration))}")
        print(f"   Average per stock: {total_duration/max(total_processed,1):.1f} seconds")
        print(f"   Throughput: {total_processed/(total_duration/60):.1f} stocks/minute")
        print(f"DATA SUMMARY:")
        print(f"   Total files: {len(list(self.data_dir.glob('*.parquet'))):,}")
        print(f"   Total size: {total_size_mb:.1f} MB")
        print(f"   Location: {self.data_dir.absolute()}")
        print(f"Report saved: {report_file}")
        print("=" * 80)
        
        return report

async def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fixed Production Data Collector')
    parser.add_argument('--max-stocks', type=int, default=500, help='Maximum number of stocks to collect')
    parser.add_argument('--batch-size', type=int, default=25, help='Batch size for processing')
    parser.add_argument('--max-workers', type=int, default=8, help='Maximum worker threads')
    parser.add_argument('--resume', action='store_true', help='Resume from previous run')
    
    args = parser.parse_args()
    
    # Create collector
    collector = ProductionDataCollectorFixed()
    
    # Run collection
    report = collector.run_production_collection(
        max_stocks=args.max_stocks,
        batch_size=args.batch_size,
        max_workers=args.max_workers
    )
    
    print(f"\nProduction collection completed!")
    print(f"Collected {report['statistics']['successfully_collected']:,} stocks")
    print(f"Data size: {report['data_summary']['total_size_mb']:.1f} MB")
    print(f"Total time: {report['collection_summary']['duration_formatted']}")

if __name__ == "__main__":
    asyncio.run(main())