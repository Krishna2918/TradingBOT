#!/usr/bin/env python3
"""
Simple demo script to start collecting stock data using yfinance.
This bypasses the complex configuration and focuses on core data collection.
"""

import asyncio
import yfinance as yf
import pandas as pd
import os
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Demo stock list
DEMO_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", 
    "META", "NVDA", "JPM", "JNJ", "V",
    "WMT", "PG", "UNH", "HD", "MA",
    "DIS", "ADBE", "NFLX", "CRM", "PYPL"
]

class SimpleDataCollector:
    def __init__(self, data_dir="data/demo/stocks"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.collected_count = 0
        self.failed_count = 0
        
    def collect_stock_data(self, symbol):
        """Collect data for a single stock."""
        try:
            logger.info(f"Collecting data for {symbol}...")
            
            # Download data using yfinance
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="max")
            
            if data.empty:
                logger.warning(f"No data available for {symbol}")
                self.failed_count += 1
                return False
                
            # Basic data quality check
            years_of_data = (data.index.max() - data.index.min()).days / 365.25
            if years_of_data < 2:
                logger.warning(f"Insufficient data for {symbol}: {years_of_data:.1f} years")
                self.failed_count += 1
                return False
                
            # Save to parquet file
            output_file = self.data_dir / f"{symbol}.parquet"
            data.to_parquet(output_file, compression='snappy')
            
            logger.info(f"✅ {symbol}: {len(data)} records, {years_of_data:.1f} years of data")
            self.collected_count += 1
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to collect {symbol}: {str(e)}")
            self.failed_count += 1
            return False
            
    async def collect_all_stocks(self, stocks):
        """Collect data for all stocks."""
        logger.info(f"Starting data collection for {len(stocks)} stocks...")
        logger.info(f"Data will be saved to: {self.data_dir.absolute()}")
        
        start_time = datetime.now()
        
        for i, symbol in enumerate(stocks, 1):
            logger.info(f"Progress: {i}/{len(stocks)} ({i/len(stocks)*100:.1f}%)")
            self.collect_stock_data(symbol)
            
            # Small delay to be respectful to the API
            await asyncio.sleep(0.5)
            
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info("="*60)
        logger.info("DATA COLLECTION COMPLETE!")
        logger.info("="*60)
        logger.info(f"Total stocks processed: {len(stocks)}")
        logger.info(f"Successfully collected: {self.collected_count}")
        logger.info(f"Failed: {self.failed_count}")
        logger.info(f"Success rate: {self.collected_count/len(stocks)*100:.1f}%")
        logger.info(f"Total time: {duration:.1f} seconds")
        logger.info(f"Average time per stock: {duration/len(stocks):.1f} seconds")
        logger.info(f"Data saved to: {self.data_dir.absolute()}")
        
        return {
            'total': len(stocks),
            'collected': self.collected_count,
            'failed': self.failed_count,
            'success_rate': self.collected_count/len(stocks),
            'duration': duration
        }

async def main():
    """Main function."""
    print("🚀 STARTING DEMO DATA COLLECTION")
    print("="*60)
    print(f"Target stocks: {len(DEMO_STOCKS)}")
    print(f"Data source: yfinance (Yahoo Finance)")
    print(f"Storage format: Parquet with Snappy compression")
    print("="*60)
    
    collector = SimpleDataCollector()
    results = await collector.collect_all_stocks(DEMO_STOCKS)
    
    # Show some sample data
    if results['collected'] > 0:
        print("\n📊 SAMPLE DATA PREVIEW:")
        print("-" * 40)
        
        # Show data for first successful stock
        for stock_file in collector.data_dir.glob("*.parquet"):
            df = pd.read_parquet(stock_file)
            symbol = stock_file.stem
            
            print(f"\n{symbol} ({len(df)} records):")
            print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
            print(f"Latest close: ${df['Close'].iloc[-1]:.2f}")
            print(f"Columns: {list(df.columns)}")
            print("\nLatest 3 days:")
            print(df.tail(3)[['Open', 'High', 'Low', 'Close', 'Volume']])
            break
            
    print(f"\n🎉 Demo complete! Check the data in: {collector.data_dir.absolute()}")

if __name__ == "__main__":
    asyncio.run(main())