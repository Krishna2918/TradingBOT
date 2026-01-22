"""
Automated Daily Data Refresh Script
Runs in background to keep all stock data up-to-date

This script:
1. Updates all existing stocks with latest data (just 1 day of new data per stock)
2. Runs silently in background
3. Logs all activity
4. Can be scheduled to run automatically every night at 1 AM

Usage:
    # Run once manually:
    python automated_data_refresh.py

    # Or schedule with Windows Task Scheduler to run daily at 1 AM
"""

import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pandas as pd
import requests

# Load environment
load_dotenv()

# Setup logging
log_dir = Path('logs/data_refresh')
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f"refresh_{datetime.now().strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # Also print to console
    ]
)

logger = logging.getLogger(__name__)

# API Configuration
PREMIUM_KEY = os.getenv('AV_PREMIUM_KEY')
DATA_DIR = Path('TrainingData/daily')


def get_latest_date(parquet_file):
    """Get the most recent date in a parquet file"""
    try:
        df = pd.read_parquet(parquet_file)
        if 'date' in df.columns:
            return pd.to_datetime(df['date']).max().date()
        else:
            return pd.to_datetime(df.index).max().date()
    except Exception as e:
        logger.error(f"Error reading {parquet_file.name}: {e}")
        return None


def fetch_latest_data(symbol, api_key):
    """Fetch latest daily data from Alpha Vantage"""
    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': symbol,
        'apikey': api_key,
        'outputsize': 'compact',  # Only last 100 days
        'datatype': 'json'
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        data = response.json()

        if 'Time Series (Daily)' not in data:
            return None, f"No data: {list(data.keys())}"

        time_series = data['Time Series (Daily)']
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        df.columns = ['open', 'high', 'low', 'close', 'volume']

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        return df, None

    except Exception as e:
        return None, str(e)


def update_stock(symbol, parquet_file, api_key):
    """Update a single stock with latest data"""
    try:
        # Read existing data
        existing_df = pd.read_parquet(parquet_file)

        # Get latest date
        if 'date' in existing_df.columns:
            latest_date = pd.to_datetime(existing_df['date']).max().date()
            existing_df['date'] = pd.to_datetime(existing_df['date'])
            existing_df.set_index('date', inplace=True)
        else:
            latest_date = pd.to_datetime(existing_df.index).max().date()
            existing_df.index = pd.to_datetime(existing_df.index)

        # Check if update needed (if data is older than 1 day)
        today = datetime.now().date()
        if latest_date >= today - timedelta(days=1):
            return 'up-to-date', f"Latest: {latest_date}"

        # Fetch new data
        new_df, error = fetch_latest_data(symbol, api_key)

        if error:
            return 'failed', error

        # Filter only new rows
        new_rows = new_df[new_df.index > pd.Timestamp(latest_date)]

        if len(new_rows) == 0:
            return 'no-new-data', f"Latest: {latest_date}"

        # Append new data
        updated_df = pd.concat([existing_df, new_rows])
        updated_df = updated_df[~updated_df.index.duplicated(keep='last')]
        updated_df.sort_index(inplace=True)

        # Save back to parquet
        updated_df.to_parquet(parquet_file, compression='snappy')

        return 'updated', f"Added {len(new_rows)} days ({new_rows.index.min().date()} to {new_rows.index.max().date()})"

    except Exception as e:
        return 'error', str(e)


def main():
    """Main refresh function"""
    logger.info("=" * 80)
    logger.info("AUTOMATED DATA REFRESH STARTED")
    logger.info("=" * 80)

    if not PREMIUM_KEY:
        logger.error("No AV_PREMIUM_KEY found in .env file!")
        sys.exit(1)

    # Get all parquet files
    parquet_files = list(DATA_DIR.glob('*.parquet'))
    total = len(parquet_files)

    logger.info(f"Found {total} stocks to check")
    logger.info(f"Using API key: {PREMIUM_KEY[:10]}...")
    logger.info("")

    stats = {
        'updated': 0,
        'up-to-date': 0,
        'no-new-data': 0,
        'failed': 0,
        'error': 0
    }

    start_time = time.time()

    # Process each stock
    for i, file in enumerate(parquet_files, 1):
        symbol = file.stem.replace('_daily', '')

        # Rate limiting (70 requests/min for premium)
        if i > 1:
            time.sleep(0.9)  # ~67 req/min to be safe

        status, message = update_stock(symbol, file, PREMIUM_KEY)
        stats[status] = stats.get(status, 0) + 1

        # Log progress
        if status == 'updated':
            logger.info(f"[{i}/{total}] {symbol}: UPDATED - {message}")
        elif status in ['failed', 'error']:
            logger.warning(f"[{i}/{total}] {symbol}: {status.upper()} - {message}")
        # Skip logging for up-to-date stocks (too verbose)

        # Progress update every 50 stocks
        if i % 50 == 0:
            elapsed = time.time() - start_time
            rate = i / elapsed * 60
            remaining = total - i
            eta_min = remaining / rate if rate > 0 else 0

            logger.info("")
            logger.info(f"Progress: {i}/{total} ({i/total*100:.1f}%)")
            logger.info(f"Stats: Updated={stats['updated']}, Up-to-date={stats['up-to-date']}, Failed={stats['failed']}")
            logger.info(f"Rate: {rate:.1f} stocks/min, ETA: {eta_min:.1f} minutes")
            logger.info("")

    # Final summary
    elapsed = time.time() - start_time
    logger.info("")
    logger.info("=" * 80)
    logger.info("REFRESH COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total stocks: {total}")
    logger.info(f"Updated: {stats['updated']}")
    logger.info(f"Already up-to-date: {stats['up-to-date']}")
    logger.info(f"No new data available: {stats.get('no-new-data', 0)}")
    logger.info(f"Failed: {stats['failed']}")
    logger.info(f"Errors: {stats.get('error', 0)}")
    logger.info(f"Time elapsed: {elapsed/60:.1f} minutes")
    logger.info(f"Log file: {log_file}")
    logger.info("")

    if stats['updated'] > 0:
        logger.info(f"SUCCESS: {stats['updated']} stocks updated with latest data!")
    elif stats['up-to-date'] == total:
        logger.info("INFO: All stocks already up-to-date!")
    else:
        logger.info("PARTIAL: Some stocks updated, check log for details")

    logger.info("=" * 80)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nRefresh interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
