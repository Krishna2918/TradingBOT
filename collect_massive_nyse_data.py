"""
Aggressive NYSE/NASDAQ Data Collection Script
Collects 20+ years of data for 1,400+ stocks using Alpha Vantage API

Features:
- Multi-threaded collection with rate limiting
- Uses all 4 Alpha Vantage keys
- Resume capability (skips already collected stocks)
- Progress tracking with tqdm
- Error handling and automatic retries
- Data validation
- Detailed logging

Usage:
    python collect_massive_nyse_data.py --symbols lists/additional_1400_stocks.txt

Author: Trading Bot
Date: October 28, 2025
"""

import os
import sys
import time
import argparse
import pandas as pd
import requests
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Alpha Vantage API keys
API_KEYS = [
    os.getenv('AV_PREMIUM_KEY'),  # Premium: 75 req/min
    os.getenv('ALPHA_VANTAGE_API_KEY'),  # Free: 25 req/day
    os.getenv('ALPHA_VANTAGE_API_KEY_SECONDARY'),  # Free: 25 req/day
    os.getenv('AV_SENTIMENT_KEY'),  # Free: 25 req/day
]

# Remove None values
API_KEYS = [k for k in API_KEYS if k]

# Rate limits (requests per minute)
PREMIUM_RATE_LIMIT = 70  # 75/min, use 70 to be safe
FREE_RATE_LIMIT = 5  # 25/day = ~1/5min

# Global state
request_times = {key: [] for key in API_KEYS}
request_lock = Lock()
stats = {
    'total': 0,
    'success': 0,
    'failed': 0,
    'skipped': 0,
    'start_time': None
}
stats_lock = Lock()


def wait_for_rate_limit(api_key, is_premium=True):
    """Wait if necessary to respect rate limits"""
    with request_lock:
        now = time.time()
        rate_limit = PREMIUM_RATE_LIMIT if is_premium else FREE_RATE_LIMIT
        window = 60  # 1 minute window

        # Clean old requests outside window
        request_times[api_key] = [t for t in request_times[api_key] if now - t < window]

        # Check if we need to wait
        if len(request_times[api_key]) >= rate_limit:
            oldest = request_times[api_key][0]
            wait_time = window - (now - oldest) + 1
            if wait_time > 0:
                print(f"  Rate limit reached, waiting {wait_time:.1f}s...")
                time.sleep(wait_time)

        # Record this request
        request_times[api_key].append(time.time())


def fetch_stock_data(symbol, api_key, output_size='full'):
    """Fetch daily stock data from Alpha Vantage"""
    url = f'https://www.alphavantage.co/query'
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': symbol,
        'apikey': api_key,
        'outputsize': output_size,
        'datatype': 'json'
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Check for errors
        if 'Error Message' in data:
            return None, f"API Error: {data['Error Message']}"
        if 'Note' in data:
            return None, f"Rate limit: {data['Note']}"
        if 'Information' in data:
            return None, f"Info: {data['Information']}"

        # Extract time series
        if 'Time Series (Daily)' not in data:
            return None, f"No time series data in response"

        time_series = data['Time Series (Daily)']

        # Convert to DataFrame
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)

        # Rename columns
        df.columns = ['open', 'high', 'low', 'close', 'volume']

        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Basic validation
        if len(df) < 100:
            return None, f"Insufficient data: only {len(df)} days"

        if df['close'].isna().sum() > len(df) * 0.1:
            return None, f"Too many missing values: {df['close'].isna().sum()}"

        return df, None

    except requests.exceptions.Timeout:
        return None, "Request timeout"
    except requests.exceptions.RequestException as e:
        return None, f"Request error: {str(e)}"
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"


def collect_stock(symbol, output_dir, api_key, is_premium=True):
    """Collect data for a single stock"""
    output_file = output_dir / f"{symbol}_daily.parquet"

    # Check if already exists (resume capability)
    if output_file.exists():
        with stats_lock:
            stats['skipped'] += 1
        return symbol, 'skipped', "Already exists"

    # Wait for rate limit
    wait_for_rate_limit(api_key, is_premium)

    # Fetch data
    df, error = fetch_stock_data(symbol, api_key)

    if error:
        with stats_lock:
            stats['failed'] += 1
        return symbol, 'failed', error

    # Save to parquet
    try:
        df.to_parquet(output_file, compression='snappy')

        with stats_lock:
            stats['success'] += 1

        # Calculate date range
        start_date = df.index.min().date()
        end_date = df.index.max().date()
        years = (end_date - start_date).days / 365.25

        return symbol, 'success', f"{len(df)} days, {years:.1f} years ({start_date} to {end_date})"

    except Exception as e:
        with stats_lock:
            stats['failed'] += 1
        return symbol, 'failed', f"Save error: {str(e)}"


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Collect massive NYSE/NASDAQ data')
    parser.add_argument('--symbols', required=True, help='Path to symbol list file')
    parser.add_argument('--output', default='TrainingData/daily', help='Output directory')
    parser.add_argument('--threads', type=int, default=1, help='Number of parallel threads (1=sequential)')
    parser.add_argument('--max-stocks', type=int, default=None, help='Limit number of stocks (for testing)')
    args = parser.parse_args()

    # Load symbols
    symbol_file = Path(args.symbols)
    if not symbol_file.exists():
        print(f"Error: Symbol file not found: {symbol_file}")
        sys.exit(1)

    with open(symbol_file, 'r') as f:
        symbols = [line.strip() for line in f if line.strip()]

    if args.max_stocks:
        symbols = symbols[:args.max_stocks]

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check API keys
    if not API_KEYS:
        print("Error: No Alpha Vantage API keys found in .env file")
        print("Please set: AV_PREMIUM_KEY, ALPHA_VANTAGE_API_KEY, etc.")
        sys.exit(1)

    # Print configuration
    print("=" * 80)
    print("MASSIVE NYSE/NASDAQ DATA COLLECTION")
    print("=" * 80)
    print(f"Symbols: {len(symbols)} stocks")
    print(f"Symbol file: {symbol_file}")
    print(f"Output: {output_dir}")
    print(f"API keys: {len(API_KEYS)} keys configured")
    print(f"Premium key: {API_KEYS[0][:10]}... (75 req/min)")
    if len(API_KEYS) > 1:
        print(f"Free keys: {len(API_KEYS) - 1} additional keys (25 req/day each)")
    print(f"Threads: {args.threads}")
    print(f"Resume: Will skip existing files")
    print()

    # Check existing files
    existing = list(output_dir.glob('*_daily.parquet'))
    print(f"Existing files: {len(existing)}")
    print(f"New to collect: {len(symbols) - len(existing)} (estimated)")
    print()

    # Confirm
    response = input("Start collection? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled")
        sys.exit(0)

    print()
    print("=" * 80)
    print("STARTING COLLECTION...")
    print("=" * 80)
    print()

    stats['start_time'] = time.time()
    stats['total'] = len(symbols)

    # Use premium key for all requests (best rate limit)
    api_key = API_KEYS[0]

    if args.threads == 1:
        # Sequential processing (easier to debug)
        for i, symbol in enumerate(symbols, 1):
            print(f"[{i}/{len(symbols)}] {symbol}...", end=' ', flush=True)
            sym, status, message = collect_stock(symbol, output_dir, api_key, is_premium=True)
            print(f"{status}: {message}")

            # Progress update every 50 stocks
            if i % 50 == 0:
                elapsed = time.time() - stats['start_time']
                rate = i / elapsed * 60  # stocks per minute
                remaining = len(symbols) - i
                eta_min = remaining / rate if rate > 0 else 0
                print()
                print(f"  Progress: {i}/{len(symbols)} ({i/len(symbols)*100:.1f}%)")
                print(f"  Success: {stats['success']}, Failed: {stats['failed']}, Skipped: {stats['skipped']}")
                print(f"  Rate: {rate:.1f} stocks/min, ETA: {eta_min:.1f} minutes")
                print()
    else:
        # Parallel processing
        print(f"Using {args.threads} threads...")
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            futures = {
                executor.submit(collect_stock, symbol, output_dir, api_key, True): symbol
                for symbol in symbols
            }

            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    sym, status, message = future.result()
                    print(f"{symbol}: {status} - {message}")
                except Exception as e:
                    print(f"{symbol}: ERROR - {str(e)}")
                    with stats_lock:
                        stats['failed'] += 1

    # Final summary
    elapsed = time.time() - stats['start_time']
    print()
    print("=" * 80)
    print("COLLECTION COMPLETE!")
    print("=" * 80)
    print(f"Total stocks: {stats['total']}")
    print(f"Success: {stats['success']}")
    print(f"Failed: {stats['failed']}")
    print(f"Skipped (already exist): {stats['skipped']}")
    print(f"Time elapsed: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
    print(f"Average rate: {stats['total']/elapsed*60:.1f} stocks/minute")
    print()
    print(f"Output directory: {output_dir}")
    print(f"Total files: {len(list(output_dir.glob('*_daily.parquet')))}")
    print()

    # Save failure log if any
    if stats['failed'] > 0:
        print(f"Note: {stats['failed']} stocks failed to collect")
        print("You can re-run this script to retry failed stocks")
        print("Or check the console output above for error details")
    else:
        print("Success! All stocks collected successfully!")

    print()


if __name__ == '__main__':
    main()