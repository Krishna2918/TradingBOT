"""
Indian Market Data Collector

Collects historical stock data for Indian markets (NSE/BSE) using Yahoo Finance.
Saves data in the same Parquet format as US stocks for ML training.
"""

import logging
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
import time
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# NIFTY 50 Components (as of 2024) - Major Indian Blue Chips
NIFTY_50 = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "SBIN", "BHARTIARTL", "KOTAKBANK", "ITC",
    "LT", "HCLTECH", "AXISBANK", "ASIANPAINT", "MARUTI",
    "SUNPHARMA", "TITAN", "BAJFINANCE", "DMART", "ULTRACEMCO",
    "NTPC", "ONGC", "WIPRO", "POWERGRID", "M&M",
    "TATAMOTORS", "ADANIENT", "ADANIPORTS", "COALINDIA", "JSWSTEEL",
    "TATASTEEL", "NESTLEIND", "TECHM", "INDUSINDBK", "BAJAJFINSV",
    "GRASIM", "HINDALCO", "CIPLA", "BRITANNIA", "EICHERMOT",
    "DRREDDY", "APOLLOHOSP", "DIVISLAB", "BPCL", "HEROMOTOCO",
    "TATACONSUM", "SBILIFE", "HDFCLIFE", "UPL", "SHREECEM"
]

# NIFTY Next 50 - Mid-cap leaders
NIFTY_NEXT_50 = [
    "ADANIGREEN", "AMBUJACEM", "AUROPHARMA", "BAJAJ-AUTO", "BANKBARODA",
    "BERGEPAINT", "BIOCON", "BOSCHLTD", "CANBK", "CHOLAFIN",
    "COLPAL", "CONCOR", "DLF", "GAIL", "GODREJCP",
    "HAVELLS", "HINDPETRO", "ICICIGI", "ICICIPRULI", "IDEA",
    "IGL", "INDHOTEL", "INDUSTOWER", "IOC", "IRCTC",
    "JINDALSTEL", "JUBLFOOD", "LTI", "LUPIN", "MARICO",
    "MCDOWELL-N", "MUTHOOTFIN", "NAUKRI", "NMDC", "OBEROIRLTY",
    "OFSS", "PAGEIND", "PEL", "PETRONET", "PFC",
    "PIDILITIND", "PIIND", "PNB", "RECLTD", "SAIL",
    "SRF", "TORNTPHARM", "TRENT", "VEDL", "ZOMATO"
]

# Additional important stocks
OTHER_IMPORTANT = [
    "IRFC", "ADANIPOWER", "PAYTM", "NYKAA", "POLICYBZR",
    "ZYDUSLIFE", "PERSISTENT", "COFORGE", "MPHASIS", "LTTS",
    "TATAELXSI", "TATAPOWER", "TATACHEM", "TATACOMM", "VOLTAS",
    "GODREJPROP", "PRESTIGE", "LODHA", "PHOENIXLTD", "BRIGADE",
    "INDIGO", "SPICEJET", "MAXHEALTH", "FORTIS", "METROPOLIS"
]

# Combine all symbols
ALL_INDIAN_SYMBOLS = NIFTY_50 + NIFTY_NEXT_50 + OTHER_IMPORTANT


def download_stock_data(symbol: str, suffix: str = ".NS",
                        start_date: str = "2000-01-01",
                        end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Download historical data for a single Indian stock.

    Args:
        symbol: Stock symbol without suffix
        suffix: .NS for NSE, .BO for BSE
        start_date: Start date for data
        end_date: End date (default: today)

    Returns:
        DataFrame with OHLCV data or None if failed
    """
    full_symbol = f"{symbol}{suffix}"

    try:
        ticker = yf.Ticker(full_symbol)

        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        df = ticker.history(start=start_date, end=end_date, interval="1d")

        if df.empty:
            # Try BSE if NSE fails
            if suffix == ".NS":
                logger.warning(f"{full_symbol} empty, trying BSE...")
                return download_stock_data(symbol, ".BO", start_date, end_date)
            logger.warning(f"No data for {full_symbol}")
            return None

        # Clean up the dataframe
        df = df.drop(columns=['Dividends', 'Stock Splits'], errors='ignore')

        # Ensure datetime index without timezone
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        logger.info(f"Downloaded {symbol}: {len(df)} rows ({df.index.min().date()} to {df.index.max().date()})")
        return df

    except Exception as e:
        logger.error(f"Error downloading {full_symbol}: {e}")
        return None


def collect_indian_data(symbols: List[str] = None,
                        output_dir: str = "TrainingData/daily",
                        max_workers: int = 5,
                        start_date: str = "2000-01-01") -> Dict[str, int]:
    """
    Collect historical data for Indian stocks and save to Parquet.

    Args:
        symbols: List of symbols to collect (default: all)
        output_dir: Directory to save parquet files
        max_workers: Number of parallel downloads
        start_date: Start date for historical data

    Returns:
        Dictionary with collection statistics
    """
    if symbols is None:
        symbols = ALL_INDIAN_SYMBOLS

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    stats = {
        'total': len(symbols),
        'success': 0,
        'failed': 0,
        'total_rows': 0,
        'failed_symbols': []
    }

    logger.info(f"Starting collection for {len(symbols)} Indian stocks...")
    logger.info(f"Output directory: {output_path}")
    logger.info(f"Date range: {start_date} to today")
    print()

    # Use ThreadPoolExecutor for parallel downloads
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}

        for symbol in symbols:
            future = executor.submit(download_stock_data, symbol, ".NS", start_date)
            futures[future] = symbol
            # Small delay to avoid rate limiting
            time.sleep(0.2)

        for future in as_completed(futures):
            symbol = futures[future]

            try:
                df = future.result()

                if df is not None and not df.empty:
                    # Save to parquet with _daily suffix to match US format
                    filename = f"{symbol}.NS_daily.parquet"
                    filepath = output_path / filename

                    df.to_parquet(filepath, compression='snappy')

                    stats['success'] += 1
                    stats['total_rows'] += len(df)

                    years = (df.index.max() - df.index.min()).days / 365
                    print(f"[{stats['success']}/{stats['total']}] {symbol}: {len(df):,} rows ({years:.1f} years)")
                else:
                    stats['failed'] += 1
                    stats['failed_symbols'].append(symbol)

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                stats['failed'] += 1
                stats['failed_symbols'].append(symbol)

    return stats


def main():
    """Main entry point for Indian data collection."""
    print("=" * 60)
    print("INDIAN MARKET DATA COLLECTOR")
    print("=" * 60)
    print()
    print(f"Symbols to collect: {len(ALL_INDIAN_SYMBOLS)}")
    print(f"  - NIFTY 50: {len(NIFTY_50)}")
    print(f"  - NIFTY Next 50: {len(NIFTY_NEXT_50)}")
    print(f"  - Other Important: {len(OTHER_IMPORTANT)}")
    print()

    # Collect data
    stats = collect_indian_data(
        symbols=ALL_INDIAN_SYMBOLS,
        output_dir="TrainingData/daily",
        max_workers=5,
        start_date="2000-01-01"  # 25 years of data
    )

    print()
    print("=" * 60)
    print("COLLECTION COMPLETE")
    print("=" * 60)
    print(f"Success: {stats['success']}/{stats['total']}")
    print(f"Failed: {stats['failed']}")
    print(f"Total rows: {stats['total_rows']:,}")

    if stats['failed_symbols']:
        print(f"\nFailed symbols: {stats['failed_symbols'][:20]}")
        if len(stats['failed_symbols']) > 20:
            print(f"  ... and {len(stats['failed_symbols']) - 20} more")

    print()
    print("Data saved to: TrainingData/daily/")
    print("=" * 60)


if __name__ == "__main__":
    main()
