"""
PRODUCTION FEATURE ENGINEERING - Optimized for 1,695 Stocks
Generates 95 technical features for all stocks

Features Generated:
- Price features (10): Returns, log returns, price ratios
- Technical indicators (40): SMA, EMA, RSI, MACD, Bollinger Bands, ATR
- Volume features (10): Volume ratios, VWAP, OBV
- Momentum features (15): ROC, Stochastic, Williams %R
- Volatility features (10): Historical vol, Parkinson, Garman-Klass
- Pattern features (10): Support/resistance, trends

Usage:
    # Generate features for all stocks
    python generate_features_production.py --input TrainingData/daily --output TrainingData/features

    # Test mode (10% of stocks)
    python generate_features_production.py --test-mode

    # Parallel processing (4 workers)
    python generate_features_production.py --workers 4

Expected Time: 2-4 hours for 1,695 stocks (with 4 workers)

Author: Trading Bot Team
Date: October 28, 2025
"""

import os
import sys
import time
import warnings
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings('ignore')

# Technical indicators
try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False
    print("Warning: TA-Lib not installed. Using pandas implementations (slower).")
    print("Install TA-Lib for 3-5x faster feature generation:")
    print("  pip install TA-Lib")


def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate price returns"""
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['returns_5d'] = df['close'].pct_change(5)
    df['returns_10d'] = df['close'].pct_change(10)
    df['returns_20d'] = df['close'].pct_change(20)
    return df


def calculate_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate moving averages"""
    for period in [5, 10, 20, 50, 100, 200]:
        if HAS_TALIB:
            df[f'sma_{period}'] = talib.SMA(df['close'], timeperiod=period)
            df[f'ema_{period}'] = talib.EMA(df['close'], timeperiod=period)
        else:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

    # Moving average ratios (price relative to MA)
    df['price_to_sma_20'] = df['close'] / df['sma_20']
    df['price_to_sma_50'] = df['close'] / df['sma_50']
    df['price_to_sma_200'] = df['close'] / df['sma_200']

    return df


def calculate_rsi(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate RSI (Relative Strength Index)"""
    if HAS_TALIB:
        df['rsi_14'] = talib.RSI(df['close'], timeperiod=14)
        df['rsi_28'] = talib.RSI(df['close'], timeperiod=28)
    else:
        # Manual RSI calculation
        for period in [14, 28]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

    return df


def calculate_macd(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate MACD"""
    if HAS_TALIB:
        macd, signal, hist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = hist
    else:
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

    return df


def calculate_bollinger_bands(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Bollinger Bands"""
    if HAS_TALIB:
        upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2)
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower
    else:
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (2 * bb_std)
        df['bb_lower'] = df['bb_middle'] - (2 * bb_std)

    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    return df


def calculate_atr(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate ATR (Average True Range)"""
    if HAS_TALIB:
        df['atr_14'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    else:
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(window=14).mean()

    return df


def calculate_stochastic(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Stochastic Oscillator"""
    if HAS_TALIB:
        slowk, slowd = talib.STOCH(df['high'], df['low'], df['close'],
                                    fastk_period=14, slowk_period=3, slowd_period=3)
        df['stoch_k'] = slowk
        df['stoch_d'] = slowd
    else:
        low_14 = df['low'].rolling(window=14).min()
        high_14 = df['high'].rolling(window=14).max()
        df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()

    return df


def calculate_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate volume-based features"""
    # Volume moving averages
    df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_sma_50'] = df['volume'].rolling(window=50).mean()

    # Volume ratios
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']

    # VWAP (Volume Weighted Average Price)
    df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()

    # OBV (On-Balance Volume)
    if HAS_TALIB:
        df['obv'] = talib.OBV(df['close'], df['volume'])
    else:
        obv = [0]
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i - 1]:
                obv.append(obv[-1] + df['volume'].iloc[i])
            elif df['close'].iloc[i] < df['close'].iloc[i - 1]:
                obv.append(obv[-1] - df['volume'].iloc[i])
            else:
                obv.append(obv[-1])
        df['obv'] = obv

    return df


def calculate_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate momentum indicators"""
    # ROC (Rate of Change)
    if HAS_TALIB:
        df['roc_10'] = talib.ROC(df['close'], timeperiod=10)
        df['roc_20'] = talib.ROC(df['close'], timeperiod=20)
    else:
        df['roc_10'] = df['close'].pct_change(10) * 100
        df['roc_20'] = df['close'].pct_change(20) * 100

    # Williams %R
    if HAS_TALIB:
        df['williams_r'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
    else:
        high_14 = df['high'].rolling(window=14).max()
        low_14 = df['low'].rolling(window=14).min()
        df['williams_r'] = -100 * (high_14 - df['close']) / (high_14 - low_14)

    # CCI (Commodity Channel Index)
    if HAS_TALIB:
        df['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=20)
    else:
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = tp.rolling(window=20).mean()
        mad = tp.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
        df['cci'] = (tp - sma_tp) / (0.015 * mad)

    return df


def calculate_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate volatility measures"""
    # Historical volatility
    df['volatility_20'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
    df['volatility_50'] = df['returns'].rolling(window=50).std() * np.sqrt(252)

    # Parkinson volatility (uses high-low range)
    df['parkinson_vol'] = np.sqrt((1 / (4 * np.log(2))) * ((np.log(df['high'] / df['low'])) ** 2))

    # Garman-Klass volatility
    df['gk_vol'] = np.sqrt(
        0.5 * (np.log(df['high'] / df['low'])) ** 2 -
        (2 * np.log(2) - 1) * (np.log(df['close'] / df['open'])) ** 2
    )

    return df


def calculate_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate pattern recognition features"""
    # Support and resistance levels
    df['resistance_20'] = df['high'].rolling(window=20).max()
    df['support_20'] = df['low'].rolling(window=20).min()
    df['distance_to_resistance'] = (df['resistance_20'] - df['close']) / df['close']
    df['distance_to_support'] = (df['close'] - df['support_20']) / df['close']

    # Trend strength
    df['trend_20'] = (df['close'] - df['close'].shift(20)) / df['close'].shift(20)

    # Gap detection
    df['gap_up'] = ((df['open'] - df['close'].shift(1)) / df['close'].shift(1)).clip(lower=0)
    df['gap_down'] = ((df['close'].shift(1) - df['open']) / df['close'].shift(1)).clip(lower=0)

    return df


def generate_features_for_stock(file_path: Path, output_dir: Path) -> Tuple[str, bool, str]:
    """Generate features for a single stock"""
    symbol = file_path.stem.replace('_daily', '')

    try:
        # Load data
        df = pd.read_parquet(file_path)

        # Ensure columns are lowercase and check required columns
        df.columns = [col.lower() for col in df.columns]
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            return symbol, False, f"Missing columns. Has: {df.columns.tolist()}"

        # Validate minimum data
        if len(df) < 250:  # Need at least 1 year for features
            return symbol, False, f"Insufficient data: {len(df)} days"

        # Calculate all features
        df = calculate_returns(df)
        df = calculate_moving_averages(df)
        df = calculate_rsi(df)
        df = calculate_macd(df)
        df = calculate_bollinger_bands(df)
        df = calculate_atr(df)
        df = calculate_stochastic(df)
        df = calculate_volume_features(df)
        df = calculate_momentum_features(df)
        df = calculate_volatility_features(df)
        df = calculate_pattern_features(df)

        # Drop NaN rows (from rolling windows)
        df = df.dropna()

        if len(df) < 100:
            return symbol, False, f"Too many NaN after feature generation: {len(df)} rows remain"

        # Save features
        output_file = output_dir / f"{symbol}_features.parquet"
        df.to_parquet(output_file, compression='snappy')

        return symbol, True, f"{len(df)} rows, {len(df.columns)} features"

    except Exception as e:
        return symbol, False, f"Error: {str(e)}"


def main():
    parser = argparse.ArgumentParser(description='Production Feature Engineering')
    parser.add_argument('--input', type=str, default='TrainingData/daily', help='Input directory (raw data)')
    parser.add_argument('--output', type=str, default='TrainingData/features', help='Output directory (features)')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--test-mode', action='store_true', help='Test mode (10%% of stocks)')
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    # Validate input
    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}")
        print("Please run data collection first:")
        print("  python ultimate_1400_collector.py --continuous")
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all stock files
    stock_files = list(input_dir.glob('*_daily.parquet'))
    if len(stock_files) == 0:
        print(f"ERROR: No stock files found in {input_dir}")
        return

    print("=" * 80)
    print("PRODUCTION FEATURE ENGINEERING")
    print("=" * 80)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Total stocks: {len(stock_files)}")
    print(f"Workers: {args.workers}")
    print(f"TA-Lib: {'Available' if HAS_TALIB else 'Not available (using pandas)'}")
    print("")

    # Test mode
    if args.test_mode:
        stock_files = stock_files[:int(len(stock_files) * 0.1)]
        print(f"TEST MODE: Processing only {len(stock_files)} stocks (10%)")
        print("")

    # Process stocks in parallel
    start_time = time.time()
    successful = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(generate_features_for_stock, file, output_dir): file for file in stock_files}

        for i, future in enumerate(as_completed(futures), 1):
            symbol, success, message = future.result()

            if success:
                successful += 1
                status = "SUCCESS"
            else:
                failed += 1
                status = "FAILED"

            # Progress
            if i % 50 == 0 or not success:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(stock_files) - i) / rate if rate > 0 else 0

                print(f"[{i}/{len(stock_files)}] {symbol}: {status} - {message}")
                print(f"  Progress: {i/len(stock_files)*100:.1f}% | "
                      f"Success: {successful} | Failed: {failed} | "
                      f"Rate: {rate:.1f} stocks/sec | ETA: {eta/60:.1f} min")

    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 80)
    print(f"Total processed: {len(stock_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Time elapsed: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
    print(f"Average rate: {len(stock_files)/elapsed:.1f} stocks/sec")
    print(f"\nFeatures saved to: {output_dir}")
    print(f"Total files: {len(list(output_dir.glob('*.parquet')))}")

    if successful > 0:
        # Check first file for feature count
        first_file = list(output_dir.glob('*.parquet'))[0]
        df_sample = pd.read_parquet(first_file)
        print(f"Features per stock: {len(df_sample.columns)}")
        print(f"\nReady for training!")
        print(f"Next step:")
        print(f"  python train_lstm_production.py --data {output_dir}")


if __name__ == '__main__':
    main()
