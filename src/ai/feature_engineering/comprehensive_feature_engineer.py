"""
Comprehensive Feature Engineering for AI Training

This module creates features from the collected market data for AI model training.
Processes daily, intraday, and technical indicator data into ML-ready features.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensiveFeatureEngineer:
    """
    Comprehensive feature engineering for AI training data preparation.
    
    Processes:
    - Daily OHLCV data with technical indicators
    - Intraday 1min and 5min data
    - Technical indicators (RSI, MACD, SMA, EMA, etc.)
    - Return-based and momentum features
    - Volatility and risk features
    """
    
    def __init__(self, data_dir: str = "TrainingData"):
        self.data_dir = Path(data_dir)
        self.daily_dir = self.data_dir / "daily"
        self.intraday_dir = self.data_dir / "intraday"
        self.indicators_dir = self.data_dir / "indicators"
        self.features_dir = self.data_dir / "features"
        
        # Create features directory
        self.features_dir.mkdir(exist_ok=True)
        
        # Feature configuration
        self.lookback_periods = [5, 10, 20, 50, 100]
        self.return_periods = [1, 2, 3, 5, 10, 20]
        self.volatility_periods = [5, 10, 20, 50]
        
        logger.info(f"Feature Engineer initialized with data directory: {self.data_dir}")
    
    def get_available_symbols(self) -> List[str]:
        """Get list of symbols with daily data available"""
        daily_files = list(self.daily_dir.glob("*_daily.parquet"))
        symbols = [f.stem.replace("_daily", "") for f in daily_files]
        symbols.sort()
        logger.info(f"Found {len(symbols)} symbols with daily data")
        return symbols
    
    def load_daily_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load daily OHLCV data for a symbol"""
        try:
            file_path = self.daily_dir / f"{symbol}_daily.parquet"
            if not file_path.exists():
                logger.warning(f"Daily data not found for {symbol}")
                return None
            
            df = pd.read_parquet(file_path)
            
            # Ensure datetime index
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            elif not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            # Sort by date
            df.sort_index(inplace=True)
            
            # Standardize column names to lowercase
            df.columns = df.columns.str.lower()
            
            # Basic validation
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.warning(f"Missing columns for {symbol}: {missing_cols}")
                return None
            
            logger.debug(f"Loaded daily data for {symbol}: {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error loading daily data for {symbol}: {e}")
            return None
    
    def load_technical_indicators(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Load all technical indicators for a symbol"""
        indicators = {}
        
        # Standard indicators to load
        indicator_names = ['RSI', 'MACD', 'SMA', 'EMA', 'BBANDS', 'STOCH', 'ADX', 'CCI', 'AROON']
        
        for indicator in indicator_names:
            try:
                file_path = self.indicators_dir / f"{symbol}_{indicator}.parquet"
                if file_path.exists():
                    df = pd.read_parquet(file_path)
                    
                    # Ensure datetime index
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                        df.set_index('date', inplace=True)
                    elif not isinstance(df.index, pd.DatetimeIndex):
                        df.index = pd.to_datetime(df.index)
                    
                    df.sort_index(inplace=True)
                    indicators[indicator] = df
                    logger.debug(f"Loaded {indicator} for {symbol}: {len(df)} rows")
                else:
                    logger.debug(f"Indicator {indicator} not found for {symbol}")
                    
            except Exception as e:
                logger.warning(f"Error loading {indicator} for {symbol}: {e}")
        
        logger.debug(f"Loaded {len(indicators)} indicators for {symbol}")
        return indicators
    
    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create price-based features"""
        features = df.copy()
        
        # Price ratios
        features['high_low_ratio'] = df['high'] / df['low']
        features['close_open_ratio'] = df['close'] / df['open']
        
        # Price positions within the day's range
        features['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        features['open_position'] = (df['open'] - df['low']) / (df['high'] - df['low'])
        
        # Typical price and weighted close
        features['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        features['weighted_close'] = (df['high'] + df['low'] + 2 * df['close']) / 4
        
        # Price gaps
        features['gap'] = df['open'] / df['close'].shift(1) - 1
        features['gap_filled'] = ((df['low'] <= df['close'].shift(1)) & 
                                 (df['close'].shift(1) <= df['high'])).astype(int)
        
        return features
    
    def create_return_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create return-based features"""
        features = df.copy()
        
        # Simple returns for different periods
        for period in self.return_periods:
            features[f'return_{period}d'] = df['close'].pct_change(period)
            features[f'log_return_{period}d'] = np.log(df['close'] / df['close'].shift(period))
        
        # Intraday returns
        features['intraday_return'] = (df['close'] - df['open']) / df['open']
        features['overnight_return'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        
        # High-low returns
        features['high_return'] = (df['high'] - df['close'].shift(1)) / df['close'].shift(1)
        features['low_return'] = (df['low'] - df['close'].shift(1)) / df['close'].shift(1)
        
        # Cumulative returns
        for period in [5, 10, 20]:
            features[f'cum_return_{period}d'] = (df['close'] / df['close'].shift(period) - 1)
        
        return features
    
    def create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volatility-based features"""
        features = df.copy()
        
        # True Range and Average True Range
        features['true_range'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        
        # Rolling volatility (standard deviation of returns)
        for period in self.volatility_periods:
            returns = df['close'].pct_change()
            features[f'volatility_{period}d'] = returns.rolling(period).std()
            features[f'volatility_{period}d_annualized'] = features[f'volatility_{period}d'] * np.sqrt(252)
        
        # Parkinson volatility (high-low based)
        for period in self.volatility_periods:
            hl_ratio = np.log(df['high'] / df['low'])
            features[f'parkinson_vol_{period}d'] = np.sqrt(
                hl_ratio.rolling(period).apply(lambda x: (x**2).sum() / (4 * len(x) * np.log(2)))
            )
        
        # Garman-Klass volatility
        for period in [10, 20]:
            gk_vol = (
                0.5 * (np.log(df['high'] / df['low']))**2 -
                (2 * np.log(2) - 1) * (np.log(df['close'] / df['open']))**2
            )
            features[f'gk_volatility_{period}d'] = np.sqrt(gk_vol.rolling(period).mean())
        
        return features
    
    def create_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create momentum-based features"""
        features = df.copy()
        
        # Price momentum
        for period in self.lookback_periods:
            features[f'momentum_{period}d'] = df['close'] / df['close'].shift(period) - 1
        
        # Rate of Change (ROC)
        for period in [5, 10, 20]:
            features[f'roc_{period}d'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period)
        
        # Moving average ratios
        for period in [5, 10, 20, 50]:
            ma = df['close'].rolling(period).mean()
            features[f'price_to_ma_{period}'] = df['close'] / ma
            features[f'ma_{period}_slope'] = (ma - ma.shift(5)) / ma.shift(5)
        
        # Relative strength vs different periods
        for period in [10, 20, 50]:
            features[f'relative_strength_{period}d'] = (
                df['close'].rolling(period).apply(
                    lambda x: len(x[x > x.iloc[0]]) / len(x)
                )
            )
        
        return features
    
    def create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volume-based features"""
        features = df.copy()
        
        # Volume ratios and changes
        for period in [5, 10, 20]:
            vol_ma = df['volume'].rolling(period).mean()
            features[f'volume_ratio_{period}d'] = df['volume'] / vol_ma
            features[f'volume_change_{period}d'] = df['volume'].pct_change(period)
        
        # Price-volume relationships
        features['price_volume'] = df['close'] * df['volume']
        features['volume_price_trend'] = (
            (df['close'] - df['close'].shift(1)) * df['volume']
        ).rolling(10).sum()
        
        # On-Balance Volume (OBV)
        price_change = df['close'].diff()
        obv_direction = np.where(price_change > 0, df['volume'],
                                np.where(price_change < 0, -df['volume'], 0))
        features['obv'] = obv_direction.cumsum()
        features['obv_ma_10'] = features['obv'].rolling(10).mean()
        
        # Volume-weighted average price approximation
        features['vwap_approx'] = (
            (df['typical_price'] * df['volume']).rolling(20).sum() /
            df['volume'].rolling(20).sum()
        )
        
        return features
    
    def integrate_technical_indicators(self, features: pd.DataFrame, 
                                     indicators: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Integrate technical indicators into feature set"""
        
        for indicator_name, indicator_df in indicators.items():
            try:
                # Align dates
                aligned_indicator = indicator_df.reindex(features.index, method='ffill')
                
                # Add indicator columns with prefix
                for col in aligned_indicator.columns:
                    if col not in ['date']:  # Skip date column if present
                        feature_name = f"{indicator_name.lower()}_{col}"
                        features[feature_name] = aligned_indicator[col]
                
                logger.debug(f"Integrated {indicator_name} with {len(aligned_indicator.columns)} features")
                
            except Exception as e:
                logger.warning(f"Error integrating {indicator_name}: {e}")
        
        return features
    
    def create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variables for different prediction horizons"""
        targets = pd.DataFrame(index=df.index)
        
        # Future returns (targets for regression)
        for horizon in [1, 2, 3, 5, 10]:
            targets[f'future_return_{horizon}d'] = df['close'].pct_change(horizon).shift(-horizon)
        
        # Direction targets (for classification)
        # WIDENED THRESHOLD: 0.5% instead of 0.1% for cleaner signals and better label quality
        # Narrower thresholds create too much noise in the neutral class
        DIRECTION_THRESHOLD = 0.005  # 0.5% threshold for cleaner signals
        for horizon in [1, 2, 3, 5]:
            future_return = df['close'].pct_change(horizon).shift(-horizon)
            targets[f'direction_{horizon}d'] = np.where(future_return > DIRECTION_THRESHOLD, 1,
                                                       np.where(future_return < -DIRECTION_THRESHOLD, -1, 0))
        
        # Volatility targets
        for horizon in [5, 10, 20]:
            future_vol = df['close'].pct_change().rolling(horizon).std().shift(-horizon)
            targets[f'future_volatility_{horizon}d'] = future_vol
        
        # High/Low targets (for range prediction)
        for horizon in [1, 2, 3]:
            targets[f'future_high_{horizon}d'] = df['high'].rolling(horizon).max().shift(-horizon) / df['close'] - 1
            targets[f'future_low_{horizon}d'] = df['low'].rolling(horizon).min().shift(-horizon) / df['close'] - 1
        
        return targets
    
    def process_symbol(self, symbol: str) -> bool:
        """Process a single symbol and create comprehensive features"""
        try:
            logger.info(f"Processing features for {symbol}")
            
            # Load data
            daily_data = self.load_daily_data(symbol)
            if daily_data is None or len(daily_data) < 100:
                logger.warning(f"Insufficient daily data for {symbol}")
                return False
            
            indicators = self.load_technical_indicators(symbol)
            
            # Create features
            logger.info(f"Creating price features for {symbol}")
            features = self.create_price_features(daily_data)
            
            logger.info(f"Creating return features for {symbol}")
            features = self.create_return_features(features)
            
            logger.info(f"Creating volatility features for {symbol}")
            features = self.create_volatility_features(features)
            
            logger.info(f"Creating momentum features for {symbol}")
            features = self.create_momentum_features(features)
            
            logger.info(f"Creating volume features for {symbol}")
            features = self.create_volume_features(features)
            
            # Integrate technical indicators
            if indicators:
                logger.info(f"Integrating {len(indicators)} technical indicators for {symbol}")
                features = self.integrate_technical_indicators(features, indicators)
            
            # Create targets
            logger.info(f"Creating target variables for {symbol}")
            targets = self.create_target_variables(daily_data)
            
            # Combine features and targets
            combined = pd.concat([features, targets], axis=1)
            
            # Remove rows with too many NaN values
            combined = combined.dropna(thresh=len(combined.columns) * 0.7)
            
            # Save processed features
            output_file = self.features_dir / f"{symbol}_features.parquet"
            combined.to_parquet(output_file, compression='snappy')
            
            logger.info(f"Saved features for {symbol}: {len(combined)} rows, {len(combined.columns)} columns")
            return True
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            return False
    
    def process_all_symbols(self, max_symbols: Optional[int] = None) -> Dict[str, bool]:
        """Process all available symbols"""
        symbols = self.get_available_symbols()
        
        if max_symbols:
            symbols = symbols[:max_symbols]
        
        logger.info(f"Processing features for {len(symbols)} symbols")
        
        results = {}
        successful = 0
        
        for i, symbol in enumerate(symbols, 1):
            logger.info(f"Processing {i}/{len(symbols)}: {symbol}")
            
            success = self.process_symbol(symbol)
            results[symbol] = success
            
            if success:
                successful += 1
            
            # Progress update
            if i % 10 == 0:
                logger.info(f"Progress: {i}/{len(symbols)} symbols processed, {successful} successful")
        
        logger.info(f"Feature engineering completed: {successful}/{len(symbols)} symbols successful")
        return results
    
    def get_feature_summary(self) -> pd.DataFrame:
        """Get summary of created features"""
        feature_files = list(self.features_dir.glob("*_features.parquet"))
        
        if not feature_files:
            logger.warning("No feature files found")
            return pd.DataFrame()
        
        summary_data = []
        
        for file_path in feature_files:
            try:
                symbol = file_path.stem.replace("_features", "")
                df = pd.read_parquet(file_path)
                
                summary_data.append({
                    'symbol': symbol,
                    'rows': len(df),
                    'features': len(df.columns),
                    'date_range_start': df.index.min(),
                    'date_range_end': df.index.max(),
                    'file_size_mb': file_path.stat().st_size / (1024 * 1024)
                })
                
            except Exception as e:
                logger.warning(f"Error reading {file_path}: {e}")
        
        summary_df = pd.DataFrame(summary_data)
        
        if not summary_df.empty:
            logger.info(f"Feature summary: {len(summary_df)} symbols, "
                       f"avg {summary_df['features'].mean():.0f} features, "
                       f"total {summary_df['file_size_mb'].sum():.1f} MB")
        
        return summary_df


def main():
    """Main feature engineering process"""
    print("Starting Comprehensive Feature Engineering")
    print("=" * 60)
    
    # Initialize feature engineer
    engineer = ComprehensiveFeatureEngineer()
    
    # Process all symbols
    results = engineer.process_all_symbols()
    
    # Generate summary
    summary = engineer.get_feature_summary()
    
    print("\nFeature Engineering Summary:")
    print("=" * 60)
    print(summary.to_string(index=False))
    
    # Success statistics
    successful = sum(results.values())
    total = len(results)
    
    print(f"\nResults: {successful}/{total} symbols processed successfully ({successful/total*100:.1f}%)")
    
    if successful > 0:
        print("\nNext steps:")
        print("1. Review feature files in TrainingData/features/")
        print("2. Start AI model training with processed features")
        print("3. Validate feature quality and distributions")


if __name__ == "__main__":
    main()