"""
REAL 58-Feature Engineering for Production Trading
Extracts the EXACT features used during LSTM training
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Try to import TA-Lib (faster), fallback to pandas
try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False
    print("Using pandas for indicators (slower than TA-Lib)")


class ProductionFeatureEngine:
    """Generate the exact 58 features used in LSTM training"""

    def __init__(self):
        self.features_58 = [
            # Raw OHLCV (5)
            'open', 'high', 'low', 'close', 'volume',

            # Returns (5)
            'returns', 'log_returns', 'returns_5d', 'returns_10d', 'returns_20d',

            # SMAs (6)
            'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_100', 'sma_200',

            # EMAs (6)
            'ema_5', 'ema_10', 'ema_20', 'ema_50', 'ema_100', 'ema_200',

            # Price to MA ratios (3)
            'price_to_sma_20', 'price_to_sma_50', 'price_to_sma_200',

            # RSI (2)
            'rsi_14', 'rsi_28',

            # MACD (3)
            'macd', 'macd_signal', 'macd_hist',

            # Bollinger Bands (5)
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',

            # ATR (1)
            'atr_14',

            # Stochastic (2)
            'stoch_k', 'stoch_d',

            # Volume (5)
            'volume_sma_20', 'volume_sma_50', 'volume_ratio', 'vwap', 'obv',

            # Momentum (4)
            'roc_10', 'roc_20', 'williams_r', 'cci',

            # Volatility (4)
            'volatility_20', 'volatility_50', 'parkinson_vol', 'gk_vol',

            # Patterns (7)
            'resistance_20', 'support_20', 'distance_to_resistance',
            'distance_to_support', 'trend_20', 'gap_up', 'gap_down'
        ]

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate all 58 features from OHLCV data"""

        df = df.copy()

        # Ensure lowercase columns
        df.columns = [col.lower() for col in df.columns]

        # 1. Returns
        df = self._calculate_returns(df)

        # 2. Moving Averages
        df = self._calculate_moving_averages(df)

        # 3. RSI
        df = self._calculate_rsi(df)

        # 4. MACD
        df = self._calculate_macd(df)

        # 5. Bollinger Bands
        df = self._calculate_bollinger(df)

        # 6. ATR
        df = self._calculate_atr(df)

        # 7. Stochastic
        df = self._calculate_stochastic(df)

        # 8. Volume features
        df = self._calculate_volume_features(df)

        # 9. Momentum
        df = self._calculate_momentum(df)

        # 10. Volatility
        df = self._calculate_volatility(df)

        # 11. Patterns
        df = self._calculate_patterns(df)

        return df

    def _calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Price returns"""
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['returns_5d'] = df['close'].pct_change(5)
        df['returns_10d'] = df['close'].pct_change(10)
        df['returns_20d'] = df['close'].pct_change(20)
        return df

    def _calculate_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """SMAs and EMAs"""
        for period in [5, 10, 20, 50, 100, 200]:
            if HAS_TALIB:
                df[f'sma_{period}'] = talib.SMA(df['close'].values, timeperiod=period)
                df[f'ema_{period}'] = talib.EMA(df['close'].values, timeperiod=period)
            else:
                df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
                df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

        # Price to MA ratios
        df['price_to_sma_20'] = df['close'] / df['sma_20']
        df['price_to_sma_50'] = df['close'] / df['sma_50']
        df['price_to_sma_200'] = df['close'] / df['sma_200']

        return df

    def _calculate_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """RSI indicators"""
        if HAS_TALIB:
            df['rsi_14'] = talib.RSI(df['close'].values, timeperiod=14)
            df['rsi_28'] = talib.RSI(df['close'].values, timeperiod=28)
        else:
            for period in [14, 28]:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        return df

    def _calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """MACD indicators"""
        if HAS_TALIB:
            macd, signal, hist = talib.MACD(df['close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
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

    def _calculate_bollinger(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bollinger Bands"""
        if HAS_TALIB:
            upper, middle, lower = talib.BBANDS(df['close'].values, timeperiod=20, nbdevup=2, nbdevdn=2)
            df['bb_upper'] = upper
            df['bb_middle'] = middle
            df['bb_lower'] = lower
        else:
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (2 * std)
            df['bb_lower'] = df['bb_middle'] - (2 * std)

        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        return df

    def _calculate_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Average True Range"""
        if HAS_TALIB:
            df['atr_14'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
        else:
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr_14'] = true_range.rolling(window=14).mean()
        return df

    def _calculate_stochastic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Stochastic Oscillator"""
        if HAS_TALIB:
            stoch_k, stoch_d = talib.STOCH(df['high'].values, df['low'].values, df['close'].values)
            df['stoch_k'] = stoch_k
            df['stoch_d'] = stoch_d
        else:
            low_min = df['low'].rolling(window=14).min()
            high_max = df['high'].rolling(window=14).max()
            df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
            df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        return df

    def _calculate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume-based features"""
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_sma_50'] = df['volume'].rolling(window=50).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']

        # VWAP
        df['vwap'] = (df['close'] * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()

        # OBV
        obv = [0]
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.append(obv[-1] + df['volume'].iloc[i])
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.append(obv[-1] - df['volume'].iloc[i])
            else:
                obv.append(obv[-1])
        df['obv'] = obv

        return df

    def _calculate_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Momentum indicators"""
        # Rate of Change
        df['roc_10'] = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10)) * 100
        df['roc_20'] = ((df['close'] - df['close'].shift(20)) / df['close'].shift(20)) * 100

        # Williams %R
        if HAS_TALIB:
            df['williams_r'] = talib.WILLR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
        else:
            high_14 = df['high'].rolling(window=14).max()
            low_14 = df['low'].rolling(window=14).min()
            df['williams_r'] = -100 * (high_14 - df['close']) / (high_14 - low_14)

        # CCI
        if HAS_TALIB:
            df['cci'] = talib.CCI(df['high'].values, df['low'].values, df['close'].values, timeperiod=20)
        else:
            tp = (df['high'] + df['low'] + df['close']) / 3
            sma = tp.rolling(window=20).mean()
            mad = (tp - sma).abs().rolling(window=20).mean()
            df['cci'] = (tp - sma) / (0.015 * mad)

        return df

    def _calculate_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volatility measures"""
        # Historical volatility
        df['volatility_20'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
        df['volatility_50'] = df['returns'].rolling(window=50).std() * np.sqrt(252)

        # Parkinson volatility
        df['parkinson_vol'] = np.sqrt((1 / (4 * np.log(2))) * ((np.log(df['high'] / df['low']) ** 2).rolling(window=20).mean()))

        # Garman-Klass volatility
        hl = np.log(df['high'] / df['low']) ** 2
        co = np.log(df['close'] / df['open']) ** 2
        df['gk_vol'] = np.sqrt((0.5 * hl - (2 * np.log(2) - 1) * co).rolling(window=20).mean())

        return df

    def _calculate_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Price pattern features"""
        # Resistance and support (20-day high/low)
        df['resistance_20'] = df['high'].rolling(window=20).max()
        df['support_20'] = df['low'].rolling(window=20).min()

        # Distance to resistance/support
        df['distance_to_resistance'] = (df['resistance_20'] - df['close']) / df['close']
        df['distance_to_support'] = (df['close'] - df['support_20']) / df['close']

        # Trend (20-day linear regression slope)
        def calculate_trend(series):
            if len(series) < 20:
                return 0
            x = np.arange(len(series))
            y = series.values
            slope = np.polyfit(x, y, 1)[0]
            return slope

        df['trend_20'] = df['close'].rolling(window=20).apply(calculate_trend, raw=False)

        # Gap up/down
        df['gap_up'] = ((df['open'] - df['close'].shift(1)) / df['close'].shift(1)).clip(lower=0)
        df['gap_down'] = ((df['close'].shift(1) - df['open']) / df['close'].shift(1)).clip(lower=0)

        return df

    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Apply z-score normalization (SAME AS TRAINING)
        Normalize using the stock's own historical mean and std
        """
        # Calculate mean and std from the entire history
        features_mean = np.nanmean(features, axis=0)
        features_std = np.nanstd(features, axis=0)

        # Handle zero std and NaN
        features_std = np.where((features_std == 0) | np.isnan(features_std), 1, features_std)
        features_mean = np.where(np.isnan(features_mean), 0, features_mean)

        # Z-score normalization
        features_normalized = (features - features_mean) / features_std

        # Replace remaining NaN
        features_normalized = np.nan_to_num(features_normalized, nan=0.0, posinf=0.0, neginf=0.0)

        return features_normalized

    def prepare_for_model(self, df: pd.DataFrame, sequence_length: int = 30) -> np.ndarray:
        """
        Generate features, normalize, and prepare sequence for LSTM
        Returns: (sequence_length, 58) array ready for model
        """
        # Generate all features
        df_features = self.generate_features(df)

        # Extract the 58 features in correct order
        features = df_features[self.features_58].values

        # Normalize
        features_normalized = self.normalize_features(features)

        # Get last sequence
        if len(features_normalized) < sequence_length:
            raise ValueError(f"Need at least {sequence_length} rows of data")

        sequence = features_normalized[-sequence_length:]

        return sequence


# Quick test
if __name__ == "__main__":
    print("Production Feature Engine Ready")
    print(f"Generates 58 features for LSTM model")
    print(f"TA-Lib available: {HAS_TALIB}")

    engine = ProductionFeatureEngine()
    print(f"\nFeature list ({len(engine.features_58)} features):")
    for i, feat in enumerate(engine.features_58, 1):
        print(f"  {i:2d}. {feat}")
