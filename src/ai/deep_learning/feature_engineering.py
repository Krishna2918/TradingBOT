"""
Advanced feature engineering for deep learning models.

This module provides comprehensive feature engineering capabilities
for financial time series data, including technical indicators,
statistical features, and domain-specific transformations.
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression

logger = logging.getLogger(__name__)

class DeepLearningFeatureEngineer:
    """
    Advanced feature engineering for deep learning models.
    
    This class provides comprehensive feature engineering capabilities
    including technical indicators, statistical features, and transformations.
    """
    
    def __init__(
        self,
        lookback_periods: List[int] = [5, 10, 20, 50],
        feature_scaling: str = 'minmax',
        include_technical_indicators: bool = True,
        include_statistical_features: bool = True,
        include_fourier_features: bool = True,
        include_wavelet_features: bool = False
    ):
        """
        Initialize Deep Learning Feature Engineer.
        
        Args:
            lookback_periods: List of lookback periods for rolling features
            feature_scaling: Scaling method ('minmax', 'standard', 'robust', 'none')
            include_technical_indicators: Whether to include technical indicators
            include_statistical_features: Whether to include statistical features
            include_fourier_features: Whether to include Fourier transform features
            include_wavelet_features: Whether to include wavelet transform features
        """
        self.lookback_periods = lookback_periods
        self.feature_scaling = feature_scaling
        self.include_technical_indicators = include_technical_indicators
        self.include_statistical_features = include_statistical_features
        self.include_fourier_features = include_fourier_features
        self.include_wavelet_features = include_wavelet_features
        
        self.scaler = None
        self.feature_names = []
        self.feature_importance = {}
        
        logger.info("Initialized Deep Learning Feature Engineer")
    
    def create_features(
        self,
        data: pd.DataFrame,
        target_column: str = 'close',
        feature_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Create comprehensive features for deep learning models.
        
        Args:
            data: Input DataFrame with OHLCV data
            target_column: Target column for prediction
            feature_columns: Base columns to use for feature engineering
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting feature engineering process")
        
        # Start with base data
        if feature_columns is None:
            feature_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Ensure we have the required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Create a copy of the data
        df = data[feature_columns].copy()
        
        # Add technical indicators
        if self.include_technical_indicators:
            df = self._add_technical_indicators(df)
        
        # Add statistical features
        if self.include_statistical_features:
            df = self._add_statistical_features(df)
        
        # Add Fourier features
        if self.include_fourier_features:
            df = self._add_fourier_features(df, target_column)
        
        # Add wavelet features
        if self.include_wavelet_features:
            df = self._add_wavelet_features(df, target_column)
        
        # Add price-based features
        df = self._add_price_features(df)
        
        # Add volume-based features
        df = self._add_volume_features(df)
        
        # Add volatility features
        df = self._add_volatility_features(df)
        
        # Add momentum features
        df = self._add_momentum_features(df)
        
        # Add market microstructure features
        df = self._add_microstructure_features(df)
        
        # Remove infinite and NaN values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Store feature names
        self.feature_names = df.columns.tolist()
        
        logger.info(f"Created {len(self.feature_names)} features")
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators using TA-Lib."""
        logger.info("Adding technical indicators")
        
        # Moving averages
        for period in [5, 10, 20, 50, 200]:
            df[f'sma_{period}'] = talib.SMA(df['close'], timeperiod=period)
            df[f'ema_{period}'] = talib.EMA(df['close'], timeperiod=period)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(df['close'], timeperiod=20)
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        df['bb_width'] = (bb_upper - bb_lower) / bb_middle
        df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # RSI
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        df['rsi_30'] = talib.RSI(df['close'], timeperiod=30)
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(df['close'])
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_hist'] = macd_hist
        
        # Stochastic
        stoch_k, stoch_d = talib.STOCH(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch_k
        df['stoch_d'] = stoch_d
        
        # Williams %R
        df['williams_r'] = talib.WILLR(df['high'], df['low'], df['close'])
        
        # CCI
        df['cci'] = talib.CCI(df['high'], df['low'], df['close'])
        
        # ATR
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'])
        
        # ADX
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'])
        
        # OBV
        df['obv'] = talib.OBV(df['close'], df['volume'])
        
        # Money Flow Index
        df['mfi'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'])
        
        return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features."""
        logger.info("Adding statistical features")
        
        # Rolling statistics for different periods
        for period in self.lookback_periods:
            # Mean
            df[f'mean_{period}'] = df['close'].rolling(period).mean()
            
            # Standard deviation
            df[f'std_{period}'] = df['close'].rolling(period).std()
            
            # Skewness
            df[f'skew_{period}'] = df['close'].rolling(period).skew()
            
            # Kurtosis
            df[f'kurt_{period}'] = df['close'].rolling(period).kurt()
            
            # Min/Max
            df[f'min_{period}'] = df['close'].rolling(period).min()
            df[f'max_{period}'] = df['close'].rolling(period).max()
            
            # Quantiles
            df[f'q25_{period}'] = df['close'].rolling(period).quantile(0.25)
            df[f'q75_{period}'] = df['close'].rolling(period).quantile(0.75)
            
            # Range
            df[f'range_{period}'] = df[f'max_{period}'] - df[f'min_{period}']
            
            # Coefficient of variation
            df[f'cv_{period}'] = df[f'std_{period}'] / df[f'mean_{period}']
        
        # Price position within range
        for period in self.lookback_periods:
            df[f'price_position_{period}'] = (df['close'] - df[f'min_{period}']) / df[f'range_{period}']
        
        return df
    
    def _add_fourier_features(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Add Fourier transform features."""
        logger.info("Adding Fourier features")
        
        # Fast Fourier Transform
        fft = np.fft.fft(df[target_column].values)
        fft_real = np.real(fft)
        fft_imag = np.imag(fft)
        
        # Add first few Fourier components
        for i in range(1, 6):  # First 5 components
            df[f'fft_real_{i}'] = fft_real[i]
            df[f'fft_imag_{i}'] = fft_imag[i]
        
        # Power spectral density
        psd = np.abs(fft) ** 2
        for i in range(1, 6):
            df[f'psd_{i}'] = psd[i]
        
        return df
    
    def _add_wavelet_features(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Add wavelet transform features."""
        logger.info("Adding wavelet features")
        
        try:
            import pywt
            
            # Discrete Wavelet Transform
            coeffs = pywt.dwt(df[target_column].values, 'db4')
            cA, cD = coeffs
            
            # Add approximation and detail coefficients
            df['wavelet_approx'] = cA[0] if len(cA) > 0 else 0
            df['wavelet_detail'] = cD[0] if len(cD) > 0 else 0
            
            # Wavelet energy
            df['wavelet_energy_approx'] = np.sum(cA ** 2) if len(cA) > 0 else 0
            df['wavelet_energy_detail'] = np.sum(cD ** 2) if len(cD) > 0 else 0
            
        except ImportError:
            logger.warning("PyWavelets not available, skipping wavelet features")
        
        return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features."""
        logger.info("Adding price features")
        
        # Price changes
        df['price_change'] = df['close'].pct_change()
        df['price_change_abs'] = df['price_change'].abs()
        
        # Log returns
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # Price ratios
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        df['high_close_ratio'] = df['high'] / df['close']
        df['low_close_ratio'] = df['low'] / df['close']
        
        # Price position
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Gap features
        df['gap'] = df['open'] - df['close'].shift(1)
        df['gap_ratio'] = df['gap'] / df['close'].shift(1)
        
        # Intraday range
        df['intraday_range'] = df['high'] - df['low']
        df['intraday_range_ratio'] = df['intraday_range'] / df['close']
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        logger.info("Adding volume features")
        
        # Volume changes
        df['volume_change'] = df['volume'].pct_change()
        df['volume_change_abs'] = df['volume_change'].abs()
        
        # Volume moving averages
        for period in [5, 10, 20]:
            df[f'volume_sma_{period}'] = df['volume'].rolling(period).mean()
            df[f'volume_ratio_{period}'] = df['volume'] / df[f'volume_sma_{period}']
        
        # Volume-price relationship
        df['volume_price_trend'] = df['volume'] * df['price_change']
        df['volume_weighted_price'] = (df['volume'] * df['close']).rolling(20).sum() / df['volume'].rolling(20).sum()
        
        # On-Balance Volume
        df['obv_change'] = df['obv'].pct_change()
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility features."""
        logger.info("Adding volatility features")
        
        # Historical volatility
        for period in self.lookback_periods:
            df[f'volatility_{period}'] = df['log_return'].rolling(period).std() * np.sqrt(252)
        
        # Parkinson volatility
        df['parkinson_vol'] = np.sqrt(
            (1 / (4 * np.log(2))) * np.log(df['high'] / df['low']) ** 2
        )
        
        # Garman-Klass volatility
        df['gk_vol'] = np.sqrt(
            0.5 * np.log(df['high'] / df['low']) ** 2 - 
            (2 * np.log(2) - 1) * np.log(df['close'] / df['open']) ** 2
        )
        
        # Volatility of volatility
        df['vol_of_vol'] = df['volatility_20'].rolling(20).std()
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum features."""
        logger.info("Adding momentum features")
        
        # Price momentum
        for period in [1, 5, 10, 20]:
            df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
        
        # Rate of change
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = talib.ROC(df['close'], timeperiod=period)
        
        # Momentum oscillator
        df['momentum_osc'] = talib.MOM(df['close'], timeperiod=10)
        
        # Commodity Channel Index
        df['cci_20'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=20)
        
        return df
    
    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features."""
        logger.info("Adding microstructure features")
        
        # Bid-ask spread proxy (using high-low)
        df['spread_proxy'] = (df['high'] - df['low']) / df['close']
        
        # Price impact
        df['price_impact'] = df['volume'] * df['price_change']
        
        # Tick direction
        df['tick_direction'] = np.where(df['close'] > df['close'].shift(1), 1, -1)
        
        # Tick intensity
        df['tick_intensity'] = df['tick_direction'].rolling(10).sum()
        
        # Volume imbalance
        df['volume_imbalance'] = df['volume'] * df['tick_direction']
        
        return df
    
    def scale_features(
        self,
        df: pd.DataFrame,
        fit_scaler: bool = True
    ) -> pd.DataFrame:
        """
        Scale features using the specified scaling method.
        
        Args:
            df: DataFrame with features to scale
            fit_scaler: Whether to fit the scaler (True for training, False for inference)
            
        Returns:
            DataFrame with scaled features
        """
        if self.feature_scaling == 'none':
            return df
        
        logger.info(f"Scaling features using {self.feature_scaling} scaler")
        
        # Select numeric columns only
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df_scaled = df.copy()
        
        if fit_scaler:
            # Fit scaler on training data
            if self.feature_scaling == 'minmax':
                self.scaler = MinMaxScaler()
            elif self.feature_scaling == 'standard':
                self.scaler = StandardScaler()
            elif self.feature_scaling == 'robust':
                self.scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown scaling method: {self.feature_scaling}")
            
            df_scaled[numeric_columns] = self.scaler.fit_transform(df[numeric_columns])
        else:
            # Use fitted scaler for inference
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call with fit_scaler=True first.")
            
            df_scaled[numeric_columns] = self.scaler.transform(df[numeric_columns])
        
        return df_scaled
    
    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = 'kbest',
        k: int = 50
    ) -> pd.DataFrame:
        """
        Select the most important features.
        
        Args:
            X: Feature matrix
            y: Target variable
            method: Feature selection method ('kbest', 'pca')
            k: Number of features to select
            
        Returns:
            DataFrame with selected features
        """
        logger.info(f"Selecting {k} features using {method}")
        
        if method == 'kbest':
            selector = SelectKBest(score_func=f_regression, k=k)
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            
            # Store feature importance
            self.feature_importance = dict(zip(
                selected_features,
                selector.scores_[selector.get_support()]
            ))
            
        elif method == 'pca':
            pca = PCA(n_components=k)
            X_selected = pca.fit_transform(X)
            selected_features = [f'pca_{i}' for i in range(k)]
            
            # Store explained variance
            self.feature_importance = dict(zip(
                selected_features,
                pca.explained_variance_ratio_
            ))
            
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
        
        # Create DataFrame with selected features
        df_selected = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        
        logger.info(f"Selected {len(selected_features)} features")
        
        return df_selected
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary with feature importance scores
        """
        return self.feature_importance.copy()
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names.
        
        Returns:
            List of feature names
        """
        return self.feature_names.copy()
    
    def create_sequences(
        self,
        df: pd.DataFrame,
        sequence_length: int = 60,
        target_column: Optional[str] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Create sequences for time series models.
        
        Args:
            df: DataFrame with features
            sequence_length: Length of sequences
            target_column: Target column (if None, only features are returned)
            
        Returns:
            Tuple of (X, y) arrays
        """
        logger.info(f"Creating sequences with length {sequence_length}")
        
        # Convert to numpy array
        feature_data = df.values
        
        # Create sequences
        X = []
        y = [] if target_column else None
        
        for i in range(sequence_length, len(feature_data)):
            X.append(feature_data[i-sequence_length:i])
            if target_column and target_column in df.columns:
                y.append(df[target_column].iloc[i])
        
        X = np.array(X)
        if y is not None:
            y = np.array(y)
        
        logger.info(f"Created {len(X)} sequences")
        
        return X, y
    
    def save_engineer(self, filepath: str) -> None:
        """Save the feature engineer to disk."""
        import joblib
        
        engineer_data = {
            'lookback_periods': self.lookback_periods,
            'feature_scaling': self.feature_scaling,
            'include_technical_indicators': self.include_technical_indicators,
            'include_statistical_features': self.include_statistical_features,
            'include_fourier_features': self.include_fourier_features,
            'include_wavelet_features': self.include_wavelet_features,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance
        }
        
        joblib.dump(engineer_data, filepath)
        logger.info(f"Feature engineer saved to {filepath}")
    
    def load_engineer(self, filepath: str) -> None:
        """Load the feature engineer from disk."""
        import joblib
        
        engineer_data = joblib.load(filepath)
        
        self.lookback_periods = engineer_data['lookback_periods']
        self.feature_scaling = engineer_data['feature_scaling']
        self.include_technical_indicators = engineer_data['include_technical_indicators']
        self.include_statistical_features = engineer_data['include_statistical_features']
        self.include_fourier_features = engineer_data['include_fourier_features']
        self.include_wavelet_features = engineer_data['include_wavelet_features']
        self.scaler = engineer_data['scaler']
        self.feature_names = engineer_data['feature_names']
        self.feature_importance = engineer_data['feature_importance']
        
        logger.info(f"Feature engineer loaded from {filepath}")

