"""Advanced Feature Engineering for Trading Models"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from scipy import stats
from scipy.signal import find_peaks
import talib

logger = logging.getLogger(__name__)

@dataclass
class FeatureSet:
    """A set of engineered features."""
    features: Dict[str, float]
    feature_names: List[str]
    feature_categories: Dict[str, List[str]]
    timestamp: datetime
    metadata: Dict[str, Any]

class TechnicalIndicatorEngine:
    """Advanced technical indicator calculations."""
    
    def __init__(self):
        self.indicators = {}
        self.lookback_periods = {
            'short': 5,
            'medium': 20,
            'long': 50
        }
    
    def calculate_all_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate all technical indicators."""
        features = {}
        
        if len(data) < 50:  # Need sufficient data
            return features
        
        # Price data
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        volume = data['volume'].values if 'volume' in data.columns else np.ones(len(data))
        
        # Trend indicators
        features.update(self._calculate_trend_indicators(close, high, low))
        
        # Momentum indicators
        features.update(self._calculate_momentum_indicators(close, high, low, volume))
        
        # Volatility indicators
        features.update(self._calculate_volatility_indicators(high, low, close))
        
        # Volume indicators
        features.update(self._calculate_volume_indicators(close, volume))
        
        # Pattern recognition
        open_prices = data['open'].values if 'open' in data.columns else close
        features.update(self._calculate_pattern_indicators(open_prices, high, low, close))
        
        return features
    
    def _calculate_trend_indicators(self, close: np.ndarray, high: np.ndarray, low: np.ndarray) -> Dict[str, float]:
        """Calculate trend-following indicators."""
        features = {}
        
        # Moving averages
        for period_name, period in self.lookback_periods.items():
            if len(close) >= period:
                sma = talib.SMA(close, timeperiod=period)[-1]
                ema = talib.EMA(close, timeperiod=period)[-1]
                features[f'sma_{period_name}'] = sma
                features[f'ema_{period_name}'] = ema
                
                # Price relative to moving averages
                if not np.isnan(sma) and sma > 0:
                    features[f'price_sma_ratio_{period_name}'] = close[-1] / sma
                if not np.isnan(ema) and ema > 0:
                    features[f'price_ema_ratio_{period_name}'] = close[-1] / ema
        
        # MACD
        if len(close) >= 26:
            macd, macd_signal, macd_hist = talib.MACD(close)
            features['macd'] = macd[-1] if not np.isnan(macd[-1]) else 0.0
            features['macd_signal'] = macd_signal[-1] if not np.isnan(macd_signal[-1]) else 0.0
            features['macd_histogram'] = macd_hist[-1] if not np.isnan(macd_hist[-1]) else 0.0
        
        # ADX (Average Directional Index)
        if len(high) >= 14 and len(low) >= 14:
            adx = talib.ADX(high, low, close, timeperiod=14)
            features['adx'] = adx[-1] if not np.isnan(adx[-1]) else 0.0
        
        # Parabolic SAR
        if len(high) >= 10:
            sar = talib.SAR(high, low)
            features['sar'] = sar[-1] if not np.isnan(sar[-1]) else close[-1]
            features['sar_signal'] = 1.0 if close[-1] > sar[-1] else -1.0
        
        return features
    
    def _calculate_momentum_indicators(self, close: np.ndarray, high: np.ndarray, 
                                     low: np.ndarray, volume: np.ndarray) -> Dict[str, float]:
        """Calculate momentum indicators."""
        features = {}
        
        # RSI
        if len(close) >= 14:
            rsi = talib.RSI(close, timeperiod=14)
            features['rsi'] = rsi[-1] if not np.isnan(rsi[-1]) else 50.0
        
        # Stochastic Oscillator
        if len(high) >= 14 and len(low) >= 14:
            slowk, slowd = talib.STOCH(high, low, close)
            features['stoch_k'] = slowk[-1] if not np.isnan(slowk[-1]) else 50.0
            features['stoch_d'] = slowd[-1] if not np.isnan(slowd[-1]) else 50.0
        
        # Williams %R
        if len(high) >= 14 and len(low) >= 14:
            willr = talib.WILLR(high, low, close, timeperiod=14)
            features['williams_r'] = willr[-1] if not np.isnan(willr[-1]) else -50.0
        
        # CCI (Commodity Channel Index)
        if len(high) >= 14 and len(low) >= 14:
            cci = talib.CCI(high, low, close, timeperiod=14)
            features['cci'] = cci[-1] if not np.isnan(cci[-1]) else 0.0
        
        # Rate of Change
        for period in [5, 10, 20]:
            if len(close) >= period + 1:
                roc = talib.ROC(close, timeperiod=period)
                features[f'roc_{period}'] = roc[-1] if not np.isnan(roc[-1]) else 0.0
        
        # Money Flow Index
        if len(high) >= 14 and len(low) >= 14 and len(volume) >= 14:
            mfi = talib.MFI(high, low, close, volume, timeperiod=14)
            features['mfi'] = mfi[-1] if not np.isnan(mfi[-1]) else 50.0
        
        return features
    
    def _calculate_volatility_indicators(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, float]:
        """Calculate volatility indicators."""
        features = {}
        
        # Bollinger Bands
        if len(close) >= 20:
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20)
            features['bb_upper'] = bb_upper[-1] if not np.isnan(bb_upper[-1]) else close[-1]
            features['bb_middle'] = bb_middle[-1] if not np.isnan(bb_middle[-1]) else close[-1]
            features['bb_lower'] = bb_lower[-1] if not np.isnan(bb_lower[-1]) else close[-1]
            
            # Bollinger Band position
            if not np.isnan(bb_upper[-1]) and not np.isnan(bb_lower[-1]) and bb_upper[-1] != bb_lower[-1]:
                bb_position = (close[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1])
                features['bb_position'] = bb_position
        
        # Average True Range
        if len(high) >= 14 and len(low) >= 14:
            atr = talib.ATR(high, low, close, timeperiod=14)
            features['atr'] = atr[-1] if not np.isnan(atr[-1]) else 0.0
            features['atr_ratio'] = atr[-1] / close[-1] if close[-1] > 0 else 0.0
        
        # True Range
        if len(high) >= 2 and len(low) >= 2:
            tr = talib.TRANGE(high, low, close)
            features['true_range'] = tr[-1] if not np.isnan(tr[-1]) else 0.0
        
        # Historical Volatility
        for period in [10, 20, 30]:
            if len(close) >= period + 1:
                returns = np.diff(np.log(close[-period-1:]))
                if len(returns) > 0:
                    hist_vol = np.std(returns) * np.sqrt(252)  # Annualized
                    features[f'hist_vol_{period}'] = hist_vol
        
        return features
    
    def _calculate_volume_indicators(self, close: np.ndarray, volume: np.ndarray) -> Dict[str, float]:
        """Calculate volume-based indicators."""
        features = {}
        
        # On-Balance Volume
        if len(close) >= 2 and len(volume) >= 2:
            obv = talib.OBV(close, volume)
            features['obv'] = obv[-1] if not np.isnan(obv[-1]) else 0.0
        
        # Accumulation/Distribution Line
        if len(close) >= 2 and len(volume) >= 2:
            ad = talib.AD(high=np.ones(len(close)) * close[-1], low=np.ones(len(close)) * close[-1], 
                         close=close, volume=volume)
            features['ad_line'] = ad[-1] if not np.isnan(ad[-1]) else 0.0
        
        # Volume Rate of Change
        for period in [5, 10, 20]:
            if len(volume) >= period + 1:
                vol_roc = talib.ROC(volume, timeperiod=period)
                features[f'volume_roc_{period}'] = vol_roc[-1] if not np.isnan(vol_roc[-1]) else 0.0
        
        # Volume Moving Average
        for period in [5, 10, 20]:
            if len(volume) >= period:
                vol_sma = talib.SMA(volume, timeperiod=period)
                features[f'volume_sma_{period}'] = vol_sma[-1] if not np.isnan(vol_sma[-1]) else volume[-1]
                
                # Volume ratio
                if not np.isnan(vol_sma[-1]) and vol_sma[-1] > 0:
                    features[f'volume_ratio_{period}'] = volume[-1] / vol_sma[-1]
        
        return features
    
    def _calculate_pattern_indicators(self, open: np.ndarray, high: np.ndarray, 
                                    low: np.ndarray, close: np.ndarray) -> Dict[str, float]:
        """Calculate candlestick pattern indicators."""
        features = {}
        
        if len(close) < 5:  # Need minimum data for patterns
            return features
        
        # Doji
        doji = talib.CDLDOJI(open, high, low, close)
        features['doji'] = doji[-1] if not np.isnan(doji[-1]) else 0.0
        
        # Hammer
        hammer = talib.CDLHAMMER(open, high, low, close)
        features['hammer'] = hammer[-1] if not np.isnan(hammer[-1]) else 0.0
        
        # Engulfing patterns
        engulfing = talib.CDLENGULFING(open, high, low, close)
        features['engulfing'] = engulfing[-1] if not np.isnan(engulfing[-1]) else 0.0
        
        # Morning/Evening Star
        morning_star = talib.CDLMORNINGSTAR(open, high, low, close)
        features['morning_star'] = morning_star[-1] if not np.isnan(morning_star[-1]) else 0.0
        
        evening_star = talib.CDLEVENINGSTAR(open, high, low, close)
        features['evening_star'] = evening_star[-1] if not np.isnan(evening_star[-1]) else 0.0
        
        return features

class StatisticalFeatureEngine:
    """Advanced statistical feature calculations."""
    
    def __init__(self):
        self.statistical_windows = [5, 10, 20, 50]
    
    def calculate_statistical_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate statistical features."""
        features = {}
        
        if len(data) < 50:
            return features
        
        # Price-based features
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        volume = data['volume'].values if 'volume' in data.columns else np.ones(len(data))
        
        # Returns
        features.update(self._calculate_return_features(close))
        
        # Statistical moments
        features.update(self._calculate_moment_features(close))
        
        # Autocorrelation features
        features.update(self._calculate_autocorrelation_features(close))
        
        # Regime detection features
        regime_features = self._calculate_regime_features(close, high, low)
        features.update(regime_features)
        
        # Fractal features
        features.update(self._calculate_fractal_features(close))
        
        return features
    
    def _calculate_return_features(self, close: np.ndarray) -> Dict[str, float]:
        """Calculate return-based features."""
        features = {}
        
        # Log returns
        log_returns = np.diff(np.log(close))
        
        for window in self.statistical_windows:
            if len(log_returns) >= window:
                window_returns = log_returns[-window:]
                
                # Mean return
                features[f'mean_return_{window}'] = np.mean(window_returns)
                
                # Volatility (standard deviation)
                features[f'volatility_{window}'] = np.std(window_returns)
                
                # Skewness
                features[f'skewness_{window}'] = stats.skew(window_returns)
                
                # Kurtosis
                features[f'kurtosis_{window}'] = stats.kurtosis(window_returns)
                
                # Sharpe ratio (simplified)
                if np.std(window_returns) > 0:
                    features[f'sharpe_ratio_{window}'] = np.mean(window_returns) / np.std(window_returns)
                else:
                    features[f'sharpe_ratio_{window}'] = 0.0
        
        return features
    
    def _calculate_moment_features(self, close: np.ndarray) -> Dict[str, float]:
        """Calculate statistical moment features."""
        features = {}
        
        for window in self.statistical_windows:
            if len(close) >= window:
                window_data = close[-window:]
                
                # Central moments
                mean_val = np.mean(window_data)
                features[f'central_moment_2_{window}'] = np.mean((window_data - mean_val) ** 2)
                features[f'central_moment_3_{window}'] = np.mean((window_data - mean_val) ** 3)
                features[f'central_moment_4_{window}'] = np.mean((window_data - mean_val) ** 4)
                
                # Percentiles
                features[f'percentile_25_{window}'] = np.percentile(window_data, 25)
                features[f'percentile_75_{window}'] = np.percentile(window_data, 75)
                features[f'percentile_90_{window}'] = np.percentile(window_data, 90)
                features[f'percentile_95_{window}'] = np.percentile(window_data, 95)
                
                # Range features
                features[f'price_range_{window}'] = np.max(window_data) - np.min(window_data)
                features[f'price_range_ratio_{window}'] = features[f'price_range_{window}'] / mean_val if mean_val > 0 else 0.0
        
        return features
    
    def _calculate_autocorrelation_features(self, close: np.ndarray) -> Dict[str, float]:
        """Calculate autocorrelation features."""
        features = {}
        
        # Log returns for autocorrelation
        log_returns = np.diff(np.log(close))
        
        for lag in [1, 2, 5, 10]:
            if len(log_returns) > lag + 10:
                autocorr = np.corrcoef(log_returns[:-lag], log_returns[lag:])[0, 1]
                features[f'autocorr_lag_{lag}'] = autocorr if not np.isnan(autocorr) else 0.0
        
        return features
    
    def _calculate_regime_features(self, close: np.ndarray, high: np.ndarray, low: np.ndarray) -> Dict[str, float]:
        """Calculate regime detection features."""
        features = {}
        
        if len(close) < 20:
            return features
        
        # Trend strength
        recent_close = close[-20:]
        trend_slope = np.polyfit(range(len(recent_close)), recent_close, 1)[0]
        features['trend_slope'] = trend_slope
        
        # Trend consistency
        price_changes = np.diff(recent_close)
        trend_consistency = np.sum(np.sign(price_changes) == np.sign(trend_slope)) / len(price_changes)
        features['trend_consistency'] = trend_consistency
        
        # Volatility regime
        returns = np.diff(np.log(close[-20:]))
        current_vol = np.std(returns)
        long_term_vol = np.std(np.diff(np.log(close[-100:]))) if len(close) >= 100 else current_vol
        features['volatility_regime'] = current_vol / long_term_vol if long_term_vol > 0 else 1.0
        
        # Range-bound vs trending
        price_range = np.max(high[-20:]) - np.min(low[-20:])
        price_movement = abs(close[-1] - close[-20])
        range_ratio = price_movement / price_range if price_range > 0 else 0.0
        features['range_bound_ratio'] = 1.0 - range_ratio
        
        return features
    
    def _calculate_fractal_features(self, close: np.ndarray) -> Dict[str, float]:
        """Calculate fractal dimension features."""
        features = {}
        
        if len(close) < 50:
            return features
        
        # Hurst exponent (simplified)
        def hurst_exponent(ts):
            """Calculate Hurst exponent."""
            lags = range(2, min(20, len(ts) // 4))
            tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        
        try:
            hurst = hurst_exponent(close[-50:])
            features['hurst_exponent'] = hurst if not np.isnan(hurst) else 0.5
        except:
            features['hurst_exponent'] = 0.5
        
        # Fractal dimension (simplified)
        def fractal_dimension(ts):
            """Calculate fractal dimension."""
            n = len(ts)
            if n < 10:
                return 1.0
            
            # Box-counting method (simplified)
            scales = [2, 4, 8, 16]
            counts = []
            
            for scale in scales:
                if scale >= n:
                    continue
                
                boxes = n // scale
                count = 0
                for i in range(boxes):
                    start = i * scale
                    end = min(start + scale, n)
                    box_data = ts[start:end]
                    if len(box_data) > 0:
                        count += 1
                
                counts.append(count)
            
            if len(counts) >= 2:
                poly = np.polyfit(np.log(scales[:len(counts)]), np.log(counts), 1)
                return -poly[0]
            else:
                return 1.0
        
        try:
            fractal_dim = fractal_dimension(close[-50:])
            features['fractal_dimension'] = fractal_dim if not np.isnan(fractal_dim) else 1.0
        except:
            features['fractal_dimension'] = 1.0
        
        return features

class AdvancedFeaturePipeline:
    """Main pipeline for advanced feature engineering."""
    
    def __init__(self):
        self.technical_engine = TechnicalIndicatorEngine()
        self.statistical_engine = StatisticalFeatureEngine()
        self.feature_cache = {}
        self.feature_importance = {}
        
    def engineer_features(self, data: pd.DataFrame, 
                         additional_data: Dict[str, Any] = None) -> FeatureSet:
        """Engineer comprehensive feature set."""
        
        features = {}
        feature_names = []
        feature_categories = {
            'technical': [],
            'statistical': [],
            'microstructure': [],
            'market_regime': [],
            'custom': []
        }
        
        # Technical indicators
        technical_features = self.technical_engine.calculate_all_indicators(data)
        features.update(technical_features)
        feature_categories['technical'] = list(technical_features.keys())
        feature_names.extend(list(technical_features.keys()))
        
        # Statistical features
        statistical_features = self.statistical_engine.calculate_statistical_features(data)
        features.update(statistical_features)
        feature_categories['statistical'] = list(statistical_features.keys())
        feature_names.extend(list(statistical_features.keys()))
        
        # Microstructure features (if order book data available)
        if additional_data and 'order_book' in additional_data:
            microstructure_features = self._calculate_microstructure_features(
                additional_data['order_book']
            )
            features.update(microstructure_features)
            feature_categories['microstructure'] = list(microstructure_features.keys())
            feature_names.extend(list(microstructure_features.keys()))
        
        # Market regime features
        regime_features = self._calculate_market_regime_features(data, additional_data)
        features.update(regime_features)
        feature_categories['market_regime'] = list(regime_features.keys())
        feature_names.extend(list(regime_features.keys()))
        
        # Custom features
        if additional_data:
            custom_features = self._calculate_custom_features(data, additional_data)
            features.update(custom_features)
            feature_categories['custom'] = list(custom_features.keys())
            feature_names.extend(list(custom_features.keys()))
        
        # Feature metadata
        metadata = {
            'total_features': len(features),
            'feature_categories': {k: len(v) for k, v in feature_categories.items()},
            'data_length': len(data),
            'timestamp': datetime.now()
        }
        
        return FeatureSet(
            features=features,
            feature_names=feature_names,
            feature_categories=feature_categories,
            timestamp=datetime.now(),
            metadata=metadata
        )
    
    def _calculate_microstructure_features(self, order_book_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate microstructure features from order book data."""
        features = {}
        
        if len(order_book_data) == 0:
            return features
        
        # Spread features
        if 'spread' in order_book_data.columns:
            features['avg_spread'] = order_book_data['spread'].mean()
            features['spread_volatility'] = order_book_data['spread'].std()
            features['max_spread'] = order_book_data['spread'].max()
            features['min_spread'] = order_book_data['spread'].min()
        
        # Depth features
        if 'depth' in order_book_data.columns:
            features['avg_depth'] = order_book_data['depth'].mean()
            features['depth_volatility'] = order_book_data['depth'].std()
            features['depth_imbalance'] = (order_book_data['depth'].max() - order_book_data['depth'].min()) / order_book_data['depth'].mean() if order_book_data['depth'].mean() > 0 else 0.0
        
        # Price impact features
        if 'price_impact' in order_book_data.columns:
            features['avg_price_impact'] = order_book_data['price_impact'].mean()
            features['price_impact_volatility'] = order_book_data['price_impact'].std()
        
        return features
    
    def _calculate_market_regime_features(self, data: pd.DataFrame, 
                                        additional_data: Dict[str, Any] = None) -> Dict[str, float]:
        """Calculate market regime features."""
        features = {}
        
        if len(data) < 20:
            return features
        
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        
        # Volatility regime
        returns = np.diff(np.log(close[-20:]))
        current_vol = np.std(returns)
        features['market_volatility_regime'] = current_vol
        
        # Trend regime
        trend_slope = np.polyfit(range(len(close[-20:])), close[-20:], 1)[0]
        features['trend_regime'] = 1.0 if trend_slope > 0 else -1.0
        
        # Range regime
        price_range = np.max(high[-20:]) - np.min(low[-20:])
        price_movement = abs(close[-1] - close[-20])
        features['range_regime'] = price_movement / price_range if price_range > 0 else 0.0
        
        # Volume regime (if available)
        if 'volume' in data.columns:
            volume = data['volume'].values
            if len(volume) >= 20:
                current_volume = np.mean(volume[-5:])
                avg_volume = np.mean(volume[-20:])
                features['volume_regime'] = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # External regime data
        if additional_data and 'market_regime' in additional_data:
            regime_data = additional_data['market_regime']
            features['external_regime'] = regime_data.get('regime_score', 0.0)
            features['regime_confidence'] = regime_data.get('confidence', 0.0)
        
        return features
    
    def _calculate_custom_features(self, data: pd.DataFrame, 
                                 additional_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate custom features from additional data."""
        features = {}
        
        # News sentiment features
        if 'news_sentiment' in additional_data:
            sentiment_data = additional_data['news_sentiment']
            features['news_sentiment_score'] = sentiment_data.get('score', 0.0)
            features['news_sentiment_volume'] = sentiment_data.get('volume', 0.0)
            features['news_sentiment_volatility'] = sentiment_data.get('volatility', 0.0)
        
        # Economic indicators
        if 'economic_indicators' in additional_data:
            econ_data = additional_data['economic_indicators']
            features['economic_momentum'] = econ_data.get('momentum', 0.0)
            features['economic_volatility'] = econ_data.get('volatility', 0.0)
        
        # Market breadth
        if 'market_breadth' in additional_data:
            breadth_data = additional_data['market_breadth']
            features['advance_decline_ratio'] = breadth_data.get('advance_decline_ratio', 0.0)
            features['new_highs_lows_ratio'] = breadth_data.get('new_highs_lows_ratio', 0.0)
        
        return features
    
    def get_feature_importance(self, feature_set: FeatureSet) -> Dict[str, float]:
        """Get feature importance scores."""
        # This would typically be calculated using a trained model
        # For now, return equal importance
        importance = {}
        if len(feature_set.feature_names) > 0:
            base_importance = 1.0 / len(feature_set.feature_names)
            for feature_name in feature_set.feature_names:
                importance[feature_name] = base_importance
        
        return importance
    
    def select_top_features(self, feature_set: FeatureSet, top_k: int = 50) -> List[str]:
        """Select top K most important features."""
        importance = self.get_feature_importance(feature_set)
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        return [feature[0] for feature in sorted_features[:top_k]]

# Global instance
_feature_pipeline = None

def get_feature_pipeline() -> AdvancedFeaturePipeline:
    """Get the global feature pipeline instance."""
    global _feature_pipeline
    if _feature_pipeline is None:
        _feature_pipeline = AdvancedFeaturePipeline()
    return _feature_pipeline