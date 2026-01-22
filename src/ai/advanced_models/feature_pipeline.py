"""
Advanced Feature Pipeline for AI Models

This module provides advanced feature engineering and pipeline optimization
for all AI models including deep learning, time series, and NLP components.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import logging
from datetime import datetime, timedelta
from collections import defaultdict
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, SelectPercentile, RFE
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
import warnings

# Feature engineering libraries
try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("Scikit-learn not available. Some features will be limited.")

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    warnings.warn("TA-Lib not available. Technical indicators will be limited.")

logger = logging.getLogger(__name__)

class AdvancedFeaturePipeline:
    """
    Advanced feature pipeline for comprehensive feature engineering.
    """
    
    def __init__(
        self,
        pipeline_name: str = "advanced_feature_pipeline",
        enable_technical_indicators: bool = True,
        enable_statistical_features: bool = True,
        enable_nlp_features: bool = True,
        enable_time_series_features: bool = True
    ):
        """
        Initialize advanced feature pipeline.
        
        Args:
            pipeline_name: Name for the pipeline
            enable_technical_indicators: Whether to enable technical indicators
            enable_statistical_features: Whether to enable statistical features
            enable_nlp_features: Whether to enable NLP features
            enable_time_series_features: Whether to enable time series features
        """
        self.pipeline_name = pipeline_name
        self.enable_technical_indicators = enable_technical_indicators
        self.enable_statistical_features = enable_statistical_features
        self.enable_nlp_features = enable_nlp_features
        self.enable_time_series_features = enable_time_series_features
        
        # Feature transformers
        self.scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        
        # Feature selectors
        self.feature_selectors = {
            'select_k_best': None,
            'select_percentile': None,
            'rfe': None
        }
        
        # Dimensionality reduction
        self.dimensionality_reducers = {
            'pca': None,
            'ica': None,
            'tsne': None
        }
        
        # NLP vectorizers
        if self.enable_nlp_features and SKLEARN_AVAILABLE:
            self.nlp_vectorizers = {
                'tfidf': TfidfVectorizer(max_features=1000, stop_words='english'),
                'count': CountVectorizer(max_features=1000, stop_words='english')
            }
        else:
            self.nlp_vectorizers = {}
        
        # Feature engineering history
        self.feature_history = []
        
        # Feature optimizer
        self.feature_optimizer = FeatureOptimizer()
        
        # Performance metrics
        self.performance_metrics = {
            'total_features_created': 0,
            'total_pipelines_executed': 0,
            'average_execution_time': 0.0,
            'feature_type_counts': {}
        }
        
        logger.info(f"Advanced Feature Pipeline initialized: {pipeline_name}")
    
    def create_features(
        self,
        data: Union[pd.DataFrame, Dict[str, Any]],
        feature_types: Optional[List[str]] = None,
        target_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create comprehensive features from input data.
        
        Args:
            data: Input data (DataFrame or dictionary)
            feature_types: List of feature types to create
            target_column: Target column for supervised feature selection
            
        Returns:
            Dictionary containing created features and metadata
        """
        start_time = datetime.now()
        
        if feature_types is None:
            feature_types = ['technical', 'statistical', 'time_series']
            if self.enable_nlp_features:
                feature_types.append('nlp')
        
        # Convert data to DataFrame if needed
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()
        
        results = {
            'pipeline_name': self.pipeline_name,
            'feature_types': feature_types,
            'input_shape': df.shape,
            'start_time': start_time.isoformat(),
            'features': {},
            'feature_metadata': {}
        }
        
        try:
            # Create different types of features
            for feature_type in feature_types:
                if feature_type == 'technical' and self.enable_technical_indicators:
                    technical_features = self._create_technical_features(df)
                    results['features']['technical'] = technical_features
                    results['feature_metadata']['technical'] = {
                        'count': len(technical_features.columns) if isinstance(technical_features, pd.DataFrame) else 0,
                        'type': 'technical_indicators'
                    }
                
                elif feature_type == 'statistical' and self.enable_statistical_features:
                    statistical_features = self._create_statistical_features(df)
                    results['features']['statistical'] = statistical_features
                    results['feature_metadata']['statistical'] = {
                        'count': len(statistical_features.columns) if isinstance(statistical_features, pd.DataFrame) else 0,
                        'type': 'statistical_features'
                    }
                
                elif feature_type == 'time_series' and self.enable_time_series_features:
                    ts_features = self._create_time_series_features(df)
                    results['features']['time_series'] = ts_features
                    results['feature_metadata']['time_series'] = {
                        'count': len(ts_features.columns) if isinstance(ts_features, pd.DataFrame) else 0,
                        'type': 'time_series_features'
                    }
                
                elif feature_type == 'nlp' and self.enable_nlp_features:
                    nlp_features = self._create_nlp_features(df)
                    results['features']['nlp'] = nlp_features
                    results['feature_metadata']['nlp'] = {
                        'count': len(nlp_features.columns) if isinstance(nlp_features, pd.DataFrame) else 0,
                        'type': 'nlp_features'
                    }
            
            # Combine all features
            combined_features = self._combine_features(results['features'])
            results['combined_features'] = combined_features
            results['combined_shape'] = combined_features.shape
            
            # Feature selection if target is provided
            if target_column and target_column in df.columns:
                selected_features = self._select_features(combined_features, df[target_column])
                results['selected_features'] = selected_features
                results['selected_shape'] = selected_features.shape
            
            # Update performance metrics
            total_features = sum(meta['count'] for meta in results['feature_metadata'].values())
            self.performance_metrics['total_features_created'] += total_features
            self.performance_metrics['total_pipelines_executed'] += 1
            
            # Update average execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            current_avg = self.performance_metrics['average_execution_time']
            total_pipelines = self.performance_metrics['total_pipelines_executed']
            self.performance_metrics['average_execution_time'] = (
                (current_avg * (total_pipelines - 1) + execution_time) / total_pipelines
            )
            
            results['execution_time'] = execution_time
            results['end_time'] = datetime.now().isoformat()
            
            # Store in history
            self.feature_history.append(results)
            
            logger.info(f"Feature pipeline completed: {total_features} features created")
            
        except Exception as e:
            logger.error(f"Feature pipeline error: {e}")
            results['error'] = str(e)
            results['execution_time'] = (datetime.now() - start_time).total_seconds()
        
        return results
    
    def _create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicator features."""
        try:
            features = pd.DataFrame(index=df.index)
            
            # Ensure required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            available_columns = [col for col in required_columns if col in df.columns]
            
            if len(available_columns) < 4:  # Need at least OHLC
                logger.warning("Insufficient data for technical indicators")
                return features
            
            # Price-based indicators
            if 'close' in df.columns:
                close_prices = df['close'].values
                
                # Moving averages
                for period in [5, 10, 20, 50]:
                    if len(close_prices) >= period:
                        if TALIB_AVAILABLE:
                            features[f'sma_{period}'] = talib.SMA(close_prices, timeperiod=period)
                        else:
                            features[f'sma_{period}'] = df['close'].rolling(window=period).mean()
                
                # RSI
                if TALIB_AVAILABLE and len(close_prices) >= 14:
                    features['rsi'] = talib.RSI(close_prices, timeperiod=14)
                
                # MACD
                if TALIB_AVAILABLE and len(close_prices) >= 26:
                    macd, macd_signal, macd_hist = talib.MACD(close_prices)
                    features['macd'] = macd
                    features['macd_signal'] = macd_signal
                    features['macd_histogram'] = macd_hist
            
            # Volume-based indicators
            if 'volume' in df.columns and 'close' in df.columns:
                volume = df['volume'].values
                close_prices = df['close'].values
                
                # Volume moving average
                for period in [5, 10, 20]:
                    if len(volume) >= period:
                        features[f'volume_sma_{period}'] = df['volume'].rolling(window=period).mean()
                
                # Price-volume trend
                if len(volume) > 1 and len(close_prices) > 1:
                    price_change = np.diff(close_prices)
                    features['price_volume_trend'] = np.concatenate([[0], price_change * volume[1:]])
            
            # Volatility indicators
            if all(col in df.columns for col in ['high', 'low', 'close']):
                high = df['high'].values
                low = df['low'].values
                close = df['close'].values
                
                # ATR (Average True Range)
                if TALIB_AVAILABLE and len(high) >= 14:
                    features['atr'] = talib.ATR(high, low, close, timeperiod=14)
                
                # Bollinger Bands
                if TALIB_AVAILABLE and len(close) >= 20:
                    bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20)
                    features['bb_upper'] = bb_upper
                    features['bb_middle'] = bb_middle
                    features['bb_lower'] = bb_lower
                    features['bb_width'] = (bb_upper - bb_lower) / bb_middle
                    features['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower)
            
            # Momentum indicators
            if 'close' in df.columns:
                close_prices = df['close'].values
                
                # Rate of change
                for period in [5, 10, 20]:
                    if len(close_prices) >= period + 1:
                        features[f'roc_{period}'] = ((close_prices[period:] - close_prices[:-period]) / close_prices[:-period])
                        # Pad with zeros for alignment
                        features[f'roc_{period}'] = np.concatenate([np.zeros(period), features[f'roc_{period}']])
            
            return features.fillna(0)
            
        except Exception as e:
            logger.error(f"Technical features creation error: {e}")
            return pd.DataFrame(index=df.index)
    
    def _create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features."""
        try:
            features = pd.DataFrame(index=df.index)
            
            # Numeric columns only
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            for column in numeric_columns:
                values = df[column].values
                
                # Basic statistics
                features[f'{column}_mean'] = np.mean(values)
                features[f'{column}_std'] = np.std(values)
                features[f'{column}_min'] = np.min(values)
                features[f'{column}_max'] = np.max(values)
                features[f'{column}_median'] = np.median(values)
                features[f'{column}_skewness'] = self._calculate_skewness(values)
                features[f'{column}_kurtosis'] = self._calculate_kurtosis(values)
                
                # Rolling statistics
                for window in [5, 10, 20]:
                    if len(values) >= window:
                        rolling_mean = pd.Series(values).rolling(window=window).mean()
                        rolling_std = pd.Series(values).rolling(window=window).std()
                        features[f'{column}_rolling_mean_{window}'] = rolling_mean
                        features[f'{column}_rolling_std_{window}'] = rolling_std
                        features[f'{column}_rolling_zscore_{window}'] = (values - rolling_mean) / rolling_std
            
            return features.fillna(0)
            
        except Exception as e:
            logger.error(f"Statistical features creation error: {e}")
            return pd.DataFrame(index=df.index)
    
    def _create_time_series_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time series features."""
        try:
            features = pd.DataFrame(index=df.index)
            
            # Time-based features
            if 'date' in df.columns or 'timestamp' in df.columns:
                time_column = 'date' if 'date' in df.columns else 'timestamp'
                time_series = pd.to_datetime(df[time_column])
                
                # Time components
                features['hour'] = time_series.dt.hour
                features['day_of_week'] = time_series.dt.dayofweek
                features['day_of_month'] = time_series.dt.day
                features['month'] = time_series.dt.month
                features['quarter'] = time_series.dt.quarter
                features['year'] = time_series.dt.year
                
                # Cyclical encoding
                features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
                features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
                features['day_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
                features['day_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
                features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
                features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
            
            # Lag features
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for column in numeric_columns:
                for lag in [1, 2, 3, 5, 10]:
                    if len(df) > lag:
                        features[f'{column}_lag_{lag}'] = df[column].shift(lag)
                
                # Difference features
                features[f'{column}_diff_1'] = df[column].diff(1)
                features[f'{column}_diff_2'] = df[column].diff(2)
                features[f'{column}_pct_change'] = df[column].pct_change()
            
            return features.fillna(0)
            
        except Exception as e:
            logger.error(f"Time series features creation error: {e}")
            return pd.DataFrame(index=df.index)
    
    def _create_nlp_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create NLP features."""
        try:
            features = pd.DataFrame(index=df.index)
            
            # Text columns
            text_columns = df.select_dtypes(include=['object']).columns
            
            for column in text_columns:
                text_data = df[column].astype(str)
                
                # Basic text features
                features[f'{column}_length'] = text_data.str.len()
                features[f'{column}_word_count'] = text_data.str.split().str.len()
                features[f'{column}_sentence_count'] = text_data.str.count(r'[.!?]+')
                features[f'{column}_avg_word_length'] = text_data.str.split().str.join(' ').str.len() / features[f'{column}_word_count']
                
                # Character features
                features[f'{column}_uppercase_ratio'] = text_data.str.count(r'[A-Z]') / features[f'{column}_length']
                features[f'{column}_digit_ratio'] = text_data.str.count(r'[0-9]') / features[f'{column}_length']
                features[f'{column}_special_char_ratio'] = text_data.str.count(r'[^a-zA-Z0-9\s]') / features[f'{column}_length']
                
                # TF-IDF features (if sklearn is available)
                if self.nlp_vectorizers and len(text_data) > 1:
                    try:
                        tfidf_vectorizer = self.nlp_vectorizers['tfidf']
                        tfidf_matrix = tfidf_vectorizer.fit_transform(text_data)
                        
                        # Get top features
                        feature_names = tfidf_vectorizer.get_feature_names_out()
                        tfidf_df = pd.DataFrame(
                            tfidf_matrix.toarray(),
                            columns=[f'{column}_tfidf_{name}' for name in feature_names],
                            index=df.index
                        )
                        
                        # Select top features to avoid too many columns
                        top_features = tfidf_df.sum().nlargest(10).index
                        features = pd.concat([features, tfidf_df[top_features]], axis=1)
                        
                    except Exception as e:
                        logger.warning(f"TF-IDF feature creation failed for {column}: {e}")
            
            return features.fillna(0)
            
        except Exception as e:
            logger.error(f"NLP features creation error: {e}")
            return pd.DataFrame(index=df.index)
    
    def _combine_features(self, features_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine all feature DataFrames."""
        try:
            if not features_dict:
                return pd.DataFrame()
            
            # Get common index
            common_index = None
            for features in features_dict.values():
                if isinstance(features, pd.DataFrame) and not features.empty:
                    if common_index is None:
                        common_index = features.index
                    else:
                        common_index = common_index.intersection(features.index)
            
            if common_index is None or len(common_index) == 0:
                return pd.DataFrame()
            
            # Combine features
            combined_features = pd.DataFrame(index=common_index)
            
            for feature_type, features in features_dict.items():
                if isinstance(features, pd.DataFrame) and not features.empty:
                    # Align with common index
                    aligned_features = features.reindex(common_index)
                    combined_features = pd.concat([combined_features, aligned_features], axis=1)
            
            return combined_features.fillna(0)
            
        except Exception as e:
            logger.error(f"Feature combination error: {e}")
            return pd.DataFrame()
    
    def _select_features(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        method: str = 'select_k_best',
        k: int = 50
    ) -> pd.DataFrame:
        """Select most relevant features."""
        try:
            if not SKLEARN_AVAILABLE:
                logger.warning("Scikit-learn not available for feature selection")
                return features
            
            # Remove any infinite or NaN values
            features_clean = features.replace([np.inf, -np.inf], np.nan).fillna(0)
            target_clean = target.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            if method == 'select_k_best':
                selector = SelectKBest(k=min(k, features_clean.shape[1]))
            elif method == 'select_percentile':
                selector = SelectPercentile(percentile=50)
            else:
                logger.warning(f"Unknown feature selection method: {method}")
                return features_clean
            
            # Fit and transform
            selected_features = selector.fit_transform(features_clean, target_clean)
            selected_columns = features_clean.columns[selector.get_support()]
            
            return pd.DataFrame(selected_features, columns=selected_columns, index=features_clean.index)
            
        except Exception as e:
            logger.error(f"Feature selection error: {e}")
            return features
    
    def _calculate_skewness(self, values: np.ndarray) -> float:
        """Calculate skewness of values."""
        try:
            mean_val = np.mean(values)
            std_val = np.std(values)
            if std_val == 0:
                return 0.0
            return np.mean(((values - mean_val) / std_val) ** 3)
        except:
            return 0.0
    
    def _calculate_kurtosis(self, values: np.ndarray) -> float:
        """Calculate kurtosis of values."""
        try:
            mean_val = np.mean(values)
            std_val = np.std(values)
            if std_val == 0:
                return 0.0
            return np.mean(((values - mean_val) / std_val) ** 4) - 3
        except:
            return 0.0
    
    def get_feature_statistics(self) -> Dict[str, Any]:
        """Get feature pipeline statistics."""
        return {
            'pipeline_name': self.pipeline_name,
            'total_features_created': self.performance_metrics['total_features_created'],
            'total_pipelines_executed': self.performance_metrics['total_pipelines_executed'],
            'average_execution_time': self.performance_metrics['average_execution_time'],
            'enabled_features': {
                'technical_indicators': self.enable_technical_indicators,
                'statistical_features': self.enable_statistical_features,
                'nlp_features': self.enable_nlp_features,
                'time_series_features': self.enable_time_series_features
            },
            'history_size': len(self.feature_history)
        }


class FeatureOptimizer:
    """
    Feature optimization for improving feature quality and model performance.
    """
    
    def __init__(
        self,
        optimizer_name: str = "feature_optimizer",
        optimization_methods: List[str] = None
    ):
        """
        Initialize feature optimizer.
        
        Args:
            optimizer_name: Name for the optimizer
            optimization_methods: List of optimization methods to use
        """
        self.optimizer_name = optimizer_name
        self.optimization_methods = optimization_methods or [
            'correlation_removal',
            'variance_threshold',
            'mutual_information',
            'recursive_elimination'
        ]
        
        # Optimization history
        self.optimization_history = []
        
        logger.info(f"Feature Optimizer initialized: {optimizer_name}")
    
    def optimize_features(
        self,
        features: pd.DataFrame,
        target: Optional[pd.Series] = None,
        optimization_methods: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Optimize features for better model performance.
        
        Args:
            features: Feature DataFrame
            target: Target variable for supervised optimization
            optimization_methods: List of optimization methods to apply
            
        Returns:
            Optimization results
        """
        if optimization_methods is None:
            optimization_methods = self.optimization_methods
        
        results = {
            'optimizer_name': self.optimizer_name,
            'input_shape': features.shape,
            'optimization_methods': optimization_methods,
            'optimization_strategies': optimization_methods,  # ADD THIS
            'start_time': datetime.now().isoformat(),
            'optimized_features': features.copy(),
            'optimization_steps': []
        }
        
        try:
            for method in optimization_methods:
                step_result = self._apply_optimization_method(
                    results['optimized_features'], target, method
                )
                results['optimization_steps'].append(step_result)
                
                if 'optimized_features' in step_result:
                    results['optimized_features'] = step_result['optimized_features']
            
            results['final_shape'] = results['optimized_features'].shape
            results['features_removed'] = results['input_shape'][1] - results['final_shape'][1]
            results['end_time'] = datetime.now().isoformat()
            
            # Store in history
            self.optimization_history.append(results)
            
            logger.info(f"Feature optimization completed: {results['features_removed']} features removed")
            
        except Exception as e:
            logger.error(f"Feature optimization error: {e}")
            results['error'] = str(e)
        
        return results
    
    def _apply_optimization_method(
        self,
        features: pd.DataFrame,
        target: Optional[pd.Series],
        method: str
    ) -> Dict[str, Any]:
        """Apply a specific optimization method."""
        try:
            if method == 'correlation_removal':
                return self._remove_correlated_features(features)
            elif method == 'variance_threshold':
                return self._remove_low_variance_features(features)
            elif method == 'mutual_information' and target is not None:
                return self._select_by_mutual_information(features, target)
            elif method == 'recursive_elimination' and target is not None:
                return self._recursive_feature_elimination(features, target)
            else:
                return {'method': method, 'status': 'skipped', 'reason': 'requirements_not_met'}
                
        except Exception as e:
            return {'method': method, 'error': str(e)}
    
    def _remove_correlated_features(self, features: pd.DataFrame, threshold: float = 0.95) -> Dict[str, Any]:
        """Remove highly correlated features."""
        try:
            # Calculate correlation matrix
            corr_matrix = features.corr().abs()
            
            # Find pairs of highly correlated features
            upper_tri = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            # Find features to drop
            to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
            
            # Remove correlated features
            optimized_features = features.drop(columns=to_drop)
            
            return {
                'method': 'correlation_removal',
                'threshold': threshold,
                'features_removed': len(to_drop),
                'removed_features': to_drop,
                'optimized_features': optimized_features,
                'optimization_strategies': ['correlation_removal']  # ADD THIS
            }
            
        except Exception as e:
            return {'method': 'correlation_removal', 'error': str(e)}
    
    def _remove_low_variance_features(self, features: pd.DataFrame, threshold: float = 0.01) -> Dict[str, Any]:
        """Remove features with low variance."""
        try:
            # Calculate variance for each feature
            variances = features.var()
            
            # Find features with low variance
            low_variance_features = variances[variances < threshold].index.tolist()
            
            # Remove low variance features
            optimized_features = features.drop(columns=low_variance_features)
            
            return {
                'method': 'variance_threshold',
                'threshold': threshold,
                'features_removed': len(low_variance_features),
                'removed_features': low_variance_features,
                'optimized_features': optimized_features,
                'optimization_strategies': ['low_variance_removal']  # ADD THIS
            }
            
        except Exception as e:
            return {'method': 'variance_threshold', 'error': str(e)}
    
    def _select_by_mutual_information(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        k: int = 50
    ) -> Dict[str, Any]:
        """Select features based on mutual information with target."""
        try:
            if not SKLEARN_AVAILABLE:
                return {'method': 'mutual_information', 'error': 'scikit-learn not available'}
            
            from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
            
            # Determine if target is continuous or categorical
            if target.dtype in ['object', 'category'] or len(target.unique()) < 10:
                # Classification
                mi_scores = mutual_info_classif(features, target)
            else:
                # Regression
                mi_scores = mutual_info_regression(features, target)
            
            # Select top k features
            k = min(k, len(features.columns))
            top_k_indices = np.argsort(mi_scores)[-k:]
            selected_features = features.iloc[:, top_k_indices]
            
            return {
                'method': 'mutual_information',
                'k': k,
                'features_selected': k,
                'selected_features': features.columns[top_k_indices].tolist(),
                'mi_scores': dict(zip(features.columns, mi_scores)),
                'optimized_features': selected_features,
                'optimization_strategies': ['mutual_information_selection']  # ADD THIS
            }
            
        except Exception as e:
            return {'method': 'mutual_information', 'error': str(e)}
    
    def _recursive_feature_elimination(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        n_features: int = 50
    ) -> Dict[str, Any]:
        """Perform recursive feature elimination."""
        try:
            if not SKLEARN_AVAILABLE:
                return {'method': 'recursive_elimination', 'error': 'scikit-learn not available'}
            
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
            from sklearn.feature_selection import RFE
            
            # Determine if target is continuous or categorical
            if target.dtype in ['object', 'category'] or len(target.unique()) < 10:
                # Classification
                estimator = RandomForestClassifier(n_estimators=10, random_state=42)
            else:
                # Regression
                estimator = RandomForestRegressor(n_estimators=10, random_state=42)
            
            # Perform RFE
            n_features = min(n_features, len(features.columns))
            rfe = RFE(estimator=estimator, n_features_to_select=n_features)
            rfe.fit(features, target)
            
            # Get selected features
            selected_features = features.iloc[:, rfe.support_]
            
            return {
                'method': 'recursive_elimination',
                'n_features': n_features,
                'features_selected': n_features,
                'selected_features': features.columns[rfe.support_].tolist(),
                'feature_ranking': dict(zip(features.columns, rfe.ranking_)),
                'optimized_features': selected_features
            }
            
        except Exception as e:
            return {'method': 'recursive_elimination', 'error': str(e)}
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get feature optimization statistics."""
        if not self.optimization_history:
            return {}
        
        # Count optimization methods used
        method_counts = defaultdict(int)
        total_features_removed = 0
        
        for optimization in self.optimization_history:
            for step in optimization.get('optimization_steps', []):
                method = step.get('method', 'unknown')
                method_counts[method] += 1
                
                if 'features_removed' in step:
                    total_features_removed += step['features_removed']
        
        return {
            'optimizer_name': self.optimizer_name,
            'total_optimizations': len(self.optimization_history),
            'method_usage_counts': dict(method_counts),
            'total_features_removed': total_features_removed,
            'average_features_removed': total_features_removed / len(self.optimization_history) if self.optimization_history else 0
        }

