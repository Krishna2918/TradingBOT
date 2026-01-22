"""
AI Features Module - Technical Indicators and Feature Engineering

This module provides technical analysis features and indicators for stock analysis.
All features are persisted to DuckDB for caching and performance.
"""

import logging
import numpy as np
import pandas as pd
import duckdb
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import os

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """Technical analysis indicators for stock data."""
    
    @staticmethod
    def sma(data: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average."""
        return data.rolling(window=window).mean()
    
    @staticmethod
    def ema(data: pd.Series, window: int) -> pd.Series:
        """Exponential Moving Average."""
        return data.ewm(span=window).mean()
    
    @staticmethod
    def rsi(data: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index."""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD (Moving Average Convergence Divergence)."""
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(data: pd.Series, window: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands."""
        sma = TechnicalIndicators.sma(data, window)
        std = data.rolling(window=window).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_window: int = 14, d_window: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator."""
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        return k_percent, d_percent
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Average True Range."""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=window).mean()
    
    @staticmethod
    def volume_profile(volume: pd.Series, price: pd.Series, bins: int = 20) -> Dict:
        """Volume Profile Analysis."""
        # Create price bins
        price_min, price_max = price.min(), price.max()
        bin_edges = np.linspace(price_min, price_max, bins + 1)
        
        # Assign prices to bins
        price_bins = pd.cut(price, bins=bin_edges, labels=False)
        
        # Calculate volume per bin
        volume_profile = volume.groupby(price_bins).sum()
        
        # Find POC (Point of Control) - highest volume bin
        poc_bin = volume_profile.idxmax()
        poc_price = bin_edges[poc_bin]
        
        return {
            'poc_price': poc_price,
            'volume_profile': volume_profile.to_dict(),
            'total_volume': volume.sum(),
            'price_range': (price_min, price_max)
        }

class FeatureEngine:
    """Feature engineering engine with DuckDB persistence."""
    
    def __init__(self, db_path: str = "data/market_data.duckdb"):
        """Initialize feature engine with DuckDB connection."""
        self.db_path = db_path
        self.conn = None
        self._ensure_db_exists()
        self._create_schema()
    
    def _ensure_db_exists(self):
        """Ensure database directory exists."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
    
    def _create_schema(self):
        """Create database schema for features."""
        self.conn = duckdb.connect(self.db_path)
        
        # Features table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS features (
                symbol VARCHAR,
                date DATE,
                feature_name VARCHAR,
                feature_value DOUBLE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, date, feature_name)
            )
        """)
        
        # Technical indicators table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS technical_indicators (
                symbol VARCHAR,
                timestamp TIMESTAMP,
                sma_20 DOUBLE,
                sma_50 DOUBLE,
                ema_12 DOUBLE,
                ema_26 DOUBLE,
                rsi_14 DOUBLE,
                macd DOUBLE,
                macd_signal DOUBLE,
                macd_histogram DOUBLE,
                bb_upper DOUBLE,
                bb_middle DOUBLE,
                bb_lower DOUBLE,
                stoch_k DOUBLE,
                stoch_d DOUBLE,
                atr_14 DOUBLE,
                volume_sma_20 DOUBLE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, timestamp)
            )
        """)
        
        # Volume profile table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS volume_profile (
                symbol VARCHAR,
                timestamp TIMESTAMP,
                poc_price DOUBLE,
                total_volume BIGINT,
                price_min DOUBLE,
                price_max DOUBLE,
                volume_profile_json VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, timestamp)
            )
        """)
        
        logger.info("Database schema created successfully")
    
    def calculate_features(self, symbol: str, data: pd.DataFrame) -> Dict:
        """Calculate all technical features for a symbol."""
        if data.empty:
            logger.warning(f"No data available for {symbol}")
            return {}
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            logger.error(f"Missing columns for {symbol}: {missing_cols}")
            return {}
        
        features = {}
        
        try:
            # Price-based indicators
            features['sma_20'] = TechnicalIndicators.sma(data['close'], 20)
            features['sma_50'] = TechnicalIndicators.sma(data['close'], 50)
            features['ema_12'] = TechnicalIndicators.ema(data['close'], 12)
            features['ema_26'] = TechnicalIndicators.ema(data['close'], 26)
            features['rsi_14'] = TechnicalIndicators.rsi(data['close'], 14)
            
            # MACD
            macd, signal, histogram = TechnicalIndicators.macd(data['close'])
            features['macd'] = macd
            features['macd_signal'] = signal
            features['macd_histogram'] = histogram
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(data['close'])
            features['bb_upper'] = bb_upper
            features['bb_middle'] = bb_middle
            features['bb_lower'] = bb_lower
            
            # Stochastic
            stoch_k, stoch_d = TechnicalIndicators.stochastic(
                data['high'], data['low'], data['close']
            )
            features['stoch_k'] = stoch_k
            features['stoch_d'] = stoch_d
            
            # ATR
            features['atr_14'] = TechnicalIndicators.atr(
                data['high'], data['low'], data['close']
            )
            
            # Volume indicators
            features['volume_sma_20'] = TechnicalIndicators.sma(data['volume'], 20)
            
            # Volume profile
            volume_profile = TechnicalIndicators.volume_profile(
                data['volume'], data['close']
            )
            features['volume_profile'] = volume_profile
            
            logger.info(f"Calculated {len(features)} features for {symbol}")
            return features
            
        except Exception as e:
            logger.error(f"Error calculating features for {symbol}: {e}")
            return {}
    
    def persist_features(self, symbol: str, features: Dict, dates: pd.DatetimeIndex):
        """Persist calculated features to DuckDB."""
        if not features:
            return
        
        try:
            # Prepare technical indicators data
            tech_data = []
            for i, date in enumerate(dates):
                row = {
                    'symbol': symbol,
                    'timestamp': date,
                    'sma_20': features.get('sma_20', {}).get(i, None),
                    'sma_50': features.get('sma_50', {}).get(i, None),
                    'ema_12': features.get('ema_12', {}).get(i, None),
                    'ema_26': features.get('ema_26', {}).get(i, None),
                    'rsi_14': features.get('rsi_14', {}).get(i, None),
                    'macd': features.get('macd', {}).get(i, None),
                    'macd_signal': features.get('macd_signal', {}).get(i, None),
                    'macd_histogram': features.get('macd_histogram', {}).get(i, None),
                    'bb_upper': features.get('bb_upper', {}).get(i, None),
                    'bb_middle': features.get('bb_middle', {}).get(i, None),
                    'bb_lower': features.get('bb_lower', {}).get(i, None),
                    'stoch_k': features.get('stoch_k', {}).get(i, None),
                    'stoch_d': features.get('stoch_d', {}).get(i, None),
                    'atr_14': features.get('atr_14', {}).get(i, None),
                    'volume_sma_20': features.get('volume_sma_20', {}).get(i, None)
                }
                tech_data.append(row)
            
            # Insert technical indicators
            if tech_data:
                tech_df = pd.DataFrame(tech_data)
                self.conn.execute("""
                    INSERT OR REPLACE INTO technical_indicators 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, tech_df.values.tolist())
            
            # Insert volume profile
            if 'volume_profile' in features:
                vp = features['volume_profile']
                self.conn.execute("""
                    INSERT OR REPLACE INTO volume_profile 
                    VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, [
                    symbol,
                    dates[-1],
                    vp.get('poc_price'),
                    vp.get('total_volume'),
                    vp.get('price_range', (None, None))[0],
                    vp.get('price_range', (None, None))[1],
                    str(vp.get('volume_profile', {}))
                ])
            
            logger.info(f"Persisted features for {symbol} to database")
            
        except Exception as e:
            logger.error(f"Error persisting features for {symbol}: {e}")
    
    def get_feature_matrix(self, symbols: List[str], feature_names: List[str], 
                          start_date: Optional[datetime] = None, 
                          end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Get feature matrix for multiple symbols."""
        try:
            # Build query
            where_conditions = []
            params = []
            
            if symbols:
                placeholders = ','.join(['?' for _ in symbols])
                where_conditions.append(f"symbol IN ({placeholders})")
                params.extend(symbols)
            
            if start_date:
                where_conditions.append("date >= ?")
                params.append(start_date.date())
            
            if end_date:
                where_conditions.append("date <= ?")
                params.append(end_date.date())
            
            where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
            
            # Get technical indicators
            query = f"""
                SELECT * FROM technical_indicators 
                WHERE {where_clause}
                ORDER BY symbol, date
            """
            
            result = self.conn.execute(query, params).fetchdf()
            
            if result.empty:
                logger.warning("No feature data found")
                return pd.DataFrame()
            
            # Pivot to get feature matrix
            feature_matrix = result.pivot_table(
                index=['symbol', 'date'],
                columns='feature_name',
                values='feature_value',
                aggfunc='first'
            ).reset_index()
            
            logger.info(f"Retrieved feature matrix: {feature_matrix.shape}")
            return feature_matrix
            
        except Exception as e:
            logger.error(f"Error getting feature matrix: {e}")
            return pd.DataFrame()
    
    def get_latest_features(self, symbol: str) -> Dict:
        """Get latest features for a symbol."""
        try:
            query = """
                SELECT * FROM technical_indicators 
                WHERE symbol = ? 
                ORDER BY timestamp DESC 
                LIMIT 1
            """
            
            result = self.conn.execute(query, [symbol]).fetchdf()
            
            if result.empty:
                logger.warning(f"No features found for {symbol}")
                return {}
            
            # Convert to dictionary
            features = result.iloc[0].to_dict()
            features.pop('symbol', None)
            features.pop('timestamp', None)
            features.pop('created_at', None)
            
            return features
            
        except Exception as e:
            logger.error(f"Error getting latest features for {symbol}: {e}")
            return {}
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

# Global feature engine instance
feature_engine = FeatureEngine()

def get_feature_matrix(symbols: List[str], feature_names: List[str] = None, 
                      start_date: Optional[datetime] = None, 
                      end_date: Optional[datetime] = None) -> pd.DataFrame:
    """Get feature matrix for multiple symbols."""
    return feature_engine.get_feature_matrix(symbols, feature_names, start_date, end_date)

def calculate_and_persist_features(symbol: str, data: pd.DataFrame) -> Dict:
    """Calculate and persist features for a symbol."""
    features = feature_engine.calculate_features(symbol, data)
    if features:
        feature_engine.persist_features(symbol, features, data.index)
    return features

def get_latest_features(symbol: str) -> Dict:
    """Get latest features for a symbol."""
    return feature_engine.get_latest_features(symbol)
