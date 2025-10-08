"""
Comprehensive Data Pipeline - Canadian Market Research Plan
TSX/TSXV, Options, Macro Data, News Sentiment
"""

import logging
import yaml
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import yfinance as yf
import requests
import time
import redis
from pathlib import Path
import duckdb
import json

# Import free APIs integration
try:
    from src.data_services.free_apis_integration import FreeAPIsIntegration
    FREE_APIS_AVAILABLE = True
except ImportError:
    FREE_APIS_AVAILABLE = False
    logger.warning("Free APIs integration not available")

logger = logging.getLogger(__name__)


class ComprehensiveDataPipeline:
    """
    Comprehensive data pipeline for Canadian market research
    """
    
    def __init__(self, config_path: str = "config/data_pipeline_config.yaml"):
        self.config = self._load_config(config_path)
        
        # Initialize data storage
        self.db_path = "data/market_data.duckdb"
        self._init_database()
        
        # Initialize Redis cache
        self.redis_client = self._init_redis()
        
        # Data quality tracking
        self.quality_metrics = {
            'missing_data_rate': 0.0,
            'stale_data_count': 0,
            'api_errors': 0,
            'last_update': None
        }
        
        # Initialize Questrade client with auto-rotation
        try:
            from src.data_pipeline.questrade_client import QuestradeClient
            self.questrade = QuestradeClient(
                allow_trading=False,  # Read-only for data fetching
                practice_mode=True
            )
            logger.info(" Questrade client initialized (auto-rotation enabled)")
        except Exception as e:
            logger.warning(f" Questrade not available: {e}")
            self.questrade = None
        
        # Initialize free APIs integration
        if FREE_APIS_AVAILABLE:
            try:
                self.free_apis = FreeAPIsIntegration(self.config)
                logger.info(" Free APIs integration initialized")
            except Exception as e:
                logger.error(f" Failed to initialize free APIs: {e}")
                self.free_apis = None
        else:
            self.free_apis = None
        
        # Initialize tick data processor
        try:
            from src.data_pipeline.tick_data_processor import TickDataManager
            self.tick_manager = TickDataManager(self.config)
            logger.info(" Tick data processor initialized")
        except Exception as e:
            logger.error(f" Failed to initialize tick data processor: {e}")
            self.tick_manager = None
        
        # Initialize options chain processor
        try:
            from src.data_pipeline.options_chain_processor import OptionsDataManager
            self.options_manager = OptionsDataManager(self.config)
            logger.info(" Options chain processor initialized")
        except Exception as e:
            logger.error(f" Failed to initialize options processor: {e}")
            self.options_manager = None
        
        # Initialize macro event calendar
        try:
            from src.data_pipeline.macro_event_calendar import MacroDataManager
            self.macro_manager = MacroDataManager(self.config)
            logger.info(" Macro event calendar initialized")
        except Exception as e:
            logger.error(f" Failed to initialize macro calendar: {e}")
            self.macro_manager = None
        
        logger.info(" Comprehensive Data Pipeline initialized")
        logger.info(f" Database: {self.db_path}")
        logger.info(f" Cache: Redis")
        logger.info(f" Free APIs: {' Available' if self.free_apis else ' Not available'}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load data pipeline configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f" Failed to load data pipeline config: {e}")
            return {}
    
    def _init_database(self):
        """Initialize DuckDB database with required tables"""
        Path("data").mkdir(exist_ok=True)
        
        conn = duckdb.connect(self.db_path)
        
        # Create tables
        conn.execute("""
            CREATE TABLE IF NOT EXISTS bars_1m (
                symbol VARCHAR,
                timestamp TIMESTAMP,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume BIGINT,
                PRIMARY KEY (symbol, timestamp)
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS bars_5m (
                symbol VARCHAR,
                timestamp TIMESTAMP,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume BIGINT,
                PRIMARY KEY (symbol, timestamp)
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS bars_daily (
                symbol VARCHAR,
                timestamp TIMESTAMP,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume BIGINT,
                PRIMARY KEY (symbol, timestamp)
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS options_5m (
                symbol VARCHAR,
                timestamp TIMESTAMP,
                strike DOUBLE,
                expiry DATE,
                option_type VARCHAR,
                open_interest BIGINT,
                implied_volatility DOUBLE,
                gamma DOUBLE,
                delta DOUBLE,
                theta DOUBLE,
                vega DOUBLE,
                PRIMARY KEY (symbol, timestamp, strike, expiry, option_type)
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS context_data (
                symbol VARCHAR,
                timestamp TIMESTAMP,
                data_type VARCHAR,
                value DOUBLE,
                metadata JSON,
                PRIMARY KEY (symbol, timestamp, data_type)
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sentiment_data (
                symbol VARCHAR,
                timestamp TIMESTAMP,
                source VARCHAR,
                sentiment_score DOUBLE,
                confidence DOUBLE,
                text_content TEXT,
                PRIMARY KEY (symbol, timestamp, source)
            )
        """)
        
        conn.close()
        logger.info(" Database tables initialized")
    
    def _init_redis(self) -> Optional[redis.Redis]:
        """Initialize Redis cache"""
        try:
            client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            client.ping()
            logger.info(" Redis cache connected")
            return client
        except Exception as e:
            logger.warning(f" Redis not available: {e}")
            return None
    
    def fetch_tsx_data(self, symbols: List[str], timeframe: str = "1m") -> Dict[str, pd.DataFrame]:
        """Fetch TSX/TSXV market data - uses Questrade (unlimited) first, Yahoo Finance as fallback"""
        data = {}
        
        # Try Questrade first for ALL symbols (no rate limiting!)
        if self.questrade:
            try:
                quotes = self.questrade.get_quotes(symbols)
                if quotes:
                    for quote in quotes:
                        symbol = quote.get('symbol', '')
                        if not symbol:
                            continue
                        
                        # Skip if no valid price data
                        last_price = quote.get('lastTradePrice')
                        if last_price is None or last_price <= 0:
                            logger.debug(f" Skipping {symbol}: no valid price from Questrade")
                            continue
                        
                        # Convert Questrade quote to DataFrame format
                        hist_data = {
                            'datetime': [datetime.now()],
                            'open': [quote.get('openPrice') or last_price],
                            'high': [quote.get('highPrice') or last_price],
                            'low': [quote.get('lowPrice') or last_price],
                            'close': [last_price],
                            'volume': [quote.get('volume', 0)],
                            'symbol': [symbol]
                        }
                        
                        hist = pd.DataFrame(hist_data)
                        data[symbol] = hist
                        
                        logger.info(f" Fetched Questrade quote for {symbol}: ${last_price:.2f}")
                    
                    # Successfully got all quotes from Questrade - return immediately!
                    if len(data) == len(symbols):
                        logger.info(f" SUCCESS: Got all {len(data)} quotes from Questrade (no rate limits!)")
                        return data
            except Exception as e:
                logger.warning(f" Questrade batch fetch failed: {e}, falling back to Yahoo Finance")
        
        # Fallback to Yahoo Finance for any missing symbols (ONLY if Questrade failed completely)
        # Skip Yahoo fallback if we got SOME data from Questrade to avoid rate limits
        if self.questrade and len(data) > 0:
            logger.info(f" Got {len(data)}/{len(symbols)} quotes from Questrade, skipping Yahoo fallback to avoid rate limits")
            return data
        
        # Only use Yahoo if Questrade completely failed
        for symbol in symbols:
            if symbol in data:
                continue  # Already have it from Questrade
            
            try:
                # Check cache first
                cache_key = f"tsx:{symbol}:{timeframe}"
                if self.redis_client:
                    cached_data = self.redis_client.get(cache_key)
                    if cached_data:
                        data[symbol] = pd.read_json(cached_data)
                        continue
                
                # Fetch from Yahoo Finance (rate limited)
                ticker = yf.Ticker(symbol)
                
                if timeframe == "1m":
                    hist = ticker.history(period="1d", interval="1m")
                elif timeframe == "5m":
                    hist = ticker.history(period="5d", interval="5m")
                elif timeframe == "1d":
                    hist = ticker.history(period="1mo", interval="1d")
                else:
                    continue
                
                if not hist.empty:
                    # Clean and prepare data
                    hist = hist.reset_index()
                    hist.columns = [col.lower() for col in hist.columns]
                    hist['symbol'] = symbol
                    
                    data[symbol] = hist
                    
                    # Cache the data
                    if self.redis_client:
                        self.redis_client.setex(
                            cache_key, 
                            300,  # 5 minutes cache
                            hist.to_json()
                        )
                    
                    logger.info(f" Fetched {timeframe} data for {symbol}: {len(hist)} bars")
                else:
                    logger.warning(f" No data for {symbol}")
                
                # Rate limiting for Yahoo Finance only (MUCH slower to avoid bans!)
                time.sleep(1.0)  # 1 second delay to stay under rate limits
                
            except Exception as e:
                logger.error(f" Error fetching {symbol}: {e}")
                self.quality_metrics['api_errors'] += 1
        
        return data
    
    def fetch_options_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Fetch options data (simplified - real implementation would use options API)"""
        options_data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                
                # Get options chain (simplified)
                try:
                    options_chain = ticker.option_chain()
                    calls = options_chain.calls
                    puts = options_chain.puts
                    
                    # Combine calls and puts
                    all_options = pd.concat([calls, puts], ignore_index=True)
                    all_options['symbol'] = symbol
                    all_options['timestamp'] = datetime.now()
                    
                    options_data[symbol] = all_options
                    logger.info(f" Fetched options data for {symbol}: {len(all_options)} contracts")
                    
                except Exception as e:
                    logger.warning(f" No options data for {symbol}: {e}")
                
                time.sleep(0.5)  # Slower polling for options data
                
            except Exception as e:
                logger.error(f" Error fetching options for {symbol}: {e}")
        
        return options_data
    
    def fetch_macro_data(self) -> Dict[str, float]:
        """Fetch macro economic data"""
        macro_data = {}
        
        try:
            # Bank of Canada interest rate (simplified)
            macro_data['boc_rate'] = 5.0  # Current rate (would fetch from API)
            
            # WTI Crude Oil
            oil_ticker = yf.Ticker("CL=F")
            oil_data = oil_ticker.history(period="1d")
            if not oil_data.empty:
                macro_data['wti_price'] = float(oil_data['Close'].iloc[-1])
            
            # USD/CAD
            fx_ticker = yf.Ticker("CADUSD=X")
            fx_data = fx_ticker.history(period="1d")
            if not fx_data.empty:
                macro_data['usd_cad'] = float(fx_data['Close'].iloc[-1])
            
            # VIX (using VIXY as proxy)
            vix_ticker = yf.Ticker("VIXY")
            vix_data = vix_ticker.history(period="1d")
            if not vix_data.empty:
                macro_data['vix'] = float(vix_data['Close'].iloc[-1])
            
            # TSX Index
            tsx_ticker = yf.Ticker("^GSPTSE")
            tsx_data = tsx_ticker.history(period="1d")
            if not tsx_data.empty:
                macro_data['tsx_index'] = float(tsx_data['Close'].iloc[-1])
            
            logger.info(f"Fetched macro data: {len(macro_data)} indicators")
            
        except Exception as e:
            logger.error(f" Error fetching macro data: {e}")
        
        return macro_data
    
    def fetch_news_sentiment(self, symbols: List[str]) -> Dict[str, Dict]:
        """Fetch news sentiment data (simplified)"""
        sentiment_data = {}
        
        for symbol in symbols:
            try:
                # Simplified sentiment scoring (would use real news API)
                sentiment_score = np.random.uniform(-1, 1)  # Random for demo
                confidence = np.random.uniform(0.6, 0.9)
                
                sentiment_data[symbol] = {
                    'sentiment_score': sentiment_score,
                    'confidence': confidence,
                    'source': 'simulated',
                    'timestamp': datetime.now(),
                    'text_content': f"Simulated news sentiment for {symbol}"
                }
                
                logger.info(f" Generated sentiment for {symbol}: {sentiment_score:.3f}")
                
            except Exception as e:
                logger.error(f" Error generating sentiment for {symbol}: {e}")
        
        return sentiment_data
    
    def calculate_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical analysis features"""
        if data.empty:
            return data
        
        df = data.copy()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # VWAP
        df['vwap'] = (df['close'] * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(window=14).mean()
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_percent'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price momentum
        df['price_change'] = df['close'].pct_change()
        df['price_change_5'] = df['close'].pct_change(5)
        df['price_change_20'] = df['close'].pct_change(20)
        
        logger.info(f" Calculated technical features: {len(df.columns)} columns")
        
        return df
    
    def store_data(self, data: Dict[str, pd.DataFrame], table_name: str):
        """Store data in DuckDB"""
        try:
            conn = duckdb.connect(self.db_path)
            
            for symbol, df in data.items():
                if not df.empty:
                    # Ensure timestamp column exists
                    if 'timestamp' not in df.columns and 'Datetime' in df.columns:
                        df['timestamp'] = df['Datetime']
                    
                    # Insert data
                    conn.execute(f"INSERT OR REPLACE INTO {table_name} SELECT * FROM df")
                    logger.info(f" Stored {len(df)} records for {symbol} in {table_name}")
            
            conn.close()
            
        except Exception as e:
            logger.error(f" Error storing data: {e}")
    
    def get_latest_data(self, symbols: List[str], table_name: str, limit: int = 100) -> Dict[str, pd.DataFrame]:
        """Get latest data from database"""
        data = {}
        
        try:
            conn = duckdb.connect(self.db_path)
            
            for symbol in symbols:
                query = f"""
                    SELECT * FROM {table_name} 
                    WHERE symbol = '{symbol}' 
                    ORDER BY timestamp DESC 
                    LIMIT {limit}
                """
                
                result = conn.execute(query).fetchdf()
                if not result.empty:
                    data[symbol] = result
                    logger.info(f" Retrieved {len(result)} records for {symbol}")
            
            conn.close()
            
        except Exception as e:
            logger.error(f" Error retrieving data: {e}")
        
        return data
    
    def check_data_quality(self) -> Dict[str, float]:
        """Check data quality metrics"""
        try:
            conn = duckdb.connect(self.db_path)
            
            # Check missing data rate
            total_records = conn.execute("SELECT COUNT(*) FROM bars_1m").fetchone()[0]
            recent_records = conn.execute("""
                SELECT COUNT(*) FROM bars_1m 
                WHERE timestamp > CURRENT_TIMESTAMP - INTERVAL '1 hour'
            """).fetchone()[0]
            
            if total_records > 0:
                missing_rate = 1 - (recent_records / (total_records * 0.1))  # Expected 10% of total in last hour
                self.quality_metrics['missing_data_rate'] = max(0, missing_rate)
            
            # Check stale data
            stale_count = conn.execute("""
                SELECT COUNT(*) FROM bars_1m 
                WHERE timestamp < CURRENT_TIMESTAMP - INTERVAL '5 minutes'
            """).fetchone()[0]
            self.quality_metrics['stale_data_count'] = stale_count
            
            self.quality_metrics['last_update'] = datetime.now()
            
            conn.close()
            
            logger.info(f" Data Quality: Missing rate {self.quality_metrics['missing_data_rate']:.3f}, Stale count {stale_count}")
            
        except Exception as e:
            logger.error(f" Error checking data quality: {e}")
        
        return self.quality_metrics
    
    def run_data_collection_cycle(self, symbols: List[str]):
        """Run complete data collection cycle"""
        logger.info(" Starting data collection cycle")
        
        # Fetch market data
        logger.info(" Fetching TSX market data...")
        bars_1m = self.fetch_tsx_data(symbols, "1m")
        bars_5m = self.fetch_tsx_data(symbols, "5m")
        bars_daily = self.fetch_tsx_data(symbols, "1d")
        
        # Fetch options data
        logger.info(" Fetching options data...")
        options_data = self.fetch_options_data(symbols[:5])  # Limit to first 5 symbols
        
        # Fetch macro data
        logger.info(" Fetching macro data...")
        macro_data = self.fetch_macro_data()
        
        # Fetch sentiment data
        logger.info(" Fetching sentiment data...")
        sentiment_data = self.fetch_news_sentiment(symbols)
        
        # Calculate technical features
        logger.info(" Calculating technical features...")
        for symbol, df in bars_1m.items():
            bars_1m[symbol] = self.calculate_technical_features(df)
        
        # Store data
        logger.info(" Storing data...")
        self.store_data(bars_1m, "bars_1m")
        self.store_data(bars_5m, "bars_5m")
        self.store_data(bars_daily, "bars_daily")
        
        # Store options data
        if options_data:
            self.store_data(options_data, "options_5m")
        
        # Store macro data as context
        context_data = {}
        for key, value in macro_data.items():
            context_data[key] = pd.DataFrame({
                'symbol': ['MACRO'],
                'timestamp': [datetime.now()],
                'data_type': [key],
                'value': [value],
                'metadata': [json.dumps({})]
            })
        self.store_data(context_data, "context_data")
        
        # Store sentiment data
        sentiment_df_data = {}
        for symbol, sentiment in sentiment_data.items():
            sentiment_df_data[symbol] = pd.DataFrame({
                'symbol': [symbol],
                'timestamp': [sentiment['timestamp']],
                'source': [sentiment['source']],
                'sentiment_score': [sentiment['sentiment_score']],
                'confidence': [sentiment['confidence']],
                'text_content': [sentiment['text_content']]
            })
        self.store_data(sentiment_df_data, "sentiment_data")
        
        # Check data quality
        self.check_data_quality()
        
        logger.info(" Data collection cycle completed")
    
    def get_enhanced_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Get enhanced data using free APIs integration"""
        if not self.free_apis:
            logger.warning("Free APIs not available, returning basic data")
            return self.get_latest_data(symbols, "bars_1m", 100)
        
        try:
            logger.info(f" Getting enhanced data for {len(symbols)} symbols")
            
            # Get comprehensive data from free APIs
            enhanced_data = self.free_apis.get_comprehensive_data(symbols)
            
            # Get basic market data
            basic_data = self.get_latest_data(symbols, "bars_1m", 100)
            
            # Combine data
            combined_data = {
                'basic_market_data': basic_data,
                'news_sentiment': enhanced_data.get('news_sentiment', {}),
                'technical_indicators': enhanced_data.get('technical_indicators', {}),
                'reddit_sentiment': enhanced_data.get('reddit_sentiment', {}),
                'timestamp': enhanced_data.get('timestamp'),
                'api_status': self.free_apis.get_api_status()
            }
            
            logger.info(" Enhanced data retrieved successfully")
            return combined_data
            
        except Exception as e:
            logger.error(f" Error getting enhanced data: {e}")
            return self.get_latest_data(symbols, "bars_1m", 100)
    
    def get_api_status(self) -> Dict[str, Any]:
        """Get status of all API services"""
        if not self.free_apis:
            return {'free_apis': 'not_available'}
        
        return self.free_apis.get_api_status()
    
    def get_tick_data(self, symbol: str, interval_minutes: int = 1, count: int = 100) -> pd.DataFrame:
        """Get tick data aggregated to bars"""
        if not self.tick_manager:
            logger.warning("Tick data manager not available")
            return pd.DataFrame()
        
        return self.tick_manager.get_recent_bars(symbol, interval_minutes, count)
    
    def get_vwap(self, symbol: str, period_minutes: int = 5) -> float:
        """Get Volume Weighted Average Price"""
        if not self.tick_manager:
            return 0.0
        
        return self.tick_manager.get_vwap(symbol, period_minutes)
    
    def get_slippage_estimate(self, symbol: str, order_size: int, order_type: str = 'MARKET') -> float:
        """Get slippage estimate for an order"""
        if not self.tick_manager:
            return 0.0
        
        return self.tick_manager.get_slippage_estimate(symbol, order_size, order_type)
    
    def get_liquidity_score(self, symbol: str) -> float:
        """Get liquidity score for a symbol"""
        if not self.tick_manager:
            return 0.0
        
        return self.tick_manager.get_liquidity_score(symbol)
    
    def get_options_chain(self, symbol: str, underlying_price: float) -> Any:
        """Get options chain for a symbol"""
        if not self.options_manager:
            return None
        
        return self.options_manager.get_options_chain(symbol, underlying_price)
    
    def get_put_call_ratio(self, symbol: str, underlying_price: float) -> float:
        """Get put/call ratio for sentiment analysis"""
        if not self.options_manager:
            return 0.0
        
        return self.options_manager.get_put_call_ratio(symbol, underlying_price)
    
    def get_delta_equivalent_exposure(self, symbol: str, underlying_price: float) -> float:
        """Get delta-equivalent exposure for a symbol"""
        if not self.options_manager:
            return 0.0
        
        return self.options_manager.get_delta_equivalent_exposure(symbol, underlying_price)
    
    def get_market_regime(self) -> str:
        """Get current market regime"""
        if not self.macro_manager:
            return "neutral"
        
        return self.macro_manager.get_market_regime()
    
    def get_trading_recommendations(self) -> Dict[str, str]:
        """Get trading recommendations based on macro events"""
        if not self.macro_manager:
            return {}
        
        return self.macro_manager.get_trading_recommendations()
    
    def get_event_heat_score(self) -> float:
        """Get current event heat score"""
        if not self.macro_manager:
            return 0.0
        
        return self.macro_manager.get_event_heat_score()
    
    def is_high_volatility_period(self) -> bool:
        """Check if we're in a high volatility period"""
        if not self.macro_manager:
            return False
        
        return self.macro_manager.is_high_volatility_period()
    
    def get_upcoming_events_summary(self, days_ahead: int = 7) -> Dict:
        """Get summary of upcoming events"""
        if not self.macro_manager:
            return {}
        
        return self.macro_manager.get_upcoming_events_summary(days_ahead)


if __name__ == "__main__":
    # Test the data pipeline
    logging.basicConfig(level=logging.INFO)
    
    pipeline = ComprehensiveDataPipeline()
    
    # Test symbols
    test_symbols = ["RY.TO", "TD.TO", "SHOP.TO", "CNR.TO", "ENB.TO"]
    
    # Run data collection cycle
    pipeline.run_data_collection_cycle(test_symbols)
    
    # Get latest data
    latest_data = pipeline.get_latest_data(test_symbols, "bars_1m", 10)
    
    print(f"\n Retrieved data for {len(latest_data)} symbols")
    for symbol, df in latest_data.items():
        print(f"  {symbol}: {len(df)} records")
        if not df.empty:
            print(f"    Latest close: ${df['close'].iloc[-1]:.2f}")
            print(f"    Latest RSI: {df['rsi'].iloc[-1]:.2f}")
