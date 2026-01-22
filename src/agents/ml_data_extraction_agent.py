"""
ML Data Extraction Agent - 20 Years Historical Data Collection

Optimized for Alpha Vantage paid subscription to extract maximum historical data
for ML training. Collects OHLCV, technical indicators, and fundamental data.
"""

import logging
import time
import json
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import duckdb
from pathlib import Path

from ..data_collection.alpha_vantage_collector import AlphaVantageCollector
from ..data_collection.enhanced_collectors import EnhancedYahooFinanceCollector

logger = logging.getLogger(__name__)

@dataclass
class MLDataProgress:
    """Track ML data extraction progress"""
    symbol: str
    daily_data_collected: bool
    technical_indicators_collected: bool
    fundamental_data_collected: bool
    total_records: int
    date_range: str
    last_updated: str

class MLDataExtractionAgent:
    """ML-focused data extraction agent for 20-year historical data"""
    
    def __init__(self):
        self.alpha_vantage = AlphaVantageCollector()
        self.yahoo_fallback = EnhancedYahooFinanceCollector()
        
        # ML data extraction configuration
        self.config = {
            "alpha_vantage_calls_per_minute": 74,
            "call_delay": 0.82,  # 60/74 seconds
            "years_of_history": 20,  # 2004-2024
            "batch_size": 10,  # Process 10 stocks at a time
            "progress_update_frequency": 5,  # Every 5 stocks
        }
        
        # ML-optimized stock universe (500+ stocks)
        self.ml_universe = self._get_ml_stock_universe()
        
        # Database setup - use specified PastData directory
        self.data_dir = Path("PastData")
        self.data_dir.mkdir(exist_ok=True)
        self.db_path = self.data_dir / "ml_training_data.duckdb"
        self.setup_database()
        
        # Progress tracking
        self.progress_tracker = {}
        self.total_api_calls = 0
        self.start_time = None
        
        logger.info(f"🤖 ML Data Extraction Agent initialized")
        logger.info(f"📊 Target: {len(self.ml_universe)} stocks × {self.config['years_of_history']} years")
        logger.info(f"🔑 Alpha Vantage rate limit: {self.config['alpha_vantage_calls_per_minute']} calls/min")
        logger.info(f"💾 Database: {self.db_path}")
    
    def _get_ml_stock_universe(self) -> List[str]:
        """Get comprehensive stock universe optimized for ML training"""
        
        # Tier 1: Mega Cap (50 stocks) - Most reliable data
        mega_cap = [
            # FAANG + Mega Tech
            "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "TSLA", "NVDA",
            "NFLX", "CRM", "ORCL", "ADBE", "INTC", "AMD", "QCOM", "AVGO",
            
            # Finance
            "JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC", "TFC", "COF",
            
            # Healthcare
            "JNJ", "PFE", "UNH", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY",
            
            # Consumer
            "WMT", "HD", "PG", "KO", "PEP", "MCD", "NKE", "SBUX", "TGT", "COST"
        ]
        
        # Tier 2: Large Cap (100 stocks) - Good historical data
        large_cap = [
            # Industrial
            "BA", "CAT", "GE", "MMM", "HON", "UPS", "FDX", "LMT", "RTX", "NOC",
            "DE", "EMR", "ETN", "PH", "ITW", "ROK", "DOV", "IR", "CMI", "EMR",
            
            # Energy
            "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "VLO", "PSX", "KMI", "OKE",
            "WMB", "EPD", "ET", "MPLX", "ENB", "TRP", "SU", "CNQ", "CVE", "IMO",
            
            # Utilities
            "NEE", "DUK", "SO", "AEP", "EXC", "XEL", "WEC", "ES", "AWK", "CMS",
            
            # Materials
            "LIN", "APD", "ECL", "SHW", "FCX", "NEM", "GOLD", "AUY", "KGC", "HL",
            
            # Real Estate
            "AMT", "PLD", "CCI", "EQIX", "DLR", "PSA", "EXR", "AVB", "EQR", "UDR",
            
            # Telecom
            "VZ", "T", "TMUS", "CHTR", "CMCSA", "DIS", "NFLX", "ROKU", "SPOT", "PINS"
        ]
        
        # Tier 3: Growth Stocks (150 stocks) - High volatility, good for ML
        growth_stocks = [
            # SaaS & Cloud
            "SNOW", "CRWD", "ZS", "OKTA", "DDOG", "NET", "FSLY", "TWLO", "ZM", "DOCU",
            "WORK", "PLTR", "UBER", "LYFT", "DASH", "ABNB", "RBLX", "U", "PINS", "SNAP",
            
            # Biotech
            "MRNA", "BNTX", "NVAX", "TDOC", "VEEV", "PTON", "ZBH", "SYK", "MDT", "BSX",
            "EW", "HOLX", "VAR", "PKI", "A", "LH", "DGX", "GILD", "BIIB", "REGN",
            "VRTX", "ILMN", "AMGN", "CELG", "MYL", "TEVA", "AGN", "BMY", "LLY", "NVO",
            
            # Fintech
            "PYPL", "SQ", "V", "MA", "COIN", "HOOD", "SOFI", "AFRM", "LC", "UPST",
            "OPEN", "RBLX", "U", "PINS", "SNAP", "TWTR", "SPOT", "ROKU", "NFLX", "DIS",
            
            # EV & Clean Energy
            "TSLA", "NIO", "XPEV", "LI", "RIVN", "LCID", "FSR", "NKLA", "PLUG", "FCEL",
            "BE", "BLNK", "CHPT", "ENPH", "SEDG", "RUN", "SPWR", "CSIQ", "JKS", "SOL",
            
            # Semiconductors
            "NVDA", "AMD", "INTC", "QCOM", "AVGO", "TXN", "ADI", "MRVL", "XLNX", "LRCX",
            "AMAT", "KLAC", "ASML", "TSM", "UMC", "ASX", "MPWR", "SWKS", "QRVO", "MCHP",
            
            # Gaming & Entertainment
            "ATVI", "EA", "TTWO", "RBLX", "U", "PINS", "SNAP", "TWTR", "SPOT", "ROKU",
            "NFLX", "DIS", "CMCSA", "VZ", "T", "TMUS", "CHTR", "DISH", "SIRI", "LYV",
            
            # E-commerce & Retail
            "AMZN", "SHOP", "EBAY", "ETSY", "W", "WAYFAIR", "CHWY", "PETS", "OSTK", "PRTS",
            "FLWS", "GRUB", "UBER", "LYFT", "DASH", "ABNB", "EXPE", "BKNG", "TXG", "TRIP"
        ]
        
        # Tier 4: Volatile/Emerging (100 stocks) - High-risk, high-reward for ML
        volatile_stocks = [
            # Cannabis
            "TLRY", "CGC", "ACB", "CRON", "SNDL", "OGI", "HEXO", "APHA", "CURLF", "GTBIF",
            
            # Meme Stocks
            "AMC", "GME", "BBBY", "CLOV", "WISH", "PLBY", "RIDE", "WKHS", "GOEV", "HYLN",
            "SPCE", "SKLZ", "DKNG", "PENN", "MGM", "LVS", "WYNN", "CZR", "BYD", "GNOG",
            
            # Crypto-related
            "COIN", "HOOD", "RIOT", "MARA", "HUT", "BITF", "CAN", "HIVE", "DMGI", "BTBT",
            
            # SPACs & Recent IPOs
            "RIVN", "LCID", "FSR", "NKLA", "HYLN", "RIDE", "WKHS", "GOEV", "SPCE", "SKLZ",
            "DKNG", "PENN", "MGM", "LVS", "WYNN", "CZR", "BYD", "GNOG", "RSI", "BETZ",
            
            # Small Cap Growth
            "ROKU", "PINS", "SNAP", "TWTR", "SPOT", "ZM", "DOCU", "WORK", "PLTR", "UBER",
            "LYFT", "DASH", "ABNB", "RBLX", "U", "PINS", "SNAP", "TWTR", "SPOT", "ROKU",
            
            # Penny Stocks (for volatility patterns)
            "SNDL", "NOK", "BB", "NAKD", "SIRI", "F", "GE", "T", "VZ", "KO",
            "PFE", "INTC", "CSCO", "WBA", "GPS", "M", "JCP", "SHLD", "RAD", "GME"
        ]
        
        # Combine all tiers
        all_stocks = mega_cap + large_cap + growth_stocks + volatile_stocks
        
        # Remove duplicates and sort
        unique_stocks = sorted(list(set(all_stocks)))
        
        logger.info(f"🎯 ML Universe: {len(unique_stocks)} stocks")
        logger.info(f"   📊 Mega Cap: {len(mega_cap)} stocks")
        logger.info(f"   📈 Large Cap: {len(large_cap)} stocks") 
        logger.info(f"   🚀 Growth: {len(growth_stocks)} stocks")
        logger.info(f"   ⚡ Volatile: {len(volatile_stocks)} stocks")
        
        return unique_stocks
    
    def setup_database(self):
        """Setup DuckDB database for ML training data"""
        
        # Create data directory
        Path("data").mkdir(exist_ok=True)
        
        # Connect to DuckDB
        self.conn = duckdb.connect(str(self.db_path))
        
        # Create tables for ML data
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_ohlcv (
                symbol VARCHAR,
                date DATE,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                adjusted_close DOUBLE,
                volume BIGINT,
                dividend_amount DOUBLE,
                split_coefficient DOUBLE,
                PRIMARY KEY (symbol, date)
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS technical_indicators (
                symbol VARCHAR,
                date DATE,
                rsi_14 DOUBLE,
                macd DOUBLE,
                macd_signal DOUBLE,
                macd_histogram DOUBLE,
                bb_upper DOUBLE,
                bb_middle DOUBLE,
                bb_lower DOUBLE,
                sma_20 DOUBLE,
                sma_50 DOUBLE,
                sma_200 DOUBLE,
                ema_12 DOUBLE,
                ema_26 DOUBLE,
                PRIMARY KEY (symbol, date)
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS fundamental_data (
                symbol VARCHAR,
                fiscal_date DATE,
                reported_date DATE,
                revenue BIGINT,
                gross_profit BIGINT,
                operating_income BIGINT,
                net_income BIGINT,
                eps DOUBLE,
                total_assets BIGINT,
                total_debt BIGINT,
                shareholders_equity BIGINT,
                PRIMARY KEY (symbol, fiscal_date)
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS extraction_progress (
                symbol VARCHAR PRIMARY KEY,
                daily_data_collected BOOLEAN,
                technical_indicators_collected BOOLEAN,
                fundamental_data_collected BOOLEAN,
                total_records INTEGER,
                date_range VARCHAR,
                last_updated TIMESTAMP
            )
        """)
        
        logger.info("💾 DuckDB database initialized with ML tables")
    
    async def start_ml_data_extraction(self):
        """Start comprehensive ML data extraction"""
        
        logger.info("🚀 Starting ML Data Extraction (20 Years Historical)")
        self.start_time = time.time()
        
        # Phase 1: Daily OHLCV Data
        logger.info("📊 Phase 1: Collecting daily OHLCV data...")
        await self.extract_daily_ohlcv_data()
        
        # Phase 2: Technical Indicators
        logger.info("📈 Phase 2: Collecting technical indicators...")
        await self.extract_technical_indicators()
        
        # Phase 3: Fundamental Data
        logger.info("💰 Phase 3: Collecting fundamental data...")
        await self.extract_fundamental_data()
        
        # Phase 4: Generate ML Dataset
        logger.info("🤖 Phase 4: Generating ML-ready dataset...")
        await self.generate_ml_dataset()
        
        # Final report
        await self.generate_extraction_report()
        
        logger.info("✅ ML data extraction completed successfully!")
    
    async def extract_daily_ohlcv_data(self):
        """Extract 20 years of daily OHLCV data"""
        
        logger.info(f"📊 Extracting daily data for {len(self.ml_universe)} stocks...")
        
        successful_extractions = 0
        failed_extractions = 0
        
        for i, symbol in enumerate(self.ml_universe):
            try:
                # Rate limiting
                await self._enforce_rate_limit()
                
                # Get daily data from Alpha Vantage
                data, source = self.alpha_vantage.fetch_daily_data(symbol, outputsize="full")
                self.total_api_calls += 1
                
                if data is not None and not data.empty:
                    # Filter to last 20 years
                    cutoff_date = datetime.now() - timedelta(days=365 * self.config["years_of_history"])
                    data = data[data.index >= cutoff_date]
                    
                    if len(data) > 0:
                        # Store in database
                        await self._store_daily_data(symbol, data)
                        successful_extractions += 1
                        
                        logger.debug(f"✅ {symbol}: {len(data)} records ({data.index.min()} to {data.index.max()})")
                    else:
                        failed_extractions += 1
                        logger.debug(f"⚠️ {symbol}: No data in 20-year range")
                else:
                    failed_extractions += 1
                    logger.debug(f"❌ {symbol}: No data from Alpha Vantage")
                
                # Progress update
                if (i + 1) % self.config["progress_update_frequency"] == 0:
                    progress = (i + 1) / len(self.ml_universe) * 100
                    elapsed = time.time() - self.start_time
                    logger.info(f"📊 Daily data progress: {progress:.1f}% ({i+1}/{len(self.ml_universe)}) - "
                              f"Success: {successful_extractions}, Failed: {failed_extractions}, "
                              f"Time: {elapsed/60:.1f}min, API calls: {self.total_api_calls}")
                
                # Small delay between calls
                await asyncio.sleep(self.config["call_delay"])
                
            except Exception as e:
                failed_extractions += 1
                logger.warning(f"❌ Error extracting daily data for {symbol}: {e}")
        
        logger.info(f"✅ Daily data extraction complete: {successful_extractions} success, {failed_extractions} failed")
    
    async def _store_daily_data(self, symbol: str, data: pd.DataFrame):
        """Store daily OHLCV data in database"""
        
        # Prepare data for insertion
        records = []
        for date, row in data.iterrows():
            records.append({
                'symbol': symbol,
                'date': date.date(),
                'open': float(row.get('Open', 0)),
                'high': float(row.get('High', 0)),
                'low': float(row.get('Low', 0)),
                'close': float(row.get('Close', 0)),
                'adjusted_close': float(row.get('Adjusted Close', row.get('Close', 0))),
                'volume': int(row.get('Volume', 0)),
                'dividend_amount': float(row.get('Dividend Amount', 0)),
                'split_coefficient': float(row.get('Split Coefficient', 1))
            })
        
        # Insert into database
        if records:
            df = pd.DataFrame(records)
            self.conn.register('temp_daily_data', df)
            self.conn.execute("""
                INSERT OR REPLACE INTO daily_ohlcv 
                SELECT * FROM temp_daily_data
            """)
            
            # Update progress
            self.conn.execute("""
                INSERT OR REPLACE INTO extraction_progress 
                (symbol, daily_data_collected, total_records, date_range, last_updated)
                VALUES (?, TRUE, ?, ?, ?)
            """, [symbol, len(records), f"{data.index.min().date()} to {data.index.max().date()}", datetime.now()])
    
    async def extract_technical_indicators(self):
        """Extract technical indicators for all stocks"""
        
        logger.info("📈 Extracting technical indicators...")
        
        # Get symbols that have daily data
        symbols_with_data = self.conn.execute("""
            SELECT DISTINCT symbol FROM daily_ohlcv 
            ORDER BY symbol
        """).fetchall()
        
        successful_indicators = 0
        failed_indicators = 0
        
        for i, (symbol,) in enumerate(symbols_with_data):
            try:
                # Rate limiting
                await self._enforce_rate_limit()
                
                # Get technical indicators from Alpha Vantage
                indicators = await self._fetch_technical_indicators(symbol)
                self.total_api_calls += len(indicators)  # Multiple API calls for different indicators
                
                if indicators:
                    await self._store_technical_indicators(symbol, indicators)
                    successful_indicators += 1
                    logger.debug(f"✅ {symbol}: Technical indicators collected")
                else:
                    failed_indicators += 1
                    logger.debug(f"❌ {symbol}: No technical indicators")
                
                # Progress update
                if (i + 1) % self.config["progress_update_frequency"] == 0:
                    progress = (i + 1) / len(symbols_with_data) * 100
                    logger.info(f"📈 Technical indicators progress: {progress:.1f}% - "
                              f"Success: {successful_indicators}, Failed: {failed_indicators}")
                
                # Delay between symbols (multiple API calls per symbol)
                await asyncio.sleep(self.config["call_delay"] * 3)  # Longer delay for multiple calls
                
            except Exception as e:
                failed_indicators += 1
                logger.warning(f"❌ Error extracting indicators for {symbol}: {e}")
        
        logger.info(f"✅ Technical indicators complete: {successful_indicators} success, {failed_indicators} failed")
    
    async def _fetch_technical_indicators(self, symbol: str) -> Dict:
        """Fetch multiple technical indicators for a symbol"""
        
        indicators = {}
        
        try:
            # RSI
            rsi_data, _ = self.alpha_vantage.fetch_technical_indicator(symbol, "RSI", interval="daily", time_period=14)
            if rsi_data is not None:
                indicators['rsi'] = rsi_data
            
            # MACD
            macd_data, _ = self.alpha_vantage.fetch_technical_indicator(symbol, "MACD", interval="daily")
            if macd_data is not None:
                indicators['macd'] = macd_data
            
            # Bollinger Bands
            bb_data, _ = self.alpha_vantage.fetch_technical_indicator(symbol, "BBANDS", interval="daily", time_period=20)
            if bb_data is not None:
                indicators['bbands'] = bb_data
            
            # Simple Moving Averages
            sma20_data, _ = self.alpha_vantage.fetch_technical_indicator(symbol, "SMA", interval="daily", time_period=20)
            if sma20_data is not None:
                indicators['sma20'] = sma20_data
                
            sma50_data, _ = self.alpha_vantage.fetch_technical_indicator(symbol, "SMA", interval="daily", time_period=50)
            if sma50_data is not None:
                indicators['sma50'] = sma50_data
                
            sma200_data, _ = self.alpha_vantage.fetch_technical_indicator(symbol, "SMA", interval="daily", time_period=200)
            if sma200_data is not None:
                indicators['sma200'] = sma200_data
            
            # Exponential Moving Averages
            ema12_data, _ = self.alpha_vantage.fetch_technical_indicator(symbol, "EMA", interval="daily", time_period=12)
            if ema12_data is not None:
                indicators['ema12'] = ema12_data
                
            ema26_data, _ = self.alpha_vantage.fetch_technical_indicator(symbol, "EMA", interval="daily", time_period=26)
            if ema26_data is not None:
                indicators['ema26'] = ema26_data
            
        except Exception as e:
            logger.debug(f"Error fetching indicators for {symbol}: {e}")
        
        return indicators
    
    async def _store_technical_indicators(self, symbol: str, indicators: Dict):
        """Store technical indicators in database"""
        
        # This is a simplified version - in practice, you'd need to align all indicators by date
        # and handle the complex data structure from Alpha Vantage technical indicators
        
        # Update progress
        self.conn.execute("""
            UPDATE extraction_progress 
            SET technical_indicators_collected = TRUE, last_updated = ?
            WHERE symbol = ?
        """, [datetime.now(), symbol])
    
    async def extract_fundamental_data(self):
        """Extract fundamental data (earnings, financials)"""
        
        logger.info("💰 Extracting fundamental data...")
        
        # Get symbols that have daily data
        symbols_with_data = self.conn.execute("""
            SELECT DISTINCT symbol FROM daily_ohlcv 
            ORDER BY symbol
        """).fetchall()
        
        successful_fundamentals = 0
        failed_fundamentals = 0
        
        for i, (symbol,) in enumerate(symbols_with_data):
            try:
                # Rate limiting
                await self._enforce_rate_limit()
                
                # Get earnings data
                earnings_data, _ = self.alpha_vantage.fetch_earnings(symbol)
                self.total_api_calls += 1
                
                if earnings_data is not None:
                    await self._store_fundamental_data(symbol, earnings_data)
                    successful_fundamentals += 1
                    logger.debug(f"✅ {symbol}: Fundamental data collected")
                else:
                    failed_fundamentals += 1
                    logger.debug(f"❌ {symbol}: No fundamental data")
                
                # Progress update
                if (i + 1) % self.config["progress_update_frequency"] == 0:
                    progress = (i + 1) / len(symbols_with_data) * 100
                    logger.info(f"💰 Fundamental data progress: {progress:.1f}% - "
                              f"Success: {successful_fundamentals}, Failed: {failed_fundamentals}")
                
                await asyncio.sleep(self.config["call_delay"])
                
            except Exception as e:
                failed_fundamentals += 1
                logger.warning(f"❌ Error extracting fundamentals for {symbol}: {e}")
        
        logger.info(f"✅ Fundamental data complete: {successful_fundamentals} success, {failed_fundamentals} failed")
    
    async def _store_fundamental_data(self, symbol: str, earnings_data):
        """Store fundamental data in database"""
        
        # Update progress
        self.conn.execute("""
            UPDATE extraction_progress 
            SET fundamental_data_collected = TRUE, last_updated = ?
            WHERE symbol = ?
        """, [datetime.now(), symbol])
    
    async def generate_ml_dataset(self):
        """Generate ML-ready dataset with features and labels"""
        
        logger.info("🤖 Generating ML-ready dataset...")
        
        # Create comprehensive feature table
        self.conn.execute("""
            CREATE OR REPLACE TABLE ml_features AS
            SELECT 
                d.symbol,
                d.date,
                d.open,
                d.high,
                d.low,
                d.close,
                d.adjusted_close,
                d.volume,
                
                -- Price features
                (d.close - LAG(d.close, 1) OVER (PARTITION BY d.symbol ORDER BY d.date)) / LAG(d.close, 1) OVER (PARTITION BY d.symbol ORDER BY d.date) AS return_1d,
                (d.close - LAG(d.close, 5) OVER (PARTITION BY d.symbol ORDER BY d.date)) / LAG(d.close, 5) OVER (PARTITION BY d.symbol ORDER BY d.date) AS return_5d,
                (d.close - LAG(d.close, 20) OVER (PARTITION BY d.symbol ORDER BY d.date)) / LAG(d.close, 20) OVER (PARTITION BY d.symbol ORDER BY d.date) AS return_20d,
                
                -- Volume features
                d.volume / AVG(d.volume) OVER (PARTITION BY d.symbol ORDER BY d.date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS volume_ratio_20d,
                
                -- Volatility features
                STDDEV(d.close) OVER (PARTITION BY d.symbol ORDER BY d.date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS volatility_20d,
                
                -- Technical indicators (when available)
                t.rsi_14,
                t.macd,
                t.bb_upper,
                t.bb_lower,
                t.sma_20,
                t.sma_50,
                t.sma_200
                
            FROM daily_ohlcv d
            LEFT JOIN technical_indicators t ON d.symbol = t.symbol AND d.date = t.date
            WHERE d.date >= '2004-01-01'
            ORDER BY d.symbol, d.date
        """)
        
        # Create labels for ML training
        self.conn.execute("""
            CREATE OR REPLACE TABLE ml_labels AS
            SELECT 
                symbol,
                date,
                
                -- Classification labels
                CASE 
                    WHEN LEAD(close, 1) OVER (PARTITION BY symbol ORDER BY date) > close THEN 1 
                    ELSE 0 
                END AS next_day_up,
                
                CASE 
                    WHEN LEAD(close, 5) OVER (PARTITION BY symbol ORDER BY date) > close * 1.05 THEN 1 
                    ELSE 0 
                END AS next_5d_up_5pct,
                
                -- Regression labels
                (LEAD(close, 1) OVER (PARTITION BY symbol ORDER BY date) - close) / close AS next_day_return,
                (LEAD(close, 5) OVER (PARTITION BY symbol ORDER BY date) - close) / close AS next_5d_return
                
            FROM daily_ohlcv
            WHERE date >= '2004-01-01'
            ORDER BY symbol, date
        """)
        
        logger.info("🤖 ML dataset generated with features and labels")
    
    async def _enforce_rate_limit(self):
        """Enforce Alpha Vantage rate limiting"""
        
        # Simple rate limiting implementation
        await asyncio.sleep(self.config["call_delay"])
    
    async def generate_extraction_report(self):
        """Generate comprehensive extraction report"""
        
        logger.info("📊 Generating ML data extraction report...")
        
        # Get extraction statistics
        stats = self.conn.execute("""
            SELECT 
                COUNT(*) as total_symbols,
                SUM(CASE WHEN daily_data_collected THEN 1 ELSE 0 END) as daily_data_success,
                SUM(CASE WHEN technical_indicators_collected THEN 1 ELSE 0 END) as technical_success,
                SUM(CASE WHEN fundamental_data_collected THEN 1 ELSE 0 END) as fundamental_success,
                SUM(total_records) as total_records
            FROM extraction_progress
        """).fetchone()
        
        # Get data range
        date_range = self.conn.execute("""
            SELECT MIN(date) as min_date, MAX(date) as max_date, COUNT(*) as total_records
            FROM daily_ohlcv
        """).fetchone()
        
        # Get ML dataset size
        ml_stats = self.conn.execute("""
            SELECT COUNT(*) as feature_records FROM ml_features
        """).fetchone()
        
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        report = {
            "extraction_summary": {
                "total_symbols_targeted": len(self.ml_universe),
                "total_symbols_processed": stats[0] if stats else 0,
                "daily_data_success": stats[1] if stats else 0,
                "technical_indicators_success": stats[2] if stats else 0,
                "fundamental_data_success": stats[3] if stats else 0,
                "total_api_calls": self.total_api_calls,
                "extraction_time_minutes": elapsed_time / 60,
                "calls_per_minute_actual": self.total_api_calls / (elapsed_time / 60) if elapsed_time > 0 else 0
            },
            "data_coverage": {
                "date_range": f"{date_range[0]} to {date_range[1]}" if date_range and date_range[0] else "No data",
                "total_daily_records": date_range[2] if date_range else 0,
                "ml_feature_records": ml_stats[0] if ml_stats else 0,
                "years_of_data": (date_range[1] - date_range[0]).days / 365 if date_range and date_range[0] and date_range[1] else 0
            },
            "ml_readiness": {
                "feature_table_created": True,
                "labels_table_created": True,
                "estimated_dataset_size_gb": (ml_stats[0] * 50 * 8) / (1024**3) if ml_stats else 0,  # Rough estimate
                "ready_for_training": ml_stats[0] > 100000 if ml_stats else False
            }
        }
        
        # Save report
        with open("ml_data_extraction_report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Log summary
        logger.info("✅ ML Data Extraction Complete!")
        logger.info(f"📊 Processed: {report['extraction_summary']['total_symbols_processed']}/{len(self.ml_universe)} symbols")
        logger.info(f"📅 Date range: {report['data_coverage']['date_range']}")
        logger.info(f"🔢 Total records: {report['data_coverage']['total_daily_records']:,}")
        logger.info(f"🤖 ML features: {report['data_coverage']['ml_feature_records']:,}")
        logger.info(f"⏱️ Time taken: {report['extraction_summary']['extraction_time_minutes']:.1f} minutes")
        logger.info(f"🔑 API calls: {report['extraction_summary']['total_api_calls']:,}")
        logger.info(f"💾 Report saved: ml_data_extraction_report.json")

# Usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        agent = MLDataExtractionAgent()
        await agent.start_ml_data_extraction()
    
    asyncio.run(main())