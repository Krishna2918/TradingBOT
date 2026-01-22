"""
Hybrid ML System - Continuous Data Extraction + ML Training

Runs 24/7 collecting historical data (20 years) + live data when market open
+ simultaneous ML training without disturbing data collection.
"""

import logging
import time
import asyncio
import threading
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import duckdb
import json

from ..data_collection.market_calendar import MarketCalendar
from ..data_collection.alpha_vantage_collector import AlphaVantageCollector
from .ml_data_extraction_agent import MLDataExtractionAgent

logger = logging.getLogger(__name__)

class HybridMLSystem:
    """Hybrid system: continuous data extraction + ML training"""
    
    def __init__(self):
        self.market_calendar = MarketCalendar()
        self.alpha_vantage = AlphaVantageCollector()
        
        # Data storage in specified directory
        self.data_dir = Path("PastData")
        self.data_dir.mkdir(exist_ok=True)
        
        # Configuration
        self.config = {
            "alpha_vantage_calls_per_minute": 74,
            "historical_extraction_priority": 0.7,  # 70% for historical, 30% for live
            "ml_training_interval_minutes": 30,     # Train ML every 30 minutes
            "live_data_interval_minutes": 5,       # Collect live data every 5 minutes
            "checkpoint_interval_minutes": 60,     # Save progress every hour
        }
        
        # System state
        self.is_running = False
        self.extraction_thread = None
        self.training_thread = None
        self.live_data_thread = None
        
        # Progress tracking
        self.extraction_progress = {"completed": 0, "total": 0, "current_phase": ""}
        self.training_progress = {"models_trained": 0, "last_training": None}
        
        logger.info("🤖 Hybrid ML System initialized")
        logger.info(f"💾 Data directory: {self.data_dir}")
    
    async def start_hybrid_system(self):
        """Start the hybrid ML system with all components"""
        
        logger.info("🚀 Starting Hybrid ML System - Continuous Operation")
        self.is_running = True
        
        # Start all components concurrently
        tasks = [
            self.run_historical_extraction(),
            self.run_live_data_collection(),
            self.run_ml_training_pipeline(),
            self.run_system_monitor()
        ]
        
        await asyncio.gather(*tasks)
    
    async def run_historical_extraction(self):
        """Run continuous historical data extraction"""
        
        logger.info("📊 Starting historical data extraction...")
        
        # Initialize ML data extraction agent
        ml_agent = MLDataExtractionAgent()
        
        # Start extraction
        await ml_agent.start_ml_data_extraction()
    
    async def run_live_data_collection(self):
        """Collect live data when market is open"""
        
        logger.info("📈 Starting live data collection...")
        
        while self.is_running:
            try:
                if self.market_calendar.is_market_open_now():
                    logger.info("🔴 Market OPEN - Collecting live data")
                    await self.collect_live_market_data()
                else:
                    logger.debug("🔵 Market CLOSED - Skipping live collection")
                
                # Wait for next collection cycle
                await asyncio.sleep(self.config["live_data_interval_minutes"] * 60)
                
            except Exception as e:
                logger.error(f"❌ Live data collection error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def collect_live_market_data(self):
        """Collect current market data for active stocks"""
        
        # Get list of stocks we're tracking
        db_path = self.data_dir / "ml_training_data.duckdb"
        if not db_path.exists():
            return
        
        conn = duckdb.connect(str(db_path))
        
        try:
            # Get unique symbols from database
            symbols = conn.execute("""
                SELECT DISTINCT symbol FROM daily_ohlcv 
                ORDER BY symbol LIMIT 100
            """).fetchall()
            
            if not symbols:
                return
            
            logger.info(f"📈 Collecting live data for {len(symbols)} stocks")
            
            # Collect current quotes (using available API calls)
            for symbol, in symbols[:20]:  # Limit to 20 stocks for live updates
                try:
                    # Get intraday data
                    data, source = self.alpha_vantage.fetch_intraday_data(symbol, interval="5min")
                    
                    if data is not None and not data.empty:
                        # Store latest data point
                        latest = data.iloc[-1]
                        
                        # Insert into live_data table
                        conn.execute("""
                            CREATE TABLE IF NOT EXISTS live_data (
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
                            INSERT OR REPLACE INTO live_data VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, [symbol, datetime.now(), latest['Open'], latest['High'], 
                             latest['Low'], latest['Close'], latest['Volume']])
                        
                        logger.debug(f"✅ Live data: {symbol} @ ${latest['Close']:.2f}")
                    
                    await asyncio.sleep(1)  # Rate limiting
                    
                except Exception as e:
                    logger.debug(f"❌ Live data error for {symbol}: {e}")
        
        finally:
            conn.close()
    
    async def run_ml_training_pipeline(self):
        """Run ML training pipeline continuously"""
        
        logger.info("🤖 Starting ML training pipeline...")
        
        while self.is_running:
            try:
                # Wait for sufficient data before training
                if await self.check_training_readiness():
                    logger.info("🧠 Starting ML training cycle...")
                    await self.train_ml_models()
                    self.training_progress["models_trained"] += 1
                    self.training_progress["last_training"] = datetime.now()
                else:
                    logger.debug("⏳ Waiting for more data before ML training")
                
                # Wait for next training cycle
                await asyncio.sleep(self.config["ml_training_interval_minutes"] * 60)
                
            except Exception as e:
                logger.error(f"❌ ML training error: {e}")
                await asyncio.sleep(600)  # Wait 10 minutes on error
    
    async def check_training_readiness(self) -> bool:
        """Check if we have enough data for ML training"""
        
        db_path = self.data_dir / "ml_training_data.duckdb"
        if not db_path.exists():
            return False
        
        conn = duckdb.connect(str(db_path))
        
        try:
            # Check if we have sufficient data
            result = conn.execute("""
                SELECT COUNT(*) as record_count,
                       COUNT(DISTINCT symbol) as symbol_count
                FROM daily_ohlcv
            """).fetchone()
            
            if result and result[0] > 10000 and result[1] > 10:  # 10K records, 10+ symbols
                return True
            
        except Exception as e:
            logger.debug(f"Training readiness check error: {e}")
        
        finally:
            conn.close()
        
        return False
    
    async def train_ml_models(self):
        """Train ML models on available data"""
        
        logger.info("🧠 Training ML models...")
        
        db_path = self.data_dir / "ml_training_data.duckdb"
        conn = duckdb.connect(str(db_path))
        
        try:
            # Create ML features if not exists
            conn.execute("""
                CREATE OR REPLACE TABLE ml_features AS
                SELECT 
                    symbol,
                    date,
                    close,
                    volume,
                    
                    -- Price features
                    (close - LAG(close, 1) OVER (PARTITION BY symbol ORDER BY date)) / LAG(close, 1) OVER (PARTITION BY symbol ORDER BY date) AS return_1d,
                    (close - LAG(close, 5) OVER (PARTITION BY symbol ORDER BY date)) / LAG(close, 5) OVER (PARTITION BY symbol ORDER BY date) AS return_5d,
                    
                    -- Volume features
                    volume / AVG(volume) OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS volume_ratio,
                    
                    -- Volatility
                    STDDEV(close) OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS volatility_20d,
                    
                    -- Labels
                    CASE WHEN LEAD(close, 1) OVER (PARTITION BY symbol ORDER BY date) > close THEN 1 ELSE 0 END AS next_day_up
                    
                FROM daily_ohlcv
                WHERE date >= '2020-01-01'  -- Use recent data for training
                ORDER BY symbol, date
            """)
            
            # Get training data
            training_data = conn.execute("""
                SELECT * FROM ml_features 
                WHERE return_1d IS NOT NULL 
                  AND volume_ratio IS NOT NULL 
                  AND volatility_20d IS NOT NULL
                  AND next_day_up IS NOT NULL
                LIMIT 50000
            """).df()
            
            if len(training_data) > 1000:
                # Simple ML training (placeholder - would use sklearn/tensorflow in production)
                logger.info(f"🧠 Training on {len(training_data)} samples")
                
                # Save training results
                training_results = {
                    "timestamp": datetime.now().isoformat(),
                    "samples_used": len(training_data),
                    "features": ["return_1d", "return_5d", "volume_ratio", "volatility_20d"],
                    "target": "next_day_up",
                    "model_type": "placeholder",
                    "accuracy": 0.55  # Placeholder
                }
                
                # Save to file
                results_file = self.data_dir / "ml_training_results.json"
                with open(results_file, 'w') as f:
                    json.dump(training_results, f, indent=2)
                
                logger.info(f"✅ ML training complete - Results saved to {results_file}")
            else:
                logger.warning("⚠️ Insufficient data for ML training")
        
        finally:
            conn.close()
    
    async def run_system_monitor(self):
        """Monitor system health and progress"""
        
        logger.info("📊 Starting system monitor...")
        
        while self.is_running:
            try:
                # Generate status report
                await self.generate_status_report()
                
                # Wait for next monitoring cycle
                await asyncio.sleep(self.config["checkpoint_interval_minutes"] * 60)
                
            except Exception as e:
                logger.error(f"❌ System monitor error: {e}")
                await asyncio.sleep(300)
    
    async def generate_status_report(self):
        """Generate comprehensive system status report"""
        
        db_path = self.data_dir / "ml_training_data.duckdb"
        
        status = {
            "timestamp": datetime.now().isoformat(),
            "system_status": "running" if self.is_running else "stopped",
            "market_status": "open" if self.market_calendar.is_market_open_now() else "closed",
            "data_extraction": self.extraction_progress,
            "ml_training": self.training_progress,
            "database_stats": {}
        }
        
        # Get database statistics
        if db_path.exists():
            conn = duckdb.connect(str(db_path))
            try:
                # Daily data stats
                daily_stats = conn.execute("""
                    SELECT 
                        COUNT(*) as total_records,
                        COUNT(DISTINCT symbol) as unique_symbols,
                        MIN(date) as earliest_date,
                        MAX(date) as latest_date
                    FROM daily_ohlcv
                """).fetchone()
                
                if daily_stats:
                    status["database_stats"]["daily_data"] = {
                        "total_records": daily_stats[0],
                        "unique_symbols": daily_stats[1],
                        "date_range": f"{daily_stats[2]} to {daily_stats[3]}"
                    }
                
                # Live data stats
                live_stats = conn.execute("""
                    SELECT COUNT(*) as live_records
                    FROM live_data
                    WHERE timestamp >= datetime('now', '-1 day')
                """).fetchone()
                
                if live_stats:
                    status["database_stats"]["live_data_24h"] = live_stats[0]
                
            except Exception as e:
                status["database_stats"]["error"] = str(e)
            finally:
                conn.close()
        
        # Save status report
        status_file = self.data_dir / "system_status.json"
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)
        
        # Log key metrics
        if "daily_data" in status["database_stats"]:
            daily_data = status["database_stats"]["daily_data"]
            logger.info(f"📊 System Status: {daily_data['total_records']:,} records, "
                       f"{daily_data['unique_symbols']} symbols, "
                       f"ML models: {self.training_progress['models_trained']}")
    
    def stop_system(self):
        """Stop the hybrid ML system"""
        
        logger.info("🛑 Stopping Hybrid ML System...")
        self.is_running = False

# Usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        system = HybridMLSystem()
        await system.start_hybrid_system()
    
    asyncio.run(main())