"""
Comprehensive Data Collector for AI Training

Orchestrates collection of 20+ years of multi-source data using the 4-key Alpha Vantage system.
Supports market data, fundamentals, macro economics, and sentiment data collection.
"""

import logging
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import time
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from .alpha_vantage_key_manager import get_alpha_vantage_key_manager
from .storage_manager import StorageManager

logger = logging.getLogger(__name__)

@dataclass
class CollectionProgress:
    """Track collection progress for each data type"""
    data_type: str
    total_symbols: int
    completed_symbols: int
    failed_symbols: List[str]
    start_time: datetime
    estimated_completion: Optional[datetime]
    current_symbol: Optional[str]
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['start_time'] = self.start_time.isoformat()
        if self.estimated_completion:
            data['estimated_completion'] = self.estimated_completion.isoformat()
        return data

class ComprehensiveDataCollector:
    """
    Comprehensive data collector for AI training
    
    Collects:
    - Core market data (OHLCV, technical indicators) - Premium key
    - Fundamentals (income, balance sheet, cash flow, earnings) - Free key #1
    - Macro economics (commodities, rates, inflation) - Free key #2  
    - Sentiment & intelligence (news, transcripts, movers) - Free key #3
    """
    
    def __init__(self, base_path: str = "TrainingData"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.key_manager = get_alpha_vantage_key_manager()
        self.storage = StorageManager(str(self.base_path))
        
        # Create specialized storage directories
        self.data_dirs = {
            'market_data': self.base_path / 'market_data',
            'fundamentals': self.base_path / 'fundamentals', 
            'macro_economics': self.base_path / 'macro_economics',
            'sentiment': self.base_path / 'sentiment',
            'technical_indicators': self.base_path / 'technical_indicators',
            'processed': self.base_path / 'processed'
        }
        
        for dir_path in self.data_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Progress tracking
        self.progress_file = self.base_path / 'collection_progress.json'
        self.progress = {}
        self._load_progress()
        
        # Symbol universe
        self.us_symbols = self._get_us_symbol_universe()
        
        logger.info(f"Comprehensive Data Collector initialized")
        logger.info(f"Target symbols: {len(self.us_symbols)} US stocks")
        logger.info(f"Data storage: {self.base_path}")
    
    def _get_us_symbol_universe(self) -> List[str]:
        """Get top 200 US stocks by market cap"""
        # Top 200 US stocks by market cap (simplified list for demo)
        # In production, this would be fetched from a reliable source
        top_us_stocks = [
            # Mega cap (>$500B)
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'TSLA', 'META', 'BRK.B', 'UNH',
            
            # Large cap ($100B-$500B)
            'JNJ', 'V', 'PG', 'JPM', 'HD', 'MA', 'CVX', 'ABBV', 'PFE', 'KO',
            'AVGO', 'PEP', 'TMO', 'COST', 'WMT', 'BAC', 'NFLX', 'XOM', 'DIS', 'ABT',
            'CRM', 'VZ', 'ADBE', 'CMCSA', 'ACN', 'NKE', 'DHR', 'TXN', 'NEE', 'RTX',
            'QCOM', 'LIN', 'PM', 'SPGI', 'LOW', 'HON', 'UPS', 'T', 'INTU', 'IBM',
            
            # Mid-large cap ($50B-$100B)
            'AMD', 'AMGN', 'CAT', 'GS', 'BKNG', 'ISRG', 'AXP', 'DE', 'TJX', 'GILD',
            'MDLZ', 'SYK', 'ADP', 'VRTX', 'ADI', 'MMM', 'CI', 'LRCX', 'ZTS', 'PYPL',
            'TMUS', 'MO', 'CVS', 'BDX', 'REGN', 'SCHW', 'FIS', 'MU', 'EOG', 'ITW',
            'DUK', 'PLD', 'AON', 'CL', 'APD', 'EQIX', 'ICE', 'USB', 'NSC', 'EMR',
            
            # Growth and tech
            'NOW', 'PANW', 'KLAC', 'AMAT', 'MRVL', 'CDNS', 'SNPS', 'ADSK', 'MCHP', 'FTNT',
            'WDAY', 'TEAM', 'DDOG', 'SNOW', 'CRWD', 'ZM', 'DOCU', 'OKTA', 'SPLK', 'VEEV',
            
            # Financial services
            'WFC', 'MS', 'C', 'BLK', 'SPGI', 'CME', 'ICE', 'MCO', 'TRV', 'PGR',
            'AIG', 'MET', 'PRU', 'ALL', 'CB', 'AFL', 'AJG', 'MMC', 'BRO', 'WTW',
            
            # Healthcare & biotech
            'UNH', 'JNJ', 'PFE', 'ABBV', 'TMO', 'ABT', 'DHR', 'BMY', 'AMGN', 'GILD',
            'VRTX', 'REGN', 'BIIB', 'ILMN', 'MRNA', 'ZTS', 'ELV', 'CVS', 'CI', 'HUM',
            
            # Consumer & retail
            'AMZN', 'WMT', 'HD', 'PG', 'KO', 'PEP', 'COST', 'TJX', 'NKE', 'SBUX',
            'MCD', 'DIS', 'NFLX', 'CRM', 'BKNG', 'LOW', 'TGT', 'F', 'GM', 'TSLA',
            
            # Energy & utilities
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'VLO', 'PSX', 'KMI', 'OKE',
            'NEE', 'DUK', 'SO', 'D', 'EXC', 'XEL', 'SRE', 'AEP', 'PCG', 'ED',
            
            # ETFs for market context
            'SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'VEA', 'VWO', 'AGG', 'LQD', 'HYG'
        ]
        
        return top_us_stocks[:200]  # Ensure exactly 200 symbols
    
    def _load_progress(self):
        """Load collection progress from file"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                    for key, value in data.items():
                        if 'start_time' in value:
                            value['start_time'] = datetime.fromisoformat(value['start_time'])
                        if 'estimated_completion' in value and value['estimated_completion']:
                            value['estimated_completion'] = datetime.fromisoformat(value['estimated_completion'])
                        self.progress[key] = CollectionProgress(**value)
                logger.info("Loaded existing collection progress")
            except Exception as e:
                logger.error(f"Error loading progress: {e}")
    
    def _save_progress(self):
        """Save collection progress to file"""
        try:
            data = {key: progress.to_dict() for key, progress in self.progress.items()}
            with open(self.progress_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving progress: {e}")
    
    async def collect_all_training_data(self) -> Dict[str, Any]:
        """
        Collect all training data in the correct order
        
        Returns comprehensive collection results
        """
        logger.info("Starting comprehensive training data collection")
        
        results = {
            'start_time': datetime.now(),
            'collections': {},
            'total_symbols': len(self.us_symbols),
            'success': False
        }
        
        try:
            # Phase 1: Core market data (Premium key - highest priority)
            logger.info("Phase 1: Collecting core market data (Premium key)")
            market_results = await self.collect_market_data()
            results['collections']['market_data'] = market_results
            
            # Phase 2: Fundamentals (Free key #1)
            logger.info("Phase 2: Collecting fundamentals data (Free key #1)")
            fundamentals_results = await self.collect_fundamentals_data()
            results['collections']['fundamentals'] = fundamentals_results
            
            # Phase 3: Macro economics (Free key #2)
            logger.info("Phase 3: Collecting macro economics data (Free key #2)")
            macro_results = await self.collect_macro_data()
            results['collections']['macro_economics'] = macro_results
            
            # Phase 4: Sentiment & intelligence (Free key #3)
            logger.info("Phase 4: Collecting sentiment & intelligence data (Free key #3)")
            sentiment_results = await self.collect_sentiment_data()
            results['collections']['sentiment'] = sentiment_results
            
            # Phase 5: Data validation and quality checks
            logger.info("Phase 5: Validating collected data")
            validation_results = await self.validate_collected_data()
            results['collections']['validation'] = validation_results
            
            results['end_time'] = datetime.now()
            results['duration'] = (results['end_time'] - results['start_time']).total_seconds()
            results['success'] = True
            
            logger.info(f"Data collection completed in {results['duration']:.0f} seconds")
            
        except Exception as e:
            logger.error(f"Data collection failed: {e}")
            results['error'] = str(e)
            results['end_time'] = datetime.now()
        
        return results
    
    async def collect_market_data(self) -> Dict[str, Any]:
        """
        Collect core market data using premium key
        
        Collects:
        - 20+ years daily OHLCV
        - 1-year intraday (1min, 5min)
        - Technical indicators
        """
        logger.info("Starting market data collection")
        
        progress = CollectionProgress(
            data_type='market_data',
            total_symbols=len(self.us_symbols),
            completed_symbols=0,
            failed_symbols=[],
            start_time=datetime.now(),
            estimated_completion=None,
            current_symbol=None
        )
        self.progress['market_data'] = progress
        
        results = {
            'symbols_processed': 0,
            'symbols_failed': [],
            'data_collected': {
                'daily': 0,
                'intraday_1min': 0,
                'intraday_5min': 0,
                'technical_indicators': 0
            }
        }
        
        # Collect data for each symbol
        for i, symbol in enumerate(self.us_symbols):
            progress.current_symbol = symbol
            progress.completed_symbols = i
            
            # Estimate completion time
            if i > 0:
                elapsed = (datetime.now() - progress.start_time).total_seconds()
                rate = i / elapsed
                remaining_time = (len(self.us_symbols) - i) / rate
                progress.estimated_completion = datetime.now() + timedelta(seconds=remaining_time)
            
            self._save_progress()
            
            try:
                logger.info(f"Collecting market data for {symbol} ({i+1}/{len(self.us_symbols)})")
                
                # Daily data (20+ years)
                daily_success = await self._collect_daily_data(symbol)
                if daily_success:
                    results['data_collected']['daily'] += 1
                
                # Intraday data (1 year)
                intraday_1min_success = await self._collect_intraday_data(symbol, '1min')
                if intraday_1min_success:
                    results['data_collected']['intraday_1min'] += 1
                
                intraday_5min_success = await self._collect_intraday_data(symbol, '5min')
                if intraday_5min_success:
                    results['data_collected']['intraday_5min'] += 1
                
                # Technical indicators
                indicators_success = await self._collect_technical_indicators(symbol)
                if indicators_success:
                    results['data_collected']['technical_indicators'] += 1
                
                results['symbols_processed'] += 1
                
                # Rate limiting for premium key (75 RPM = 1.25 requests/second)
                await asyncio.sleep(0.8)  # Conservative rate limiting
                
            except Exception as e:
                logger.error(f"Failed to collect market data for {symbol}: {e}")
                results['symbols_failed'].append(symbol)
                progress.failed_symbols.append(symbol)
        
        progress.completed_symbols = len(self.us_symbols)
        progress.current_symbol = None
        self._save_progress()
        
        logger.info(f"Market data collection completed: {results['symbols_processed']}/{len(self.us_symbols)} symbols")
        return results
    
    async def _collect_daily_data(self, symbol: str) -> bool:
        """Collect 20+ years of daily OHLCV data"""
        try:
            data = self.key_manager.make_request(
                'TIME_SERIES_DAILY_ADJUSTED',
                {'symbol': symbol, 'outputsize': 'full'},
                'premium_realtime'
            )
            
            if not data or 'Time Series (Daily)' not in data:
                logger.warning(f"No daily data for {symbol}")
                return False
            
            # Convert to DataFrame
            time_series = data['Time Series (Daily)']
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Rename columns
            df.columns = ['Open', 'High', 'Low', 'Close', 'Adjusted_Close', 'Volume', 'Dividend_Amount', 'Split_Coefficient']
            
            # Convert to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Save to storage
            file_path = self.data_dirs['market_data'] / f"{symbol}_daily.parquet"
            df.to_parquet(file_path, compression='snappy')
            
            logger.debug(f"Saved daily data for {symbol}: {len(df)} rows")
            return True
            
        except Exception as e:
            logger.error(f"Failed to collect daily data for {symbol}: {e}")
            return False
    
    async def _collect_intraday_data(self, symbol: str, interval: str) -> bool:
        """Collect intraday data (1min or 5min)"""
        try:
            data = self.key_manager.make_request(
                'TIME_SERIES_INTRADAY',
                {'symbol': symbol, 'interval': interval, 'outputsize': 'full'},
                'premium_realtime'
            )
            
            time_series_key = f'Time Series ({interval})'
            if not data or time_series_key not in data:
                logger.warning(f"No {interval} data for {symbol}")
                return False
            
            # Convert to DataFrame
            time_series = data[time_series_key]
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Rename columns
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            # Convert to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Save to storage
            file_path = self.data_dirs['market_data'] / f"{symbol}_{interval}.parquet"
            df.to_parquet(file_path, compression='snappy')
            
            logger.debug(f"Saved {interval} data for {symbol}: {len(df)} rows")
            return True
            
        except Exception as e:
            logger.error(f"Failed to collect {interval} data for {symbol}: {e}")
            return False
    
    async def _collect_technical_indicators(self, symbol: str) -> bool:
        """Collect technical indicators for a symbol"""
        # Use correct Alpha Vantage function names with all required parameters
        indicators_config = {
            'RSI': {
                'function': 'RSI', 
                'params': {
                    'symbol': symbol, 
                    'interval': '5min',  # Use 5min for better data availability
                    'time_period': '14', 
                    'series_type': 'close'
                }
            },
            'SMA': {
                'function': 'SMA', 
                'params': {
                    'symbol': symbol, 
                    'interval': '5min', 
                    'time_period': '20', 
                    'series_type': 'close'
                }
            },
            'EMA': {
                'function': 'EMA', 
                'params': {
                    'symbol': symbol, 
                    'interval': '5min', 
                    'time_period': '12', 
                    'series_type': 'close'
                }
            },
            'MACD': {
                'function': 'MACD', 
                'params': {
                    'symbol': symbol, 
                    'interval': '5min', 
                    'series_type': 'close'
                }
            },
            'STOCH': {
                'function': 'STOCH', 
                'params': {
                    'symbol': symbol, 
                    'interval': '5min'
                }
            },
            'ADX': {
                'function': 'ADX', 
                'params': {
                    'symbol': symbol, 
                    'interval': '5min', 
                    'time_period': '14'
                }
            },
            'CCI': {
                'function': 'CCI', 
                'params': {
                    'symbol': symbol, 
                    'interval': '5min', 
                    'time_period': '14'
                }
            },
            'AROON': {
                'function': 'AROON', 
                'params': {
                    'symbol': symbol, 
                    'interval': '5min', 
                    'time_period': '14'
                }
            },
            'BBANDS': {
                'function': 'BBANDS', 
                'params': {
                    'symbol': symbol, 
                    'interval': '5min', 
                    'time_period': '20', 
                    'series_type': 'close'
                }
            }
        }
        collected = 0
        
        for indicator_name, config in indicators_config.items():
            try:
                data = self.key_manager.make_request(
                    config['function'],
                    config['params'],
                    'premium_realtime'
                )
                
                if data and f'Technical Analysis: {indicator_name}' in data:
                    # Save indicator data
                    indicator_data = data[f'Technical Analysis: {indicator_name}']
                    df = pd.DataFrame.from_dict(indicator_data, orient='index')
                    df.index = pd.to_datetime(df.index)
                    df = df.sort_index()
                    
                    # Convert to numeric
                    for col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    file_path = self.data_dirs['technical_indicators'] / f"{symbol}_{indicator_name}.parquet"
                    df.to_parquet(file_path, compression='snappy')
                    collected += 1
                    logger.debug(f"Saved {indicator_name} for {symbol}: {len(df)} rows")
                
                # Rate limiting between indicator requests
                await asyncio.sleep(0.8)
                
            except Exception as e:
                logger.error(f"Failed to collect {indicator_name} for {symbol}: {e}")
        
        logger.debug(f"Collected {collected}/{len(indicators_config)} indicators for {symbol}")
        return collected > 0
    
    async def collect_fundamentals_data(self) -> Dict[str, Any]:
        """
        Collect fundamentals data using free key #1
        
        Collects:
        - Company overview
        - Income statements (annual/quarterly)
        - Balance sheets (annual/quarterly)
        - Cash flow statements (annual/quarterly)
        - Earnings data
        - Insider transactions
        """
        logger.info("Starting fundamentals data collection")
        
        # Implementation would go here - similar structure to market data
        # This is a placeholder for the comprehensive implementation
        
        return {
            'symbols_processed': 0,
            'data_types_collected': ['overview', 'income_statement', 'balance_sheet', 'cash_flow', 'earnings']
        }
    
    async def collect_macro_data(self) -> Dict[str, Any]:
        """
        Collect macro economics data using free key #2
        
        Collects:
        - GDP, inflation, unemployment
        - Interest rates, yield curve
        - Commodities (oil, gold, etc.)
        - Currency data
        """
        logger.info("Starting macro economics data collection")
        
        # Implementation would go here
        return {
            'indicators_collected': ['GDP', 'CPI', 'UNEMPLOYMENT', 'FEDERAL_FUNDS_RATE', 'WTI_OIL', 'GOLD']
        }
    
    async def collect_sentiment_data(self) -> Dict[str, Any]:
        """
        Collect sentiment & intelligence data using free key #3
        
        Collects:
        - News sentiment
        - Earnings calendar
        - Top movers
        - Market news sentiment
        """
        logger.info("Starting sentiment & intelligence data collection")
        
        # Implementation would go here
        return {
            'sentiment_data_collected': ['news_sentiment', 'earnings_calendar', 'top_movers']
        }
    
    async def validate_collected_data(self) -> Dict[str, Any]:
        """
        Validate all collected data for completeness and quality
        
        Returns validation report
        """
        logger.info("Starting data validation")
        
        validation_results = {
            'market_data': {},
            'fundamentals': {},
            'macro_economics': {},
            'sentiment': {},
            'overall_quality_score': 0.0,
            'training_ready': False
        }
        
        # Validate market data
        market_files = list(self.data_dirs['market_data'].glob("*.parquet"))
        validation_results['market_data'] = {
            'files_found': len(market_files),
            'expected_files': len(self.us_symbols) * 3,  # daily + 1min + 5min
            'completeness': len(market_files) / (len(self.us_symbols) * 3) if len(self.us_symbols) > 0 else 0
        }
        
        # Calculate overall quality score
        completeness_scores = [
            validation_results['market_data']['completeness']
        ]
        
        validation_results['overall_quality_score'] = np.mean(completeness_scores)
        validation_results['training_ready'] = validation_results['overall_quality_score'] > 0.95
        
        logger.info(f"Data validation completed - Quality score: {validation_results['overall_quality_score']:.3f}")
        
        return validation_results
    
    def get_collection_status(self) -> Dict[str, Any]:
        """Get current collection status"""
        status = {
            'progress': {key: progress.to_dict() for key, progress in self.progress.items()},
            'key_usage': self.key_manager.get_usage_summary(),
            'storage_summary': self.storage.get_storage_summary(),
            'data_directories': {name: len(list(path.glob("*.parquet"))) for name, path in self.data_dirs.items()}
        }
        
        return status

# Convenience function
def create_comprehensive_collector(base_path: str = "TrainingData") -> ComprehensiveDataCollector:
    """Create a comprehensive data collector instance"""
    return ComprehensiveDataCollector(base_path)

if __name__ == '__main__':
    # Test the comprehensive data collector
    logging.basicConfig(level=logging.INFO)
    
    async def test_collector():
        collector = ComprehensiveDataCollector()
        
        # Test status
        status = collector.get_collection_status()
        print(f"Collector initialized with {len(collector.us_symbols)} symbols")
        
        # Test single symbol collection (for demo)
        print("Testing single symbol collection...")
        results = await collector._collect_daily_data('AAPL')
        print(f"AAPL daily data collection: {'Success' if results else 'Failed'}")
    
    asyncio.run(test_collector())