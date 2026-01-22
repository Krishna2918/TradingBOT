"""
Market Data Processor for Portfolio Optimization

Integrates with existing data collection infrastructure to provide
market data for 200 companies with 25 years of historical data.
"""

import os
import pandas as pd
import numpy as np
import duckdb
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from pathlib import Path
import logging

from ..interfaces.data_provider import IDataProvider
from ..config.settings import get_config
from ..utils.logger import get_logger
from ..utils.cache_manager import get_cache_manager
from ..utils.resource_monitor import get_resource_monitor
from ..exceptions.optimization_errors import DataError
from ..models.portfolio_state import PortfolioState


class MarketDataProcessor(IDataProvider):
    """
    Market data processor that integrates with existing data collection
    infrastructure and provides data for portfolio optimization.
    
    Supports 200 companies with up to 25 years of historical data.
    """
    
    def __init__(self, data_source: str = 'auto'):
        """
        Initialize market data processor.
        
        Args:
            data_source: Data source ('parquet', 'duckdb', 'auto')
        """
        self.config = get_config()
        self.logger = get_logger('market_data_processor')
        self.cache = get_cache_manager()
        self.resource_monitor = get_resource_monitor()
        
        self.data_source = data_source
        self._setup_data_sources()
        
        # Symbol management
        self._load_symbol_universe()
        
        # Data quality tracking
        self.data_quality_scores = {}
        self.last_update_times = {}
        
        self.logger.info(f"MarketDataProcessor initialized with {len(self.symbol_universe)} symbols")
    
    def _setup_data_sources(self) -> None:
        """Set up data source connections"""
        # Parquet data directory
        self.parquet_dir = Path('PastData/daily')
        
        # DuckDB connections
        self.main_db_path = 'data/market_data.duckdb'
        self.ml_db_path = 'PastData/ml_training_data.duckdb'
        
        # Check available data sources
        self.has_parquet = self.parquet_dir.exists()
        self.has_main_db = os.path.exists(self.main_db_path)
        self.has_ml_db = os.path.exists(self.ml_db_path)
        
        if self.data_source == 'auto':
            if self.has_parquet:
                self.data_source = 'parquet'
                self.logger.info("Using parquet files as primary data source")
            elif self.has_ml_db:
                self.data_source = 'duckdb_ml'
                self.logger.info("Using ML training database as primary data source")
            elif self.has_main_db:
                self.data_source = 'duckdb_main'
                self.logger.info("Using main database as primary data source")
            else:
                raise DataError("No data sources available", data_source="none")
    
    def _load_symbol_universe(self) -> None:
        """Load the symbol universe for portfolio optimization"""
        # Import existing symbol manager
        try:
            import sys
            sys.path.append('src')
            from data_collection.symbol_manager import SymbolManager
            
            symbol_manager = SymbolManager()
            existing_symbols = symbol_manager.get_all_symbols()
            
            # Expand to 200 companies by adding US large caps
            us_large_caps = self._get_us_large_cap_symbols()
            
            self.symbol_universe = existing_symbols + us_large_caps
            self.symbol_universe = list(set(self.symbol_universe))  # Remove duplicates
            
            # Limit to 200 companies for now
            self.symbol_universe = self.symbol_universe[:200]
            
            self.logger.info(f"Symbol universe loaded: {len(existing_symbols)} TSX + {len(us_large_caps)} US = {len(self.symbol_universe)} total")
            
        except ImportError:
            # Fallback symbol list if import fails
            self.symbol_universe = self._get_fallback_symbols()
            self.logger.warning("Using fallback symbol list")
    
    def _get_us_large_cap_symbols(self) -> List[str]:
        """Get US large cap symbols to expand to 200 companies"""
        # Top US large cap stocks to complement Canadian holdings
        us_symbols = [
            # Tech Giants
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "NFLX",
            "ADBE", "CRM", "ORCL", "INTC", "AMD", "QCOM", "AVGO", "TXN",
            
            # Financial Services
            "JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC", "TFC", "COF",
            "AXP", "BLK", "SCHW", "CB", "MMC", "AON", "SPGI", "ICE", "CME",
            
            # Healthcare
            "JNJ", "PFE", "UNH", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY",
            "AMGN", "GILD", "BIIB", "REGN", "VRTX", "ISRG", "SYK", "BSX",
            
            # Consumer
            "PG", "KO", "PEP", "WMT", "HD", "MCD", "NKE", "SBUX", "TGT",
            "LOW", "COST", "DIS", "CMCSA", "VZ", "T", "PM", "MO", "CL",
            
            # Industrial
            "BA", "CAT", "GE", "MMM", "HON", "UPS", "RTX", "LMT", "NOC",
            "GD", "DE", "EMR", "ETN", "ITW", "PH", "ROK", "DOV", "XYL",
            
            # Energy & Materials
            "XOM", "CVX", "COP", "EOG", "SLB", "PSX", "VLO", "MPC", "OXY",
            "BHP", "RIO", "FCX", "NEM", "GOLD", "AA", "X", "CLF", "NUE",
            
            # Utilities & REITs
            "NEE", "DUK", "SO", "D", "EXC", "SRE", "AEP", "XEL", "ED",
            "AMT", "PLD", "CCI", "EQIX", "PSA", "EXR", "AVB", "EQR", "UDR"
        ]
        
        return us_symbols[:100]  # Take top 100 to reach ~200 total
    
    def _get_fallback_symbols(self) -> List[str]:
        """Fallback symbol list if symbol manager import fails"""
        return [
            # Canadian Core
            "RY.TO", "TD.TO", "BNS.TO", "BMO.TO", "CM.TO", "SHOP.TO", "CSU.TO",
            "CNQ.TO", "SU.TO", "ENB.TO", "TRP.TO", "AEM.TO", "ABX.TO", "WPM.TO",
            
            # US Large Caps
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM",
            "JNJ", "PG", "KO", "WMT", "HD", "BA", "XOM", "CVX"
        ]
    
    @property
    def provider_name(self) -> str:
        """Get the name of the data provider"""
        return f"MarketDataProcessor({self.data_source})"
    
    @property
    def rate_limit_per_minute(self) -> int:
        """Get the rate limit for this provider"""
        return 0  # No rate limit for local data
    
    @property
    def supported_frequencies(self) -> List[str]:
        """Get list of supported data frequencies"""
        return ['daily', 'weekly', 'monthly']
    
    def get_market_data(
        self, 
        symbols: List[str], 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        frequency: str = 'daily'
    ) -> pd.DataFrame:
        """
        Retrieve market data for specified symbols.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date for data retrieval
            end_date: End date for data retrieval  
            frequency: Data frequency ('daily', 'weekly', 'monthly')
            
        Returns:
            DataFrame with OHLCV data indexed by date
        """
        # Check cache first
        cache_key = f"market_data_{hash(tuple(symbols))}_{start_date}_{end_date}_{frequency}"
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            self.logger.debug(f"Cache hit for market data: {len(symbols)} symbols")
            return cached_data
        
        # Record resource usage
        self.resource_monitor.record_api_call('local_data', 'market_data')
        
        try:
            if self.data_source == 'parquet':
                data = self._load_from_parquet(symbols, start_date, end_date, frequency)
            elif self.data_source.startswith('duckdb'):
                data = self._load_from_duckdb(symbols, start_date, end_date, frequency)
            else:
                raise DataError(f"Unsupported data source: {self.data_source}")
            
            # Cache the result
            self.cache.set(cache_key, data, ttl_seconds=3600)  # 1 hour cache
            
            # Update data quality scores
            self._update_data_quality_scores(symbols, data)
            
            self.logger.info(f"Loaded market data: {len(symbols)} symbols, {len(data)} records")
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading market data: {e}")
            raise DataError(f"Failed to load market data: {e}", data_source=self.data_source, symbols=symbols)
    
    def _load_from_parquet(
        self, 
        symbols: List[str], 
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        frequency: str
    ) -> pd.DataFrame:
        """Load data from parquet files"""
        all_data = []
        
        for symbol in symbols:
            # Handle both TSX (.TO) and US symbols
            if symbol.endswith('.TO'):
                file_path = self.parquet_dir / f"{symbol}.parquet"
            else:
                # For US symbols, might need different directory or naming
                file_path = self.parquet_dir / f"{symbol}.parquet"
                if not file_path.exists():
                    # Try without extension
                    file_path = self.parquet_dir / f"{symbol}.parquet"
            
            if file_path.exists():
                try:
                    df = pd.read_parquet(file_path)
                    
                    # Ensure date index
                    if not isinstance(df.index, pd.DatetimeIndex):
                        if 'Date' in df.columns:
                            df.set_index('Date', inplace=True)
                        elif 'date' in df.columns:
                            df.set_index('date', inplace=True)
                    
                    # Filter by date range
                    if start_date:
                        df = df[df.index >= start_date]
                    if end_date:
                        df = df[df.index <= end_date]
                    
                    # Add symbol column
                    df['Symbol'] = symbol
                    
                    # Resample if needed
                    if frequency == 'weekly':
                        df = df.resample('W').agg({
                            'Open': 'first',
                            'High': 'max',
                            'Low': 'min',
                            'Close': 'last',
                            'Volume': 'sum'
                        }).dropna()
                    elif frequency == 'monthly':
                        df = df.resample('M').agg({
                            'Open': 'first',
                            'High': 'max',
                            'Low': 'min',
                            'Close': 'last',
                            'Volume': 'sum'
                        }).dropna()
                    
                    all_data.append(df)
                    
                except Exception as e:
                    self.logger.warning(f"Error loading {symbol} from parquet: {e}")
            else:
                self.logger.warning(f"Parquet file not found for symbol: {symbol}")
        
        if all_data:
            combined_data = pd.concat(all_data, sort=True)
            return combined_data.sort_index()
        else:
            return pd.DataFrame()
    
    def _load_from_duckdb(
        self, 
        symbols: List[str], 
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        frequency: str
    ) -> pd.DataFrame:
        """Load data from DuckDB database"""
        db_path = self.ml_db_path if self.data_source == 'duckdb_ml' else self.main_db_path
        
        try:
            conn = duckdb.connect(db_path)
            
            # Build query
            symbol_list = "', '".join(symbols)
            query = f"""
            SELECT * FROM daily_ohlcv 
            WHERE symbol IN ('{symbol_list}')
            """
            
            if start_date:
                query += f" AND date >= '{start_date.strftime('%Y-%m-%d')}'"
            if end_date:
                query += f" AND date <= '{end_date.strftime('%Y-%m-%d')}'"
            
            query += " ORDER BY date, symbol"
            
            df = conn.execute(query).df()
            conn.close()
            
            if not df.empty:
                # Set date as index
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                
                # Rename columns to match expected format
                column_mapping = {
                    'open': 'Open',
                    'high': 'High', 
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume',
                    'symbol': 'Symbol'
                }
                df.rename(columns=column_mapping, inplace=True)
                
                # Handle frequency resampling
                if frequency != 'daily':
                    # Group by symbol and resample
                    resampled_data = []
                    for symbol in df['Symbol'].unique():
                        symbol_data = df[df['Symbol'] == symbol].copy()
                        
                        if frequency == 'weekly':
                            resampled = symbol_data.resample('W').agg({
                                'Open': 'first',
                                'High': 'max',
                                'Low': 'min',
                                'Close': 'last',
                                'Volume': 'sum'
                            }).dropna()
                        elif frequency == 'monthly':
                            resampled = symbol_data.resample('M').agg({
                                'Open': 'first',
                                'High': 'max',
                                'Low': 'min',
                                'Close': 'last',
                                'Volume': 'sum'
                            }).dropna()
                        
                        resampled['Symbol'] = symbol
                        resampled_data.append(resampled)
                    
                    if resampled_data:
                        df = pd.concat(resampled_data, sort=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading from DuckDB: {e}")
            return pd.DataFrame()
    
    def get_factor_data(
        self, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Retrieve factor data (momentum, value, quality, etc.).
        
        For now, we'll calculate basic factors from price data.
        In production, this would integrate with factor data providers.
        """
        # Get market data for factor calculation
        factor_symbols = self.symbol_universe[:50]  # Use subset for factor calculation
        
        market_data = self.get_market_data(
            factor_symbols, 
            start_date=start_date or datetime.now() - timedelta(days=365),
            end_date=end_date
        )
        
        if market_data.empty:
            return pd.DataFrame()
        
        # Calculate basic factors
        factor_data = self._calculate_basic_factors(market_data)
        
        return factor_data
    
    def _calculate_basic_factors(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic factor returns from market data"""
        factors = []
        
        # Get unique dates
        dates = market_data.index.unique().sort_values()
        
        for date in dates:
            daily_data = market_data[market_data.index == date]
            
            if len(daily_data) < 10:  # Need minimum number of stocks
                continue
            
            # Calculate returns
            prev_date_data = market_data[market_data.index < date].groupby('Symbol').last()
            current_data = daily_data.set_index('Symbol')
            
            # Only include symbols with both current and previous data
            common_symbols = prev_date_data.index.intersection(current_data.index)
            
            if len(common_symbols) < 10:
                continue
            
            prev_prices = prev_date_data.loc[common_symbols, 'Close']
            current_prices = current_data.loc[common_symbols, 'Close']
            
            returns = (current_prices / prev_prices - 1).dropna()
            
            if len(returns) < 10:
                continue
            
            # Market factor (equal-weighted market return)
            market_return = returns.mean()
            
            # Size factor (small minus big - simplified)
            # In production, this would use market cap data
            size_factor = returns.quantile(0.3) - returns.quantile(0.7)
            
            # Momentum factor (simplified - would use longer lookback)
            momentum_factor = returns.quantile(0.8) - returns.quantile(0.2)
            
            # Value factor (placeholder - would use P/B, P/E ratios)
            value_factor = np.random.normal(0, 0.01)  # Placeholder
            
            # Quality factor (placeholder - would use ROE, debt ratios)
            quality_factor = np.random.normal(0, 0.01)  # Placeholder
            
            factors.append({
                'date': date,
                'market': market_return,
                'size': size_factor,
                'momentum': momentum_factor,
                'value': value_factor,
                'quality': quality_factor,
                'low_volatility': -returns.std()  # Negative volatility as factor
            })
        
        if factors:
            factor_df = pd.DataFrame(factors)
            factor_df.set_index('date', inplace=True)
            return factor_df
        else:
            return pd.DataFrame()
    
    def get_risk_free_rate(
        self, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.Series:
        """
        Retrieve risk-free rate data.
        
        For now, returns a constant rate. In production, would integrate
        with government bond data or central bank rates.
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)
        if end_date is None:
            end_date = datetime.now()
        
        # Create date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Use different rates for different currencies/regions
        # Canadian 3-month treasury bill rate (approximate)
        risk_free_rate = 0.045 / 252  # 4.5% annual, converted to daily
        
        return pd.Series(risk_free_rate, index=date_range, name='risk_free_rate')
    
    def get_sector_data(self, symbols: List[str]) -> Dict[str, str]:
        """Get sector classification for symbols"""
        # Simplified sector mapping - in production would use data provider
        sector_mapping = {
            # Canadian Banks
            'RY.TO': 'Financials', 'TD.TO': 'Financials', 'BNS.TO': 'Financials',
            'BMO.TO': 'Financials', 'CM.TO': 'Financials', 'NA.TO': 'Financials',
            
            # Canadian Energy
            'CNQ.TO': 'Energy', 'SU.TO': 'Energy', 'ENB.TO': 'Energy',
            'TRP.TO': 'Energy', 'CVE.TO': 'Energy',
            
            # Canadian Tech
            'SHOP.TO': 'Information Technology', 'CSU.TO': 'Information Technology',
            'LSPD.TO': 'Information Technology',
            
            # US Tech
            'AAPL': 'Information Technology', 'MSFT': 'Information Technology',
            'GOOGL': 'Information Technology', 'AMZN': 'Consumer Discretionary',
            'META': 'Information Technology', 'TSLA': 'Consumer Discretionary',
            'NVDA': 'Information Technology',
            
            # US Financials
            'JPM': 'Financials', 'BAC': 'Financials', 'WFC': 'Financials',
            'GS': 'Financials', 'MS': 'Financials',
            
            # US Healthcare
            'JNJ': 'Health Care', 'PFE': 'Health Care', 'UNH': 'Health Care',
            'ABBV': 'Health Care', 'MRK': 'Health Care',
            
            # US Consumer
            'PG': 'Consumer Staples', 'KO': 'Consumer Staples', 'PEP': 'Consumer Staples',
            'WMT': 'Consumer Staples', 'HD': 'Consumer Discretionary',
            
            # US Industrial
            'BA': 'Industrials', 'CAT': 'Industrials', 'GE': 'Industrials',
            'MMM': 'Industrials', 'HON': 'Industrials',
            
            # US Energy
            'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy'
        }
        
        result = {}
        for symbol in symbols:
            result[symbol] = sector_mapping.get(symbol, 'Unknown')
        
        return result
    
    def get_market_cap_data(self, symbols: List[str]) -> Dict[str, float]:
        """Get market capitalization data for symbols"""
        # Simplified market cap data - in production would use real-time data
        market_caps = {}
        
        for symbol in symbols:
            if symbol in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']:
                market_caps[symbol] = 2e12  # $2T+ mega caps
            elif symbol in ['TSLA', 'NVDA', 'JPM', 'JNJ', 'PG']:
                market_caps[symbol] = 500e9  # $500B+ large caps
            elif symbol.endswith('.TO'):
                if symbol in ['RY.TO', 'TD.TO', 'SHOP.TO']:
                    market_caps[symbol] = 100e9  # $100B+ Canadian large caps
                else:
                    market_caps[symbol] = 20e9  # $20B+ Canadian mid caps
            else:
                market_caps[symbol] = 50e9  # $50B default for US stocks
        
        return market_caps
    
    def get_trading_volume_data(
        self, 
        symbols: List[str],
        lookback_days: int = 30
    ) -> Dict[str, float]:
        """Get average daily trading volume for symbols"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        market_data = self.get_market_data(symbols, start_date, end_date)
        
        volume_data = {}
        for symbol in symbols:
            symbol_data = market_data[market_data['Symbol'] == symbol]
            if not symbol_data.empty:
                avg_volume = symbol_data['Volume'].mean()
                volume_data[symbol] = avg_volume
            else:
                volume_data[symbol] = 1e6  # Default 1M shares
        
        return volume_data
    
    def is_data_stale(self, symbol: str, max_age_minutes: int = 60) -> bool:
        """Check if data for a symbol is stale"""
        if symbol in self.last_update_times:
            age = datetime.now() - self.last_update_times[symbol]
            return age.total_seconds() > (max_age_minutes * 60)
        return True  # No data means stale
    
    def get_data_quality_score(self, symbols: List[str]) -> Dict[str, float]:
        """Get data quality scores for symbols"""
        return {symbol: self.data_quality_scores.get(symbol, 0.5) for symbol in symbols}
    
    def _update_data_quality_scores(self, symbols: List[str], data: pd.DataFrame) -> None:
        """Update data quality scores based on loaded data"""
        for symbol in symbols:
            symbol_data = data[data['Symbol'] == symbol] if 'Symbol' in data.columns else data
            
            if symbol_data.empty:
                self.data_quality_scores[symbol] = 0.0
            else:
                # Calculate quality score based on data completeness and recency
                completeness = 1.0 - (symbol_data.isnull().sum().sum() / symbol_data.size)
                recency_score = 1.0 if not symbol_data.empty else 0.0
                
                quality_score = (completeness * 0.7) + (recency_score * 0.3)
                self.data_quality_scores[symbol] = min(1.0, max(0.0, quality_score))
            
            self.last_update_times[symbol] = datetime.now()
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols"""
        return self.symbol_universe.copy()
    
    def get_data_coverage_report(self) -> Dict[str, Any]:
        """Get comprehensive data coverage report"""
        report = {
            'total_symbols': len(self.symbol_universe),
            'data_source': self.data_source,
            'has_parquet': self.has_parquet,
            'has_main_db': self.has_main_db,
            'has_ml_db': self.has_ml_db,
            'quality_scores': self.data_quality_scores.copy(),
            'last_updates': {k: v.isoformat() for k, v in self.last_update_times.items()},
            'cache_stats': self.cache.get_stats()
        }
        
        # Add symbol breakdown
        canadian_symbols = [s for s in self.symbol_universe if s.endswith('.TO')]
        us_symbols = [s for s in self.symbol_universe if not s.endswith('.TO')]
        
        report['symbol_breakdown'] = {
            'canadian_symbols': len(canadian_symbols),
            'us_symbols': len(us_symbols),
            'sample_canadian': canadian_symbols[:10],
            'sample_us': us_symbols[:10]
        }
        
        return report