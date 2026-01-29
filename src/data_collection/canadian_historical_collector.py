"""
Historical Canadian Market Data Collector

Collects 20 years of historical data for:
- TSX Composite Index
- TSX 60 (major stocks)
- Key Canadian sectors (Energy, Banks, Mining, Tech)
- Individual Canadian stocks

Data sources:
- Yahoo Finance (free, reliable for historical)
- Alpha Vantage (backup)
- Questrade (if configured)
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import os

logger = logging.getLogger(__name__)

try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logger.warning("pandas not available")

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("yfinance not available - historical data limited")


@dataclass
class HistoricalDataConfig:
    """Configuration for historical data collection"""
    years_of_history: int = 20
    data_directory: str = "data/historical"
    include_dividends: bool = True
    include_splits: bool = True
    auto_adjust: bool = True  # Adjust for splits/dividends


class CanadianMarketHistoricalCollector:
    """
    Collects and manages 20 years of Canadian market historical data
    """

    # TSX Composite and major indices
    INDICES = {
        'TSX_COMPOSITE': '^GSPTSE',      # S&P/TSX Composite Index
        'TSX_60': '^TX60',               # S&P/TSX 60 Index
        'TSX_VENTURE': '^JX',            # S&P/TSX Venture Composite
        'TSX_CAPPED_ENERGY': 'XEG.TO',   # Energy sector ETF (proxy)
        'TSX_CAPPED_FINANCIALS': 'XFN.TO', # Financials sector ETF
        'TSX_CAPPED_MATERIALS': 'XMA.TO',  # Materials (mining) ETF
    }

    # Top 60 TSX stocks by sector
    TSX_UNIVERSE = {
        'banks': [
            'RY.TO',   # Royal Bank of Canada
            'TD.TO',   # Toronto-Dominion Bank
            'BNS.TO',  # Bank of Nova Scotia
            'BMO.TO',  # Bank of Montreal
            'CM.TO',   # CIBC
            'NA.TO',   # National Bank
        ],
        'energy': [
            'CNQ.TO',  # Canadian Natural Resources
            'SU.TO',   # Suncor Energy
            'IMO.TO',  # Imperial Oil
            'CVE.TO',  # Cenovus Energy
            'TOU.TO',  # Tourmaline Oil
            'ARX.TO',  # ARC Resources
            'OVV.TO',  # Ovintiv
            'MEG.TO',  # MEG Energy
        ],
        'pipelines': [
            'ENB.TO',  # Enbridge
            'TRP.TO',  # TC Energy
            'PPL.TO',  # Pembina Pipeline
            'KEY.TO',  # Keyera
            'IPL.TO',  # Inter Pipeline
        ],
        'mining': [
            'ABX.TO',  # Barrick Gold
            'WPM.TO',  # Wheaton Precious Metals
            'FNV.TO',  # Franco-Nevada
            'AEM.TO',  # Agnico Eagle
            'K.TO',    # Kinross Gold
            'NTR.TO',  # Nutrien (fertilizers)
            'TECK-B.TO', # Teck Resources
            'FM.TO',   # First Quantum Minerals
            'LUN.TO',  # Lundin Mining
        ],
        'technology': [
            'SHOP.TO', # Shopify
            'CSU.TO',  # Constellation Software
            'OTEX.TO', # Open Text
            'DSG.TO',  # Descartes Systems
            'KXS.TO',  # Kinaxis
            'LSPD.TO', # Lightspeed Commerce
        ],
        'telecom': [
            'BCE.TO',  # BCE Inc
            'T.TO',    # TELUS
            'RCI-B.TO', # Rogers Communications
            'QBR-B.TO', # Quebecor
        ],
        'utilities': [
            'FTS.TO',  # Fortis
            'EMA.TO',  # Emera
            'H.TO',    # Hydro One
            'AQN.TO',  # Algonquin Power
            'CPX.TO',  # Capital Power
        ],
        'industrials': [
            'CNR.TO',  # Canadian National Railway
            'CP.TO',   # Canadian Pacific Railway
            'WCN.TO',  # Waste Connections
            'TIH.TO',  # Toromont Industries
            'WSP.TO',  # WSP Global
            'STN.TO',  # Stantec
            'TFII.TO', # TFI International
        ],
        'real_estate': [
            'RioCan.TO',  # RioCan REIT (REI-UN.TO)
            'CAR-UN.TO',  # Canadian Apartment REIT
            'BPY-UN.TO',  # Brookfield Property Partners
            'HR-UN.TO',   # H&R REIT
        ],
        'consumer': [
            'ATD.TO',  # Alimentation Couche-Tard
            'L.TO',    # Loblaw Companies
            'MRU.TO',  # Metro Inc
            'DOL.TO',  # Dollarama
            'CTC-A.TO', # Canadian Tire
            'QSR.TO',  # Restaurant Brands International
        ],
        'healthcare': [
            'WSP.TO',  # Bausch Health (BHC.TO)
            # Cannabis (volatile)
            # 'WEED.TO', # Canopy Growth
            # 'ACB.TO',  # Aurora Cannabis
        ],
        'etfs': [
            'XIU.TO',  # iShares S&P/TSX 60
            'XIC.TO',  # iShares Core S&P/TSX
            'VCN.TO',  # Vanguard FTSE Canada
            'ZCN.TO',  # BMO S&P/TSX Capped Composite
            'XEG.TO',  # iShares S&P/TSX Capped Energy
            'XFN.TO',  # iShares S&P/TSX Capped Financials
            'XGD.TO',  # iShares S&P/TSX Global Gold
            'ZAG.TO',  # BMO Aggregate Bond
            'VFV.TO',  # Vanguard S&P 500 (USD hedged)
        ]
    }

    def __init__(self, config: Optional[HistoricalDataConfig] = None):
        """
        Initialize historical data collector

        Args:
            config: Configuration for data collection
        """
        self.config = config or HistoricalDataConfig()
        self.data_dir = Path(self.config.data_directory)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Calculate date range
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=365 * self.config.years_of_history)

        # Data storage
        self.historical_data: Dict[str, pd.DataFrame] = {}
        self.metadata: Dict[str, Any] = {}

        logger.info(f"Historical Collector initialized")
        logger.info(f"  Date range: {self.start_date.date()} to {self.end_date.date()}")
        logger.info(f"  Data directory: {self.data_dir}")

    def collect_all(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Collect all historical data for Canadian markets

        Args:
            force_refresh: If True, re-download even if cached

        Returns:
            Summary of collected data
        """
        logger.info("Starting comprehensive Canadian market data collection...")

        summary = {
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'indices': {},
            'stocks': {},
            'sectors': {},
            'errors': []
        }

        # 1. Collect indices
        logger.info("Collecting index data...")
        for name, symbol in self.INDICES.items():
            try:
                data = self._fetch_historical(symbol, name, force_refresh)
                if data is not None and not data.empty:
                    summary['indices'][name] = {
                        'symbol': symbol,
                        'records': len(data),
                        'start': data.index[0].isoformat() if hasattr(data.index[0], 'isoformat') else str(data.index[0]),
                        'end': data.index[-1].isoformat() if hasattr(data.index[-1], 'isoformat') else str(data.index[-1])
                    }
            except Exception as e:
                logger.error(f"Error collecting {name}: {e}")
                summary['errors'].append(f"{name}: {str(e)}")

        # 2. Collect stocks by sector
        logger.info("Collecting stock data by sector...")
        for sector, symbols in self.TSX_UNIVERSE.items():
            summary['sectors'][sector] = {'stocks': [], 'errors': []}

            for symbol in symbols:
                try:
                    data = self._fetch_historical(symbol, symbol, force_refresh)
                    if data is not None and not data.empty:
                        summary['stocks'][symbol] = {
                            'sector': sector,
                            'records': len(data),
                            'start': data.index[0].isoformat() if hasattr(data.index[0], 'isoformat') else str(data.index[0]),
                            'end': data.index[-1].isoformat() if hasattr(data.index[-1], 'isoformat') else str(data.index[-1])
                        }
                        summary['sectors'][sector]['stocks'].append(symbol)
                except Exception as e:
                    logger.error(f"Error collecting {symbol}: {e}")
                    summary['sectors'][sector]['errors'].append(f"{symbol}: {str(e)}")

        # Save summary
        summary_file = self.data_dir / "collection_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Data collection complete. Summary saved to {summary_file}")
        return summary

    def _fetch_historical(
        self,
        symbol: str,
        name: str,
        force_refresh: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical data for a symbol

        Args:
            symbol: Yahoo Finance symbol
            name: Friendly name for caching
            force_refresh: Force re-download

        Returns:
            DataFrame with OHLCV data
        """
        if not YFINANCE_AVAILABLE:
            logger.warning(f"yfinance not available, skipping {symbol}")
            return None

        # Check cache
        cache_file = self.data_dir / f"{name.replace('.', '_').replace('^', '')}.parquet"

        if not force_refresh and cache_file.exists():
            try:
                data = pd.read_parquet(cache_file)
                logger.debug(f"Loaded cached data for {symbol}")
                self.historical_data[symbol] = data
                return data
            except Exception as e:
                logger.warning(f"Cache read failed for {symbol}: {e}")

        # Fetch from Yahoo Finance
        try:
            logger.info(f"Fetching {symbol} ({self.config.years_of_history} years)...")
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=self.start_date,
                end=self.end_date,
                auto_adjust=self.config.auto_adjust,
                actions=self.config.include_dividends
            )

            if data.empty:
                logger.warning(f"No data returned for {symbol}")
                return None

            # Save to cache
            data.to_parquet(cache_file)
            logger.info(f"  {symbol}: {len(data)} records saved")

            self.historical_data[symbol] = data
            return data

        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return None

    def get_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get historical data for a symbol (from cache or memory)"""
        if symbol in self.historical_data:
            return self.historical_data[symbol]

        # Try to load from disk
        for name, sym in {**self.INDICES, **{s: s for sector in self.TSX_UNIVERSE.values() for s in sector}}.items():
            if sym == symbol:
                cache_file = self.data_dir / f"{name.replace('.', '_').replace('^', '')}.parquet"
                if cache_file.exists():
                    data = pd.read_parquet(cache_file)
                    self.historical_data[symbol] = data
                    return data

        return None

    def get_sector_data(self, sector: str) -> Dict[str, pd.DataFrame]:
        """Get all stock data for a sector"""
        if sector not in self.TSX_UNIVERSE:
            logger.warning(f"Unknown sector: {sector}")
            return {}

        result = {}
        for symbol in self.TSX_UNIVERSE[sector]:
            data = self.get_data(symbol)
            if data is not None:
                result[symbol] = data

        return result

    def calculate_sector_performance(
        self,
        sector: str,
        period_days: int = 252  # 1 year
    ) -> Dict[str, float]:
        """Calculate sector performance metrics"""
        sector_data = self.get_sector_data(sector)

        if not sector_data:
            return {}

        performance = {}
        for symbol, data in sector_data.items():
            if len(data) >= period_days:
                recent_close = data['Close'].iloc[-1]
                past_close = data['Close'].iloc[-period_days]
                pct_return = ((recent_close - past_close) / past_close) * 100
                performance[symbol] = round(pct_return, 2)

        return performance

    def get_tsx_composite_data(self) -> Optional[pd.DataFrame]:
        """Get TSX Composite Index data"""
        return self.get_data('^GSPTSE')

    def export_to_csv(self, output_dir: str = None) -> List[str]:
        """Export all data to CSV files"""
        output_dir = Path(output_dir or self.data_dir / "csv")
        output_dir.mkdir(parents=True, exist_ok=True)

        exported = []
        for symbol, data in self.historical_data.items():
            filename = output_dir / f"{symbol.replace('.', '_').replace('^', '')}.csv"
            data.to_csv(filename)
            exported.append(str(filename))

        return exported


def collect_canadian_historical_data(years: int = 20, force_refresh: bool = False) -> Dict[str, Any]:
    """
    Convenience function to collect all Canadian historical data

    Args:
        years: Number of years of history
        force_refresh: Force re-download

    Returns:
        Collection summary
    """
    config = HistoricalDataConfig(years_of_history=years)
    collector = CanadianMarketHistoricalCollector(config)
    return collector.collect_all(force_refresh=force_refresh)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("="*60)
    print("CANADIAN MARKET HISTORICAL DATA COLLECTOR")
    print("="*60)
    print(f"Collecting {20} years of data...")
    print()

    summary = collect_canadian_historical_data(years=20)

    print(f"\nIndices collected: {len(summary['indices'])}")
    print(f"Stocks collected: {len(summary['stocks'])}")
    print(f"Errors: {len(summary['errors'])}")

    if summary['errors']:
        print("\nErrors:")
        for error in summary['errors'][:5]:
            print(f"  - {error}")
