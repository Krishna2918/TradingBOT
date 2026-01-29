"""
Yahoo Finance Data Collection Plan

Efficient data collection strategy that respects API rate limits:
- Yahoo Finance: ~2000 requests/hour (unofficial), recommend 1 req/2 sec
- Batch requests where possible
- Cache aggressively to avoid re-fetching
- Priority-based collection (critical data first)

Collection Phases:
1. Phase 1 (Critical): TSX Index + Major Canadian stocks (30 min)
2. Phase 2 (Important): Global indices + US data (20 min)
3. Phase 3 (Supporting): Commodities + Currencies (15 min)
4. Phase 4 (Background): Remaining stocks + Yields (35 min)

Total: ~100 minutes for full collection, ~250 symbols
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import json
import random

logger = logging.getLogger(__name__)

try:
    import pandas as pd
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("yfinance not available")


class CollectionPriority(Enum):
    """Priority levels for data collection"""
    CRITICAL = 1    # Must have - TSX index, major banks, energy
    HIGH = 2        # Important - US indices, oil, gold
    MEDIUM = 3      # Supporting - other global indices, currencies
    LOW = 4         # Background - remaining stocks, yields


@dataclass
class CollectionTask:
    """A data collection task"""
    symbol: str
    name: str
    category: str
    priority: CollectionPriority
    years: int = 20
    interval: str = '1d'  # Daily data
    status: str = 'pending'
    error: str = None
    records_collected: int = 0


@dataclass
class RateLimiter:
    """Rate limiter for API calls"""
    requests_per_minute: int = 30  # Conservative: 30/min = 1 every 2 sec
    min_delay_seconds: float = 2.0
    max_delay_seconds: float = 3.0
    burst_limit: int = 5  # Max requests before mandatory pause
    burst_pause_seconds: float = 10.0

    request_count: int = 0
    last_request_time: float = 0
    burst_count: int = 0

    def wait(self):
        """Wait appropriate time before next request"""
        now = time.time()

        # Check burst limit
        self.burst_count += 1
        if self.burst_count >= self.burst_limit:
            logger.debug(f"Burst limit reached, pausing {self.burst_pause_seconds}s")
            time.sleep(self.burst_pause_seconds)
            self.burst_count = 0
            return

        # Calculate time since last request
        time_since_last = now - self.last_request_time

        # Add random jitter to avoid patterns
        delay = random.uniform(self.min_delay_seconds, self.max_delay_seconds)

        if time_since_last < delay:
            sleep_time = delay - time_since_last
            time.sleep(sleep_time)

        self.last_request_time = time.time()
        self.request_count += 1


class YahooDataCollectionPlan:
    """
    Orchestrates efficient data collection from Yahoo Finance

    Rate Limit Strategy:
    - 30 requests/minute (conservative)
    - 2-3 second delay between requests
    - Burst pauses every 5 requests
    - Total ~250 symbols in ~100 minutes
    """

    # ==================== SYMBOL DEFINITIONS ====================

    # Phase 1: Critical Canadian Data (highest priority)
    PHASE1_CRITICAL = {
        'indices': {
            '^GSPTSE': 'TSX Composite',
            '^TX60': 'TSX 60',
        },
        'banks': {
            'RY.TO': 'Royal Bank',
            'TD.TO': 'TD Bank',
            'BNS.TO': 'Bank of Nova Scotia',
            'BMO.TO': 'Bank of Montreal',
            'CM.TO': 'CIBC',
        },
        'energy_major': {
            'CNQ.TO': 'Canadian Natural Resources',
            'SU.TO': 'Suncor',
            'ENB.TO': 'Enbridge',
        },
        'mining_major': {
            'ABX.TO': 'Barrick Gold',
            'WPM.TO': 'Wheaton Precious Metals',
        },
        'tech': {
            'SHOP.TO': 'Shopify',
        },
    }  # ~15 symbols

    # Phase 2: US & Global Critical
    PHASE2_GLOBAL = {
        'us_indices': {
            '^GSPC': 'S&P 500',
            '^IXIC': 'NASDAQ',
            '^DJI': 'Dow Jones',
            '^VIX': 'VIX',
        },
        'commodities_critical': {
            'CL=F': 'WTI Crude Oil',
            'GC=F': 'Gold',
            'NG=F': 'Natural Gas',
        },
        'currencies_critical': {
            'USDCAD=X': 'USD/CAD',
        },
    }  # ~8 symbols

    # Phase 3: Supporting Global Data
    PHASE3_SUPPORTING = {
        'global_indices': {
            '^FTSE': 'FTSE 100',
            '^GDAXI': 'DAX',
            '^FCHI': 'CAC 40',
            '^N225': 'Nikkei 225',
            '^HSI': 'Hang Seng',
            '^AXJO': 'ASX 200',
            '^NSEI': 'Nifty 50',
            '^STOXX50E': 'Euro Stoxx 50',
        },
        'commodities_other': {
            'SI=F': 'Silver',
            'HG=F': 'Copper',
            'BZ=F': 'Brent Crude',
        },
        'currencies_other': {
            'EURCAD=X': 'EUR/CAD',
            'GBPCAD=X': 'GBP/CAD',
            'JPYCAD=X': 'JPY/CAD',
            'AUDCAD=X': 'AUD/CAD',
            'CNYCAD=X': 'CNY/CAD',
        },
        'us_sectors': {
            'XLF': 'Financials ETF',
            'XLE': 'Energy ETF',
            'XLK': 'Technology ETF',
        },
    }  # ~19 symbols

    # Phase 4: Remaining Canadian Stocks
    PHASE4_CANADIAN = {
        'energy_other': {
            'TRP.TO': 'TC Energy',
            'CVE.TO': 'Cenovus',
            'IMO.TO': 'Imperial Oil',
            'TOU.TO': 'Tourmaline',
            'PPL.TO': 'Pembina',
            'ARX.TO': 'ARC Resources',
            'KEY.TO': 'Keyera',
        },
        'mining_other': {
            'FNV.TO': 'Franco-Nevada',
            'AEM.TO': 'Agnico Eagle',
            'K.TO': 'Kinross',
            'NTR.TO': 'Nutrien',
            'TECK-B.TO': 'Teck Resources',
            'FM.TO': 'First Quantum',
            'LUN.TO': 'Lundin Mining',
        },
        'banks_other': {
            'NA.TO': 'National Bank',
        },
        'industrials': {
            'CNR.TO': 'CN Railway',
            'CP.TO': 'CP Railway',
            'WCN.TO': 'Waste Connections',
            'TIH.TO': 'Toromont',
            'WSP.TO': 'WSP Global',
            'TFII.TO': 'TFI International',
        },
        'telecom': {
            'BCE.TO': 'BCE',
            'T.TO': 'TELUS',
            'RCI-B.TO': 'Rogers',
        },
        'utilities': {
            'FTS.TO': 'Fortis',
            'EMA.TO': 'Emera',
            'H.TO': 'Hydro One',
        },
        'consumer': {
            'ATD.TO': 'Couche-Tard',
            'L.TO': 'Loblaw',
            'DOL.TO': 'Dollarama',
            'MRU.TO': 'Metro',
        },
        'tech_other': {
            'CSU.TO': 'Constellation Software',
            'OTEX.TO': 'Open Text',
        },
        'etfs': {
            'XIU.TO': 'iShares TSX 60',
            'XIC.TO': 'iShares Core TSX',
            'XEG.TO': 'iShares Energy',
            'XFN.TO': 'iShares Financials',
            'XGD.TO': 'iShares Gold',
            'ZAG.TO': 'BMO Bonds',
            'VFV.TO': 'Vanguard S&P 500',
        },
    }  # ~42 symbols

    # Phase 5: Yields and Additional
    PHASE5_YIELDS = {
        'treasury': {
            '^TNX': 'US 10Y Yield',
            '^TYX': 'US 30Y Yield',
            '^IRX': 'US 13W T-Bill',
        },
        'bond_etfs': {
            'TLT': 'iShares 20Y+ Treasury',
            'IEF': 'iShares 7-10Y Treasury',
        },
    }  # ~5 symbols

    def __init__(
        self,
        data_dir: str = "data/historical",
        years: int = 20,
        demo_mode: bool = False
    ):
        """
        Initialize collection plan

        Args:
            data_dir: Directory to store collected data
            years: Years of history to collect
            demo_mode: Skip actual API calls if True
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.years = years
        self.demo_mode = demo_mode

        self.rate_limiter = RateLimiter()
        self.tasks: List[CollectionTask] = []
        self.results: Dict[str, Any] = {}

        # Progress tracking
        self.total_symbols = 0
        self.completed_symbols = 0
        self.failed_symbols = 0

        # Build task list
        self._build_task_list()

        logger.info(f"Collection plan initialized:")
        logger.info(f"  Total symbols: {self.total_symbols}")
        logger.info(f"  Years of history: {years}")
        logger.info(f"  Estimated time: {self._estimate_time()} minutes")

    def _build_task_list(self):
        """Build prioritized task list"""

        def add_phase(phase_data: Dict, priority: CollectionPriority):
            for category, symbols in phase_data.items():
                for symbol, name in symbols.items():
                    self.tasks.append(CollectionTask(
                        symbol=symbol,
                        name=name,
                        category=category,
                        priority=priority,
                        years=self.years
                    ))

        # Add phases in priority order
        add_phase(self.PHASE1_CRITICAL, CollectionPriority.CRITICAL)
        add_phase(self.PHASE2_GLOBAL, CollectionPriority.HIGH)
        add_phase(self.PHASE3_SUPPORTING, CollectionPriority.MEDIUM)
        add_phase(self.PHASE4_CANADIAN, CollectionPriority.LOW)
        add_phase(self.PHASE5_YIELDS, CollectionPriority.LOW)

        # Sort by priority
        self.tasks.sort(key=lambda t: t.priority.value)
        self.total_symbols = len(self.tasks)

    def _estimate_time(self) -> int:
        """Estimate collection time in minutes"""
        # ~2.5 seconds per request average
        return int((self.total_symbols * 2.5) / 60) + 5  # +5 for overhead

    def collect_all(
        self,
        max_symbols: int = None,
        priority_filter: CollectionPriority = None,
        resume: bool = True
    ) -> Dict[str, Any]:
        """
        Collect all data according to plan

        Args:
            max_symbols: Limit number of symbols (for testing)
            priority_filter: Only collect symbols of this priority or higher
            resume: Skip symbols that already have cached data

        Returns:
            Collection summary
        """
        if not YFINANCE_AVAILABLE:
            return {'error': 'yfinance not available'}

        start_time = datetime.now()
        logger.info("="*60)
        logger.info("STARTING DATA COLLECTION")
        logger.info(f"Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*60)

        tasks_to_run = self.tasks.copy()

        # Apply filters
        if priority_filter:
            tasks_to_run = [t for t in tasks_to_run
                          if t.priority.value <= priority_filter.value]

        if max_symbols:
            tasks_to_run = tasks_to_run[:max_symbols]

        # Check cache if resume enabled
        if resume:
            tasks_to_run = [t for t in tasks_to_run
                          if not self._is_cached(t.symbol)]
            logger.info(f"Skipping {len(self.tasks) - len(tasks_to_run)} cached symbols")

        logger.info(f"Collecting {len(tasks_to_run)} symbols...")

        # Collection loop
        for i, task in enumerate(tasks_to_run, 1):
            try:
                # Rate limiting
                self.rate_limiter.wait()

                # Progress log every 10 symbols
                if i % 10 == 0:
                    elapsed = (datetime.now() - start_time).seconds / 60
                    remaining = ((len(tasks_to_run) - i) * 2.5) / 60
                    logger.info(f"Progress: {i}/{len(tasks_to_run)} "
                              f"({elapsed:.1f}min elapsed, ~{remaining:.1f}min remaining)")

                # Collect data
                if self.demo_mode:
                    self._collect_demo(task)
                else:
                    self._collect_real(task)

                task.status = 'completed'
                self.completed_symbols += 1

            except Exception as e:
                logger.error(f"Error collecting {task.symbol}: {e}")
                task.status = 'failed'
                task.error = str(e)
                self.failed_symbols += 1

                # Extra delay on error
                time.sleep(5)

        # Generate summary
        end_time = datetime.now()
        duration = (end_time - start_time).seconds / 60

        summary = {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_minutes': round(duration, 1),
            'total_symbols': len(tasks_to_run),
            'completed': self.completed_symbols,
            'failed': self.failed_symbols,
            'success_rate': round(self.completed_symbols / max(len(tasks_to_run), 1) * 100, 1),
            'tasks': [
                {
                    'symbol': t.symbol,
                    'name': t.name,
                    'category': t.category,
                    'priority': t.priority.name,
                    'status': t.status,
                    'records': t.records_collected,
                    'error': t.error
                }
                for t in tasks_to_run
            ]
        }

        # Save summary
        summary_file = self.data_dir / f"collection_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info("="*60)
        logger.info("COLLECTION COMPLETE")
        logger.info(f"Duration: {duration:.1f} minutes")
        logger.info(f"Success rate: {summary['success_rate']}%")
        logger.info(f"Summary saved: {summary_file}")
        logger.info("="*60)

        return summary

    def _collect_real(self, task: CollectionTask):
        """Collect real data from Yahoo Finance"""
        logger.debug(f"Collecting {task.symbol} ({task.name})...")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * task.years)

        ticker = yf.Ticker(task.symbol)
        data = ticker.history(
            start=start_date,
            end=end_date,
            interval=task.interval,
            auto_adjust=True
        )

        if data.empty:
            raise ValueError(f"No data returned for {task.symbol}")

        # Save to parquet
        filename = self._get_cache_filename(task.symbol)
        data.to_parquet(filename)

        task.records_collected = len(data)
        self.results[task.symbol] = {
            'records': len(data),
            'start': data.index[0].isoformat() if hasattr(data.index[0], 'isoformat') else str(data.index[0]),
            'end': data.index[-1].isoformat() if hasattr(data.index[-1], 'isoformat') else str(data.index[-1]),
            'file': str(filename)
        }

        logger.debug(f"  {task.symbol}: {len(data)} records saved")

    def _collect_demo(self, task: CollectionTask):
        """Generate demo data (for testing)"""
        import numpy as np

        n_days = task.years * 252
        dates = pd.date_range(
            end=datetime.now(),
            periods=n_days,
            freq='B'  # Business days
        )

        # Generate random price data
        np.random.seed(hash(task.symbol) % 2**32)
        returns = np.random.normal(0.0003, 0.015, n_days)
        prices = 100 * np.exp(np.cumsum(returns))

        data = pd.DataFrame({
            'Open': prices * (1 + np.random.uniform(-0.01, 0.01, n_days)),
            'High': prices * (1 + np.random.uniform(0, 0.02, n_days)),
            'Low': prices * (1 - np.random.uniform(0, 0.02, n_days)),
            'Close': prices,
            'Volume': np.random.randint(100000, 10000000, n_days)
        }, index=dates)

        filename = self._get_cache_filename(task.symbol)
        data.to_parquet(filename)

        task.records_collected = len(data)

    def _get_cache_filename(self, symbol: str) -> Path:
        """Get cache filename for a symbol"""
        safe_symbol = symbol.replace('^', 'IDX_').replace('=', '_').replace('.', '_')
        return self.data_dir / f"{safe_symbol}.parquet"

    def _is_cached(self, symbol: str) -> bool:
        """Check if symbol data is already cached"""
        return self._get_cache_filename(symbol).exists()

    def collect_critical_only(self) -> Dict[str, Any]:
        """Quick collection of only critical data (~15 symbols, ~1 min)"""
        return self.collect_all(priority_filter=CollectionPriority.CRITICAL)

    def collect_high_priority(self) -> Dict[str, Any]:
        """Collect critical + high priority (~23 symbols, ~2 min)"""
        return self.collect_all(priority_filter=CollectionPriority.HIGH)

    def get_collection_status(self) -> Dict[str, Any]:
        """Get current collection status"""
        cached = sum(1 for t in self.tasks if self._is_cached(t.symbol))

        return {
            'total_symbols': self.total_symbols,
            'cached': cached,
            'pending': self.total_symbols - cached,
            'completion_pct': round(cached / self.total_symbols * 100, 1),
            'by_priority': {
                p.name: sum(1 for t in self.tasks if t.priority == p and self._is_cached(t.symbol))
                for p in CollectionPriority
            }
        }


# ==================== CONVENIENCE FUNCTIONS ====================

def collect_critical_data(years: int = 20) -> Dict[str, Any]:
    """Quick collection of critical Canadian + US data"""
    plan = YahooDataCollectionPlan(years=years)
    return plan.collect_critical_only()


def collect_all_data(years: int = 20, resume: bool = True) -> Dict[str, Any]:
    """Full collection with resume support"""
    plan = YahooDataCollectionPlan(years=years)
    return plan.collect_all(resume=resume)


def get_collection_status() -> Dict[str, Any]:
    """Check what data is already cached"""
    plan = YahooDataCollectionPlan()
    return plan.get_collection_status()


# ==================== SCHEDULED COLLECTION ====================

class ScheduledDataRefresh:
    """
    Scheduled data refresh strategy

    Daily refresh schedule (minimal API usage):
    - 6:00 AM: Critical data only (15 symbols)
    - Weekend: Full refresh (all symbols)
    """

    @staticmethod
    def get_daily_refresh_symbols() -> List[str]:
        """Symbols to refresh daily (before market open)"""
        return [
            # TSX Index
            '^GSPTSE',
            # Top 5 stocks by volume
            'RY.TO', 'TD.TO', 'ENB.TO', 'CNQ.TO', 'SHOP.TO',
            # US indices
            '^GSPC', '^VIX',
            # Critical commodities
            'CL=F', 'GC=F',
            # Currency
            'USDCAD=X',
        ]  # 11 symbols, ~30 seconds

    @staticmethod
    def run_daily_refresh():
        """Run minimal daily refresh"""
        symbols = ScheduledDataRefresh.get_daily_refresh_symbols()

        plan = YahooDataCollectionPlan(years=1)  # Only need recent data

        # Filter tasks to only daily symbols
        plan.tasks = [t for t in plan.tasks if t.symbol in symbols]

        return plan.collect_all(resume=False)  # Always refresh


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("="*60)
    print("YAHOO DATA COLLECTION PLAN")
    print("="*60)

    plan = YahooDataCollectionPlan(years=20, demo_mode=True)

    print(f"\nTotal symbols to collect: {plan.total_symbols}")
    print(f"Estimated time: {plan._estimate_time()} minutes")

    print("\nSymbols by priority:")
    for priority in CollectionPriority:
        count = sum(1 for t in plan.tasks if t.priority == priority)
        print(f"  {priority.name}: {count} symbols")

    print("\n" + "="*60)
    print("Running demo collection (first 10 symbols)...")
    print("="*60)

    summary = plan.collect_all(max_symbols=10)

    print(f"\nCollection complete!")
    print(f"Duration: {summary['duration_minutes']} minutes")
    print(f"Success rate: {summary['success_rate']}%")
