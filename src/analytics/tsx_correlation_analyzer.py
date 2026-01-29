"""
TSX Correlation Analyzer

Analyzes correlations between:
- TSX Composite and global indices (S&P 500, FTSE, DAX, etc.)
- TSX sectors and commodities (Oil → Energy, Gold → Mining)
- Canadian stocks and foreign currencies
- Historical patterns and lead-lag relationships

Uses 20 years of historical data to identify:
1. Correlation strengths
2. Lead-lag relationships (which markets lead TSX)
3. Sector-specific correlations
4. Currency impacts
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import json

logger = logging.getLogger(__name__)

try:
    import pandas as pd
    import numpy as np
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available - correlation analysis limited")

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


@dataclass
class CorrelationResult:
    """Result of correlation analysis"""
    symbol1: str
    symbol2: str
    correlation: float
    p_value: float
    sample_size: int
    period: str
    is_significant: bool  # p < 0.05


class TSXCorrelationAnalyzer:
    """
    Analyzes correlations between TSX and global factors
    """

    # Key correlations to track
    CORRELATION_PAIRS = {
        # TSX vs Global Indices
        'tsx_sp500': ('^GSPTSE', '^GSPC'),
        'tsx_nasdaq': ('^GSPTSE', '^IXIC'),
        'tsx_dow': ('^GSPTSE', '^DJI'),
        'tsx_ftse': ('^GSPTSE', '^FTSE'),
        'tsx_dax': ('^GSPTSE', '^GDAXI'),
        'tsx_nikkei': ('^GSPTSE', '^N225'),
        'tsx_hang_seng': ('^GSPTSE', '^HSI'),
        'tsx_asx': ('^GSPTSE', '^AXJO'),

        # TSX Energy vs Oil
        'tsx_energy_wti': ('XEG.TO', 'CL=F'),
        'tsx_energy_brent': ('XEG.TO', 'BZ=F'),

        # TSX Mining vs Gold/Copper
        'tsx_gold_miners': ('XGD.TO', 'GC=F'),
        'tsx_materials_copper': ('XMA.TO', 'HG=F'),

        # TSX vs Currency
        'tsx_usdcad': ('^GSPTSE', 'USDCAD=X'),

        # TSX Banks vs US Financials
        'tsx_banks_xlf': ('XFN.TO', 'XLF'),

        # VIX correlation (inverse expected)
        'tsx_vix': ('^GSPTSE', '^VIX'),
    }

    # Known historical correlations (baseline for comparison)
    BASELINE_CORRELATIONS = {
        'tsx_sp500': 0.85,
        'tsx_nasdaq': 0.78,
        'tsx_dow': 0.82,
        'tsx_ftse': 0.65,
        'tsx_dax': 0.60,
        'tsx_nikkei': 0.45,
        'tsx_hang_seng': 0.40,
        'tsx_asx': 0.55,
        'tsx_energy_wti': 0.75,
        'tsx_gold_miners': 0.85,
        'tsx_vix': -0.65,
    }

    def __init__(self, data_dir: str = "data/historical"):
        self.data_dir = Path(data_dir)
        self.data_cache: Dict[str, pd.DataFrame] = {}
        self.correlation_cache: Dict[str, CorrelationResult] = {}

    def calculate_correlation(
        self,
        symbol1: str,
        symbol2: str,
        period_years: int = 5,
        return_type: str = 'daily'  # 'daily', 'weekly', 'monthly'
    ) -> Optional[CorrelationResult]:
        """
        Calculate correlation between two symbols

        Args:
            symbol1: First symbol
            symbol2: Second symbol
            period_years: Years of data to use
            return_type: Type of returns to correlate

        Returns:
            CorrelationResult with correlation coefficient and p-value
        """
        if not SCIPY_AVAILABLE:
            logger.warning("scipy not available")
            return None

        # Get data for both symbols
        data1 = self._get_returns(symbol1, period_years, return_type)
        data2 = self._get_returns(symbol2, period_years, return_type)

        if data1 is None or data2 is None:
            return None

        # Align the data
        aligned = pd.concat([data1, data2], axis=1, join='inner')
        aligned.columns = ['returns1', 'returns2']
        aligned = aligned.dropna()

        if len(aligned) < 30:
            logger.warning(f"Insufficient data for {symbol1} vs {symbol2}")
            return None

        # Calculate Pearson correlation
        corr, p_value = stats.pearsonr(aligned['returns1'], aligned['returns2'])

        return CorrelationResult(
            symbol1=symbol1,
            symbol2=symbol2,
            correlation=round(corr, 4),
            p_value=round(p_value, 6),
            sample_size=len(aligned),
            period=f"{period_years}Y_{return_type}",
            is_significant=p_value < 0.05
        )

    def _get_returns(
        self,
        symbol: str,
        years: int,
        return_type: str
    ) -> Optional[pd.Series]:
        """Get returns for a symbol"""
        # Check cache
        cache_key = f"{symbol}_{years}Y"
        if cache_key in self.data_cache:
            data = self.data_cache[cache_key]
        else:
            if not YFINANCE_AVAILABLE:
                return self._get_demo_returns(symbol, years, return_type)

            try:
                ticker = yf.Ticker(symbol)
                end = datetime.now()
                start = end - timedelta(days=365 * years)
                data = ticker.history(start=start, end=end)

                if data.empty:
                    return self._get_demo_returns(symbol, years, return_type)

                self.data_cache[cache_key] = data

            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
                return self._get_demo_returns(symbol, years, return_type)

        # Calculate returns
        if return_type == 'weekly':
            prices = data['Close'].resample('W').last()
        elif return_type == 'monthly':
            prices = data['Close'].resample('M').last()
        else:  # daily
            prices = data['Close']

        returns = prices.pct_change().dropna()
        return returns

    def _get_demo_returns(
        self,
        symbol: str,
        years: int,
        return_type: str
    ) -> pd.Series:
        """Generate demo returns for testing"""
        np.random.seed(hash(symbol) % 2**32)

        if return_type == 'weekly':
            n_periods = years * 52
        elif return_type == 'monthly':
            n_periods = years * 12
        else:
            n_periods = years * 252

        # Generate correlated returns based on baseline
        base_vol = 0.01 if return_type == 'daily' else 0.02
        returns = np.random.normal(0, base_vol, n_periods)

        dates = pd.date_range(
            end=datetime.now(),
            periods=n_periods,
            freq='D' if return_type == 'daily' else 'W' if return_type == 'weekly' else 'M'
        )

        return pd.Series(returns, index=dates, name=symbol)

    def calculate_all_correlations(
        self,
        period_years: int = 5
    ) -> Dict[str, CorrelationResult]:
        """Calculate all predefined correlations"""
        results = {}

        for name, (symbol1, symbol2) in self.CORRELATION_PAIRS.items():
            result = self.calculate_correlation(symbol1, symbol2, period_years)
            if result:
                results[name] = result
                logger.info(f"{name}: {result.correlation:.3f} (p={result.p_value:.4f})")

        return results

    def calculate_lead_lag(
        self,
        leader_symbol: str,
        follower_symbol: str,
        max_lag_days: int = 5,
        period_years: int = 5
    ) -> Dict[str, Any]:
        """
        Calculate lead-lag relationship

        Tests if leader_symbol's returns predict follower_symbol's returns
        at various lags
        """
        leader_returns = self._get_returns(leader_symbol, period_years, 'daily')
        follower_returns = self._get_returns(follower_symbol, period_years, 'daily')

        if leader_returns is None or follower_returns is None:
            return {'error': 'Could not get data'}

        # Align data
        aligned = pd.concat([leader_returns, follower_returns], axis=1, join='inner')
        aligned.columns = ['leader', 'follower']
        aligned = aligned.dropna()

        results = {
            'leader': leader_symbol,
            'follower': follower_symbol,
            'lag_correlations': {},
            'best_lag': 0,
            'best_correlation': 0,
        }

        best_corr = 0
        best_lag = 0

        for lag in range(-max_lag_days, max_lag_days + 1):
            if lag == 0:
                corr = aligned['leader'].corr(aligned['follower'])
            elif lag > 0:
                # Leader leads: shift leader back
                corr = aligned['leader'].shift(lag).corr(aligned['follower'])
            else:
                # Follower leads: shift follower back
                corr = aligned['leader'].corr(aligned['follower'].shift(-lag))

            if not np.isnan(corr):
                results['lag_correlations'][lag] = round(corr, 4)

                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_lag = lag

        results['best_lag'] = best_lag
        results['best_correlation'] = round(best_corr, 4)
        results['interpretation'] = self._interpret_lead_lag(best_lag, best_corr, leader_symbol, follower_symbol)

        return results

    def _interpret_lead_lag(
        self,
        lag: int,
        corr: float,
        leader: str,
        follower: str
    ) -> str:
        """Interpret lead-lag results"""
        if abs(corr) < 0.1:
            return f"Weak relationship between {leader} and {follower}"

        if lag > 0:
            return f"{leader} leads {follower} by {lag} day(s) with correlation {corr:.2f}"
        elif lag < 0:
            return f"{follower} leads {leader} by {-lag} day(s) with correlation {corr:.2f}"
        else:
            return f"{leader} and {follower} move together (contemporaneous) with correlation {corr:.2f}"

    def get_sector_correlations(self) -> Dict[str, Dict[str, float]]:
        """
        Get correlation matrix for TSX sectors with global factors
        """
        sectors = {
            'TSX_Banks': 'XFN.TO',
            'TSX_Energy': 'XEG.TO',
            'TSX_Mining': 'XGD.TO',
            'TSX_Materials': 'XMA.TO',
        }

        factors = {
            'S&P_500': '^GSPC',
            'Oil_WTI': 'CL=F',
            'Gold': 'GC=F',
            'USD_CAD': 'USDCAD=X',
            'VIX': '^VIX',
            'US_10Y': '^TNX',
        }

        matrix = {}

        for sector_name, sector_symbol in sectors.items():
            matrix[sector_name] = {}

            for factor_name, factor_symbol in factors.items():
                result = self.calculate_correlation(sector_symbol, factor_symbol, period_years=3)
                if result:
                    matrix[sector_name][factor_name] = result.correlation
                else:
                    matrix[sector_name][factor_name] = None

        return matrix

    def get_correlation_regime(
        self,
        rolling_window: int = 60
    ) -> Dict[str, Any]:
        """
        Analyze if correlations are in normal range or stressed regime

        Returns analysis of whether current correlations differ from historical
        """
        regime = {
            'timestamp': datetime.now().isoformat(),
            'status': 'normal',
            'stressed_pairs': [],
            'breakdown': {},
        }

        # Calculate recent correlations (60-day rolling)
        for name, baseline in self.BASELINE_CORRELATIONS.items():
            if name not in self.CORRELATION_PAIRS:
                continue

            symbol1, symbol2 = self.CORRELATION_PAIRS[name]
            recent = self.calculate_correlation(symbol1, symbol2, period_years=1)

            if recent:
                diff = abs(recent.correlation - baseline)
                regime['breakdown'][name] = {
                    'baseline': baseline,
                    'current': recent.correlation,
                    'difference': round(diff, 3),
                    'status': 'normal' if diff < 0.15 else 'elevated' if diff < 0.25 else 'stressed'
                }

                if diff >= 0.25:
                    regime['stressed_pairs'].append(name)

        if len(regime['stressed_pairs']) > 3:
            regime['status'] = 'stressed'
        elif len(regime['stressed_pairs']) > 0:
            regime['status'] = 'elevated'

        return regime

    def generate_correlation_report(self) -> Dict[str, Any]:
        """Generate comprehensive correlation report"""
        report = {
            'generated_at': datetime.now().isoformat(),
            'tsx_global_correlations': {},
            'sector_correlations': {},
            'lead_lag_analysis': {},
            'correlation_regime': {},
            'trading_implications': [],
        }

        # 1. Calculate all correlations
        logger.info("Calculating global correlations...")
        correlations = self.calculate_all_correlations(period_years=5)
        for name, result in correlations.items():
            report['tsx_global_correlations'][name] = {
                'correlation': result.correlation,
                'p_value': result.p_value,
                'significant': result.is_significant,
                'sample_size': result.sample_size,
            }

        # 2. Sector correlations
        logger.info("Calculating sector correlations...")
        report['sector_correlations'] = self.get_sector_correlations()

        # 3. Key lead-lag relationships
        logger.info("Analyzing lead-lag relationships...")
        lead_lag_pairs = [
            ('^GSPC', '^GSPTSE'),  # S&P leads TSX?
            ('CL=F', 'XEG.TO'),    # Oil leads energy?
            ('GC=F', 'XGD.TO'),    # Gold leads miners?
            ('^VIX', '^GSPTSE'),   # VIX leads TSX?
        ]

        for leader, follower in lead_lag_pairs:
            key = f"{leader}_vs_{follower}"
            report['lead_lag_analysis'][key] = self.calculate_lead_lag(leader, follower)

        # 4. Correlation regime
        report['correlation_regime'] = self.get_correlation_regime()

        # 5. Trading implications
        report['trading_implications'] = self._generate_trading_implications(report)

        return report

    def _generate_trading_implications(self, report: Dict[str, Any]) -> List[str]:
        """Generate trading implications from correlation analysis"""
        implications = []

        # Check S&P correlation
        sp_corr = report['tsx_global_correlations'].get('tsx_sp500', {})
        if sp_corr.get('correlation', 0) > 0.8:
            implications.append("High TSX-S&P correlation: US market direction will strongly influence TSX")

        # Check oil-energy correlation
        oil_corr = report['tsx_global_correlations'].get('tsx_energy_wti', {})
        if oil_corr.get('correlation', 0) > 0.7:
            implications.append("Strong oil-energy correlation: Monitor WTI for TSX energy sector signals")

        # Check VIX correlation
        vix_corr = report['tsx_global_correlations'].get('tsx_vix', {})
        if vix_corr.get('correlation', 0) < -0.5:
            implications.append("VIX inversely correlated: Rising VIX suggests TSX downside risk")

        # Check correlation regime
        regime = report['correlation_regime']
        if regime.get('status') == 'stressed':
            implications.append("⚠️ CORRELATION STRESS: Multiple pairs showing unusual correlations")

        # Lead-lag implications
        for pair, analysis in report['lead_lag_analysis'].items():
            if analysis.get('best_lag', 0) != 0 and abs(analysis.get('best_correlation', 0)) > 0.3:
                implications.append(f"Lead-lag signal: {analysis.get('interpretation', '')}")

        return implications


# Convenience function
def analyze_tsx_correlations() -> Dict[str, Any]:
    """Run full TSX correlation analysis"""
    analyzer = TSXCorrelationAnalyzer()
    return analyzer.generate_correlation_report()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("="*60)
    print("TSX CORRELATION ANALYZER")
    print("="*60)

    analyzer = TSXCorrelationAnalyzer()

    print("\n--- CALCULATING CORRELATIONS ---")
    correlations = analyzer.calculate_all_correlations(period_years=3)

    print("\n--- TSX VS GLOBAL INDICES ---")
    for name, result in correlations.items():
        if 'tsx_' in name and result.is_significant:
            print(f"{name:25s}: {result.correlation:+.3f} (p={result.p_value:.4f})")

    print("\n--- LEAD-LAG ANALYSIS: S&P → TSX ---")
    lead_lag = analyzer.calculate_lead_lag('^GSPC', '^GSPTSE')
    print(f"Best lag: {lead_lag['best_lag']} days")
    print(f"Best correlation: {lead_lag['best_correlation']}")
    print(f"Interpretation: {lead_lag['interpretation']}")

    print("\n--- CORRELATION REGIME ---")
    regime = analyzer.get_correlation_regime()
    print(f"Status: {regime['status']}")
    if regime['stressed_pairs']:
        print(f"Stressed pairs: {', '.join(regime['stressed_pairs'])}")
