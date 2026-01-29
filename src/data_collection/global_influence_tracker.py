"""
Global Influence Tracker for Canadian Market (TSX)

Tracks factors from 8 key countries/regions that influence Canadian stocks:

Influence Rankings:
1. United States – dominant (trade, S&P, Fed, USD, tech)
2. China – commodities demand, global growth
3. United Kingdom – finance, mining, FX flows
4. European Union – macro sentiment, banks
5. Japan – global liquidity, risk cycles
6. India – energy & metals demand (growing)
7. OPEC members – oil prices → TSX energy
8. Australia – mining & commodity correlation

Each country has specific factors tracked:
- Market indices
- Economic indicators
- Central bank policy
- Currency movements
- Sector-specific impacts
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


class InfluenceLevel(Enum):
    """Level of influence on TSX"""
    DOMINANT = 5    # United States
    HIGH = 4        # China
    SIGNIFICANT = 3 # UK, EU
    MODERATE = 2    # Japan, India
    SECTOR = 1      # OPEC, Australia (sector-specific)


@dataclass
class CountryInfluence:
    """Country influence configuration"""
    name: str
    code: str
    influence_level: InfluenceLevel
    tsx_correlation: float  # Historical correlation with TSX
    primary_sectors: List[str]  # Canadian sectors most affected
    key_indices: Dict[str, str]  # Name -> Yahoo symbol
    currency_pair: str  # vs CAD
    factors: List[str]  # Key factors to track


class GlobalInfluenceTracker:
    """
    Tracks global factors that influence Canadian markets

    Monitors 8 key countries/regions and their impact on TSX
    """

    # Country configurations with factors
    COUNTRIES = {
        'US': CountryInfluence(
            name="United States",
            code="US",
            influence_level=InfluenceLevel.DOMINANT,
            tsx_correlation=0.85,  # Very high correlation
            primary_sectors=['all'],  # Affects all sectors
            key_indices={
                'SP500': '^GSPC',
                'NASDAQ': '^IXIC',
                'DOW': '^DJI',
                'VIX': '^VIX',
                'RUSSELL2000': '^RUT',
            },
            currency_pair='USDCAD=X',
            factors=[
                'fed_interest_rate',
                'usd_strength',
                'sp500_direction',
                'us_gdp_growth',
                'tech_earnings',
                'us_canada_trade_policy',
                'treasury_yields',
                'unemployment',
                'inflation_cpi',
            ]
        ),
        'CN': CountryInfluence(
            name="China",
            code="CN",
            influence_level=InfluenceLevel.HIGH,
            tsx_correlation=0.45,
            primary_sectors=['mining', 'energy', 'materials'],
            key_indices={
                'SHANGHAI': '000001.SS',
                'SHENZHEN': '399001.SZ',
                'HANG_SENG': '^HSI',
                'CSI300': '000300.SS',
            },
            currency_pair='CNYCAD=X',
            factors=[
                'commodity_demand',
                'manufacturing_pmi',
                'property_sector',
                'gdp_growth',
                'iron_ore_demand',
                'copper_demand',
                'oil_imports',
            ]
        ),
        'UK': CountryInfluence(
            name="United Kingdom",
            code="UK",
            influence_level=InfluenceLevel.SIGNIFICANT,
            tsx_correlation=0.55,
            primary_sectors=['banks', 'mining', 'energy'],
            key_indices={
                'FTSE100': '^FTSE',
                'FTSE250': '^FTMC',
            },
            currency_pair='GBPCAD=X',
            factors=[
                'mining_sector',
                'financial_sector',
                'gbp_movement',
                'risk_sentiment',
                'boe_policy',
            ]
        ),
        'EU': CountryInfluence(
            name="European Union",
            code="EU",
            influence_level=InfluenceLevel.SIGNIFICANT,
            tsx_correlation=0.50,
            primary_sectors=['banks', 'industrials'],
            key_indices={
                'EURO_STOXX50': '^STOXX50E',
                'DAX': '^GDAXI',
                'CAC40': '^FCHI',
            },
            currency_pair='EURCAD=X',
            factors=[
                'ecb_interest_rate',
                'eu_gdp_growth',
                'banking_stability',
                'euro_strength',
                'german_manufacturing',
            ]
        ),
        'JP': CountryInfluence(
            name="Japan",
            code="JP",
            influence_level=InfluenceLevel.MODERATE,
            tsx_correlation=0.35,
            primary_sectors=['banks', 'technology'],
            key_indices={
                'NIKKEI225': '^N225',
                'TOPIX': '^TOPX',
            },
            currency_pair='JPYCAD=X',
            factors=[
                'yen_carry_trade',
                'boj_policy',
                'global_liquidity',
                'risk_on_off',
                'jgb_yields',
            ]
        ),
        'IN': CountryInfluence(
            name="India",
            code="IN",
            influence_level=InfluenceLevel.MODERATE,
            tsx_correlation=0.30,
            primary_sectors=['energy', 'mining'],
            key_indices={
                'NIFTY50': '^NSEI',
                'SENSEX': '^BSESN',
            },
            currency_pair='INRCAD=X',
            factors=[
                'energy_demand',
                'metals_consumption',
                'em_growth_signal',
                'oil_imports',
                'gold_demand',
            ]
        ),
        'OPEC': CountryInfluence(
            name="OPEC",
            code="OPEC",
            influence_level=InfluenceLevel.SECTOR,
            tsx_correlation=0.60,  # High for energy sector
            primary_sectors=['energy', 'pipelines'],
            key_indices={
                'CRUDE_OIL_WTI': 'CL=F',
                'CRUDE_OIL_BRENT': 'BZ=F',
                'NATURAL_GAS': 'NG=F',
            },
            currency_pair='USDCAD=X',  # Oil priced in USD
            factors=[
                'oil_production_decisions',
                'crude_oil_price',
                'energy_profitability',
                'supply_cuts',
                'demand_forecasts',
            ]
        ),
        'AU': CountryInfluence(
            name="Australia",
            code="AU",
            influence_level=InfluenceLevel.SECTOR,
            tsx_correlation=0.55,  # High for mining
            primary_sectors=['mining', 'materials'],
            key_indices={
                'ASX200': '^AXJO',
                'ASX_MATERIALS': '^AXMJ',
            },
            currency_pair='AUDCAD=X',
            factors=[
                'mining_cycle',
                'china_commodity_demand',
                'aud_trend',
                'iron_ore_price',
                'coal_price',
            ]
        ),
    }

    # Commodity symbols for tracking
    COMMODITIES = {
        'GOLD': 'GC=F',
        'SILVER': 'SI=F',
        'COPPER': 'HG=F',
        'IRON_ORE': 'TIO=F',  # SGX Iron Ore
        'CRUDE_OIL_WTI': 'CL=F',
        'CRUDE_OIL_BRENT': 'BZ=F',
        'NATURAL_GAS': 'NG=F',
        'WHEAT': 'ZW=F',
        'LUMBER': 'LBS=F',
        'URANIUM': 'UX=F',
    }

    # Treasury/Bond yields
    YIELDS = {
        'US_10Y': '^TNX',
        'US_2Y': '^IRX',  # 13-week T-bill as proxy
        'US_30Y': '^TYX',
    }

    def __init__(self, demo_mode: bool = True, data_dir: str = "data/global_influence"):
        """
        Initialize global influence tracker

        Args:
            demo_mode: Use simulated data if True
            data_dir: Directory for storing data
        """
        self.demo_mode = demo_mode
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Cache
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_expiry = timedelta(minutes=15)

        # Historical data storage
        self.historical_data: Dict[str, pd.DataFrame] = {}

        logger.info(f"Global Influence Tracker initialized (demo_mode={demo_mode})")

    def get_global_snapshot(self) -> Dict[str, Any]:
        """
        Get current global market snapshot

        Returns:
            Comprehensive view of all global influences
        """
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'countries': {},
            'commodities': {},
            'yields': {},
            'tsx_impact_summary': {},
        }

        # Get data for each country
        for code, country in self.COUNTRIES.items():
            snapshot['countries'][code] = self._get_country_data(country)

        # Get commodity prices
        snapshot['commodities'] = self._get_commodity_prices()

        # Get yields
        snapshot['yields'] = self._get_yield_data()

        # Calculate TSX impact summary
        snapshot['tsx_impact_summary'] = self._calculate_tsx_impact(snapshot)

        return snapshot

    def _get_country_data(self, country: CountryInfluence) -> Dict[str, Any]:
        """Get data for a specific country"""
        data = {
            'name': country.name,
            'influence_level': country.influence_level.name,
            'tsx_correlation': country.tsx_correlation,
            'primary_sectors': country.primary_sectors,
            'indices': {},
            'currency': {},
            'factors_status': {},
            'overall_signal': 'neutral',
        }

        # Get index data
        for name, symbol in country.key_indices.items():
            index_data = self._fetch_market_data(symbol)
            if index_data:
                data['indices'][name] = index_data

        # Get currency data
        currency_data = self._fetch_market_data(country.currency_pair)
        if currency_data:
            data['currency'] = currency_data

        # Calculate overall signal based on indices
        signals = []
        for idx_name, idx_data in data['indices'].items():
            change = idx_data.get('change_pct', 0)
            if change > 0.5:
                signals.append(1)
            elif change < -0.5:
                signals.append(-1)
            else:
                signals.append(0)

        if signals:
            avg_signal = sum(signals) / len(signals)
            if avg_signal > 0.3:
                data['overall_signal'] = 'bullish'
            elif avg_signal < -0.3:
                data['overall_signal'] = 'bearish'

        return data

    def _fetch_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch current market data for a symbol"""
        # Check cache
        cached = self.cache.get(symbol)
        if cached and (datetime.now() - cached['timestamp']) < self.cache_expiry:
            return cached['data']

        if not YFINANCE_AVAILABLE or self.demo_mode:
            data = self._get_demo_market_data(symbol)
        else:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='5d')

                if hist.empty:
                    return self._get_demo_market_data(symbol)

                current = hist['Close'].iloc[-1]
                prev = hist['Close'].iloc[-2] if len(hist) > 1 else current
                change = current - prev
                change_pct = (change / prev) * 100 if prev != 0 else 0

                data = {
                    'price': round(current, 4),
                    'prev_close': round(prev, 4),
                    'change': round(change, 4),
                    'change_pct': round(change_pct, 2),
                    'high_5d': round(hist['High'].max(), 4),
                    'low_5d': round(hist['Low'].min(), 4),
                    'volume': int(hist['Volume'].iloc[-1]) if 'Volume' in hist else 0,
                }

            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
                data = self._get_demo_market_data(symbol)

        # Cache the data
        self.cache[symbol] = {
            'data': data,
            'timestamp': datetime.now()
        }

        return data

    def _get_demo_market_data(self, symbol: str) -> Dict[str, Any]:
        """Generate demo market data"""
        import random

        # Base prices for different asset types
        if 'CAD' in symbol or '=X' in symbol:
            base = 1.0 + random.uniform(-0.1, 0.5)
        elif symbol.startswith('^'):
            base = random.uniform(1000, 50000)
        elif '=F' in symbol:
            base = random.uniform(20, 2000)
        else:
            base = random.uniform(50, 500)

        change_pct = random.uniform(-2, 2)
        change = base * (change_pct / 100)

        return {
            'price': round(base, 4),
            'prev_close': round(base - change, 4),
            'change': round(change, 4),
            'change_pct': round(change_pct, 2),
            'high_5d': round(base * 1.02, 4),
            'low_5d': round(base * 0.98, 4),
            'volume': random.randint(1000000, 100000000),
            'demo': True,
        }

    def _get_commodity_prices(self) -> Dict[str, Dict[str, Any]]:
        """Get all commodity prices"""
        commodities = {}
        for name, symbol in self.COMMODITIES.items():
            data = self._fetch_market_data(symbol)
            if data:
                commodities[name] = data
        return commodities

    def _get_yield_data(self) -> Dict[str, Dict[str, Any]]:
        """Get treasury yield data"""
        yields = {}
        for name, symbol in self.YIELDS.items():
            data = self._fetch_market_data(symbol)
            if data:
                yields[name] = data
        return yields

    def _calculate_tsx_impact(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate the overall impact on TSX from global factors"""
        impact = {
            'overall_sentiment': 'neutral',
            'sentiment_score': 0.0,
            'sector_impacts': {},
            'key_drivers': [],
            'risk_factors': [],
        }

        weighted_score = 0.0
        total_weight = 0.0

        # Calculate weighted impact from each country
        for code, country_data in snapshot['countries'].items():
            country = self.COUNTRIES[code]
            weight = country.tsx_correlation

            # Get signal as numeric
            signal = country_data.get('overall_signal', 'neutral')
            if signal == 'bullish':
                signal_value = 1
            elif signal == 'bearish':
                signal_value = -1
            else:
                signal_value = 0

            weighted_score += signal_value * weight
            total_weight += weight

            # Track significant drivers
            if abs(signal_value) > 0:
                for sector in country.primary_sectors:
                    if sector not in impact['sector_impacts']:
                        impact['sector_impacts'][sector] = []
                    impact['sector_impacts'][sector].append({
                        'country': country.name,
                        'signal': signal,
                        'weight': weight
                    })

                if signal == 'bullish':
                    impact['key_drivers'].append(f"{country.name} markets positive")
                elif signal == 'bearish':
                    impact['risk_factors'].append(f"{country.name} markets negative")

        # Calculate overall sentiment
        if total_weight > 0:
            impact['sentiment_score'] = round(weighted_score / total_weight, 2)

        if impact['sentiment_score'] > 0.2:
            impact['overall_sentiment'] = 'bullish'
        elif impact['sentiment_score'] < -0.2:
            impact['overall_sentiment'] = 'bearish'

        # Add commodity impacts
        commodities = snapshot.get('commodities', {})
        oil = commodities.get('CRUDE_OIL_WTI', {})
        gold = commodities.get('GOLD', {})

        if oil.get('change_pct', 0) > 2:
            impact['key_drivers'].append(f"Oil up {oil['change_pct']:.1f}% - energy positive")
            impact['sector_impacts'].setdefault('energy', []).append({
                'factor': 'oil_price', 'signal': 'bullish', 'change': oil['change_pct']
            })
        elif oil.get('change_pct', 0) < -2:
            impact['risk_factors'].append(f"Oil down {oil['change_pct']:.1f}% - energy negative")
            impact['sector_impacts'].setdefault('energy', []).append({
                'factor': 'oil_price', 'signal': 'bearish', 'change': oil['change_pct']
            })

        if gold.get('change_pct', 0) > 1:
            impact['key_drivers'].append(f"Gold up {gold['change_pct']:.1f}% - miners positive")
            impact['sector_impacts'].setdefault('mining', []).append({
                'factor': 'gold_price', 'signal': 'bullish', 'change': gold['change_pct']
            })

        return impact

    def get_country_analysis(self, country_code: str) -> Dict[str, Any]:
        """Get detailed analysis for a specific country"""
        if country_code not in self.COUNTRIES:
            return {'error': f'Unknown country code: {country_code}'}

        country = self.COUNTRIES[country_code]
        return self._get_country_data(country)

    def get_sector_impact_analysis(self, sector: str) -> Dict[str, Any]:
        """Get impact analysis for a specific Canadian sector"""
        snapshot = self.get_global_snapshot()

        impact = {
            'sector': sector,
            'timestamp': datetime.now().isoformat(),
            'country_influences': [],
            'commodity_influences': [],
            'overall_outlook': 'neutral',
            'outlook_score': 0.0,
        }

        # Find countries that affect this sector
        for code, country in self.COUNTRIES.items():
            if sector in country.primary_sectors or 'all' in country.primary_sectors:
                country_data = snapshot['countries'].get(code, {})
                influence = {
                    'country': country.name,
                    'correlation': country.tsx_correlation,
                    'signal': country_data.get('overall_signal', 'neutral'),
                    'indices': country_data.get('indices', {}),
                }
                impact['country_influences'].append(influence)

        # Add relevant commodity influences
        if sector in ['energy', 'pipelines']:
            for name in ['CRUDE_OIL_WTI', 'CRUDE_OIL_BRENT', 'NATURAL_GAS']:
                if name in snapshot['commodities']:
                    impact['commodity_influences'].append({
                        'commodity': name,
                        'data': snapshot['commodities'][name]
                    })
        elif sector in ['mining', 'materials']:
            for name in ['GOLD', 'SILVER', 'COPPER', 'IRON_ORE']:
                if name in snapshot['commodities']:
                    impact['commodity_influences'].append({
                        'commodity': name,
                        'data': snapshot['commodities'][name]
                    })

        # Calculate outlook score
        scores = []
        for inf in impact['country_influences']:
            if inf['signal'] == 'bullish':
                scores.append(inf['correlation'])
            elif inf['signal'] == 'bearish':
                scores.append(-inf['correlation'])

        for inf in impact['commodity_influences']:
            change = inf['data'].get('change_pct', 0)
            if change > 1:
                scores.append(0.3)
            elif change < -1:
                scores.append(-0.3)

        if scores:
            impact['outlook_score'] = round(sum(scores) / len(scores), 2)
            if impact['outlook_score'] > 0.2:
                impact['overall_outlook'] = 'bullish'
            elif impact['outlook_score'] < -0.2:
                impact['overall_outlook'] = 'bearish'

        return impact

    def collect_historical_influence_data(
        self,
        years: int = 20,
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Collect historical data for all global influence factors

        Args:
            years: Years of history to collect
            force_refresh: Force re-download

        Returns:
            Summary of collected data
        """
        if not YFINANCE_AVAILABLE:
            return {'error': 'yfinance not available'}

        logger.info(f"Collecting {years} years of global influence data...")

        summary = {
            'start_date': (datetime.now() - timedelta(days=365*years)).isoformat(),
            'end_date': datetime.now().isoformat(),
            'countries': {},
            'commodities': {},
            'yields': {},
        }

        start = datetime.now() - timedelta(days=365*years)
        end = datetime.now()

        # Collect country index data
        for code, country in self.COUNTRIES.items():
            summary['countries'][code] = {'indices': {}}

            for name, symbol in country.key_indices.items():
                try:
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(start=start, end=end)

                    if not data.empty:
                        # Save to file
                        filename = self.data_dir / f"{code}_{name}.parquet"
                        data.to_parquet(filename)
                        self.historical_data[f"{code}_{name}"] = data

                        summary['countries'][code]['indices'][name] = {
                            'records': len(data),
                            'file': str(filename)
                        }
                        logger.info(f"  {code} {name}: {len(data)} records")

                except Exception as e:
                    logger.error(f"Error collecting {code} {name}: {e}")

        # Collect commodity data
        for name, symbol in self.COMMODITIES.items():
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start, end=end)

                if not data.empty:
                    filename = self.data_dir / f"commodity_{name}.parquet"
                    data.to_parquet(filename)
                    self.historical_data[f"commodity_{name}"] = data

                    summary['commodities'][name] = {
                        'records': len(data),
                        'file': str(filename)
                    }
                    logger.info(f"  Commodity {name}: {len(data)} records")

            except Exception as e:
                logger.error(f"Error collecting commodity {name}: {e}")

        # Collect yield data
        for name, symbol in self.YIELDS.items():
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start, end=end)

                if not data.empty:
                    filename = self.data_dir / f"yield_{name}.parquet"
                    data.to_parquet(filename)
                    self.historical_data[f"yield_{name}"] = data

                    summary['yields'][name] = {
                        'records': len(data),
                        'file': str(filename)
                    }
                    logger.info(f"  Yield {name}: {len(data)} records")

            except Exception as e:
                logger.error(f"Error collecting yield {name}: {e}")

        # Save summary
        summary_file = self.data_dir / "global_influence_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        return summary


# Convenience functions
def get_global_market_impact() -> Dict[str, Any]:
    """Quick function to get current global market impact on TSX"""
    tracker = GlobalInfluenceTracker(demo_mode=True)
    return tracker.get_global_snapshot()


def collect_all_global_data(years: int = 20) -> Dict[str, Any]:
    """Collect all historical global influence data"""
    tracker = GlobalInfluenceTracker(demo_mode=False)
    return tracker.collect_historical_influence_data(years=years)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("="*60)
    print("GLOBAL INFLUENCE TRACKER - TSX IMPACT ANALYSIS")
    print("="*60)

    tracker = GlobalInfluenceTracker(demo_mode=True)
    snapshot = tracker.get_global_snapshot()

    print("\n--- COUNTRY SIGNALS ---")
    for code, data in snapshot['countries'].items():
        print(f"{data['name']:20s} | Signal: {data['overall_signal']:10s} | "
              f"TSX Correlation: {data['tsx_correlation']:.2f}")

    print("\n--- TSX IMPACT SUMMARY ---")
    impact = snapshot['tsx_impact_summary']
    print(f"Overall Sentiment: {impact['overall_sentiment']}")
    print(f"Sentiment Score: {impact['sentiment_score']}")

    print("\nKey Drivers:")
    for driver in impact['key_drivers'][:5]:
        print(f"  + {driver}")

    print("\nRisk Factors:")
    for risk in impact['risk_factors'][:5]:
        print(f"  - {risk}")
