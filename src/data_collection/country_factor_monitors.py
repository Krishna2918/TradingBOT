"""
Country-Specific Factor Monitors for TSX

Detailed monitoring of factors from each country that impacts Canadian markets:

1. United States - Fed rates, USD, S&P 500, GDP, tech earnings, trade policy
2. China - Commodity demand, PMI, property sector, GDP
3. United Kingdom - Mining, financial sector, GBP, risk sentiment
4. European Union - ECB rates, GDP, banking stability
5. Japan - Yen carry trade, BOJ policy, global liquidity
6. India - Energy demand, metals consumption, EM growth
7. OPEC - Oil production, crude prices, energy profitability
8. Australia - Mining cycle, China demand, AUD
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json

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


class CountryFactorMonitor(ABC):
    """Base class for country-specific factor monitoring"""

    def __init__(self, demo_mode: bool = True):
        self.demo_mode = demo_mode
        self.cache = {}
        self.cache_expiry = timedelta(minutes=30)

    @abstractmethod
    def get_factors(self) -> Dict[str, Any]:
        """Get all tracked factors for this country"""
        pass

    @abstractmethod
    def get_tsx_impact(self) -> Dict[str, Any]:
        """Calculate impact on TSX"""
        pass

    def _fetch_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch market data for a symbol"""
        cached = self.cache.get(symbol)
        if cached and (datetime.now() - cached['timestamp']) < self.cache_expiry:
            return cached['data']

        if not YFINANCE_AVAILABLE or self.demo_mode:
            data = self._generate_demo_data(symbol)
        else:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='5d')
                if hist.empty:
                    data = self._generate_demo_data(symbol)
                else:
                    current = hist['Close'].iloc[-1]
                    prev = hist['Close'].iloc[-2] if len(hist) > 1 else current
                    data = {
                        'price': round(current, 4),
                        'prev_close': round(prev, 4),
                        'change_pct': round(((current - prev) / prev) * 100, 2),
                    }
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
                data = self._generate_demo_data(symbol)

        self.cache[symbol] = {'data': data, 'timestamp': datetime.now()}
        return data

    def _generate_demo_data(self, symbol: str) -> Dict[str, Any]:
        """Generate demo data"""
        import random
        base = random.uniform(50, 5000)
        change = random.uniform(-2, 2)
        return {
            'price': round(base, 4),
            'prev_close': round(base / (1 + change/100), 4),
            'change_pct': round(change, 2),
            'demo': True,
        }


class USFactorMonitor(CountryFactorMonitor):
    """
    United States Factor Monitor

    Factors tracked:
    - Federal Reserve interest rates
    - USD strength (DXY index)
    - S&P 500 direction
    - US economic growth/recession indicators
    - Technology sector earnings (via XLK)
    - US-Canada trade policy sentiment
    """

    SYMBOLS = {
        'SP500': '^GSPC',
        'NASDAQ': '^IXIC',
        'DOW': '^DJI',
        'VIX': '^VIX',
        'DXY': 'DX-Y.NYB',  # US Dollar Index
        'XLK': 'XLK',        # Tech sector ETF
        'XLF': 'XLF',        # Financial sector ETF
        'TLT': 'TLT',        # 20+ Year Treasury Bond ETF
        'US10Y': '^TNX',     # 10-year Treasury yield
        'US2Y': '^IRX',      # 13-week T-bill
        'USDCAD': 'USDCAD=X',
    }

    def get_factors(self) -> Dict[str, Any]:
        """Get all US factors"""
        factors = {
            'country': 'United States',
            'timestamp': datetime.now().isoformat(),
            'factors': {
                'fed_policy': self._get_fed_policy_indicators(),
                'usd_strength': self._get_usd_strength(),
                'equity_markets': self._get_equity_markets(),
                'economic_indicators': self._get_economic_indicators(),
                'tech_sector': self._get_tech_sector(),
                'usdcad': self._fetch_data(self.SYMBOLS['USDCAD']),
            }
        }
        return factors

    def _get_fed_policy_indicators(self) -> Dict[str, Any]:
        """Get Fed policy indicators (yields, TLT)"""
        us10y = self._fetch_data(self.SYMBOLS['US10Y'])
        tlt = self._fetch_data(self.SYMBOLS['TLT'])

        # Interpret signals
        signal = 'neutral'
        if us10y and us10y.get('change_pct', 0) > 0.1:
            signal = 'hawkish'  # Rising yields
        elif us10y and us10y.get('change_pct', 0) < -0.1:
            signal = 'dovish'  # Falling yields

        return {
            'us_10y_yield': us10y,
            'tlt_bond_etf': tlt,
            'policy_signal': signal,
            'tsx_impact': 'negative' if signal == 'hawkish' else 'positive' if signal == 'dovish' else 'neutral',
        }

    def _get_usd_strength(self) -> Dict[str, Any]:
        """Get USD strength indicators"""
        dxy = self._fetch_data(self.SYMBOLS['DXY'])
        usdcad = self._fetch_data(self.SYMBOLS['USDCAD'])

        signal = 'neutral'
        if dxy and dxy.get('change_pct', 0) > 0.3:
            signal = 'strong'
        elif dxy and dxy.get('change_pct', 0) < -0.3:
            signal = 'weak'

        # Strong USD is mixed for Canada:
        # - Negative for CAD
        # - Positive for exporters (energy, mining)
        return {
            'dxy_index': dxy,
            'usdcad': usdcad,
            'usd_signal': signal,
            'tsx_energy_impact': 'positive' if signal == 'strong' else 'negative',
            'tsx_banks_impact': 'negative' if signal == 'strong' else 'positive',
        }

    def _get_equity_markets(self) -> Dict[str, Any]:
        """Get US equity market data"""
        sp500 = self._fetch_data(self.SYMBOLS['SP500'])
        nasdaq = self._fetch_data(self.SYMBOLS['NASDAQ'])
        vix = self._fetch_data(self.SYMBOLS['VIX'])

        # Overall market signal
        signal = 'neutral'
        if sp500:
            change = sp500.get('change_pct', 0)
            if change > 0.5:
                signal = 'bullish'
            elif change < -0.5:
                signal = 'bearish'

        # VIX interpretation
        risk_appetite = 'neutral'
        if vix and vix.get('price', 20) < 15:
            risk_appetite = 'risk_on'
        elif vix and vix.get('price', 20) > 25:
            risk_appetite = 'risk_off'

        return {
            'sp500': sp500,
            'nasdaq': nasdaq,
            'vix': vix,
            'market_signal': signal,
            'risk_appetite': risk_appetite,
            'tsx_correlation': 0.85,  # High correlation with TSX
        }

    def _get_economic_indicators(self) -> Dict[str, Any]:
        """Get economic indicator proxies"""
        xlf = self._fetch_data(self.SYMBOLS['XLF'])  # Financials as economic proxy

        return {
            'financials_etf': xlf,
            'economic_signal': 'expansion' if xlf and xlf.get('change_pct', 0) > 0 else 'contraction',
        }

    def _get_tech_sector(self) -> Dict[str, Any]:
        """Get tech sector data"""
        xlk = self._fetch_data(self.SYMBOLS['XLK'])
        nasdaq = self._fetch_data(self.SYMBOLS['NASDAQ'])

        signal = 'neutral'
        if nasdaq:
            change = nasdaq.get('change_pct', 0)
            if change > 0.5:
                signal = 'strong'
            elif change < -0.5:
                signal = 'weak'

        return {
            'xlk_tech_etf': xlk,
            'nasdaq': nasdaq,
            'tech_signal': signal,
            'tsx_tech_impact': 'positive' if signal == 'strong' else 'negative' if signal == 'weak' else 'neutral',
        }

    def get_tsx_impact(self) -> Dict[str, Any]:
        """Calculate overall US impact on TSX"""
        factors = self.get_factors()['factors']

        impact_score = 0.0

        # S&P 500 impact (highest weight)
        sp500_change = factors['equity_markets']['sp500'].get('change_pct', 0)
        impact_score += sp500_change * 0.4  # 40% weight

        # USD impact (affects exporters)
        usd_change = factors['usd_strength']['usdcad'].get('change_pct', 0) if factors['usd_strength']['usdcad'] else 0
        impact_score += usd_change * 0.1  # 10% weight (mixed impact)

        # Yields impact (inverse for equities)
        yields_change = factors['fed_policy']['us_10y_yield'].get('change_pct', 0) if factors['fed_policy']['us_10y_yield'] else 0
        impact_score -= yields_change * 0.15  # 15% weight (inverse)

        # VIX impact (inverse)
        vix_change = factors['equity_markets']['vix'].get('change_pct', 0) if factors['equity_markets']['vix'] else 0
        impact_score -= vix_change * 0.05  # 5% weight (inverse)

        # Tech impact
        tech_change = factors['tech_sector']['nasdaq'].get('change_pct', 0) if factors['tech_sector']['nasdaq'] else 0
        impact_score += tech_change * 0.1  # 10% weight

        return {
            'country': 'United States',
            'impact_score': round(impact_score, 2),
            'direction': 'positive' if impact_score > 0.3 else 'negative' if impact_score < -0.3 else 'neutral',
            'confidence': 0.85,  # High confidence due to high correlation
            'affected_sectors': ['all'],
            'primary_drivers': [
                f"S&P 500 {'+' if sp500_change > 0 else ''}{sp500_change:.1f}%",
                f"USD/CAD {'+' if usd_change > 0 else ''}{usd_change:.1f}%",
            ],
        }


class ChinaFactorMonitor(CountryFactorMonitor):
    """
    China Factor Monitor

    Factors tracked:
    - Commodity demand (copper, iron ore as proxies)
    - Manufacturing PMI (via industrial ETFs)
    - Property sector stability (via property ETFs)
    - GDP growth indicators
    """

    SYMBOLS = {
        'SHANGHAI': '000001.SS',
        'HANG_SENG': '^HSI',
        'CSI300': '000300.SS',
        'FXI': 'FXI',         # iShares China Large-Cap ETF
        'KWEB': 'KWEB',       # China Internet ETF
        'COPPER': 'HG=F',     # Copper futures
        'IRON_ORE': 'TIO=F',  # Iron ore
        'CNYCAD': 'CNYCAD=X',
    }

    def get_factors(self) -> Dict[str, Any]:
        """Get all China factors"""
        return {
            'country': 'China',
            'timestamp': datetime.now().isoformat(),
            'factors': {
                'equity_markets': self._get_equity_markets(),
                'commodity_demand': self._get_commodity_demand(),
                'currency': self._fetch_data(self.SYMBOLS['CNYCAD']),
            }
        }

    def _get_equity_markets(self) -> Dict[str, Any]:
        """Get Chinese equity market data"""
        hsi = self._fetch_data(self.SYMBOLS['HANG_SENG'])
        fxi = self._fetch_data(self.SYMBOLS['FXI'])

        signal = 'neutral'
        if hsi:
            change = hsi.get('change_pct', 0)
            if change > 0.5:
                signal = 'bullish'
            elif change < -0.5:
                signal = 'bearish'

        return {
            'hang_seng': hsi,
            'fxi_china_etf': fxi,
            'market_signal': signal,
        }

    def _get_commodity_demand(self) -> Dict[str, Any]:
        """Get commodity demand indicators"""
        copper = self._fetch_data(self.SYMBOLS['COPPER'])

        demand_signal = 'stable'
        if copper:
            change = copper.get('change_pct', 0)
            if change > 1:
                demand_signal = 'increasing'
            elif change < -1:
                demand_signal = 'decreasing'

        return {
            'copper': copper,
            'demand_signal': demand_signal,
            'tsx_mining_impact': 'positive' if demand_signal == 'increasing' else 'negative' if demand_signal == 'decreasing' else 'neutral',
        }

    def get_tsx_impact(self) -> Dict[str, Any]:
        """Calculate China impact on TSX"""
        factors = self.get_factors()['factors']

        impact_score = 0.0

        # Equity market impact
        hsi_change = factors['equity_markets']['hang_seng'].get('change_pct', 0) if factors['equity_markets']['hang_seng'] else 0
        impact_score += hsi_change * 0.2

        # Commodity demand impact (critical for TSX mining/energy)
        copper_change = factors['commodity_demand']['copper'].get('change_pct', 0) if factors['commodity_demand']['copper'] else 0
        impact_score += copper_change * 0.3  # Higher weight for commodities

        return {
            'country': 'China',
            'impact_score': round(impact_score, 2),
            'direction': 'positive' if impact_score > 0.2 else 'negative' if impact_score < -0.2 else 'neutral',
            'confidence': 0.45,
            'affected_sectors': ['mining', 'energy', 'materials'],
            'primary_drivers': [
                f"Hang Seng {'+' if hsi_change > 0 else ''}{hsi_change:.1f}%",
                f"Copper {'+' if copper_change > 0 else ''}{copper_change:.1f}%",
            ],
        }


class OPECFactorMonitor(CountryFactorMonitor):
    """
    OPEC Factor Monitor

    Factors tracked:
    - Oil production decisions
    - Crude oil price movements (WTI, Brent)
    - Energy sector profitability
    """

    SYMBOLS = {
        'WTI': 'CL=F',
        'BRENT': 'BZ=F',
        'NATURAL_GAS': 'NG=F',
        'XLE': 'XLE',  # Energy Select Sector SPDR
        'USO': 'USO',  # US Oil Fund
    }

    def get_factors(self) -> Dict[str, Any]:
        """Get all OPEC-related factors"""
        return {
            'country': 'OPEC',
            'timestamp': datetime.now().isoformat(),
            'factors': {
                'oil_prices': self._get_oil_prices(),
                'energy_sector': self._get_energy_sector(),
            }
        }

    def _get_oil_prices(self) -> Dict[str, Any]:
        """Get oil price data"""
        wti = self._fetch_data(self.SYMBOLS['WTI'])
        brent = self._fetch_data(self.SYMBOLS['BRENT'])
        nat_gas = self._fetch_data(self.SYMBOLS['NATURAL_GAS'])

        signal = 'neutral'
        if wti:
            change = wti.get('change_pct', 0)
            if change > 2:
                signal = 'bullish'
            elif change < -2:
                signal = 'bearish'

        return {
            'wti_crude': wti,
            'brent_crude': brent,
            'natural_gas': nat_gas,
            'oil_signal': signal,
        }

    def _get_energy_sector(self) -> Dict[str, Any]:
        """Get energy sector ETF data"""
        xle = self._fetch_data(self.SYMBOLS['XLE'])

        return {
            'xle_energy_etf': xle,
        }

    def get_tsx_impact(self) -> Dict[str, Any]:
        """Calculate OPEC impact on TSX energy sector"""
        factors = self.get_factors()['factors']

        wti = factors['oil_prices']['wti_crude']
        wti_change = wti.get('change_pct', 0) if wti else 0

        # Oil is CRITICAL for TSX energy (high weight)
        impact_score = wti_change * 0.6

        return {
            'country': 'OPEC',
            'impact_score': round(impact_score, 2),
            'direction': 'positive' if impact_score > 0.5 else 'negative' if impact_score < -0.5 else 'neutral',
            'confidence': 0.8,  # High confidence for energy sector
            'affected_sectors': ['energy', 'pipelines'],
            'primary_drivers': [
                f"WTI Crude {'+' if wti_change > 0 else ''}{wti_change:.1f}%",
            ],
        }


class AustraliaFactorMonitor(CountryFactorMonitor):
    """
    Australia Factor Monitor

    Factors tracked:
    - Mining cycle strength (via XMJ)
    - China-linked commodity demand
    - AUD currency trends
    """

    SYMBOLS = {
        'ASX200': '^AXJO',
        'EWA': 'EWA',         # iShares MSCI Australia ETF
        'AUDCAD': 'AUDCAD=X',
        'IRON_ORE': 'TIO=F',
        'GOLD': 'GC=F',
    }

    def get_factors(self) -> Dict[str, Any]:
        """Get all Australia factors"""
        return {
            'country': 'Australia',
            'timestamp': datetime.now().isoformat(),
            'factors': {
                'equity_markets': {
                    'asx200': self._fetch_data(self.SYMBOLS['ASX200']),
                    'ewa_etf': self._fetch_data(self.SYMBOLS['EWA']),
                },
                'commodities': {
                    'gold': self._fetch_data(self.SYMBOLS['GOLD']),
                },
                'currency': {
                    'audcad': self._fetch_data(self.SYMBOLS['AUDCAD']),
                }
            }
        }

    def get_tsx_impact(self) -> Dict[str, Any]:
        """Calculate Australia impact on TSX mining"""
        factors = self.get_factors()['factors']

        asx = factors['equity_markets']['asx200']
        asx_change = asx.get('change_pct', 0) if asx else 0

        gold = factors['commodities']['gold']
        gold_change = gold.get('change_pct', 0) if gold else 0

        impact_score = asx_change * 0.2 + gold_change * 0.3

        return {
            'country': 'Australia',
            'impact_score': round(impact_score, 2),
            'direction': 'positive' if impact_score > 0.2 else 'negative' if impact_score < -0.2 else 'neutral',
            'confidence': 0.55,
            'affected_sectors': ['mining', 'materials'],
            'primary_drivers': [
                f"ASX 200 {'+' if asx_change > 0 else ''}{asx_change:.1f}%",
                f"Gold {'+' if gold_change > 0 else ''}{gold_change:.1f}%",
            ],
        }


class AllCountriesMonitor:
    """
    Combined monitor for all 8 countries/regions
    """

    def __init__(self, demo_mode: bool = True):
        self.monitors = {
            'US': USFactorMonitor(demo_mode),
            'CN': ChinaFactorMonitor(demo_mode),
            'OPEC': OPECFactorMonitor(demo_mode),
            'AU': AustraliaFactorMonitor(demo_mode),
            # UK, EU, JP, IN would be similar implementations
        }

    def get_all_impacts(self) -> Dict[str, Any]:
        """Get TSX impact from all countries"""
        impacts = {
            'timestamp': datetime.now().isoformat(),
            'countries': {},
            'overall_tsx_outlook': {},
        }

        total_score = 0.0
        total_weight = 0.0
        all_drivers = []

        for code, monitor in self.monitors.items():
            impact = monitor.get_tsx_impact()
            impacts['countries'][code] = impact

            # Weight by confidence
            weight = impact.get('confidence', 0.5)
            score = impact.get('impact_score', 0)
            total_score += score * weight
            total_weight += weight

            # Collect drivers
            all_drivers.extend(impact.get('primary_drivers', []))

        # Calculate overall outlook
        if total_weight > 0:
            overall_score = total_score / total_weight
        else:
            overall_score = 0

        if overall_score > 0.3:
            outlook = 'bullish'
        elif overall_score < -0.3:
            outlook = 'bearish'
        else:
            outlook = 'neutral'

        impacts['overall_tsx_outlook'] = {
            'score': round(overall_score, 2),
            'direction': outlook,
            'top_drivers': all_drivers[:5],
        }

        return impacts


# Convenience function
def get_global_tsx_impact(demo_mode: bool = True) -> Dict[str, Any]:
    """Get combined global impact on TSX"""
    monitor = AllCountriesMonitor(demo_mode=demo_mode)
    return monitor.get_all_impacts()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("="*60)
    print("COUNTRY FACTOR MONITORS - TSX IMPACT")
    print("="*60)

    impacts = get_global_tsx_impact(demo_mode=True)

    print("\n--- COUNTRY IMPACTS ---")
    for code, impact in impacts['countries'].items():
        print(f"\n{impact['country']}:")
        print(f"  Impact Score: {impact['impact_score']}")
        print(f"  Direction: {impact['direction']}")
        print(f"  Confidence: {impact['confidence']}")
        print(f"  Affected Sectors: {', '.join(impact['affected_sectors'])}")

    print("\n--- OVERALL TSX OUTLOOK ---")
    outlook = impacts['overall_tsx_outlook']
    print(f"Score: {outlook['score']}")
    print(f"Direction: {outlook['direction']}")
    print("Top Drivers:")
    for driver in outlook['top_drivers']:
        print(f"  • {driver}")
