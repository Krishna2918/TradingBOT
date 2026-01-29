"""
Currency & FX Tracker

Tracks currency exchange rates that impact Canadian stocks:
- USD/CAD (critical for all cross-listed and export companies)
- EUR/CAD
- CNY/CAD (China trade impact)

Provides currency impact scores for trading signals
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import os

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("yfinance not available for FX data")


class CurrencyTracker:
    """
    Tracks currency exchange rates and their impact on Canadian stocks
    """

    # Currency pairs relevant to Canadian markets
    CURRENCY_PAIRS = {
        'USDCAD': 'USDCAD=X',  # US Dollar / Canadian Dollar
        'EURCAD': 'EURCAD=X',  # Euro / Canadian Dollar
        'GBPCAD': 'GBPCAD=X',  # British Pound / Canadian Dollar
        'CNYCAD': 'CNYCAD=X',  # Chinese Yuan / Canadian Dollar
        'JPYCAD': 'JPYCAD=X',  # Japanese Yen / Canadian Dollar
    }

    # Sector sensitivity to USD/CAD
    SECTOR_FX_SENSITIVITY = {
        # Exporters benefit from weak CAD (negative correlation)
        'energy': -0.8,      # Oil priced in USD
        'mining': -0.7,      # Commodities priced in USD
        'forestry': -0.6,    # Export-heavy
        'agriculture': -0.5, # Export-heavy

        # Importers hurt by weak CAD (positive correlation)
        'retail': 0.4,       # Import goods
        'consumer': 0.3,     # Import goods

        # Banks - mixed (US operations but CAD base)
        'banks': 0.2,

        # Utilities - mostly domestic
        'utilities': 0.1,

        # Tech - depends on export/import mix
        'technology': 0.0,
    }

    # Symbol to sector mapping for major Canadian stocks
    SYMBOL_SECTORS = {
        # Energy
        'CNQ.TO': 'energy', 'SU.TO': 'energy', 'IMO.TO': 'energy',
        'CVE.TO': 'energy', 'TOU.TO': 'energy', 'ARX.TO': 'energy',
        'ENB.TO': 'energy', 'TRP.TO': 'energy', 'PPL.TO': 'energy',

        # Mining
        'ABX.TO': 'mining', 'WPM.TO': 'mining', 'FNV.TO': 'mining',
        'K.TO': 'mining', 'NTR.TO': 'mining', 'TECK.TO': 'mining',

        # Banks
        'RY.TO': 'banks', 'TD.TO': 'banks', 'BNS.TO': 'banks',
        'BMO.TO': 'banks', 'CM.TO': 'banks', 'NA.TO': 'banks',

        # Retail
        'L.TO': 'retail', 'CTC-A.TO': 'retail', 'DOL.TO': 'retail',

        # Utilities
        'FTS.TO': 'utilities', 'EMA.TO': 'utilities', 'H.TO': 'utilities',

        # Tech
        'SHOP.TO': 'technology', 'CSU.TO': 'technology', 'OTEX.TO': 'technology',
    }

    def __init__(self, demo_mode: bool = True):
        """
        Initialize currency tracker

        Args:
            demo_mode: If True, generates simulated data when API unavailable
        """
        self.demo_mode = demo_mode
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_expiry = timedelta(minutes=5)  # FX moves fast
        self.historical_rates: Dict[str, List[float]] = {}

        logger.info(f"Currency Tracker initialized (demo_mode={demo_mode})")

    def get_current_rates(self) -> Dict[str, Dict[str, Any]]:
        """
        Get current exchange rates for all tracked currency pairs

        Returns:
            Dict mapping pair name to:
                - rate: Current exchange rate
                - change_1d: 1-day percentage change
                - change_5d: 5-day percentage change
                - trend: 'strengthening', 'weakening', or 'stable'
        """
        rates = {}

        for pair_name, yahoo_symbol in self.CURRENCY_PAIRS.items():
            # Check cache
            cached = self.cache.get(pair_name)
            if cached and (datetime.now() - cached['timestamp']) < self.cache_expiry:
                rates[pair_name] = cached['data']
                continue

            # Fetch fresh data
            if YFINANCE_AVAILABLE and not self.demo_mode:
                data = self._fetch_fx_data(yahoo_symbol, pair_name)
            else:
                data = self._get_demo_data(pair_name)

            # Cache it
            self.cache[pair_name] = {
                'data': data,
                'timestamp': datetime.now()
            }

            rates[pair_name] = data

        return rates

    def get_usdcad(self) -> Dict[str, Any]:
        """Get USD/CAD rate specifically (most important for Canadian stocks)"""
        rates = self.get_current_rates()
        return rates.get('USDCAD', self._get_demo_data('USDCAD'))

    def get_fx_impact(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Calculate FX impact score for given symbols

        Args:
            symbols: List of stock symbols

        Returns:
            Dict mapping symbol to:
                - impact_score: -1 to 1 (negative = FX headwind, positive = tailwind)
                - usdcad_sensitivity: How sensitive the stock is to USD/CAD
                - recommendation: 'favorable', 'unfavorable', or 'neutral'
        """
        results = {}

        # Get current FX rates
        usdcad = self.get_usdcad()
        usdcad_change = usdcad.get('change_1d', 0.0)
        usdcad_trend = usdcad.get('trend', 'stable')

        for symbol in symbols:
            # Get sector for symbol
            sector = self.SYMBOL_SECTORS.get(symbol, 'unknown')
            sensitivity = self.SECTOR_FX_SENSITIVITY.get(sector, 0.0)

            # Calculate impact
            # If CAD weakening (USDCAD rising) and negative sensitivity (exporter)
            # → positive impact (tailwind)
            impact_score = -sensitivity * (usdcad_change / 100)  # Normalize
            impact_score = max(-1.0, min(1.0, impact_score * 10))  # Scale and clamp

            # Determine recommendation
            if impact_score > 0.2:
                recommendation = 'favorable'
            elif impact_score < -0.2:
                recommendation = 'unfavorable'
            else:
                recommendation = 'neutral'

            results[symbol] = {
                'impact_score': round(impact_score, 3),
                'usdcad_sensitivity': sensitivity,
                'usdcad_rate': usdcad.get('rate', 1.35),
                'usdcad_change_1d': usdcad_change,
                'usdcad_trend': usdcad_trend,
                'sector': sector,
                'recommendation': recommendation,
                'timestamp': datetime.now().isoformat()
            }

        return results

    def _fetch_fx_data(self, yahoo_symbol: str, pair_name: str) -> Dict[str, Any]:
        """Fetch real FX data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(yahoo_symbol)
            hist = ticker.history(period='5d')

            if hist.empty:
                logger.warning(f"No FX data for {yahoo_symbol}")
                return self._get_demo_data(pair_name)

            current_rate = hist['Close'].iloc[-1]
            prev_rate = hist['Close'].iloc[-2] if len(hist) > 1 else current_rate
            rate_5d_ago = hist['Close'].iloc[0] if len(hist) >= 5 else current_rate

            change_1d = ((current_rate - prev_rate) / prev_rate) * 100
            change_5d = ((current_rate - rate_5d_ago) / rate_5d_ago) * 100

            # Determine trend
            if change_5d > 0.5:
                trend = 'strengthening' if pair_name.startswith('USD') else 'weakening'
            elif change_5d < -0.5:
                trend = 'weakening' if pair_name.startswith('USD') else 'strengthening'
            else:
                trend = 'stable'

            return {
                'rate': round(current_rate, 4),
                'change_1d': round(change_1d, 3),
                'change_5d': round(change_5d, 3),
                'trend': trend,
                'high_5d': round(hist['High'].max(), 4),
                'low_5d': round(hist['Low'].min(), 4),
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error fetching FX data for {yahoo_symbol}: {e}")
            return self._get_demo_data(pair_name)

    def _get_demo_data(self, pair_name: str) -> Dict[str, Any]:
        """Generate simulated FX data"""
        import random

        # Base rates (approximate)
        base_rates = {
            'USDCAD': 1.35,
            'EURCAD': 1.47,
            'GBPCAD': 1.72,
            'CNYCAD': 0.19,
            'JPYCAD': 0.0091,
        }

        base = base_rates.get(pair_name, 1.0)

        # Add small random variation
        rate = base * (1 + random.uniform(-0.01, 0.01))
        change_1d = random.uniform(-0.5, 0.5)
        change_5d = random.uniform(-1.0, 1.0)

        if change_5d > 0.3:
            trend = 'strengthening'
        elif change_5d < -0.3:
            trend = 'weakening'
        else:
            trend = 'stable'

        return {
            'rate': round(rate, 4),
            'change_1d': round(change_1d, 3),
            'change_5d': round(change_5d, 3),
            'trend': trend,
            'high_5d': round(rate * 1.01, 4),
            'low_5d': round(rate * 0.99, 4),
            'demo': True,
            'timestamp': datetime.now().isoformat()
        }


class USPreMarketTracker:
    """
    Tracks US pre-market futures for gap prediction on Canadian markets

    Key futures:
    - ES (S&P 500 E-mini) - Overall market direction
    - NQ (Nasdaq E-mini) - Tech sentiment
    - CL (Crude Oil) - Energy sector impact
    - GC (Gold) - Mining sector impact
    """

    FUTURES_SYMBOLS = {
        'SP500': 'ES=F',      # S&P 500 E-mini futures
        'NASDAQ': 'NQ=F',     # Nasdaq 100 E-mini futures
        'DOW': 'YM=F',        # Dow Jones E-mini futures
        'CRUDE_OIL': 'CL=F',  # Crude Oil futures
        'GOLD': 'GC=F',       # Gold futures
        'NATURAL_GAS': 'NG=F', # Natural Gas futures
        'VIX': '^VIX',        # Volatility Index
    }

    # Impact on Canadian sectors
    FUTURES_SECTOR_IMPACT = {
        'SP500': {
            'banks': 0.7, 'technology': 0.6, 'consumer': 0.5,
            'energy': 0.4, 'mining': 0.3
        },
        'NASDAQ': {
            'technology': 0.9, 'banks': 0.3, 'consumer': 0.4
        },
        'CRUDE_OIL': {
            'energy': 0.95, 'pipelines': 0.8, 'banks': 0.2
        },
        'GOLD': {
            'mining': 0.9, 'banks': 0.1
        },
        'NATURAL_GAS': {
            'energy': 0.6, 'pipelines': 0.7, 'utilities': 0.4
        },
        'VIX': {
            'banks': -0.5, 'energy': -0.4, 'technology': -0.6,
            'mining': -0.3, 'consumer': -0.4
        }
    }

    def __init__(self, demo_mode: bool = True):
        """Initialize US pre-market tracker"""
        self.demo_mode = demo_mode
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_expiry = timedelta(minutes=2)  # Pre-market data changes fast

        logger.info(f"US Pre-Market Tracker initialized (demo_mode={demo_mode})")

    def get_premarket_data(self) -> Dict[str, Dict[str, Any]]:
        """
        Get current pre-market futures data

        Returns:
            Dict mapping future name to:
                - price: Current price
                - change: Dollar change from previous close
                - change_pct: Percentage change
                - signal: 'bullish', 'bearish', or 'neutral'
        """
        results = {}

        for name, symbol in self.FUTURES_SYMBOLS.items():
            # Check cache
            cached = self.cache.get(name)
            if cached and (datetime.now() - cached['timestamp']) < self.cache_expiry:
                results[name] = cached['data']
                continue

            # Fetch data
            if YFINANCE_AVAILABLE and not self.demo_mode:
                data = self._fetch_futures_data(symbol, name)
            else:
                data = self._get_demo_data(name)

            # Cache
            self.cache[name] = {
                'data': data,
                'timestamp': datetime.now()
            }

            results[name] = data

        return results

    def predict_tsx_gap(self) -> Dict[str, Any]:
        """
        Predict TSX opening gap based on overnight US futures

        Returns:
            Dict with:
                - predicted_gap_pct: Expected gap percentage
                - confidence: 0-1 confidence score
                - direction: 'up', 'down', or 'flat'
                - drivers: List of main contributing factors
        """
        futures = self.get_premarket_data()

        # Weight the futures impact
        weighted_impact = 0.0
        total_weight = 0.0
        drivers = []

        weights = {
            'SP500': 0.35,      # S&P 500 has highest correlation
            'CRUDE_OIL': 0.25,  # Oil critical for TSX
            'GOLD': 0.15,       # Mining stocks
            'NASDAQ': 0.10,     # Tech stocks
            'VIX': 0.10,        # Fear gauge (inverse)
            'DOW': 0.05,        # Small weight
        }

        for name, weight in weights.items():
            if name in futures:
                change_pct = futures[name].get('change_pct', 0)

                # VIX is inverse (high VIX = bearish)
                if name == 'VIX':
                    change_pct = -change_pct * 0.5  # Dampen VIX impact

                weighted_impact += change_pct * weight
                total_weight += weight

                # Track significant drivers
                if abs(change_pct) > 0.5:
                    direction = 'up' if change_pct > 0 else 'down'
                    drivers.append(f"{name} {direction} {abs(change_pct):.1f}%")

        # Calculate predicted gap
        if total_weight > 0:
            predicted_gap = weighted_impact / total_weight
        else:
            predicted_gap = 0.0

        # Dampen prediction (futures don't translate 1:1)
        predicted_gap *= 0.7

        # Determine direction and confidence
        if predicted_gap > 0.3:
            direction = 'up'
            confidence = min(0.8, 0.5 + abs(predicted_gap) / 5)
        elif predicted_gap < -0.3:
            direction = 'down'
            confidence = min(0.8, 0.5 + abs(predicted_gap) / 5)
        else:
            direction = 'flat'
            confidence = 0.6

        return {
            'predicted_gap_pct': round(predicted_gap, 2),
            'confidence': round(confidence, 2),
            'direction': direction,
            'drivers': drivers[:3],  # Top 3 drivers
            'futures_summary': {
                name: f"{data.get('change_pct', 0):+.2f}%"
                for name, data in futures.items()
            },
            'timestamp': datetime.now().isoformat()
        }

    def get_sector_outlook(self, sectors: List[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Get pre-market outlook for Canadian sectors

        Args:
            sectors: List of sectors (default: all)

        Returns:
            Dict mapping sector to outlook
        """
        if sectors is None:
            sectors = ['energy', 'mining', 'banks', 'technology', 'pipelines']

        futures = self.get_premarket_data()
        results = {}

        for sector in sectors:
            impact = 0.0
            contributors = []

            for future_name, sector_impacts in self.FUTURES_SECTOR_IMPACT.items():
                if sector in sector_impacts and future_name in futures:
                    sensitivity = sector_impacts[sector]
                    change = futures[future_name].get('change_pct', 0)
                    contribution = sensitivity * change
                    impact += contribution

                    if abs(contribution) > 0.1:
                        contributors.append(f"{future_name}: {contribution:+.2f}%")

            # Determine outlook
            if impact > 0.5:
                outlook = 'bullish'
            elif impact < -0.5:
                outlook = 'bearish'
            else:
                outlook = 'neutral'

            results[sector] = {
                'outlook': outlook,
                'expected_impact_pct': round(impact, 2),
                'contributors': contributors,
                'timestamp': datetime.now().isoformat()
            }

        return results

    def _fetch_futures_data(self, symbol: str, name: str) -> Dict[str, Any]:
        """Fetch real futures data"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='2d')

            if hist.empty:
                return self._get_demo_data(name)

            current = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current

            change = current - prev_close
            change_pct = (change / prev_close) * 100

            if change_pct > 0.3:
                signal = 'bullish'
            elif change_pct < -0.3:
                signal = 'bearish'
            else:
                signal = 'neutral'

            return {
                'price': round(current, 2),
                'prev_close': round(prev_close, 2),
                'change': round(change, 2),
                'change_pct': round(change_pct, 2),
                'signal': signal,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return self._get_demo_data(name)

    def _get_demo_data(self, name: str) -> Dict[str, Any]:
        """Generate demo futures data"""
        import random

        base_prices = {
            'SP500': 5200,
            'NASDAQ': 18500,
            'DOW': 39000,
            'CRUDE_OIL': 78,
            'GOLD': 2050,
            'NATURAL_GAS': 2.5,
            'VIX': 15,
        }

        base = base_prices.get(name, 100)
        change_pct = random.uniform(-1.5, 1.5)
        change = base * (change_pct / 100)

        if change_pct > 0.3:
            signal = 'bullish'
        elif change_pct < -0.3:
            signal = 'bearish'
        else:
            signal = 'neutral'

        return {
            'price': round(base + change, 2),
            'prev_close': round(base, 2),
            'change': round(change, 2),
            'change_pct': round(change_pct, 2),
            'signal': signal,
            'demo': True,
            'timestamp': datetime.now().isoformat()
        }


# Convenience function for quick access
def get_market_context() -> Dict[str, Any]:
    """
    Get complete market context for pre-market analysis

    Returns comprehensive view of:
    - Currency rates (USD/CAD)
    - US futures (overnight moves)
    - TSX gap prediction
    - Sector outlooks
    """
    currency_tracker = CurrencyTracker(demo_mode=True)
    premarket_tracker = USPreMarketTracker(demo_mode=True)

    return {
        'currencies': currency_tracker.get_current_rates(),
        'usdcad': currency_tracker.get_usdcad(),
        'futures': premarket_tracker.get_premarket_data(),
        'tsx_gap_prediction': premarket_tracker.predict_tsx_gap(),
        'sector_outlook': premarket_tracker.get_sector_outlook(),
        'timestamp': datetime.now().isoformat()
    }
