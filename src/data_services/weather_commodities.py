"""
Weather & Commodity Correlation Tracker

Tracks weather patterns and commodity prices that affect stock sectors:
- Cold snaps → Natural gas demand → Pipeline stocks (ENB, TRP)
- Droughts → Agriculture impacts
- Oil price spikes → Energy sector
- Gold prices → Mining stocks
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import random

logger = logging.getLogger(__name__)


class WeatherCommodityTracker:
    """
    Tracks weather and commodity correlations for sector-specific trading signals
    """
    
    def __init__(self, demo_mode: bool = True):
        """
        Initialize weather/commodity tracker
        
        Args:
            demo_mode: If True, generates simulated data
        """
        self.demo_mode = demo_mode
        self.cache: Dict[str, Any] = {}
        self.cache_expiry = timedelta(hours=1)  # Cache for 1 hour
        
        # Sector mappings
        self.sector_correlations = {
            'energy': ['CNQ.TO', 'SU.TO', 'IMO.TO', 'CVE.TO', 'TOU.TO', 'ARX.TO'],
            'pipelines': ['ENB.TO', 'TRP.TO', 'PPL.TO', 'KEY.TO'],
            'miners': ['ABX.TO', 'WPM.TO', 'FNV.TO', 'K.TO', 'NTR.TO'],
            'agriculture': ['NTR.TO'],  # Fertilizers, food producers
        }
        
        logger.info(f"Weather/Commodity Tracker initialized (demo_mode={demo_mode})")
    
    def get_impact_score(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get weather/commodity impact scores for given symbols
        
        Args:
            symbols: List of stock symbols
        
        Returns:
            Dict mapping symbol to:
                - impact: -1 to 1 (negative to positive impact)
                - factors: List of impacting factors
                - commodity_prices: Dict of relevant commodity prices
        """
        results = {}
        
        # Get current weather/commodity data (cached)
        market_conditions = self._get_market_conditions()
        
        for symbol in symbols:
            impact_score = 0.0
            factors = []
            
            # Determine sector
            sector = self._get_sector(symbol)
            
            if sector == 'energy':
                # Oil price impact
                oil_change = market_conditions['oil_change_pct']
                if abs(oil_change) > 2.0:
                    impact_score += oil_change / 10.0  # Scale to -1 to 1
                    factors.append(f"oil_price_{'+' if oil_change > 0 else ''}{oil_change:.1f}%")
            
            elif sector == 'pipelines':
                # Natural gas + weather
                gas_change = market_conditions['gas_change_pct']
                temp_impact = market_conditions['temp_impact']
                
                impact_score += gas_change / 10.0
                impact_score += temp_impact * 0.5
                
                if abs(gas_change) > 2.0:
                    factors.append(f"nat_gas_{'+' if gas_change > 0 else ''}{gas_change:.1f}%")
                if abs(temp_impact) > 0.3:
                    factors.append(f"temp_{'cold_snap' if temp_impact > 0 else 'warm_spell'}")
            
            elif sector == 'miners':
                # Gold price impact
                gold_change = market_conditions['gold_change_pct']
                if abs(gold_change) > 1.0:
                    impact_score += gold_change / 10.0
                    factors.append(f"gold_price_{'+' if gold_change > 0 else ''}{gold_change:.1f}%")
            
            elif sector == 'agriculture':
                # Weather impact (drought, flooding)
                weather_impact = market_conditions['weather_severity']
                impact_score += weather_impact
                if abs(weather_impact) > 0.3:
                    factors.append(f"weather_{'adverse' if weather_impact < 0 else 'favorable'}")
            
            # Clip to [-1, 1]
            impact_score = max(-1.0, min(1.0, impact_score))
            
            results[symbol] = {
                'impact': impact_score,
                'factors': factors,
                'commodity_prices': {
                    'oil_wti': market_conditions['oil_price'],
                    'natural_gas': market_conditions['gas_price'],
                    'gold': market_conditions['gold_price'],
                },
            }
        
        return results
    
    def _get_market_conditions(self) -> Dict[str, Any]:
        """Get current market conditions (cached)"""
        
        # Check cache
        if self.cache and (datetime.now() - self.cache.get('timestamp', datetime.min)) < self.cache_expiry:
            return self.cache
        
        # Fetch fresh data
        if self.demo_mode:
            conditions = self._get_demo_conditions()
        else:
            conditions = self._fetch_real_conditions()
        
        conditions['timestamp'] = datetime.now()
        self.cache = conditions
        
        return conditions
    
    def _get_demo_conditions(self) -> Dict[str, Any]:
        """Generate simulated market conditions"""
        return {
            'oil_price': random.uniform(70.0, 90.0),
            'oil_change_pct': random.uniform(-3.0, 3.0),
            'gas_price': random.uniform(2.0, 5.0),
            'gas_change_pct': random.uniform(-4.0, 4.0),
            'gold_price': random.uniform(1800.0, 2100.0),
            'gold_change_pct': random.uniform(-2.0, 2.0),
            'temp_impact': random.uniform(-0.5, 0.5),  # Cold snap vs warm spell
            'weather_severity': random.uniform(-0.3, 0.3),  # Drought vs favorable
        }
    
    def _fetch_real_conditions(self) -> Dict[str, Any]:
        """
        Fetch real weather and commodity data
        
        NOTE: This is a placeholder for future implementation.
        Real implementation would:
        1. Query OpenWeatherMap API for severe weather
        2. Query Alpha Vantage for commodity prices (WTI, Brent, Gold, Nat Gas)
        3. Calculate % changes from previous day
        4. Map weather patterns to sector impacts
        """
        logger.warning("Real weather/commodity APIs not yet implemented, using demo data")
        return self._get_demo_conditions()
    
    def _get_sector(self, symbol: str) -> Optional[str]:
        """Determine sector for a symbol"""
        for sector, stocks in self.sector_correlations.items():
            if symbol in stocks:
                return sector
        return None


def create_weather_commodity_tracker(demo_mode: bool = True) -> WeatherCommodityTracker:
    """Factory function to create WeatherCommodityTracker"""
    return WeatherCommodityTracker(demo_mode=demo_mode)

