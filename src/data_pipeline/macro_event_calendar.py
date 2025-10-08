"""
Macro & Event Calendar Processor
Handles economic calendar events and macro data for regime switching
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum
import requests
import json

logger = logging.getLogger(__name__)

class EventImpact(Enum):
    """Event impact levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class EventType(Enum):
    """Types of economic events"""
    INTEREST_RATE = "interest_rate"
    INFLATION = "inflation"
    EMPLOYMENT = "employment"
    GDP = "gdp"
    TRADE = "trade"
    CENTRAL_BANK = "central_bank"
    EARNINGS = "earnings"
    DIVIDEND = "dividend"
    SPLIT = "split"
    IPO = "ipo"

@dataclass
class EconomicEvent:
    """Economic calendar event"""
    date: datetime
    time: str
    country: str
    event: str
    impact: EventImpact
    event_type: EventType
    previous: Optional[float]
    forecast: Optional[float]
    actual: Optional[float]
    currency: str
    description: str

@dataclass
class MacroIndicator:
    """Macro economic indicator"""
    indicator: str
    value: float
    previous: float
    change: float
    change_pct: float
    date: datetime
    source: str

class MacroEventCalendar:
    """Manages economic calendar and macro events"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.events: List[EconomicEvent] = []
        self.macro_indicators: Dict[str, MacroIndicator] = {}
        self.regime_indicators = {}
        
        # Load static economic calendar
        self._load_static_calendar()
        
        logger.info("Macro Event Calendar initialized")
    
    def _load_static_calendar(self):
        """Load static economic calendar for Canadian events"""
        # Bank of Canada events
        boc_events = [
            {
                'event': 'Bank of Canada Interest Rate Decision',
                'impact': EventImpact.HIGH,
                'event_type': EventType.INTEREST_RATE,
                'country': 'Canada',
                'currency': 'CAD',
                'description': 'BoC monetary policy decision'
            },
            {
                'event': 'Bank of Canada Monetary Policy Report',
                'impact': EventImpact.HIGH,
                'event_type': EventType.CENTRAL_BANK,
                'country': 'Canada',
                'currency': 'CAD',
                'description': 'BoC quarterly economic outlook'
            }
        ]
        
        # Economic indicators
        economic_indicators = [
            {
                'event': 'Consumer Price Index (CPI)',
                'impact': EventImpact.HIGH,
                'event_type': EventType.INFLATION,
                'country': 'Canada',
                'currency': 'CAD',
                'description': 'Monthly inflation data'
            },
            {
                'event': 'Gross Domestic Product (GDP)',
                'impact': EventImpact.HIGH,
                'event_type': EventType.GDP,
                'country': 'Canada',
                'currency': 'CAD',
                'description': 'Quarterly economic growth'
            },
            {
                'event': 'Employment Change',
                'impact': EventImpact.MEDIUM,
                'event_type': EventType.EMPLOYMENT,
                'country': 'Canada',
                'currency': 'CAD',
                'description': 'Monthly employment data'
            },
            {
                'event': 'Unemployment Rate',
                'impact': EventImpact.MEDIUM,
                'event_type': EventType.EMPLOYMENT,
                'country': 'Canada',
                'currency': 'CAD',
                'description': 'Monthly unemployment rate'
            },
            {
                'event': 'Trade Balance',
                'impact': EventImpact.MEDIUM,
                'event_type': EventType.TRADE,
                'country': 'Canada',
                'currency': 'CAD',
                'description': 'Monthly trade data'
            }
        ]
        
        # Generate events for next 3 months
        today = datetime.now()
        for i in range(90):  # 3 months
            event_date = today + timedelta(days=i)
            
            # Add BoC events (typically 8 times per year)
            if i % 45 == 0:  # Roughly every 6 weeks
                for boc_event in boc_events:
                    event = EconomicEvent(
                        date=event_date,
                        time="10:00",
                        country=boc_event['country'],
                        event=boc_event['event'],
                        impact=boc_event['impact'],
                        event_type=boc_event['event_type'],
                        previous=None,
                        forecast=None,
                        actual=None,
                        currency=boc_event['currency'],
                        description=boc_event['description']
                    )
                    self.events.append(event)
            
            # Add economic indicators (monthly)
            if event_date.day == 15:  # Mid-month releases
                for indicator in economic_indicators:
                    event = EconomicEvent(
                        date=event_date,
                        time="08:30",
                        country=indicator['country'],
                        event=indicator['event'],
                        impact=indicator['impact'],
                        event_type=indicator['event_type'],
                        previous=None,
                        forecast=None,
                        actual=None,
                        currency=indicator['currency'],
                        description=indicator['description']
                    )
                    self.events.append(event)
    
    def get_upcoming_events(self, days_ahead: int = 7) -> List[EconomicEvent]:
        """Get upcoming events in the next N days"""
        today = datetime.now()
        end_date = today + timedelta(days=days_ahead)
        
        upcoming = []
        for event in self.events:
            if today <= event.date <= end_date:
                upcoming.append(event)
        
        return sorted(upcoming, key=lambda x: x.date)
    
    def get_high_impact_events(self, days_ahead: int = 7) -> List[EconomicEvent]:
        """Get high impact events in the next N days"""
        upcoming = self.get_upcoming_events(days_ahead)
        return [e for e in upcoming if e.impact in [EventImpact.HIGH, EventImpact.CRITICAL]]
    
    def is_event_day(self, date: datetime = None) -> bool:
        """Check if a given date has high-impact events"""
        if date is None:
            date = datetime.now()
        
        for event in self.events:
            if event.date.date() == date.date() and event.impact in [EventImpact.HIGH, EventImpact.CRITICAL]:
                return True
        
        return False
    
    def get_event_heat_score(self, date: datetime = None) -> float:
        """Calculate event heat score (0-1) for a given date"""
        if date is None:
            date = datetime.now()
        
        heat_score = 0.0
        
        for event in self.events:
            if event.date.date() == date.date():
                if event.impact == EventImpact.CRITICAL:
                    heat_score += 1.0
                elif event.impact == EventImpact.HIGH:
                    heat_score += 0.7
                elif event.impact == EventImpact.MEDIUM:
                    heat_score += 0.4
                elif event.impact == EventImpact.LOW:
                    heat_score += 0.1
        
        return min(heat_score, 1.0)
    
    def update_macro_indicators(self):
        """Update macro economic indicators"""
        try:
            # Simulate macro data updates
            indicators = {
                'CAD_CPI': MacroIndicator(
                    indicator='Consumer Price Index',
                    value=3.2,
                    previous=3.1,
                    change=0.1,
                    change_pct=3.23,
                    date=datetime.now(),
                    source='Statistics Canada'
                ),
                'CAD_GDP': MacroIndicator(
                    indicator='GDP Growth',
                    value=2.1,
                    previous=1.8,
                    change=0.3,
                    change_pct=16.67,
                    date=datetime.now(),
                    source='Statistics Canada'
                ),
                'CAD_UNEMPLOYMENT': MacroIndicator(
                    indicator='Unemployment Rate',
                    value=5.4,
                    previous=5.6,
                    change=-0.2,
                    change_pct=-3.57,
                    date=datetime.now(),
                    source='Statistics Canada'
                ),
                'CAD_INTEREST_RATE': MacroIndicator(
                    indicator='Interest Rate',
                    value=5.0,
                    previous=4.75,
                    change=0.25,
                    change_pct=5.26,
                    date=datetime.now(),
                    source='Bank of Canada'
                )
            }
            
            self.macro_indicators.update(indicators)
            
            logger.info("Macro indicators updated")
            
        except Exception as e:
            logger.error(f"Error updating macro indicators: {e}")
    
    def get_regime_indicators(self) -> Dict[str, float]:
        """Get regime detection indicators"""
        try:
            # Calculate regime indicators based on macro data
            indicators = {}
            
            # Interest rate regime
            if 'CAD_INTEREST_RATE' in self.macro_indicators:
                rate = self.macro_indicators['CAD_INTEREST_RATE'].value
                if rate > 4.0:
                    indicators['rate_regime'] = 1.0  # High rate regime
                elif rate < 2.0:
                    indicators['rate_regime'] = -1.0  # Low rate regime
                else:
                    indicators['rate_regime'] = 0.0  # Neutral
            
            # Inflation regime
            if 'CAD_CPI' in self.macro_indicators:
                cpi = self.macro_indicators['CAD_CPI'].value
                if cpi > 3.0:
                    indicators['inflation_regime'] = 1.0  # High inflation
                elif cpi < 1.0:
                    indicators['inflation_regime'] = -1.0  # Low inflation
                else:
                    indicators['inflation_regime'] = 0.0  # Target range
            
            # Growth regime
            if 'CAD_GDP' in self.macro_indicators:
                gdp = self.macro_indicators['CAD_GDP'].value
                if gdp > 3.0:
                    indicators['growth_regime'] = 1.0  # High growth
                elif gdp < 1.0:
                    indicators['growth_regime'] = -1.0  # Low growth
                else:
                    indicators['growth_regime'] = 0.0  # Moderate growth
            
            # Employment regime
            if 'CAD_UNEMPLOYMENT' in self.macro_indicators:
                unemployment = self.macro_indicators['CAD_UNEMPLOYMENT'].value
                if unemployment < 5.0:
                    indicators['employment_regime'] = 1.0  # Strong employment
                elif unemployment > 7.0:
                    indicators['employment_regime'] = -1.0  # Weak employment
                else:
                    indicators['employment_regime'] = 0.0  # Normal
            
            # Event heat
            indicators['event_heat'] = self.get_event_heat_score()
            
            # Overall regime score
            regime_scores = [v for v in indicators.values() if isinstance(v, (int, float))]
            if regime_scores:
                indicators['overall_regime'] = np.mean(regime_scores)
            else:
                indicators['overall_regime'] = 0.0
            
            self.regime_indicators = indicators
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating regime indicators: {e}")
            return {}
    
    def get_market_regime(self) -> str:
        """Determine current market regime based on macro indicators"""
        try:
            indicators = self.get_regime_indicators()
            
            if not indicators:
                return "neutral"
            
            overall_regime = indicators.get('overall_regime', 0.0)
            event_heat = indicators.get('event_heat', 0.0)
            
            # Determine regime
            if event_heat > 0.7:
                return "high_volatility"
            elif overall_regime > 0.5:
                return "bullish"
            elif overall_regime < -0.5:
                return "bearish"
            else:
                return "neutral"
                
        except Exception as e:
            logger.error(f"Error determining market regime: {e}")
            return "neutral"
    
    def get_trading_recommendations(self) -> Dict[str, str]:
        """Get trading recommendations based on macro events"""
        try:
            recommendations = {}
            
            # Check for upcoming high-impact events
            high_impact_events = self.get_high_impact_events(3)
            
            if high_impact_events:
                recommendations['position_sizing'] = "reduce"
                recommendations['risk_management'] = "increase"
                recommendations['strategy'] = "defensive"
            else:
                recommendations['position_sizing'] = "normal"
                recommendations['risk_management'] = "standard"
                recommendations['strategy'] = "aggressive"
            
            # Check market regime
            regime = self.get_market_regime()
            if regime == "high_volatility":
                recommendations['volatility_handling'] = "increase_hedging"
            elif regime == "bullish":
                recommendations['volatility_handling'] = "reduce_hedging"
            else:
                recommendations['volatility_handling'] = "maintain_hedging"
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting trading recommendations: {e}")
            return {}
    
    def add_custom_event(self, event: EconomicEvent):
        """Add a custom economic event"""
        self.events.append(event)
        logger.info(f"Added custom event: {event.event} on {event.date}")
    
    def get_events_by_type(self, event_type: EventType, days_ahead: int = 30) -> List[EconomicEvent]:
        """Get events of a specific type"""
        today = datetime.now()
        end_date = today + timedelta(days=days_ahead)
        
        events = []
        for event in self.events:
            if (today <= event.date <= end_date and 
                event.event_type == event_type):
                events.append(event)
        
        return sorted(events, key=lambda x: x.date)
    
    def export_calendar(self, filename: str = None) -> str:
        """Export economic calendar to CSV"""
        try:
            if filename is None:
                filename = f"economic_calendar_{datetime.now().strftime('%Y%m%d')}.csv"
            
            # Convert events to DataFrame
            data = []
            for event in self.events:
                data.append({
                    'date': event.date.strftime('%Y-%m-%d'),
                    'time': event.time,
                    'country': event.country,
                    'event': event.event,
                    'impact': event.impact.value,
                    'event_type': event.event_type.value,
                    'currency': event.currency,
                    'description': event.description
                })
            
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
            
            logger.info(f"Economic calendar exported to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error exporting calendar: {e}")
            return ""

class MacroDataManager:
    """Manages macro data collection and processing"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.calendar = MacroEventCalendar(config)
        
        logger.info("Macro Data Manager initialized")
    
    def update_all_data(self):
        """Update all macro data"""
        self.calendar.update_macro_indicators()
        self.calendar.get_regime_indicators()
    
    def get_market_regime(self) -> str:
        """Get current market regime"""
        return self.calendar.get_market_regime()
    
    def get_trading_recommendations(self) -> Dict[str, str]:
        """Get trading recommendations"""
        return self.calendar.get_trading_recommendations()
    
    def is_high_volatility_period(self) -> bool:
        """Check if we're in a high volatility period"""
        return self.calendar.get_market_regime() == "high_volatility"
    
    def get_event_heat_score(self) -> float:
        """Get current event heat score"""
        return self.calendar.get_event_heat_score()
    
    def get_upcoming_events_summary(self, days_ahead: int = 7) -> Dict:
        """Get summary of upcoming events"""
        events = self.calendar.get_upcoming_events(days_ahead)
        high_impact = self.calendar.get_high_impact_events(days_ahead)
        
        return {
            'total_events': len(events),
            'high_impact_events': len(high_impact),
            'event_heat_score': self.calendar.get_event_heat_score(),
            'market_regime': self.calendar.get_market_regime(),
            'recommendations': self.calendar.get_trading_recommendations()
        }
