"""
Event Calendar Module

Tracks and manages important market events:
- Economic data releases (GDP, CPI, employment)
- Central bank meetings (Bank of Canada)
- Earnings announcements
- Holidays and market closures
- Custom events
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class EventType:
    """Event types"""
    ECONOMIC = "economic"
    EARNINGS = "earnings"
    CENTRAL_BANK = "central_bank"
    HOLIDAY = "holiday"
    DIVIDEND = "dividend"
    CUSTOM = "custom"

class EventImportance:
    """Event importance levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class Event:
    """Market event"""
    
    def __init__(
        self,
        event_id: str,
        title: str,
        event_type: str,
        scheduled_time: datetime,
        importance: int = EventImportance.MEDIUM,
        description: str = "",
        symbols_affected: Optional[List[str]] = None,
        metadata: Optional[Dict] = None
    ):
        self.event_id = event_id
        self.title = title
        self.event_type = event_type
        self.scheduled_time = scheduled_time
        self.importance = importance
        self.description = description
        self.symbols_affected = symbols_affected or []
        self.metadata = metadata or {}
        
        self.created_at = datetime.now()
        self.is_active = True
    
    def is_upcoming(self, hours_ahead: int = 24) -> bool:
        """Check if event is upcoming within specified hours"""
        now = datetime.now()
        future_time = now + timedelta(hours=hours_ahead)
        return now <= self.scheduled_time <= future_time
    
    def is_past(self) -> bool:
        """Check if event has passed"""
        return datetime.now() > self.scheduled_time
    
    def time_until(self) -> timedelta:
        """Get time until event"""
        return self.scheduled_time - datetime.now()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'event_id': self.event_id,
            'title': self.title,
            'event_type': self.event_type,
            'scheduled_time': self.scheduled_time.isoformat(),
            'importance': self.importance,
            'description': self.description,
            'symbols_affected': self.symbols_affected,
            'metadata': self.metadata,
            'is_active': self.is_active
        }

class EventCalendar:
    """
    Event calendar for tracking market events
    
    Features:
    - Economic calendar
    - Earnings calendar
    - Holiday calendar
    - Event filtering and querying
    - Event notifications
    """
    
    def __init__(self, calendar_file: str = "data/event_calendar.json"):
        self.calendar_file = Path(calendar_file)
        self.events: Dict[str, Event] = {}
        
        # Create directory if needed
        self.calendar_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing events
        self._load_events()
        
        # Initialize default Canadian market events
        self._initialize_default_events()
        
        logger.info(" Event Calendar initialized")
    
    def add_event(self, event: Event):
        """Add event to calendar"""
        self.events[event.event_id] = event
        self._save_events()
        logger.info(f" Event added: {event.title} on {event.scheduled_time}")
    
    def remove_event(self, event_id: str):
        """Remove event from calendar"""
        if event_id in self.events:
            event = self.events.pop(event_id)
            self._save_events()
            logger.info(f" Event removed: {event.title}")
    
    def get_event(self, event_id: str) -> Optional[Event]:
        """Get event by ID"""
        return self.events.get(event_id)
    
    def get_upcoming_events(
        self,
        hours_ahead: int = 24,
        event_type: Optional[str] = None,
        min_importance: int = EventImportance.LOW
    ) -> List[Event]:
        """Get upcoming events"""
        events = []
        
        for event in self.events.values():
            if not event.is_active:
                continue
            
            if not event.is_upcoming(hours_ahead):
                continue
            
            if event_type and event.event_type != event_type:
                continue
            
            if event.importance < min_importance:
                continue
            
            events.append(event)
        
        # Sort by scheduled time
        events.sort(key=lambda e: e.scheduled_time)
        
        return events
    
    def get_events_today(self) -> List[Event]:
        """Get all events scheduled for today"""
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        today_end = today_start + timedelta(days=1)
        
        events = [
            event for event in self.events.values()
            if event.is_active and today_start <= event.scheduled_time < today_end
        ]
        
        events.sort(key=lambda e: e.scheduled_time)
        return events
    
    def get_events_by_symbol(self, symbol: str) -> List[Event]:
        """Get events affecting a specific symbol"""
        events = [
            event for event in self.events.values()
            if event.is_active and symbol in event.symbols_affected
        ]
        
        events.sort(key=lambda e: e.scheduled_time)
        return events
    
    def get_high_impact_events(self, days_ahead: int = 7) -> List[Event]:
        """Get high impact events in the next N days"""
        return self.get_upcoming_events(
            hours_ahead=days_ahead * 24,
            min_importance=EventImportance.HIGH
        )
    
    def is_market_holiday(self, date: Optional[datetime] = None) -> bool:
        """Check if a date is a market holiday"""
        if date is None:
            date = datetime.now()
        
        date_start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        date_end = date_start + timedelta(days=1)
        
        for event in self.events.values():
            if event.event_type == EventType.HOLIDAY:
                if date_start <= event.scheduled_time < date_end:
                    return True
        
        return False
    
    def cleanup_old_events(self, days_old: int = 30):
        """Remove events older than specified days"""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        old_events = [
            event_id for event_id, event in self.events.items()
            if event.scheduled_time < cutoff_date
        ]
        
        for event_id in old_events:
            self.remove_event(event_id)
        
        logger.info(f" Cleaned up {len(old_events)} old events")
    
    def _initialize_default_events(self):
        """Initialize default Canadian market events"""
        
        # Add Canadian holidays for 2025
        canadian_holidays_2025 = [
            ("2025-01-01", "New Year's Day"),
            ("2025-02-17", "Family Day"),
            ("2025-04-18", "Good Friday"),
            ("2025-05-19", "Victoria Day"),
            ("2025-07-01", "Canada Day"),
            ("2025-08-04", "Civic Holiday"),
            ("2025-09-01", "Labour Day"),
            ("2025-10-13", "Thanksgiving"),
            ("2025-12-25", "Christmas Day"),
            ("2025-12-26", "Boxing Day")
        ]
        
        for date_str, title in canadian_holidays_2025:
            event_id = f"holiday_{date_str}"
            
            # Skip if already exists
            if event_id in self.events:
                continue
            
            event = Event(
                event_id=event_id,
                title=title,
                event_type=EventType.HOLIDAY,
                scheduled_time=datetime.strptime(date_str, "%Y-%m-%d"),
                importance=EventImportance.HIGH,
                description=f"Canadian market holiday: {title}"
            )
            
            self.events[event_id] = event
        
        # Add Bank of Canada interest rate announcements (example dates)
        boc_dates_2025 = [
            ("2025-01-29", "Bank of Canada Interest Rate Decision"),
            ("2025-03-12", "Bank of Canada Interest Rate Decision"),
            ("2025-04-16", "Bank of Canada Interest Rate Decision"),
            ("2025-06-04", "Bank of Canada Interest Rate Decision"),
            ("2025-07-16", "Bank of Canada Interest Rate Decision"),
            ("2025-09-03", "Bank of Canada Interest Rate Decision"),
            ("2025-10-29", "Bank of Canada Interest Rate Decision"),
            ("2025-12-10", "Bank of Canada Interest Rate Decision")
        ]
        
        for date_str, title in boc_dates_2025:
            event_id = f"boc_{date_str}"
            
            if event_id in self.events:
                continue
            
            event = Event(
                event_id=event_id,
                title=title,
                event_type=EventType.CENTRAL_BANK,
                scheduled_time=datetime.strptime(f"{date_str} 10:00", "%Y-%m-%d %H:%M"),
                importance=EventImportance.CRITICAL,
                description="Bank of Canada interest rate announcement - high market impact expected"
            )
            
            self.events[event_id] = event
        
        # Save events
        self._save_events()
        
        logger.info(f" Initialized {len(self.events)} default events")
    
    def _load_events(self):
        """Load events from file"""
        if not self.calendar_file.exists():
            return
        
        try:
            with open(self.calendar_file, 'r') as f:
                data = json.load(f)
            
            for event_data in data:
                event = Event(
                    event_id=event_data['event_id'],
                    title=event_data['title'],
                    event_type=event_data['event_type'],
                    scheduled_time=datetime.fromisoformat(event_data['scheduled_time']),
                    importance=event_data['importance'],
                    description=event_data.get('description', ''),
                    symbols_affected=event_data.get('symbols_affected', []),
                    metadata=event_data.get('metadata', {})
                )
                event.is_active = event_data.get('is_active', True)
                
                self.events[event.event_id] = event
            
            logger.info(f" Loaded {len(self.events)} events from file")
        
        except Exception as e:
            logger.error(f" Failed to load events: {e}")
    
    def _save_events(self):
        """Save events to file"""
        try:
            data = [event.to_dict() for event in self.events.values()]
            
            with open(self.calendar_file, 'w') as f:
                json.dump(data, f, indent=2)
        
        except Exception as e:
            logger.error(f" Failed to save events: {e}")

# Global calendar instance
_calendar_instance = None

def get_event_calendar() -> EventCalendar:
    """Get global event calendar instance"""
    global _calendar_instance
    if _calendar_instance is None:
        _calendar_instance = EventCalendar()
    return _calendar_instance

