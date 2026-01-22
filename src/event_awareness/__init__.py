"""
Event Awareness Package

Tracks market events, volatility, and anomalies
"""

from .event_calendar import (
    EventCalendar,
    Event,
    EventType,
    EventImportance,
    get_event_calendar
)
from .volatility_detector import (
    VolatilityDetector,
    VolatilityRegime,
    get_volatility_detector
)
from .anomaly_detector import (
    AnomalyDetector,
    AnomalyType,
    get_anomaly_detector
)

__all__ = [
    'EventCalendar',
    'Event',
    'EventType',
    'EventImportance',
    'get_event_calendar',
    'VolatilityDetector',
    'VolatilityRegime',
    'get_volatility_detector',
    'AnomalyDetector',
    'AnomalyType',
    'get_anomaly_detector'
]

