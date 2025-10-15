"""
Trade Surveillance System

This module implements comprehensive trade surveillance and anomaly detection
for monitoring suspicious trading activities, market manipulation, and
regulatory compliance violations.

Author: AI Trading System
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from enum import Enum
import json
import sqlite3
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AlertType(Enum):
    """Types of surveillance alerts."""
    WASH_SALE = "WASH_SALE"
    SPOOFING = "SPOOFING"
    LAYERING = "LAYERING"
    FRONT_RUNNING = "FRONT_RUNNING"
    MARKING_THE_CLOSE = "MARKING_THE_CLOSE"
    UNUSUAL_VOLUME = "UNUSUAL_VOLUME"
    PRICE_MANIPULATION = "PRICE_MANIPULATION"
    INSIDER_TRADING = "INSIDER_TRADING"
    CROSS_MARKET_ABUSE = "CROSS_MARKET_ABUSE"
    MOMENTUM_IGNITION = "MOMENTUM_IGNITION"
    STATISTICAL_ARBITRAGE = "STATISTICAL_ARBITRAGE"
    ANOMALOUS_PATTERN = "ANOMALOUS_PATTERN"

class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class AlertStatus(Enum):
    """Alert status."""
    NEW = "NEW"
    INVESTIGATING = "INVESTIGATING"
    RESOLVED = "RESOLVED"
    FALSE_POSITIVE = "FALSE_POSITIVE"
    ESCALATED = "ESCALATED"

@dataclass
class SurveillanceAlert:
    """Trade surveillance alert."""
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    status: AlertStatus
    timestamp: datetime
    symbol: str
    description: str
    confidence_score: float
    suspicious_activities: List[Dict[str, Any]]
    affected_orders: List[str]
    affected_trades: List[str]
    market_impact: Dict[str, float]
    regulatory_relevance: bool
    investigation_notes: Optional[str] = None
    resolution_notes: Optional[str] = None
    escalated_to: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class AnomalyDetector:
    """Anomaly detection configuration."""
    detector_id: str
    name: str
    alert_type: AlertType
    parameters: Dict[str, Any]
    thresholds: Dict[str, float]
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)

class TradeSurveillance:
    """
    Comprehensive trade surveillance system with anomaly detection.
    
    Features:
    - Real-time trade monitoring and analysis
    - Advanced anomaly detection algorithms
    - Market manipulation pattern recognition
    - Regulatory compliance monitoring
    - Alert generation and management
    - Investigation workflow support
    """
    
    def __init__(self, db_path: str = "data/trade_surveillance.db"):
        """
        Initialize trade surveillance system.
        
        Args:
            db_path: Path to surveillance database
        """
        self.db_path = db_path
        self.alerts: List[SurveillanceAlert] = []
        self.anomaly_detectors: Dict[str, AnomalyDetector] = {}
        self.trade_data_buffer: List[Dict[str, Any]] = []
        self.order_data_buffer: List[Dict[str, Any]] = []
        
        # Surveillance parameters
        self.surveillance_params = {
            'wash_sale_window': 30,  # days
            'unusual_volume_threshold': 3.0,  # standard deviations
            'price_impact_threshold': 0.05,  # 5%
            'spoofing_detection_window': 60,  # seconds
            'layering_detection_threshold': 5,  # orders
            'anomaly_confidence_threshold': 0.7
        }
        
        # Initialize database
        self._init_database()
        
        # Load default anomaly detectors
        self._load_default_detectors()
        
        logger.info("Trade Surveillance system initialized")
    
    def _init_database(self) -> None:
        """Initialize surveillance database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create alerts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS surveillance_alerts (
                alert_id TEXT PRIMARY KEY,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                status TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                description TEXT NOT NULL,
                confidence_score REAL NOT NULL,
                suspicious_activities TEXT,
                affected_orders TEXT,
                affected_trades TEXT,
                market_impact TEXT,
                regulatory_relevance INTEGER,
                investigation_notes TEXT,
                resolution_notes TEXT,
                escalated_to TEXT,
                created_at TEXT NOT NULL
            )
        """)
        
        # Create anomaly detectors table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS anomaly_detectors (
                detector_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                parameters TEXT,
                thresholds TEXT,
                is_active INTEGER,
                created_at TEXT
            )
        """)
        
        # Create trade data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade_data (
                trade_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                price REAL NOT NULL,
                quantity INTEGER NOT NULL,
                side TEXT NOT NULL,
                order_type TEXT,
                execution_venue TEXT,
                market_impact REAL,
                created_at TEXT
            )
        """)
        
        # Create order data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS order_data (
                order_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                price REAL,
                quantity INTEGER NOT NULL,
                side TEXT NOT NULL,
                order_type TEXT NOT NULL,
                time_in_force TEXT,
                status TEXT,
                venue TEXT,
                created_at TEXT
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON surveillance_alerts(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_symbol ON surveillance_alerts(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_type ON surveillance_alerts(alert_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trade_data(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trade_data(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_timestamp ON order_data(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_symbol ON order_data(symbol)")
        
        conn.commit()
        conn.close()
    
    def _load_default_detectors(self) -> None:
        """Load default anomaly detectors."""
        default_detectors = [
            AnomalyDetector(
                detector_id="DETECTOR_001",
                name="Wash Sale Detector",
                alert_type=AlertType.WASH_SALE,
                parameters={
                    'window_days': 30,
                    'loss_threshold': 0.01,
                    'same_security': True
                },
                thresholds={
                    'confidence_threshold': 0.8,
                    'volume_threshold': 1000
                }
            ),
            AnomalyDetector(
                detector_id="DETECTOR_002",
                name="Spoofing Detector",
                alert_type=AlertType.SPOOFING,
                parameters={
                    'detection_window': 60,
                    'cancellation_rate': 0.8,
                    'price_improvement': 0.001
                },
                thresholds={
                    'confidence_threshold': 0.7,
                    'order_count_threshold': 10
                }
            ),
            AnomalyDetector(
                detector_id="DETECTOR_003",
                name="Layering Detector",
                alert_type=AlertType.LAYERING,
                parameters={
                    'detection_window': 300,
                    'order_count_threshold': 5,
                    'price_increment': 0.01
                },
                thresholds={
                    'confidence_threshold': 0.75,
                    'volume_threshold': 5000
                }
            ),
            AnomalyDetector(
                detector_id="DETECTOR_004",
                name="Unusual Volume Detector",
                alert_type=AlertType.UNUSUAL_VOLUME,
                parameters={
                    'lookback_days': 30,
                    'std_dev_multiplier': 3.0,
                    'min_volume': 10000
                },
                thresholds={
                    'confidence_threshold': 0.6,
                    'volume_threshold': 2.0
                }
            ),
            AnomalyDetector(
                detector_id="DETECTOR_005",
                name="Price Manipulation Detector",
                alert_type=AlertType.PRICE_MANIPULATION,
                parameters={
                    'detection_window': 300,
                    'price_change_threshold': 0.05,
                    'volume_ratio_threshold': 2.0
                },
                thresholds={
                    'confidence_threshold': 0.8,
                    'impact_threshold': 0.03
                }
            )
        ]
        
        for detector in default_detectors:
            self.add_anomaly_detector(detector)
    
    def add_anomaly_detector(self, detector: AnomalyDetector) -> None:
        """
        Add a new anomaly detector.
        
        Args:
            detector: Anomaly detector configuration
        """
        self.anomaly_detectors[detector.detector_id] = detector
        
        # Save to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO anomaly_detectors 
            (detector_id, name, alert_type, parameters, thresholds, is_active, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            detector.detector_id, detector.name, detector.alert_type.value,
            json.dumps(detector.parameters), json.dumps(detector.thresholds),
            detector.is_active, detector.created_at.isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Added anomaly detector: {detector.detector_id} - {detector.name}")
    
    def process_trade(self, trade_data: Dict[str, Any]) -> List[SurveillanceAlert]:
        """
        Process a trade for surveillance analysis.
        
        Args:
            trade_data: Trade execution data
            
        Returns:
            List of generated alerts
        """
        # Store trade data
        self._store_trade_data(trade_data)
        self.trade_data_buffer.append(trade_data)
        
        # Run surveillance checks
        alerts = []
        
        for detector_id, detector in self.anomaly_detectors.items():
            if not detector.is_active:
                continue
            
            # Run specific detector
            detector_alerts = self._run_detector(detector, trade_data)
            alerts.extend(detector_alerts)
        
        # Store alerts
        for alert in alerts:
            self._store_alert(alert)
        
        return alerts
    
    def process_order(self, order_data: Dict[str, Any]) -> List[SurveillanceAlert]:
        """
        Process an order for surveillance analysis.
        
        Args:
            order_data: Order data
            
        Returns:
            List of generated alerts
        """
        # Store order data
        self._store_order_data(order_data)
        self.order_data_buffer.append(order_data)
        
        # Run surveillance checks
        alerts = []
        
        for detector_id, detector in self.anomaly_detectors.items():
            if not detector.is_active:
                continue
            
            # Run specific detector
            detector_alerts = self._run_detector(detector, order_data)
            alerts.extend(detector_alerts)
        
        # Store alerts
        for alert in alerts:
            self._store_alert(alert)
        
        return alerts
    
    def _run_detector(self, detector: AnomalyDetector, data: Dict[str, Any]) -> List[SurveillanceAlert]:
        """Run a specific anomaly detector."""
        alerts = []
        
        try:
            if detector.alert_type == AlertType.WASH_SALE:
                alerts.extend(self._detect_wash_sales(detector, data))
            elif detector.alert_type == AlertType.SPOOFING:
                alerts.extend(self._detect_spoofing(detector, data))
            elif detector.alert_type == AlertType.LAYERING:
                alerts.extend(self._detect_layering(detector, data))
            elif detector.alert_type == AlertType.UNUSUAL_VOLUME:
                alerts.extend(self._detect_unusual_volume(detector, data))
            elif detector.alert_type == AlertType.PRICE_MANIPULATION:
                alerts.extend(self._detect_price_manipulation(detector, data))
        except Exception as e:
            logger.error(f"Error running detector {detector.detector_id}: {e}")
        
        return alerts
    
    def _detect_wash_sales(self, detector: AnomalyDetector, data: Dict[str, Any]) -> List[SurveillanceAlert]:
        """Detect wash sale violations."""
        alerts = []
        
        symbol = data.get('symbol')
        trade_type = data.get('side', 'BUY')
        quantity = data.get('quantity', 0)
        
        if trade_type == 'BUY' and symbol:
            # Check for recent sales of same security
            recent_sales = self._get_recent_sales(symbol, detector.parameters.get('window_days', 30))
            
            for sale in recent_sales:
                if (sale['quantity'] == quantity and 
                    sale['loss'] > detector.parameters.get('loss_threshold', 0.01)):
                    
                    confidence = min(0.95, 0.5 + (sale['loss'] * 10))
                    
                    if confidence >= detector.thresholds.get('confidence_threshold', 0.8):
                        alert = SurveillanceAlert(
                            alert_id=f"WASH_SALE_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                            alert_type=AlertType.WASH_SALE,
                            severity=AlertSeverity.HIGH,
                            status=AlertStatus.NEW,
                            timestamp=datetime.now(),
                            symbol=symbol,
                            description=f"Wash sale detected: buying {quantity} shares within {detector.parameters.get('window_days')} days of loss sale",
                            confidence_score=confidence,
                            suspicious_activities=[{
                                'type': 'wash_sale',
                                'recent_sale': sale,
                                'current_buy': data
                            }],
                            affected_orders=[data.get('order_id', '')],
                            affected_trades=[sale.get('trade_id', ''), data.get('trade_id', '')],
                            market_impact={'price_impact': 0.0, 'volume_impact': 0.0},
                            regulatory_relevance=True
                        )
                        alerts.append(alert)
        
        return alerts
    
    def _detect_spoofing(self, detector: AnomalyDetector, data: Dict[str, Any]) -> List[SurveillanceAlert]:
        """Detect spoofing patterns."""
        alerts = []
        
        symbol = data.get('symbol')
        order_type = data.get('order_type', 'LIMIT')
        
        if order_type == 'LIMIT' and symbol:
            # Check for spoofing patterns in recent orders
            recent_orders = self._get_recent_orders(symbol, detector.parameters.get('detection_window', 60))
            
            # Analyze cancellation rate and price improvement
            cancellations = [o for o in recent_orders if o.get('status') == 'CANCELLED']
            total_orders = len(recent_orders)
            
            if total_orders > 0:
                cancellation_rate = len(cancellations) / total_orders
                
                if cancellation_rate >= detector.parameters.get('cancellation_rate', 0.8):
                    # Check for price improvement
                    price_improvement = self._calculate_price_improvement(recent_orders)
                    
                    if price_improvement >= detector.parameters.get('price_improvement', 0.001):
                        confidence = min(0.95, cancellation_rate + price_improvement * 100)
                        
                        if confidence >= detector.thresholds.get('confidence_threshold', 0.7):
                            alert = SurveillanceAlert(
                                alert_id=f"SPOOFING_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                                alert_type=AlertType.SPOOFING,
                                severity=AlertSeverity.CRITICAL,
                                status=AlertStatus.NEW,
                                timestamp=datetime.now(),
                                symbol=symbol,
                                description=f"Spoofing detected: {cancellation_rate:.1%} cancellation rate with price improvement",
                                confidence_score=confidence,
                                suspicious_activities=[{
                                    'type': 'spoofing',
                                    'cancellation_rate': cancellation_rate,
                                    'price_improvement': price_improvement,
                                    'recent_orders': recent_orders
                                }],
                                affected_orders=[o.get('order_id', '') for o in recent_orders],
                                affected_trades=[],
                                market_impact={'price_impact': price_improvement, 'volume_impact': 0.0},
                                regulatory_relevance=True
                            )
                            alerts.append(alert)
        
        return alerts
    
    def _detect_layering(self, detector: AnomalyDetector, data: Dict[str, Any]) -> List[SurveillanceAlert]:
        """Detect layering patterns."""
        alerts = []
        
        symbol = data.get('symbol')
        
        if symbol:
            # Check for layering patterns in recent orders
            recent_orders = self._get_recent_orders(symbol, detector.parameters.get('detection_window', 300))
            
            # Group orders by side and analyze patterns
            buy_orders = [o for o in recent_orders if o.get('side') == 'BUY']
            sell_orders = [o for o in recent_orders if o.get('side') == 'SELL']
            
            # Check for layering on one side
            for side, orders in [('BUY', buy_orders), ('SELL', sell_orders)]:
                if len(orders) >= detector.parameters.get('order_count_threshold', 5):
                    # Analyze price progression
                    prices = [o.get('price', 0) for o in orders if o.get('price')]
                    
                    if len(prices) >= 3:
                        # Check for consistent price increments
                        price_increments = [prices[i+1] - prices[i] for i in range(len(prices)-1)]
                        increment_consistency = self._calculate_increment_consistency(price_increments)
                        
                        if increment_consistency >= 0.8:  # 80% consistency
                            confidence = min(0.95, len(orders) / 10 + increment_consistency)
                            
                            if confidence >= detector.thresholds.get('confidence_threshold', 0.75):
                                alert = SurveillanceAlert(
                                    alert_id=f"LAYERING_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                                    alert_type=AlertType.LAYERING,
                                    severity=AlertSeverity.HIGH,
                                    status=AlertStatus.NEW,
                                    timestamp=datetime.now(),
                                    symbol=symbol,
                                    description=f"Layering detected: {len(orders)} {side} orders with consistent price increments",
                                    confidence_score=confidence,
                                    suspicious_activities=[{
                                        'type': 'layering',
                                        'side': side,
                                        'order_count': len(orders),
                                        'price_increments': price_increments,
                                        'consistency': increment_consistency
                                    }],
                                    affected_orders=[o.get('order_id', '') for o in orders],
                                    affected_trades=[],
                                    market_impact={'price_impact': 0.0, 'volume_impact': 0.0},
                                    regulatory_relevance=True
                                )
                                alerts.append(alert)
        
        return alerts
    
    def _detect_unusual_volume(self, detector: AnomalyDetector, data: Dict[str, Any]) -> List[SurveillanceAlert]:
        """Detect unusual volume patterns."""
        alerts = []
        
        symbol = data.get('symbol')
        volume = data.get('quantity', 0)
        
        if symbol and volume > 0:
            # Get historical volume data
            historical_volumes = self._get_historical_volumes(symbol, detector.parameters.get('lookback_days', 30))
            
            if len(historical_volumes) >= 10:  # Need sufficient data
                # Calculate volume statistics
                mean_volume = np.mean(historical_volumes)
                std_volume = np.std(historical_volumes)
                
                if std_volume > 0:
                    # Calculate z-score
                    z_score = (volume - mean_volume) / std_volume
                    std_multiplier = detector.parameters.get('std_dev_multiplier', 3.0)
                    
                    if z_score >= std_multiplier:
                        confidence = min(0.95, (z_score - std_multiplier) / std_multiplier)
                        
                        if confidence >= detector.thresholds.get('confidence_threshold', 0.6):
                            alert = SurveillanceAlert(
                                alert_id=f"UNUSUAL_VOLUME_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                                alert_type=AlertType.UNUSUAL_VOLUME,
                                severity=AlertSeverity.MEDIUM,
                                status=AlertStatus.NEW,
                                timestamp=datetime.now(),
                                symbol=symbol,
                                description=f"Unusual volume detected: {volume:,} shares ({z_score:.1f} std devs above mean)",
                                confidence_score=confidence,
                                suspicious_activities=[{
                                    'type': 'unusual_volume',
                                    'current_volume': volume,
                                    'mean_volume': mean_volume,
                                    'std_volume': std_volume,
                                    'z_score': z_score
                                }],
                                affected_orders=[data.get('order_id', '')],
                                affected_trades=[data.get('trade_id', '')],
                                market_impact={'price_impact': 0.0, 'volume_impact': z_score},
                                regulatory_relevance=False
                            )
                            alerts.append(alert)
        
        return alerts
    
    def _detect_price_manipulation(self, detector: AnomalyDetector, data: Dict[str, Any]) -> List[SurveillanceAlert]:
        """Detect price manipulation patterns."""
        alerts = []
        
        symbol = data.get('symbol')
        price = data.get('price', 0)
        volume = data.get('quantity', 0)
        
        if symbol and price > 0 and volume > 0:
            # Get recent price and volume data
            recent_data = self._get_recent_market_data(symbol, detector.parameters.get('detection_window', 300))
            
            if len(recent_data) >= 5:
                # Calculate price change and volume ratio
                price_changes = [d['price_change'] for d in recent_data if 'price_change' in d]
                volume_ratios = [d['volume_ratio'] for d in recent_data if 'volume_ratio' in d]
                
                if price_changes and volume_ratios:
                    max_price_change = max(price_changes)
                    max_volume_ratio = max(volume_ratios)
                    
                    price_threshold = detector.parameters.get('price_change_threshold', 0.05)
                    volume_threshold = detector.parameters.get('volume_ratio_threshold', 2.0)
                    
                    if (max_price_change >= price_threshold and 
                        max_volume_ratio >= volume_threshold):
                        
                        confidence = min(0.95, (max_price_change / price_threshold + max_volume_ratio / volume_threshold) / 2)
                        
                        if confidence >= detector.thresholds.get('confidence_threshold', 0.8):
                            alert = SurveillanceAlert(
                                alert_id=f"PRICE_MANIP_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                                alert_type=AlertType.PRICE_MANIPULATION,
                                severity=AlertSeverity.CRITICAL,
                                status=AlertStatus.NEW,
                                timestamp=datetime.now(),
                                symbol=symbol,
                                description=f"Price manipulation detected: {max_price_change:.1%} price change with {max_volume_ratio:.1f}x volume",
                                confidence_score=confidence,
                                suspicious_activities=[{
                                    'type': 'price_manipulation',
                                    'price_change': max_price_change,
                                    'volume_ratio': max_volume_ratio,
                                    'recent_data': recent_data
                                }],
                                affected_orders=[data.get('order_id', '')],
                                affected_trades=[data.get('trade_id', '')],
                                market_impact={'price_impact': max_price_change, 'volume_impact': max_volume_ratio},
                                regulatory_relevance=True
                            )
                            alerts.append(alert)
        
        return alerts
    
    def _get_recent_sales(self, symbol: str, days: int) -> List[Dict[str, Any]]:
        """Get recent sales for wash sale detection."""
        # In a real implementation, this would query the database
        return []
    
    def _get_recent_orders(self, symbol: str, seconds: int) -> List[Dict[str, Any]]:
        """Get recent orders for pattern detection."""
        # In a real implementation, this would query the database
        return []
    
    def _calculate_price_improvement(self, orders: List[Dict[str, Any]]) -> float:
        """Calculate price improvement from orders."""
        # Simplified calculation
        return 0.0
    
    def _calculate_increment_consistency(self, increments: List[float]) -> float:
        """Calculate consistency of price increments."""
        if not increments:
            return 0.0
        
        # Calculate coefficient of variation
        mean_inc = np.mean(increments)
        std_inc = np.std(increments)
        
        if mean_inc == 0:
            return 0.0
        
        cv = std_inc / abs(mean_inc)
        return max(0.0, 1.0 - cv)
    
    def _get_historical_volumes(self, symbol: str, days: int) -> List[float]:
        """Get historical volume data."""
        # In a real implementation, this would query the database
        return []
    
    def _get_recent_market_data(self, symbol: str, seconds: int) -> List[Dict[str, Any]]:
        """Get recent market data for analysis."""
        # In a real implementation, this would query the database
        return []
    
    def _store_trade_data(self, trade_data: Dict[str, Any]) -> None:
        """Store trade data in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO trade_data 
            (trade_id, symbol, timestamp, price, quantity, side, order_type, execution_venue, market_impact, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade_data.get('trade_id', ''),
            trade_data.get('symbol', ''),
            trade_data.get('timestamp', datetime.now().isoformat()),
            trade_data.get('price', 0),
            trade_data.get('quantity', 0),
            trade_data.get('side', ''),
            trade_data.get('order_type', ''),
            trade_data.get('execution_venue', ''),
            trade_data.get('market_impact', 0),
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def _store_order_data(self, order_data: Dict[str, Any]) -> None:
        """Store order data in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO order_data 
            (order_id, symbol, timestamp, price, quantity, side, order_type, time_in_force, status, venue, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            order_data.get('order_id', ''),
            order_data.get('symbol', ''),
            order_data.get('timestamp', datetime.now().isoformat()),
            order_data.get('price', 0),
            order_data.get('quantity', 0),
            order_data.get('side', ''),
            order_data.get('order_type', ''),
            order_data.get('time_in_force', ''),
            order_data.get('status', ''),
            order_data.get('venue', ''),
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def _store_alert(self, alert: SurveillanceAlert) -> None:
        """Store surveillance alert in database."""
        self.alerts.append(alert)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO surveillance_alerts 
            (alert_id, alert_type, severity, status, timestamp, symbol, description, confidence_score,
             suspicious_activities, affected_orders, affected_trades, market_impact, regulatory_relevance,
             investigation_notes, resolution_notes, escalated_to, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            alert.alert_id, alert.alert_type.value, alert.severity.value, alert.status.value,
            alert.timestamp.isoformat(), alert.symbol, alert.description, alert.confidence_score,
            json.dumps(alert.suspicious_activities), json.dumps(alert.affected_orders),
            json.dumps(alert.affected_trades), json.dumps(alert.market_impact),
            alert.regulatory_relevance, alert.investigation_notes, alert.resolution_notes,
            alert.escalated_to, alert.created_at.isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        logger.warning(f"Surveillance alert generated: {alert.alert_id} - {alert.description}")
    
    def get_alerts(self, start_date: datetime = None, end_date: datetime = None,
                   alert_type: AlertType = None, severity: AlertSeverity = None,
                   status: AlertStatus = None, symbol: str = None) -> List[SurveillanceAlert]:
        """
        Get surveillance alerts with filters.
        
        Args:
            start_date: Filter by start date
            end_date: Filter by end date
            alert_type: Filter by alert type
            severity: Filter by severity
            status: Filter by status
            symbol: Filter by symbol
            
        Returns:
            List of matching alerts
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM surveillance_alerts WHERE 1=1"
        params = []
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())
        
        if alert_type:
            query += " AND alert_type = ?"
            params.append(alert_type.value)
        
        if severity:
            query += " AND severity = ?"
            params.append(severity.value)
        
        if status:
            query += " AND status = ?"
            params.append(status.value)
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        
        query += " ORDER BY timestamp DESC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        # Convert rows to SurveillanceAlert objects
        alerts = []
        for row in rows:
            alert = self._row_to_alert(row)
            alerts.append(alert)
        
        return alerts
    
    def _row_to_alert(self, row: Tuple) -> SurveillanceAlert:
        """Convert database row to SurveillanceAlert object."""
        return SurveillanceAlert(
            alert_id=row[0],
            alert_type=AlertType(row[1]),
            severity=AlertSeverity(row[2]),
            status=AlertStatus(row[3]),
            timestamp=datetime.fromisoformat(row[4]),
            symbol=row[5],
            description=row[6],
            confidence_score=row[7],
            suspicious_activities=json.loads(row[8]) if row[8] else [],
            affected_orders=json.loads(row[9]) if row[9] else [],
            affected_trades=json.loads(row[10]) if row[10] else [],
            market_impact=json.loads(row[11]) if row[11] else {},
            regulatory_relevance=bool(row[12]),
            investigation_notes=row[13],
            resolution_notes=row[14],
            escalated_to=row[15],
            created_at=datetime.fromisoformat(row[16])
        )
    
    def update_alert_status(self, alert_id: str, status: AlertStatus, 
                           notes: str = None, escalated_to: str = None) -> bool:
        """
        Update alert status.
        
        Args:
            alert_id: Alert ID
            status: New status
            notes: Optional notes
            escalated_to: Optional escalation target
            
        Returns:
            True if updated successfully
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE surveillance_alerts
            SET status = ?, investigation_notes = ?, escalated_to = ?
            WHERE alert_id = ?
        """, (status.value, notes, escalated_to, alert_id))
        
        updated = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        if updated:
            logger.info(f"Alert status updated: {alert_id} -> {status.value}")
        
        return updated
    
    def get_surveillance_summary(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Get surveillance summary for a period.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            Surveillance summary dictionary
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get alert counts by type
        cursor.execute("""
            SELECT alert_type, COUNT(*) as count
            FROM surveillance_alerts
            WHERE timestamp BETWEEN ? AND ?
            GROUP BY alert_type
        """, (start_date.isoformat(), end_date.isoformat()))
        
        alert_type_counts = dict(cursor.fetchall())
        
        # Get alert counts by severity
        cursor.execute("""
            SELECT severity, COUNT(*) as count
            FROM surveillance_alerts
            WHERE timestamp BETWEEN ? AND ?
            GROUP BY severity
        """, (start_date.isoformat(), end_date.isoformat()))
        
        severity_counts = dict(cursor.fetchall())
        
        # Get alert counts by status
        cursor.execute("""
            SELECT status, COUNT(*) as count
            FROM surveillance_alerts
            WHERE timestamp BETWEEN ? AND ?
            GROUP BY status
        """, (start_date.isoformat(), end_date.isoformat()))
        
        status_counts = dict(cursor.fetchall())
        
        # Get regulatory relevance
        cursor.execute("""
            SELECT COUNT(*) as total, COUNT(CASE WHEN regulatory_relevance = 1 THEN 1 END) as regulatory
            FROM surveillance_alerts
            WHERE timestamp BETWEEN ? AND ?
        """, (start_date.isoformat(), end_date.isoformat()))
        
        total_alerts, regulatory_alerts = cursor.fetchone()
        
        conn.close()
        
        return {
            'summary': {
                'total_alerts': total_alerts or 0,
                'regulatory_alerts': regulatory_alerts or 0,
                'regulatory_percentage': (regulatory_alerts / total_alerts * 100) if total_alerts > 0 else 0
            },
            'alert_type_breakdown': alert_type_counts,
            'severity_breakdown': severity_counts,
            'status_breakdown': status_counts,
            'period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            },
            'generated_at': datetime.now().isoformat()
        }
