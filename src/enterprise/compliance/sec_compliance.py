"""
SEC Compliance Automation System

This module implements comprehensive SEC rule compliance automation including
rule validation, violation detection, reporting, and enforcement mechanisms.

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
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class RuleSeverity(Enum):
    """SEC rule violation severity levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class RuleStatus(Enum):
    """SEC rule status."""
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    SUSPENDED = "SUSPENDED"
    PENDING = "PENDING"

@dataclass
class SECRule:
    """SEC rule definition with validation logic."""
    rule_id: str
    name: str
    description: str
    category: str
    severity: RuleSeverity
    status: RuleStatus
    effective_date: datetime
    validation_logic: Dict[str, Any]
    enforcement_actions: List[str]
    reporting_requirements: List[str]
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class ComplianceViolation:
    """SEC compliance violation record."""
    violation_id: str
    rule_id: str
    rule_name: str
    severity: RuleSeverity
    violation_type: str
    description: str
    affected_entities: List[str]
    violation_data: Dict[str, Any]
    detected_at: datetime
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None
    enforcement_action: Optional[str] = None
    reported_to_sec: bool = False
    created_at: datetime = field(default_factory=datetime.now)

class SECCompliance:
    """
    Comprehensive SEC compliance automation system.
    
    Features:
    - Real-time SEC rule validation
    - Automated violation detection and reporting
    - Compliance monitoring and alerting
    - Regulatory reporting automation
    - Enforcement action tracking
    - Historical compliance analysis
    """
    
    def __init__(self, db_path: str = "data/compliance.db"):
        """
        Initialize SEC compliance system.
        
        Args:
            db_path: Path to compliance database
        """
        self.db_path = db_path
        self.rules: Dict[str, SECRule] = {}
        self.violations: List[ComplianceViolation] = []
        
        # Compliance thresholds
        self.compliance_thresholds = {
            RuleSeverity.CRITICAL: 0,  # Zero tolerance
            RuleSeverity.HIGH: 1,      # 1 violation per day
            RuleSeverity.MEDIUM: 5,    # 5 violations per day
            RuleSeverity.LOW: 10       # 10 violations per day
        }
        
        # Initialize database
        self._init_database()
        
        # Load default SEC rules
        self._load_default_rules()
        
        logger.info("SEC Compliance system initialized")
    
    def _init_database(self) -> None:
        """Initialize compliance database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create rules table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sec_rules (
                rule_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                category TEXT,
                severity TEXT,
                status TEXT,
                effective_date TEXT,
                validation_logic TEXT,
                enforcement_actions TEXT,
                reporting_requirements TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)
        
        # Create violations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS compliance_violations (
                violation_id TEXT PRIMARY KEY,
                rule_id TEXT,
                rule_name TEXT,
                severity TEXT,
                violation_type TEXT,
                description TEXT,
                affected_entities TEXT,
                violation_data TEXT,
                detected_at TEXT,
                resolved_at TEXT,
                resolution_notes TEXT,
                enforcement_action TEXT,
                reported_to_sec INTEGER,
                created_at TEXT,
                FOREIGN KEY (rule_id) REFERENCES sec_rules (rule_id)
            )
        """)
        
        # Create compliance metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS compliance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                total_violations INTEGER,
                critical_violations INTEGER,
                high_violations INTEGER,
                medium_violations INTEGER,
                low_violations INTEGER,
                resolved_violations INTEGER,
                compliance_score REAL,
                created_at TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _load_default_rules(self) -> None:
        """Load default SEC rules for trading compliance."""
        default_rules = [
            SECRule(
                rule_id="SEC_RULE_001",
                name="Position Size Limits",
                description="Enforce maximum position size limits per SEC regulations",
                category="POSITION_LIMITS",
                severity=RuleSeverity.HIGH,
                status=RuleStatus.ACTIVE,
                effective_date=datetime(2020, 1, 1),
                validation_logic={
                    "max_position_size": 0.05,  # 5% of portfolio
                    "max_sector_exposure": 0.20,  # 20% per sector
                    "max_single_stock": 0.10  # 10% per stock
                },
                enforcement_actions=["REJECT_ORDER", "REDUCE_POSITION", "ALERT_COMPLIANCE"],
                reporting_requirements=["DAILY_POSITION_REPORT", "VIOLATION_REPORT"]
            ),
            SECRule(
                rule_id="SEC_RULE_002",
                name="Wash Sale Prevention",
                description="Prevent wash sale violations per IRS and SEC rules",
                category="WASH_SALES",
                severity=RuleSeverity.CRITICAL,
                status=RuleStatus.ACTIVE,
                effective_date=datetime(2020, 1, 1),
                validation_logic={
                    "wash_sale_window": 30,  # 30 days
                    "loss_threshold": 0.01,  # 1% loss
                    "same_security": True
                },
                enforcement_actions=["REJECT_ORDER", "ALERT_COMPLIANCE", "REPORT_TO_IRS"],
                reporting_requirements=["WASH_SALE_REPORT", "TAX_REPORTING"]
            ),
            SECRule(
                rule_id="SEC_RULE_003",
                name="Market Manipulation Prevention",
                description="Prevent market manipulation and spoofing",
                category="MARKET_MANIPULATION",
                severity=RuleSeverity.CRITICAL,
                status=RuleStatus.ACTIVE,
                effective_date=datetime(2020, 1, 1),
                validation_logic={
                    "max_order_size": 1000000,  # $1M per order
                    "max_daily_volume": 0.10,  # 10% of daily volume
                    "spoofing_detection": True,
                    "layering_detection": True
                },
                enforcement_actions=["REJECT_ORDER", "SUSPEND_TRADING", "REPORT_TO_SEC"],
                reporting_requirements=["MANIPULATION_REPORT", "SUSPICIOUS_ACTIVITY_REPORT"]
            ),
            SECRule(
                rule_id="SEC_RULE_004",
                name="Insider Trading Prevention",
                description="Prevent insider trading violations",
                category="INSIDER_TRADING",
                severity=RuleSeverity.CRITICAL,
                status=RuleStatus.ACTIVE,
                effective_date=datetime(2020, 1, 1),
                validation_logic={
                    "restricted_securities": True,
                    "blackout_periods": True,
                    "insider_list_check": True,
                    "material_nonpublic_info": True
                },
                enforcement_actions=["REJECT_ORDER", "ALERT_COMPLIANCE", "REPORT_TO_SEC"],
                reporting_requirements=["INSIDER_TRADING_REPORT", "COMPLIANCE_REPORT"]
            ),
            SECRule(
                rule_id="SEC_RULE_005",
                name="Best Execution",
                description="Ensure best execution for all trades",
                category="BEST_EXECUTION",
                severity=RuleSeverity.HIGH,
                status=RuleStatus.ACTIVE,
                effective_date=datetime(2020, 1, 1),
                validation_logic={
                    "price_improvement": 0.001,  # 0.1% price improvement
                    "execution_quality": True,
                    "market_impact": 0.005,  # 0.5% max market impact
                    "timing_analysis": True
                },
                enforcement_actions=["ROUTE_TO_BETTER_VENUE", "ALERT_COMPLIANCE"],
                reporting_requirements=["BEST_EXECUTION_REPORT", "EXECUTION_QUALITY_REPORT"]
            ),
            SECRule(
                rule_id="SEC_RULE_006",
                name="Anti-Money Laundering",
                description="AML compliance and suspicious activity monitoring",
                category="AML",
                severity=RuleSeverity.HIGH,
                status=RuleStatus.ACTIVE,
                effective_date=datetime(2020, 1, 1),
                validation_logic={
                    "suspicious_amount": 10000,  # $10K threshold
                    "unusual_patterns": True,
                    "sanctions_screening": True,
                    "pep_screening": True
                },
                enforcement_actions=["FLAG_TRANSACTION", "REPORT_TO_FINRA", "FREEZE_ACCOUNT"],
                reporting_requirements=["SAR_REPORT", "AML_REPORT"]
            )
        ]
        
        for rule in default_rules:
            self.add_rule(rule)
    
    def add_rule(self, rule: SECRule) -> None:
        """
        Add a new SEC rule to the compliance system.
        
        Args:
            rule: SEC rule definition
        """
        self.rules[rule.rule_id] = rule
        
        # Save to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO sec_rules 
            (rule_id, name, description, category, severity, status, effective_date,
             validation_logic, enforcement_actions, reporting_requirements, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            rule.rule_id, rule.name, rule.description, rule.category,
            rule.severity.value, rule.status.value, rule.effective_date.isoformat(),
            json.dumps(rule.validation_logic), json.dumps(rule.enforcement_actions),
            json.dumps(rule.reporting_requirements), rule.created_at.isoformat(),
            rule.updated_at.isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Added SEC rule: {rule.rule_id} - {rule.name}")
    
    def validate_trade(self, trade_data: Dict[str, Any]) -> Tuple[bool, List[ComplianceViolation]]:
        """
        Validate a trade against all applicable SEC rules.
        
        Args:
            trade_data: Trade data to validate
            
        Returns:
            Tuple of (is_compliant, violations)
        """
        violations = []
        
        for rule_id, rule in self.rules.items():
            if rule.status != RuleStatus.ACTIVE:
                continue
            
            # Validate against specific rule
            rule_violations = self._validate_rule(rule, trade_data)
            violations.extend(rule_violations)
        
        is_compliant = len(violations) == 0
        
        # Log violations
        for violation in violations:
            self._log_violation(violation)
        
        return is_compliant, violations
    
    def _validate_rule(self, rule: SECRule, trade_data: Dict[str, Any]) -> List[ComplianceViolation]:
        """
        Validate trade data against a specific SEC rule.
        
        Args:
            rule: SEC rule to validate against
            trade_data: Trade data to validate
            
        Returns:
            List of compliance violations
        """
        violations = []
        
        try:
            if rule.rule_id == "SEC_RULE_001":  # Position Size Limits
                violations.extend(self._validate_position_limits(rule, trade_data))
            elif rule.rule_id == "SEC_RULE_002":  # Wash Sale Prevention
                violations.extend(self._validate_wash_sales(rule, trade_data))
            elif rule.rule_id == "SEC_RULE_003":  # Market Manipulation
                violations.extend(self._validate_market_manipulation(rule, trade_data))
            elif rule.rule_id == "SEC_RULE_004":  # Insider Trading
                violations.extend(self._validate_insider_trading(rule, trade_data))
            elif rule.rule_id == "SEC_RULE_005":  # Best Execution
                violations.extend(self._validate_best_execution(rule, trade_data))
            elif rule.rule_id == "SEC_RULE_006":  # AML
                violations.extend(self._validate_aml(rule, trade_data))
        except Exception as e:
            logger.error(f"Error validating rule {rule.rule_id}: {e}")
        
        return violations
    
    def _validate_position_limits(self, rule: SECRule, trade_data: Dict[str, Any]) -> List[ComplianceViolation]:
        """Validate position size limits."""
        violations = []
        logic = rule.validation_logic
        
        # Check position size
        position_size = trade_data.get('position_size', 0)
        max_position = logic.get('max_position_size', 0.05)
        
        if position_size > max_position:
            violation = ComplianceViolation(
                violation_id=f"POS_LIMIT_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                rule_id=rule.rule_id,
                rule_name=rule.name,
                severity=rule.severity,
                violation_type="POSITION_SIZE_EXCEEDED",
                description=f"Position size {position_size:.2%} exceeds limit {max_position:.2%}",
                affected_entities=[trade_data.get('symbol', 'UNKNOWN')],
                violation_data={
                    'position_size': position_size,
                    'max_position': max_position,
                    'excess': position_size - max_position
                },
                detected_at=datetime.now()
            )
            violations.append(violation)
        
        return violations
    
    def _validate_wash_sales(self, rule: SECRule, trade_data: Dict[str, Any]) -> List[ComplianceViolation]:
        """Validate wash sale prevention."""
        violations = []
        logic = rule.validation_logic
        
        # Check for wash sale conditions
        symbol = trade_data.get('symbol')
        trade_type = trade_data.get('trade_type', 'BUY')
        quantity = trade_data.get('quantity', 0)
        
        if trade_type == 'BUY' and symbol:
            # Check recent sales of same security
            recent_sales = self._get_recent_sales(symbol, logic.get('wash_sale_window', 30))
            
            for sale in recent_sales:
                if sale['quantity'] == quantity and sale['loss'] > logic.get('loss_threshold', 0.01):
                    violation = ComplianceViolation(
                        violation_id=f"WASH_SALE_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        rule_id=rule.rule_id,
                        rule_name=rule.name,
                        severity=rule.severity,
                        violation_type="WASH_SALE_DETECTED",
                        description=f"Wash sale detected for {symbol}: buying {quantity} shares within {logic.get('wash_sale_window')} days of loss sale",
                        affected_entities=[symbol],
                        violation_data={
                            'symbol': symbol,
                            'quantity': quantity,
                            'recent_sale': sale,
                            'loss_amount': sale['loss']
                        },
                        detected_at=datetime.now()
                    )
                    violations.append(violation)
        
        return violations
    
    def _validate_market_manipulation(self, rule: SECRule, trade_data: Dict[str, Any]) -> List[ComplianceViolation]:
        """Validate market manipulation prevention."""
        violations = []
        logic = rule.validation_logic
        
        # Check order size
        order_value = trade_data.get('order_value', 0)
        max_order_size = logic.get('max_order_size', 1000000)
        
        if order_value > max_order_size:
            violation = ComplianceViolation(
                violation_id=f"MANIPULATION_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                rule_id=rule.rule_id,
                rule_name=rule.name,
                severity=rule.severity,
                violation_type="ORDER_SIZE_EXCEEDED",
                description=f"Order value ${order_value:,.2f} exceeds manipulation threshold ${max_order_size:,.2f}",
                affected_entities=[trade_data.get('symbol', 'UNKNOWN')],
                violation_data={
                    'order_value': order_value,
                    'max_order_size': max_order_size,
                    'excess': order_value - max_order_size
                },
                detected_at=datetime.now()
            )
            violations.append(violation)
        
        # Check for spoofing patterns
        if self._detect_spoofing(trade_data):
            violation = ComplianceViolation(
                violation_id=f"SPOOFING_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                rule_id=rule.rule_id,
                rule_name=rule.name,
                severity=rule.severity,
                violation_type="SPOOFING_DETECTED",
                description="Spoofing pattern detected in order flow",
                affected_entities=[trade_data.get('symbol', 'UNKNOWN')],
                violation_data=trade_data,
                detected_at=datetime.now()
            )
            violations.append(violation)
        
        return violations
    
    def _validate_insider_trading(self, rule: SECRule, trade_data: Dict[str, Any]) -> List[ComplianceViolation]:
        """Validate insider trading prevention."""
        violations = []
        
        # Check if security is restricted
        symbol = trade_data.get('symbol')
        if self._is_restricted_security(symbol):
            violation = ComplianceViolation(
                violation_id=f"INSIDER_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                rule_id=rule.rule_id,
                rule_name=rule.name,
                severity=rule.severity,
                violation_type="RESTRICTED_SECURITY",
                description=f"Trading in restricted security {symbol}",
                affected_entities=[symbol],
                violation_data={'symbol': symbol, 'restriction_reason': 'INSIDER_TRADING'},
                detected_at=datetime.now()
            )
            violations.append(violation)
        
        # Check blackout periods
        if self._is_blackout_period(symbol):
            violation = ComplianceViolation(
                violation_id=f"BLACKOUT_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                rule_id=rule.rule_id,
                rule_name=rule.name,
                severity=rule.severity,
                violation_type="BLACKOUT_PERIOD",
                description=f"Trading during blackout period for {symbol}",
                affected_entities=[symbol],
                violation_data={'symbol': symbol, 'blackout_reason': 'EARNINGS_PERIOD'},
                detected_at=datetime.now()
            )
            violations.append(violation)
        
        return violations
    
    def _validate_best_execution(self, rule: SECRule, trade_data: Dict[str, Any]) -> List[ComplianceViolation]:
        """Validate best execution requirements."""
        violations = []
        logic = rule.validation_logic
        
        # Check price improvement
        execution_price = trade_data.get('execution_price', 0)
        market_price = trade_data.get('market_price', 0)
        
        if execution_price > 0 and market_price > 0:
            price_improvement = (market_price - execution_price) / market_price
            min_improvement = logic.get('price_improvement', 0.001)
            
            if price_improvement < min_improvement:
                violation = ComplianceViolation(
                    violation_id=f"BEST_EXEC_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    rule_id=rule.rule_id,
                    rule_name=rule.name,
                    severity=rule.severity,
                    violation_type="INSUFFICIENT_PRICE_IMPROVEMENT",
                    description=f"Price improvement {price_improvement:.4f} below minimum {min_improvement:.4f}",
                    affected_entities=[trade_data.get('symbol', 'UNKNOWN')],
                    violation_data={
                        'execution_price': execution_price,
                        'market_price': market_price,
                        'price_improvement': price_improvement,
                        'min_improvement': min_improvement
                    },
                    detected_at=datetime.now()
                )
                violations.append(violation)
        
        return violations
    
    def _validate_aml(self, rule: SECRule, trade_data: Dict[str, Any]) -> List[ComplianceViolation]:
        """Validate anti-money laundering requirements."""
        violations = []
        logic = rule.validation_logic
        
        # Check suspicious amount
        trade_amount = trade_data.get('trade_amount', 0)
        suspicious_threshold = logic.get('suspicious_amount', 10000)
        
        if trade_amount >= suspicious_threshold:
            # Check for unusual patterns
            if self._detect_unusual_patterns(trade_data):
                violation = ComplianceViolation(
                    violation_id=f"AML_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    rule_id=rule.rule_id,
                    rule_name=rule.name,
                    severity=rule.severity,
                    violation_type="SUSPICIOUS_ACTIVITY",
                    description=f"Suspicious activity detected: ${trade_amount:,.2f} trade with unusual patterns",
                    affected_entities=[trade_data.get('symbol', 'UNKNOWN')],
                    violation_data={
                        'trade_amount': trade_amount,
                        'threshold': suspicious_threshold,
                        'patterns': self._get_unusual_patterns(trade_data)
                    },
                    detected_at=datetime.now()
                )
                violations.append(violation)
        
        return violations
    
    def _get_recent_sales(self, symbol: str, days: int) -> List[Dict[str, Any]]:
        """Get recent sales for wash sale detection."""
        # In a real implementation, this would query the database
        # For now, return empty list
        return []
    
    def _detect_spoofing(self, trade_data: Dict[str, Any]) -> bool:
        """Detect spoofing patterns in trade data."""
        # Simplified spoofing detection
        # In a real implementation, this would analyze order flow patterns
        return False
    
    def _is_restricted_security(self, symbol: str) -> bool:
        """Check if security is restricted for insider trading."""
        # In a real implementation, this would check against restricted securities list
        restricted_securities = ['AAPL', 'GOOGL', 'MSFT']  # Example
        return symbol in restricted_securities
    
    def _is_blackout_period(self, symbol: str) -> bool:
        """Check if security is in blackout period."""
        # In a real implementation, this would check earnings calendars
        return False
    
    def _detect_unusual_patterns(self, trade_data: Dict[str, Any]) -> bool:
        """Detect unusual trading patterns for AML."""
        # Simplified pattern detection
        return False
    
    def _get_unusual_patterns(self, trade_data: Dict[str, Any]) -> List[str]:
        """Get list of unusual patterns detected."""
        return []
    
    def _log_violation(self, violation: ComplianceViolation) -> None:
        """Log compliance violation to database."""
        self.violations.append(violation)
        
        # Save to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO compliance_violations 
            (violation_id, rule_id, rule_name, severity, violation_type, description,
             affected_entities, violation_data, detected_at, resolved_at, resolution_notes,
             enforcement_action, reported_to_sec, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            violation.violation_id, violation.rule_id, violation.rule_name,
            violation.severity.value, violation.violation_type, violation.description,
            json.dumps(violation.affected_entities), json.dumps(violation.violation_data),
            violation.detected_at.isoformat(), violation.resolved_at.isoformat() if violation.resolved_at else None,
            violation.resolution_notes, violation.enforcement_action, violation.reported_to_sec,
            violation.created_at.isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        logger.warning(f"Compliance violation logged: {violation.violation_id} - {violation.description}")
    
    def get_compliance_metrics(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Get compliance metrics for a date range.
        
        Args:
            start_date: Start date for metrics
            end_date: End date for metrics
            
        Returns:
            Compliance metrics dictionary
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT severity, COUNT(*) as count
            FROM compliance_violations
            WHERE detected_at BETWEEN ? AND ?
            GROUP BY severity
        """, (start_date.isoformat(), end_date.isoformat()))
        
        severity_counts = dict(cursor.fetchall())
        
        cursor.execute("""
            SELECT COUNT(*) as total, COUNT(CASE WHEN resolved_at IS NOT NULL THEN 1 END) as resolved
            FROM compliance_violations
            WHERE detected_at BETWEEN ? AND ?
        """, (start_date.isoformat(), end_date.isoformat()))
        
        total_violations, resolved_violations = cursor.fetchone()
        
        conn.close()
        
        # Calculate compliance score
        total_violations = total_violations or 0
        resolved_violations = resolved_violations or 0
        compliance_score = (resolved_violations / total_violations * 100) if total_violations > 0 else 100
        
        return {
            'total_violations': total_violations,
            'resolved_violations': resolved_violations,
            'compliance_score': compliance_score,
            'severity_breakdown': {
                'CRITICAL': severity_counts.get('CRITICAL', 0),
                'HIGH': severity_counts.get('HIGH', 0),
                'MEDIUM': severity_counts.get('MEDIUM', 0),
                'LOW': severity_counts.get('LOW', 0)
            },
            'date_range': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            }
        }
    
    def generate_compliance_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Generate comprehensive compliance report.
        
        Args:
            start_date: Start date for report
            end_date: End date for report
            
        Returns:
            Compliance report dictionary
        """
        metrics = self.get_compliance_metrics(start_date, end_date)
        
        # Get detailed violation data
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT rule_id, rule_name, severity, violation_type, description, detected_at
            FROM compliance_violations
            WHERE detected_at BETWEEN ? AND ?
            ORDER BY detected_at DESC
        """, (start_date.isoformat(), end_date.isoformat()))
        
        violations = cursor.fetchall()
        conn.close()
        
        return {
            'report_period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            },
            'summary_metrics': metrics,
            'detailed_violations': [
                {
                    'rule_id': v[0],
                    'rule_name': v[1],
                    'severity': v[2],
                    'violation_type': v[3],
                    'description': v[4],
                    'detected_at': v[5]
                }
                for v in violations
            ],
            'recommendations': self._generate_recommendations(metrics),
            'generated_at': datetime.now().isoformat()
        }
    
    def _generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate compliance recommendations based on metrics."""
        recommendations = []
        
        if metrics['severity_breakdown']['CRITICAL'] > 0:
            recommendations.append("Immediate action required: Critical violations detected")
        
        if metrics['compliance_score'] < 90:
            recommendations.append("Compliance score below 90% - review and improve processes")
        
        if metrics['severity_breakdown']['HIGH'] > metrics['total_violations'] * 0.1:
            recommendations.append("High severity violations exceed 10% - review risk controls")
        
        if metrics['resolved_violations'] < metrics['total_violations'] * 0.8:
            recommendations.append("Resolution rate below 80% - improve violation resolution process")
        
        return recommendations
    
    def get_active_rules(self) -> List[SECRule]:
        """Get list of active SEC rules."""
        return [rule for rule in self.rules.values() if rule.status == RuleStatus.ACTIVE]
    
    def get_violations_by_severity(self, severity: RuleSeverity) -> List[ComplianceViolation]:
        """Get violations by severity level."""
        return [v for v in self.violations if v.severity == severity]
    
    def resolve_violation(self, violation_id: str, resolution_notes: str, enforcement_action: str = None) -> bool:
        """
        Resolve a compliance violation.
        
        Args:
            violation_id: Violation ID to resolve
            resolution_notes: Notes about resolution
            enforcement_action: Enforcement action taken
            
        Returns:
            True if resolved successfully
        """
        for violation in self.violations:
            if violation.violation_id == violation_id:
                violation.resolved_at = datetime.now()
                violation.resolution_notes = resolution_notes
                violation.enforcement_action = enforcement_action
                
                # Update database
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE compliance_violations
                    SET resolved_at = ?, resolution_notes = ?, enforcement_action = ?
                    WHERE violation_id = ?
                """, (violation.resolved_at.isoformat(), resolution_notes, enforcement_action, violation_id))
                
                conn.commit()
                conn.close()
                
                logger.info(f"Violation resolved: {violation_id}")
                return True
        
        return False
