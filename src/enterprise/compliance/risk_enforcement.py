"""
Risk Enforcement System

This module implements comprehensive risk limit enforcement including
real-time risk monitoring, limit validation, violation detection, and
automated risk management actions.

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

class RiskLimitType(Enum):
    """Types of risk limits."""
    POSITION_SIZE = "POSITION_SIZE"
    PORTFOLIO_VALUE = "PORTFOLIO_VALUE"
    DAILY_LOSS = "DAILY_LOSS"
    VAR = "VAR"
    BETA = "BETA"
    SECTOR_EXPOSURE = "SECTOR_EXPOSURE"
    CONCENTRATION = "CONCENTRATION"
    LEVERAGE = "LEVERAGE"
    CORRELATION = "CORRELATION"
    VOLATILITY = "VOLATILITY"

class RiskSeverity(Enum):
    """Risk violation severity levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class EnforcementAction(Enum):
    """Risk enforcement actions."""
    WARNING = "WARNING"
    REDUCE_POSITION = "REDUCE_POSITION"
    REJECT_ORDER = "REJECT_ORDER"
    CLOSE_POSITION = "CLOSE_POSITION"
    HALT_TRADING = "HALT_TRADING"
    EMERGENCY_STOP = "EMERGENCY_STOP"

@dataclass
class RiskLimit:
    """Risk limit definition."""
    limit_id: str
    name: str
    limit_type: RiskLimitType
    value: float
    threshold: float
    severity: RiskSeverity
    enforcement_action: EnforcementAction
    description: str
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class RiskViolation:
    """Risk limit violation record."""
    violation_id: str
    limit_id: str
    limit_name: str
    limit_type: RiskLimitType
    current_value: float
    limit_value: float
    excess_amount: float
    severity: RiskSeverity
    enforcement_action: EnforcementAction
    description: str
    affected_positions: List[str]
    violation_data: Dict[str, Any]
    detected_at: datetime
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None
    action_taken: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)

class RiskEnforcer:
    """
    Comprehensive risk limit enforcement system.
    
    Features:
    - Real-time risk limit monitoring
    - Automated risk violation detection
    - Enforcement action execution
    - Risk metrics calculation
    - Portfolio risk analysis
    - Historical risk tracking
    """
    
    def __init__(self, db_path: str = "data/risk_enforcement.db"):
        """
        Initialize risk enforcement system.
        
        Args:
            db_path: Path to risk enforcement database
        """
        self.db_path = db_path
        self.risk_limits: Dict[str, RiskLimit] = {}
        self.violations: List[RiskViolation] = []
        self.portfolio_data: Dict[str, Any] = {}
        
        # Risk calculation parameters
        self.risk_params = {
            'confidence_level': 0.95,
            'lookback_days': 252,
            'var_method': 'historical',
            'correlation_window': 60
        }
        
        # Initialize database
        self._init_database()
        
        # Load default risk limits
        self._load_default_limits()
        
        logger.info("Risk Enforcement system initialized")
    
    def _init_database(self) -> None:
        """Initialize risk enforcement database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create risk limits table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS risk_limits (
                limit_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                limit_type TEXT,
                value REAL,
                threshold REAL,
                severity TEXT,
                enforcement_action TEXT,
                description TEXT,
                is_active INTEGER,
                created_at TEXT,
                updated_at TEXT
            )
        """)
        
        # Create risk violations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS risk_violations (
                violation_id TEXT PRIMARY KEY,
                limit_id TEXT,
                limit_name TEXT,
                limit_type TEXT,
                current_value REAL,
                limit_value REAL,
                excess_amount REAL,
                severity TEXT,
                enforcement_action TEXT,
                description TEXT,
                affected_positions TEXT,
                violation_data TEXT,
                detected_at TEXT,
                resolved_at TEXT,
                resolution_notes TEXT,
                action_taken TEXT,
                created_at TEXT,
                FOREIGN KEY (limit_id) REFERENCES risk_limits (limit_id)
            )
        """)
        
        # Create risk metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS risk_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                portfolio_value REAL,
                total_var REAL,
                max_drawdown REAL,
                sharpe_ratio REAL,
                beta REAL,
                volatility REAL,
                correlation_risk REAL,
                concentration_risk REAL,
                created_at TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _load_default_limits(self) -> None:
        """Load default risk limits."""
        default_limits = [
            RiskLimit(
                limit_id="RISK_LIMIT_001",
                name="Maximum Position Size",
                limit_type=RiskLimitType.POSITION_SIZE,
                value=0.05,  # 5% of portfolio
                threshold=0.04,  # Warning at 4%
                severity=RiskSeverity.HIGH,
                enforcement_action=EnforcementAction.REJECT_ORDER,
                description="Maximum position size as percentage of portfolio value"
            ),
            RiskLimit(
                limit_id="RISK_LIMIT_002",
                name="Daily Loss Limit",
                limit_type=RiskLimitType.DAILY_LOSS,
                value=0.02,  # 2% daily loss
                threshold=0.015,  # Warning at 1.5%
                severity=RiskSeverity.CRITICAL,
                enforcement_action=EnforcementAction.HALT_TRADING,
                description="Maximum daily loss as percentage of portfolio value"
            ),
            RiskLimit(
                limit_id="RISK_LIMIT_003",
                name="Value at Risk (VaR)",
                limit_type=RiskLimitType.VAR,
                value=0.03,  # 3% VaR
                threshold=0.025,  # Warning at 2.5%
                severity=RiskSeverity.HIGH,
                enforcement_action=EnforcementAction.REDUCE_POSITION,
                description="Maximum Value at Risk at 95% confidence level"
            ),
            RiskLimit(
                limit_id="RISK_LIMIT_004",
                name="Portfolio Beta",
                limit_type=RiskLimitType.BETA,
                value=1.5,  # Maximum beta
                threshold=1.3,  # Warning at 1.3
                severity=RiskSeverity.MEDIUM,
                enforcement_action=EnforcementAction.WARNING,
                description="Maximum portfolio beta relative to market"
            ),
            RiskLimit(
                limit_id="RISK_LIMIT_005",
                name="Sector Concentration",
                limit_type=RiskLimitType.SECTOR_EXPOSURE,
                value=0.20,  # 20% per sector
                threshold=0.15,  # Warning at 15%
                severity=RiskSeverity.MEDIUM,
                enforcement_action=EnforcementAction.WARNING,
                description="Maximum exposure to any single sector"
            ),
            RiskLimit(
                limit_id="RISK_LIMIT_006",
                name="Portfolio Volatility",
                limit_type=RiskLimitType.VOLATILITY,
                value=0.25,  # 25% annual volatility
                threshold=0.20,  # Warning at 20%
                severity=RiskSeverity.MEDIUM,
                enforcement_action=EnforcementAction.REDUCE_POSITION,
                description="Maximum portfolio volatility"
            ),
            RiskLimit(
                limit_id="RISK_LIMIT_007",
                name="Leverage Limit",
                limit_type=RiskLimitType.LEVERAGE,
                value=2.0,  # 2x leverage
                threshold=1.5,  # Warning at 1.5x
                severity=RiskSeverity.HIGH,
                enforcement_action=EnforcementAction.REDUCE_POSITION,
                description="Maximum portfolio leverage"
            ),
            RiskLimit(
                limit_id="RISK_LIMIT_008",
                name="Correlation Risk",
                limit_type=RiskLimitType.CORRELATION,
                value=0.8,  # 80% max correlation
                threshold=0.7,  # Warning at 70%
                severity=RiskSeverity.MEDIUM,
                enforcement_action=EnforcementAction.WARNING,
                description="Maximum correlation between positions"
            )
        ]
        
        for limit in default_limits:
            self.add_risk_limit(limit)
    
    def add_risk_limit(self, limit: RiskLimit) -> None:
        """
        Add a new risk limit.
        
        Args:
            limit: Risk limit definition
        """
        self.risk_limits[limit.limit_id] = limit
        
        # Save to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO risk_limits 
            (limit_id, name, limit_type, value, threshold, severity, enforcement_action,
             description, is_active, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            limit.limit_id, limit.name, limit.limit_type.value, limit.value,
            limit.threshold, limit.severity.value, limit.enforcement_action.value,
            limit.description, limit.is_active, limit.created_at.isoformat(),
            limit.updated_at.isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Added risk limit: {limit.limit_id} - {limit.name}")
    
    def update_portfolio_data(self, portfolio_data: Dict[str, Any]) -> None:
        """
        Update portfolio data for risk calculations.
        
        Args:
            portfolio_data: Current portfolio data
        """
        self.portfolio_data = portfolio_data
        logger.debug("Portfolio data updated for risk calculations")
    
    def validate_risk_limits(self, trade_data: Dict[str, Any] = None) -> Tuple[bool, List[RiskViolation]]:
        """
        Validate current portfolio against all risk limits.
        
        Args:
            trade_data: Optional trade data for pre-trade validation
            
        Returns:
            Tuple of (is_compliant, violations)
        """
        violations = []
        
        # Calculate current risk metrics
        risk_metrics = self._calculate_risk_metrics()
        
        # Validate against each risk limit
        for limit_id, limit in self.risk_limits.items():
            if not limit.is_active:
                continue
            
            # Calculate current value for this limit type
            current_value = self._get_current_risk_value(limit.limit_type, risk_metrics, trade_data)
            
            # Check if limit is violated
            if current_value > limit.value:
                excess_amount = current_value - limit.value
                
                violation = RiskViolation(
                    violation_id=f"RISK_VIOL_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    limit_id=limit.limit_id,
                    limit_name=limit.name,
                    limit_type=limit.limit_type,
                    current_value=current_value,
                    limit_value=limit.value,
                    excess_amount=excess_amount,
                    severity=limit.severity,
                    enforcement_action=limit.enforcement_action,
                    description=f"{limit.name}: {current_value:.4f} exceeds limit {limit.value:.4f}",
                    affected_positions=self._get_affected_positions(limit.limit_type),
                    violation_data={
                        'risk_metrics': risk_metrics,
                        'trade_data': trade_data,
                        'excess_percentage': (excess_amount / limit.value) * 100
                    },
                    detected_at=datetime.now()
                )
                violations.append(violation)
        
        # Log violations
        for violation in violations:
            self._log_violation(violation)
        
        is_compliant = len(violations) == 0
        
        return is_compliant, violations
    
    def _calculate_risk_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive risk metrics for the portfolio."""
        if not self.portfolio_data:
            return {}
        
        positions = self.portfolio_data.get('positions', [])
        portfolio_value = self.portfolio_data.get('total_value', 0)
        
        if not positions or portfolio_value == 0:
            return {}
        
        # Calculate position sizes
        position_sizes = []
        position_values = []
        position_weights = []
        
        for position in positions:
            value = position.get('market_value', 0)
            position_values.append(value)
            position_weights.append(value / portfolio_value)
            position_sizes.append(value / portfolio_value)
        
        # Calculate risk metrics
        risk_metrics = {
            'portfolio_value': portfolio_value,
            'position_count': len(positions),
            'max_position_size': max(position_sizes) if position_sizes else 0,
            'concentration_risk': self._calculate_concentration_risk(position_weights),
            'portfolio_volatility': self._calculate_portfolio_volatility(positions),
            'portfolio_beta': self._calculate_portfolio_beta(positions),
            'var_95': self._calculate_var(positions, 0.95),
            'var_99': self._calculate_var(positions, 0.99),
            'max_drawdown': self._calculate_max_drawdown(positions),
            'leverage': self._calculate_leverage(positions),
            'sector_exposure': self._calculate_sector_exposure(positions),
            'correlation_risk': self._calculate_correlation_risk(positions)
        }
        
        return risk_metrics
    
    def _get_current_risk_value(self, limit_type: RiskLimitType, risk_metrics: Dict[str, float], 
                               trade_data: Dict[str, Any] = None) -> float:
        """Get current value for a specific risk limit type."""
        if limit_type == RiskLimitType.POSITION_SIZE:
            if trade_data:
                # Pre-trade validation
                trade_value = trade_data.get('trade_value', 0)
                portfolio_value = risk_metrics.get('portfolio_value', 1)
                return trade_value / portfolio_value
            else:
                return risk_metrics.get('max_position_size', 0)
        
        elif limit_type == RiskLimitType.DAILY_LOSS:
            return abs(risk_metrics.get('daily_pnl', 0) / risk_metrics.get('portfolio_value', 1))
        
        elif limit_type == RiskLimitType.VAR:
            return risk_metrics.get('var_95', 0)
        
        elif limit_type == RiskLimitType.BETA:
            return abs(risk_metrics.get('portfolio_beta', 0))
        
        elif limit_type == RiskLimitType.SECTOR_EXPOSURE:
            return risk_metrics.get('sector_exposure', 0)
        
        elif limit_type == RiskLimitType.VOLATILITY:
            return risk_metrics.get('portfolio_volatility', 0)
        
        elif limit_type == RiskLimitType.LEVERAGE:
            return risk_metrics.get('leverage', 0)
        
        elif limit_type == RiskLimitType.CORRELATION:
            return risk_metrics.get('correlation_risk', 0)
        
        elif limit_type == RiskLimitType.CONCENTRATION:
            return risk_metrics.get('concentration_risk', 0)
        
        else:
            return 0.0
    
    def _calculate_concentration_risk(self, position_weights: List[float]) -> float:
        """Calculate concentration risk using Herfindahl index."""
        if not position_weights:
            return 0.0
        
        # Herfindahl-Hirschman Index
        hhi = sum(w**2 for w in position_weights)
        return hhi
    
    def _calculate_portfolio_volatility(self, positions: List[Dict[str, Any]]) -> float:
        """Calculate portfolio volatility."""
        if not positions:
            return 0.0
        
        # Simplified volatility calculation
        # In a real implementation, this would use historical returns and covariance matrix
        total_volatility = 0.0
        total_weight = 0.0
        
        for position in positions:
            weight = position.get('weight', 0)
            volatility = position.get('volatility', 0.2)  # Default 20% volatility
            total_volatility += weight * volatility
            total_weight += weight
        
        return total_volatility / total_weight if total_weight > 0 else 0.0
    
    def _calculate_portfolio_beta(self, positions: List[Dict[str, Any]]) -> float:
        """Calculate portfolio beta."""
        if not positions:
            return 0.0
        
        total_beta = 0.0
        total_weight = 0.0
        
        for position in positions:
            weight = position.get('weight', 0)
            beta = position.get('beta', 1.0)  # Default beta of 1.0
            total_beta += weight * beta
            total_weight += weight
        
        return total_beta / total_weight if total_weight > 0 else 0.0
    
    def _calculate_var(self, positions: List[Dict[str, Any]], confidence_level: float) -> float:
        """Calculate Value at Risk."""
        if not positions:
            return 0.0
        
        # Simplified VaR calculation
        # In a real implementation, this would use historical simulation or parametric methods
        portfolio_value = sum(pos.get('market_value', 0) for pos in positions)
        
        # Assume normal distribution for simplicity
        z_score = 1.96 if confidence_level == 0.95 else 2.33  # 99% confidence
        portfolio_volatility = self._calculate_portfolio_volatility(positions)
        
        var = z_score * portfolio_volatility * portfolio_value
        return var / portfolio_value if portfolio_value > 0 else 0.0
    
    def _calculate_max_drawdown(self, positions: List[Dict[str, Any]]) -> float:
        """Calculate maximum drawdown."""
        # In a real implementation, this would use historical portfolio values
        return 0.0
    
    def _calculate_leverage(self, positions: List[Dict[str, Any]]) -> float:
        """Calculate portfolio leverage."""
        if not positions:
            return 0.0
        
        total_market_value = sum(pos.get('market_value', 0) for pos in positions)
        total_cash = self.portfolio_data.get('cash', 0)
        
        if total_cash == 0:
            return 0.0
        
        leverage = total_market_value / (total_market_value + total_cash)
        return leverage
    
    def _calculate_sector_exposure(self, positions: List[Dict[str, Any]]) -> float:
        """Calculate maximum sector exposure."""
        if not positions:
            return 0.0
        
        sector_exposures = {}
        total_value = sum(pos.get('market_value', 0) for pos in positions)
        
        for position in positions:
            sector = position.get('sector', 'UNKNOWN')
            value = position.get('market_value', 0)
            weight = value / total_value if total_value > 0 else 0
            
            if sector in sector_exposures:
                sector_exposures[sector] += weight
            else:
                sector_exposures[sector] = weight
        
        return max(sector_exposures.values()) if sector_exposures else 0.0
    
    def _calculate_correlation_risk(self, positions: List[Dict[str, Any]]) -> float:
        """Calculate correlation risk."""
        if len(positions) < 2:
            return 0.0
        
        # Simplified correlation risk calculation
        # In a real implementation, this would calculate actual correlations
        return 0.0
    
    def _get_affected_positions(self, limit_type: RiskLimitType) -> List[str]:
        """Get list of positions affected by a risk limit violation."""
        positions = self.portfolio_data.get('positions', [])
        
        if limit_type == RiskLimitType.POSITION_SIZE:
            # Return largest positions
            sorted_positions = sorted(positions, key=lambda x: x.get('market_value', 0), reverse=True)
            return [pos.get('symbol', 'UNKNOWN') for pos in sorted_positions[:5]]
        
        elif limit_type == RiskLimitType.SECTOR_EXPOSURE:
            # Return positions in the largest sector
            sector_exposures = {}
            for position in positions:
                sector = position.get('sector', 'UNKNOWN')
                if sector in sector_exposures:
                    sector_exposures[sector].append(position.get('symbol', 'UNKNOWN'))
                else:
                    sector_exposures[sector] = [position.get('symbol', 'UNKNOWN')]
            
            if sector_exposures:
                largest_sector = max(sector_exposures.keys(), key=lambda x: len(sector_exposures[x]))
                return sector_exposures[largest_sector]
        
        else:
            # Return all positions for other limit types
            return [pos.get('symbol', 'UNKNOWN') for pos in positions]
    
    def _log_violation(self, violation: RiskViolation) -> None:
        """Log risk violation to database."""
        self.violations.append(violation)
        
        # Save to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO risk_violations 
            (violation_id, limit_id, limit_name, limit_type, current_value, limit_value,
             excess_amount, severity, enforcement_action, description, affected_positions,
             violation_data, detected_at, resolved_at, resolution_notes, action_taken, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            violation.violation_id, violation.limit_id, violation.limit_name,
            violation.limit_type.value, violation.current_value, violation.limit_value,
            violation.excess_amount, violation.severity.value, violation.enforcement_action.value,
            violation.description, json.dumps(violation.affected_positions),
            json.dumps(violation.violation_data), violation.detected_at.isoformat(),
            violation.resolved_at.isoformat() if violation.resolved_at else None,
            violation.resolution_notes, violation.action_taken, violation.created_at.isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        logger.warning(f"Risk violation logged: {violation.violation_id} - {violation.description}")
    
    def execute_enforcement_action(self, violation: RiskViolation) -> bool:
        """
        Execute enforcement action for a risk violation.
        
        Args:
            violation: Risk violation to act upon
            
        Returns:
            True if action executed successfully
        """
        try:
            if violation.enforcement_action == EnforcementAction.WARNING:
                self._execute_warning(violation)
            elif violation.enforcement_action == EnforcementAction.REDUCE_POSITION:
                self._execute_reduce_position(violation)
            elif violation.enforcement_action == EnforcementAction.REJECT_ORDER:
                self._execute_reject_order(violation)
            elif violation.enforcement_action == EnforcementAction.CLOSE_POSITION:
                self._execute_close_position(violation)
            elif violation.enforcement_action == EnforcementAction.HALT_TRADING:
                self._execute_halt_trading(violation)
            elif violation.enforcement_action == EnforcementAction.EMERGENCY_STOP:
                self._execute_emergency_stop(violation)
            
            # Update violation with action taken
            violation.action_taken = violation.enforcement_action.value
            self._update_violation(violation)
            
            logger.info(f"Enforcement action executed: {violation.enforcement_action.value} for {violation.violation_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute enforcement action: {e}")
            return False
    
    def _execute_warning(self, violation: RiskViolation) -> None:
        """Execute warning action."""
        logger.warning(f"RISK WARNING: {violation.description}")
        # In a real implementation, this would send alerts to risk managers
    
    def _execute_reduce_position(self, violation: RiskViolation) -> None:
        """Execute position reduction action."""
        logger.warning(f"POSITION REDUCTION REQUIRED: {violation.description}")
        # In a real implementation, this would trigger position reduction orders
    
    def _execute_reject_order(self, violation: RiskViolation) -> None:
        """Execute order rejection action."""
        logger.warning(f"ORDER REJECTED: {violation.description}")
        # In a real implementation, this would reject the pending order
    
    def _execute_close_position(self, violation: RiskViolation) -> None:
        """Execute position closure action."""
        logger.warning(f"POSITION CLOSURE REQUIRED: {violation.description}")
        # In a real implementation, this would trigger position closure orders
    
    def _execute_halt_trading(self, violation: RiskViolation) -> None:
        """Execute trading halt action."""
        logger.critical(f"TRADING HALTED: {violation.description}")
        # In a real implementation, this would halt all trading activities
    
    def _execute_emergency_stop(self, violation: RiskViolation) -> None:
        """Execute emergency stop action."""
        logger.critical(f"EMERGENCY STOP: {violation.description}")
        # In a real implementation, this would trigger emergency procedures
    
    def _update_violation(self, violation: RiskViolation) -> None:
        """Update violation record in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE risk_violations
            SET action_taken = ?
            WHERE violation_id = ?
        """, (violation.action_taken, violation.violation_id))
        
        conn.commit()
        conn.close()
    
    def get_risk_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of current risk metrics."""
        risk_metrics = self._calculate_risk_metrics()
        
        return {
            'portfolio_value': risk_metrics.get('portfolio_value', 0),
            'position_count': risk_metrics.get('position_count', 0),
            'max_position_size': risk_metrics.get('max_position_size', 0),
            'portfolio_volatility': risk_metrics.get('portfolio_volatility', 0),
            'portfolio_beta': risk_metrics.get('portfolio_beta', 0),
            'var_95': risk_metrics.get('var_95', 0),
            'leverage': risk_metrics.get('leverage', 0),
            'sector_exposure': risk_metrics.get('sector_exposure', 0),
            'concentration_risk': risk_metrics.get('concentration_risk', 0),
            'calculated_at': datetime.now().isoformat()
        }
    
    def get_violations_summary(self, start_date: datetime = None, end_date: datetime = None) -> Dict[str, Any]:
        """Get summary of risk violations."""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT severity, COUNT(*) as count
            FROM risk_violations
            WHERE detected_at BETWEEN ? AND ?
            GROUP BY severity
        """, (start_date.isoformat(), end_date.isoformat()))
        
        severity_counts = dict(cursor.fetchall())
        
        cursor.execute("""
            SELECT COUNT(*) as total, COUNT(CASE WHEN action_taken IS NOT NULL THEN 1 END) as acted_upon
            FROM risk_violations
            WHERE detected_at BETWEEN ? AND ?
        """, (start_date.isoformat(), end_date.isoformat()))
        
        total_violations, acted_upon = cursor.fetchone()
        
        conn.close()
        
        return {
            'total_violations': total_violations or 0,
            'acted_upon': acted_upon or 0,
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
    
    def resolve_violation(self, violation_id: str, resolution_notes: str) -> bool:
        """
        Resolve a risk violation.
        
        Args:
            violation_id: Violation ID to resolve
            resolution_notes: Notes about resolution
            
        Returns:
            True if resolved successfully
        """
        for violation in self.violations:
            if violation.violation_id == violation_id:
                violation.resolved_at = datetime.now()
                violation.resolution_notes = resolution_notes
                
                # Update database
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE risk_violations
                    SET resolved_at = ?, resolution_notes = ?
                    WHERE violation_id = ?
                """, (violation.resolved_at.isoformat(), resolution_notes, violation_id))
                
                conn.commit()
                conn.close()
                
                logger.info(f"Risk violation resolved: {violation_id}")
                return True
        
        return False
