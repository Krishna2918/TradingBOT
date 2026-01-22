"""
Complex Compliance Automation Module

This module contains enterprise-grade compliance automation systems for:
- SEC rule compliance and enforcement
- Risk limit enforcement and monitoring
- Comprehensive audit trail generation
- Regulatory reporting automation
- Trade surveillance and anomaly detection

Author: AI Trading System
Version: 1.0.0
"""

from .sec_compliance import SECCompliance, SECRule, ComplianceViolation
from .risk_enforcement import RiskEnforcer, RiskLimit, RiskViolation
from .audit_trail import AuditTrail, AuditEvent, AuditLogger
from .regulatory_reporting import RegulatoryReporter, ReportType, ReportGenerator
from .trade_surveillance import TradeSurveillance, SurveillanceAlert, AnomalyDetector

__all__ = [
    # SEC Compliance
    "SECCompliance",
    "SECRule", 
    "ComplianceViolation",
    
    # Risk Enforcement
    "RiskEnforcer",
    "RiskLimit",
    "RiskViolation",
    
    # Audit Trail
    "AuditTrail",
    "AuditEvent",
    "AuditLogger",
    
    # Regulatory Reporting
    "RegulatoryReporter",
    "ReportType",
    "ReportGenerator",
    
    # Trade Surveillance
    "TradeSurveillance",
    "SurveillanceAlert",
    "AnomalyDetector"
]
