"""
Enterprise Features Module

This module contains enterprise-grade features for production-ready trading systems:
- Advanced ML Predictive Models
- Complex Compliance Automation  
- Professional Penetration Testing
- Enterprise-Grade SLA Monitoring
- Enterprise Integration & Deployment

Author: AI Trading System
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "AI Trading System"

# Import main enterprise components
from .advanced_ml import *
from .compliance import *
from .security import *
from .monitoring import *
from .integration import *

__all__ = [
    # Advanced ML
    "CrashDetector",
    "BubbleDetector", 
    "RegimePredictor",
    "VolatilityForecaster",
    "CorrelationAnalyzer",
    
    # Compliance
    "SECCompliance",
    "RiskEnforcer",
    "AuditTrail",
    "RegulatoryReporter",
    "TradeSurveillance",
    
    # Security
    "PenetrationTester",
    "VulnerabilityAssessor",
    "SecurityMonitor",
    "IncidentResponder",
    
    # Monitoring
    "SLAMonitor",
    "PerformanceTracker",
    "AlertManager",
    "HealthChecker",
    
    # Integration
    "EnterpriseManager",
    "DeploymentTools",
    "ProductionConfig"
]
