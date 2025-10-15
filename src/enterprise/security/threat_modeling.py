"""
Threat Modeling System

This module implements a comprehensive threat modeling system with
STRIDE methodology, attack vector analysis, and security control
recommendations for enterprise applications.

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
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ThreatCategory(Enum):
    """Threat categories based on STRIDE methodology."""
    SPOOFING = "SPOOFING"
    TAMPERING = "TAMPERING"
    REPUDIATION = "REPUDIATION"
    INFORMATION_DISCLOSURE = "INFORMATION_DISCLOSURE"
    DENIAL_OF_SERVICE = "DENIAL_OF_SERVICE"
    ELEVATION_OF_PRIVILEGE = "ELEVATION_OF_PRIVILEGE"

class AttackVector(Enum):
    """Attack vectors."""
    NETWORK = "NETWORK"
    LOCAL = "LOCAL"
    PHYSICAL = "PHYSICAL"
    ADJACENT_NETWORK = "ADJACENT_NETWORK"
    SOCIAL_ENGINEERING = "SOCIAL_ENGINEERING"

class SecurityControl(Enum):
    """Security control types."""
    PREVENTIVE = "PREVENTIVE"
    DETECTIVE = "DETECTIVE"
    CORRECTIVE = "CORRECTIVE"
    COMPENSATING = "COMPENSATING"

class RiskLevel(Enum):
    """Risk levels."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    MINIMAL = "MINIMAL"

@dataclass
class Threat:
    """Threat definition."""
    threat_id: str
    name: str
    description: str
    category: ThreatCategory
    attack_vector: AttackVector
    likelihood: float  # 0.0 to 1.0
    impact: float  # 0.0 to 1.0
    risk_score: float  # Calculated from likelihood * impact
    affected_components: List[str]
    attack_scenarios: List[str]
    prerequisites: List[str]
    mitigations: List[str]
    references: List[str]
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class SecurityControl:
    """Security control definition."""
    control_id: str
    name: str
    description: str
    control_type: SecurityControl
    category: str
    implementation_effort: str
    effectiveness: float  # 0.0 to 1.0
    cost: str
    prerequisites: List[str]
    implementation_guide: str
    references: List[str]
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class RiskAssessment:
    """Risk assessment definition."""
    assessment_id: str
    threat_id: str
    component: str
    risk_level: RiskLevel
    likelihood: float
    impact: float
    risk_score: float
    current_controls: List[str]
    residual_risk: float
    recommendations: List[str]
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class ThreatModel:
    """Threat model definition."""
    model_id: str
    application_name: str
    version: str
    description: str
    architecture: Dict[str, Any]
    data_flows: List[Dict[str, Any]]
    trust_boundaries: List[Dict[str, Any]]
    threats: List[Threat]
    security_controls: List[SecurityControl]
    risk_assessments: List[RiskAssessment]
    created_at: datetime = field(default_factory=datetime.now)

class ThreatModeler:
    """
    Comprehensive threat modeling system.
    
    Features:
    - STRIDE threat analysis
    - Attack vector identification
    - Security control recommendations
    - Risk assessment and scoring
    - Threat model visualization
    - Mitigation strategies
    """
    
    def __init__(self, db_path: str = "data/threat_modeling.db"):
        """
        Initialize threat modeler.
        
        Args:
            db_path: Path to threat modeling database
        """
        self.db_path = db_path
        self.threat_models: Dict[str, ThreatModel] = {}
        self.threat_library: Dict[str, Threat] = {}
        self.security_controls: Dict[str, SecurityControl] = {}
        
        # Initialize database
        self._init_database()
        
        # Load default threat library
        self._load_default_threats()
        
        # Load default security controls
        self._load_default_security_controls()
        
        logger.info("Threat Modeling system initialized")
    
    def _init_database(self) -> None:
        """Initialize threat modeling database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create threat models table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS threat_models (
                model_id TEXT PRIMARY KEY,
                application_name TEXT NOT NULL,
                version TEXT,
                description TEXT,
                architecture TEXT,
                data_flows TEXT,
                trust_boundaries TEXT,
                threats TEXT,
                security_controls TEXT,
                risk_assessments TEXT,
                created_at TEXT NOT NULL
            )
        """)
        
        # Create threats table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS threats (
                threat_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                category TEXT NOT NULL,
                attack_vector TEXT,
                likelihood REAL,
                impact REAL,
                risk_score REAL,
                affected_components TEXT,
                attack_scenarios TEXT,
                prerequisites TEXT,
                mitigations TEXT,
                references TEXT,
                created_at TEXT NOT NULL
            )
        """)
        
        # Create security controls table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS security_controls (
                control_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                control_type TEXT,
                category TEXT,
                implementation_effort TEXT,
                effectiveness REAL,
                cost TEXT,
                prerequisites TEXT,
                implementation_guide TEXT,
                references TEXT,
                created_at TEXT NOT NULL
            )
        """)
        
        # Create risk assessments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS risk_assessments (
                assessment_id TEXT PRIMARY KEY,
                threat_id TEXT,
                component TEXT,
                risk_level TEXT,
                likelihood REAL,
                impact REAL,
                risk_score REAL,
                current_controls TEXT,
                residual_risk REAL,
                recommendations TEXT,
                created_at TEXT NOT NULL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _load_default_threats(self) -> None:
        """Load default threat library."""
        default_threats = [
            Threat(
                threat_id="THREAT_001",
                name="Identity Spoofing",
                description="An attacker impersonates a legitimate user or system",
                category=ThreatCategory.SPOOFING,
                attack_vector=AttackVector.NETWORK,
                likelihood=0.7,
                impact=0.8,
                risk_score=0.56,
                affected_components=["Authentication System", "User Interface"],
                attack_scenarios=[
                    "Brute force password attacks",
                    "Credential theft",
                    "Session hijacking",
                    "Social engineering"
                ],
                prerequisites=["Weak authentication", "No MFA", "Predictable credentials"],
                mitigations=[
                    "Strong authentication mechanisms",
                    "Multi-factor authentication",
                    "Account lockout policies",
                    "Regular password changes"
                ],
                references=["https://owasp.org/www-community/attacks/Identity_Spoofing"]
            ),
            Threat(
                threat_id="THREAT_002",
                name="Data Tampering",
                description="An attacker modifies data in transit or at rest",
                category=ThreatCategory.TAMPERING,
                attack_vector=AttackVector.NETWORK,
                likelihood=0.6,
                impact=0.9,
                risk_score=0.54,
                affected_components=["Database", "API", "File System"],
                attack_scenarios=[
                    "SQL injection",
                    "Man-in-the-middle attacks",
                    "File system access",
                    "API manipulation"
                ],
                prerequisites=["Unencrypted data", "Weak access controls", "No integrity checks"],
                mitigations=[
                    "Data encryption",
                    "Digital signatures",
                    "Access controls",
                    "Input validation"
                ],
                references=["https://owasp.org/www-community/attacks/Data_Tampering"]
            ),
            Threat(
                threat_id="THREAT_003",
                name="Repudiation",
                description="An attacker denies performing an action",
                category=ThreatCategory.REPUDIATION,
                attack_vector=AttackVector.LOCAL,
                likelihood=0.4,
                impact=0.6,
                risk_score=0.24,
                affected_components=["Audit Logs", "Transaction Records"],
                attack_scenarios=[
                    "Log deletion",
                    "Audit trail manipulation",
                    "Transaction modification",
                    "Digital signature forgery"
                ],
                prerequisites=["Weak logging", "No digital signatures", "Poor audit controls"],
                mitigations=[
                    "Comprehensive logging",
                    "Digital signatures",
                    "Immutable audit trails",
                    "Regular log reviews"
                ],
                references=["https://owasp.org/www-community/attacks/Repudiation"]
            ),
            Threat(
                threat_id="THREAT_004",
                name="Information Disclosure",
                description="Sensitive information is exposed to unauthorized parties",
                category=ThreatCategory.INFORMATION_DISCLOSURE,
                attack_vector=AttackVector.NETWORK,
                likelihood=0.8,
                impact=0.7,
                risk_score=0.56,
                affected_components=["Database", "API", "File System", "Memory"],
                attack_scenarios=[
                    "SQL injection",
                    "Directory traversal",
                    "Information leakage",
                    "Memory dumps"
                ],
                prerequisites=["Weak access controls", "Information leakage", "No data classification"],
                mitigations=[
                    "Access controls",
                    "Data encryption",
                    "Input validation",
                    "Error handling"
                ],
                references=["https://owasp.org/www-community/attacks/Information_Disclosure"]
            ),
            Threat(
                threat_id="THREAT_005",
                name="Denial of Service",
                description="An attacker disrupts service availability",
                category=ThreatCategory.DENIAL_OF_SERVICE,
                attack_vector=AttackVector.NETWORK,
                likelihood=0.9,
                impact=0.5,
                risk_score=0.45,
                affected_components=["Web Server", "Database", "Network", "API"],
                attack_scenarios=[
                    "DDoS attacks",
                    "Resource exhaustion",
                    "Application crashes",
                    "Network flooding"
                ],
                prerequisites=["No rate limiting", "Resource constraints", "Weak monitoring"],
                mitigations=[
                    "Rate limiting",
                    "Load balancing",
                    "Resource monitoring",
                    "DDoS protection"
                ],
                references=["https://owasp.org/www-community/attacks/Denial_of_Service"]
            ),
            Threat(
                threat_id="THREAT_006",
                name="Elevation of Privilege",
                description="An attacker gains unauthorized elevated privileges",
                category=ThreatCategory.ELEVATION_OF_PRIVILEGE,
                attack_vector=AttackVector.LOCAL,
                likelihood=0.5,
                impact=0.9,
                risk_score=0.45,
                affected_components=["Operating System", "Application", "Database"],
                attack_scenarios=[
                    "Buffer overflow",
                    "Privilege escalation",
                    "Configuration errors",
                    "Vulnerability exploitation"
                ],
                prerequisites=["Weak access controls", "Unpatched systems", "Configuration errors"],
                mitigations=[
                    "Principle of least privilege",
                    "Regular patching",
                    "Access controls",
                    "Vulnerability management"
                ],
                references=["https://owasp.org/www-community/attacks/Elevation_of_Privilege"]
            )
        ]
        
        for threat in default_threats:
            self.threat_library[threat.threat_id] = threat
            self._store_threat(threat)
    
    def _load_default_security_controls(self) -> None:
        """Load default security controls."""
        default_controls = [
            SecurityControl(
                control_id="CONTROL_001",
                name="Multi-Factor Authentication",
                description="Require multiple authentication factors for user access",
                control_type=SecurityControl.PREVENTIVE,
                category="Authentication",
                implementation_effort="Medium",
                effectiveness=0.8,
                cost="Medium",
                prerequisites=["Authentication system", "User devices"],
                implementation_guide="Implement MFA using SMS, email, or authenticator apps",
                references=["https://owasp.org/www-community/controls/Authentication"]
            ),
            SecurityControl(
                control_id="CONTROL_002",
                name="Input Validation",
                description="Validate and sanitize all user inputs",
                control_type=SecurityControl.PREVENTIVE,
                category="Input Validation",
                implementation_effort="Medium",
                effectiveness=0.7,
                cost="Low",
                prerequisites=["Application code", "Validation framework"],
                implementation_guide="Implement server-side validation for all inputs",
                references=["https://owasp.org/www-community/controls/Input_Validation"]
            ),
            SecurityControl(
                control_id="CONTROL_003",
                name="Data Encryption",
                description="Encrypt sensitive data at rest and in transit",
                control_type=SecurityControl.PREVENTIVE,
                category="Cryptography",
                implementation_effort="High",
                effectiveness=0.9,
                cost="High",
                prerequisites=["Encryption libraries", "Key management"],
                implementation_guide="Use AES-256 for data at rest and TLS 1.3 for data in transit",
                references=["https://owasp.org/www-community/controls/Transport_Layer_Protection"]
            ),
            SecurityControl(
                control_id="CONTROL_004",
                name="Access Controls",
                description="Implement role-based access controls",
                control_type=SecurityControl.PREVENTIVE,
                category="Access Control",
                implementation_effort="High",
                effectiveness=0.8,
                cost="Medium",
                prerequisites=["User management system", "Role definitions"],
                implementation_guide="Implement RBAC with principle of least privilege",
                references=["https://owasp.org/www-community/controls/Access_Control"]
            ),
            SecurityControl(
                control_id="CONTROL_005",
                name="Security Monitoring",
                description="Monitor system for security events and anomalies",
                control_type=SecurityControl.DETECTIVE,
                category="Monitoring",
                implementation_effort="High",
                effectiveness=0.7,
                cost="High",
                prerequisites=["SIEM system", "Log aggregation"],
                implementation_guide="Implement SIEM with real-time monitoring and alerting",
                references=["https://owasp.org/www-community/controls/Security_Monitoring"]
            ),
            SecurityControl(
                control_id="CONTROL_006",
                name="Vulnerability Management",
                description="Regular vulnerability scanning and patch management",
                control_type=SecurityControl.PREVENTIVE,
                category="Vulnerability Management",
                implementation_effort="Medium",
                effectiveness=0.8,
                cost="Medium",
                prerequisites=["Vulnerability scanner", "Patch management system"],
                implementation_guide="Implement regular scanning and automated patching",
                references=["https://owasp.org/www-community/controls/Vulnerability_Management"]
            ),
            SecurityControl(
                control_id="CONTROL_007",
                name="Incident Response",
                description="Procedures for responding to security incidents",
                control_type=SecurityControl.CORRECTIVE,
                category="Incident Response",
                implementation_effort="High",
                effectiveness=0.6,
                cost="Medium",
                prerequisites=["Incident response team", "Communication plan"],
                implementation_guide="Develop and test incident response procedures",
                references=["https://owasp.org/www-community/controls/Incident_Response"]
            ),
            SecurityControl(
                control_id="CONTROL_008",
                name="Backup and Recovery",
                description="Regular backups and disaster recovery procedures",
                control_type=SecurityControl.CORRECTIVE,
                category="Backup and Recovery",
                implementation_effort="Medium",
                effectiveness=0.8,
                cost="Medium",
                prerequisites=["Backup system", "Recovery procedures"],
                implementation_guide="Implement automated backups and test recovery procedures",
                references=["https://owasp.org/www-community/controls/Backup_and_Recovery"]
            )
        ]
        
        for control in default_controls:
            self.security_controls[control.control_id] = control
            self._store_security_control(control)
    
    def create_threat_model(self, application_name: str, version: str, 
                           description: str, architecture: Dict[str, Any]) -> str:
        """
        Create a new threat model.
        
        Args:
            application_name: Name of the application
            version: Application version
            description: Application description
            architecture: Application architecture
            
        Returns:
            Threat model ID
        """
        model_id = f"TM_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        threat_model = ThreatModel(
            model_id=model_id,
            application_name=application_name,
            version=version,
            description=description,
            architecture=architecture,
            data_flows=[],
            trust_boundaries=[],
            threats=[],
            security_controls=[],
            risk_assessments=[]
        )
        
        self.threat_models[model_id] = threat_model
        self._store_threat_model(threat_model)
        
        logger.info(f"Created threat model: {model_id} - {application_name}")
        return model_id
    
    def add_data_flow(self, model_id: str, data_flow: Dict[str, Any]) -> bool:
        """
        Add a data flow to the threat model.
        
        Args:
            model_id: Threat model ID
            data_flow: Data flow definition
            
        Returns:
            True if added successfully
        """
        if model_id not in self.threat_models:
            return False
        
        threat_model = self.threat_models[model_id]
        threat_model.data_flows.append(data_flow)
        
        self._update_threat_model(threat_model)
        
        logger.info(f"Added data flow to threat model: {model_id}")
        return True
    
    def add_trust_boundary(self, model_id: str, trust_boundary: Dict[str, Any]) -> bool:
        """
        Add a trust boundary to the threat model.
        
        Args:
            model_id: Threat model ID
            trust_boundary: Trust boundary definition
            
        Returns:
            True if added successfully
        """
        if model_id not in self.threat_models:
            return False
        
        threat_model = self.threat_models[model_id]
        threat_model.trust_boundaries.append(trust_boundary)
        
        self._update_threat_model(threat_model)
        
        logger.info(f"Added trust boundary to threat model: {model_id}")
        return True
    
    def analyze_threats(self, model_id: str) -> List[Threat]:
        """
        Analyze threats for a threat model using STRIDE methodology.
        
        Args:
            model_id: Threat model ID
            
        Returns:
            List of identified threats
        """
        if model_id not in self.threat_models:
            return []
        
        threat_model = self.threat_models[model_id]
        
        # Analyze threats based on data flows and trust boundaries
        identified_threats = []
        
        for data_flow in threat_model.data_flows:
            # Analyze each data flow for STRIDE threats
            flow_threats = self._analyze_data_flow_threats(data_flow, threat_model)
            identified_threats.extend(flow_threats)
        
        # Add identified threats to the model
        threat_model.threats = identified_threats
        
        # Generate risk assessments
        risk_assessments = self._generate_risk_assessments(threat_model)
        threat_model.risk_assessments = risk_assessments
        
        # Recommend security controls
        recommended_controls = self._recommend_security_controls(threat_model)
        threat_model.security_controls = recommended_controls
        
        self._update_threat_model(threat_model)
        
        logger.info(f"Analyzed threats for threat model: {model_id} - Found {len(identified_threats)} threats")
        return identified_threats
    
    def _analyze_data_flow_threats(self, data_flow: Dict[str, Any], 
                                  threat_model: ThreatModel) -> List[Threat]:
        """Analyze threats for a specific data flow."""
        threats = []
        
        # Get data flow components
        source = data_flow.get('source', '')
        destination = data_flow.get('destination', '')
        data_type = data_flow.get('data_type', '')
        protocol = data_flow.get('protocol', '')
        
        # Analyze for each STRIDE category
        for threat_id, threat in self.threat_library.items():
            # Check if threat applies to this data flow
            if self._threat_applies_to_flow(threat, data_flow):
                # Create a specific instance of the threat
                specific_threat = Threat(
                    threat_id=f"{threat_id}_{data_flow.get('flow_id', 'unknown')}",
                    name=f"{threat.name} - {data_flow.get('flow_id', 'Flow')}",
                    description=f"{threat.description} affecting data flow from {source} to {destination}",
                    category=threat.category,
                    attack_vector=threat.attack_vector,
                    likelihood=threat.likelihood,
                    impact=threat.impact,
                    risk_score=threat.risk_score,
                    affected_components=[source, destination],
                    attack_scenarios=threat.attack_scenarios,
                    prerequisites=threat.prerequisites,
                    mitigations=threat.mitigations,
                    references=threat.references
                )
                threats.append(specific_threat)
        
        return threats
    
    def _threat_applies_to_flow(self, threat: Threat, data_flow: Dict[str, Any]) -> bool:
        """Check if a threat applies to a specific data flow."""
        # This is a simplified check - in a real implementation, this would be more sophisticated
        data_type = data_flow.get('data_type', '').lower()
        protocol = data_flow.get('protocol', '').lower()
        
        # Check if threat category matches data flow characteristics
        if threat.category == ThreatCategory.SPOOFING:
            return 'authentication' in data_type or 'login' in data_type
        elif threat.category == ThreatCategory.TAMPERING:
            return 'database' in data_type or 'api' in data_type
        elif threat.category == ThreatCategory.INFORMATION_DISCLOSURE:
            return 'sensitive' in data_type or 'personal' in data_type
        elif threat.category == ThreatCategory.DENIAL_OF_SERVICE:
            return 'http' in protocol or 'api' in data_type
        elif threat.category == ThreatCategory.ELEVATION_OF_PRIVILEGE:
            return 'admin' in data_type or 'privilege' in data_type
        elif threat.category == ThreatCategory.REPUDIATION:
            return 'transaction' in data_type or 'audit' in data_type
        
        return True  # Default to applying the threat
    
    def _generate_risk_assessments(self, threat_model: ThreatModel) -> List[RiskAssessment]:
        """Generate risk assessments for threats."""
        risk_assessments = []
        
        for threat in threat_model.threats:
            for component in threat.affected_components:
                # Calculate risk level based on risk score
                risk_level = self._calculate_risk_level(threat.risk_score)
                
                # Calculate residual risk (after applying controls)
                residual_risk = self._calculate_residual_risk(threat, threat_model.security_controls)
                
                assessment = RiskAssessment(
                    assessment_id=f"RA_{threat.threat_id}_{component}",
                    threat_id=threat.threat_id,
                    component=component,
                    risk_level=risk_level,
                    likelihood=threat.likelihood,
                    impact=threat.impact,
                    risk_score=threat.risk_score,
                    current_controls=[],
                    residual_risk=residual_risk,
                    recommendations=self._generate_threat_recommendations(threat)
                )
                risk_assessments.append(assessment)
        
        return risk_assessments
    
    def _calculate_risk_level(self, risk_score: float) -> RiskLevel:
        """Calculate risk level based on risk score."""
        if risk_score >= 0.8:
            return RiskLevel.CRITICAL
        elif risk_score >= 0.6:
            return RiskLevel.HIGH
        elif risk_score >= 0.4:
            return RiskLevel.MEDIUM
        elif risk_score >= 0.2:
            return RiskLevel.LOW
        else:
            return RiskLevel.MINIMAL
    
    def _calculate_residual_risk(self, threat: Threat, controls: List[SecurityControl]) -> float:
        """Calculate residual risk after applying security controls."""
        # This is a simplified calculation
        # In a real implementation, this would consider control effectiveness and coverage
        
        total_effectiveness = 0.0
        for control in controls:
            # Check if control mitigates this threat
            if self._control_mitigates_threat(control, threat):
                total_effectiveness += control.effectiveness
        
        # Cap effectiveness at 1.0
        total_effectiveness = min(1.0, total_effectiveness)
        
        # Calculate residual risk
        residual_risk = threat.risk_score * (1.0 - total_effectiveness)
        return residual_risk
    
    def _control_mitigates_threat(self, control: SecurityControl, threat: Threat) -> bool:
        """Check if a security control mitigates a specific threat."""
        # This is a simplified check - in a real implementation, this would be more sophisticated
        
        if threat.category == ThreatCategory.SPOOFING:
            return 'authentication' in control.category.lower() or 'mfa' in control.name.lower()
        elif threat.category == ThreatCategory.TAMPERING:
            return 'validation' in control.category.lower() or 'encryption' in control.category.lower()
        elif threat.category == ThreatCategory.INFORMATION_DISCLOSURE:
            return 'encryption' in control.category.lower() or 'access' in control.category.lower()
        elif threat.category == ThreatCategory.DENIAL_OF_SERVICE:
            return 'monitoring' in control.category.lower() or 'rate' in control.name.lower()
        elif threat.category == ThreatCategory.ELEVATION_OF_PRIVILEGE:
            return 'access' in control.category.lower() or 'privilege' in control.name.lower()
        elif threat.category == ThreatCategory.REPUDIATION:
            return 'logging' in control.category.lower() or 'audit' in control.name.lower()
        
        return False
    
    def _generate_threat_recommendations(self, threat: Threat) -> List[str]:
        """Generate recommendations for a specific threat."""
        recommendations = []
        
        # Add threat-specific mitigations
        recommendations.extend(threat.mitigations)
        
        # Add general recommendations based on threat category
        if threat.category == ThreatCategory.SPOOFING:
            recommendations.extend([
                "Implement strong authentication mechanisms",
                "Use multi-factor authentication",
                "Implement account lockout policies"
            ])
        elif threat.category == ThreatCategory.TAMPERING:
            recommendations.extend([
                "Implement input validation",
                "Use data encryption",
                "Implement integrity checks"
            ])
        elif threat.category == ThreatCategory.INFORMATION_DISCLOSURE:
            recommendations.extend([
                "Implement access controls",
                "Use data encryption",
                "Implement proper error handling"
            ])
        elif threat.category == ThreatCategory.DENIAL_OF_SERVICE:
            recommendations.extend([
                "Implement rate limiting",
                "Use load balancing",
                "Implement monitoring and alerting"
            ])
        elif threat.category == ThreatCategory.ELEVATION_OF_PRIVILEGE:
            recommendations.extend([
                "Implement principle of least privilege",
                "Regular security updates",
                "Implement access controls"
            ])
        elif threat.category == ThreatCategory.REPUDIATION:
            recommendations.extend([
                "Implement comprehensive logging",
                "Use digital signatures",
                "Implement audit trails"
            ])
        
        return recommendations
    
    def _recommend_security_controls(self, threat_model: ThreatModel) -> List[SecurityControl]:
        """Recommend security controls based on identified threats."""
        recommended_controls = []
        
        # Analyze threats and recommend appropriate controls
        threat_categories = set(threat.category for threat in threat_model.threats)
        
        for category in threat_categories:
            # Find controls that address this threat category
            for control in self.security_controls.values():
                if self._control_addresses_category(control, category):
                    recommended_controls.append(control)
        
        # Remove duplicates
        unique_controls = []
        seen_controls = set()
        for control in recommended_controls:
            if control.control_id not in seen_controls:
                unique_controls.append(control)
                seen_controls.add(control.control_id)
        
        return unique_controls
    
    def _control_addresses_category(self, control: SecurityControl, category: ThreatCategory) -> bool:
        """Check if a control addresses a specific threat category."""
        # This is a simplified check - in a real implementation, this would be more sophisticated
        
        if category == ThreatCategory.SPOOFING:
            return 'authentication' in control.category.lower()
        elif category == ThreatCategory.TAMPERING:
            return 'validation' in control.category.lower() or 'encryption' in control.category.lower()
        elif category == ThreatCategory.INFORMATION_DISCLOSURE:
            return 'encryption' in control.category.lower() or 'access' in control.category.lower()
        elif category == ThreatCategory.DENIAL_OF_SERVICE:
            return 'monitoring' in control.category.lower()
        elif category == ThreatCategory.ELEVATION_OF_PRIVILEGE:
            return 'access' in control.category.lower()
        elif category == ThreatCategory.REPUDIATION:
            return 'logging' in control.category.lower() or 'audit' in control.name.lower()
        
        return False
    
    def generate_threat_model_report(self, model_id: str) -> Dict[str, Any]:
        """
        Generate comprehensive threat model report.
        
        Args:
            model_id: Threat model ID
            
        Returns:
            Threat model report dictionary
        """
        if model_id not in self.threat_models:
            return {}
        
        threat_model = self.threat_models[model_id]
        
        # Calculate overall risk metrics
        total_threats = len(threat_model.threats)
        critical_threats = len([t for t in threat_model.threats if t.risk_score >= 0.8])
        high_threats = len([t for t in threat_model.threats if 0.6 <= t.risk_score < 0.8])
        medium_threats = len([t for t in threat_model.threats if 0.4 <= t.risk_score < 0.6])
        low_threats = len([t for t in threat_model.threats if t.risk_score < 0.4])
        
        # Calculate average risk score
        avg_risk_score = sum(t.risk_score for t in threat_model.threats) / total_threats if total_threats > 0 else 0.0
        
        # Create report
        report = {
            'model_id': threat_model.model_id,
            'application_name': threat_model.application_name,
            'version': threat_model.version,
            'description': threat_model.description,
            'architecture': threat_model.architecture,
            'data_flows': threat_model.data_flows,
            'trust_boundaries': threat_model.trust_boundaries,
            'summary': {
                'total_threats': total_threats,
                'critical_threats': critical_threats,
                'high_threats': high_threats,
                'medium_threats': medium_threats,
                'low_threats': low_threats,
                'average_risk_score': avg_risk_score,
                'overall_risk_level': self._calculate_risk_level(avg_risk_score).value
            },
            'threats': [
                {
                    'threat_id': t.threat_id,
                    'name': t.name,
                    'description': t.description,
                    'category': t.category.value,
                    'attack_vector': t.attack_vector.value,
                    'likelihood': t.likelihood,
                    'impact': t.impact,
                    'risk_score': t.risk_score,
                    'affected_components': t.affected_components,
                    'attack_scenarios': t.attack_scenarios,
                    'prerequisites': t.prerequisites,
                    'mitigations': t.mitigations
                }
                for t in threat_model.threats
            ],
            'security_controls': [
                {
                    'control_id': c.control_id,
                    'name': c.name,
                    'description': c.description,
                    'control_type': c.control_type.value,
                    'category': c.category,
                    'implementation_effort': c.implementation_effort,
                    'effectiveness': c.effectiveness,
                    'cost': c.cost,
                    'prerequisites': c.prerequisites,
                    'implementation_guide': c.implementation_guide
                }
                for c in threat_model.security_controls
            ],
            'risk_assessments': [
                {
                    'assessment_id': ra.assessment_id,
                    'threat_id': ra.threat_id,
                    'component': ra.component,
                    'risk_level': ra.risk_level.value,
                    'likelihood': ra.likelihood,
                    'impact': ra.impact,
                    'risk_score': ra.risk_score,
                    'current_controls': ra.current_controls,
                    'residual_risk': ra.residual_risk,
                    'recommendations': ra.recommendations
                }
                for ra in threat_model.risk_assessments
            ],
            'generated_at': datetime.now().isoformat()
        }
        
        # Save report
        report_path = f"reports/threat_model_{model_id}.json"
        Path("reports").mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def _store_threat_model(self, threat_model: ThreatModel) -> None:
        """Store threat model in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO threat_models 
            (model_id, application_name, version, description, architecture, data_flows,
             trust_boundaries, threats, security_controls, risk_assessments, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            threat_model.model_id, threat_model.application_name, threat_model.version,
            threat_model.description, json.dumps(threat_model.architecture),
            json.dumps(threat_model.data_flows), json.dumps(threat_model.trust_boundaries),
            json.dumps([t.__dict__ for t in threat_model.threats]),
            json.dumps([c.__dict__ for c in threat_model.security_controls]),
            json.dumps([ra.__dict__ for ra in threat_model.risk_assessments]),
            threat_model.created_at.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def _update_threat_model(self, threat_model: ThreatModel) -> None:
        """Update threat model in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE threat_models
            SET data_flows = ?, trust_boundaries = ?, threats = ?, 
                security_controls = ?, risk_assessments = ?
            WHERE model_id = ?
        """, (
            json.dumps(threat_model.data_flows), json.dumps(threat_model.trust_boundaries),
            json.dumps([t.__dict__ for t in threat_model.threats]),
            json.dumps([c.__dict__ for c in threat_model.security_controls]),
            json.dumps([ra.__dict__ for ra in threat_model.risk_assessments]),
            threat_model.model_id
        ))
        
        conn.commit()
        conn.close()
    
    def _store_threat(self, threat: Threat) -> None:
        """Store threat in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO threats 
            (threat_id, name, description, category, attack_vector, likelihood, impact,
             risk_score, affected_components, attack_scenarios, prerequisites, mitigations, references, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            threat.threat_id, threat.name, threat.description, threat.category.value,
            threat.attack_vector.value, threat.likelihood, threat.impact, threat.risk_score,
            json.dumps(threat.affected_components), json.dumps(threat.attack_scenarios),
            json.dumps(threat.prerequisites), json.dumps(threat.mitigations),
            json.dumps(threat.references), threat.created_at.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def _store_security_control(self, control: SecurityControl) -> None:
        """Store security control in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO security_controls 
            (control_id, name, description, control_type, category, implementation_effort,
             effectiveness, cost, prerequisites, implementation_guide, references, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            control.control_id, control.name, control.description, control.control_type.value,
            control.category, control.implementation_effort, control.effectiveness,
            control.cost, json.dumps(control.prerequisites), control.implementation_guide,
            json.dumps(control.references), control.created_at.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def get_threat_model_summary(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Get threat model summary for a date range.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            Threat model summary dictionary
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get threat model counts
        cursor.execute("""
            SELECT COUNT(*) as total_models
            FROM threat_models
            WHERE created_at BETWEEN ? AND ?
        """, (start_date.isoformat(), end_date.isoformat()))
        
        total_models = cursor.fetchone()[0]
        
        # Get threat statistics
        cursor.execute("""
            SELECT category, COUNT(*) as count
            FROM threats
            WHERE created_at BETWEEN ? AND ?
            GROUP BY category
        """, (start_date.isoformat(), end_date.isoformat()))
        
        threat_categories = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            'summary': {
                'total_models': total_models,
                'total_threats': sum(threat_categories.values())
            },
            'threat_categories': threat_categories,
            'period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            },
            'generated_at': datetime.now().isoformat()
        }
